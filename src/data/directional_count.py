"""Directional packet-count representation used by the current model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CACHE_SCHEMA = "directional_count"
COUNT_CHANNEL_NAMES = ("incoming_count", "outgoing_count")


def source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_metadata(source_path: str | Path, *, seq_len: int,
                   max_matrix_len: int, maximum_load_time: float,
                   log_transform: bool = True) -> dict:
    return {
        "schema": CACHE_SCHEMA,
        "count_channel_names": list(COUNT_CHANNEL_NAMES),
        "seq_len": int(seq_len),
        "max_matrix_len": int(max_matrix_len),
        "maximum_load_time": float(maximum_load_time),
        "log_transform": bool(log_transform),
        "source_sha256": source_sha256(source_path),
    }


def metadata_json(metadata: dict) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _valid_trace(packet_length, timestamps, seq_len: int):
    packet_length = np.asarray(packet_length)[:seq_len]
    timestamps = np.asarray(timestamps)[:seq_len]
    if packet_length.shape != timestamps.shape:
        raise ValueError("packet_length and timestamps must have the same shape")
    stop = len(packet_length)
    padding = np.flatnonzero((packet_length == 0) & (timestamps == 0))
    if len(padding):
        stop = int(padding[0])
    packet_length = packet_length[:stop]
    timestamps = timestamps[:stop]
    nonzero = packet_length != 0
    return packet_length[nonzero], timestamps[nonzero]


def build_directional_count(packet_length, timestamps, *, seq_len: int,
                            max_matrix_len: int,
                            maximum_load_time: float):
    """Build a raw ``(2, max_matrix_len)`` directional count map.

    Negative packet lengths are placed in row 0 and positive packet lengths
    in row 1.
    """
    if seq_len <= 0 or max_matrix_len <= 0:
        raise ValueError("representation capacities must be positive")
    if maximum_load_time <= 0:
        raise ValueError("maximum_load_time must be positive")

    lengths, times = _valid_trace(packet_length, timestamps, seq_len)
    counts = np.zeros((2, max_matrix_len), dtype=np.float32)
    if len(lengths) == 0:
        return counts, -1, 0

    times = times.astype(np.float64)
    bin_width = float(maximum_load_time) / max_matrix_len
    bins = np.floor(times / bin_width).astype(np.int64)
    bins = np.clip(bins, 0, max_matrix_len - 1)
    directions = np.where(lengths < 0, -1, 1).astype(np.int8)
    np.add.at(counts, ((directions > 0).astype(np.int64), bins), 1.0)
    count_idx = int(bins.max())
    nonmonotonic = int(np.count_nonzero(np.diff(times) < 0))
    return counts, count_idx, nonmonotonic


def directional_count_global_time_scale(counts, count_idx, scale: float):
    """Apply global time scaling while preserving both direction totals."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != 2:
        raise ValueError("counts must have shape (2, L)")
    count_idx = min(max(int(count_idx), -1), counts.shape[1] - 1)
    if scale == 1.0:
        return counts.copy(), count_idx
    if count_idx < 0:
        return np.zeros_like(counts), -1

    source = np.arange(count_idx + 1, dtype=np.int64)
    target = np.floor(source * float(scale)).astype(np.int64)
    target = np.clip(target, 0, counts.shape[1] - 1)
    warped = np.zeros_like(counts)
    for row in range(2):
        np.add.at(warped[row], target, counts[row, :count_idx + 1])
    return warped, int(target[-1])
