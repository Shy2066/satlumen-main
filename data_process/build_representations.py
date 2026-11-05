#!/usr/bin/env python3
"""Build strict directional-count caches from existing dataset splits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.directional_count import (
    build_directional_count, cache_metadata, metadata_json,
)
from src.utils.common import load_config
from data_process.storage import mapped_samples
from data_process.convert_to_npz import safe_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="预计算 directional-count 表示")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--dataset", required=True, type=safe_dataset)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"),
                        default=["train", "valid"])
    parser.add_argument("--refresh", action="store_true")
    return parser


def build_cache(source: Path, destination: Path, representation: dict,
                refresh=False):
    if destination.exists() and not refresh:
        raise FileExistsError(f"缓存已存在，拒绝覆盖: {destination}")
    with np.load(source) as data:
        y = data["y"]
    n = len(y)
    max_matrix_len = int(representation["max_matrix_len"])
    counts = np.zeros((n, 2, max_matrix_len), dtype=np.float32)
    count_idx = np.zeros(n, dtype=np.int64)
    nonmonotonic = np.zeros(n, dtype=np.int64)
    with mapped_samples(source) as X:
        for index in tqdm(range(n), desc=f"  {source.stem}"):
            result = build_directional_count(
                X[index, :, 1], X[index, :, 0],
                seq_len=int(representation["seq_len"]),
                max_matrix_len=max_matrix_len,
                maximum_load_time=float(representation["maximum_load_time"]))
            counts[index], count_idx[index], nonmonotonic[index] = result
    metadata = cache_metadata(
        source, seq_len=int(representation["seq_len"]),
        max_matrix_len=max_matrix_len,
        maximum_load_time=float(representation["maximum_load_time"]),
        log_transform=bool(representation.get("log_transform", True)))
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary, counts=counts, count_idx=count_idx,
        nonmonotonic_count=nonmonotonic, y=y,
        metadata_json=np.asarray(metadata_json(metadata)))
    os.replace(temporary, destination)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_config(str(config), profile="validation_candidate")
    representation = cfg["representation"]
    if representation.get("kind") != "directional_count":
        raise ValueError(
            "build_representations 仅支持 representation.kind=directional_count")
    data_root = Path(cfg["dataset"]["data_dir"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for split in args.splits:
        source = data_root / args.dataset / f"{split}.npz"
        destination = data_root / args.dataset / (
            f"{split}_directional_count_precomputed.npz")
        if not source.is_file():
            raise FileNotFoundError(source)
        build_cache(source, destination, representation, args.refresh)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
