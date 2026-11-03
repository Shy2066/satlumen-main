"""Datasets for the directional-count SatLumen candidate."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .directional_count import (
    CACHE_SCHEMA,
    COUNT_CHANNEL_NAMES,
    build_directional_count,
    cache_metadata,
    directional_count_global_time_scale,
)
from .representations import sample_global_time_scale


def _validate_time_scale_options(mode, probability, minimum):
    if mode not in (None, "global_time_scale"):
        raise ValueError(f"unsupported time_scale_mode: {mode}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("time scale probability must be in [0, 1]")
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum time scale must be in (0, 1]")


def load_data(data_path):
    with np.load(data_path) as data:
        return data["X"], data["y"]


def _directional_count_precomputed_path(data_path):
    directory, name = os.path.split(os.fspath(data_path))
    return os.path.join(directory, name.replace(
        ".npz", "_directional_count_precomputed.npz"))


def _expected_metadata(source_path, args):
    return cache_metadata(
        source_path,
        seq_len=args.seq_len,
        max_matrix_len=args.max_matrix_len,
        maximum_load_time=args.maximum_load_time,
        log_transform=getattr(args, "log_transform", True),
    )


def load_directional_count_cache(cache_path, source_path, args):
    required = {"counts", "count_idx", "nonmonotonic_count", "y",
                "metadata_json"}
    with np.load(cache_path) as data:
        actual = set(data.files)
        missing = required.difference(actual)
        extra = actual.difference(required)
        if missing or extra:
            raise ValueError(
                f"directional-count cache 字段非法: missing={sorted(missing)}, "
                f"extra={sorted(extra)}")
        try:
            metadata = json.loads(str(data["metadata_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"directional-count cache metadata_json 无效: {cache_path}") from exc
        expected = _expected_metadata(source_path, args)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"directional-count cache metadata 不匹配 {key}: "
                    f"{metadata.get(key)!r} != {value!r}")

        arrays = {key: data[key] for key in required if key != "metadata_json"}
        sample_count = len(arrays["y"])
        expected_shapes = {
            "counts": (sample_count, 2, int(args.max_matrix_len)),
            "count_idx": (sample_count,),
            "nonmonotonic_count": (sample_count,),
            "y": (sample_count,),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(
                    f"directional-count cache {key} shape 非法: "
                    f"{arrays[key].shape} != {shape}")
        if np.any(arrays["count_idx"] < -1) or np.any(
                arrays["count_idx"] >= int(args.max_matrix_len)):
            raise ValueError(f"directional-count cache count_idx 非法: {cache_path}")
        if np.any(arrays["nonmonotonic_count"] < 0):
            raise ValueError(
                f"directional-count cache nonmonotonic_count 非法: {cache_path}")
        return arrays


class DirectionalCountDataset(Dataset):
    def __init__(self, X, labels, args, apply_time_scale=False,
                 time_scale_probability=0.35, time_scale_min=0.85):
        self.X = X
        self.labels = labels
        self.args = args
        self.apply_time_scale = apply_time_scale
        self.time_scale_probability = time_scale_probability
        self.time_scale_min = time_scale_min
        _validate_time_scale_options(
            "global_time_scale" if apply_time_scale else None,
            time_scale_probability, time_scale_min)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.X[index]
        counts, count_idx, _ = build_directional_count(
            sample[:, 1], sample[:, 0], seq_len=self.args.seq_len,
            max_matrix_len=self.args.max_matrix_len,
            maximum_load_time=self.args.maximum_load_time)
        return self._render(counts, count_idx), int(self.labels[index])

    def _render(self, counts, count_idx):
        if self.apply_time_scale:
            scale = sample_global_time_scale(
                self.time_scale_probability, self.time_scale_min)
            counts, count_idx = directional_count_global_time_scale(
                counts, count_idx, scale)
        if getattr(self.args, "log_transform", True):
            counts = np.log1p(counts)
        return counts.astype(np.float32), np.int64(count_idx)


class PrecomputedDirectionalCountDataset(DirectionalCountDataset):
    def __init__(self, counts, count_indices, labels, args,
                 apply_time_scale=False, time_scale_probability=0.35,
                 time_scale_min=0.85):
        super().__init__(None, labels, args, apply_time_scale,
                         time_scale_probability, time_scale_min)
        self.counts = counts
        self.count_indices = count_indices

    def __getitem__(self, index):
        return self._render(
            self.counts[index].copy(), int(self.count_indices[index])), int(self.labels[index])


def create_directional_count_data_loaders(
        train_path, val_path, args, batch_size=128, num_workers=4,
        apply_time_scale=False, worker_init_fn=None, seed=None,
        time_scale_probability=0.35, time_scale_min=0.85):
    train_cache = _directional_count_precomputed_path(train_path)
    val_cache = _directional_count_precomputed_path(val_path)
    train_exists, val_exists = os.path.isfile(train_cache), os.path.isfile(val_cache)
    if train_exists != val_exists:
        raise ValueError("train/valid directional-count cache 必须同时存在或同时不存在")

    if train_exists:
        train = load_directional_count_cache(train_cache, train_path, args)
        valid = load_directional_count_cache(val_cache, val_path, args)
        train_dataset = PrecomputedDirectionalCountDataset(
            train["counts"], train["count_idx"], train["y"], args,
            apply_time_scale, time_scale_probability, time_scale_min)
        valid_dataset = PrecomputedDirectionalCountDataset(
            valid["counts"], valid["count_idx"], valid["y"], args)
        num_classes = len(np.unique(train["y"]))
    else:
        train_X, train_y = load_data(train_path)
        val_X, val_y = load_data(val_path)
        train_dataset = DirectionalCountDataset(
            train_X, train_y, args, apply_time_scale,
            time_scale_probability, time_scale_min)
        valid_dataset = DirectionalCountDataset(val_X, val_y, args)
        num_classes = len(np.unique(train_y))

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, worker_init_fn=worker_init_fn)
    if seed is not None:
        common["generator"] = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=True, **common),
        DataLoader(valid_dataset, shuffle=False, drop_last=False, **common),
        num_classes,
    )


def validate_schema_constants():
    return {"schema": CACHE_SCHEMA, "count_channels": COUNT_CHANNEL_NAMES}
