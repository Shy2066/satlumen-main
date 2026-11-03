"""Create stratified train/valid/test splits with optional provenance groups.

Each dataset is split independently.  The first split reserves 10 percent for
test; the remaining 90 percent is split again, reserving 10 percent of that
portion for validation (81/9/10 overall).
"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_process.storage import mapped_samples
from data_process.convert_to_npz import sha256_file, safe_dataset


DEFAULT_SEED = 2024


def split_dataset(dataset: str, seed: int = DEFAULT_SEED, root: Path | None = None) -> None:
    safe_dataset(dataset)
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    dataset_path = project_root / "npz_dataset" / dataset
    source = dataset_path / "data.npz"
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source) as data:
        y = data["y"]
        original_y = data["original_y"] if "original_y" in data.files else None
        identities = {key: data[key] for key in ("sample_ids", "origin_groups", "trace_sha256")
                      if key in data.files}
    if y.ndim != 1 or not len(y) or not np.array_equal(np.unique(y), np.arange(int(y.max()) + 1)):
        raise ValueError("Labels must be a continuous one-dimensional range")
    if original_y is not None and original_y.shape != y.shape:
        raise ValueError("original_y 与标签长度不匹配")
    for key, values in identities.items():
        if values.shape != y.shape:
            raise ValueError(f"{key} 与标签长度不匹配")
    if "sample_ids" in identities and len(np.unique(identities["sample_ids"])) != len(y):
        raise ValueError("sample_ids 重复")
    if dataset == "93_1100_wang" and len(identities) != 3:
        raise ValueError("93 类数据必须重新转换以保留样本身份与同源组")
    # Union both metadata groups and actual byte hashes, so stale grouping cannot
    # place a local duplicate on both sides of a split.
    parents = np.arange(len(y))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return int(i)

    for key in ("origin_groups", "trace_sha256"):
        seen = {}
        for index, value in enumerate(identities.get(key, [])):
            if not str(value):
                raise ValueError(f"空的 {key}")
            if value in seen:
                parents[find(index)] = find(seen[value])
            else:
                seen[value] = index
    groups = np.asarray([find(i) for i in range(len(y))])
    unique_groups = np.unique(groups)
    group_labels = []
    for group in unique_groups:
        group_labels.append(int(y[group]))
    if np.any(y != np.asarray(group_labels)[np.searchsorted(unique_groups, groups)]):
        raise ValueError("同源组跨越不同类别，请复核清单")
    train_groups, test_groups = train_test_split(
        unique_groups, train_size=0.9, random_state=seed, stratify=group_labels)
    train_groups, valid_groups = train_test_split(
        train_groups, train_size=0.9, random_state=seed, stratify=y[train_groups])
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(int(group), []).append(index)
    # Preserve sklearn's original shuffled row order when every group is unique.
    indices = {name: np.asarray([index for group in values for index in members[int(group)]]) for name, values in
               (("train", train_groups), ("valid", valid_groups), ("test", test_groups))}
    for name, rows in indices.items():
        if not np.array_equal(np.unique(y[rows]), np.unique(y)):
            raise ValueError(f"{name} 缺少类别")
    manifest = {"dataset": dataset, "seed": seed, "ratios": [0.81, 0.09, 0.10],
                "unit": "origin_group_and_sha256" if identities else "sample",
                "source_sha256": sha256_file(source), "splits": {}}
    if original_y is not None:
        manifest["model_to_original_label"] = {
            str(int(label)): int(original_y[np.flatnonzero(y == label)[0]]) for label in np.unique(y)}
    with mapped_samples(source) as X, TemporaryDirectory(prefix=".split-", dir=dataset_path) as temp:
        if len(X) != len(y):
            raise ValueError("X/y 长度不匹配")
        for name, rows in indices.items():
            buffer = np.lib.format.open_memmap(Path(temp) / "split.npy", mode="w+",
                                               dtype=X.dtype, shape=(len(rows), *X.shape[1:]))
            try:
                for start in range(0, len(rows), 256):
                    buffer[start:start + 256] = X[rows[start:start + 256]]
                staged = Path(temp) / f"{name}.npz"
                np.savez_compressed(staged, X=buffer, y=y[rows],
                                    **({"original_y": original_y[rows]} if original_y is not None else {}),
                                    **{key: values[rows] for key, values in identities.items()})
            finally:
                buffer._mmap.close()
            manifest["splits"][name] = {
                "count": len(rows), "sha256": sha256_file(staged),
                "class_counts": np.bincount(y[rows].astype(int)).tolist(),
                "source_indices": rows.tolist(),
                "sample_ids": identities["sample_ids"][rows].tolist() if "sample_ids" in identities else None}
        for name in indices:
            (Path(temp) / f"{name}.npz").replace(dataset_path / f"{name}.npz")
    (dataset_path / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{dataset}: " + " ".join(f"{name}={len(rows)}" for name, rows in indices.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立分层划分 train/valid/test")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    split_dataset(args.dataset, args.seed)
