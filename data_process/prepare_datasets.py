#!/usr/bin/env python3
"""Prepare one or both datasets through strict directional-count caches.

The normal mode is resumable: an already complete stage is skipped.  Use
``--refresh`` only when raw data or the requested split seed changed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("tor_fiber", "tor_starlink")
CACHE_SPLITS = ("train", "valid")
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Job:
    dataset: str
    stage: str
    command: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键执行原始轨迹转换、独立划分和 directional-count 预计算")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DEFAULT_DATASETS), help="默认同时处理两个数据集")
    parser.add_argument("--seed", type=int, default=2024, help="独立分层划分 seed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(CACHE_SPLITS),
                        help="预计算的划分，默认 train valid")
    parser.add_argument("--skip-convert", action="store_true", help="跳过原始轨迹转 data.npz")
    parser.add_argument("--skip-split", action="store_true", help="跳过 train/valid/test 划分")
    parser.add_argument("--skip-precompute", action="store_true", help="跳过 directional-count 缓存")
    parser.add_argument("--refresh", action="store_true",
                        help="重新生成已存在的数据、划分和缓存")
    parser.add_argument("--remap-labels", action="store_true",
                        help="转换时显式重映射子集标签；已有 data.npz 时需 --refresh")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    return parser


def _files(directory: Path, names: tuple[str, ...], suffix: str = ".npz") -> list[Path]:
    return [directory / f"{name}{suffix}" for name in names]


def build_jobs(args: argparse.Namespace, root: Path = ROOT) -> list[Job]:
    """Create only the missing-stage jobs unless refresh was requested."""
    config = args.config if args.config.is_absolute() else root / args.config
    python = sys.executable
    jobs: list[Job] = []
    for dataset in args.datasets:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset):
            raise ValueError("dataset 必须是单个有效目录名")
        directory = root / "npz_dataset" / dataset
        data_file = directory / "data.npz"
        if args.remap_labels and data_file.is_file() and not args.refresh and not args.skip_convert:
            raise ValueError("--remap-labels 需要重新转换已有 data.npz，请加 --refresh")
        if args.remap_labels and args.skip_convert:
            raise ValueError("--remap-labels 不能与 --skip-convert 同时使用")
        split_files = _files(directory, SPLITS)
        cache_files = _files(directory, tuple(args.splits),
                              "_directional_count_precomputed.npz")

        convert = not args.skip_convert and (args.refresh or not data_file.is_file())
        split = not args.skip_split and (args.refresh or convert or not all(path.is_file() for path in split_files))
        if convert and args.skip_split:
            raise ValueError("重新生成 data.npz 时不能使用 --skip-split")
        if split and args.skip_precompute:
            raise ValueError("重新划分时不能使用 --skip-precompute")

        if convert:
            jobs.append(Job(dataset, "convert", [python, "data_process/convert_to_npz.py",
                                                  "--dataset", dataset]))
            if args.refresh:
                jobs[-1].command.append("--refresh")
            if args.remap_labels:
                jobs[-1].command.append("--remap-labels")
        if split:
            jobs.append(Job(dataset, "split", [python, "data_process/dataset_split.py",
                                                "--dataset", dataset, "--seed", str(args.seed)]))

        if args.skip_precompute:
            continue
        cache_splits = list(args.splits) if (args.refresh or split) else [
            name for name, path in zip(args.splits, cache_files)
            if not path.is_file()]
        if cache_splits:
            command = [python, "data_process/build_representations.py",
                       "--config", str(config), "--dataset", dataset,
                       "--splits", *cache_splits]
            if args.refresh:
                command.append("--refresh")
            jobs.append(Job(dataset, "precompute", command))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        jobs = build_jobs(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not jobs:
        print("所有请求的数据集阶段均已完成；使用 --refresh 强制重新生成。")
        return 0
    for job in jobs:
        command_text = " ".join(job.command)
        print(f"[{job.dataset}:{job.stage}] {command_text}")
        if args.dry_run:
            continue
        completed = subprocess.run(job.command, cwd=ROOT, check=False)
        if completed.returncode:
            print(f"失败：{job.dataset} 的 {job.stage} 阶段", file=sys.stderr)
            return completed.returncode
    print("数据集处理完成。" if not args.dry_run else "dry-run 完成；未写入文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
