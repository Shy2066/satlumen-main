#!/usr/bin/env python3
"""Evaluate a frozen directional-count SatLumen checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
sys.path.insert(0, str(ROOT))

from scripts.train import build_model, safe_token
from src.data.datasets import (
    DirectionalCountDataset,
    PrecomputedDirectionalCountDataset,
    _directional_count_precomputed_path,
    load_data,
    load_directional_count_cache,
)
from src.utils.common import Args, compute_metrics, load_config, worker_init_fn
from src.models.networks import load_directional_count_checkpoint


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_loader(data_path: Path, cfg: dict, seed: int | None = None) -> tuple[DataLoader, np.ndarray, Path]:
    representation, training = cfg["representation"], cfg["training"]
    precomputed = Path(_directional_count_precomputed_path(str(data_path)))
    if precomputed.is_file():
        data = load_directional_count_cache(
            precomputed, data_path, Args(representation))
        labels = data["y"]
        dataset = PrecomputedDirectionalCountDataset(
            data["counts"], data["count_idx"], labels, Args(representation),
            apply_time_scale=False)
        resolved_path = precomputed
    else:
        samples, labels = load_data(str(data_path))
        dataset = DirectionalCountDataset(
            samples, labels, Args(representation), apply_time_scale=False)
        resolved_path = data_path
    return (DataLoader(dataset, batch_size=training["batch_size"], shuffle=False,
                       num_workers=training["num_workers"], pin_memory=True,
                       drop_last=False, worker_init_fn=worker_init_fn), labels, resolved_path)


def evaluate_checkpoint(config_path: Path, dataset: str, checkpoint: Path, output: Path,
                        source_dataset: str | None = None, split: str | None = None,
                        seed: int | None = None) -> dict:
    cfg = load_config(str(config_path), profile="formal_frozen")
    safe_token(dataset, "--dataset")
    if cfg.get("experiment", {}).get("stage") != "formal_frozen":
        raise ValueError(
            "candidate 配置禁止读取 test；测试只允许未来的 formal_frozen 配置")
    if cfg.get("representation", {}).get("kind") != "directional_count":
        raise ValueError("活动测试只支持 directional_count 配置")
    if output.exists():
        raise RuntimeError(f"拒绝覆盖已有结果: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"检查点不存在: {checkpoint}")
    data_root = Path(cfg["dataset"]["data_dir"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    filename = split or cfg["dataset"].get("test_file", "test.npz")
    if Path(filename).name != filename:
        raise ValueError("--split 必须是数据集目录内的文件名")
    data_path = data_root / dataset / filename
    if not data_path.is_file():
        raise FileNotFoundError(f"测试数据不存在: {data_path}")
    if cfg["dataset"].get("require_cache") and not Path(
            _directional_count_precomputed_path(str(data_path))).is_file():
        raise FileNotFoundError(f"请先生成测试缓存: {data_path}")
    loader, labels, resolved_data_path = make_loader(data_path, cfg, seed)
    expected_classes = cfg["dataset"].get("num_classes", len(np.unique(labels)))
    if not np.array_equal(np.unique(labels), np.arange(expected_classes)):
        raise ValueError(f"测试标签必须覆盖 0..{expected_classes - 1}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(np.unique(labels)), cfg, device)
    load_directional_count_checkpoint(model, checkpoint)
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(*(value.to(device) for value in inputs))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_type": "test", "dataset": dataset,
        "source_dataset": source_dataset, "split": filename,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()), "data_file": str(resolved_data_path.resolve()),
        "test_samples": int(len(labels)), "num_classes": int(len(np.unique(labels))),
        "metrics": compute_metrics(labels, predicted),
    }
    atomic_json(output.resolve(), result)
    np.savez_compressed(output.with_suffix(".predictions.npz"), predictions=predicted, labels=labels)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估一个 SatLumen 单模型")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, help="测试数据集")
    parser.add_argument("--source-dataset", help="训练数据集，仅写入结果元数据")
    parser.add_argument("--seed", type=int, help="训练 seed")
    parser.add_argument("--split", help="覆盖配置中的测试文件名，默认 test.npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_checkpoint(args.config.resolve(), args.dataset, args.checkpoint,
                                     args.output, args.source_dataset, args.split, args.seed)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    else:
        print(json.dumps(result["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
