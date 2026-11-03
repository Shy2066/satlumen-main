#!/usr/bin/env python3
"""Train one ordinary SatLumen classifier on a selected dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
sys.path.insert(0, str(ROOT))

from src.data.datasets import create_directional_count_data_loaders
from src.data.directional_count import source_sha256
from src.models.networks import SatLumenConvClassifier
from src.trainers.satlumen import SatLumenTrainer
from src.utils.common import Args, load_config, set_seed, worker_init_fn
from src.utils.logger import ExperimentLogger


def safe_token(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} 只能包含字母、数字、点、下划线和连字符")
    return value


def build_model(num_classes: int, cfg: dict, device: torch.device) -> torch.nn.Module:
    representation, model = cfg["representation"], cfg["model"]
    if representation.get("kind") != "directional_count":
        raise ValueError("活动训练只支持 directional_count 表示")
    if representation.get("cache_schema") != "directional_count":
        raise ValueError("cache_schema 必须为 directional_count")
    if model.get("name") != "satlumen_conv_classifier":
        raise ValueError("directional_count 使用了未知模型")
    if representation.get("max_matrix_len") != 2700:
        raise ValueError("输入必须固定为 2x2700 direction count")
    if model.get("count_dilations") != [1, 2, 4]:
        raise ValueError("count dilations 必须为 [1,2,4]")
    if model.get("pooling") != "masked_gated_mean_max":
        raise ValueError("pooling 必须为 masked_gated_mean_max")
    common = {
        "num_classes": num_classes,
        "count_dim": model["count_dim"],
        "dropout": model["dropout"],
        "max_matrix_len": representation["max_matrix_len"],
    }
    if "mamba_depth" in model or "drop_path_rate" in model:
        raise ValueError("主模型不得配置 mamba_depth/drop_path_rate")
    instance = SatLumenConvClassifier(
        **common, temporal_dilations=model["temporal_dilations"])
    parameter_count = sum(parameter.numel() for parameter in instance.parameters())
    if parameter_count >= 1_000_000:
        raise ValueError(f"主模型参数量必须小于 1,000,000，实际为 {parameter_count}")
    return instance.to(device)


def run_training(config_path: Path, dataset: str, seed: int, run_id: str = "manual",
                 init_checkpoint: Path | None = None, batch_size: int | None = None) -> Path:
    cfg = load_config(str(config_path))
    safe_token(dataset, "--dataset")
    cfg.setdefault("dataset", {})["name"] = dataset
    cfg["training"]["seed"] = seed
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    data_root = Path(cfg["dataset"]["data_dir"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    dataset_root = data_root / dataset
    train_path = dataset_root / cfg["dataset"]["train_file"]
    valid_path = dataset_root / cfg["dataset"]["val_file"]
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(f"缺少训练或验证数据: {train_path}, {valid_path}")
    if cfg["dataset"].get("require_cache"):
        for name in ("train", "valid"):
            if not (dataset_root / f"{name}_directional_count_precomputed.npz").is_file():
                raise FileNotFoundError(f"请先生成 {dataset} 的 {name} 缓存")

    checkpoint_dir = ROOT / cfg["logging"]["checkpoint_dir"] / run_id / dataset / f"seed_{seed}"
    checkpoint = checkpoint_dir / "best_model.pth"
    if checkpoint.exists():
        raise RuntimeError(f"拒绝覆盖已有训练制品: {checkpoint}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation, training = cfg["representation"], cfg["training"]
    if init_checkpoint is not None:
        raise ValueError("候选训练必须从随机初始化开始；兼容加载仅供只读检查")
    augmentation = cfg.get("augmentation", {})
    apply_time_scale = augmentation.get("mode", "none") == "global_time_scale"
    train_loader, valid_loader, num_classes = create_directional_count_data_loaders(
        str(train_path), str(valid_path), Args(representation),
        batch_size=training["batch_size"], num_workers=training["num_workers"],
        apply_time_scale=apply_time_scale, worker_init_fn=worker_init_fn, seed=seed,
        time_scale_probability=augmentation.get("probability", 0.0),
        time_scale_min=augmentation.get("min_scale", 0.85))
    expected_classes = cfg["dataset"].get("num_classes", num_classes)
    for loader in (train_loader, valid_loader):
        if not np.array_equal(np.unique(loader.dataset.labels), np.arange(expected_classes)):
            raise ValueError(f"训练/验证标签必须覆盖 0..{expected_classes - 1}")
    logger = ExperimentLogger(str(ROOT / cfg["logging"]["log_dir"] / run_id / "training"),
                              f"{dataset}_seed_{seed}")
    try:
        model = build_model(num_classes, cfg, device)
        logger.log_config(cfg)
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg["loss"]["label_smoothing"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                                      weight_decay=training["weight_decay"])
        trainer = SatLumenTrainer(
            model, optimizer, criterion, device, logger=logger,
            train_cfg=training)
        best_metric = trainer.train(train_loader, valid_loader, training["epochs"],
                                    save_dir=str(checkpoint_dir))
    finally:
        logger.close()
    model_cfg = cfg["model"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {"created_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id, "dataset": dataset, "seed": seed,
                "num_classes": num_classes,
                "train_sha256": source_sha256(train_path),
                "valid_sha256": source_sha256(valid_path),
                "config": str(config_path.resolve()), "checkpoint": str(checkpoint.resolve()),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "best_validation_f1": best_metric, "best_epoch": trainer.best_epoch,
                "model_name": model_cfg["name"],
                "representation_kind": "directional_count",
                "cache_schema": representation["cache_schema"],
                "count_shape": [2, representation["max_matrix_len"]],
                "parameter_count": parameter_count,
                "mamba_depth": model_cfg.get("mamba_depth"),
                "temporal_dilations": model_cfg.get("temporal_dilations"),
                "count_dilations": model_cfg["count_dilations"],
                "label_smoothing": float(cfg["loss"]["label_smoothing"])}
    (checkpoint_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练一个 SatLumen 单模型")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-id", default="manual")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_training(args.config.resolve(), args.dataset, args.seed,
                     safe_token(args.run_id, "--run-id"), args.init_checkpoint,
                     args.batch_size)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
