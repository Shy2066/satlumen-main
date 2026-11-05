#!/usr/bin/env python3
"""Train the directional-count models and run the full frozen test matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.common import compute_metrics, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
DOMAINS = ("tor_fiber", "tor_starlink")
EXPERIMENT_SEEDS = (42, 2024, 2025)
DIRECTION_EVALUATIONS = (
    ("F_to_F", "tor_fiber", "tor_fiber"),
    ("S_to_S", "tor_starlink", "tor_starlink"),
    ("F_to_S", "tor_starlink", "tor_fiber"),
    ("S_to_F", "tor_fiber", "tor_starlink"),
)
RUNNER_WORKFLOW = "three_seed_train_then_formal_test"


def safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("运行目录名只能包含字母、数字、点、下划线和连字符")
    return value


def timestamp_run_id(experiment_name: str = "satlumen_directional_count_candidate",
                     now: datetime | None = None) -> str:
    if isinstance(experiment_name, datetime):
        now = experiment_name
        experiment_name = "independent"
    now = now or datetime.now()
    return safe_run_id(experiment_name) + now.strftime("_%Y%m%d_%H%M%S")


def run_job(command: list[str], log_path: Path, gpu: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=ROOT, env=environment,
                              stdout=handle, stderr=subprocess.STDOUT,
                              check=False).returncode


def run_parallel(jobs: list[tuple[str, list[str], Path]], gpus: list[int]) -> None:
    if not jobs:
        return
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)
    failures = []

    def worker(job):
        label, command, log = job
        gpu = available.get()
        try:
            return label, run_job(command, log, gpu)
        finally:
            available.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            label, code = future.result()
            print(f"[{'OK' if code == 0 else 'FAIL'}] {label}", flush=True)
            if code:
                failures.append(label)
    if failures:
        raise RuntimeError("任务失败: " + ", ".join(failures))


def verify_torch_interpreter(python: str) -> None:
    probe = subprocess.run(
        [python, "-c", "import torch; print(torch.__version__)"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    if probe.returncode:
        detail = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(
            f"PyTorch could not be imported by Python interpreter {python}:\n{detail}\n"
            "Create a virtual environment, install dependencies from requirements.txt, "
            "and select that Python interpreter")


def checkpoint_path(cfg: dict, run_id: str, dataset: str, seed: int) -> Path:
    return (ROOT / cfg["logging"]["checkpoint_dir"] / run_id / dataset /
            f"seed_{seed}" / "best_model.pth")


def _data_root(cfg: dict) -> Path:
    data_root = Path(cfg["dataset"]["data_dir"])
    return data_root if data_root.is_absolute() else ROOT / data_root


def prepare_caches(python: str, config: Path) -> None:
    cfg = load_config(str(config), profile="validation_candidate")
    if cfg.get("experiment", {}).get("stage") != "validation_candidate":
        raise ValueError("候选 runner 只接受 stage=validation_candidate")
    if cfg.get("representation", {}).get("kind") != "directional_count":
        raise ValueError("runner 只支持 directional_count")
    for dataset in DOMAINS:
        directory = _data_root(cfg) / dataset
        missing = [
            name for name in ("train", "valid")
            if not (directory / f"{name}_directional_count_precomputed.npz").is_file()
        ]
        if missing:
            completed = subprocess.run(
                [python, "data_process/build_representations.py", "--config",
                 str(config), "--dataset", dataset, "--splits", *missing],
                cwd=ROOT, check=False)
            if completed.returncode:
                raise RuntimeError(f"{dataset} directional-count 预计算失败")


def resolved_runtime(cfg: dict) -> dict:
    representation, model = cfg["representation"], cfg["model"]
    if representation.get("kind") != "directional_count":
        raise ValueError("representation.kind 必须为 directional_count")
    if representation.get("cache_schema") != "directional_count":
        raise ValueError("cache_schema 必须为 directional_count")
    if model.get("name") != "satlumen_conv_classifier":
        raise ValueError("当前主模型必须为 satlumen_conv_classifier")
    if representation.get("max_matrix_len") != 2700:
        raise ValueError("max_matrix_len 必须为 2700")
    if model.get("count_dilations") != [1, 2, 4]:
        raise ValueError("count_dilations 必须为 [1,2,4]")
    if model.get("temporal_dilations") != [1, 2, 4, 1, 2, 4]:
        raise ValueError("当前主模型 temporal_dilations 必须为 [1,2,4,1,2,4]")
    if "mamba_depth" in model or "drop_path_rate" in model:
        raise ValueError("当前主模型不得包含 Mamba 配置字段")
    if model.get("pooling") != "masked_gated_mean_max":
        raise ValueError("pooling 必须为 masked_gated_mean_max")
    return {
        "representation": "directional_count",
        "count_shape": [2, representation["max_matrix_len"]],
        "count_stem": "depthwise_residual_d1_d2_d4_avgpool6",
        "model_name": model["name"],
        "temporal_dilations": model["temporal_dilations"],
        "pooling": "masked_gated_mean_max",
        "loss": "label_smoothed_cross_entropy",
    }


def audit_caches(cfg: dict) -> list[dict]:
    records = []
    for dataset in DOMAINS:
        for split in ("train", "valid"):
            path = (_data_root(cfg) / dataset /
                    f"{split}_directional_count_precomputed.npz")
            with np.load(path) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                records.append({
                    "dataset": dataset,
                    "split": split,
                    "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                    else str(path),
                    "schema": metadata.get("schema"),
                    "source_sha256": metadata.get("source_sha256"),
                    "fields": sorted(data.files),
                })
    return records


def collect_validation_summary(cfg: dict, run_id: str,
                               seeds: tuple[int, ...]) -> dict:
    rows = []
    for seed in seeds:
        for dataset in DOMAINS:
            metadata_path = (ROOT / cfg["logging"]["checkpoint_dir"] / run_id /
                             dataset / f"seed_{seed}" / "training_metadata.json")
            row = {
                "dataset": dataset,
                "seed": seed,
                "metadata": str(metadata_path),
                "metadata_status": "available",
                "identity_matches": None,
                "best_valid_macro_f1": None,
            }
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                row["metadata_status"] = f"unavailable: {exc}"
                rows.append(row)
                continue

            expected_identity = {
                "run_id": run_id,
                "dataset": dataset,
                "seed": seed,
                "model_name": cfg["model"]["name"],
                "representation_kind": "directional_count",
                "cache_schema": "directional_count",
            }
            mismatches = {
                key: {"expected": expected, "actual": metadata.get(key)}
                for key, expected in expected_identity.items()
                if metadata.get(key) != expected
            } if isinstance(metadata, dict) else {"metadata": "not an object"}
            row["identity_matches"] = not mismatches
            if mismatches:
                row["identity_mismatches"] = mismatches
            try:
                score = float(metadata["best_validation_f1"])
            except (KeyError, TypeError, ValueError):
                row["metric_status"] = "unavailable"
            else:
                if np.isfinite(score):
                    row["best_valid_macro_f1"] = score
                    row["metric_status"] = "available"
                else:
                    row["metric_status"] = "nonfinite"
            rows.append(row)
    return {
        "seeds": list(seeds),
        "purpose": "checkpoint_selection_record_only",
        "results": rows,
    }


def build_formal_test_jobs(python: str, cfg: dict, run_id: str,
                           seeds: tuple[int, ...], config_path: Path,
                           run_dir: Path, result_dir: Path,
                           resume: bool = False) -> tuple[list, list[dict]]:
    jobs = []
    plan = []
    for seed in seeds:
        for direction, target, source in DIRECTION_EVALUATIONS:
            checkpoint = checkpoint_path(cfg, run_id, source, seed)
            output = result_dir / f"{direction}_seed_{seed}.json"
            prediction_file = output.with_suffix(".predictions.npz")
            log_path = run_dir / "jobs" / f"test_{direction}_seed_{seed}.log"
            artifacts_exist = output.exists() or prediction_file.exists()
            status = "pending"
            if artifacts_exist:
                if not resume or not output.is_file() or not prediction_file.is_file():
                    raise RuntimeError(
                        f"测试产物已存在或不完整，拒绝覆盖: {output} / {prediction_file}")
                result = json.loads(output.read_text(encoding="utf-8"))
                expected = {
                    "result_type": "test",
                    "dataset": target,
                    "source_dataset": source,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                }
                mismatches = {
                    key: {"expected": value, "actual": result.get(key)}
                    for key, value in expected.items()
                    if result.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(f"已有测试结果身份不匹配: {output}: {mismatches}")
                status = "completed"
            else:
                command = [
                    python, "scripts/test.py", "--config", str(config_path),
                    "--dataset", target, "--source-dataset", source,
                    "--seed", str(seed), "--checkpoint", str(checkpoint),
                    "--output", str(output),
                ]
                jobs.append((f"test {direction} seed={seed}", command, log_path))
            plan.append({
                "direction": direction,
                "dataset": target,
                "source_dataset": source,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "output": str(output),
                "prediction_file": str(prediction_file),
                "log": str(log_path),
                "status": status,
            })
    return jobs, plan


def metrics_from_prediction_file(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"predictions", "labels"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} 缺少预测字段: {sorted(missing)}")
        return compute_metrics(archive["labels"], archive["predictions"])


def summarize_test_metrics(results: list[dict]) -> dict:
    summary = {}
    metric_names = ("Accuracy", "Precision", "Recall", "F1-score")
    for direction, _, _ in DIRECTION_EVALUATIONS:
        direction_results = [item for item in results
                             if item["direction"] == direction]
        if len(direction_results) != len(EXPERIMENT_SEEDS):
            raise ValueError(
                f"{direction} 汇总需要 {len(EXPERIMENT_SEEDS)} 个 seed 结果，"
                f"实际为 {len(direction_results)}")
        metrics = {}
        for name in metric_names:
            values = [float(item["metrics"][name])
                      for item in direction_results]
            metrics[name] = {
                "mean": round(float(np.mean(values)), 2),
                "sample_std": round(float(np.std(values, ddof=1)), 2)
                if len(values) > 1 else 0.0,
            }
        summary[direction] = {
            "seed_count": len(direction_results),
            "metrics": metrics,
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练方向计数模型并运行完整的三种子正式测试矩阵")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = args.config if args.config.is_absolute() else ROOT / args.config
    config = config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    cfg = load_config(str(config), profile="validation_candidate")
    runtime = resolved_runtime(cfg)
    if cfg.get("experiment", {}).get("stage") != "validation_candidate":
        raise RuntimeError("训练配置必须使用 stage=validation_candidate")
    formal_cfg = load_config(str(config), profile="formal_frozen")
    if formal_cfg.get("experiment", {}).get("stage") != "formal_frozen":
        raise RuntimeError("正式评估配置必须使用 stage=formal_frozen")
    if formal_cfg.get("experiment", {}).get("seeds") != list(EXPERIMENT_SEEDS):
        raise ValueError("正式评估配置的 seeds 必须与训练配置一致")
    if (formal_cfg.get("representation") != cfg.get("representation") or
            formal_cfg.get("model") != cfg.get("model") or
            formal_cfg.get("dataset", {}).get("data_dir") !=
            cfg.get("dataset", {}).get("data_dir")):
        raise ValueError("正式评估与训练配置的表示、模型或数据根目录不一致")
    seeds = tuple(cfg["experiment"].get("seeds", []))
    if seeds != EXPERIMENT_SEEDS:
        raise ValueError(
            f"完整训练与测试必须使用 seeds={list(EXPERIMENT_SEEDS)}")
    run_id = safe_run_id(args.resume) if args.resume else timestamp_run_id(
        cfg["experiment"]["name"])
    try:
        gpus = [int(value) for value in args.gpus.split(",") if value]
    except ValueError as exc:
        raise ValueError("--gpus 必须是逗号分隔的整数") from exc
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus 必须非空且不能重复")

    result_dir = ROOT / "results/directional_count" / run_id
    evaluation_plan = [
        {
            "direction": direction,
            "dataset": target,
            "source_dataset": source,
            "seed": seed,
            "checkpoint": str(checkpoint_path(cfg, run_id, source, seed)),
            "output": str(result_dir / f"{direction}_seed_{seed}.json"),
        }
        for seed in seeds
        for direction, target, source in DIRECTION_EVALUATIONS
    ]
    counts = {
        "training": len(DOMAINS) * len(seeds),
        "evaluation": len(evaluation_plan),
    }
    payload = {
        "workflow": RUNNER_WORKFLOW,
        "run_id": run_id,
        "seeds": list(seeds),
        "counts": counts,
        "split": {"training": "train/valid", "evaluation": "test"},
        "models_per_seed": len(DOMAINS),
        "evaluation_directions": [item[0] for item in DIRECTION_EVALUATIONS],
        "config": str(config.relative_to(ROOT)) if config.is_relative_to(ROOT)
        else str(config),
        "config_profiles": {
            "training": "validation_candidate",
            "formal_evaluation": "formal_frozen",
        },
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "evaluation_plan": evaluation_plan,
        "augmentation": cfg.get("augmentation", {}),
        "resolved_runtime": runtime,
        "stage": "train_then_formal_test",
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    python = sys.executable
    verify_torch_interpreter(python)
    run_dir = ROOT / cfg["logging"]["log_dir"] / run_id
    manifest = run_dir / "manifest.json"
    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"时间戳目录已存在，请稍后重试: {run_dir}")
    if args.resume and not run_dir.is_dir():
        raise RuntimeError(f"--resume 目录不存在: {run_dir}")
    if result_dir.exists() and not args.resume:
        raise RuntimeError(f"测试结果目录已存在，拒绝覆盖: {result_dir}")
    if args.resume:
        if not manifest.is_file():
            raise RuntimeError(f"--resume 缺少 manifest: {manifest}")
        previous_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if previous_manifest.get("workflow") != RUNNER_WORKFLOW:
            raise RuntimeError(
                "该运行目录属于旧流程，不能原位续跑；请使用新 RUN_ID 以保留旧日志")
        if (previous_manifest.get("seeds") != list(seeds) or
                previous_manifest.get("config") != payload["config"] or
                previous_manifest.get("config_profiles") != payload["config_profiles"] or
                previous_manifest.get("config_sha256") != payload["config_sha256"]):
            raise RuntimeError("续跑配置与原 manifest 不一致；请使用新 RUN_ID")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        manifest.write_text(json.dumps(
            {**payload, "cache_audit": None, "completed": False},
            ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

    prepare_caches(python, config)
    manifest_payload["cache_audit"] = audit_caches(cfg)
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    training_jobs = []
    for seed in seeds:
        for dataset in DOMAINS:
            checkpoint = checkpoint_path(cfg, run_id, dataset, seed)
            if checkpoint.exists():
                if not args.resume:
                    raise RuntimeError(f"拒绝覆盖已有训练制品: {checkpoint}")
                continue
            training_jobs.append((
                f"train {dataset} seed={seed}",
                [python, "scripts/train.py", "--config", str(config),
                 "--dataset", dataset, "--seed", str(seed), "--run-id", run_id],
                 run_dir / "jobs" / f"train_{dataset}_seed_{seed}.log"))
    run_parallel(training_jobs, gpus)
    validation_record = collect_validation_summary(cfg, run_id, seeds)
    validation_path = run_dir / "validation_summary.json"
    validation_path.write_text(
        json.dumps(validation_record, ensure_ascii=False, indent=2),
        encoding="utf-8")
    manifest_payload["validation_summary"] = str(validation_path)

    test_jobs, planned_tests = build_formal_test_jobs(
        python, cfg, run_id, seeds, config, run_dir, result_dir,
        resume=bool(args.resume))
    manifest_payload["formal_test_plan"] = planned_tests
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    run_parallel(test_jobs, gpus)

    test_results = []
    for item in planned_tests:
        result = json.loads(Path(item["output"]).read_text(encoding="utf-8"))
        item["status"] = "completed"
        test_results.append({
            "direction": item["direction"],
            "seed": item["seed"],
            "dataset": item["dataset"],
            "source_dataset": item["source_dataset"],
            "test_samples": result["test_samples"],
            "metrics": metrics_from_prediction_file(Path(item["prediction_file"])),
            "output": item["output"],
            "prediction_file": item["prediction_file"],
        })
    test_summary = {
        "run_id": run_id,
        "seeds": list(seeds),
        "evaluation_count": len(test_results),
        "directions": [item[0] for item in DIRECTION_EVALUATIONS],
        "by_direction": summarize_test_metrics(test_results),
        "results": test_results,
    }
    summary_path = run_dir / "test_summary.json"
    summary_path.write_text(
        json.dumps(test_summary, ensure_ascii=False, indent=2),
        encoding="utf-8")
    manifest_payload["formal_test_plan"] = planned_tests
    manifest_payload["test_summary"] = str(summary_path)
    manifest_payload["completed"] = True
    manifest_payload["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"训练与完整测试完成: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
