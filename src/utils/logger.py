"""
实验日志记录模块
支持记录训练损失、运行时间、准确率、精确率、召回率、F1等关键指标
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """实验日志记录器，支持控制台输出和文件记录"""

    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.json"

        self.terminal = sys.stdout
        self.log = open(self.log_file, "w", encoding="utf-8")

        self.metrics_history = {
            "experiment_name": experiment_name,
            "start_time": timestamp,
            "config": {},
            "epochs": [],
            "best_metrics": {},
        }

        self.epoch_start_time = None
        self.total_start_time = time.time()

    def write(self, message: str):
        """同时输出到控制台和日志文件"""
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()

    def log_config(self, config: dict):
        """记录实验配置"""
        self.metrics_history["config"] = config
        self.write(f"\n{'='*60}\n")
        self.write(f"实验配置:\n")
        for key, value in config.items():
            self.write(f"  {key}: {value}\n")
        self.write(f"{'='*60}\n\n")
        self._save_metrics()

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.epoch_start_time = time.time()
        self.write(f"\n--- Epoch {epoch}/{total_epochs} ---\n")

    def log_train_loss(self, epoch: int, loss: float, lr: float):
        """记录训练损失和学习率"""
        self.write(f"  Train Loss: {loss:.4f} | LR: {lr:.6f}\n")

    def log_validation(self, epoch: int, metrics: dict):
        """记录验证指标"""
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_elapsed = time.time() - self.total_start_time

        epoch_record = {
            "epoch": epoch,
            "elapsed_time": round(elapsed, 2),
            "total_time": round(total_elapsed, 2),
            "metrics": metrics,
        }

        self.metrics_history["epochs"].append(epoch_record)

        self.write(f"  Validation Metrics:\n")
        for name, value in metrics.items():
            self.write(f"    {name}: {value:.4f}\n")
        self.write(f"  Epoch Time: {elapsed:.1f}s | Total Time: {total_elapsed:.1f}s\n")

        self._save_metrics()

    def log_best_model(self, epoch: int, metric_name: str, value: float):
        """记录最佳模型信息"""
        self.metrics_history["best_metrics"] = {
            "epoch": epoch,
            "metric_name": metric_name,
            "value": value,
        }
        self.write(f"  ★ New Best Model! {metric_name}: {value:.4f} at epoch {epoch}\n")
        self._save_metrics()

    def log_summary(self):
        """记录实验总结"""
        total_time = time.time() - self.total_start_time
        self.metrics_history["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_history["total_training_time"] = round(total_time, 2)

        self.write(f"\n{'='*60}\n")
        self.write(f"实验总结:\n")
        self.write(f"  总训练时间: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        if self.metrics_history["best_metrics"]:
            bm = self.metrics_history["best_metrics"]
            self.write(f"  最佳模型: Epoch {bm['epoch']}, {bm['metric_name']}: {bm['value']:.4f}\n")
        self.write(f"  日志文件: {self.log_file}\n")
        self.write(f"  指标文件: {self.metrics_file}\n")
        self.write(f"{'='*60}\n")
        self._save_metrics()

    def _save_metrics(self):
        """保存指标到JSON文件"""
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

    def close(self):
        """关闭日志文件"""
        self.log_summary()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False