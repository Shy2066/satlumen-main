"""
通用工具函数
供 scripts/ 和 src/ 共享使用
"""

import copy
import random
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _merge_config(base, overrides):
    """Recursively merge a profile into shared YAML settings."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path, profile=None):
    """Load a YAML config, optionally resolving one named profile."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    profiles = config.pop("profiles", None)
    if profiles is None:
        return config
    if profile is None:
        raise ValueError(f"Config {path} defines profiles; select one explicitly")
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile {profile!r} in {path}; available profiles: "
            f"{', '.join(sorted(profiles))}")
    return _merge_config(config, profiles[profile])


def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 固定所有可见 GPU，避免多卡/换卡时结果漂移
    cudnn.deterministic = True
    cudnn.benchmark = False
    # 关闭 TF32，保证卷积/矩阵乘使用 FP32 位级计算，跨驱动/跨机器更可复现
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker_init_fn(worker_id):
    """DataLoader worker 种子初始化，确保多进程数据加载的可复现性"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


from typing import Any


class Args:
    """将嵌套 dict 转为可点访问的参数对象"""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v if not isinstance(v, dict) else Args(v))

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"Args 对象没有属性 '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


def compute_metrics(y_true, y_pred, average="macro"):
    """计算分类指标（百分制，保留全精度）"""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred) * 100),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0) * 100),  # type: ignore
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0) * 100),  # type: ignore
        "F1-score": float(f1_score(y_true, y_pred, average=average, zero_division=0) * 100),  # type: ignore
    }
