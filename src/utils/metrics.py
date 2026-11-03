"""
工具函数模块
包含位置编码和学习率调度
"""

import math
import numpy as np


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """生成1D正弦余弦位置编码"""
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 输出维度
    pos: 位置列表 (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def adjust_learning_rate(optimizer, epoch, cfg):
    """余弦退火学习率调度（带warmup）"""
    # 支持字典和对象两种形式
    warmup_epochs = cfg["warmup_epochs"] if isinstance(cfg, dict) else cfg.warmup_epochs
    lr_init = cfg["lr"] if isinstance(cfg, dict) else cfg.lr
    min_lr = cfg["min_lr"] if isinstance(cfg, dict) else cfg.min_lr
    epochs = cfg["epochs"] if isinstance(cfg, dict) else cfg.epochs

    if epochs <= warmup_epochs:
        lr = lr_init
    elif epoch < warmup_epochs:
        lr = lr_init * (epoch + 1) / warmup_epochs
    else:
        lr = min_lr + (lr_init - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr