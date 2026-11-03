"""Directional-count SatLumen classifier and its checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from .heads.classification import ClassificationHead
from src.utils.metrics import get_1d_sincos_pos_embed


def _sequence_mask(lengths, capacity):
    positions = torch.arange(capacity, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


class MaskedGatedMeanMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, features, mask):
        mask = mask.bool()
        weights = mask.unsqueeze(-1).to(features.dtype)
        count = weights.sum(dim=1).clamp(min=1.0)
        mean = (features * weights).sum(dim=1) / count
        maximum = features.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        valid = mask.any(dim=1, keepdim=True)
        maximum = torch.where(valid, maximum, torch.zeros_like(maximum))
        mean = torch.where(valid, mean, torch.zeros_like(mean))
        gate = torch.sigmoid(self.gate(mean))
        pooled = gate * maximum + (1.0 - gate) * mean
        return torch.where(valid, pooled, torch.zeros_like(pooled))


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size, padding=padding, dilation=dilation,
            groups=dim, bias=False)
        self.norm = nn.GroupNorm(1, dim)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(self.pointwise(x)) + residual
        if mask is not None:
            x = x * mask.unsqueeze(1).to(x.dtype)
        return x


class MultiScaleCountStem(nn.Module):
    def __init__(self, dim=256, dilations=(1, 2, 4), dropout=0.1,
                 downsample_ratio=6):
        super().__init__()
        if tuple(dilations) != (1, 2, 4):
            raise ValueError("directional-count stem dilations must be [1, 2, 4]")
        self.downsample_ratio = int(downsample_ratio)
        self.input_projection = nn.Conv1d(2, dim, 1)
        self.blocks = nn.ModuleList([
            DepthwiseResidualBlock(dim, 5, dilation, dropout)
            for dilation in dilations
        ])
        self.downsample = nn.AvgPool1d(
            kernel_size=self.downsample_ratio, stride=self.downsample_ratio)

    def forward(self, counts, count_idx):
        if counts.ndim != 3 or counts.shape[1] != 2:
            raise ValueError(
                f"counts must have shape (B, 2, L), got {tuple(counts.shape)}")
        lengths = (count_idx.long() + 1).clamp(min=0, max=counts.shape[-1])
        input_mask = _sequence_mask(lengths, counts.shape[-1])
        x = self.input_projection(counts) * input_mask.unsqueeze(1).to(counts.dtype)
        for block in self.blocks:
            x = block(x, input_mask)
        x = self.downsample(x)
        output_lengths = torch.div(
            lengths + self.downsample_ratio - 1, self.downsample_ratio,
            rounding_mode="floor")
        output_lengths = output_lengths.clamp(max=x.shape[-1])
        output_mask = _sequence_mask(output_lengths, x.shape[-1])
        x = x * output_mask.unsqueeze(1).to(x.dtype)
        return x.transpose(1, 2), output_mask


class SatLumenConvClassifier(nn.Module):
    """Directional-count stem, six temporal blocks and masked gated readout."""

    def __init__(self, num_classes, count_dim=256, temporal_dilations=(),
                 dropout=0.1, max_matrix_len=2700):
        super().__init__()
        if max_matrix_len != 2700:
            raise ValueError("SatLumenConvClassifier requires max_matrix_len=2700")
        temporal_dilations = tuple(int(value) for value in temporal_dilations)
        if any(value <= 0 for value in temporal_dilations):
            raise ValueError("temporal_dilations must contain positive integers")
        self.temporal_dilations = temporal_dilations
        self.count_stem = MultiScaleCountStem(
            count_dim, dilations=(1, 2, 4), dropout=dropout)
        count_tokens = max_matrix_len // self.count_stem.downsample_ratio
        position = get_1d_sincos_pos_embed(count_dim, count_tokens)
        self.register_buffer(
            "count_pos_embed", torch.from_numpy(position).float().unsqueeze(0),
            persistent=False)
        self.temporal_blocks = nn.ModuleList([
            DepthwiseResidualBlock(count_dim, 5, dilation, dropout)
            for dilation in temporal_dilations
        ])
        self.count_norm = nn.LayerNorm(count_dim)
        self.count_pool = MaskedGatedMeanMax(count_dim)
        self.pre_head_norm = nn.LayerNorm(count_dim)
        self.head = ClassificationHead(count_dim, num_classes)

    def forward(self, counts, count_idx):
        count_tokens, count_mask = self.count_stem(counts, count_idx)
        position = self.count_pos_embed[:, :count_tokens.shape[1]].to(
            count_tokens.dtype)
        x = count_tokens + position * count_mask.unsqueeze(-1).to(
            count_tokens.dtype)
        x = x.transpose(1, 2)
        for block in self.temporal_blocks:
            x = block(x, count_mask)
        x = x.transpose(1, 2)
        z_count = self.count_pool(self.count_norm(x), count_mask)
        return self.head(self.pre_head_norm(z_count))


def load_directional_count_checkpoint(model: nn.Module,
                                      checkpoint: str | Path | dict):
    """Load a directional-count state, mapping a compatible fusion norm name.

    This loader is intentionally strict: any Burst or alpha parameter rejects
    the checkpoint instead of silently dropping it.
    """
    if isinstance(checkpoint, (str, Path)):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        state = checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state-dict mapping")

    forbidden = ["burst_encoder.", "burst_projection.", "alpha_logit"]
    if isinstance(model, SatLumenConvClassifier):
        forbidden.extend(("mamba_blocks.", "mamba_droppaths."))
    forbidden_keys = [key for key in state if any(token in key for token in forbidden)]
    if forbidden_keys:
        raise ValueError(
            "拒绝不兼容架构参数的 checkpoint: " + ", ".join(forbidden_keys))
    mapped = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("fusion_norm."):
            key = "pre_head_norm." + key[len("fusion_norm."):]
        if key in mapped:
            raise ValueError(f"checkpoint 字段映射冲突: {key}")
        mapped[key] = value
    try:
        return model.load_state_dict(mapped, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint 不是兼容的 directional-count state") from exc
