"""分类头模块"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """分类头：两层 MLP + BN + GELU + Dropout（比单层线性头更强的判别能力）"""

    def __init__(self, embed_dim, num_classes, hidden_ratio=2, dropout=0.3):
        super().__init__()
        hidden_dim = embed_dim * hidden_ratio
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.fc(x)
