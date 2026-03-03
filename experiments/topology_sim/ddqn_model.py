"""DDQN 网络结构定义（Dueling 架构）。

输出每个动作的 Q 值，供训练与推理共用。
"""

from __future__ import annotations

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, head_dim: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, act_dim),
        )
        self.val_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        adv = self.adv_head(h)
        val = self.val_head(h)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q
