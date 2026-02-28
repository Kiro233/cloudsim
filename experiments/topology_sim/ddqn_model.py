from __future__ import annotations

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        self.val_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        adv = self.adv_head(h)
        val = self.val_head(h)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q
