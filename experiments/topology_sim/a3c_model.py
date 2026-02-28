from __future__ import annotations

import torch
from torch import nn


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value
