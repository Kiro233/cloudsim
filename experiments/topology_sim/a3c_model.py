"""A3C 模型结构定义。

包含共享特征提取骨干网络与策略/价值双头输出，
供训练与在线推理共用。
"""

from __future__ import annotations

import torch
from torch import nn


class ActorCriticNet(nn.Module):
    """A3C 的 Actor-Critic 网络。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        # 共享特征提取层：先抽取状态表示，再交给策略头与价值头。
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value
