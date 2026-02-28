from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float
    next_action_mask: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.data: List[Transition] = []
        self.priorities: List[float] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, tr: Transition) -> None:
        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.data) < self.capacity:
            self.data.append(tr)
            self.priorities.append(max_prio)
        else:
            self.data[self.pos] = tr
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        prios = np.asarray(self.priorities, dtype=np.float64)
        probs = prios ** self.alpha
        probs = probs / probs.sum()

        idxs = np.random.choice(len(self.data), size=batch_size, p=probs)
        samples = [self.data[i] for i in idxs]

        weights = (len(self.data) * probs[idxs]) ** (-beta)
        weights = weights / weights.max()
        return samples, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        for i, e in zip(idxs.tolist(), td_errors.tolist()):
            self.priorities[i] = float(abs(e) + 1e-6)
