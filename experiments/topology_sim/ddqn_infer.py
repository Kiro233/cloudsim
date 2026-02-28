from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch

from .ddqn_model import DuelingQNetwork


@dataclass
class DdqnInferenceResult:
    action_node_id: Optional[int]
    error: Optional[str] = None


class DdqnPolicy:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model: Optional[DuelingQNetwork] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_error: Optional[str] = None
        self.node_count: Optional[int] = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            payload = torch.load(self.model_path, map_location=self.device)
            obs_dim = int(payload["obs_dim"])
            act_dim = int(payload["act_dim"])
            self.model = DuelingQNetwork(obs_dim, act_dim).to(self.device)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
            self.node_count = act_dim - 1
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = f"DDQN model load failed: {exc}"

    @staticmethod
    def _build_obs(req: dict[str, Any], node_count: int) -> np.ndarray:
        request_obj = req.get("request", {})
        tier = request_obj.get("tier", "L")
        tier_val = 0.0 if tier == "L" else 1.0

        obs: List[float] = []
        for node in req.get("node_states", []):
            obs.extend([
                float(node.get("nL", 0.0)),
                float(node.get("nH", 0.0)),
                float(node.get("remaining_capacity", 0.0)),
                1.0 if bool(node.get("enabled", False)) else 0.0,
            ])

        obs.append(tier_val)
        candidate_mask = request_obj.get("candidate_mask", [0] * node_count)
        for i in range(node_count):
            obs.append(float(candidate_mask[i] if i < len(candidate_mask) else 0))

        return np.asarray(obs, dtype=np.float32)

    def predict(self, req: dict[str, Any]) -> DdqnInferenceResult:
        if self.model is None:
            return DdqnInferenceResult(action_node_id=None, error=self.load_error or "DDQN model not loaded")

        node_states = req.get("node_states", [])
        node_count = len(node_states)
        valid_mask = req.get("valid_action_mask", [])
        if len(valid_mask) < node_count + 1:
            return DdqnInferenceResult(action_node_id=None, error="invalid valid_action_mask")

        if self.node_count is not None and node_count != self.node_count:
            return DdqnInferenceResult(action_node_id=None, error="node_count mismatch with trained model")

        try:
            obs = self._build_obs(req, node_count)
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(x).squeeze(0)

            mask = torch.tensor(valid_mask[: node_count + 1], dtype=torch.bool, device=self.device)
            q_masked = q.clone()
            q_masked[~mask] = -1e9
            action = int(torch.argmax(q_masked).item())

            if action >= node_count:
                return DdqnInferenceResult(action_node_id=None)
            return DdqnInferenceResult(action_node_id=action)
        except Exception as exc:
            return DdqnInferenceResult(action_node_id=None, error=f"DDQN inference failed: {exc}")
