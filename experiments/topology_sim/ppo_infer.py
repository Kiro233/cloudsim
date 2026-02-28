from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class PpoInferenceResult:
    action_node_id: Optional[int]
    error: Optional[str] = None


class PpoPolicy:
    """Thin inference wrapper for MaskablePPO models.

    Expected model type: sb3_contrib.ppo_mask.MaskablePPO (or compatible predict API).
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.load_error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from sb3_contrib import MaskablePPO  # type: ignore

            self.model = MaskablePPO.load(self.model_path)
            self.load_error = None
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.model = None
            self.load_error = f"PPO model load failed: {exc}"

    @staticmethod
    def _build_obs(req: dict[str, Any], node_count: int) -> np.ndarray:
        request_obj = req.get("request", {})
        tier = request_obj.get("tier", "L")
        tier_val = 0.0 if tier == "L" else 1.0

        obs: List[float] = []
        for node in req.get("node_states", []):
            n_l = float(node.get("nL", 0.0))
            n_h = float(node.get("nH", 0.0))
            rem = float(node.get("remaining_capacity", 0.0))
            enabled = 1.0 if bool(node.get("enabled", False)) else 0.0
            obs.extend([n_l, n_h, rem, enabled])

        obs.append(tier_val)

        candidate_mask = request_obj.get("candidate_mask", [0] * node_count)
        for i in range(node_count):
            v = candidate_mask[i] if i < len(candidate_mask) else 0
            obs.append(float(v))

        return np.asarray(obs, dtype=np.float32)

    def predict(self, req: dict[str, Any]) -> PpoInferenceResult:
        node_states = req.get("node_states", [])
        node_count = len(node_states)

        if self.model is None:
            return PpoInferenceResult(action_node_id=None, error=self.load_error or "PPO model not loaded")

        valid_mask = req.get("valid_action_mask", [])
        if len(valid_mask) < node_count + 1:
            return PpoInferenceResult(action_node_id=None, error="invalid valid_action_mask")

        try:
            obs = self._build_obs(req, node_count)
            action_masks = np.asarray(valid_mask[: node_count + 1], dtype=np.int8)
            action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
            action_int = int(action)
            if action_int >= node_count:
                return PpoInferenceResult(action_node_id=None)
            return PpoInferenceResult(action_node_id=action_int)
        except Exception as exc:  # pragma: no cover - inference error path
            return PpoInferenceResult(action_node_id=None, error=f"PPO inference failed: {exc}")
