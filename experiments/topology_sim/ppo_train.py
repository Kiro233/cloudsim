from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from .config import SimConfig, TIER_H, TIER_L


@dataclass
class RewardWeights:
    # Reward rebalance: stronger acceptance incentive, milder reject/SLA penalties.
    w_a: float = 3.0
    w_v: float = 1.5
    w_r: float = 1.0
    w_c: float = 0.2


class TopologyDispatchEnv(gym.Env[np.ndarray, int]):
    """Single-step dispatch environment for masked PPO training.

    Observation format is strictly aligned with `ppo_infer.PpoPolicy._build_obs`:
    [node0(nL,nH,remaining,enabled), ..., nodeN(...), tier_val, candidate_mask[0..N-1]]

    Action space:
      0..num_nodes-1 => assign node
      num_nodes       => reject
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig, reward_weights: RewardWeights | None = None, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.reward_weights = reward_weights or RewardWeights()

        self.num_nodes = cfg.num_nodes
        self.reject_action = self.num_nodes

        obs_dim = self.num_nodes * 4 + 1 + self.num_nodes
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes + 1)

        self._obs: np.ndarray | None = None
        self._valid_action_mask: np.ndarray | None = None
        self._ctx: Dict[str, Any] = {}

    def _sample_node_state(self) -> Dict[str, Any]:
        # Sample a feasible pair under nL/3 + nH/2 <= 1.
        feasible_pairs: List[Tuple[int, int]] = []
        for n_l in range(0, 4):
            for n_h in range(0, 3):
                if (n_l / 3.0 + n_h / 2.0) <= 1.0 + 1e-9:
                    feasible_pairs.append((n_l, n_h))

        n_l, n_h = self.rng.choice(feasible_pairs)
        rem = 1.0 - (n_l / 3.0 + n_h / 2.0)
        enabled = 1.0 if (n_l + n_h) > 0 else 0.0
        return {"nL": n_l, "nH": n_h, "remaining_capacity": rem, "enabled": enabled}

    def _sample_request(self) -> Dict[str, Any]:
        tier = "L" if self.rng.random() < self.cfg.tier_l_ratio else "H"
        k = min(self.cfg.avg_candidate_nodes, self.num_nodes)
        candidates = set(self.rng.sample(range(self.num_nodes), k=k))
        candidate_mask = np.zeros(self.num_nodes, dtype=np.int32)
        for i in candidates:
            candidate_mask[i] = 1
        return {"tier": tier, "candidate_mask": candidate_mask}

    def _compute_valid_action_mask(self, node_states: List[Dict[str, Any]], req: Dict[str, Any]) -> np.ndarray:
        tier = req["tier"]
        step = 1.0 / 3.0 if tier == "L" else 1.0 / 2.0
        valid = np.zeros(self.num_nodes + 1, dtype=np.int8)

        feasible_count = 0
        for i in range(self.num_nodes):
            candidate = req["candidate_mask"][i] == 1
            feasible = (node_states[i]["remaining_capacity"] - step) >= -1e-9
            is_valid = candidate and feasible
            valid[i] = 1 if is_valid else 0
            if is_valid:
                feasible_count += 1

        # During training, only expose `reject` when no feasible assignment exists.
        # This prevents policy collapse to always-reject solutions.
        valid[self.reject_action] = 1 if feasible_count == 0 else 0
        return valid

    def _build_obs(self, node_states: List[Dict[str, Any]], req: Dict[str, Any]) -> np.ndarray:
        tier_val = 0.0 if req["tier"] == "L" else 1.0
        obs: List[float] = []

        for node in node_states:
            obs.extend([
                float(node["nL"]),
                float(node["nH"]),
                float(node["remaining_capacity"]),
                float(node["enabled"]),
            ])

        obs.append(tier_val)
        obs.extend([float(x) for x in req["candidate_mask"].tolist()])
        return np.asarray(obs, dtype=np.float32)

    def _compute_reward(self, action: int) -> float:
        w = self.reward_weights
        valid = self._valid_action_mask
        if valid is None:
            return 0.0

        # invalid -> treat as reject
        if action < 0 or action >= len(valid) or valid[action] != 1:
            a_t = 0.0
            r_t = 1.0
            v_t = 0.0
            c_t = 0.0
            return w.w_a * a_t - w.w_c * c_t - w.w_v * v_t - w.w_r * r_t

        if action == self.reject_action:
            a_t = 0.0
            r_t = 1.0
            v_t = 0.0
            c_t = 0.0
            return w.w_a * a_t - w.w_c * c_t - w.w_v * v_t - w.w_r * r_t

        # assigned
        node = self._ctx["node_states"][action]
        tier = self._ctx["request"]["tier"]
        req_work = TIER_L.compute_tflops_s if tier == "L" else TIER_H.compute_tflops_s
        tx_delay = TIER_L.tx_delay_ms if tier == "L" else TIER_H.tx_delay_ms

        total_work = node["nL"] * TIER_L.compute_tflops_s + node["nH"] * TIER_H.compute_tflops_s + req_work
        d_compute = (total_work / self.cfg.gpu_tflops) * 1000.0
        d_e2e = d_compute + tx_delay

        a_t = 1.0
        r_t = 0.0
        v_t = 1.0 if d_e2e > self.cfg.sla_ms else 0.0
        c_t = self.cfg.node_cost_cny_per_minute if node["enabled"] < 0.5 else 0.0

        return w.w_a * a_t - w.w_c * c_t - w.w_v * v_t - w.w_r * r_t

    def action_masks(self) -> np.ndarray:
        if self._valid_action_mask is None:
            return np.ones(self.num_nodes + 1, dtype=np.int8)
        return self._valid_action_mask

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng.seed(seed)

        node_states = [self._sample_node_state() for _ in range(self.num_nodes)]
        req = self._sample_request()

        self._valid_action_mask = self._compute_valid_action_mask(node_states, req)
        self._obs = self._build_obs(node_states, req)
        self._ctx = {"node_states": node_states, "request": req}

        return self._obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = self._compute_reward(int(action))
        obs, info = self.reset()
        # One dispatch decision per episode; this is required for stable eval callback.
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info


def build_env(cfg: SimConfig, seed: int) -> Monitor:
    env = TopologyDispatchEnv(cfg=cfg, seed=seed)
    return Monitor(env)


def train(args: argparse.Namespace) -> Path:
    cfg = SimConfig()
    env = build_env(cfg, seed=args.seed)
    eval_env = build_env(cfg, seed=args.seed + 1000)

    # Single-step dispatch is closer to a contextual bandit than a long-horizon MDP.
    # Use bandit-friendly PPO hyperparameters for better stability/generalization.
    ppo_hyperparams = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "target_kl": args.target_kl,
    }
    policy_kwargs = {
        "net_arch": [args.hidden_dim, args.hidden_dim],
    }

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        **ppo_hyperparams,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir),
        eval_freq=max(10_000, args.total_timesteps // 20),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)

    model_path = out_dir / "ppo_topology_model"
    model.save(str(model_path))

    meta = {
        "model_type": "MaskablePPO",
        "model_path": str(model_path.with_suffix(".zip")),
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "sim_config": asdict(cfg),
        "ppo_hyperparams": ppo_hyperparams,
        "network": {"hidden_dim": args.hidden_dim},
    }
    (out_dir / "ppo_training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return model_path.with_suffix(".zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO for topology dispatch.")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="experiments/topology_sim/models")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--vf-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_zip = train(args)
    print(f"Saved PPO model: {model_zip}")


if __name__ == "__main__":
    main()
