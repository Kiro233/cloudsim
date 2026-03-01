from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SimConfig, TIER_H, TIER_L
from .ddqn_model import DuelingQNetwork
from .ddqn_replay import PrioritizedReplayBuffer, Transition


@dataclass
class DdqnConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 256
    target_update_interval: int = 2000
    train_freq: int = 4
    gradient_steps: int = 1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_portion: float = 0.2
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_final: float = 1.0


class TopologySampleEnv:
    def __init__(self, cfg: SimConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.num_nodes = cfg.num_nodes
        self.act_dim = self.num_nodes + 1

    def _sample_node_state(self) -> dict:
        feasible = []
        for n_l in range(0, 4):
            for n_h in range(0, 3):
                if (n_l / 3.0 + n_h / 2.0) <= 1.0 + 1e-9:
                    feasible.append((n_l, n_h))
        n_l, n_h = self.rng.choice(feasible)
        rem = 1.0 - (n_l / 3.0 + n_h / 2.0)
        enabled = 1.0 if (n_l + n_h) > 0 else 0.0
        return {"nL": n_l, "nH": n_h, "remaining_capacity": rem, "enabled": enabled}

    def reset(self) -> tuple[np.ndarray, np.ndarray, dict]:
        nodes = [self._sample_node_state() for _ in range(self.num_nodes)]
        tier = "L" if self.rng.random() < self.cfg.tier_l_ratio else "H"
        candidates = set(self.rng.sample(range(self.num_nodes), k=min(self.cfg.avg_candidate_nodes, self.num_nodes)))
        candidate_mask = np.zeros(self.num_nodes, dtype=np.int8)
        for i in candidates:
            candidate_mask[i] = 1

        tier_val = 0.0 if tier == "L" else 1.0
        obs = []
        for n in nodes:
            obs.extend([float(n["nL"]), float(n["nH"]), float(n["remaining_capacity"]), float(n["enabled"])])
        obs.append(tier_val)
        obs.extend([float(x) for x in candidate_mask.tolist()])
        obs_arr = np.asarray(obs, dtype=np.float32)

        step = 1.0 / 3.0 if tier == "L" else 1.0 / 2.0
        valid = np.zeros(self.act_dim, dtype=np.int8)
        feasible_count = 0
        for i in range(self.num_nodes):
            is_valid = candidate_mask[i] == 1 and (nodes[i]["remaining_capacity"] - step) >= -1e-9
            valid[i] = 1 if is_valid else 0
            if is_valid:
                feasible_count += 1
        # During training, only expose reject when no feasible assignment exists.
        valid[self.num_nodes] = 1 if feasible_count == 0 else 0

        ctx = {"nodes": nodes, "tier": tier}
        return obs_arr, valid, ctx

    def reward(self, action: int, valid_mask: np.ndarray, ctx: dict) -> float:
        # Reward rebalance: stronger acceptance incentive, milder reject/SLA penalties.
        w_a, w_v, w_r, w_c = 3.0, 1.5, 1.0, 0.2
        if action < 0 or action >= len(valid_mask) or valid_mask[action] != 1 or action == self.num_nodes:
            return w_a * 0.0 - w_c * 0.0 - w_v * 0.0 - w_r * 1.0

        node = ctx["nodes"][action]
        tier = ctx["tier"]
        req_work = TIER_L.compute_tflops_s if tier == "L" else TIER_H.compute_tflops_s
        tx_delay = TIER_L.tx_delay_ms if tier == "L" else TIER_H.tx_delay_ms
        total_work = node["nL"] * TIER_L.compute_tflops_s + node["nH"] * TIER_H.compute_tflops_s + req_work
        d_compute = (total_work / self.cfg.gpu_tflops) * 1000.0
        v_t = 1.0 if (d_compute + tx_delay) > self.cfg.sla_ms else 0.0
        c_t = self.cfg.node_cost_cny_per_minute if node["enabled"] < 0.5 else 0.0
        return w_a * 1.0 - w_c * c_t - w_v * v_t - w_r * 0.0


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.full_like(q_values, -1e9)
    masked = torch.where(mask > 0, q_values, neg_inf)
    return masked.argmax(dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDQN (dueling+double+PER) for topology dispatch.")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="experiments/topology_sim/models")
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = SimConfig()
    ddqn_cfg = DdqnConfig()
    env = TopologySampleEnv(cfg, seed=args.seed)

    obs_dim = cfg.num_nodes * 4 + 1 + cfg.num_nodes
    act_dim = cfg.num_nodes + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online = DuelingQNetwork(obs_dim, act_dim).to(device)
    target = DuelingQNetwork(obs_dim, act_dim).to(device)
    target.load_state_dict(online.state_dict())

    optimizer = torch.optim.Adam(online.parameters(), lr=ddqn_cfg.lr)
    replay = PrioritizedReplayBuffer(capacity=ddqn_cfg.buffer_size, alpha=ddqn_cfg.per_alpha)

    eps_decay_steps = max(1, int(ddqn_cfg.epsilon_decay_portion * args.total_steps))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "ddqn_training_log.csv"

    obs, mask, ctx = env.reset()
    last_loss = 0.0

    with log_file.open("w", encoding="utf-8", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["step", "algorithm", "reward", "loss", "epsilon", "beta", "replay_size"])

        for step in range(1, args.total_steps + 1):
            frac = min(1.0, step / eps_decay_steps)
            epsilon = ddqn_cfg.epsilon_start + frac * (ddqn_cfg.epsilon_final - ddqn_cfg.epsilon_start)
            beta = ddqn_cfg.per_beta_start + (step / args.total_steps) * (ddqn_cfg.per_beta_final - ddqn_cfg.per_beta_start)

            if random.random() < epsilon:
                valid_actions = np.where(mask == 1)[0]
                action = int(random.choice(valid_actions.tolist()))
            else:
                with torch.no_grad():
                    q = online(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                    m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(masked_argmax(q, m).item())

            reward = env.reward(action, mask, ctx)
            next_obs, next_mask, next_ctx = env.reset()
            done = 0.0

            replay.add(
                Transition(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=done,
                    next_action_mask=next_mask,
                )
            )

            obs, mask, ctx = next_obs, next_mask, next_ctx

            if len(replay) >= ddqn_cfg.batch_size and step % ddqn_cfg.train_freq == 0:
                for _ in range(ddqn_cfg.gradient_steps):
                    batch, idxs, is_weights = replay.sample(ddqn_cfg.batch_size, beta=beta)

                    b_obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device)
                    b_act = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
                    b_rew = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
                    b_next = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device)
                    b_done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
                    b_nmask = torch.tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.float32, device=device)
                    b_w = torch.tensor(is_weights, dtype=torch.float32, device=device).unsqueeze(1)

                    q_pred = online(b_obs).gather(1, b_act)

                    with torch.no_grad():
                        next_q_online = online(b_next)
                        next_act = masked_argmax(next_q_online, b_nmask).unsqueeze(1)
                        next_q_target = target(b_next).gather(1, next_act)
                        y = b_rew + (1.0 - b_done) * ddqn_cfg.gamma * next_q_target

                    td = q_pred - y
                    loss = (b_w * F.smooth_l1_loss(q_pred, y, reduction="none")).mean()
                    last_loss = float(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    optimizer.step()

                    replay.update_priorities(idxs, td.detach().abs().cpu().numpy().reshape(-1))

            if step % ddqn_cfg.target_update_interval == 0:
                target.load_state_dict(online.state_dict())

            if step % args.log_interval == 0 or step == args.total_steps:
                writer.writerow([step, "ddqn", f"{reward:.6f}", f"{last_loss:.6f}", f"{epsilon:.6f}", f"{beta:.6f}", len(replay)])

    model_file = out_dir / "ddqn_topology_model.pt"
    torch.save(
        {
            "state_dict": online.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "algorithm": "DDQN",
            "seed": args.seed,
            "sim_config": asdict(cfg),
            "ddqn_config": asdict(ddqn_cfg),
        },
        model_file,
    )

    meta_file = out_dir / "ddqn_training_meta.json"
    meta_file.write_text(
        json.dumps(
            {
                "model_path": str(model_file),
                "algorithm": "DDQN",
                "seed": args.seed,
                "total_steps": args.total_steps,
                "log_file": str(log_file),
                "log_schema": ["step", "algorithm", "reward", "loss", "epsilon", "beta", "replay_size"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved DDQN model: {model_file}")
    print(f"Saved DDQN log: {log_file}")


if __name__ == "__main__":
    main()
