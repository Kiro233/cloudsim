from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .a3c_model import ActorCriticNet
from .config import SimConfig, TIER_H, TIER_L


@dataclass
class A3cConfig:
    num_workers: int = 8
    lr: float = 7e-4
    gamma: float = 0.99
    t_max: int = 20
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 40.0
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-5


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
        for i in range(self.num_nodes):
            valid[i] = 1 if candidate_mask[i] == 1 and (nodes[i]["remaining_capacity"] - step) >= -1e-9 else 0
        valid[self.num_nodes] = 1

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


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(mask <= 0, -1e9)
    return torch.softmax(masked_logits, dim=-1)


def worker_loop(
    worker_id: int,
    global_model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    global_step: list[int],
    step_lock: threading.Lock,
    update_lock: threading.Lock,
    total_steps: int,
    cfg: SimConfig,
    a3c_cfg: A3cConfig,
    seed: int,
    stat_lock: threading.Lock,
    stats: dict[str, float],
    worker_device: torch.device,
) -> None:
    device = worker_device
    env = TopologySampleEnv(cfg, seed=seed + worker_id * 97)
    local_model = ActorCriticNet(cfg.num_nodes * 4 + 1 + cfg.num_nodes, cfg.num_nodes + 1).to(device)

    obs, mask, ctx = env.reset()
    x_buf = torch.empty((1, obs.shape[0]), dtype=torch.float32, device=device)
    m_buf = torch.empty((1, mask.shape[0]), dtype=torch.float32, device=device)

    while True:
        with step_lock:
            if global_step[0] >= total_steps:
                break

        trajectories = []
        with torch.no_grad():
            local_model.load_state_dict(global_model.state_dict())

        for _ in range(a3c_cfg.t_max):
            x_buf[0].copy_(torch.from_numpy(obs).to(device=device, dtype=torch.float32))
            m_buf[0].copy_(torch.from_numpy(mask).to(device=device, dtype=torch.float32))

            with torch.no_grad():
                logits, value = local_model(x_buf)

            probs = masked_softmax(logits, m_buf)
            dist = torch.distributions.Categorical(probs=probs)
            action = int(dist.sample().item())

            reward = env.reward(action, mask, ctx)
            next_obs, next_mask, next_ctx = env.reset()

            trajectories.append((obs, mask, action, reward, value.squeeze(0)))
            obs, mask, ctx = next_obs, next_mask, next_ctx

            with stat_lock:
                stats["latest_reward"] = float(reward)

            with step_lock:
                global_step[0] += 1
                if global_step[0] >= total_steps:
                    break

        if not trajectories:
            continue

        returns = []
        ret = 0.0
        for _, _, _, r, _ in reversed(trajectories):
            ret = r + a3c_cfg.gamma * ret
            returns.append(ret)
        returns.reverse()

        b_obs = torch.tensor(np.stack([t[0] for t in trajectories]), dtype=torch.float32, device=device)
        b_mask = torch.tensor(np.stack([t[1] for t in trajectories]), dtype=torch.float32, device=device)
        b_act = torch.tensor([t[2] for t in trajectories], dtype=torch.int64, device=device)
        b_ret = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)

        logits, values = local_model(b_obs)
        probs = masked_softmax(logits, b_mask)
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(b_act).unsqueeze(1)
        entropy = dist.entropy().mean()

        advantage = b_ret - values
        policy_loss = -(log_prob * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + a3c_cfg.value_coef * value_loss - a3c_cfg.entropy_coef * entropy

        local_model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), a3c_cfg.max_grad_norm)

        with stat_lock:
            stats["latest_loss"] = float(loss.item())

        with update_lock:
            optimizer.zero_grad(set_to_none=False)
            for gp, lp in zip(global_model.parameters(), local_model.parameters()):
                if lp.grad is None:
                    if gp.grad is None:
                        gp.grad = torch.zeros_like(gp)
                    else:
                        gp.grad.zero_()
                    continue
                if gp.grad is None:
                    gp.grad = lp.grad.detach().to(gp.device).clone()
                else:
                    gp.grad.copy_(lp.grad.detach().to(gp.device))
            optimizer.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A3C for topology dispatch.")
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
    a3c_cfg = A3cConfig()
    obs_dim = cfg.num_nodes * 4 + 1 + cfg.num_nodes
    act_dim = cfg.num_nodes + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = ActorCriticNet(obs_dim, act_dim).to(device)
    optimizer = torch.optim.RMSprop(
        global_model.parameters(),
        lr=a3c_cfg.lr,
        alpha=a3c_cfg.rmsprop_alpha,
        eps=a3c_cfg.rmsprop_eps,
    )

    global_step = [0]
    step_lock = threading.Lock()
    update_lock = threading.Lock()
    stat_lock = threading.Lock()
    stats = {"latest_reward": 0.0, "latest_loss": 0.0}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "a3c_training_log.csv"

    workers = []
    for wid in range(a3c_cfg.num_workers):
        t = threading.Thread(
            target=worker_loop,
            args=(
                wid,
                global_model,
                optimizer,
                global_step,
                step_lock,
                update_lock,
                args.total_steps,
                cfg,
                a3c_cfg,
                args.seed,
                stat_lock,
                stats,
                device,
            ),
            daemon=True,
        )
        t.start()
        workers.append(t)

    next_log_step = args.log_interval
    with log_file.open("w", encoding="utf-8", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["step", "algorithm", "reward", "loss", "epsilon", "beta", "replay_size"])
        f_log.flush()

        while True:
            with step_lock:
                step_val = global_step[0]
            if step_val >= next_log_step or step_val >= args.total_steps:
                with stat_lock:
                    reward = stats["latest_reward"]
                    loss = stats["latest_loss"]
                writer.writerow([step_val, "a3c", f"{reward:.6f}", f"{loss:.6f}", "", "", ""])
                f_log.flush()
                next_log_step += args.log_interval

            if step_val >= args.total_steps:
                break

            if not any(t.is_alive() for t in workers):
                raise RuntimeError(
                    f"All A3C workers stopped early at step={step_val}. Check thread traceback above."
                )

            time.sleep(0.05)

    for t in workers:
        t.join()

    model_file = out_dir / "a3c_topology_model.pt"
    torch.save(
        {
            "state_dict": global_model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "algorithm": "A3C",
            "seed": args.seed,
            "sim_config": asdict(cfg),
            "a3c_config": asdict(a3c_cfg),
        },
        model_file,
    )

    meta_file = out_dir / "a3c_training_meta.json"
    meta_file.write_text(
        json.dumps(
            {
                "model_path": str(model_file),
                "algorithm": "A3C",
                "seed": args.seed,
                "total_steps": args.total_steps,
                "log_file": str(log_file),
                "log_schema": ["step", "algorithm", "reward", "loss", "epsilon", "beta", "replay_size"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved A3C model: {model_file}")
    print(f"Saved A3C log: {log_file}")


if __name__ == "__main__":
    main()
