from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TierSpec:
    name: str
    compute_tflops_s: float
    length_mi: int
    downlink_mb: float
    tx_delay_ms: float


@dataclass(frozen=True)
class LocalRegressionConfig:
    a0: float = 0.0
    a1: float = 20.0
    a2: float = 1.5
    a3: float = 8.0
    a4: float = 1.0
    alpha: float = 1.0
    beta: float = 0.8
    gamma: float = 0.2


@dataclass(frozen=True)
class SimConfig:
    num_nodes: int = 50
    num_potential_users: int = 100
    avg_candidate_nodes: int = 5
    slot_minutes: int = 1
    sim_days: int = 3
    bandwidth_mbps: float = 100.0
    sla_ms: float = 50.0
    node_cost_cny_per_hour: float = 5.35
    lambda_per_hour: float = 8.0
    tier_l_ratio: float = 0.6
    tier_h_ratio: float = 0.4
    session_hours_range: Tuple[float, float] = (1.0, 3.0)
    gpu_tflops: float = 31.2
    decision_timeout_ms: int = 50
    python_policy_url: str = "http://127.0.0.1:8000/act"
    local_regression: LocalRegressionConfig = LocalRegressionConfig()

    @property
    def node_cost_cny_per_minute(self) -> float:
        return self.node_cost_cny_per_hour / 60.0

    @property
    def total_slots(self) -> int:
        return int(self.sim_days * 24 * 60 / self.slot_minutes)


TIER_L = TierSpec(
    name="L",
    compute_tflops_s=0.31758,
    length_mi=317_580,
    downlink_mb=0.0958,
    tx_delay_ms=0.958,
)

TIER_H = TierSpec(
    name="H",
    compute_tflops_s=0.59362,
    length_mi=593_620,
    downlink_mb=0.1916,
    tx_delay_ms=1.916,
)

TIERS = (TIER_L, TIER_H)
