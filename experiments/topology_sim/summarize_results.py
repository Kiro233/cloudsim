from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


METRICS = [
    "accept_rate",
    "sla_rate",
    "avg_delay_ms",
    "p95_delay_ms",
    "avg_control_delay_ms",
    "p95_control_delay_ms",
    "cost_per_accepted_cny",
    "fallback_timeout",
    "fallback_service_unavailable",
    "fallback_illegal_action",
    "policy_reject",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Java experiment CSV results by lambda and scheduler.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="experiments/topology_sim/results_java.csv",
        help="Path to per-seed Java results CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="experiments/topology_sim/results_java_summary.csv",
        help="Path to write grouped summary CSV.",
    )
    return parser.parse_args()


def _to_float(row: Dict[str, str], key: str) -> float:
    v = row.get(key, "0").strip()
    return float(v) if v else 0.0


def summarize(input_csv: Path, output_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    groups: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lam = row.get("lambda_per_hour", "")
            scheduler = row.get("scheduler", "")
            key = (lam, scheduler)
            for metric in METRICS:
                groups[key][metric].append(_to_float(row, metric))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["lambda_per_hour", "scheduler", "n_seeds"]
    for metric in METRICS:
        header.extend([f"{metric}_mean", f"{metric}_std"])

    rows: List[List[str]] = [header]

    for (lam, scheduler) in sorted(groups.keys(), key=lambda x: (float(x[0]), x[1])):
        metric_values = groups[(lam, scheduler)]
        n = len(next(iter(metric_values.values()))) if metric_values else 0
        row: List[str] = [lam, scheduler, str(n)]
        for metric in METRICS:
            vals = metric_values.get(metric, [])
            m = mean(vals) if vals else 0.0
            s = pstdev(vals) if len(vals) > 1 else 0.0
            row.extend([f"{m:.6f}", f"{s:.6f}"])
        rows.append(row)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote summary CSV: {output_csv}")


if __name__ == "__main__":
    args = parse_args()
    summarize(Path(args.input_csv), Path(args.output_csv))
