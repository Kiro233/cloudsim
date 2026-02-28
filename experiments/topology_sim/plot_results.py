from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


METRIC_PLOTS = [
    ("accept_rate_mean", "accept_rate_std", "接纳率", "比率"),
    ("sla_rate_mean", "sla_rate_std", "SLA满足率", "比率"),
    ("avg_delay_ms_mean", "avg_delay_ms_std", "平均端到端时延", "毫秒"),
    ("p95_delay_ms_mean", "p95_delay_ms_std", "P95端到端时延", "毫秒"),
    ("avg_control_delay_ms_mean", "avg_control_delay_ms_std", "平均控制时延", "毫秒"),
    ("p95_control_delay_ms_mean", "p95_control_delay_ms_std", "P95控制时延", "毫秒"),
    ("cost_per_accepted_cny_mean", "cost_per_accepted_cny_std", "单位接纳会话成本", "元"),
]

METRIC_LABELS = {
    "accept_rate_mean": ("接纳率", "比率"),
    "sla_rate_mean": ("SLA满足率", "比率"),
    "avg_delay_ms_mean": ("平均端到端时延", "毫秒"),
    "p95_delay_ms_mean": ("P95端到端时延", "毫秒"),
    "avg_control_delay_ms_mean": ("平均控制时延", "毫秒"),
    "p95_control_delay_ms_mean": ("P95控制时延", "毫秒"),
    "cost_per_accepted_cny_mean": ("单位接纳会话成本", "元"),
}

SCHEDULER_ORDER = ["best_fit", "random", "random_feasible", "ppo", "ddqn", "a3c", "python_policy"]
SCHEDULER_LABELS = {
    "best_fit": "Best-Fit（基线）",
    "random": "随机策略",
    "random_feasible": "随机可行策略",
    "ppo": "PPO",
    "ddqn": "DDQN",
    "a3c": "A3C",
    "python_policy": "Python策略",
}
SCHEDULER_COLORS = {
    "best_fit": "#1f77b4",
    "random": "#ff7f0e",
    "random_feasible": "#ff7f0e",
    "ppo": "#2ca02c",
    "ddqn": "#d62728",
    "a3c": "#9467bd",
    "python_policy": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metric comparison charts from summary CSV.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="experiments/topology_sim/results_java_summary.csv",
        help="Input summary CSV produced by summarize_results.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/topology_sim/plots",
        help="Directory to write PNG charts",
    )
    parser.add_argument(
        "--include-schedulers",
        type=str,
        default="best_fit,ppo,ddqn,a3c",
        help="Comma-separated schedulers to plot. Empty means all.",
    )
    parser.add_argument(
        "--ppo-eval-npz",
        type=str,
        default="experiments/topology_sim/models/evaluations.npz",
        help="PPO 评估曲线文件（由 MaskableEvalCallback 生成）",
    )
    parser.add_argument(
        "--bar-metric",
        type=str,
        default="accept_rate_mean",
        help="柱状图比较指标（summary 中 *_mean 列名）",
    )
    parser.add_argument(
        "--bar-lambda-mode",
        type=str,
        default="ppo_best",
        choices=["ppo_best", "fixed"],
        help="柱状图选择哪一个 λ：ppo_best=取 PPO 在该指标最优的 λ；fixed=用 --bar-lambda。",
    )
    parser.add_argument("--bar-lambda", type=float, default=8.0, help="当 --bar-lambda-mode=fixed 时使用。")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _read_summary(path: Path, include_schedulers: set[str] | None = None) -> Dict[str, List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheduler = row["scheduler"]
            if include_schedulers is not None and scheduler not in include_schedulers:
                continue
            grouped[scheduler].append(row)

    for scheduler in grouped:
        grouped[scheduler].sort(key=lambda r: float(r["lambda_per_hour"]))
    return grouped


def _plot_metric(grouped: Dict[str, List[dict]], metric_mean: str, metric_std: str, title: str, unit: str, out_path: Path, dpi: int) -> None:
    plt.figure(figsize=(8, 5))

    schedulers = [s for s in SCHEDULER_ORDER if s in grouped] + [s for s in grouped.keys() if s not in SCHEDULER_ORDER]
    for scheduler in schedulers:
        rows = grouped[scheduler]
        xs = [float(r["lambda_per_hour"]) for r in rows]
        ys = [float(r.get(metric_mean, "0") or 0.0) for r in rows]
        es = [float(r.get(metric_std, "0") or 0.0) for r in rows]

        plt.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=SCHEDULER_COLORS.get(scheduler, None),
            label=SCHEDULER_LABELS.get(scheduler, scheduler),
        )

    plt.title(title)
    plt.xlabel("到达率 λ（次/小时）")
    plt.ylabel(unit)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _plot_ppo_convergence(ppo_eval_npz: Path, output_dir: Path, dpi: int) -> None:
    if not ppo_eval_npz.exists():
        print(f"Skip PPO 收敛曲线：文件不存在 {ppo_eval_npz}")
        return

    d = np.load(ppo_eval_npz)
    if "timesteps" not in d or "results" not in d:
        print(f"Skip PPO 收敛曲线：缺少 timesteps/results 字段 {ppo_eval_npz}")
        return

    xs = d["timesteps"]
    ys = d["results"].mean(axis=1)
    es = d["results"].std(axis=1)

    out_path = output_dir / "ppo_convergence_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, color=SCHEDULER_COLORS["ppo"], linewidth=2.0, marker="o", label="PPO评估均值")
    plt.fill_between(xs, ys - es, ys + es, color=SCHEDULER_COLORS["ppo"], alpha=0.2, label="±1σ")
    plt.title("PPO训练收敛曲线")
    plt.xlabel("训练步数")
    plt.ylabel("评估回报")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Wrote plot: {out_path}")


def _best_lambda_for_ppo(grouped: Dict[str, List[dict]], metric_mean: str) -> float | None:
    ppo_rows = grouped.get("ppo", [])
    if not ppo_rows:
        return None

    best_row = max(ppo_rows, key=lambda r: float(r.get(metric_mean, "0") or 0.0))
    return float(best_row["lambda_per_hour"])


def _plot_best_case_bar(grouped: Dict[str, List[dict]], metric_mean: str, lambda_val: float, output_dir: Path, dpi: int) -> None:
    metric_title, unit = METRIC_LABELS.get(metric_mean, (metric_mean, "值"))

    schedulers = [s for s in SCHEDULER_ORDER if s in grouped] + [s for s in grouped.keys() if s not in SCHEDULER_ORDER]
    x_labels: List[str] = []
    y_vals: List[float] = []
    colors: List[str] = []

    target_lambda_str = f"{lambda_val:g}"
    for scheduler in schedulers:
        rows = grouped[scheduler]
        row = next((r for r in rows if f"{float(r['lambda_per_hour']):g}" == target_lambda_str), None)
        if row is None:
            continue

        x_labels.append(SCHEDULER_LABELS.get(scheduler, scheduler))
        y_vals.append(float(row.get(metric_mean, "0") or 0.0))
        colors.append(SCHEDULER_COLORS.get(scheduler, "#666666"))

    if not x_labels:
        print(f"Skip 柱状图：在 λ={lambda_val:g} 没有可绘制数据")
        return

    out_path = output_dir / f"bar_{metric_mean}_lambda_{target_lambda_str}.png"
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_labels, y_vals, color=colors)
    plt.title(f"算法对比柱状图（λ={target_lambda_str}，指标：{metric_title}）")
    plt.xlabel("算法")
    plt.ylabel(unit)
    plt.grid(True, axis="y", alpha=0.3)

    for b, v in zip(bars, y_vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Wrote plot: {out_path}")


def plot_all(
    summary_csv: Path,
    output_dir: Path,
    dpi: int,
    include_schedulers: set[str] | None = None,
    ppo_eval_npz: Path | None = None,
    bar_metric: str = "accept_rate_mean",
    bar_lambda_mode: str = "ppo_best",
    bar_lambda: float = 8.0,
) -> None:
    grouped = _read_summary(summary_csv, include_schedulers=include_schedulers)
    if not grouped:
        raise RuntimeError("No scheduler data left after filtering. Please check --include-schedulers.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_mean, metric_std, title, unit in METRIC_PLOTS:
        file_name = metric_mean.replace("_mean", "") + ".png"
        out_path = output_dir / file_name
        _plot_metric(grouped, metric_mean, metric_std, title, unit, out_path, dpi)
        print(f"Wrote plot: {out_path}")

    if ppo_eval_npz is not None:
        _plot_ppo_convergence(ppo_eval_npz, output_dir, dpi)

    selected_lambda = bar_lambda
    if bar_lambda_mode == "ppo_best":
        best_lambda = _best_lambda_for_ppo(grouped, bar_metric)
        if best_lambda is None:
            print("Skip 柱状图：当前筛选结果中没有 PPO 数据")
            return
        selected_lambda = best_lambda

    _plot_best_case_bar(grouped, bar_metric, selected_lambda, output_dir, dpi)


def main() -> None:
    args = parse_args()
    include_schedulers = {s.strip() for s in args.include_schedulers.split(",") if s.strip()}
    include = include_schedulers if include_schedulers else None

    plot_all(
        summary_csv=Path(args.summary_csv),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
        include_schedulers=include,
        ppo_eval_npz=Path(args.ppo_eval_npz),
        bar_metric=args.bar_metric,
        bar_lambda_mode=args.bar_lambda_mode,
        bar_lambda=args.bar_lambda,
    )


if __name__ == "__main__":
    main()
