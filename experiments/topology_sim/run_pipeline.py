from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


TRAIN_ALGOS = ["none", "ppo", "ddqn", "a3c"]
POLICY_MODES = ["best_fit", "random", "ppo", "ddqn", "a3c"]
LEARNING_MODES = ["ppo", "ddqn", "a3c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click pipeline: train -> start policy service -> run Java -> summarize -> plot")

    parser.add_argument("--train-algo", type=str, default="none", choices=TRAIN_ALGOS)
    parser.add_argument("--total-steps", type=int, default=300_000, help="Training timesteps/steps for selected algorithm.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", type=str, default="experiments/topology_sim/models")

    parser.add_argument("--policy-mode", type=str, default="ppo", choices=POLICY_MODES)
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)

    parser.add_argument("--compare-four", action="store_true", help="Run ppo/ddqn/a3c sequentially and auto-merge to 4+ algorithm comparison CSV.")

    parser.add_argument(
        "--java-cmd",
        type=str,
        default=(
            "mvn -pl modules/cloudsim-examples -am -DskipTests install && "
            "mvn -f modules/cloudsim-examples/pom.xml -DskipTests exec:java "
            "-Dexec.mainClass=org.cloudbus.cloudsim.examples.topologyrl.RunExperiments"
        ),
        help="Command to run Java experiments.",
    )
    parser.add_argument("--skip-java", action="store_true", help="Skip Java experiment execution.")

    parser.add_argument("--results-csv", type=str, default="experiments/topology_sim/results_java.csv")
    parser.add_argument("--summary-csv", type=str, default="experiments/topology_sim/results_java_summary.csv")
    parser.add_argument("--plots-dir", type=str, default="experiments/topology_sim/plots")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plot generation from summary CSV.")

    return parser.parse_args()


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def wait_health(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except URLError as e:
            last_err = str(e)
        time.sleep(0.5)
    raise RuntimeError(f"Policy service health check failed: {url}, last_error={last_err}")


def maybe_train(args: argparse.Namespace, models_dir: Path) -> None:
    if args.train_algo == "none":
        return

    if args.train_algo == "ppo":
        run_cmd(
            [
                sys.executable,
                "-m",
                "experiments.topology_sim.ppo_train",
                "--total-timesteps",
                str(args.total_steps),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(models_dir),
            ]
        )
        return

    if args.train_algo == "ddqn":
        run_cmd(
            [
                sys.executable,
                "-m",
                "experiments.topology_sim.ddqn_train",
                "--total-steps",
                str(args.total_steps),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(models_dir),
            ]
        )
        return

    if args.train_algo == "a3c":
        run_cmd(
            [
                sys.executable,
                "-m",
                "experiments.topology_sim.a3c_train",
                "--total-steps",
                str(args.total_steps),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(models_dir),
            ]
        )
        return


def build_policy_env(base_env: dict[str, str], policy_mode: str, models_dir: Path) -> dict[str, str]:
    env = base_env.copy()
    env["TOPOLOGY_POLICY_MODE"] = policy_mode

    if policy_mode == "ppo":
        env["TOPOLOGY_PPO_MODEL_PATH"] = str(models_dir / "ppo_topology_model.zip")
    if policy_mode == "ddqn":
        env["TOPOLOGY_DDQN_MODEL_PATH"] = str(models_dir / "ddqn_topology_model.pt")
    if policy_mode == "a3c":
        env["TOPOLOGY_A3C_MODEL_PATH"] = str(models_dir / "a3c_topology_model.pt")

    return env


def run_single_mode(args: argparse.Namespace, models_dir: Path, policy_mode: str) -> Path:
    env = build_policy_env(os.environ.copy(), policy_mode, models_dir)

    policy_proc = subprocess.Popen(
        [sys.executable, "-m", "experiments.topology_sim.policy_service"],
        env=env,
    )

    try:
        health_url = f"http://{args.policy_host}:{args.policy_port}/health"
        wait_health(health_url)
        print(f"Policy service is healthy: {health_url} ({policy_mode})")

        if not args.skip_java:
            subprocess.run(args.java_cmd, check=True, shell=True)

        result_path = Path(args.results_csv)
        if not result_path.exists():
            raise FileNotFoundError(f"Expected Java results csv not found: {result_path}")
        return result_path
    finally:
        policy_proc.terminate()
        try:
            policy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            policy_proc.kill()


def rewrite_python_policy_rows(input_csv: Path, output_csv: Path, new_name: str) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("scheduler") == "python_policy":
                row["scheduler"] = new_name
            writer.writerow(row)


def merge_four_compare_csvs(raw_dir: Path, merged_out: Path) -> None:
    mode_files = {
        "ppo": raw_dir / "results_java_ppo.csv",
        "ddqn": raw_dir / "results_java_ddqn.csv",
        "a3c": raw_dir / "results_java_a3c.csv",
    }

    merged_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for mode in LEARNING_MODES:
        file_path = mode_files[mode]
        if not file_path.exists():
            raise FileNotFoundError(f"Missing raw result for mode={mode}: {file_path}")

        with file_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            if rows and fieldnames is None:
                fieldnames = list(rows[0].keys())

            for row in rows:
                scheduler = row.get("scheduler", "")
                if scheduler == "python_policy" or scheduler == mode:
                    row["scheduler"] = mode
                    merged_rows.append(row)
                elif mode == "ppo":
                    merged_rows.append(row)

    if fieldnames is None:
        raise RuntimeError("No rows available to merge.")

    merged_out.parent.mkdir(parents=True, exist_ok=True)
    with merged_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


def summarize_and_plot(args: argparse.Namespace) -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "experiments.topology_sim.summarize_results",
            "--input-csv",
            args.results_csv,
            "--output-csv",
            args.summary_csv,
        ]
    )

    if not args.skip_plot:
        run_cmd(
            [
                sys.executable,
                "-m",
                "experiments.topology_sim.plot_results",
                "--summary-csv",
                args.summary_csv,
                "--output-dir",
                args.plots_dir,
            ]
        )


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_four:
        raw_dir = Path("experiments/topology_sim/raw_runs")
        raw_dir.mkdir(parents=True, exist_ok=True)

        for mode in LEARNING_MODES:
            if args.train_algo == mode:
                maybe_train(args, models_dir)
            elif args.train_algo != "none":
                sub_args = argparse.Namespace(**vars(args))
                sub_args.train_algo = mode
                maybe_train(sub_args, models_dir)

            run_single_mode(args, models_dir, mode)
            mode_raw = raw_dir / f"results_java_{mode}.csv"
            rewrite_python_policy_rows(Path(args.results_csv), mode_raw, mode)

        merge_four_compare_csvs(raw_dir, Path(args.results_csv))
        summarize_and_plot(args)
    else:
        maybe_train(args, models_dir)
        run_single_mode(args, models_dir, args.policy_mode)
        summarize_and_plot(args)

    print("Pipeline completed successfully.")
    print(f"Results CSV : {root / args.results_csv}")
    print(f"Summary CSV : {root / args.summary_csv}")
    if not args.skip_plot:
        print(f"Plots Dir   : {root / args.plots_dir}")


if __name__ == "__main__":
    main()
