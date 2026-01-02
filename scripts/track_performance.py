#!/usr/bin/env python3
"""
Track performance metrics for each commit.

This script runs the benchmark suite and appends results to the history file
for tracking performance over time.

Usage
-----
    python scripts/track_performance.py

    # Specify custom paths
    python scripts/track_performance.py --history .benchmarks/history.jsonl

    # Run only light benchmarks
    python scripts/track_performance.py --light-only
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def get_git_info() -> tuple:
    """Get current git commit and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        branch = "unknown"

    return commit, branch


def run_benchmarks(output_file: Path, light_only: bool = False) -> bool:
    """
    Run benchmarks and save JSON results.

    Returns True if benchmarks ran successfully.
    """
    benchmark_paths = ["benchmarks/"]
    if light_only:
        benchmark_paths = [
            "benchmarks/test_kalman_bench.py",
            "benchmarks/test_gating_bench.py",
            "benchmarks/test_rotations_bench.py",
        ]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *benchmark_paths,
        "--benchmark-only",
        "--benchmark-json",
        str(output_file),
        "--benchmark-warmup=on",
        "-q",
    ]

    if light_only:
        cmd.extend(["-m", "light"])

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark run failed with exit code {e.returncode}")
        return False


def parse_benchmark_results(json_file: Path) -> list:
    """Parse pytest-benchmark JSON output into history records."""
    with open(json_file) as f:
        data = json.load(f)

    commit, branch = get_git_info()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Get machine info
    machine_info = data.get("machine_info", {})
    runner = machine_info.get("node", "local")

    records = []
    for bench in data.get("benchmarks", []):
        # Extract function name and parameters
        name = bench.get("name", "unknown")
        group = bench.get("group", None)
        fullname = bench.get("fullname", name)

        # Extract parameters from name or params dict
        params = bench.get("params", {})
        if params:
            param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        else:
            # Try to extract from test name
            param_str = "default"
            if "[" in name and "]" in name:
                param_str = name[name.index("[") + 1 : name.index("]")]

        # Extract stats
        stats = bench.get("stats", {})
        mean_s = stats.get("mean", 0)
        stddev_s = stats.get("stddev", 0)
        min_s = stats.get("min", 0)
        max_s = stats.get("max", 0)
        rounds = stats.get("rounds", 0)

        record = {
            "timestamp": timestamp,
            "commit": commit,
            "branch": branch,
            "function": fullname.split("::")[0] if "::" in fullname else name,
            "test": name,
            "params": param_str,
            "mean_ms": mean_s * 1000,
            "stddev_ms": stddev_s * 1000,
            "min_ms": min_s * 1000,
            "max_ms": max_s * 1000,
            "rounds": rounds,
            "runner": runner,
        }
        records.append(record)

    return records


def append_to_history(records: list, history_file: Path) -> None:
    """Append records to JSONL history file."""
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def print_summary(records: list) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Test':<45} {'Mean':>10} {'StdDev':>10}")
    print("-" * 65)

    for record in sorted(records, key=lambda r: r["test"]):
        test = record["test"][:44]
        mean = f"{record['mean_ms']:.3f}ms"
        stddev = f"{record['stddev_ms']:.3f}ms"
        print(f"{test:<45} {mean:>10} {stddev:>10}")

    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track performance metrics for each commit"
    )
    parser.add_argument(
        "--history",
        default=".benchmarks/history.jsonl",
        help="Path to history JSONL file (default: .benchmarks/history.jsonl)",
    )
    parser.add_argument(
        "--light-only",
        action="store_true",
        help="Run only light benchmarks (kalman, gating, rotations)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Don't append results to history file (dry run)",
    )
    args = parser.parse_args()

    history_file = Path(args.history)

    # Create temporary file for benchmark output
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        benchmark_output = Path(tmp.name)

    try:
        print("Running benchmarks...")
        if not run_benchmarks(benchmark_output, args.light_only):
            return 1

        print("Parsing results...")
        records = parse_benchmark_results(benchmark_output)

        if not records:
            print("No benchmark results found!")
            return 1

        print_summary(records)

        if not args.no_append:
            print(f"\nAppending {len(records)} records to {history_file}")
            append_to_history(records, history_file)

        print("\nDone!")
        return 0

    finally:
        # Clean up temporary file
        if benchmark_output.exists():
            benchmark_output.unlink()


if __name__ == "__main__":
    sys.exit(main())
