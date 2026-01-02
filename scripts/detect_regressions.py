#!/usr/bin/env python3
"""
Detect performance regressions by comparing against SLOs and history.

This script checks benchmark results against:
1. SLO definitions (mean/p99 thresholds)
2. Historical baseline (rolling average of recent runs)

Usage
-----
    python scripts/detect_regressions.py benchmark_results.json

    # Strict mode (exit with error on any violation)
    python scripts/detect_regressions.py benchmark_results.json --strict

    # Custom paths
    python scripts/detect_regressions.py results.json \\
        --slos .benchmarks/slos.json \\
        --history .benchmarks/history.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_slos(slo_file: Path) -> dict:
    """Load SLO definitions."""
    if not slo_file.exists():
        print(f"Warning: SLO file {slo_file} not found, skipping SLO checks")
        return {}

    with open(slo_file) as f:
        return json.load(f)


def load_history(history_file: Path, limit: int = 100) -> list:
    """Load recent history records."""
    records = []
    if not history_file.exists():
        return records

    with open(history_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return records[-limit:]


def load_current_results(results_file: Path) -> list:
    """Load current benchmark results from pytest-benchmark JSON."""
    with open(results_file) as f:
        data = json.load(f)

    results = []
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "unknown")
        fullname = bench.get("fullname", name)

        # Extract parameters
        params = bench.get("params", {})
        if params:
            param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        else:
            param_str = "default"
            if "[" in name and "]" in name:
                param_str = name[name.index("[") + 1 : name.index("]")]

        stats = bench.get("stats", {})
        results.append(
            {
                "function": fullname.split("::")[0] if "::" in fullname else name,
                "test": name,
                "params": param_str,
                "mean_ms": stats.get("mean", 0) * 1000,
                "max_ms": stats.get("max", 0) * 1000,
            }
        )

    return results


def _match_score(test_name: str, param_key: str) -> int:
    """Calculate match score between test name and SLO param key.

    Higher score = better match. Returns 0 for no match.
    """
    test_lower = test_name.lower()
    key_lower = param_key.lower()

    # Exact match in test name (highest priority)
    if key_lower in test_lower:
        return 100 + len(key_lower)  # Longer matches score higher

    # Check if all parts of the key appear in the test name
    key_parts = [p for p in key_lower.split("_") if len(p) > 1]
    if key_parts:
        matches = sum(1 for p in key_parts if p in test_lower)
        if matches == len(key_parts):
            return 50 + matches

    return 0


def check_slo_violations(results: list, slos: dict) -> list:
    """Check for SLO violations."""
    issues = []

    if not slos:
        return issues

    thresholds = slos.get("regression_thresholds", {})
    warning_pct = thresholds.get("warning_percent", 15)
    failure_pct = thresholds.get("failure_percent", 30)

    slo_defs = slos.get("slos", {})

    for result in results:
        test_name = result["test"]
        params = result["params"]
        mean_ms = result["mean_ms"]

        # Try to find matching SLO with best score
        matched_slo = None
        best_score = 0

        for func_path, func_slos in slo_defs.items():
            if isinstance(func_slos, dict):
                # Check if any param key matches
                for param_key, param_slo in func_slos.items():
                    if param_key in ["description"]:
                        continue
                    if isinstance(param_slo, dict) and "mean_ms" in param_slo:
                        score = _match_score(test_name, param_key)
                        if score > best_score:
                            best_score = score
                            matched_slo = param_slo

        if matched_slo:
            target_mean = matched_slo.get("mean_ms")
            if target_mean:
                deviation_pct = ((mean_ms - target_mean) / target_mean) * 100

                if deviation_pct > failure_pct:
                    issues.append(
                        {
                            "level": "FAILURE",
                            "test": test_name,
                            "params": params,
                            "actual_ms": mean_ms,
                            "slo_ms": target_mean,
                            "deviation_pct": deviation_pct,
                            "message": f"FAILURE: {test_name}[{params}] mean={mean_ms:.3f}ms "
                            f"exceeds SLO={target_mean:.3f}ms by {deviation_pct:.1f}%",
                        }
                    )
                elif deviation_pct > warning_pct:
                    issues.append(
                        {
                            "level": "WARNING",
                            "test": test_name,
                            "params": params,
                            "actual_ms": mean_ms,
                            "slo_ms": target_mean,
                            "deviation_pct": deviation_pct,
                            "message": f"WARNING: {test_name}[{params}] mean={mean_ms:.3f}ms "
                            f"exceeds SLO={target_mean:.3f}ms by {deviation_pct:.1f}%",
                        }
                    )

    return issues


def check_historical_regressions(
    results: list, history: list, thresholds: dict
) -> list:
    """Check for regressions against recent history."""
    issues = []

    if not history:
        return issues

    failure_pct = thresholds.get("failure_percent", 30)
    min_samples = thresholds.get("min_samples", 5)

    # Build historical baseline (mean of recent runs per test)
    baseline: dict[tuple, list] = defaultdict(list)
    for record in history:
        key = (record.get("test", ""), record.get("params", ""))
        if record.get("mean_ms"):
            baseline[key].append(record["mean_ms"])

    # Compute baselines (use last N samples)
    baselines = {}
    for key, values in baseline.items():
        if len(values) >= min_samples:
            # Use average of last min_samples runs
            baselines[key] = sum(values[-min_samples:]) / min(len(values), min_samples)

    # Compare current results
    for result in results:
        key = (result["test"], result["params"])
        if key in baselines:
            base = baselines[key]
            current = result["mean_ms"]
            change_pct = ((current - base) / base) * 100

            if change_pct > failure_pct:
                issues.append(
                    {
                        "level": "REGRESSION",
                        "test": result["test"],
                        "params": result["params"],
                        "actual_ms": current,
                        "baseline_ms": base,
                        "change_pct": change_pct,
                        "message": f"REGRESSION: {result['test']}[{result['params']}] "
                        f"mean={current:.3f}ms vs baseline={base:.3f}ms "
                        f"(+{change_pct:.1f}%)",
                    }
                )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect performance regressions against SLOs and history"
    )
    parser.add_argument("results", help="Path to benchmark results JSON")
    parser.add_argument(
        "--slos",
        default=".benchmarks/slos.json",
        help="Path to SLO definitions (default: .benchmarks/slos.json)",
    )
    parser.add_argument(
        "--history",
        default=".benchmarks/history.jsonl",
        help="Path to history JSONL (default: .benchmarks/history.jsonl)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error on any violation or regression",
    )
    args = parser.parse_args()

    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file {results_file} not found")
        return 1

    # Load data
    results = load_current_results(results_file)
    slos = load_slos(Path(args.slos))
    history = load_history(Path(args.history))

    # Get thresholds
    thresholds = slos.get("regression_thresholds", {})

    # Check for issues
    slo_issues = check_slo_violations(results, slos)
    regression_issues = check_historical_regressions(results, history, thresholds)

    all_issues = slo_issues + regression_issues

    # Print results
    print("\n" + "=" * 70)
    print("PERFORMANCE REGRESSION CHECK")
    print("=" * 70)

    if all_issues:
        print(f"\nFound {len(all_issues)} issue(s):\n")
        for issue in all_issues:
            print(issue["message"])
    else:
        print("\nNo performance issues detected.")

    print("\n" + "=" * 70)

    # Summary
    failures = [i for i in all_issues if i["level"] in ("FAILURE", "REGRESSION")]
    warnings = [i for i in all_issues if i["level"] == "WARNING"]

    print(f"Summary: {len(failures)} failures, {len(warnings)} warnings")

    if args.strict and failures:
        print(f"\nStrict mode: {len(failures)} failures detected. Exiting with error.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
