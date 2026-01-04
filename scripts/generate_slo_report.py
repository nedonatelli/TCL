#!/usr/bin/env python3
"""
Generate SLO compliance reports from benchmark results.

This script produces formatted reports showing:
1. Overall compliance status (pass/fail/warning counts)
2. Per-category SLO compliance tables
3. Trend analysis from historical data
4. Markdown output for PR comments and documentation

Usage
-----
    # Generate report from benchmark JSON
    python scripts/generate_slo_report.py benchmark_results.json

    # Output markdown for PR comment
    python scripts/generate_slo_report.py benchmark_results.json --format markdown

    # Generate HTML report
    python scripts/generate_slo_report.py benchmark_results.json --format html -o report.html

    # Include trend analysis
    python scripts/generate_slo_report.py benchmark_results.json --include-trends
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass
class SLOResult:
    """Result of an SLO check for a single benchmark."""

    test: str
    params: str
    category: str
    actual_ms: float
    slo_mean_ms: float | None
    slo_p99_ms: float | None
    status: Literal["pass", "warning", "fail", "no_slo"]
    deviation_pct: float | None = None


@dataclass
class ComplianceReport:
    """Complete compliance report."""

    timestamp: str
    commit: str
    branch: str
    total_benchmarks: int
    passed: int
    warnings: int
    failures: int
    no_slo: int
    results: list[SLOResult]
    compliance_pct: float

    @property
    def status_emoji(self) -> str:
        if self.failures > 0:
            return "❌"
        if self.warnings > 0:
            return "⚠️"
        return "✅"

    @property
    def status_text(self) -> str:
        if self.failures > 0:
            return "FAILING"
        if self.warnings > 0:
            return "WARNING"
        return "PASSING"


def load_slos(slo_file: Path) -> dict:
    """Load SLO definitions."""
    if not slo_file.exists():
        return {}

    with open(slo_file) as f:
        return json.load(f)


def load_history(history_file: Path, limit: int = 50) -> list:
    """Load recent history records."""
    records: list[dict] = []
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


def load_benchmark_results(results_file: Path) -> tuple[list, dict]:
    """Load benchmark results from pytest-benchmark JSON."""
    with open(results_file) as f:
        data = json.load(f)

    # Get machine/commit info
    machine_info = data.get("machine_info", {})
    commit_info = data.get("commit_info", {})

    meta = {
        "commit": commit_info.get("id", "unknown")[:7],
        "branch": commit_info.get("branch", "unknown"),
        "runner": machine_info.get("node", "local"),
    }

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
                "stddev_ms": stats.get("stddev", 0) * 1000,
                "group": bench.get("group", "other"),
            }
        )

    return results, meta


def categorize_benchmark(test_name: str) -> str:
    """Categorize a benchmark by its test name."""
    name_lower = test_name.lower()

    if "kalman" in name_lower or "kf_" in name_lower:
        return "Kalman Filters"
    if "gating" in name_lower or "mahalanobis" in name_lower:
        return "Gating"
    if "rotation" in name_lower or "euler" in name_lower or "quat" in name_lower:
        return "Rotations"
    if "jpda" in name_lower:
        return "JPDA"
    if "cfar" in name_lower:
        return "CFAR Detection"
    if "cluster" in name_lower or "dbscan" in name_lower or "kmeans" in name_lower:
        return "Clustering"
    if "hyp" in name_lower or "pochhammer" in name_lower or "3f2" in name_lower:
        return "Special Functions"

    return "Other"


def find_matching_slo(
    test_name: str, params: str, slos: dict
) -> tuple[float | None, float | None]:
    """Find matching SLO for a benchmark."""
    slo_defs = slos.get("slos", {})

    best_match = (None, None)
    best_score = 0

    for func_path, func_slos in slo_defs.items():
        if not isinstance(func_slos, dict):
            continue

        for param_key, param_slo in func_slos.items():
            if param_key == "description":
                continue
            if not isinstance(param_slo, dict) or "mean_ms" not in param_slo:
                continue

            # Calculate match score
            score = 0
            key_lower = param_key.lower()
            test_lower = test_name.lower()

            if key_lower in test_lower:
                score = 100 + len(key_lower)
            else:
                key_parts = [p for p in key_lower.split("_") if len(p) > 1]
                if key_parts:
                    matches = sum(1 for p in key_parts if p in test_lower)
                    if matches == len(key_parts):
                        score = 50 + matches

            if score > best_score:
                best_score = score
                best_match = (param_slo.get("mean_ms"), param_slo.get("p99_ms"))

    return best_match


def check_compliance(
    results: list, slos: dict, warning_pct: float = 25.0, failure_pct: float = 50.0
) -> list[SLOResult]:
    """Check all benchmarks against SLOs."""
    slo_results = []

    for result in results:
        test_name = result["test"]
        params = result["params"]
        mean_ms = result["mean_ms"]
        category = categorize_benchmark(test_name)

        slo_mean, slo_p99 = find_matching_slo(test_name, params, slos)

        if slo_mean is None:
            slo_results.append(
                SLOResult(
                    test=test_name,
                    params=params,
                    category=category,
                    actual_ms=mean_ms,
                    slo_mean_ms=None,
                    slo_p99_ms=None,
                    status="no_slo",
                )
            )
            continue

        deviation_pct = ((mean_ms - slo_mean) / slo_mean) * 100

        if deviation_pct > failure_pct:
            status = "fail"
        elif deviation_pct > warning_pct:
            status = "warning"
        else:
            status = "pass"

        slo_results.append(
            SLOResult(
                test=test_name,
                params=params,
                category=category,
                actual_ms=mean_ms,
                slo_mean_ms=slo_mean,
                slo_p99_ms=slo_p99,
                status=status,
                deviation_pct=deviation_pct,
            )
        )

    return slo_results


def compute_trends(results: list, history: list) -> dict:
    """Compute performance trends from history."""
    if not history:
        return {}

    # Group history by test
    by_test: dict[str, list] = defaultdict(list)
    for record in history:
        key = record.get("test", "")
        if record.get("mean_ms"):
            by_test[key].append(record["mean_ms"])

    trends = {}
    for result in results:
        test = result["test"]
        if test in by_test and len(by_test[test]) >= 3:
            recent = by_test[test][-5:]
            avg = sum(recent) / len(recent)
            current = result["mean_ms"]
            change_pct = ((current - avg) / avg) * 100

            if abs(change_pct) > 5:
                trends[test] = {
                    "baseline_ms": avg,
                    "current_ms": current,
                    "change_pct": change_pct,
                    "trend": "up" if change_pct > 0 else "down",
                }

    return trends


def build_report(
    results: list, meta: dict, slos: dict, history: list | None = None
) -> ComplianceReport:
    """Build a complete compliance report."""
    thresholds = slos.get("regression_thresholds", {})
    warning_pct = thresholds.get("warning_percent", 25)
    failure_pct = thresholds.get("failure_percent", 50)

    slo_results = check_compliance(results, slos, warning_pct, failure_pct)

    passed = sum(1 for r in slo_results if r.status == "pass")
    warnings = sum(1 for r in slo_results if r.status == "warning")
    failures = sum(1 for r in slo_results if r.status == "fail")
    no_slo = sum(1 for r in slo_results if r.status == "no_slo")

    # Compliance % = passed / (total with SLOs)
    with_slo = passed + warnings + failures
    compliance_pct = (passed / with_slo * 100) if with_slo > 0 else 100.0

    return ComplianceReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit=meta.get("commit", "unknown"),
        branch=meta.get("branch", "unknown"),
        total_benchmarks=len(slo_results),
        passed=passed,
        warnings=warnings,
        failures=failures,
        no_slo=no_slo,
        results=slo_results,
        compliance_pct=compliance_pct,
    )


def format_markdown(report: ComplianceReport, trends: dict | None = None) -> str:
    """Format report as Markdown."""
    lines = []

    # Header
    lines.append(f"## {report.status_emoji} Performance SLO Report")
    lines.append("")
    lines.append(f"**Commit:** `{report.commit}` | **Branch:** `{report.branch}`")
    lines.append(
        f"**Status:** {report.status_text} | **Compliance:** {report.compliance_pct:.1f}%"
    )
    lines.append("")

    # Summary table
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| ✅ Passed | {report.passed} |")
    lines.append(f"| ⚠️ Warnings | {report.warnings} |")
    lines.append(f"| ❌ Failures | {report.failures} |")
    lines.append(f"| ❓ No SLO | {report.no_slo} |")
    lines.append(f"| **Total** | **{report.total_benchmarks}** |")
    lines.append("")

    # Group results by category
    by_category: dict[str, list] = defaultdict(list)
    for result in report.results:
        by_category[result.category].append(result)

    # Failures and warnings section
    issues = [r for r in report.results if r.status in ("fail", "warning")]
    if issues:
        lines.append("### Issues")
        lines.append("")
        lines.append("| Status | Test | Actual | SLO | Deviation |")
        lines.append("|--------|------|--------|-----|-----------|")
        for r in sorted(issues, key=lambda x: x.status):
            status_icon = "❌" if r.status == "fail" else "⚠️"
            dev = f"+{r.deviation_pct:.1f}%" if r.deviation_pct else "N/A"
            lines.append(
                f"| {status_icon} | `{r.test[:40]}` | {r.actual_ms:.3f}ms | {r.slo_mean_ms:.3f}ms | {dev} |"
            )
        lines.append("")

    # Per-category breakdown (collapsed)
    lines.append("<details>")
    lines.append("<summary>📊 Detailed Results by Category</summary>")
    lines.append("")

    for category in sorted(by_category.keys()):
        cat_results = by_category[category]
        passed = sum(1 for r in cat_results if r.status == "pass")
        total = len([r for r in cat_results if r.status != "no_slo"])

        if total > 0:
            pct = passed / total * 100
            emoji = "✅" if pct == 100 else ("⚠️" if pct >= 75 else "❌")
        else:
            pct = 100
            emoji = "❓"

        lines.append(f"#### {emoji} {category} ({passed}/{total} passed)")
        lines.append("")
        lines.append("| Test | Actual | SLO | Status |")
        lines.append("|------|--------|-----|--------|")

        for r in sorted(cat_results, key=lambda x: x.test):
            if r.status == "no_slo":
                status = "❓"
                slo = "—"
            else:
                status = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[r.status]
                slo = f"{r.slo_mean_ms:.3f}ms"
            lines.append(
                f"| `{r.test[:35]}` | {r.actual_ms:.3f}ms | {slo} | {status} |"
            )

        lines.append("")

    lines.append("</details>")
    lines.append("")

    # Trends section
    if trends:
        lines.append("<details>")
        lines.append("<summary>📈 Performance Trends</summary>")
        lines.append("")
        lines.append("| Test | Baseline | Current | Change |")
        lines.append("|------|----------|---------|--------|")

        for test, trend in sorted(trends.items()):
            arrow = "📈" if trend["trend"] == "up" else "📉"
            change = f"{trend['change_pct']:+.1f}%"
            lines.append(
                f"| `{test[:35]}` | {trend['baseline_ms']:.3f}ms | {trend['current_ms']:.3f}ms | {arrow} {change} |"
            )

        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated at {report.timestamp}*")

    return "\n".join(lines)


def format_text(report: ComplianceReport) -> str:
    """Format report as plain text."""
    lines = []

    lines.append("=" * 70)
    lines.append(f"PERFORMANCE SLO REPORT - {report.status_text}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Commit: {report.commit} | Branch: {report.branch}")
    lines.append(f"Compliance: {report.compliance_pct:.1f}%")
    lines.append("")
    lines.append(f"  Passed:   {report.passed:3d}")
    lines.append(f"  Warnings: {report.warnings:3d}")
    lines.append(f"  Failures: {report.failures:3d}")
    lines.append(f"  No SLO:   {report.no_slo:3d}")
    lines.append(f"  Total:    {report.total_benchmarks:3d}")
    lines.append("")

    # Show issues
    issues = [r for r in report.results if r.status in ("fail", "warning")]
    if issues:
        lines.append("-" * 70)
        lines.append("ISSUES:")
        lines.append("-" * 70)
        for r in sorted(issues, key=lambda x: x.status):
            prefix = "FAIL" if r.status == "fail" else "WARN"
            dev = f"+{r.deviation_pct:.1f}%" if r.deviation_pct else "N/A"
            lines.append(f"  [{prefix}] {r.test}")
            lines.append(
                f"          actual={r.actual_ms:.3f}ms slo={r.slo_mean_ms:.3f}ms ({dev})"
            )
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_json(report: ComplianceReport) -> str:
    """Format report as JSON."""
    data = {
        "timestamp": report.timestamp,
        "commit": report.commit,
        "branch": report.branch,
        "status": report.status_text.lower(),
        "compliance_pct": report.compliance_pct,
        "summary": {
            "total": report.total_benchmarks,
            "passed": report.passed,
            "warnings": report.warnings,
            "failures": report.failures,
            "no_slo": report.no_slo,
        },
        "results": [
            {
                "test": r.test,
                "params": r.params,
                "category": r.category,
                "actual_ms": r.actual_ms,
                "slo_mean_ms": r.slo_mean_ms,
                "status": r.status,
                "deviation_pct": r.deviation_pct,
            }
            for r in report.results
        ],
    }
    return json.dumps(data, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SLO compliance reports")
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
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--include-trends",
        action="store_true",
        help="Include trend analysis from history",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error on any failures",
    )

    args = parser.parse_args()

    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file {results_file} not found", file=sys.stderr)
        return 1

    # Load data
    results, meta = load_benchmark_results(results_file)
    slos = load_slos(Path(args.slos))
    history = load_history(Path(args.history)) if args.include_trends else None

    # Build report
    report = build_report(results, meta, slos, history)

    # Compute trends if requested
    trends = (
        compute_trends(results, history) if args.include_trends and history else None
    )

    # Format output
    if args.format == "markdown":
        output = format_markdown(report, trends)
    elif args.format == "json":
        output = format_json(report)
    else:
        output = format_text(report)

    # Write output
    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Exit code
    if args.strict and report.failures > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
