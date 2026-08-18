"""The benchmark SLO gate must actually trip on a violation.

`scripts/detect_regressions.py` and `scripts/generate_slo_report.py` both
used to read `slos.get("slos", {})`, expecting a shape
`{"slos": {<func_path>: {<param_key>: {"mean_ms": ..., "p99_ms": ...}}}}`.
`.benchmarks/slos.json` has never had a `"slos"` key -- it is
`{"benchmarks": {<test_name>: {"max_mean_us": ..., "max_p99_us": ...}}}`
(exact test name, microseconds). The lookup silently returned `{}`, so the
gate reported "No performance issues detected" no matter how slow a
benchmark was -- including a synthetic 1000ms result against a 50us SLO,
running with `--strict` on every push to `main`
(`.github/workflows/benchmark-full.yml`).

These tests pin two things: that a fabricated violation actually produces a
FAILURE and a nonzero exit code under `--strict`, and that a compliant
result stays clean. Either script regressing back to the old key/shape (or
losing the unit conversion) makes at least one of these fail.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.detect_regressions import check_slo_violations
from scripts.detect_regressions import main as detect_main
from scripts.generate_slo_report import check_compliance, find_matching_slo
from scripts.generate_slo_report import main as report_main

REPO_ROOT = Path(__file__).resolve().parents[2]
SLOS_FILE = REPO_ROOT / ".benchmarks" / "slos.json"

# Mirrors the real .benchmarks/slos.json shape: flat "benchmarks" dict keyed
# by exact pytest-benchmark name, thresholds in microseconds.
_SLOS = {
    "description": "test SLOs",
    "benchmarks": {
        "test_kf_predict": {"max_mean_us": 50.0, "max_p99_us": 100.0},
    },
    "regression_thresholds": {
        "warning_percent": 15,
        "failure_percent": 30,
        "min_samples": 5,
    },
}


def _result(mean_ms: float, test: str = "test_kf_predict") -> dict:
    return {"function": "bench", "test": test, "params": "default", "mean_ms": mean_ms}


def _write_benchmark_json(path, mean_seconds: float, name: str = "test_kf_predict"):
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": name,
                        "fullname": f"benchmarks/test_kalman_bench.py::{name}",
                        "params": {},
                        "stats": {"mean": mean_seconds, "max": mean_seconds * 1.2},
                    }
                ]
            }
        )
    )


class TestDetectRegressionsSLOMatching:
    """Unit-level: check_slo_violations against the real on-disk schema shape."""

    def test_gross_violation_is_flagged(self):
        # 1000ms mean vs a 0.050ms (50us) SLO -- the audit's exact probe.
        issues = check_slo_violations([_result(mean_ms=1000.0)], _SLOS)
        assert len(issues) == 1
        assert issues[0]["level"] == "FAILURE"

    def test_compliant_result_is_not_flagged(self):
        # 0.03ms (30us) is comfortably under the 50us SLO.
        issues = check_slo_violations([_result(mean_ms=0.03)], _SLOS)
        assert issues == []

    def test_old_nested_slos_key_matches_nothing(self):
        # Regression guard for the actual bug: a payload shaped like the
        # pre-break "slos" nesting must not be silently accepted as if it
        # were today's "benchmarks" shape.
        old_shape = {
            "slos": {
                "pytcl.dynamic_estimation.kalman.linear.kf_predict": {
                    "state_dim_4": {"mean_ms": 0.02, "p99_ms": 0.1}
                }
            }
        }
        issues = check_slo_violations([_result(mean_ms=1000.0)], old_shape)
        assert issues == []


class TestDetectRegressionsMainExitCode:
    """End-to-end: main() must exit nonzero under --strict on a real violation."""

    def test_strict_mode_exits_nonzero_on_violation(self, tmp_path, monkeypatch):
        results_file = tmp_path / "results.json"
        slos_file = tmp_path / "slos.json"
        _write_benchmark_json(results_file, mean_seconds=1.0)  # 1000ms
        slos_file.write_text(json.dumps(_SLOS))

        monkeypatch.setattr(
            "sys.argv",
            [
                "detect_regressions.py",
                str(results_file),
                "--slos",
                str(slos_file),
                "--history",
                str(tmp_path / "missing_history.jsonl"),
                "--strict",
            ],
        )
        assert detect_main() == 1

    def test_strict_mode_exits_zero_when_compliant(self, tmp_path, monkeypatch):
        results_file = tmp_path / "results.json"
        slos_file = tmp_path / "slos.json"
        _write_benchmark_json(results_file, mean_seconds=0.00003)  # 30us
        slos_file.write_text(json.dumps(_SLOS))

        monkeypatch.setattr(
            "sys.argv",
            [
                "detect_regressions.py",
                str(results_file),
                "--slos",
                str(slos_file),
                "--history",
                str(tmp_path / "missing_history.jsonl"),
                "--strict",
            ],
        )
        assert detect_main() == 0


class TestGenerateSLOReportMatching:
    """Unit-level: find_matching_slo / check_compliance against the real schema."""

    def test_find_matching_slo_converts_us_to_ms(self):
        mean_ms, p99_ms = find_matching_slo("test_kf_predict", "default", _SLOS)
        assert mean_ms == pytest.approx(0.050)
        assert p99_ms == pytest.approx(0.100)

    def test_check_compliance_fails_gross_violation(self):
        results = [
            {
                "function": "bench",
                "test": "test_kf_predict",
                "params": "default",
                "mean_ms": 1000.0,
                "max_ms": 1200.0,
                "stddev_ms": 1.0,
                "group": "other",
            }
        ]
        slo_results = check_compliance(results, _SLOS)
        assert len(slo_results) == 1
        assert slo_results[0].status == "fail"

    def test_check_compliance_passes_compliant_result(self):
        results = [
            {
                "function": "bench",
                "test": "test_kf_predict",
                "params": "default",
                "mean_ms": 0.03,
                "max_ms": 0.04,
                "stddev_ms": 0.001,
                "group": "other",
            }
        ]
        slo_results = check_compliance(results, _SLOS)
        assert len(slo_results) == 1
        assert slo_results[0].status == "pass"


class TestGenerateSLOReportMainExitCode:
    """End-to-end: main() must exit nonzero under --strict on a real violation."""

    def test_strict_mode_exits_nonzero_on_violation(self, tmp_path, monkeypatch):
        results_file = tmp_path / "results.json"
        slos_file = tmp_path / "slos.json"
        _write_benchmark_json(results_file, mean_seconds=1.0)  # 1000ms
        slos_file.write_text(json.dumps(_SLOS))

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_slo_report.py",
                str(results_file),
                "--slos",
                str(slos_file),
                "--history",
                str(tmp_path / "missing_history.jsonl"),
                "--strict",
            ],
        )
        assert report_main() == 1

    def test_strict_mode_exits_zero_when_compliant(self, tmp_path, monkeypatch):
        results_file = tmp_path / "results.json"
        slos_file = tmp_path / "slos.json"
        _write_benchmark_json(results_file, mean_seconds=0.00003)  # 30us
        slos_file.write_text(json.dumps(_SLOS))

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_slo_report.py",
                str(results_file),
                "--slos",
                str(slos_file),
                "--history",
                str(tmp_path / "missing_history.jsonl"),
                "--strict",
            ],
        )
        assert report_main() == 0


def _live_benchmark_names() -> set[str]:
    """The real benchmark names pytest would collect from benchmarks/.

    Collected live (rather than a hand-maintained static list) specifically
    so that renaming a `@pytest.mark.parametrize` id in benchmarks/*.py --
    the exact way `test_kf_predict` / `test_kf_update` went orphaned in
    `.benchmarks/slos.json` once their bare names stopped existing -- makes
    this test fail immediately instead of quietly producing a new orphan.

    `-o addopts=""` strips the repo's default `--strict-markers` etc, which
    otherwise collapse `--collect-only -q` down to a one-line-per-file
    count instead of full node IDs. pytest-benchmark's JSON `name` field
    (what `.benchmarks/slos.json` keys must match) is `item.name`: the
    node ID's final `::`-separated segment, with no class/file prefix.

    Run under `pytest -n auto` (xdist), this subprocess inherits the
    worker's `PYTEST_*` environment (`PYTEST_XDIST_WORKER`,
    `PYTEST_CURRENT_TEST`, ...) and the child pytest picks up on being
    "inside" a run already in progress, collecting nothing. Scrub every
    `PYTEST_`-prefixed variable so the child collects standalone regardless
    of what invoked this test; `-p no:cacheprovider` keeps it from touching
    the parent run's `.pytest_cache` for good measure.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/",
            "--collect-only",
            "--benchmark-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    names = set()
    for line in result.stdout.splitlines():
        if "::" not in line:
            continue
        names.add(line.strip().rsplit("::", 1)[-1])
    return names


class TestSLOEntriesNotOrphaned:
    """Every .benchmarks/slos.json key must match a real benchmark name.

    This is the residual-vacuity case the exact-name-match fix doesn't
    cover by construction: a key that matches nothing is not a schema
    mismatch (check_slo_violations/find_matching_slo work fine), it is a
    threshold that silently never fires. That's exactly how
    `test_kf_predict` / `test_kf_update` went dead -- the schema was right,
    the key just stopped matching any live benchmark.
    """

    def test_guard_the_guard_collection_finds_benchmarks(self):
        names = _live_benchmark_names()
        assert len(names) > 50, f"only collected {len(names)} benchmark names"

    def test_no_slo_entry_is_orphaned(self):
        slos = json.loads(SLOS_FILE.read_text())
        slo_keys = set(slos.get("benchmarks", {}))
        live_names = _live_benchmark_names()

        orphaned = slo_keys - live_names
        assert not orphaned, (
            f"{sorted(orphaned)} in {SLOS_FILE} match no benchmark pytest "
            f"would actually collect -- these SLOs can never fire"
        )
