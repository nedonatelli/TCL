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

import pytest

from scripts.detect_regressions import check_slo_violations
from scripts.detect_regressions import main as detect_main
from scripts.generate_slo_report import check_compliance, find_matching_slo
from scripts.generate_slo_report import main as report_main

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
