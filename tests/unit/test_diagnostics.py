"""pytcl.diagnostics: silence by default, enable/disable, instrumentation."""

import subprocess
import sys

import pytest
from loguru import logger as _loguru_logger

import pytcl
from pytcl.diagnostics import (
    diagnostics_enabled,
    disable_debug_logging,
    enable_debug_logging,
)


@pytest.fixture(autouse=True)
def _always_silent_after():
    yield
    disable_debug_logging()


class TestSilenceByDefault:
    def test_import_emits_nothing_and_installs_no_handlers(self):
        # Fresh interpreter: importing pytcl and running a filter step must
        # print nothing and leave loguru handler state untouched.
        code = (
            "from loguru import logger; before = len(logger._core.handlers); "
            "import numpy as np; import pytcl; "
            "from pytcl.dynamic_estimation.kalman.linear import kf_predict; "
            "kf_predict(np.zeros(2), np.eye(2), np.eye(2), np.eye(2)); "
            "after = len(logger._core.handlers); "
            "assert after == before, f'{before} -> {after} handlers'; "
            "print('SILENT-OK', end='')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "SILENT-OK"
        assert result.stderr == ""

    def test_disabled_flag_by_default(self):
        assert diagnostics_enabled() is False


class TestEnableDisable:
    def test_round_trip(self, capsys):
        records = []
        enable_debug_logging()
        assert diagnostics_enabled() is True
        handle = _loguru_logger.add(records.append, format="{message}")
        _loguru_logger.bind(name="pytcl").debug("probe")
        _loguru_logger.remove(handle)
        disable_debug_logging()
        assert diagnostics_enabled() is False

    def test_enable_twice_does_not_stack_handlers(self):
        enable_debug_logging()
        n1 = len(_loguru_logger._core.handlers)
        enable_debug_logging()
        n2 = len(_loguru_logger._core.handlers)
        assert n2 == n1
        disable_debug_logging()

    def test_reexported_from_top_level(self):
        assert pytcl.enable_debug_logging is enable_debug_logging
        assert pytcl.disable_debug_logging is disable_debug_logging


class TestRendering:
    def _tracks(self):
        import numpy as np

        from pytcl.trackers import MultiTargetTracker

        tracker = MultiTargetTracker(
            state_dim=4,
            meas_dim=2,
            F=np.eye(4),
            H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
            Q=np.eye(4) * 0.01,
            R=np.eye(2) * 2.0,
        )
        tracks = tracker.process([np.array([1.0, 2.0])], dt=1.0)
        return tracks

    def test_track_table_is_ascii_only(self):
        import io

        from rich.console import Console

        from pytcl.diagnostics import track_table

        buf = io.StringIO()
        track_table(self._tracks(), console=Console(file=buf, width=100))
        out = buf.getvalue()
        assert len(out) > 0
        out.encode("cp1252")  # raises UnicodeEncodeError on any unsafe char

    def test_progress_bar_yields_all_items_and_is_ascii(self, capsys):
        from pytcl.diagnostics import progress_bar

        items = list(progress_bar(range(5), description="test"))
        assert items == [0, 1, 2, 3, 4]
        err = capsys.readouterr().err
        err.encode("cp1252")

    def test_terrain_progress_param_accepted(self):
        import inspect

        from pytcl.terrain.loaders import load_earth2014, load_gebco

        assert "progress" in inspect.signature(load_gebco).parameters
        assert "progress" in inspect.signature(load_earth2014).parameters


class TestDataFileInstrumentation:
    def test_missing_file_resolution_is_logged(self, tmp_path, monkeypatch):
        from loguru import logger as _l

        from pytcl.terrain.loaders import load_gebco

        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            with pytest.raises((FileNotFoundError, Exception)):
                load_gebco(0.0, 0.1, 0.0, 0.1)
        finally:
            _l.remove(handle)
            disable_debug_logging()
        text = " ".join(str(r) for r in records)
        assert str(tmp_path) in text  # the resolved directory
        assert "PYTCL_DATA_DIR" in text  # the override was named
        assert "missing" in text or "not found" in text

    def test_silent_when_disabled(self, tmp_path, monkeypatch):
        from loguru import logger as _l

        from pytcl.terrain.loaders import load_gebco

        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        records = []
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            with pytest.raises((FileNotFoundError, Exception)):
                load_gebco(0.0, 0.1, 0.0, 0.1)
        finally:
            _l.remove(handle)
        assert records == []  # disabled namespace emits nothing


class TestGatingAssociationInstrumentation:
    def _run_scenario(self, capture):
        import numpy as np

        from pytcl.trackers import MultiTargetTracker

        tracker = MultiTargetTracker(
            state_dim=4,
            meas_dim=2,
            F=np.array(
                [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
            ),
            H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
            Q=np.eye(4) * 0.01,
            R=np.eye(2) * 1.0,
            confirm_hits=1,
        )
        # Scan 1 starts a track at origin; scan 2 has one near measurement
        # and one far measurement that MUST be gated out.
        tracker.process([np.array([0.0, 0.0])], dt=1.0)
        tracker.process([np.array([0.1, 0.1]), np.array([500.0, 500.0])], dt=1.0)

    def test_gating_rejection_logged_when_enabled(self):
        from loguru import logger as _l

        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            self._run_scenario(records)
        finally:
            _l.remove(handle)
            disable_debug_logging()
        text = " ".join(str(r) for r in records)
        assert "gate" in text.lower()
        assert "assign" in text.lower() or "association" in text.lower()

    def test_zero_records_and_zero_payload_when_disabled(self, monkeypatch):
        from loguru import logger as _l

        from pytcl.diagnostics import diagnostics_enabled as real

        calls = []
        # Patch the CONSUMING module's imported name -- the tracker holds a
        # direct reference, so patching pytcl.diagnostics itself would miss.
        monkeypatch.setattr(
            "pytcl.trackers.multi_target.diagnostics_enabled",
            lambda: (calls.append(1), real())[1],
        )
        records = []
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            self._run_scenario(records)
        finally:
            _l.remove(handle)
        assert records == []
        assert len(calls) > 0  # the guard IS consulted on the hot path

    def test_jpda_summary_logged(self):
        import numpy as np
        from loguru import logger as _l

        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            likelihood = np.array([[0.8, 0.1], [0.2, 0.7]])
            gated = np.array([[True, True], [True, True]])
            jpda_probabilities(
                likelihood, gated, detection_prob=0.9, clutter_density=1e-6
            )
        finally:
            _l.remove(handle)
            disable_debug_logging()
        assert any("jpda" in str(r).lower() for r in records)

    def _run_mht_scenario(self):
        import numpy as np

        from pytcl.trackers import MHTTracker

        tracker = MHTTracker(
            state_dim=4,
            meas_dim=2,
            F=np.array(
                [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
            ),
            H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
            Q=np.eye(4) * 0.01,
            R=np.eye(2) * 1.0,
        )
        tracker.process([np.array([0.0, 0.0])], dt=1.0)
        tracker.process([np.array([0.1, 0.1])], dt=1.0)

    def test_mht_hypothesis_summary_logged_when_enabled(self):
        import re

        from loguru import logger as _l

        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            self._run_mht_scenario()
        finally:
            _l.remove(handle)
            disable_debug_logging()
        text = " ".join(str(r) for r in records)
        # Match the real payload (counts + best_score), not just the static
        # "hypotheses" literal in the format string -- a garbled or empty
        # payload must fail this.
        assert re.search(r"\d+ hypotheses, pruned \d+, best_score=[-\d.eE+]+", text)

    def test_mht_zero_records_when_disabled(self):
        from loguru import logger as _l

        records = []
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            self._run_mht_scenario()
        finally:
            _l.remove(handle)
        assert records == []


class TestFilterHealth:
    def test_health_logged_and_warning_on_symptoms(self):
        from loguru import logger as _l

        from pytcl.diagnostics import log_filter_health

        records = []
        enable_debug_logging()
        handle = _l.add(
            lambda m: records.append((m.record["level"].name, m.record["message"])),
            level="DEBUG",
        )
        try:
            log_filter_health(
                1, nis_value=1.0, nis_window=[1.0] * 5, cov_condition=10.0
            )
            log_filter_health(
                2, nis_value=99.0, nis_window=[50.0] * 5, cov_condition=1e15
            )
        finally:
            _l.remove(handle)
            disable_debug_logging()
        levels = [lvl for lvl, _ in records]
        assert "DEBUG" in levels and "WARNING" in levels

    def test_behavioral_neutrality(self):
        # Identical numerical results with diagnostics enabled vs disabled.
        import numpy as np

        from pytcl.trackers import MultiTargetTracker

        def run():
            rng = np.random.default_rng(11)
            tracker = MultiTargetTracker(
                state_dim=4,
                meas_dim=2,
                F=np.array(
                    [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                    dtype=float,
                ),
                H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
                Q=np.eye(4) * 0.01,
                R=np.eye(2) * 1.0,
            )
            out = []
            for k in range(20):
                z = [np.array([k + rng.normal(0, 0.5), k + rng.normal(0, 0.5)])]
                out.append(tracker.process(z, dt=1.0))
            return out

        baseline = run()
        enable_debug_logging()
        try:
            with_diag = run()
        finally:
            disable_debug_logging()
        for a, b in zip(baseline, with_diag):
            for ta, tb in zip(a, b):
                np.testing.assert_array_equal(ta.state, tb.state)
                np.testing.assert_array_equal(ta.covariance, tb.covariance)
