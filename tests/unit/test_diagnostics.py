"""pytcl.diagnostics: silence by default, enable/disable, instrumentation."""

import subprocess
import sys

import pytest
from loguru import logger as _loguru_logger

import pytcl
from pytcl.core.exceptions import DependencyError
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
    def test_round_trip(self):
        from pytcl.diagnostics import log_filter_health

        records = []
        enable_debug_logging()
        assert diagnostics_enabled() is True
        handle = _loguru_logger.add(records.append, format="{message}")
        try:
            # log_filter_health lives in pytcl.diagnostics, so this is a
            # real emission through the pytcl namespace -- not a `.bind()`
            # on an unrelated `name` key, which never touches the
            # `record["name"]` the enable/disable filter actually checks.
            log_filter_health(1, nis_value=0.1, nis_window=[0.1], cov_condition=1.0)
            assert len(records) == 1  # enabled: the emission reached the sink
        finally:
            _loguru_logger.remove(handle)
        disable_debug_logging()
        assert diagnostics_enabled() is False

        records2 = []
        handle2 = _loguru_logger.add(records2.append, format="{message}")
        try:
            log_filter_health(1, nis_value=0.1, nis_window=[0.1], cov_condition=1.0)
        finally:
            _loguru_logger.remove(handle2)
        assert records2 == []  # disabled: the emission never reached the sink

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


class TestStderrRealityAndLifecycle:
    """OS-level (capfd) checks of the actual bytes on stderr, plus the
    lifecycle regressions the reviewer probed for: a leaked default
    handler double-printing every record, a bad level leaving the
    namespace half-enabled, and a host application ripping our handler
    out from under us."""

    def test_single_emission_is_exactly_one_stderr_line_then_silent(self, capfd):
        from pytcl.diagnostics import log_filter_health

        enable_debug_logging()
        log_filter_health(7, nis_value=0.25, nis_window=[0.25], cov_condition=1.0)
        out, err = capfd.readouterr()
        assert out == ""
        # Exactly one line for the record -- a leaked default handler (id 0)
        # alongside pytcl's own would print this substring twice.
        assert err.count("track 7") == 1
        assert "pytcl.diagnostics" in err  # {name} renders the full module path

        disable_debug_logging()
        log_filter_health(7, nis_value=0.25, nis_window=[0.25], cov_condition=1.0)
        out2, err2 = capfd.readouterr()
        assert out2 == ""
        assert err2 == ""

    def test_bad_level_raises_and_leaves_state_fully_off(self):
        with pytest.raises(ValueError):
            enable_debug_logging(level="NOT_A_REAL_LEVEL")
        assert diagnostics_enabled() is False

        records = []
        handle = _loguru_logger.add(records.append, format="{message}")
        try:
            from pytcl.diagnostics import log_filter_health

            log_filter_health(9, nis_value=0.1, nis_window=[0.1], cov_condition=1.0)
        finally:
            _loguru_logger.remove(handle)
        assert records == []  # namespace still disabled -> no sink reached

    def test_host_removing_our_handler_then_disable_reenable_does_not_raise(self):
        import pytcl.diagnostics as diag_module

        enable_debug_logging()
        # Simulate a host application that reaches into loguru directly and
        # removes pytcl's handler out from under it.
        _loguru_logger.remove(diag_module._handler_id)

        disable_debug_logging()  # must not raise despite the handler being gone
        assert diagnostics_enabled() is False

        enable_debug_logging()  # must not raise either
        assert diagnostics_enabled() is True
        disable_debug_logging()


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
            with pytest.raises((FileNotFoundError, DependencyError)):
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
            with pytest.raises((FileNotFoundError, DependencyError)):
                load_gebco(0.0, 0.1, 0.0, 0.1)
        finally:
            _l.remove(handle)
        assert records == []  # disabled namespace emits nothing

    def test_gebco_fixture_exercises_found_and_parse_records(
        self, tmp_path, monkeypatch
    ):
        """Real GEBCO files are ~7.5 GB, so no CI runner has one and the
        "found" + parse start/finish DEBUG records (and the
        ``progress=True`` log-pair fallback) never execute end-to-end in
        CI -- only ever against a developer's local download. A tiny
        synthetic NetCDF satisfying ``parse_gebco_netcdf``'s schema
        (tests/fixtures/terrain/synthetic_gebco.nc, see that directory's
        SOURCES.md) lets the real loader run this path in CI permanently.
        """
        pytest.importorskip("netCDF4")
        import shutil
        from pathlib import Path

        import numpy as np
        from loguru import logger as _l

        from pytcl.terrain.loaders import _load_gebco_cached, load_gebco

        fixture = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "terrain"
            / "synthetic_gebco.nc"
        )
        # _find_gebco_file's first pattern for the default version is
        # "{version}.nc" -- name the copy so the loader finds it unaided.
        shutil.copy(fixture, tmp_path / "GEBCO2025.nc")
        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        _load_gebco_cached.cache_clear()

        lat_min, lat_max = np.radians(10.0), np.radians(12.0)
        lon_min, lon_max = np.radians(20.0), np.radians(22.0)

        # Plain (non-diagnostics, cached) load succeeds against the real
        # loader and parser -- no special-casing for the synthetic file.
        dem = load_gebco(lat_min, lat_max, lon_min, lon_max)
        assert dem.data.shape == (21, 21)

        # progress=True bypasses the load cache, so this is a fresh parse
        # even though the call above already populated the cache.
        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            dem2 = load_gebco(lat_min, lat_max, lon_min, lon_max, progress=True)
        finally:
            _l.remove(handle)
            disable_debug_logging()
        assert dem2.data.shape == (21, 21)

        text = " ".join(str(r) for r in records)
        assert "GEBCO file found" in text
        assert "GEBCO read starting" in text
        assert "GEBCO read finished" in text


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

    def test_warning_from_nis_outlier_alone(self):
        # Isolate the NIS-outlier branch: cov_condition stays benign, so a
        # WARNING here can only come from nis_value >> mean(nis_window).
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
                3, nis_value=100.0, nis_window=[1.0] * 5, cov_condition=10.0
            )
        finally:
            _l.remove(handle)
            disable_debug_logging()
        levels = [lvl for lvl, _ in records]
        assert levels == ["WARNING"]

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
