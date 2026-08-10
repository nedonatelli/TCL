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
