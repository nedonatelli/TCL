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
