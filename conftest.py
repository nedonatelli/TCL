"""Repo-level pytest configuration.

Applies to doctests collected from pytcl/ (via --doctest-modules), which the
tests/conftest.py does not cover.
"""

from typing import Iterator

import numpy as np
import pytest

_BACKENDS_KEY = pytest.StashKey[list]()


# Everything in pytcl/gpu goes through get_compute_backend(), which raises
# DependencyError unless CuPy or MLX is importable -- there is no NumPy
# fallback. So on a machine with neither, these doctests cannot run at all and
# collecting them would report 35 failures that say nothing about the code.
#
# The gate used to test for CuPy alone. That skipped the whole package on
# Apple Silicon, where MLX is a working backend and the examples do run --
# so the developers most able to exercise this code were the ones getting no
# feedback from it (gh-66). It now tests for any backend.
#
# CI has neither and still skips. That is a real limit rather than an
# oversight: GPU code cannot be doctested on a runner without a GPU. What CI
# does cover is the CPU-side contracts, in tests/validation/test_gpu_audit.py.
def _has_compute_backend() -> bool:
    for module in ("cupy", "mlx.core"):
        try:
            __import__(module)
            return True
        except Exception:
            continue
    return False


collect_ignore_glob = [] if _has_compute_backend() else ["pytcl/gpu/*"]


# ---------------------------------------------------------------------------
# Backend coverage
# ---------------------------------------------------------------------------
#
# 149 tests do not execute on a machine without MLX, and nothing announces it.
# No CI runner has MLX, so those 149 skip on every runner and the build is
# green regardless of their state. That is how 36 broken tests reached main:
# the callback-contract refactor missed three files under tests/unit/, every
# runner skipped them, and the only machine that runs them is a developer's.
#
# Making the absence of MLX an error unconditionally would break CI, which
# legitimately has no MLX. So the strictness is opt-in: set PYTCL_REQUIRE_MLX=1
# and a run that cannot exercise the MLX layer aborts instead of quietly
# passing. That is the pre-merge command on an Apple Silicon machine:
#
#     PYTCL_REQUIRE_MLX=1 pytest
#
# It cannot be satisfied by accident, which is the property the plain suite
# lacks.


def _backend_names() -> "list[str]":
    present = []
    for label, module in (("mlx", "mlx.core"), ("cupy", "cupy")):
        try:
            __import__(module)
            present.append(label)
        except Exception:
            continue
    return present


def pytest_configure(config: "pytest.Config") -> None:
    import os

    present = _backend_names()
    config.stash[_BACKENDS_KEY] = present

    if os.environ.get("PYTCL_REQUIRE_MLX") == "1" and "mlx" not in present:
        raise pytest.UsageError(
            "PYTCL_REQUIRE_MLX=1 but MLX is not importable, so the MLX layer "
            "would be skipped and this run would pass without exercising it. "
            "Run on Apple Silicon with MLX installed, or unset the variable "
            "and accept that the MLX-gated tests are not covered."
        )
    if os.environ.get("PYTCL_REQUIRE_CUPY") == "1" and "cupy" not in present:
        raise pytest.UsageError(
            "PYTCL_REQUIRE_CUPY=1 but CuPy is not importable, so the CuPy "
            "device layer would be skipped and this run would pass without "
            "exercising it."
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Say plainly when a GPU layer did not run.

    Not pytest_report_header: addopts carries -q permanently, which suppresses
    the header entirely. The terminal summary still prints, and it is where
    the skip counts already appear.

    Only reports absence. A run with everything present says nothing, so the
    message means "a layer was not exercised" rather than being noise every
    developer learns to scroll past.
    """
    present = config.stash.get(_BACKENDS_KEY, None)
    if present is None:
        present = _backend_names()
    missing = [b for b in ("mlx", "cupy") if b not in present]
    if not missing:
        return
    terminalreporter.write_sep("=", "gpu backend coverage", yellow=True)
    for backend in missing:
        var = "PYTCL_REQUIRE_MLX" if backend == "mlx" else "PYTCL_REQUIRE_CUPY"
        terminalreporter.write_line(
            f"{backend} is not installed: that layer was SKIPPED, not verified. "
            f"Set {var}=1 to make this an error instead of a silent pass."
        )


@pytest.fixture(autouse=True)
def _numpy_legacy_scalar_repr() -> Iterator[None]:
    """Print NumPy scalars as plain values (1.0, True) in doctests.

    NumPy 2 changed scalar reprs to np.float64(1.0)/np.True_, which would
    require version-specific expected output in every docstring example.
    """
    old = np.get_printoptions()["legacy"]
    np.set_printoptions(legacy="1.25")
    yield
    np.set_printoptions(legacy=old)
