"""Repo-level pytest configuration.

Applies to doctests collected from pytcl/ (via --doctest-modules), which the
tests/conftest.py does not cover.
"""

from typing import Iterator

import numpy as np
import pytest


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
