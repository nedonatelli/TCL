"""Repo-level pytest configuration.

Applies to doctests collected from pytcl/ (via --doctest-modules), which the
tests/conftest.py does not cover.
"""

from typing import Iterator

import numpy as np
import pytest

# GPU doctests require a CuPy/MLX backend; mirror the skip behavior of
# tests/test_gpu.py at collection time so --doctest-modules works everywhere.
try:
    import cupy  # noqa: F401

    _HAS_GPU_BACKEND = True
except Exception:
    try:
        import mlx.core  # noqa: F401

        _HAS_GPU_BACKEND = True
    except Exception:
        _HAS_GPU_BACKEND = False

collect_ignore_glob = [] if _HAS_GPU_BACKEND else ["pytcl/gpu/*"]


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
