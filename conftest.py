"""Repo-level pytest configuration.

Applies to doctests collected from pytcl/ (via --doctest-modules), which the
tests/conftest.py does not cover.
"""

from typing import Iterator

import numpy as np
import pytest

# GPU compute doctests require CuPy specifically: the batch filter examples
# are CuPy-gated even on Apple Silicon (MLX covers only transfer/detection
# utilities today — see AUDIT.md). Collect them only when CuPy is importable.
try:
    import cupy  # noqa: F401

    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

collect_ignore_glob = [] if _HAS_CUPY else ["pytcl/gpu/*"]


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
