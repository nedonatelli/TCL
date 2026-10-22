"""SO-CFAR edge cases that the numba ``fastmath`` flag used to turn into noise.

``fastmath=True`` lets the compiler assume no infinities and no NaNs, so the
``noise_est == np.inf`` sentinel never matched under the JIT and NaN handling
was undefined. The JIT returned all-infinity thresholds where the pure-Python
path (CI's ``NUMBA_DISABLE_JIT=1`` coverage cell) returned 0.0, so the covered
numeric path was not the shipped one. Both paths are exercised here through
the kernels' ``py_func``.
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.signal_processing import cfar_2d, cfar_so
from pytcl.mathematical_functions.signal_processing import detection as det


def _py(kernel):
    return getattr(kernel, "py_func", kernel)


def _run_1d(kernel, x, guard, ref, alpha):
    noise = np.zeros(len(x))
    thr = np.zeros(len(x))
    kernel(x, guard, ref, alpha, noise, thr)
    return noise, thr


def _run_2d(kernel, img, guard, ref, alpha):
    noise = np.zeros_like(img)
    thr = np.zeros_like(img)
    kernel(img, guard[0], guard[1], ref[0], ref[1], alpha, noise, thr)
    return noise, thr


@pytest.mark.parametrize(
    "x, guard, ref",
    [
        (np.array([1.0, 2.0, 3.0]), 5, 2),
        (np.array([1.0, 2.0, 3.0]), 2, 1),
    ],
)
def test_so_cfar_window_that_never_fits_raises(x, guard, ref):
    with pytest.raises(ValueError, match="reference window"):
        cfar_so(x, guard_cells=guard, ref_cells=ref, pfa=1e-3)


def test_so_cfar_2d_window_that_never_fits_raises():
    with pytest.raises(ValueError, match="reference window"):
        cfar_2d(np.ones((3, 3)), (5, 5), (2, 2), pfa=1e-3, method="so")


def test_so_cfar_jit_and_python_paths_agree_on_noise():
    x = np.abs(np.random.default_rng(0).normal(size=64)) + 1.0
    jit = _run_1d(det._cfar_so_kernel, x, 2, 8, 3.0)
    plain = _run_1d(_py(det._cfar_so_kernel), x, 2, 8, 3.0)
    assert np.allclose(jit[0], plain[0], equal_nan=True)
    assert np.allclose(jit[1], plain[1], equal_nan=True)


def test_so_cfar_nan_in_reference_window_propagates_on_both_paths():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    jit = _run_1d(det._cfar_so_kernel, x, 1, 1, 3.0)
    plain = _run_1d(_py(det._cfar_so_kernel), x, 1, 1, 3.0)
    assert np.allclose(jit[0], plain[0], equal_nan=True)
    assert np.allclose(jit[1], plain[1], equal_nan=True)
    # Cells whose reference window touches index 1 (cells 0, 2, 3 via the
    # leading/lagging side) must carry NaN, not a finite or infinite value.
    for i in (3,):
        assert np.isnan(jit[1][i])


def test_so_cfar_2d_jit_and_python_paths_agree_with_nan():
    img = np.abs(np.random.default_rng(1).normal(size=(12, 12))) + 1.0
    img[5, 5] = np.nan
    jit = _run_2d(det._cfar_2d_so_kernel, img, (1, 1), (2, 2), 3.0)
    plain = _run_2d(_py(det._cfar_2d_so_kernel), img, (1, 1), (2, 2), 3.0)
    assert np.isnan(jit[1]).any()
    assert np.allclose(jit[0], plain[0], equal_nan=True)
    assert np.allclose(jit[1], plain[1], equal_nan=True)
