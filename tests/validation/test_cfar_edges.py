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


@pytest.mark.parametrize("kernel", [det._cfar_so_kernel, _py(det._cfar_so_kernel)])
def test_so_cfar_nan_in_reference_window_propagates(kernel):
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    noise, thr = _run_1d(kernel, x, 1, 1, 3.0)
    # Only cell 3 has the NaN (index 1) in a reference half; the rest are the
    # min of the two single-cell halves, or the lone half at the edges.
    np.testing.assert_array_equal(noise, [3.0, 4.0, 1.0, np.nan, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(thr, [9.0, 12.0, 3.0, np.nan, 9.0, 12.0, 15.0])


def test_so_cfar_reports_no_detection_at_nan_threshold_cell():
    x = np.array([1.0, np.nan, 3.0, 1e9, 5.0, 6.0, 7.0])
    result = cfar_so(x, guard_cells=1, ref_cells=1, pfa=1e-3, alpha=3.0)
    assert np.isnan(result.threshold[3])
    assert not result.detections[3]
    assert 3 not in result.detection_indices


@pytest.mark.parametrize("kernel", [det._cfar_so_kernel, _py(det._cfar_so_kernel)])
def test_so_cfar_empty_reference_window_gives_zero_noise_and_threshold(kernel):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    noise, thr = _run_1d(kernel, x, 2, 2, 3.0)
    np.testing.assert_array_equal(noise, [4.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(thr, [12.0, 0.0, 0.0, 3.0])


def test_so_cfar_flags_every_positive_sample_in_an_empty_window():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = cfar_so(x, guard_cells=2, ref_cells=2, pfa=1e-3, alpha=3.0)
    assert result.detections[1] and result.detections[2]


def test_so_cfar_2d_jit_and_python_paths_agree_with_nan():
    img = np.abs(np.random.default_rng(1).normal(size=(12, 12))) + 1.0
    img[5, 5] = np.nan
    jit = _run_2d(det._cfar_2d_so_kernel, img, (1, 1), (2, 2), 3.0)
    plain = _run_2d(_py(det._cfar_2d_so_kernel), img, (1, 1), (2, 2), 3.0)
    assert np.isnan(jit[1]).any()
    assert np.allclose(jit[0], plain[0], equal_nan=True)
    assert np.allclose(jit[1], plain[1], equal_nan=True)
