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
from hypothesis import given
from hypothesis import strategies as st

from pytcl.mathematical_functions.signal_processing import (
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
)
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


def test_so_cfar_empty_reference_window_gives_nan_noise_and_threshold():
    # guard 2 on 4 samples: cells 1 and 2 have no reference cell; cells 0
    # and 3 keep their one half (x[3] and x[0]).
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = cfar_so(x, guard_cells=2, ref_cells=2, pfa=1e-3, alpha=3.0)
    np.testing.assert_array_equal(result.noise_estimate, [4.0, np.nan, np.nan, 1.0])
    np.testing.assert_array_equal(result.threshold, [12.0, np.nan, np.nan, 3.0])


def test_so_cfar_never_flags_a_sample_in_an_empty_window():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = cfar_so(x, guard_cells=2, ref_cells=2, pfa=1e-3, alpha=3.0)
    assert not result.detections[1] and not result.detections[2]
    np.testing.assert_array_equal(result.detection_indices, [3])


def test_so_cfar_2d_jit_and_python_paths_agree_with_nan():
    img = np.abs(np.random.default_rng(1).normal(size=(12, 12))) + 1.0
    img[5, 5] = np.nan
    jit = _run_2d(det._cfar_2d_so_kernel, img, (1, 1), (2, 2), 3.0)
    plain = _run_2d(_py(det._cfar_2d_so_kernel), img, (1, 1), (2, 2), 3.0)
    assert np.isnan(jit[1]).any()
    assert np.allclose(jit[0], plain[0], equal_nan=True)
    assert np.allclose(jit[1], plain[1], equal_nan=True)


@pytest.mark.parametrize("kernel", [det._cfar_go_kernel, _py(det._cfar_go_kernel)])
@pytest.mark.parametrize(
    "x, noise",
    [
        # NaN in cell 3's leading half.
        (
            [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, np.nan, 7.0, 4.0, 5.0],
        ),
        # NaN in cell 3's lagging half: Python's max(2.0, nan) is 2.0.
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0],
            [3.0, 4.0, 5.0, np.nan, 7.0, 4.0, 5.0],
        ),
    ],
    ids=["leading", "lagging"],
)
def test_go_cfar_nan_in_either_reference_half_propagates(kernel, x, noise):
    # guard 1, ref 1: cell i's halves are x[i-2] and x[i+2]; a missing half
    # counts as 0.0, so the edge cells take the lone half.
    got_noise, got_thr = _run_1d(kernel, np.array(x), 1, 1, 3.0)
    np.testing.assert_array_equal(got_noise, noise)
    np.testing.assert_array_equal(got_thr, 3.0 * np.array(noise))


@pytest.mark.parametrize(
    "kernel", [det._cfar_2d_go_kernel, _py(det._cfar_2d_go_kernel)]
)
def test_go_cfar_2d_nan_in_lagging_half_propagates(kernel):
    # Cell (2, 2), guard (1, 1), ref (1, 1): the leading half is row 0 (mean
    # 1.0), the lagging half holds row 4, where the NaN sits.
    img = np.ones((5, 5))
    img[4, 2] = np.nan
    noise, thr = _run_2d(kernel, img, (1, 1), (1, 1), 3.0)
    assert np.isnan(noise[2, 2]) and np.isnan(thr[2, 2])


def test_go_cfar_reports_no_detection_at_nan_threshold_cell():
    x = np.array([1.0, 2.0, 3.0, 1e9, 5.0, np.nan, 7.0])
    result = cfar_go(x, guard_cells=1, ref_cells=1, alpha=3.0)
    assert np.isnan(result.threshold[3])
    assert not result.detections[3]


# =============================================================================
# Empty reference windows: NaN threshold, never a detection
# =============================================================================


def _brute_force_empty(n, guard, ref):
    """Count each cell's reference cells by enumerating the signal."""
    return np.array(
        [
            not any(guard < abs(j - i) <= guard + ref for j in range(n))
            for i in range(n)
        ],
        dtype=bool,
    )


def _brute_force_empty_2d(shape, guard, ref):
    """Count each cell's reference cells by enumerating the image."""
    n_rows, n_cols = shape
    out = np.empty(shape, dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            out[i, j] = not any(
                abs(r - i) <= guard[0] + ref[0]
                and abs(c - j) <= guard[1] + ref[1]
                and not (abs(r - i) <= guard[0] and abs(c - j) <= guard[1])
                for r in range(n_rows)
                for c in range(n_cols)
            )
    return out


@pytest.mark.parametrize("fn", [cfar_ca, cfar_go, cfar_so, cfar_os])
def test_empty_window_is_nan_and_never_a_detection(fn):
    # 2.11.1: cell 2 had threshold 0.0 and was a detection in all four.
    r = fn(np.array([1.0, 1.0, 5.0, 1.0, 1.0]), guard_cells=2, ref_cells=1, pfa=1e-3)
    assert np.isnan(r.threshold[2]) and np.isnan(r.noise_estimate[2])
    assert not r.detections[2] and 2 not in r.detection_indices
    assert np.all(np.isfinite(np.delete(r.threshold, 2)))
    assert np.all(np.isfinite(np.delete(r.noise_estimate, 2)))


@pytest.mark.parametrize("fn", [cfar_ca, cfar_go, cfar_os])
def test_window_that_never_fits_gives_all_nan(fn):
    r = fn(np.array([1.0, 2.0, 3.0]), guard_cells=5, ref_cells=2, pfa=1e-3)
    assert np.isnan(r.threshold).all() and np.isnan(r.noise_estimate).all()
    assert not r.detections.any() and r.detection_indices.size == 0


@pytest.mark.parametrize("method", ["ca", "go", "so"])
def test_2d_empty_window_is_nan(method):
    # 2.11.1 ('ca'): row 2 threshold [8.0045, 8.0045, 0., 8.0045, 8.0045].
    img = np.ones((5, 5))
    img[2, 2] = 3.0
    r = cfar_2d(img, guard_cells=(2, 2), ref_cells=(1, 1), pfa=1e-3, method=method)
    assert np.isnan(r.threshold[2, 2]) and np.isnan(r.noise_estimate[2, 2])
    assert not r.detections[2, 2]
    others = np.ones((5, 5), dtype=bool)
    others[2, 2] = False
    assert np.all(np.isfinite(r.threshold[others]))
    assert np.all(np.isfinite(r.noise_estimate[others]))


@given(n=st.integers(1, 40), guard=st.integers(0, 8), ref=st.integers(0, 8))
def test_mask_equals_a_brute_force_count(n, guard, ref):
    assert np.array_equal(
        det._empty_window_mask_1d(n, guard, ref), _brute_force_empty(n, guard, ref)
    )


@given(
    shape=st.tuples(st.integers(1, 9), st.integers(1, 9)),
    guard=st.tuples(st.integers(0, 5), st.integers(0, 5)),
    ref=st.tuples(st.integers(0, 4), st.integers(0, 4)),
)
def test_2d_mask_equals_a_brute_force_count(shape, guard, ref):
    assert np.array_equal(
        det._empty_window_mask_2d(shape, guard, ref),
        _brute_force_empty_2d(shape, guard, ref),
    )


_KERNEL_NAMES = [
    "_cfar_ca_kernel",
    "_cfar_go_kernel",
    "_cfar_so_kernel",
    "_cfar_os_kernel",
    "_cfar_2d_ca_kernel",
    "_cfar_2d_go_kernel",
    "_cfar_2d_so_kernel",
]


def _public_results():
    rng = np.random.default_rng(20261027)
    x = rng.exponential(1.0, 40)
    x_nan = x.copy()
    x_nan[[5, 20]] = np.nan
    img = rng.exponential(1.0, (9, 11))
    img_nan = img.copy()
    img_nan[4, 5] = np.nan
    out = []
    for sig in (x, x_nan, x[:5]):
        for guard, ref in [(2, 4), (2, 1), (0, 3), (25, 25), (3, 0)]:
            alpha = 3.5
            for fn in (cfar_ca, cfar_go, cfar_so):
                if fn is cfar_so and len(sig) < guard + 2:
                    continue
                out.append(fn(sig, guard, ref, alpha=alpha))
            out.append(cfar_os(sig, guard, ref, k=max(1, ref), alpha=alpha))
    for im in (img, img_nan, img[:2, :5]):
        for guard, ref in [((1, 1), (2, 2)), ((2, 2), (1, 1)), ((9, 9), (1, 1))]:
            for method in ("ca", "go", "so"):
                if method == "so" and im.shape[0] < guard[0] + 2:
                    if im.shape[1] < guard[1] + 2:
                        continue
                out.append(cfar_2d(im, guard, ref, method=method, alpha=3.5))
    return out


@pytest.mark.skipif(
    _py(det._cfar_ca_kernel) is det._cfar_ca_kernel,
    reason="NUMBA_DISABLE_JIT=1: no compiled path to compare",
)
def test_compiled_and_interpreted_paths_agree_on_the_nan_cells(monkeypatch):
    compiled = _public_results()
    for name in _KERNEL_NAMES:
        monkeypatch.setattr(det, name, getattr(det, name).py_func)
    interpreted = _public_results()
    assert len(compiled) == len(interpreted)
    n_nan = 0
    for c, p in zip(compiled, interpreted):
        # fastmath reassociates the window sums, so finite values agree to
        # rounding; the NaN cells and the detections agree exactly.
        for field in ("threshold", "noise_estimate"):
            np.testing.assert_allclose(
                getattr(c, field), getattr(p, field), rtol=1e-12, atol=0
            )
        np.testing.assert_array_equal(c.detections, p.detections, strict=True)
        n_nan += int(np.isnan(c.threshold).sum())
    assert n_nan > 0
