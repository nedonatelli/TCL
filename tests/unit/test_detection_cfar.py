"""Oracle-based coverage for detection.py's private kernels and edge behavior.

``tests/unit/test_cfar_detection.py`` already exercises the public API with
mostly smoke-style assertions (shapes, dtypes, "detects a strong target").
This file targets what that one does not: closed-form/Monte-Carlo checks on
the Pfa solvers (``_pfa_so``, ``_pfa_go``, ``_pfa_os``, ``_solve_alpha``), the
documented CA/GO/SO/OS discriminating behavior on a shared synthetic scene,
and the 2D window-clipping arithmetic at image edges -- all driven through
the public API, since the kernels are private.

Every Monte Carlo tolerance below is derived from the binomial standard
deviation of a false-alarm count over ``n_trials`` independent draws:
``sigma_count = sqrt(n_trials * pfa * (1 - pfa))``, so the measured rate's
standard deviation is ``sigma_count / n_trials``. A 3-sigma band is used.
"""

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from pytcl.mathematical_functions.signal_processing.detection import (
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    cluster_detections,
    snr_loss,
    threshold_factor,
)

# =============================================================================
# Independent reference (oracle) window arithmetic
#
# These re-derive the noise estimate from the documented window definition
# (guard-cell-excluded annulus around the cell under test, clamped to array
# bounds) using plain NumPy/Python, independently of the njit kernels under
# test. Any indexing or clipping bug in the kernels shows up as a mismatch.
# =============================================================================


def _ref_noise_1d(signal, i, guard_cells, ref_cells, method):
    n = len(signal)
    half_window = guard_cells + ref_cells
    left = signal[max(0, i - half_window) : max(0, i - guard_cells)]
    right = signal[min(n, i + guard_cells + 1) : min(n, i + half_window + 1)]
    if method == "ca":
        combined = np.concatenate([left, right])
        return float(combined.mean()) if len(combined) else 0.0
    left_avg = float(left.mean()) if len(left) else np.inf
    right_avg = float(right.mean()) if len(right) else np.inf
    if method == "go":
        return (
            max(left_avg, right_avg) if np.isfinite(max(left_avg, right_avg)) else 0.0
        )
    if method == "so":
        val = min(left_avg, right_avg)
        return 0.0 if val == np.inf else val
    raise ValueError(method)


def _ref_noise_os_1d(signal, i, guard_cells, ref_cells, k):
    n = len(signal)
    half_window = guard_cells + ref_cells
    left = signal[max(0, i - half_window) : max(0, i - guard_cells)]
    right = signal[min(n, i + guard_cells + 1) : min(n, i + half_window + 1)]
    combined = np.sort(np.concatenate([left, right]))
    idx = min(k - 1, len(combined) - 1)
    return float(combined[idx])


def _ref_noise_2d(image, i, j, guard_rows, guard_cols, ref_rows, ref_cols, method):
    n_rows, n_cols = image.shape
    half_row, half_col = guard_rows + ref_rows, guard_cols + ref_cols
    row_min, row_max = max(0, i - half_row), min(n_rows, i + half_row + 1)
    col_min, col_max = max(0, j - half_col), min(n_cols, j + half_col + 1)
    g_row_min, g_row_max = max(0, i - guard_rows), min(n_rows, i + guard_rows + 1)
    g_col_min, g_col_max = max(0, j - guard_cols), min(n_cols, j + guard_cols + 1)

    if method == "ca":
        window = image[row_min:row_max, col_min:col_max]
        mask = np.ones(window.shape, dtype=bool)
        mask[
            g_row_min - row_min : g_row_max - row_min,
            g_col_min - col_min : g_col_max - col_min,
        ] = False
        return float(window[mask].mean())

    if method == "so":
        top, bottom = [], []
        for ri in range(row_min, row_max):
            for ci in range(col_min, col_max):
                if g_row_min <= ri < g_row_max and g_col_min <= ci < g_col_max:
                    continue
                (top if ri < i else bottom).append(image[ri, ci])
        top_avg = float(np.mean(top)) if top else np.inf
        bottom_avg = float(np.mean(bottom)) if bottom else np.inf
        val = min(top_avg, bottom_avg)
        return 0.0 if val == np.inf else val

    raise ValueError(method)


# =============================================================================
# threshold_factor: closed form (CA) and Monte-Carlo inversion (GO/SO/OS)
# =============================================================================


class TestThresholdFactorClosedForm:
    def test_ca_matches_closed_form(self):
        # CA-CFAR: Pfa = (1 + alpha/N)^-N  =>  alpha = N*(Pfa^(-1/N) - 1)
        n, pfa = 16, 1e-4
        expected = n * (pfa ** (-1.0 / n) - 1.0)
        assert threshold_factor(pfa, n, method="ca") == pytest.approx(expected)

    def test_os_default_k_is_resolved_inside_threshold_factor(self):
        # threshold_factor's own k=None branch (as opposed to cfar_os's,
        # which resolves k before calling threshold_factor): default is
        # 0.75 * n_ref, rounded down.
        n_ref = 20
        alpha_default = threshold_factor(1e-3, n_ref, method="os")
        alpha_explicit = threshold_factor(1e-3, n_ref, method="os", k=15)
        assert alpha_default == pytest.approx(alpha_explicit)

    @pytest.mark.parametrize("method", ["so", "go", "os"])
    def test_alpha_reproduces_target_pfa(self, method):
        # Inverse check: alpha from the solver, plugged back into a Monte
        # Carlo simulation of the same statistic threshold_factor solved
        # for, must reproduce the requested Pfa on exponential (unit-mean)
        # noise. GO/SO use n_ref // 2 cells per half window; OS uses all
        # n_ref cells and its k-th order statistic.
        pfa = 1e-3
        n_ref = 16
        n_trials = 1_000_000
        rng = Generator(PCG64(20260816))

        if method == "os":
            k = int(0.75 * n_ref)
            alpha = threshold_factor(pfa, n_ref, method="os", k=k)
            ref_cells = rng.exponential(1.0, size=(n_trials, n_ref))
            ref_cells.sort(axis=1)
            noise_stat = ref_cells[:, k - 1]
        else:
            n_half = n_ref // 2
            alpha = threshold_factor(pfa, n_ref, method=method)
            left = rng.exponential(1.0, size=(n_trials, n_half)).mean(axis=1)
            right = rng.exponential(1.0, size=(n_trials, n_half)).mean(axis=1)
            noise_stat = (
                np.maximum(left, right) if method == "go" else np.minimum(left, right)
            )

        cell_under_test = rng.exponential(1.0, size=n_trials)
        false_alarms = int(np.sum(cell_under_test > alpha * noise_stat))
        measured_pfa = false_alarms / n_trials

        # Binomial std dev of the false-alarm COUNT over n_trials draws is
        # sqrt(n_trials * pfa * (1 - pfa)); dividing by n_trials gives the
        # std dev of the measured RATE. Allow a 3-sigma band.
        sigma_rate = np.sqrt(pfa * (1 - pfa) / n_trials)
        assert abs(measured_pfa - pfa) < 3 * sigma_rate


# =============================================================================
# CA/GO/SO/OS discriminating behavior on a shared synthetic scene
# =============================================================================


class TestCFARVariantsOnSyntheticScenes:
    """One noise floor, a clutter edge, and two injected targets.

    Background is a two-level clutter step (edge at index 150: 1.0 before,
    8.0 after) with:

    - a clutter transient at the edge (index 150, value 45.0) that is not a
      real target -- used to show GO suppresses the false alarm CA raises
      there;
    - a target close enough to the edge (index 142, value 18.0) that its
      reference window straddles both clutter levels -- used to show SO
      recovers it where CA and GO do not;
    - an interferer/target pair (index 60 = 40.0, index 70 = 12.0) inside
      the homogeneous low-clutter region -- used to show OS survives the
      interferer where CA is masked by it.

    All four detectors share guard_cells=2, ref_cells=10, pfa=1e-3, so the
    comparison is apples-to-apples. Noise-estimate values are cross-checked
    against the independent reference window arithmetic (`_ref_noise_1d`),
    not just the resulting detection booleans.
    """

    guard_cells = 2
    ref_cells = 10
    pfa = 1e-3

    @pytest.fixture
    def scene(self):
        n = 300
        signal = np.where(np.arange(n) < 150, 1.0, 8.0).astype(np.float64)
        signal[150] = 45.0
        signal[142] = 18.0
        signal[60] = 40.0
        signal[70] = 12.0

        kwargs = dict(
            guard_cells=self.guard_cells, ref_cells=self.ref_cells, pfa=self.pfa
        )
        results = {
            "ca": cfar_ca(signal, **kwargs),
            "go": cfar_go(signal, **kwargs),
            "so": cfar_so(signal, **kwargs),
            "os": cfar_os(signal, **kwargs),
        }
        return signal, results

    def test_go_suppresses_clutter_edge_false_alarm_ca_raises(self, scene):
        signal, r = scene
        idx = 150

        expected_ca_noise = _ref_noise_1d(
            signal, idx, self.guard_cells, self.ref_cells, "ca"
        )
        expected_go_noise = _ref_noise_1d(
            signal, idx, self.guard_cells, self.ref_cells, "go"
        )
        assert r["ca"].noise_estimate[idx] == pytest.approx(expected_ca_noise)
        assert r["go"].noise_estimate[idx] == pytest.approx(expected_go_noise)

        # CA blends the low-clutter left window with the partially
        # high-clutter right window into a noise estimate below the true
        # local (high-clutter) level, so the 45.0 transient clears its
        # threshold. GO takes the max of the two half-window averages,
        # which is dominated by the high-clutter side and correctly
        # thresholds above 45.0.
        assert r["ca"].detections[idx]
        assert not r["go"].detections[idx]

    def test_so_recovers_target_masked_by_ca_and_go_near_edge(self, scene):
        signal, r = scene
        idx = 142

        expected_so_noise = _ref_noise_1d(
            signal, idx, self.guard_cells, self.ref_cells, "so"
        )
        assert r["so"].noise_estimate[idx] == pytest.approx(expected_so_noise)

        # At idx=142 the right half-window is already inside the
        # high-clutter region, so CA's blended mean and GO's max are both
        # pulled up enough to threshold above the 18.0 target. SO takes
        # the min of the two half-window averages -- the untouched
        # low-clutter left half -- so its threshold stays low enough to
        # detect it.
        assert not r["ca"].detections[idx]
        assert not r["go"].detections[idx]
        assert r["so"].detections[idx]

    def test_os_survives_interfering_target_that_masks_ca(self, scene):
        signal, r = scene
        idx = 70
        k_os = int(0.75 * 2 * self.ref_cells)  # cfar_os's own default k

        expected_ca_noise = _ref_noise_1d(
            signal, idx, self.guard_cells, self.ref_cells, "ca"
        )
        expected_os_noise = _ref_noise_os_1d(
            signal, idx, self.guard_cells, self.ref_cells, k_os
        )
        assert r["ca"].noise_estimate[idx] == pytest.approx(expected_ca_noise)
        assert r["os"].noise_estimate[idx] == pytest.approx(expected_os_noise)

        # The interferer at idx=60 sits inside idx=70's reference window
        # and pulls CA's mean-based noise estimate up enough to mask the
        # 12.0 target. OS-CFAR's order statistic (k = 0.75 * n_ref) is a
        # near-top-of-window rank that a single outlier does not reach,
        # so its noise estimate -- and threshold -- stay near the true
        # 1.0 noise floor.
        assert not r["ca"].detections[idx]
        assert r["os"].detections[idx]


# =============================================================================
# 2D CFAR: window-clipping arithmetic at image edges
# =============================================================================


class TestCfar2D:
    """CA and SO methods on a small range-Doppler image (OS is not
    implemented for cfar_2d -- see test_cfar_2d_os_method_not_implemented).

    The background is a row-dependent gradient (not homogeneous), so a
    window clipped by the image boundary genuinely averages a different set
    of cells than an unclipped one -- exercising the row_min/col_min clamps
    rather than merely running through them.
    """

    guard_cells = (1, 1)
    ref_cells = (4, 4)
    pfa = 1e-3

    @pytest.fixture
    def image(self):
        n = 20
        img = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            img[i, :] = 1.0 + 0.1 * i
        img[0, 0] = 50.0  # corner: both row and col windows clipped
        img[0, 10] = 50.0  # edge: only the row window clipped
        img[10, 10] = 50.0  # interior: unclipped, full window
        return img

    @pytest.mark.parametrize(
        "position", ["corner", "edge", "interior"], ids=["corner", "edge", "interior"]
    )
    @pytest.mark.parametrize("method", ["ca", "so"])
    def test_window_clipping_matches_reference_arithmetic(
        self, image, method, position
    ):
        coords = {"corner": (0, 0), "edge": (0, 10), "interior": (10, 10)}
        i, j = coords[position]

        result = cfar_2d(
            image, self.guard_cells, self.ref_cells, pfa=self.pfa, method=method
        )
        expected_noise = _ref_noise_2d(
            image,
            i,
            j,
            *self.guard_cells,
            *self.ref_cells,
            method,
        )

        assert result.noise_estimate[i, j] == pytest.approx(expected_noise)
        # The injected 50.0 target clears any of these thresholds -- the
        # background gradient tops out at 1.0 + 0.1*19 = 2.9.
        assert result.detections[i, j]

    def test_cfar_2d_os_method_not_implemented(self, image):
        # cfar_2d's method dispatch only implements 'ca', 'go', 'so'; 'os'
        # falls through to the same "Unknown method" branch as a typo. This
        # documents that OS-CFAR is not available in 2D, rather than
        # assuming it exists the way the 1D functions do.
        with pytest.raises(ValueError, match="Unknown method"):
            cfar_2d(image, self.guard_cells, self.ref_cells, pfa=self.pfa, method="os")


# =============================================================================
# cluster_detections and snr_loss
# =============================================================================


class TestClusterAndLoss:
    def test_cluster_detections_collapses_runs(self):
        # cluster_detections only reads VALUES from a non-boolean input as
        # a dense per-position array (det_indices = arange(len(...))); it
        # does not treat integer entries as sparse index positions. So the
        # brief's literal `det = np.array([3, 4, 5, 20, 40, 41])` passed
        # directly would cluster on array POSITIONS (0..5, all within
        # min_separation=2 of each other) rather than on those values,
        # collapsing to a single peak -- not the 3-cluster behavior being
        # tested here. A boolean detection mask, as the function's own
        # docstring example uses, gets the intended semantics.
        det = np.array([3, 4, 5, 20, 40, 41])
        mask = np.zeros(det.max() + 1, dtype=bool)
        mask[det] = True

        peaks = cluster_detections(mask, min_separation=2)

        assert set(np.asarray(peaks).tolist()) <= set(det.tolist())
        assert len(peaks) == 3

    def test_cluster_detections_non_boolean_input_clusters_by_position(self):
        # The complementary, documented case: a dense non-boolean array is
        # clustered by its own position, not by its values.
        values = np.array([0.1, 0.2, 5.0, 6.0, 0.1])
        peaks = cluster_detections(values, min_separation=1)
        np.testing.assert_array_equal(peaks, [2])

    def test_snr_loss_positive_and_decreasing_in_n(self):
        losses = [snr_loss(n, 1e-6) for n in (8, 16, 32, 64)]
        assert all(loss > 0 for loss in losses)
        assert losses == sorted(losses, reverse=True)
