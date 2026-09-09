"""MOSPA metrics and interval scheduling against MATLAB.

Fixtures captured by scripts/matlab_capture/
capture_mospa_scheduling.m; inputs are sin-formula generated and
reconstructed here exactly. MATLAB job/order indices are 1-based and
compared after subtracting one.

Brute-force enumeration over random small instances provides the
independent optimality oracle for the scheduling algorithms, and
exact-vs-approximate agreement on well-separated targets for MMOSPA.
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from pytcl.performance_evaluation.mospa import (
    calc_mospa_error,
    mmospa2tar_2d,
    mmospa_approx,
)
from pytcl.scheduling import (
    partition_intervals,
    schedule_intervals,
    schedule_min_lateness_dense,
    schedule_weighted_intervals,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "matlab" / "mospa_scheduling.csv"


def _fixtures() -> dict:
    out = {}
    with open(FIXTURE) as f:
        for row in csv.reader(f):
            if row[0] == "label":
                continue
            out[row[0]] = np.array([float(v) for v in row[1:]])
    return out


FIX = _fixtures()

# The capture driver's sin-formula inputs, verbatim.
_K = np.arange(1, 13)
STARTS = 5 * (1 + np.sin(1.7 * _K))
ENDS = STARTS + 2 * (1.05 + np.sin(3.1 * _K))
INTERVALS = np.vstack([STARTS, ENDS])
WEIGHTS = 1.5 + np.sin(2.3 * _K)
DEADLINES = 4 * (1.1 + np.sin(1.3 * _K))
DURATIONS = 0.55 + 0.5 * np.sin(0.9 * _K)

X_DIM, NUM_TAR, NUM_HYP = 4, 3, 6
_D = np.arange(1, X_DIM + 1)[:, None, None]
_T = np.arange(1, NUM_TAR + 1)[None, :, None]
_H = np.arange(1, NUM_HYP + 1)[None, None, :]
X_HYP = 10 * np.sin(0.7 * _D + 1.9 * _T + 3.7 * _H)
W_HYP = 0.5 + 0.5 * np.sin(1.1 * np.arange(1, NUM_HYP + 1))
W_HYP = W_HYP / W_HYP.sum()
X_EST = 10 * np.sin(
    0.3 * np.arange(1, X_DIM + 1)[:, None] + 2.9 * np.arange(1, NUM_TAR + 1)[None, :]
)

_J = np.arange(1, 41)
PARTICLES = np.vstack(
    [
        2 + np.sin(1.3 * _J),
        np.sin(2.1 * _J),
        -2 + np.sin(0.7 * _J),
        np.sin(2.9 * _J),
    ]
)


class TestSchedulingAgainstMatlab:
    def test_schedule_intervals(self):
        got = schedule_intervals(INTERVALS)
        np.testing.assert_array_equal(got, FIX["schedule_intervals"] - 1)

    def test_partition_intervals(self):
        got = partition_intervals(INTERVALS)
        np.testing.assert_array_equal(got, FIX["partition_intervals"] - 1)

    def test_weighted_intervals(self):
        wv, jobs = schedule_weighted_intervals(INTERVALS, WEIGHTS)
        np.testing.assert_allclose(wv, FIX["weighted_val"][0], rtol=1e-14)
        np.testing.assert_array_equal(jobs, FIX["weighted_jobs"] - 1)

    def test_min_lateness(self):
        jobs, t_starts = schedule_min_lateness_dense(DEADLINES, DURATIONS, 0.25)
        np.testing.assert_array_equal(jobs, FIX["min_lateness_jobs"] - 1)
        np.testing.assert_allclose(t_starts, FIX["min_lateness_starts"], rtol=1e-14)


class TestMospaAgainstMatlab:
    def test_calc_mospa_error(self):
        got = calc_mospa_error(X_EST, X_HYP, W_HYP)
        np.testing.assert_allclose(got, FIX["calc_mospa_error"][0], rtol=1e-13)

    @pytest.mark.parametrize("scans", [1, 3])
    def test_mmospa_approx(self, scans):
        est, orders = mmospa_approx(X_HYP, W_HYP, scans)
        ref_est = FIX[f"mmospa_approx_est_s{scans}"].reshape(
            (X_DIM, NUM_TAR), order="F"
        )
        ref_orders = FIX[f"mmospa_approx_orders_s{scans}"].reshape(
            (NUM_TAR, NUM_HYP), order="F"
        )
        np.testing.assert_allclose(est, ref_est, rtol=1e-13)
        np.testing.assert_array_equal(orders, ref_orders - 1)

    def test_mmospa_2tar_2d(self):
        got = mmospa2tar_2d(PARTICLES)
        np.testing.assert_allclose(got, FIX["mmospa_2tar_2d"], rtol=1e-13)


class TestSchedulingOptimality:
    """Brute-force enumeration as the independent oracle."""

    def _compatible(self, starts, ends, jobs):
        js = sorted(jobs, key=lambda j: ends[j])
        return all(starts[js[i + 1]] >= ends[js[i]] for i in range(len(js) - 1))

    def test_randomized_instances(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            n = int(rng.integers(1, 9))
            starts = rng.uniform(0, 10, n)
            ends = starts + rng.uniform(0.1, 5, n)
            iv = np.vstack([starts, ends])

            sel = schedule_intervals(iv)
            best = max(
                (
                    len(c)
                    for r in range(n + 1)
                    for c in combinations(range(n), r)
                    if self._compatible(starts, ends, c)
                ),
                default=0,
            )
            assert len(sel) == best
            assert self._compatible(starts, ends, sel)

            wts = rng.uniform(0.1, 5, n)
            wv, jobs = schedule_weighted_intervals(iv, wts)
            best_w = max(
                (
                    wts[list(c)].sum()
                    for r in range(n + 1)
                    for c in combinations(range(n), r)
                    if self._compatible(starts, ends, c)
                ),
                default=0.0,
            )
            np.testing.assert_allclose(wv, best_w, rtol=1e-12)
            assert self._compatible(starts, ends, jobs)
            np.testing.assert_allclose(wts[jobs].sum(), wv, rtol=1e-12)

            part = partition_intervals(iv)
            for g in range(int(part.max()) + 1):
                assert self._compatible(starts, ends, np.where(part == g)[0])
            # The number of groups equals the maximum overlap depth.
            times = np.union1d(starts, ends)
            depth = max(
                int(((starts < tt + 1e-12) & (ends > tt + 1e-12)).sum()) for tt in times
            )
            assert int(part.max()) + 1 == depth

    def test_min_lateness_is_edf(self):
        jobs, t_starts = schedule_min_lateness_dense([6.0, 2.0, 9.0], [1.0, 2.0, 1.0])
        np.testing.assert_array_equal(jobs, [1, 0, 2])
        np.testing.assert_allclose(t_starts, [0.0, 2.0, 3.0])

    def test_single_flat_interval(self):
        # A bare (2,) interval is accepted as one column.
        np.testing.assert_array_equal(schedule_intervals([1.0, 2.0]), [0])
        np.testing.assert_array_equal(partition_intervals([1.0, 2.0]), [0])

    def test_empty_inputs(self):
        assert schedule_intervals(np.empty((2, 0))).size == 0
        assert partition_intervals(np.empty((2, 0))).size == 0
        wv, jobs = schedule_weighted_intervals(np.empty((2, 0)), [])
        assert wv == 0.0 and jobs.size == 0
        j, t = schedule_min_lateness_dense([], [])
        assert j.size == 0 and t.size == 0


class TestMospaProperties:
    def test_exact_and_approx_agree_when_separated(self):
        rng = np.random.default_rng(11)
        n = 100
        base = rng.normal(0, 1, (2, n))
        p = np.vstack(
            [base + np.array([[3.0], [0.0]]), base - np.array([[3.0], [0.0]])]
        )
        exact = mmospa2tar_2d(p)
        x = np.stack([p[0:2, :], p[2:4, :]], axis=1)
        w = np.full(n, 1.0 / n)
        approx, _ = mmospa_approx(x, w, num_scans=3)
        # The target labeling is a global two-fold ambiguity: both
        # orderings are equally optimal, and the two algorithms may
        # settle on different ones.
        a = approx.reshape(-1, order="F")
        swapped = np.concatenate([a[2:4], a[0:2]])
        err = min(np.abs(a - exact).max(), np.abs(swapped - exact).max())
        assert err < 1e-10
        np.testing.assert_allclose(
            calc_mospa_error(approx, x, w),
            calc_mospa_error(exact.reshape((2, 2), order="F"), x, w),
            rtol=1e-12,
        )

    def test_mmospa_beats_the_naive_mean(self):
        # With hypothesis orderings scrambled, the naive weighted mean
        # collapses the targets; the MMOSPA estimate must score better.
        rng = np.random.default_rng(5)
        n = 60
        base = rng.normal(0, 0.3, (2, 2, n))
        base[:, 0, :] += np.array([[2.0], [0.0]])
        base[:, 1, :] -= np.array([[2.0], [0.0]])
        swap = rng.random(n) < 0.5
        base[:, :, swap] = base[:, ::-1, swap]
        w = np.full(n, 1.0 / n)
        est, _ = mmospa_approx(base, w, num_scans=3)
        naive = (base * w).sum(axis=2)
        assert calc_mospa_error(est, base, w) < calc_mospa_error(naive, base, w)

    def test_2tar_sweep_improves_on_the_canonical_mean(self):
        # Targets on a circle with displacement directions spanning
        # half the plane: the optimal split of the direction-sorted
        # sweep is interior, so the estimate must beat the naive
        # canonicalized mean (the sweep's improvement branch fires).
        th = np.linspace(0.1, np.pi - 0.1, 25)
        p = np.vstack([np.cos(th), np.sin(th), -np.cos(th), -np.sin(th)])
        est = mmospa2tar_2d(p)
        canon = p.copy()
        for j in range(p.shape[1]):
            if -(canon[0, j] - canon[2, j]) < 0:
                canon[:, j] = canon[[2, 3, 0, 1], j]
        init = canon.mean(axis=1)
        assert est @ est > init @ init + 1e-6
        x = np.stack([p[0:2, :], p[2:4, :]], axis=1)
        w = np.full(len(th), 1.0 / len(th))
        assert calc_mospa_error(
            est.reshape((2, 2), order="F"), x, w
        ) < calc_mospa_error(init.reshape((2, 2), order="F"), x, w)

    def test_2tar_requires_stacked_2d(self):
        with pytest.raises(ValueError):
            mmospa2tar_2d(np.zeros((6, 4)))
