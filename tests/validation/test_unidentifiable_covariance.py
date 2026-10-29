"""One convention for a covariance whose information matrix is singular.

When a point estimate exists but the information is singular in some
directions, the estimate is returned with ``inf`` variance on the
unidentified parameters, NaN on their cross terms, the identified block
intact, and a ``RuntimeWarning``. v2.11.1 returned a finite pseudo-inverse
from ``mle_scoring`` and ``cramer_rao_bound`` (0.25 on every entry of the
model below). Where no point estimate exists the function raises:
``total_least_squares`` for an infeasible rank, and
``range_only_static_loc_est_np`` for coplanar sensors, a geometry MATLAB's
algorithm excludes.
"""

import numpy as np
import pytest

from pytcl.static_estimation import (
    cramer_rao_bound,
    mle_scoring,
    range_only_static_loc_est_np,
    total_least_squares,
)
from tests.validation._v2111 import shipped_module

# ll = -0.5 (t0 + t1 - 1)^2 identifies t0 + t1 only.
FLAT_FISHER = np.ones((2, 2))

# Receiver and transmitters all on the plane z = 0, outside MATLAB's domain.
RX = np.array([100.0, 200.0, 0.0])
TX = np.array(
    [
        [0.0, 8e3, -6e3, 2e3, -3e3],
        [0.0, 1e3, 5e3, -7e3, 2e3],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
TARGET = np.array([4e3, -2e3, 0.0])
R_COV = np.diag([9.0, 25.0, 4.0, 1.0, 16.0])
R_BI = np.linalg.norm(TARGET[:, None] - TX, axis=0) + np.linalg.norm(TARGET - RX)


def _assert_unidentified(cov, unidentified):
    unidentified = np.asarray(unidentified)
    identified = ~unidentified
    assert np.all(np.isposinf(np.diag(cov)[unidentified]))
    cross = cov[np.ix_(unidentified, unidentified)]
    assert np.all(np.isnan(cross[~np.eye(len(cross), dtype=bool)]))
    assert np.all(np.isnan(cov[np.ix_(unidentified, identified)]))
    assert np.all(np.isnan(cov[np.ix_(identified, unidentified)]))
    assert np.all(np.isfinite(cov[np.ix_(identified, identified)]))


class TestMleScoring:
    @staticmethod
    def _run():
        return mle_scoring(
            lambda t: -0.5 * (t[0] + t[1] - 1.0) ** 2,
            lambda t: -(t[0] + t[1] - 1.0) * np.ones(2),
            lambda t: FLAT_FISHER,
            np.zeros(2),
        )

    def test_unidentified_model_has_infinite_variance_and_warns(self):
        with pytest.warns(RuntimeWarning, match="mle_scoring"):
            res = self._run()
        assert res.theta[0] + res.theta[1] == pytest.approx(1.0, rel=1e-15, abs=0)
        assert not res.converged
        _assert_unidentified(res.covariance, [True, True])

    def test_v2111_returned_a_finite_pseudo_inverse(self):
        old = shipped_module("pytcl/static_estimation/maximum_likelihood.py")
        res = old.mle_scoring(
            lambda t: -0.5 * (t[0] + t[1] - 1.0) ** 2,
            lambda t: -(t[0] + t[1] - 1.0) * np.ones(2),
            lambda t: FLAT_FISHER,
            np.zeros(2),
        )
        np.testing.assert_allclose(res.covariance, np.full((2, 2), 0.25), rtol=1e-12)


class TestCramerRaoBound:
    def test_unidentified_model_has_infinite_variance_and_warns(self):
        with pytest.warns(RuntimeWarning, match="cramer_rao_bound"):
            res = cramer_rao_bound(FLAT_FISHER)
        _assert_unidentified(res.crb_matrix, [True, True])
        assert np.all(np.isposinf(res.variances))
        assert np.all(np.isposinf(res.std_bounds))

    def test_identified_block_is_intact(self):
        F = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        with pytest.warns(RuntimeWarning, match="cramer_rao_bound"):
            res = cramer_rao_bound(F)
        _assert_unidentified(res.crb_matrix, [False, True, True])
        assert res.variances[0] == 0.5

    def test_identified_model_keeps_v2111_bits(self):
        old = shipped_module("pytcl/static_estimation/maximum_likelihood.py")
        F = np.array([[4.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 3.0]])
        np.testing.assert_array_equal(
            cramer_rao_bound(F).crb_matrix, old.cramer_rao_bound(F).crb_matrix
        )


class TestRangeOnlyCoplanarRaises:
    @pytest.mark.parametrize("height", [0.0, 3e3], ids=["in_plane", "off_plane"])
    @pytest.mark.parametrize("r_cov", [None, R_COV], ids=["no_r_cov", "r_cov"])
    def test_coplanar_sensors_raise(self, height, r_cov):
        target = np.array([4e3, -2e3, height])
        r = np.linalg.norm(target[:, None] - TX, axis=0) + np.linalg.norm(target - RX)
        with pytest.raises(np.linalg.LinAlgError, match="sensors are coplanar"):
            range_only_static_loc_est_np(r, TX, RX, 1, r_cov)

    def test_method_zero_raises(self):
        with pytest.raises(np.linalg.LinAlgError, match="sensors are coplanar"):
            range_only_static_loc_est_np(R_BI, TX, RX, 0)

    def test_v2111_returned_a_biased_estimate_without_r_cov(self):
        # Why the geometry is rejected: the pseudo-inverse confines the
        # estimate to the plane and the Rt = |x| constraint then biases y.
        old = shipped_module("pytcl/static_estimation/localization.py")
        target = np.array([4e3, -2e3, 3e3])
        r = np.linalg.norm(target[:, None] - TX, axis=0) + np.linalg.norm(target - RX)
        x_est = old.range_only_static_loc_est_np(r, TX, RX, 1).x_est
        assert x_est[2] == 0.0
        assert abs(x_est[1] - target[1]) > 100.0


class TestTotalLeastSquaresStillRaises:
    def test_infeasible_rank_raises(self):
        # [A | b] has orthogonal columns with singular values 3 (b), 2, 1:
        # the trailing right singular vector for rank 2 is e_2, whose last
        # entry is zero, so no rank-2 TLS solution exists.
        A = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        b = np.array([0.0, 0.0, 3.0])
        with pytest.raises(ValueError, match="does not exist"):
            total_least_squares(A, b, rank=2)
