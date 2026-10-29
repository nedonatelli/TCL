"""One convention for a covariance whose information matrix is singular.

When a point estimate exists but the information is singular in some
directions, the estimate is returned with ``inf`` variance on the
unidentified parameters, NaN on their cross terms, the identified block
intact, and a ``RuntimeWarning``. v2.11.1 returned a finite pseudo-inverse
from ``mle_scoring`` and ``cramer_rao_bound`` (0.25 on every entry of the
model below) and raised ``LinAlgError`` from ``range_only_static_loc_est_np``
for coplanar sensors. ``total_least_squares`` still raises, because no
solution exists for an infeasible rank.

REFERENCE class: the range-only x-y block is compared with an independent
2-D solve of the same geometry, the problem restricted to the identified
plane.
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

# Receiver, transmitters and target all on the plane z = 0: z has zero
# Fisher information, x and y are identified.
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

# Both sides are closed-form linear algebra on ranges of ~1e4 m.
RTOL_ORACLE = 1e-10


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


def _oracle_2d():
    """Taylor covariance and CRLB of the planar problem, from first principles.

    The estimator is the fixed point ``x = S+ (z(r) + r |x|)`` with
    ``z_i = (|s_i|^2 - r_i^2) / 2``, ``S`` the transmitters relative to the
    receiver and ``S+`` its pseudo-inverse; the implicit function theorem
    gives ``dx/dr = -(dG/dx)^-1 dG/dr`` for ``G(x, r) = S+ (z + r |x|) - x``.
    """
    s = (TX[:2] - RX[:2, None]).T
    x = TARGET[:2] - RX[:2]
    norm_x = np.linalg.norm(x)
    s_plus = np.linalg.pinv(s)
    dg_dx = s_plus @ np.outer(R_BI, x / norm_x) - np.eye(2)
    dg_dr = s_plus @ (norm_x * np.eye(len(R_BI)) - np.diag(R_BI))
    dxdr = -np.linalg.solve(dg_dx, dg_dr)
    p_taylor = dxdr @ R_COV @ dxdr.T

    to_tx = x[:, None] - s.T
    jac = x / norm_x + (to_tx / np.linalg.norm(to_tx, axis=0)).T
    p_crlb = np.linalg.inv(jac.T @ np.linalg.inv(R_COV) @ jac)
    return p_taylor, p_crlb


class TestRangeOnlyCoplanar:
    def test_returns_estimate_with_infinite_z_variance_and_warns(self):
        with pytest.warns(RuntimeWarning, match="range_only_static_loc_est_np"):
            res = range_only_static_loc_est_np(R_BI, TX, RX, 1, R_COV)
        np.testing.assert_allclose(res.x_est, TARGET, rtol=1e-12, atol=1e-9)
        for p in (res.p_taylor[:, :, 0], res.p_crlb[:, :, 0]):
            _assert_unidentified(p, [False, False, True])

    def test_xy_block_matches_planar_solve(self):
        with pytest.warns(RuntimeWarning):
            res = range_only_static_loc_est_np(R_BI, TX, RX, 1, R_COV)
        p_taylor, p_crlb = _oracle_2d()
        np.testing.assert_allclose(
            res.p_taylor[:2, :2, 0], p_taylor, rtol=RTOL_ORACLE, atol=0
        )
        np.testing.assert_allclose(
            res.p_crlb[:2, :2, 0], p_crlb, rtol=RTOL_ORACLE, atol=0
        )

    def test_clipped_root_branch_marks_z(self):
        # Ranges no target fits: Equation 21's discriminant is negative, so the
        # root is clipped and the Taylor derivative takes its other branch.
        # v2.11.1 returned 0 for the z variances here, a certainty about the
        # one coordinate the sensors cannot see.
        r = np.array([846877.0, 1663346.0, 826160.0, 1106634.0, 58703.0])
        s = (TX - RX[:, None]).T
        s_plus = np.linalg.pinv(s)
        a = s_plus @ (0.5 * (np.sum(s * s, axis=1) - r**2))
        b = s_plus @ r
        assert (a @ b) ** 2 - (b @ b - 1.0) * (a @ a) < 0.0
        with pytest.warns(RuntimeWarning, match="range_only_static_loc_est_np"):
            res = range_only_static_loc_est_np(r, TX, RX, 1, R_COV)
        for p in (res.p_taylor[:, :, 0], res.p_crlb[:, :, 0]):
            _assert_unidentified(p, [False, False, True])

    def test_v2111_raised(self):
        old = shipped_module("pytcl/static_estimation/localization.py")
        with pytest.raises(np.linalg.LinAlgError):
            old.range_only_static_loc_est_np(R_BI, TX, RX, 1, R_COV)


class TestTotalLeastSquaresStillRaises:
    def test_infeasible_rank_raises(self):
        # [A | b] has orthogonal columns with singular values 3 (b), 2, 1:
        # the trailing right singular vector for rank 2 is e_2, whose last
        # entry is zero, so no rank-2 TLS solution exists.
        A = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        b = np.array([0.0, 0.0, 3.0])
        with pytest.raises(ValueError, match="does not exist"):
            total_least_squares(A, b, rank=2)
