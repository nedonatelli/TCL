"""Close the honest-coverage gaps in the STABLE-classified modules.

The maturity rubric requires >=90% coverage (measured with
``NUMBA_DISABLE_JIT=1`` and ``--cov-branch``) for MATURE; STABLE promises
more, yet five STABLE modules sat below that bar. These tests target the
specific unexecuted branches -- the same exercise that surfaced the ZXZ
gimbal bug, so each test asserts values, not just execution.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.assignment_algorithms.two_dimensional.assignment import (
    assign2d,
    auction,
)
from pytcl.core.optional_deps import (
    _AvailabilityFlags,
    _get_install_command,
    check_dependencies,
)
from pytcl.dynamic_estimation.kalman.unscented import (
    ckf_update,
    sigma_points_julier,
    sigma_points_merwe,
    ukf_update,
)
from pytcl.dynamic_models.process_noise.polynomial import q_poly_kal


class TestQPolyKalMultiDim:
    """The num_dims > 1 block-diagonal path had never executed."""

    def test_block_diagonal_of_the_1d_matrix(self):
        Q1 = q_poly_kal(1, 0.5, 2.0, num_dims=1)
        Q3 = q_poly_kal(1, 0.5, 2.0, num_dims=3)

        n = Q1.shape[0]
        assert Q3.shape == (3 * n, 3 * n)
        expected = np.zeros_like(Q3)
        for d in range(3):
            expected[d * n : (d + 1) * n, d * n : (d + 1) * n] = Q1
        assert_allclose(Q3, expected, rtol=1e-15)
        # off-diagonal blocks are exactly zero: dimensions are independent
        assert np.count_nonzero(Q3) == 3 * np.count_nonzero(Q1)


class TestAuctionRectangularAndDegenerate:
    """The transpose path (n > m) and the single-column bid path."""

    def test_tall_matrix_transposes_and_matches_wide(self):
        cost = np.array([[4.0, 1.0], [2.0, 3.0], [5.0, 2.0]])  # 3x2: n > m
        rows_t, cols_t, total_t = auction(cost)
        rows_w, cols_w, total_w = auction(cost.T)

        assert total_t == pytest.approx(total_w)
        # assignments correspond under transposition
        assert set(zip(rows_t, cols_t)) == set(zip(cols_w, rows_w))
        # and the reported total is the sum of the assigned entries
        assert total_t == pytest.approx(cost[rows_t, cols_t].sum())

    def test_single_column_uses_the_one_value_bid_path(self):
        """len(values) < 2: second-best is -inf by construction."""
        cost = np.array([[3.0]])
        rows, cols, total = auction(cost)
        assert (list(rows), list(cols)) == ([0], [0])
        assert total == pytest.approx(3.0)


class TestAssign2dMaximize:
    """The maximize branch of the augmented (finite non-assignment) path."""

    def test_maximize_picks_the_large_entries(self):
        rewards = np.array([[10.0, 1.0], [1.0, 10.0]])
        result = assign2d(rewards, cost_of_non_assignment=0.5, maximize=True)
        assert set(zip(result.row_indices, result.col_indices)) == {(0, 0), (1, 1)}
        assert result.cost == pytest.approx(20.0)

    def test_maximize_reports_the_reward_sum_not_a_cost(self):
        """Named-field access, after the first draft of this test unpacked
        the NamedTuple positionally in the wrong order and 'passed' by
        comparing pytest.approx against an empty unassigned array. The
        field order is (row_indices, col_indices, cost, unassigned_rows,
        unassigned_cols)."""
        rewards = np.array([[0.1, 0.2], [0.3, 5.0]])
        result = assign2d(rewards, cost_of_non_assignment=1.0, maximize=True)
        assert (1, 1) in set(zip(result.row_indices, result.col_indices))
        assert (
            result.cost >= rewards[result.row_indices, result.col_indices].sum() - 1e-9
        )


class TestSigmaPointNearSingularFallbacks:
    """Cholesky-failure eigendecomposition paths in both sigma-point sets.

    A rank-deficient covariance is legitimate (a perfectly known component),
    and the fallback must produce points whose sample covariance still
    reconstructs P to the clamped tolerance.
    """

    P_SINGULAR = np.array([[2.0, 0.0], [0.0, 0.0]])

    def _check(self, pts):
        x = np.zeros(2)
        assert np.all(np.isfinite(pts.points))
        # weighted reconstruction of P from the sigma points (fields Wm/Wc)
        diffs = pts.points - x
        P_rec = (pts.Wc[:, None, None] * diffs[:, :, None] * diffs[:, None, :]).sum(0)
        assert_allclose(P_rec[0, 0], 2.0, atol=1e-6)
        assert abs(P_rec[1, 1]) < 1e-6

    def test_merwe_fallback(self):
        self._check(sigma_points_merwe(np.zeros(2), self.P_SINGULAR))

    def test_julier_fallback(self):
        self._check(sigma_points_julier(np.zeros(2), self.P_SINGULAR))


class TestCKFUpdateValidationAndSingularPaths:
    """ckf_update's points/weights validation and both updates' singular-S
    likelihood guards. (A first draft aimed these at ckf_predict; the
    validation lives in ckf_update, which coverage exposed.)"""

    X = np.zeros(2)
    P = np.eye(2)
    Z = np.array([0.1, -0.2])
    R = np.eye(2) * 0.1

    @staticmethod
    def _h(x):
        return x

    def test_points_without_weights_rejected(self):
        with pytest.raises(ValueError, match="together"):
            ckf_update(self.X, self.P, self.Z, self._h, self.R, points=np.zeros((4, 2)))

    def test_wrong_point_dimension_rejected(self):
        with pytest.raises(ValueError, match="incompatible"):
            ckf_update(
                self.X,
                self.P,
                self.Z,
                self._h,
                self.R,
                points=np.zeros((4, 3)),
                weights=np.full(4, 0.25),
            )

    def test_mismatched_weight_count_rejected(self):
        with pytest.raises(ValueError, match="does not match"):
            ckf_update(
                self.X,
                self.P,
                self.Z,
                self._h,
                self.R,
                points=np.zeros((4, 2)),
                weights=np.full(3, 1 / 3),
            )

    def test_singular_P_takes_the_eigh_fallback(self):
        upd = ckf_update(
            self.X, np.array([[1.0, 0.0], [0.0, 0.0]]), self.Z, self._h, self.R
        )
        assert np.all(np.isfinite(upd.x))
        assert np.all(np.isfinite(upd.P))

    def test_singular_innovation_covariance_gives_zero_likelihood(self):
        """Both updates' det(S) <= 0 guard: R = 0 with a singular P makes S
        singular, and the documented contract is likelihood 0.0, not a
        division by zero."""
        P_sing = np.array([[1.0, 0.0], [0.0, 0.0]])
        R_zero = np.zeros((2, 2))
        for update in (ukf_update, ckf_update):
            result = update(self.X, P_sing, self.Z, self._h, R_zero)
            assert result.likelihood == 0.0, update.__name__


class TestOptionalDepsBranches:
    """Install-command composition and the availability-flag properties."""

    def test_install_command_for_a_known_extra(self):
        cmd = _get_install_command("plotly", extra="visualization")
        assert "visualization" in cmd

    def test_install_command_for_a_package_with_registered_extra(self):
        # the PACKAGE_EXTRAS lookup branches (extra name + pip name)
        cmd = _get_install_command("plotly")
        assert "install" in cmd

    def test_every_availability_flag_answers_a_bool(self):
        flags = _AvailabilityFlags()
        for name in dir(flags):
            if name.startswith("HAS_") or name.endswith("_AVAILABLE"):
                assert isinstance(getattr(flags, name), bool), name

    def test_check_dependencies_lists_all_missing_packages(self):
        with pytest.raises(Exception) as excinfo:
            check_dependencies("nonexistent_package_alpha", "nonexistent_package_beta")
        message = str(excinfo.value)
        assert "nonexistent_package_alpha" in message
        assert "nonexistent_package_beta" in message
