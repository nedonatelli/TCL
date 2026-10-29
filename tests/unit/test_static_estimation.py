"""Tests for static estimation module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from pytcl.static_estimation import (  # Least squares; Robust estimation
    cauchy_weight,
    generalized_least_squares,
    huber_regression,
    huber_rho,
    huber_weight,
    irls,
    mad,
    ordinary_least_squares,
    ransac,
    ransac_n_trials,
    recursive_least_squares,
    ridge_regression,
    tau_scale,
    total_least_squares,
    tukey_regression,
    tukey_rho,
    tukey_weight,
    weighted_least_squares,
)


def _exact_mad(e):
    """Median-centred MAD with the exact Gaussian constant (statsmodels')."""
    return float(np.median(np.abs(e - np.median(e))) / norm.ppf(0.75))


class TestOrdinaryLeastSquares:
    """Tests for ordinary least squares."""

    def test_simple_line(self):
        """Fits a simple line."""
        A = np.array([[1, 0], [1, 1], [1, 2]])
        b = np.array([1, 2, 3])

        result = ordinary_least_squares(A, b)

        # y = 1 + 1*x
        assert_allclose(result.x, [1, 1], atol=1e-10)

    def test_overdetermined(self):
        """Handles overdetermined system."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        b = np.array([2.1, 2.9, 4.1, 4.9])

        result = ordinary_least_squares(A, b)

        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        assert_allclose(result.x, x_ref, rtol=1e-12)
        assert_allclose(result.residuals, b - A @ x_ref, rtol=1e-10, atol=1e-15)

    def test_rank_determination(self):
        """Correctly determines rank."""
        A = np.array([[1, 1], [2, 2], [3, 3]])  # Rank 1
        b = np.array([1, 2, 3])

        result = ordinary_least_squares(A, b)

        assert result.rank == 1

    def test_singular_values(self):
        """Returns singular values."""
        A = np.array([[1, 0], [0, 2], [0, 0]])
        b = np.array([1, 2, 0])

        result = ordinary_least_squares(A, b)

        assert_allclose(result.singular_values, [2.0, 1.0], rtol=1e-14)


class TestWeightedLeastSquares:
    """Tests for weighted least squares."""

    def test_identity_weights(self):
        """Identity weights give OLS solution."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        result_ols = ordinary_least_squares(A, b)
        result_wls = weighted_least_squares(A, b)

        assert_allclose(result_wls.x, result_ols.x, atol=1e-10)

    def test_diagonal_weights(self):
        """Diagonal weights work correctly."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        weights = np.array([1, 10, 1])  # High weight on middle

        result = weighted_least_squares(A, b, weights=weights)

        W = np.diag(weights.astype(float))
        x_ref = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
        assert_allclose(result.x, x_ref, rtol=1e-12)
        assert_allclose(result.residuals, b - A @ x_ref, rtol=1e-10, atol=1e-15)

    def test_covariance_output(self):
        """Returns covariance matrix."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        result = weighted_least_squares(A, b)

        # Unit weights: Cov = (A'A)^-1 = [[14, -6], [-6, 3]] / 6.
        assert_allclose(result.covariance, np.array([[14.0, -6.0], [-6.0, 3.0]]) / 6.0)

    def test_weight_matrix(self):
        """Full weight matrix works."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        W = np.array([[1, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 1]])

        result = weighted_least_squares(A, b, W=W)

        assert_allclose(result.x, np.linalg.solve(A.T @ W @ A, A.T @ W @ b), rtol=1e-12)


class TestTotalLeastSquares:
    """Tests for total least squares."""

    def test_no_error(self):
        """Exact solution when data lies on line."""
        A = np.array([[1, 1], [2, 1], [3, 1]])
        b = np.array([2, 3, 4])  # b = x0 + 1

        result = total_least_squares(A, b)

        assert_allclose(result.x, [1, 1], atol=1e-10)

    def test_returns_corrections(self):
        """Returns corrections to A and b."""
        A = np.array([[1, 1], [2, 1], [3, 1]])
        b = np.array([2.1, 2.9, 4.1])

        result = total_least_squares(A, b)

        # Closed form: the smallest singular triplet of [A | b] gives the
        # minimal correction [dA | db] = -s_min u_min v_min'.
        U, sv, Vt = np.linalg.svd(np.column_stack([A, b]), full_matrices=False)
        delta = -sv[-1] * np.outer(U[:, -1], Vt[-1])
        assert_allclose(result.residuals_A, delta[:, :2], rtol=1e-9, atol=1e-15)
        assert_allclose(result.residuals_b, delta[:, 2], rtol=1e-9, atol=1e-15)

    def test_rank_output(self):
        """Returns effective rank."""
        A = np.array([[1, 1], [2, 1], [3, 1]])
        b = np.array([2, 3, 4])

        result = total_least_squares(A, b)

        assert result.rank == 2


def _system_with_singular_values(sv, m=8, seed=0):
    """Build (A, b) so that [A | b] has exactly the singular values ``sv``."""
    rng = np.random.default_rng(seed)
    k = len(sv)
    U, _ = np.linalg.qr(rng.standard_normal((m, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    C = U @ np.diag(sv) @ V.T
    return C[:, : k - 1], C[:, k - 1]


def _ttls_reference(A, b, k):
    """Min-norm LS solution of the best rank-k approximation of [A | b]."""
    U, s, Vt = np.linalg.svd(np.column_stack([A, b]), full_matrices=False)
    s = np.where(np.arange(len(s)) < k, s, 0.0)
    Ck = (U * s) @ Vt
    return np.linalg.lstsq(Ck[:, :-1], Ck[:, -1], rcond=None)[0]


class TestOrdinaryLeastSquaresResiduals:
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

    @pytest.mark.parametrize(
        "b",
        [
            np.array([1.0, 2.0, 2.0]),
            np.array([[1.0, 0.0], [2.0, 1.0], [2.0, 5.0]]),
        ],
    )
    def test_residuals_are_b_minus_ax(self, b):
        out = ordinary_least_squares(self.A, b)
        assert out.residuals.shape == b.shape
        assert_allclose(out.residuals, b - self.A @ out.x, rtol=0, atol=1e-15)
        assert_allclose(self.A.T @ out.residuals, 0.0, rtol=0, atol=1e-14)

    def test_rank_deficient_returns_residual_vector(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        b = np.array([1.0, 2.0, 2.0])
        out = ordinary_least_squares(A, b)
        assert out.residuals.shape == (3,)
        assert_allclose(out.residuals, b - A @ out.x, rtol=1e-12)


class TestTotalLeastSquaresRank:
    """The ``rank`` argument selects truncated TLS (Fierro et al., 1997)."""

    def test_system_has_requested_singular_values(self):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        s = np.linalg.svd(np.column_stack([A, b]), compute_uv=False)
        # Weyl: every computed singular value is within ~eps * sigma_max of
        # the exact one, an absolute bound; rtol=1e-12 on 1e-4 is tighter than
        # that and failed on OpenBLAS (error 1.3e-16 absolute).
        assert_allclose(
            s, [5.0, 1.0, 1e-4], rtol=0, atol=10 * np.finfo(float).eps * 5.0
        )

    def test_rank_truncation_changes_the_solution(self):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        full = total_least_squares(A, b)
        truncated = total_least_squares(A, b, rank=1)
        assert_allclose(full.x, _ttls_reference(A, b, 2), rtol=1e-9)
        assert_allclose(truncated.x, _ttls_reference(A, b, 1), rtol=1e-9)
        assert np.max(np.abs(full.x - truncated.x)) > 1e-3

    def test_default_and_full_rank_are_identical(self):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        V = np.linalg.svd(np.column_stack([A, b]))[2].T
        legacy = -V[:2, 2] / V[2, 2]
        assert np.array_equal(total_least_squares(A, b).x, legacy)
        assert np.array_equal(total_least_squares(A, b, rank=2).x, legacy)

    def test_full_rank_matches_closed_form_tls(self):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        v = np.linalg.svd(np.column_stack([A, b]))[2][-1]
        assert_allclose(total_least_squares(A, b).x, -v[:2] / v[2], rtol=1e-12)

    @pytest.mark.parametrize("k", [1, 2])
    def test_truncated_solution_matches_rank_k_lstsq_oracle(self, k):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        out = total_least_squares(A, b, rank=k)
        assert out.x == pytest.approx(_ttls_reference(A, b, k), rel=1e-9, abs=0)

    def test_corrections_make_truncated_system_consistent(self):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        out = total_least_squares(A, b, rank=1)
        resid = (A + out.residuals_A) @ out.x - (b + out.residuals_b)
        assert_allclose(resid, 0.0, atol=1e-12)

    @pytest.mark.parametrize("rank", [0, 3, -1, 1.5, True])
    def test_invalid_rank_raises(self, rank):
        A, b = _system_with_singular_values([5.0, 1.0, 1e-4])
        with pytest.raises(ValueError, match="rank"):
            total_least_squares(A, b, rank=rank)

    def test_nonexistent_truncated_solution_raises(self):
        # e_3 is the leading right singular vector, so v22 vanishes for k=1.
        A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        b = np.array([0.0, 0.0, 3.0])
        with pytest.raises(ValueError, match="does not exist"):
            total_least_squares(A, b, rank=1)


class TestGeneralizedLeastSquares:
    """Tests for generalized least squares."""

    def test_identity_covariance(self):
        """Identity error covariance gives OLS."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        Sigma = np.eye(3)

        result_gls = generalized_least_squares(A, b, Sigma)
        result_ols = ordinary_least_squares(A, b)

        assert_allclose(result_gls.x, result_ols.x, atol=1e-10)

    def test_correlated_errors(self):
        """Handles correlated errors."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        Sigma = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])

        result = generalized_least_squares(A, b, Sigma)

        W = np.linalg.inv(Sigma)
        assert_allclose(result.x, np.linalg.solve(A.T @ W @ A, A.T @ W @ b), rtol=1e-12)


class TestRecursiveLeastSquares:
    """Tests for recursive least squares."""

    def test_batch_equivalence(self):
        """RLS converges to batch OLS."""
        # Generate data
        rng = np.random.default_rng(42)
        n_samples = 20
        true_x = np.array([2.0, 0.5])
        A = np.column_stack([np.ones(n_samples), np.arange(n_samples)])
        b = A @ true_x + rng.normal(0, 0.1, n_samples)

        # RLS
        x = np.zeros(2)
        P = np.eye(2) * 100

        for i in range(n_samples):
            x, P = recursive_least_squares(x, P, A[i], b[i])

        # With prior N(0, 100 I) the RLS estimate is exactly the batch
        # solution of (A'A + I/100) x = A'b.
        x_batch = np.linalg.solve(A.T @ A + np.eye(2) / 100.0, A.T @ b)

        assert_allclose(x, x_batch, rtol=1e-9)
        assert_allclose(
            P, np.linalg.inv(A.T @ A + np.eye(2) / 100.0), rtol=1e-8, atol=1e-15
        )

    def test_forgetting_factor(self):
        """Forgetting factor emphasizes recent data."""
        x = np.array([1.0, 0.0])
        P = np.eye(2)

        # Old data pointing one direction
        for _ in range(10):
            x, P = recursive_least_squares(
                x, P, np.array([1, 0]), 0.0, forgetting_factor=0.95
            )

        # New data pointing different direction
        for _ in range(10):
            x, P = recursive_least_squares(
                x, P, np.array([1, 0]), 1.0, forgetting_factor=0.95
            )

        # Exponentially weighted batch solution: minimise
        # sum_i lam^(N-i) (b_i - x0)^2 + lam^N (x0 - 1)^2 over the first
        # component (the second is unobserved and keeps its prior of 0).
        lam, n = 0.95, 20
        w = lam ** (n - np.arange(1, n + 1))
        b_obs = np.r_[np.zeros(10), np.ones(10)]
        expected = (w @ b_obs + lam**n) / (w.sum() + lam**n)
        assert_allclose(x[0], expected, rtol=1e-10)


class TestRidgeRegression:
    """Tests for ridge regression."""

    def test_zero_regularization(self):
        """Zero regularization gives OLS."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        x_ridge = ridge_regression(A, b, alpha=1e-10)
        x_ols = np.linalg.lstsq(A, b, rcond=None)[0]

        # First-order shrinkage bound: x_ridge - x_ols ~ -alpha (A'A)^-1 x_ols.
        assert_allclose(x_ridge, x_ols, rtol=1e-9)

    def test_shrinkage(self):
        """Higher regularization shrinks parameters."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        x_low = ridge_regression(A, b, alpha=0.1)
        x_high = ridge_regression(A, b, alpha=10.0)

        for x, alpha in ((x_low, 0.1), (x_high, 10.0)):
            assert_allclose(
                x, np.linalg.solve(A.T @ A + alpha * np.eye(2), A.T @ b), rtol=1e-12
            )

    def test_ill_conditioned(self):
        """Handles ill-conditioned problems."""
        # Collinear columns
        A = np.array([[1, 1], [2, 2.0001], [3, 3]])
        b = np.array([1, 2, 3])

        # Should not fail
        x = ridge_regression(A, b, alpha=1.0)
        assert_allclose(x, np.linalg.solve(A.T @ A + np.eye(2), A.T @ b), rtol=1e-12)


class TestHuberWeight:
    """Tests for Huber weight function."""

    def test_small_residuals(self):
        """Small residuals get weight 1."""
        r = np.array([0.0, 0.5, 1.0])
        weights = huber_weight(r, c=1.345)

        assert_allclose(weights, [1.0, 1.0, 1.0])

    def test_large_residuals(self):
        """Large residuals get reduced weight."""
        r = np.array([2.0, 4.0])
        weights = huber_weight(r, c=1.345)

        # w = c / |r| beyond the threshold.
        assert_allclose(weights, [1.345 / 2.0, 1.345 / 4.0], rtol=1e-14)

    def test_symmetry(self):
        """Weights are symmetric."""
        r = np.array([2.0, -2.0])
        weights = huber_weight(r, c=1.345)

        assert_allclose(weights, [1.345 / 2.0, 1.345 / 2.0], rtol=1e-14)


class TestTukeyWeight:
    """Tests for Tukey bisquare weight function."""

    def test_small_residuals(self):
        """Small residuals get positive weight."""
        r = np.array([0.0, 1.0, 2.0])
        weights = tukey_weight(r, c=4.685)

        assert_allclose(weights, [(1 - (r_ / 4.685) ** 2) ** 2 for r_ in r], rtol=1e-14)

    def test_large_residuals(self):
        """Large residuals get zero weight."""
        r = np.array([5.0, 10.0])
        weights = tukey_weight(r, c=4.685)

        assert_allclose(weights, [0.0, 0.0])

    def test_continuity(self):
        """Weight function is continuous at boundary."""
        c = 4.685
        r = np.array([c - 0.01, c, c + 0.01])
        weights = tukey_weight(r, c=c)

        assert_allclose(
            weights, [(1 - ((c - 0.01) / c) ** 2) ** 2, 0.0, 0.0], rtol=1e-12
        )


class TestMAD:
    """Tests for MAD scale estimator."""

    def test_normal_data(self):
        """Consistent for normal data."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)

        scale = mad(data)

        assert_allclose(
            scale, 1.4826 * np.median(np.abs(data - np.median(data))), rtol=1e-14
        )

    def test_with_outliers(self):
        """Robust to outliers."""
        data = np.array([1, 2, 3, 4, 5, 1000])

        scale = mad(data)

        # median = 3.5, |dev| sorted = [.5, .5, 1.5, 1.5, 2.5, 996.5],
        # median |dev| = 1.5, so the outlier does not enter.
        assert_allclose(scale, 1.4826 * 1.5, rtol=1e-14)


class TestIRLS:
    """Tests for IRLS algorithm."""

    def test_convergence(self):
        """Algorithm converges."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        b = np.array([2, 3, 4, 5])

        result = irls(A, b)

        # b = 1 + x exactly: zero residuals, unit weights, x = [1, 1].
        assert result.converged
        assert_allclose(result.x, [1.0, 1.0], rtol=1e-12)
        assert_allclose(result.weights, np.ones(4), rtol=1e-12)

    def test_outlier_downweighting(self):
        """Outliers get lower weights."""
        x = np.arange(1, 9, dtype=float)
        A = np.column_stack([np.ones(8), x])
        b = 2 + x + np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 0.0])
        b[-1] = 60.0  # outlier

        result = irls(A, b, scale_func=_exact_mad, max_iter=2000, tol=1e-14)

        # statsmodels 0.15.0 RLM(b, A, M=HuberT(1.345)) with the same
        # median-centred MAD / Phi^-1(0.75) scale, tol=1e-14.
        assert result.converged
        assert_allclose(result.x, [1.8701454105365432, 1.0466251207990838], rtol=1e-9)
        assert_allclose(
            result.weights,
            [1.0] * 6 + [0.9117467064280294, 0.006349670266076819],
            rtol=1e-9,
        )


class TestHuberRegression:
    """Tests for Huber regression."""

    def test_clean_data(self):
        """Matches OLS on clean data."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        b = np.array([2, 3, 4, 5])

        result_huber = huber_regression(A, b)
        result_ols = ordinary_least_squares(A, b)

        # b = 1 + x is an exact line: Huber and OLS both recover [1, 1].
        assert_allclose(result_huber.x, [1.0, 1.0], rtol=1e-12)
        assert_allclose(result_ols.x, [1.0, 1.0], rtol=1e-12)

    def test_outlier_resistance(self):
        """Resistant to outliers."""
        rng = np.random.default_rng(42)
        n = 50
        A = np.column_stack([np.ones(n), np.arange(n)])
        b = A @ np.array([1.0, 1.0]) + rng.normal(0, 0.5, n)
        # Add a few outliers
        b[-5:] = 100

        result = huber_regression(A, b, max_iter=500, tol=1e-14)

        # statsmodels 0.15.0 RLM(b, A, M=HuberT(1.345)) with the median-centred
        # MAD scale (pytcl's 1.4826 differs from 1/Phi^-1(0.75) by 1.5e-6).
        assert_allclose(result.x, [0.7411332403203881, 1.0171454460762526], rtol=1e-5)


class TestTukeyRegression:
    """Tests for Tukey bisquare regression."""

    def test_outlier_rejection(self):
        """Completely rejects gross outliers."""
        rng = np.random.default_rng(42)
        n = 50
        A = np.column_stack([np.ones(n), np.arange(n)])
        b = A @ np.array([1.0, 1.0]) + rng.normal(0, 0.5, n)
        # Add one gross outlier
        b[-1] = 1000

        result = tukey_regression(A, b, max_iter=500, tol=1e-14)

        # statsmodels 0.15.0 RLM(b, A, M=TukeyBiweight(4.685)): the outlier's
        # weight is exactly 0, which IRLS floors at 1e-10; the median inlier
        # weight is 0.96001692787.
        assert_allclose(result.x, [0.9521377101870738, 1.0039663570020796], rtol=1e-5)
        assert result.weights[-1] == 1e-10
        assert_allclose(np.median(result.weights[:-1]), 0.9600169278268497, rtol=1e-5)


class TestRANSAC:
    """Tests for RANSAC."""

    def test_finds_inliers(self):
        """Correctly identifies inliers."""
        rng = np.random.default_rng(42)

        # Generate inlier data
        n_inliers = 80
        A_inliers = np.column_stack([np.ones(n_inliers), rng.uniform(0, 10, n_inliers)])
        b_inliers = 2 + 3 * A_inliers[:, 1] + rng.normal(0, 0.5, n_inliers)

        # Add outliers
        n_outliers = 20
        A_outliers = np.column_stack(
            [np.ones(n_outliers), rng.uniform(0, 10, n_outliers)]
        )
        b_outliers = rng.uniform(-100, 100, n_outliers)

        A = np.vstack([A_inliers, A_outliers])
        b = np.concatenate([b_inliers, b_outliers])

        result = ransac(A, b, random_state=42)

        # The fit is the least-squares solution on exactly the points it
        # calls inliers, and those are the points within the threshold.
        mask = result.inliers
        assert result.n_inliers == mask.sum()
        x_ref = np.linalg.lstsq(A[mask], b[mask], rcond=None)[0]
        assert_allclose(result.x, x_ref, rtol=1e-9)
        assert_allclose(result.residuals, b - A @ result.x, rtol=1e-9)
        assert np.all(mask[:n_inliers])
        assert not np.any(mask[n_inliers:])

    def test_parameter_estimation(self):
        """Estimates parameters correctly."""
        rng = np.random.default_rng(42)

        # True parameters
        true_x = np.array([2.0, 3.0])

        # Generate data
        n = 100
        A = np.column_stack([np.ones(n), rng.uniform(0, 10, n)])
        b = A @ true_x + rng.normal(0, 0.5, n)

        # Add 20% outliers
        outlier_idx = rng.choice(n, size=20, replace=False)
        b[outlier_idx] = rng.uniform(-100, 100, 20)

        result = ransac(A, b, random_state=42)

        # The returned fit is least squares on exactly the points it calls
        # inliers (RANSAC refits on the consensus set).
        x_ref = np.linalg.lstsq(A[result.inliers], b[result.inliers], rcond=None)[0]
        assert_allclose(result.x, x_ref, rtol=1e-9)
        # Points more than 10 off the true line can never be consensus members
        # (the default threshold is a couple of noise sigmas).
        far = np.abs(b - A @ true_x) > 10
        assert far.sum() >= 10
        assert not np.any(result.inliers[far])


class TestRANSACNTrials:
    """Tests for RANSAC trial computation."""

    def test_low_outlier_rate(self):
        """Few trials for low outlier rate."""
        # ceil(log(0.01) / log(1 - 0.9^2)) = ceil(2.77)
        assert ransac_n_trials(100, 10, 2) == 3

    def test_high_outlier_rate(self):
        """More trials for high outlier rate."""
        n_trials_low = ransac_n_trials(100, 10, 2)
        n_trials_high = ransac_n_trials(100, 40, 2)

        assert n_trials_low == 3
        # ceil(log(0.01) / log(1 - 0.6^2)) = ceil(10.3)
        assert n_trials_high == 11

    def test_more_samples_need_more_trials(self):
        """More samples per trial needs more trials."""
        n_trials_2 = ransac_n_trials(100, 30, 2)
        n_trials_5 = ransac_n_trials(100, 30, 5)

        # ceil(log(0.01) / log(1 - 0.7^2)) = ceil(6.84); with 5 samples
        # ceil(log(0.01) / log(1 - 0.7^5)) = ceil(25.05).
        assert n_trials_2 == 7
        assert n_trials_5 == 26


class TestRhoFunctions:
    """Tests for rho (loss) functions."""

    def test_huber_rho_quadratic_region(self):
        """Huber rho is quadratic for small residuals."""
        r = np.array([0.5, 1.0])
        rho = huber_rho(r, c=1.345)

        assert_allclose(rho, r**2 / 2)

    def test_tukey_rho_bounded(self):
        """Tukey rho is bounded."""
        c = 4.685
        r = np.array([0.0, 10.0, 100.0])
        rho = tukey_rho(r, c=c)

        assert_allclose(rho, [0.0, c**2 / 6, c**2 / 6], rtol=1e-14)


class TestCauchyWeight:
    """Tests for Cauchy weight function."""

    def test_decays_smoothly(self):
        """Weights decay smoothly."""
        r = np.array([0.0, 1.0, 2.0, 4.0])
        weights = cauchy_weight(r, c=2.385)

        assert_allclose(weights, 1.0 / (1.0 + (r / 2.385) ** 2), rtol=1e-14)


class TestTauScale:
    """Tests for tau scale estimator."""

    def test_robust_to_outliers(self):
        """Tau scale is robust to outliers."""
        clean = np.array([1, 2, 3, 4, 5])
        with_outlier = np.array([1, 2, 3, 4, 100])

        scale_clean = tau_scale(clean)
        scale_outlier = tau_scale(with_outlier)

        # statsmodels 0.15.0 scale_tau(...)[1]; see tests/validation/test_tau_scale.py
        assert scale_clean == pytest.approx(1.470655510277057, rel=1e-12, abs=0)
        assert scale_outlier == pytest.approx(1.744097372590729, rel=1e-12, abs=0)
