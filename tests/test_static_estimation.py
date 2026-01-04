"""Tests for static estimation module."""

import numpy as np
from numpy.testing import assert_allclose

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

        # Check residuals are small
        assert np.linalg.norm(result.residuals) < 0.5

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

        assert len(result.singular_values) == 2
        assert_allclose(sorted(result.singular_values, reverse=True), [2, 1])


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

        # Middle point should be fit more closely
        residuals = b - A @ result.x
        assert abs(residuals[1]) < abs(residuals[0])

    def test_covariance_output(self):
        """Returns covariance matrix."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        result = weighted_least_squares(A, b)

        assert result.covariance.shape == (2, 2)
        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert np.all(eigenvalues > 0)

    def test_weight_matrix(self):
        """Full weight matrix works."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        W = np.array([[1, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 1]])

        result = weighted_least_squares(A, b, W=W)

        assert result.x.shape == (2,)


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

        assert result.residuals_A.shape == A.shape
        assert result.residuals_b.shape == b.shape

    def test_rank_output(self):
        """Returns effective rank."""
        A = np.array([[1, 1], [2, 1], [3, 1]])
        b = np.array([2, 3, 4])

        result = total_least_squares(A, b)

        assert result.rank == 2


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

        assert result.x.shape == (2,)


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

        # Batch OLS
        x_batch = ordinary_least_squares(A, b).x

        assert_allclose(x, x_batch, atol=0.1)

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

        # Should be closer to 1 than to 0
        assert x[0] > 0.5


class TestRidgeRegression:
    """Tests for ridge regression."""

    def test_zero_regularization(self):
        """Zero regularization gives OLS."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        x_ridge = ridge_regression(A, b, alpha=1e-10)
        x_ols = ordinary_least_squares(A, b).x

        assert_allclose(x_ridge, x_ols, atol=1e-6)

    def test_shrinkage(self):
        """Higher regularization shrinks parameters."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])

        x_low = ridge_regression(A, b, alpha=0.1)
        x_high = ridge_regression(A, b, alpha=10.0)

        assert np.linalg.norm(x_high) < np.linalg.norm(x_low)

    def test_ill_conditioned(self):
        """Handles ill-conditioned problems."""
        # Collinear columns
        A = np.array([[1, 1], [2, 2.0001], [3, 3]])
        b = np.array([1, 2, 3])

        # Should not fail
        x = ridge_regression(A, b, alpha=1.0)
        assert np.isfinite(x).all()


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

        assert weights[0] < 1.0
        assert weights[1] < weights[0]

    def test_symmetry(self):
        """Weights are symmetric."""
        r = np.array([2.0, -2.0])
        weights = huber_weight(r, c=1.345)

        assert_allclose(weights[0], weights[1])


class TestTukeyWeight:
    """Tests for Tukey bisquare weight function."""

    def test_small_residuals(self):
        """Small residuals get positive weight."""
        r = np.array([0.0, 1.0, 2.0])
        weights = tukey_weight(r, c=4.685)

        assert np.all(weights > 0)

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

        assert weights[1] < 0.01  # Very small at boundary
        assert weights[2] == 0.0  # Zero beyond


class TestMAD:
    """Tests for MAD scale estimator."""

    def test_normal_data(self):
        """Consistent for normal data."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)

        scale = mad(data)

        # Should be close to 1 for standard normal
        assert 0.9 < scale < 1.1

    def test_with_outliers(self):
        """Robust to outliers."""
        data = np.array([1, 2, 3, 4, 5, 1000])

        scale = mad(data)

        # Should not be dominated by outlier
        assert scale < 10


class TestIRLS:
    """Tests for IRLS algorithm."""

    def test_convergence(self):
        """Algorithm converges."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        b = np.array([2, 3, 4, 5])

        result = irls(A, b)

        assert result.converged

    def test_outlier_downweighting(self):
        """Outliers get lower weights."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
        b = np.array([2, 3, 4, 5, 100])  # Last is outlier

        result = irls(A, b)

        # Outlier should have low weight
        assert result.weights[-1] < result.weights[0]


class TestHuberRegression:
    """Tests for Huber regression."""

    def test_clean_data(self):
        """Matches OLS on clean data."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        b = np.array([2, 3, 4, 5])

        result_huber = huber_regression(A, b)
        result_ols = ordinary_least_squares(A, b)

        assert_allclose(result_huber.x, result_ols.x, atol=0.1)

    def test_outlier_resistance(self):
        """Resistant to outliers."""
        rng = np.random.default_rng(42)
        n = 50
        A = np.column_stack([np.ones(n), np.arange(n)])
        b = A @ np.array([1.0, 1.0]) + rng.normal(0, 0.5, n)
        # Add a few outliers
        b[-5:] = 100

        result = huber_regression(A, b)

        # Slope should be close to 1, not pulled by outliers
        assert 0.8 < result.x[1] < 1.2


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

        result = tukey_regression(A, b)

        # Outlier should have much lower weight than typical inliers
        median_weight = np.median(result.weights[:-1])
        assert result.weights[-1] < median_weight * 0.5


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

        # Should find most inliers
        assert result.n_inliers >= 60

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

        # Should be close to true parameters
        assert_allclose(result.x, true_x, atol=0.5)


class TestRANSACNTrials:
    """Tests for RANSAC trial computation."""

    def test_low_outlier_rate(self):
        """Few trials for low outlier rate."""
        n_trials = ransac_n_trials(100, 10, 2)  # 10% outliers

        assert n_trials < 20

    def test_high_outlier_rate(self):
        """More trials for high outlier rate."""
        n_trials_low = ransac_n_trials(100, 10, 2)
        n_trials_high = ransac_n_trials(100, 40, 2)

        assert n_trials_high > n_trials_low

    def test_more_samples_need_more_trials(self):
        """More samples per trial needs more trials."""
        n_trials_2 = ransac_n_trials(100, 30, 2)
        n_trials_5 = ransac_n_trials(100, 30, 5)

        assert n_trials_5 > n_trials_2


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

        assert rho[0] < rho[1]
        assert_allclose(rho[1], rho[2])  # Both at max


class TestCauchyWeight:
    """Tests for Cauchy weight function."""

    def test_decays_smoothly(self):
        """Weights decay smoothly."""
        r = np.array([0.0, 1.0, 2.0, 4.0])
        weights = cauchy_weight(r, c=2.385)

        assert weights[0] == 1.0
        assert weights[1] > weights[2] > weights[3]
        assert np.all(weights > 0)  # Never zero


class TestTauScale:
    """Tests for tau scale estimator."""

    def test_robust_to_outliers(self):
        """Tau scale is robust to outliers."""
        clean = np.array([1, 2, 3, 4, 5])
        with_outlier = np.array([1, 2, 3, 4, 100])

        scale_clean = tau_scale(clean)
        scale_outlier = tau_scale(with_outlier)

        # Should not be heavily affected by outlier
        assert scale_outlier < 10 * scale_clean
