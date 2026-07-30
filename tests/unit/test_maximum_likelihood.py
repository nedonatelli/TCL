"""Tests for maximum likelihood estimation module."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.static_estimation import (
    aic,
    aicc,
    bic,
    cramer_rao_bound,
    cramer_rao_bound_biased,
    efficiency,
    fisher_information_gaussian,
    fisher_information_numerical,
    mle_gaussian,
    mle_newton_raphson,
    mle_scoring,
)


class TestFisherInformation:
    """Tests for Fisher information computation."""

    def test_numerical_gaussian(self):
        """Fisher info for Gaussian mean estimation."""
        # For N(mu, sigma^2), I(mu) = n/sigma^2
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = 1.0

        def log_lik(theta):
            return -0.5 * np.sum((data - theta[0]) ** 2) / sigma**2

        F = fisher_information_numerical(log_lik, np.array([3.0]))

        # Expected: n/sigma^2 = 5
        assert_allclose(F[0, 0], 5.0, rtol=0.1)

    def test_gaussian_linear_model(self):
        """Fisher info for linear Gaussian model."""
        H = np.array([[1, 0], [0, 1], [1, 1]])
        R = np.eye(3) * 0.5

        F = fisher_information_gaussian(H, R)

        # F = H^T R^{-1} H
        expected = H.T @ np.linalg.inv(R) @ H
        assert_allclose(F, expected)

    def test_fisher_positive_definite(self):
        """Fisher info should be positive semi-definite."""
        H = np.random.randn(5, 3)
        R = np.eye(5)

        F = fisher_information_gaussian(H, R)

        eigenvalues = np.linalg.eigvalsh(F)
        assert np.all(eigenvalues >= -1e-10)


class TestCramerRaoBound:
    """Tests for Cramer-Rao bound."""

    def test_basic_crb(self):
        """Basic CRB computation."""
        F = np.array([[10, 0], [0, 5]])

        result = cramer_rao_bound(F)

        assert_allclose(result.variances, [0.1, 0.2])
        assert_allclose(result.std_bounds, [np.sqrt(0.1), np.sqrt(0.2)])

    def test_crb_matrix_inverse(self):
        """CRB is inverse of Fisher info."""
        F = np.array([[4, 1], [1, 2]])

        result = cramer_rao_bound(F)

        assert_allclose(result.crb_matrix, np.linalg.inv(F))

    def test_efficiency_computation(self):
        """Efficiency relative to CRB."""
        var_est = np.array([0.12, 0.25])
        crb_vals = np.array([0.1, 0.2])

        eff = efficiency(var_est, crb_vals)

        # Efficiency = CRB / actual variance
        assert_allclose(eff, crb_vals / var_est)
        assert np.all(eff <= 1.0)

    def test_biased_crb(self):
        """CRB for biased estimator."""
        F = np.array([[10, 0], [0, 5]])
        bias_grad = np.array([[0.1, 0], [0, 0.2]])

        crb = cramer_rao_bound_biased(F, bias_grad)

        assert crb.shape == (2, 2)


class TestMLEGaussian:
    """Tests for Gaussian MLE."""

    def test_mean_estimation(self):
        """MLE estimates mean correctly."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        data = rng.normal(true_mean, 1.0, 1000)

        result = mle_gaussian(data, estimate_variance=False)

        assert_allclose(result.theta[0], true_mean, atol=0.1)
        assert result.converged

    def test_variance_estimation(self):
        """MLE estimates variance correctly."""
        rng = np.random.default_rng(42)
        true_var = 4.0
        data = rng.normal(0, np.sqrt(true_var), 1000)

        result = mle_gaussian(data)

        assert_allclose(result.theta[1], true_var, atol=0.5)

    def test_fisher_info_returned(self):
        """MLE returns Fisher information."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        result = mle_gaussian(data)

        assert result.fisher_info.shape[0] == len(result.theta)


class TestMLENewtonRaphson:
    """Tests for Newton-Raphson MLE."""

    def test_convergence(self):
        """Newton-Raphson converges."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 100)

        def log_lik(theta):
            return -0.5 * np.sum((data - theta[0]) ** 2)

        def score(theta):
            return np.array([np.sum(data - theta[0])])

        result = mle_newton_raphson(log_lik, score, np.array([0.0]), max_iter=50)

        assert_allclose(result.theta[0], np.mean(data), atol=0.1)

    def test_returns_covariance(self):
        """Returns covariance estimate."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        def log_lik(theta):
            return -0.5 * np.sum((data - theta[0]) ** 2)

        def score(theta):
            return np.array([np.sum(data - theta[0])])

        result = mle_newton_raphson(log_lik, score, np.array([0.0]))

        assert result.covariance.shape == (1, 1)
        assert result.covariance[0, 0] > 0


class TestMLEScoring:
    """Tests for Fisher scoring MLE."""

    def test_convergence(self):
        """Fisher scoring converges."""
        rng = np.random.default_rng(42)
        data = rng.normal(3.0, 1.0, 100)

        def log_lik(theta):
            return -0.5 * np.sum((data - theta[0]) ** 2)

        def score(theta):
            return np.array([np.sum(data - theta[0])])

        def fisher(theta):
            return np.array([[len(data)]])

        result = mle_scoring(log_lik, score, fisher, np.array([0.0]))

        assert_allclose(result.theta[0], np.mean(data), atol=0.1)


class TestInformationCriteria:
    """Tests for information criteria."""

    def test_aic_formula(self):
        """AIC follows correct formula."""
        log_lik = -100.0
        k = 3

        result = aic(log_lik, k)

        assert_allclose(result, -2 * log_lik + 2 * k)

    def test_bic_formula(self):
        """BIC follows correct formula."""
        log_lik = -100.0
        k = 3
        n = 50

        result = bic(log_lik, k, n)

        assert_allclose(result, -2 * log_lik + k * np.log(n))

    def test_aicc_correction(self):
        """AICc adds small-sample correction."""
        log_lik = -100.0
        k = 3
        n = 50

        aic_val = aic(log_lik, k)
        aicc_val = aicc(log_lik, k, n)

        # AICc should be larger than AIC
        assert aicc_val > aic_val

    def test_aicc_small_sample(self):
        """AICc handles edge cases."""
        # When n - k - 1 <= 0, should return inf
        result = aicc(-100.0, 10, 10)
        assert result == np.inf

    def test_model_selection(self):
        """Lower AIC/BIC indicates better model."""
        # Better model has higher log-likelihood
        aic1 = aic(-100.0, 2)  # Good model
        aic2 = aic(-150.0, 2)  # Worse model

        assert aic1 < aic2  # Lower is better
