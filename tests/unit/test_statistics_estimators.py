"""
Tests for statistical estimator functions.

Tests cover:
- Weighted mean, variance, covariance
- Sample mean, variance, covariance, correlation
- Median, MAD, IQR
- Skewness, kurtosis, moments
- NEES and NIS
"""

import numpy as np

from pytcl.mathematical_functions.statistics.estimators import (
    iqr,
    kurtosis,
    mad,
    median,
    moment,
    nees,
    nis,
    sample_corr,
    sample_cov,
    sample_mean,
    sample_var,
    skewness,
    weighted_cov,
    weighted_mean,
    weighted_var,
)


class TestWeightedEstimators:
    """Tests for weighted statistical estimators."""

    def test_weighted_mean(self):
        """Test weighted mean computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        mean = weighted_mean(x, weights)
        assert np.isclose(mean, 2.5)

        weights = np.array([0.1, 0.2, 0.3, 0.4])
        mean = weighted_mean(x, weights)
        expected = 0.1 * 1 + 0.2 * 2 + 0.3 * 3 + 0.4 * 4
        assert np.isclose(mean, expected)

    def test_weighted_var(self):
        """Test weighted variance computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        var = weighted_var(x, weights)
        assert var > 0

        var_corrected = weighted_var(x, weights, ddof=1)
        assert var_corrected > var

    def test_weighted_cov(self):
        """Test weighted covariance matrix."""
        np.random.seed(42)
        x = np.random.randn(100, 3)
        weights = np.ones(100)
        cov = weighted_cov(x, weights)

        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_weighted_cov_1d(self):
        """Test weighted covariance with 1D input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        cov = weighted_cov(x, weights)
        assert cov.shape == (1, 1)


class TestSampleEstimators:
    """Tests for sample statistical estimators."""

    def test_sample_mean(self):
        """Test sample mean."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sample_mean(x)
        assert np.isclose(mean, 3.0)

    def test_sample_var(self):
        """Test sample variance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = sample_var(x, ddof=0)
        assert np.isclose(var, 2.0)

        var_sample = sample_var(x, ddof=1)
        assert np.isclose(var_sample, 2.5)

    def test_sample_cov(self):
        """Test sample covariance."""
        np.random.seed(42)
        x = np.random.randn(100, 2)
        cov = sample_cov(x)

        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)

    def test_sample_cov_1d(self):
        """Test sample covariance with 1D input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = sample_cov(x)
        assert np.isclose(var, 2.5)

    def test_sample_cov_cross(self):
        """Test cross-covariance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cov = sample_cov(x, y)

        assert cov.shape == (2, 2)
        assert cov[0, 1] > 0

    def test_sample_corr(self):
        """Test sample correlation."""
        np.random.seed(42)
        x = np.random.randn(100, 3)
        corr = sample_corr(x)

        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T)
        assert np.all(corr >= -1) and np.all(corr <= 1)


class TestRobustEstimators:
    """Tests for robust statistical estimators."""

    def test_median(self):
        """Test median computation."""
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        med = median(x)
        assert med == 3.0

        x_even = np.array([1.0, 2.0, 3.0, 4.0])
        med_even = median(x_even)
        assert med_even == 2.5

    def test_mad(self):
        """Test median absolute deviation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad(x)
        assert result > 0

        np.random.seed(42)
        normal_data = np.random.randn(10000)
        mad_scaled = mad(normal_data)
        assert np.isclose(mad_scaled, 1.0, atol=0.1)

    def test_iqr(self):
        """Test interquartile range."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = iqr(x)
        assert result > 0


class TestHigherOrderEstimators:
    """Tests for skewness, kurtosis, and moments."""

    def test_skewness(self):
        """Test skewness computation."""
        symmetric = np.array([-2, -1, 0, 1, 2])
        skew = skewness(symmetric)
        assert np.isclose(skew, 0.0, atol=1e-10)

        right_skewed = np.array([1, 2, 2, 3, 3, 3, 10])
        skew_right = skewness(right_skewed)
        assert skew_right > 0

    def test_kurtosis(self):
        """Test kurtosis computation."""
        np.random.seed(42)
        normal_data = np.random.randn(10000)

        kurt = kurtosis(normal_data, fisher=True)
        assert np.isclose(kurt, 0.0, atol=0.2)

        kurt_pearson = kurtosis(normal_data, fisher=False)
        assert np.isclose(kurt_pearson, 3.0, atol=0.2)

    def test_moment(self):
        """Test moment computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        m2 = moment(x, order=2, central=True)
        var = np.var(x, ddof=0)
        assert np.isclose(m2, var)

        m2_raw = moment(x, order=2, central=False)
        assert np.isclose(m2_raw, np.mean(x**2))


class TestConsistencyMetrics:
    """Tests for NEES and NIS."""

    def test_nees(self):
        """Test Normalized Estimation Error Squared."""
        error = np.array([1.0, 0.0])
        cov = np.eye(2)
        result = nees(error, cov)
        assert np.isclose(result, 1.0)

        errors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        results = nees(errors, cov)
        assert results.shape == (3,)
        assert np.allclose(results, [1.0, 1.0, 2.0])

    def test_nis(self):
        """Test Normalized Innovation Squared."""
        innovation = np.array([0.5, 0.5])
        S = np.eye(2) * 0.25
        result = nis(innovation, S)
        expected = 0.5**2 / 0.25 + 0.5**2 / 0.25
        assert np.isclose(result, expected)
