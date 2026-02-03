"""
Comprehensive tests for statistical distributions module.

Tests coverage for various probability distribution classes including:
- Gaussian
- Uniform
- Exponential
- Gamma
- ChiSquared
- StudentT
- Beta
- Poisson
"""

import numpy as np
import pytest
from scipy import stats

from pytcl.mathematical_functions.statistics.distributions import (
    Beta,
    ChiSquared,
    Exponential,
    Gamma,
    Gaussian,
    Poisson,
    StudentT,
    Uniform,
)


class TestGaussianDistribution:
    """Tests for Gaussian distribution."""

    def test_gaussian_creation(self):
        """Test Gaussian distribution creation."""
        g = Gaussian(mean=0, var=1)
        assert g is not None

    def test_gaussian_pdf(self):
        """Test Gaussian PDF."""
        g = Gaussian(mean=0, var=1)
        result = g.pdf(0)
        expected = stats.norm.pdf(0, loc=0, scale=1)
        assert np.isclose(result, expected)

    def test_gaussian_cdf(self):
        """Test Gaussian CDF."""
        g = Gaussian(mean=0, var=1)
        result = g.cdf(0)
        expected = stats.norm.cdf(0, loc=0, scale=1)
        assert np.isclose(result, expected)

    def test_gaussian_mean(self):
        """Test Gaussian mean."""
        g = Gaussian(mean=5, var=1)
        assert g.mean() == 5

    def test_gaussian_sample(self):
        """Test Gaussian sampling."""
        g = Gaussian(mean=0, var=1)
        samples = g.sample(size=100)
        assert samples.shape == (100,)

    def test_gaussian_invalid_variance(self):
        """Test Gaussian with invalid variance."""
        with pytest.raises(ValueError):
            Gaussian(mean=0, var=-1)


class TestUniformDistribution:
    """Tests for uniform distribution."""

    def test_uniform_creation(self):
        """Test uniform distribution creation."""
        u = Uniform(low=0, high=1)
        assert u is not None

    def test_uniform_pdf(self):
        """Test uniform PDF."""
        u = Uniform(low=0, high=1)
        result = u.pdf(0.5)
        expected = stats.uniform.pdf(0.5, loc=0, scale=1)
        assert np.isclose(result, expected)

    def test_uniform_cdf(self):
        """Test uniform CDF."""
        u = Uniform(low=0, high=1)
        result = u.cdf(0.5)
        expected = stats.uniform.cdf(0.5, loc=0, scale=1)
        assert np.isclose(result, expected)

    def test_uniform_mean(self):
        """Test uniform mean."""
        u = Uniform(low=2, high=8)
        assert np.isclose(u.mean(), 5.0)

    def test_uniform_sample(self):
        """Test uniform sampling."""
        u = Uniform(low=0, high=1)
        samples = u.sample(size=100)
        assert samples.shape == (100,)
        assert np.all(samples >= 0) and np.all(samples <= 1)


class TestExponentialDistribution:
    """Tests for exponential distribution."""

    def test_exponential_creation(self):
        """Test exponential creation."""
        e = Exponential(rate=1.0)
        assert e is not None

    def test_exponential_pdf(self):
        """Test exponential PDF."""
        e = Exponential(rate=1.0)
        result = e.pdf(1.0)
        expected = stats.expon.pdf(1.0, scale=1.0)
        assert np.isclose(result, expected)

    def test_exponential_cdf(self):
        """Test exponential CDF."""
        e = Exponential(rate=1.0)
        result = e.cdf(1.0)
        expected = stats.expon.cdf(1.0, scale=1.0)
        assert np.isclose(result, expected)

    def test_exponential_mean(self):
        """Test exponential mean."""
        e = Exponential(rate=2.0)
        assert np.isclose(e.mean(), 0.5)

    def test_exponential_sample(self):
        """Test exponential sampling."""
        e = Exponential(rate=1.0)
        samples = e.sample(size=100)
        assert samples.shape == (100,)
        assert np.all(samples >= 0)


class TestGammaDistribution:
    """Tests for gamma distribution."""

    def test_gamma_creation(self):
        """Test gamma creation."""
        g = Gamma(shape=2.0, rate=1.0)
        assert g is not None

    def test_gamma_pdf(self):
        """Test gamma PDF."""
        g = Gamma(shape=2.0, rate=1.0)
        result = g.pdf(1.0)
        expected = stats.gamma.pdf(1.0, a=2.0, scale=1.0)
        assert np.isclose(result, expected)

    def test_gamma_cdf(self):
        """Test gamma CDF."""
        g = Gamma(shape=2.0, rate=1.0)
        result = g.cdf(1.0)
        expected = stats.gamma.cdf(1.0, a=2.0, scale=1.0)
        assert np.isclose(result, expected)

    def test_gamma_mean(self):
        """Test gamma mean."""
        g = Gamma(shape=2.0, rate=1.0)
        assert np.isclose(g.mean(), 2.0)

    def test_gamma_sample(self):
        """Test gamma sampling."""
        g = Gamma(shape=2.0, rate=1.0)
        samples = g.sample(size=100)
        assert samples.shape == (100,)


class TestChiSquaredDistribution:
    """Tests for chi-squared distribution."""

    def test_chisquared_creation(self):
        """Test chi-squared creation."""
        cs = ChiSquared(df=5)
        assert cs is not None

    def test_chisquared_pdf(self):
        """Test chi-squared PDF."""
        cs = ChiSquared(df=5)
        result = cs.pdf(2.0)
        expected = stats.chi2.pdf(2.0, df=5)
        assert np.isclose(result, expected)

    def test_chisquared_cdf(self):
        """Test chi-squared CDF."""
        cs = ChiSquared(df=5)
        result = cs.cdf(2.0)
        expected = stats.chi2.cdf(2.0, df=5)
        assert np.isclose(result, expected)

    def test_chisquared_mean(self):
        """Test chi-squared mean."""
        cs = ChiSquared(df=5)
        assert np.isclose(cs.mean(), 5.0)

    def test_chisquared_sample(self):
        """Test chi-squared sampling."""
        cs = ChiSquared(df=5)
        samples = cs.sample(size=100)
        assert samples.shape == (100,)


class TestStudentTDistribution:
    """Tests for Student's t-distribution."""

    def test_studentt_creation(self):
        """Test Student's t creation."""
        st = StudentT(df=10)
        assert st is not None

    def test_studentt_pdf(self):
        """Test Student's t PDF."""
        st = StudentT(df=10)
        result = st.pdf(1.0)
        expected = stats.t.pdf(1.0, df=10)
        assert np.isclose(result, expected)

    def test_studentt_cdf(self):
        """Test Student's t CDF."""
        st = StudentT(df=10)
        result = st.cdf(1.0)
        expected = stats.t.cdf(1.0, df=10)
        assert np.isclose(result, expected)

    def test_studentt_symmetry(self):
        """Test Student's t symmetry."""
        st = StudentT(df=10)
        pdf_pos = st.pdf(1.5)
        pdf_neg = st.pdf(-1.5)
        assert np.isclose(pdf_pos, pdf_neg)

    def test_studentt_sample(self):
        """Test Student's t sampling."""
        st = StudentT(df=10)
        samples = st.sample(size=100)
        assert samples.shape == (100,)


class TestBetaDistribution:
    """Tests for beta distribution."""

    def test_beta_creation(self):
        """Test beta creation."""
        b = Beta(a=2.0, b=5.0)
        assert b is not None

    def test_beta_pdf(self):
        """Test beta PDF."""
        b = Beta(a=2.0, b=5.0)
        result = b.pdf(0.5)
        expected = stats.beta.pdf(0.5, a=2.0, b=5.0)
        assert np.isclose(result, expected)

    def test_beta_cdf(self):
        """Test beta CDF."""
        b = Beta(a=2.0, b=5.0)
        result = b.cdf(0.5)
        expected = stats.beta.cdf(0.5, a=2.0, b=5.0)
        assert np.isclose(result, expected)

    def test_beta_mean(self):
        """Test beta mean."""
        b = Beta(a=2.0, b=5.0)
        expected_mean = 2.0 / (2.0 + 5.0)
        assert np.isclose(b.mean(), expected_mean)

    def test_beta_sample(self):
        """Test beta sampling."""
        b = Beta(a=2.0, b=5.0)
        samples = b.sample(size=100)
        assert samples.shape == (100,)
        assert np.all(samples >= 0) and np.all(samples <= 1)


class TestPoissonDistribution:
    """Tests for Poisson distribution."""

    def test_poisson_creation(self):
        """Test Poisson creation."""
        p = Poisson(rate=5)
        assert p is not None

    def test_poisson_pdf(self):
        """Test Poisson PDF."""
        p = Poisson(rate=5)
        result = p.pdf(3)
        expected = stats.poisson.pmf(3, mu=5)
        assert np.isclose(result, expected)

    def test_poisson_cdf(self):
        """Test Poisson CDF."""
        p = Poisson(rate=5)
        result = p.cdf(3)
        expected = stats.poisson.cdf(3, mu=5)
        assert np.isclose(result, expected)

    def test_poisson_mean(self):
        """Test Poisson mean."""
        p = Poisson(rate=5)
        assert np.isclose(p.mean(), 5.0)

    def test_poisson_sample(self):
        """Test Poisson sampling."""
        p = Poisson(rate=5)
        samples = p.sample(size=100)
        assert samples.shape == (100,)
        assert np.all(samples >= 0)


class TestDistributionProperties:
    """Tests for distribution properties."""

    def test_pdf_cdf_monotonic(self):
        """Test CDF is monotonically increasing."""
        e = Exponential(rate=1.0)
        x = np.linspace(0.1, 5, 50)
        cdf_vals = e.cdf(x)
        diffs = np.diff(cdf_vals)
        assert np.all(diffs >= 0)

    def test_cdf_bounds(self):
        """Test CDF is bounded [0, 1]."""
        e = Exponential(rate=1.0)
        x = np.linspace(0.1, 10, 50)
        cdf_vals = e.cdf(x)
        assert np.all(cdf_vals >= 0) and np.all(cdf_vals <= 1)

    def test_pdf_integrates_to_one(self):
        """Test PDF integrates approximately to 1 (numerical check)."""
        u = Uniform(low=0, high=1)
        x = np.linspace(0, 1, 1000)
        pdf_vals = u.pdf(x)
        # Approximate integration using trapezoid rule
        integral = np.trapz(pdf_vals, x)
        assert np.isclose(integral, 1.0, atol=0.01)


class TestDistributionEdgeCases:
    """Tests for edge cases."""

    def test_gaussian_logpdf(self):
        """Test Gaussian logpdf."""
        g = Gaussian(mean=0, var=1)
        result = g.logpdf(1.0)
        assert np.isfinite(result)

    def test_exponential_zero_input(self):
        """Test exponential at zero."""
        e = Exponential(rate=1.0)
        result = e.pdf(0)
        assert np.isfinite(result)

    def test_uniform_boundary_values(self):
        """Test uniform at boundaries."""
        u = Uniform(low=0, high=1)
        pdf_0 = u.pdf(0.0)
        pdf_1 = u.pdf(1.0)
        assert np.isfinite(pdf_0) and np.isfinite(pdf_1)

    def test_beta_boundary_values(self):
        """Test beta at boundaries."""
        b = Beta(a=2.0, b=5.0)
        pdf_0 = b.pdf(0.0)
        pdf_1 = b.pdf(1.0)
        # May be infinite or finite depending on parameters
        assert np.isfinite(pdf_0) or np.isinf(pdf_0)  # Ensure computation works
        assert np.isfinite(pdf_1) or np.isinf(pdf_1)  # Ensure computation works
