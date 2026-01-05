"""
Tests for gamma and related functions.

Tests cover:
- Gamma function and log gamma
- Incomplete gamma functions
- Digamma and polygamma
- Beta function and incomplete beta
- Factorial and double factorial
- Combinations and permutations
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.gamma_functions import (
    beta,
    betainc,
    betaincinv,
    betaln,
    comb,
    digamma,
    factorial,
    factorial2,
    gamma,
    gammainc,
    gammaincc,
    gammaincinv,
    gammaln,
    perm,
    polygamma,
)

# =============================================================================
# Tests for gamma function
# =============================================================================


class TestGamma:
    """Tests for gamma function."""

    def test_gamma_integers(self):
        """Test Γ(n) = (n-1)! for positive integers."""
        import math

        for n in range(1, 8):
            result = gamma(n)
            expected = math.factorial(n - 1)
            assert result == pytest.approx(expected, rel=1e-10)

    def test_gamma_half(self):
        """Test Γ(1/2) = √π."""
        result = gamma(0.5)
        assert result == pytest.approx(np.sqrt(np.pi), rel=1e-10)

    def test_gamma_array(self):
        """Test gamma with array input."""
        x = np.array([1, 2, 3, 4, 5])
        result = gamma(x)
        expected = np.array([1, 1, 2, 6, 24])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_reflection(self):
        """Test Γ(x)Γ(1-x) = π/sin(πx)."""
        x = 0.3
        result = gamma(x) * gamma(1 - x)
        expected = np.pi / np.sin(np.pi * x)
        assert result == pytest.approx(expected, rel=1e-10)


class TestGammaln:
    """Tests for log gamma function."""

    def test_gammaln_positive(self):
        """Test log gamma for positive values."""
        x = 5
        result = gammaln(x)
        expected = np.log(gamma(x))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_gammaln_large(self):
        """Test log gamma for large values."""
        x = 100
        result = gammaln(x)
        # Should be finite
        assert np.isfinite(result)
        # Stirling's approximation: log(n!) ≈ n*log(n) - n
        assert result > 300

    def test_gammaln_array(self):
        """Test log gamma with array input."""
        x = np.array([1, 2, 5, 10])
        result = gammaln(x)
        assert len(result) == 4
        assert np.all(np.isfinite(result))


# =============================================================================
# Tests for incomplete gamma functions
# =============================================================================


class TestGammainc:
    """Tests for regularized lower incomplete gamma."""

    def test_gammainc_exponential_cdf(self):
        """Test gammainc(1, x) = 1 - exp(-x) (exponential CDF)."""
        x = 1.0
        result = gammainc(1, x)
        expected = 1 - np.exp(-x)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_gammainc_zero(self):
        """Test gammainc(a, 0) = 0."""
        result = gammainc(2, 0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_gammainc_large_x(self):
        """Test gammainc approaches 1 for large x."""
        result = gammainc(2, 100)
        assert result == pytest.approx(1.0, rel=1e-10)


class TestGammaincc:
    """Tests for regularized upper incomplete gamma."""

    def test_gammaincc_complement(self):
        """Test gammainc + gammaincc = 1."""
        a, x = 2.0, 1.5
        lower = gammainc(a, x)
        upper = gammaincc(a, x)
        assert lower + upper == pytest.approx(1.0, rel=1e-10)

    def test_gammaincc_exponential(self):
        """Test gammaincc(1, x) = exp(-x)."""
        x = 1.0
        result = gammaincc(1, x)
        expected = np.exp(-x)
        assert result == pytest.approx(expected, rel=1e-10)


class TestGammaincinv:
    """Tests for inverse of lower incomplete gamma."""

    def test_gammaincinv_roundtrip(self):
        """Test gammaincinv is inverse of gammainc."""
        a = 2.0
        x = 1.5
        y = gammainc(a, x)
        x_recovered = gammaincinv(a, y)
        assert x_recovered == pytest.approx(x, rel=1e-10)

    def test_gammaincinv_median(self):
        """Test finding median of gamma distribution."""
        a = 2.0
        median = gammaincinv(a, 0.5)
        assert np.isfinite(median)
        assert median > 0


# =============================================================================
# Tests for digamma and polygamma
# =============================================================================


class TestDigamma:
    """Tests for digamma (psi) function."""

    def test_digamma_one(self):
        """Test ψ(1) = -γ (negative Euler-Mascheroni)."""
        result = digamma(1)
        euler_gamma = 0.5772156649015329
        assert result == pytest.approx(-euler_gamma, rel=1e-6)

    def test_digamma_recurrence(self):
        """Test ψ(x+1) = ψ(x) + 1/x."""
        x = 2.5
        result1 = digamma(x + 1)
        result2 = digamma(x) + 1 / x
        assert result1 == pytest.approx(result2, rel=1e-10)

    def test_digamma_array(self):
        """Test digamma with array input."""
        x = np.array([1, 2, 3, 4])
        result = digamma(x)
        assert len(result) == 4
        assert np.all(np.isfinite(result))


class TestPolygamma:
    """Tests for polygamma function."""

    def test_polygamma_0_is_digamma(self):
        """Test polygamma(0, x) = digamma(x)."""
        x = 2.5
        result = polygamma(0, x)
        expected = digamma(x)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_trigamma(self):
        """Test trigamma ψ'(1) = π²/6."""
        result = polygamma(1, 1)
        expected = np.pi**2 / 6
        assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Tests for beta functions
# =============================================================================


class TestBeta:
    """Tests for beta function."""

    def test_beta_symmetry(self):
        """Test B(a, b) = B(b, a)."""
        a, b = 2.5, 3.5
        assert beta(a, b) == pytest.approx(beta(b, a), rel=1e-10)

    def test_beta_ones(self):
        """Test B(1, 1) = 1."""
        result = beta(1, 1)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_beta_half_half(self):
        """Test B(1/2, 1/2) = π."""
        result = beta(0.5, 0.5)
        assert result == pytest.approx(np.pi, rel=1e-10)

    def test_beta_gamma_relation(self):
        """Test B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
        a, b = 2.0, 3.0
        result = beta(a, b)
        expected = gamma(a) * gamma(b) / gamma(a + b)
        assert result == pytest.approx(expected, rel=1e-10)


class TestBetaln:
    """Tests for log beta function."""

    def test_betaln_equals_log_beta(self):
        """Test betaln(a, b) = log(beta(a, b))."""
        a, b = 2.0, 3.0
        result = betaln(a, b)
        expected = np.log(beta(a, b))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_betaln_large_values(self):
        """Test betaln for large values."""
        result = betaln(100, 100)
        assert np.isfinite(result)


class TestBetainc:
    """Tests for incomplete beta function."""

    def test_betainc_uniform(self):
        """Test betainc(1, 1, x) = x (uniform CDF)."""
        x = 0.7
        result = betainc(1, 1, x)
        assert result == pytest.approx(x, rel=1e-10)

    def test_betainc_endpoints(self):
        """Test betainc at endpoints."""
        assert betainc(2, 3, 0) == pytest.approx(0.0, abs=1e-10)
        assert betainc(2, 3, 1) == pytest.approx(1.0, rel=1e-10)


class TestBetaincinv:
    """Tests for inverse incomplete beta function."""

    def test_betaincinv_roundtrip(self):
        """Test betaincinv is inverse of betainc."""
        a, b, x = 2.0, 3.0, 0.4
        y = betainc(a, b, x)
        x_recovered = betaincinv(a, b, y)
        assert x_recovered == pytest.approx(x, rel=1e-6)

    def test_betaincinv_uniform_median(self):
        """Test median of uniform is 0.5."""
        result = betaincinv(1, 1, 0.5)
        assert result == pytest.approx(0.5, rel=1e-10)


# =============================================================================
# Tests for factorial functions
# =============================================================================


class TestFactorial:
    """Tests for factorial function."""

    def test_factorial_small(self):
        """Test factorial for small values."""
        expected = [1, 1, 2, 6, 24, 120]
        for n, exp in enumerate(expected):
            assert factorial(n) == pytest.approx(exp, rel=1e-10)

    def test_factorial_array(self):
        """Test factorial with array input."""
        n = np.array([1, 2, 3, 4, 5])
        result = factorial(n)
        expected = np.array([1, 2, 6, 24, 120])
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestFactorial2:
    """Tests for double factorial."""

    def test_factorial2_odd(self):
        """Test n!! for odd n."""
        assert factorial2(5) == pytest.approx(15, rel=1e-10)  # 5*3*1
        assert factorial2(7) == pytest.approx(105, rel=1e-10)  # 7*5*3*1

    def test_factorial2_even(self):
        """Test n!! for even n."""
        assert factorial2(6) == pytest.approx(48, rel=1e-10)  # 6*4*2
        assert factorial2(8) == pytest.approx(384, rel=1e-10)  # 8*6*4*2


# =============================================================================
# Tests for combinations and permutations
# =============================================================================


class TestComb:
    """Tests for binomial coefficient."""

    def test_comb_basic(self):
        """Test basic combinations."""
        assert comb(5, 2) == pytest.approx(10, rel=1e-10)
        assert comb(10, 3) == pytest.approx(120, rel=1e-10)

    def test_comb_symmetry(self):
        """Test C(n, k) = C(n, n-k)."""
        n = 8
        for k in range(n + 1):
            assert comb(n, k) == pytest.approx(comb(n, n - k), rel=1e-10)

    def test_comb_boundary(self):
        """Test boundary cases."""
        assert comb(5, 0) == pytest.approx(1, rel=1e-10)
        assert comb(5, 5) == pytest.approx(1, rel=1e-10)


class TestPerm:
    """Tests for permutation coefficient."""

    def test_perm_basic(self):
        """Test basic permutations."""
        assert perm(5, 2) == pytest.approx(20, rel=1e-10)
        assert perm(10, 3) == pytest.approx(720, rel=1e-10)

    def test_perm_full(self):
        """Test P(n, n) = n!."""
        for n in range(1, 8):
            result = perm(n, n)
            expected = factorial(n)
            assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Integration tests
# =============================================================================


class TestGammaFunctionsIntegration:
    """Integration tests for gamma functions."""

    def test_gamma_beta_relation(self):
        """Test Γ(a)Γ(b)/Γ(a+b) = B(a,b)."""
        a, b = 3.5, 2.5
        gamma_ratio = gamma(a) * gamma(b) / gamma(a + b)
        beta_val = beta(a, b)
        assert gamma_ratio == pytest.approx(beta_val, rel=1e-10)

    def test_incomplete_gamma_chi2(self):
        """Test incomplete gamma relates to chi-squared distribution."""
        # For chi-squared with k degrees of freedom:
        # CDF = gammainc(k/2, x/2)
        k = 4
        x = 5.0
        cdf = gammainc(k / 2, x / 2)
        assert 0 < cdf < 1

    def test_comb_perm_relation(self):
        """Test C(n,k) = P(n,k) / k!."""
        n, k = 10, 4
        result = comb(n, k)
        expected = perm(n, k) / factorial(k)
        assert result == pytest.approx(expected, rel=1e-10)
