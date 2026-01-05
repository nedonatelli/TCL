"""
Tests for hypergeometric functions.

Tests cover:
- 0F1 (confluent hypergeometric limit function)
- 1F1 (Kummer's function M)
- 2F1 (Gauss hypergeometric function)
- U (Tricomi function)
- Regularized 1F1
- Pochhammer symbol (rising factorial)
- Falling factorial
- Generalized pFq hypergeometric function
"""

import numpy as np
import pytest
from scipy.special import gamma

from pytcl.mathematical_functions.special_functions.hypergeometric import (
    falling_factorial,
    generalized_hypergeometric,
    hyp0f1,
    hyp1f1,
    hyp1f1_regularized,
    hyp2f1,
    hyperu,
    pochhammer,
)

# =============================================================================
# Tests for 0F1
# =============================================================================


class TestHyp0F1:
    """Tests for confluent hypergeometric limit function 0F1."""

    def test_hyp0f1_at_zero(self):
        """Test 0F1(b; 0) = 1."""
        result = hyp0f1(1.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_hyp0f1_basic(self):
        """Test basic 0F1 evaluation."""
        result = hyp0f1(1.0, 1.0)
        # 0F1(1; 1) is related to Bessel function I_0
        assert result > 1.0
        assert np.isfinite(result)

    def test_hyp0f1_negative_z(self):
        """Test 0F1 with negative argument (related to J Bessel)."""
        result = hyp0f1(1.0, -1.0)
        assert np.isfinite(result)

    def test_hyp0f1_array_input(self):
        """Test 0F1 with array input."""
        z = np.array([0.0, 0.5, 1.0, 2.0])
        result = hyp0f1(1.0, z)
        assert len(result) == 4
        assert result[0] == pytest.approx(1.0)

    def test_hyp0f1_various_b(self):
        """Test 0F1 with various b values."""
        for b in [0.5, 1.0, 2.0, 3.5]:
            result = hyp0f1(b, 1.0)
            assert np.isfinite(result)


# =============================================================================
# Tests for 1F1 (Kummer's function M)
# =============================================================================


class TestHyp1F1:
    """Tests for confluent hypergeometric function 1F1."""

    def test_hyp1f1_exp_identity(self):
        """Test 1F1(a; a; z) = exp(z)."""
        z = 1.0
        result = hyp1f1(2.0, 2.0, z)
        expected = np.exp(z)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_hyp1f1_zero_a(self):
        """Test 1F1(0; b; z) = 1."""
        result = hyp1f1(0.0, 1.0, 1.0)
        assert result == pytest.approx(1.0)

    def test_hyp1f1_at_zero(self):
        """Test 1F1(a; b; 0) = 1."""
        result = hyp1f1(1.0, 2.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_hyp1f1_basic(self):
        """Test basic 1F1 evaluation."""
        result = hyp1f1(1.0, 2.0, 1.0)
        # (exp(1) - 1) for 1F1(1; 2; 1)
        expected = np.exp(1.0) - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_hyp1f1_array_input(self):
        """Test 1F1 with array input."""
        z = np.array([0.0, 0.5, 1.0])
        result = hyp1f1(1.0, 2.0, z)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)

    def test_hyp1f1_negative_z(self):
        """Test 1F1 with negative argument."""
        result = hyp1f1(0.5, 1.5, -1.0)
        assert np.isfinite(result)
        assert 0 < result < 1  # Decreasing for negative z


# =============================================================================
# Tests for 2F1 (Gauss hypergeometric)
# =============================================================================


class TestHyp2F1:
    """Tests for Gauss hypergeometric function 2F1."""

    def test_hyp2f1_at_zero(self):
        """Test 2F1(a, b; c; 0) = 1."""
        result = hyp2f1(1.0, 1.0, 2.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_hyp2f1_log_identity(self):
        """Test 2F1(1, 1; 2; z) = -log(1-z)/z."""
        z = 0.5
        result = hyp2f1(1.0, 1.0, 2.0, z)
        expected = -np.log(1 - z) / z
        assert result == pytest.approx(expected, rel=1e-10)

    def test_hyp2f1_power_identity(self):
        """Test 2F1(a, b; b; z) = (1-z)^(-a)."""
        a = 2.0
        b = 3.0
        z = 0.3
        result = hyp2f1(a, b, b, z)
        expected = (1 - z) ** (-a)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_hyp2f1_array_input(self):
        """Test 2F1 with array input."""
        z = np.array([0.0, 0.25, 0.5])
        result = hyp2f1(1.0, 1.0, 2.0, z)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)

    def test_hyp2f1_arcsin_related(self):
        """Test 2F1 related to arcsin."""
        # arcsin(z)/z = 2F1(1/2, 1/2; 3/2; z^2)
        z = 0.5
        result = hyp2f1(0.5, 0.5, 1.5, z**2)
        expected = np.arcsin(z) / z
        assert result == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Tests for U (Tricomi function)
# =============================================================================


class TestHyperU:
    """Tests for Tricomi's confluent hypergeometric function U."""

    def test_hyperu_basic(self):
        """Test basic U function evaluation."""
        result = hyperu(1.0, 1.0, 1.0)
        assert np.isfinite(result)
        assert result > 0

    def test_hyperu_asymptotic(self):
        """Test U asymptotic behavior U(a,b,z) ~ z^(-a) for large z."""
        a = 2.0
        b = 1.0
        z = 100.0
        result = hyperu(a, b, z)
        expected_asymp = z ** (-a)
        # Should be close to z^(-a) for large z
        assert result == pytest.approx(expected_asymp, rel=0.1)

    def test_hyperu_array_input(self):
        """Test U with array input."""
        z = np.array([1.0, 2.0, 5.0])
        result = hyperu(1.0, 1.0, z)
        assert len(result) == 3
        assert np.all(np.isfinite(result))


# =============================================================================
# Tests for regularized 1F1
# =============================================================================


class TestHyp1F1Regularized:
    """Tests for regularized confluent hypergeometric function."""

    def test_hyp1f1_regularized_basic(self):
        """Test basic regularized 1F1."""
        result = hyp1f1_regularized(1.0, 2.0, 1.0)
        expected = hyp1f1(1.0, 2.0, 1.0) / gamma(2.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_hyp1f1_regularized_at_zero(self):
        """Test regularized 1F1 at z=0."""
        result = hyp1f1_regularized(1.0, 2.0, 0.0)
        expected = 1.0 / gamma(2.0)
        assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Tests for Pochhammer symbol
# =============================================================================


class TestPochhammer:
    """Tests for Pochhammer symbol (rising factorial)."""

    def test_pochhammer_zero_n(self):
        """Test (a)_0 = 1."""
        result = pochhammer(5.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_pochhammer_one_n(self):
        """Test (a)_1 = a."""
        result = pochhammer(5.0, 1.0)
        assert result == pytest.approx(5.0)

    def test_pochhammer_factorial(self):
        """Test (1)_n = n!."""
        import math

        result = pochhammer(1.0, 5.0)
        expected = math.factorial(5)
        assert result == pytest.approx(expected)

    def test_pochhammer_general(self):
        """Test (3)_4 = 3*4*5*6 = 360."""
        result = pochhammer(3.0, 4.0)
        expected = 3 * 4 * 5 * 6
        assert result == pytest.approx(expected)

    def test_pochhammer_array_input(self):
        """Test Pochhammer with array input."""
        a = np.array([1.0, 2.0, 3.0])
        n = 3.0
        result = pochhammer(a, n)
        assert len(result) == 3
        # (1)_3 = 6, (2)_3 = 24, (3)_3 = 60
        expected = np.array([6.0, 24.0, 60.0])
        np.testing.assert_allclose(result, expected)


# =============================================================================
# Tests for falling factorial
# =============================================================================


class TestFallingFactorial:
    """Tests for falling factorial."""

    def test_falling_factorial_basic(self):
        """Test (5)_3 falling = 5*4*3 = 60."""
        result = falling_factorial(5.0, 3.0)
        expected = 5 * 4 * 3
        assert result == pytest.approx(expected)

    def test_falling_factorial_zero_n(self):
        """Test (a)_0 falling = 1."""
        result = falling_factorial(5.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_falling_factorial_one_n(self):
        """Test (a)_1 falling = a."""
        result = falling_factorial(5.0, 1.0)
        assert result == pytest.approx(5.0)

    def test_falling_factorial_array_input(self):
        """Test falling factorial with array input."""
        a = np.array([5.0, 6.0, 7.0])
        n = 2.0
        result = falling_factorial(a, n)
        assert len(result) == 3
        # (5)_2 = 20, (6)_2 = 30, (7)_2 = 42
        expected = np.array([20.0, 30.0, 42.0])
        np.testing.assert_allclose(result, expected)


# =============================================================================
# Tests for generalized hypergeometric pFq
# =============================================================================


class TestGeneralizedHypergeometric:
    """Tests for generalized hypergeometric function pFq."""

    def test_pFq_reduces_to_0F1(self):
        """Test pFq reduces to 0F1 for p=0, q=1."""
        result = generalized_hypergeometric([], [1.0], 1.0)
        expected = hyp0f1(1.0, 1.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_pFq_reduces_to_1F1(self):
        """Test pFq reduces to 1F1 for p=1, q=1."""
        result = generalized_hypergeometric([1.0], [2.0], 1.0)
        expected = hyp1f1(1.0, 2.0, 1.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_pFq_reduces_to_2F1(self):
        """Test pFq reduces to 2F1 for p=2, q=1."""
        result = generalized_hypergeometric([1.0, 1.0], [2.0], 0.5)
        expected = hyp2f1(1.0, 1.0, 2.0, 0.5)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_pFq_general_case(self):
        """Test generalized pFq for p=3, q=2."""
        # 3F2 with known values
        a = [1.0, 1.0, 1.0]
        b = [2.0, 2.0]
        z = 0.5
        result = generalized_hypergeometric(a, b, z)
        assert np.isfinite(result)
        assert result > 1  # Should be > 1 for positive a, b, z

    def test_pFq_array_z(self):
        """Test pFq with array argument z."""
        a = [1.0, 1.0]
        b = [2.0]
        z = np.array([0.0, 0.25, 0.5])
        result = generalized_hypergeometric(a, b, z)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)

    def test_pFq_convergence(self):
        """Test pFq convergence for p <= q."""
        # For p < q, series converges for all z
        a = [1.0]
        b = [2.0, 3.0]  # 1F2
        z = 10.0  # Large z
        result = generalized_hypergeometric(a, b, z)
        assert np.isfinite(result)

    def test_pFq_custom_tolerance(self):
        """Test pFq with custom tolerance."""
        a = [1.0, 1.0, 1.0]
        b = [2.0, 2.0]
        z = 0.3
        result1 = generalized_hypergeometric(a, b, z, tol=1e-10)
        result2 = generalized_hypergeometric(a, b, z, tol=1e-15)
        # Both should give similar results
        assert result1 == pytest.approx(result2, rel=1e-6)

    def test_pFq_max_terms(self):
        """Test pFq with custom max terms."""
        a = [1.0, 1.0]
        b = [2.0]
        z = 0.5
        result = generalized_hypergeometric(a, b, z, max_terms=100)
        expected = hyp2f1(1.0, 1.0, 2.0, 0.5)
        assert result == pytest.approx(expected, rel=1e-8)


# =============================================================================
# Integration tests
# =============================================================================


class TestHypergeometricIntegration:
    """Integration tests for hypergeometric functions."""

    def test_kummer_relation(self):
        """Test Kummer's transformation: M(a,b,z) = e^z * M(b-a,b,-z)."""
        a, b, z = 1.0, 3.0, 1.0
        lhs = hyp1f1(a, b, z)
        rhs = np.exp(z) * hyp1f1(b - a, b, -z)
        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_gauss_summation(self):
        """Test Gauss's summation: 2F1(a,b;c;1) for convergent case."""
        # 2F1(a, b; c; 1) = Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b))
        # when Re(c-a-b) > 0
        from scipy.special import gamma

        a, b, c = 0.5, 0.5, 2.0  # c - a - b = 1 > 0
        result = hyp2f1(a, b, c, 1.0)
        expected = gamma(c) * gamma(c - a - b) / (gamma(c - a) * gamma(c - b))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_bessel_relation(self):
        """Test relation between 0F1 and Bessel function."""
        from scipy.special import iv

        # I_0(2) = 0F1(1; 1)
        result = hyp0f1(1.0, 1.0)
        expected = iv(0, 2.0)
        assert result == pytest.approx(expected, rel=1e-10)
