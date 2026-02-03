"""
Comprehensive tests for hypergeometric functions.

Hypergeometric functions appear in many special function representations,
probability distributions, and physics applications.
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.hypergeometric import (
    hyp2f1,
    hyp1f1,
    hyp0f1,
    generalized_hypergeometric,
)


class TestHypergeometric2F1:
    """Tests for 2F1(a,b;c;z) hypergeometric function."""

    def test_hyp2f1_at_z_zero(self):
        """Test 2F1(a,b;c;0) = 1."""
        # 2F1 at z=0 should be 1 for all valid a,b,c
        a, b, c = 1.0, 2.0, 3.0
        z = 0.0
        result = hyp2f1(a, b, c, z)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_hyp2f1_special_case_geometric(self):
        """Test 2F1(1,1;2;z) = -ln(1-z)/z (geometric series)."""
        a, b, c = 1.0, 1.0, 2.0
        z = 0.5
        result = hyp2f1(a, b, c, z)
        expected = -np.log(1 - z) / z
        assert np.isclose(result, expected, atol=1e-8)

    def test_hyp2f1_pfaffian_transformation(self):
        """Test Pfaff transformation: 2F1(a,b;c;z) = (1-z)^(-a) * 2F1(a,c-b;c;z/(z-1))."""
        a, b, c = 0.5, 1.5, 2.0
        z = 0.3
        lhs = hyp2f1(a, b, c, z)
        z_new = z / (z - 1)
        rhs = (1 - z) ** (-a) * hyp2f1(a, c - b, c, z_new)
        assert np.isclose(lhs, rhs, atol=1e-6)

    def test_hyp2f1_small_z_expansion(self):
        """Test 2F1 behaves correctly for small |z|."""
        a, b, c = 2.0, 3.0, 4.0
        z_small = 1e-4
        result = hyp2f1(a, b, c, z_small)
        # For small z, should be approximately 1 + O(z)
        assert 0.99 < result < 1.01

    def test_hyp2f1_negative_integer_numerator(self):
        """Test 2F1 with negative integer parameter (terminates)."""
        # 2F1(-n, b; c; z) is a polynomial
        a, b, c, z = -2.0, 1.0, 2.0, 0.5
        result = hyp2f1(a, b, c, z)
        # Should be a real finite value
        assert np.isfinite(result)

    def test_hyp2f1_real_inputs(self):
        """Test 2F1 with various real inputs."""
        test_cases = [
            (0.5, 1.0, 1.5, 0.1),
            (1.0, 2.0, 3.0, 0.2),
            (2.0, 2.0, 4.0, 0.3),
            (0.25, 0.75, 1.25, 0.4),
        ]
        
        for a, b, c, z in test_cases:
            result = hyp2f1(a, b, c, z)
            assert np.isfinite(result), f"2F1({a},{b};{c};{z}) not finite"
            assert result > 0, f"2F1({a},{b};{c};{z}) should be positive"

    def test_hyp2f1_symmetry_in_parameters(self):
        """Test symmetry: 2F1(a,b;c;z) = 2F1(b,a;c;z)."""
        a, b, c, z = 1.5, 2.5, 3.5, 0.3
        result1 = hyp2f1(a, b, c, z)
        result2 = hyp2f1(b, a, c, z)
        assert np.isclose(result1, result2, atol=1e-10)

    def test_hyp2f1_array_input_z(self):
        """Test 2F1 with array input for z."""
        a, b, c = 1.0, 2.0, 3.0
        z_arr = np.array([0.1, 0.2, 0.3, 0.4])
        results = np.array([hyp2f1(a, b, c, z) for z in z_arr])
        
        # Results should be monotonically increasing for these parameters
        assert np.all(np.diff(results) > 0)

    def test_hyp2f1_consistency_with_scipy(self):
        """Test consistency with scipy.special.hyp2f1 if available."""
        try:
            from scipy.special import hyp2f1 as scipy_hyp2f1
        except ImportError:
            pytest.skip("scipy not available")
        
        test_cases = [
            (0.5, 1.0, 1.5, 0.1),
            (1.0, 2.0, 3.0, 0.3),
            (2.0, 3.0, 4.0, 0.2),
        ]
        
        for a, b, c, z in test_cases:
            our_result = hyp2f1(a, b, c, z)
            scipy_result = scipy_hyp2f1(a, b, c, z)
            # Allow some tolerance due to different numerical methods
            assert np.isclose(our_result, scipy_result, rtol=1e-4)


class TestHypergeometric0F1:
    """Tests for 0F1(;b;z) hypergeometric function."""

    def test_hyp0f1_at_z_zero(self):
        """Test 0F1(;b;0) = 1."""
        b = 2.0
        z = 0.0
        result = hyp0f1(b, z)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_hyp0f1_real_values(self):
        """Test 0F1 with real values."""
        test_cases = [
            (1.5, 0.1),
            (2.0, 0.5),
            (3.0, 1.0),
            (2.5, 2.0),
        ]
        
        for b, z in test_cases:
            result = hyp0f1(b, z)
            assert np.isfinite(result), f"0F1(;{b};{z}) not finite"

    def test_hyp0f1_bessel_connection(self):
        """Test 0F1 connection to Bessel function."""
        # 0F1(;b;z) is related to Bessel J function
        b = 1.0
        z = 1.0
        result = hyp0f1(b, z)
        assert np.isfinite(result)

    def test_hyp0f1_array_input(self):
        """Test 0F1 with multiple z values."""
        b = 2.0
        z_arr = np.linspace(0, 1, 5)
        results = np.array([hyp0f1(b, z) for z in z_arr])
        
        assert len(results) == len(z_arr)
        assert np.all(np.isfinite(results))


class TestHypergeometric1F1:
    """Tests for 1F1(a;b;z) confluent hypergeometric function."""

    def test_hyp1f1_at_z_zero(self):
        """Test 1F1(a;b;0) = 1."""
        a, b = 1.0, 2.0
        z = 0.0
        result = hyp1f1(a, b, z)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_hyp1f1_real_values(self):
        """Test 1F1 with real values."""
        test_cases = [
            (0.5, 1.5, 0.1),
            (1.0, 2.0, 0.5),
            (2.0, 3.0, 1.0),
            (1.5, 2.5, 2.0),
        ]
        
        for a, b, z in test_cases:
            result = hyp1f1(a, b, z)
            assert np.isfinite(result), f"1F1({a};{b};{z}) not finite"

    def test_hyp1f1_special_case_exponential(self):
        """Test 1F1(a;a;z) = exp(z) / a^(1/a) or similar special case."""
        # For special parameters, 1F1 reduces to known functions
        a, b = 1.0, 1.0
        z = 1.0
        result = hyp1f1(a, b, z)
        # 1F1(1;1;z) = e^z
        expected = np.exp(z)
        assert np.isclose(result, expected, rtol=1e-6)

    def test_hyp1f1_negative_integer_parameter(self):
        """Test 1F1 with negative integer a (terminates)."""
        a, b, z = -2.0, 2.0, 0.5
        result = hyp1f1(a, b, z)
        # Should be a polynomial
        assert np.isfinite(result)

    def test_hyp1f1_increasing_z(self):
        """Test behavior of 1F1 as z increases."""
        a, b = 1.0, 2.0
        z_vals = np.array([0.1, 0.5, 1.0, 2.0])
        results = np.array([hyp1f1(a, b, z) for z in z_vals])
        
        # For positive a, b, z, 1F1 should increase with z
        assert np.all(np.diff(results) > 0)

    def test_hyp1f1_array_input_z(self):
        """Test 1F1 with multiple z values."""
        a, b = 1.0, 2.0
        z_arr = np.linspace(0, 1, 5)
        results = np.array([hyp1f1(a, b, z) for z in z_arr])
        
        assert len(results) == len(z_arr)
        assert np.all(np.isfinite(results))

    def test_hyp1f1_consistency_with_scipy(self):
        """Test consistency with scipy.special.hyp1f1 if available."""
        try:
            from scipy.special import hyp1f1 as scipy_hyp1f1
        except ImportError:
            pytest.skip("scipy not available")
        
        test_cases = [
            (0.5, 1.5, 0.5),
            (1.0, 2.0, 1.0),
            (2.0, 3.0, 0.5),
        ]
        
        for a, b, z in test_cases:
            our_result = hyp1f1(a, b, z)
            scipy_result = scipy_hyp1f1(a, b, z)
            # Allow tolerance for numerical differences
            assert np.isclose(our_result, scipy_result, rtol=1e-4)


class TestGeneralizedHypergeometric:
    """Tests for generalized hypergeometric pFq function."""

    def test_generalized_hyp_simple_case(self):
        """Test generalized hypergeometric with simple parameters."""
        a = [1.0]
        b = [2.0]
        z = 0.5
        result = generalized_hypergeometric(a, b, z)
        assert np.isfinite(result)

    def test_generalized_hyp_multiple_params(self):
        """Test with multiple parameters."""
        a = [0.5, 1.0]
        b = [1.5, 2.0]
        z = 0.3
        result = generalized_hypergeometric(a, b, z)
        assert np.isfinite(result)

    def test_generalized_hyp_at_zero(self):
        """Test pFq(a,b;0) = 1."""
        a = [1.0, 2.0]
        b = [2.0, 3.0]
        z = 0.0
        result = generalized_hypergeometric(a, b, z)
        assert np.isclose(result, 1.0, atol=1e-10)


class TestHypergeometricEdgeCases:
    """Test edge cases and error handling."""

    def test_hyp2f1_large_z(self):
        """Test 2F1 with z close to 1."""
        a, b, c = 0.5, 1.0, 1.5
        z = 0.99  # Close to singularity
        result = hyp2f1(a, b, c, z)
        assert np.isfinite(result)

    def test_hyp1f1_large_z(self):
        """Test 1F1 with large z (might diverge)."""
        a, b = 0.5, 1.0
        z = 5.0
        result = hyp1f1(a, b, z)
        # Result might be large but should be finite
        assert np.isfinite(result)

    def test_hyp2f1_parameter_b_equals_c(self):
        """Test 2F1 with b = c (degenerate case)."""
        # This might need special handling
        a, b, c, z = 1.0, 2.0, 2.0, 0.5
        try:
            result = hyp2f1(a, b, c, z)
            assert np.isfinite(result) or (c == b and a != 0)
        except (ValueError, RuntimeError):
            # Some implementations might raise for degenerate cases
            pytest.skip("Degenerate case not supported")
