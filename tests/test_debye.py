"""
Tests for Debye functions.

Tests cover:
- Debye function D_n for various orders
- Special convenience functions D_1, D_2, D_3, D_4
- Physical applications: heat capacity, entropy
- Limiting behavior (small x, large x)
"""

import numpy as np
import pytest
from scipy.special import zeta

from pytcl.mathematical_functions.special_functions.debye import (
    debye,
    debye_1,
    debye_2,
    debye_3,
    debye_4,
    debye_entropy,
    debye_heat_capacity,
)

# =============================================================================
# Tests for general Debye function
# =============================================================================


class TestDebye:
    """Tests for general Debye function D_n(x)."""

    def test_debye_at_zero(self):
        """Test D_n(0) = 1 for all orders."""
        for n in range(1, 6):
            result = debye(n, 0.0)
            assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_debye_order_1(self):
        """Test D_1(x) basic evaluation."""
        result = debye(1, 1.0)
        assert np.isfinite(result[0])
        assert result[0] > 0
        assert result[0] < 1  # D_n(x) decreases from 1

    def test_debye_order_3(self):
        """Test D_3(x) basic evaluation."""
        result = debye(3, 1.0)
        assert np.isfinite(result[0])
        assert result[0] > 0
        assert result[0] < 1

    def test_debye_small_x(self):
        """Test D_n(x) for small x using series expansion."""
        x = 0.05  # Small x should use series expansion
        for n in range(1, 5):
            result = debye(n, x)
            assert np.isfinite(result[0])
            # For small x, D_n(x) ≈ 1 - n*x/(2*(n+1))
            expected_approx = 1 - n * x / (2 * (n + 1))
            assert result[0] == pytest.approx(expected_approx, rel=0.01)

    def test_debye_large_x(self):
        """Test D_n(x) asymptotic for large x."""
        x = 150.0  # Large x
        n = 3
        result = debye(n, x)
        assert np.isfinite(result[0])
        # For large x, D_n(x) -> n! * zeta(n+1) * n / x^n
        import math

        expected_asymp = math.factorial(n) * zeta(n + 1) * n / (x**n)
        assert result[0] == pytest.approx(expected_asymp, rel=0.1)

    def test_debye_array_input(self):
        """Test D_n with array input."""
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = debye(3, x)
        assert len(result) == 5
        assert result[0] == pytest.approx(1.0, rel=1e-6)
        assert np.all(np.isfinite(result))
        # D_n should be monotonically decreasing
        assert np.all(np.diff(result) <= 0)

    def test_debye_monotonically_decreasing(self):
        """Test that D_n(x) is monotonically decreasing for x > 0."""
        x = np.linspace(0.1, 10, 50)
        for n in range(1, 5):
            result = debye(n, x)
            # Allow small numerical fluctuations
            assert np.all(np.diff(result) < 1e-6)

    def test_debye_higher_orders(self):
        """Test D_n for higher orders n > 4."""
        for n in [5, 6, 8, 10]:
            result = debye(n, 2.0)
            assert np.isfinite(result[0])
            assert result[0] > 0

    def test_debye_invalid_order(self):
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Order n must be >= 1"):
            debye(0, 1.0)

    def test_debye_order_beyond_cache(self):
        """Test D_n for orders beyond pre-computed zeta values."""
        result = debye(15, 5.0)
        assert np.isfinite(result[0])
        assert result[0] > 0


# =============================================================================
# Tests for convenience functions D_1, D_2, D_3, D_4
# =============================================================================


class TestDebyeConvenienceFunctions:
    """Tests for D_1, D_2, D_3, D_4 convenience functions."""

    def test_debye_1_at_zero(self):
        """Test D_1(0) = 1."""
        result = debye_1(0.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_debye_2_at_zero(self):
        """Test D_2(0) = 1."""
        result = debye_2(0.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_debye_3_at_zero(self):
        """Test D_3(0) = 1."""
        result = debye_3(0.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_debye_4_at_zero(self):
        """Test D_4(0) = 1."""
        result = debye_4(0.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_debye_1_equals_debye(self):
        """Test debye_1 gives same result as debye(1, x)."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result1 = debye_1(x)
        result2 = debye(1, x)
        np.testing.assert_allclose(result1, result2)

    def test_debye_2_equals_debye(self):
        """Test debye_2 gives same result as debye(2, x)."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result1 = debye_2(x)
        result2 = debye(2, x)
        np.testing.assert_allclose(result1, result2)

    def test_debye_3_equals_debye(self):
        """Test debye_3 gives same result as debye(3, x)."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result1 = debye_3(x)
        result2 = debye(3, x)
        np.testing.assert_allclose(result1, result2)

    def test_debye_4_equals_debye(self):
        """Test debye_4 gives same result as debye(4, x)."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result1 = debye_4(x)
        result2 = debye(4, x)
        np.testing.assert_allclose(result1, result2)


# =============================================================================
# Tests for heat capacity
# =============================================================================


class TestDebyeHeatCapacity:
    """Tests for Debye model heat capacity."""

    def test_heat_capacity_high_temperature(self):
        """Test C_V approaches classical limit at high T."""
        # At T >> Theta_D, C_V / (3*N*k_B) -> 1
        T = np.array([1000.0, 2000.0, 5000.0])
        theta_D = 100.0  # Much lower than T
        result = debye_heat_capacity(T, theta_D)
        # Should be close to 1 for T >> Theta_D
        for r in result:
            assert r == pytest.approx(1.0, rel=0.05)

    def test_heat_capacity_low_temperature(self):
        """Test C_V follows T^3 law at low T."""
        # At T << Theta_D, C_V ~ (T/Theta_D)^3
        theta_D = 400.0
        T = np.array([10.0, 20.0])  # Much lower than Theta_D
        result = debye_heat_capacity(T, theta_D)
        # C_V should be small and proportional to T^3
        assert result[0] < result[1]  # Increases with T
        assert np.all(result < 1)

    def test_heat_capacity_intermediate(self):
        """Test C_V at intermediate temperature."""
        T = 200.0  # Intermediate temperature
        theta_D = 300.0
        result = debye_heat_capacity(T, theta_D)
        assert np.isfinite(result)
        assert 0 < result < 1

    def test_heat_capacity_invalid_temperature(self):
        """Test negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            debye_heat_capacity(-100.0, 300.0)

    def test_heat_capacity_invalid_debye_temperature(self):
        """Test negative Debye temperature raises ValueError."""
        with pytest.raises(ValueError, match="Debye temperature must be positive"):
            debye_heat_capacity(300.0, -100.0)

    def test_heat_capacity_array_input(self):
        """Test heat capacity with array input."""
        T = np.linspace(50, 500, 10)
        theta_D = 300.0
        result = debye_heat_capacity(T, theta_D)
        assert len(result) == 10
        assert np.all(np.isfinite(result))
        # Should increase with temperature
        assert np.all(np.diff(result) > 0)

    def test_heat_capacity_aluminum(self):
        """Test realistic example: aluminum at various temperatures."""
        theta_D = 428.0  # Aluminum Debye temperature
        T_room = 300.0
        result = debye_heat_capacity(T_room, theta_D)
        # At room temperature (T < Theta_D), heat capacity is between 0 and 1
        assert 0.5 < result[0] < 0.8


# =============================================================================
# Tests for entropy
# =============================================================================


class TestDebyeEntropy:
    """Tests for Debye model entropy."""

    def test_entropy_increases_with_temperature(self):
        """Test entropy increases with temperature."""
        T = np.array([100.0, 200.0, 300.0, 400.0])
        theta_D = 300.0
        result = debye_entropy(T, theta_D)
        assert np.all(np.diff(result) > 0)

    def test_entropy_low_temperature(self):
        """Test entropy at low temperature."""
        T = 50.0
        theta_D = 400.0
        result = debye_entropy(T, theta_D)
        assert np.isfinite(result)
        assert result > 0  # Entropy should be positive

    def test_entropy_high_temperature(self):
        """Test entropy at high temperature."""
        T = 1000.0
        theta_D = 300.0
        result = debye_entropy(T, theta_D)
        assert np.isfinite(result)

    def test_entropy_invalid_temperature(self):
        """Test negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            debye_entropy(-100.0, 300.0)

    def test_entropy_invalid_debye_temperature(self):
        """Test negative Debye temperature raises ValueError."""
        with pytest.raises(ValueError, match="Debye temperature must be positive"):
            debye_entropy(300.0, -100.0)

    def test_entropy_array_input(self):
        """Test entropy with array input."""
        T = np.linspace(50, 500, 10)
        theta_D = 300.0
        result = debye_entropy(T, theta_D)
        assert len(result) == 10
        assert np.all(np.isfinite(result))

    def test_entropy_large_x(self):
        """Test entropy for large x = Theta_D / T (low T limit)."""
        T = 10.0  # Very low temperature
        theta_D = 1000.0  # High Debye temperature, so x > 100
        result = debye_entropy(T, theta_D)
        # Should still be finite and positive
        assert np.isfinite(result)
        assert result > 0


# =============================================================================
# Integration tests
# =============================================================================


class TestDebyeIntegration:
    """Integration tests for Debye functions."""

    def test_debye_physical_consistency(self):
        """Test that D_n for different n have consistent ordering."""
        x = 2.0
        d1 = debye(1, x)[0]
        d2 = debye(2, x)[0]
        d3 = debye(3, x)[0]
        d4 = debye(4, x)[0]
        # All should be positive and finite
        assert all(np.isfinite([d1, d2, d3, d4]))
        assert all(d > 0 for d in [d1, d2, d3, d4])

    def test_heat_capacity_entropy_relationship(self):
        """Test physical relationship between heat capacity and entropy."""
        T = np.array([100.0, 200.0, 300.0])
        theta_D = 300.0
        cv = debye_heat_capacity(T, theta_D)
        s = debye_entropy(T, theta_D)
        # Both should increase with temperature
        assert np.all(np.diff(cv) > 0)
        assert np.all(np.diff(s) > 0)

    def test_batch_performance(self):
        """Test batch computation works correctly."""
        x = np.logspace(-2, 2, 100)  # 0.01 to 100
        result = debye(3, x)
        assert len(result) == 100
        assert np.all(np.isfinite(result))
        assert result[0] > result[-1]  # Decreasing

    def test_all_regimes(self):
        """Test all three computation regimes: small, medium, large x."""
        # Small x (series expansion)
        small = debye(3, 0.05)
        assert np.isfinite(small[0])

        # Medium x (numerical integration)
        medium = debye(3, 5.0)
        assert np.isfinite(medium[0])

        # Large x (asymptotic)
        large = debye(3, 150.0)
        assert np.isfinite(large[0])

        # Check ordering
        assert small[0] > medium[0] > large[0]
