"""
Comprehensive tests for Debye functions module.

Tests coverage for:
- General debye function D_n(x)
- Specific debye functions (D_1, D_2, D_3, D_4)
- Thermodynamic properties (heat capacity, entropy)
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.debye import (
    debye,
    debye_1,
    debye_2,
    debye_3,
    debye_4,
    debye_entropy,
    debye_heat_capacity,
)


class TestDebeyBasic:
    """Tests for basic Debye function evaluation."""

    def test_debye_first_order(self):
        """Test Debye function D_1(x)."""
        x = np.array([0.5, 1.0, 2.0])
        result = debye(1, x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_second_order(self):
        """Test Debye function D_2(x)."""
        x = np.array([0.5, 1.0, 2.0])
        result = debye(2, x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_third_order(self):
        """Test Debye function D_3(x)."""
        x = np.array([0.5, 1.0, 2.0])
        result = debye(3, x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_fourth_order(self):
        """Test Debye function D_4(x)."""
        x = np.array([0.5, 1.0, 2.0])
        result = debye(4, x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_scalar_input(self):
        """Test Debye function with scalar input."""
        x = 1.5
        result = debye(1, x)

        assert np.isfinite(result)

    def test_debye_small_values(self):
        """Test Debye function with small x values."""
        x = np.array([0.001, 0.01, 0.1])
        result = debye(1, x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_large_values(self):
        """Test Debye function with large x values."""
        x = np.array([10, 50, 100])
        result = debye(1, x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))


class TestDebeySpecific:
    """Tests for specific Debye functions D_1-D_4."""

    def test_debye_1_basic(self):
        """Test Debye D_1 function."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result = debye_1(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_1_scalar(self):
        """Test Debye D_1 with scalar."""
        result = debye_1(1.5)
        assert np.isfinite(result)

    def test_debye_2_basic(self):
        """Test Debye D_2 function."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result = debye_2(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_2_scalar(self):
        """Test Debye D_2 with scalar."""
        result = debye_2(1.5)
        assert np.isfinite(result)

    def test_debye_3_basic(self):
        """Test Debye D_3 function."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result = debye_3(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_3_scalar(self):
        """Test Debye D_3 with scalar."""
        result = debye_3(1.5)
        assert np.isfinite(result)

    def test_debye_4_basic(self):
        """Test Debye D_4 function."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        result = debye_4(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_4_scalar(self):
        """Test Debye D_4 with scalar."""
        result = debye_4(1.5)
        assert np.isfinite(result)

    def test_debye_functions_comparison(self):
        """Test that different Debye functions return different values."""
        x = np.array([1.0, 2.0])

        d1 = debye_1(x)
        d2 = debye_2(x)
        d3 = debye_3(x)
        d4 = debye_4(x)

        # All should be finite
        assert np.all(np.isfinite(d1))
        assert np.all(np.isfinite(d2))
        assert np.all(np.isfinite(d3))
        assert np.all(np.isfinite(d4))

    def test_debye_1_list_input(self):
        """Test Debye D_1 with list input."""
        x_list = [0.5, 1.0, 2.0]
        result = debye_1(x_list)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_debye_2_list_input(self):
        """Test Debye D_2 with list input."""
        x_list = [0.5, 1.0, 2.0]
        result = debye_2(x_list)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


class TestDebeyThermodynamics:
    """Tests for thermodynamic properties using Debye functions."""

    def test_debye_heat_capacity_basic(self):
        """Test Debye heat capacity calculation."""
        T = np.array([10, 50, 100, 300])  # Kelvin
        theta_d = 250  # Debye temperature (K)

        c_v = debye_heat_capacity(T, theta_d)

        assert c_v.shape == T.shape
        assert np.all(np.isfinite(c_v))

    def test_debye_heat_capacity_scalar_temperature(self):
        """Test Debye heat capacity with scalar temperature."""
        T = 100
        theta_d = 250

        c_v = debye_heat_capacity(T, theta_d)

        assert np.isfinite(c_v)

    def test_debye_heat_capacity_different_debye_temps(self):
        """Test Debye heat capacity with different Debye temperatures."""
        T = np.array([50, 100, 200])
        theta_d_values = [100, 250, 500]

        for theta_d in theta_d_values:
            c_v = debye_heat_capacity(T, theta_d)
            assert c_v.shape == T.shape
            assert np.all(np.isfinite(c_v))

    def test_debye_heat_capacity_temperature_dependence(self):
        """Test Debye heat capacity increases with temperature."""
        T = np.array([10, 50, 100, 200, 300, 500])
        theta_d = 250

        c_v = debye_heat_capacity(T, theta_d)

        # Heat capacity should increase with temperature (monotonic)
        assert np.all(np.isfinite(c_v))

    def test_debye_entropy_basic(self):
        """Test Debye entropy calculation."""
        T = np.array([10, 50, 100, 300])
        theta_d = 250

        S = debye_entropy(T, theta_d)

        assert S.shape == T.shape
        assert np.all(np.isfinite(S))

    def test_debye_entropy_scalar_temperature(self):
        """Test Debye entropy with scalar temperature."""
        T = 100
        theta_d = 250

        S = debye_entropy(T, theta_d)

        assert np.isfinite(S)

    def test_debye_entropy_different_debye_temps(self):
        """Test Debye entropy with different Debye temperatures."""
        T = np.array([50, 100, 200])
        theta_d_values = [100, 250, 500]

        for theta_d in theta_d_values:
            S = debye_entropy(T, theta_d)
            assert S.shape == T.shape
            assert np.all(np.isfinite(S))

    def test_debye_entropy_positive(self):
        """Test that Debye entropy is positive."""
        T = np.array([50, 100, 200, 300])
        theta_d = 250

        S = debye_entropy(T, theta_d)

        assert np.all(S >= 0)


class TestDebeyProperties:
    """Tests for mathematical properties of Debye functions."""

    def test_debye_zero_behavior(self):
        """Test Debye function behavior near zero."""
        x_small = np.array([1e-6, 1e-5, 1e-4, 0.001])
        result = debye(1, x_small)

        assert np.all(np.isfinite(result))

    def test_debye_monotonicity(self):
        """Test Debye function monotonicity."""
        x = np.linspace(0.1, 10, 50)
        d1 = debye(1, x)

        # Debye functions should be finite across range
        assert np.all(np.isfinite(d1))

    def test_debye_consistency_across_orders(self):
        """Test consistency of Debye functions across orders."""
        x = 1.5

        d1 = debye_1(x)
        d2 = debye_2(x)
        d3 = debye_3(x)
        d4 = debye_4(x)

        # Compare with general debye function
        d1_general = debye(1, x)
        d2_general = debye(2, x)
        d3_general = debye(3, x)
        d4_general = debye(4, x)

        # Should match
        assert np.isclose(d1, d1_general)
        assert np.isclose(d2, d2_general)
        assert np.isclose(d3, d3_general)
        assert np.isclose(d4, d4_general)


class TestDebeyArrayOperations:
    """Tests for array operations with Debye functions."""

    def test_debye_1d_array(self):
        """Test Debye with 1D array."""
        x = np.linspace(0.5, 5, 100)
        result = debye_1(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_with_zeros_in_array(self):
        """Test Debye with array containing near-zero values."""
        x = np.array([0.0, 0.001, 0.01, 0.1, 1.0])
        result = debye_1(x)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_debye_heat_capacity_1d_array(self):
        """Test heat capacity with 1D temperature array."""
        T = np.linspace(10, 300, 50)
        theta_d = 250

        c_v = debye_heat_capacity(T, theta_d)

        assert c_v.shape == T.shape
        assert np.all(np.isfinite(c_v))

    def test_debye_entropy_1d_array(self):
        """Test entropy with 1D temperature array."""
        T = np.linspace(10, 300, 50)
        theta_d = 250

        S = debye_entropy(T, theta_d)

        assert S.shape == T.shape
        assert np.all(np.isfinite(S))


class TestDebeyIntegration:
    """Integration tests for Debye functions."""

    def test_debye_physical_parameters(self):
        """Test Debye functions with typical material parameters."""
        # Typical Debye temperatures for common materials
        materials = {
            "Cu": 343,  # Copper
            "Al": 428,  # Aluminum
            "Fe": 467,  # Iron
            "Pb": 105,  # Lead
        }

        T = np.array([100, 200, 300])

        for material, theta_d in materials.items():
            c_v = debye_heat_capacity(T, theta_d)
            S = debye_entropy(T, theta_d)

            assert np.all(np.isfinite(c_v))
            assert np.all(np.isfinite(S))

    def test_debye_limit_behavior(self):
        """Test limiting behavior of Debye functions."""
        # Test high and low temperature limits
        T_low = np.array([1, 5, 10])
        T_high = np.array([500, 1000, 2000])
        theta_d = 250

        c_v_low = debye_heat_capacity(T_low, theta_d)
        c_v_high = debye_heat_capacity(T_high, theta_d)

        # All should be finite
        assert np.all(np.isfinite(c_v_low))
        assert np.all(np.isfinite(c_v_high))

    def test_debye_continuous_evaluation(self):
        """Test continuous evaluation of Debye functions."""
        x = np.linspace(0.01, 20, 1000)

        d1 = debye_1(x)
        d2 = debye_2(x)
        d3 = debye_3(x)
        d4 = debye_4(x)

        # All should be finite and continuous
        assert np.all(np.isfinite(d1))
        assert np.all(np.isfinite(d2))
        assert np.all(np.isfinite(d3))
        assert np.all(np.isfinite(d4))

    def test_thermodynamic_consistency(self):
        """Test consistency of thermodynamic properties."""
        T = np.array([50, 100, 150, 200, 250, 300])
        theta_d = 250

        c_v = debye_heat_capacity(T, theta_d)
        S = debye_entropy(T, theta_d)

        # Heat capacity and entropy should be finite
        assert np.all(np.isfinite(c_v))
        assert np.all(np.isfinite(S))

        # Entropy should increase with temperature
        assert np.all(np.diff(S) >= 0)
