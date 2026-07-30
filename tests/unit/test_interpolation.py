"""
Tests for interpolation functions.

Tests cover:
- 1D interpolation (linear, cubic spline, PCHIP, Akima, barycentric)
- 2D interpolation
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.interpolation.interpolation import (
    akima,
    barycentric,
    cubic_spline,
    interp1d,
    interp2d,
    linear_interp,
    pchip,
)


class TestInterpolation:
    """Tests for interpolation functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for interpolation tests."""
        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        y = np.array([0, 1, 4, 9, 16], dtype=np.float64)
        return x, y

    def test_linear_interp(self, sample_data):
        """Test linear interpolation."""
        xp, yp = sample_data
        result = linear_interp(1.5, xp, yp)
        np.testing.assert_allclose(result, 2.5, rtol=1e-10)

    def test_interp1d_linear(self, sample_data):
        """Test 1D interpolation with linear method."""
        x, y = sample_data
        f = interp1d(x, y, kind="linear")
        result = f(1.5)
        np.testing.assert_allclose(result, 2.5, rtol=1e-10)

    def test_cubic_spline(self, sample_data):
        """Test cubic spline interpolation."""
        x, y = sample_data
        cs = cubic_spline(x, y)
        result = cs(2.5)
        assert 5 < result < 8

    def test_pchip(self, sample_data):
        """Test PCHIP interpolation."""
        x, y = sample_data
        p = pchip(x, y)
        result = p(2.5)
        assert 5 < result < 8

    def test_akima(self, sample_data):
        """Test Akima interpolation."""
        x, y = sample_data
        a = akima(x, y)
        result = a(2.5)
        assert 5 < result < 8

    def test_barycentric(self, sample_data):
        """Test barycentric interpolation."""
        x, y = sample_data
        b = barycentric(x, y)
        result = b(2.0)
        np.testing.assert_allclose(result, 4.0, rtol=1e-6)


class TestInterp2D:
    """Tests for 2D interpolation."""

    @pytest.fixture
    def grid_data(self):
        """Sample grid data."""
        x = np.array([0, 1, 2], dtype=np.float64)
        y = np.array([0, 1, 2], dtype=np.float64)
        z = np.array([[0, 1, 4], [1, 2, 5], [4, 5, 8]], dtype=np.float64)
        return x, y, z

    def test_interp2d_linear(self, grid_data):
        """Test 2D linear interpolation."""
        x, y, z = grid_data
        f = interp2d(x, y, z, kind="linear")
        result = f([[0.5, 0.5]])
        assert isinstance(result, np.ndarray)
