"""
Tests for numerical integration (quadrature) methods.

Tests cover:
- Gauss-Legendre quadrature
- Gauss-Hermite quadrature
- Gauss-Laguerre quadrature
- Gauss-Chebyshev quadrature
- Adaptive quadrature (quad, dblquad, tplquad)
- Fixed quadrature
- Romberg integration
- Simpson's rule
- Trapezoidal rule
- Multi-dimensional cubature
- Unscented transform points
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.numerical_integration.quadrature import (
    cubature_gauss_hermite,
    dblquad,
    fixed_quad,
    gauss_chebyshev,
    gauss_hermite,
    gauss_laguerre,
    gauss_legendre,
    quad,
    romberg,
    simpson,
    spherical_cubature,
    tplquad,
    trapezoid,
    unscented_transform_points,
)

# =============================================================================
# Tests for Gauss-Legendre quadrature
# =============================================================================


class TestGaussLegendre:
    """Tests for Gauss-Legendre quadrature."""

    def test_points_in_range(self):
        """Test points are in [-1, 1]."""
        x, w = gauss_legendre(5)
        assert np.all(x >= -1)
        assert np.all(x <= 1)

    def test_weights_positive(self):
        """Test weights are positive."""
        x, w = gauss_legendre(5)
        assert np.all(w > 0)

    def test_weights_sum(self):
        """Test weights sum to 2 (length of interval)."""
        x, w = gauss_legendre(10)
        assert np.sum(w) == pytest.approx(2.0, rel=1e-10)

    def test_integrate_x_squared(self):
        """Test integrating x^2 from -1 to 1."""
        x, w = gauss_legendre(5)
        result = np.sum(w * x**2)
        expected = 2 / 3  # ∫_{-1}^{1} x^2 dx = 2/3
        assert result == pytest.approx(expected, rel=1e-10)

    def test_integrate_polynomial(self):
        """Test exact integration of polynomial."""
        # n-point Gauss rule is exact for polynomials up to degree 2n-1
        x, w = gauss_legendre(4)
        # Integrate x^6 (degree 6, needs at least n=4)
        result = np.sum(w * x**6)
        expected = 2 / 7  # ∫_{-1}^{1} x^6 dx = 2/7
        assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Tests for Gauss-Hermite quadrature
# =============================================================================


class TestGaussHermite:
    """Tests for Gauss-Hermite quadrature."""

    def test_symmetric_points(self):
        """Test points are symmetric around 0."""
        x, w = gauss_hermite(5)
        for xi in x:
            assert -xi in x or xi == pytest.approx(0, abs=1e-10)

    def test_weights_positive(self):
        """Test weights are positive."""
        x, w = gauss_hermite(5)
        assert np.all(w > 0)

    def test_gaussian_expectation(self):
        """Test computing E[X^2] for X ~ N(0, 1)."""
        x, w = gauss_hermite(5)
        # E[X^2] = ∫ x^2 * exp(-x^2) dx / √π = 1/2
        # With substitution x -> x*√2 for standard normal:
        result = np.sum(w * (np.sqrt(2) * x) ** 2) / np.sqrt(np.pi)
        assert result == pytest.approx(1.0, rel=1e-6)


# =============================================================================
# Tests for Gauss-Laguerre quadrature
# =============================================================================


class TestGaussLaguerre:
    """Tests for Gauss-Laguerre quadrature."""

    def test_points_positive(self):
        """Test points are positive."""
        x, w = gauss_laguerre(5)
        assert np.all(x >= 0)

    def test_weights_positive(self):
        """Test weights are positive."""
        x, w = gauss_laguerre(5)
        assert np.all(w > 0)

    def test_integrate_x_exp(self):
        """Test ∫ x * exp(-x) dx from 0 to inf = 1."""
        x, w = gauss_laguerre(5)
        result = np.sum(w * x)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_weights_sum(self):
        """Test weights sum to Γ(1) = 1."""
        x, w = gauss_laguerre(10)
        assert np.sum(w) == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# Tests for Gauss-Chebyshev quadrature
# =============================================================================


class TestGaussChebyshev:
    """Tests for Gauss-Chebyshev quadrature."""

    def test_kind_1_shape(self):
        """Test type 1 Chebyshev returns correct shape."""
        x, w = gauss_chebyshev(5, kind=1)
        assert len(x) == 5
        assert len(w) == 5

    def test_kind_2_shape(self):
        """Test type 2 Chebyshev returns correct shape."""
        x, w = gauss_chebyshev(5, kind=2)
        assert len(x) == 5
        assert len(w) == 5

    def test_kind_1_points_in_range(self):
        """Test type 1 points are in (-1, 1)."""
        x, w = gauss_chebyshev(5, kind=1)
        assert np.all(x > -1)
        assert np.all(x < 1)

    def test_kind_2_points_in_range(self):
        """Test type 2 points are in (-1, 1)."""
        x, w = gauss_chebyshev(5, kind=2)
        assert np.all(x > -1)
        assert np.all(x < 1)

    def test_invalid_kind_raises(self):
        """Test invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be 1 or 2"):
            gauss_chebyshev(5, kind=3)


# =============================================================================
# Tests for adaptive quadrature
# =============================================================================


class TestQuad:
    """Tests for adaptive quadrature."""

    def test_integrate_polynomial(self):
        """Test integrating x^2 from 0 to 1."""
        result, error = quad(lambda x: x**2, 0, 1)
        assert result == pytest.approx(1 / 3, rel=1e-6)
        assert error < 1e-10

    def test_integrate_sin(self):
        """Test integrating sin(x) from 0 to pi."""
        result, error = quad(lambda x: np.sin(x), 0, np.pi)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_integrate_exp(self):
        """Test integrating exp(-x) from 0 to 1."""
        result, error = quad(lambda x: np.exp(-x), 0, 1)
        expected = 1 - np.exp(-1)
        assert result == pytest.approx(expected, rel=1e-6)


class TestDblQuad:
    """Tests for double integration."""

    def test_integrate_unit_square(self):
        """Test integrating x*y over unit square."""
        result, error = dblquad(lambda y, x: x * y, 0, 1, lambda x: 0, lambda x: 1)
        assert result == pytest.approx(0.25, rel=1e-6)

    def test_integrate_triangle(self):
        """Test integrating over triangle."""
        # ∫_0^1 ∫_0^x 1 dy dx = ∫_0^1 x dx = 0.5
        result, error = dblquad(lambda y, x: 1, 0, 1, lambda x: 0, lambda x: x)
        assert result == pytest.approx(0.5, rel=1e-6)


class TestTplQuad:
    """Tests for triple integration."""

    def test_integrate_unit_cube(self):
        """Test integrating x*y*z over unit cube."""
        result, error = tplquad(
            lambda z, y, x: x * y * z,
            0,
            1,
            lambda x: 0,
            lambda x: 1,
            lambda x, y: 0,
            lambda x, y: 1,
        )
        assert result == pytest.approx(0.125, rel=1e-6)


# =============================================================================
# Tests for fixed quadrature
# =============================================================================


class TestFixedQuad:
    """Tests for fixed-order quadrature."""

    def test_integrate_polynomial(self):
        """Test integrating x^2 from 0 to 1."""
        result, _ = fixed_quad(lambda x: x**2, 0, 1, n=5)
        assert result == pytest.approx(1 / 3, rel=1e-6)

    def test_returns_none_error(self):
        """Test returns None for error estimate."""
        result, error = fixed_quad(lambda x: x**2, 0, 1)
        assert error is None


# =============================================================================
# Tests for Romberg integration
# =============================================================================


class TestRomberg:
    """Tests for Romberg integration."""

    def test_integrate_polynomial(self):
        """Test integrating x^2 from 0 to 1."""
        result = romberg(lambda x: x**2, 0, 1)
        assert result == pytest.approx(1 / 3, rel=1e-6)

    def test_integrate_sin(self):
        """Test integrating sin(x) from 0 to pi."""
        result = romberg(lambda x: np.sin(x), 0, np.pi)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_with_tolerance(self):
        """Test with specified tolerance."""
        result = romberg(lambda x: x**4, 0, 1, tol=1e-10)
        assert result == pytest.approx(0.2, rel=1e-8)


# =============================================================================
# Tests for Simpson's rule
# =============================================================================


class TestSimpson:
    """Tests for Simpson's rule integration."""

    def test_integrate_sin(self):
        """Test integrating sin(x) from 0 to pi."""
        x = np.linspace(0, np.pi, 101)
        y = np.sin(x)
        result = simpson(y, x)
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_with_uniform_spacing(self):
        """Test with uniform spacing."""
        x = np.linspace(0, 1, 101)
        y = x**2
        result = simpson(y, dx=0.01)
        assert result == pytest.approx(1 / 3, rel=0.01)


# =============================================================================
# Tests for trapezoidal rule
# =============================================================================


class TestTrapezoid:
    """Tests for trapezoidal rule integration."""

    def test_integrate_x_squared(self):
        """Test integrating x^2 from 0 to 1."""
        x = np.linspace(0, 1, 101)
        y = x**2
        result = trapezoid(y, x)
        assert result == pytest.approx(1 / 3, rel=0.01)

    def test_with_uniform_spacing(self):
        """Test with uniform spacing."""
        y = np.array([0, 1, 4, 9, 16])  # x^2 at x = 0, 1, 2, 3, 4
        result = trapezoid(y, dx=1)
        # Trapezoidal approximation
        assert result == pytest.approx(22.0, rel=1e-10)


# =============================================================================
# Tests for cubature methods
# =============================================================================


class TestCubatureGaussHermite:
    """Tests for tensor product Gauss-Hermite cubature."""

    def test_shape_2d(self):
        """Test shape for 2D cubature."""
        points, weights = cubature_gauss_hermite(2, 3)
        assert points.shape == (9, 2)  # 3^2 = 9 points
        assert weights.shape == (9,)

    def test_shape_3d(self):
        """Test shape for 3D cubature."""
        points, weights = cubature_gauss_hermite(3, 2)
        assert points.shape == (8, 3)  # 2^3 = 8 points
        assert weights.shape == (8,)

    def test_weights_positive(self):
        """Test weights are positive."""
        points, weights = cubature_gauss_hermite(2, 3)
        assert np.all(weights > 0)


class TestSphericalCubature:
    """Tests for spherical cubature rule."""

    def test_shape(self):
        """Test correct shape: 2n points in n dimensions."""
        points, weights = spherical_cubature(3)
        assert points.shape == (6, 3)  # 2*3 = 6 points
        assert weights.shape == (6,)

    def test_weights_sum_to_one(self):
        """Test weights sum to 1."""
        points, weights = spherical_cubature(4)
        assert np.sum(weights) == pytest.approx(1.0, rel=1e-10)

    def test_equal_weights(self):
        """Test weights are equal."""
        points, weights = spherical_cubature(5)
        assert np.allclose(weights, 1 / 10)  # 1/(2*5)

    def test_point_magnitude(self):
        """Test point magnitudes are sqrt(n)."""
        n = 4
        points, weights = spherical_cubature(n)
        magnitudes = np.linalg.norm(points, axis=1)
        assert np.allclose(magnitudes, np.sqrt(n))


# =============================================================================
# Tests for unscented transform
# =============================================================================


class TestUnscentedTransformPoints:
    """Tests for unscented transform sigma points."""

    def test_shape(self):
        """Test correct shape: 2n+1 points in n dimensions."""
        sigma_points, wm, wc = unscented_transform_points(3)
        assert sigma_points.shape == (7, 3)  # 2*3+1 = 7 points
        assert wm.shape == (7,)
        assert wc.shape == (7,)

    def test_center_point_at_origin(self):
        """Test center point is at origin."""
        sigma_points, wm, wc = unscented_transform_points(4)
        np.testing.assert_allclose(sigma_points[0], np.zeros(4))

    def test_mean_weights_sum(self):
        """Test mean weights sum to 1."""
        sigma_points, wm, wc = unscented_transform_points(3)
        assert np.sum(wm) == pytest.approx(1.0, rel=1e-10)

    def test_symmetric_points(self):
        """Test points are symmetric around center."""
        n = 3
        sigma_points, wm, wc = unscented_transform_points(n)
        # Points 1 to n should be opposite of n+1 to 2n
        for i in range(n):
            np.testing.assert_allclose(sigma_points[1 + i], -sigma_points[1 + n + i])

    def test_custom_parameters(self):
        """Test with custom alpha, beta, kappa."""
        sigma_points, wm, wc = unscented_transform_points(
            3, alpha=0.5, beta=3.0, kappa=1.0
        )
        assert sigma_points.shape == (7, 3)
        assert np.sum(wm) == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# Integration tests
# =============================================================================


class TestQuadratureIntegration:
    """Integration tests for quadrature methods."""

    def test_gauss_legendre_exactness(self):
        """Test Gauss-Legendre exactness for polynomials."""
        for n in range(2, 8):
            x, w = gauss_legendre(n)
            # Should be exact for polynomials up to degree 2n-1
            for degree in range(2 * n):
                result = np.sum(w * x**degree)
                # ∫_{-1}^{1} x^k dx = 2/(k+1) if k even, 0 if k odd
                if degree % 2 == 0:
                    expected = 2 / (degree + 1)
                else:
                    expected = 0
                assert result == pytest.approx(expected, abs=1e-10)

    def test_quad_vs_fixed_quad(self):
        """Test quad and fixed_quad give similar results."""

        def gaussian(x):
            return np.exp(-(x**2))

        result_quad, _ = quad(gaussian, 0, 2)
        result_fixed, _ = fixed_quad(gaussian, 0, 2, n=10)
        assert result_quad == pytest.approx(result_fixed, rel=0.01)

    def test_simpson_vs_trapezoid_accuracy(self):
        """Test Simpson's rule is more accurate than trapezoidal."""
        x = np.linspace(0, 1, 11)
        y = np.exp(x)

        result_simp = simpson(y, x)
        result_trap = trapezoid(y, x)
        exact = np.exp(1) - 1

        error_simp = abs(result_simp - exact)
        error_trap = abs(result_trap - exact)

        # Simpson should be more accurate
        assert error_simp < error_trap
