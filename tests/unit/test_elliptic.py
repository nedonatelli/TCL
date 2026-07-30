"""
Tests for elliptic integrals and functions.

Tests cover:
- Complete elliptic integrals (ellipk, ellipe)
- Incomplete elliptic integrals (ellipkinc, ellipeinc)
- Carlson symmetric integrals (elliprf, elliprd, elliprg, elliprj, elliprc)
- Edge cases and mathematical identities
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.elliptic import (
    ellipe,
    ellipeinc,
    ellipk,
    ellipkinc,
    ellipkm1,
    elliprc,
    elliprd,
    elliprf,
    elliprg,
    elliprj,
)

# =============================================================================
# Tests for complete elliptic integral of the first kind
# =============================================================================


class TestEllipK:
    """Tests for complete elliptic integral of the first kind."""

    def test_ellipk_zero(self):
        """Test K(0) = π/2."""
        result = ellipk(0)
        assert result == pytest.approx(np.pi / 2, rel=1e-10)

    def test_ellipk_half(self):
        """Test K(0.5) has known value."""
        result = ellipk(0.5)
        assert result == pytest.approx(1.8540746773013719, rel=1e-10)

    def test_ellipk_array(self):
        """Test ellipk with array input."""
        m = np.array([0, 0.25, 0.5, 0.75])
        result = ellipk(m)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(np.pi / 2, rel=1e-10)

    def test_ellipk_approaches_infinity(self):
        """Test K(m) increases as m approaches 1."""
        m_values = np.array([0.9, 0.99, 0.999])
        results = ellipk(m_values)
        # Should be strictly increasing
        assert np.all(np.diff(results) > 0)
        # K(0.999) should be large
        assert results[-1] > 4.0

    def test_ellipk_negative_m(self):
        """Test ellipk with negative m (valid for m < 0)."""
        result = ellipk(-0.5)
        assert np.isfinite(result)


class TestEllipKm1:
    """Tests for ellipk near m = 1."""

    def test_ellipkm1_small_p(self):
        """Test ellipkm1 for small p = 1-m."""
        result = ellipkm1(0.1)  # K(0.9)
        expected = ellipk(0.9)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_ellipkm1_one(self):
        """Test ellipkm1(1) = K(0) = π/2."""
        result = ellipkm1(1.0)
        assert result == pytest.approx(np.pi / 2, rel=1e-10)


# =============================================================================
# Tests for complete elliptic integral of the second kind
# =============================================================================


class TestEllipE:
    """Tests for complete elliptic integral of the second kind."""

    def test_ellipe_zero(self):
        """Test E(0) = π/2."""
        result = ellipe(0)
        assert result == pytest.approx(np.pi / 2, rel=1e-10)

    def test_ellipe_one(self):
        """Test E(1) = 1."""
        result = ellipe(1)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_ellipe_half(self):
        """Test E(0.5) has known value."""
        result = ellipe(0.5)
        assert result == pytest.approx(1.3506438810476757, rel=1e-10)

    def test_ellipe_array(self):
        """Test ellipe with array input."""
        m = np.array([0, 0.5, 1.0])
        result = ellipe(m)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(np.pi / 2, rel=1e-10)
        assert result[2] == pytest.approx(1.0, rel=1e-10)

    def test_ellipe_decreasing(self):
        """Test E(m) is decreasing in m."""
        m = np.linspace(0, 0.99, 50)
        result = ellipe(m)
        assert np.all(np.diff(result) < 0)


# =============================================================================
# Tests for incomplete elliptic integrals
# =============================================================================


class TestEllipEinc:
    """Tests for incomplete elliptic integral of the second kind."""

    def test_ellipeinc_complete(self):
        """Test E(π/2, m) = ellipe(m)."""
        m = 0.5
        result = ellipeinc(np.pi / 2, m)
        expected = ellipe(m)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ellipeinc_zero_phi(self):
        """Test E(0, m) = 0."""
        result = ellipeinc(0, 0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_ellipeinc_zero_m(self):
        """Test E(φ, 0) = φ."""
        phi = 0.7
        result = ellipeinc(phi, 0)
        assert result == pytest.approx(phi, rel=1e-10)

    def test_ellipeinc_array(self):
        """Test ellipeinc with array inputs."""
        phi = np.array([0, np.pi / 4, np.pi / 2])
        result = ellipeinc(phi, 0.5)
        assert result.shape == (3,)


class TestEllipKinc:
    """Tests for incomplete elliptic integral of the first kind."""

    def test_ellipkinc_complete(self):
        """Test F(π/2, m) = ellipk(m)."""
        m = 0.5
        result = ellipkinc(np.pi / 2, m)
        expected = ellipk(m)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ellipkinc_zero_phi(self):
        """Test F(0, m) = 0."""
        result = ellipkinc(0, 0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_ellipkinc_zero_m(self):
        """Test F(φ, 0) = φ."""
        phi = 0.7
        result = ellipkinc(phi, 0)
        assert result == pytest.approx(phi, rel=1e-10)

    def test_ellipkinc_array(self):
        """Test ellipkinc with array inputs."""
        phi = np.linspace(0, np.pi / 2, 10)
        result = ellipkinc(phi, 0.5)
        assert result.shape == (10,)
        # Should be increasing in phi
        assert np.all(np.diff(result) >= 0)


# =============================================================================
# Tests for Carlson symmetric integrals
# =============================================================================


class TestEllipRF:
    """Tests for Carlson R_F integral."""

    def test_elliprf_symmetric(self):
        """Test R_F is symmetric in its arguments."""
        result1 = elliprf(1, 2, 3)
        result2 = elliprf(2, 3, 1)
        result3 = elliprf(3, 1, 2)
        assert result1 == pytest.approx(result2, rel=1e-10)
        assert result1 == pytest.approx(result3, rel=1e-10)

    def test_elliprf_equal_args(self):
        """Test R_F(a, a, a) = 1/sqrt(a)."""
        a = 4.0
        result = elliprf(a, a, a)
        expected = 1.0 / np.sqrt(a)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_elliprf_homogeneity(self):
        """Test R_F(λx, λy, λz) = λ^(-1/2) * R_F(x, y, z)."""
        x, y, z = 1, 2, 3
        lam = 4.0
        result_scaled = elliprf(lam * x, lam * y, lam * z)
        result_unscaled = elliprf(x, y, z)
        assert result_scaled == pytest.approx(
            lam ** (-0.5) * result_unscaled, rel=1e-10
        )

    def test_elliprf_relation_to_ellipk(self):
        """Test K(m) = R_F(0, 1-m, 1)."""
        m = 0.5
        result = elliprf(0, 1 - m, 1)
        expected = ellipk(m)
        assert result == pytest.approx(expected, rel=1e-10)


class TestEllipRD:
    """Tests for Carlson R_D integral."""

    def test_elliprd_known_value(self):
        """Test R_D with known inputs."""
        result = elliprd(1, 2, 3)
        assert 0.2 < result < 0.4

    def test_elliprd_equal_xy(self):
        """Test R_D with x = y."""
        result = elliprd(1, 1, 2)
        assert np.isfinite(result)


class TestEllipRG:
    """Tests for Carlson R_G integral."""

    def test_elliprg_equal_args(self):
        """Test R_G(a, a, a) = sqrt(a)."""
        a = 4.0
        result = elliprg(a, a, a)
        expected = np.sqrt(a)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_elliprg_relation_to_ellipe(self):
        """Test E(m) = 2 * R_G(0, 1-m, 1)."""
        m = 0.5
        result = 2 * elliprg(0, 1 - m, 1)
        expected = ellipe(m)
        assert result == pytest.approx(expected, rel=1e-10)


class TestEllipRJ:
    """Tests for Carlson R_J integral."""

    def test_elliprj_known_value(self):
        """Test R_J with known inputs."""
        result = elliprj(1, 2, 3, 4)
        assert 0.1 < result < 0.3


class TestEllipRC:
    """Tests for Carlson R_C integral."""

    def test_elliprc_equal_args(self):
        """Test R_C(a, a) = 1/sqrt(a)."""
        a = 4.0
        result = elliprc(a, a)
        expected = 1.0 / np.sqrt(a)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_elliprc_x_greater_y(self):
        """Test R_C when x > y."""
        result = elliprc(4, 1)
        assert np.isfinite(result)

    def test_elliprc_x_less_y(self):
        """Test R_C when x < y."""
        result = elliprc(1, 4)
        assert np.isfinite(result)


# =============================================================================
# Integration tests
# =============================================================================


class TestEllipticIntegration:
    """Integration tests for elliptic functions."""

    def test_legendre_relation(self):
        """Test Legendre's relation: K(m)*E(1-m) + E(m)*K(1-m) - K(m)*K(1-m) = π/2."""
        m = 0.3
        K_m = ellipk(m)
        K_1m = ellipk(1 - m)
        E_m = ellipe(m)
        E_1m = ellipe(1 - m)
        result = K_m * E_1m + E_m * K_1m - K_m * K_1m
        assert result == pytest.approx(np.pi / 2, rel=1e-10)

    def test_incomplete_to_complete(self):
        """Test incomplete integrals reduce to complete at φ = π/2."""
        m_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for m in m_values:
            assert ellipkinc(np.pi / 2, m) == pytest.approx(ellipk(m), rel=1e-10)
            assert ellipeinc(np.pi / 2, m) == pytest.approx(ellipe(m), rel=1e-10)

    def test_ellipe_bounds(self):
        """Test E(m) is bounded by 1 and π/2."""
        m = np.linspace(0, 1, 100)
        result = ellipe(m)
        assert np.all(result >= 1.0 - 1e-10)
        assert np.all(result <= np.pi / 2 + 1e-10)

    def test_complementary_modulus(self):
        """Test K(m) relation with complementary modulus."""
        m = 0.25
        result1 = ellipkm1(1 - m)
        result2 = ellipk(m)
        assert result1 == pytest.approx(result2, rel=1e-10)
