"""
Tests for error functions and related special functions.

Tests cover:
- Error function (erf) and complementary (erfc)
- Scaled complementary error function (erfcx)
- Imaginary error function (erfi)
- Inverse error functions (erfinv, erfcinv)
- Dawson's integral
- Fresnel integrals
- Faddeeva function (wofz)
- Voigt profile
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.error_functions import (
    dawsn,
    erf,
    erfc,
    erfcinv,
    erfcx,
    erfi,
    erfinv,
    fresnel,
    voigt_profile,
    wofz,
)

# =============================================================================
# Tests for error function
# =============================================================================


class TestErf:
    """Tests for error function."""

    def test_erf_zero(self):
        """Test erf(0) = 0."""
        result = erf(0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_erf_one(self):
        """Test erf(1) has known value."""
        result = erf(1)
        assert result == pytest.approx(0.8427007929497149, rel=1e-10)

    def test_erf_symmetry(self):
        """Test erf(-x) = -erf(x)."""
        x = 1.5
        assert erf(-x) == pytest.approx(-erf(x), rel=1e-10)

    def test_erf_bounds(self):
        """Test erf is bounded by -1 and 1."""
        x = np.linspace(-10, 10, 100)
        result = erf(x)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_erf_approaches_one(self):
        """Test erf(x) approaches 1 for large x."""
        result = erf(5)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_erf_array(self):
        """Test erf with array input."""
        x = np.array([-1, 0, 1])
        result = erf(x)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.0, abs=1e-15)


# =============================================================================
# Tests for complementary error function
# =============================================================================


class TestErfc:
    """Tests for complementary error function."""

    def test_erfc_zero(self):
        """Test erfc(0) = 1."""
        result = erfc(0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_erfc_relation(self):
        """Test erfc(x) = 1 - erf(x)."""
        x_values = np.array([-1, 0, 0.5, 1, 2])
        for x in x_values:
            assert erfc(x) == pytest.approx(1 - erf(x), rel=1e-10)

    def test_erfc_large_x(self):
        """Test erfc(x) is small for large x."""
        result = erfc(3)
        assert result == pytest.approx(2.2090496998585438e-05, rel=1e-6)

    def test_erfc_bounds(self):
        """Test erfc is bounded by 0 and 2."""
        x = np.linspace(-5, 5, 100)
        result = erfc(x)
        assert np.all(result >= 0.0 - 1e-15)
        assert np.all(result <= 2.0 + 1e-15)


# =============================================================================
# Tests for scaled complementary error function
# =============================================================================


class TestErfcx:
    """Tests for scaled complementary error function."""

    def test_erfcx_zero(self):
        """Test erfcx(0) = 1."""
        result = erfcx(0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_erfcx_relation(self):
        """Test erfcx(x) = exp(x^2) * erfc(x) for moderate x."""
        x = 1.0
        result = erfcx(x)
        expected = np.exp(x**2) * erfc(x)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_erfcx_large_x(self):
        """Test erfcx remains finite for large x."""
        result = erfcx(10)
        assert np.isfinite(result)
        assert 0 < result < 0.1

    def test_erfcx_asymptotic(self):
        """Test erfcx(x) ~ 1/(sqrt(pi)*x) for large x."""
        x = 50.0
        result = erfcx(x)
        asymptotic = 1.0 / (np.sqrt(np.pi) * x)
        assert result == pytest.approx(asymptotic, rel=0.05)


# =============================================================================
# Tests for imaginary error function
# =============================================================================


class TestErfi:
    """Tests for imaginary error function."""

    def test_erfi_zero(self):
        """Test erfi(0) = 0."""
        result = erfi(0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_erfi_one(self):
        """Test erfi(1) has known value."""
        result = erfi(1)
        assert result == pytest.approx(1.6504257587975428, rel=1e-10)

    def test_erfi_symmetry(self):
        """Test erfi(-x) = -erfi(x)."""
        x = 1.5
        assert erfi(-x) == pytest.approx(-erfi(x), rel=1e-10)


# =============================================================================
# Tests for inverse error functions
# =============================================================================


class TestErfinv:
    """Tests for inverse error function."""

    def test_erfinv_zero(self):
        """Test erfinv(0) = 0."""
        result = erfinv(0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_erfinv_roundtrip(self):
        """Test erf(erfinv(y)) = y."""
        y_values = np.array([-0.5, 0.0, 0.3, 0.7])
        for y in y_values:
            x = erfinv(y)
            assert erf(x) == pytest.approx(y, rel=1e-10)

    def test_erfinv_inverse_roundtrip(self):
        """Test erfinv(erf(x)) = x."""
        x_values = np.array([-1, 0, 0.5, 1])
        for x in x_values:
            y = erf(x)
            assert erfinv(y) == pytest.approx(x, rel=1e-10)

    def test_erfinv_approaches_infinity(self):
        """Test erfinv approaches infinity as y approaches 1."""
        result = erfinv(0.999999)
        assert result > 3.0


class TestErfcinv:
    """Tests for inverse complementary error function."""

    def test_erfcinv_one(self):
        """Test erfcinv(1) = 0."""
        result = erfcinv(1)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_erfcinv_roundtrip(self):
        """Test erfc(erfcinv(y)) = y."""
        y_values = np.array([0.5, 1.0, 1.5])
        for y in y_values:
            x = erfcinv(y)
            assert erfc(x) == pytest.approx(y, rel=1e-10)


# =============================================================================
# Tests for Dawson's integral
# =============================================================================


class TestDawsn:
    """Tests for Dawson's integral."""

    def test_dawsn_zero(self):
        """Test F(0) = 0."""
        result = dawsn(0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_dawsn_one(self):
        """Test F(1) has known value."""
        result = dawsn(1)
        assert result == pytest.approx(0.5380795069127683, rel=1e-10)

    def test_dawsn_symmetry(self):
        """Test F(-x) = -F(x)."""
        x = 1.5
        assert dawsn(-x) == pytest.approx(-dawsn(x), rel=1e-10)

    def test_dawsn_maximum(self):
        """Test Dawson function has maximum near x = 0.924."""
        x = np.linspace(0, 2, 100)
        result = dawsn(x)
        max_idx = np.argmax(result)
        assert 0.8 < x[max_idx] < 1.0

    def test_dawsn_relation_to_erfi(self):
        """Test F(x) = sqrt(pi)/2 * exp(-x^2) * erfi(x)."""
        x = 0.5
        result = dawsn(x)
        expected = np.sqrt(np.pi) / 2 * np.exp(-(x**2)) * erfi(x)
        assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Tests for Fresnel integrals
# =============================================================================


class TestFresnel:
    """Tests for Fresnel integrals."""

    def test_fresnel_zero(self):
        """Test S(0) = C(0) = 0."""
        S, C = fresnel(0)
        assert S == pytest.approx(0.0, abs=1e-15)
        assert C == pytest.approx(0.0, abs=1e-15)

    def test_fresnel_one(self):
        """Test S(1) and C(1) have known values."""
        S, C = fresnel(1)
        assert S == pytest.approx(0.4382591473903548, rel=1e-10)
        assert C == pytest.approx(0.7798934003768228, rel=1e-10)

    def test_fresnel_symmetry(self):
        """Test S(-x) = -S(x) and C(-x) = -C(x)."""
        x = 1.5
        S_pos, C_pos = fresnel(x)
        S_neg, C_neg = fresnel(-x)
        assert S_neg == pytest.approx(-S_pos, rel=1e-10)
        assert C_neg == pytest.approx(-C_pos, rel=1e-10)

    def test_fresnel_asymptotic(self):
        """Test Fresnel integrals approach 0.5 for large x."""
        S, C = fresnel(50)
        assert S == pytest.approx(0.5, abs=0.01)
        assert C == pytest.approx(0.5, abs=0.01)

    def test_fresnel_array(self):
        """Test Fresnel integrals with array input."""
        x = np.array([0, 1, 2])
        S, C = fresnel(x)
        assert S.shape == (3,)
        assert C.shape == (3,)


# =============================================================================
# Tests for Faddeeva function
# =============================================================================


class TestWofz:
    """Tests for Faddeeva function."""

    def test_wofz_zero(self):
        """Test w(0) = 1."""
        result = wofz(0)
        assert result.real == pytest.approx(1.0, rel=1e-10)
        assert result.imag == pytest.approx(0.0, abs=1e-15)

    def test_wofz_real_positive(self):
        """Test w(x) for real positive x."""
        result = wofz(1.0)
        assert np.isfinite(result.real)
        assert np.isfinite(result.imag)

    def test_wofz_complex(self):
        """Test w(z) for complex z."""
        z = 1 + 1j
        result = wofz(z)
        assert np.isfinite(result.real)
        assert np.isfinite(result.imag)

    def test_wofz_purely_imaginary(self):
        """Test w(iy) for purely imaginary input."""
        y = 1.0
        result = wofz(1j * y)
        assert np.isfinite(result)


# =============================================================================
# Tests for Voigt profile
# =============================================================================


class TestVoigtProfile:
    """Tests for Voigt profile."""

    def test_voigt_pure_gaussian(self):
        """Test Voigt profile with gamma=0 is Gaussian."""
        sigma = 1.0
        x = 0.0
        result = voigt_profile(x, sigma, 0.0)
        expected = 1.0 / (sigma * np.sqrt(2 * np.pi))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_voigt_symmetric(self):
        """Test Voigt profile is symmetric."""
        sigma, gamma = 1.0, 0.5
        x = 1.5
        result_pos = voigt_profile(x, sigma, gamma)
        result_neg = voigt_profile(-x, sigma, gamma)
        assert result_pos == pytest.approx(result_neg, rel=1e-10)

    def test_voigt_maximum_at_zero(self):
        """Test Voigt profile has maximum at x=0."""
        sigma, gamma = 1.0, 0.5
        x = np.linspace(-5, 5, 101)
        result = voigt_profile(x, sigma, gamma)
        max_idx = np.argmax(result)
        assert x[max_idx] == pytest.approx(0.0, abs=0.1)

    def test_voigt_normalized(self):
        """Test Voigt profile integrates to approximately 1."""
        sigma, gamma = 1.0, 0.5
        x = np.linspace(-50, 50, 20000)
        dx = x[1] - x[0]
        result = voigt_profile(x, sigma, gamma)
        integral = np.sum(result) * dx
        assert integral == pytest.approx(1.0, rel=0.02)

    def test_voigt_array(self):
        """Test Voigt profile with array input."""
        x = np.array([-1, 0, 1])
        result = voigt_profile(x, 1.0, 0.5)
        assert result.shape == (3,)

    def test_voigt_wider_with_gamma(self):
        """Test larger gamma makes profile wider."""
        x = 2.0
        result_small_gamma = voigt_profile(x, 1.0, 0.1)
        result_large_gamma = voigt_profile(x, 1.0, 1.0)
        # At x=2, larger gamma should give larger value (wider tail)
        assert result_large_gamma > result_small_gamma


# =============================================================================
# Integration tests
# =============================================================================


class TestErrorFunctionsIntegration:
    """Integration tests for error functions."""

    def test_normal_cdf_relation(self):
        """Test Φ(x) = (1 + erf(x/√2)) / 2."""
        from scipy.stats import norm

        x = 1.5
        result = (1 + erf(x / np.sqrt(2))) / 2
        expected = norm.cdf(x)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_complementary_relation(self):
        """Test erf(x) + erfc(x) = 1."""
        x_values = np.linspace(-3, 3, 50)
        for x in x_values:
            assert erf(x) + erfc(x) == pytest.approx(1.0, rel=1e-10)

    def test_erfcx_stability(self):
        """Test erfcx provides stable results when erfc underflows."""
        x = 30.0
        # erfc(30) would underflow
        erfcx_result = erfcx(x)
        assert np.isfinite(erfcx_result)
        assert erfcx_result > 0

    def test_inverse_functions_consistency(self):
        """Test erfinv and erfcinv are consistent."""
        y = 0.3
        x1 = erfinv(y)
        x2 = erfcinv(1 - y)
        assert x1 == pytest.approx(x2, rel=1e-10)
