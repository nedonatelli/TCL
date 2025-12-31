"""Tests for EGM (Earth Gravitational Model) support.

Tests Clenshaw summation, coefficient loading, geoid height computation,
and gravity disturbance calculations.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.gravity import (  # Clenshaw summation; EGM functions; Scaling factors; Existing functions for comparison
    EGMCoefficients,
    GravityDisturbance,
    associated_legendre,
    associated_legendre_scaled,
    clenshaw_gravity,
    clenshaw_potential,
    clenshaw_sum_order,
    clenshaw_sum_order_derivative,
    create_test_coefficients,
    deflection_of_vertical,
    geoid_height,
    geoid_heights,
    get_data_dir,
    gravity_anomaly,
    gravity_disturbance,
    legendre_scaling_factors,
    spherical_harmonic_sum,
)


class TestClenshawSumOrder:
    """Tests for Clenshaw summation for fixed order m."""

    def test_returns_two_values(self):
        """clenshaw_sum_order returns (s_c, s_s) tuple."""
        coef = create_test_coefficients(n_max=10)
        cos_theta = 0.5
        sin_theta = np.sqrt(1 - cos_theta**2)

        result = clenshaw_sum_order(0, cos_theta, sin_theta, coef.C, coef.S, 10)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_m0_no_nan(self):
        """Order m=0 produces no NaN values."""
        coef = create_test_coefficients(n_max=20)
        cos_theta = 0.3
        sin_theta = np.sqrt(1 - cos_theta**2)

        s_c, s_s = clenshaw_sum_order(0, cos_theta, sin_theta, coef.C, coef.S, 20)

        assert not np.isnan(s_c)
        assert not np.isnan(s_s)

    def test_higher_order(self):
        """Higher orders m > 0 compute without error."""
        coef = create_test_coefficients(n_max=15)
        cos_theta = 0.6
        sin_theta = np.sqrt(1 - cos_theta**2)

        for m in range(1, 10):
            s_c, s_s = clenshaw_sum_order(m, cos_theta, sin_theta, coef.C, coef.S, 15)
            assert not np.isnan(s_c)
            assert not np.isnan(s_s)

    def test_pole_cos_theta_1(self):
        """Computation at cos(theta)=1 (north pole) succeeds."""
        coef = create_test_coefficients(n_max=10)
        cos_theta = 1.0
        sin_theta = 0.0

        # Should not raise or produce NaN
        s_c, s_s = clenshaw_sum_order(0, cos_theta, sin_theta, coef.C, coef.S, 10)
        assert not np.isnan(s_c)

    def test_pole_cos_theta_minus1(self):
        """Computation at cos(theta)=-1 (south pole) succeeds."""
        coef = create_test_coefficients(n_max=10)
        cos_theta = -1.0
        sin_theta = 0.0

        s_c, s_s = clenshaw_sum_order(0, cos_theta, sin_theta, coef.C, coef.S, 10)
        assert not np.isnan(s_c)


class TestClenshawSumOrderDerivative:
    """Tests for Clenshaw summation with derivatives."""

    def test_returns_four_values(self):
        """clenshaw_sum_order_derivative returns 4-tuple."""
        coef = create_test_coefficients(n_max=10)
        cos_theta = 0.5
        sin_theta = np.sqrt(1 - cos_theta**2)

        result = clenshaw_sum_order_derivative(
            0, cos_theta, sin_theta, coef.C, coef.S, 10
        )

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_no_nan_values(self):
        """Derivatives produce no NaN values."""
        coef = create_test_coefficients(n_max=15)
        cos_theta = 0.4
        sin_theta = np.sqrt(1 - cos_theta**2)

        for m in range(5):
            s_c, s_s, ds_c, ds_s = clenshaw_sum_order_derivative(
                m, cos_theta, sin_theta, coef.C, coef.S, 15
            )
            assert not np.isnan(s_c)
            assert not np.isnan(s_s)
            assert not np.isnan(ds_c)
            assert not np.isnan(ds_s)


class TestClenshawPotential:
    """Tests for gravitational potential using Clenshaw summation."""

    def test_returns_float(self):
        """clenshaw_potential returns a float."""
        coef = create_test_coefficients(n_max=10)
        lat = np.radians(45)
        lon = np.radians(-75)
        r = 6378137.0  # WGS84 semi-major axis

        V = clenshaw_potential(lat, lon, r, coef.C, coef.S, coef.R, coef.GM, 10)

        assert isinstance(V, float)
        assert not np.isnan(V)

    def test_potential_decreases_with_radius(self):
        """Gravitational potential decreases with distance."""
        coef = create_test_coefficients(n_max=10)
        lat = np.radians(30)
        lon = 0

        V_close = clenshaw_potential(
            lat, lon, coef.R, coef.C, coef.S, coef.R, coef.GM, 10
        )
        V_far = clenshaw_potential(
            lat, lon, coef.R * 1.1, coef.C, coef.S, coef.R, coef.GM, 10
        )

        # Potential magnitude should decrease with distance
        assert abs(V_far) < abs(V_close)

    def test_potential_approximately_symmetric(self):
        """Potential is approximately symmetric about equator."""
        coef = create_test_coefficients(n_max=10)
        lon = 0
        r = coef.R

        V_north = clenshaw_potential(
            np.radians(45), lon, r, coef.C, coef.S, coef.R, coef.GM, 10
        )
        V_south = clenshaw_potential(
            np.radians(-45), lon, r, coef.C, coef.S, coef.R, coef.GM, 10
        )

        # For Earth's gravity field (with small asymmetries), should be close
        # The potential is dominated by the central term, so relative difference is small
        assert_allclose(V_north, V_south, rtol=1e-4)


class TestClenshawGravity:
    """Tests for gravity disturbance using Clenshaw summation."""

    def test_returns_three_values(self):
        """clenshaw_gravity returns (g_r, g_lat, g_lon) tuple."""
        coef = create_test_coefficients(n_max=10)
        lat = np.radians(45)
        lon = np.radians(-75)
        r = coef.R

        result = clenshaw_gravity(lat, lon, r, coef.C, coef.S, coef.R, coef.GM, 10)

        assert isinstance(result, tuple)
        assert len(result) == 3
        for g in result:
            assert isinstance(g, float)
            assert not np.isnan(g)

    def test_full_gravity_magnitude_reasonable(self):
        """Full gravity (including central term) is in reasonable range."""
        coef = create_test_coefficients(n_max=10)
        lat = np.radians(45)
        lon = np.radians(0)
        r = coef.R

        g_r, g_lat, g_lon = clenshaw_gravity(
            lat, lon, r, coef.C, coef.S, coef.R, coef.GM, 10
        )

        # Full gravity includes the central term GM/r^2 which is about 9.8 m/s^2
        # The sign depends on convention - we compute -dV/dr
        magnitude = np.sqrt(g_r**2 + g_lat**2 + g_lon**2)
        assert magnitude > 9.0  # At least 9 m/s^2


class TestLegendreScalingFactors:
    """Tests for Legendre polynomial scaling factors."""

    def test_low_degree_unity(self):
        """Scaling factors are 1.0 for low degrees."""
        scale = legendre_scaling_factors(50)

        # For n_max <= 150, all factors should be 1.0
        assert_allclose(scale, np.ones(51))

    def test_high_degree_scaling(self):
        """Scaling factors are < 1 for high degrees."""
        scale = legendre_scaling_factors(500)

        # First factor should still be 1
        assert scale[0] == 1.0

        # Higher degrees should have smaller scale factors
        assert scale[500] < scale[250]
        assert scale[250] < scale[100]

    def test_shape(self):
        """Scaling factors have correct shape."""
        for n_max in [10, 100, 500]:
            scale = legendre_scaling_factors(n_max)
            assert scale.shape == (n_max + 1,)


class TestAssociatedLegendreScaled:
    """Tests for scaled associated Legendre polynomials."""

    def test_returns_two_arrays(self):
        """associated_legendre_scaled returns (P, exponents)."""
        P, exponents = associated_legendre_scaled(10, 10, 0.5)

        assert isinstance(P, np.ndarray)
        assert isinstance(exponents, np.ndarray)
        assert P.shape == (11, 11)
        # exponents is 1D, one per degree
        assert exponents.shape == (11,)

    def test_matches_unscaled_low_degree(self):
        """Scaled results match unscaled for low degrees."""
        x = 0.6
        P_scaled, exp = associated_legendre_scaled(10, 10, x)
        P_unscaled = associated_legendre(10, 10, x, normalized=True)

        # For low degrees (n < 150), exponents should be 0 and P_scaled should match
        for n in range(11):
            for m in range(n + 1):
                # exp is indexed by degree only (1D array)
                P_reconstructed = P_scaled[n, m] * (10.0 ** exp[n])
                assert_allclose(
                    P_reconstructed, P_unscaled[n, m], rtol=1e-8, atol=1e-12
                )

    def test_no_overflow_high_degree(self):
        """High degree computation doesn't overflow."""
        # This would overflow without scaling
        P, exp = associated_legendre_scaled(300, 300, 0.5)

        # Should not have any NaN or Inf
        assert not np.any(np.isnan(P))
        assert not np.any(np.isinf(P))


class TestCreateTestCoefficients:
    """Tests for test coefficient generation."""

    def test_returns_egm_coefficients(self):
        """create_test_coefficients returns EGMCoefficients."""
        coef = create_test_coefficients(n_max=10)

        assert isinstance(coef, EGMCoefficients)
        assert coef.n_max == 10
        assert coef.C.shape == (11, 11)
        assert coef.S.shape == (11, 11)

    def test_c00_is_one(self):
        """C[0,0] = 1 for normalized coefficients."""
        coef = create_test_coefficients(n_max=10)
        assert coef.C[0, 0] == 1.0

    def test_s_diagonal_zero(self):
        """S[n,0] = 0 for all n (sine of zonal harmonics)."""
        coef = create_test_coefficients(n_max=10)
        for n in range(11):
            assert coef.S[n, 0] == 0.0

    def test_j2_present(self):
        """J2 (C[2,0]) is non-zero and negative."""
        coef = create_test_coefficients(n_max=10)
        # J2 ≈ -1.08263e-3 (unnormalized), normalized is different
        assert coef.C[2, 0] != 0
        assert coef.C[2, 0] < 0


class TestGeoidHeight:
    """Tests for geoid height computation."""

    def test_returns_float(self):
        """geoid_height returns a float."""
        # Use test coefficients
        coef = create_test_coefficients(n_max=10)
        N = geoid_height(np.radians(45), np.radians(0), coefficients=coef, n_max=10)
        assert isinstance(N, float)
        assert not np.isnan(N)

    def test_geoid_not_nan_or_inf(self):
        """Geoid height computation produces finite values."""
        coef = create_test_coefficients(n_max=36)

        # Test several locations
        for lat in [0, 45, -45, 90]:
            for lon in [0, 90, -90, 180]:
                N = geoid_height(
                    np.radians(lat), np.radians(lon), coefficients=coef, n_max=36
                )
                assert not np.isnan(N)
                assert not np.isinf(N)


class TestGeoidHeights:
    """Tests for batch geoid height computation."""

    def test_returns_array(self):
        """geoid_heights returns numpy array."""
        coef = create_test_coefficients(n_max=10)
        lats = np.radians([0, 30, 45, 60, 90])
        lons = np.radians([0, 0, 0, 0, 0])

        N = geoid_heights(lats, lons, coefficients=coef, n_max=10)

        assert isinstance(N, np.ndarray)
        assert N.shape == (5,)

    def test_matches_single_point(self):
        """Batch results match single-point computation."""
        coef = create_test_coefficients(n_max=10)
        lat = np.radians(45)
        lon = np.radians(-75)

        N_single = geoid_height(lat, lon, coefficients=coef, n_max=10)
        N_batch = geoid_heights(
            np.array([lat]), np.array([lon]), coefficients=coef, n_max=10
        )

        assert_allclose(N_batch[0], N_single)


class TestGravityDisturbance:
    """Tests for gravity disturbance computation."""

    def test_returns_gravity_disturbance(self):
        """gravity_disturbance returns GravityDisturbance."""
        coef = create_test_coefficients(n_max=10)

        result = gravity_disturbance(
            np.radians(45), np.radians(-75), h=0.0, coefficients=coef, n_max=10
        )

        assert isinstance(result, GravityDisturbance)
        assert hasattr(result, "delta_g_r")
        assert hasattr(result, "delta_g_lat")
        assert hasattr(result, "delta_g_lon")
        assert hasattr(result, "magnitude")

    def test_magnitude_formula(self):
        """Magnitude equals sqrt(sum of squares)."""
        coef = create_test_coefficients(n_max=10)

        result = gravity_disturbance(
            np.radians(30), np.radians(60), h=0.0, coefficients=coef, n_max=10
        )

        expected_mag = np.sqrt(
            result.delta_g_r**2 + result.delta_g_lat**2 + result.delta_g_lon**2
        )
        assert_allclose(result.magnitude, expected_mag)

    def test_disturbance_decreases_with_altitude(self):
        """Gravity disturbance decreases with altitude."""
        coef = create_test_coefficients(n_max=20)
        lat = np.radians(45)
        lon = np.radians(0)

        result_0 = gravity_disturbance(lat, lon, h=0.0, coefficients=coef, n_max=20)
        result_10km = gravity_disturbance(
            lat, lon, h=10000.0, coefficients=coef, n_max=20
        )

        # Disturbance magnitude should decrease with altitude
        assert result_10km.magnitude < result_0.magnitude


class TestGravityAnomaly:
    """Tests for gravity anomaly computation."""

    def test_returns_float(self):
        """gravity_anomaly returns a float."""
        coef = create_test_coefficients(n_max=10)

        anom = gravity_anomaly(
            np.radians(45), np.radians(-75), h=0.0, coefficients=coef, n_max=10
        )

        assert isinstance(anom, float)
        assert not np.isnan(anom)

    def test_anomaly_returns_finite(self):
        """Gravity anomaly computation produces finite values."""
        coef = create_test_coefficients(n_max=36)

        anom = gravity_anomaly(
            np.radians(45), np.radians(0), h=0.0, coefficients=coef, n_max=36
        )

        # Note: With test coefficients (including full J2), values are large
        # because J2 represents the equatorial bulge which is part of the
        # reference ellipsoid. Real EGM files subtract the reference ellipsoid.
        assert not np.isnan(anom)
        assert not np.isinf(anom)


class TestDeflectionOfVertical:
    """Tests for deflection of the vertical."""

    def test_returns_two_values(self):
        """deflection_of_vertical returns (xi, eta)."""
        coef = create_test_coefficients(n_max=10)

        xi, eta = deflection_of_vertical(
            np.radians(45), np.radians(-75), coefficients=coef, n_max=10
        )

        assert isinstance(xi, float)
        assert isinstance(eta, float)
        assert not np.isnan(xi)
        assert not np.isnan(eta)

    def test_deflection_returns_finite(self):
        """Deflection values are finite."""
        coef = create_test_coefficients(n_max=36)

        xi, eta = deflection_of_vertical(
            np.radians(45), np.radians(0), coefficients=coef, n_max=36
        )

        # Note: Test coefficients include full J2, so deflections are larger
        # than real-world values which are typically < 60 arcseconds.
        # Real EGM files subtract the reference ellipsoid contribution.
        assert not np.isnan(xi)
        assert not np.isnan(eta)
        assert not np.isinf(xi)
        assert not np.isinf(eta)


class TestGetDataDir:
    """Tests for data directory management."""

    def test_returns_path(self):
        """get_data_dir returns a Path object."""
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)

    def test_path_in_home(self):
        """Data directory is under user's home."""
        data_dir = get_data_dir()
        home = Path.home()
        # Should be somewhere under home
        assert str(home) in str(data_dir)


class TestNumericalStability:
    """Tests for numerical stability at high degrees."""

    def test_no_overflow_degree_100(self):
        """Degree 100 computation doesn't overflow."""
        coef = create_test_coefficients(n_max=100)

        N = geoid_height(np.radians(45), np.radians(0), coefficients=coef, n_max=100)

        assert not np.isnan(N)
        assert not np.isinf(N)

    def test_no_overflow_degree_200(self):
        """Degree 200 computation doesn't overflow."""
        coef = create_test_coefficients(n_max=200)

        N = geoid_height(np.radians(45), np.radians(0), coefficients=coef, n_max=200)

        assert not np.isnan(N)
        assert not np.isinf(N)

    @pytest.mark.slow
    def test_no_overflow_degree_360(self):
        """Degree 360 (EGM96 full) computation doesn't overflow."""
        coef = create_test_coefficients(n_max=360)

        N = geoid_height(np.radians(45), np.radians(0), coefficients=coef, n_max=360)

        assert not np.isnan(N)
        assert not np.isinf(N)


class TestClenshawVsNaive:
    """Tests comparing Clenshaw summation to naive approach."""

    def test_low_degree_matches_naive(self):
        """Clenshaw matches naive summation for low degrees."""
        coef = create_test_coefficients(n_max=10)
        lat = np.radians(45)
        lon = np.radians(30)
        r = coef.R

        # Compute potential using Clenshaw
        V_clenshaw = clenshaw_potential(
            lat, lon, r, coef.C, coef.S, coef.R, coef.GM, 10
        )

        # Compute using naive spherical harmonic sum
        # spherical_harmonic_sum takes (lat, lon, r, C, S, R, GM, n_max)
        # and returns (V, dV_r, dV_lat)
        V_naive, _, _ = spherical_harmonic_sum(
            lat, lon, r, coef.C, coef.S, coef.R, coef.GM, 10
        )

        # Should match to reasonable precision
        # (Clenshaw and naive may have different numerical properties)
        assert_allclose(V_clenshaw, V_naive, rtol=1e-6)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_north_pole(self):
        """Computation at north pole succeeds."""
        coef = create_test_coefficients(n_max=20)

        N = geoid_height(np.radians(90), 0, coefficients=coef, n_max=20)

        assert not np.isnan(N)
        assert not np.isinf(N)

    def test_south_pole(self):
        """Computation at south pole succeeds."""
        coef = create_test_coefficients(n_max=20)

        N = geoid_height(np.radians(-90), 0, coefficients=coef, n_max=20)

        assert not np.isnan(N)
        assert not np.isinf(N)

    def test_equator_prime_meridian(self):
        """Computation at (0, 0) succeeds."""
        coef = create_test_coefficients(n_max=20)

        N = geoid_height(0, 0, coefficients=coef, n_max=20)

        assert not np.isnan(N)

    def test_date_line(self):
        """Computation at date line (lon=180) succeeds."""
        coef = create_test_coefficients(n_max=20)

        N = geoid_height(np.radians(45), np.pi, coefficients=coef, n_max=20)

        assert not np.isnan(N)

    def test_negative_longitude(self):
        """Negative longitude works correctly."""
        coef = create_test_coefficients(n_max=20)

        N_pos = geoid_height(
            np.radians(45), np.radians(90), coefficients=coef, n_max=20
        )
        N_neg = geoid_height(
            np.radians(45), np.radians(-270), coefficients=coef, n_max=20
        )

        # 90° and -270° are the same longitude
        assert_allclose(N_pos, N_neg, rtol=1e-10)
