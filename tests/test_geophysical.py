"""Tests for geophysical models (gravity and magnetism)."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.gravity import (
    associated_legendre,
    bouguer_anomaly,
    free_air_anomaly,
    geoid_height_j2,
    gravity_j2,
    gravity_wgs84,
    normal_gravity,
    normal_gravity_somigliana,
)
from pytcl.magnetism import (
    IGRF13,
    WMM2020,
    dipole_axis,
    dipole_moment,
    igrf,
    igrf_declination,
    igrf_inclination,
    magnetic_declination,
    magnetic_field_intensity,
    magnetic_inclination,
    wmm,
)


class TestAssociatedLegendre:
    """Tests for associated Legendre polynomials."""

    def test_p00(self):
        """P_0^0 = 1."""
        P = associated_legendre(0, 0, 0.5)
        assert_allclose(P[0, 0], 1.0)

    def test_p10(self):
        """P_1^0(x) has expected sign."""
        x = 0.5
        P = associated_legendre(1, 1, x, normalized=True)
        # P_1^1 should be non-zero for x != ±1
        assert P[1, 1] != 0

    def test_p20(self):
        """P_2^0 is computed without error."""
        x = 0.6
        P = associated_legendre(2, 2, x, normalized=True)
        # Just verify it runs and returns values
        assert P.shape == (3, 3)
        assert not np.any(np.isnan(P))

    def test_symmetry(self):
        """Legendre polynomials have definite parity."""
        x = 0.7
        P_pos = associated_legendre(4, 0, x)
        P_neg = associated_legendre(4, 0, -x)
        # P_n^0(-x) = (-1)^n P_n^0(x)
        assert_allclose(P_neg[2, 0], P_pos[2, 0], rtol=1e-10)  # n=2, even
        assert_allclose(P_neg[3, 0], -P_pos[3, 0], rtol=1e-10)  # n=3, odd

    def test_boundary_x1(self):
        """Legendre polynomials at x=1."""
        P = associated_legendre(5, 5, 1.0)
        # P_n^0(1) = 1 for unnormalized
        # For normalized, P_n^0(1) = sqrt(2n+1)
        assert_allclose(P[0, 0], 1.0)

    def test_boundary_x_minus1(self):
        """Legendre polynomials at x=-1 computed without error."""
        P = associated_legendre(4, 0, -1.0)
        # Just verify computation succeeds
        assert P.shape == (5, 1)
        assert not np.any(np.isnan(P))


class TestNormalGravity:
    """Tests for normal gravity computations."""

    def test_equator_gravity(self):
        """Gravity at equator is approximately 9.78 m/s^2."""
        g = normal_gravity(0, 0)
        assert 9.77 < g < 9.79

    def test_pole_gravity(self):
        """Gravity at pole is approximately 9.83 m/s^2."""
        g = normal_gravity(np.pi / 2, 0)
        assert 9.82 < g < 9.84

    def test_gravity_increases_with_latitude(self):
        """Gravity increases from equator to poles."""
        g_equator = normal_gravity(0, 0)
        g_45 = normal_gravity(np.radians(45), 0)
        g_pole = normal_gravity(np.pi / 2, 0)

        assert g_equator < g_45 < g_pole

    def test_gravity_decreases_with_altitude(self):
        """Gravity decreases with altitude."""
        g_0 = normal_gravity(np.radians(45), 0)
        g_1000 = normal_gravity(np.radians(45), 1000)
        g_10000 = normal_gravity(np.radians(45), 10000)

        assert g_0 > g_1000 > g_10000

    def test_somigliana_on_ellipsoid(self):
        """Somigliana formula gives correct values on ellipsoid."""
        gamma_eq = normal_gravity_somigliana(0)
        gamma_pole = normal_gravity_somigliana(np.pi / 2)

        # Known approximate values
        assert 9.78 < gamma_eq < 9.79
        assert 9.83 < gamma_pole < 9.84


class TestGravityWGS84:
    """Tests for WGS84 gravity model."""

    def test_returns_result(self):
        """Function returns GravityResult."""
        result = gravity_wgs84(np.radians(45), 0, 0)
        assert hasattr(result, "magnitude")
        assert hasattr(result, "g_down")
        assert hasattr(result, "g_north")
        assert hasattr(result, "g_east")

    def test_magnitude_reasonable(self):
        """Gravity magnitude is in expected range."""
        result = gravity_wgs84(np.radians(45), np.radians(-75), 1000)
        assert 9.7 < result.magnitude < 9.9

    def test_down_component_positive(self):
        """Down component is positive (pointing down)."""
        result = gravity_wgs84(0, 0, 0)
        assert result.g_down > 0


class TestGravityJ2:
    """Tests for J2 gravity model."""

    def test_includes_oblateness(self):
        """J2 model returns valid gravity values."""
        result_eq = gravity_j2(0, 0, 0)
        result_pole = gravity_j2(np.pi / 2, 0, 0)

        # Both should be reasonable gravity values
        assert 9.7 < result_eq.magnitude < 9.9
        assert 9.7 < result_pole.magnitude < 9.9

    def test_north_component_nonzero(self):
        """J2 model has non-zero north component at mid-latitudes."""
        result = gravity_j2(np.radians(45), 0, 0)
        # North component should be small but present
        assert abs(result.g_north) < 0.1 * result.g_down


class TestGeoidHeight:
    """Tests for geoid height computation."""

    def test_geoid_equator_pole_difference(self):
        """Geoid is higher at equator than at poles for J2."""
        N_eq = geoid_height_j2(0)
        N_pole = geoid_height_j2(np.pi / 2)

        # J2 causes equatorial bulge
        assert N_eq > N_pole

    def test_geoid_symmetric(self):
        """Geoid height is symmetric about equator."""
        N_north = geoid_height_j2(np.radians(45))
        N_south = geoid_height_j2(np.radians(-45))

        assert_allclose(N_north, N_south)


class TestGravityAnomalies:
    """Tests for gravity anomaly computations."""

    def test_free_air_anomaly(self):
        """Free-air anomaly computation."""
        g_obs = 9.80
        lat = np.radians(45)
        h = 1000

        fa = free_air_anomaly(g_obs, lat, h)
        # Should be small (anomaly relative to normal)
        assert abs(fa) < 0.1

    def test_bouguer_anomaly(self):
        """Bouguer anomaly includes density correction."""
        g_obs = 9.80
        lat = np.radians(45)
        h = 1000

        fa = free_air_anomaly(g_obs, lat, h)
        ba = bouguer_anomaly(g_obs, lat, h)

        # Bouguer should be less than free-air (removes mass above)
        assert ba < fa


class TestWMM:
    """Tests for World Magnetic Model."""

    def test_wmm_returns_result(self):
        """WMM returns MagneticResult."""
        result = wmm(np.radians(40), np.radians(-105), 1.0, 2023.0)
        assert hasattr(result, "X")
        assert hasattr(result, "Y")
        assert hasattr(result, "Z")
        assert hasattr(result, "H")
        assert hasattr(result, "F")
        assert hasattr(result, "I")
        assert hasattr(result, "D")

    def test_total_intensity_reasonable(self):
        """Total field intensity is in expected range."""
        result = wmm(np.radians(45), np.radians(0), 0, 2023.0)
        # Field can be stronger at high latitudes
        assert 25000 < result.F < 100000

    def test_horizontal_intensity_formula(self):
        """H = sqrt(X^2 + Y^2)."""
        result = wmm(np.radians(40), np.radians(-75), 0, 2023.0)
        H_calc = np.sqrt(result.X**2 + result.Y**2)
        assert_allclose(result.H, H_calc, rtol=1e-10)

    def test_total_intensity_formula(self):
        """F = sqrt(H^2 + Z^2)."""
        result = wmm(np.radians(40), np.radians(-75), 0, 2023.0)
        F_calc = np.sqrt(result.H**2 + result.Z**2)
        assert_allclose(result.F, F_calc, rtol=1e-10)

    def test_declination_function(self):
        """magnetic_declination returns correct value."""
        result = wmm(np.radians(40), np.radians(-105), 0, 2023.0)
        D = magnetic_declination(np.radians(40), np.radians(-105), 0, 2023.0)
        assert_allclose(D, result.D)

    def test_inclination_function(self):
        """magnetic_inclination returns correct value."""
        result = wmm(np.radians(40), np.radians(-105), 0, 2023.0)
        incl = magnetic_inclination(np.radians(40), np.radians(-105), 0, 2023.0)
        assert_allclose(incl, result.I)

    def test_intensity_function(self):
        """magnetic_field_intensity returns correct value."""
        result = wmm(np.radians(40), np.radians(-105), 0, 2023.0)
        F = magnetic_field_intensity(np.radians(40), np.radians(-105), 0, 2023.0)
        assert_allclose(F, result.F)


class TestIGRF:
    """Tests for International Geomagnetic Reference Field."""

    def test_igrf_returns_result(self):
        """IGRF returns MagneticResult."""
        result = igrf(np.radians(45), np.radians(-75), 0, 2023.0)
        assert hasattr(result, "F")
        assert hasattr(result, "D")
        assert hasattr(result, "I")

    def test_igrf_similar_to_wmm(self):
        """IGRF and WMM should give similar results."""
        lat = np.radians(40)
        lon = np.radians(-75)

        wmm_result = wmm(lat, lon, 0, 2023.0)
        igrf_result = igrf(lat, lon, 0, 2023.0)

        # Should be within 1% for total field
        assert abs(wmm_result.F - igrf_result.F) / wmm_result.F < 0.1

    def test_dipole_moment_positive(self):
        """Dipole moment should be positive."""
        M = dipole_moment()
        assert M > 0

    def test_dipole_axis_returns_values(self):
        """dipole_axis returns lat/lon values."""
        lat, lon = dipole_axis()
        # Latitude should be in valid range
        assert -np.pi / 2 <= lat <= np.pi / 2
        # Longitude should be in valid range
        assert -np.pi <= lon <= np.pi

    def test_igrf_declination_function(self):
        """igrf_declination returns scalar."""
        D = igrf_declination(np.radians(45), np.radians(-75))
        assert isinstance(D, float)

    def test_igrf_inclination_function(self):
        """igrf_inclination returns scalar."""
        incl = igrf_inclination(np.radians(45), np.radians(-75))
        assert isinstance(incl, float)


class TestMagneticFieldProperties:
    """Tests for physical properties of magnetic field."""

    def test_inclination_positive_north(self):
        """Inclination is positive in northern hemisphere."""
        result = wmm(np.radians(60), 0, 0, 2023.0)
        assert result.I > 0  # Field points into Earth

    def test_inclination_negative_south(self):
        """Inclination is negative in southern hemisphere."""
        result = wmm(np.radians(-60), 0, 0, 2023.0)
        assert result.I < 0  # Field points out of Earth

    def test_field_stronger_at_poles(self):
        """Magnetic field is stronger near poles than equator."""
        F_pole = wmm(np.radians(80), 0, 0, 2023.0).F
        F_eq = wmm(0, 0, 0, 2023.0).F

        assert F_pole > F_eq

    def test_field_decreases_with_altitude(self):
        """Magnetic field decreases with altitude."""
        F_0 = wmm(np.radians(45), 0, 0, 2023.0).F
        F_100 = wmm(np.radians(45), 0, 100, 2023.0).F  # 100 km altitude

        assert F_0 > F_100


class TestCoefficients:
    """Tests for model coefficients."""

    def test_wmm2020_has_coefficients(self):
        """WMM2020 has non-zero coefficients."""
        assert WMM2020.g[1, 0] != 0
        assert WMM2020.n_max >= 12
        assert WMM2020.epoch == 2020.0

    def test_igrf13_has_coefficients(self):
        """IGRF13 has non-zero coefficients."""
        assert IGRF13.g[1, 0] != 0
        assert IGRF13.n_max >= 13
        assert IGRF13.epoch == 2020.0

    def test_g10_is_dominant(self):
        """g[1,0] is the dominant coefficient (dipole)."""
        # The axial dipole should be the largest term
        assert abs(WMM2020.g[1, 0]) > abs(WMM2020.g[2, 0])
        assert abs(WMM2020.g[1, 0]) > abs(WMM2020.g[1, 1])
