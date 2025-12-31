"""Tests for tidal effects module."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.gravity.tides import (
    GRAVIMETRIC_FACTOR,
    LOVE_H2,
    LOVE_K2,
    SHIDA_L2,
    TIDAL_CONSTITUENTS,
    OceanTideLoading,
    TidalDisplacement,
    TidalGravity,
    atmospheric_pressure_loading,
    fundamental_arguments,
    julian_centuries_j2000,
    moon_position_approximate,
    ocean_tide_loading_displacement,
    pole_tide_displacement,
    solid_earth_tide_displacement,
    solid_earth_tide_gravity,
    sun_position_approximate,
    tidal_gravity_correction,
    total_tidal_displacement,
)


class TestConstants:
    """Tests for tidal constants."""

    def test_love_number_h2_range(self):
        """Love number h2 should be approximately 0.6."""
        assert 0.5 < LOVE_H2 < 0.7

    def test_love_number_k2_range(self):
        """Love number k2 should be approximately 0.3."""
        assert 0.2 < LOVE_K2 < 0.4

    def test_shida_number_l2_range(self):
        """Shida number l2 should be approximately 0.08."""
        assert 0.05 < SHIDA_L2 < 0.12

    def test_gravimetric_factor(self):
        """Gravimetric factor should equal 1 + h - 3k/2."""
        expected = 1.0 + LOVE_H2 - 1.5 * LOVE_K2
        assert_allclose(GRAVIMETRIC_FACTOR, expected, rtol=1e-10)

    def test_tidal_constituents_defined(self):
        """Major tidal constituents should be defined."""
        assert "M2" in TIDAL_CONSTITUENTS
        assert "S2" in TIDAL_CONSTITUENTS
        assert "K1" in TIDAL_CONSTITUENTS
        assert "O1" in TIDAL_CONSTITUENTS

    def test_m2_frequency(self):
        """M2 frequency should be approximately 1.93 cycles/day."""
        assert_allclose(TIDAL_CONSTITUENTS["M2"], 1.9322736, rtol=1e-6)


class TestTimeConversions:
    """Tests for time conversion functions."""

    def test_julian_centuries_j2000(self):
        """Julian centuries at J2000.0 should be zero."""
        T = julian_centuries_j2000(51544.5)  # J2000.0
        assert_allclose(T, 0.0, atol=1e-10)

    def test_julian_centuries_one_century(self):
        """One century after J2000.0."""
        # 36525 days = 1 Julian century
        T = julian_centuries_j2000(51544.5 + 36525)
        assert_allclose(T, 1.0, atol=1e-10)

    def test_fundamental_arguments_range(self):
        """Fundamental arguments should be in [0, 2*pi)."""
        T = 0.1  # Some time
        l, l_prime, F, D, Omega = fundamental_arguments(T)

        assert 0 <= l < 2 * np.pi
        assert 0 <= l_prime < 2 * np.pi
        assert 0 <= F < 2 * np.pi
        assert 0 <= D < 2 * np.pi
        assert 0 <= Omega < 2 * np.pi


class TestMoonPosition:
    """Tests for Moon position computation."""

    def test_moon_distance_reasonable(self):
        """Moon distance should be approximately 384,400 km."""
        r, lat, lon = moon_position_approximate(58000)  # Arbitrary MJD
        # Within 10% of mean distance
        assert 350000e3 < r < 420000e3

    def test_moon_latitude_bounded(self):
        """Moon latitude should be bounded by inclination (~5.1°)."""
        r, lat, lon = moon_position_approximate(58000)
        assert abs(lat) < np.radians(6)  # Max ~5.1° + small perturbations

    def test_moon_longitude_range(self):
        """Moon longitude should be in [0, 2*pi)."""
        r, lat, lon = moon_position_approximate(58000)
        assert 0 <= lon < 2 * np.pi


class TestSunPosition:
    """Tests for Sun position computation."""

    def test_sun_distance_reasonable(self):
        """Sun distance should be approximately 1 AU."""
        r, lat, lon = sun_position_approximate(58000)
        AU = 1.496e11
        # Within 3% of 1 AU (accounts for eccentricity)
        assert 0.97 * AU < r < 1.03 * AU

    def test_sun_latitude_zero(self):
        """Sun ecliptic latitude should be approximately zero."""
        r, lat, lon = sun_position_approximate(58000)
        assert_allclose(lat, 0.0, atol=0.01)

    def test_sun_longitude_range(self):
        """Sun longitude should be in [0, 2*pi)."""
        r, lat, lon = sun_position_approximate(58000)
        assert 0 <= lon < 2 * np.pi


class TestSolidEarthTideDisplacement:
    """Tests for solid Earth tide displacement."""

    def test_result_type(self):
        """Result should be TidalDisplacement named tuple."""
        disp = solid_earth_tide_displacement(np.radians(45), 0, 58000)
        assert isinstance(disp, TidalDisplacement)
        assert hasattr(disp, "radial")
        assert hasattr(disp, "north")
        assert hasattr(disp, "east")

    def test_radial_displacement_magnitude(self):
        """Radial displacement should be typically < 50 cm."""
        disp = solid_earth_tide_displacement(np.radians(45), 0, 58000)
        assert abs(disp.radial) < 0.5  # 50 cm

    def test_horizontal_displacement_smaller(self):
        """Horizontal displacements typically smaller than radial."""
        disp = solid_earth_tide_displacement(np.radians(45), 0, 58000)
        # Horizontal typically 1/3 to 1/5 of radial
        horiz = np.sqrt(disp.north**2 + disp.east**2)
        assert horiz < 0.3  # 30 cm

    def test_displacement_varies_with_time(self):
        """Displacement should vary with time (tidal cycle)."""
        mjd = 58000
        disp1 = solid_earth_tide_displacement(np.radians(45), 0, mjd)
        disp2 = solid_earth_tide_displacement(
            np.radians(45), 0, mjd + 0.5
        )  # 12 hours later
        # M2 tide has ~12.4 hour period, displacements should differ
        assert disp1.radial != disp2.radial

    def test_displacement_varies_with_location(self):
        """Displacement should vary with location."""
        mjd = 58000
        disp1 = solid_earth_tide_displacement(np.radians(45), 0, mjd)
        disp2 = solid_earth_tide_displacement(
            np.radians(45), np.pi, mjd
        )  # Opposite side
        assert disp1.radial != disp2.radial


class TestSolidEarthTideGravity:
    """Tests for solid Earth tide gravity effect."""

    def test_result_type(self):
        """Result should be TidalGravity named tuple."""
        grav = solid_earth_tide_gravity(np.radians(45), 0, 58000)
        assert isinstance(grav, TidalGravity)
        assert hasattr(grav, "delta_g")
        assert hasattr(grav, "delta_g_north")
        assert hasattr(grav, "delta_g_east")

    def test_gravity_magnitude(self):
        """Gravity effect should be typically < 3 microGal (3e-8 m/s^2)."""
        grav = solid_earth_tide_gravity(np.radians(45), 0, 58000)
        # Tidal gravity typically ~100-300 microGal peak-to-peak
        assert abs(grav.delta_g) < 3e-6  # 300 microGal

    def test_gravity_varies_with_time(self):
        """Gravity effect should vary with time."""
        mjd = 58000
        grav1 = solid_earth_tide_gravity(np.radians(45), 0, mjd)
        grav2 = solid_earth_tide_gravity(np.radians(45), 0, mjd + 0.5)
        assert grav1.delta_g != grav2.delta_g


class TestOceanTideLoading:
    """Tests for ocean tide loading."""

    def test_result_type(self):
        """Result should be TidalDisplacement."""
        amp = np.array([[0.01, 0.005], [0.002, 0.001], [0.002, 0.001]])
        phase = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        disp = ocean_tide_loading_displacement(58000, amp, phase, ("M2", "S2"))
        assert isinstance(disp, TidalDisplacement)

    def test_zero_amplitude_zero_displacement(self):
        """Zero amplitude should give zero displacement."""
        amp = np.zeros((3, 2))
        phase = np.zeros((3, 2))
        disp = ocean_tide_loading_displacement(58000, amp, phase, ("M2", "S2"))
        assert_allclose(disp.radial, 0.0, atol=1e-15)
        assert_allclose(disp.north, 0.0, atol=1e-15)
        assert_allclose(disp.east, 0.0, atol=1e-15)

    def test_displacement_bounded_by_amplitude(self):
        """Displacement should be bounded by amplitude."""
        amp = np.array([[0.01, 0.005], [0.002, 0.001], [0.002, 0.001]])
        phase = np.zeros((3, 2))
        disp = ocean_tide_loading_displacement(58000, amp, phase, ("M2", "S2"))
        # Total amplitude for each component
        assert abs(disp.radial) <= amp[0, :].sum() * 1.1  # Allow small numerical error
        assert abs(disp.north) <= amp[1, :].sum() * 1.1
        assert abs(disp.east) <= amp[2, :].sum() * 1.1


class TestAtmosphericPressureLoading:
    """Tests for atmospheric pressure loading."""

    def test_result_type(self):
        """Result should be TidalDisplacement."""
        disp = atmospheric_pressure_loading(np.radians(45), 0, 101325)
        assert isinstance(disp, TidalDisplacement)

    def test_reference_pressure_zero_displacement(self):
        """At reference pressure, radial displacement should be zero."""
        disp = atmospheric_pressure_loading(np.radians(45), 0, 101325)
        assert_allclose(disp.radial, 0.0, atol=1e-10)

    def test_high_pressure_negative_displacement(self):
        """High pressure should cause negative (downward) displacement."""
        disp = atmospheric_pressure_loading(np.radians(45), 0, 102325)  # +10 hPa
        assert disp.radial < 0

    def test_low_pressure_positive_displacement(self):
        """Low pressure should cause positive (upward) displacement."""
        disp = atmospheric_pressure_loading(np.radians(45), 0, 100325)  # -10 hPa
        assert disp.radial > 0

    def test_displacement_magnitude(self):
        """10 hPa pressure change should give ~3.5 mm displacement."""
        disp = atmospheric_pressure_loading(np.radians(45), 0, 102325)  # +10 hPa
        # Default admittance is -0.35 mm/hPa = -0.35e-3 m/Pa
        expected = -0.35e-3 * 1000  # 10 hPa = 1000 Pa
        assert_allclose(disp.radial, expected, rtol=0.01)


class TestPoleTide:
    """Tests for pole tide displacement."""

    def test_result_type(self):
        """Result should be TidalDisplacement."""
        disp = pole_tide_displacement(np.radians(45), 0, 0.1, 0.1)
        assert isinstance(disp, TidalDisplacement)

    def test_zero_pole_offset_zero_displacement(self):
        """Zero pole offset should give zero displacement."""
        disp = pole_tide_displacement(np.radians(45), 0, 0, 0)
        assert_allclose(disp.radial, 0.0, atol=1e-15)
        assert_allclose(disp.north, 0.0, atol=1e-15)
        assert_allclose(disp.east, 0.0, atol=1e-15)

    def test_displacement_magnitude(self):
        """Pole tide displacement should be typically < 3 cm."""
        disp = pole_tide_displacement(np.radians(45), 0, 0.3, 0.3)  # ~0.3 arcsec
        assert abs(disp.radial) < 0.03  # 3 cm
        assert abs(disp.north) < 0.03
        assert abs(disp.east) < 0.03


class TestTotalTidalDisplacement:
    """Tests for total tidal displacement."""

    def test_result_type(self):
        """Result should be TidalDisplacement."""
        disp = total_tidal_displacement(np.radians(45), 0, 58000)
        assert isinstance(disp, TidalDisplacement)

    def test_includes_solid_earth_tide(self):
        """Total should include solid Earth tide contribution."""
        disp_total = total_tidal_displacement(np.radians(45), 0, 58000)
        disp_solid = solid_earth_tide_displacement(np.radians(45), 0, 58000)
        # If no other contributions, should be equal
        assert_allclose(disp_total.radial, disp_solid.radial, rtol=1e-10)

    def test_with_pressure(self):
        """Adding pressure should change displacement."""
        disp1 = total_tidal_displacement(np.radians(45), 0, 58000)
        disp2 = total_tidal_displacement(np.radians(45), 0, 58000, pressure=102325)
        assert disp1.radial != disp2.radial

    def test_with_pole_motion(self):
        """Adding pole motion should change displacement."""
        disp1 = total_tidal_displacement(np.radians(45), 0, 58000)
        disp2 = total_tidal_displacement(np.radians(45), 0, 58000, xp=0.3, yp=0.2)
        # Note: For some locations, the change might be very small
        # but at 45 degrees, there should be a measurable difference
        total_diff = abs(disp1.radial - disp2.radial) + abs(disp1.north - disp2.north)
        assert total_diff > 0


class TestTidalGravityCorrection:
    """Tests for tidal gravity correction."""

    def test_opposite_sign_to_effect(self):
        """Correction should have opposite sign to effect."""
        grav = solid_earth_tide_gravity(np.radians(45), 0, 58000)
        corr = tidal_gravity_correction(np.radians(45), 0, 58000)
        assert_allclose(corr, -grav.delta_g, rtol=1e-10)

    def test_correction_magnitude(self):
        """Correction should be typically < 300 microGal."""
        corr = tidal_gravity_correction(np.radians(45), 0, 58000)
        assert abs(corr) < 3e-6


class TestOceanTideLoadingType:
    """Tests for OceanTideLoading named tuple."""

    def test_create_ocean_loading(self):
        """Should be able to create OceanTideLoading."""
        amp = np.array([[0.01, 0.005], [0.002, 0.001], [0.002, 0.001]])
        phase = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        loading = OceanTideLoading(
            amplitude=amp, phase=phase, constituents=("M2", "S2")
        )
        assert loading.constituents == ("M2", "S2")
        assert loading.amplitude.shape == (3, 2)

    def test_use_in_total_displacement(self):
        """OceanTideLoading should work with total_tidal_displacement."""
        amp = np.array([[0.01, 0.005], [0.002, 0.001], [0.002, 0.001]])
        phase = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        loading = OceanTideLoading(
            amplitude=amp, phase=phase, constituents=("M2", "S2")
        )
        disp = total_tidal_displacement(np.radians(45), 0, 58000, ocean_loading=loading)
        assert isinstance(disp, TidalDisplacement)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_equator(self):
        """Should work at equator."""
        disp = solid_earth_tide_displacement(0, 0, 58000)
        assert np.isfinite(disp.radial)
        assert np.isfinite(disp.north)
        assert np.isfinite(disp.east)

    def test_pole(self):
        """Should work at poles."""
        disp = solid_earth_tide_displacement(np.pi / 2, 0, 58000)
        assert np.isfinite(disp.radial)
        assert np.isfinite(disp.north)
        assert np.isfinite(disp.east)

    def test_date_line(self):
        """Should work at date line (lon = pi)."""
        disp = solid_earth_tide_displacement(np.radians(45), np.pi, 58000)
        assert np.isfinite(disp.radial)

    def test_negative_longitude(self):
        """Should work with negative longitude."""
        disp = solid_earth_tide_displacement(np.radians(45), -np.pi / 2, 58000)
        assert np.isfinite(disp.radial)

    def test_different_mjd_epochs(self):
        """Should work for different MJD epochs."""
        # Past
        disp1 = solid_earth_tide_displacement(np.radians(45), 0, 40000)
        assert np.isfinite(disp1.radial)

        # Future
        disp2 = solid_earth_tide_displacement(np.radians(45), 0, 70000)
        assert np.isfinite(disp2.radial)
