"""Test suite for relativistic corrections module.

Tests cover all public functions with physical validation against known results
and edge case handling.
"""

import numpy as np
import pytest

from pytcl.astronomical.relativity import (
    AU,
    C_LIGHT,
    GM_EARTH,
    GM_SUN,
    geodetic_precession,
    gravitational_time_dilation,
    lense_thirring_precession,
    post_newtonian_acceleration,
    proper_time_rate,
    relativistic_range_correction,
    schwarzschild_precession_per_orbit,
    schwarzschild_radius,
    shapiro_delay,
)


class TestSchwarzchildRadius:
    """Test Schwarzschild radius calculations."""

    def test_earth_schwarzschild_radius(self):
        """Schwarzschild radius for Earth should be ~8.87 mm."""
        mass_earth = 5.972e24  # kg
        r_s = schwarzschild_radius(mass_earth)
        assert 8.8e-3 < r_s < 8.9e-3, f"Expected ~8.87 mm, got {r_s:.3e} m"

    def test_sun_schwarzschild_radius(self):
        """Schwarzschild radius for Sun should be ~2.95 km."""
        mass_sun = 1.989e30  # kg
        r_s = schwarzschild_radius(mass_sun)
        assert 2.9e3 < r_s < 3.0e3, f"Expected ~2.95 km, got {r_s:.3e} m"

    def test_zero_mass(self):
        """Zero mass should give zero radius."""
        assert schwarzschild_radius(0.0) == 0.0

    def test_radius_increases_with_mass(self):
        """Schwarzschild radius should increase linearly with mass."""
        r_s1 = schwarzschild_radius(1e24)
        r_s2 = schwarzschild_radius(2e24)
        assert abs(r_s2 / r_s1 - 2.0) < 1e-10


class TestGravitationalTimeDilation:
    """Test gravitational time dilation calculations."""

    def test_dilation_at_infinity(self):
        """Time should pass normally at infinity (dilation = 1)."""
        # Test at very large distance
        r_large = 1e20  # Very far
        dilation = gravitational_time_dilation(r_large, GM_EARTH)
        assert abs(dilation - 1.0) < 1e-15

    def test_dilation_at_earth_surface(self):
        """Time dilation at Earth's surface should be ~0.9999999993."""
        r_earth = 6.371e6  # meters
        dilation = gravitational_time_dilation(r_earth, GM_EARTH)
        # Expected: sqrt(1 - 2*3.986e14/(3e8^2*6.371e6))
        expected_squared = 1.0 - 2.0 * GM_EARTH / (C_LIGHT**2 * r_earth)
        expected = np.sqrt(expected_squared)
        assert abs(dilation - expected) < 1e-15

    def test_dilation_increases_outward(self):
        """Time dilation should increase (approach 1) as distance increases."""
        r1 = 1e7
        r2 = 1e8
        d1 = gravitational_time_dilation(r1, GM_EARTH)
        d2 = gravitational_time_dilation(r2, GM_EARTH)
        assert d2 > d1  # More distant point has larger dilation

    def test_dilation_below_schwarzschild_radius(self):
        """Should raise error for r <= Schwarzschild radius."""
        r_s_earth = schwarzschild_radius(5.972e24)
        with pytest.raises(ValueError):
            gravitational_time_dilation(r_s_earth - 1e-3, GM_EARTH)

    def test_sun_vs_earth_dilation(self):
        """At same distance, Sun's gravity gives stronger dilation than Earth's."""
        r = 1e7
        dilation_earth = gravitational_time_dilation(r, GM_EARTH)
        dilation_sun = gravitational_time_dilation(r, GM_SUN)
        assert dilation_sun < dilation_earth  # Stronger effect from Sun


class TestProperTimeRate:
    """Test proper time rate calculations (SR + GR combined)."""

    def test_stationary_at_infinity(self):
        """Proper time at rest at infinity should equal coordinate time."""
        rate = proper_time_rate(0.0, 1e20, GM_EARTH)
        assert abs(rate - 1.0) < 1e-15

    def test_gps_satellite(self):
        """GPS satellite experiences both SR and GR time dilation effects."""
        v_gps = 3870.0  # m/s, typical GPS speed
        r_gps = 26.56e6  # meters, ~20,200 km altitude
        rate = proper_time_rate(v_gps, r_gps, GM_EARTH)

        # Both effects slow down time
        assert rate < 1.0

        # SR effect: ~-v^2/(2c^2) ≈ -8.3e-11
        # GR effect: ~-GM/(c^2*r) ≈ -5.3e-10
        # Net should be between SR-only and GR dominant
        sr_only = 1.0 - (v_gps**2) / (2.0 * C_LIGHT**2)
        # Rate is less than sr_only because GR adds negative time dilation
        # Rate = sr_only + gr_effect, where gr_effect is negative
        assert rate < sr_only  # GR adds more time dilation

    def test_high_velocity_dominates_at_small_r(self):
        """At very small radius with high velocity, SR should dominate."""
        v = 0.1 * C_LIGHT  # Relativistic velocity
        r = 1e7  # Distance to GR effect
        rate = proper_time_rate(v, r, GM_EARTH)

        # SR effect should be significant
        sr_component = 1.0 - (v**2) / (2.0 * C_LIGHT**2)
        gr_component = -GM_EARTH / (C_LIGHT**2 * r)

        expected = sr_component + gr_component
        assert abs(rate - expected) < 1e-15


class TestShapiroDelay:
    """Test Shapiro delay (light bending in gravitational field)."""

    def test_shapiro_delay_positive(self):
        """Shapiro delay should be positive when light bends through gravity field."""
        # Create a geometry where light path passes near the gravitating body
        obs = np.array([2.0e11, 0.0, 0.0])  # Observer far on one side
        source = np.array([0.0, 2.0e11, 0.0])  # Source far on perpendicular side
        sun = np.array([0.0, 0.0, 0.0])  # Sun at origin

        delay = shapiro_delay(obs, source, sun, GM_SUN)
        assert delay >= 0.0  # Should be positive or zero

    def test_shapiro_delay_superior_conjunction(self):
        """Shapiro delay is larger when light path passes closer to Sun."""
        # Two scenarios: one with more straight path, one with closer passage

        # More distant geometry
        obs1 = np.array([1.496e11, 0.5e11, 0.0])
        source1 = np.array([-1.496e11, 0.5e11, 0.0])

        # Closer passage (light bends more)
        obs2 = np.array([1.496e11, 0.1e11, 0.0])
        source2 = np.array([-1.496e11, 0.1e11, 0.0])

        sun = np.array([0.0, 0.0, 0.0])

        delay1 = shapiro_delay(obs1, source1, sun, GM_SUN)
        delay2 = shapiro_delay(obs2, source2, sun, GM_SUN)

        # Closer passage should have larger delay
        assert delay2 >= delay1

    def test_shapiro_delay_no_body(self):
        """Shapiro delay should be ~0 when gravitating body is far from light path."""
        obs = np.array([1.0, 0.0, 0.0])
        source = np.array([0.0, 1.0, 0.0])
        sun_far = np.array([1000.0, 1000.0, 0.0])  # Very far from light path

        delay = shapiro_delay(obs, source, sun_far, GM_SUN)
        # Should be very small (essentially zero for light path far from body)
        assert abs(delay) < 1e-8

    def test_shapiro_collinear_error(self):
        """Shapiro delay formula handles collinear geometries."""
        obs = np.array([2.0e11, 0.0, 0.0])
        body = np.array([0.0, 0.0, 0.0])
        source = np.array(
            [0.5e11, 0.0, 0.0]
        )  # Between body and observer (on same line)

        # For collinear geometry, formula is less meaningful but should still return finite value
        delay = shapiro_delay(obs, source, body, GM_SUN)
        assert np.isfinite(delay)


class TestSchwarzchildPrecession:
    """Test perihelion precession due to general relativity."""

    def test_mercury_precession(self):
        """Mercury's GR perihelion precession should be ~43 arcsec/century."""
        a_mercury = 0.38709927 * AU  # Semi-major axis
        e_mercury = 0.20563593  # Eccentricity

        # Precession per orbit
        precession_rad = schwarzschild_precession_per_orbit(
            a_mercury, e_mercury, GM_SUN
        )
        precession_arcsec = precession_rad * 206265  # Convert to arcseconds

        # Mercury orbital period
        orbital_period = 87.969 / 365.25  # years
        orbits_per_century = 100.0 / orbital_period

        precession_per_century = precession_arcsec * orbits_per_century

        # Expected: ~43 arcsec/century from GR
        # (Observed is ~5600 arcsec/century, but includes Newtonian precession)
        assert (
            40 < precession_per_century < 45
        ), f"Got {precession_per_century:.1f} arcsec/century"

    def test_circular_orbit_zero_ecc(self):
        """Circular orbit (e=0) should give precession for any semi-major axis."""
        a = 1.0e7  # arbitrary
        e = 0.0
        precession = schwarzschild_precession_per_orbit(a, e, GM_EARTH)
        assert precession > 0.0

    def test_eccentricity_effect(self):
        """Higher eccentricity should increase precession rate."""
        a = 1e7
        e1 = 0.1
        e2 = 0.5

        p1 = schwarzschild_precession_per_orbit(a, e1, GM_EARTH)
        p2 = schwarzschild_precession_per_orbit(a, e2, GM_EARTH)

        assert p2 > p1  # Higher e gives larger precession

    def test_invalid_eccentricity(self):
        """Should raise error for invalid eccentricity."""
        with pytest.raises(ValueError):
            schwarzschild_precession_per_orbit(1e7, 1.5, GM_SUN)

        with pytest.raises(ValueError):
            schwarzschild_precession_per_orbit(1e7, -0.1, GM_SUN)


class TestPostNewtonianAcceleration:
    """Test 1PN order acceleration corrections."""

    def test_circular_leo_orbit(self):
        """For LEO circular orbit, PN correction should be measurable."""
        # ~300 km altitude circular orbit
        r = 6.678e6
        v = 7.7e3  # Circular orbit velocity

        r_vec = np.array([r, 0.0, 0.0])
        v_vec = np.array([0.0, v, 0.0])

        a_total = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)
        a_newt = -GM_EARTH / r**2 * np.array([1.0, 0.0, 0.0])

        # PN corrections exist but should be very small
        correction = a_total - a_newt

        # Check that PN effects exist and are measurable (ppm level)
        assert np.linalg.norm(correction) > 0.0

    def test_zero_velocity(self):
        """At zero velocity, PN terms should reduce with Newtonian as dominant."""
        r_vec = np.array([1e7, 0.0, 0.0])
        v_vec = np.array([0.0, 0.0, 0.0])

        a = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)
        a_newt = -GM_EARTH / np.linalg.norm(r_vec) ** 2 * np.array([1.0, 0.0, 0.0])

        # With zero velocity, there are still some PN terms from the metric
        # They should be reasonably small relative to Newtonian
        ratio = np.linalg.norm(a - a_newt) / np.linalg.norm(a_newt)
        assert ratio < 0.05  # Less than 5% correction

    def test_radial_velocity(self):
        """Radial velocity components affect acceleration direction."""
        r_vec = np.array([1e7, 0.0, 0.0])
        v_rad = np.array([1e3, 0.0, 0.0])  # Outward velocity
        v_tan = np.array([0.0, 1e3, 0.0])  # Tangential velocity

        a_rad = post_newtonian_acceleration(r_vec, v_rad, GM_EARTH)
        a_tan = post_newtonian_acceleration(r_vec, v_tan, GM_EARTH)

        # Radial vs tangential velocity should affect the PN term differently
        # At minimum, they should produce different accelerations
        a_newt = -GM_EARTH / np.linalg.norm(r_vec) ** 2 * np.array([1.0, 0.0, 0.0])

        # Both should be different from pure Newtonian
        assert not np.allclose(a_rad, a_newt, atol=1e-10)
        assert not np.allclose(a_tan, a_newt, atol=1e-10)


class TestGeodeticPrecession:
    """Test geodetic (de Sitter) precession."""

    def test_equatorial_orbit(self):
        """Equatorial orbit (i=0) should have maximum geodetic precession."""
        a = 6.678e6
        e = 0.0
        i = 0.0

        precession = geodetic_precession(a, e, i, GM_EARTH)
        assert precession < 0.0  # Negative (retrograde)

    def test_polar_orbit(self):
        """Polar orbit (i=90°) should have zero geodetic precession."""
        a = 6.678e6
        e = 0.0
        i = np.pi / 2

        precession = geodetic_precession(a, e, i, GM_EARTH)
        assert abs(precession) < 1e-15

    def test_inclination_effect(self):
        """Precession should scale with cos(inclination)."""
        a = 6.678e6
        e = 0.0
        i1 = np.radians(30)
        i2 = np.radians(60)

        p1 = geodetic_precession(a, e, i1, GM_EARTH)
        p2 = geodetic_precession(a, e, i2, GM_EARTH)

        # |p2| < |p1| because cos(60°) < cos(30°)
        assert abs(p2) < abs(p1)

    def test_altitude_effect(self):
        """Higher altitude should reduce precession magnitude."""
        e = 0.0
        i = np.radians(45)

        a_leo = 6.678e6  # LEO
        a_geo = 42.164e6  # GEO

        p_leo = geodetic_precession(a_leo, e, i, GM_EARTH)
        p_geo = geodetic_precession(a_geo, e, i, GM_EARTH)

        # GEO precession should be smaller magnitude
        assert abs(p_geo) < abs(p_leo)


class TestLenseThirringPrecession:
    """Test Lense-Thirring (frame-dragging) precession."""

    def test_lense_thirring_positive(self):
        """Lense-Thirring effect should cause prograde precession."""
        a = 12.27e6  # LAGEOS
        e = 0.0045
        i = np.radians(109.9)
        L = 7.05e33  # Earth's angular momentum

        precession = lense_thirring_precession(a, e, i, L, GM_EARTH)
        assert precession > 0.0  # Prograde

    def test_no_rotation_no_precession(self):
        """Zero angular momentum should give zero Lense-Thirring effect."""
        a = 6.678e6
        e = 0.0
        i = np.radians(51.6)

        precession = lense_thirring_precession(a, e, i, 0.0, GM_EARTH)
        assert precession == 0.0

    def test_altitude_effect(self):
        """Higher altitude should reduce Lense-Thirring effect."""
        e = 0.0
        i = np.radians(51.6)
        L = 7.05e33

        a_leo = 6.678e6
        a_geo = 42.164e6

        p_leo = lense_thirring_precession(a_leo, e, i, L, GM_EARTH)
        p_geo = lense_thirring_precession(a_geo, e, i, L, GM_EARTH)

        assert p_geo < p_leo


class TestRelativisticsRangeCorrection:
    """Test relativistic range corrections for ranging measurements."""

    def test_range_correction_positive(self):
        """Range correction should be non-negative."""
        distance = 3.84e8  # Lunar distance
        velocity = 0.0

        correction = relativistic_range_correction(distance, velocity, GM_EARTH)
        # Should be a small positive value
        assert correction >= 0.0

    def test_lunar_laser_ranging(self):
        """Lunar laser ranging correction should be small but measurable."""
        distance = 3.84e8  # meters
        velocity = 0.0

        correction = relativistic_range_correction(distance, velocity, GM_EARTH)

        # Should be a small positive number (order of cm to m)
        # Gravitational correction is ~gm/c^2 for Earth ~1.5 mm
        assert 0.0 <= correction <= 0.1  # Allow up to 10 cm

    def test_velocity_effect(self):
        """Higher radial velocity should increase correction."""
        distance = 3.84e8
        v1 = 0.0
        v2 = 100.0  # m/s

        c1 = relativistic_range_correction(distance, v1, GM_EARTH)
        c2 = relativistic_range_correction(distance, v2, GM_EARTH)

        assert c2 > c1  # Velocity increases correction

    def test_distance_effect(self):
        """Range correction doesn't depend on distance (weak-field approximation)."""
        v = 0.0

        # In weak-field approximation, correction is constant (gm/c^2)
        d_earth = 6.371e6  # At Earth surface
        d_moon = 3.84e8  # To Moon

        c_earth = relativistic_range_correction(d_earth, v, GM_EARTH)
        c_moon = relativistic_range_correction(d_moon, v, GM_EARTH)

        # Should be approximately the same (weak-field limit)
        assert np.isclose(c_moon, c_earth, rtol=1e-10)


class TestPhysicalConsistency:
    """Cross-cutting tests for physical consistency."""

    def test_weak_field_limit(self):
        """At large distances, PN effects should match weak-field expansion."""
        # At r >> r_s, all relativistic effects should be small
        r = 1e12  # Very large distance

        dilation = gravitational_time_dilation(r, GM_EARTH)
        expected = 1.0 - GM_EARTH / (C_LIGHT**2 * r)  # Weak field approximation

        assert abs(dilation - expected) < 1e-15

    def test_schwarzschild_precession_dimensionless(self):
        """Precession per orbit should be dimensionless (radians)."""
        a = 1e7
        e = 0.3
        precession = schwarzschild_precession_per_orbit(a, e, GM_EARTH)

        # Should be small fraction of 2π
        assert 0 < precession < 0.1

    def test_shapiro_delay_causality(self):
        """Shapiro delay should satisfy causality (finite, non-negative)."""
        positions = [
            (np.array([1.0e11, 0.5e11, 0.0]), np.array([-1.0e11, 0.5e11, 0.0])),
            (np.array([1.0e11, 0.1e11, 0.0]), np.array([-1.0e11, 0.1e11, 0.0])),
            (np.array([1.0e11, 0.2e10, 0.0]), np.array([-1.0e11, 0.2e10, 0.0])),
        ]

        center = np.array([0.0, 0.0, 0.0])

        for obs, src in positions:
            delay = shapiro_delay(obs, src, center, GM_SUN)
            assert np.isfinite(delay) and delay >= 0.0
