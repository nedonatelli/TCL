"""Test suite for relativistic corrections module.

Tests cover all public functions with physical validation against known results
and edge case handling.
"""

import pytest
import numpy as np
from pytcl.astronomical.relativity import (
    schwarzschild_radius,
    gravitational_time_dilation,
    proper_time_rate,
    shapiro_delay,
    schwarzschild_precession_per_orbit,
    post_newtonian_acceleration,
    geodetic_precession,
    lense_thirring_precession,
    relativistic_range_correction,
    C_LIGHT, G_GRAV, GM_EARTH, GM_SUN, AU
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
        expected_squared = 1.0 - 2.0 * GM_EARTH / (C_LIGHT ** 2 * r_earth)
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
        # Net: GR dominates, but SR is non-negligible
        sr_only = 1.0 - (v_gps ** 2) / (2.0 * C_LIGHT ** 2)
        assert sr_only < rate < 1.0  # GR is dominant
    
    def test_high_velocity_dominates_at_small_r(self):
        """At very small radius with high velocity, SR should dominate."""
        v = 0.1 * C_LIGHT  # Relativistic velocity
        r = 1e7  # Distance to GR effect
        rate = proper_time_rate(v, r, GM_EARTH)
        
        # SR effect should be significant
        sr_component = 1.0 - (v ** 2) / (2.0 * C_LIGHT ** 2)
        gr_component = -GM_EARTH / (C_LIGHT ** 2 * r)
        
        expected = sr_component + gr_component
        assert abs(rate - expected) < 1e-15


class TestShapiroDelay:
    """Test Shapiro delay (light bending in gravitational field)."""
    
    def test_shapiro_delay_positive(self):
        """Shapiro delay should always be positive (increases light travel time)."""
        obs = np.array([1.496e11, 0.0, 0.0])  # Earth
        source = np.array([-1.496e11, 0.0, 0.0])  # Far side of Sun
        sun = np.array([0.0, 0.0, 0.0])
        
        delay = shapiro_delay(obs, source, sun, GM_SUN)
        assert delay > 0.0
    
    def test_shapiro_delay_superior_conjunction(self):
        """Shapiro delay at superior conjunction should be ~250 microseconds."""
        # Earth-Sun-Spacecraft geometry
        obs = np.array([1.496e11, 0.0, 0.0])
        source = np.array([-(1.496e11 + 0.8e11), 0.0, 0.0])  # Beyond Sun, ~0.8 AU
        sun = np.array([0.0, 0.0, 0.0])
        
        delay = shapiro_delay(obs, source, sun, GM_SUN)
        
        # Should be on order of 100+ microseconds for interplanetary distances
        assert delay > 100e-6 and delay < 500e-6
    
    def test_shapiro_delay_no_body(self):
        """Shapiro delay should be ~0 when gravitating body is far away."""
        obs = np.array([1.0, 0.0, 0.0])
        source = np.array([0.0, 1.0, 0.0])
        sun_far = np.array([1000.0, 1000.0, 0.0])  # Very far
        
        delay = shapiro_delay(obs, source, sun_far, GM_SUN)
        assert delay < 1e-9  # Nearly zero
    
    def test_shapiro_collinear_error(self):
        """Should raise error if source is between observer and body."""
        obs = np.array([2.0, 0.0, 0.0])
        body = np.array([0.0, 0.0, 0.0])
        source = np.array([0.5, 0.0, 0.0])  # Between body and observer
        
        with pytest.raises(ValueError):
            shapiro_delay(obs, source, body, GM_SUN)


class TestSchwarzchildPrecession:
    """Test perihelion precession due to general relativity."""
    
    def test_mercury_precession(self):
        """Mercury's GR perihelion precession should be ~43 arcsec/century."""
        a_mercury = 0.38709927 * AU  # Semi-major axis
        e_mercury = 0.20563593  # Eccentricity
        
        # Precession per orbit
        precession_rad = schwarzschild_precession_per_orbit(a_mercury, e_mercury, GM_SUN)
        precession_arcsec = precession_rad * 206265  # Convert to arcseconds
        
        # Mercury orbital period
        orbital_period = 87.969 / 365.25  # years
        orbits_per_century = 100.0 / orbital_period
        
        precession_per_century = precession_arcsec * orbits_per_century
        
        # Expected: ~43 arcsec/century from GR
        # (Observed is ~5600 arcsec/century, but includes Newtonian precession)
        assert 40 < precession_per_century < 45, f"Got {precession_per_century:.1f} arcsec/century"
    
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
        """For LEO circular orbit, PN correction should be small but measurable."""
        # ~300 km altitude circular orbit
        r = 6.678e6
        v = 7.7e3  # Circular orbit velocity
        
        r_vec = np.array([r, 0.0, 0.0])
        v_vec = np.array([0.0, v, 0.0])
        
        a_total = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)
        a_newt = -GM_EARTH / r ** 2 * np.array([1.0, 0.0, 0.0])
        
        correction = np.linalg.norm(a_total - a_newt)
        relative_correction = correction / np.linalg.norm(a_newt)
        
        # Should be on order of 1e-6 to 1e-7
        assert 1e-8 < relative_correction < 1e-4
    
    def test_zero_velocity(self):
        """At zero velocity, PN terms should reduce correctly."""
        r_vec = np.array([1e7, 0.0, 0.0])
        v_vec = np.array([0.0, 0.0, 0.0])
        
        a = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)
        a_newt = -GM_EARTH / np.sum(r_vec**2) ** 1.5 * r_vec
        
        # Should be close to Newtonian with small corrections
        assert np.allclose(a, a_newt, rtol=1e-4)
    
    def test_radial_velocity(self):
        """Radial velocity should affect PN acceleration."""
        r_vec = np.array([1e7, 0.0, 0.0])
        v_rad = np.array([1e3, 0.0, 0.0])
        v_tan = np.array([0.0, 1e3, 0.0])
        
        a_rad = post_newtonian_acceleration(r_vec, v_rad, GM_EARTH)
        a_tan = post_newtonian_acceleration(r_vec, v_tan, GM_EARTH)
        
        # Different velocity directions should give different accelerations
        assert not np.allclose(a_rad, a_tan)


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
        """Range correction should be positive (increases measured range)."""
        distance = 3.84e8  # Lunar distance
        velocity = 0.0
        
        correction = relativistic_range_correction(distance, velocity, GM_EARTH)
        assert correction > 0.0
    
    def test_lunar_laser_ranging(self):
        """Lunar laser ranging correction should be on order of meters."""
        distance = 3.84e8  # meters
        velocity = 0.0
        
        correction = relativistic_range_correction(distance, velocity, GM_EARTH)
        
        # Should be on order of 1-10 meters
        assert 0.1 < correction < 100.0
    
    def test_velocity_effect(self):
        """Higher radial velocity should increase correction."""
        distance = 3.84e8
        v1 = 0.0
        v2 = 100.0  # m/s
        
        c1 = relativistic_range_correction(distance, v1, GM_EARTH)
        c2 = relativistic_range_correction(distance, v2, GM_EARTH)
        
        assert c2 > c1  # Velocity increases correction
    
    def test_distance_effect(self):
        """Larger distance should increase gravitational correction."""
        v = 0.0
        
        d_earth = 6.371e6  # At Earth surface
        d_moon = 3.84e8  # To Moon
        
        c_earth = relativistic_range_correction(d_earth, v, GM_EARTH)
        c_moon = relativistic_range_correction(d_moon, v, GM_EARTH)
        
        # Larger distance means larger logarithmic correction
        assert c_moon > c_earth


class TestPhysicalConsistency:
    """Cross-cutting tests for physical consistency."""
    
    def test_weak_field_limit(self):
        """At large distances, PN effects should match weak-field expansion."""
        # At r >> r_s, all relativistic effects should be small
        r = 1e12  # Very large distance
        
        dilation = gravitational_time_dilation(r, GM_EARTH)
        expected = 1.0 - GM_EARTH / (C_LIGHT ** 2 * r)  # Weak field approximation
        
        assert abs(dilation - expected) < 1e-15
    
    def test_schwarzschild_precession_dimensionless(self):
        """Precession per orbit should be dimensionless (radians)."""
        a = 1e7
        e = 0.3
        precession = schwarzschild_precession_per_orbit(a, e, GM_EARTH)
        
        # Should be small fraction of 2π
        assert 0 < precession < 0.1
    
    def test_shapiro_delay_causality(self):
        """Shapiro delay should satisfy causality (finite, positive)."""
        positions = [
            (np.array([1.0e11, 0.0, 0.0]), np.array([-1.0e11, 0.0, 0.0])),
            (np.array([1.0e11, 1.0e10, 0.0]), np.array([-1.0e11, 1.0e10, 0.0])),
            (np.array([1.0e11, 0.0, 0.0]), np.array([-1.0e11, 1.0e10, 1.0e10])),
        ]
        
        center = np.array([0.0, 0.0, 0.0])
        
        for obs, src in positions:
            delay = shapiro_delay(obs, src, center, GM_SUN)
            assert np.isfinite(delay) and delay > 0.0
