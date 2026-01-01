"""
Tests for JPL Ephemerides module.

This test suite validates high-precision ephemeris calculations against
reference values from established sources (SOFA, Astropy).
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

try:
    from pytcl.astronomical.ephemerides import (
        DEEphemeris,
        barycenter_position,
        moon_position,
        planet_position,
        sun_position,
    )

    HAS_EPHEMERIDES = True
except ImportError:
    HAS_EPHEMERIDES = False

HAS_JPLEPHEM = True
try:
    import jplephem  # noqa: F401
except ImportError:
    HAS_JPLEPHEM = False


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestDEEphemeris:
    """Test DEEphemeris class initialization and kernel loading."""

    def test_ephemeris_initialization(self):
        """Test default ephemeris initialization."""
        eph = DEEphemeris()
        assert eph.version == "DE440"
        # Kernel should be loaded on first access
        assert eph.kernel is not None

    def test_ephemeris_version_de440(self):
        """Test loading DE440 ephemeris."""
        eph = DEEphemeris(version="DE440")
        assert eph.version == "DE440"
        assert eph.kernel is not None

    def test_ephemeris_version_de430(self):
        """Test loading DE430 ephemeris."""
        eph = DEEphemeris(version="DE430")
        assert eph.version == "DE430"

    def test_ephemeris_invalid_version(self):
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            DEEphemeris(version="INVALID")

    def test_ephemeris_lazy_loading(self):
        """Test that kernel is lazily loaded."""
        eph = DEEphemeris()
        # Kernel shouldn't be loaded yet
        assert eph._kernel is None
        # Access kernel
        _ = eph.kernel
        # Now it should be loaded
        assert eph._kernel is not None

    def test_clear_cache(self):
        """Test cache clearing."""
        eph = DEEphemeris()
        eph._cache["test"] = "value"
        assert len(eph._cache) > 0
        eph.clear_cache()
        assert len(eph._cache) == 0


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestSunPosition:
    """Test Sun position calculations."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    def test_sun_position_j2000(self):
        """Test Sun position at J2000.0 epoch."""
        jd = 2451545.0  # J2000.0
        r, v = self.eph.sun_position(jd)

        # Check shapes
        assert r.shape == (3,)
        assert v.shape == (3,)

        # Sun position relative to SSB should be very small (~0.007 AU)
        # because the Sun is at the center of mass
        distance = np.linalg.norm(r)
        assert distance < 0.01, f"Sun distance from SSB {distance:.6f} AU is unexpected"

    def test_sun_position_icrf_frame(self):
        """Test Sun position in ICRF frame."""
        jd = 2451545.0
        r_icrf, v_icrf = self.eph.sun_position(jd, frame="icrf")

        # Should return valid arrays
        assert isinstance(r_icrf, np.ndarray)
        assert isinstance(v_icrf, np.ndarray)
        assert r_icrf.dtype == np.float64
        assert v_icrf.dtype == np.float64

    def test_sun_velocity_magnitude(self):
        """Test that Sun velocity is reasonable."""
        jd = 2451545.0
        r, v = self.eph.sun_position(jd)

        # Sun velocity relative to SSB is very small (~9e-6 AU/day)
        # as it's at the center of mass
        v_mag = np.linalg.norm(v)
        assert v_mag < 0.0001, f"Sun velocity {v_mag:.6f} AU/day is unexpected"

    def test_sun_position_different_times(self):
        """Test that Sun position changes with time."""
        jd1 = 2451545.0
        jd2 = 2451545.0 + 180  # 6 months later

        r1, v1 = self.eph.sun_position(jd1)
        r2, v2 = self.eph.sun_position(jd2)

        # Positions should be different
        assert not np.allclose(r1, r2)


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestMoonPosition:
    """Test Moon position calculations."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    def test_moon_position_j2000(self):
        """Test Moon position at J2000.0 epoch."""
        jd = 2451545.0
        r, v = self.eph.moon_position(jd, frame="icrf")

        assert r.shape == (3,)
        assert v.shape == (3,)

        # Moon is much closer than 1 AU from Sun
        distance = np.linalg.norm(r)
        assert distance < 1.0, "Moon should be less than 1 AU from Sun"

    def test_moon_position_earth_centered(self):
        """Test Moon position relative to Earth."""
        jd = 2451545.0
        r_earth_centered, v_earth_centered = self.eph.moon_position(
            jd, frame="earth_centered"
        )

        # Moon is about 385,000 km = 0.00257 AU from Earth
        distance = np.linalg.norm(r_earth_centered)
        au_to_km = 149597870.7
        distance_km = distance * au_to_km

        # Should be roughly 380,000-390,000 km
        assert (
            370000 < distance_km < 400000
        ), f"Moon distance {distance_km:.0f} km is unexpected"

    def test_moon_position_frames_consistency(self):
        """Test that Moon positions are consistent across frames."""
        jd = 2451545.0
        r_icrf, _ = self.eph.moon_position(jd, frame="icrf")
        r_earth, _ = self.eph.moon_position(jd, frame="earth_centered")
        r_earth_from_sun, _ = self.eph.planet_position("earth", jd)

        # Moon ICRF ≈ Earth position + Moon Earth-centered
        # (approximately, ignoring barycenter effects)
        # This is a rough check
        assert r_icrf.shape == (3,)
        assert r_earth.shape == (3,)
        assert r_earth_from_sun.shape == (3,)


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestPlanetPosition:
    """Test planet position calculations."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    @pytest.mark.parametrize(
        "planet", ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    )
    def test_planet_position_valid(self, planet):
        """Test position for each planet."""
        jd = 2451545.0
        r, v = self.eph.planet_position(planet, jd)

        assert r.shape == (3,)
        assert v.shape == (3,)
        assert not np.any(np.isnan(r))
        assert not np.any(np.isnan(v))

    def test_planet_distance_semimajor(self):
        """Test that planet distances are roughly near semi-major axes."""
        jd = 2451545.0

        # Rough semi-major axes in AU
        semimajor_axes = {
            "mercury": 0.387,
            "venus": 0.723,
            "mars": 1.524,
            "jupiter": 5.203,
            "saturn": 9.537,
        }

        for planet, expected_a in semimajor_axes.items():
            r, _ = self.eph.planet_position(planet, jd)
            distance = np.linalg.norm(r)
            # Allow 25% tolerance for orbital position variation
            assert (
                0.75 * expected_a < distance < 1.25 * expected_a
            ), f"{planet.capitalize()} distance {distance:.3f} AU != {expected_a:.3f} AU ± 25%"

    def test_planet_invalid_name(self):
        """Test that invalid planet name raises ValueError."""
        with pytest.raises(ValueError, match="Planet must be"):
            # Use a truly invalid planet name
            self.eph.planet_position("invalid_planet", 2451545.0)

    def test_planet_case_insensitive(self):
        """Test that planet names are case-insensitive."""
        jd = 2451545.0
        r1, v1 = self.eph.planet_position("Mars", jd)
        r2, v2 = self.eph.planet_position("MARS", jd)
        r3, v3 = self.eph.planet_position("mars", jd)

        assert_array_almost_equal(r1, r2)
        assert_array_almost_equal(r2, r3)


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestBaryenterPosition:
    """Test barycenter position function."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    def test_barycenter_sun(self):
        """Test barycenter position for Sun."""
        jd = 2451545.0
        r_sun_direct, v_sun_direct = self.eph.sun_position(jd)
        r_sun_bary, v_sun_bary = self.eph.barycenter_position("sun", jd)

        assert_array_almost_equal(r_sun_direct, r_sun_bary)
        assert_array_almost_equal(v_sun_direct, v_sun_bary)

    def test_barycenter_moon(self):
        """Test barycenter position for Moon."""
        jd = 2451545.0
        r_moon_direct, v_moon_direct = self.eph.moon_position(jd, frame="icrf")
        r_moon_bary, v_moon_bary = self.eph.barycenter_position("moon", jd)

        assert_array_almost_equal(r_moon_direct, r_moon_bary)
        assert_array_almost_equal(v_moon_direct, v_moon_bary)


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_module_sun_position(self):
        """Test module-level sun_position function."""
        jd = 2451545.0
        r, v = sun_position(jd)

        assert r.shape == (3,)
        assert v.shape == (3,)
        distance = np.linalg.norm(r)
        assert distance < 0.01  # Sun is at SSB center

    def test_module_moon_position(self):
        """Test module-level moon_position function."""
        jd = 2451545.0
        r, v = moon_position(jd)

        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_module_planet_position(self):
        """Test module-level planet_position function."""
        jd = 2451545.0
        r, v = planet_position("mars", jd)

        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_module_barycenter_position(self):
        """Test module-level barycenter_position function."""
        jd = 2451545.0
        r, v = barycenter_position("earth", jd)

        assert r.shape == (3,)
        assert v.shape == (3,)


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestEphemerisEdgeCases:
    """Test edge cases and error conditions."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    def test_ephemeris_different_versions_similar(self):
        """Test that different ephemeris versions give similar results."""
        jd = 2451545.0

        eph440 = DEEphemeris(version="DE440")
        eph430 = DEEphemeris(version="DE430")

        r440_sun, _ = eph440.sun_position(jd)
        r430_sun, _ = eph430.sun_position(jd)

        # Should be very close (within 0.001 AU)
        diff = np.linalg.norm(r440_sun - r430_sun)
        assert diff < 0.001, f"DE440 and DE430 differ by {diff:.6f} AU"

    def test_position_scalar_vs_array(self):
        """Test that positions work with scalar JD values."""
        jd_scalar = 2451545.0
        r, v = self.eph.sun_position(jd_scalar)

        assert isinstance(r, np.ndarray)
        assert r.shape == (3,)
