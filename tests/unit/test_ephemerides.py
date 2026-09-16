"""
Tests for JPL Ephemerides module.

This test suite validates high-precision ephemeris calculations against
reference values from established sources (SOFA, Astropy).
"""

from pathlib import Path

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


# DEEphemeris downloads de4NN.bsp from naif.jpl.nasa.gov on first use, into
# ~/.jplephem/. That made the unit suite depend on a 114 MB network fetch:
# when NAIF was unreachable on 2026-09-14, 24 tests here failed and took CI
# red on the v2.11.0 release commit (run 34799347373).
#
# The kernel is now a prerequisite rather than something a test run acquires.
# Absent, these tests skip; set PYTCL_REQUIRE_EPHEMERIS=1 to make that
# absence an error instead, the way PYTCL_REQUIRE_MLX and
# PYTCL_REQUIRE_CUPY already work for the GPU layers. CI caches the
# directory, so the fetch happens once per cache lifetime, not once per run.
#
# The path is the one DEEphemeris itself uses; it is hardcoded there, so
# there is nothing to configure and nothing to keep in sync beyond this.
_KERNEL_NAMES = ("de440.bsp", "de430.bsp")


def _kernel_available() -> bool:
    """True when a DE kernel is already on disk, so no test needs the network."""
    if not HAS_JPLEPHEM:
        return False
    cache = Path.home() / ".jplephem"
    return any((cache / name).exists() for name in _KERNEL_NAMES)


requires_kernel = pytest.mark.skipif(
    not _kernel_available(),
    reason=(
        "no DE kernel cached in ~/.jplephem; these tests will not download one. "
        "Set PYTCL_REQUIRE_EPHEMERIS=1 to make this an error instead of a skip."
    ),
)


@requires_kernel
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

    def test_ephemeris_lazy_loading(self):
        """Test that kernel is lazily loaded."""
        eph = DEEphemeris()
        # Kernel shouldn't be loaded yet
        assert eph._kernel is None
        # Access kernel
        _ = eph.kernel
        # Now it should be loaded
        assert eph._kernel is not None


@requires_kernel
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


@requires_kernel
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

        # True perigee-apogee range (356500-406700 km). The old bound
        # (370000-400000) bracketed the EMB->Moon defect, not the Earth->Moon
        # distance; see test_ephemerides_oracle.py for the reference check.
        assert 350000 < distance_km < 410000, (
            f"Moon distance {distance_km:.0f} km is unexpected"
        )

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


@requires_kernel
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
            assert 0.75 * expected_a < distance < 1.25 * expected_a, (
                f"{planet.capitalize()} distance {distance:.3f} AU != {expected_a:.3f} AU ± 25%"
            )

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


@requires_kernel
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


@requires_kernel
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


@requires_kernel
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


@pytest.mark.skipif(not HAS_JPLEPHEM, reason="jplephem not installed")
class TestDEEphemerisWithoutKernel:
    """Construction and cache behaviour that never touches a kernel file.

    Separated from the kernel-gated classes so these still run on a machine
    with jplephem installed but no cached ephemeris.
    """

    def test_ephemeris_version_de430(self):
        """Selecting DE430 records the version without loading the kernel."""
        eph = DEEphemeris(version="DE430")
        assert eph.version == "DE430"
        assert eph._kernel is None

    def test_ephemeris_invalid_version(self):
        """An unknown version raises before any file access."""
        with pytest.raises(ValueError, match="must be one of"):
            DEEphemeris(version="INVALID")

    def test_clear_cache(self):
        """clear_cache empties the result cache."""
        eph = DEEphemeris()
        eph._cache["test"] = "value"
        assert len(eph._cache) > 0
        eph.clear_cache()
        assert len(eph._cache) == 0
