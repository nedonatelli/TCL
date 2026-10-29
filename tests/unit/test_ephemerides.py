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


AU_KM = 149597870.7
_DE440_PATH = Path.home() / ".jplephem" / "de440.bsp"
_PLANET_IDS = {
    "mercury": 1,
    "venus": 2,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
}


def _spk_state(jd, *chain):
    """Position (AU) and velocity (AU/day) summed straight from DE440 SPK segments.

    ``chain`` is a sequence of ``(center, target, sign)``. This bypasses
    ``DEEphemeris`` entirely, so it is an oracle for its frame and unit
    handling.
    """
    from jplephem.spk import SPK

    kernel = SPK.open(str(_DE440_PATH))
    try:
        pos = np.zeros(3)
        vel = np.zeros(3)
        for center, target, sign in chain:
            p, v = kernel[center, target].compute_and_differentiate(jd)
            pos += sign * p
            vel += sign * v
    finally:
        kernel.close()
    return pos / AU_KM, vel / AU_KM


requires_de440 = pytest.mark.skipif(
    not (HAS_JPLEPHEM and _DE440_PATH.exists()),
    reason="DE440 kernel not cached at ~/.jplephem/de440.bsp",
)

_SUN = ((0, 10, 1.0),)
_MOON_ICRF = ((0, 3, 1.0), (3, 301, 1.0))
_MOON_GEOCENTRIC = ((3, 301, 1.0), (3, 399, -1.0))
_EARTH = ((0, 3, 1.0), (3, 399, 1.0))


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

    @requires_de440
    @pytest.mark.parametrize("jd", [2451545.0, 2451545.0 + 180, 2460311.0])
    def test_sun_state_matches_spk_segment(self, jd):
        """Sun state is the SSB->Sun segment, in AU and AU/day."""
        r, v = self.eph.sun_position(jd)
        r_ref, v_ref = _spk_state(jd, *_SUN)

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

    @requires_de440
    def test_sun_icrf_frame_is_default(self):
        jd = 2451545.0
        r_default, v_default = self.eph.sun_position(jd)
        r_icrf, v_icrf = self.eph.sun_position(jd, frame="icrf")
        r_ref, v_ref = _spk_state(jd, *_SUN)

        assert r_icrf == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v_icrf == pytest.approx(v_ref, rel=1e-12, abs=0)
        assert r_default == pytest.approx(r_icrf, rel=1e-15, abs=0)


@requires_kernel
class TestMoonPosition:
    """Test Moon position calculations."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    @requires_de440
    def test_moon_icrf_is_ssb_to_moon(self):
        """ICRF Moon is SSB->EMB + EMB->Moon, about 1 AU from the barycentre.

        The earlier bound ``distance < 1.0`` rejected the correct value.
        """
        jd = 2451545.0
        r, v = self.eph.moon_position(jd, frame="icrf")
        r_ref, v_ref = _spk_state(jd, *_MOON_ICRF)

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

    @requires_de440
    def test_moon_earth_centered_j2000(self):
        jd = 2451545.0
        r, v = self.eph.moon_position(jd, frame="earth_centered")
        r_ref, v_ref = _spk_state(jd, *_MOON_GEOCENTRIC)

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

    @requires_de440
    def test_moon_distance_jd_2460311(self):
        """Earth-Moon distance of 404896.86 km at JD 2460311.0 (DE440).

        The reference distance comes from the DE440 segments directly
        (task 1.1 of v2.11.1, tests/validation/test_ephemerides_oracle.py).
        """
        r, _ = self.eph.moon_position(2460311.0, frame="earth_centered")

        assert np.linalg.norm(r) * AU_KM == pytest.approx(404896.86, rel=1e-7, abs=0)

    @requires_de440
    def test_moon_frames_are_mutually_consistent(self):
        """Moon(ICRF) - Earth(SSB) = Moon(geocentric): the segments cancel."""
        jd = 2451545.0
        r_icrf, v_icrf = self.eph.moon_position(jd, frame="icrf")
        r_geo, v_geo = self.eph.moon_position(jd, frame="earth_centered")
        r_earth, v_earth = self.eph.planet_position("earth", jd)

        assert r_icrf - r_earth == pytest.approx(r_geo, rel=1e-10, abs=0)
        assert v_icrf - v_earth == pytest.approx(v_geo, rel=1e-10, abs=0)


@requires_kernel
class TestPlanetPosition:
    """Test planet position calculations."""

    @classmethod
    def setup_class(cls):
        """Set up ephemeris for tests."""
        cls.eph = DEEphemeris(version="DE440")

    @requires_de440
    @pytest.mark.parametrize("planet", list(_PLANET_IDS))
    def test_planet_state_matches_spk_segment(self, planet):
        jd = 2451545.0
        r, v = self.eph.planet_position(planet, jd)
        r_ref, v_ref = _spk_state(jd, (0, _PLANET_IDS[planet], 1.0))

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

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

    @requires_de440
    def test_module_sun_position(self):
        r, v = sun_position(2451545.0)
        r_ref, v_ref = _spk_state(2451545.0, *_SUN)

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

    @requires_de440
    def test_module_moon_position(self):
        r, v = moon_position(2451545.0)
        r_ref, v_ref = _spk_state(2451545.0, *_MOON_ICRF)

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

    @requires_de440
    def test_module_planet_position(self):
        r, v = planet_position("mars", 2451545.0)
        r_ref, v_ref = _spk_state(2451545.0, (0, 4, 1.0))

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)

    @requires_de440
    def test_module_barycenter_position_earth(self):
        r, v = barycenter_position("earth", 2451545.0)
        r_ref, v_ref = _spk_state(2451545.0, *_EARTH)

        assert r == pytest.approx(r_ref, rel=1e-12, abs=0)
        assert v == pytest.approx(v_ref, rel=1e-12, abs=0)


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
