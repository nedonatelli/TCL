"""Time and astronomy helpers consolidated to one body each (v2.12.0 Task 6.3).

Deliberate value changes, each pinned against an external oracle (astropy)
or a published constant:

* Tide GMST is now IAU 1982 (``reference_frames.gmst_iau82``). The
  v2.11.1 tide polynomial lacked the T^2 term; its error against astropy's
  IAU1982 model was -1.8e-5" at J2000, -0.0436" at MJD 58000, -0.0747" at
  60000 and -0.0937" at 61000.7.
* ``EPSILON_J2000`` is the IAU 1976 84381.448" (0.40909280422232897 rad);
  v2.11.1's 0.4090910179 was 0.3685" smaller. At JD 2460311 the ecliptic
  vectors from ``moon_position``/``planet_position`` (SSB-relative, ICRF
  rotated by the obliquity) move by 258 km / 0.362" for the Moon and
  389 km / 0.361" for Mars.
"""

import warnings

import numpy as np
import pytest

from pytcl.astronomical import orbital_mechanics, reference_frames, sgp4, special_orbits
from pytcl.astronomical.ephemerides import (
    EPSILON_J2000,
    moon_position,
    planet_position,
)
from pytcl.core.exceptions import PytclDeprecationWarning
from pytcl.gravity import tides
from tests.validation._v2111 import shipped_module

ARCSEC = np.pi / (180.0 * 3600.0)
MJDS = (51544.5, 58000.0, 60000.0, 61000.7)
OLD_EPSILON = 0.4090910179


@pytest.fixture(scope="module")
def old_tides():
    return shipped_module("pytcl/gravity/tides.py")


def _wrapped_arcsec(a, b):
    return ((np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi) / ARCSEC


def _astropy_gmst(jd):
    time = pytest.importorskip("astropy.time")
    return (
        time.Time(jd, format="jd", scale="ut1")
        .sidereal_time("mean", "greenwich", model="IAU1982")
        .rad
    )


@pytest.mark.parametrize(
    ("mjd", "tol"),
    [(51544.5, 1e-6), (58000.0, 1e-6), (60000.0, 1e-6), (61000.7, 2e-5)],
)
def test_tide_gmst_matches_astropy_iau82(mjd, tol):
    # 61000.7 is looser: gmst_iau82 splits a float JD (ulp 4.7e-10 d), which
    # costs 1.46e-5" there; the MJD input is not the cause.
    err = _wrapped_arcsec(tides._gmst_radians(mjd), _astropy_gmst(mjd + 2400000.5))
    assert abs(float(err)) < tol


@pytest.mark.parametrize("mjd", MJDS)
def test_v2111_tide_gmst_error_was_the_missing_t2_term(old_tides, mjd):
    got = _wrapped_arcsec(old_tides._gmst_radians(mjd), _astropy_gmst(mjd + 2400000.5))
    expected = {
        51544.5: -1.8e-5,
        58000.0: -0.0436,
        60000.0: -0.0747,
        61000.7: -0.0937,
    }[mjd]
    assert float(got) == pytest.approx(expected, abs=2e-4)


def test_tide_gmst_scalar_and_array_shapes_match_v2111(old_tides):
    mjds = np.array(MJDS)
    assert np.shape(tides._gmst_radians(51544.5)) == np.shape(
        old_tides._gmst_radians(51544.5)
    )
    new, old = tides._gmst_radians(mjds), old_tides._gmst_radians(mjds)
    assert new.shape == old.shape == (4,)
    assert new.dtype == old.dtype
    for mjd, value in zip(mjds, new, strict=True):
        assert value == tides._gmst_radians(float(mjd))
    assert np.all(np.abs(_wrapped_arcsec(new, old)) < 0.1)


def test_sgp4_gstime_stays_the_spec_gmst_within_documented_bound():
    rng = np.random.default_rng(2123)
    for jd in 2451545.0 + rng.uniform(-20000, 25000, 100):
        err = _wrapped_arcsec(sgp4._gstime(jd), _astropy_gmst(jd))
        assert abs(float(err)) < 1.5e-5


def test_epsilon_j2000_is_iau1976():
    assert EPSILON_J2000 == pytest.approx(0.40909280422232897, abs=0, rel=1e-15)
    assert EPSILON_J2000 == pytest.approx(84381.448 * ARCSEC, abs=1.2e-16, rel=0)
    assert abs(
        EPSILON_J2000 - reference_frames.mean_obliquity_iau80(2451545.0)
    ) <= np.spacing(EPSILON_J2000)
    assert (OLD_EPSILON - EPSILON_J2000) / ARCSEC == pytest.approx(-0.3685, abs=1e-4)


def test_tides_obliquity_is_the_shared_constant():
    assert tides._OBLIQUITY_J2000 is EPSILON_J2000


def test_solid_earth_tide_phase_shift_is_small_and_shape_preserved(old_tides):
    lat, lon = np.radians(40.0), np.radians(-105.0)
    for mjd in MJDS:
        new = tides.solid_earth_tide_displacement(lat, lon, mjd)
        old = old_tides.solid_earth_tide_displacement(lat, lon, mjd)
        assert type(new).__name__ == type(old).__name__
        for field in new._fields:
            assert np.shape(getattr(new, field)) == np.shape(getattr(old, field))
            assert abs(getattr(new, field) - getattr(old, field)) < 2e-4
        gnew = tides.solid_earth_tide_gravity(lat, lon, mjd)
        gold = old_tides.solid_earth_tide_gravity(lat, lon, mjd)
        for field in gnew._fields:
            assert abs(getattr(gnew, field) - getattr(gold, field)) < 1e-6


def test_julian_century_users_in_tides_unchanged_to_rounding(old_tides):
    for mjd in MJDS:
        for name in ("moon_position_approximate", "sun_position_approximate"):
            new = getattr(tides, name)(mjd)
            old = getattr(old_tides, name)(mjd)
            np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-11)
    amp = np.array([1e-3, 5e-4, 2e-4, 1e-4, 3e-3, 2e-3, 1e-3, 5e-4]) * np.ones((3, 1))
    ph = np.linspace(0.1, 3.0, 24).reshape(3, 8)
    for mjd in MJDS:
        new = tides.ocean_tide_loading_displacement(mjd, amp, ph)
        old = old_tides.ocean_tide_loading_displacement(mjd, amp, ph)
        for field in new._fields:
            np.testing.assert_allclose(
                getattr(new, field), getattr(old, field), rtol=0, atol=1e-11
            )


@pytest.mark.parametrize("mjd", [51544.5, 58000.0])
def test_gravity_julian_centuries_matches_v2111_and_warns(old_tides, mjd):
    with pytest.warns(PytclDeprecationWarning, match=r"mjd \+ 2400000\.5"):
        got = tides.julian_centuries_j2000(mjd)
    assert got == old_tides.julian_centuries_j2000(mjd)


def test_gravity_julian_centuries_array_input_matches_v2111(old_tides):
    mjds = np.array(MJDS)
    with pytest.warns(PytclDeprecationWarning):
        got = tides.julian_centuries_j2000(mjds)
    np.testing.assert_array_equal(got, old_tides.julian_centuries_j2000(mjds))


def test_gravity_package_export_is_the_deprecated_function():
    import pytcl.gravity as gravity

    with pytest.warns(PytclDeprecationWarning):
        gravity.julian_centuries_j2000(51544.5)


def test_astronomical_julian_centuries_takes_jd_not_mjd():
    assert reference_frames.julian_centuries_j2000(2451545.0) == 0.0
    assert reference_frames.julian_centuries_j2000(51544.5) == pytest.approx(
        -65.7084, abs=1e-4
    )


def test_tide_internals_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tides.moon_position_approximate(58000.0)
        tides.sun_position_approximate(58000.0)
        tides.solid_earth_tide_displacement(0.5, 0.5, 58000.0)


def test_escape_velocity_at_radius_matches_v2111():
    old = shipped_module("pytcl/astronomical/special_orbits.py")
    for mu, r in [(398600.4418, 6371.0), (132712440041.94, 1.496e8), (4902.8, 1737.4)]:
        assert special_orbits.escape_velocity_at_radius(mu, r) == (
            old.escape_velocity_at_radius(mu, r)
        )
    assert special_orbits.escape_velocity_at_radius(398600.4418, 6371.0) == (
        pytest.approx(11.186135691389076, rel=1e-14)
    )


def test_escape_velocity_argument_orders_are_reversed():
    mu, r = 398600.4418, 6371.0
    assert special_orbits.escape_velocity_at_radius(mu, r) == (
        orbital_mechanics.escape_velocity(r, mu)
    )
    assert "(mu, r)" in special_orbits.escape_velocity_at_radius.__doc__
    assert "(r, mu)" in orbital_mechanics.escape_velocity.__doc__


def test_escape_velocity_at_radius_array_input_matches_v2111():
    old = shipped_module("pytcl/astronomical/special_orbits.py")
    r = np.array([6371.0, 7000.0, 42164.0])
    np.testing.assert_array_equal(
        special_orbits.escape_velocity_at_radius(398600.4418, r),
        old.escape_velocity_at_radius(398600.4418, r),
    )


def _kernel_present():
    import os

    return os.path.exists(os.path.expanduser("~/.jplephem/de440.bsp"))


@pytest.mark.skipif(not _kernel_present(), reason="DE440 kernel not cached")
def test_ephemeris_ecliptic_shift_is_the_obliquity_change():
    jd = 2460311.0
    pytest.importorskip("jplephem")
    cases = (
        ("moon", moon_position(jd, frame="icrf"), moon_position(jd, frame="ecliptic")),
        (
            "mars",
            planet_position("mars", jd, frame="icrf"),
            planet_position("mars", jd, frame="ecliptic"),
        ),
    )
    expected = {"moon": (258.05, 0.3619), "mars": (388.82, 0.3611)}
    for name, (icrf, _), (ecl, ecl_v) in cases:
        old = reference_frames.equatorial_to_ecliptic(icrf, OLD_EPSILON)
        shift = np.linalg.norm(ecl - old)
        km = shift * 149597870.7
        arcsec = shift / np.linalg.norm(icrf) / ARCSEC
        assert km == pytest.approx(expected[name][0], abs=0.05)
        assert arcsec == pytest.approx(expected[name][1], abs=2e-4)
        np.testing.assert_allclose(
            ecl, reference_frames.equatorial_to_ecliptic(icrf, EPSILON_J2000), rtol=0
        )
