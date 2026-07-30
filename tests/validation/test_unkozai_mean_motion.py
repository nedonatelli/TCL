"""``unkozai_mean_motion`` against the official ``sgp4`` package.

A TLE carries a Kozai mean motion; SGP4 propagates a Brouwer mean motion, and
the conversion between them removes the secular J2 contribution. The recovered
value also decides whether the deep-space model applies, so an error here does
not merely perturb a propagation -- it can route an orbit through the wrong
model entirely. The deep-space path in this package previously had no physics at
all and produced 15-75 km/day errors for GEO and Molniya orbits (gh-13).

The function was exported and referenced by no test, which is what gh-47 is
about. The reference is `sgp4.io.twoline2rv`, which exposes ``no_unkozai``
computed by the canonical implementation. The C-accelerated ``Satrec`` does not
expose it, hence the pure-Python entry point.
"""

import numpy as np
import pytest

from pytcl.astronomical.sgp4 import unkozai_mean_motion

sgp4_io = pytest.importorskip("sgp4.io")
sgp4_gravity = pytest.importorskip("sgp4.earth_gravity")

# Real TLEs spanning the regimes the conversion has to handle. Inclination
# enters through (3 cos^2 i - 1), which changes sign near 54.7 degrees, so the
# set deliberately straddles it.
TLES = {
    "ISS (LEO, 51.6 deg, near-circular)": (
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
        "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003",
    ),
    "GEO (0.02 deg, geosynchronous)": (
        "1 25872U 99046A   24001.50000000 -.00000098  00000-0  00000-0 0  9992",
        "2 25872   0.0182 254.6431 0002269 195.0731 178.1197  1.00270298 88927",
    ),
    "Molniya (63.2 deg, e = 0.71)": (
        "1 25485U 98054A   24001.50000000  .00000200  00000-0  00000-0 0  9994",
        "2 25485  63.1706 156.0074 7148063 264.9797  15.4390  2.00612394185378",
    ),
    "GPS (55.5 deg, 12-hour non-resonant)": (
        "1 24876U 97035A   24001.50000000 -.00000029  00000-0  00000-0 0  9995",
        "2 24876  55.5460 106.1477 0102184  54.6647 306.4059  2.00565165194155",
    ),
}


def _reference(line1, line2):
    """(kozai, inclination, eccentricity, un-Kozai'd) from the sgp4 package."""
    sat = sgp4_io.twoline2rv(line1, line2, sgp4_gravity.wgs72)
    return sat.no_kozai, sat.inclo, sat.ecco, sat.no_unkozai


@pytest.mark.parametrize("name", sorted(TLES), ids=lambda n: n.split()[0])
def test_matches_the_official_sgp4_implementation(name):
    """The canonical implementation is the reference, across four regimes."""
    kozai, inclination, eccentricity, expected = _reference(*TLES[name])
    got = unkozai_mean_motion(kozai, inclination, eccentricity)
    assert got == pytest.approx(expected, rel=1e-12), (
        f"{name}: un-Kozai'd mean motion {got!r} does not match the sgp4 "
        f"package's {expected!r}"
    )


@pytest.mark.parametrize("name", sorted(TLES), ids=lambda n: n.split()[0])
def test_correction_follows_the_sign_of_the_j2_secular_term(name):
    """Direction of the correction is set by (3 cos^2 i - 1), not fixed.

    Below the critical inclination (~54.736 deg) the term is positive and the
    recovered mean motion is smaller than the TLE's; above it the term is
    negative and the recovered value is larger. Asserting "always smaller" --
    which the docstring's LEO example alone would suggest -- is wrong for the
    Molniya and GPS cases here, both of which sit above the critical value.
    """
    kozai, inclination, eccentricity, _ = _reference(*TLES[name])
    got = unkozai_mean_motion(kozai, inclination, eccentricity)

    j2_term = 3.0 * np.cos(inclination) ** 2 - 1.0
    if j2_term > 0:
        assert got < kozai, f"{name}: i below critical, expected a decrease"
    else:
        assert got > kozai, f"{name}: i above critical, expected an increase"


def test_correction_scales_with_inclination_through_the_j2_term():
    """The J2 secular term enters as (3 cos^2 i - 1), and the output shows it.

    An equatorial orbit gets the largest correction; near the critical
    inclination the term passes through zero and the correction nearly
    vanishes. A conversion that ignored inclination would give a flat profile.
    """
    kozai, eccentricity = 0.0676, 0.0007
    critical = np.arccos(np.sqrt(1.0 / 3.0))  # ~54.736 degrees

    corrections = {
        inc: kozai - unkozai_mean_motion(kozai, inc, eccentricity)
        for inc in (0.0, np.radians(30.0), critical, np.radians(80.0))
    }

    assert corrections[0.0] > corrections[np.radians(30.0)] > 0
    # at the critical inclination the J2 secular term vanishes
    assert abs(corrections[critical]) < 1e-9
    # beyond it the term reverses sign, so the correction does too
    assert corrections[np.radians(80.0)] < 0


def test_zero_eccentricity_is_handled():
    """A circular orbit is a valid input; (1 - e^2) must not be special-cased."""
    result = unkozai_mean_motion(0.0676, 0.9013, 0.0)
    assert np.isfinite(result)
    assert result < 0.0676
