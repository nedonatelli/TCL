"""
Reference-oracle tests for ``pytcl.astronomical.ephemerides``.

Compares ``moon_position``/``planet_position`` directly against the SPK
segments they wrap (and, transitively, astropy's own DE440 read) to catch
geocentric/barycentric frame confusion and wrong NAIF body ids -- the kind
of defect a shape- or finiteness-only test cannot see.

Requires jplephem and a DE440 kernel cached at ``~/.jplephem/de440.bsp``;
skips (never downloads) when either is absent.
"""

import os

import numpy as np
import pytest

from pytcl.astronomical.ephemerides import moon_position, planet_position

AU_KM = 149597870.7

HAS_JPLEPHEM = True
try:
    from jplephem.spk import SPK
except ImportError:
    HAS_JPLEPHEM = False


def _kernel_path(version: str = "DE440") -> str:
    """Local path DEEphemeris resolves for ``version`` (see ephemerides.py)."""
    return os.path.expanduser(f"~/.jplephem/de{version[2:]}.bsp")


requires_kernel = pytest.mark.skipif(
    not (HAS_JPLEPHEM and os.path.exists(_kernel_path())),
    reason="jplephem not installed or DE440 kernel not cached at ~/.jplephem/de440.bsp",
)


@requires_kernel
def test_moon_earth_centered_is_geocentric_not_barycentric():
    """Earth->Moon, not EMB->Moon: the two differ by ~4900 km."""
    jd = 2460311.0
    pos, _ = moon_position(jd, frame="earth_centered")
    got_km = np.linalg.norm(pos) * AU_KM

    kernel = SPK.open(_kernel_path())
    emb_to_moon = kernel[3, 301].compute(jd)
    emb_to_earth = kernel[3, 399].compute(jd)
    expected_km = np.linalg.norm(emb_to_moon - emb_to_earth)

    assert expected_km == pytest.approx(404896.86, abs=1.0)
    assert got_km == pytest.approx(expected_km, rel=1e-12)


@requires_kernel
def test_planet_position_earth_is_earth_not_the_barycentre():
    """planet_position('earth') must chain 0->3->399, not stop at the EMB."""
    jd = 2460311.0
    earth, _ = planet_position("earth", jd)
    earth_km = np.array(earth) * AU_KM

    kernel = SPK.open(_kernel_path())
    expected_km = kernel[0, 3].compute(jd) + kernel[3, 399].compute(jd)
    assert np.allclose(earth_km, expected_km, rtol=1e-12)


@requires_kernel
def test_planet_position_moon_raises_value_error():
    """The docstring promises ValueError; it used to leak KeyError (0, 301)."""
    with pytest.raises(ValueError, match="moon"):
        planet_position("moon", 2460311.0)
