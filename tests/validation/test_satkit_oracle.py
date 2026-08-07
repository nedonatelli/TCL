"""Cross-validation against satkit (https://github.com/ssmichael1/satkit).

satkit is an independent Rust implementation of SGP4/SDP4 and IAU frame
transformations, giving a second oracle alongside the official ``sgp4``
package (test_astro_audit.py) with broader surface: it also covers the
TEME -> ITRF and TEME -> GCRF chains with its own EOP handling.

Measured agreement this suite locks in (ISS/Vanguard/GPS TLEs, 2024 epochs):

- SGP4/SDP4 TEME states: < 1e-4 m across near-Earth, deep-space, and
  high-eccentricity regimes (both are Vallado ports).
- teme_to_itrf, polar motion supplied from satkit's EOP: ~4e-3 m.
- teme_to_gcrf: ~8 m at LEO radius. pytcl uses the IAU-76/FK5 series
  without EOP nutation corrections (ddpsi/ddeps); satkit applies its full
  model, and ~0.2 arcsec of model difference is ~8 m at 6800 km.

Requires the ``validation`` dependency group: ``uv sync --group validation``.
Skips when satkit (or its bundled EOP/ephemeris data) is absent.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.astronomical.reference_frames import teme_to_gcrf, teme_to_itrf
from pytcl.astronomical.sgp4 import sgp4_propagate
from pytcl.astronomical.tle import parse_tle

satkit = pytest.importorskip("satkit")

pytestmark = pytest.mark.skipif(
    not satkit.utils.datafiles_exist(),
    reason="satkit EOP/ephemeris data files absent",
)

ARCSEC = np.pi / 180.0 / 3600.0

# Same synthetic TLEs (correct checksums) as test_sgp4.py: one per SGP4 regime.
TLES = {
    "iss-near-earth": (
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
        "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003",
    ),
    "vanguard-high-eccentricity": (
        "1 00005U 58002B   24001.00000000  .00000023  00000-0  28605-4 0  9997",
        "2 00005  34.2500  40.4560 1845947 262.5280  75.7980 10.84861856344568",
    ),
    "gps-deep-space": (
        "1 28474U 04045A   24001.00000000 -.00000037  00000-0  00000-0 0  9996",
        "2 28474  55.4330 143.3940 0056500 248.5560 110.9890  2.00563774144569",
    ),
}

TSINCE_MIN = [0.0, 30.0, 90.0, 720.0, 1440.0, 4320.0]


def _satkit_state(l1, l2, tsince):
    """satkit TEME position/velocity (km, km/s) at epoch + tsince minutes."""
    tle = satkit.TLE.from_lines([l1, l2])
    t = tle.epoch + satkit.duration.from_minutes(tsince)
    pos_m, vel_ms = satkit.sgp4(tle, t)
    return np.asarray(pos_m) / 1e3, np.asarray(vel_ms) / 1e3, t


class TestSGP4AgainstSatkit:
    """Both are Vallado ports; anything beyond float noise is a defect."""

    @pytest.mark.parametrize("name", TLES)
    @pytest.mark.parametrize("tsince", TSINCE_MIN)
    def test_teme_state_matches(self, name, tsince):
        l1, l2 = TLES[name]
        state = sgp4_propagate(parse_tle(l1, l2), tsince)
        r_ref, v_ref, _ = _satkit_state(l1, l2, tsince)

        # Observed < 1e-7 km; 1e-6 km (1 mm) allows platform float noise
        # while catching any real modeling divergence outright.
        assert_allclose(state.r, r_ref, atol=1e-6, rtol=0)
        assert_allclose(state.v, v_ref, atol=1e-9, rtol=0)


class TestFrameTransformsAgainstSatkit:
    """TEME -> ITRF / GCRF for an ISS state 90 minutes past epoch."""

    @pytest.fixture()
    def iss_state(self):
        l1, l2 = TLES["iss-near-earth"]
        state = sgp4_propagate(parse_tle(l1, l2), 90.0)
        r_ref, _, t = _satkit_state(l1, l2, 90.0)
        assert_allclose(state.r, r_ref, atol=1e-6, rtol=0)
        return state.r, t

    def test_teme_to_itrf(self, iss_state):
        r_teme, t = iss_state
        # Feed pytcl the same UT1 and polar motion satkit resolves from its
        # EOP tables, so the comparison isolates the rotation itself.
        jd_ut1 = t.as_jd(satkit.timescale.UT1)
        xp_as, yp_as = satkit.frametransform.earth_orientation_params(t)[1:3]

        r_itrf = teme_to_itrf(r_teme, jd_ut1, xp=xp_as * ARCSEC, yp=yp_as * ARCSEC)
        r_ref = np.asarray(satkit.frametransform.qteme2itrf(t) * (r_teme * 1e3)) / 1e3

        # Observed ~4e-6 km; a wrong rotation ordering or GMST error would
        # show up at km scale.
        assert np.linalg.norm(r_itrf - r_ref) < 1e-3  # km

    def test_teme_to_gcrf(self, iss_state):
        r_teme, t = iss_state
        jd_tt = t.as_jd(satkit.timescale.TT)

        r_gcrf = teme_to_gcrf(r_teme, jd_tt)
        r_ref = np.asarray(satkit.frametransform.qteme2gcrf(t) * (r_teme * 1e3)) / 1e3

        # Observed ~8e-3 km: IAU-76/FK5 without EOP nutation corrections vs
        # satkit's full model (~0.2 arcsec). 0.05 km still catches a wrong
        # equation-of-equinoxes sign or precession error (both >= km scale).
        assert np.linalg.norm(r_gcrf - r_ref) < 0.05  # km
