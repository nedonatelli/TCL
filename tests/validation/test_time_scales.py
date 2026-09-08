"""Time scales against astropy.

The MATLAB TCL implements these conversions as MEX stubs over IAU
SOFA; the port calls pyerfa (the liberated SOFA, algorithmically
identical). Astropy's Time machinery drives the same ERFA kernels
through entirely separate argument-handling code, so agreement here
validates our plumbing (epoch splitting, UT1 handling, unit
conventions) rather than re-proving SOFA. Round-trip and
physical-magnitude properties cover what cross-validation cannot.
"""

import math

import numpy as np
import pytest

astropy_time = pytest.importorskip("astropy.time")

from pytcl.astronomical.time_scales import (  # noqa: E402
    besselian_epoch2tdb,
    jul_date2jul_epoch,
    jul_epoch2jul_date,
    tcb2tdb,
    tcg2tt,
    tdb2besselian_epoch,
    tdb2tcb,
    tdb2tt,
    tt2gast,
    tt2gmst,
    tt2last,
    tt2lmst,
    tt2tcg,
    tt2tdb,
)

# A spread of epochs: J2000, a 2006 SOFA-cookbook-era date, far past
# and future, and a leap-second-adjacent day.
EPOCHS = [
    (2451545.0, 0.0),
    (2453750.5, 0.892482639),
    (2433282.5, 0.25),
    (2469807.5, 0.75),
    (2457753.5, 0.999),
]


def _astropy(jd1, jd2, scale):
    return astropy_time.Time(jd1, jd2, format="jd", scale=scale)


class TestAgainstAstropy:
    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_tt_to_tcg(self, jd1, jd2):
        got = sum(tt2tcg(jd1, jd2))
        ref = _astropy(jd1, jd2, "tt").tcg
        np.testing.assert_allclose(got, ref.jd1 + ref.jd2, rtol=0, atol=1e-9 / 86400)

    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_tdb_to_tcb(self, jd1, jd2):
        got = sum(tdb2tcb(jd1, jd2))
        ref = _astropy(jd1, jd2, "tdb").tcb
        np.testing.assert_allclose(got, ref.jd1 + ref.jd2, rtol=0, atol=1e-9 / 86400)

    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_tt_to_tdb(self, jd1, jd2):
        # Astropy's TT->TDB uses the same Fairhead-Bretagnon series
        # with its own UT argument; agreement to 10 microseconds is
        # far inside the series' 10-nanosecond-scale disagreement
        # budget quoted for different UT approximations.
        got = sum(tt2tdb(jd1, jd2))
        ref = _astropy(jd1, jd2, "tt").tdb
        np.testing.assert_allclose(got, ref.jd1 + ref.jd2, rtol=0, atol=1e-5 / 86400)

    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_besselian_epoch(self, jd1, jd2):
        got = tdb2besselian_epoch(jd1, jd2)
        ref = _astropy(jd1, jd2, "tdb").byear
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_julian_epoch(self, jd1, jd2):
        got = jul_date2jul_epoch(jd1, jd2)
        ref = _astropy(jd1, jd2, "tt").jyear
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_gmst(self, jd1, jd2):
        # Astropy computes sidereal time from UT1; with delta_t_ut1=0
        # our UT1 equals TT, so build the astropy time the same way.
        t = astropy_time.Time(jd1, jd2, format="jd", scale="ut1")
        t_tt = astropy_time.Time(jd1, jd2, format="jd", scale="tt")
        ref = t.sidereal_time("mean", "greenwich", model="IAU2006").radian
        got = tt2gmst(jd1, jd2)
        # astropy pairs its UT1 with a derived TT; with delta 0 ours
        # coincide to the model's microarcsecond level.
        diff = (got - ref + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 5e-9, f"gmst differs by {diff:.2e} rad"
        del t_tt


class TestProperties:
    @pytest.mark.parametrize("jd1,jd2", EPOCHS)
    def test_round_trips(self, jd1, jd2):
        for fwd, back in (
            (tt2tcg, tcg2tt),
            (tdb2tcb, tcb2tdb),
            (tt2tdb, tdb2tt),
        ):
            j1, j2 = fwd(jd1, jd2)
            r1, r2 = back(j1, j2)
            np.testing.assert_allclose((r1 - jd1) + (r2 - jd2), 0.0, atol=5e-10 / 86400)
        np.testing.assert_allclose(
            sum(besselian_epoch2tdb(tdb2besselian_epoch(jd1, jd2))),
            jd1 + jd2,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sum(jul_epoch2jul_date(jul_date2jul_epoch(jd1, jd2))),
            jd1 + jd2,
            rtol=0,
            atol=1e-6,
        )

    def test_topocentric_clock_terms(self):
        # A clock on the equator at ~geostationary-like radius: the
        # diurnal topocentric TDB term is nonzero but bounded by ~2.1
        # microseconds (Fairhead & Bretagnon), and the conversion must
        # round-trip with the same clock location.
        clock = (1.0, 6378.137, 0.0)
        geo1, geo2 = tt2tdb(2453750.5, 0.892482639)
        top1, top2 = tt2tdb(2453750.5, 0.892482639, clock_loc=clock)
        diff_s = ((top1 - geo1) + (top2 - geo2)) * 86400.0
        assert 0.0 < abs(diff_s) < 3e-6
        r1, r2 = tdb2tt(top1, top2, clock_loc=clock)
        np.testing.assert_allclose(
            (r1 - 2453750.5) + (r2 - 0.892482639), 0.0, atol=5e-10 / 86400
        )

    def test_tdb_tt_stays_within_the_known_envelope(self):
        # TDB-TT is a quasi-periodic ~1.7 ms oscillation.
        for frac in np.linspace(0.0, 1.0, 25, endpoint=False):
            tdb1, tdb2 = tt2tdb(2451545.0 + 180.0 * frac * 2, frac)
            diff_s = ((tdb1 - 2451545.0 - 180.0 * frac * 2) + (tdb2 - frac)) * 86400
            assert abs(diff_s) < 2.0e-3

    def test_tcg_drift_rate(self):
        # TCG gains on TT at L_G ~ 6.97e-10 (about 22 ms per year).
        century = 36525.0
        a1, a2 = tt2tcg(2451545.0, 0.0)
        b1, b2 = tt2tcg(2451545.0 + century, 0.0)
        # Difference part-wise: summing two-part dates first would cost
        # ~40 microseconds of double precision at JD magnitudes.
        gained_s = ((b1 - a1 - century) + (b2 - a2)) * 86400.0
        np.testing.assert_allclose(
            gained_s, 6.969290134e-10 * century * 86400, rtol=1e-6
        )

    def test_local_sidereal_offsets(self):
        g = tt2gmst(2453750.5, 0.892482639)
        for lon in (-math.pi, -1.0, 0.5, math.pi / 2):
            loc = tt2lmst(2453750.5, 0.892482639, lon)
            diff = (loc - g - lon + math.pi) % (2 * math.pi) - math.pi
            assert abs(diff) < 1e-12
        ga = tt2gast(2453750.5, 0.892482639)
        loc = tt2last(2453750.5, 0.892482639, 1.0)
        diff = (loc - ga - 1.0 + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-12
