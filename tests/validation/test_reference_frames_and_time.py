"""IAU 1976 precession angles and GPS-to-UTC, against ERFA and astropy.

Two exported functions that no test reached (gh-49). Both are small, both sit
underneath things that are tested, and both are the kind of routine where an
error is invisible downstream: a precession angle that is wrong in the third
polynomial term still produces a valid rotation matrix, and a leap-second count
that is one too low still produces a plausible date.

`precession_angles_iau76` is checked against ERFA's `prec76`, the reference C
implementation of the same Lieske et al. (1977) model, reached through the
`pyerfa` wheel that ships with astropy. `gps_to_utc` is checked against
astropy's own GPS time scale, which carries the IERS leap-second table.

The angles agree with ERFA to the last bit or two of a double, so the tolerance
here is tight on purpose. A loose tolerance would let a genuine coefficient typo
through: the cubic terms contribute only milliarcseconds over a century, and
that is precisely the magnitude a wrong digit would move them by.
"""

import numpy as np
import pytest

from pytcl.astronomical.time_systems import JD_GPS_EPOCH, gps_to_utc

erfa = pytest.importorskip("erfa", reason="pyerfa provides the IAU reference model")
astropy_time = pytest.importorskip("astropy.time", reason="astropy provides GPS scale")

from pytcl.astronomical.reference_frames import precession_angles_iau76  # noqa: E402

J2000_JD = 2451545.0
JULIAN_CENTURY_DAYS = 36525.0


class TestPrecessionAnglesIau76:
    """Lieske et al. (1977) precession angles, against ERFA's `prec76`."""

    # Julian centuries from J2000.0. Includes negative epochs, because the
    # odd-power terms change sign there and a transcription that dropped a
    # minus would still pass a forward-only check.
    CENTURIES = [-1.0, -0.5, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 2.0]

    @pytest.mark.parametrize("T", CENTURIES)
    def test_angles_match_the_erfa_reference_implementation(self, T):
        """All three angles, to within a few ulp of ERFA.

        Note the ordering difference: ERFA returns (zeta, z, theta) while this
        library returns (zeta, theta, z). Getting that wrong would swap two
        angles that are numerically close -- zeta and z differ by under 2% at
        one century -- so a loose tolerance really would hide it.
        """
        zeta_ref, z_ref, theta_ref = erfa.prec76(
            J2000_JD, 0.0, J2000_JD, T * JULIAN_CENTURY_DAYS
        )
        zeta, theta, z = precession_angles_iau76(T)

        np.testing.assert_allclose(zeta, zeta_ref, rtol=1e-14, atol=1e-18)
        np.testing.assert_allclose(theta, theta_ref, rtol=1e-14, atol=1e-18)
        np.testing.assert_allclose(z, z_ref, rtol=1e-14, atol=1e-18)

    def test_the_returned_order_is_zeta_theta_z(self):
        """Pin the tuple order the docstring promises.

        zeta and z share the same linear term and separate only in the
        quadratic, so at small T they are nearly equal and a swap between them
        is invisible. At a full century they differ by 0.79 arcseconds, which
        is enough to tell them apart.

        The expected values are the Lieske et al. (1977) polynomials evaluated
        at T = 1, written out so this test states the published model rather
        than re-deriving it from the code under test:

            zeta  = 2306.2181 + 0.30188 + 0.017998 = 2306.537978
            theta = 2004.3109 - 0.42665 - 0.041833 = 2003.842417
            z     = 2306.2181 + 1.09468 + 0.018203 = 2307.330983
        """
        zeta, theta, z = precession_angles_iau76(1.0)
        arcsec = np.degrees(1.0) * 3600.0

        assert zeta * arcsec == pytest.approx(2306.537978, abs=1e-6)
        assert theta * arcsec == pytest.approx(2003.842417, abs=1e-6)
        assert z * arcsec == pytest.approx(2307.330983, abs=1e-6)
        assert z > zeta, "z exceeds zeta at T=1 because its quadratic term is larger"

    def test_all_angles_vanish_at_the_j2000_epoch(self):
        """The model is a polynomial in T with no constant term."""
        assert precession_angles_iau76(0.0) == (0.0, 0.0, 0.0)

    def test_the_angles_reverse_sign_before_the_epoch(self):
        """The linear term dominates, so a century back mirrors a century on.

        Not exactly, because the even-power terms do not change sign; the check
        is that the sign flips and the magnitude is close.
        """
        forward = precession_angles_iau76(1.0)
        backward = precession_angles_iau76(-1.0)
        for ahead, behind in zip(forward, backward):
            assert behind < 0.0 < ahead
            assert abs(behind) == pytest.approx(abs(ahead), rel=1e-3)


class TestGpsToUtc:
    """GPS to UTC, against astropy's IERS-backed leap-second table."""

    # Dates chosen to straddle the two most recent leap seconds: one was
    # inserted at the end of June 2015 and another at the end of December 2016.
    # A conversion that hard-coded a single offset passes at one date and fails
    # at the others.
    DATES = [
        ("2010-01-01T00:00:00", 34),
        ("2015-01-01T00:00:00", 35),
        ("2015-07-01T12:00:00", 36),
        ("2017-01-01T00:00:00", 37),
        ("2020-01-01T00:00:00", 37),
        ("2026-01-01T00:00:00", 37),
    ]
    IDS = [iso[:10] for iso, _ in DATES]

    # A Julian Date near 2020 is about 2.46e6, so one ulp of a float64 JD is
    # 4.0e-5 s. Differencing two such dates cannot resolve better than that,
    # and a tolerance tighter than one ulp would be passing by luck rather than
    # by correctness. This is still four orders of magnitude below the one
    # second that any real leap-second error would move the answer by.
    JD_RESOLUTION_SECONDS = 1e-4

    @staticmethod
    def _jd_gps(iso: str) -> float:
        """The GPS-scale Julian Date astropy assigns to a UTC instant."""
        return astropy_time.Time(iso, scale="utc").gps / 86400.0 + JD_GPS_EPOCH

    @pytest.mark.parametrize("iso,expected_leap", DATES, ids=IDS)
    def test_converted_utc_matches_astropy(self, iso, expected_leap):
        """The returned UTC must land on the instant astropy says it is."""
        reference = astropy_time.Time(iso, scale="utc")
        jd_utc, _ = gps_to_utc(self._jd_gps(iso))

        error_seconds = (jd_utc - reference.jd) * 86400.0
        assert abs(error_seconds) < self.JD_RESOLUTION_SECONDS, (
            f"{iso}: converted UTC is off by {error_seconds:+.9f} s"
        )

    @pytest.mark.parametrize("iso,expected_leap", DATES, ids=IDS)
    def test_the_reported_leap_second_count_is_the_published_one(
        self, iso, expected_leap
    ):
        """TAI-UTC at each date, from the IERS table.

        This is the second return value, and a caller can use it without ever
        looking at the first, so it needs its own assertion.
        """
        _, leap_seconds = gps_to_utc(self._jd_gps(iso))
        assert leap_seconds == expected_leap, (
            f"{iso}: reported {leap_seconds} leap seconds, published value is "
            f"{expected_leap}"
        )

    def test_gps_runs_ahead_of_utc_by_the_leap_count_less_nineteen(self):
        """GPS-UTC = (TAI-UTC) - 19 s, because GPS was set to TAI-19 in 1980.

        Checking the relation rather than a stored offset means this still
        holds after the next leap second, and fails if the 19 s constant is
        ever lost.
        """
        jd_gps = self._jd_gps("2020-01-01T00:00:00")
        jd_utc, leap_seconds = gps_to_utc(jd_gps)

        offset_seconds = (jd_gps - jd_utc) * 86400.0
        assert offset_seconds == pytest.approx(
            leap_seconds - 19, abs=self.JD_RESOLUTION_SECONDS
        )

    def test_a_leap_second_makes_the_conversion_step(self):
        """Across an insertion the UTC offset must change by exactly one second.

        The June 2015 leap second is the boundary; sampling either side of it
        catches a table that is interpolated instead of stepped.
        """
        before = gps_to_utc(self._jd_gps("2015-06-30T00:00:00"))[1]
        after = gps_to_utc(self._jd_gps("2015-07-01T00:00:00"))[1]
        assert after - before == 1, (
            f"leap-second count went {before} -> {after} across 2015-06-30; a "
            f"leap second is a step of exactly one"
        )
