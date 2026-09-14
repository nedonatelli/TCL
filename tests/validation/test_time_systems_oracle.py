"""Leap-second lookup correctness, against astropy.

`pytcl.astronomical.time_systems` keeps its own leap-second table and
Julian-date arithmetic rather than delegating to pyerfa (that layer is
`pytcl.astronomical.time_scales`, covered by test_time_scales.py). Two
defects lived here: the leap-second count for a TAI/TT/GPS instant was
looked up on the input scale's calendar date rather than UTC's, which is
wrong for the ~37 seconds after every insertion instant (Task 1.2); and
dates before 1972 silently returned a 0 s offset instead of signaling that
the table does not cover them (Task 1.3).

Why these tests compare *calendar* representations, not raw JD floats
-------------------------------------------------------------------
SOFA's UTC "pseudo-Julian Date" (which astropy's `Time.jd` for a UTC scale
inherits) deliberately encodes a leap day's extra second by dividing the
day into 86401 parts instead of 86400 -- see sofa/src/taiutc.c's docstring.
That makes astropy's raw `.jd` for an instant *on* a leap day disagree with
this module's fixed-86400-seconds-per-day JD arithmetic by close to a full
second, even when the leap-second *count* this module picked is correct --
a representational difference the existing "Notes" caveat about
23:59:60 already gestures at. Comparing rounded calendar tuples (derived
from astropy's ISO string, which SOFA's d2dtf renders leap-second-aware)
isolates the thing under test -- did the lookup pick the right leap count
-- from that unrelated encoding difference. `cal_to_jd` is used to turn
both sides' calendar tuples into a leap-second count in a single
self-consistent way, again avoiding astropy's raw pseudo-JD.
"""

import re
import warnings

import pytest

astropy_time = pytest.importorskip("astropy.time")
from astropy.time import Time  # noqa: E402

from pytcl.astronomical.time_systems import (  # noqa: E402
    cal_to_jd,
    get_leap_seconds,
    gps_to_utc,
    jd_to_cal,
    tai_to_tt,
    tai_to_utc,
    tt_to_utc,
)

_ISO_RE = re.compile(r"(\d+)-(\d+)-(\d+)[T ](\d+):(\d+):([\d.]+)")


def _parse_iso(s):
    y, mo, d, h, mi, sec = _ISO_RE.match(s).groups()
    return int(y), int(mo), int(d), int(h), int(mi), float(sec)


def _expected_leap_and_utc_calendar(iso_tai):
    """The leap-second count and UTC calendar tuple astropy implies for a
    TAI instant, derived without ever reading astropy's raw pseudo-JD."""
    tai_tuple = _parse_iso(iso_tai)
    utc_tuple = _parse_iso(Time(iso_tai, scale="tai").utc.iso)
    leap = round((cal_to_jd(*tai_tuple) - cal_to_jd(*utc_tuple)) * 86400.0)
    return leap, utc_tuple


def _rounded_calendar(jd_utc):
    year, month, day, hour, minute, second = jd_to_cal(jd_utc)
    return (year, month, day, hour, minute, round(second))


# TAI instants spanning the 2017-01-01, 2015-07-01, 2012-07-01 and
# 1999-01-01 insertions. Each block's first two entries sit inside the
# window where the TAI calendar date has already rolled over but the
# correct (UTC) leap count has not; the last is the far edge of that
# window, where the new count becomes correct again. TAI 00:00:36 (the
# instant astropy itself resolves to the literal leap second, 23:59:60)
# is deliberately excluded: this module has no distinct representation
# for that second and folds it into the next one by design (gh-25, see
# tai_to_utc's docstring), so it is not a case with a single right answer.
BOUNDARY_CASES = [
    "2017-01-01T00:00:00",  # the instant the audit caught
    "2017-01-01T00:00:20",  # well inside the pre-rollover window
    "2017-01-01T00:00:37",  # first instant needing the new count
    "2015-07-01T00:00:00",
    "2015-07-01T00:00:20",
    "2012-07-01T00:00:00",
    "1999-01-01T00:00:00",
]

# Ordinary instants, at least a day clear of any insertion, where this
# module's fixed-86400-s-per-day JD arithmetic and astropy's raw JD agree
# to well under a millisecond.
JD_TOLERANCE_DAYS = 2e-3 / 86400.0
ORDINARY_ISO_TAI = [
    "2020-06-15T12:00:00",
    "1999-06-15T00:00:00",
    "2010-01-01T00:00:00",
]


class TestTaiToUtcMatchesAstropy:
    @pytest.mark.parametrize("iso_tai", BOUNDARY_CASES)
    def test_leap_count_and_calendar_at_leap_boundaries(self, iso_tai):
        expected_leap, expected_cal = _expected_leap_and_utc_calendar(iso_tai)
        jd_tai = Time(iso_tai, scale="tai").jd

        jd_utc, leap = tai_to_utc(jd_tai)

        assert leap == expected_leap
        assert _rounded_calendar(jd_utc) == expected_cal

    @pytest.mark.parametrize("iso_tai", ORDINARY_ISO_TAI)
    def test_matches_astropy_raw_jd_away_from_any_boundary(self, iso_tai):
        t = Time(iso_tai, scale="tai")
        jd_utc, _leap = tai_to_utc(t.jd)
        assert jd_utc == pytest.approx(t.utc.jd, abs=JD_TOLERANCE_DAYS)


class TestDelegatingEntryPointsShareTheFix:
    """tt_to_utc and gps_to_utc only inherit the fix by delegating to
    tai_to_utc (time_systems.py:517 and :605); exercise them directly so a
    future change that breaks the delegation is caught here."""

    @pytest.mark.parametrize("iso_tai", BOUNDARY_CASES)
    def test_tt_to_utc_at_leap_boundaries(self, iso_tai):
        expected_leap, expected_cal = _expected_leap_and_utc_calendar(iso_tai)
        jd_tai = Time(iso_tai, scale="tai").jd
        jd_tt = tai_to_tt(jd_tai)

        jd_utc, leap = tt_to_utc(jd_tt)

        assert leap == expected_leap
        assert _rounded_calendar(jd_utc) == expected_cal

    @pytest.mark.parametrize("iso_tai", BOUNDARY_CASES)
    def test_gps_to_utc_at_leap_boundaries(self, iso_tai):
        expected_leap, expected_cal = _expected_leap_and_utc_calendar(iso_tai)
        jd_tai = Time(iso_tai, scale="tai").jd
        jd_gps = jd_tai - 19.0 / 86400.0

        jd_utc, leap = gps_to_utc(jd_gps)

        assert leap == expected_leap
        assert _rounded_calendar(jd_utc) == expected_cal


class TestPre1972LeapLookup:
    """Dates before the 1972 start of the leap-second table."""

    def test_pre_1972_leap_lookup_warns_instead_of_returning_zero(self):
        with pytest.warns(UserWarning, match="before 1972"):
            get_leap_seconds(1970, 1, 1)

    def test_post_1972_lookup_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_leap_seconds(2026, 9, 14) == 37

    def test_the_1972_epoch_itself_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_leap_seconds(1972, 1, 1) == 10
