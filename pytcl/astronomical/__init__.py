"""
Astronomical calculations for target tracking.

This module provides time system conversions and astronomical utilities
commonly needed in tracking applications.
"""

from pytcl.astronomical.time_systems import (
    # Julian dates
    cal_to_jd,
    jd_to_cal,
    mjd_to_jd,
    jd_to_mjd,
    # Time scales
    utc_to_tai,
    tai_to_utc,
    tai_to_tt,
    tt_to_tai,
    utc_to_tt,
    tt_to_utc,
    tai_to_gps,
    gps_to_tai,
    utc_to_gps,
    gps_to_utc,
    # Unix time
    unix_to_jd,
    jd_to_unix,
    # GPS week
    gps_week_seconds,
    gps_week_to_utc,
    # Sidereal time
    gmst,
    gast,
    # Leap seconds
    get_leap_seconds,
    LeapSecondTable,
)

__all__ = [
    # Julian dates
    "cal_to_jd",
    "jd_to_cal",
    "mjd_to_jd",
    "jd_to_mjd",
    # Time scales
    "utc_to_tai",
    "tai_to_utc",
    "tai_to_tt",
    "tt_to_tai",
    "utc_to_tt",
    "tt_to_utc",
    "tai_to_gps",
    "gps_to_tai",
    "utc_to_gps",
    "gps_to_utc",
    # Unix time
    "unix_to_jd",
    "jd_to_unix",
    # GPS week
    "gps_week_seconds",
    "gps_week_to_utc",
    # Sidereal time
    "gmst",
    "gast",
    # Leap seconds
    "get_leap_seconds",
    "LeapSecondTable",
]
