"""
Time system conversions for astronomical and tracking applications.

This module provides conversions between various time systems:
- UTC (Coordinated Universal Time)
- TAI (International Atomic Time)
- TT (Terrestrial Time)
- GPS (GPS Time)
- Julian Date (JD) and Modified Julian Date (MJD)
- Unix time
- Sidereal time (GMST, GAST)
"""

from typing import List, Tuple

import numpy as np

# Constants
JD_UNIX_EPOCH = 2440587.5  # Julian date of Unix epoch (1970-01-01 00:00:00 UTC)
JD_GPS_EPOCH = 2444244.5  # Julian date of GPS epoch (1980-01-06 00:00:00 UTC)
JD_J2000 = 2451545.0  # Julian date of J2000.0 epoch (2000-01-01 12:00:00 TT)
MJD_OFFSET = 2400000.5  # JD - MJD offset

# TAI-UTC offset at GPS epoch (1980-01-06)
TAI_UTC_AT_GPS_EPOCH = 19  # seconds

# TT-TAI offset (constant)
TT_TAI_OFFSET = 32.184  # seconds


class LeapSecondTable:
    """
    Table of leap seconds for UTC-TAI conversion.

    The leap second table contains the dates when leap seconds were added
    and the cumulative TAI-UTC offset.

    Attributes
    ----------
    entries : list of tuple
        List of (year, month, day, tai_utc_offset) tuples.

    Notes
    -----
    This table must be updated when new leap seconds are announced.
    Last update: 2017-01-01 (37 seconds).

    As of 2024, no new leap seconds have been added since 2017.
    """

    def __init__(self) -> None:
        # Leap second table: (year, month, day, TAI-UTC offset in seconds)
        # From IERS Bulletin C
        self.entries: List[Tuple[int, int, int, int]] = [
            (1972, 1, 1, 10),
            (1972, 7, 1, 11),
            (1973, 1, 1, 12),
            (1974, 1, 1, 13),
            (1975, 1, 1, 14),
            (1976, 1, 1, 15),
            (1977, 1, 1, 16),
            (1978, 1, 1, 17),
            (1979, 1, 1, 18),
            (1980, 1, 1, 19),
            (1981, 7, 1, 20),
            (1982, 7, 1, 21),
            (1983, 7, 1, 22),
            (1985, 7, 1, 23),
            (1988, 1, 1, 24),
            (1990, 1, 1, 25),
            (1991, 1, 1, 26),
            (1992, 7, 1, 27),
            (1993, 7, 1, 28),
            (1994, 7, 1, 29),
            (1996, 1, 1, 30),
            (1997, 7, 1, 31),
            (1999, 1, 1, 32),
            (2006, 1, 1, 33),
            (2009, 1, 1, 34),
            (2012, 7, 1, 35),
            (2015, 7, 1, 36),
            (2017, 1, 1, 37),
        ]

    def get_offset(self, year: int, month: int, day: int) -> int:
        """
        Get TAI-UTC offset for a given date.

        Parameters
        ----------
        year : int
            Year.
        month : int
            Month (1-12).
        day : int
            Day of month.

        Returns
        -------
        int
            TAI-UTC offset in seconds.
        """
        offset = 0
        for y, m, d, o in self.entries:
            if (year, month, day) >= (y, m, d):
                offset = o
            else:
                break
        return offset


# Global leap second table instance
_LEAP_SECOND_TABLE = LeapSecondTable()


def get_leap_seconds(year: int, month: int, day: int) -> int:
    """
    Get the number of leap seconds (TAI-UTC offset) for a given date.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month (1-12).
    day : int
        Day of month.

    Returns
    -------
    int
        TAI-UTC offset in seconds.

    Examples
    --------
    >>> get_leap_seconds(2020, 1, 1)
    37
    >>> get_leap_seconds(1980, 1, 6)
    19
    """
    return _LEAP_SECOND_TABLE.get_offset(year, month, day)


def cal_to_jd(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Convert calendar date to Julian Date.

    Parameters
    ----------
    year : int
        Year (negative for BCE).
    month : int
        Month (1-12).
    day : int
        Day of month (1-31).
    hour : int, optional
        Hour (0-23).
    minute : int, optional
        Minute (0-59).
    second : float, optional
        Second (0-59.999...).

    Returns
    -------
    float
        Julian Date.

    Examples
    --------
    >>> cal_to_jd(2000, 1, 1, 12, 0, 0)  # J2000.0
    2451545.0
    >>> cal_to_jd(1970, 1, 1, 0, 0, 0)  # Unix epoch
    2440587.5

    Notes
    -----
    Uses the algorithm from Meeus, "Astronomical Algorithms", 2nd ed.
    Valid for dates after October 15, 1582 (Gregorian calendar).
    """
    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    # Add time of day
    jd += (hour + minute / 60.0 + second / 3600.0) / 24.0

    return jd


def jd_to_cal(jd: float) -> Tuple[int, int, int, int, int, float]:
    """
    Convert Julian Date to calendar date.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    tuple
        (year, month, day, hour, minute, second).

    Examples
    --------
    >>> jd_to_cal(2451545.0)  # J2000.0
    (2000, 1, 1, 12, 0, 0.0)
    """
    jd = jd + 0.5
    Z = int(jd)
    F = jd - Z

    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day = B - D - int(30.6001 * E)

    if E < 14:
        month = E - 1
    else:
        month = E - 13

    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    # Convert fractional day to time
    hours_frac = F * 24.0
    hour = int(hours_frac)
    minutes_frac = (hours_frac - hour) * 60.0
    minute = int(minutes_frac)
    second = (minutes_frac - minute) * 60.0

    return year, month, day, hour, minute, second


def mjd_to_jd(mjd: float) -> float:
    """
    Convert Modified Julian Date to Julian Date.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.

    Returns
    -------
    float
        Julian Date.

    Notes
    -----
    MJD = JD - 2400000.5
    """
    return mjd + MJD_OFFSET


def jd_to_mjd(jd: float) -> float:
    """
    Convert Julian Date to Modified Julian Date.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    float
        Modified Julian Date.
    """
    return jd - MJD_OFFSET


def unix_to_jd(unix_time: float) -> float:
    """
    Convert Unix timestamp to Julian Date.

    Parameters
    ----------
    unix_time : float
        Unix timestamp (seconds since 1970-01-01 00:00:00 UTC).

    Returns
    -------
    float
        Julian Date.

    Examples
    --------
    >>> unix_to_jd(0.0)  # Unix epoch
    2440587.5
    """
    return JD_UNIX_EPOCH + unix_time / 86400.0


def jd_to_unix(jd: float) -> float:
    """
    Convert Julian Date to Unix timestamp.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    float
        Unix timestamp.
    """
    return (jd - JD_UNIX_EPOCH) * 86400.0


def utc_to_tai(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Convert UTC to TAI (Julian Date).

    Parameters
    ----------
    year, month, day : int
        Calendar date.
    hour, minute : int, optional
        Time of day.
    second : float, optional
        Seconds (including fractional).

    Returns
    -------
    float
        TAI as Julian Date.

    Notes
    -----
    TAI = UTC + leap_seconds
    """
    leap_seconds = get_leap_seconds(year, month, day)
    jd_utc = cal_to_jd(year, month, day, hour, minute, second)
    return jd_utc + leap_seconds / 86400.0


def tai_to_utc(jd_tai: float) -> Tuple[float, int]:
    """
    Convert TAI (Julian Date) to UTC.

    Parameters
    ----------
    jd_tai : float
        TAI as Julian Date.

    Returns
    -------
    jd_utc : float
        UTC as Julian Date.
    leap_seconds : int
        Number of leap seconds applied.

    Notes
    -----
    This is an approximate conversion that may have small errors
    near leap second boundaries.
    """
    # First approximation
    jd_utc_approx = jd_tai
    year, month, day, _, _, _ = jd_to_cal(jd_utc_approx)
    leap_seconds = get_leap_seconds(year, month, day)

    jd_utc = jd_tai - leap_seconds / 86400.0
    return jd_utc, leap_seconds


def tai_to_tt(jd_tai: float) -> float:
    """
    Convert TAI (Julian Date) to TT (Terrestrial Time).

    Parameters
    ----------
    jd_tai : float
        TAI as Julian Date.

    Returns
    -------
    float
        TT as Julian Date.

    Notes
    -----
    TT = TAI + 32.184 seconds (constant offset)
    """
    return jd_tai + TT_TAI_OFFSET / 86400.0


def tt_to_tai(jd_tt: float) -> float:
    """
    Convert TT (Terrestrial Time) to TAI (Julian Date).

    Parameters
    ----------
    jd_tt : float
        TT as Julian Date.

    Returns
    -------
    float
        TAI as Julian Date.
    """
    return jd_tt - TT_TAI_OFFSET / 86400.0


def utc_to_tt(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Convert UTC to TT (Terrestrial Time).

    Parameters
    ----------
    year, month, day : int
        Calendar date.
    hour, minute : int, optional
        Time of day.
    second : float, optional
        Seconds.

    Returns
    -------
    float
        TT as Julian Date.

    Notes
    -----
    TT = UTC + leap_seconds + 32.184
    """
    jd_tai = utc_to_tai(year, month, day, hour, minute, second)
    return tai_to_tt(jd_tai)


def tt_to_utc(jd_tt: float) -> Tuple[float, int]:
    """
    Convert TT to UTC.

    Parameters
    ----------
    jd_tt : float
        TT as Julian Date.

    Returns
    -------
    jd_utc : float
        UTC as Julian Date.
    leap_seconds : int
        Number of leap seconds applied.
    """
    jd_tai = tt_to_tai(jd_tt)
    return tai_to_utc(jd_tai)


def tai_to_gps(jd_tai: float) -> float:
    """
    Convert TAI to GPS time.

    Parameters
    ----------
    jd_tai : float
        TAI as Julian Date.

    Returns
    -------
    float
        GPS time as Julian Date.

    Notes
    -----
    GPS time = TAI - 19 seconds (offset at GPS epoch)
    GPS time does not include leap seconds after 1980.
    """
    return jd_tai - TAI_UTC_AT_GPS_EPOCH / 86400.0


def gps_to_tai(jd_gps: float) -> float:
    """
    Convert GPS time to TAI.

    Parameters
    ----------
    jd_gps : float
        GPS time as Julian Date.

    Returns
    -------
    float
        TAI as Julian Date.
    """
    return jd_gps + TAI_UTC_AT_GPS_EPOCH / 86400.0


def utc_to_gps(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Convert UTC to GPS time.

    Parameters
    ----------
    year, month, day : int
        Calendar date.
    hour, minute : int, optional
        Time of day.
    second : float, optional
        Seconds.

    Returns
    -------
    float
        GPS time as Julian Date.
    """
    jd_tai = utc_to_tai(year, month, day, hour, minute, second)
    return tai_to_gps(jd_tai)


def gps_to_utc(jd_gps: float) -> Tuple[float, int]:
    """
    Convert GPS time to UTC.

    Parameters
    ----------
    jd_gps : float
        GPS time as Julian Date.

    Returns
    -------
    jd_utc : float
        UTC as Julian Date.
    leap_seconds : int
        Number of leap seconds applied.
    """
    jd_tai = gps_to_tai(jd_gps)
    return tai_to_utc(jd_tai)


def gps_week_seconds(jd_gps: float) -> Tuple[int, float]:
    """
    Convert GPS Julian Date to GPS week and seconds of week.

    Parameters
    ----------
    jd_gps : float
        GPS time as Julian Date.

    Returns
    -------
    week : int
        GPS week number.
    seconds : float
        Seconds into the week.

    Examples
    --------
    >>> gps_week_seconds(JD_GPS_EPOCH)
    (0, 0.0)
    """
    days_since_epoch = jd_gps - JD_GPS_EPOCH
    week = int(days_since_epoch / 7)
    day_of_week = days_since_epoch - week * 7
    seconds = day_of_week * 86400.0
    return week, seconds


def gps_week_to_utc(week: int, seconds: float) -> Tuple[float, int]:
    """
    Convert GPS week and seconds to UTC.

    Parameters
    ----------
    week : int
        GPS week number.
    seconds : float
        Seconds into the week.

    Returns
    -------
    jd_utc : float
        UTC as Julian Date.
    leap_seconds : int
        Number of leap seconds applied.
    """
    jd_gps = JD_GPS_EPOCH + week * 7 + seconds / 86400.0
    return gps_to_utc(jd_gps)


def gmst(jd_ut1: float) -> float:
    """
    Compute Greenwich Mean Sidereal Time.

    Parameters
    ----------
    jd_ut1 : float
        UT1 as Julian Date.

    Returns
    -------
    float
        GMST in radians.

    Notes
    -----
    Uses the IAU 1982 expression for GMST.

    References
    ----------
    .. [1] Explanatory Supplement to the Astronomical Almanac, 3rd ed.
    """
    # Julian centuries from J2000.0
    T = (jd_ut1 - JD_J2000) / 36525.0

    # GMST in seconds at 0h UT1
    gmst_sec = 24110.54841 + 8640184.812866 * T + 0.093104 * T**2 - 6.2e-6 * T**3

    # Add rotation for time of day
    jd_frac = jd_ut1 - int(jd_ut1) - 0.5
    if jd_frac < 0:
        jd_frac += 1.0
    gmst_sec += jd_frac * 86400.0 * 1.00273790935

    # Normalize to [0, 86400)
    gmst_sec = gmst_sec % 86400.0

    # Convert to radians
    return gmst_sec * 2 * np.pi / 86400.0


def gast(jd_ut1: float, dpsi: float = 0.0, eps: float = 0.0) -> float:
    """
    Compute Greenwich Apparent Sidereal Time.

    Parameters
    ----------
    jd_ut1 : float
        UT1 as Julian Date.
    dpsi : float, optional
        Nutation in longitude (radians). Default 0.
    eps : float, optional
        Mean obliquity of ecliptic (radians). Default uses approximate value.

    Returns
    -------
    float
        GAST in radians.

    Notes
    -----
    GAST = GMST + equation of equinoxes
    The equation of equinoxes = dpsi * cos(eps)

    For high precision applications, nutation parameters should be computed
    from the IAU 2006/2000A precession-nutation model.
    """
    gmst_val = gmst(jd_ut1)

    if eps == 0.0:
        # Approximate mean obliquity
        T = (jd_ut1 - JD_J2000) / 36525.0
        eps = np.radians(23.439291 - 0.0130042 * T)

    # Equation of equinoxes
    eq_eq = dpsi * np.cos(eps)

    return gmst_val + eq_eq


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
    # Constants
    "JD_UNIX_EPOCH",
    "JD_GPS_EPOCH",
    "JD_J2000",
    "MJD_OFFSET",
    "TT_TAI_OFFSET",
]
