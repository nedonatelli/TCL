"""
Two-Line Element (TLE) set parsing and data structures.

This module provides functions for parsing NORAD Two-Line Element sets,
which are the standard format for distributing satellite orbital data.

TLE Format
----------
A TLE consists of two 69-character lines containing orbital elements
in a specialized format used by NORAD/USSPACECOM.

Line 1 format:
    1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN

    Column  Description
    01      Line number (1)
    03-07   Satellite catalog number
    08      Classification (U=unclassified)
    10-11   International designator (last two digits of launch year)
    12-14   International designator (launch number of the year)
    15-17   International designator (piece of the launch)
    19-20   Epoch year (last two digits)
    21-32   Epoch (day of the year and fractional portion of the day)
    34-43   First derivative of mean motion (ballistic coefficient)
    45-52   Second derivative of mean motion (decimal point assumed)
    54-61   BSTAR drag term (decimal point assumed)
    63      Ephemeris type
    65-68   Element set number
    69      Checksum (modulo 10)

Line 2 format:
    2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN

    Column  Description
    01      Line number (2)
    03-07   Satellite catalog number
    09-16   Inclination (degrees)
    18-25   Right ascension of ascending node (degrees)
    27-33   Eccentricity (decimal point assumed)
    35-42   Argument of perigee (degrees)
    44-51   Mean anomaly (degrees)
    53-63   Mean motion (revolutions per day)
    64-68   Revolution number at epoch
    69      Checksum (modulo 10)

References
----------
.. [1] Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
       4th ed., Microcosm Press, 2013, Appendix C.
.. [2] Hoots, F. R. and Roehrich, R. L., "Spacetrack Report No. 3:
       Models for Propagation of NORAD Element Sets," 1980.
.. [3] CelesTrak, https://celestrak.org/NORAD/documentation/tle-fmt.php
"""

from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np


class TLE(NamedTuple):
    """Two-Line Element set data structure.

    Contains all orbital elements and metadata from a NORAD TLE.

    Attributes
    ----------
    name : str
        Satellite name (from line 0, if present).
    catalog_number : int
        NORAD catalog number.
    classification : str
        Classification ('U' = unclassified).
    int_designator : str
        International designator (e.g., '98067A').
    epoch_year : int
        Epoch year (4-digit).
    epoch_day : float
        Epoch day of year (fractional).
    ndot : float
        First derivative of mean motion (rev/day^2).
    nddot : float
        Second derivative of mean motion (rev/day^3).
    bstar : float
        BSTAR drag coefficient (1/Earth radii).
    ephemeris_type : int
        Ephemeris type (usually 0 for SGP4).
    element_set_number : int
        Element set number.
    inclination : float
        Inclination (radians).
    raan : float
        Right ascension of ascending node (radians).
    eccentricity : float
        Eccentricity (dimensionless).
    arg_perigee : float
        Argument of perigee (radians).
    mean_anomaly : float
        Mean anomaly (radians).
    mean_motion : float
        Mean motion (radians/minute).
    revolution_number : int
        Revolution number at epoch.
    line1 : str
        Original TLE line 1.
    line2 : str
        Original TLE line 2.

    Notes
    -----
    Angular quantities are stored in radians for consistency with the
    rest of the pytcl library. Mean motion is in radians/minute as
    required by SGP4/SDP4.
    """

    name: str
    catalog_number: int
    classification: str
    int_designator: str
    epoch_year: int
    epoch_day: float
    ndot: float
    nddot: float
    bstar: float
    ephemeris_type: int
    element_set_number: int
    inclination: float
    raan: float
    eccentricity: float
    arg_perigee: float
    mean_anomaly: float
    mean_motion: float
    revolution_number: int
    line1: str
    line2: str


def _parse_decimal_with_exponent(s: str) -> float:
    """Parse TLE decimal format with implicit decimal point and exponent.

    TLE format uses: +NNNNN-N meaning 0.NNNNN * 10^-N

    Parameters
    ----------
    s : str
        String in TLE decimal format (e.g., '+12345-4' or ' 12345-4').

    Returns
    -------
    float
        Parsed value.
    """
    s = s.strip()
    if not s or s == "00000-0" or s == "+00000-0" or s == " 00000-0":
        return 0.0

    # Handle sign
    if s[0] == "-":
        sign = -1
        s = s[1:]
    elif s[0] == "+" or s[0] == " ":
        sign = 1
        s = s[1:]
    else:
        sign = 1

    # Find exponent marker (- or +)
    for i in range(len(s) - 1, 0, -1):
        if s[i] == "-" or s[i] == "+":
            mantissa = float("0." + s[:i])
            exponent = int(s[i:])
            return sign * mantissa * (10**exponent)

    # No exponent found, just parse as decimal
    return sign * float("0." + s)


def _verify_checksum(line: str) -> bool:
    """Verify TLE line checksum.

    Parameters
    ----------
    line : str
        TLE line (69 characters).

    Returns
    -------
    bool
        True if checksum is valid.
    """
    if len(line) < 69:
        return False

    checksum = 0
    for c in line[:68]:
        if c.isdigit():
            checksum += int(c)
        elif c == "-":
            checksum += 1

    expected = int(line[68])
    return (checksum % 10) == expected


def _epoch_year_to_full_year(year_2digit: int) -> int:
    """Convert 2-digit epoch year to 4-digit year.

    Uses the convention that years < 57 are in 2000s, otherwise 1900s.
    This matches the NORAD convention (Sputnik launched in 1957).

    Parameters
    ----------
    year_2digit : int
        Two-digit year (0-99).

    Returns
    -------
    int
        Four-digit year.
    """
    if year_2digit < 57:
        return 2000 + year_2digit
    else:
        return 1900 + year_2digit


def parse_tle(
    line1: str,
    line2: str,
    name: str = "",
    verify_checksum: bool = True,
) -> TLE:
    """Parse a Two-Line Element set.

    Parameters
    ----------
    line1 : str
        First line of the TLE (69 characters).
    line2 : str
        Second line of the TLE (69 characters).
    name : str, optional
        Satellite name. Default empty.
    verify_checksum : bool, optional
        Whether to verify line checksums. Default True.

    Returns
    -------
    tle : TLE
        Parsed TLE data structure.

    Raises
    ------
    ValueError
        If TLE format is invalid or checksum fails.

    Examples
    --------
    >>> line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
    >>> line2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479001"
    >>> tle = parse_tle(line1, line2, name="ISS (ZARYA)")
    >>> print(f"Inclination: {np.degrees(tle.inclination):.4f} deg")
    """
    # Validate line lengths
    line1 = line1.rstrip()
    line2 = line2.rstrip()

    if len(line1) < 69:
        raise ValueError(f"Line 1 too short: {len(line1)} characters (need 69)")
    if len(line2) < 69:
        raise ValueError(f"Line 2 too short: {len(line2)} characters (need 69)")

    # Verify line numbers
    if line1[0] != "1":
        raise ValueError(f"Line 1 should start with '1', got '{line1[0]}'")
    if line2[0] != "2":
        raise ValueError(f"Line 2 should start with '2', got '{line2[0]}'")

    # Verify checksums
    if verify_checksum:
        if not _verify_checksum(line1):
            raise ValueError("Line 1 checksum failed")
        if not _verify_checksum(line2):
            raise ValueError("Line 2 checksum failed")

    # Parse line 1
    catalog_number = int(line1[2:7])
    classification = line1[7]
    int_designator = line1[9:17].strip()

    epoch_year_2digit = int(line1[18:20])
    epoch_year = _epoch_year_to_full_year(epoch_year_2digit)
    epoch_day = float(line1[20:32])

    # First derivative of mean motion (revs/day^2, divided by 2)
    ndot_str = line1[33:43].strip()
    ndot = float(ndot_str) * 2  # Multiply by 2 (stored as ndot/2)

    # Second derivative of mean motion (revs/day^3, divided by 6)
    nddot = _parse_decimal_with_exponent(line1[44:52]) * 6  # Multiply by 6

    # BSTAR drag coefficient
    bstar = _parse_decimal_with_exponent(line1[53:61])

    ephemeris_type = int(line1[62])
    element_set_number = int(line1[64:68].strip() or "0")

    # Parse line 2
    catalog_number_2 = int(line2[2:7])
    if catalog_number_2 != catalog_number:
        raise ValueError(
            f"Catalog number mismatch: {catalog_number} vs {catalog_number_2}"
        )

    # Angles in degrees
    inclination_deg = float(line2[8:16])
    raan_deg = float(line2[17:25])

    # Eccentricity (decimal point assumed)
    eccentricity = float("0." + line2[26:33])

    arg_perigee_deg = float(line2[34:42])
    mean_anomaly_deg = float(line2[43:51])

    # Mean motion (revs/day) -> radians/minute
    mean_motion_revs_day = float(line2[52:63])
    mean_motion = mean_motion_revs_day * 2 * np.pi / 1440.0  # rad/min

    revolution_number = int(line2[63:68].strip() or "0")

    # Convert angles to radians
    deg_to_rad = np.pi / 180.0

    return TLE(
        name=name,
        catalog_number=catalog_number,
        classification=classification,
        int_designator=int_designator,
        epoch_year=epoch_year,
        epoch_day=epoch_day,
        ndot=ndot,
        nddot=nddot,
        bstar=bstar,
        ephemeris_type=ephemeris_type,
        element_set_number=element_set_number,
        inclination=inclination_deg * deg_to_rad,
        raan=raan_deg * deg_to_rad,
        eccentricity=eccentricity,
        arg_perigee=arg_perigee_deg * deg_to_rad,
        mean_anomaly=mean_anomaly_deg * deg_to_rad,
        mean_motion=mean_motion,
        revolution_number=revolution_number,
        line1=line1,
        line2=line2,
    )


def parse_tle_3line(lines: str) -> TLE:
    """Parse a three-line TLE (with satellite name).

    Parameters
    ----------
    lines : str
        Three-line TLE string (name, line1, line2).

    Returns
    -------
    tle : TLE
        Parsed TLE data structure.

    Examples
    --------
    >>> tle_text = '''ISS (ZARYA)
    ... 1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993
    ... 2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479001'''
    >>> tle = parse_tle_3line(tle_text)
    >>> print(tle.name)
    ISS (ZARYA)
    """
    line_list = lines.strip().split("\n")

    if len(line_list) < 2:
        raise ValueError("TLE must have at least 2 lines")

    if len(line_list) == 2:
        return parse_tle(line_list[0], line_list[1])
    else:
        name = line_list[0].strip()
        return parse_tle(line_list[1], line_list[2], name=name)


def tle_epoch_to_jd(tle: TLE) -> float:
    """Convert TLE epoch to Julian date.

    Parameters
    ----------
    tle : TLE
        Parsed TLE.

    Returns
    -------
    jd : float
        Julian date of TLE epoch.

    Examples
    --------
    >>> tle = parse_tle(line1, line2)
    >>> jd = tle_epoch_to_jd(tle)
    """
    # Start of year
    year = tle.epoch_year

    # Julian date at midnight Jan 1 of epoch year
    # Using algorithm from time_systems module
    a = (14 - 1) // 12
    y = year + 4800 - a
    m = 1 + 12 * a - 3

    jd_jan1 = 1 + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd_jan1 = float(jd_jan1) - 0.5  # Midnight

    # Add day of year (1-based, so subtract 1)
    jd = jd_jan1 + (tle.epoch_day - 1.0)

    return jd


def tle_epoch_to_datetime(tle: TLE) -> datetime:
    """Convert TLE epoch to Python datetime.

    Parameters
    ----------
    tle : TLE
        Parsed TLE.

    Returns
    -------
    dt : datetime
        TLE epoch as UTC datetime.
    """
    year = tle.epoch_year
    day_of_year = tle.epoch_day

    # Integer day and fractional part
    day_int = int(day_of_year)
    day_frac = day_of_year - day_int

    # Convert to datetime
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(
        days=day_int - 1 + day_frac
    )

    return dt


def format_tle(tle: TLE, include_name: bool = True) -> str:
    """Format TLE data structure back to TLE string.

    Parameters
    ----------
    tle : TLE
        TLE data structure.
    include_name : bool, optional
        Whether to include satellite name as line 0. Default True.

    Returns
    -------
    str
        Formatted TLE string.
    """
    lines = []

    if include_name and tle.name:
        lines.append(tle.name)

    lines.append(tle.line1)
    lines.append(tle.line2)

    return "\n".join(lines)


def is_deep_space(tle: TLE) -> bool:
    """Determine if TLE requires deep-space (SDP4) propagation.

    Satellites with orbital period >= 225 minutes use SDP4 instead
    of SGP4 due to lunar-solar perturbations.

    Parameters
    ----------
    tle : TLE
        Parsed TLE.

    Returns
    -------
    bool
        True if deep-space propagation (SDP4) is required.
    """
    # Mean motion in rad/min, period = 2*pi / n
    period_minutes = 2 * np.pi / tle.mean_motion
    return period_minutes >= 225.0


def semi_major_axis_from_mean_motion(n: float, mu: float = 398600.4418) -> float:
    """Compute semi-major axis from mean motion.

    Uses the relationship n = sqrt(mu / a^3) where n is in rad/s.

    Parameters
    ----------
    n : float
        Mean motion (radians/minute).
    mu : float, optional
        Gravitational parameter (km^3/s^2). Default is Earth.

    Returns
    -------
    a : float
        Semi-major axis (km).
    """
    # Convert to rad/s
    n_rad_s = n / 60.0

    # a = (mu / n^2)^(1/3)
    return (mu / (n_rad_s**2)) ** (1.0 / 3.0)


def orbital_period_from_tle(tle: TLE) -> float:
    """Compute orbital period from TLE mean motion.

    Parameters
    ----------
    tle : TLE
        Parsed TLE.

    Returns
    -------
    period : float
        Orbital period (seconds).
    """
    # Mean motion in rad/min, period = 2*pi / n (in minutes)
    period_minutes = 2 * np.pi / tle.mean_motion
    return period_minutes * 60.0  # Convert to seconds


__all__ = [
    # Types
    "TLE",
    # Parsing
    "parse_tle",
    "parse_tle_3line",
    # Conversion
    "tle_epoch_to_jd",
    "tle_epoch_to_datetime",
    "format_tle",
    # Utilities
    "is_deep_space",
    "semi_major_axis_from_mean_motion",
    "orbital_period_from_tle",
]
