"""
Reference frame transformations.

This module provides transformations between inertial and Earth-fixed
reference frames, including precession, nutation, and Earth rotation.

Frames:
- GCRF: Geocentric Celestial Reference Frame (inertial)
- ITRF: International Terrestrial Reference Frame (Earth-fixed)
- J2000: Mean equator and equinox of J2000.0
- TOD: True of Date (true equator and equinox)
- MOD: Mean of Date (mean equator and equinox)
- PEF: Pseudo Earth-Fixed (rotated by Earth rotation only)

References
----------
.. [1] Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
       4th ed., Microcosm Press, 2013.
.. [2] IERS Conventions (2010), IERS Technical Note No. 36.
.. [3] Capitaine et al., "Expressions for IAU 2000 precession quantities,"
       A&A, 2003.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from pytcl.astronomical.time_systems import JD_J2000


def julian_centuries_j2000(jd: float) -> float:
    """
    Compute Julian centuries since J2000.0.

    Parameters
    ----------
    jd : float
        Julian date.

    Returns
    -------
    T : float
        Julian centuries since J2000.0.
    """
    return (jd - JD_J2000) / 36525.0


def precession_angles_iau76(T: float) -> Tuple[float, float, float]:
    """
    Compute IAU 1976 precession angles.

    Parameters
    ----------
    T : float
        Julian centuries since J2000.0.

    Returns
    -------
    zeta : float
        Precession angle zeta (radians).
    theta : float
        Precession angle theta (radians).
    z : float
        Precession angle z (radians).
    """
    # Angles in arcseconds
    zeta_arcsec = 2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3
    theta_arcsec = 2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3
    z_arcsec = 2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3

    # Convert to radians
    arcsec_to_rad = np.pi / (180 * 3600)
    return (
        zeta_arcsec * arcsec_to_rad,
        theta_arcsec * arcsec_to_rad,
        z_arcsec * arcsec_to_rad,
    )


def precession_matrix_iau76(jd: float) -> NDArray[np.floating]:
    """
    Compute IAU 1976 precession matrix from J2000 to date.

    Parameters
    ----------
    jd : float
        Julian date of epoch.

    Returns
    -------
    P : ndarray
        Precession rotation matrix (3x3).
        Transforms from J2000 (GCRF) to mean of date.
    """
    T = julian_centuries_j2000(jd)
    zeta, theta, z = precession_angles_iau76(T)

    cos_zeta = np.cos(zeta)
    sin_zeta = np.sin(zeta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_z = np.cos(z)
    sin_z = np.sin(z)

    P = np.array(
        [
            [
                cos_zeta * cos_theta * cos_z - sin_zeta * sin_z,
                -sin_zeta * cos_theta * cos_z - cos_zeta * sin_z,
                -sin_theta * cos_z,
            ],
            [
                cos_zeta * cos_theta * sin_z + sin_zeta * cos_z,
                -sin_zeta * cos_theta * sin_z + cos_zeta * cos_z,
                -sin_theta * sin_z,
            ],
            [cos_zeta * sin_theta, -sin_zeta * sin_theta, cos_theta],
        ]
    )

    return P


def nutation_angles_iau80(jd: float) -> Tuple[float, float]:
    """
    Compute IAU 1980 nutation angles (simplified).

    Parameters
    ----------
    jd : float
        Julian date.

    Returns
    -------
    dpsi : float
        Nutation in longitude (radians).
    deps : float
        Nutation in obliquity (radians).
    """
    T = julian_centuries_j2000(jd)

    # Fundamental arguments (degrees)
    # Mean anomaly of the Moon
    l_moon = 134.96340251 + 477198.8675605 * T
    # Mean anomaly of the Sun
    l_prime = 357.52910918 + 35999.0502911 * T
    # Mean argument of latitude of the Moon
    F = 93.27209062 + 483202.0174577 * T
    # Mean elongation of the Moon from the Sun
    D = 297.85019547 + 445267.1114469 * T
    # Mean longitude of ascending node of the Moon
    Omega = 125.04455501 - 1934.1361851 * T

    # Convert to radians
    deg_to_rad = np.pi / 180
    l_moon = l_moon * deg_to_rad
    l_prime = l_prime * deg_to_rad
    F = F * deg_to_rad
    D = D * deg_to_rad
    Omega = Omega * deg_to_rad

    # Nutation in longitude and obliquity (arcseconds)
    # Using only the largest terms (106 terms in full theory)
    dpsi_arcsec = (
        -17.2 * np.sin(Omega)
        - 1.32 * np.sin(2 * l_moon)
        - 0.23 * np.sin(2 * F)
        + 0.21 * np.sin(2 * Omega)
    )

    deps_arcsec = (
        9.2 * np.cos(Omega)
        + 0.57 * np.cos(2 * l_moon)
        + 0.10 * np.cos(2 * F)
        - 0.09 * np.cos(2 * Omega)
    )

    # Convert to radians
    arcsec_to_rad = np.pi / (180 * 3600)
    return dpsi_arcsec * arcsec_to_rad, deps_arcsec * arcsec_to_rad


def mean_obliquity_iau80(jd: float) -> float:
    """
    Compute mean obliquity of the ecliptic (IAU 1980).

    Parameters
    ----------
    jd : float
        Julian date.

    Returns
    -------
    eps0 : float
        Mean obliquity (radians).
    """
    T = julian_centuries_j2000(jd)

    # Mean obliquity in arcseconds
    eps0_arcsec = 84381.448 - 46.8150 * T - 0.00059 * T**2 + 0.001813 * T**3

    return eps0_arcsec * np.pi / (180 * 3600)


def nutation_matrix(jd: float) -> NDArray[np.floating]:
    """
    Compute nutation matrix.

    Parameters
    ----------
    jd : float
        Julian date.

    Returns
    -------
    N : ndarray
        Nutation rotation matrix (3x3).
        Transforms from mean of date to true of date.
    """
    dpsi, deps = nutation_angles_iau80(jd)
    eps0 = mean_obliquity_iau80(jd)
    eps = eps0 + deps

    cos_eps0 = np.cos(eps0)
    sin_eps0 = np.sin(eps0)
    cos_eps = np.cos(eps)
    sin_eps = np.sin(eps)
    cos_dpsi = np.cos(dpsi)
    sin_dpsi = np.sin(dpsi)

    N = np.array(
        [
            [cos_dpsi, -sin_dpsi * cos_eps0, -sin_dpsi * sin_eps0],
            [
                sin_dpsi * cos_eps,
                cos_dpsi * cos_eps0 * cos_eps + sin_eps0 * sin_eps,
                cos_dpsi * sin_eps0 * cos_eps - cos_eps0 * sin_eps,
            ],
            [
                sin_dpsi * sin_eps,
                cos_dpsi * cos_eps0 * sin_eps - sin_eps0 * cos_eps,
                cos_dpsi * sin_eps0 * sin_eps + cos_eps0 * cos_eps,
            ],
        ]
    )

    return N


def earth_rotation_angle(jd_ut1: float) -> float:
    """
    Compute Earth Rotation Angle (ERA).

    Parameters
    ----------
    jd_ut1 : float
        Julian date in UT1.

    Returns
    -------
    theta : float
        Earth rotation angle (radians), in [0, 2*pi).
    """
    # Days since J2000.0 at 0h UT1
    Du = jd_ut1 - JD_J2000

    # ERA in turns (IERS 2003 model)
    theta_turns = 0.7790572732640 + 1.00273781191135448 * Du

    # Convert to radians and normalize
    theta = (theta_turns % 1.0) * 2 * np.pi
    return theta


def gmst_iau82(jd_ut1: float) -> float:
    """
    Compute Greenwich Mean Sidereal Time (GMST) using IAU 1982 model.

    Parameters
    ----------
    jd_ut1 : float
        Julian date in UT1.

    Returns
    -------
    gmst : float
        GMST (radians), in [0, 2*pi).
    """
    # Julian centuries since J2000 at 0h UT1
    jd_0h = np.floor(jd_ut1 - 0.5) + 0.5
    T_u = (jd_0h - JD_J2000) / 36525.0

    # GMST at 0h UT1 (seconds)
    gmst_0h_sec = (
        24110.54841 + 8640184.812866 * T_u + 0.093104 * T_u**2 - 6.2e-6 * T_u**3
    )

    # Add UT1 fraction
    ut1_fraction = (jd_ut1 - jd_0h) * 86400.0
    omega_earth = 1.00273790935  # Ratio of sidereal to solar day
    gmst_sec = gmst_0h_sec + omega_earth * ut1_fraction

    # Convert to radians
    gmst = (gmst_sec % 86400.0) / 86400.0 * 2 * np.pi
    return gmst


def gast_iau82(jd_ut1: float, jd_tt: float) -> float:
    """
    Compute Greenwich Apparent Sidereal Time (GAST).

    Parameters
    ----------
    jd_ut1 : float
        Julian date in UT1.
    jd_tt : float
        Julian date in TT.

    Returns
    -------
    gast : float
        GAST (radians), in [0, 2*pi).
    """
    gmst = gmst_iau82(jd_ut1)
    dpsi, _ = nutation_angles_iau80(jd_tt)
    eps0 = mean_obliquity_iau80(jd_tt)

    # Equation of the equinoxes
    eq_eq = dpsi * np.cos(eps0)

    gast = (gmst + eq_eq) % (2 * np.pi)
    return gast


def sidereal_rotation_matrix(theta: float) -> NDArray[np.floating]:
    """
    Compute Earth rotation matrix (sidereal rotation).

    Parameters
    ----------
    theta : float
        Sidereal angle (GMST or GAST) in radians.

    Returns
    -------
    R : ndarray
        Rotation matrix (3x3).
        Transforms from true of date to pseudo Earth-fixed (PEF).
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    R = np.array(
        [
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [0, 0, 1],
        ]
    )

    return R


def polar_motion_matrix(xp: float, yp: float) -> NDArray[np.floating]:
    """
    Compute polar motion rotation matrix.

    Parameters
    ----------
    xp : float
        Polar motion x coordinate (radians).
    yp : float
        Polar motion y coordinate (radians).

    Returns
    -------
    W : ndarray
        Polar motion matrix (3x3).
        Transforms from PEF to ITRF.
    """
    cos_xp = np.cos(xp)
    sin_xp = np.sin(xp)
    cos_yp = np.cos(yp)
    sin_yp = np.sin(yp)

    W = np.array(
        [
            [cos_xp, sin_xp * sin_yp, sin_xp * cos_yp],
            [0, cos_yp, -sin_yp],
            [-sin_xp, cos_xp * sin_yp, cos_xp * cos_yp],
        ]
    )

    return W


def gcrf_to_itrf(
    r_gcrf: NDArray[np.floating],
    jd_ut1: float,
    jd_tt: float,
    xp: float = 0.0,
    yp: float = 0.0,
) -> NDArray[np.floating]:
    """
    Transform position from GCRF (inertial) to ITRF (Earth-fixed).

    Parameters
    ----------
    r_gcrf : ndarray
        Position in GCRF (km), shape (3,).
    jd_ut1 : float
        Julian date in UT1.
    jd_tt : float
        Julian date in TT.
    xp : float, optional
        Polar motion x (radians). Default 0.
    yp : float, optional
        Polar motion y (radians). Default 0.

    Returns
    -------
    r_itrf : ndarray
        Position in ITRF (km), shape (3,).

    Examples
    --------
    >>> r_gcrf = np.array([5102.5096, 6123.01152, 6378.1363])
    >>> r_itrf = gcrf_to_itrf(r_gcrf, 2453101.827406783, 2453101.828154745)
    """
    # Precession: J2000 -> Mean of Date
    P = precession_matrix_iau76(jd_tt)

    # Nutation: Mean of Date -> True of Date
    N = nutation_matrix(jd_tt)

    # Sidereal rotation: True of Date -> PEF
    gast = gast_iau82(jd_ut1, jd_tt)
    R = sidereal_rotation_matrix(gast)

    # Polar motion: PEF -> ITRF
    W = polar_motion_matrix(xp, yp)

    # Combined transformation: ITRF = W * R * N * P * GCRF
    r_mod = P @ r_gcrf
    r_tod = N @ r_mod
    r_pef = R @ r_tod
    r_itrf = W @ r_pef

    return r_itrf


def itrf_to_gcrf(
    r_itrf: NDArray[np.floating],
    jd_ut1: float,
    jd_tt: float,
    xp: float = 0.0,
    yp: float = 0.0,
) -> NDArray[np.floating]:
    """
    Transform position from ITRF (Earth-fixed) to GCRF (inertial).

    Parameters
    ----------
    r_itrf : ndarray
        Position in ITRF (km), shape (3,).
    jd_ut1 : float
        Julian date in UT1.
    jd_tt : float
        Julian date in TT.
    xp : float, optional
        Polar motion x (radians). Default 0.
    yp : float, optional
        Polar motion y (radians). Default 0.

    Returns
    -------
    r_gcrf : ndarray
        Position in GCRF (km), shape (3,).
    """
    # Compute rotation matrices
    P = precession_matrix_iau76(jd_tt)
    N = nutation_matrix(jd_tt)
    gast = gast_iau82(jd_ut1, jd_tt)
    R = sidereal_rotation_matrix(gast)
    W = polar_motion_matrix(xp, yp)

    # Inverse transformation: GCRF = P.T * N.T * R.T * W.T * ITRF
    r_pef = W.T @ r_itrf
    r_tod = R.T @ r_pef
    r_mod = N.T @ r_tod
    r_gcrf = P.T @ r_mod

    return r_gcrf


def eci_to_ecef(
    r_eci: NDArray[np.floating],
    gmst: float,
) -> NDArray[np.floating]:
    """
    Simple ECI to ECEF transformation (rotation only).

    This is a simplified transformation that only accounts for
    Earth rotation, ignoring precession, nutation, and polar motion.

    Parameters
    ----------
    r_eci : ndarray
        Position in ECI (km), shape (3,).
    gmst : float
        Greenwich Mean Sidereal Time (radians).

    Returns
    -------
    r_ecef : ndarray
        Position in ECEF (km), shape (3,).
    """
    R = sidereal_rotation_matrix(gmst)
    return R @ r_eci


def ecef_to_eci(
    r_ecef: NDArray[np.floating],
    gmst: float,
) -> NDArray[np.floating]:
    """
    Simple ECEF to ECI transformation (rotation only).

    Parameters
    ----------
    r_ecef : ndarray
        Position in ECEF (km), shape (3,).
    gmst : float
        Greenwich Mean Sidereal Time (radians).

    Returns
    -------
    r_eci : ndarray
        Position in ECI (km), shape (3,).
    """
    R = sidereal_rotation_matrix(gmst)
    return R.T @ r_ecef


def equation_of_equinoxes(jd_tt: float) -> float:
    """
    Compute equation of the equinoxes.

    Parameters
    ----------
    jd_tt : float
        Julian date in TT.

    Returns
    -------
    eq_eq : float
        Equation of the equinoxes (radians).
    """
    dpsi, _ = nutation_angles_iau80(jd_tt)
    eps0 = mean_obliquity_iau80(jd_tt)
    return dpsi * np.cos(eps0)


def true_obliquity(jd_tt: float) -> float:
    """
    Compute true obliquity of the ecliptic.

    Parameters
    ----------
    jd_tt : float
        Julian date in TT.

    Returns
    -------
    eps : float
        True obliquity (radians).
    """
    eps0 = mean_obliquity_iau80(jd_tt)
    _, deps = nutation_angles_iau80(jd_tt)
    return eps0 + deps


def ecliptic_to_equatorial(
    r_ecl: NDArray[np.floating],
    obliquity: float,
) -> NDArray[np.floating]:
    """
    Transform from ecliptic to equatorial coordinates.

    Parameters
    ----------
    r_ecl : ndarray
        Position in ecliptic frame (km), shape (3,).
    obliquity : float
        Obliquity of the ecliptic (radians).

    Returns
    -------
    r_eq : ndarray
        Position in equatorial frame (km), shape (3,).
    """
    cos_eps = np.cos(obliquity)
    sin_eps = np.sin(obliquity)

    R = np.array(
        [
            [1, 0, 0],
            [0, cos_eps, -sin_eps],
            [0, sin_eps, cos_eps],
        ]
    )

    return R @ r_ecl


def equatorial_to_ecliptic(
    r_eq: NDArray[np.floating],
    obliquity: float,
) -> NDArray[np.floating]:
    """
    Transform from equatorial to ecliptic coordinates.

    Parameters
    ----------
    r_eq : ndarray
        Position in equatorial frame (km), shape (3,).
    obliquity : float
        Obliquity of the ecliptic (radians).

    Returns
    -------
    r_ecl : ndarray
        Position in ecliptic frame (km), shape (3,).
    """
    cos_eps = np.cos(obliquity)
    sin_eps = np.sin(obliquity)

    R = np.array(
        [
            [1, 0, 0],
            [0, cos_eps, sin_eps],
            [0, -sin_eps, cos_eps],
        ]
    )

    return R @ r_eq


__all__ = [
    # Time utilities
    "julian_centuries_j2000",
    # Precession
    "precession_angles_iau76",
    "precession_matrix_iau76",
    # Nutation
    "nutation_angles_iau80",
    "nutation_matrix",
    "mean_obliquity_iau80",
    "true_obliquity",
    # Earth rotation
    "earth_rotation_angle",
    "gmst_iau82",
    "gast_iau82",
    "sidereal_rotation_matrix",
    "equation_of_equinoxes",
    # Polar motion
    "polar_motion_matrix",
    # Full transformations
    "gcrf_to_itrf",
    "itrf_to_gcrf",
    "eci_to_ecef",
    "ecef_to_eci",
    # Ecliptic/equatorial
    "ecliptic_to_equatorial",
    "equatorial_to_ecliptic",
]
