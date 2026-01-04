"""
Geodetic coordinate conversions.

This module provides functions for converting between geodetic (latitude,
longitude, altitude) and Earth-centered coordinate systems (ECEF), as well
as local tangent plane coordinates (ENU, NED).
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.constants import WGS84


def geodetic2ecef(
    lat: ArrayLike,
    lon: ArrayLike,
    alt: ArrayLike,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> NDArray[np.floating]:
    """
    Convert geodetic coordinates to ECEF (Earth-Centered Earth-Fixed).

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    lon : array_like
        Geodetic longitude in radians.
    alt : array_like
        Altitude above the reference ellipsoid in meters.
    a : float, optional
        Semi-major axis of the reference ellipsoid in meters.
        Default is WGS84 value.
    f : float, optional
        Flattening of the reference ellipsoid.
        Default is WGS84 value.

    Returns
    -------
    ecef : ndarray
        ECEF coordinates [x, y, z] in meters.
        Shape is (3,) for single point or (3, n) for multiple points.

    Examples
    --------
    >>> lat, lon, alt = np.radians(45), np.radians(-75), 100.0
    >>> ecef = geodetic2ecef(lat, lon, alt)
    >>> ecef / 1e6  # In millions of meters
    array([ 1.14..., -4.29...,  4.48...])

    See Also
    --------
    ecef2geodetic : Inverse conversion.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)

    # Eccentricity squared
    e2 = 2 * f - f**2

    # Prime vertical radius of curvature
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    # ECEF coordinates
    x = (N + alt) * cos_lat * np.cos(lon)
    y = (N + alt) * cos_lat * np.sin(lon)
    z = (N * (1 - e2) + alt) * sin_lat

    if np.isscalar(lat) or lat.size == 1:
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    return np.array([x, y, z], dtype=np.float64)


def ecef2geodetic(
    ecef: ArrayLike,
    a: float = WGS84.a,
    f: float = WGS84.f,
    method: str = "iterative",
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert ECEF coordinates to geodetic coordinates.

    Parameters
    ----------
    ecef : array_like
        ECEF coordinates [x, y, z] in meters.
    a : float, optional
        Semi-major axis of the reference ellipsoid.
    f : float, optional
        Flattening of the reference ellipsoid.
    method : str, optional
        Algorithm to use:
        - 'iterative': Bowring's iterative method (default)
        - 'direct': Closed-form solution (Vermeille's method)

    Returns
    -------
    lat : ndarray
        Geodetic latitude in radians.
    lon : ndarray
        Geodetic longitude in radians.
    alt : ndarray
        Altitude above the ellipsoid in meters.

    Examples
    --------
    >>> ecef = np.array([1.14e6, -4.29e6, 4.48e6])
    >>> lat, lon, alt = ecef2geodetic(ecef)
    >>> np.degrees(lat), np.degrees(lon)
    (45.0..., -75.0...)

    See Also
    --------
    geodetic2ecef : Inverse conversion.
    """
    ecef = np.asarray(ecef, dtype=np.float64)

    if ecef.ndim == 1:
        ecef = ecef.reshape(3, 1)
    elif ecef.shape[0] != 3 and ecef.shape[1] == 3:
        ecef = ecef.T

    x = ecef[0]
    y = ecef[1]
    z = ecef[2]

    # Derived constants
    e2 = 2 * f - f**2  # First eccentricity squared
    b = a * (1 - f)  # Semi-minor axis
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared

    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Distance from z-axis
    p = np.sqrt(x**2 + y**2)

    if method == "iterative":
        # Bowring's iterative method
        # Initial estimate
        theta = np.arctan2(z * a, p * b)
        lat = np.arctan2(
            z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3
        )

        # Iterate for improved accuracy
        for _ in range(5):
            sin_lat = np.sin(lat)
            N = a / np.sqrt(1 - e2 * sin_lat**2)
            lat_new = np.arctan2(z + e2 * N * sin_lat, p)
            if np.all(np.abs(lat_new - lat) < 1e-12):
                break
            lat = lat_new

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)

        # Altitude
        # Use cos_lat when available, otherwise use sin_lat with guard against division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            alt = np.where(
                np.abs(cos_lat) > 1e-10,
                p / cos_lat - N,
                np.abs(z) / np.abs(sin_lat) - N * (1 - e2),
            )
    else:
        # Direct/closed-form method (simplified Vermeille)
        zp = np.abs(z)

        z2 = z**2
        r2 = p**2 + z2
        r = np.sqrt(r2)

        s2 = z2 / r2
        c2 = p**2 / r2
        u = a**2 / r
        v = b**2 / r

        if np.any(c2 > 0.3):
            s = (zp / r) * (1 + c2 * (a - b) / r)
            lat = np.arcsin(s)
            ss = s**2
            c = np.sqrt(1 - ss)
        else:
            c = (p / r) * (1 - s2 * (a - b) / r)
            lat = np.arccos(c)
            ss = 1 - c**2
            s = np.sqrt(ss)

        g = 1 - e2 * ss
        rg = a / np.sqrt(g)
        u = p - rg * c
        v = zp - rg * (1 - e2) * s
        f_val = c * u + s * v
        m = c * v - s * u
        p2 = f_val + 0.5 * m**2 / rg

        lat = np.arctan2(zp * (1 - e2), p * np.sqrt(g))
        alt = p2 - a + rg * (1 - np.cos(lat - np.arctan2(zp, p)))

        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        alt = p / np.cos(lat) - N

    # Handle sign of latitude for southern hemisphere
    lat = np.sign(z) * np.abs(lat)

    if lat.size == 1:
        return float(lat), float(lon), float(alt)

    return lat, lon, alt


def geodetic2enu(
    lat: ArrayLike,
    lon: ArrayLike,
    alt: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> NDArray[np.floating]:
    """
    Convert geodetic coordinates to local ENU (East-North-Up) coordinates.

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    lon : array_like
        Geodetic longitude in radians.
    alt : array_like
        Altitude in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    alt_ref : float
        Reference point altitude in meters.
    a : float, optional
        Semi-major axis of the reference ellipsoid.
    f : float, optional
        Flattening of the reference ellipsoid.

    Returns
    -------
    enu : ndarray
        Local ENU coordinates [east, north, up] in meters.

    See Also
    --------
    enu2geodetic : Inverse conversion.
    ecef2enu : ECEF to ENU conversion.
    """
    # Convert both to ECEF
    ecef = geodetic2ecef(lat, lon, alt, a, f)
    ecef_ref = geodetic2ecef(lat_ref, lon_ref, alt_ref, a, f)

    # Get ENU from ECEF difference
    return ecef2enu(ecef, lat_ref, lon_ref, ecef_ref)


def ecef2enu(
    ecef: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    ecef_ref: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert ECEF coordinates to local ENU coordinates.

    Parameters
    ----------
    ecef : array_like
        ECEF coordinates [x, y, z] in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    ecef_ref : array_like, optional
        Reference point ECEF coordinates. If None, computed from lat_ref, lon_ref.

    Returns
    -------
    enu : ndarray
        Local ENU coordinates [east, north, up] in meters.

    See Also
    --------
    enu2ecef : Inverse conversion.
    """
    ecef = np.asarray(ecef, dtype=np.float64)

    if ecef.ndim == 1:
        ecef = ecef.reshape(3, 1)
    elif ecef.shape[0] != 3:
        ecef = ecef.T

    if ecef_ref is None:
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)
    ecef_ref = np.asarray(ecef_ref, dtype=np.float64).reshape(3, 1)

    # Difference vector
    d = ecef - ecef_ref

    # Rotation matrix from ECEF to ENU
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # ENU = R @ (ECEF - ECEF_ref)
    east = -sin_lon * d[0] + cos_lon * d[1]
    north = -sin_lat * cos_lon * d[0] - sin_lat * sin_lon * d[1] + cos_lat * d[2]
    up = cos_lat * cos_lon * d[0] + cos_lat * sin_lon * d[1] + sin_lat * d[2]

    if east.size == 1:
        return np.array([float(east), float(north), float(up)], dtype=np.float64)

    return np.array([east.flatten(), north.flatten(), up.flatten()], dtype=np.float64)


def enu2ecef(
    enu: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    ecef_ref: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert local ENU coordinates to ECEF coordinates.

    Parameters
    ----------
    enu : array_like
        Local ENU coordinates [east, north, up] in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    ecef_ref : array_like, optional
        Reference point ECEF coordinates.

    Returns
    -------
    ecef : ndarray
        ECEF coordinates [x, y, z] in meters.

    See Also
    --------
    ecef2enu : Inverse conversion.
    """
    enu = np.asarray(enu, dtype=np.float64)

    if enu.ndim == 1:
        enu = enu.reshape(3, 1)
    elif enu.shape[0] != 3:
        enu = enu.T

    if ecef_ref is None:
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)
    ecef_ref = np.asarray(ecef_ref, dtype=np.float64).reshape(3, 1)

    east = enu[0]
    north = enu[1]
    up = enu[2]

    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # ECEF = R^T @ ENU + ECEF_ref
    x = (
        -sin_lon * east
        - sin_lat * cos_lon * north
        + cos_lat * cos_lon * up
        + ecef_ref[0]
    )
    y = (
        cos_lon * east
        - sin_lat * sin_lon * north
        + cos_lat * sin_lon * up
        + ecef_ref[1]
    )
    z = cos_lat * north + sin_lat * up + ecef_ref[2]

    if x.size == 1:
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    return np.array([x.flatten(), y.flatten(), z.flatten()], dtype=np.float64)


def ecef2ned(
    ecef: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    ecef_ref: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert ECEF coordinates to local NED (North-East-Down) coordinates.

    Parameters
    ----------
    ecef : array_like
        ECEF coordinates [x, y, z] in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    ecef_ref : array_like, optional
        Reference point ECEF coordinates.

    Returns
    -------
    ned : ndarray
        Local NED coordinates [north, east, down] in meters.

    See Also
    --------
    ned2ecef : Inverse conversion.
    ecef2enu : Similar, but ENU frame.
    """
    enu = ecef2enu(ecef, lat_ref, lon_ref, ecef_ref)

    if enu.ndim == 1:
        return np.array([enu[1], enu[0], -enu[2]], dtype=np.float64)

    return np.array([enu[1], enu[0], -enu[2]], dtype=np.float64)


def ned2ecef(
    ned: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    ecef_ref: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert local NED coordinates to ECEF coordinates.

    Parameters
    ----------
    ned : array_like
        Local NED coordinates [north, east, down] in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    ecef_ref : array_like, optional
        Reference point ECEF coordinates.

    Returns
    -------
    ecef : ndarray
        ECEF coordinates [x, y, z] in meters.

    See Also
    --------
    ecef2ned : Inverse conversion.
    """
    ned = np.asarray(ned, dtype=np.float64)

    if ned.ndim == 1:
        enu = np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)
    else:
        if ned.shape[0] != 3:
            ned = ned.T
        enu = np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)

    return enu2ecef(enu, lat_ref, lon_ref, ecef_ref)


def enu2ned(enu: ArrayLike) -> NDArray[np.floating]:
    """
    Convert ENU coordinates to NED coordinates.

    Parameters
    ----------
    enu : array_like
        ENU coordinates [east, north, up].

    Returns
    -------
    ned : ndarray
        NED coordinates [north, east, down].
    """
    enu = np.asarray(enu, dtype=np.float64)

    if enu.ndim == 1:
        return np.array([enu[1], enu[0], -enu[2]], dtype=np.float64)

    if enu.shape[0] != 3:
        enu = enu.T
    return np.array([enu[1], enu[0], -enu[2]], dtype=np.float64)


def ned2enu(ned: ArrayLike) -> NDArray[np.floating]:
    """
    Convert NED coordinates to ENU coordinates.

    Parameters
    ----------
    ned : array_like
        NED coordinates [north, east, down].

    Returns
    -------
    enu : ndarray
        ENU coordinates [east, north, up].
    """
    ned = np.asarray(ned, dtype=np.float64)

    if ned.ndim == 1:
        return np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)

    if ned.shape[0] != 3:
        ned = ned.T
    return np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)


def geodetic2sez(
    lat: ArrayLike,
    lon: ArrayLike,
    alt: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> NDArray[np.floating]:
    """
    Convert geodetic coordinates to local SEZ (South-East-Zenith) coordinates.

    SEZ is a horizon-relative coordinate frame where:
    - S (South) points in the southward direction
    - E (East) points in the eastward direction
    - Z (Zenith) points upward (away from Earth center)

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    lon : array_like
        Geodetic longitude in radians.
    alt : array_like
        Altitude in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    alt_ref : float
        Reference point altitude in meters.
    a : float, optional
        Semi-major axis of the reference ellipsoid.
    f : float, optional
        Flattening of the reference ellipsoid.

    Returns
    -------
    sez : ndarray
        Local SEZ coordinates [south, east, zenith] in meters.

    See Also
    --------
    sez2geodetic : Inverse conversion.
    ecef2sez : ECEF to SEZ conversion.

    Notes
    -----
    SEZ is equivalent to NED when azimuth is measured from south.
    Conversion: SEZ = [S, E, Z] = [NED[0], NED[1], -NED[2]]

    Examples
    --------
    >>> sez = geodetic2sez(lat, lon, alt, lat_ref, lon_ref, alt_ref)
    """
    # Convert both to ECEF
    ecef = geodetic2ecef(lat, lon, alt, a, f)
    ecef_ref = geodetic2ecef(lat_ref, lon_ref, alt_ref, a, f)

    # Get SEZ from ECEF difference
    return ecef2sez(ecef, lat_ref, lon_ref, ecef_ref)


def ecef2sez(
    ecef: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    ecef_ref: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert ECEF coordinates to local SEZ coordinates.

    Parameters
    ----------
    ecef : array_like
        ECEF coordinates [X, Y, Z] in meters, shape (3,) or (3, N).
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    ecef_ref : array_like, optional
        Reference ECEF position. If None, the reference point is
        at (lat_ref, lon_ref) with zero altitude.

    Returns
    -------
    sez : ndarray
        SEZ coordinates [south, east, zenith] in meters.

    See Also
    --------
    sez2ecef : Inverse conversion.
    """
    ecef = np.asarray(ecef, dtype=np.float64)

    if ecef_ref is None:
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)
    else:
        ecef_ref = np.asarray(ecef_ref, dtype=np.float64)

    # Relative position in ECEF
    if ecef.ndim == 1:
        delta_ecef = ecef - ecef_ref
    else:
        if ecef.shape[0] != 3:
            ecef = ecef.T
        if ecef_ref.ndim == 1:
            delta_ecef = ecef - ecef_ref[:, np.newaxis]
        else:
            delta_ecef = ecef - ecef_ref[:, np.newaxis]

    # Rotation matrix from ECEF to SEZ
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # SEZ rotation matrix (transforms ECEF delta to SEZ)
    # S = -sin(lat)*cos(lon)*dX - sin(lat)*sin(lon)*dY + cos(lat)*dZ
    # E = -sin(lon)*dX + cos(lon)*dY
    # Z = cos(lat)*cos(lon)*dX + cos(lat)*sin(lon)*dY + sin(lat)*dZ

    if delta_ecef.ndim == 1:
        s = (
            -sin_lat * cos_lon * delta_ecef[0]
            - sin_lat * sin_lon * delta_ecef[1]
            + cos_lat * delta_ecef[2]
        )
        e = -sin_lon * delta_ecef[0] + cos_lon * delta_ecef[1]
        z = (
            cos_lat * cos_lon * delta_ecef[0]
            + cos_lat * sin_lon * delta_ecef[1]
            + sin_lat * delta_ecef[2]
        )
        return np.array([s, e, z], dtype=np.float64)
    else:
        s = (
            -sin_lat * cos_lon * delta_ecef[0, :]
            - sin_lat * sin_lon * delta_ecef[1, :]
            + cos_lat * delta_ecef[2, :]
        )
        e = -sin_lon * delta_ecef[0, :] + cos_lon * delta_ecef[1, :]
        z = (
            cos_lat * cos_lon * delta_ecef[0, :]
            + cos_lat * sin_lon * delta_ecef[1, :]
            + sin_lat * delta_ecef[2, :]
        )
        return np.array([s, e, z], dtype=np.float64)


def sez2ecef(
    sez: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    ecef_ref: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert local SEZ coordinates to ECEF coordinates.

    Parameters
    ----------
    sez : array_like
        SEZ coordinates [south, east, zenith] in meters, shape (3,) or (3, N).
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    ecef_ref : array_like, optional
        Reference ECEF position. If None, the reference point is
        at (lat_ref, lon_ref) with zero altitude.

    Returns
    -------
    ecef : ndarray
        ECEF coordinates [X, Y, Z] in meters.

    See Also
    --------
    ecef2sez : Forward conversion.
    """
    sez = np.asarray(sez, dtype=np.float64)

    if ecef_ref is None:
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)
    else:
        ecef_ref = np.asarray(ecef_ref, dtype=np.float64)

    # Rotation matrix from SEZ to ECEF (transpose of ECEF to SEZ)
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # Inverse rotation: ECEF = ECEF_ref + R_inv @ SEZ
    if sez.ndim == 1:
        dX = -sin_lat * cos_lon * sez[0] - sin_lon * sez[1] + cos_lat * cos_lon * sez[2]
        dY = -sin_lat * sin_lon * sez[0] + cos_lon * sez[1] + cos_lat * sin_lon * sez[2]
        dZ = cos_lat * sez[0] + sin_lat * sez[2]
        return ecef_ref + np.array([dX, dY, dZ], dtype=np.float64)
    else:
        if sez.shape[0] != 3:
            sez = sez.T
        dX = (
            -sin_lat * cos_lon * sez[0, :]
            - sin_lon * sez[1, :]
            + cos_lat * cos_lon * sez[2, :]
        )
        dY = (
            -sin_lat * sin_lon * sez[0, :]
            + cos_lon * sez[1, :]
            + cos_lat * sin_lon * sez[2, :]
        )
        dZ = cos_lat * sez[0, :] + sin_lat * sez[2, :]
        return ecef_ref[:, np.newaxis] + np.array([dX, dY, dZ], dtype=np.float64)


def sez2geodetic(
    sez: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert local SEZ coordinates to geodetic coordinates.

    Parameters
    ----------
    sez : array_like
        SEZ coordinates [south, east, zenith] in meters.
    lat_ref : float
        Reference point latitude in radians.
    lon_ref : float
        Reference point longitude in radians.
    alt_ref : float
        Reference point altitude in meters.
    a : float, optional
        Semi-major axis.
    f : float, optional
        Flattening.

    Returns
    -------
    lat : ndarray
        Geodetic latitude in radians.
    lon : ndarray
        Geodetic longitude in radians.
    alt : ndarray
        Altitude in meters.

    See Also
    --------
    geodetic2sez : Forward conversion.
    """
    ecef_ref = geodetic2ecef(lat_ref, lon_ref, alt_ref, a, f)
    ecef = sez2ecef(sez, lat_ref, lon_ref, ecef_ref)
    return ecef2geodetic(ecef, a, f)


def geocentric_radius(
    lat: ArrayLike,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> NDArray[np.floating]:
    """
    Compute the geocentric radius at a given geodetic latitude.

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    a : float, optional
        Semi-major axis.
    f : float, optional
        Flattening.

    Returns
    -------
    r : ndarray
        Geocentric radius in meters.
    """
    lat = np.asarray(lat, dtype=np.float64)

    b = a * (1 - f)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)

    num = (a**2 * cos_lat) ** 2 + (b**2 * sin_lat) ** 2
    den = (a * cos_lat) ** 2 + (b * sin_lat) ** 2

    return np.sqrt(num / den)


def prime_vertical_radius(
    lat: ArrayLike,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> NDArray[np.floating]:
    """
    Compute the prime vertical radius of curvature.

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    a : float, optional
        Semi-major axis.
    f : float, optional
        Flattening.

    Returns
    -------
    N : ndarray
        Prime vertical radius of curvature in meters.
    """
    lat = np.asarray(lat, dtype=np.float64)
    e2 = 2 * f - f**2
    sin_lat = np.sin(lat)
    return a / np.sqrt(1 - e2 * sin_lat**2)


def meridional_radius(
    lat: ArrayLike,
    a: float = WGS84.a,
    f: float = WGS84.f,
) -> NDArray[np.floating]:
    """
    Compute the meridional radius of curvature.

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    a : float, optional
        Semi-major axis.
    f : float, optional
        Flattening.

    Returns
    -------
    M : ndarray
        Meridional radius of curvature in meters.
    """
    lat = np.asarray(lat, dtype=np.float64)
    e2 = 2 * f - f**2
    sin_lat = np.sin(lat)
    return a * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5


__all__ = [
    "geodetic2ecef",
    "ecef2geodetic",
    "geodetic2enu",
    "ecef2enu",
    "enu2ecef",
    "ecef2ned",
    "ned2ecef",
    "enu2ned",
    "ned2enu",
    "geodetic2sez",
    "ecef2sez",
    "sez2ecef",
    "sez2geodetic",
    "geocentric_radius",
    "prime_vertical_radius",
    "meridional_radius",
]
