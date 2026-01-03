"""
Jacobian matrices for coordinate transformations.

This module provides functions for computing Jacobian matrices of
coordinate transformations, essential for error propagation in tracking
filters (e.g., converting measurement covariances between coordinate systems).
"""

from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def spherical_jacobian(
    cart_point: ArrayLike,
    system_type: Literal["standard", "az-el"] = "az-el",
) -> NDArray[np.floating]:
    """
    Compute Jacobian of Cartesian to spherical transformation.

    Returns the Jacobian matrix J where d[r, az, el] = J @ d[x, y, z].

    Parameters
    ----------
    cart_point : array_like
        Cartesian coordinates [x, y, z].
    system_type : {'standard', 'az-el'}, optional
        Spherical coordinate convention. Default is 'az-el'.

    Returns
    -------
    J : ndarray
        3x3 Jacobian matrix.

    Notes
    -----
    For tracking convention ('az-el'):
    - r = sqrt(x² + y² + z²)
    - az = atan2(y, x)
    - el = atan2(z, sqrt(x² + y²))

    Examples
    --------
    >>> J = spherical_jacobian([1, 1, 1])
    >>> J.shape
    (3, 3)
    """
    cart_point = np.asarray(cart_point, dtype=np.float64)
    x, y, z = cart_point

    r = np.sqrt(x**2 + y**2 + z**2)
    r_xy = np.sqrt(x**2 + y**2)

    if r < 1e-15:
        # At origin, Jacobian is undefined
        return np.eye(3, dtype=np.float64) * np.nan

    J = np.zeros((3, 3), dtype=np.float64)

    # dr/dx, dr/dy, dr/dz
    J[0, 0] = x / r
    J[0, 1] = y / r
    J[0, 2] = z / r

    # daz/dx, daz/dy, daz/dz
    if r_xy > 1e-15:
        J[1, 0] = -y / (r_xy**2)
        J[1, 1] = x / (r_xy**2)
        J[1, 2] = 0
    # else: azimuth undefined at z-axis

    if system_type == "standard":
        # del/dx, del/dy, del/dz (polar angle from +z)
        if r > 1e-15:
            J[2, 0] = x * z / (r**2 * r_xy) if r_xy > 1e-15 else 0
            J[2, 1] = y * z / (r**2 * r_xy) if r_xy > 1e-15 else 0
            J[2, 2] = -r_xy / r**2
    else:  # 'az-el'
        # del/dx, del/dy, del/dz (elevation from xy-plane)
        if r > 1e-15:
            J[2, 0] = -x * z / (r**2 * r_xy) if r_xy > 1e-15 else 0
            J[2, 1] = -y * z / (r**2 * r_xy) if r_xy > 1e-15 else 0
            J[2, 2] = r_xy / r**2

    return J


def spherical_jacobian_inv(
    r: float,
    az: float,
    el: float,
    system_type: Literal["standard", "az-el"] = "az-el",
) -> NDArray[np.floating]:
    """
    Compute Jacobian of spherical to Cartesian transformation.

    Returns the Jacobian matrix J where d[x, y, z] = J @ d[r, az, el].

    Parameters
    ----------
    r : float
        Range.
    az : float
        Azimuth in radians.
    el : float
        Elevation in radians.
    system_type : {'standard', 'az-el'}, optional
        Spherical coordinate convention.

    Returns
    -------
    J : ndarray
        3x3 Jacobian matrix.
    """
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    cos_el = np.cos(el)
    sin_el = np.sin(el)

    J = np.zeros((3, 3), dtype=np.float64)

    if system_type == "standard":
        # x = r * sin(el) * cos(az)
        # y = r * sin(el) * sin(az)
        # z = r * cos(el)
        J[0, 0] = sin_el * cos_az  # dx/dr
        J[0, 1] = -r * sin_el * sin_az  # dx/daz
        J[0, 2] = r * cos_el * cos_az  # dx/del

        J[1, 0] = sin_el * sin_az  # dy/dr
        J[1, 1] = r * sin_el * cos_az  # dy/daz
        J[1, 2] = r * cos_el * sin_az  # dy/del

        J[2, 0] = cos_el  # dz/dr
        J[2, 1] = 0  # dz/daz
        J[2, 2] = -r * sin_el  # dz/del
    else:  # 'az-el'
        # x = r * cos(el) * cos(az)
        # y = r * cos(el) * sin(az)
        # z = r * sin(el)
        J[0, 0] = cos_el * cos_az  # dx/dr
        J[0, 1] = -r * cos_el * sin_az  # dx/daz
        J[0, 2] = -r * sin_el * cos_az  # dx/del

        J[1, 0] = cos_el * sin_az  # dy/dr
        J[1, 1] = r * cos_el * cos_az  # dy/daz
        J[1, 2] = -r * sin_el * sin_az  # dy/del

        J[2, 0] = sin_el  # dz/dr
        J[2, 1] = 0  # dz/daz
        J[2, 2] = r * cos_el  # dz/del

    return J


def polar_jacobian(cart_point: ArrayLike) -> NDArray[np.floating]:
    """
    Compute Jacobian of 2D Cartesian to polar transformation.

    Returns J where d[r, theta] = J @ d[x, y].

    Parameters
    ----------
    cart_point : array_like
        Cartesian coordinates [x, y].

    Returns
    -------
    J : ndarray
        2x2 Jacobian matrix.
    """
    cart_point = np.asarray(cart_point, dtype=np.float64)
    x, y = cart_point[:2]

    r = np.sqrt(x**2 + y**2)

    if r < 1e-15:
        return np.eye(2, dtype=np.float64) * np.nan

    J = np.array([[x / r, y / r], [-y / r**2, x / r**2]], dtype=np.float64)

    return J


def polar_jacobian_inv(r: float, theta: float) -> NDArray[np.floating]:
    """
    Compute Jacobian of polar to 2D Cartesian transformation.

    Returns J where d[x, y] = J @ d[r, theta].

    Parameters
    ----------
    r : float
        Radial distance.
    theta : float
        Angle in radians.

    Returns
    -------
    J : ndarray
        2x2 Jacobian matrix.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    J = np.array([[cos_t, -r * sin_t], [sin_t, r * cos_t]], dtype=np.float64)

    return J


def ruv_jacobian(cart_point: ArrayLike) -> NDArray[np.floating]:
    """
    Compute Jacobian of Cartesian to r-u-v transformation.

    The r-u-v representation uses range and direction cosines:
    - u = x/r (direction cosine along x)
    - v = y/r (direction cosine along y)

    Parameters
    ----------
    cart_point : array_like
        Cartesian coordinates [x, y, z].

    Returns
    -------
    J : ndarray
        3x3 Jacobian matrix where d[r, u, v] = J @ d[x, y, z].
    """
    cart_point = np.asarray(cart_point, dtype=np.float64)
    x, y, z = cart_point

    r = np.sqrt(x**2 + y**2 + z**2)

    if r < 1e-15:
        return np.eye(3, dtype=np.float64) * np.nan

    r2 = r**2
    r3 = r**3

    J = np.array(
        [
            [x / r, y / r, z / r],  # dr/d(x,y,z)
            [(r2 - x**2) / r3, -x * y / r3, -x * z / r3],  # du/d(x,y,z)
            [-x * y / r3, (r2 - y**2) / r3, -y * z / r3],  # dv/d(x,y,z)
        ],
        dtype=np.float64,
    )

    return J


def enu_jacobian(
    lat: float,
    lon: float,
) -> NDArray[np.floating]:
    """
    Compute Jacobian of ECEF to ENU transformation.

    Returns J where d[e, n, u] = J @ d[x, y, z].

    Parameters
    ----------
    lat : float
        Reference latitude in radians.
    lon : float
        Reference longitude in radians.

    Returns
    -------
    J : ndarray
        3x3 rotation matrix (Jacobian is constant for this linear transformation).
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # This is actually the rotation matrix from ECEF to ENU
    J = np.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )

    return J


def ned_jacobian(
    lat: float,
    lon: float,
) -> NDArray[np.floating]:
    """
    Compute Jacobian of ECEF to NED transformation.

    Parameters
    ----------
    lat : float
        Reference latitude in radians.
    lon : float
        Reference longitude in radians.

    Returns
    -------
    J : ndarray
        3x3 rotation matrix.
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Rotation matrix from ECEF to NED
    J = np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ],
        dtype=np.float64,
    )

    return J


def geodetic_jacobian(
    lat: float,
    lon: float,
    alt: float,
    a: float = 6378137.0,
    f: float = 1 / 298.257223563,
) -> NDArray[np.floating]:
    """
    Compute Jacobian of geodetic to ECEF transformation.

    Returns J where d[x, y, z] = J @ d[lat, lon, alt].

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Geodetic longitude in radians.
    alt : float
        Altitude above ellipsoid in meters.
    a : float, optional
        Semi-major axis (default: WGS84).
    f : float, optional
        Flattening (default: WGS84).

    Returns
    -------
    J : ndarray
        3x3 Jacobian matrix.
    """
    e2 = 2 * f - f**2

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = a / np.sqrt(1 - e2 * sin_lat**2)
    dN_dlat = a * e2 * sin_lat * cos_lat / (1 - e2 * sin_lat**2) ** 1.5

    # Partial derivatives
    # x = (N + alt) * cos(lat) * cos(lon)
    dx_dlat = (dN_dlat * cos_lat - (N + alt) * sin_lat) * cos_lon
    dx_dlon = -(N + alt) * cos_lat * sin_lon
    dx_dalt = cos_lat * cos_lon

    # y = (N + alt) * cos(lat) * sin(lon)
    dy_dlat = (dN_dlat * cos_lat - (N + alt) * sin_lat) * sin_lon
    dy_dlon = (N + alt) * cos_lat * cos_lon
    dy_dalt = cos_lat * sin_lon

    # z = (N*(1-e2) + alt) * sin(lat)
    dN1e2_dlat = dN_dlat * (1 - e2)
    dz_dlat = dN1e2_dlat * sin_lat + (N * (1 - e2) + alt) * cos_lat
    dz_dlon = 0
    dz_dalt = sin_lat

    J = np.array(
        [
            [dx_dlat, dx_dlon, dx_dalt],
            [dy_dlat, dy_dlon, dy_dalt],
            [dz_dlat, dz_dlon, dz_dalt],
        ],
        dtype=np.float64,
    )

    return J


def cross_covariance_transform(
    J: ArrayLike,
    P: ArrayLike,
) -> NDArray[np.floating]:
    """
    Transform a covariance matrix through a Jacobian.

    Computes P_new = J @ P @ J.T for error propagation.

    Parameters
    ----------
    J : array_like
        Jacobian matrix of the transformation.
    P : array_like
        Original covariance matrix.

    Returns
    -------
    P_new : ndarray
        Transformed covariance matrix.

    Examples
    --------
    >>> # Transform spherical covariance to Cartesian
    >>> P_sph = np.diag([1, 0.01, 0.01])  # [r, az, el] variances
    >>> r, az, el = 1000, np.radians(45), np.radians(30)
    >>> J = spherical_jacobian_inv(r, az, el)
    >>> P_cart = cross_covariance_transform(J, P_sph)
    """
    J = np.asarray(J, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    return J @ P @ J.T


def numerical_jacobian(
    func: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    dx: float = 1e-7,
) -> NDArray[np.floating]:
    """
    Compute Jacobian numerically using central differences.

    Parameters
    ----------
    func : callable
        Function f(x) -> y.
    x : array_like
        Point at which to compute Jacobian.
    dx : float, optional
        Step size for finite differences.

    Returns
    -------
    J : ndarray
        Jacobian matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(func(x), dtype=np.float64)

    n = len(x)
    m = len(f0) if f0.ndim > 0 else 1

    J = np.zeros((m, n), dtype=np.float64)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += dx
        x_minus[i] -= dx

        f_plus = np.asarray(func(x_plus), dtype=np.float64)
        f_minus = np.asarray(func(x_minus), dtype=np.float64)

        J[:, i] = (f_plus - f_minus) / (2 * dx)

    return J


__all__ = [
    "spherical_jacobian",
    "spherical_jacobian_inv",
    "polar_jacobian",
    "polar_jacobian_inv",
    "ruv_jacobian",
    "enu_jacobian",
    "ned_jacobian",
    "geodetic_jacobian",
    "cross_covariance_transform",
    "numerical_jacobian",
]
