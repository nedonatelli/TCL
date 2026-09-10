"""
Jacobian matrices for coordinate transformations.

This module provides functions for computing Jacobian matrices of
coordinate transformations, essential for error propagation in tracking
filters (e.g., converting measurement covariances between coordinate systems).

Performance Notes
-----------------
ENU and NED Jacobians use lru_cache with quantized inputs. Measured
speedup (timeit, 200000 calls, repeated identical lat/lon -- the best
case for the cache) is 12-31% across two machines/runs (2026-08-17);
machine-dependent, not a guarantee, and well short of the previously
claimed 25-40% at the low end.
"""

from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Cache precision: quantize lat/lon to 1e-5 rad, about 64 m on Earth --
# NOT ~1 m as previously claimed (that needs 7 decimals). Frame error is
# bounded by 5e-6 rad (~1 arcsec); measured worst deviation 5.0e-6.
_JACOBIAN_CACHE_DECIMALS = 5


def _quantize_angle(angle: float) -> float:
    """Quantize angle for cache key compatibility."""
    return round(angle, _JACOBIAN_CACHE_DECIMALS)


@lru_cache(maxsize=256)
def _enu_jacobian_cached(
    lat_q: float, lon_q: float
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Cached ENU Jacobian computation with quantized inputs."""
    sin_lat = np.sin(lat_q)
    cos_lat = np.cos(lat_q)
    sin_lon = np.sin(lon_q)
    cos_lon = np.cos(lon_q)

    return (
        (-sin_lon, cos_lon, 0.0),
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
        (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat),
    )


@lru_cache(maxsize=256)
def _ned_jacobian_cached(
    lat_q: float, lon_q: float
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Cached NED Jacobian computation with quantized inputs."""
    sin_lat = np.sin(lat_q)
    cos_lat = np.cos(lat_q)
    sin_lon = np.sin(lon_q)
    cos_lon = np.cos(lon_q)

    return (
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
        (-sin_lon, cos_lon, 0.0),
        (-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat),
    )


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

    Notes
    -----
    Uses cached computation with quantized inputs for performance.
    """
    # Use cached version with quantized inputs
    cached_result = _enu_jacobian_cached(_quantize_angle(lat), _quantize_angle(lon))
    return np.array(cached_result, dtype=np.float64)


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

    Notes
    -----
    Uses cached computation with quantized inputs for performance.
    """
    # Use cached version with quantized inputs
    cached_result = _ned_jacobian_cached(_quantize_angle(lat), _quantize_angle(lon))
    return np.array(cached_result, dtype=np.float64)


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
    >>> from pytcl.coordinate_systems.jacobians import calc_spher_inv_jacob
    >>> P_sph = np.diag([1, 0.01, 0.01])  # [r, az, el] variances
    >>> r, az, el = 1000, np.radians(45), np.radians(30)
    >>> J = calc_spher_inv_jacob([r, az, el])
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
    "enu_jacobian",
    "ned_jacobian",
    "geodetic_jacobian",
    "cross_covariance_transform",
    "numerical_jacobian",
]
