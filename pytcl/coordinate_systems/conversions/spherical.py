"""
Spherical and polar coordinate conversions.

This module provides functions for converting between Cartesian and
spherical/polar coordinate systems, following tracking conventions.
"""

from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cart2sphere(
    cart_points: ArrayLike,
    system_type: Literal["standard", "az-el", "range-az-el"] = "standard",
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_points : array_like
        Cartesian coordinates. Can be:
        - Shape (3,) for a single point [x, y, z]
        - Shape (3, n) for n points (each column is a point)
        - Shape (n, 3) will be transposed automatically
    system_type : {'standard', 'az-el', 'range-az-el'}, optional
        Spherical coordinate convention:
        - 'standard': Physics convention (r, θ polar from +z, φ azimuth from +x)
        - 'az-el': Tracking convention (r, azimuth from +x, elevation from xy-plane)
        - 'range-az-el': Same as 'az-el' (alias)
        Default is 'standard'.

    Returns
    -------
    r : ndarray
        Range (radial distance from origin).
    az : ndarray
        Azimuth angle in radians.
        - 'standard': Angle in xy-plane from +x axis [0, 2π)
        - 'az-el': Angle in xy-plane from +x axis [-π, π]
    el : ndarray
        Elevation/polar angle in radians.
        - 'standard': Polar angle from +z axis [0, π]
        - 'az-el': Elevation from xy-plane [-π/2, π/2]

    Examples
    --------
    >>> x, y, z = 1.0, 1.0, 1.0
    >>> r, az, el = cart2sphere([x, y, z], system_type='az-el')
    >>> r
    1.7320508075688772
    >>> np.degrees(az)
    45.0
    >>> np.degrees(el)
    35.26438968275465

    See Also
    --------
    sphere2cart : Inverse conversion.
    """
    cart_points = np.asarray(cart_points, dtype=np.float64)

    # Handle different input shapes
    if cart_points.ndim == 1:
        cart_points = cart_points.reshape(3, 1)
    elif cart_points.shape[0] != 3 and cart_points.shape[1] == 3:
        cart_points = cart_points.T

    x = cart_points[0]
    y = cart_points[1]
    z = cart_points[2]

    # Range
    r = np.sqrt(x**2 + y**2 + z**2)

    # Azimuth (angle in xy-plane from +x)
    az = np.arctan2(y, x)

    if system_type == "standard":
        # Standard physics convention: polar angle from +z
        el = np.arccos(np.clip(z / np.maximum(r, 1e-15), -1, 1))
        # Wrap azimuth to [0, 2π)
        az = np.mod(az, 2 * np.pi)
    else:  # 'az-el' or 'range-az-el'
        # Tracking convention: elevation from xy-plane
        xy_range = np.sqrt(x**2 + y**2)
        el = np.arctan2(z, xy_range)

    # Squeeze if single point
    if r.size == 1:
        return float(r), float(az), float(el)

    return r, az, el


def sphere2cart(
    r: ArrayLike,
    az: ArrayLike,
    el: ArrayLike,
    system_type: Literal["standard", "az-el", "range-az-el"] = "standard",
) -> NDArray[np.floating]:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : array_like
        Range (radial distance).
    az : array_like
        Azimuth angle in radians.
    el : array_like
        Elevation/polar angle in radians.
    system_type : {'standard', 'az-el', 'range-az-el'}, optional
        Spherical coordinate convention (see cart2sphere).

    Returns
    -------
    cart_points : ndarray
        Cartesian coordinates of shape (3,) or (3, n).

    Examples
    --------
    >>> r, az, el = 1.732, np.radians(45), np.radians(35.26)
    >>> cart2sphere(sphere2cart(r, az, el, 'az-el'), 'az-el')
    (1.732..., 0.785..., 0.615...)

    See Also
    --------
    cart2sphere : Inverse conversion.
    """
    r = np.asarray(r, dtype=np.float64)
    az = np.asarray(az, dtype=np.float64)
    el = np.asarray(el, dtype=np.float64)

    if system_type == "standard":
        # Standard physics: el is polar angle from +z
        x = r * np.sin(el) * np.cos(az)
        y = r * np.sin(el) * np.sin(az)
        z = r * np.cos(el)
    else:  # 'az-el' or 'range-az-el'
        # Tracking: el is elevation from xy-plane
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

    if np.isscalar(r) or r.size == 1:
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    return np.array([x, y, z], dtype=np.float64)


def cart2pol(
    cart_points: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert 2D Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    cart_points : array_like
        Cartesian coordinates. Can be:
        - Shape (2,) for a single point [x, y]
        - Shape (2, n) for n points
        - Shape (n, 2) will be transposed

    Returns
    -------
    r : ndarray
        Radial distance from origin.
    theta : ndarray
        Angle in radians from +x axis, in range [-π, π].

    Examples
    --------
    >>> r, theta = cart2pol([1, 1])
    >>> r
    1.4142135623730951
    >>> np.degrees(theta)
    45.0

    See Also
    --------
    pol2cart : Inverse conversion.
    """
    cart_points = np.asarray(cart_points, dtype=np.float64)

    if cart_points.ndim == 1:
        cart_points = cart_points.reshape(2, 1)
    elif cart_points.shape[0] != 2 and cart_points.shape[1] == 2:
        cart_points = cart_points.T

    x = cart_points[0]
    y = cart_points[1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    if r.size == 1:
        return float(r), float(theta)

    return r, theta


def pol2cart(
    r: ArrayLike,
    theta: ArrayLike,
) -> NDArray[np.floating]:
    """
    Convert polar coordinates to 2D Cartesian coordinates.

    Parameters
    ----------
    r : array_like
        Radial distance.
    theta : array_like
        Angle in radians from +x axis.

    Returns
    -------
    cart_points : ndarray
        Cartesian coordinates of shape (2,) or (2, n).

    Examples
    --------
    >>> x, y = pol2cart(1.414, np.radians(45))
    >>> x, y
    (0.999..., 0.999...)

    See Also
    --------
    cart2pol : Inverse conversion.
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if np.isscalar(r) or r.size == 1:
        return np.array([float(x), float(y)], dtype=np.float64)

    return np.array([x, y], dtype=np.float64)


def cart2cyl(
    cart_points: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert 3D Cartesian coordinates to cylindrical coordinates.

    Parameters
    ----------
    cart_points : array_like
        Cartesian coordinates [x, y, z].

    Returns
    -------
    rho : ndarray
        Radial distance in xy-plane.
    phi : ndarray
        Azimuth angle in radians from +x axis.
    z : ndarray
        Height (same as Cartesian z).

    See Also
    --------
    cyl2cart : Inverse conversion.
    """
    cart_points = np.asarray(cart_points, dtype=np.float64)

    if cart_points.ndim == 1:
        cart_points = cart_points.reshape(3, 1)
    elif cart_points.shape[0] != 3 and cart_points.shape[1] == 3:
        cart_points = cart_points.T

    x = cart_points[0]
    y = cart_points[1]
    z = cart_points[2]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    if rho.size == 1:
        return float(rho), float(phi), float(z)

    return rho, phi, z


def cyl2cart(
    rho: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Convert cylindrical coordinates to 3D Cartesian coordinates.

    Parameters
    ----------
    rho : array_like
        Radial distance in xy-plane.
    phi : array_like
        Azimuth angle in radians from +x axis.
    z : array_like
        Height.

    Returns
    -------
    cart_points : ndarray
        Cartesian coordinates of shape (3,) or (3, n).

    See Also
    --------
    cart2cyl : Inverse conversion.
    """
    rho = np.asarray(rho, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    if np.isscalar(rho) or rho.size == 1:
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    return np.array([x, y, z], dtype=np.float64)


def ruv2cart(
    r: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
) -> NDArray[np.floating]:
    """
    Convert r-u-v (range, direction cosines) to Cartesian coordinates.

    The r-u-v system uses direction cosines where:
    - u = cos(az) * cos(el) = x/r
    - v = sin(az) * cos(el) = y/r
    - w = sin(el) = z/r (derived from u, v)

    Parameters
    ----------
    r : array_like
        Range.
    u : array_like
        Direction cosine along x-axis.
    v : array_like
        Direction cosine along y-axis.

    Returns
    -------
    cart_points : ndarray
        Cartesian coordinates.

    Notes
    -----
    This representation is common in radar tracking systems.
    """
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # w = sqrt(1 - u^2 - v^2), assuming positive z
    w_sq = 1.0 - u**2 - v**2
    w = np.sqrt(np.maximum(w_sq, 0.0))

    x = r * u
    y = r * v
    z = r * w

    if np.isscalar(r) or r.size == 1:
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    return np.array([x, y, z], dtype=np.float64)


def cart2ruv(
    cart_points: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert Cartesian coordinates to r-u-v (range, direction cosines).

    Parameters
    ----------
    cart_points : array_like
        Cartesian coordinates [x, y, z].

    Returns
    -------
    r : ndarray
        Range.
    u : ndarray
        Direction cosine along x-axis (x/r).
    v : ndarray
        Direction cosine along y-axis (y/r).

    See Also
    --------
    ruv2cart : Inverse conversion.
    """
    cart_points = np.asarray(cart_points, dtype=np.float64)

    if cart_points.ndim == 1:
        cart_points = cart_points.reshape(3, 1)
    elif cart_points.shape[0] != 3 and cart_points.shape[1] == 3:
        cart_points = cart_points.T

    x = cart_points[0]
    y = cart_points[1]
    z = cart_points[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.maximum(r, 1e-15)

    u = x / r_safe
    v = y / r_safe

    if r.size == 1:
        return float(r), float(u), float(v)

    return r, u, v


__all__ = [
    "cart2sphere",
    "sphere2cart",
    "cart2pol",
    "pol2cart",
    "cart2cyl",
    "cyl2cart",
    "ruv2cart",
    "cart2ruv",
]
