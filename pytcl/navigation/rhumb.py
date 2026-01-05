"""
Rhumb line (loxodrome) navigation algorithms.

This module provides rhumb line navigation functions for computing
constant-bearing paths on a sphere and ellipsoid, including:
- Rhumb line distance and bearing
- Direct and inverse rhumb problems
- Rhumb line intersection
- Spherical and ellipsoidal formulations
"""

from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.navigation.geodesy import WGS84, Ellipsoid


class RhumbResult(NamedTuple):
    """
    Result of rhumb line computation.

    Attributes
    ----------
    distance : float
        Rhumb line distance in meters.
    bearing : float
        Constant bearing in radians (from north, clockwise).
    """

    distance: float
    bearing: float


class RhumbDirectResult(NamedTuple):
    """
    Result of direct rhumb problem.

    Attributes
    ----------
    lat : float
        Destination latitude in radians.
    lon : float
        Destination longitude in radians.
    """

    lat: float
    lon: float


class RhumbIntersectionResult(NamedTuple):
    """
    Result of rhumb line intersection.

    Attributes
    ----------
    lat : float
        Intersection latitude in radians.
    lon : float
        Intersection longitude in radians.
    valid : bool
        True if intersection exists.
    """

    lat: float
    lon: float
    valid: bool


# Default Earth radius (mean radius in meters)
EARTH_RADIUS = 6371000.0


def _isometric_latitude(lat: float, e2: float = 0.0) -> float:
    """
    Compute isometric latitude (Mercator projection latitude).

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    e2 : float, optional
        First eccentricity squared (0 for sphere).

    Returns
    -------
    float
        Isometric latitude.
    """
    if e2 == 0:
        # Spherical case
        return np.log(np.tan(np.pi / 4 + lat / 2))
    else:
        # Ellipsoidal case
        e = np.sqrt(e2)
        sin_lat = np.sin(lat)
        return np.log(np.tan(np.pi / 4 + lat / 2)) - e / 2 * np.log(
            (1 + e * sin_lat) / (1 - e * sin_lat)
        )


def _inverse_isometric_latitude(
    psi: float, e2: float = 0.0, max_iter: int = 20
) -> float:
    """
    Compute geodetic latitude from isometric latitude.

    Parameters
    ----------
    psi : float
        Isometric latitude.
    e2 : float, optional
        First eccentricity squared (0 for sphere).
    max_iter : int, optional
        Maximum iterations for ellipsoidal case.

    Returns
    -------
    float
        Geodetic latitude in radians.
    """
    if e2 == 0:
        # Spherical case
        return 2 * np.arctan(np.exp(psi)) - np.pi / 2
    else:
        # Ellipsoidal case - iterative solution
        e = np.sqrt(e2)
        lat = 2 * np.arctan(np.exp(psi)) - np.pi / 2

        for _ in range(max_iter):
            sin_lat = np.sin(lat)
            lat_new = (
                2
                * np.arctan(
                    ((1 + e * sin_lat) / (1 - e * sin_lat)) ** (e / 2) * np.exp(psi)
                )
                - np.pi / 2
            )
            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new

        return lat


def rhumb_distance_spherical(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS,
) -> float:
    """
    Compute rhumb line distance on a sphere.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    float
        Rhumb line distance in meters.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)
    >>> lat2, lon2 = np.radians(51), np.radians(0)
    >>> dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2)
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Handle wraparound
    if abs(dlon) > np.pi:
        dlon = dlon - np.sign(dlon) * 2 * np.pi

    # Compute q (scaling factor for longitude)
    dpsi = _isometric_latitude(lat2) - _isometric_latitude(lat1)

    if abs(dpsi) > 1e-12:
        q = dlat / dpsi
    else:
        # At nearly the same latitude
        q = np.cos(lat1)

    distance = np.sqrt(dlat**2 + q**2 * dlon**2) * radius

    return float(distance)


def rhumb_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute rhumb line bearing from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.

    Returns
    -------
    float
        Constant bearing in radians (from north, clockwise, 0 to 2π).

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)
    >>> lat2, lon2 = np.radians(51), np.radians(0)
    >>> bearing = rhumb_bearing(lat1, lon1, lat2, lon2)
    >>> print(f"Bearing: {np.degrees(bearing):.1f}°")
    """
    dlon = lon2 - lon1

    # Handle wraparound
    if abs(dlon) > np.pi:
        dlon = dlon - np.sign(dlon) * 2 * np.pi

    dpsi = _isometric_latitude(lat2) - _isometric_latitude(lat1)

    bearing = np.arctan2(dlon, dpsi)

    return bearing % (2 * np.pi)


def indirect_rhumb_spherical(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS,
) -> RhumbResult:
    """
    Solve the indirect rhumb problem on a sphere.

    Given two points, find the rhumb line distance and bearing.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    RhumbResult
        Distance and constant bearing.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)  # New York
    >>> lat2, lon2 = np.radians(51), np.radians(0)   # London
    >>> result = indirect_rhumb_spherical(lat1, lon1, lat2, lon2)
    >>> result.distance > 5000000  # Over 5000 km
    True
    """
    distance = rhumb_distance_spherical(lat1, lon1, lat2, lon2, radius)
    bearing = rhumb_bearing(lat1, lon1, lat2, lon2)

    return RhumbResult(distance, bearing)


def direct_rhumb_spherical(
    lat1: float,
    lon1: float,
    bearing: float,
    distance: float,
    radius: float = EARTH_RADIUS,
) -> RhumbDirectResult:
    """
    Solve the direct rhumb problem on a sphere.

    Given starting point, bearing, and distance, find destination.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    bearing : float
        Constant bearing in radians (from north, clockwise).
    distance : float
        Distance in meters.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    RhumbDirectResult
        Destination latitude and longitude.

    Examples
    --------
    >>> import numpy as np
    >>> lat, lon = np.radians(40), np.radians(-74)
    >>> bearing = np.radians(90)  # Due east
    >>> dest = direct_rhumb_spherical(lat, lon, bearing, 1000000)
    """
    delta = distance / radius  # Angular distance

    dlat = delta * np.cos(bearing)
    lat2 = lat1 + dlat

    # Check for pole crossing
    if abs(lat2) > np.pi / 2:
        lat2 = np.sign(lat2) * np.pi - lat2

    dpsi = _isometric_latitude(lat2) - _isometric_latitude(lat1)

    if abs(dpsi) > 1e-12:
        q = dlat / dpsi
    else:
        q = np.cos(lat1)

    dlon = delta * np.sin(bearing) / q

    lon2 = lon1 + dlon

    # Normalize longitude to [-π, π)
    lon2 = ((lon2 + np.pi) % (2 * np.pi)) - np.pi

    return RhumbDirectResult(float(lat2), float(lon2))


def rhumb_distance_ellipsoidal(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    ellipsoid: Ellipsoid = WGS84,
) -> float:
    """
    Compute rhumb line distance on an ellipsoid.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    float
        Rhumb line distance in meters.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)
    >>> lat2, lon2 = np.radians(51), np.radians(0)
    >>> dist = rhumb_distance_ellipsoidal(lat1, lon1, lat2, lon2)
    >>> dist > 5000000  # Over 5000 km
    True
    """
    a = ellipsoid.a
    e2 = ellipsoid.e2

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Handle wraparound
    if abs(dlon) > np.pi:
        dlon = dlon - np.sign(dlon) * 2 * np.pi

    # Isometric latitudes
    psi1 = _isometric_latitude(lat1, e2)
    psi2 = _isometric_latitude(lat2, e2)
    dpsi = psi2 - psi1

    if abs(dpsi) > 1e-12:
        q = dlat / dpsi
    else:
        # Use midpoint approximation
        lat_mid = (lat1 + lat2) / 2
        sin_lat = np.sin(lat_mid)
        q = np.cos(lat_mid) / np.sqrt(1 - e2 * sin_lat**2)

    # Meridional arc length (simplified approximation)
    # For high accuracy, use elliptic integrals
    lat_mid = (lat1 + lat2) / 2
    sin_lat = np.sin(lat_mid)
    M = a * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5  # Meridional radius

    # Rhumb distance
    if abs(dlat) > 1e-12:
        distance = M * np.sqrt(dlat**2 + (q * dlon) ** 2 * (dlat / abs(dlat)) ** 2)
        distance = abs(dlat) * M / np.cos(np.arctan2(q * dlon, dlat))
    else:
        # East-west rhumb line
        N = a / np.sqrt(1 - e2 * sin_lat**2)  # Transverse radius
        distance = N * np.cos(lat_mid) * abs(dlon)

    return float(abs(distance))


def indirect_rhumb(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    ellipsoid: Ellipsoid = WGS84,
) -> RhumbResult:
    """
    Solve the indirect rhumb problem on an ellipsoid.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    RhumbResult
        Distance and constant bearing.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)  # New York
    >>> lat2, lon2 = np.radians(51), np.radians(0)   # London
    >>> result = indirect_rhumb(lat1, lon1, lat2, lon2)
    >>> 0 < result.bearing < np.pi  # Eastward bearing
    True
    """
    distance = rhumb_distance_ellipsoidal(lat1, lon1, lat2, lon2, ellipsoid)

    # Bearing uses isometric latitude difference
    dlon = lon2 - lon1
    if abs(dlon) > np.pi:
        dlon = dlon - np.sign(dlon) * 2 * np.pi

    dpsi = _isometric_latitude(lat2, ellipsoid.e2) - _isometric_latitude(
        lat1, ellipsoid.e2
    )
    bearing = np.arctan2(dlon, dpsi) % (2 * np.pi)

    return RhumbResult(distance, bearing)


def direct_rhumb(
    lat1: float,
    lon1: float,
    bearing: float,
    distance: float,
    ellipsoid: Ellipsoid = WGS84,
) -> RhumbDirectResult:
    """
    Solve the direct rhumb problem on an ellipsoid.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    bearing : float
        Constant bearing in radians (from north, clockwise).
    distance : float
        Distance in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    RhumbDirectResult
        Destination latitude and longitude.

    Examples
    --------
    >>> import numpy as np
    >>> lat, lon = np.radians(40), np.radians(-74)
    >>> bearing = np.radians(90)  # Due east
    >>> dest = direct_rhumb(lat, lon, bearing, 100000)  # 100 km
    >>> np.degrees(dest.lon) > -74  # Moved east
    True
    """
    a = ellipsoid.a
    e2 = ellipsoid.e2

    # Meridional radius at starting latitude
    sin_lat = np.sin(lat1)
    M = a * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5

    # Initial latitude increment
    dlat = distance * np.cos(bearing) / M

    # Iterate to refine
    lat2 = lat1 + dlat
    for _ in range(5):
        lat_mid = (lat1 + lat2) / 2
        sin_lat = np.sin(lat_mid)
        M = a * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5
        dlat = distance * np.cos(bearing) / M
        lat2_new = lat1 + dlat
        if abs(lat2_new - lat2) < 1e-12:
            break
        lat2 = lat2_new

    # Check for pole crossing
    if abs(lat2) > np.pi / 2:
        lat2 = np.sign(lat2) * np.pi - lat2

    # Compute longitude change
    psi1 = _isometric_latitude(lat1, e2)
    psi2 = _isometric_latitude(lat2, e2)
    dpsi = psi2 - psi1

    if abs(dpsi) > 1e-12:
        dlon = np.tan(bearing) * dpsi
    else:
        # East-west rhumb
        N = a / np.sqrt(1 - e2 * np.sin(lat1) ** 2)
        dlon = distance * np.sin(bearing) / (N * np.cos(lat1))

    lon2 = lon1 + dlon

    # Normalize longitude
    lon2 = ((lon2 + np.pi) % (2 * np.pi)) - np.pi

    return RhumbDirectResult(float(lat2), float(lon2))


def rhumb_intersect(
    lat1: float,
    lon1: float,
    bearing1: float,
    lat2: float,
    lon2: float,
    bearing2: float,
) -> RhumbIntersectionResult:
    """
    Find intersection of two rhumb lines.

    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in radians.
    bearing1 : float
        Bearing from first point in radians.
    lat2, lon2 : float
        Second point coordinates in radians.
    bearing2 : float
        Bearing from second point in radians.

    Returns
    -------
    RhumbIntersectionResult
        Intersection point and validity flag.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)
    >>> lat2, lon2 = np.radians(51), np.radians(0)
    >>> bearing1 = np.radians(45)
    >>> bearing2 = np.radians(270)
    >>> result = rhumb_intersect(lat1, lon1, bearing1, lat2, lon2, bearing2)
    >>> result.valid  # May or may not intersect
    True

    Notes
    -----
    Unlike great circles, two rhumb lines may not intersect (if bearings
    are parallel or if the lines diverge before meeting).
    """
    # Convert to isometric coordinates
    psi1 = _isometric_latitude(lat1)
    psi2 = _isometric_latitude(lat2)

    # Rhumb lines in isometric coordinates are straight lines:
    # psi = psi0 + (lon - lon0) / tan(bearing)
    # or lon = lon0 + (psi - psi0) * tan(bearing)

    tan_b1 = np.tan(bearing1)
    tan_b2 = np.tan(bearing2)

    # Check for parallel lines
    if abs(tan_b1 - tan_b2) < 1e-12:
        return RhumbIntersectionResult(0.0, 0.0, False)

    # Check for meridional lines (bearing = 0 or π)
    if abs(np.cos(bearing1)) > 0.99999:
        # First rhumb is nearly meridional (north-south)
        lon_i = lon1
        if abs(np.cos(bearing2)) > 0.99999:
            # Both are meridional - no intersection unless same longitude
            if abs(lon1 - lon2) < 1e-12:
                # Same meridian, infinite intersections
                return RhumbIntersectionResult((lat1 + lat2) / 2, lon1, True)
            return RhumbIntersectionResult(0.0, 0.0, False)
        psi_i = psi2 + (lon_i - lon2) / tan_b2
    elif abs(np.cos(bearing2)) > 0.99999:
        # Second rhumb is nearly meridional
        lon_i = lon2
        psi_i = psi1 + (lon_i - lon1) / tan_b1
    else:
        # General case
        # lon = lon1 + (psi - psi1) * tan_b1
        # lon = lon2 + (psi - psi2) * tan_b2
        # lon1 + (psi - psi1) * tan_b1 = lon2 + (psi - psi2) * tan_b2
        # psi * (tan_b1 - tan_b2) = lon2 - lon1 + psi1*tan_b1 - psi2*tan_b2

        psi_i = (lon2 - lon1 + psi1 * tan_b1 - psi2 * tan_b2) / (tan_b1 - tan_b2)
        lon_i = lon1 + (psi_i - psi1) * tan_b1

    # Convert back to geodetic latitude
    lat_i = _inverse_isometric_latitude(psi_i)

    # Check if valid (within reasonable range)
    if abs(lat_i) > np.pi / 2:
        return RhumbIntersectionResult(0.0, 0.0, False)

    # Normalize longitude
    lon_i = ((lon_i + np.pi) % (2 * np.pi)) - np.pi

    return RhumbIntersectionResult(float(lat_i), float(lon_i), True)


def rhumb_midpoint(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> RhumbDirectResult:
    """
    Compute midpoint along a rhumb line.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.

    Returns
    -------
    RhumbDirectResult
        Midpoint latitude and longitude.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(0), np.radians(0)
    >>> lat2, lon2 = np.radians(10), np.radians(10)
    >>> mid = rhumb_midpoint(lat1, lon1, lat2, lon2)
    >>> np.isclose(np.degrees(mid.lat), 5, atol=0.1)
    True
    """
    result = indirect_rhumb_spherical(lat1, lon1, lat2, lon2)
    return direct_rhumb_spherical(lat1, lon1, result.bearing, result.distance / 2)


def rhumb_waypoints(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    n_points: int,
    radius: float = EARTH_RADIUS,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute multiple waypoints along a rhumb line.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    n_points : int
        Number of waypoints (including start and end).
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    lats, lons : ndarray
        Arrays of waypoint latitudes and longitudes in radians.

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)
    >>> lat2, lon2 = np.radians(51), np.radians(0)
    >>> lats, lons = rhumb_waypoints(lat1, lon1, lat2, lon2, 5)
    >>> len(lats)
    5
    """
    result = indirect_rhumb_spherical(lat1, lon1, lat2, lon2, radius)

    distances = np.linspace(0, result.distance, n_points)
    lats = np.zeros(n_points)
    lons = np.zeros(n_points)

    for i, d in enumerate(distances):
        wp = direct_rhumb_spherical(lat1, lon1, result.bearing, d, radius)
        lats[i] = wp.lat
        lons[i] = wp.lon

    return lats, lons


def compare_great_circle_rhumb(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS,
) -> Tuple[float, float, float]:
    """
    Compare great circle and rhumb line paths.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    gc_distance : float
        Great circle distance in meters.
    rhumb_distance : float
        Rhumb line distance in meters.
    difference_percent : float
        Percentage difference (rhumb is longer).

    Examples
    --------
    >>> import numpy as np
    >>> lat1, lon1 = np.radians(40), np.radians(-74)  # NYC
    >>> lat2, lon2 = np.radians(51), np.radians(0)   # London
    >>> gc, rhumb, diff = compare_great_circle_rhumb(lat1, lon1, lat2, lon2)
    >>> rhumb > gc  # Rhumb is always longer
    True
    """
    from pytcl.navigation.great_circle import great_circle_distance

    gc_dist = great_circle_distance(lat1, lon1, lat2, lon2, radius)
    rhumb_dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2, radius)

    diff_pct = (rhumb_dist - gc_dist) / gc_dist * 100 if gc_dist > 0 else 0.0

    return gc_dist, rhumb_dist, diff_pct


__all__ = [
    # Constants
    "EARTH_RADIUS",
    # Result types
    "RhumbResult",
    "RhumbDirectResult",
    "RhumbIntersectionResult",
    # Spherical
    "rhumb_distance_spherical",
    "rhumb_bearing",
    "indirect_rhumb_spherical",
    "direct_rhumb_spherical",
    # Ellipsoidal
    "rhumb_distance_ellipsoidal",
    "indirect_rhumb",
    "direct_rhumb",
    # Intersection
    "rhumb_intersect",
    # Waypoints
    "rhumb_midpoint",
    "rhumb_waypoints",
    # Comparison
    "compare_great_circle_rhumb",
]
