"""
Great circle navigation algorithms.

This module provides great circle (orthodrome) navigation functions for
computing the shortest path on a sphere, including:
- Great circle distance and initial azimuth
- Intermediate waypoint computation
- Great circle intersection
- Cross-track and along-track distances
- TDOA localization on a sphere
"""

import logging
from functools import lru_cache
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Module logger
_logger = logging.getLogger("pytcl.navigation.great_circle")

# Cache configuration for great circle calculations
_GC_CACHE_DECIMALS = 10  # ~0.01mm precision at Earth's surface
_GC_CACHE_MAXSIZE = 256  # Max cached coordinate pairs


class GreatCircleResult(NamedTuple):
    """
    Result of great circle computation.

    Attributes
    ----------
    distance : float
        Great circle distance in meters.
    azimuth1 : float
        Initial azimuth in radians (from north, clockwise).
    azimuth2 : float
        Final azimuth in radians.
    """

    distance: float
    azimuth1: float
    azimuth2: float


class WaypointResult(NamedTuple):
    """
    Result of waypoint computation.

    Attributes
    ----------
    lat : float
        Latitude of waypoint in radians.
    lon : float
        Longitude of waypoint in radians.
    """

    lat: float
    lon: float


class IntersectionResult(NamedTuple):
    """
    Result of great circle intersection.

    Attributes
    ----------
    lat1 : float
        Latitude of first intersection in radians.
    lon1 : float
        Longitude of first intersection in radians.
    lat2 : float
        Latitude of second intersection (antipodal) in radians.
    lon2 : float
        Longitude of second intersection (antipodal) in radians.
    valid : bool
        True if intersection exists.
    """

    lat1: float
    lon1: float
    lat2: float
    lon2: float
    valid: bool


class CrossTrackResult(NamedTuple):
    """
    Result of cross-track distance computation.

    Attributes
    ----------
    cross_track : float
        Cross-track distance in meters (positive = right of path).
    along_track : float
        Along-track distance from start in meters.
    """

    cross_track: float
    along_track: float


# Default Earth radius (mean radius in meters)
EARTH_RADIUS = 6371000.0


def _quantize_coord(val: float) -> float:
    """Quantize coordinate value for cache key compatibility."""
    return round(val, _GC_CACHE_DECIMALS)


@lru_cache(maxsize=_GC_CACHE_MAXSIZE)
def _gc_distance_cached(
    lat1_q: float,
    lon1_q: float,
    lat2_q: float,
    lon2_q: float,
) -> float:
    """Cached great circle distance computation (internal).

    Uses haversine formula for numerical stability.
    Returns angular distance in radians.
    """
    dlat = lat2_q - lat1_q
    dlon = lon2_q - lon1_q

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_q) * np.cos(lat2_q) * np.sin(dlon / 2) ** 2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@lru_cache(maxsize=_GC_CACHE_MAXSIZE)
def _gc_azimuth_cached(
    lat1_q: float,
    lon1_q: float,
    lat2_q: float,
    lon2_q: float,
) -> float:
    """Cached great circle azimuth computation (internal).

    Returns azimuth in radians [0, 2π).
    """
    dlon = lon2_q - lon1_q

    x = np.sin(dlon) * np.cos(lat2_q)
    y = np.cos(lat1_q) * np.sin(lat2_q) - np.sin(lat1_q) * np.cos(lat2_q) * np.cos(dlon)

    azimuth = np.arctan2(x, y)
    return azimuth % (2 * np.pi)


def great_circle_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS,
) -> float:
    """
    Compute great circle distance between two points.

    Uses the haversine formula for numerical stability at small distances.
    Results are cached for repeated queries with the same coordinates.

    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in radians.
    lat2, lon2 : float
        Second point coordinates in radians.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    float
        Great circle distance in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # New York to London
    >>> lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
    >>> lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)
    >>> dist = great_circle_distance(lat1, lon1, lat2, lon2)
    >>> print(f"Distance: {dist/1000:.0f} km")
    """
    # Use cached angular distance computation
    angular_dist = _gc_distance_cached(
        _quantize_coord(lat1),
        _quantize_coord(lon1),
        _quantize_coord(lat2),
        _quantize_coord(lon2),
    )
    return radius * angular_dist


def great_circle_azimuth(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute initial azimuth (bearing) from point 1 to point 2.

    Results are cached for repeated queries with the same coordinates.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.

    Returns
    -------
    float
        Initial azimuth in radians (from north, clockwise, 0 to 2π).

    Examples
    --------
    >>> import numpy as np
    >>> # Bearing from New York to London
    >>> lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
    >>> lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)
    >>> az = great_circle_azimuth(lat1, lon1, lat2, lon2)
    >>> print(f"Initial bearing: {np.degrees(az):.1f}°")
    """
    return _gc_azimuth_cached(
        _quantize_coord(lat1),
        _quantize_coord(lon1),
        _quantize_coord(lat2),
        _quantize_coord(lon2),
    )


def great_circle_inverse(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS,
) -> GreatCircleResult:
    """
    Solve the inverse great circle problem.

    Given two points, compute distance and azimuths.

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
    GreatCircleResult
        Distance and azimuths.

    Examples
    --------
    >>> lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)  # NYC
    >>> lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)   # London
    >>> result = great_circle_inverse(lat1, lon1, lat2, lon2)
    >>> result.distance > 5000000  # Over 5000 km
    True
    """
    distance = great_circle_distance(lat1, lon1, lat2, lon2, radius)
    azimuth1 = great_circle_azimuth(lat1, lon1, lat2, lon2)
    azimuth2 = great_circle_azimuth(lat2, lon2, lat1, lon1)

    # Convert back azimuth to forward direction at destination
    azimuth2 = (azimuth2 + np.pi) % (2 * np.pi)

    return GreatCircleResult(distance, azimuth1, azimuth2)


def great_circle_waypoint(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    fraction: float,
) -> WaypointResult:
    """
    Compute intermediate waypoint along a great circle.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    fraction : float
        Fraction of distance (0 = start, 1 = end).

    Returns
    -------
    WaypointResult
        Latitude and longitude of waypoint.

    Examples
    --------
    >>> import numpy as np
    >>> # Midpoint between New York and London
    >>> lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
    >>> lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)
    >>> mid = great_circle_waypoint(lat1, lon1, lat2, lon2, 0.5)
    >>> print(f"Midpoint: {np.degrees(mid.lat):.2f}°, {np.degrees(mid.lon):.2f}°")
    """
    # Angular distance
    d = great_circle_distance(lat1, lon1, lat2, lon2, radius=1.0)

    if d < 1e-12:
        return WaypointResult(lat1, lon1)

    a = np.sin((1 - fraction) * d) / np.sin(d)
    b = np.sin(fraction * d) / np.sin(d)

    x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
    y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
    z = a * np.sin(lat1) + b * np.sin(lat2)

    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)

    return WaypointResult(float(lat), float(lon))


def great_circle_waypoints(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    n_points: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute multiple waypoints along a great circle.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    lat2, lon2 : float
        Destination point coordinates in radians.
    n_points : int
        Number of waypoints (including start and end).

    Returns
    -------
    lats, lons : ndarray
        Arrays of waypoint latitudes and longitudes in radians.

    Examples
    --------
    >>> lat1, lon1 = 0.0, 0.0
    >>> lat2, lon2 = np.pi/4, np.pi/4
    >>> lats, lons = great_circle_waypoints(lat1, lon1, lat2, lon2, 5)
    >>> len(lats)
    5
    """
    fractions = np.linspace(0, 1, n_points)
    lats = np.zeros(n_points)
    lons = np.zeros(n_points)

    for i, f in enumerate(fractions):
        wp = great_circle_waypoint(lat1, lon1, lat2, lon2, f)
        lats[i] = wp.lat
        lons[i] = wp.lon

    return lats, lons


def great_circle_direct(
    lat1: float,
    lon1: float,
    azimuth: float,
    distance: float,
    radius: float = EARTH_RADIUS,
) -> WaypointResult:
    """
    Solve the direct great circle problem.

    Given starting point, azimuth, and distance, find destination.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point coordinates in radians.
    azimuth : float
        Initial azimuth in radians (from north, clockwise).
    distance : float
        Distance in meters.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    WaypointResult
        Destination latitude and longitude.

    Examples
    --------
    >>> import numpy as np
    >>> # 1000 km northeast from origin
    >>> lat, lon = 0.0, 0.0
    >>> az = np.radians(45)  # Northeast
    >>> dest = great_circle_direct(lat, lon, az, 1000000)
    >>> print(f"Destination: {np.degrees(dest.lat):.2f}°, {np.degrees(dest.lon):.2f}°")
    """
    d = distance / radius  # Angular distance

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(azimuth)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(azimuth) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2),
    )

    # Normalize longitude to [-π, π)
    lon2 = ((lon2 + np.pi) % (2 * np.pi)) - np.pi

    return WaypointResult(float(lat2), float(lon2))


def cross_track_distance(
    lat_point: float,
    lon_point: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS,
) -> CrossTrackResult:
    """
    Compute cross-track and along-track distances.

    The cross-track distance is the perpendicular distance from a point
    to the great circle path between two other points.

    Parameters
    ----------
    lat_point, lon_point : float
        Point coordinates in radians.
    lat1, lon1 : float
        Path start coordinates in radians.
    lat2, lon2 : float
        Path end coordinates in radians.
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    CrossTrackResult
        Cross-track distance (positive = right of path) and along-track distance.

    Notes
    -----
    Positive cross-track means the point is to the right of the path
    (when traveling from start to end).

    Examples
    --------
    >>> # Point near a path from origin to northeast
    >>> lat_pt, lon_pt = np.radians(5), np.radians(2)
    >>> lat1, lon1 = 0.0, 0.0
    >>> lat2, lon2 = np.radians(10), np.radians(10)
    >>> result = cross_track_distance(lat_pt, lon_pt, lat1, lon1, lat2, lon2)
    >>> abs(result.cross_track) < 500000  # Within 500 km
    True
    """
    # Angular distance from start to point
    d13 = great_circle_distance(lat1, lon1, lat_point, lon_point, radius=1.0)

    # Bearings
    theta13 = great_circle_azimuth(lat1, lon1, lat_point, lon_point)
    theta12 = great_circle_azimuth(lat1, lon1, lat2, lon2)

    # Cross-track distance
    dxt = np.arcsin(np.sin(d13) * np.sin(theta13 - theta12))

    # Along-track distance
    dat = np.arccos(np.cos(d13) / np.cos(dxt))

    return CrossTrackResult(
        cross_track=float(dxt * radius), along_track=float(dat * radius)
    )


def great_circle_intersect(
    lat1: float,
    lon1: float,
    azimuth1: float,
    lat2: float,
    lon2: float,
    azimuth2: float,
) -> IntersectionResult:
    """
    Find intersection of two great circles.

    Given two points with initial bearings, find where the great circles
    defined by those bearings intersect.

    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in radians.
    azimuth1 : float
        Bearing from first point in radians.
    lat2, lon2 : float
        Second point coordinates in radians.
    azimuth2 : float
        Bearing from second point in radians.

    Returns
    -------
    IntersectionResult
        Two intersection points (antipodal) and validity flag.

    Notes
    -----
    Great circles always intersect at two antipodal points (unless they
    are identical or parallel). The returned points are the intersections
    closest to the given points.

    Examples
    --------
    >>> lat1, lon1 = 0.0, 0.0
    >>> az1 = np.radians(45)  # Northeast
    >>> lat2, lon2 = 0.0, np.radians(10)
    >>> az2 = np.radians(315)  # Northwest
    >>> result = great_circle_intersect(lat1, lon1, az1, lat2, lon2, az2)
    >>> result.valid
    True
    """

    # Convert to Cartesian unit vectors
    def to_cartesian(lat: Any, lon: Any) -> NDArray[np.float64]:
        return np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )

    # Normal vectors to the great circles
    p1 = to_cartesian(lat1, lon1)
    p2 = to_cartesian(lat2, lon2)

    # Direction vectors along the great circles
    # North pole in Cartesian
    north = np.array([0.0, 0.0, 1.0])

    # East vector at point 1
    east1 = np.cross(north, p1)
    if np.linalg.norm(east1) > 1e-12:
        east1 = east1 / np.linalg.norm(east1)
    else:
        east1 = np.array([0.0, 1.0, 0.0])

    # North vector at point 1
    north1 = np.cross(p1, east1)

    # Direction of great circle from point 1
    d1 = np.sin(azimuth1) * east1 + np.cos(azimuth1) * north1

    # Same for point 2
    east2 = np.cross(north, p2)
    if np.linalg.norm(east2) > 1e-12:
        east2 = east2 / np.linalg.norm(east2)
    else:
        east2 = np.array([0.0, 1.0, 0.0])

    north2 = np.cross(p2, east2)
    d2 = np.sin(azimuth2) * east2 + np.cos(azimuth2) * north2

    # Normal vectors to the great circle planes
    n1 = np.cross(p1, d1)
    n2 = np.cross(p2, d2)

    # Intersection is the cross product of normals
    intersection = np.cross(n1, n2)
    norm = np.linalg.norm(intersection)

    if norm < 1e-12:
        # Parallel or identical great circles
        return IntersectionResult(0.0, 0.0, 0.0, 0.0, False)

    intersection = intersection / norm

    # Two antipodal points
    lat_i1 = np.arcsin(intersection[2])
    lon_i1 = np.arctan2(intersection[1], intersection[0])

    lat_i2 = -lat_i1
    lon_i2 = ((lon_i1 + np.pi) % (2 * np.pi)) - np.pi

    return IntersectionResult(
        float(lat_i1), float(lon_i1), float(lat_i2), float(lon_i2), True
    )


def great_circle_path_intersect(
    lat1a: float,
    lon1a: float,
    lat2a: float,
    lon2a: float,
    lat1b: float,
    lon1b: float,
    lat2b: float,
    lon2b: float,
) -> IntersectionResult:
    """
    Find intersection of two great circle paths (defined by endpoints).

    Parameters
    ----------
    lat1a, lon1a : float
        Start of first path in radians.
    lat2a, lon2a : float
        End of first path in radians.
    lat1b, lon1b : float
        Start of second path in radians.
    lat2b, lon2b : float
        End of second path in radians.

    Returns
    -------
    IntersectionResult
        Intersection points and validity.

    Examples
    --------
    >>> # Two crossing paths
    >>> result = great_circle_path_intersect(
    ...     0.0, 0.0, np.radians(10), np.radians(10),  # Path A
    ...     0.0, np.radians(10), np.radians(10), 0.0    # Path B
    ... )
    >>> result.valid
    True
    """
    # Get bearings from start points
    az1 = great_circle_azimuth(lat1a, lon1a, lat2a, lon2a)
    az2 = great_circle_azimuth(lat1b, lon1b, lat2b, lon2b)

    return great_circle_intersect(lat1a, lon1a, az1, lat1b, lon1b, az2)


def great_circle_tdoa_loc(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    lat3: float,
    lon3: float,
    tdoa12: float,
    tdoa13: float,
    speed: float = 299792458.0,
    radius: float = EARTH_RADIUS,
) -> Tuple[Optional[WaypointResult], Optional[WaypointResult]]:
    """
    TDOA localization on a sphere.

    Given three receivers and two time-difference-of-arrival measurements,
    estimate the emitter location.

    Parameters
    ----------
    lat1, lon1 : float
        First receiver coordinates in radians.
    lat2, lon2 : float
        Second receiver coordinates in radians.
    lat3, lon3 : float
        Third receiver coordinates in radians.
    tdoa12 : float
        Time difference of arrival between receivers 1 and 2 (seconds).
        Positive means signal arrived at receiver 1 first.
    tdoa13 : float
        Time difference of arrival between receivers 1 and 3 (seconds).
    speed : float, optional
        Signal propagation speed in m/s (default: speed of light).
    radius : float, optional
        Sphere radius in meters (default: 6371 km).

    Returns
    -------
    loc1, loc2 : Optional[WaypointResult]
        Two possible emitter locations, or None if no solution.

    Notes
    -----
    TDOA localization finds the intersection of two hyperbolas. On a sphere,
    these are represented by the locus of points where the difference in
    distance to two receivers is constant.

    This implementation uses a grid search followed by Newton-Raphson
    refinement for robustness.
    """
    # Range differences
    delta_r12 = tdoa12 * speed
    delta_r13 = tdoa13 * speed

    # Convert to angular differences
    delta_d12 = delta_r12 / radius
    delta_d13 = delta_r13 / radius

    def objective(lat: Any, lon: Any) -> Any:
        """Objective function: difference between computed and observed TDOAs."""
        d1 = great_circle_distance(lat, lon, lat1, lon1, radius=1.0)
        d2 = great_circle_distance(lat, lon, lat2, lon2, radius=1.0)
        d3 = great_circle_distance(lat, lon, lat3, lon3, radius=1.0)

        err12 = (d1 - d2) - delta_d12
        err13 = (d1 - d3) - delta_d13

        return err12**2 + err13**2

    # Grid search for initial guess
    best_lat, best_lon = 0.0, 0.0
    best_val = float("inf")

    lats = np.linspace(-np.pi / 2, np.pi / 2, 37)
    lons = np.linspace(-np.pi, np.pi, 73)

    for lat in lats:
        for lon in lons:
            val = objective(lat, lon)
            if val < best_val:
                best_val = val
                best_lat = lat
                best_lon = lon

    # Newton-Raphson refinement
    lat, lon = best_lat, best_lon
    h = 1e-6

    for _ in range(50):
        f = objective(lat, lon)

        if f < 1e-20:
            break

        # Numerical gradient
        df_dlat = (objective(lat + h, lon) - f) / h
        df_dlon = (objective(lat, lon + h) - f) / h

        grad_norm = np.sqrt(df_dlat**2 + df_dlon**2)
        if grad_norm < 1e-12:
            break

        # Step (gradient descent with backtracking)
        step = 0.1
        for _ in range(10):
            new_lat = np.clip(lat - step * df_dlat / grad_norm, -np.pi / 2, np.pi / 2)
            new_lon = lon - step * df_dlon / grad_norm

            if objective(new_lat, new_lon) < f:
                lat, lon = new_lat, new_lon
                break
            step *= 0.5

    # Normalize longitude
    lon = ((lon + np.pi) % (2 * np.pi)) - np.pi

    # Check if solution is valid
    if objective(lat, lon) > 1e-6:
        return None, None

    loc1 = WaypointResult(float(lat), float(lon))

    # Find second solution (antipodal search)
    lat2_init, lon2_init = -lat, ((lon + np.pi) % (2 * np.pi)) - np.pi
    lat2, lon2 = lat2_init, lon2_init

    for _ in range(50):
        f = objective(lat2, lon2)
        if f < 1e-20:
            break

        df_dlat = (objective(lat2 + h, lon2) - f) / h
        df_dlon = (objective(lat2, lon2 + h) - f) / h

        grad_norm = np.sqrt(df_dlat**2 + df_dlon**2)
        if grad_norm < 1e-12:
            break

        step = 0.1
        for _ in range(10):
            new_lat = np.clip(lat2 - step * df_dlat / grad_norm, -np.pi / 2, np.pi / 2)
            new_lon = lon2 - step * df_dlon / grad_norm
            if objective(new_lat, new_lon) < f:
                lat2, lon2 = new_lat, new_lon
                break
            step *= 0.5

    lon2 = ((lon2 + np.pi) % (2 * np.pi)) - np.pi

    if objective(lat2, lon2) > 1e-6:
        return loc1, None

    # Check if second solution is actually different
    if great_circle_distance(lat, lon, lat2, lon2, radius=1.0) < 0.01:
        return loc1, None

    loc2 = WaypointResult(float(lat2), float(lon2))
    return loc1, loc2


def angular_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute angular distance (central angle) between two points.

    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in radians.
    lat2, lon2 : float
        Second point coordinates in radians.

    Returns
    -------
    float
        Angular distance in radians.

    Examples
    --------
    Compute angular distance between New York and London:

    >>> import numpy as np
    >>> # NYC: 40.7°N, 74.0°W; London: 51.5°N, 0.1°W
    >>> lat1, lon1 = np.radians(40.7), np.radians(-74.0)
    >>> lat2, lon2 = np.radians(51.5), np.radians(-0.1)
    >>> angle = angular_distance(lat1, lon1, lat2, lon2)
    >>> np.degrees(angle)  # about 50 degrees
    49.9...

    See Also
    --------
    great_circle_distance : Compute distance on sphere with given radius.
    """
    return great_circle_distance(lat1, lon1, lat2, lon2, radius=1.0)


def destination_point(
    lat: float,
    lon: float,
    bearing: float,
    angular_distance: float,
) -> WaypointResult:
    """
    Compute destination point given start, bearing, and angular distance.

    Parameters
    ----------
    lat, lon : float
        Starting point coordinates in radians.
    bearing : float
        Initial bearing in radians.
    angular_distance : float
        Angular distance in radians (distance/radius).

    Returns
    -------
    WaypointResult
        Destination coordinates.

    Examples
    --------
    >>> lat, lon = 0.0, 0.0
    >>> bearing = np.radians(90)  # Due East
    >>> ang_dist = np.radians(10)  # 10 degrees
    >>> dest = destination_point(lat, lon, bearing, ang_dist)
    >>> np.degrees(dest.lon)  # Should be ~10 degrees East
    10.0
    """
    lat2 = np.arcsin(
        np.sin(lat) * np.cos(angular_distance)
        + np.cos(lat) * np.sin(angular_distance) * np.cos(bearing)
    )

    lon2 = lon + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat),
        np.cos(angular_distance) - np.sin(lat) * np.sin(lat2),
    )

    lon2 = ((lon2 + np.pi) % (2 * np.pi)) - np.pi

    return WaypointResult(float(lat2), float(lon2))


def clear_great_circle_cache() -> None:
    """Clear all great circle computation caches.

    This can be useful to free memory after processing large datasets
    or when cache statistics are being monitored.
    """
    _gc_distance_cached.cache_clear()
    _gc_azimuth_cached.cache_clear()
    _logger.debug("Great circle caches cleared")


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics for great circle computations.

    Returns
    -------
    dict[str, Any]
        Dictionary with cache statistics for distance and azimuth caches.
    """
    return {
        "distance": _gc_distance_cached.cache_info()._asdict(),
        "azimuth": _gc_azimuth_cached.cache_info()._asdict(),
    }


__all__ = [
    # Constants
    "EARTH_RADIUS",
    # Result types
    "GreatCircleResult",
    "WaypointResult",
    "IntersectionResult",
    "CrossTrackResult",
    # Distance and bearing
    "great_circle_distance",
    "great_circle_azimuth",
    "great_circle_inverse",
    "angular_distance",
    # Waypoints
    "great_circle_waypoint",
    "great_circle_waypoints",
    "great_circle_direct",
    "destination_point",
    # Cross-track
    "cross_track_distance",
    # Intersection
    "great_circle_intersect",
    "great_circle_path_intersect",
    # TDOA
    "great_circle_tdoa_loc",
    # Cache management
    "clear_great_circle_cache",
    "get_cache_info",
]
