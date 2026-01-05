"""
Geodetic calculations for navigation and tracking.

This module provides geodetic utilities including:
- Geodetic to ECEF conversions
- Direct and inverse geodetic problems
- Local tangent plane (ENU/NED) conversions
- Earth ellipsoid parameters
"""

import logging
from functools import lru_cache
from typing import Any, NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Module logger
_logger = logging.getLogger("pytcl.navigation.geodesy")

# Cache configuration for Vincenty geodetic calculations
_VINCENTY_CACHE_DECIMALS = 10  # ~0.01mm precision
_VINCENTY_CACHE_MAXSIZE = 128  # Max cached coordinate pairs


class Ellipsoid(NamedTuple):
    """
    Earth ellipsoid parameters.

    Attributes
    ----------
    a : float
        Semi-major axis (equatorial radius) in meters.
    f : float
        Flattening (a-b)/a.
    """

    a: float
    f: float

    @property
    def b(self) -> float:
        """Semi-minor axis (polar radius) in meters."""
        return self.a * (1 - self.f)

    @property
    def e2(self) -> float:
        """First eccentricity squared."""
        return self.f * (2 - self.f)

    @property
    def ep2(self) -> float:
        """Second eccentricity squared."""
        return self.e2 / (1 - self.e2)


# Standard ellipsoids
WGS84 = Ellipsoid(a=6378137.0, f=1.0 / 298.257223563)
GRS80 = Ellipsoid(a=6378137.0, f=1.0 / 298.257222101)
SPHERE = Ellipsoid(a=6371000.0, f=0.0)


def _quantize_geodetic(val: float) -> float:
    """Quantize geodetic coordinate for cache key compatibility."""
    return round(val, _VINCENTY_CACHE_DECIMALS)


@lru_cache(maxsize=_VINCENTY_CACHE_MAXSIZE)
def _inverse_geodetic_cached(
    lat1_q: float,
    lon1_q: float,
    lat2_q: float,
    lon2_q: float,
    a: float,
    f: float,
) -> Tuple[float, float, float]:
    """Cached Vincenty inverse geodetic computation (internal).

    Returns (distance, azimuth1, azimuth2).
    """
    b = a * (1 - f)

    # Reduced latitudes
    U1 = np.arctan((1 - f) * np.tan(lat1_q))
    U2 = np.arctan((1 - f) * np.tan(lat2_q))
    sin_U1, cos_U1 = np.sin(U1), np.cos(U1)
    sin_U2, cos_U2 = np.sin(U2), np.cos(U2)

    L = lon2_q - lon1_q
    lam = L

    for _ in range(100):
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)

        sin_sigma = np.sqrt(
            (cos_U2 * sin_lam) ** 2 + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam) ** 2
        )

        if sin_sigma == 0:
            # Coincident points
            return 0.0, 0.0, 0.0

        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha**2

        if cos2_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha

        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))

        lam_new = L + (1 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )

        if abs(lam_new - lam) < 1e-12:
            break
        lam = lam_new

    u2 = cos2_alpha * (a**2 - b**2) / b**2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + B
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                - B
                / 6
                * cos_2sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_2sigma_m**2)
            )
        )
    )

    distance = b * A * (sigma - delta_sigma)

    # Azimuths
    azimuth1 = np.arctan2(cos_U2 * sin_lam, cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam)
    azimuth2 = np.arctan2(
        cos_U1 * sin_lam, -sin_U1 * cos_U2 + cos_U1 * sin_U2 * cos_lam
    )

    return float(distance), float(azimuth1), float(azimuth2)


@lru_cache(maxsize=_VINCENTY_CACHE_MAXSIZE)
def _direct_geodetic_cached(
    lat1_q: float,
    lon1_q: float,
    azimuth_q: float,
    distance_q: float,
    a: float,
    f: float,
) -> Tuple[float, float, float]:
    """Cached Vincenty direct geodetic computation (internal).

    Returns (lat2, lon2, azimuth2).
    """
    b = a * (1 - f)

    sin_alpha1 = np.sin(azimuth_q)
    cos_alpha1 = np.cos(azimuth_q)

    # Reduced latitude
    tan_U1 = (1 - f) * np.tan(lat1_q)
    cos_U1 = 1.0 / np.sqrt(1 + tan_U1**2)
    sin_U1 = tan_U1 * cos_U1

    sigma1 = np.arctan2(tan_U1, cos_alpha1)
    sin_alpha = cos_U1 * sin_alpha1
    cos2_alpha = 1 - sin_alpha**2

    u2 = cos2_alpha * (a**2 - b**2) / b**2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    sigma = distance_q / (b * A)

    for _ in range(100):
        cos_2sigma_m = np.cos(2 * sigma1 + sigma)
        sin_sigma = np.sin(sigma)
        cos_sigma = np.cos(sigma)

        delta_sigma = (
            B
            * sin_sigma
            * (
                cos_2sigma_m
                + B
                / 4
                * (
                    cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                    - B
                    / 6
                    * cos_2sigma_m
                    * (-3 + 4 * sin_sigma**2)
                    * (-3 + 4 * cos_2sigma_m**2)
                )
            )
        )

        sigma_new = distance_q / (b * A) + delta_sigma
        if abs(sigma_new - sigma) < 1e-12:
            break
        sigma = sigma_new

    cos_2sigma_m = np.cos(2 * sigma1 + sigma)
    sin_sigma = np.sin(sigma)
    cos_sigma = np.cos(sigma)

    sin_U2 = sin_U1 * cos_sigma + cos_U1 * sin_sigma * cos_alpha1
    lat2 = np.arctan2(
        sin_U2,
        (1 - f)
        * np.sqrt(
            sin_alpha**2 + (sin_U1 * sin_sigma - cos_U1 * cos_sigma * cos_alpha1) ** 2
        ),
    )

    lam = np.arctan2(
        sin_sigma * sin_alpha1, cos_U1 * cos_sigma - sin_U1 * sin_sigma * cos_alpha1
    )

    C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
    L = lam - (1 - C) * f * sin_alpha * (
        sigma
        + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
    )

    lon2 = lon1_q + L

    # Back azimuth
    azimuth2 = np.arctan2(
        sin_alpha, -sin_U1 * sin_sigma + cos_U1 * cos_sigma * cos_alpha1
    )

    return float(lat2), float(lon2), float(azimuth2)


def geodetic_to_ecef(
    lat: ArrayLike,
    lon: ArrayLike,
    alt: ArrayLike,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed).

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in radians.
    lon : array_like
        Geodetic longitude in radians.
    alt : array_like
        Altitude above ellipsoid in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    x, y, z : ndarray
        ECEF coordinates in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # Philadelphia (40°N, 75°W) at 100m altitude
    >>> lat, lon, alt = np.radians(40.0), np.radians(-75.0), 100.0
    >>> x, y, z = geodetic_to_ecef(lat, lon, alt)
    >>> x / 1e6  # ~1.2 million meters
    1.24...
    >>> # Equator at prime meridian
    >>> x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
    >>> x  # Semi-major axis (equatorial radius)
    6378137.0
    >>> y, z
    (0.0, 0.0)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Radius of curvature in the prime vertical
    N = ellipsoid.a / np.sqrt(1 - ellipsoid.e2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - ellipsoid.e2) + alt) * sin_lat

    return x, y, z


def ecef_to_geodetic(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert ECEF coordinates to geodetic.

    Parameters
    ----------
    x, y, z : array_like
        ECEF coordinates in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    lat : ndarray
        Geodetic latitude in radians.
    lon : ndarray
        Geodetic longitude in radians.
    alt : ndarray
        Altitude above ellipsoid in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # Point on equator at prime meridian
    >>> lat, lon, alt = ecef_to_geodetic(6378137.0, 0.0, 0.0)
    >>> np.degrees(lat), np.degrees(lon), alt
    (0.0, 0.0, 0.0)
    >>> # Round-trip conversion
    >>> x, y, z = geodetic_to_ecef(np.radians(45.0), np.radians(90.0), 1000.0)
    >>> lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
    >>> np.degrees(lat2), np.degrees(lon2), alt2
    (45.0..., 90.0..., 1000.0...)

    Notes
    -----
    Uses Bowring's iterative algorithm for robust conversion.

    References
    ----------
    .. [1] Bowring, B.R., "Transformation from spatial to geographical
           coordinates", Survey Review, 1976.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    a = ellipsoid.a
    e2 = ellipsoid.e2

    # Longitude
    lon = np.arctan2(y, x)

    # Distance from z-axis
    p = np.sqrt(x**2 + y**2)

    # Bowring's method (iterative)
    # Initial approximation
    lat = np.arctan2(z, p * (1 - e2))

    for _ in range(10):  # Usually converges in 2-3 iterations
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        lat_new = np.arctan2(z + e2 * N * sin_lat, p)
        if np.all(np.abs(lat_new - lat) < 1e-12):
            break
        lat = lat_new

    # Altitude
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    # Handle points near poles
    with np.errstate(divide="ignore", invalid="ignore"):
        alt = np.where(
            np.abs(cos_lat) > 1e-10,
            p / cos_lat - N,
            np.abs(z) / np.abs(sin_lat) - N * (1 - e2),
        )

    return lat, lon, alt


def ecef_to_enu(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert ECEF to local ENU (East-North-Up) coordinates.

    Parameters
    ----------
    x, y, z : array_like
        ECEF coordinates in meters.
    lat_ref : float
        Reference latitude in radians.
    lon_ref : float
        Reference longitude in radians.
    alt_ref : float
        Reference altitude in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    east, north, up : ndarray
        ENU coordinates in meters relative to reference point.

    Examples
    --------
    >>> import numpy as np
    >>> # Reference point: Philadelphia
    >>> lat_ref, lon_ref, alt_ref = np.radians(40.0), np.radians(-75.0), 0.0
    >>> # Target point slightly east
    >>> lat, lon, alt = np.radians(40.0), np.radians(-74.99), 0.0
    >>> x, y, z = geodetic_to_ecef(lat, lon, alt)
    >>> e, n, u = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)
    >>> e  # East displacement in meters
    850...
    >>> abs(n) < 10  # North displacement should be ~0
    True
    >>> abs(u) < 10  # Up displacement should be ~0
    True
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # Reference point in ECEF
    x_ref, y_ref, z_ref = geodetic_to_ecef(lat_ref, lon_ref, alt_ref, ellipsoid)

    # Vector from reference to point
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref

    # Rotation matrix
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # ENU = R @ [dx, dy, dz]
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def enu_to_ecef(
    east: ArrayLike,
    north: ArrayLike,
    up: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert local ENU to ECEF coordinates.

    Parameters
    ----------
    east, north, up : array_like
        ENU coordinates in meters relative to reference point.
    lat_ref : float
        Reference latitude in radians.
    lon_ref : float
        Reference longitude in radians.
    alt_ref : float
        Reference altitude in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    x, y, z : ndarray
        ECEF coordinates in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # Reference point
    >>> lat_ref, lon_ref, alt_ref = np.radians(40.0), np.radians(-75.0), 0.0
    >>> # 1 km east, 500 m north, 100 m up
    >>> x, y, z = enu_to_ecef(1000.0, 500.0, 100.0, lat_ref, lon_ref, alt_ref)
    >>> # Convert back to verify
    >>> e, n, u = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)
    >>> e, n, u
    (1000.0..., 500.0..., 100.0...)
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Reference point in ECEF
    x_ref, y_ref, z_ref = geodetic_to_ecef(lat_ref, lon_ref, alt_ref, ellipsoid)

    # Rotation matrix (transpose of ENU->ECEF)
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # ECEF = R^T @ [e, n, u] + [x_ref, y_ref, z_ref]
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    return x_ref + dx, y_ref + dy, z_ref + dz


def ecef_to_ned(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert ECEF to local NED (North-East-Down) coordinates.

    Parameters
    ----------
    x, y, z : array_like
        ECEF coordinates in meters.
    lat_ref : float
        Reference latitude in radians.
    lon_ref : float
        Reference longitude in radians.
    alt_ref : float
        Reference altitude in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    north, east, down : ndarray
        NED coordinates in meters relative to reference point.

    Examples
    --------
    >>> import numpy as np
    >>> # Reference point
    >>> lat_ref, lon_ref, alt_ref = np.radians(40.0), np.radians(-75.0), 0.0
    >>> # Target above reference
    >>> x, y, z = geodetic_to_ecef(lat_ref, lon_ref, 1000.0)  # 1km above
    >>> n, e, d = ecef_to_ned(x, y, z, lat_ref, lon_ref, alt_ref)
    >>> abs(n) < 1, abs(e) < 1, d  # Should be ~0, ~0, -1000
    (True, True, -1000.0...)
    """
    east, north, up = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref, ellipsoid)
    return north, east, -up


def ned_to_ecef(
    north: ArrayLike,
    east: ArrayLike,
    down: ArrayLike,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert local NED to ECEF coordinates.

    Parameters
    ----------
    north, east, down : array_like
        NED coordinates in meters relative to reference point.
    lat_ref : float
        Reference latitude in radians.
    lon_ref : float
        Reference longitude in radians.
    alt_ref : float
        Reference altitude in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    x, y, z : ndarray
        ECEF coordinates in meters.

    Examples
    --------
    >>> import numpy as np
    >>> lat_ref, lon_ref, alt_ref = np.radians(40.0), np.radians(-75.0), 0.0
    >>> # 100m north, 50m east, 10m down
    >>> x, y, z = ned_to_ecef(100.0, 50.0, 10.0, lat_ref, lon_ref, alt_ref)
    >>> # Verify round-trip
    >>> n, e, d = ecef_to_ned(x, y, z, lat_ref, lon_ref, alt_ref)
    >>> n, e, d
    (100.0..., 50.0..., 10.0...)
    """
    return enu_to_ecef(
        east, north, -np.asarray(down), lat_ref, lon_ref, alt_ref, ellipsoid
    )


def direct_geodetic(
    lat1: float,
    lon1: float,
    azimuth: float,
    distance: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """
    Solve the direct geodetic problem (Vincenty).

    Given a starting point, azimuth, and distance, find the destination point.
    Results are cached for repeated queries with the same parameters.

    Parameters
    ----------
    lat1 : float
        Starting latitude in radians.
    lon1 : float
        Starting longitude in radians.
    azimuth : float
        Forward azimuth in radians (from north, clockwise).
    distance : float
        Distance in meters.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    lat2 : float
        Destination latitude in radians.
    lon2 : float
        Destination longitude in radians.
    azimuth2 : float
        Back azimuth at destination in radians.

    Examples
    --------
    >>> import numpy as np
    >>> # From New York, travel 1000 km northeast
    >>> lat1, lon1 = np.radians(40.7), np.radians(-74.0)
    >>> azimuth = np.radians(45)  # Northeast
    >>> distance = 1_000_000  # 1000 km
    >>> lat2, lon2, az2 = direct_geodetic(lat1, lon1, azimuth, distance)
    >>> np.degrees(lat2), np.degrees(lon2)  # Destination
    (47.0..., -62.6...)

    References
    ----------
    .. [1] Vincenty, T., "Direct and Inverse Solutions of Geodesics on the
           Ellipsoid with Application of Nested Equations", Survey Review, 1975.
    """
    return _direct_geodetic_cached(
        _quantize_geodetic(lat1),
        _quantize_geodetic(lon1),
        _quantize_geodetic(azimuth),
        round(distance, 3),  # 1mm precision for distance
        ellipsoid.a,
        ellipsoid.f,
    )


def inverse_geodetic(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """
    Solve the inverse geodetic problem (Vincenty).

    Given two points, find the distance and azimuths between them.
    Results are cached for repeated queries with the same coordinates.

    Parameters
    ----------
    lat1 : float
        Starting latitude in radians.
    lon1 : float
        Starting longitude in radians.
    lat2 : float
        Destination latitude in radians.
    lon2 : float
        Destination longitude in radians.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    distance : float
        Geodesic distance in meters.
    azimuth1 : float
        Forward azimuth at start in radians.
    azimuth2 : float
        Back azimuth at destination in radians.

    Examples
    --------
    >>> import numpy as np
    >>> # Distance from New York to London
    >>> lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)  # NYC
    >>> lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)   # London
    >>> dist, az1, az2 = inverse_geodetic(lat1, lon1, lat2, lon2)
    >>> dist / 1000  # Distance in km
    5570...
    >>> np.degrees(az1)  # Initial heading from NYC
    51.2...

    Notes
    -----
    May fail to converge for nearly antipodal points.

    References
    ----------
    .. [1] Vincenty, T., "Direct and Inverse Solutions of Geodesics on the
           Ellipsoid with Application of Nested Equations", Survey Review, 1975.
    """
    return _inverse_geodetic_cached(
        _quantize_geodetic(lat1),
        _quantize_geodetic(lon1),
        _quantize_geodetic(lat2),
        _quantize_geodetic(lon2),
        ellipsoid.a,
        ellipsoid.f,
    )


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = 6371000.0,
) -> float:
    """
    Compute great-circle distance using the haversine formula.

    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in radians.
    lat2, lon2 : float
        Second point coordinates in radians.
    radius : float, optional
        Earth radius in meters (default: 6371 km).

    Returns
    -------
    float
        Great-circle distance in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # Distance from equator to 45°N along prime meridian
    >>> lat1, lon1 = 0.0, 0.0
    >>> lat2, lon2 = np.radians(45.0), 0.0
    >>> dist = haversine_distance(lat1, lon1, lat2, lon2)
    >>> dist / 1000  # ~5000 km
    5003...
    >>> # Same point -> 0 distance
    >>> haversine_distance(0.0, 0.0, 0.0, 0.0)
    0.0

    Notes
    -----
    This is a spherical approximation. For higher accuracy on an ellipsoid,
    use inverse_geodetic().
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius * c


def clear_geodesy_cache() -> None:
    """Clear all geodesy computation caches.

    This can be useful to free memory after processing large datasets
    or when cache statistics are being monitored.
    """
    _inverse_geodetic_cached.cache_clear()
    _direct_geodetic_cached.cache_clear()
    _logger.debug("Geodesy caches cleared")


def get_geodesy_cache_info() -> dict[str, Any]:
    """Get cache statistics for geodesy computations.

    Returns
    -------
    dict[str, Any]
        Dictionary with cache statistics for inverse and direct geodetic caches.
    """
    return {
        "inverse_geodetic": _inverse_geodetic_cached.cache_info()._asdict(),
        "direct_geodetic": _direct_geodetic_cached.cache_info()._asdict(),
    }


__all__ = [
    # Ellipsoids
    "Ellipsoid",
    "WGS84",
    "GRS80",
    "SPHERE",
    # Coordinate conversions
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_enu",
    "enu_to_ecef",
    "ecef_to_ned",
    "ned_to_ecef",
    # Geodetic problems
    "direct_geodetic",
    "inverse_geodetic",
    "haversine_distance",
    # Cache management
    "clear_geodesy_cache",
    "get_geodesy_cache_info",
]
