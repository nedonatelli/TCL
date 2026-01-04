"""
Map Projections for Tracking and Navigation.

This module provides map projection functions commonly used in tracking,
navigation, and geospatial applications. All projections convert between
geodetic coordinates (latitude, longitude) and planar map coordinates (x, y).

Projections included:
- Mercator: Conformal cylindrical projection, good for small-scale maps
- Transverse Mercator: Conformal projection for UTM zones
- UTM: Universal Transverse Mercator with zone handling
- Stereographic: Conformal azimuthal projection, good for polar regions
- Lambert Conformal Conic: Conformal conic projection for mid-latitudes
- Azimuthal Equidistant: Preserves distances from center point

All angles are in radians unless otherwise noted.

References
----------
.. [1] Snyder, J. P. "Map Projections: A Working Manual." U.S. Geological
       Survey Professional Paper 1395, 1987.
.. [2] NIMA Technical Report 8350.2, "Department of Defense World Geodetic
       System 1984," Third Edition, 2000.
.. [3] Karney, C. F. F. "Transverse Mercator with an accuracy of a few
       nanometers." Journal of Geodesy 85.8 (2011): 475-485.
"""

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis (m)
WGS84_E = np.sqrt(2 * WGS84_F - WGS84_F**2)  # First eccentricity
WGS84_E2 = WGS84_E**2  # First eccentricity squared
WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2)  # Second eccentricity squared


class ProjectionResult(NamedTuple):
    """Result of a map projection.

    Parameters
    ----------
    x : float
        Easting coordinate in meters.
    y : float
        Northing coordinate in meters.
    scale : float
        Scale factor at the point (ratio of map distance to ground distance).
    convergence : float
        Grid convergence angle in radians (angle from true north to grid north).
    """

    x: float
    y: float
    scale: float
    convergence: float


class UTMResult(NamedTuple):
    """Result of UTM projection.

    Parameters
    ----------
    easting : float
        UTM easting coordinate in meters.
    northing : float
        UTM northing coordinate in meters.
    zone : int
        UTM zone number (1-60).
    hemisphere : str
        'N' for northern hemisphere, 'S' for southern.
    scale : float
        Scale factor at the point.
    convergence : float
        Grid convergence angle in radians.
    """

    easting: float
    northing: float
    zone: int
    hemisphere: str
    scale: float
    convergence: float


# =============================================================================
# Mercator Projection
# =============================================================================


def mercator(
    lat: float,
    lon: float,
    lon0: float = 0.0,
    a: float = WGS84_A,
    e: float = WGS84_E,
) -> ProjectionResult:
    """
    Ellipsoidal Mercator projection (forward).

    The Mercator projection is a conformal cylindrical projection where
    rhumb lines (lines of constant bearing) appear as straight lines.
    Scale increases with latitude, becoming infinite at the poles.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    lon0 : float, optional
        Central meridian in radians. Default is 0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e : float, optional
        First eccentricity. Default is WGS84.

    Returns
    -------
    ProjectionResult
        Projected coordinates with scale and convergence.

    Notes
    -----
    The Mercator projection is conformal, meaning it preserves local angles.
    However, it distorts areas, especially at high latitudes.

    Examples
    --------
    >>> import numpy as np
    >>> result = mercator(np.radians(45), np.radians(-75))
    >>> print(f"x={result.x:.1f}, y={result.y:.1f}")
    """
    # Easting
    x = a * (lon - lon0)

    # Northing using isometric latitude
    sin_lat = np.sin(lat)
    y = a * np.log(
        np.tan(np.pi / 4 + lat / 2) * ((1 - e * sin_lat) / (1 + e * sin_lat)) ** (e / 2)
    )

    # Scale factor
    cos_lat = np.cos(lat)
    w = np.sqrt(1 - WGS84_E2 * sin_lat**2)
    scale = w / cos_lat if cos_lat > 1e-10 else np.inf

    # Convergence is zero for Mercator
    convergence = 0.0

    return ProjectionResult(x, y, scale, convergence)


def mercator_inverse(
    x: float,
    y: float,
    lon0: float = 0.0,
    a: float = WGS84_A,
    e: float = WGS84_E,
    tol: float = 1e-12,
    max_iter: int = 10,
) -> Tuple[float, float]:
    """
    Ellipsoidal Mercator projection (inverse).

    Parameters
    ----------
    x : float
        Easting coordinate in meters.
    y : float
        Northing coordinate in meters.
    lon0 : float, optional
        Central meridian in radians. Default is 0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e : float, optional
        First eccentricity. Default is WGS84.
    tol : float, optional
        Convergence tolerance. Default is 1e-12.
    max_iter : int, optional
        Maximum iterations. Default is 10.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in radians.

    Examples
    --------
    >>> import numpy as np
    >>> lat, lon = mercator_inverse(1000000, 5000000)
    """
    # Longitude
    lon = x / a + lon0

    # Latitude using iterative solution
    t = np.exp(-y / a)
    lat = np.pi / 2 - 2 * np.arctan(t)

    for _ in range(max_iter):
        sin_lat = np.sin(lat)
        lat_new = np.pi / 2 - 2 * np.arctan(
            t * ((1 - e * sin_lat) / (1 + e * sin_lat)) ** (e / 2)
        )
        if abs(lat_new - lat) < tol:
            break
        lat = lat_new

    return lat, lon


# =============================================================================
# Transverse Mercator Projection
# =============================================================================


def transverse_mercator(
    lat: float,
    lon: float,
    lat0: float = 0.0,
    lon0: float = 0.0,
    k0: float = 1.0,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> ProjectionResult:
    """
    Transverse Mercator projection (forward).

    The Transverse Mercator is a conformal projection that rotates the
    cylinder to be tangent along a meridian. It's the basis for UTM.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    lat0 : float, optional
        Origin latitude in radians. Default is 0.
    lon0 : float, optional
        Central meridian in radians. Default is 0.
    k0 : float, optional
        Scale factor at central meridian. Default is 1.0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    ProjectionResult
        Projected coordinates with scale and convergence.

    Notes
    -----
    This implementation uses the Redfearn series expansion, accurate to
    about 1 meter within 4 degrees of the central meridian.

    Examples
    --------
    >>> import numpy as np
    >>> result = transverse_mercator(np.radians(45), np.radians(-75),
    ...                              lon0=np.radians(-75))
    """
    # Derived constants
    ep2 = e2 / (1 - e2)  # Second eccentricity squared
    n = (a - WGS84_B) / (a + WGS84_B)  # Third flattening

    # Compute meridian arc length from equator
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    alpha = [
        0,  # alpha[0] unused
        n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16,
        13 * n**2 / 48 - 3 * n**3 / 5,
        61 * n**3 / 240,
    ]

    # Conformal latitude
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    tan_lat = np.tan(lat)
    eta2 = ep2 * cos_lat**2

    # Longitude difference
    dlon = lon - lon0

    # Compute projection using series
    t = tan_lat
    t2 = t**2
    t4 = t**4

    c = eta2
    c2 = c**2

    cos2 = cos_lat**2
    cos4 = cos_lat**4
    cos6 = cos_lat**6

    dl = dlon
    dl2 = dl**2
    dl4 = dl**4
    dl6 = dl**6

    # Meridian arc length
    N = a / np.sqrt(1 - e2 * sin_lat**2)  # Radius of curvature in prime vertical

    # Arc length from equator to latitude
    sigma = lat
    for i in range(1, 4):
        sigma -= alpha[i] * np.sin(2 * i * lat)
    M = A * sigma

    # Arc length from origin latitude
    sigma0 = lat0
    for i in range(1, 4):
        sigma0 -= alpha[i] * np.sin(2 * i * lat0)
    M0 = A * sigma0

    # Easting
    x = (
        k0
        * N
        * (
            dl * cos_lat
            + dl**3 * cos_lat**3 / 6 * (1 - t2 + c)
            + dl**5 * cos_lat**5 / 120 * (5 - 18 * t2 + t4 + 14 * c - 58 * t2 * c)
        )
    )

    # Northing
    y = k0 * (
        (M - M0)
        + N * tan_lat * (dl2 * cos2 / 2 + dl4 * cos4 / 24 * (5 - t2 + 9 * c + 4 * c2))
        + N * tan_lat * dl6 * cos6 / 720 * (61 - 58 * t2 + t4 + 270 * c - 330 * t2 * c)
    )

    # Scale factor
    scale = k0 * (
        1
        + dl2 * cos2 / 2 * (1 + c)
        + dl4 * cos4 / 24 * (5 - 4 * t2 + 14 * c + 13 * c2 - 28 * t2 * c)
    )

    # Convergence
    convergence = (
        dl * sin_lat
        + dl**3 * sin_lat * cos2 / 3 * (1 + 3 * c + 2 * c2)
        + dl**5 * sin_lat * cos4 / 15 * (2 - t2)
    )

    return ProjectionResult(x, y, scale, convergence)


def transverse_mercator_inverse(
    x: float,
    y: float,
    lat0: float = 0.0,
    lon0: float = 0.0,
    k0: float = 1.0,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> Tuple[float, float]:
    """
    Transverse Mercator projection (inverse).

    Parameters
    ----------
    x : float
        Easting coordinate in meters.
    y : float
        Northing coordinate in meters.
    lat0 : float, optional
        Origin latitude in radians. Default is 0.
    lon0 : float, optional
        Central meridian in radians. Default is 0.
    k0 : float, optional
        Scale factor at central meridian. Default is 1.0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in radians.
    """
    # Derived constants
    ep2 = e2 / (1 - e2)
    n = (a - WGS84_B) / (a + WGS84_B)

    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    beta = [
        0,
        n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96,
        n**2 / 48 + n**3 / 15,
        17 * n**3 / 480,
    ]

    # Arc length from origin latitude
    sigma0 = lat0
    alpha = [
        0,
        n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16,
        13 * n**2 / 48 - 3 * n**3 / 5,
        61 * n**3 / 240,
    ]
    for i in range(1, 4):
        sigma0 -= alpha[i] * np.sin(2 * i * lat0)
    M0 = A * sigma0

    # Footprint latitude
    M = M0 + y / k0
    mu = M / A

    lat_fp = mu
    for i in range(1, 4):
        lat_fp += beta[i] * np.sin(2 * i * mu)

    # Parameters at footprint latitude
    sin_fp = np.sin(lat_fp)
    cos_fp = np.cos(lat_fp)
    tan_fp = np.tan(lat_fp)

    N_fp = a / np.sqrt(1 - e2 * sin_fp**2)
    R_fp = a * (1 - e2) / (1 - e2 * sin_fp**2) ** 1.5

    t = tan_fp
    t2 = t**2
    t4 = t**4

    c = ep2 * cos_fp**2
    c2 = c**2

    d = x / (k0 * N_fp)
    d2 = d**2
    d4 = d**4
    d6 = d**6

    # Latitude
    lat = lat_fp - N_fp * tan_fp / R_fp * (
        d2 / 2
        - d4 / 24 * (5 + 3 * t2 + 10 * c - 4 * c2 - 9 * ep2)
        + d6 / 720 * (61 + 90 * t2 + 298 * c + 45 * t4 - 252 * ep2 - 3 * c2)
    )

    # Longitude
    lon = (
        lon0
        + (
            d
            - d**3 / 6 * (1 + 2 * t2 + c)
            + d**5 / 120 * (5 - 2 * c + 28 * t2 - 3 * c2 + 8 * ep2 + 24 * t4)
        )
        / cos_fp
    )

    return lat, lon


# =============================================================================
# UTM Projection
# =============================================================================


def utm_zone(lon: float, lat: float = 0.0) -> int:
    """
    Determine UTM zone number from longitude.

    Parameters
    ----------
    lon : float
        Longitude in radians.
    lat : float, optional
        Latitude in radians (used for Norway/Svalbard exceptions).

    Returns
    -------
    int
        UTM zone number (1-60).

    Notes
    -----
    Standard zones are 6 degrees wide. Special zones exist for Norway
    and Svalbard.
    """
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)

    # Normalize longitude to -180 to 180
    while lon_deg < -180:
        lon_deg += 360
    while lon_deg >= 180:
        lon_deg -= 360

    # Standard zone calculation
    zone = int((lon_deg + 180) / 6) + 1

    # Norway exception (zone 32 extended)
    if 56 <= lat_deg < 64 and 3 <= lon_deg < 12:
        zone = 32

    # Svalbard exceptions
    if 72 <= lat_deg < 84:
        if 0 <= lon_deg < 9:
            zone = 31
        elif 9 <= lon_deg < 21:
            zone = 33
        elif 21 <= lon_deg < 33:
            zone = 35
        elif 33 <= lon_deg < 42:
            zone = 37

    return zone


def utm_central_meridian(zone: int) -> float:
    """
    Get central meridian for UTM zone.

    Parameters
    ----------
    zone : int
        UTM zone number (1-60).

    Returns
    -------
    float
        Central meridian in radians.
    """
    return np.radians((zone - 1) * 6 - 180 + 3)


def geodetic2utm(lat: float, lon: float, zone: Optional[int] = None) -> UTMResult:
    """
    Convert geodetic coordinates to UTM.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    zone : int, optional
        Force specific UTM zone. If None, computed from longitude.

    Returns
    -------
    UTMResult
        UTM coordinates with zone information.

    Examples
    --------
    >>> import numpy as np
    >>> result = geodetic2utm(np.radians(45.0), np.radians(-75.5))
    >>> print(f"Zone {result.zone}{result.hemisphere}: "
    ...       f"E={result.easting:.1f}, N={result.northing:.1f}")
    """
    # Determine zone
    if zone is None:
        zone = utm_zone(lon, lat)

    # Central meridian
    lon0 = utm_central_meridian(zone)

    # UTM scale factor
    k0 = 0.9996

    # Project
    result = transverse_mercator(lat, lon, lat0=0.0, lon0=lon0, k0=k0)

    # Apply false easting/northing
    easting = result.x + 500000.0

    if lat >= 0:
        northing = result.y
        hemisphere = "N"
    else:
        northing = result.y + 10000000.0
        hemisphere = "S"

    return UTMResult(
        easting, northing, zone, hemisphere, result.scale, result.convergence
    )


def utm2geodetic(
    easting: float,
    northing: float,
    zone: int,
    hemisphere: str = "N",
) -> Tuple[float, float]:
    """
    Convert UTM coordinates to geodetic.

    Parameters
    ----------
    easting : float
        UTM easting in meters.
    northing : float
        UTM northing in meters.
    zone : int
        UTM zone number (1-60).
    hemisphere : str, optional
        'N' for northern, 'S' for southern hemisphere.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in radians.

    Examples
    --------
    >>> lat, lon = utm2geodetic(500000, 5000000, 18, 'N')
    >>> print(f"Lat: {np.degrees(lat):.4f}, Lon: {np.degrees(lon):.4f}")
    """
    # Remove false easting/northing
    x = easting - 500000.0

    if hemisphere.upper() == "S":
        y = northing - 10000000.0
    else:
        y = northing

    # Central meridian
    lon0 = utm_central_meridian(zone)

    # UTM scale factor
    k0 = 0.9996

    return transverse_mercator_inverse(x, y, lat0=0.0, lon0=lon0, k0=k0)


# =============================================================================
# Stereographic Projection
# =============================================================================


def stereographic(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
    k0: float = 1.0,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> ProjectionResult:
    """
    Oblique stereographic projection (forward).

    The stereographic projection is conformal and azimuthal. It's commonly
    used for polar regions and local surveys.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    lat0 : float
        Center latitude in radians.
    lon0 : float
        Center longitude in radians.
    k0 : float, optional
        Scale factor at center. Default is 1.0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    ProjectionResult
        Projected coordinates with scale and convergence.

    Notes
    -----
    For polar stereographic, use lat0 = +-pi/2.

    Examples
    --------
    >>> import numpy as np
    >>> # Polar stereographic centered at North Pole
    >>> result = stereographic(np.radians(85), np.radians(45),
    ...                        np.radians(90), 0)
    """
    e = np.sqrt(e2)

    # Conformal latitudes
    def conformal_lat(phi: float) -> float:
        sin_phi = np.sin(phi)
        return (
            2
            * np.arctan(
                np.tan(np.pi / 4 + phi / 2)
                * ((1 - e * sin_phi) / (1 + e * sin_phi)) ** (e / 2)
            )
            - np.pi / 2
        )

    chi = conformal_lat(lat)
    chi0 = conformal_lat(lat0)

    # Radius of curvature
    sin_lat0 = np.sin(lat0)
    N0 = a / np.sqrt(1 - e2 * sin_lat0**2)
    R = N0 * np.sqrt(1 - e2) / (1 - e2 * sin_lat0**2)

    # Projection
    sin_chi = np.sin(chi)
    cos_chi = np.cos(chi)
    sin_chi0 = np.sin(chi0)
    cos_chi0 = np.cos(chi0)

    dlon = lon - lon0
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)

    k_denom = 1 + sin_chi0 * sin_chi + cos_chi0 * cos_chi * cos_dlon
    k = 2 * k0 / k_denom

    x = k * R * cos_chi * sin_dlon
    y = k * R * (cos_chi0 * sin_chi - sin_chi0 * cos_chi * cos_dlon)

    # Scale factor
    scale = k

    # Convergence
    convergence = np.arctan2(
        cos_chi0 * sin_dlon,
        sin_chi0 * cos_chi - cos_chi0 * sin_chi * cos_dlon,
    )

    return ProjectionResult(x, y, scale, convergence)


def stereographic_inverse(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
    k0: float = 1.0,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
    tol: float = 1e-12,
    max_iter: int = 10,
) -> Tuple[float, float]:
    """
    Oblique stereographic projection (inverse).

    Parameters
    ----------
    x : float
        Easting in meters.
    y : float
        Northing in meters.
    lat0 : float
        Center latitude in radians.
    lon0 : float
        Center longitude in radians.
    k0 : float, optional
        Scale factor at center. Default is 1.0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.
    tol : float, optional
        Convergence tolerance. Default is 1e-12.
    max_iter : int, optional
        Maximum iterations. Default is 10.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in radians.
    """
    e = np.sqrt(e2)

    # Conformal latitude of center
    sin_lat0 = np.sin(lat0)
    chi0 = (
        2
        * np.arctan(
            np.tan(np.pi / 4 + lat0 / 2)
            * ((1 - e * sin_lat0) / (1 + e * sin_lat0)) ** (e / 2)
        )
        - np.pi / 2
    )

    sin_chi0 = np.sin(chi0)
    cos_chi0 = np.cos(chi0)

    # Radius
    N0 = a / np.sqrt(1 - e2 * sin_lat0**2)
    R = N0 * np.sqrt(1 - e2) / (1 - e2 * sin_lat0**2)

    # Distance from center
    rho = np.sqrt(x**2 + y**2)

    if rho < 1e-10:
        return lat0, lon0

    c = 2 * np.arctan(rho / (2 * k0 * R))
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    # Conformal latitude
    chi = np.arcsin(cos_c * sin_chi0 + y * sin_c * cos_chi0 / rho)

    # Longitude
    lon = lon0 + np.arctan2(x * sin_c, rho * cos_chi0 * cos_c - y * sin_chi0 * sin_c)

    # Invert conformal latitude
    lat = chi
    for _ in range(max_iter):
        sin_lat = np.sin(lat)
        lat_new = (
            2
            * np.arctan(
                np.tan(np.pi / 4 + chi / 2)
                * ((1 + e * sin_lat) / (1 - e * sin_lat)) ** (e / 2)
            )
            - np.pi / 2
        )
        if abs(lat_new - lat) < tol:
            break
        lat = lat_new

    return lat, lon


def polar_stereographic(
    lat: float,
    lon: float,
    north: bool = True,
    k0: float = 0.994,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> ProjectionResult:
    """
    Polar stereographic projection (forward).

    Standard polar stereographic used for polar regions.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    north : bool, optional
        True for North Pole center, False for South Pole. Default is True.
    k0 : float, optional
        Scale factor at pole. Default is 0.994 (UPS standard).
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    ProjectionResult
        Projected coordinates with scale and convergence.

    Examples
    --------
    >>> import numpy as np
    >>> # Arctic location
    >>> result = polar_stereographic(np.radians(80), np.radians(45))
    """
    lat0 = np.pi / 2 if north else -np.pi / 2
    lon0 = 0.0

    if not north:
        lat = -lat
        lon = -lon

    result = stereographic(lat, lon, lat0, lon0, k0, a, e2)

    if not north:
        return ProjectionResult(-result.x, -result.y, result.scale, -result.convergence)

    return result


# =============================================================================
# Lambert Conformal Conic Projection
# =============================================================================


def lambert_conformal_conic(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
    lat1: float,
    lat2: float,
    k0: float = 1.0,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> ProjectionResult:
    """
    Lambert Conformal Conic projection (forward).

    A conformal conic projection with two standard parallels where
    scale is exact. Good for mid-latitude regions with east-west extent.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    lat0 : float
        Origin latitude in radians.
    lon0 : float
        Central meridian in radians.
    lat1 : float
        First standard parallel in radians.
    lat2 : float
        Second standard parallel in radians.
    k0 : float, optional
        Scale factor. Default is 1.0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    ProjectionResult
        Projected coordinates with scale and convergence.

    Examples
    --------
    >>> import numpy as np
    >>> # Continental US projection
    >>> result = lambert_conformal_conic(
    ...     np.radians(40), np.radians(-100),
    ...     lat0=np.radians(39), lon0=np.radians(-96),
    ...     lat1=np.radians(33), lat2=np.radians(45)
    ... )
    """
    e = np.sqrt(e2)

    # Compute m and t for each latitude
    def compute_m(phi: float) -> float:
        sin_phi = np.sin(phi)
        return np.cos(phi) / np.sqrt(1 - e2 * sin_phi**2)

    def compute_t(phi: float) -> float:
        sin_phi = np.sin(phi)
        return np.tan(np.pi / 4 - phi / 2) / (
            ((1 - e * sin_phi) / (1 + e * sin_phi)) ** (e / 2)
        )

    m1 = compute_m(lat1)
    m2 = compute_m(lat2)
    t0 = compute_t(lat0)
    t1 = compute_t(lat1)
    t2 = compute_t(lat2)
    t = compute_t(lat)

    # Cone constant
    if abs(lat1 - lat2) < 1e-10:
        n = np.sin(lat1)
    else:
        n = (np.log(m1) - np.log(m2)) / (np.log(t1) - np.log(t2))

    # Projection constants
    F = m1 / (n * t1**n)
    rho0 = a * F * t0**n * k0
    rho = a * F * t**n * k0

    # Coordinates
    theta = n * (lon - lon0)
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)

    # Scale factor
    m = compute_m(lat)
    scale = k0 * n * F * t**n / m

    # Convergence
    convergence = theta

    return ProjectionResult(x, y, scale, convergence)


def lambert_conformal_conic_inverse(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
    lat1: float,
    lat2: float,
    k0: float = 1.0,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
    tol: float = 1e-12,
    max_iter: int = 10,
) -> Tuple[float, float]:
    """
    Lambert Conformal Conic projection (inverse).

    Parameters
    ----------
    x : float
        Easting in meters.
    y : float
        Northing in meters.
    lat0 : float
        Origin latitude in radians.
    lon0 : float
        Central meridian in radians.
    lat1 : float
        First standard parallel in radians.
    lat2 : float
        Second standard parallel in radians.
    k0 : float, optional
        Scale factor. Default is 1.0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.
    tol : float, optional
        Convergence tolerance. Default is 1e-12.
    max_iter : int, optional
        Maximum iterations. Default is 10.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in radians.
    """
    e = np.sqrt(e2)

    def compute_m(phi: float) -> float:
        sin_phi = np.sin(phi)
        return np.cos(phi) / np.sqrt(1 - e2 * sin_phi**2)

    def compute_t(phi: float) -> float:
        sin_phi = np.sin(phi)
        return np.tan(np.pi / 4 - phi / 2) / (
            ((1 - e * sin_phi) / (1 + e * sin_phi)) ** (e / 2)
        )

    m1 = compute_m(lat1)
    m2 = compute_m(lat2)
    t0 = compute_t(lat0)
    t1 = compute_t(lat1)
    t2 = compute_t(lat2)

    if abs(lat1 - lat2) < 1e-10:
        n = np.sin(lat1)
    else:
        n = (np.log(m1) - np.log(m2)) / (np.log(t1) - np.log(t2))

    F = m1 / (n * t1**n)
    rho0 = a * F * t0**n * k0

    # Compute rho and theta
    rho_prime = rho0 - y
    rho = np.sign(n) * np.sqrt(x**2 + rho_prime**2)
    theta = np.arctan2(np.sign(n) * x, np.sign(n) * rho_prime)

    # Compute t from rho
    t = (rho / (a * F * k0)) ** (1 / n)

    # Compute latitude iteratively
    lat = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(max_iter):
        sin_lat = np.sin(lat)
        lat_new = np.pi / 2 - 2 * np.arctan(
            t * ((1 - e * sin_lat) / (1 + e * sin_lat)) ** (e / 2)
        )
        if abs(lat_new - lat) < tol:
            break
        lat = lat_new

    # Longitude
    lon = theta / n + lon0

    return lat, lon


# =============================================================================
# Azimuthal Equidistant Projection
# =============================================================================


def azimuthal_equidistant(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> ProjectionResult:
    """
    Azimuthal equidistant projection (forward).

    Distances from the center point are preserved. Useful for showing
    distances from a specific location (e.g., radio coverage).

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    lat0 : float
        Center latitude in radians.
    lon0 : float
        Center longitude in radians.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    ProjectionResult
        Projected coordinates with scale and convergence.

    Notes
    -----
    This implementation uses the spherical approximation for simplicity.
    For high accuracy over long distances, geodesic calculations should
    be used.

    Examples
    --------
    >>> import numpy as np
    >>> result = azimuthal_equidistant(np.radians(40), np.radians(-75),
    ...                                np.radians(38.9), np.radians(-77))
    """
    # Use spherical approximation with authalic radius
    R = a * np.sqrt(
        (
            1
            + (1 - e2)
            / (2 * np.sqrt(1 - e2))
            * np.log((1 + np.sqrt(1 - e2)) / np.sqrt(e2))
        )
        / 2
    )

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)

    dlon = lon - lon0
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)

    # Angular distance from center
    cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon
    cos_c = np.clip(cos_c, -1, 1)
    c = np.arccos(cos_c)

    if c < 1e-10:
        return ProjectionResult(0.0, 0.0, 1.0, 0.0)

    # Azimuth from center
    k = R * c / np.sin(c)

    x = k * cos_lat * sin_dlon
    y = k * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon)

    # Scale (radial = 1, tangential varies)
    scale = 1.0  # Radial scale is exactly 1

    # Convergence
    convergence = np.arctan2(
        cos_lat0 * sin_dlon,
        sin_lat0 * cos_lat - cos_lat0 * sin_lat * cos_dlon,
    )

    return ProjectionResult(x, y, scale, convergence)


def azimuthal_equidistant_inverse(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
    a: float = WGS84_A,
    e2: float = WGS84_E2,
) -> Tuple[float, float]:
    """
    Azimuthal equidistant projection (inverse).

    Parameters
    ----------
    x : float
        Easting in meters.
    y : float
        Northing in meters.
    lat0 : float
        Center latitude in radians.
    lon0 : float
        Center longitude in radians.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    e2 : float, optional
        First eccentricity squared. Default is WGS84.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in radians.
    """
    R = a * np.sqrt(
        (
            1
            + (1 - e2)
            / (2 * np.sqrt(1 - e2))
            * np.log((1 + np.sqrt(1 - e2)) / np.sqrt(e2))
        )
        / 2
    )

    rho = np.sqrt(x**2 + y**2)

    if rho < 1e-10:
        return lat0, lon0

    c = rho / R
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)

    lat = np.arcsin(cos_c * sin_lat0 + y * sin_c * cos_lat0 / rho)
    lon = lon0 + np.arctan2(x * sin_c, rho * cos_lat0 * cos_c - y * sin_lat0 * sin_c)

    return lat, lon


# =============================================================================
# Batch Operations
# =============================================================================


def geodetic2utm_batch(
    lats: NDArray[np.floating],
    lons: NDArray[np.floating],
    zone: Optional[int] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.intp], NDArray[Any]]:
    """
    Batch convert geodetic coordinates to UTM.

    Parameters
    ----------
    lats : ndarray
        Geodetic latitudes in radians.
    lons : ndarray
        Longitudes in radians.
    zone : int, optional
        Force specific UTM zone. If None, computed per point.

    Returns
    -------
    eastings : ndarray
        UTM eastings in meters.
    northings : ndarray
        UTM northings in meters.
    zones : ndarray
        UTM zone numbers.
    hemispheres : ndarray
        Hemisphere characters ('N' or 'S').
    """
    n = len(lats)
    eastings = np.zeros(n)
    northings = np.zeros(n)
    zones = np.zeros(n, dtype=np.int_)
    hemispheres = np.empty(n, dtype="U1")

    for i in range(n):
        result = geodetic2utm(lats[i], lons[i], zone)
        eastings[i] = result.easting
        northings[i] = result.northing
        zones[i] = result.zone
        hemispheres[i] = result.hemisphere

    return eastings, northings, zones, hemispheres


__all__ = [
    # Constants
    "WGS84_A",
    "WGS84_B",
    "WGS84_F",
    "WGS84_E",
    "WGS84_E2",
    "WGS84_EP2",
    # Result types
    "ProjectionResult",
    "UTMResult",
    # Mercator
    "mercator",
    "mercator_inverse",
    # Transverse Mercator
    "transverse_mercator",
    "transverse_mercator_inverse",
    # UTM
    "utm_zone",
    "utm_central_meridian",
    "geodetic2utm",
    "utm2geodetic",
    "geodetic2utm_batch",
    # Stereographic
    "stereographic",
    "stereographic_inverse",
    "polar_stereographic",
    # Lambert Conformal Conic
    "lambert_conformal_conic",
    "lambert_conformal_conic_inverse",
    # Azimuthal Equidistant
    "azimuthal_equidistant",
    "azimuthal_equidistant_inverse",
]
