"""
Gravity field models.

This module provides implementations of standard gravity models including
WGS84 normal gravity and spherical harmonic gravity models.

References
----------
- NIMA, "Department of Defense World Geodetic System 1984," TR8350.2, 2000.
- Pavlis et al., "The development and evaluation of the Earth
  Gravitational Model 2008 (EGM2008)," JGR, 2012.
"""

from typing import NamedTuple

import numpy as np

from pytcl.core.exceptions import ConvergenceError
from pytcl.gravity.spherical_harmonics import spherical_harmonic_sum


class GravityConstants(NamedTuple):
    """Constants for a gravity model.

    Attributes
    ----------
    GM : float
        Gravitational parameter (m^3/s^2).
    a : float
        Semi-major axis / equatorial radius (m).
    f : float
        Flattening.
    omega : float
        Angular velocity (rad/s).
    J2 : float
        Second degree zonal harmonic (unnormalized).
    """

    GM: float
    a: float
    f: float
    omega: float
    J2: float


# WGS84 constants
WGS84 = GravityConstants(
    GM=3.986004418e14,  # m^3/s^2
    a=6378137.0,  # m
    f=1 / 298.257223563,
    omega=7.292115e-5,  # rad/s
    J2=1.08263e-3,
)

# GRS80 constants (used in some applications)
GRS80 = GravityConstants(
    GM=3.986005e14,
    a=6378137.0,
    f=1 / 298.257222101,
    omega=7.292115e-5,
    J2=1.08263e-3,
)


class GravityResult(NamedTuple):
    """Result of gravity computation.

    Attributes
    ----------
    magnitude : float
        Total gravity magnitude (m/s^2).
    g_down : float
        Downward component (positive down) (m/s^2).
    g_north : float
        Northward component (m/s^2).
    g_east : float
        Eastward component (m/s^2).
    """

    magnitude: float
    g_down: float
    g_north: float
    g_east: float


def normal_gravity_somigliana(
    lat: float,
    constants: GravityConstants = WGS84,
) -> float:
    """
    Compute normal gravity on the ellipsoid using Somigliana's formula.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    constants : GravityConstants, optional
        Gravity model constants. Default WGS84.

    Returns
    -------
    gamma : float
        Normal gravity on the ellipsoid surface (m/s^2).

    Notes
    -----
    Somigliana's closed formula gives the exact normal gravity on the
    reference ellipsoid without iteration.

    Examples
    --------
    >>> gamma = normal_gravity_somigliana(0)  # At equator
    >>> abs(gamma - 9.7803) < 0.001
    True

    >>> gamma = normal_gravity_somigliana(np.pi/2)  # At pole
    >>> abs(gamma - 9.8322) < 0.001
    True
    """
    a = constants.a
    f = constants.f
    GM = constants.GM
    omega = constants.omega

    # Derived quantities
    b = a * (1 - f)  # Semi-minor axis
    e2 = 2 * f - f * f  # First eccentricity squared

    # Gravity at equator and pole
    # Using closed-form expressions
    m = omega * omega * a * a * b / GM

    # Normal gravity at equator
    gamma_a = GM / (a * b) * (1 - 3 / 2 * m - 3 / 14 * e2 * m)

    # Normal gravity at pole
    gamma_b = GM / (a * a) * (1 + m + 3 / 7 * e2 * m)

    # Somigliana formula
    sin_lat = np.sin(lat)

    k = (b * gamma_b - a * gamma_a) / (a * gamma_a)

    gamma = gamma_a * (1 + k * sin_lat * sin_lat) / np.sqrt(1 - e2 * sin_lat * sin_lat)

    return gamma


def normal_gravity(
    lat: float,
    h: float = 0.0,
    constants: GravityConstants = WGS84,
) -> float:
    """
    Compute normal gravity at a given latitude and height.

    Uses the International Gravity Formula with free-air correction.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    h : float, optional
        Height above ellipsoid in meters. Default 0.
    constants : GravityConstants, optional
        Gravity model constants. Default WGS84.

    Returns
    -------
    g : float
        Normal gravity (m/s^2).

    Examples
    --------
    >>> g = normal_gravity(0, 0)  # Sea level at equator
    >>> abs(g - 9.78) < 0.01
    True

    >>> g = normal_gravity(np.radians(45), 1000)  # 1km altitude, 45° lat
    >>> g < normal_gravity(np.radians(45), 0)  # Decreases with altitude
    True
    """
    # Gravity on ellipsoid
    gamma_0 = normal_gravity_somigliana(lat, constants)

    # Free-air correction (second-order)
    a = constants.a
    f = constants.f
    b = a * (1 - f)
    # Geodetic parameter m = omega^2 a^2 b / GM (Heiskanen & Moritz 2-123)
    m = constants.omega**2 * a**2 * b / constants.GM

    sin2_lat = np.sin(lat) ** 2

    # Height correction
    gamma = gamma_0 * (
        1 - 2 / a * (1 + f + m - 2 * f * sin2_lat) * h + 3 / (a * a) * h * h
    )

    return gamma


def gravity_wgs84(
    lat: float,
    lon: float,
    h: float = 0.0,
) -> GravityResult:
    """
    Compute gravity using WGS84 model.

    This computes the full gravity vector including the centrifugal
    acceleration due to Earth's rotation.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in meters. Default 0.

    Returns
    -------
    result : GravityResult
        Gravity components and magnitude.

    Examples
    --------
    >>> result = gravity_wgs84(0, 0, 0)
    >>> abs(result.magnitude - 9.78) < 0.01
    True
    """
    g = normal_gravity(lat, h, WGS84)

    # For normal gravity model, gravity is purely radial (downward)
    # in the local level frame
    return GravityResult(
        magnitude=g,
        g_down=g,
        g_north=0.0,
        g_east=0.0,
    )


def gravity_j2(
    lat: float,
    lon: float,
    h: float = 0.0,
    constants: GravityConstants = WGS84,
) -> GravityResult:
    """
    Compute gravity using J2 (oblateness) model.

    This simplified model includes only the J2 zonal harmonic,
    which accounts for Earth's equatorial bulge.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above ellipsoid in meters. Default 0.
    constants : GravityConstants, optional
        Model constants. Default WGS84.

    Returns
    -------
    result : GravityResult
        Gravity components and magnitude.

    Examples
    --------
    >>> result = gravity_j2(0, 0, 0)  # At equator, sea level
    >>> abs(result.magnitude - 9.78) < 0.01
    True
    """
    GM = constants.GM
    a = constants.a
    J2 = constants.J2
    omega = constants.omega

    # Approximate geocentric radius
    r = a + h  # Simplified

    # Geocentric latitude (approximate)
    lat_gc = lat  # Simplified, should account for flattening

    sin_lat = np.sin(lat_gc)
    cos_lat = np.cos(lat_gc)

    # J2 gravity in spherical coordinates
    r2 = r * r
    a2_r2 = (a / r) ** 2

    # Radial component (positive outward)
    g_r = -GM / r2 * (1 - 3 / 2 * J2 * a2_r2 * (3 * sin_lat**2 - 1))

    # Latitudinal component (northward): (1/r) dV/dlat with
    # V = GM/r * (1 - J2 (a/r)^2 P2(sin(lat)))
    g_lat = -GM / r2 * 3 * J2 * a2_r2 * sin_lat * cos_lat

    # Centrifugal acceleration (points away from the rotation axis):
    # local up component +centrifugal*cos_lat, north component
    # -centrifugal*sin_lat (toward the equator)
    centrifugal = omega**2 * r * cos_lat

    # Convert to local level frame (down, north, east)
    g_down = -g_r - centrifugal * cos_lat
    g_north = g_lat - centrifugal * sin_lat

    magnitude = np.sqrt(g_down**2 + g_north**2)

    return GravityResult(
        magnitude=magnitude,
        g_down=g_down,
        g_north=g_north,
        g_east=0.0,  # J2 is zonally symmetric
    )


def geoid_height_j2(
    lat: float,
    constants: GravityConstants = WGS84,
) -> float:
    """
    Compute approximate geoid height using J2 model.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    constants : GravityConstants, optional
        Model constants. Default WGS84.

    Returns
    -------
    N : float
        Deviation of the J2 equipotential surface from a mean sphere,
        in meters (equal to -a * J2 * P2(sin(lat))).

    Examples
    --------
    >>> N = geoid_height_j2(0)  # At equator
    >>> N > 0  # Equator bulges outward
    True

    Notes
    -----
    This is a simplified model: it gives the shape of the J2-only
    (non-rotating) level surface relative to a sphere of radius a, NOT
    the geoid height above the reference ellipsoid (which is what full
    geoid models like EGM2008 provide; those values are within ~±100 m).
    """
    a = constants.a
    J2 = constants.J2

    sin_lat = np.sin(lat)

    # J2 contribution to geoid height
    N = -a * J2 * (3 * sin_lat**2 - 1) / 2

    return N


def gravitational_potential(
    lat: float,
    lon: float,
    r: float,
    constants: GravityConstants = WGS84,
    n_max: int = 2,
) -> float:
    """
    Compute gravitational potential.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from Earth's center in meters.
    constants : GravityConstants, optional
        Model constants. Default WGS84.
    n_max : int, optional
        Maximum spherical harmonic degree. Default 2.

    Returns
    -------
    U : float
        Gravitational potential (m^2/s^2).

    Examples
    --------
    >>> U = gravitational_potential(0, 0, 6.4e6)  # Near Earth surface
    >>> bool(6.1e7 < U < 6.4e7)  # Positive ~GM/r (geodesy sign convention)
    True
    """
    GM = constants.GM
    a = constants.a
    J2 = constants.J2

    # Zonal harmonics only (simplified)
    # C_20 = -J2 / sqrt(5) for normalized coefficients
    C = np.zeros((n_max + 1, n_max + 1))
    S = np.zeros((n_max + 1, n_max + 1))

    C[0, 0] = 1.0  # Central term
    if n_max >= 2:
        C[2, 0] = -J2 / np.sqrt(5)  # Normalized J2

    V, _, _ = spherical_harmonic_sum(lat, lon, r, C, S, a, GM, n_max)

    return V


def free_air_anomaly(
    g_observed: float,
    lat: float,
    h: float,
    constants: GravityConstants = WGS84,
) -> float:
    """
    Compute free-air gravity anomaly.

    Parameters
    ----------
    g_observed : float
        Observed gravity in m/s^2.
    lat : float
        Geodetic latitude in radians.
    h : float
        Height above geoid in meters.
    constants : GravityConstants, optional
        Model constants. Default WGS84.

    Returns
    -------
    delta_g : float
        Free-air anomaly in m/s^2 (or mGal if multiplied by 1e5).

    Examples
    --------
    >>> import numpy as np
    >>> # Observed gravity slightly higher than normal
    >>> delta_g = free_air_anomaly(9.81, np.radians(45), 100)
    >>> isinstance(delta_g, float)
    True

    Notes
    -----
    The free-air anomaly is the difference between observed gravity
    and normal gravity at the observation point.
    """
    gamma = normal_gravity(lat, h, constants)
    return g_observed - gamma


def bouguer_anomaly(
    g_observed: float,
    lat: float,
    h: float,
    rho: float = 2670.0,
    constants: GravityConstants = WGS84,
) -> float:
    """
    Compute simple Bouguer gravity anomaly.

    Parameters
    ----------
    g_observed : float
        Observed gravity in m/s^2.
    lat : float
        Geodetic latitude in radians.
    h : float
        Height above geoid in meters.
    rho : float, optional
        Crustal density in kg/m^3. Default 2670 (average crustal density).
    constants : GravityConstants, optional
        Model constants. Default WGS84.

    Returns
    -------
    delta_g : float
        Bouguer anomaly in m/s^2.

    Examples
    --------
    >>> import numpy as np
    >>> # Bouguer anomaly at mountain location
    >>> delta_g = bouguer_anomaly(9.81, np.radians(45), 1000)
    >>> isinstance(delta_g, float)
    True

    Notes
    -----
    The Bouguer anomaly removes the gravitational effect of the
    topographic mass between the observation point and the geoid.
    """
    G = 6.67430e-11  # Gravitational constant

    # Free-air anomaly
    fa = free_air_anomaly(g_observed, lat, h, constants)

    # Bouguer plate correction
    bouguer_correction = 2 * np.pi * G * rho * h

    return fa - bouguer_correction


def alt_ellips_param_to_flattening(
    omega: float,
    a: float,
    c20_bar: float,
    gm: float,
) -> float:
    """
    Flattening from the alternative ellipsoid parameter set.

    Port of ``altEllipsParam2Flattening``. Some reference ellipsoids
    (EGM2008 among them) are defined by ``(omega, a, C20bar, GM)`` --
    a fully normalized second-degree zonal harmonic coefficient instead
    of a flattening. This converts to the flattening ``f`` of the
    standard parameterization by Moritz's iterative method. The
    unnormalized ``J2 = -C20bar * sqrt(5)``.

    Parameters
    ----------
    omega : float
        Average rotation rate of the planet in rad/s.
    a : float
        Semi-major axis of the reference ellipsoid in meters.
    c20_bar : float
        Fully normalized second-degree zonal coefficient.
    gm : float
        Universal gravitational constant times planet mass, m^3/s^2.

    Returns
    -------
    f : float
        Flattening factor of the reference ellipsoid.

    References
    ----------
    - H. Moritz, "Geodetic reference system 1980," Bulletin geodesique
      54(3):395-405, 1980.

    Examples
    --------
    >>> # EGM2008's defining constants recover (nearly) the WGS84 flattening
    >>> f = alt_ellips_param_to_flattening(
    ...     7292115e-11, 6378136.3, -484.1654767e-6, 3986004.415e8
    ... )
    >>> round(1.0 / f, 3)
    298.258
    """
    j2 = -c20_bar * np.sqrt(5.0)

    # The MATLAB original iterates with no cap under a tolerance of
    # 1e3 ulps; for some parameter sets the fixed-point map settles
    # into a 2-cycle a few hundred ulps wide, just above that
    # tolerance, and the original then loops forever. This port runs
    # the same criterion for 500 iterations, then accepts an ulp-scale
    # cycle (relative amplitude below 1e-10 -- physically nothing) and
    # fails loudly on anything else.
    e = 0.8
    e_old = np.inf
    for _ in range(500):
        if abs(e_old - e) <= 1e3 * np.spacing(e):
            return float(1 - np.sqrt(1 - e**2))
        e_tilde = e / np.sqrt(1 - e**2)  # The second eccentricity.
        q0 = 0.5 * ((1 + 3 / e_tilde**2) * np.arctan(e_tilde) - 3 / e_tilde)
        e_old = e
        e = np.sqrt(3 * j2 + (4 / 15) * (omega**2 * a**3 / gm) * (e**3 / (2 * q0)))

    if np.isfinite(e) and abs(e_old - e) <= 1e-10 * abs(e):
        return float(1 - np.sqrt(1 - e**2))

    raise ConvergenceError(
        "alt_ellips_param_to_flattening: Moritz's eccentricity iteration "
        "did not converge in 500 iterations (the MATLAB original loops "
        "forever on such inputs). Check that the parameter set describes "
        "a physically plausible rotating oblate ellipsoid."
    )


def ellips_grav_coeffs(
    max_order: int = 2,
    is_normalized: bool = True,
    omega: float = WGS84.omega,
    a: float = WGS84.a,
    f: float = WGS84.f,
    gm: float = WGS84.GM,
) -> tuple:
    """
    Spherical harmonic coefficients of the ellipsoidal gravity field.

    Port of ``ellipsGravCoeffs``: the (unnormalized or fully
    normalized) spherical harmonic coefficients implied by an
    ellipsoidal approximation to the geoid with rotation rate
    ``omega``. Only the even zonal terms ``C_{2n,0}`` are nonzero; all
    sine coefficients are zero by symmetry. ``max_order=2`` is the J2
    model, ``max_order=4`` the J4 model. The returned ``(C, S, a, c)``
    plug directly into :func:`pytcl.gravity.spherical_harmonics.
    spherical_harmonic_sum` (``R=a``, ``GM=c``).

    Parameters
    ----------
    max_order : int, optional
        Maximum degree of the coefficients. Default 2.
    is_normalized : bool, optional
        If True (default), return fully normalized coefficients (for
        fully normalized associated Legendre functions).
    omega : float, optional
        Rotation rate in rad/s. Default WGS84.
    a : float, optional
        Semi-major axis in meters. Default WGS84.
    f : float, optional
        Flattening. Default WGS84.
    gm : float, optional
        GM in m^3/s^2. Default WGS84.

    Returns
    -------
    C : ndarray
        Cosine coefficients, shape (max_order+1, max_order+1);
        ``C[n, m]`` is the degree-n order-m coefficient.
    S : ndarray
        Sine coefficients, same shape, all zero.
    a : float
        The numerator of the ``(a/r)^n`` term, meters (the input ``a``).
    c : float
        The multiplier of the series, m^3/s^2 (equals ``gm``).

    References
    ----------
    - B. Hofmann-Wellenhof and H. Moritz, Physical Geodesy, 2nd ed.,
      Springer, 2006 (Chapters 2.5, 2.7-2.9).

    Examples
    --------
    >>> C, S, R, GM = ellips_grav_coeffs()
    >>> round(C[2, 0] * 1e6, 4)  # the WGS84-derived normalized C20
    -484.1668
    >>> bool(np.all(S == 0))
    True
    """
    C = np.zeros((max_order + 1, max_order + 1))
    S = np.zeros((max_order + 1, max_order + 1))

    b = a * (1 - f)  # Semi-minor axis.
    E = np.sqrt(a**2 - b**2)  # Linear eccentricity (Eq. 2-174).
    e = E / a  # First numerical eccentricity.
    epsilon = E / b  # Second numerical eccentricity.

    # Equations 2-137 and 2-113.
    m = omega**2 * a**2 * b / gm
    q0 = 0.5 * ((1 + 3 * b**2 / E**2) * np.arctan(E / b) - 3 * b / E)

    inner_term = (1 / 3) * (1 - (2 / 15) * (m * epsilon / q0))
    for n in range(max_order // 2 + 1):
        # Equations 2-170 and 2-167.
        C[2 * n, 0] = (
            (-1) ** n
            * 3
            * e ** (2 * n)
            / ((2 * n + 1) * (2 * n + 3))
            * (1 - n + 5 * n * inner_term)
        )
        if is_normalized:
            # Normalization per Equation 2-80.
            C[2 * n, 0] /= np.sqrt(2 * (2 * n) + 1)

    return C, S, float(a), float(gm)


__all__ = [
    "GravityConstants",
    "GravityResult",
    "WGS84",
    "GRS80",
    "normal_gravity_somigliana",
    "normal_gravity",
    "gravity_wgs84",
    "gravity_j2",
    "geoid_height_j2",
    "gravitational_potential",
    "free_air_anomaly",
    "bouguer_anomaly",
    "alt_ellips_param_to_flattening",
    "ellips_grav_coeffs",
]
