"""Relativistic corrections for precision astronomy and satellite positioning.

This module provides utilities for computing relativistic effects in orbital mechanics,
including gravitational time dilation, Shapiro delay, and coordinate transformations
in the Schwarzschild metric. These effects are critical for high-precision applications
such as GPS, pulsar timing, and celestial mechanics.

Key Physical Constants:
    - Schwarzschild radius: r_s = 2GM/c^2
    - Gravitational parameter for Earth: μ = GM = 3.986004418e14 m^3/s^2
    - Speed of light: c = 299792458 m/s
    - Gravitational constant: G = 6.67430e-11 m^3/(kg·s^2)

References:
    - Soffel et al. (2003): The IAU 2000 Resolutions for Astrometry, Celestial Mechanics,
      and Reference Frames
    - Will, C. M. (2014): The Confrontation between General Relativity and Experiment
    - Ries et al. (1992): Preliminary Analysis of LAGEOS II Observations for the
      Determination of Relativistic Effects
"""

import numpy as np
from numpy.typing import NDArray

# Physical constants (CODATA 2018 values)
C_LIGHT = 299792458.0  # Speed of light (m/s)
G_GRAV = 6.67430e-11  # Gravitational constant (m^3/(kg·s^2))
GM_EARTH = 3.986004418e14  # Gravitational parameter for Earth (m^3/s^2)
GM_SUN = 1.32712440018e20  # Gravitational parameter for Sun (m^3/s^2)
AU = 1.495978707e11  # Astronomical unit (m)


def schwarzschild_radius(mass: float) -> float:
    """Compute Schwarzschild radius for a given mass.

    The Schwarzschild radius is the radius at which an object becomes a black hole.
    It is given by r_s = 2GM/c^2.

    Parameters
    ----------
    mass : float
        Mass of the object (kg)

    Returns
    -------
    float
        Schwarzschild radius (m)

    Examples
    --------
    >>> r_s_earth = schwarzschild_radius(5.972e24)  # Earth's mass
    >>> print(f"Earth's Schwarzschild radius: {r_s_earth:.3e} m")
    Earth's Schwarzschild radius: 8.870e-03 m

    >>> r_s_sun = schwarzschild_radius(1.989e30)  # Sun's mass
    >>> print(f"Sun's Schwarzschild radius: {r_s_sun:.3e} m")
    Sun's Schwarzschild radius: 2.954e+03 m
    """
    return 2.0 * G_GRAV * mass / (C_LIGHT**2)


def gravitational_time_dilation(r: float, gm: float = GM_EARTH) -> float:
    """Compute gravitational time dilation factor sqrt(1 - 2GM/(rc^2)).

    In general relativity, time passes slower in stronger gravitational fields.
    This function computes the metric coefficient g_00 for the Schwarzschild metric,
    which determines proper time relative to coordinate time.

    Parameters
    ----------
    r : float
        Distance from the gravitational body (m)
    gm : float, optional
        Gravitational parameter GM of the body (m^3/s^2).
        Default is GM_EARTH.

    Returns
    -------
    float
        Time dilation factor in [0, 1]. A value less than 1 indicates time
        passes slower at radius r compared to infinity.

    Raises
    ------
    ValueError
        If r is less than or equal to Schwarzschild radius

    Examples
    --------
    Compute time dilation at Earth's surface (6371 km):

    >>> r_earth = 6.371e6  # meters
    >>> dilation = gravitational_time_dilation(r_earth)
    >>> print(f"Time dilation at surface: {dilation:.15f}")
    Time dilation at surface: 0.999999999303873

    At GPS orbital altitude (~20,200 km):

    >>> r_gps = 26.56e6  # meters
    >>> dilation_gps = gravitational_time_dilation(r_gps)
    >>> time_shift = (1 - dilation_gps) * 86400 * 1e9  # nanoseconds per day
    >>> print(f"Time shift: {time_shift:.1f} ns/day")
    Time shift: 14427.2 ns/day
    """
    r_s = schwarzschild_radius(gm / G_GRAV)
    if r <= r_s:
        raise ValueError(f"Radius {r} m is at or within Schwarzschild radius {r_s} m")

    dilation_squared = 1.0 - 2.0 * gm / (C_LIGHT**2 * r)
    return np.sqrt(dilation_squared)


def proper_time_rate(v: float, r: float, gm: float = GM_EARTH) -> float:
    """Compute proper time rate accounting for both velocity and gravity.

    The proper time rate combines special relativistic time dilation from velocity
    and general relativistic time dilation from the gravitational potential.

    d(tau)/d(t) = sqrt(1 - v^2/c^2) * sqrt(1 - 2GM/(rc^2))

    For small velocities and weak fields: 1 - v^2/(2c^2) - GM/(rc^2)

    Parameters
    ----------
    v : float
        Velocity magnitude (m/s)
    r : float
        Distance from gravitational body (m)
    gm : float, optional
        Gravitational parameter GM (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    float
        Proper time rate. A value less than 1 indicates proper time passes
        slower than coordinate time.

    Examples
    --------
    Proper time rate for a GPS satellite at ~3.87 km/s and 26.56 Mm altitude:

    >>> v_gps = 3870.0  # m/s
    >>> r_gps = 26.56e6  # m
    >>> rate = proper_time_rate(v_gps, r_gps)
    >>> print(f"Proper time rate: {rate:.15f}")
    Proper time rate: 0.999999999749698
    >>> time_shift = (1 - rate) * 86400 * 1e6  # microseconds per day
    >>> print(f"Daily time shift: {time_shift:.1f} us/day")
    Daily time shift: 21.6 us/day
    """
    # Special relativistic effect
    special_rel = 1.0 - (v**2) / (2.0 * C_LIGHT**2)

    # General relativistic effect
    general_rel = -gm / (C_LIGHT**2 * r)

    return special_rel + general_rel


def shapiro_delay(
    observer_pos: NDArray[np.floating],
    light_source_pos: NDArray[np.floating],
    gravitating_body_pos: NDArray[np.floating],
    gm: float = GM_SUN,
) -> float:
    """Compute Shapiro time delay for light propagation through gravitational field.

    The Shapiro delay is the additional propagation time experienced by light
    traveling through a gravitational field, compared to flat spacetime.

    delay = (2GM/c^3) * ln((r_o + r_s + r_os) / (r_o + r_s - r_os))

    where r_o is distance from body to observer, r_s is distance from body to
    source, and r_os is distance from observer to source.

    Parameters
    ----------
    observer_pos : np.ndarray
        Position of observer (m), shape (3,)
    light_source_pos : np.ndarray
        Position of light source (m), shape (3,)
    gravitating_body_pos : np.ndarray
        Position of gravitating body (m), shape (3,)
    gm : float, optional
        Gravitational parameter GM (m^3/s^2). Default is GM_SUN.

    Returns
    -------
    float
        Shapiro delay (seconds)

    Examples
    --------
    Earth-Sun-Spacecraft signal at superior conjunction (worst case):

    >>> # Simplified geometry: Sun at origin, Earth at 1 AU, spacecraft beyond at distance
    >>> sun_pos = np.array([0.0, 0.0, 0.0])
    >>> earth_pos = np.array([1.496e11, 0.0, 0.0])  # 1 AU
    >>> spacecraft_pos = np.array([1.496e11, 1.0e11, 0.0])  # Far from sun
    >>> delay = shapiro_delay(earth_pos, spacecraft_pos, sun_pos, GM_SUN)
    >>> print(f"Shapiro delay: {delay:.3e} seconds")
    Shapiro delay: 6.173e-06 seconds
    >>> print(f"Shapiro delay: {delay*1e6:.1f} microseconds")
    Shapiro delay: 6.2 microseconds
    """
    # Compute distances
    r_observer = np.linalg.norm(observer_pos - gravitating_body_pos)
    r_source = np.linalg.norm(light_source_pos - gravitating_body_pos)
    r_os = np.linalg.norm(observer_pos - light_source_pos)

    # Shapiro delay formula (second-order PN)
    # Check for valid geometry (gravitating body should affect path)
    # The formula is valid when the impact parameter is close to the body
    numerator = r_observer + r_source + r_os
    denominator = r_observer + r_source - r_os

    # If denominator <= 0, it means the path doesn't pass near the gravitating body
    if denominator <= 0.0:
        # Return zero delay if geometry is invalid (light path doesn't bend)
        return 0.0

    delay = (2.0 * gm / (C_LIGHT**3)) * np.log(numerator / denominator)
    return delay


def schwarzschild_precession_per_orbit(a: float, e: float, gm: float = GM_SUN) -> float:
    """Compute perihelion precession per orbit due to general relativity.

    The advance of perihelion for an orbit around a central mass M is:

    Δφ = (6π * GM) / (c^2 * a * (1 - e^2))

    This effect is a key test of general relativity. For Mercury,
    the predicted precession is ~43 arcseconds per century.

    Parameters
    ----------
    a : float
        Semi-major axis (m)
    e : float
        Eccentricity (dimensionless), must be in [0, 1)
    gm : float, optional
        Gravitational parameter GM (m^3/s^2). Default is GM_SUN.

    Returns
    -------
    float
        Perihelion precession per orbit (radians)

    Examples
    --------
    Mercury's perihelion precession (GR contribution):

    >>> a_mercury = 0.38709927 * AU  # Semi-major axis in meters
    >>> e_mercury = 0.20563593     # Eccentricity
    >>> precession_rad = schwarzschild_precession_per_orbit(a_mercury, e_mercury, GM_SUN)
    >>> precession_arcsec = precession_rad * 206265  # Convert to arcseconds
    >>> orbital_period = 87.969  # days
    >>> centuries = 36525 / orbital_period  # Orbits per century
    >>> precession_per_century = precession_arcsec * centuries
    >>> print(f"GR perihelion precession: {precession_per_century:.2f} arcsec/century")
    GR perihelion precession: 42.98 arcsec/century
    """
    if e < 0 or e >= 1:
        raise ValueError(f"Eccentricity {e} must be in [0, 1)")

    precession = (6.0 * np.pi * gm) / (C_LIGHT**2 * a * (1.0 - e**2))
    return precession


def post_newtonian_acceleration(
    r_vec: NDArray[np.floating], v_vec: NDArray[np.floating], gm: float = GM_EARTH
) -> NDArray[np.floating]:
    """Compute total acceleration including first post-Newtonian corrections.

    Extends Newtonian gravity with the first post-Newtonian (Schwarzschild)
    correction in harmonic coordinates:

    a_1PN = (GM / (c^2 * r^3)) * [(4*GM/r - v.v) * r_vec + 4*(r_vec . v_vec) * v_vec]

    Parameters
    ----------
    r_vec : np.ndarray
        Position vector (m), shape (3,)
    v_vec : np.ndarray
        Velocity vector (m/s), shape (3,)
    gm : float, optional
        Gravitational parameter GM (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    np.ndarray
        Total acceleration including 1PN corrections (m/s^2), shape (3,)

    Examples
    --------
    Compare Newtonian and PN acceleration for a circular LEO satellite:

    >>> r = np.array([6.678e6, 0.0, 0.0])  # ~300 km altitude
    >>> v = np.array([0.0, 7.7e3, 0.0])    # Circular orbit velocity
    >>> a_total = post_newtonian_acceleration(r, v)
    >>> a_newt = -GM_EARTH / np.linalg.norm(r)**3 * r
    >>> ratio = np.linalg.norm(a_total - a_newt) / np.linalg.norm(a_newt)
    >>> print(f"PN correction ratio: {ratio:.3e}")
    PN correction ratio: 1.997e-09
    """
    r = np.linalg.norm(r_vec)
    v_squared = np.dot(v_vec, v_vec)
    r_dot_v = np.dot(r_vec, v_vec)

    # Newtonian acceleration
    a_newt = -gm / (r**3) * r_vec

    # First post-Newtonian (Schwarzschild) correction, harmonic coordinates
    a_1pn = (gm / (C_LIGHT**2 * r**3)) * (
        (4.0 * gm / r - v_squared) * r_vec + 4.0 * r_dot_v * v_vec
    )

    return a_newt + a_1pn


def geodetic_precession(
    a: float, e: float, inclination: float, gm: float = GM_EARTH
) -> float:
    """Compute geodetic (de Sitter) precession per orbit.

    A gyroscope (or the orbital angular momentum vector of a satellite)
    transported around a central mass precesses due to spacetime curvature.
    The geodetic precession accumulated over one orbit is:

    Δω = 3π * GM / (c^2 * a * (1 - e^2))

    The result is positive (prograde) and its magnitude is independent of
    the orbital inclination.

    Parameters
    ----------
    a : float
        Semi-major axis (m)
    e : float
        Eccentricity (dimensionless)
    inclination : float
        Orbital inclination (radians). Retained for API compatibility;
        the geodetic precession magnitude does not depend on it.
    gm : float, optional
        Gravitational parameter (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    float
        Geodetic precession per orbit (radians), positive (prograde).

    Examples
    --------
    Geodetic precession for a typical Earth satellite:

    >>> a = 6.678e6  # ~300 km altitude
    >>> e = 0.0      # Circular
    >>> i = np.radians(51.6)  # ISS-like inclination
    >>> rate = geodetic_precession(a, e, i)
    >>> print(f"Precession per orbit: {rate:.3e} rad")
    Precession per orbit: 6.259e-09 rad

    De Sitter precession of the Earth-Moon gyroscope orbiting the Sun:

    >>> rate = geodetic_precession(AU, 0.0167, 0.0, GM_SUN)
    >>> per_century = rate * 100 * 206264.806  # ~100 orbits per century
    >>> print(f"De Sitter precession: {per_century:.2f} arcsec/century")
    De Sitter precession: 1.92 arcsec/century
    """
    precession = 3.0 * np.pi * gm / (C_LIGHT**2 * a * (1.0 - e**2))
    return precession


def lense_thirring_precession(
    a: float,
    e: float,
    inclination: float,
    angular_momentum: float,
    gm: float = GM_EARTH,
) -> float:
    """Compute Lense-Thirring (frame-dragging) nodal precession rate.

    A rotating central body drags the orbital plane of nearby objects.
    The precession rate of the ascending node due to this effect is:

    Ω̇_LT = 2*G*J / (c^2 * a^3 * (1 - e^2)^(3/2))

    where J is the spin angular momentum of the central body. The rate is
    independent of the orbital inclination. Multiply by the orbital period
    to obtain the precession per orbit.

    Parameters
    ----------
    a : float
        Semi-major axis (m)
    e : float
        Eccentricity (dimensionless)
    inclination : float
        Orbital inclination (radians). Retained for API compatibility;
        the nodal Lense-Thirring rate does not depend on it.
    angular_momentum : float
        Spin angular momentum J of central body (kg·m^2/s).
        For Earth, J ≈ 5.86e33 kg·m^2/s.
    gm : float, optional
        Gravitational parameter (m^3/s^2). Default is GM_EARTH.
        Retained for API compatibility; the rate depends on G*J, not GM.

    Returns
    -------
    float
        Lense-Thirring nodal precession rate (radians per second),
        positive (prograde).

    Notes
    -----
    For Earth satellites, classical J_2 nodal precession dominates by many
    orders of magnitude; Lense-Thirring is a small correction (~31 mas/year
    for LAGEOS), first measured with the LAGEOS satellites.

    Examples
    --------
    Lense-Thirring effect for the LAGEOS satellite:

    >>> a = 12.27e6  # Semi-major axis (m)
    >>> e = 0.0045
    >>> i = np.radians(109.9)
    >>> J_earth = 5.86e33  # Earth's spin angular momentum (kg m^2/s)
    >>> rate = lense_thirring_precession(a, e, i, J_earth)
    >>> mas_per_year = rate * 86400 * 365.25 * 206264.806 * 1e3
    >>> print(f"LT nodal precession: {mas_per_year:.1f} mas/year")
    LT nodal precession: 30.7 mas/year
    """
    precession_rate = (2.0 * G_GRAV * angular_momentum) / (
        C_LIGHT**2 * a**3 * (1.0 - e**2) ** 1.5
    )

    return precession_rate


def relativistic_range_correction(
    r1: float, r2: float, rho: float, gm: float = GM_EARTH
) -> float:
    """Compute relativistic (Shapiro) range correction for ranging measurements.

    When measuring the distance between two points through a gravitational
    field (e.g., satellite laser ranging or GNSS pseudoranging), spacetime
    curvature delays the signal. The equivalent one-way range correction is:

    Δρ = (2*GM/c^2) * ln((r1 + r2 + ρ) / (r1 + r2 - ρ))

    where r1 and r2 are the distances of the endpoints from the center of
    the gravitating body and ρ is the geometric distance between them.

    Parameters
    ----------
    r1 : float
        Distance of first endpoint from the mass center (m)
    r2 : float
        Distance of second endpoint from the mass center (m)
    rho : float
        Geometric distance between the two endpoints (m)
    gm : float, optional
        Gravitational parameter (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    float
        One-way range correction (m), always positive (the measured range
        is longer than the geometric range).

    Examples
    --------
    Shapiro range correction for GPS-to-ground ranging (satellite overhead):

    >>> r_ground = 6.371e6   # Ground station radius (m)
    >>> r_gps = 26.561e6     # GPS orbit radius (m)
    >>> rho = r_gps - r_ground
    >>> correction = relativistic_range_correction(r_ground, r_gps, rho)
    >>> print(f"Range correction: {correction*1e3:.2f} mm")
    Range correction: 12.66 mm
    """
    return (2.0 * gm / C_LIGHT**2) * np.log((r1 + r2 + rho) / (r1 + r2 - rho))


__all__ = [
    # Constants
    "C_LIGHT",
    "G_GRAV",
    "GM_EARTH",
    "GM_SUN",
    "AU",
    # Functions
    "schwarzschild_radius",
    "gravitational_time_dilation",
    "proper_time_rate",
    "shapiro_delay",
    "schwarzschild_precession_per_orbit",
    "post_newtonian_acceleration",
    "geodetic_precession",
    "lense_thirring_precession",
    "relativistic_range_correction",
]
