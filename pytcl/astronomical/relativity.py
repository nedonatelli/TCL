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
    Sun's Schwarzschild radius: 2.952e+03 m
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
    Time dilation at surface: 0.999999999300693

    At GPS orbital altitude (~20,200 km):

    >>> r_gps = 26.56e6  # meters
    >>> dilation_gps = gravitational_time_dilation(r_gps)
    >>> time_shift = (1 - dilation_gps) * 86400 * 1e9  # nanoseconds per day
    >>> print(f"Time shift: {time_shift:.1f} ns/day")
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
    >>> time_shift = (1 - rate) * 86400  # seconds per day
    >>> print(f"Daily time shift: {time_shift:.3f} s/day")
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
    >>> print(f"Shapiro delay: {delay*1e6:.1f} microseconds")
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
    >>> print(f"GR perihelion precession: {precession_per_century:.1f} arcsec/century")
    GR perihelion precession: 42.98 arcsec/century
    """
    if e < 0 or e >= 1:
        raise ValueError(f"Eccentricity {e} must be in [0, 1)")

    precession = (6.0 * np.pi * gm) / (C_LIGHT**2 * a * (1.0 - e**2))
    return precession


def post_newtonian_acceleration(
    r_vec: NDArray[np.floating], v_vec: NDArray[np.floating], gm: float = GM_EARTH
) -> NDArray[np.floating]:
    """Compute post-Newtonian acceleration corrections (1PN order).

    Extends Newtonian gravity with first-order post-Newtonian corrections.

    a_PN = -GM/r^2 * u_r + a_1PN

    where a_1PN includes velocity-dependent and metric perturbation terms.

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
    Compare Newtonian and PN acceleration for LEO satellite:

    >>> r = np.array([6.678e6, 0.0, 0.0])  # ~300 km altitude
    >>> v = np.array([0.0, 7.7e3, 0.0])    # Circular orbit velocity
    >>> a_total = post_newtonian_acceleration(r, v)
    >>> a_newt = -GM_EARTH / np.linalg.norm(r)**3 * r
    >>> correction_ratio = np.linalg.norm(a_total - a_newt) / np.linalg.norm(a_newt)
    >>> print(f"PN correction: {correction_ratio*1e6:.1f} ppm")
    """
    r = np.linalg.norm(r_vec)
    v_squared = np.sum(v_vec**2)

    # Unit vector
    u_r = r_vec / r

    # Newtonian acceleration
    a_newt = -gm / (r**2) * u_r

    # 1PN corrections (in m/s^2)
    c2 = C_LIGHT**2

    # Term 1: Velocity squared effect on metric
    term1 = (gm / c2) * (2.0 * v_squared / r - 4.0 * gm / r) * u_r / r

    # Term 2: Radial velocity coupling
    v_dot_r = np.dot(v_vec, u_r)
    term2 = (4.0 * gm / c2) * v_dot_r * v_vec / r

    # Combine corrections (these are small corrections to Newtonian acceleration)
    a_1pn = term1 + term2

    return a_newt + a_1pn


def geodetic_precession(
    a: float, e: float, inclination: float, gm: float = GM_EARTH
) -> float:
    """Compute geodetic (de Sitter) precession rate of orbital plane.

    The orbital plane of a satellite precesses due to frame-dragging effects
    and spacetime curvature. The geodetic precession rate is:

    Ω_geodetic = -GM/(c^2 * a^3 * (1 - e^2)^2) * cos(i)

    Parameters
    ----------
    a : float
        Semi-major axis (m)
    e : float
        Eccentricity (dimensionless)
    inclination : float
        Orbital inclination (radians)
    gm : float, optional
        Gravitational parameter (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    float
        Geodetic precession rate (radians per orbit)

    Examples
    --------
    Geodetic precession for a typical Earth satellite:

    >>> a = 6.678e6  # ~300 km altitude
    >>> e = 0.0      # Circular
    >>> i = np.radians(51.6)  # ISS-like inclination
    >>> rate = geodetic_precession(a, e, i)
    >>> print(f"Precession per orbit: {rate*206265:.3f} arcsec")
    """
    p = a * (1.0 - e**2)
    precession = -(gm / (C_LIGHT**2 * p**2)) * np.cos(inclination)
    return precession


def lense_thirring_precession(
    a: float,
    e: float,
    inclination: float,
    angular_momentum: float,
    gm: float = GM_EARTH,
) -> float:
    """Compute Lense-Thirring (frame-dragging) precession of orbital node.

    A rotating central body drags the orbital plane of nearby objects.
    The nodal precession rate due to this effect is:

    Ω_LT = (2GM * J_2 * a * ω) / (c^2 * p^2) * f(e, i)

    where J_2 is the quadrupole moment, ω is angular velocity, and f depends
    on eccentricity and inclination.

    Parameters
    ----------
    a : float
        Semi-major axis (m)
    e : float
        Eccentricity (dimensionless)
    inclination : float
        Orbital inclination (radians)
    angular_momentum : float
        Angular momentum of central body (kg·m^2/s)
    gm : float, optional
        Gravitational parameter (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    float
        Lense-Thirring precession rate (radians per orbit)

    Notes
    -----
    This is a simplified version. For Earth, J_2 effects typically dominate
    classical nodal precession, while Lense-Thirring is a small correction
    (~1 arcsec per year for typical satellites).

    Examples
    --------
    Lense-Thirring effect for LAGEOS satellite:

    >>> # LAGEOS parameters
    >>> a = 12.27e6  # Semi-major axis
    >>> e = 0.0045
    >>> i = np.radians(109.9)
    >>> L_earth = 7.05e33  # Earth's angular momentum
    >>> rate = lense_thirring_precession(a, e, i, L_earth)
    >>> print(f"LT precession per orbit: {rate*206265*1e3:.1f} milliarcsec")
    """
    p = a * (1.0 - e**2)

    # Simplified Lense-Thirring term (second-order PN effect)
    # For a sphere: Lense-Thirring parameter = 2GM*L/(c^2*M*r^3)
    precession = (2.0 * angular_momentum * gm) / (C_LIGHT**2 * p**3)

    return precession


def relativistic_range_correction(
    distance: float, relative_velocity: float, gm: float = GM_EARTH
) -> float:
    """Compute relativistic range correction for ranging measurements.

    When measuring distance to a satellite or spacecraft using ranging
    (e.g., laser ranging), relativistic effects introduce corrections to
    the measured range.

    The main contributions are:
    - Gravitational time dilation
    - Relativistic Doppler effect

    Parameters
    ----------
    distance : float
        Distance to object (m)
    relative_velocity : float
        Radial velocity component (m/s, positive = receding)
    gm : float, optional
        Gravitational parameter (m^3/s^2). Default is GM_EARTH.

    Returns
    -------
    float
        Range correction (m)

    Examples
    --------
    Range correction for lunar laser ranging:

    >>> distance_to_moon = 3.84e8  # meters
    >>> radial_velocity = 0.0  # Average over orbit
    >>> correction = relativistic_range_correction(distance_to_moon, radial_velocity, GM_EARTH)
    >>> print(f"Range correction: {correction:.1f} m")
    """
    # Gravitational correction (positive because the signal is delayed)
    # Uses weak-field approximation
    grav_correction = gm / (C_LIGHT**2)

    # Doppler correction (second order effect, small)
    doppler_correction = (relative_velocity**2) / (3.0 * C_LIGHT**2)

    return grav_correction + doppler_correction
