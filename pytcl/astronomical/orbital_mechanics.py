"""
Two-body orbital mechanics and Kepler's equation.

This module provides functions for two-body orbit propagation, Kepler's
equation solvers, and orbital element conversions.

References
----------
.. [1] Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
       4th ed., Microcosm Press, 2013.
.. [2] Curtis, H. D., "Orbital Mechanics for Engineering Students,"
       3rd ed., Butterworth-Heinemann, 2014.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# Standard gravitational parameters (km^3/s^2)
GM_SUN = 1.32712440018e11
GM_EARTH = 3.986004418e5
GM_MOON = 4.9048695e3
GM_MARS = 4.282837e4
GM_JUPITER = 1.26686534e8


class OrbitalElements(NamedTuple):
    """Classical (Keplerian) orbital elements.

    Attributes
    ----------
    a : float
        Semi-major axis (km). Negative for hyperbolic orbits.
    e : float
        Eccentricity. 0=circular, 0<e<1=elliptic, e=1=parabolic, e>1=hyperbolic.
    i : float
        Inclination (radians), 0 to pi.
    raan : float
        Right ascension of ascending node (radians), 0 to 2*pi.
    omega : float
        Argument of periapsis (radians), 0 to 2*pi.
    nu : float
        True anomaly (radians), 0 to 2*pi.
    """

    a: float
    e: float
    i: float
    raan: float
    omega: float
    nu: float


class StateVector(NamedTuple):
    """Cartesian state vector.

    Attributes
    ----------
    r : ndarray
        Position vector (km), shape (3,).
    v : ndarray
        Velocity vector (km/s), shape (3,).
    """

    r: NDArray[np.floating]
    v: NDArray[np.floating]


def mean_to_eccentric_anomaly(
    M: float,
    e: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> float:
    """
    Solve Kepler's equation: M = E - e*sin(E).

    Uses Newton-Raphson iteration to find eccentric anomaly E
    given mean anomaly M and eccentricity e.

    Parameters
    ----------
    M : float
        Mean anomaly (radians).
    e : float
        Eccentricity (0 <= e < 1 for elliptic orbits).
    tol : float, optional
        Convergence tolerance. Default 1e-12.
    max_iter : int, optional
        Maximum iterations. Default 50.

    Returns
    -------
    E : float
        Eccentric anomaly (radians).

    Raises
    ------
    ValueError
        If eccentricity is not in valid range or iteration fails.

    Examples
    --------
    >>> import numpy as np
    >>> E = mean_to_eccentric_anomaly(np.pi/4, 0.5)
    >>> print(f"E = {np.degrees(E):.4f} degrees")
    """
    if e < 0 or e >= 1:
        raise ValueError(f"Eccentricity must be in [0, 1) for elliptic orbits, got {e}")

    # Normalize M to [0, 2*pi)
    M = M % (2 * np.pi)

    # Initial guess
    if e < 0.8:
        E = M
    else:
        E = np.pi

    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E = E - delta
        if abs(delta) < tol:
            return E

    raise ValueError(f"Kepler's equation did not converge after {max_iter} iterations")


def mean_to_hyperbolic_anomaly(
    M: float,
    e: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> float:
    """
    Solve hyperbolic Kepler's equation: M = e*sinh(H) - H.

    Uses Newton-Raphson iteration to find hyperbolic anomaly H
    given mean anomaly M and eccentricity e.

    Parameters
    ----------
    M : float
        Mean anomaly (radians).
    e : float
        Eccentricity (e > 1 for hyperbolic orbits).
    tol : float, optional
        Convergence tolerance. Default 1e-12.
    max_iter : int, optional
        Maximum iterations. Default 50.

    Returns
    -------
    H : float
        Hyperbolic anomaly (radians).

    Examples
    --------
    >>> H = mean_to_hyperbolic_anomaly(1.0, 1.5)
    >>> abs(1.5 * np.sinh(H) - H - 1.0) < 1e-10
    True
    """
    if e <= 1:
        raise ValueError(f"Eccentricity must be > 1 for hyperbolic orbits, got {e}")

    # Initial guess
    if e < 1.6:
        H = M if abs(M) < np.pi else np.sign(M) * np.pi
    else:
        H = np.sign(M) * np.log(2 * abs(M) / e + 1.8)

    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = e * np.sinh(H) - H - M
        f_prime = e * np.cosh(H) - 1
        delta = f / f_prime
        H = H - delta
        if abs(delta) < tol:
            return H

    raise ValueError(
        f"Hyperbolic Kepler's equation did not converge after {max_iter} iterations"
    )


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly to true anomaly.

    Parameters
    ----------
    E : float
        Eccentric anomaly (radians).
    e : float
        Eccentricity.

    Returns
    -------
    nu : float
        True anomaly (radians), in [0, 2*pi).

    Examples
    --------
    >>> nu = eccentric_to_true_anomaly(np.pi/4, 0.5)
    >>> 0 <= nu < 2 * np.pi
    True
    """
    # Use half-angle formula for numerical stability
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return nu % (2 * np.pi)


def true_to_eccentric_anomaly(nu: float, e: float) -> float:
    """
    Convert true anomaly to eccentric anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians).
    e : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly (radians), in [0, 2*pi).

    Examples
    --------
    >>> E = true_to_eccentric_anomaly(np.pi/3, 0.5)
    >>> 0 <= E < 2 * np.pi
    True
    """
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    return E % (2 * np.pi)


def hyperbolic_to_true_anomaly(H: float, e: float) -> float:
    """
    Convert hyperbolic anomaly to true anomaly.

    Parameters
    ----------
    H : float
        Hyperbolic anomaly (radians).
    e : float
        Eccentricity (e > 1).

    Returns
    -------
    nu : float
        True anomaly (radians).

    Examples
    --------
    >>> nu = hyperbolic_to_true_anomaly(0.5, 1.5)
    >>> isinstance(nu, float)
    True
    """
    nu = 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(H / 2))
    return nu


def true_to_hyperbolic_anomaly(nu: float, e: float) -> float:
    """
    Convert true anomaly to hyperbolic anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians).
    e : float
        Eccentricity (e > 1).

    Returns
    -------
    H : float
        Hyperbolic anomaly (radians).
    """
    H = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(nu / 2))
    return H


def eccentric_to_mean_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly to mean anomaly (Kepler's equation).

    Parameters
    ----------
    E : float
        Eccentric anomaly (radians).
    e : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly (radians).

    Examples
    --------
    >>> M = eccentric_to_mean_anomaly(np.pi/4, 0.5)
    >>> 0 <= M < 2 * np.pi
    True
    """
    M = E - e * np.sin(E)
    return M % (2 * np.pi)


def mean_to_true_anomaly(M: float, e: float) -> float:
    """
    Convert mean anomaly to true anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly (radians).
    e : float
        Eccentricity.

    Returns
    -------
    nu : float
        True anomaly (radians).

    Examples
    --------
    >>> nu = mean_to_true_anomaly(np.pi/4, 0.1)
    >>> 0 <= nu < 2 * np.pi
    True
    """
    if e < 1:
        E = mean_to_eccentric_anomaly(M, e)
        return eccentric_to_true_anomaly(E, e)
    else:
        H = mean_to_hyperbolic_anomaly(M, e)
        return hyperbolic_to_true_anomaly(H, e)


def true_to_mean_anomaly(nu: float, e: float) -> float:
    """
    Convert true anomaly to mean anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians).
    e : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly (radians).
    """
    if e < 1:
        E = true_to_eccentric_anomaly(nu, e)
        return eccentric_to_mean_anomaly(E, e)
    else:
        H = true_to_hyperbolic_anomaly(nu, e)
        return e * np.sinh(H) - H


def orbital_elements_to_state(
    elements: OrbitalElements,
    mu: float = GM_EARTH,
) -> StateVector:
    """
    Convert orbital elements to Cartesian state vector.

    Parameters
    ----------
    elements : OrbitalElements
        Classical orbital elements.
    mu : float, optional
        Gravitational parameter (km^3/s^2). Default is Earth.

    Returns
    -------
    state : StateVector
        Position and velocity vectors in inertial frame.

    Examples
    --------
    >>> elements = OrbitalElements(a=7000, e=0.01, i=0.5, raan=0, omega=0, nu=0)
    >>> state = orbital_elements_to_state(elements)
    >>> print(f"r = {state.r}")
    """
    a, e, i, raan, omega, nu = elements

    # Semi-latus rectum
    if abs(e - 1) < 1e-10:
        raise ValueError("Parabolic orbits (e=1) not supported")
    p = a * (1 - e * e)

    # Position and velocity in perifocal frame
    r_pqw = p / (1 + e * np.cos(nu))

    r_perifocal = np.array([r_pqw * np.cos(nu), r_pqw * np.sin(nu), 0.0])

    v_perifocal = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix from perifocal to inertial (ECI)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    R = np.array(
        [
            [
                cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
                -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
                sin_raan * sin_i,
            ],
            [
                sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
                -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
                -cos_raan * sin_i,
            ],
            [sin_omega * sin_i, cos_omega * sin_i, cos_i],
        ]
    )

    r = R @ r_perifocal
    v = R @ v_perifocal

    return StateVector(r=r, v=v)


def state_to_orbital_elements(
    state: StateVector,
    mu: float = GM_EARTH,
) -> OrbitalElements:
    """
    Convert Cartesian state vector to orbital elements.

    Parameters
    ----------
    state : StateVector
        Position and velocity vectors.
    mu : float, optional
        Gravitational parameter (km^3/s^2). Default is Earth.

    Returns
    -------
    elements : OrbitalElements
        Classical orbital elements.

    Examples
    --------
    >>> r = np.array([7000, 0, 0])
    >>> v = np.array([0, 7.5, 0])
    >>> state = StateVector(r=r, v=v)
    >>> elements = state_to_orbital_elements(state)
    """
    r = np.asarray(state.r, dtype=float)
    v = np.asarray(state.v, dtype=float)

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = ((v_mag * v_mag - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # Specific mechanical energy
    energy = v_mag * v_mag / 2 - mu / r_mag

    # Semi-major axis
    if abs(e - 1) < 1e-10:
        # Parabolic
        a = np.inf
    else:
        a = -mu / (2 * energy)

    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    # Right ascension of ascending node
    if n_mag > 1e-10:
        raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        # Equatorial orbit
        raan = 0.0

    # Argument of periapsis
    if n_mag > 1e-10 and e > 1e-10:
        omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    elif e > 1e-10:
        # Equatorial orbit with eccentricity
        omega = np.arctan2(e_vec[1], e_vec[0])
        if omega < 0:
            omega += 2 * np.pi
    else:
        omega = 0.0

    # True anomaly
    if e > 1e-10:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        # Circular orbit
        if n_mag > 1e-10:
            nu = np.arccos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1, 1))
            if r[2] < 0:
                nu = 2 * np.pi - nu
        else:
            # Circular equatorial
            nu = np.arctan2(r[1], r[0])
            if nu < 0:
                nu += 2 * np.pi

    return OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu)


def orbital_period(a: float, mu: float = GM_EARTH) -> float:
    """
    Compute orbital period for elliptic orbit.

    Parameters
    ----------
    a : float
        Semi-major axis (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    T : float
        Orbital period (seconds).

    Examples
    --------
    >>> T = orbital_period(7000)  # LEO satellite
    >>> T / 60  # Convert to minutes  # doctest: +SKIP
    97.8...
    """
    if a <= 0:
        raise ValueError("Semi-major axis must be positive for elliptic orbits")
    return 2 * np.pi * np.sqrt(a**3 / mu)


def mean_motion(a: float, mu: float = GM_EARTH) -> float:
    """
    Compute mean motion.

    Parameters
    ----------
    a : float
        Semi-major axis (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    n : float
        Mean motion (radians/second).

    Examples
    --------
    >>> n = mean_motion(42164)  # GEO orbit
    >>> revs_per_day = n * 86400 / (2 * np.pi)
    >>> abs(revs_per_day - 1.0) < 0.01  # Approximately 1 rev/day
    True
    """
    return np.sqrt(mu / abs(a) ** 3)


def kepler_propagate(
    elements: OrbitalElements,
    dt: float,
    mu: float = GM_EARTH,
) -> OrbitalElements:
    """
    Propagate orbital elements using Kepler's equation.

    This performs two-body propagation by advancing the mean anomaly
    and solving Kepler's equation for the new true anomaly.

    Parameters
    ----------
    elements : OrbitalElements
        Initial orbital elements.
    dt : float
        Time step (seconds).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    new_elements : OrbitalElements
        Propagated orbital elements.

    Examples
    --------
    >>> elements = OrbitalElements(a=7000, e=0.01, i=0.5, raan=0, omega=0, nu=0)
    >>> new_elements = kepler_propagate(elements, 3600)  # 1 hour
    """
    a, e, i, raan, omega, nu = elements

    # Compute mean motion
    n = mean_motion(a, mu)

    # Convert true anomaly to mean anomaly
    M0 = true_to_mean_anomaly(nu, e)

    # Propagate mean anomaly
    M = M0 + n * dt

    # Convert back to true anomaly
    nu_new = mean_to_true_anomaly(M, e)

    return OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu_new)


def kepler_propagate_state(
    state: StateVector,
    dt: float,
    mu: float = GM_EARTH,
) -> StateVector:
    """
    Propagate state vector using Kepler's equation.

    Parameters
    ----------
    state : StateVector
        Initial state vector.
    dt : float
        Time step (seconds).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    new_state : StateVector
        Propagated state vector.

    Examples
    --------
    >>> r = np.array([7000.0, 0.0, 0.0])
    >>> v = np.array([0.0, 7.5, 0.0])
    >>> state = StateVector(r=r, v=v)
    >>> new_state = kepler_propagate_state(state, 3600)
    >>> np.linalg.norm(new_state.r) > 0
    True
    """
    elements = state_to_orbital_elements(state, mu)
    new_elements = kepler_propagate(elements, dt, mu)
    return orbital_elements_to_state(new_elements, mu)


def vis_viva(r: float, a: float, mu: float = GM_EARTH) -> float:
    """
    Compute orbital velocity using vis-viva equation.

    Parameters
    ----------
    r : float
        Current orbital radius (km).
    a : float
        Semi-major axis (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    v : float
        Orbital velocity (km/s).

    Examples
    --------
    >>> v = vis_viva(7000, 7000)  # Circular orbit
    >>> abs(v - circular_velocity(7000)) < 0.01
    True
    """
    return np.sqrt(mu * (2 / r - 1 / a))


def specific_angular_momentum(
    state: StateVector,
) -> NDArray[np.floating]:
    """
    Compute specific angular momentum vector.

    Parameters
    ----------
    state : StateVector
        State vector.

    Returns
    -------
    h : ndarray
        Specific angular momentum vector (km^2/s).

    Examples
    --------
    >>> r = np.array([7000.0, 0.0, 0.0])
    >>> v = np.array([0.0, 7.5, 0.0])
    >>> state = StateVector(r=r, v=v)
    >>> h = specific_angular_momentum(state)
    >>> h[2]  # Angular momentum in z-direction
    52500.0
    """
    return np.cross(state.r, state.v)


def specific_orbital_energy(
    state: StateVector,
    mu: float = GM_EARTH,
) -> float:
    """
    Compute specific orbital energy.

    Parameters
    ----------
    state : StateVector
        State vector.
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    energy : float
        Specific orbital energy (km^2/s^2).
        Negative for bound orbits, positive for escape trajectories.

    Examples
    --------
    >>> r = np.array([7000.0, 0.0, 0.0])
    >>> v = np.array([0.0, 7.5, 0.0])
    >>> state = StateVector(r=r, v=v)
    >>> energy = specific_orbital_energy(state)
    >>> energy < 0  # Bound orbit
    True
    """
    r_mag = np.linalg.norm(state.r)
    v_mag = np.linalg.norm(state.v)
    return v_mag * v_mag / 2 - mu / r_mag


def flight_path_angle(state: StateVector) -> float:
    """
    Compute flight path angle.

    Parameters
    ----------
    state : StateVector
        State vector.

    Returns
    -------
    gamma : float
        Flight path angle (radians).
        Positive when climbing, negative when descending.

    Examples
    --------
    >>> r = np.array([7000.0, 0.0, 0.0])
    >>> v = np.array([0.0, 7.5, 0.0])  # Tangential velocity
    >>> state = StateVector(r=r, v=v)
    >>> gamma = flight_path_angle(state)
    >>> abs(gamma) < 0.01  # Nearly zero for circular motion
    True
    """
    r = np.asarray(state.r)
    v = np.asarray(state.v)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Radial and transverse velocity components
    v_r = np.dot(r, v) / r_mag
    v_t = np.sqrt(v_mag * v_mag - v_r * v_r)

    return np.arctan2(v_r, v_t)


def periapsis_radius(a: float, e: float) -> float:
    """
    Compute periapsis radius.

    Parameters
    ----------
    a : float
        Semi-major axis (km).
    e : float
        Eccentricity.

    Returns
    -------
    r_p : float
        Periapsis radius (km).

    Examples
    --------
    >>> r_p = periapsis_radius(10000, 0.3)
    >>> r_p
    7000.0
    """
    return a * (1 - e)


def apoapsis_radius(a: float, e: float) -> float:
    """
    Compute apoapsis radius.

    Parameters
    ----------
    a : float
        Semi-major axis (km).
    e : float
        Eccentricity.

    Returns
    -------
    r_a : float
        Apoapsis radius (km). Infinite for parabolic/hyperbolic orbits.

    Examples
    --------
    >>> r_a = apoapsis_radius(10000, 0.3)
    >>> r_a
    13000.0
    """
    if e >= 1:
        return np.inf
    return a * (1 + e)


def time_since_periapsis(
    nu: float,
    a: float,
    e: float,
    mu: float = GM_EARTH,
) -> float:
    """
    Compute time since periapsis passage.

    Parameters
    ----------
    nu : float
        True anomaly (radians).
    a : float
        Semi-major axis (km).
    e : float
        Eccentricity.
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    t : float
        Time since periapsis (seconds).

    Examples
    --------
    >>> t = time_since_periapsis(np.pi, 7000, 0.1)  # At apoapsis
    >>> T = orbital_period(7000)
    >>> abs(t - T/2) < 1  # Approximately half the period
    True
    """
    M = true_to_mean_anomaly(nu, e)
    n = mean_motion(a, mu)
    return M / n


def orbit_radius(nu: float, a: float, e: float) -> float:
    """
    Compute orbital radius at given true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians).
    a : float
        Semi-major axis (km).
    e : float
        Eccentricity.

    Returns
    -------
    r : float
        Orbital radius (km).

    Examples
    --------
    >>> r = orbit_radius(0, 10000, 0.3)  # At periapsis
    >>> r
    7000.0
    >>> r = orbit_radius(np.pi, 10000, 0.3)  # At apoapsis
    >>> r
    13000.0
    """
    p = a * (1 - e * e)
    return p / (1 + e * np.cos(nu))


def escape_velocity(r: float, mu: float = GM_EARTH) -> float:
    """
    Compute escape velocity at given radius.

    Parameters
    ----------
    r : float
        Radial distance (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    v_esc : float
        Escape velocity (km/s).

    Examples
    --------
    >>> v_esc = escape_velocity(6378 + 400)  # At ISS altitude
    >>> 10 < v_esc < 12  # About 11 km/s
    True
    """
    return np.sqrt(2 * mu / r)


def circular_velocity(r: float, mu: float = GM_EARTH) -> float:
    """
    Compute circular orbital velocity at given radius.

    Parameters
    ----------
    r : float
        Orbital radius (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    v_circ : float
        Circular velocity (km/s).

    Examples
    --------
    >>> v_circ = circular_velocity(6378 + 400)  # At ISS altitude
    >>> 7 < v_circ < 8  # About 7.7 km/s
    True
    """
    return np.sqrt(mu / r)


__all__ = [
    # Constants
    "GM_SUN",
    "GM_EARTH",
    "GM_MOON",
    "GM_MARS",
    "GM_JUPITER",
    # Types
    "OrbitalElements",
    "StateVector",
    # Anomaly conversions
    "mean_to_eccentric_anomaly",
    "mean_to_hyperbolic_anomaly",
    "eccentric_to_true_anomaly",
    "true_to_eccentric_anomaly",
    "hyperbolic_to_true_anomaly",
    "true_to_hyperbolic_anomaly",
    "eccentric_to_mean_anomaly",
    "mean_to_true_anomaly",
    "true_to_mean_anomaly",
    # Element conversions
    "orbital_elements_to_state",
    "state_to_orbital_elements",
    # Propagation
    "kepler_propagate",
    "kepler_propagate_state",
    # Orbital quantities
    "orbital_period",
    "mean_motion",
    "vis_viva",
    "specific_angular_momentum",
    "specific_orbital_energy",
    "flight_path_angle",
    "periapsis_radius",
    "apoapsis_radius",
    "time_since_periapsis",
    "orbit_radius",
    "escape_velocity",
    "circular_velocity",
]
