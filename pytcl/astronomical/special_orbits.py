"""
Special orbit cases: parabolic and advanced hyperbolic orbits.

This module extends orbital_mechanics.py with handling for edge cases:
- Parabolic orbits (e = 1, unbounded trajectory with zero energy)
- Advanced hyperbolic orbit calculations (escape trajectories)
- Unified orbit type detection and handling

References
----------
.. [1] Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
       4th ed., Microcosm Press, 2013.
.. [2] Curtis, H. D., "Orbital Mechanics for Engineering Students,"
       3rd ed., Butterworth-Heinemann, 2014.
.. [3] Battin, R. H., "An Introduction to the Mathematics and Methods
       of Astrodynamics," 2nd ed., AIAA, 1999.
"""

from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class OrbitType(Enum):
    """Classification of orbit types based on eccentricity."""

    CIRCULAR = 0  # e = 0
    ELLIPTICAL = 1  # 0 < e < 1
    PARABOLIC = 2  # e = 1 (boundary case)
    HYPERBOLIC = 3  # e > 1


class ParabolicElements(NamedTuple):
    """Parabolic (escape) orbit elements.

    For a parabolic orbit (e=1), the semi-major axis is infinite.
    Instead, we use the periapsis distance and orientation parameters.

    Attributes
    ----------
    rp : float
        Periapsis distance (km). Also called pericenter or closest approach.
    i : float
        Inclination (radians), 0 to pi.
    raan : float
        Right ascension of ascending node (radians), 0 to 2*pi.
    omega : float
        Argument of periapsis (radians), 0 to 2*pi.
    nu : float
        True anomaly (radians), typically in [-pi, pi] for parabolic orbits.
    """

    rp: float
    i: float
    raan: float
    omega: float
    nu: float


def classify_orbit(e: float, tol: float = 1e-9) -> OrbitType:
    """
    Classify orbit type based on eccentricity.

    Parameters
    ----------
    e : float
        Eccentricity value.
    tol : float, optional
        Tolerance for parabolic classification (default 1e-9).
        Orbits with abs(e - 1) < tol are classified as parabolic.

    Returns
    -------
    OrbitType
        Classified orbit type.

    Raises
    ------
    ValueError
        If eccentricity is negative or NaN.
    """
    if np.isnan(e) or e < 0:
        raise ValueError(f"Eccentricity must be non-negative, got {e}")

    if e < tol:
        return OrbitType.CIRCULAR
    elif abs(e - 1.0) < tol:
        return OrbitType.PARABOLIC
    elif e < 1 - tol:
        return OrbitType.ELLIPTICAL
    else:
        return OrbitType.HYPERBOLIC


def mean_to_parabolic_anomaly(
    M: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """
    Solve parabolic Kepler's equation: M = D + (1/3)*D^3.

    For parabolic orbits (e=1), the "anomaly" is the parabolic anomaly D,
    related to true anomaly by: D = tan(nu/2).

    The equation relates mean anomaly to parabolic anomaly:
    M = D + (1/3)*D^3

    This is solved numerically using Newton-Raphson iteration.

    Parameters
    ----------
    M : float
        Mean anomaly (radians).
    tol : float, optional
        Convergence tolerance (default 1e-12).
    max_iter : int, optional
        Maximum iterations (default 100).

    Returns
    -------
    D : float
        Parabolic anomaly (the parameter D such that tan(nu/2) = D).

    Notes
    -----
    For parabolic orbits, mean anomaly relates to time as:
    M = sqrt(mu/rp^3) * t where rp is periapsis distance and mu is GM.

    The solution D satisfies: D + (1/3)*D^3 = M
    """
    # Newton-Raphson for parabolic anomaly
    # f(D) = D + (1/3)*D^3 - M = 0
    # f'(D) = 1 + D^2

    D = M  # Initial guess

    for _ in range(max_iter):
        f = D + (1.0 / 3.0) * D**3 - M
        f_prime = 1.0 + D**2
        delta = f / f_prime
        D = D - delta

        if abs(delta) < tol:
            return D

    raise ValueError(
        f"Parabolic Kepler's equation did not converge after {max_iter} iterations"
    )


def parabolic_anomaly_to_true_anomaly(D: float) -> float:
    """
    Convert parabolic anomaly to true anomaly.

    For parabolic orbits, the parabolic anomaly D relates to true anomaly by:
    tan(nu/2) = D

    Parameters
    ----------
    D : float
        Parabolic anomaly (the parameter such that tan(nu/2) = D).

    Returns
    -------
    nu : float
        True anomaly (radians), in [-pi, pi].
    """
    return 2.0 * np.arctan(D)


def true_anomaly_to_parabolic_anomaly(nu: float) -> float:
    """
    Convert true anomaly to parabolic anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians).

    Returns
    -------
    D : float
        Parabolic anomaly.
    """
    return np.tan(nu / 2.0)


def mean_to_true_anomaly_parabolic(M: float, tol: float = 1e-12) -> float:
    """
    Direct conversion from mean to true anomaly for parabolic orbits.

    Parameters
    ----------
    M : float
        Mean anomaly (radians).
    tol : float, optional
        Convergence tolerance (default 1e-12).

    Returns
    -------
    nu : float
        True anomaly (radians).
    """
    D = mean_to_parabolic_anomaly(M, tol=tol)
    return parabolic_anomaly_to_true_anomaly(D)


def radius_parabolic(rp: float, nu: float) -> float:
    """
    Compute radius for parabolic orbit.

    For a parabolic orbit with periapsis distance rp and true anomaly nu:
    r = 2*rp / (1 + cos(nu))

    This formula is consistent with the general conic section equation
    with e=1: r = p/(1 + e*cos(nu)) where p = 2*rp (semi-latus rectum).

    Parameters
    ----------
    rp : float
        Periapsis distance (km).
    nu : float
        True anomaly (radians).

    Returns
    -------
    r : float
        Orbital radius (km).

    Raises
    ------
    ValueError
        If radius would be negative (nu near +pi for parabolic orbit).
    """
    denom = 1.0 + np.cos(nu)

    if denom <= 0:
        raise ValueError(
            f"Parabolic orbit undefined at true anomaly nu={np.degrees(nu):.2f}°"
        )

    r = 2.0 * rp / denom

    if r < 0:
        raise ValueError(f"Computed radius is negative: r={r}")

    return r


def velocity_parabolic(mu: float, rp: float, nu: float) -> float:
    """
    Compute velocity magnitude for parabolic orbit.

    For a parabolic orbit (e=1), the specific orbital energy is zero,
    and the velocity relates to radius by:
    v = sqrt(2*mu/r)

    Parameters
    ----------
    mu : float
        Standard gravitational parameter (km^3/s^2).
    rp : float
        Periapsis distance (km).
    nu : float
        True anomaly (radians).

    Returns
    -------
    v : float
        Velocity magnitude (km/s).
    """
    r = radius_parabolic(rp, nu)
    return np.sqrt(2.0 * mu / r)


def hyperbolic_anomaly_to_true_anomaly(H: float, e: float) -> float:
    """
    Convert hyperbolic anomaly to true anomaly.

    For hyperbolic orbits (e > 1), hyperbolic anomaly H relates to true anomaly by:
    tan(nu/2) = sqrt((e+1)/(e-1)) * tanh(H/2)

    Parameters
    ----------
    H : float
        Hyperbolic anomaly (radians).
    e : float
        Eccentricity (e > 1 for hyperbolic).

    Returns
    -------
    nu : float
        True anomaly (radians).

    Raises
    ------
    ValueError
        If eccentricity is not hyperbolic (e <= 1).
    """
    if e <= 1:
        raise ValueError(f"Eccentricity must be > 1 for hyperbolic orbits, got {e}")

    nu = 2.0 * np.arctan(np.sqrt((e + 1.0) / (e - 1.0)) * np.tanh(H / 2.0))

    return nu


def true_anomaly_to_hyperbolic_anomaly(nu: float, e: float) -> float:
    """
    Convert true anomaly to hyperbolic anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians).
    e : float
        Eccentricity (e > 1 for hyperbolic).

    Returns
    -------
    H : float
        Hyperbolic anomaly (radians).

    Raises
    ------
    ValueError
        If eccentricity is not hyperbolic.
    """
    if e <= 1:
        raise ValueError(f"Eccentricity must be > 1 for hyperbolic orbits, got {e}")

    H = 2.0 * np.arctanh(np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu / 2.0))

    return H


def escape_velocity_at_radius(mu: float, r: float) -> float:
    """
    Compute escape velocity at a given radius.

    Escape velocity is the minimum velocity needed to reach infinity
    with zero velocity, corresponding to a parabolic orbit:
    v_esc = sqrt(2*mu/r)

    Parameters
    ----------
    mu : float
        Standard gravitational parameter (km^3/s^2).
    r : float
        Orbital radius (km).

    Returns
    -------
    v_esc : float
        Escape velocity (km/s).
    """
    return np.sqrt(2.0 * mu / r)


def hyperbolic_excess_velocity(mu: float, a: float) -> float:
    """
    Compute hyperbolic excess velocity.

    For a hyperbolic orbit with semi-major axis a (negative for hyperbolic),
    the excess velocity at infinity is:
    v_inf = sqrt(-mu/a)

    Parameters
    ----------
    mu : float
        Standard gravitational parameter (km^3/s^2).
    a : float
        Semi-major axis (km). Must be negative for hyperbolic orbits.

    Returns
    -------
    v_inf : float
        Hyperbolic excess velocity (km/s).

    Raises
    ------
    ValueError
        If semi-major axis is not negative.
    """
    if a >= 0:
        raise ValueError(
            f"Semi-major axis must be negative for hyperbolic orbits, got {a}"
        )

    v_inf = np.sqrt(-mu / a)
    return v_inf


def hyperbolic_asymptote_angle(e: float) -> float:
    """
    Compute the asymptote angle for a hyperbolic orbit.

    For a hyperbolic orbit with eccentricity e, the true anomaly
    asymptotically approaches ±nu_inf where:
    cos(nu_inf) = -1/e

    The asymptote angle is nu_inf.

    Parameters
    ----------
    e : float
        Eccentricity (e > 1 for hyperbolic).

    Returns
    -------
    nu_inf : float
        Asymptote angle (radians), in (0, pi).

    Raises
    ------
    ValueError
        If eccentricity is not hyperbolic.
    """
    if e <= 1:
        raise ValueError(f"Eccentricity must be > 1 for hyperbolic orbits, got {e}")

    nu_inf = np.arccos(-1.0 / e)
    return nu_inf


def hyperbolic_deflection_angle(e: float) -> float:
    """
    Compute the deflection angle for a hyperbolic orbit.

    The deflection angle is the angle through which the velocity vector
    is deflected from its asymptotic direction:
    delta = pi - 2*nu_inf = pi - 2*arccos(-1/e)

    Parameters
    ----------
    e : float
        Eccentricity (e > 1 for hyperbolic).

    Returns
    -------
    delta : float
        Deflection angle (radians), in (0, pi).

    Raises
    ------
    ValueError
        If eccentricity is not hyperbolic.
    """
    if e <= 1:
        raise ValueError(f"Eccentricity must be > 1 for hyperbolic orbits, got {e}")

    nu_inf = hyperbolic_asymptote_angle(e)
    delta = np.pi - 2.0 * nu_inf

    return delta


def semi_major_axis_from_energy(mu: float, specific_energy: float) -> float:
    """
    Compute semi-major axis from specific orbital energy.

    The specific orbital energy relates to semi-major axis by:
    epsilon = -mu / (2*a)

    Rearranging: a = -mu / (2*epsilon)

    Parameters
    ----------
    mu : float
        Standard gravitational parameter (km^3/s^2).
    specific_energy : float
        Specific orbital energy (km^2/s^2).

    Returns
    -------
    a : float
        Semi-major axis (km).
        - a > 0 for elliptical orbits (epsilon < 0)
        - a < 0 for hyperbolic orbits (epsilon > 0)
        - a → ∞ for parabolic orbits (epsilon = 0)

    Raises
    ------
    ValueError
        If specific energy is exactly zero (parabolic case).
    """
    if abs(specific_energy) < 1e-15:
        raise ValueError(
            "Specific energy is zero (parabolic orbit); use alternative methods"
        )

    a = -mu / (2.0 * specific_energy)
    return a


def eccentricity_vector(
    r: NDArray[np.floating],
    v: NDArray[np.floating],
    mu: float,
) -> NDArray[np.floating]:
    """
    Compute eccentricity vector from position and velocity.

    The eccentricity vector e is defined as:
    e = (v^2/mu - 1/r) * r - (r·v/mu) * v

    This works for all orbit types: elliptical, parabolic, and hyperbolic.

    Parameters
    ----------
    r : ndarray
        Position vector (km), shape (3,).
    v : ndarray
        Velocity vector (km/s), shape (3,).
    mu : float
        Standard gravitational parameter (km^3/s^2).

    Returns
    -------
    e : ndarray
        Eccentricity vector, shape (3,).
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    rv_dot = np.dot(r, v)

    e_vec = (v_mag**2 / mu - 1.0 / r_mag) * r - (rv_dot / mu) * v

    return e_vec
