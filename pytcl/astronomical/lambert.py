"""
Lambert's problem solver.

Lambert's problem determines the orbit connecting two position vectors
given the time of flight. This is fundamental for orbital transfer
calculations, trajectory design, and orbit determination.

References
----------
.. [1] Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
       4th ed., Microcosm Press, 2013.
.. [2] Izzo, D., "Revisiting Lambert's problem," Celestial Mechanics and
       Dynamical Astronomy, 2015.
.. [3] Gooding, R. H., "A procedure for the solution of Lambert's
       orbital boundary-value problem," Celestial Mechanics, 1990.
"""

from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.astronomical.orbital_mechanics import GM_EARTH


class LambertSolution(NamedTuple):
    """Solution to Lambert's problem.

    Attributes
    ----------
    v1 : ndarray
        Velocity at first position (km/s), shape (3,).
    v2 : ndarray
        Velocity at second position (km/s), shape (3,).
    a : float
        Semi-major axis of transfer orbit (km).
    e : float
        Eccentricity of transfer orbit.
    tof : float
        Time of flight (seconds).
    """

    v1: NDArray[np.floating]
    v2: NDArray[np.floating]
    a: float
    e: float
    tof: float


def _stumpff_c2(psi: float) -> float:
    """Stumpff function c2(psi)."""
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        return (1 - np.cos(sqrt_psi)) / psi
    elif psi < -1e-6:
        sqrt_neg_psi = np.sqrt(-psi)
        return (1 - np.cosh(sqrt_neg_psi)) / psi
    else:
        # Taylor series for small psi
        return 1 / 2 - psi / 24 + psi * psi / 720


def _stumpff_c3(psi: float) -> float:
    """Stumpff function c3(psi)."""
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        return (sqrt_psi - np.sin(sqrt_psi)) / (psi * sqrt_psi)
    elif psi < -1e-6:
        sqrt_neg_psi = np.sqrt(-psi)
        return (np.sinh(sqrt_neg_psi) - sqrt_neg_psi) / ((-psi) * sqrt_neg_psi)
    else:
        # Taylor series for small psi
        return 1 / 6 - psi / 120 + psi * psi / 5040


def lambert_universal(
    r1: NDArray[np.floating],
    r2: NDArray[np.floating],
    tof: float,
    mu: float = GM_EARTH,
    prograde: bool = True,
    low_path: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> LambertSolution:
    """
    Solve Lambert's problem using universal variables.

    Given two position vectors and time of flight, determine the
    transfer orbit connecting them.

    Parameters
    ----------
    r1 : ndarray
        Initial position vector (km), shape (3,).
    r2 : ndarray
        Final position vector (km), shape (3,).
    tof : float
        Time of flight (seconds). Must be positive.
    mu : float, optional
        Gravitational parameter (km^3/s^2). Default is Earth.
    prograde : bool, optional
        If True, use prograde (counterclockwise) transfer.
        If False, use retrograde transfer. Default True.
    low_path : bool, optional
        If True, use low energy (short way) transfer.
        If False, use high energy (long way) transfer. Default True.
    max_iter : int, optional
        Maximum iterations. Default 100.
    tol : float, optional
        Convergence tolerance. Default 1e-10.

    Returns
    -------
    solution : LambertSolution
        Solution containing velocities and orbital parameters.

    Raises
    ------
    ValueError
        If solution does not converge.

    Examples
    --------
    >>> r1 = np.array([5000, 10000, 2100])  # km
    >>> r2 = np.array([-14600, 2500, 7000])  # km
    >>> tof = 3600  # 1 hour
    >>> sol = lambert_universal(r1, r2, tof)
    >>> print(f"v1 = {sol.v1} km/s")
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Cross product to determine direction
    cross = np.cross(r1, r2)

    # Determine transfer angle
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1, 1)

    # Determine direction of motion
    if prograde:
        if cross[2] >= 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2 * np.pi - np.arccos(cos_dnu)
    else:
        if cross[2] < 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2 * np.pi - np.arccos(cos_dnu)

    # Short way vs long way
    if not low_path:
        dnu = 2 * np.pi - dnu

    sin_dnu = np.sin(dnu)

    # Chord and semi-perimeter
    A = sin_dnu * np.sqrt(r1_mag * r2_mag / (1 - cos_dnu))

    if A == 0:
        raise ValueError("Cannot solve Lambert problem: A = 0 (degenerate case)")

    # Initial guess for psi (universal variable)
    psi = 0.0
    psi_low = -4 * np.pi * np.pi
    psi_high = 4 * np.pi * np.pi

    # Newton iteration
    for iteration in range(max_iter):
        c2 = _stumpff_c2(psi)
        c3 = _stumpff_c3(psi)

        y = r1_mag + r2_mag + A * (psi * c3 - 1) / np.sqrt(c2)

        if y < 0:
            # Adjust bounds
            psi_low = psi
            psi = (psi_low + psi_high) / 2
            continue

        chi = np.sqrt(y / c2)
        tof_calc = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)

        if abs(tof_calc - tof) < tol:
            break

        # Newton-Raphson update
        if tof_calc <= tof:
            psi_low = psi
        else:
            psi_high = psi

        psi = (psi_low + psi_high) / 2

    else:
        raise ValueError(
            f"Lambert's problem did not converge after {max_iter} iterations"
        )

    # Compute f, g, f_dot, g_dot
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag

    # Compute velocities
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    # Compute orbital elements of transfer orbit
    # Semi-major axis from energy
    v1_mag = np.linalg.norm(v1)
    energy = v1_mag * v1_mag / 2 - mu / r1_mag
    if abs(energy) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = np.inf

    # Eccentricity from angular momentum and energy
    h = np.cross(r1, v1)
    h_mag = np.linalg.norm(h)
    if abs(energy) > 1e-10:
        ecc = np.sqrt(1 + 2 * energy * h_mag * h_mag / (mu * mu))
    else:
        ecc = 1.0

    return LambertSolution(v1=v1, v2=v2, a=a, e=ecc, tof=tof)


def lambert_izzo(
    r1: NDArray[np.floating],
    r2: NDArray[np.floating],
    tof: float,
    mu: float = GM_EARTH,
    prograde: bool = True,
    multi_rev: int = 0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> LambertSolution:
    """
    Solve Lambert's problem using Izzo's algorithm.

    This is a more robust algorithm that handles multi-revolution
    transfers and edge cases better than the universal variable method.

    Parameters
    ----------
    r1 : ndarray
        Initial position vector (km), shape (3,).
    r2 : ndarray
        Final position vector (km), shape (3,).
    tof : float
        Time of flight (seconds).
    mu : float, optional
        Gravitational parameter (km^3/s^2). Default is Earth.
    prograde : bool, optional
        If True, use prograde transfer. Default True.
    multi_rev : int, optional
        Number of complete revolutions. Default 0 (direct transfer).
    max_iter : int, optional
        Maximum iterations. Default 100.
    tol : float, optional
        Convergence tolerance. Default 1e-10.

    Returns
    -------
    solution : LambertSolution
        Solution containing velocities and orbital parameters.

    Notes
    -----
    For multi-revolution transfers, there may be two solutions
    (low and high energy). This returns the low energy solution.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Unit vectors
    r1_hat = r1 / r1_mag
    r2_hat = r2 / r2_mag

    # Cross product for angular momentum direction
    cross = np.cross(r1, r2)
    h_hat = (
        cross / np.linalg.norm(cross)
        if np.linalg.norm(cross) > 1e-10
        else np.array([0, 0, 1])
    )

    # Transfer angle
    cos_dnu = np.dot(r1_hat, r2_hat)
    cos_dnu = np.clip(cos_dnu, -1, 1)

    # Determine direction
    if prograde:
        if h_hat[2] >= 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2 * np.pi - np.arccos(cos_dnu)
    else:
        if h_hat[2] < 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2 * np.pi - np.arccos(cos_dnu)

    # Non-dimensional parameter
    c = np.sqrt(r1_mag * r1_mag + r2_mag * r2_mag - 2 * r1_mag * r2_mag * np.cos(dnu))
    s = (r1_mag + r2_mag + c) / 2

    # Characteristic velocity
    lambda_param = np.sqrt(r1_mag * r2_mag) * np.cos(dnu / 2) / s

    # Time of flight normalization
    T = np.sqrt(2 * mu / s**3) * tof

    # Use Householder iteration for x parameter
    if multi_rev == 0:
        # Single revolution - start with parabolic guess
        x = 0.0
    else:
        # Multi-revolution - start closer to elliptic
        x = 0.5

    # Householder iteration
    for _ in range(max_iter):
        # Battin's x-parameter equations
        y = np.sqrt(1 - lambda_param * lambda_param * (1 - x * x))

        # Time of flight equation
        if x < 1:
            psi = np.arccos(
                x * lambda_param + y * np.sqrt(1 - lambda_param * lambda_param)
            )
        else:
            psi = np.arccosh(
                x * lambda_param + y * np.sqrt(lambda_param * lambda_param - 1)
            )

        T_x = (
            psi + multi_rev * np.pi - (x - lambda_param * y) * np.sqrt(abs(1 - x * x))
        ) / np.sqrt(abs(1 - x * x)) ** 3

        # Check convergence
        if abs(T_x - T) < tol:
            break

        # Derivative for Newton update
        if abs(1 - x * x) > 1e-10:
            dT_dx = (3 * T_x * x - 2 - 2 * lambda_param**3 * y) / (1 - x * x)
        else:
            dT_dx = 1.0  # Avoid division by zero

        x = x - (T_x - T) / dT_dx
        x = np.clip(x, -0.999, 0.999)  # Keep in bounds

    # Compute velocities
    gamma = np.sqrt(mu * s / 2)

    rho = (r1_mag - r2_mag) / c
    sigma = np.sqrt(1 - rho * rho)

    # Radial and transverse velocity components
    v_r1 = gamma * ((lambda_param * y - x) - rho * (lambda_param * y + x)) / r1_mag
    v_r2 = -gamma * ((lambda_param * y - x) + rho * (lambda_param * y + x)) / r2_mag

    v_t1 = gamma * sigma * (y + lambda_param * x) / r1_mag
    v_t2 = gamma * sigma * (y + lambda_param * x) / r2_mag

    # Construct velocity vectors
    # Transverse unit vector
    t1_hat = np.cross(h_hat, r1_hat)
    t2_hat = np.cross(h_hat, r2_hat)

    v1 = v_r1 * r1_hat + v_t1 * t1_hat
    v2 = v_r2 * r2_hat + v_t2 * t2_hat

    # Compute orbital elements
    energy = np.linalg.norm(v1) ** 2 / 2 - mu / r1_mag
    if abs(energy) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = np.inf

    h_vec = np.cross(r1, v1)
    h_mag = np.linalg.norm(h_vec)
    if abs(energy) > 1e-10:
        ecc = np.sqrt(1 + 2 * energy * h_mag * h_mag / (mu * mu))
    else:
        ecc = 1.0

    return LambertSolution(v1=v1, v2=v2, a=a, e=ecc, tof=tof)


def minimum_energy_transfer(
    r1: NDArray[np.floating],
    r2: NDArray[np.floating],
    mu: float = GM_EARTH,
    prograde: bool = True,
) -> Tuple[float, LambertSolution]:
    """
    Compute minimum energy transfer between two positions.

    Parameters
    ----------
    r1 : ndarray
        Initial position vector (km).
    r2 : ndarray
        Final position vector (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).
    prograde : bool, optional
        If True, use prograde transfer.

    Returns
    -------
    tof_min : float
        Minimum energy time of flight (seconds).
    solution : LambertSolution
        Lambert solution at minimum energy.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Chord
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    c = np.sqrt(r1_mag**2 + r2_mag**2 - 2 * r1_mag * r2_mag * cos_dnu)

    # Semi-perimeter
    s = (r1_mag + r2_mag + c) / 2

    # Minimum energy semi-major axis
    a_min = s / 2

    # Minimum energy time of flight (parabolic)
    alpha = 2 * np.arcsin(np.sqrt(s / (2 * a_min)))
    beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a_min)))

    tof_min = np.sqrt(a_min**3 / mu) * (alpha - np.sin(alpha) - (beta - np.sin(beta)))

    # Solve Lambert at minimum energy TOF
    solution = lambert_universal(r1, r2, tof_min, mu, prograde)

    return tof_min, solution


def hohmann_transfer(
    r1: float,
    r2: float,
    mu: float = GM_EARTH,
) -> Tuple[float, float, float]:
    """
    Compute Hohmann transfer between two circular orbits.

    Parameters
    ----------
    r1 : float
        Initial orbit radius (km).
    r2 : float
        Final orbit radius (km).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    dv1 : float
        Delta-v at first burn (km/s).
    dv2 : float
        Delta-v at second burn (km/s).
    tof : float
        Transfer time of flight (seconds).

    Examples
    --------
    >>> dv1, dv2, tof = hohmann_transfer(6678, 42164)  # LEO to GEO
    >>> print(f"Total dv = {dv1 + dv2:.3f} km/s")
    """
    # Transfer orbit semi-major axis
    a_transfer = (r1 + r2) / 2

    # Circular velocities
    v1_circ = np.sqrt(mu / r1)
    v2_circ = np.sqrt(mu / r2)

    # Transfer orbit velocities at periapsis and apoapsis
    v1_transfer = np.sqrt(mu * (2 / r1 - 1 / a_transfer))
    v2_transfer = np.sqrt(mu * (2 / r2 - 1 / a_transfer))

    # Delta-v's
    dv1 = abs(v1_transfer - v1_circ)
    dv2 = abs(v2_circ - v2_transfer)

    # Transfer time (half orbital period)
    tof = np.pi * np.sqrt(a_transfer**3 / mu)

    return dv1, dv2, tof


def bi_elliptic_transfer(
    r1: float,
    r2: float,
    r_intermediate: float,
    mu: float = GM_EARTH,
) -> Tuple[float, float, float, float]:
    """
    Compute bi-elliptic transfer between two circular orbits.

    Parameters
    ----------
    r1 : float
        Initial orbit radius (km).
    r2 : float
        Final orbit radius (km).
    r_intermediate : float
        Intermediate apoapsis radius (km). Must be > max(r1, r2).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    dv1 : float
        Delta-v at first burn (km/s).
    dv2 : float
        Delta-v at intermediate apoapsis (km/s).
    dv3 : float
        Delta-v at final circularization (km/s).
    tof : float
        Total transfer time (seconds).
    """
    if r_intermediate < max(r1, r2):
        raise ValueError("Intermediate radius must be greater than both orbit radii")

    # First transfer ellipse
    a1 = (r1 + r_intermediate) / 2
    v1_circ = np.sqrt(mu / r1)
    v1_transfer = np.sqrt(mu * (2 / r1 - 1 / a1))
    v_int_1 = np.sqrt(mu * (2 / r_intermediate - 1 / a1))

    # Second transfer ellipse
    a2 = (r_intermediate + r2) / 2
    v_int_2 = np.sqrt(mu * (2 / r_intermediate - 1 / a2))
    v2_transfer = np.sqrt(mu * (2 / r2 - 1 / a2))
    v2_circ = np.sqrt(mu / r2)

    # Delta-v's
    dv1 = abs(v1_transfer - v1_circ)
    dv2 = abs(v_int_2 - v_int_1)
    dv3 = abs(v2_circ - v2_transfer)

    # Transfer times
    tof1 = np.pi * np.sqrt(a1**3 / mu)
    tof2 = np.pi * np.sqrt(a2**3 / mu)
    tof = tof1 + tof2

    return dv1, dv2, dv3, tof


__all__ = [
    "LambertSolution",
    "lambert_universal",
    "lambert_izzo",
    "minimum_energy_transfer",
    "hohmann_transfer",
    "bi_elliptic_transfer",
]
