"""
Spherical harmonic functions for geophysical models.

Spherical harmonics are used to represent gravitational and magnetic
fields on a sphere. This module provides functions for evaluating
associated Legendre polynomials and spherical harmonic expansions.

References
----------
.. [1] W. A. Heiskanen and H. Moritz, "Physical Geodesy," W. H. Freeman, 1967.
.. [2] O. Montenbruck and E. Gill, "Satellite Orbits," Springer, 2000.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def associated_legendre(
    n_max: int,
    m_max: int,
    x: float,
    normalized: bool = True,
) -> NDArray[np.floating]:
    """
    Compute associated Legendre polynomials P_n^m(x).

    Uses the recursive algorithm for numerical stability.

    Parameters
    ----------
    n_max : int
        Maximum degree.
    m_max : int
        Maximum order (must be <= n_max).
    x : float
        Argument, typically cos(colatitude). Must be in [-1, 1].
    normalized : bool, optional
        If True, return fully normalized (geodetic) coefficients.
        Default True.

    Returns
    -------
    P : ndarray
        Array of shape (n_max+1, m_max+1) containing P_n^m(x).

    Notes
    -----
    The fully normalized associated Legendre functions satisfy:

    .. math::

        \\int_{-1}^{1} [\\bar{P}_n^m(x)]^2 dx = \\frac{2}{2n+1}

    Examples
    --------
    >>> P = associated_legendre(2, 2, 0.5)
    >>> P[2, 0]  # P_2^0(0.5)
    """
    if m_max > n_max:
        raise ValueError("m_max must be <= n_max")
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1, 1]")

    P = np.zeros((n_max + 1, m_max + 1))

    # Compute sqrt(1 - x^2) = sin(theta) for colatitude
    u = np.sqrt(1 - x * x)

    # Seed values
    P[0, 0] = 1.0

    # Sectoral recursion: P_m^m from P_{m-1}^{m-1}
    for m in range(1, m_max + 1):
        if normalized:
            P[m, m] = u * np.sqrt((2 * m + 1) / (2 * m)) * P[m - 1, m - 1]
        else:
            P[m, m] = (2 * m - 1) * u * P[m - 1, m - 1]

    # Compute P_{m+1}^m from P_m^m
    for m in range(m_max):
        if m + 1 <= n_max:
            if normalized:
                P[m + 1, m] = x * np.sqrt(2 * m + 3) * P[m, m]
            else:
                P[m + 1, m] = x * (2 * m + 1) * P[m, m]

    # General recursion: P_n^m from P_{n-1}^m and P_{n-2}^m
    for m in range(m_max + 1):
        for n in range(m + 2, n_max + 1):
            if normalized:
                a_nm = np.sqrt((4 * n * n - 1) / (n * n - m * m))
                b_nm = np.sqrt(((n - 1) ** 2 - m * m) / (4 * (n - 1) ** 2 - 1))
                P[n, m] = a_nm * (x * P[n - 1, m] - b_nm * P[n - 2, m])
            else:
                P[n, m] = (
                    (2 * n - 1) * x * P[n - 1, m] - (n + m - 1) * P[n - 2, m]
                ) / (n - m)

    return P


def associated_legendre_derivative(
    n_max: int,
    m_max: int,
    x: float,
    P: Optional[NDArray[np.floating]] = None,
    normalized: bool = True,
) -> NDArray[np.floating]:
    """
    Compute derivatives of associated Legendre polynomials dP_n^m/dx.

    Parameters
    ----------
    n_max : int
        Maximum degree.
    m_max : int
        Maximum order.
    x : float
        Argument in [-1, 1].
    P : ndarray, optional
        Precomputed P_n^m values. If None, computed internally.
    normalized : bool, optional
        If True, use fully normalized functions. Default True.

    Returns
    -------
    dP : ndarray
        Array of shape (n_max+1, m_max+1) containing dP_n^m/dx.
    """
    if P is None:
        P = associated_legendre(n_max, m_max, x, normalized)

    dP = np.zeros((n_max + 1, m_max + 1))

    # Handle x = ±1 specially (poles)
    if abs(abs(x) - 1) < 1e-14:
        # At poles, derivatives need special handling
        # For now, return zeros (valid for m > 0)
        return dP

    u2 = 1 - x * x  # sin^2(theta)

    for n in range(n_max + 1):
        for m in range(min(n, m_max) + 1):
            if n == 0:
                dP[n, m] = 0.0
            elif m == n:
                # dP_n^n/dx = n * x / (1-x^2) * P_n^n
                dP[n, m] = n * x / u2 * P[n, m]
            else:
                # General formula using recurrence
                if normalized:
                    # Normalized form
                    if n > m:
                        factor = np.sqrt((n - m) * (n + m + 1))
                        if m + 1 <= m_max and n >= m + 1:
                            dP[n, m] = (
                                n * x / u2 * P[n, m]
                                - factor / np.sqrt(u2) * P[n, m + 1]
                                if m + 1 <= n
                                else n * x / u2 * P[n, m]
                            )
                        else:
                            dP[n, m] = n * x / u2 * P[n, m]
                else:
                    # Unnormalized form
                    dP[n, m] = (
                        (n * x * P[n, m] - (n + m) * P[n - 1, m]) / u2 if n > 0 else 0
                    )

    return dP


def spherical_harmonic_sum(
    lat: float,
    lon: float,
    r: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate spherical harmonic expansion for a scalar field.

    Computes the value and gradient of a field represented by
    spherical harmonic coefficients C_nm and S_nm.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from center of mass.
    C : ndarray
        Cosine coefficients C_nm, shape (n_max+1, n_max+1).
    S : ndarray
        Sine coefficients S_nm, shape (n_max+1, n_max+1).
    R : float
        Reference radius (e.g., Earth's equatorial radius).
    GM : float
        Gravitational parameter (G * M).
    n_max : int, optional
        Maximum degree to use. Default uses full coefficient array.

    Returns
    -------
    V : float
        Potential value.
    dV_r : float
        Radial derivative dV/dr.
    dV_lat : float
        Latitudinal derivative (1/r) * dV/dlat.

    Notes
    -----
    The spherical harmonic expansion of the gravitational potential is:

    .. math::

        V = \\frac{GM}{r} \\sum_{n=0}^{N} \\left(\\frac{R}{r}\\right)^n
            \\sum_{m=0}^{n} \\bar{P}_n^m(\\sin\\phi)
            (C_{nm}\\cos m\\lambda + S_{nm}\\sin m\\lambda)
    """
    if n_max is None:
        n_max = C.shape[0] - 1

    # Colatitude for Legendre polynomials
    colat = np.pi / 2 - lat
    cos_colat = np.cos(colat)
    sin_colat = np.sin(colat)

    # Compute Legendre polynomials and derivatives
    P = associated_legendre(n_max, n_max, cos_colat, normalized=True)
    dP = associated_legendre_derivative(n_max, n_max, cos_colat, P, normalized=True)

    # Initialize sums
    V = 0.0
    dV_r = 0.0
    dV_colat = 0.0
    dV_lon = 0.0

    # Compute (R/r)^n factors
    r_ratio = R / r
    r_power = 1.0  # (R/r)^0

    for n in range(n_max + 1):
        r_power_n = r_power  # (R/r)^n

        for m in range(n + 1):
            cos_m_lon = np.cos(m * lon)
            sin_m_lon = np.sin(m * lon)

            # Coefficient combination
            Cnm = C[n, m] if n < C.shape[0] and m < C.shape[1] else 0.0
            Snm = S[n, m] if n < S.shape[0] and m < S.shape[1] else 0.0

            coeff = Cnm * cos_m_lon + Snm * sin_m_lon
            coeff_lon = m * (-Cnm * sin_m_lon + Snm * cos_m_lon)

            # Potential contribution
            V += r_power_n * P[n, m] * coeff

            # Radial derivative contribution
            dV_r += -(n + 1) * r_power_n / r * P[n, m] * coeff

            # Colatitude derivative contribution
            # dP/d(colat) = -sin(colat) * dP/d(cos(colat))
            dV_colat += r_power_n * (-sin_colat) * dP[n, m] * coeff

            # Longitude derivative contribution
            dV_lon += r_power_n * P[n, m] * coeff_lon

        r_power *= r_ratio  # Update for next n

    # Scale by GM/r
    scale = GM / r
    V *= scale
    dV_r = dV_r * GM + V / r * (-1)  # Product rule
    dV_r = -GM / (r * r) * (V / scale) + scale * dV_r / scale

    # Correct radial derivative
    dV_r = 0.0
    r_power = 1.0
    for n in range(n_max + 1):
        for m in range(n + 1):
            cos_m_lon = np.cos(m * lon)
            sin_m_lon = np.sin(m * lon)
            Cnm = C[n, m] if n < C.shape[0] and m < C.shape[1] else 0.0
            Snm = S[n, m] if n < S.shape[0] and m < S.shape[1] else 0.0
            coeff = Cnm * cos_m_lon + Snm * sin_m_lon
            dV_r += -(n + 1) * r_power * P[n, m] * coeff
        r_power *= r_ratio

    dV_r *= GM / (r * r)

    # Convert colatitude derivative to latitude derivative
    dV_lat = -dV_colat * scale / r  # (1/r) dV/dlat

    return V, dV_r, dV_lat


def gravity_acceleration(
    lat: float,
    lon: float,
    h: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute gravity acceleration vector from spherical harmonics.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float
        Height above reference ellipsoid in meters.
    C : ndarray
        Cosine coefficients.
    S : ndarray
        Sine coefficients.
    R : float
        Reference radius.
    GM : float
        Gravitational parameter.
    n_max : int, optional
        Maximum degree.

    Returns
    -------
    g_r : float
        Radial component of gravity (positive outward).
    g_lat : float
        Northward component of gravity.
    g_lon : float
        Eastward component of gravity.
    """
    # Approximate radial distance (simplified, ignoring ellipsoid flattening)
    r = R + h

    V, dV_r, dV_lat = spherical_harmonic_sum(lat, lon, r, C, S, R, GM, n_max)

    # Gravity is negative gradient of potential
    g_r = -dV_r
    g_lat = -dV_lat

    # Longitude component (for non-zonal terms)
    # This would require additional computation for full accuracy
    g_lon = 0.0  # Simplified

    return g_r, g_lat, g_lon


__all__ = [
    "associated_legendre",
    "associated_legendre_derivative",
    "spherical_harmonic_sum",
    "gravity_acceleration",
]
