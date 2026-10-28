"""Shared Schmidt semi-normalized Legendre synthesis pieces.

Geomagnetic Gauss coefficients (WMM, IGRF, EMM, WMMHR) are given in the
Schmidt semi-normalized convention, while
:func:`pytcl.gravity.spherical_harmonics.associated_legendre` implements
the geodesy fully normalized one,
``sqrt((2 - delta_0m)(2n+1)(n-m)!/(n+m)!)``. Schmidt keeps the
``sqrt(2 - delta_0m)`` factor only.

This module is the single home of the Schmidt functions and their
colatitude derivatives for wmm.py, emm.py and coordinates.py. Since
v2.12.0 it no longer rescales the gravity table, whose cache rounds
``cos(theta)`` to 12 decimals (an error that grows like
``1 / sin(theta)**2`` toward the poles).
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from pytcl.core.array_utils import make_readonly

# Geomagnetic reference sphere radius in km (the WMM/IGRF convention).
GEOMAGNETIC_REFERENCE_RADIUS_KM = 6371.2


@lru_cache(maxsize=8)
def _recursion_coefficients(
    n_max: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Sectoral factors and column-recursion coefficients for degree n_max.

    ``sectoral[k]`` is ``P[k, k] / sin(theta)**k``; row ``k`` of the
    column recursion is ``P[k, m] = a[k, m] cos(theta) P[k-1, m]
    - b[k, m] P[k-2, m]`` for ``m < k``.
    """
    k = np.arange(n_max + 1)
    ratios = np.ones(n_max + 1)
    ratios[2:] = np.sqrt((2 * k[2:] - 1) / (2 * k[2:]))
    sectoral = np.cumprod(ratios)
    n = k[:, np.newaxis].astype(np.float64)
    m = k[np.newaxis, :].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(n**2 - m**2)
        a = np.where(m < n, (2 * n - 1) / denom, 0.0)
        b = np.where(m < n, np.sqrt(np.maximum((n - 1) ** 2 - m**2, 0.0)) / denom, 0.0)
    make_readonly(sectoral, a, b)
    return sectoral, a, b


@lru_cache(maxsize=16)
def _schmidt_tables(
    n_max: int, cos_theta: float, sin_theta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read-only (P, dP/dtheta), memoised on the exact (cos, sin) pair."""
    sectoral, a, b = _recursion_coefficients(n_max)
    P = np.zeros((n_max + 2, n_max + 2))
    k = np.arange(n_max + 1)
    with np.errstate(under="ignore"):
        P[k, k] = sectoral * sin_theta**k
    for row in range(1, n_max + 1):
        P[row, :row] = (
            a[row, :row] * cos_theta * P[row - 1, :row]
            - b[row, :row] * P[row - 2, :row]
        )
    P = P[: n_max + 1, : n_max + 1]

    n = k[:, np.newaxis]
    m = k[np.newaxis, :]
    lower = np.zeros_like(P)
    lower[:, 1:] = P[:, :-1]
    upper = np.zeros_like(P)
    upper[:, :-1] = P[:, 1:]
    c = np.where(m == 1, np.sqrt(2.0), 1.0)
    d = np.where(m == 0, np.sqrt(2.0), 1.0)
    alpha = np.sqrt(np.maximum((n + m) * (n - m + 1), 0))
    beta = np.sqrt(np.maximum((n - m) * (n + m + 1), 0))
    dP = 0.5 * (c * alpha * lower - d * beta * upper)
    make_readonly(P, dP)
    return P, dP


def _schmidt_legendre(
    n_max: int, cos_theta: float, sin_theta: float
) -> NDArray[np.float64]:
    """Schmidt semi-normalized associated Legendre functions.

    The standard Schmidt column recursion, seeded with ``sin_theta``
    itself rather than ``sqrt(1 - cos_theta**2)``, which loses the
    sine's relative precision near the poles. The result is read-only
    and shared between calls with the same arguments.

    Parameters
    ----------
    n_max : int
        Maximum degree and order.
    cos_theta, sin_theta : float
        Cosine and sine of colatitude.

    Returns
    -------
    P : ndarray
        Array of shape (n_max+1, n_max+1); ``P[n, m]`` is the Schmidt
        semi-normalized function of degree n, order m.
    """
    return _schmidt_tables(n_max, float(cos_theta), float(sin_theta))[0]


def _schmidt_legendre_with_derivative(
    n_max: int, cos_theta: float, sin_theta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Schmidt functions and their colatitude derivatives.

    ``dP/dtheta`` comes from the same-degree, neighbouring-order identity

    ``dP[n,m]/dtheta = (c_m sqrt((n+m)(n-m+1)) P[n,m-1]
    - d_m sqrt((n-m)(n+m+1)) P[n,m+1]) / 2``

    with ``c_1 = sqrt(2)``, ``d_0 = sqrt(2)`` and 1 otherwise. It never
    divides by ``sin(theta)``, so it holds at the poles, where the
    ``m = 1`` derivatives are nonzero. Both arrays are read-only and
    shared between calls with the same arguments.

    Returns
    -------
    P, dP : ndarray
        Shape (n_max+1, n_max+1) each: the Schmidt semi-normalized
        functions and their derivatives with respect to colatitude.
    """
    return _schmidt_tables(n_max, float(cos_theta), float(sin_theta))
