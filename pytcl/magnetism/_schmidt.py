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

import numpy as np
from numpy.typing import NDArray

# Geomagnetic reference sphere radius in km (the WMM/IGRF convention).
GEOMAGNETIC_REFERENCE_RADIUS_KM = 6371.2


def _schmidt_legendre(
    n_max: int, cos_theta: float, sin_theta: float
) -> NDArray[np.float64]:
    """Schmidt semi-normalized associated Legendre functions.

    The standard Schmidt column recursion, seeded with ``sin_theta``
    itself rather than ``sqrt(1 - cos_theta**2)``, which loses the
    sine's relative precision near the poles.

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
    P = np.zeros((n_max + 2, n_max + 2))
    P[0, 0] = 1.0
    if n_max >= 1:
        P[1, 1] = sin_theta
    for k in range(2, n_max + 1):
        P[k, k] = np.sqrt((2 * k - 1) / (2 * k)) * sin_theta * P[k - 1, k - 1]
    for k in range(1, n_max + 1):
        m = np.arange(k)
        P[k, :k] = (
            (2 * k - 1) * cos_theta * P[k - 1, :k]
            - np.sqrt((k - 1) ** 2 - m**2) * P[k - 2, :k]
        ) / np.sqrt(k**2 - m**2)
    return P[: n_max + 1, : n_max + 1]


def _schmidt_legendre_with_derivative(
    n_max: int, cos_theta: float, sin_theta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Schmidt functions and their colatitude derivatives.

    ``dP/dtheta`` comes from the same-degree, neighbouring-order identity

    ``dP[n,m]/dtheta = (c_m sqrt((n+m)(n-m+1)) P[n,m-1]
    - d_m sqrt((n-m)(n+m+1)) P[n,m+1]) / 2``

    with ``c_1 = sqrt(2)``, ``d_0 = sqrt(2)`` and 1 otherwise. It never
    divides by ``sin(theta)``, so it holds at the poles, where the
    ``m = 1`` derivatives are nonzero.

    Returns
    -------
    P, dP : ndarray
        Shape (n_max+1, n_max+1) each: the Schmidt semi-normalized
        functions and their derivatives with respect to colatitude.
    """
    P = _schmidt_legendre(n_max, cos_theta, sin_theta)
    n = np.arange(n_max + 1)[:, np.newaxis]
    m = np.arange(n_max + 1)[np.newaxis, :]
    lower = np.zeros_like(P)
    lower[:, 1:] = P[:, :-1]
    upper = np.zeros_like(P)
    upper[:, :-1] = P[:, 1:]
    c = np.where(m == 1, np.sqrt(2.0), 1.0)
    d = np.where(m == 0, np.sqrt(2.0), 1.0)
    a = np.sqrt(np.maximum((n + m) * (n - m + 1), 0))
    b = np.sqrt(np.maximum((n - m) * (n + m + 1), 0))
    dP = 0.5 * (c * a * lower - d * b * upper)
    return P, dP
