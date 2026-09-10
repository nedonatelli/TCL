"""Shared Schmidt semi-normalized Legendre synthesis pieces.

Geomagnetic Gauss coefficients (WMM, IGRF, EMM, WMMHR) are given in the
Schmidt semi-normalized convention, while
:func:`pytcl.gravity.spherical_harmonics.associated_legendre` implements
the geodesy fully normalized one,
``sqrt((2 - delta_0m)(2n+1)(n-m)!/(n+m)!)``. Schmidt keeps the
``sqrt(2 - delta_0m)`` factor only, so ``Schmidt = full / sqrt(2n+1)``.

This module is the single home of that conversion; wmm.py, emm.py and
coordinates.py each carried an identical copy of it before v2.11.0.
"""

import numpy as np
from numpy.typing import NDArray

from pytcl.gravity.spherical_harmonics import (
    associated_legendre,
    associated_legendre_derivative,
)

# Geomagnetic reference sphere radius in km (the WMM/IGRF convention).
GEOMAGNETIC_REFERENCE_RADIUS_KM = 6371.2


def _schmidt_legendre(n_max: int, cos_theta: float) -> NDArray[np.float64]:
    """Schmidt semi-normalized associated Legendre functions.

    Parameters
    ----------
    n_max : int
        Maximum degree and order.
    cos_theta : float
        Cosine of colatitude.

    Returns
    -------
    P : ndarray
        Array of shape (n_max+1, n_max+1); ``P[n, m]`` is the Schmidt
        semi-normalized function of degree n, order m at ``cos_theta``.
    """
    P_full = associated_legendre(n_max, n_max, cos_theta, normalized=True)
    scale = 1.0 / np.sqrt(2 * np.arange(n_max + 1) + 1)
    return P_full * scale[:, np.newaxis]


def _schmidt_legendre_with_derivative(
    n_max: int, cos_theta: float, sin_theta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Schmidt functions and their colatitude derivatives.

    ``dP/dtheta`` is derived from the fully normalized ``dP/dx`` via
    ``dP/dtheta = -sin(theta) * dP/dx`` with ``x = cos(theta)``.

    Returns
    -------
    P, dP : ndarray
        Shape (n_max+1, n_max+1) each: the Schmidt semi-normalized
        functions and their derivatives with respect to colatitude.
    """
    P_full = associated_legendre(n_max, n_max, cos_theta, normalized=True)
    dP_full = associated_legendre_derivative(
        n_max, n_max, cos_theta, P_full, normalized=True
    )
    scale = 1.0 / np.sqrt(2 * np.arange(n_max + 1) + 1)
    P = P_full * scale[:, np.newaxis]
    dP = -sin_theta * dP_full * scale[:, np.newaxis]
    return P, dP
