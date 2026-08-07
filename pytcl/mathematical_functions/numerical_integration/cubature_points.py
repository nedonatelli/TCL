"""
Cubature point sets for Gaussian-weighted integration.

Every generator in this module targets the standard multivariate normal
weight N(0, I): points have shape ``(num_points, n)`` and weights sum to 1,
so ``E[f(x)] ~= sum_i w_i f(x_i)`` directly. This matches
``ckf_spherical_cubature_points`` and differs from the 1-D
:func:`~pytcl.mathematical_functions.numerical_integration.gauss_hermite`,
which uses the physicists' ``exp(-x**2)`` weight (map with
``x -> sqrt(2) x`` and divide weights by ``sqrt(pi)`` per dimension).

Ported from the Tracker Component Library's Cubature_Points collection.

References
----------
.. [1] A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
   Prentice-Hall, 1971.
.. [2] J. McNamee and F. Stenger, "Construction of fully symmetric
   numerical integration formulas," Numerische Mathematik 10, 1967.
.. [3] D. F. Crouse, "The Tracker Component Library," IEEE AESS Magazine,
   2017.
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def transform_cubature_points(
    points: ArrayLike,
    weights: ArrayLike,
    mean: ArrayLike,
    sqrt_cov: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Affinely map unit cubature points to a given mean and covariance.

    Parameters
    ----------
    points : array_like
        Unit points for N(0, I), shape (num_points, n).
    weights : array_like
        Weights, shape (num_points,).
    mean : array_like
        Target mean, shape (n,).
    sqrt_cov : array_like
        Square root of the target covariance (lower-triangular Cholesky
        factor S with S @ S.T = P), shape (n, n).

    Returns
    -------
    points : ndarray
        Transformed points ``mean + points @ sqrt_cov.T``.
    weights : ndarray
        Unchanged weights (copied).

    Examples
    --------
    >>> unit = np.array([[1.0], [-1.0]])
    >>> w = np.array([0.5, 0.5])
    >>> pts, wts = transform_cubature_points(unit, w, [10.0], [[2.0]])
    >>> pts.ravel().tolist()
    [12.0, 8.0]
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).ravel()
    sqrt_cov = np.asarray(sqrt_cov, dtype=np.float64)

    if points.ndim != 2:
        raise ValueError(f"points must be 2-D, got shape {points.shape}")
    num_points, n = points.shape
    if weights.shape != (num_points,):
        raise ValueError(
            f"weights shape {weights.shape} does not match {num_points} points"
        )
    if mean.shape != (n,) or sqrt_cov.shape != (n, n):
        raise ValueError(
            f"mean/sqrt_cov dimensions {mean.shape}/{sqrt_cov.shape} do not "
            f"match points dimension {n}"
        )

    return mean + points @ sqrt_cov.T, weights.copy()


def fifth_order_cubature_points(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Degree-5 cubature points for the standard normal N(0, I).

    The 2n^2 + 1 point fully-symmetric rule E_n^{r^2} 5-3 of Stroud [1]_,
    the counterpart of the MATLAB TCL's ``fifthOrderCubPoints``. Exactly
    integrates every polynomial of total degree <= 5 against N(0, I).

    Parameters
    ----------
    n : int
        Dimension, n >= 1.

    Returns
    -------
    points : ndarray
        Shape (2*n*n + 1, n).
    weights : ndarray
        Shape (2*n*n + 1,), summing to 1. For n > 4 the axis-point weight
        (4 - n)/(2 (n+2)^2) is negative; this is inherent to the rule, not
        an error. Covariances assembled from these points must not use a
        sqrt-of-weights factorization.

    Examples
    --------
    >>> pts, w = fifth_order_cubature_points(3)
    >>> pts.shape
    (19, 3)
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 4)), 12)  # E[x^4] = 3
    3.0
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")

    lam = np.sqrt(n + 2.0)
    mu = np.sqrt((n + 2.0) / 2.0)
    w_center = 2.0 / (n + 2.0)
    w_axis = (4.0 - n) / (2.0 * (n + 2.0) ** 2)
    w_pair = 1.0 / (n + 2.0) ** 2

    points = [np.zeros((1, n))]
    weights = [np.array([w_center])]

    axis = lam * np.eye(n)
    points.append(np.vstack([axis, -axis]))
    weights.append(np.full(2 * n, w_axis))

    pair_pts = []
    for i in range(n):
        for j in range(i + 1, n):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    p = np.zeros(n)
                    p[i] = si * mu
                    p[j] = sj * mu
                    pair_pts.append(p)
    if pair_pts:
        points.append(np.array(pair_pts))
        weights.append(np.full(len(pair_pts), w_pair))

    return np.vstack(points), np.concatenate(weights)
