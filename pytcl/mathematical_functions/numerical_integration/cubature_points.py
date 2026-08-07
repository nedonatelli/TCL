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

import itertools
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma, roots_genlaguerre, roots_jacobi


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


def _seventh_order_unit_sphere_points(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Degree-7 rule for the uniform measure on the unit sphere S^(n-1).

    Stroud's surface Formula I (E_n^{r^2} 7-1's spherical building block),
    the counterpart of the MATLAB TCL's ``seventhOrderSpherSurfCubPoints``
    (algorithm 0), n >= 3. 2^n + 2n^2 points: axis points e_i (weight A1),
    pairwise points (e_i + e_j)/sqrt(2) (weight A2), and the all-nonzero
    point (1,...,1)/sqrt(n) (weight A3), each fully signed. Weights are
    normalized to sum to 1.
    """
    axis = np.eye(n)
    points = [axis, -axis]

    s = 1.0 / np.sqrt(2.0)
    pair_pts = []
    for i in range(n):
        for j in range(i + 1, n):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    p = np.zeros(n)
                    p[i] = si * s
                    p[j] = sj * s
                    pair_pts.append(p)
    points.append(np.array(pair_pts))

    t = 1.0 / np.sqrt(n)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    points.append(t * signs)

    i1 = 2.0 * np.pi ** (n / 2.0) / gamma(n / 2.0)  # surface area of S^(n-1)
    denom = n * (n + 2.0) * (n + 4.0)
    a1 = (8.0 - n) / denom * i1
    a2 = 4.0 / denom * i1
    a3 = 2.0 ** (-n) * n**3 / denom * i1

    weights = np.concatenate(
        [
            np.full(2 * n, a1),
            np.full(len(pair_pts), a2),
            np.full(2**n, a3),
        ]
    )
    return np.vstack(points), weights / i1


def seventh_order_cubature_points(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Degree-7 cubature points for the standard normal N(0, I).

    The 2*(2^n + 2n^2) point fully-symmetric rule E_n^{r^2} 7-1 of Stroud
    [1]_ (McNamee-Stenger [2]_ construction), the counterpart of the
    MATLAB TCL's ``seventhOrderCubPoints`` (algorithm 0, the default for
    n > 2). Two concentric copies of the degree-7 spherical-surface rule
    ``seventhOrderSpherSurfCubPoints`` (algorithm 0, formula I of Stroud's
    1968 paper cited there) are scaled to radii r1 and r2 -- the roots of
    the radial polynomial in Stroud's construction -- and blended with
    weights that sum to 1. Exactly integrates every polynomial of total
    degree <= 7 against N(0, I).

    Parameters
    ----------
    n : int
        Dimension, n >= 3.

    Returns
    -------
    points : ndarray
        Shape (2*(2**n + 2*n*n), n).
    weights : ndarray
        Shape (2*(2**n + 2*n*n),), summing to 1. The axis shell's surface
        weight (8 - n)/(n(n+2)(n+4)) is negative for n > 8; this is
        inherent to the rule, not an error. Covariances assembled from
        these points must not use a sqrt-of-weights factorization.

    Examples
    --------
    >>> pts, w = seventh_order_cubature_points(3)
    >>> pts.shape
    (52, 3)
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 6)), 6)  # E[x^6] = 15
    15.0

    References
    ----------
    .. [1] A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
       Prentice-Hall, 1971, Formula E_n^{r^2} 7-1, p. 318.
    .. [2] J. McNamee and F. Stenger, "Construction of fully symmetric
       numerical integration formulas," Numerische Mathematik 10, 1967.
    """
    if n < 3:
        raise ValueError(f"dimension must be >= 3, got {n}")

    u, wu = _seventh_order_unit_sphere_points(n)

    root = np.sqrt(2.0 * (n + 2.0))
    r1 = np.sqrt((n + 2.0 - root) / 2.0)
    r2 = np.sqrt((n + 2.0 + root) / 2.0)
    a1 = (n + 2.0 + root) / (2.0 * (n + 2.0))
    a2 = (n + 2.0 - root) / (2.0 * (n + 2.0))

    points = np.sqrt(2.0) * np.vstack([r1 * u, r2 * u])
    weights = np.concatenate([wu * a1, wu * a2])

    return points, weights


def _sphere_surface_points(
    n: int, degree: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Degree-``degree`` rule for the uniform measure on S^(n-1).

    Dimension-recursive spherical-coordinate product construction:
    x = (t, sqrt(1 - t^2) * y) with t from Gauss-Jacobi quadrature with
    weight (1 - t^2)^((n-3)/2) and y a degree-``degree`` rule on S^(n-2).
    Weights are normalized to sum to 1.
    """
    if n == 1:
        return np.array([[1.0], [-1.0]]), np.array([0.5, 0.5])
    if n == 2:
        m = 2 * ((degree + 1) // 2 + 1)  # uniform points, exact for trig deg < m
        theta = 2.0 * np.pi * np.arange(m) / m
        return np.column_stack([np.cos(theta), np.sin(theta)]), np.full(m, 1.0 / m)

    m = (degree + 1) // 2  # Gauss-Jacobi exact through poly degree 2m-1
    t, wt = roots_jacobi(m, (n - 3.0) / 2.0, (n - 3.0) / 2.0)
    sub_pts, sub_w = _sphere_surface_points(n - 1, degree)

    pts = []
    wts = []
    for tk, wk in zip(t, wt):
        s = np.sqrt(1.0 - tk * tk)
        block = np.column_stack([np.full(len(sub_pts), tk), s * sub_pts])
        pts.append(block)
        wts.append(wk * sub_w)
    points = np.vstack(pts)
    weights = np.concatenate(wts)
    return points, weights / weights.sum()


def spherical_radial_points(
    n: int, degree: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Arbitrary-odd-degree spherical-radial cubature points for N(0, I).

    Product of a generalized Gauss-Laguerre radial rule (exact for all
    required even powers of r) with a dimension-recursive surface rule on
    the unit sphere. Generalizes the 3rd-degree spherical-radial rule of
    the CKF to any odd degree.

    The point count grows roughly as ``(degree/2)^(n-1)`` from the surface
    rule; for the common degrees 5 and 7 prefer
    :func:`fifth_order_cubature_points` and
    :func:`seventh_order_cubature_points`, which grow polynomially in n.

    Parameters
    ----------
    n : int
        Dimension, n >= 1.
    degree : int
        Odd polynomial degree >= 3 the rule integrates exactly.

    Returns
    -------
    points : ndarray, shape (num_points, n)
    weights : ndarray, shape (num_points,), summing to 1.

    Examples
    --------
    >>> pts, w = spherical_radial_points(2, 5)
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 4)), 10)  # E[x^4] = 3
    3.0
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if degree < 3 or degree % 2 == 0:
        raise ValueError(f"degree must be an odd integer >= 3, got {degree}")

    # Radial part: substitute t = r^2/2 in the integral of g(r) r^(n-1)
    # exp(-r^2/2); Gauss-Laguerre with alpha = n/2 - 1 handles t^j exactly.
    # Even powers r^(2j) with 2j <= degree - 1 must be exact => j <=
    # (degree-1)/2 => m_r points with 2*m_r - 1 >= (degree-1)/2.
    # Checked: degree 3 -> 1 node (r = sqrt(n), the CKF radius); 5,7 -> 2;
    # 9 -> 3.
    m_r = (degree + 3) // 4
    t, wt = roots_genlaguerre(m_r, n / 2.0 - 1.0)
    radii = np.sqrt(2.0 * t)
    w_rad = wt / gamma(n / 2.0)

    surf_pts, surf_w = _sphere_surface_points(n, degree)

    points = np.vstack([r * surf_pts for r in radii])
    weights = np.concatenate([wr * surf_w for wr in w_rad])
    return points, weights / weights.sum()
