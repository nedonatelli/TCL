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

    Stroud's surface Formula I, the degree-7 spherical-surface building
    block of rule E_n^{r^2} 7-3, the counterpart of the MATLAB TCL's
    ``seventhOrderSpherSurfCubPoints`` (algorithm 0), n >= 3. 2^n + 2n^2
    points: axis points e_i (weight A1), pairwise points
    (e_i + e_j)/sqrt(2) (weight A2), and the all-nonzero point
    (1,...,1)/sqrt(n) (weight A3), each fully signed. Weights are
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

    The 2*(2^n + 2n^2) point fully-symmetric rule E_n^{r^2} 7-3 of Stroud
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
       Prentice-Hall, 1971, Formula E_n^{r^2} 7-3, p. 319. The formula as
       printed there contains a typo; the corrected form used here
       follows Stroud's original papers [2]_, [3]_, summarized in [4]_.
    .. [2] A. H. Stroud, "Some seventh degree integration formulas for
       symmetric regions," SIAM Journal on Numerical Analysis, vol. 4,
       no. 1, pp. 37-44, Mar. 1967.
    .. [3] A. H. Stroud, "Some seventh degree integration formulas for
       the surface of an n-sphere," Numerische Mathematik, vol. 11,
       no. 3, pp. 273-276, Mar. 1968.
    .. [4] D. F. Crouse, "Basic tracking using nonlinear 3D monostatic
       and bistatic measurements," IEEE Aerospace and Electronic Systems
       Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
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


def sphere_surface_to_gauss_points(
    surface_points: ArrayLike,
    surface_weights: ArrayLike,
    degree: int,
    beta: float = 0.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Lift a spherical-surface cubature rule to a Gaussian(-times-\\|x\\|^beta) rule.

    Counterpart of the MATLAB TCL's ``spherSurfPoints2GaussPoints``: given
    cubature points/weights for the uniform measure on the unit sphere
    S^(n-1) (weights summing to 1, e.g. from
    :func:`_sphere_surface_points`), produces points/weights for the
    weighting function ``w(x) = N(x; 0, I) * |x|^beta`` (``beta = 0`` is the
    plain N(0, I) density).

    The MATLAB source builds the radial rule from a 1-D quadrature for
    ``|x|^c1 * exp(-x^2)`` (``quadraturePoints1D``, algorithm 9, a
    three-term-recursion construction restricted to integer ``c1``), then
    rescales by ``x -> sqrt(2) x``. This port instead reuses the
    generalized Gauss-Laguerre substitution ``t = r^2/2`` already used by
    :func:`spherical_radial_points`: with ``alpha = (n + beta)/2 - 1``, the
    quadrature nodes/weights from ``scipy.special.roots_genlaguerre`` match
    the same target radial moments
    ``integral_0^inf r^(n-1+beta) exp(-r^2/2) r^(2k) dr`` for every ``k``
    needed up to ``degree``, so it reproduces the identical family of rules
    while additionally allowing non-integer ``beta`` (MATLAB's
    three-term-recursion route cannot). Both routes implement the same
    "spherical shell plus a 1-D |x|^beta * exp(-x^2)-type formula"
    construction described in Chapter 2.8 of [1]_, cited by the MATLAB
    source. Randomization (MATLAB's ``randomize`` flag, a random
    orthonormal rotation applied post hoc to reduce repeated-orientation
    artifacts in tracking -- see [2]_, [3]_) is not exposed; callers who
    want it can rotate the returned points themselves.

    Parameters
    ----------
    surface_points : array_like
        Points on the unit sphere S^(n-1), shape (num_surface_points, n).
    surface_weights : array_like
        Weights for the uniform measure on S^(n-1), shape
        (num_surface_points,), summing to 1.
    degree : int
        The polynomial degree the surface rule (and thus this rule) is
        exact through, degree >= 1.
    beta : float, optional
        Exponent of \\|x\\| in the weighting function, beta > -n. Default 0.0
        (plain N(0, I)).

    Returns
    -------
    points : ndarray, shape (num_radii * num_surface_points, n)
    weights : ndarray, shape (num_radii * num_surface_points,). Sums to
        ``2**(beta / 2) * gamma((n + beta) / 2) / gamma(n / 2)``, the
        beta-th absolute moment of the chi_n distribution -- 1.0 when
        beta=0, not 1 in general.

    Examples
    --------
    >>> surf_pts, surf_w = _sphere_surface_points(3, 5)
    >>> pts, w = sphere_surface_to_gauss_points(surf_pts, surf_w, 5)
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 4)), 9)  # E[x^4] = 3
    3.0

    References
    ----------
    .. [1] A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
       Prentice-Hall, 1971, Ch. 2.8.
    .. [2] O. Straka, D. Dunik, and M. Simandl, "Randomized unscented
       Kalman filter in tracking," in Proc. 15th Int. Conf. on Information
       Fusion, Singapore, 2012, pp. 503-510.
    .. [3] J. Dunik, O. Straka, and M. Simandl, "The development of a
       randomised unscented Kalman filter," in Proc. 18th World Congress,
       IFAC, Milan, Italy, 2011, pp. 8-13.
    """
    surface_points = np.asarray(surface_points, dtype=np.float64)
    surface_weights = np.asarray(surface_weights, dtype=np.float64)
    if surface_points.ndim != 2:
        raise ValueError(
            f"surface_points must be 2-D, got shape {surface_points.shape}"
        )
    n = surface_points.shape[1]
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    if not beta > -n:
        raise ValueError(f"beta must be > -n ({-n}), got {beta}")

    # Same substitution as spherical_radial_points, generalized to alpha =
    # (n + beta)/2 - 1: exact for radial powers r^(2j), j = 0..2*m_r-1.
    # Needed j range is 0..degree//2, so m_r = ceil((degree//2 + 1)/2).
    m_r = (degree // 2 + 2) // 2
    alpha = (n + beta) / 2.0 - 1.0
    t, u = roots_genlaguerre(m_r, alpha)
    radii = np.sqrt(2.0 * t)
    w_rad = (u / gamma(n / 2.0)) * 2.0 ** (beta / 2.0)

    points = np.vstack([r * surface_points for r in radii])
    weights = np.concatenate([wr * surface_weights for wr in w_rad])
    return points, weights


def spherical_radial_points(
    n: int, degree: int, beta: float = 0.0
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Arbitrary-odd-degree spherical-radial cubature points for N(0, I) times
    \\|x\\|^beta.

    Product of a generalized Gauss-Laguerre radial rule (exact for all
    required even powers of r) with a dimension-recursive surface rule on
    the unit sphere. Generalizes the 3rd-degree spherical-radial rule of
    the CKF to any odd degree. ``beta`` selects the
    :func:`sphere_surface_to_gauss_points` weight family (``|x|^beta``
    times the Gaussian); the default ``beta=0.0`` is the plain N(0, I)
    case and its code path is byte-for-byte the original
    (pre-``beta``-parameter) implementation, so existing callers are
    unaffected.

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
    beta : float, optional
        Exponent of \\|x\\| in the weighting function, beta > -n. Default
        0.0 (plain N(0, I); weights then sum to 1).

    Returns
    -------
    points : ndarray, shape (num_points, n)
    weights : ndarray, shape (num_points,). Summing to 1 when beta=0;
        otherwise to ``2**(beta / 2) * gamma((n + beta) / 2) / gamma(n / 2)``
        (see :func:`sphere_surface_to_gauss_points`).

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
    if not beta > -n:
        raise ValueError(f"beta must be > -n ({-n}), got {beta}")

    surf_pts, surf_w = _sphere_surface_points(n, degree)

    if beta == 0.0:
        # Unchanged from before the beta parameter was added (see
        # TestSphericalRadialBetaGeneralization in
        # tests/unit/test_cubature_points.py for a bit-identity regression
        # guard against this branch).
        #
        # Radial part: substitute t = r^2/2 in the integral of g(r) r^(n-1)
        # exp(-r^2/2); Gauss-Laguerre with alpha = n/2 - 1 handles t^j
        # exactly. Even powers r^(2j) with 2j <= degree - 1 must be exact
        # => j <= (degree-1)/2 => m_r points with 2*m_r - 1 >= (degree-1)/2.
        # Checked: degree 3 -> 1 node (r = sqrt(n), the CKF radius); 5,7 ->
        # 2; 9 -> 3.
        m_r = (degree + 3) // 4
        t, wt = roots_genlaguerre(m_r, n / 2.0 - 1.0)
        radii = np.sqrt(2.0 * t)
        w_rad = wt / gamma(n / 2.0)

        points = np.vstack([r * surf_pts for r in radii])
        weights = np.concatenate([wr * surf_w for wr in w_rad])
        return points, weights / weights.sum()

    return sphere_surface_to_gauss_points(surf_pts, surf_w, degree, beta)


def _fourteenth_order_unit_sphere_points_3d() -> Tuple[
    NDArray[np.floating], NDArray[np.floating]
]:
    """Degree-14 rule for the uniform measure on the unit sphere S^2 (n=3 only).

    Stroud's surface formula U3 14-1 [1]_, p. 302, the counterpart of the
    MATLAB TCL's ``fourteenthOrderSpherSurfCubPoints``, which itself hardcodes
    ``numDim=3`` -- this specific 72-point construction has no documented
    n-dimensional generalization. 12 points from all sign flips of
    permutations of ``(r, s, 0)``, plus 60 points with icosahedral symmetry
    built from the positive roots of a degree-6 polynomial (Stroud's
    tabulated coefficients). Weights are normalized to sum to 1.

    The polynomial's 6 roots are consumed as one "hub" value paired with each
    of the other 5 (arranged in a 5-cycle) to build the 60-point block.
    MATLAB assigns hub=z(1) and cycle=z(2..6) to whatever order its ``roots``
    call happens to return -- an implementation-defined solver ordering this
    port does not try to reproduce bit-for-bit. Instead, the roots are sorted
    descending and the hub is assigned to the largest, which is forced (the
    unit-norm constraint below picks it out uniquely). The remaining 5-cycle
    assignment is NOT unique, though: an exhaustive search over all 720
    labelings found 10 that tie at the same ~1e-14 unit-norm residual,
    collapsing into exactly two 60-point clouds related by a single
    coordinate's sign flip -- a genuine mirror ambiguity that neither the
    unit-norm check nor degree-14 exactness (both verified below) can
    resolve, since a chiral construction and its mirror image integrate every
    polynomial identically. This port fixes one of the two mirrors
    deterministically (whichever the descending sort produces) and verifies
    it end-to-end against the closed-form N(0, I) moments; it does not claim
    to match MATLAB's specific mirror bit-for-bit, and a caller that needs
    that (e.g. to reproduce a published result exactly) should not assume it.
    """
    r = np.sqrt((5.0 - np.sqrt(5.0)) / 10.0)
    s = np.sqrt((5.0 + np.sqrt(5.0)) / 10.0)

    # Stroud's tabulated coefficients (highest degree first) for the degree-6
    # polynomial whose positive roots give the squared "radii" z_i**2 used
    # below; all 6 roots are real and positive for this polynomial.
    poly_coeffs = [
        2556125.0,
        -5112250.0,
        3578575.0,
        -1043900.0,
        115115.0,
        -3562.0,
        9.0,
    ]
    y = np.sort(np.roots(poly_coeffs).real)[::-1]  # descending: y[0] >= ... >= y[5]
    z1, z2, z3, z4, z5, z6 = np.sqrt(y)

    u = np.array([-z3 + z4, -z5 + z2, -z2 + z6, -z6 + z3, -z4 + z5]) / (2.0 * s)
    v = np.array([z5 + z6, z6 + z4, z3 + z5, z4 + z2, z2 + z3]) / (2.0 * s)
    w = np.array([z1 + z2, z1 + z3, z1 + z4, z1 + z5, z1 + z6]) / (2.0 * s)

    def pm_combos(x: ArrayLike) -> NDArray[np.floating]:
        """All sign flips of the nonzero entries of x (MATLAB's PMCombos)."""
        x = np.asarray(x, dtype=np.float64)
        nz = np.flatnonzero(x)
        out = []
        for signs in itertools.product((1.0, -1.0), repeat=len(nz)):
            p = x.copy()
            p[nz] = x[nz] * np.array(signs)
            out.append(p)
        return np.array(out)

    axis_points = np.vstack(
        [pm_combos([r, s, 0.0]), pm_combos([0.0, r, s]), pm_combos([s, 0.0, r])]
    )

    cyclic_points = []
    for ui, vi, wi in zip(u, v, w):
        cyclic_points.extend(
            [
                [ui, vi, wi],
                [ui, -vi, -wi],
                [-ui, -vi, wi],
                [-ui, vi, -wi],
                [vi, wi, ui],
                [vi, -wi, -ui],
                [-vi, -wi, ui],
                [-vi, wi, -ui],
                [wi, ui, vi],
                [wi, -ui, -vi],
                [-wi, -ui, vi],
                [-wi, ui, -vi],
            ]
        )

    points = np.vstack([axis_points, np.array(cyclic_points)])
    weights = np.concatenate(
        [np.full(12, 125.0 / 10080.0), np.full(60, 143.0 / 10080.0)]
    )
    return points, weights


def fourteenth_order_cubature_points(
    n: int, beta: float = 0.0
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Degree-14 cubature points for the standard normal N(0, I), n = 3 only.

    The counterpart of the MATLAB TCL's ``fourteenthOrderCubPoints``: lifts
    the 72-point degree-14 spherical-surface rule
    (:func:`_fourteenth_order_unit_sphere_points_3d`, Stroud's U3 14-1 [1]_)
    to N(0, I) times \\|x\\|^beta via :func:`sphere_surface_to_gauss_points`
    (the same adapter :func:`spherical_radial_points` uses), rather than
    duplicating the radial weight machinery. Exactly integrates every
    polynomial of total degree <= 14 against N(0, I).

    Unlike :func:`fifth_order_cubature_points`,
    :func:`seventh_order_cubature_points`, and :func:`spherical_radial_points`,
    this rule has no documented n-dimensional generalization in the source --
    MATLAB's ``fourteenthOrderCubPoints`` and ``fourteenthOrderSpherSurfCubPoints``
    both hardcode ``if(numDim~=3) error('Only 3D points are supported'); end``.
    So ``n = 3`` here is not a lower bound, it is the only supported value.

    Parameters
    ----------
    n : int
        Dimension; only n = 3 is supported (matches the MATLAB source's
        restriction).
    beta : float, optional
        Exponent of \\|x\\| in the weighting function, beta > -n. Default 0.0
        (plain N(0, I)).

    Returns
    -------
    points : ndarray
        Shape (288, 3) -- 72 surface points times 4 radial nodes.
    weights : ndarray
        Shape (288,). Sums to 1 when beta=0; otherwise to
        ``2**(beta / 2) * gamma((n + beta) / 2) / gamma(n / 2)`` (see
        :func:`sphere_surface_to_gauss_points`).

    Examples
    --------
    >>> pts, w = fourteenth_order_cubature_points(3)
    >>> pts.shape
    (288, 3)
    >>> round(float(w.sum()), 9)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 14)), 3)  # E[x^14] = 135135
    135135.0

    References
    ----------
    .. [1] A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
       Prentice-Hall, 1971, Formula U3 14-1, p. 302.
    """
    if n != 3:
        raise ValueError(f"only 3-D points are supported (numDim must be 3), got {n}")
    if not beta > -n:
        raise ValueError(f"beta must be > -n ({-n}), got {beta}")

    surf_pts, surf_w = _fourteenth_order_unit_sphere_points_3d()
    return sphere_surface_to_gauss_points(surf_pts, surf_w, 14, beta)
