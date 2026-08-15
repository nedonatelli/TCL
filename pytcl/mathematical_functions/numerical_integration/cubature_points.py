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
from collections import Counter
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma, roots_genlaguerre, roots_jacobi

from pytcl.mathematical_functions.basic_matrix.decompositions import chol_semi_def


def _pm_combos(x: ArrayLike) -> NDArray[np.floating]:
    """All sign flips of the nonzero entries of x (MATLAB's PMCombos)."""
    x = np.asarray(x, dtype=np.float64)
    nz = np.flatnonzero(x)
    out = []
    for signs in itertools.product((1.0, -1.0), repeat=len(nz)):
        p = x.copy()
        p[nz] = x[nz] * np.array(signs)
        out.append(p)
    return np.array(out)


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


def cubature_point_moments(
    points: ArrayLike,
    weights: ArrayLike,
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    mean: Optional[ArrayLike] = None,
    cov: Optional[ArrayLike] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Propagate a distribution's first two moments through ``func`` by
    cubature integration.

    Counterpart of the MATLAB TCL's ``calcCubPointMoments``: given a set of
    cubature points/weights, transform them through ``func`` and return the
    resulting mean and covariance. This is the public, filter-independent
    form of the transform-then-propagate pattern ``ckf_predict``/
    ``ckf_update`` (:mod:`pytcl.dynamic_estimation.kalman.unscented`) apply
    internally to a specific dynamics or measurement function.

    MATLAB's ``z``/``S`` (mean and lower-triangular covariance square root)
    are mandatory positional arguments there -- ``calcCubPointMoments``
    always affinely maps its ``xi`` unit points by them before applying
    ``h``. This port makes that step optional via ``mean``/``cov``: when
    both are given, ``points`` are treated as unit points for N(0, I) (or
    the corresponding unit rule for whatever distribution ``points``/
    ``weights`` target) and are mapped through
    :func:`transform_cubature_points` first, exactly as MATLAB's
    ``transformCubPoints`` does; when both are omitted, ``points`` are
    passed to ``func`` unchanged, e.g. because they are already points of
    the target distribution (as when reusing a filter's own cubature
    points, already scaled by its state covariance).

    MATLAB's optional ``innovTrans`` (custom difference function, e.g. for
    circular quantities) and ``meanFun`` (custom weighted-average function,
    e.g. for angular means) are not exposed: ``func``'s output is always
    averaged and differenced with plain arithmetic here. Callers needing a
    non-Euclidean mean/innovation should wrap ``func`` accordingly or
    average/difference the returned points by hand.

    Parameters
    ----------
    points : array_like
        Cubature points, shape (num_points, n). Unit points for N(0, I) if
        ``mean``/``cov`` are given; already-in-distribution points
        otherwise.
    weights : array_like
        Weights matching ``points``, shape (num_points,), normally summing
        to 1. May contain negative values (e.g. higher-degree rules); never
        suppressed.
    func : callable
        Maps a single point, shape (n,), to a transformed point, shape
        (m,).
    mean : array_like, optional
        Target mean, shape (n,). Must be given together with ``cov``.
    cov : array_like, optional
        Target covariance, shape (n, n). Must be given together with
        ``mean``. Need only be positive *semi*-definite -- factored with
        :func:`~pytcl.mathematical_functions.basic_matrix.decompositions.chol_semi_def`,
        matching MATLAB's own docstring recommendation
        (``cholSemiDef(R,'lower')``) and the same fallback
        ``ckf_predict``/``ckf_update`` already use for a filter's own
        (possibly rank-deficient) state covariance -- rather than a raw
        Cholesky that would raise on a singular or near-singular ``cov``.

    Returns
    -------
    mean_out : ndarray
        Weighted mean of ``func`` applied to the (possibly transformed)
        points, shape (m,).
    cov_out : ndarray
        Weighted covariance of ``func`` applied to the (possibly
        transformed) points, shape (m, m).

    Examples
    --------
    >>> pts, w = second_order_cubature_points(2)
    >>> mean = np.array([1.0, -2.0])
    >>> cov = np.diag([0.5, 2.0])
    >>> A = np.array([[2.0, 0.0], [1.0, 1.0]])
    >>> b = np.array([0.5, 0.5])
    >>> mu, P = cubature_point_moments(pts, w, lambda x: A @ x + b, mean, cov)
    >>> np.allclose(mu, A @ mean + b, atol=1e-10)
    True
    >>> np.allclose(P, A @ cov @ A.T, atol=1e-10)
    True

    See Also
    --------
    transform_cubature_points : The affine unit-point mapping applied when
        ``mean``/``cov`` are given.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D, got shape {points.shape}")
    num_points, n = points.shape
    if weights.shape != (num_points,):
        raise ValueError(
            f"weights shape {weights.shape} does not match {num_points} points"
        )
    if (mean is None) != (cov is None):
        raise ValueError("mean and cov must be provided together")

    if mean is not None:
        mean_arr = np.asarray(mean, dtype=np.float64).ravel()
        cov_arr = np.asarray(cov, dtype=np.float64)
        if mean_arr.shape != (n,) or cov_arr.shape != (n, n):
            raise ValueError(
                f"mean/cov dimensions {mean_arr.shape}/{cov_arr.shape} do not "
                f"match points dimension {n}"
            )
        sqrt_cov = chol_semi_def(cov_arr)
        eval_points, _ = transform_cubature_points(points, weights, mean_arr, sqrt_cov)
    else:
        eval_points = points

    transformed = np.array([func(p) for p in eval_points], dtype=np.float64)
    if transformed.ndim != 2 or transformed.shape[0] != num_points:
        raise ValueError(
            f"func must map each point to a fixed-size vector; got output "
            f"shape {transformed.shape} for {num_points} points"
        )

    mean_out = np.sum(weights[:, np.newaxis] * transformed, axis=0)
    resid = transformed - mean_out
    cov_out = resid.T @ (weights[:, np.newaxis] * resid)

    return mean_out, cov_out


def second_order_cubature_points(
    n: int,
    w0: float = 1.0 / 3.0,
    alpha: float = 1.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Degree-2 cubature points for the standard normal N(0, I): the scaled
    unscented transformation's minimal (n+2)-point spherical-simplex rule.

    Counterpart of the MATLAB TCL's ``secondOrderCubPoints``: the scaled
    unscented transformation of Julier [1]_, itself a scaled generalization
    of the *spherical simplex sigma points* of Appendix III of Julier and
    Uhlmann [2]_ (``alpha=1`` reproduces those unscaled). A center point at
    the origin plus n+1 simplex vertices, built recursively one dimension at
    a time and then rescaled so the point set has unit sample covariance.

    This is a **genuinely different construction** from both cubature rules
    already in pytcl, not an alternate parameterization of either -- see
    Notes.

    Parameters
    ----------
    n : int
        Dimension, n >= 1.
    w0 : float, optional
        Weight of the center point before ``alpha``-scaling, 0 < w0 < 1.
        Default 1/3 (MATLAB's default). The rescaling below divides by
        ``sqrt((1/w0 - 1) / (n + 1))``, which -> 0 as ``w0 -> 1``, so
        values close to the open upper bound blow the non-center points up
        without limit and degrade the sample covariance accordingly (e.g.
        ``w0 = 1 - 1e-12`` at n=3 places points near 1.7e6 and leaves the
        sample covariance off from the identity by ~1.1e-4) -- the same
        kind of near-boundary blow-up :func:`student_t_cubature_points`
        documents for ``dof`` near its own open lower bound. The ``0 < w0
        < 1`` check does not, and is not meant to, guard against this.
    alpha : float, optional
        Positive spread factor for the non-center points. Default 1.0
        (unscaled spherical simplex points). Values of ``alpha`` and ``w0``
        with ``alpha**2 < 1 - w0`` drive the *center* weight negative --
        see Notes.

    Returns
    -------
    points : ndarray
        Shape (n + 2, n).
    weights : ndarray
        Shape (n + 2,), summing to 1. The center weight
        ``w0 / alpha**2 + (1 - 1 / alpha**2)`` can be negative (see Notes);
        this is inherent to the scaled construction, not an error.
        Covariances assembled from these points must not use a
        sqrt-of-weights factorization.

    Notes
    -----
    **Relation to pytcl's other sigma/cubature-point generators.** All three
    trace to the same Julier & Uhlmann unscented-transform lineage but are
    structurally distinct rules, not reparameterizations of one another:

    - :func:`~pytcl.mathematical_functions.numerical_integration.quadrature.unscented_transform_points`
      is the classic *symmetric* sigma-point set: 2n+1 points (center plus
      an antipodal pair per axis), exact through degree 3 for N(0, I)
      because the antipodal pairs cancel every odd-order term.
    - :func:`~pytcl.dynamic_estimation.kalman.unscented.ckf_spherical_cubature_points`
      is the CKF's 2n-point spherical-radial rule (no center point, all
      weights equal), also exact through degree 3.
    - This function is the (n+2)-point *spherical simplex* set: no
      antipodal symmetry (only the first coordinate axis has a mirrored
      pair; every later dimension's simplex vertices are one-sided), so it
      is exact through degree 2 only -- matching the mean and covariance
      exactly but *not* third moments in general. E.g. at n=3, w0=1/3,
      alpha=1, the rule gives ``E[x2^3] = sqrt(2)`` against a true value of
      0, even though ``E[x0^3] = 0`` by the one axis that does have a
      mirrored pair.

    A user choosing between these should pick
    ``unscented_transform_points`` or ``ckf_spherical_cubature_points``
    (degree-3 exact, standard choices for the UKF/CKF) unless the point
    budget must be as small as possible: this rule's n+2 points is the
    fewest of the three, at the cost of the third-moment accuracy the other
    two get for free from symmetry.

    **Negative center weight.** Unlike the unscaled (``alpha=1``) spherical
    simplex points, for which ``0 < w0 < 1`` keeps every weight positive,
    the scaled center weight ``w0 / alpha**2 + (1 - 1 / alpha**2)`` goes
    negative whenever ``alpha**2 < 1 - w0`` (e.g. ``w0=1/3``, ``alpha=0.5``
    gives a center weight of -5/3). This is a real, disclosed property of
    Julier's scaled construction -- corrected here per Equation 15 of [1]_
    (MATLAB's own header notes that Equation 24 of the same paper has a
    typo the code does not follow), not suppressed or clamped.

    ``randomize`` (MATLAB's optional post hoc random-orthonormal-rotation
    parameter, used in [3]_ and [4]_ to avoid repeated-orientation artifacts
    in tracking) is not exposed; callers who want it can rotate the
    returned points themselves.

    Examples
    --------
    >>> pts, w = second_order_cubature_points(3)
    >>> pts.shape
    (5, 3)
    >>> round(float(w.sum()), 12)
    1.0
    >>> np.allclose(np.sum(w[:, None] * pts, axis=0), 0.0, atol=1e-12)  # E[x] = 0
    True
    >>> resid = pts - np.sum(w[:, None] * pts, axis=0)
    >>> np.allclose(resid.T @ (w[:, None] * resid), np.eye(3), atol=1e-12)  # Cov = I
    True

    References
    ----------
    .. [1] S. J. Julier, "The scaled unscented transformation," in Proc.
       American Control Conference, Anchorage, AK, 8-10 May 2002,
       pp. 4555-4559.
    .. [2] S. J. Julier and J. K. Uhlmann, "Unscented filtering and
       nonlinear estimation," Proceedings of the IEEE, vol. 92, no. 3,
       pp. 401-422, Mar. 2004.
    .. [3] O. Straka, D. Dunik, and M. Simandl, "Randomized unscented
       Kalman filter in tracking," in Proc. 15th Int. Conf. on Information
       Fusion, Singapore, 2012, pp. 503-510.
    .. [4] J. Dunik, O. Straka, and M. Simandl, "The development of a
       randomised unscented Kalman filter," in Proc. 18th World Congress,
       IFAC, Milan, Italy, 2011, pp. 8-13.
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if not 0.0 < w0 < 1.0:
        raise ValueError(f"w0 must satisfy 0 < w0 < 1, got {w0}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    num_points = n + 2
    w = np.zeros(num_points)
    xi = np.zeros((n, num_points))  # dim x point, MATLAB's layout; transposed on return

    w[0] = w0
    w[1:] = (1.0 - w0) / (n + 1)

    # Dimension 1 (row 0): the one axis with a mirrored pair, at columns 1, 2.
    xi[0, 1] = -1.0 / np.sqrt(2.0 * w0)
    xi[0, 2] = 1.0 / np.sqrt(2.0 * w0)

    # Dimensions 2..n: each adds one new simplex vertex (column j+1) while
    # extending the negative value into the columns already in use.
    for j in range(2, n + 1):
        row = j - 1
        xi[row, 1 : j + 1] = -1.0 / np.sqrt(j * (j + 1) * w0)
        xi[row, j + 1] = j / np.sqrt(j * (j + 1) * w0)

    # Rescale so the unscaled (alpha=1) point set has unit sample covariance.
    xi = xi / np.sqrt((1.0 / w0 - 1.0) / (n + 1))

    # Scaled unscented transformation: spread the non-center points by
    # alpha, then correct the weights per Equation 15 of [1] (not the
    # typo'd Equation 24).
    xi = alpha * xi
    w[0] = w[0] / alpha**2 + (1.0 - 1.0 / alpha**2)
    w[1:] = w[1:] / alpha**2

    return xi.T, w


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
        Precondition (not validated): every row has unit norm -- a rule
        built from points off the sphere silently integrates a different
        weighting than the one this function documents.
    surface_weights : array_like
        Weights for the uniform measure on S^(n-1), shape
        (num_surface_points,), summing to 1 (checked: must be 1-D and its
        length must match ``surface_points``'s row count, or ``ValueError``
        is raised). Precondition (not validated): the weights actually sum
        to 1.
    degree : int
        The polynomial degree the surface rule (and thus this rule) is
        exact through, an integer >= 1. Unlike
        :func:`spherical_radial_points`, EVEN values are accepted here --
        this is the lower-level adapter :func:`fourteenth_order_cubature_points`
        itself calls with ``degree=14`` -- so only integer-ness is checked,
        not oddness. Precondition (not validated): this must match the
        degree the supplied ``surface_points``/``surface_weights`` rule was
        actually built for -- an inconsistent value is accepted silently
        and just mislabels the resulting rule's true degree.
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
    num_surface_points = surface_points.shape[0]
    if surface_weights.ndim != 1 or surface_weights.shape[0] != num_surface_points:
        raise ValueError(
            f"surface_weights shape {surface_weights.shape} does not match "
            f"{num_surface_points} surface points"
        )
    if not float(degree).is_integer() or degree < 1:
        raise ValueError(f"degree must be an integer >= 1, got {degree}")
    degree = int(degree)
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

    Notes
    -----
    **Deviation from MATLAB at n=1.** MATLAB's ``arbOrderGaussCubPoints``
    (the ``beta != 0`` routine this function's ``beta`` path delegates to,
    via :func:`sphere_surface_to_gauss_points`) hard-errors at
    ``numDim==1`` (``error('numDim must be >1.')``); this port accepts
    ``n=1`` for both the ``beta=0`` and ``beta != 0`` paths and returns a
    correct rule instead -- S^0 = {-1, +1} is a valid, if degenerate,
    sphere, and the construction handles it without special-casing. This
    was already true of the pre-existing ``beta=0`` path; the ``beta``
    path added alongside this note now exercises the same n=1 case (via
    :func:`sphere_surface_to_gauss_points`, which never had MATLAB's
    ``numDim>1`` restriction to begin with), so it inherits the same
    disclosed divergence.

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
    if not float(degree).is_integer() or degree < 3 or degree % 2 == 0:
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

    axis_points = np.vstack(
        [_pm_combos([r, s, 0.0]), _pm_combos([0.0, r, s]), _pm_combos([s, 0.0, r])]
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


# Tabulated Genz-Keister generators, transcribed verbatim from
# GenzKeisterPoints.m. lambda_0 = 0 and lambda_1 = sqrt(3) are common to
# both (the 3-point Gauss-Hermite rule); the remaining entries were computed
# by Genz and Keister [1]_ in extended precision and are reproduced here at
# the double precision the MATLAB source publishes them in -- see the
# "Precision at the top of the range" note in `genz_keister_points`'s
# docstring for what that costs at the very last entry of each table.
_GK_LAMBDA_0 = np.array(
    [
        0.0,
        1.7320508075688773,
        4.1849560176727319,
        0.74109534999454084,
        2.8612795760570581,
        6.3633944943363700,
        1.2304236340273060,
        5.1870160399136561,
        2.5960831150492022,
        3.2053337944991945,
        9.0169397898903025,
        0.24899229757996061,
        7.9807717985905609,
        2.2336260616769417,
        7.1221067008046167,
        3.6353185190372782,
        5.6981777684881096,
        4.7364330859522971,
    ]
)
_GK_A_0 = np.array(
    [
        1.0,
        1.0,
        0.0,
        6.0,
        -48.378475125832451,
        0.0,
        0.0,
        0.0,
        34020.0,
        -986064.53173677489,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.2912054173706603e12,
        -1.1268664521456168e14,
        2.9248520348796280e15,
    ]
)
_GK_LAMBDA_1 = np.array(
    [
        0.0,
        1.7320508075688773,
        4.9791465117195582,
        0.84628809835102170,
        3.7355715460409573,
        2.6840395601585692,
        9.0508037980317400,
        0.47371420996884380,
        8.0130130598043254,
        1.2435457006528093,
        7.1482776511870860,
        2.2210157242456798,
        6.3725842092196923,
        3.1782891110545301,
        5.6545621267720157,
        4.3394221426603945,
    ]
)
_GK_A_1 = np.array(
    [
        1.0,
        1.0,
        0.0,
        6.0,
        -93.0486211834777976,
        504.496566347049718,
        0.0,
        0.0,
        0.0,
        0.0,
        1.93536000000001776e6,
        -2.38763644847775079e8,
        4.62442819320708296e9,
        8.22485150843440875e9,
        -4.90446886942675039e12,
        7.61797098142559229e13,
    ]
)
# MATLAB's documented per-algorithm bound on m ("<=17 for algorithm 0 and
# <=15 for algorithm 1"): len(lambda) - 1, since lambda is indexed 0..m.
_GK_TABLES = {
    0: (_GK_LAMBDA_0, _GK_A_0),
    1: (_GK_LAMBDA_1, _GK_A_1),
}


def _gk_inner_terms(
    a: NDArray[np.floating], lam2: NDArray[np.floating], m: int
) -> NDArray[np.floating]:
    """b[p, k] = a[k] / prod_{idx=0..k, idx != p} (lam2[p] - lam2[idx]), for k >= p.

    This is the closed form of MATLAB's incrementally-accumulated
    ``innerTerms4Weights(pSubi, pkSum)``: tracing through its mutable
    ``prodVal`` loop shows the product it accumulates for entry (p, k) runs
    over idx = 0..k (excluding p), NOT the full 0..m -- i.e. each column k
    is a Newton-form divided-difference coefficient built from only the
    first k+1 generator points, not all m+1. Verified against a literal,
    line-for-line translation of MATLAB's loop (mutable ``prodVal`` /
    ``cardinalityLeft`` state machine and all) on randomized inputs before
    trusting this closed form in the port.
    """
    b = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for p in range(k + 1):
            prod = 1.0
            for idx in range(k + 1):
                if idx != p:
                    prod *= lam2[p] - lam2[idx]
            b[p, k] = a[k] / prod
    return b


def _gk_partition_weight(
    m: int, p: Tuple[int, ...], inner_terms: NDArray[np.floating]
) -> float:
    """The shared weight for every point generated from partition p (MATLAB's computeW).

    Equation (unnumbered, before Eq. 1 in [1]_, = Eq. 2.4 in [2]_):
    ``w(p) = 2^(-K) * sum_{k: k_i>=p_i, sum(k)<=m} prod_i b(p_i, k_i)``,
    K = number of nonzero entries of p. The sum over admissible k-vectors is
    computed here as a truncated polynomial convolution (one factor per
    dimension, each factor's coefficients are ``b[p_i, p_i:p_i+budget+1]``)
    rather than MATLAB's mutable nested-loop state machine
    (``cardinalityLeft`` / ``kpSum`` / ``prodSums``) -- verified to agree
    with a literal translation of that state machine on randomized (m, p)
    inputs across n = 1..4 before trusting it here.
    """
    active = sum(1 for pi in p if pi != 0)
    budget = m - sum(p)
    poly = np.array([1.0])
    for pi in p:
        row = inner_terms[pi, pi : pi + budget + 1]
        poly = np.convolve(poly, row)[: budget + 1]
    return float(2.0 ** (-active) * poly.sum())


def _gk_partitions_at_most_n(total: int, n: int):
    """Non-increasing tuples of length n, each entry >= 0, summing to total.

    One representative per partition of `total` into at most n parts
    (zero-padded to length n) -- the multiset class that
    `_multiset_permutations` then expands to every distinct placement
    across the n dimensions, matching MATLAB's
    ``getNextMPartition(n+curCard, n) - 1`` + ``genAllMultisetPermutations``
    pairing (that pairing is what lets `_gk_partition_weight` be computed
    once per partition and reused for every point in its symmetry orbit).
    """

    def helper(remaining: int, parts_left: int, max_val: int):
        if parts_left == 0:
            if remaining == 0:
                yield ()
            return
        if parts_left == 1:
            if remaining <= max_val:
                yield (remaining,)
            return
        for first in range(min(remaining, max_val), -1, -1):
            for rest in helper(remaining - first, parts_left - 1, first):
                yield (first,) + rest

    yield from helper(total, n, total)


def _multiset_permutations(seq: Tuple[int, ...]):
    """Every distinct permutation of seq, each yielded exactly once.

    Counterpart of MATLAB's ``genAllMultisetPermutations``: seq typically
    has repeated entries (zero-padding, repeated partition parts), so a
    plain ``itertools.permutations`` would yield many duplicates.
    """
    counts = Counter(seq)
    values = sorted(counts)
    length = len(seq)
    current = [0] * length

    def rec(pos: int):
        if pos == length:
            yield tuple(current)
            return
        for v in values:
            if counts[v] > 0:
                counts[v] -= 1
                current[pos] = v
                yield from rec(pos + 1)
                counts[v] += 1

    yield from rec(0)


def genz_keister_points(
    n: int,
    m: int,
    algorithm: int = 0,
    eps_val: Optional[float] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""
    Genz-Keister nested cubature points for the standard normal N(0, I).

    Counterpart of the MATLAB TCL's ``GenzKeisterPoints``: a fully-symmetric
    interpolatory rule (Genz [2]_, extended by Genz and Keister [1]_) built
    from a single 1-D sequence of NESTED generator magnitudes
    ``lambda_0=0, lambda_1=sqrt(3), lambda_2, ..., lambda_M`` -- each
    ``algorithm``'s full table is a strict superset of the previous one's
    prefix, which is what makes the resulting n-D point sets nest across
    increasing ``m`` too (see Notes). This is the enabling primitive for
    Smolyak sparse grids: sparse-grid construction reuses function
    evaluations across levels only when the levels' point sets nest, which
    plain (non-nested) Gauss-Hermite point sets do not do.

    Parameters
    ----------
    n : int
        Dimension, n >= 1.
    m : int
        Controls both the number of generator magnitudes available and the
        rule's exactness degree -- see Notes for exactly how. Must satisfy
        ``1 <= m <= 17`` for ``algorithm=0``, ``1 <= m <= 15`` for
        ``algorithm=1`` (``len(lambda table) - 1`` for each; MATLAB's own
        documented bounds).
    algorithm : int, optional
        Which tabulated generator to use (MATLAB's ``algorithm`` selector):

        - 0 (default): :math:`Q_P`, built from the increment vector
          ``nu = [3, 5, 8]`` -- the first column of Table 3.4 in [1]_.
        - 1: :math:`\hat{Q}_P`, built from ``nu = [4, 10]`` -- the second
          column of Table 3.4 in [1]_.

        MATLAB also accepts a vector of custom ``nu`` increments (via the
        internal ``getGenzKeisterGenerators``/``computeAValues`` routines)
        to derive a new generator from scratch. **This port does not
        implement that path** -- MATLAB's own docstring calls it "generally
        a bad idea" and warns of "a severe loss of precision" for the
        higher-order lambda terms it produces; the two tabulated generators
        above were computed by Genz and Keister in extended precision
        specifically to avoid that, and are what every practical use of
        this rule wants. ``algorithm`` values other than 0 or 1 raise
        ``ValueError``.
    eps_val : float, optional
        Points whose |weight| is at or below this threshold are dropped
        (MATLAB's pruning of "numerically zero" weights, which is what
        keeps the point count far below the naive :math:`(2m+1)^n` bound --
        see Notes). Default ``None`` uses ``eps(1)`` in double precision
        (``numpy.finfo(numpy.float64).eps``), matching MATLAB's default.

    Returns
    -------
    points : ndarray
        Shape (num_points, n).
    weights : ndarray
        Shape (num_points,), summing to 1. Genz-Keister rules commonly
        produce negative weights (visible directly in the worked examples
        below); these are inherent to the construction and are never
        suppressed, only the numerically-zero ones are dropped per
        ``eps_val``.

    Notes
    -----
    **What `m` means.** Write a "level" as a partition
    :math:`p = (p_1, ..., p_n)` of nonnegative integers with
    :math:`p_1 + ... + p_n \le m`; each :math:`p_i` selects
    :math:`\lambda_{p_i}` for dimension :math:`i`. The rule sums, over every
    such partition and every one of its signed permutations, a shared
    per-partition weight (MATLAB's ``computeW``) -- so ``m`` is a *shared
    budget* for how many generator "levels" the n dimensions may jointly
    spend, not a per-dimension point count. This module verified two
    independent, useful consequences of that budget structure numerically
    (both cross-checked against a literal, line-for-line translation of
    MATLAB's ``computeW``/``innerTerms4Weights`` state machine before being
    trusted, and against an unoptimized brute-force reconstruction that
    evaluates every raw n-tuple in ``{0,...,m}^n`` instead of one
    representative per symmetry orbit):

    1. **General guarantee, any n, up to an n-DEPENDENT ceiling well below
       each algorithm's maximum m (verified empirically for n = 1..4; do
       not assume it extrapolates past n=4 or past this floor without
       checking):** the rule is exact through total polynomial degree
       :math:`2m + 1` (equivalently, since odd-total-degree monomials are
       already exact at every ``m`` via antipodal symmetry, through the
       largest even degree :math:`2m`), and this bound is sharp there (a
       *mixed*-exponent monomial of degree :math:`2m+2` fails) -- e.g. at
       n=2, algorithm=0, m=4: ``E[x1^8 x2^2]`` rule-integrates to ``153.38``
       against a true value of ``105``, even though every *single-axis*
       monomial through degree 16 is still exact there (next point).

       This does NOT hold all the way up to each algorithm's true maximum
       ``m`` -- an earlier version of this note claimed the guarantee held
       for every ``m`` except the algorithm's own maximum (17 for
       algorithm 0, 15 for algorithm 1), which is false: the breakdown
       starts *before* the maximum once ``n >= 2``, and gets worse, not
       better, as ``n`` grows further. Measured largest ``m`` for which
       every degree-:math:`2m` monomial (mixed-exponent included -- the
       worst case is usually mixed, not single-axis; see the worked
       examples below) is exact to a relative error at or below the
       ``~1e-12`` roundoff-noise floor, by ``n``:

       ========= ====== ========= ====== ======
       algorithm  n=1    n=2       n=3    n=4
       ========= ====== ========= ====== ======
       0 (max 17) 16     15        15     15
       1 (max 15) 14     15 [#]_   13     13
       ========= ====== ========= ====== ======

       .. [#] No violation was found anywhere in algorithm 1's valid range
          at n=2 (worst measured relative error ``2.1e-13``, at ``m=15``,
          its true maximum) -- unlike every other column, this one is not
          *known* to break before the algorithm's own ``m`` ceiling, it was
          simply never observed to in the swept range.

       A single conservative number usable without checking ``n`` first:
       the guarantee above is verified for **m <= 15 (algorithm 0),
       m <= 13 (algorithm 1), for every n in 1..4**. Above that floor,
       consult the table (or remeasure for your own ``n``) rather than
       assume the generic bound holds -- e.g. at n=3, algorithm=0, m=16
       (one step above n=3's floor), the worst degree-32 monomial --
       ``E[x2^32]`` (0, 32, 0), mixed-exponent monomials are not always
       worse here -- is off by a relative 3.2e-2; at n=3, algorithm=1,
       m=14 (one step above n=3's floor there), the worst degree-28
       monomial is off by a relative 4.5e-2 -- notably higher than the
       3.1e-3 a single-axis-only probe (28, 0, 0) would suggest, because
       the true worst case there, (0, 6, 22), is mixed-exponent. See
       "Precision at the top of the range" below for why this degrades
       smoothly rather than cutting off, and why it starts sooner as ``n``
       grows.
    2. **Bonus at specific "milestone" m values, single-axis moments only:**
       MATLAB's tabulated :math:`\lambda`/:math:`a` values were tuned by
       Genz and Keister so that at the m where each ``nu`` stage completes,
       *single-axis* moments (equivalently, the n=1 rule itself) are exact
       to a degree well beyond :math:`2m+1` -- this is what "the first
       column of Table 3.4" in [1]_ actually tabulates (this port does not
       have that table's literal text, only the MATLAB source's transcribed
       :math:`\lambda`/:math:`a` constants, so the milestone degrees below
       were independently determined by direct computation against the
       closed-form N(0,1) moments, not copied from the paper):

       ================  =====  ==========  ======  ========================
       algorithm          m     points (n=1) degree  nu stage
       ================  =====  ==========  ======  ========================
       0 (:math:`Q_P`)    1      3            5      base (0, +-sqrt(3))
       0                  3      7            7      (no bonus -- not a stage boundary)
       0                  4      9           15      nu[0]=3 complete
       0                  8     17           17      (no bonus)
       0                  9     19           29      nu[1]=5 complete
       0                 15     31           31      (no bonus)
       0                 16     33           33      (no bonus)
       0                 17     33         see below  nu[2]=8 "complete" -- precision breakdown
       1 (:math:`\hat{Q}_P`)  1  3            5      base
       1                  3      7            7      (no bonus)
       1                  4      9            9      (no bonus)
       1                  5     11           19      nu[0]=4 complete
       1                 10     21           21      (no bonus)
       1                 14     29           29      (no bonus)
       1                 15     29         see below  nu[1]=10 "complete" -- precision breakdown
       ================  =====  ==========  ======  ========================

       (m values between milestones, and m=2/m=1 or m=14/m=... duplicates,
       reuse the previous milestone's point set post-pruning -- see the
       nesting property below.) Mixed-exponent moments do **not** inherit
       this bonus at any m, milestone or not (point 1 above).

    **Precision at the top of the range (disclosed, not a porting bug).**
    The rows marked "see below" in the milestone table above are real and
    reproducible, not an artifact of this port, but they are not a
    "degree" claim at all: at ``m = 17`` (algorithm 0) and ``m = 15``
    (algorithm 1) -- the *last* stage boundary, which is also the maximum
    ``m`` MATLAB's own docstring allows -- there is no accuracy cliff to
    report a single degree for, for the n=1 marginal those milestone rows
    describe. Every other tested ``m`` has relative error staying flat
    (``~1e-16``) up to some degree and then cutting off sharply; at the
    maximum ``m`` of each algorithm the relative error instead grows
    *smoothly* with degree from the start, so "the degree this rule is
    exact to" becomes purely a function of whatever tolerance is used to
    define "exact". Measured directly for algorithm 0 at m=17 (n=1
    marginal, single-axis moments): relative error is ``2.2e-16`` at
    degree 0 and grows to ``5.6e-7`` by degree 24 with no step anywhere in
    between; the resulting degree at a chosen relative tolerance is 35 @
    ``1e-4``, 25 @ ``1e-6``, 19 @ ``1e-8``, 13 @ ``1e-10``, 9 @ ``1e-12``,
    5 @ ``1e-14`` -- no tolerance reproduces a "15" here; there is no such
    number, at any tolerance.

    The SAME smooth-growth-not-cliff pattern, with the SAME underlying
    cause, is what makes item 1's n-dependent floor above sit below each
    algorithm's true maximum ``m`` once ``n >= 2``: e.g. at n=3,
    algorithm=0, m=16 (n=3's floor there is 15, one below), the
    single-axis marginal ``E[x2^d]`` measured at even ``d`` from 0 to 32
    grows from ``4.9e-15`` to ``3.2e-2`` with no step in between either --
    it is the identical phenomenon as the n=1/m=17 case above, just
    triggered one ``m`` earlier. This traces to the published
    double-precision :math:`\lambda`/:math:`a` constants themselves:
    achieving the top stage's designed accuracy requires near-exact
    cancellation among terms spanning roughly 15 to 16 orders of magnitude
    (``a`` ranges up to ``2.92e15`` at algorithm 0's last entry), which is
    at the edge of what float64 can represent faithfully even with
    exact-precision *inputs*; at n=1 only that single generator sequence
    is ever combined against itself, but at higher ``n`` the shared
    per-partition weight table (``computeW``) combines it across
    combinatorially more partitions, so the same precision loss surfaces
    at a lower ``m`` (this explains the *direction* of the n-dependence
    empirically measured in item 1, not a claim independently re-derived
    from the constants). Do not rely on exactness AT ALL above the
    n-dependent floor established in item 1 above -- not even up to the
    generic :math:`2m+1` bound that holds at or below it (see the
    counterexamples there: a 3.2e-2 relative error at n=3, algorithm=0,
    m=16, degree 32 (one step above n=3's floor of 15); a 4.5e-2 relative
    error at n=3, algorithm=1, m=14, degree 28 (one step above n=3's floor
    of 13 there)).

    **Nesting.** For every consecutive pair ``m-1, m`` **except the last
    one** (``m = 17`` for algorithm 0, ``m = 15`` for algorithm 1), the
    point set at ``m`` contains every point of ``m - 1`` to floating
    tolerance -- verified explicitly for n = 1, 2, 3 across the full valid
    range. At the excluded top pair, the newly available generator
    magnitude does not simply add to the previous point set: recomputing
    the shared weight table with the larger ``m`` shifts an existing
    extremal point's weight across the ``eps_val`` pruning threshold,
    dropping it as a new, unrelated extremal point becomes available. This
    is the same precision effect described above, viewed through its effect
    on the point *set* rather than on exactness degree, and follows
    directly from it (both are consequences of the last stage's constants
    being computed by Genz and Keister in extended precision, but published
    here, and by MATLAB, at double precision).

    **Point count.** MATLAB's docstring describes the point-count formula
    "given in the text after Equation 1" of [1]_; this port does not
    preallocate (MATLAB does, to size ``xi``/``w`` before filling them) and
    instead grows the point list directly, so that formula was not needed
    here.

    ``randomize`` (MATLAB's optional post hoc random-orthonormal-rotation
    parameter, see [3]_, [4]_) is not exposed; callers who want it can
    rotate the returned points themselves.

    Examples
    --------
    >>> pts, w = genz_keister_points(1, 1)
    >>> sorted(pts.ravel().round(10).tolist())
    [-1.7320508076, 0.0, 1.7320508076]
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 4)), 9)  # E[x^4] = 3, exact at m=1
    3.0

    >>> pts9, w9 = genz_keister_points(1, 4)  # milestone m=4: bonus degree 15
    >>> pts9.shape
    (9, 1)
    >>> round(float(np.sum(w9 * pts9[:, 0] ** 14)), 3)  # E[x^14] = 135135
    135135.0

    >>> _, w19 = genz_keister_points(1, 9)  # milestone m=9: bonus degree 29
    >>> bool((w19 < 0).any())  # Genz-Keister rules commonly have negative weights
    True

    >>> pts1, w1 = genz_keister_points(1, 1)
    >>> pts3, w3 = genz_keister_points(1, 3)  # nesting: m=1's points subset of m=3's
    >>> all(np.any(np.isclose(pts3, p, atol=1e-12)) for p in pts1)
    True

    References
    ----------
    .. [1] A. Genz and B. D. Keister, "Fully symmetric interpolatory rules
       for multiple integrals over infinite regions with Gaussian weight,"
       Journal of Computational and Applied Mathematics, vol. 71, no. 2,
       pp. 299-309, Jul. 1996.
    .. [2] A. Genz, "Fully symmetric interpolatory rules for multiple
       integrals," SIAM Journal on Numerical Analysis, vol. 23, no. 6, pp.
       1273-1283, Dec. 1986.
    .. [3] O. Straka, D. Dunik, and M. Simandl, "Randomized unscented
       Kalman filter in tracking," in Proc. 15th Int. Conf. on Information
       Fusion, Singapore, 2012, pp. 503-510.
    .. [4] J. Dunik, O. Straka, and M. Simandl, "The development of a
       randomised unscented Kalman filter," in Proc. 18th World Congress,
       IFAC, Milan, Italy, 2011, pp. 8-13.
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if algorithm not in _GK_TABLES:
        raise ValueError(
            "algorithm must be 0 (Q_P, nu=[3,5,8]) or 1 (Q-hat_P, nu=[4,10]); "
            "the custom-nu-vector generator path is not ported (MATLAB's own "
            "docstring calls it 'generally a bad idea' due to severe precision "
            f"loss), got {algorithm!r}"
        )
    lam, a = _GK_TABLES[algorithm]
    max_m = len(lam) - 1
    if not isinstance(m, (int, np.integer)) or m < 1 or m > max_m:
        raise ValueError(f"m must be an integer with 1 <= m <= {max_m}, got {m}")
    if eps_val is None:
        eps_val = float(np.finfo(np.float64).eps)

    lam2 = lam[: m + 1] ** 2
    inner_terms = _gk_inner_terms(a, lam2, m)

    point_blocks = []
    weight_blocks = []
    for card in range(m + 1):
        for p in _gk_partitions_at_most_n(card, n):
            w_p = _gk_partition_weight(m, p, inner_terms)
            for perm in _multiset_permutations(p):
                combos = _pm_combos(lam[list(perm)])
                point_blocks.append(combos)
                weight_blocks.append(np.full(len(combos), w_p))

    points = np.vstack(point_blocks)
    weights = np.concatenate(weight_blocks)
    keep = np.abs(weights) > eps_val
    return points[keep], weights[keep]


def student_t_cubature_points(
    n: int, dof: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Third-order cubature points for the standard multivariate Student-t.

    Counterpart of the MATLAB TCL's ``thirdOrderStudentTCubPoints``: unit
    points for the multivariate Student-t distribution with zero mean,
    identity scale matrix, and ``dof`` degrees of freedom -- the Student-t
    analogue of :func:`~pytcl.dynamic_estimation.kalman.unscented.ckf_spherical_cubature_points`'s
    2n-point Gaussian rule. To use Student-t process or measurement noise
    in a cubature filter, swap these points/weights in wherever the
    Gaussian cubature points would otherwise go (e.g. via ``ckf_predict``'s
    ``points``/``weights`` arguments), then affinely map by the target mean
    and scale-matrix square root exactly as the Gaussian points are (see
    :func:`transform_cubature_points`).

    MATLAB's ``mu`` (mean) and ``SR`` (lower-triangular scale-matrix square
    root) arguments are folded into that same affine-map step used
    everywhere else in this module rather than taken here directly: with
    ``SR = I`` MATLAB's ``xi(:,curPoint) = nuConst * SR(:,curDim)`` reduces
    to ``nuConst`` times the standard basis vectors, which is exactly what
    this function returns before the caller (or ``transform_cubature_points``)
    maps them to a specific mean/scale.

    Parameters
    ----------
    n : int
        Dimension, n >= 1.
    dof : float
        Degrees of freedom, dof > 2. The constraint comes directly from
        ``nuConst = sqrt(dof * n / (dof - 2))`` in the source: below
        ``dof = 2`` the multivariate Student-t's covariance
        ``dof / (dof - 2) * Sigma`` itself does not exist (the term the
        rule is built to reproduce, see Notes), so there is no covariance
        left for a third-order rule to match.

    Returns
    -------
    points : ndarray
        Shape (2n, n): pairs (+nuConst * e_i, -nuConst * e_i) for each axis
        i, interleaved in that order (MATLAB's ``curPoint`` loop -- not
        grouped into all-positive-then-all-negative blocks the way
        :func:`~pytcl.dynamic_estimation.kalman.unscented.ckf_spherical_cubature_points`
        is).
    weights : ndarray
        Shape (2n,), each ``1 / (2n)``, summing to 1.

    Notes
    -----
    **Why third order, not higher.** The rule matches the distribution's
    mean (0, by antipodal symmetry -- every odd-total-degree monomial
    vanishes for the same reason it does in
    :func:`~pytcl.dynamic_estimation.kalman.unscented.ckf_spherical_cubature_points`)
    and its per-axis second moment, ``E[x_i^2] = dof / (dof - 2)``, via
    ``nuConst**2 = dof * n / (dof - 2)`` (see the worked example below). It
    is not exact at degree 4: e.g. the rule's ``E[x_i^4]`` reduces to
    ``nuConst**4 / n``, which generally disagrees with the Student-t
    distribution's true fourth moment ``3 * dof**2 / ((dof-2)*(dof-4))``
    (itself only defined for ``dof > 4``) -- see
    ``tests/unit/test_cubature_points.py::TestStudentT`` for the verified
    numeric gap. As ``dof -> inf`` the Student-t distribution converges to
    N(0, I) and ``nuConst -> sqrt(n)``, recovering
    ``ckf_spherical_cubature_points`` exactly.

    Examples
    --------
    >>> pts, w = student_t_cubature_points(3, 6.0)
    >>> pts.shape
    (6, 3)
    >>> round(float(w.sum()), 12)
    1.0
    >>> nu_const_sq = 6.0 * 3 / (6.0 - 2)
    >>> round(float(np.sum(w * pts[:, 0] ** 2)), 10) == round(nu_const_sq / 3, 10)
    True

    References
    ----------
    .. [1] Y. Huang, Y. Zhang, N. Li, S. M. Naqvi, and J. Chambers, "A
       robust Student's t based cubature filter," in Proc. 19th Int. Conf.
       on Information Fusion, Heidelberg, Germany, 5-8 Jul. 2016.
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if not dof > 2:
        raise ValueError(f"dof must be > 2, got {dof}")

    nu_const = np.sqrt(dof * n / (dof - 2.0))

    points = np.zeros((2 * n, n), dtype=np.float64)
    for i in range(n):
        points[2 * i, i] = nu_const
        points[2 * i + 1, i] = -nu_const

    weights = np.full(2 * n, 1.0 / (2 * n), dtype=np.float64)

    return points, weights
