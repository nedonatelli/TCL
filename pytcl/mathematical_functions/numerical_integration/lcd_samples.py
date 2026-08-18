"""Gaussian LCD samples: objective, analytic gradients, and optimizer wrapper.

Faithful transcription of the modified Cramer-von Mises (CvM) objective and
its four analytic gradient routines from the MATLAB Tracker Component
Library's ``GaussianLCDSamples.m`` (commit 593ce51,
Mathematical_Functions/Numerical_Integration/Cubature_Points/Gaussian_Weight/),
per the hybrid design spec (docs/superpowers/specs/2026-08-16-lcd-samples-design.md):
the math is ported as ordinary Python/NumPy; the optimizer (MATLAB calls a
MEX build of the third-party liblbfgs library, not MATLAB code) is
deliberately NOT ported and is instead wrapped as
``scipy.optimize.minimize(method="L-BFGS-B")`` in the public
:func:`gaussian_lcd_samples` below.

Everything except :func:`gaussian_lcd_samples` is private. The objective
operates on the symmetric half-sample parameterization: ``s`` is the
``(num_dim, num_samples // 2)`` matrix of free points; the full sample set is
``[s, -s]`` (plus a zero point when ``num_samples`` is odd). The "even"/"odd"
split throughout refers to the parity of ``num_samples`` (MATLAB's
``isEven = mod(numSamples,2)==0``; the MATLAB subfunction comments say
"dimensionality" but the dispatch is on sample count). The odd case differs
because of the extra fixed point at the origin.

Flattening convention: stacked vectors use Fortran (column-major) order,
matching MATLAB's ``s(:)`` / ``reshape`` semantics.

The constant terms ``computeD1`` and ``computeDo2ContTerm`` never enter the
optimized cost or its gradient; they are transcribed here because
:func:`gaussian_lcd_samples` adds them back internally to compute the exit
condition, matching what ``GaussianLCDSamples.m`` reports as ``CvMDistMin``
(the value itself is not part of this module's public return contract --
see that function's docstring).

MATLAB's ``integral(...,'AbsTol',1e-14,'RelTol',1e-14)`` calls are mirrored
with ``scipy.integrate.quad(..., epsabs=1e-14, epsrel=1e-14)``. QUADPACK
cannot certify 1e-14 relative error near machine precision and raises an
``IntegrationWarning``; measured achieved accuracy on the D2-type integrands
is ~4e-16 relative (values agree to <1e-12 absolute against looser-tolerance
runs; n=4, L=10, b_max=70, macOS/Apple Silicon, 2026-08-18), so the warning
is suppressed at the single call-site helper below.

Single-pass value/gradient merge (investigated, not applied): measured on
this campaign's grid (macOS/Apple Silicon, 2026-08-18, 20-rep mean per
cell), the D3/Do3 closed-form pieces -- the ones cheap enough to merge
safely, since they are pure vectorized NumPy with no adaptive quadrature --
account for well under 2% of one ``_lcd_objective`` call's wall time at
every grid cell (0.76%-1.57%, worst at (1,5)). The dominant cost (>=98%) is
the D2/Do2 quadrature: one ``scipy.integrate.quad`` call for the value plus
one *per free point* (``L``) for the gradient, i.e. the "L+1 quads per
call" this module was flagged for. Merging *that* into a true single pass
would require switching from ``scipy.integrate.quad`` to a vector-valued
quadrature routine (``scipy.integrate.quad_vec``) with different tolerance
semantics, and re-validating the merged result against the ~2e-16 accuracy
bar the existing ``quad``-based D2/D2Grad functions were reviewed to -- not
a straightforward refactor, and out of scope here. Left as-is and
documented per the task brief's explicit "if invasive, leave and document"
instruction; the D3/Do3 duplication was left alone too since merging it
would not move the measured number.

References
----------
J. Steinbring, M. Pander, and U. D. Hanebeck, "The smart sampling Kalman
filter with symmetric samples," arXiv:1506.03254, 10 Jun. 2015.

J. Steinbring and U. D. Hanebeck, "LRKF revisited: The smart sampling
Kalman filter (S2KF)," Journal of Advances in Information Fusion, vol. 9,
no. 2, pp. 106-123, Dec. 2014.

D. F. Crouse, "The Tracker Component Library," IEEE AESS Magazine, 2017.
"""

import warnings
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import minimize
from scipy.special import expi

from pytcl.core.exceptions import ConvergenceError, SingularMatrixError


def _ei(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exponential integral Ei with Ei(0) redefined to 0.

    Transcribed from ``GaussianLCDSamples.m`` lines 481-487 (``Ei``):
    ``val = -real(expint(-x)); val(~isfinite(val)) = 0``. For the x <= 0
    arguments used throughout this module, ``-real(expint(-x))`` equals the
    Cauchy principal value Ei(x), which is ``scipy.special.expi``. The zero
    redefinition kills the 0*Inf terms on the i == j diagonal of the D3
    double sums (derived from the D3 theorem's own limiting behavior, per
    the MATLAB header comment -- not an ad hoc patch).

    Parameters
    ----------
    x : ndarray
        Argument array (all call sites pass x <= 0).

    Returns
    -------
    ndarray
        Ei(x) elementwise, with non-finite values (x == 0) replaced by 0.
    """
    val = expi(x)
    return np.where(np.isfinite(val), val, 0.0)


def _integral(
    f: Callable[[float], float], b_max: float, abs_tol: float, rel_tol: float
) -> float:
    """Mirror MATLAB ``integral(f, 0, bMax, 'AbsTol', ..., 'RelTol', ...)``.

    Suppresses ``IntegrationWarning`` only: at the MATLAB-default 1e-14
    tolerances QUADPACK reports it cannot certify the request even though
    the achieved accuracy is at machine precision (see module docstring).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(f, 0.0, b_max, epsabs=abs_tol, epsrel=rel_tol, limit=200)
    return val


def _compute_d1(b_max: float, num_dim: int, abs_tol: float, rel_tol: float) -> float:
    """Sample-independent D1 constant of the CvM distance.

    Transcribed from ``GaussianLCDSamples.m`` lines 296-308 (``computeD1``),
    the D1 term on page 9 of Steinbring et al. 2015. Never enters the
    optimized cost or gradient; added back to the reported distance.

    Parameters
    ----------
    b_max : float
        Upper integration bound b_max from the reference.
    num_dim : int
        Dimensionality N of the samples.
    abs_tol, rel_tol : float
        Quadrature tolerances (MATLAB default 1e-14 for both).

    Returns
    -------
    float
        The D1 constant.
    """

    def f(b: float) -> float:
        return b * (b**2 / (1.0 + b**2)) ** (num_dim / 2.0)

    return _integral(f, b_max, abs_tol, rel_tol)


def _compute_de2(
    s: NDArray[np.floating], b_max: float, abs_tol: float, rel_tol: float
) -> float:
    """D2 term of the CvM distance, even sample count.

    Transcribed from ``GaussianLCDSamples.m`` lines 311-331 (``computeDe2``),
    the page-9 formula of Steinbring et al. 2015, by numerical integration
    over b in [0, b_max]. Depends on ``s`` only through the squared column
    norms (hence orthogonally invariant).

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.
    abs_tol, rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        The De2 term.
    """
    num_dim, length = s.shape
    col_sq = np.sum(s * s, axis=0)

    def f(b: float) -> float:
        b_sq_ratio = 2.0 * b**2 / (1.0 + 2.0 * b**2)
        return (
            (b / length)
            * b_sq_ratio ** (num_dim / 2.0)
            * np.sum(np.exp(-0.5 * col_sq / (1.0 + 2.0 * b**2)))
        )

    return _integral(f, b_max, abs_tol, rel_tol)


def _compute_de3(s: NDArray[np.floating], b_max: float) -> float:
    """D3 term of the CvM distance, even sample count (closed form).

    Transcribed from ``GaussianLCDSamples.m`` lines 334-358 (``computeDe3``),
    Theorem 3.1 of Steinbring et al. 2015: an O(L^2) pairwise sum over
    ``||s_i - s_j||^2`` and ``||s_i + s_j||^2`` using the exponential
    integral (vectorized here; the MATLAB double loop sums identical terms).

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound of the reference (enters the closed form).

    Returns
    -------
    float
        The De3 term.
    """
    length = s.shape[1]

    diff = s[:, :, None] - s[:, None, :]
    total = s[:, :, None] + s[:, None, :]
    diff_sq = np.sum(diff * diff, axis=0)
    sum_sq = np.sum(total * total, axis=0)

    arg1 = -0.5 * diff_sq / (2.0 * b_max**2)
    arg2 = -0.5 * sum_sq / (2.0 * b_max**2)

    term1 = (b_max**2 / 2.0) * (np.exp(arg1) + np.exp(arg2))
    term2 = (1.0 / 8.0) * (diff_sq * _ei(arg1) + sum_sq * _ei(arg2))

    return (2.0 / (2.0 * length) ** 2) * float(np.sum(term1 + term2))


def _compute_do2_simp(
    s: NDArray[np.floating], b_max: float, abs_tol: float, rel_tol: float
) -> float:
    """Sample-dependent D2 term, odd sample count.

    Transcribed from ``GaussianLCDSamples.m`` lines 360-368
    (``computeDo2Simp``), the page-10 formula: ``(2L/(2L+1)) * De2``. The
    constant part (``_compute_do2_cont_term``) is omitted from the
    optimized cost.

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.
    abs_tol, rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        The sample-dependent part of the Do2 term.
    """
    length = s.shape[1]
    return (2.0 * length / (2.0 * length + 1.0)) * _compute_de2(
        s, b_max, abs_tol, rel_tol
    )


def _compute_do2_cont_term(
    num_dim: int, num_half: int, b_max: float, abs_tol: float, rel_tol: float
) -> float:
    """Constant part of the D2 term, odd sample count.

    Transcribed from ``GaussianLCDSamples.m`` lines 370-385
    (``computeDo2ContTerm``), the page-10 formula. Sample-independent:
    never enters the optimized cost or gradient; subtracted (times 2) from
    the reported distance by the caller, matching the MATLAB main function
    (line 202).

    Parameters
    ----------
    num_dim : int
        Dimensionality N of the samples.
    num_half : int
        Number of free points L (= ``num_samples // 2``).
    b_max : float
        Upper integration bound.
    abs_tol, rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        The constant part of the Do2 term.
    """

    def f(b: float) -> float:
        b_sq_ratio = 2.0 * b**2 / (1.0 + 2.0 * b**2)
        return (b / (2.0 * num_half + 1.0)) * b_sq_ratio ** (num_dim / 2.0)

    return _integral(f, b_max, abs_tol, rel_tol)


def _compute_do3(s: NDArray[np.floating], b_max: float) -> float:
    """D3 term of the CvM distance, odd sample count.

    Transcribed from ``GaussianLCDSamples.m`` lines 387-406 (``computeDo3``),
    Theorem 3.2 of Steinbring et al. 2015: a rescaled De3 plus a constant
    plus a single sum over the squared column norms (the extra fixed origin
    point interacting with each mirrored pair).

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.

    Returns
    -------
    float
        The Do3 term.
    """
    length = s.shape[1]
    two_l1_sq = (2.0 * length + 1.0) ** 2

    term1 = ((2.0 * length) ** 2 / two_l1_sq) * _compute_de3(s, b_max)
    term2 = b_max**2 / (2.0 * two_l1_sq)

    si_sq = np.sum(s * s, axis=0)
    arg1 = -0.5 * si_sq / (2.0 * b_max**2)
    term3 = float(
        np.sum((b_max**2 / 2.0) * np.exp(arg1) + (1.0 / 8.0) * si_sq * _ei(arg1))
    ) * (4.0 / two_l1_sq)

    return term1 + term2 + term3


def _compute_de2_grad(
    s: NDArray[np.floating], b_max: float, abs_tol: float, rel_tol: float
) -> NDArray[np.floating]:
    """Analytic gradient of ``_compute_de2`` w.r.t. every entry of ``s``.

    Transcribed from ``GaussianLCDSamples.m`` lines 408-436
    (``computeDe2Grad``), the formula at the bottom of page 10 of
    Steinbring et al. 2015: one scalar quadrature per point i, then
    ``grad_i = -(s_i / (2L)) * intVal_i``.

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.
    abs_tol, rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    ndarray
        Gradient, same shape as ``s``.
    """
    num_dim, length = s.shape
    col_sq = np.sum(s * s, axis=0)

    grad = np.zeros_like(s)
    for i in range(length):
        si_sq = col_sq[i]

        def f(b: float) -> float:
            denom = 1.0 + 2.0 * b**2
            return (
                (2.0 * b / denom)
                * (2.0 * b**2 / denom) ** (num_dim / 2.0)
                * np.exp(-0.5 * si_sq / denom)
            )

        int_val = _integral(f, b_max, abs_tol, rel_tol)
        grad[:, i] = -(s[:, i] / (2.0 * length)) * int_val

    return grad


def _compute_de3_grad(s: NDArray[np.floating], b_max: float) -> NDArray[np.floating]:
    """Analytic gradient of ``_compute_de3`` w.r.t. every entry of ``s``.

    Transcribed from ``GaussianLCDSamples.m`` lines 438-459
    (``computeDe3Grad``), Theorem 3.3 of Steinbring et al. 2015:
    ``grad_i = (1/(2L)^2) * sum_j [ (s_i - s_j) Ei(arg_diff) +
    (s_i + s_j) Ei(arg_sum) ]`` (vectorized here; the i == j diff term
    vanishes identically as 0 * Ei(0) = 0).

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.

    Returns
    -------
    ndarray
        Gradient, same shape as ``s``.
    """
    length = s.shape[1]

    diff = s[:, :, None] - s[:, None, :]
    total = s[:, :, None] + s[:, None, :]
    diff_sq = np.sum(diff * diff, axis=0)
    sum_sq = np.sum(total * total, axis=0)

    ei_diff = _ei(-0.5 * diff_sq / (2.0 * b_max**2))
    ei_sum = _ei(-0.5 * sum_sq / (2.0 * b_max**2))

    grad = np.sum(diff * ei_diff[None, :, :] + total * ei_sum[None, :, :], axis=2)
    return grad / (2.0 * length) ** 2


def _compute_do2_grad(
    s: NDArray[np.floating], b_max: float, abs_tol: float, rel_tol: float
) -> NDArray[np.floating]:
    """Analytic gradient of ``_compute_do2_simp`` w.r.t. every entry of ``s``.

    Transcribed from ``GaussianLCDSamples.m`` lines 461-469
    (``computeDo2Grad``), the page-11 formula: ``(2L/(2L+1)) * De2Grad``.

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.
    abs_tol, rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    ndarray
        Gradient, same shape as ``s``.
    """
    length = s.shape[1]
    return (2.0 * length / (2.0 * length + 1.0)) * _compute_de2_grad(
        s, b_max, abs_tol, rel_tol
    )


def _compute_do3_grad(s: NDArray[np.floating], b_max: float) -> NDArray[np.floating]:
    """Analytic gradient of ``_compute_do3`` w.r.t. every entry of ``s``.

    Transcribed from ``GaussianLCDSamples.m`` lines 471-479
    (``computeDo3Grad``), Theorem 3.4 of Steinbring et al. 2015:
    ``((2L)^2/(2L+1)^2) * De3Grad + (s/(2L+1)^2) .* Ei(arg_col)`` with
    the Ei factor broadcast per column (MATLAB's ``bsxfun(@times, ...)``).

    Parameters
    ----------
    s : ndarray
        ``(num_dim, num_half_samples)`` free-point matrix.
    b_max : float
        Upper integration bound.

    Returns
    -------
    ndarray
        Gradient, same shape as ``s``.
    """
    length = s.shape[1]
    two_l1_sq = (2.0 * length + 1.0) ** 2

    col_sq = np.sum(s * s, axis=0)
    ei_col = _ei(-0.5 * col_sq / (2.0 * b_max**2))

    return ((2.0 * length) ** 2 / two_l1_sq) * _compute_de3_grad(s, b_max) + (
        s / two_l1_sq
    ) * ei_col[None, :]


def _mod_cvm_dist(
    s_flat: NDArray[np.floating],
    b_max: float,
    is_even: bool,
    abs_tol: float,
    rel_tol: float,
    s_dims: Tuple[int, int],
) -> float:
    """Modified CvM distance between the Gaussian LCD and the Dirac-mixture LCD.

    Transcribed from ``GaussianLCDSamples.m`` lines 246-272 (``modCvMDist``):
    ``D = -2*D2 + D3``, omitting the constant D1 term (and, for odd sample
    counts, the constant part of D2) since constants do not affect the
    minimizer.

    Parameters
    ----------
    s_flat : ndarray
        Stacked free-point vector, Fortran (column-major) order, matching
        MATLAB's ``s(:)``.
    b_max : float
        Upper integration bound.
    is_even : bool
        True when the total sample count is even (selects the De vs Do
        branch).
    abs_tol, rel_tol : float
        Quadrature tolerances.
    s_dims : tuple of int
        ``(num_dim, num_half_samples)`` shape of the free-point matrix.

    Returns
    -------
    float
        The modified CvM distance (constants omitted).
    """
    s = np.asarray(s_flat, dtype=np.float64).reshape(s_dims, order="F")

    if is_even:
        d2 = _compute_de2(s, b_max, abs_tol, rel_tol)
        d3 = _compute_de3(s, b_max)
    else:
        d2 = _compute_do2_simp(s, b_max, abs_tol, rel_tol)
        d3 = _compute_do3(s, b_max)

    return -2.0 * d2 + d3


def _mod_cvm_dist_grad(
    s_flat: NDArray[np.floating],
    b_max: float,
    is_even: bool,
    abs_tol: float,
    rel_tol: float,
    s_dims: Tuple[int, int],
) -> NDArray[np.floating]:
    """Gradient of ``_mod_cvm_dist`` w.r.t. every stacked entry of ``s_flat``.

    Transcribed from ``GaussianLCDSamples.m`` lines 274-294
    (``modCvMDistGrad``): ``-2 * D2Grad + D3Grad``, branch-selected on
    sample-count parity, restacked in the same (Fortran) order as the input.

    Parameters
    ----------
    s_flat : ndarray
        Stacked free-point vector, Fortran (column-major) order.
    b_max : float
        Upper integration bound.
    is_even : bool
        True when the total sample count is even.
    abs_tol, rel_tol : float
        Quadrature tolerances.
    s_dims : tuple of int
        ``(num_dim, num_half_samples)`` shape of the free-point matrix.

    Returns
    -------
    ndarray
        Stacked gradient, same shape as ``s_flat``.
    """
    s = np.asarray(s_flat, dtype=np.float64).reshape(s_dims, order="F")

    if is_even:
        grad = -2.0 * _compute_de2_grad(s, b_max, abs_tol, rel_tol)
        grad = grad + _compute_de3_grad(s, b_max)
    else:
        grad = -2.0 * _compute_do2_grad(s, b_max, abs_tol, rel_tol)
        grad = grad + _compute_do3_grad(s, b_max)

    return grad.flatten(order="F")


def _lcd_objective(
    s_flat: NDArray[np.floating],
    num_dim: int,
    num_samples: int,
    b_max: float = 70.0,
    abs_tol: float = 1e-14,
    rel_tol: float = 1e-14,
) -> Tuple[float, NDArray[np.floating]]:
    """Value and gradient of the modified CvM cost for the LCD optimization.

    Combined objective/gradient pair mirroring the MATLAB cost-function
    handle ``f = @(s) deal(modCvMDist(...), modCvMDistGrad(...))``
    (``GaussianLCDSamples.m`` line 192), with the same defaults the MATLAB
    main function applies: ``b_max = 70`` (line 132) and quadrature
    tolerances ``1e-14`` (lines 136-137).

    Parameters
    ----------
    s_flat : ndarray
        Stacked ``(num_dim * (num_samples // 2),)`` free-point vector in
        Fortran (column-major) order, matching MATLAB's ``sInit(:)``.
    num_dim : int
        Dimensionality of the cubature points.
    num_samples : int
        Total number of samples the caller will build as ``[s, -s]`` (plus
        a zero point when odd). Its parity selects the even/odd branch.
    b_max : float, optional
        Upper integration bound (default 70, the reference's suggestion for
        ``num_dim <= 1000``).
    abs_tol, rel_tol : float, optional
        Quadrature tolerances (default 1e-14, the MATLAB defaults).

    Returns
    -------
    value : float
        Modified CvM distance (constant D1 and odd-count D2 constant
        omitted, exactly as in the optimized MATLAB cost).
    grad : ndarray
        Analytic gradient w.r.t. ``s_flat``, same shape.
    """
    num_half = num_samples // 2
    is_even = num_samples % 2 == 0
    s_dims = (num_dim, num_half)

    value = _mod_cvm_dist(s_flat, b_max, is_even, abs_tol, rel_tol, s_dims)
    grad = _mod_cvm_dist_grad(s_flat, b_max, is_even, abs_tol, rel_tol, s_dims)
    return value, grad


def gaussian_lcd_samples(
    n: int,
    num_points: int,
    *,
    force_cov_match: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_iter: int = 1000,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Localized cumulative distribution (LCD) cubature points for N(0, I).

    Port of the MATLAB Tracker Component Library's ``GaussianLCDSamples.m``
    (commit 593ce51) entry point, per the hybrid design spec
    (docs/superpowers/specs/2026-08-16-lcd-samples-design.md): the CvM
    objective/gradient are a faithful transcription (this module's private
    functions); the optimizer is ``scipy.optimize.minimize(method=
    "L-BFGS-B", jac=True)`` in place of MATLAB's MEX-only ``liblbfgs`` call,
    since ``liblbfgs`` is a vendored third-party C library with no MATLAB
    source of its own to port fidelity against (spec Section 6).

    Points are generated as ``2*(num_points // 2)`` symmetric pairs
    ``+-s_i`` obtained by minimizing the modified Cramer-von Mises distance
    between the points' localized CDF and a standard normal's, plus one
    fixed point at the origin when `num_points` is odd (MATLAB lines
    212-218). All weights are uniform, ``1 / num_points`` (MATLAB line 222:
    ``w=1/numSamples``), summing to 1 -- the Gaussian-weight convention this
    module's sibling ``cubature_points`` functions use (region-measure
    weighting, where weights sum to a region's volume rather than 1, is a
    separate contract used only by ``region_cubature.py``; see that
    module's docstring).

    **Rotation-invariance caveat (read before comparing outputs across
    runs or against MATLAB).** For `n` >= 2 the CvM cost is provably
    invariant under any global orthogonal transform applied to every point
    simultaneously (design spec Section 3): ``O(n)`` is a continuous
    symmetry group of the objective, so a minimizer sits on a flat manifold
    of equally-optimal solutions, not an isolated point. Two calls that
    converge correctly -- including one call of this function versus a
    MATLAB ``GaussianLCDSamples`` run given the *same* starting matrix --
    generically land on *different* points of that manifold: same CvM cost,
    different raw coordinates, and this is expected, not a bug. Only `n`
    == 1 has a discrete symmetry group (``O(1) = {+1, -1}``) where raw
    coordinates (up to sign/permutation) are a meaningful comparison.

    **Seeding is NOT cross-compatible with MATLAB.** `rng` (default: a
    fresh ``numpy.random.default_rng()``, PCG64-backed per this repo's
    convention) mirrors MATLAB's ``sInit=randn(sDims)`` initialization
    scheme -- drawing a ``(n, num_points // 2)`` matrix of standard normal
    variates via ``rng.standard_normal`` -- but NumPy's Generator and
    MATLAB's ``randn`` use different underlying bit generators and
    uniform-to-Gaussian transforms. No seed value reproduces the same
    ``sInit`` matrix in both ecosystems; giving this function and a MATLAB
    call "the same seed" only means "the same kind of random start," not
    "the same numbers." A fixed `rng` does give bit-identical output across
    repeated Python calls (checked in the test suite).

    **L-BFGS-B option mapping (honest, not exact, parity with liblbfgs).**
    MATLAB's ``quasiNewtonLBFGS`` defaults (all left at default by
    ``GaussianLCDSamples.m``) come from a MEX wrapper around Naoaki
    Okazaki's ``liblbfgs`` (a C port of Nocedal's L-BFGS with a
    More-Thuente line search). ``scipy``'s L-BFGS-B uses a different
    Fortran implementation (Zhu/Byrd/Lu/Nocedal) with its own line search,
    so no option mapping can reproduce ``liblbfgs``'s exact step sequence;
    the mapping below matches intent, not mechanics:

    - ``numCorr=6`` (history size) -> ``maxcor=6`` (scipy's history-size
      option shares the same meaning and default value).
    - ``max_iterations=1000`` -> ``maxiter=max_iter`` (this function's
      parameter, default 1000, matching MATLAB's default exactly) and
      ``max_linesearch=20`` -> ``maxfun=max_iter*20``. liblbfgs's
      ``max_iterations`` bounds only outer iterations; scipy's L-BFGS-B
      additionally caps total function evaluations via ``maxfun``
      (default 15000, independent of ``maxiter``), which would silently
      cut optimization short before liblbfgs's own worst-case evaluation
      budget (``max_iterations * max_linesearch`` = 20000 at the
      defaults) is reached, so ``maxfun`` is raised to match that budget
      rather than left at scipy's unrelated default.
    - ``epsilon=1e-6`` (stop when the Euclidean gradient norm drops below
      ``epsilon * max(1, ||x||)``) -> ``gtol=1e-6`` (the ``minimize``
      option name; the underlying Fortran variable is called ``pgtol``,
      which is also the keyword the legacy, non-``minimize``
      ``scipy.optimize.fmin_l_bfgs_b`` entry point uses -- ``minimize``
      renames it to ``gtol``). This is a genuinely different criterion,
      not just a differently-named equivalent: scipy's ``gtol`` stops on
      the max-absolute-component of the (here unconstrained, so
      unprojected) gradient, with no scaling by the parameter norm. Both
      use the same numeric threshold as a reasonable value-level match;
      they are not the same stopping rule.
    - ``delta=0`` / ``past=0`` (liblbfgs's relative-function-decrease test
      explicitly disabled) -> ``ftol=0.0``. This one *is* a faithful
      mapping: scipy computes ``factr = ftol / eps`` internally and the
      underlying Fortran L-BFGS-B code treats ``factr=0`` as "suppress
      this termination test" (its own documented convention), mirroring
      liblbfgs's disabled delta-test exactly.
    - ``wolfe=0.9`` (curvature/Wolfe condition in the More-Thuente line
      search) and ``ftol=1e-6`` / ``xtol=1e-16`` / ``min_step`` /
      ``max_step`` (liblbfgs's own internal line-search parameters) have
      **no exposed equivalent** in scipy's L-BFGS-B ``minimize``
      interface -- its internal line search (``dcsrch``) hardcodes its
      own strong-Wolfe curvature parameter (conventionally 0.9,
      coincidentally the same value liblbfgs defaults to, but not
      user-settable through this wrapper) and is not configurable from
      Python. Not mapped; noted here so a future reader does not assume
      silent parity.

    Parameters
    ----------
    n : int
        Dimensionality of the cubature points, n >= 1.
    num_points : int
        Total number of points to generate, num_points >= 2. For a
        non-singular sample covariance when `force_cov_match` is True,
        MATLAB's own documented requirement is num_points >= 2*n (see
        Raises below).
    force_cov_match : bool, optional
        When True (the default -- MATLAB's default is the same True only
        when ``num_points >= 2*n``; this port always defaults True and
        raises instead of silently falling back to False), whiten the
        optimized points post-hoc by Cholesky factorization so their
        sample covariance is exactly the identity (to float64 rounding),
        correcting the base algorithm's tendency to underestimate the
        diagonal of the covariance (MATLAB lines 224-240). When False, the
        raw optimized (mirrored) points are returned with whatever sample
        covariance the CvM optimum happens to produce.
    rng : numpy.random.Generator, optional
        Generator used to draw the MATLAB-``randn``-equivalent
        initialization matrix (see the seeding caveat above). Default
        None constructs a fresh ``numpy.random.default_rng()``.
    max_iter : int, optional
        Maximum L-BFGS-B outer iterations, default 1000 (MATLAB's
        ``max_iterations`` default, see the option-mapping notes above).

    Returns
    -------
    points : ndarray
        Shape ``(num_points, n)``.
    weights : ndarray
        Shape ``(num_points,)``, every entry exactly ``1 / num_points``
        (uniform weighting; MATLAB line 222), summing to 1.

    Raises
    ------
    ValueError
        If `n` < 1 or `num_points` < 2.
    SingularMatrixError
        If `force_cov_match` is True and the optimized points' sample
        covariance is singular (MATLAB's ``exitCode=-2`` case, lines
        230-235) -- generically occurs when ``num_points < 2*n``, since a
        symmetric point set spanning fewer than `n` free directions cannot
        have a full-rank covariance.
    ConvergenceError
        If L-BFGS-B does not report success within `max_iter` iterations.

    Notes
    -----
    Measured on this campaign's validation grid
    ``{(1,5),(2,10),(2,20),(3,15),(4,20)}`` (macOS/Apple Silicon,
    2026-08-18): every case converges (``result.success`` True) with the
    optimized objective strictly below its value at the random
    initialization, and, under `force_cov_match` (default), the returned
    points' sample mean matches 0 and sample covariance matches the
    identity to float64 rounding (~1e-14 or tighter absolute, exact test
    tolerances recorded in ``tests/unit/test_lcd_samples.py``). No claim is
    made about convergence quality outside this grid.

    Examples
    --------
    >>> pts, w = gaussian_lcd_samples(2, 10, rng=np.random.default_rng(0))
    >>> pts.shape
    (10, 2)
    >>> round(float(w.sum()), 12)
    1.0
    >>> bool(np.allclose(pts.mean(axis=0), 0.0, atol=1e-10))
    True
    """
    if n < 1:
        raise ValueError(f"dimension n must be >= 1, got {n}")
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")

    num_half = num_points // 2
    is_even = num_points % 2 == 0
    s_dims = (n, num_half)

    generator = rng if rng is not None else np.random.default_rng()
    s_init = generator.standard_normal(s_dims)
    s_init_flat = s_init.flatten(order="F")

    result = minimize(
        _lcd_objective,
        s_init_flat,
        args=(n, num_points),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxcor": 6,
            "maxiter": max_iter,
            "maxfun": max_iter * 20,
            "gtol": 1e-6,
            "ftol": 0.0,
        },
    )

    if not result.success:
        raise ConvergenceError(
            "L-BFGS-B did not converge for the Gaussian LCD sample "
            f"optimization (n={n}, num_points={num_points}): {result.message}",
            algorithm="L-BFGS-B",
            iterations=int(result.nit),
            max_iterations=max_iter,
            residual=float(np.max(np.abs(result.jac))),
            tolerance=1e-6,
        )

    s_min = result.x.reshape(s_dims, order="F")

    if is_even:
        xi = np.hstack([s_min, -s_min])
    else:
        xi = np.hstack([np.zeros((n, 1)), s_min, -s_min])

    w_scalar = 1.0 / num_points

    if force_cov_match:
        cov = w_scalar * (xi @ xi.T)
        if np.linalg.matrix_rank(cov) != n:
            raise SingularMatrixError(
                "sample covariance of the optimized LCD points is singular; "
                "cannot force a covariance match with symmetric samples "
                f"(n={n}, num_points={num_points}; MATLAB requires "
                f"num_points >= 2*n = {2 * n} for this to be non-singular)",
                matrix_name="R",
            )
        cov_inv = np.linalg.inv(cov)
        s_r_inv = np.linalg.cholesky(cov_inv)
        xi = s_r_inv.T @ xi

    points = xi.T
    weights = np.full(num_points, w_scalar)
    return points, weights
