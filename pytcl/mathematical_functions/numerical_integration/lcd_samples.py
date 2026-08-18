"""Gaussian LCD sample objective and analytic gradients (private, no optimizer).

Faithful transcription of the modified Cramer-von Mises (CvM) objective and
its four analytic gradient routines from the MATLAB Tracker Component
Library's ``GaussianLCDSamples.m`` (commit 593ce51,
Mathematical_Functions/Numerical_Integration/Cubature_Points/Gaussian_Weight/),
per the hybrid design spec (docs/superpowers/specs/2026-08-16-lcd-samples-design.md):
the math is ported here; the optimizer (MATLAB calls a MEX build of the
third-party liblbfgs library, not MATLAB code) is deliberately NOT ported and
lands separately as a ``scipy.optimize.minimize(method="L-BFGS-B")`` wrapper
with the public API.

Everything in this module is private. The objective operates on the symmetric
half-sample parameterization: ``s`` is the ``(num_dim, num_samples // 2)``
matrix of free points; the full sample set is ``[s, -s]`` (plus a zero point
when ``num_samples`` is odd). The "even"/"odd" split throughout refers to the
parity of ``num_samples`` (MATLAB's ``isEven = mod(numSamples,2)==0``; the
MATLAB subfunction comments say "dimensionality" but the dispatch is on
sample count). The odd case differs because of the extra fixed point at the
origin.

Flattening convention: stacked vectors use Fortran (column-major) order,
matching MATLAB's ``s(:)`` / ``reshape`` semantics.

The constant terms ``computeD1`` and ``computeDo2ContTerm`` never enter the
optimized cost or its gradient; they are transcribed here because the public
wrapper (next task) adds them back to report the full CvM distance.

MATLAB's ``integral(...,'AbsTol',1e-14,'RelTol',1e-14)`` calls are mirrored
with ``scipy.integrate.quad(..., epsabs=1e-14, epsrel=1e-14)``. QUADPACK
cannot certify 1e-14 relative error near machine precision and raises an
``IntegrationWarning``; measured achieved accuracy on the D2-type integrands
is ~4e-16 relative (values agree to <1e-12 absolute against looser-tolerance
runs; n=4, L=10, b_max=70, macOS/Apple Silicon, 2026-08-18), so the warning
is suppressed at the single call-site helper below.

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
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.special import expi


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
