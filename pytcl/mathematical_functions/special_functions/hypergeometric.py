"""
Hypergeometric functions.

This module provides hypergeometric functions commonly used in
mathematical physics, probability theory, and special function evaluation.

Performance
-----------
The generalized hypergeometric function uses Numba JIT compilation for
the series summation loop, providing significant speedup for the general
case (p > 2 or q > 1).
"""

import math
import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import scipy.special as sp
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit(cache=True)
def _hypergeometric_series(
    a: np.ndarray[Any, Any],
    b: np.ndarray[Any, Any],
    z: np.ndarray[Any, Any],
    max_terms: int,
    tol: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
    """
    Numba-optimized series summation for generalized hypergeometric function.

    Parameters
    ----------
    a : ndarray
        Numerator parameters (1D array).
    b : ndarray
        Denominator parameters (1D array).
    z : ndarray
        Argument values (1D array).
    max_terms : int
        Maximum number of series terms.
    tol : float
        Convergence tolerance.

    Returns
    -------
    result : ndarray
        Computed pFq values for each z.
    max_term : ndarray
        Largest term magnitude seen for each z; the rounding error of
        ``result`` scales with it, not with ``result``.
    converged : bool
        False when max_terms was exhausted before the tolerance held.
    """
    n_z = len(z)
    p = len(a)
    q = len(b)

    result = np.ones(n_z, dtype=np.float64)
    term = np.ones(n_z, dtype=np.float64)
    max_term = np.ones(n_z, dtype=np.float64)

    for k in range(1, max_terms):
        # Compute numerator product: prod(a_i + k - 1)
        num_factor = 1.0
        for i in range(p):
            num_factor *= a[i] + k - 1

        # Compute denominator product: prod(b_i + k - 1) * k
        den_factor = float(k)
        for i in range(q):
            den_factor *= b[i] + k - 1

        # Update term and result for each z value
        ratio = num_factor / den_factor
        converged = True
        for j in range(n_z):
            term[j] = term[j] * z[j] * ratio
            result[j] += term[j]

            # Relative to the largest term, not the partial sum: an
            # alternating series cancels the sum far below its terms.
            if np.abs(term[j]) > max_term[j]:
                max_term[j] = np.abs(term[j])
            if np.abs(term[j]) >= tol * max_term[j]:
                converged = False

        if converged:
            break

    return result, max_term, converged


def hyp0f1(
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Confluent hypergeometric limit function 0F1(b; z).

    The function 0F1(b; z) is defined by the series:
    0F1(b; z) = sum_{k=0}^inf z^k / ((b)_k * k!)

    where (b)_k is the Pochhammer symbol (rising factorial).

    Parameters
    ----------
    b : array_like
        Numerator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function.

    Returns
    -------
    F : ndarray
        Values of 0F1(b; z).

    Notes
    -----
    Related to Bessel functions:
    J_n(x) = (x/2)^n / Gamma(n+1) * 0F1(n+1; -x^2/4)
    I_n(x) = (x/2)^n / Gamma(n+1) * 0F1(n+1; x^2/4)

    Examples
    --------
    >>> float(hyp0f1(1, 0))  # 0F1(1; 0) = 1
    1.0
    >>> round(float(hyp0f1(1, 1)), 6)
    2.279585

    References
    ----------
    - NIST Digital Library of Mathematical Functions, Chapter 16.
    """
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyp0f1(b, z), dtype=np.float64)


def hyp1f1(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Confluent hypergeometric function 1F1(a; b; z) (Kummer's function M).

    The function 1F1(a; b; z) is defined by the series:
    1F1(a; b; z) = sum_{k=0}^inf (a)_k * z^k / ((b)_k * k!)

    Parameters
    ----------
    a : array_like
        Numerator parameter.
    b : array_like
        Denominator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function.

    Returns
    -------
    F : ndarray
        Values of 1F1(a; b; z).

    Notes
    -----
    Also known as Kummer's function M(a, b, z).

    Special cases:
    - 1F1(0; b; z) = 1
    - 1F1(a; a; z) = exp(z)
    - 1F1(1; 2; 2z) = sinh(z) * exp(z) / z

    Related to incomplete gamma:
    gammainc(a, z) = z^a * exp(-z) * 1F1(1; 1+a; z) / (a * Gamma(a))

    Examples
    --------
    >>> round(float(hyp1f1(1, 1, 1)), 6)  # exp(1)
    2.718282
    >>> round(float(hyp1f1(0.5, 1.5, -1)), 6)  # erf(1) * sqrt(pi) / 2
    0.746824

    References
    ----------
    - Abramowitz & Stegun, "Handbook of Mathematical Functions", Ch. 13.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyp1f1(a, b, z), dtype=np.float64)


def hyp2f1(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Gauss hypergeometric function 2F1(a, b; c; z).

    The function 2F1(a, b; c; z) is defined by the series::

        2F1(a, b; c; z) = sum_{k=0}^inf (a)_k * (b)_k * z^k / ((c)_k * k!)

    converging for ``|z| < 1``.

    Parameters
    ----------
    a : array_like
        First numerator parameter.
    b : array_like
        Second numerator parameter.
    c : array_like
        Denominator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function. For ``|z| >= 1``, analytic continuation
        is used.

    Returns
    -------
    F : ndarray
        Values of 2F1(a, b; c; z).

    Notes
    -----
    Many elementary and special functions are special cases:
    - (1-z)^(-a) = 2F1(a, b; b; z)
    - log(1+z)/z = 2F1(1, 1; 2; -z)
    - arcsin(z)/z = 2F1(1/2, 1/2; 3/2; z^2)
    - Complete elliptic integrals K(k) and E(k)

    Examples
    --------
    >>> round(float(hyp2f1(1, 1, 2, 0.5)), 6)  # -log(1-0.5)/0.5 = 2*log(2)
    1.386294
    >>> round(float(hyp2f1(0.5, 0.5, 1.5, 0.25)), 6)  # arcsin(0.5)/0.5 = pi/3
    1.047198

    References
    ----------
    - NIST DLMF, Chapter 15.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyp2f1(a, b, c, z), dtype=np.float64)


def hyperu(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Confluent hypergeometric function U(a, b, z) (Tricomi function).

    The function U(a, b, z) is defined as::

        U(a, b, z) = Gamma(1-b)/Gamma(a-b+1) * 1F1(a; b; z)
                     + Gamma(b-1)/Gamma(a) * z^(1-b) * 1F1(a-b+1; 2-b; z)

    Parameters
    ----------
    a : array_like
        First parameter.
    b : array_like
        Second parameter.
    z : array_like
        Argument of the function (must be positive for real result).

    Returns
    -------
    U : ndarray
        Values of U(a, b, z).

    Notes
    -----
    Also known as Tricomi's function or Kummer's function of the second kind.

    Asymptotic behavior for large z:
    U(a, b, z) ~ z^(-a) as z -> inf

    Examples
    --------
    >>> round(float(hyperu(1, 1, 1)), 6)
    0.596347
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyperu(a, b, z), dtype=np.float64)


def hyp1f1_regularized(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Regularized confluent hypergeometric function 1F1(a; b; z) / Gamma(b).

    This is useful when b may be near a non-positive integer.

    Parameters
    ----------
    a : array_like
        Numerator parameter.
    b : array_like
        Denominator parameter.
    z : array_like
        Argument of the function.

    Returns
    -------
    F : ndarray
        Values of 1F1(a; b; z) / Gamma(b).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.mathematical_functions.special_functions import hyp1f1_regularized
    >>> # Regularized form avoids overflow for problematic b values
    >>> a, b, z = 0.5, 1.5, 1.0
    >>> f_reg = hyp1f1_regularized(a, b, z)
    >>> # Should give finite, non-overflowing result
    >>> np.isfinite(f_reg)
    True
    >>> # Compare to regular hypergeometric computation
    >>> from pytcl.mathematical_functions.special_functions import hyp1f1
    >>> import scipy.special as sp
    >>> f_normal = hyp1f1(a, b, z) / sp.gamma(b)
    >>> np.allclose(f_reg, f_normal)
    True

    Notes
    -----
    This function remains finite even when b is a non-positive integer,
    unlike the standard 1F1. For b = -m (m = 0, 1, ...) the limit is

        (a)_{m+1} z^{m+1} / (m+1)! * 1F1(a+m+1; m+2; z)

    so that ``hyp1f1_regularized(1, 0, 1)`` is e (it was nan, from
    ``hyp1f1/gamma`` evaluating inf/inf).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = sp.hyp1f1(a, b, z) / sp.gamma(b)
        m1 = 1.0 - b
        limit = (
            sp.poch(a, m1)
            * z**m1
            * sp.rgamma(m1 + 1.0)
            * sp.hyp1f1(a + m1, m1 + 1.0, z)
        )
    pole = (b <= 0.0) & (b == np.round(b))
    return np.where(pole, limit, result)[()]


def pochhammer(
    a: ArrayLike,
    n: ArrayLike,
) -> NDArray[np.floating]:
    """
    Pochhammer symbol (rising factorial) (a)_n.

    The Pochhammer symbol is defined as:
    (a)_n = a * (a+1) * (a+2) * ... * (a+n-1) = Gamma(a+n) / Gamma(a)

    Parameters
    ----------
    a : array_like
        Base value.
    n : array_like
        Number of terms (can be non-integer for generalization).

    Returns
    -------
    p : ndarray
        Values of (a)_n.

    Notes
    -----
    Special cases:
    - (a)_0 = 1
    - (1)_n = n!
    - (a)_1 = a

    Examples
    --------
    >>> float(pochhammer(1, 5))  # 5!
    120.0
    >>> float(pochhammer(3, 4))  # 3*4*5*6
    360.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    return np.asarray(sp.poch(a, n), dtype=np.float64)


def falling_factorial(
    a: ArrayLike,
    n: ArrayLike,
) -> NDArray[np.floating]:
    """
    Falling factorial (a)_n (Pochhammer symbol variant).

    The falling factorial is defined as:
    (a)_n = a * (a-1) * (a-2) * ... * (a-n+1)

    Parameters
    ----------
    a : array_like
        Base value.
    n : array_like
        Number of terms.

    Returns
    -------
    f : ndarray
        Values of the falling factorial.

    Notes
    -----
    Related to rising factorial:
    (a)_n (falling) = (-1)^n * (-a)_n (rising)

    Examples
    --------
    >>> float(falling_factorial(5, 3))  # 5*4*3
    60.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    # (a)_n falling = a!/(a-n)! = Gamma(a+1)/Gamma(a-n+1)
    return np.asarray(sp.poch(a - n + 1, n), dtype=np.float64)


_EPS = float(np.finfo(np.float64).eps)
_CANCELLATION_WARN = 1e-8
_EXACT_MAX_TERMS = 500


def _exact_polynomial_value(a: Any, b: Any, z: float) -> float:
    """Evaluate a terminating pFq in exact rational arithmetic.

    Floats are exact rationals, so the only rounding is the final
    conversion. Returns NaN when an input is not finite, a denominator
    parameter hits a pole, or the series has more than
    ``_EXACT_MAX_TERMS`` terms (the rational arithmetic is superlinear in
    the term count: 0.1 s at 500 terms, 4.6 s at 2000).
    """
    if not (np.isfinite(z) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    n = int(min(-ai for ai in a if ai <= 0 and float(ai).is_integer()))
    if n > _EXACT_MAX_TERMS:
        return float("nan")
    zf = Fraction(float(z))
    term = Fraction(1)
    total = Fraction(1)
    for k in range(1, n + 1):
        den = Fraction(k)
        for ai in a:
            term *= Fraction(float(ai)) + (k - 1)
        for bi in b:
            den *= Fraction(float(bi)) + (k - 1)
        if den == 0:
            return float("nan")
        term = term * zf / den
        total += term
    return float(total)


def _flag_cancellation(
    result: NDArray[np.floating],
    max_term: NDArray[np.floating],
    a: Any,
    b: Any,
    z: NDArray[np.floating],
    terminates: bool,
) -> NDArray[np.floating]:
    """Blank out sums whose rounding error swamps the value.

    A terminating series with no surviving digits is re-evaluated in exact
    arithmetic instead, so an exact zero (``1 - z`` at ``z = 1``) stays 0
    while a non-terminating sum that merely rounds to 0.0 still becomes NaN.

    Summing terms as large as ``max_term`` leaves an absolute error of about
    ``eps * max_term``. When that reaches ``abs(result)`` no significant
    digit survives, so the entry becomes NaN; when it exceeds
    ``_CANCELLATION_WARN`` of the result, the value is kept with a warning.
    Entries with a non-finite ``z`` or parameter are NaN without a warning:
    their NaN comes from the input, not from cancellation.
    """
    bad = ~np.isfinite(z)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        bad[:] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(bad, 0.0, _EPS * max_term / np.abs(result))
    result = np.where(bad, np.nan, result)
    lost = ~(loss < 1.0)
    if terminates and np.any(lost):
        result = result.copy()
        for j in np.flatnonzero(lost):
            result[j] = _exact_polynomial_value(a, b, z[j])
        recovered = lost & ~np.isnan(result)
        lost = lost & ~recovered
        loss = np.where(recovered, 0.0, loss)
    degraded = ~lost & (loss > _CANCELLATION_WARN)
    if np.any(lost):
        warnings.warn(
            "generalized_hypergeometric: catastrophic cancellation; the "
            "largest series term exceeds the sum by more than 1/eps, so no "
            "significant digits remain. Returning NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
        result = np.where(lost, np.nan, result)
    if np.any(degraded):
        warnings.warn(
            "generalized_hypergeometric: cancellation in the series sum "
            "has cost more than half of the available digits.",
            RuntimeWarning,
            stacklevel=3,
        )
    return result


def _negative_factors(x: float, n: int) -> int:
    """Number of negative factors in the Pochhammer symbol (x)_n."""
    return 0 if x >= 0 else min(n, math.ceil(-x))


def _pfq_limit_at_infinity(
    a: NDArray[np.floating], b: NDArray[np.floating], z: float
) -> float:
    """Limit of pFq(a; b; z) as z -> +inf or -inf (DLMF 16.11).

    The caller has already rejected the divergent non-terminating cases
    (p = q + 1 other than 2F1, and p > q + 1).
    """
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    p, q = len(a), len(b)
    degrees = [int(-ai) for ai in a if ai <= 0 and float(ai).is_integer()]
    if degrees:
        n = min(degrees)
        if any(-n < bj <= 0 and float(bj).is_integer() for bj in b):
            return np.nan
        if n == 0:
            return 1.0
        negatives = sum(_negative_factors(x, n) for x in (*a, *b))
        if z < 0:
            negatives += n
        return -np.inf if negatives % 2 else np.inf
    if not (np.all(a > 0) and np.all(b > 0)):
        return np.nan
    if (p, q) == (2, 1):
        return np.nan if z > 0 else 0.0
    if z > 0:
        return np.inf
    if p == q:
        return 0.0
    if p == q - 1 and a.sum() - b.sum() + 0.5 < 0:
        return 0.0
    return np.nan


def generalized_hypergeometric(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
    max_terms: int = 500,
    tol: float = 1e-15,
) -> NDArray[np.floating]:
    """
    Generalized hypergeometric function pFq(a; b; z).

    Computes the generalized hypergeometric function with p numerator
    and q denominator parameters.

    Parameters
    ----------
    a : array_like
        Numerator parameters (1D array of length p).
    b : array_like
        Denominator parameters (1D array of length q).
    z : array_like
        Argument of the function.
    max_terms : int, optional
        Maximum number of series terms. Default is 500.
    tol : float, optional
        Tolerance for series convergence. Default is 1e-15.

    Returns
    -------
    F : ndarray
        Values of pFq(a; b; z).

    Notes
    -----
    The series converges for::

        - p <= q: all z
        - p = q + 1: |z| < 1
        - p > q + 1: diverges except for polynomial cases

    Uses Numba JIT compilation for the general case (p > 2 or q > 1),
    providing 5-10x speedup over pure Python loops.

    Outside the convergence domain (p = q + 1 with ``|z| >= 1``, or
    p > q + 1 with ``z != 0``) a ``ValueError`` is raised unless the
    series terminates (some ``a[i]`` a non-positive integer). A
    ``RuntimeWarning`` is emitted when ``max_terms`` is exhausted
    before the tolerance holds.

    Convergence is judged against the largest term seen, not the partial
    sum. An alternating series (negative ``z``) sums to a value far below
    its terms, and the old test against the partial sum stopped on a term
    still enormous relative to the true value (``0F0(-40)`` returned
    -2.23 against ``exp(-40) = 4.2e-18``; ``3F3([1,2,3],[4,5,6],-100)``
    returned 3.2e12 against 0.1993). The rounding error of the sum is about
    ``eps`` times the largest term: when that reaches the magnitude of the
    sum (largest term more than ``1/eps`` times the sum) the result is NaN
    with a ``RuntimeWarning``, and a ``RuntimeWarning`` is emitted with the
    value kept once more than half the digits are lost. A terminating
    series (some ``a[i]`` a non-positive integer) that loses every digit is
    re-evaluated exactly in rational arithmetic instead (Laguerre
    ``L_80(20)`` gives -1986.4165), so the NaN applies only to
    non-terminating series and to terminating ones longer than 500 terms.
    A NaN ``z`` or a non-finite parameter gives NaN without a cancellation
    warning. ``0F0(z)`` is evaluated as ``exp(z)``.

    At ``z = +inf`` or ``z = -inf`` the limit is returned where one exists
    (DLMF 16.11). Parameters are real (the function casts them to
    float64). With ``n`` the smallest ``abs(a[i])`` over the non-positive
    integer ``a[i]`` and ``c_n = prod (a_i)_n / (prod (b_j)_n * n!)`` the
    leading coefficient of the terminating polynomial:

    ======================================  ===================  ===========================================
    Case                                    ``z = +inf``         ``z = -inf``
    ======================================  ===================  ===========================================
    terminating                             ``sign(c_n) * inf``  ``sign(c_n) * (-1)**n * inf``
    terminating, ``n = 0``                  1.0                  1.0
    non-terminating, ``p == q``             ``+inf``             0.0
    non-terminating, ``p == q - 1``         ``+inf``             0.0 if ``sum(a) - sum(b) + 1/2 < 0``, else NaN (oscillates)
    non-terminating, ``p <= q - 2``         ``+inf``             NaN (oscillating growth)
    non-terminating, ``(p, q) == (2, 1)``   NaN (branch cut)     0.0
    non-terminating, other ``p >= q + 1``   ``ValueError``       ``ValueError``
    anything else                           NaN                  NaN
    ======================================  ===================  ===========================================

    The non-terminating rows other than the ``ValueError`` one require
    every ``a[i] > 0`` and every ``b[j] > 0``; a negative non-integer
    ``a[i]``, a non-positive ``b[j]``, or a terminating series with a pole
    (``(b_j)_n = 0``) is "anything else".

    Examples
    --------
    >>> round(float(generalized_hypergeometric([1], [2], 1)), 6)  # 1F1(1; 2; 1) = e - 1
    1.718282
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    p = len(a)
    q = len(b)

    # The series terminates (is a polynomial) when any numerator
    # parameter is a non-positive integer; only then does it converge
    # outside the |z| < 1 (p = q+1) / z = 0 (p > q+1) domains. 2F1 is
    # analytically continued by scipy instead.
    z_arr = np.atleast_1d(z)
    terminates = any(ai <= 0 and float(ai).is_integer() for ai in a)
    if not terminates and (p, q) != (2, 1):
        if p == q + 1 and np.any(np.abs(z_arr) >= 1.0):
            raise ValueError(
                "pFq with p = q + 1 diverges for |z| >= 1 (the series "
                "is not a polynomial); no value exists to return."
            )
        if p > q + 1 and np.any(z_arr != 0.0):
            raise ValueError(
                "pFq with p > q + 1 diverges for z != 0 (the series "
                "is not a polynomial); no value exists to return."
            )

    infinite = np.isinf(z)
    if np.any(infinite):
        out = np.empty(z.shape)
        if not np.all(infinite):
            out[...] = generalized_hypergeometric(
                a, b, np.where(infinite, 0.0, z), max_terms, tol
            )
        out[infinite] = [_pfq_limit_at_infinity(a, b, zi) for zi in z[infinite]]
        return out if out.ndim else out[()]

    # For special cases, use scipy's optimized functions
    if p == 0 and q == 0:
        return np.exp(z)
    if p == 0 and q == 1:
        return hyp0f1(b[0], z)
    elif p == 1 and q == 1:
        return hyp1f1(a[0], b[0], z)
    elif p == 2 and q == 1:
        return hyp2f1(a[0], a[1], b[0], z)

    # General case: use Numba-optimized series summation
    result, max_term, converged = _hypergeometric_series(a, b, z_arr, max_terms, tol)
    if not converged:
        warnings.warn(
            f"generalized_hypergeometric did not reach tol={tol:g} "
            f"within max_terms={max_terms}; the returned partial sum "
            "may be inaccurate. Increase max_terms.",
            RuntimeWarning,
            stacklevel=2,
        )

    result = _flag_cancellation(result, max_term, a, b, z_arr, terminates)
    return result if result.size > 1 else result[0]


__all__ = [
    "hyp0f1",
    "hyp1f1",
    "hyp2f1",
    "hyperu",
    "hyp1f1_regularized",
    "pochhammer",
    "falling_factorial",
    "generalized_hypergeometric",
]
