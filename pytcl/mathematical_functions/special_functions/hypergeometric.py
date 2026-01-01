"""
Hypergeometric functions.

This module provides hypergeometric functions commonly used in
mathematical physics, probability theory, and special function evaluation.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


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
    >>> hyp0f1(1, 0)  # 0F1(1; 0) = 1
    1.0
    >>> hyp0f1(1, 1)
    2.279585...

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Chapter 16.
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
    >>> hyp1f1(1, 1, 1)  # exp(1)
    2.718281828...
    >>> hyp1f1(0.5, 1.5, -1)  # Related to erf
    0.842700...

    References
    ----------
    .. [1] Abramowitz & Stegun, "Handbook of Mathematical Functions", Ch. 13.
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

    The function 2F1(a, b; c; z) is defined by the series:
    2F1(a, b; c; z) = sum_{k=0}^inf (a)_k * (b)_k * z^k / ((c)_k * k!)

    converging for |z| < 1.

    Parameters
    ----------
    a : array_like
        First numerator parameter.
    b : array_like
        Second numerator parameter.
    c : array_like
        Denominator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function. For |z| >= 1, analytic continuation is used.

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
    >>> hyp2f1(1, 1, 2, 0.5)  # -log(1-0.5)/0.5 = log(2)
    1.386294...
    >>> hyp2f1(0.5, 0.5, 1.5, 0.25)  # Related to arcsin
    1.072379...

    References
    ----------
    .. [1] NIST DLMF, Chapter 15.
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

    The function U(a, b, z) is defined as:
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
    >>> hyperu(1, 1, 1)
    0.596347...
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

    Notes
    -----
    This function remains finite even when b is a non-positive integer,
    unlike the standard 1F1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return sp.hyp1f1(a, b, z) / sp.gamma(b)


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
    >>> pochhammer(1, 5)  # 5!
    120.0
    >>> pochhammer(3, 4)  # 3*4*5*6
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
    >>> falling_factorial(5, 3)  # 5*4*3
    60.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    # (a)_n falling = a!/(a-n)! = Gamma(a+1)/Gamma(a-n+1)
    return np.asarray(sp.poch(a - n + 1, n), dtype=np.float64)


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
    The series converges for:
    - p <= q: all z
    - p = q + 1: |z| < 1
    - p > q + 1: diverges except for polynomial cases

    Examples
    --------
    >>> generalized_hypergeometric([1], [2], 1)  # 1F1(1; 2; 1) ~ 1.718...
    1.718281...
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    p = len(a)
    q = len(b)

    # For special cases, use scipy's optimized functions
    if p == 0 and q == 1:
        return hyp0f1(b[0], z)
    elif p == 1 and q == 1:
        return hyp1f1(a[0], b[0], z)
    elif p == 2 and q == 1:
        return hyp2f1(a[0], a[1], b[0], z)

    # General case: series summation
    z = np.atleast_1d(z)
    result = np.ones_like(z, dtype=np.float64)
    term = np.ones_like(z, dtype=np.float64)

    for k in range(1, max_terms):
        # Compute ratio term_k / term_{k-1}
        num_factor = np.prod(a + k - 1)
        den_factor = np.prod(b + k - 1) * k
        term = term * z * num_factor / den_factor

        result += term

        if np.all(np.abs(term) < tol * np.abs(result)):
            break

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
