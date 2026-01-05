"""
Gamma and related functions.

This module provides gamma functions, factorials, and related special
functions used in statistics and probability calculations.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


def gamma(x: ArrayLike) -> NDArray[np.floating]:
    """
    Gamma function.

    Computes Γ(x) = ∫_0^∞ t^(x-1) * e^(-t) dt.

    Parameters
    ----------
    x : array_like
        Argument of the gamma function.

    Returns
    -------
    Γ : ndarray
        Values of Γ(x).

    Notes
    -----
    For positive integers, Γ(n) = (n-1)!

    Examples
    --------
    >>> gamma(5)  # 4! = 24
    24.0
    >>> gamma(0.5)  # sqrt(pi)
    1.7724538509055159

    See Also
    --------
    scipy.special.gamma : Gamma function.
    """
    return np.asarray(sp.gamma(x), dtype=np.float64)


def gammaln(x: ArrayLike) -> NDArray[np.floating]:
    """
    Natural logarithm of the absolute value of the gamma function.

    Computes ln|Γ(x)|. This is more numerically stable than computing
    log(gamma(x)) for large x.

    Parameters
    ----------
    x : array_like
        Argument of the function.

    Returns
    -------
    lng : ndarray
        Values of ln|Γ(x)|.

    Examples
    --------
    >>> gammaln(100)  # log(99!)
    359.1342053695754

    See Also
    --------
    scipy.special.gammaln : Log of gamma function.
    """
    return np.asarray(sp.gammaln(x), dtype=np.float64)


def gammainc(a: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """
    Regularized lower incomplete gamma function.

    Computes P(a, x) = γ(a, x) / Γ(a), where γ(a, x) = ∫_0^x t^(a-1) * e^(-t) dt.

    Parameters
    ----------
    a : array_like
        Parameter of the function (must be positive).
    x : array_like
        Upper limit of integration (must be non-negative).

    Returns
    -------
    P : ndarray
        Values of the regularized lower incomplete gamma function.

    Notes
    -----
    This is the CDF of the gamma distribution.

    Examples
    --------
    >>> gammainc(1, 1)  # 1 - exp(-1)
    0.632...

    See Also
    --------
    scipy.special.gammainc : Regularized lower incomplete gamma function.
    gammaincc : Upper incomplete gamma (complement).
    """
    return np.asarray(sp.gammainc(a, x), dtype=np.float64)


def gammaincc(a: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """
    Regularized upper incomplete gamma function.

    Computes Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x).

    Parameters
    ----------
    a : array_like
        Parameter of the function (must be positive).
    x : array_like
        Lower limit of integration (must be non-negative).

    Returns
    -------
    Q : ndarray
        Values of the regularized upper incomplete gamma function.

    Examples
    --------
    >>> gammaincc(1, 1)  # exp(-1)
    0.367...

    See Also
    --------
    scipy.special.gammaincc : Regularized upper incomplete gamma function.
    """
    return np.asarray(sp.gammaincc(a, x), dtype=np.float64)


def gammaincinv(a: ArrayLike, y: ArrayLike) -> NDArray[np.floating]:
    """
    Inverse of the regularized lower incomplete gamma function.

    Finds x such that P(a, x) = y.

    Parameters
    ----------
    a : array_like
        Parameter of the function.
    y : array_like
        Target probability (between 0 and 1).

    Returns
    -------
    x : ndarray
        Values where P(a, x) = y.

    Examples
    --------
    >>> gammaincinv(1, 0.5)  # Median of exponential distribution
    0.693...

    See Also
    --------
    scipy.special.gammaincinv : Inverse of lower incomplete gamma.
    """
    return np.asarray(sp.gammaincinv(a, y), dtype=np.float64)


def digamma(x: ArrayLike) -> NDArray[np.floating]:
    """
    Digamma (psi) function.

    Computes ψ(x) = d/dx ln(Γ(x)) = Γ'(x) / Γ(x).

    Parameters
    ----------
    x : array_like
        Argument of the function.

    Returns
    -------
    ψ : ndarray
        Values of the digamma function.

    Examples
    --------
    >>> digamma(1)  # -γ (negative Euler-Mascheroni constant)
    -0.577...

    See Also
    --------
    scipy.special.digamma : Digamma function.
    polygamma : Higher derivatives.
    """
    return np.asarray(sp.digamma(x), dtype=np.float64)


def polygamma(n: int, x: ArrayLike) -> NDArray[np.floating]:
    """
    Polygamma function.

    Computes ψ^(n)(x) = d^(n+1)/dx^(n+1) ln(Γ(x)).

    Parameters
    ----------
    n : int
        Order of the derivative (n=0 gives digamma, n=1 gives trigamma, etc.).
    x : array_like
        Argument of the function.

    Returns
    -------
    ψn : ndarray
        Values of the n-th polygamma function.

    Examples
    --------
    >>> polygamma(0, 1)  # Digamma at 1 = -γ
    -0.577...
    >>> polygamma(1, 1)  # Trigamma at 1 = π²/6
    1.644...

    See Also
    --------
    scipy.special.polygamma : Polygamma function.
    """
    return np.asarray(sp.polygamma(n, x), dtype=np.float64)


def beta(a: ArrayLike, b: ArrayLike) -> NDArray[np.floating]:
    """
    Beta function.

    Computes B(a, b) = Γ(a) * Γ(b) / Γ(a + b).

    Parameters
    ----------
    a : array_like
        First parameter.
    b : array_like
        Second parameter.

    Returns
    -------
    B : ndarray
        Values of the beta function.

    Examples
    --------
    >>> beta(1, 1)
    1.0
    >>> beta(0.5, 0.5)  # pi
    3.141592653589793

    See Also
    --------
    scipy.special.beta : Beta function.
    """
    return np.asarray(sp.beta(a, b), dtype=np.float64)


def betaln(a: ArrayLike, b: ArrayLike) -> NDArray[np.floating]:
    """
    Natural logarithm of the beta function.

    Computes ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a + b)).

    Parameters
    ----------
    a : array_like
        First parameter.
    b : array_like
        Second parameter.

    Returns
    -------
    lnB : ndarray
        Values of ln(B(a, b)).

    Examples
    --------
    >>> import numpy as np
    >>> betaln(100, 100)  # More stable than log(beta(100, 100))
    -137.74...

    See Also
    --------
    scipy.special.betaln : Log of beta function.
    """
    return np.asarray(sp.betaln(a, b), dtype=np.float64)


def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """
    Regularized incomplete beta function.

    Computes I_x(a, b) = B(x; a, b) / B(a, b), where
    B(x; a, b) = ∫_0^x t^(a-1) * (1-t)^(b-1) dt.

    Parameters
    ----------
    a : array_like
        First parameter (must be positive).
    b : array_like
        Second parameter (must be positive).
    x : array_like
        Upper limit of integration (between 0 and 1).

    Returns
    -------
    I : ndarray
        Values of the regularized incomplete beta function.

    Notes
    -----
    This is the CDF of the beta distribution.

    Examples
    --------
    >>> betainc(1, 1, 0.5)  # Uniform distribution CDF at 0.5
    0.5

    See Also
    --------
    scipy.special.betainc : Regularized incomplete beta function.
    """
    return np.asarray(sp.betainc(a, b, x), dtype=np.float64)


def betaincinv(a: ArrayLike, b: ArrayLike, y: ArrayLike) -> NDArray[np.floating]:
    """
    Inverse of the regularized incomplete beta function.

    Finds x such that I_x(a, b) = y.

    Parameters
    ----------
    a : array_like
        First parameter.
    b : array_like
        Second parameter.
    y : array_like
        Target probability (between 0 and 1).

    Returns
    -------
    x : ndarray
        Values where I_x(a, b) = y.

    Examples
    --------
    >>> betaincinv(1, 1, 0.5)  # Median of uniform distribution
    0.5

    See Also
    --------
    scipy.special.betaincinv : Inverse of incomplete beta function.
    """
    return np.asarray(sp.betaincinv(a, b, y), dtype=np.float64)


def factorial(n: ArrayLike, exact: bool = False) -> NDArray[np.floating]:
    """
    Factorial function.

    Computes n! = n * (n-1) * ... * 2 * 1.

    Parameters
    ----------
    n : array_like
        Input values (non-negative integers).
    exact : bool, optional
        If True, compute exact integer factorial (may overflow for large n).
        If False (default), use gamma function approximation.

    Returns
    -------
    nfact : ndarray
        Values of n!.

    Examples
    --------
    >>> factorial(5)
    120.0
    >>> factorial(np.array([1, 2, 3, 4, 5]))
    array([  1.,   2.,   6.,  24., 120.])

    See Also
    --------
    scipy.special.factorial : Factorial function.
    """
    return np.asarray(sp.factorial(n, exact=exact), dtype=np.float64)


def factorial2(n: ArrayLike, exact: bool = False) -> NDArray[np.floating]:
    """
    Double factorial.

    Computes n!! = n * (n-2) * (n-4) * ... * (2 or 1).

    Parameters
    ----------
    n : array_like
        Input values (non-negative integers).
    exact : bool, optional
        If True, compute exact integer result.

    Returns
    -------
    nfact2 : ndarray
        Values of n!!.

    Examples
    --------
    >>> factorial2(5)  # 5 * 3 * 1 = 15
    15.0
    >>> factorial2(6)  # 6 * 4 * 2 = 48
    48.0

    See Also
    --------
    scipy.special.factorial2 : Double factorial.
    """
    return np.asarray(sp.factorial2(n, exact=exact), dtype=np.float64)


def comb(
    n: ArrayLike,
    k: ArrayLike,
    exact: bool = False,
    repetition: bool = False,
) -> NDArray[np.floating]:
    """
    Binomial coefficient (combinations).

    Computes C(n, k) = n! / (k! * (n-k)!).

    Parameters
    ----------
    n : array_like
        Number of elements to choose from.
    k : array_like
        Number of elements to choose.
    exact : bool, optional
        If True, compute exact integer result.
    repetition : bool, optional
        If True, compute combinations with repetition.

    Returns
    -------
    C : ndarray
        Values of C(n, k).

    Examples
    --------
    >>> comb(5, 2)
    10.0
    >>> comb(10, 3)
    120.0

    See Also
    --------
    scipy.special.comb : Combinations.
    """
    return np.asarray(
        sp.comb(n, k, exact=exact, repetition=repetition), dtype=np.float64
    )


def perm(n: ArrayLike, k: ArrayLike, exact: bool = False) -> NDArray[np.floating]:
    """
    Permutation coefficient.

    Computes P(n, k) = n! / (n-k)!.

    Parameters
    ----------
    n : array_like
        Number of elements to arrange.
    k : array_like
        Number of elements in arrangement.
    exact : bool, optional
        If True, compute exact integer result.

    Returns
    -------
    P : ndarray
        Values of P(n, k).

    Examples
    --------
    >>> perm(5, 2)
    20.0

    See Also
    --------
    scipy.special.perm : Permutations.
    """
    return np.asarray(sp.perm(n, k, exact=exact), dtype=np.float64)


__all__ = [
    "gamma",
    "gammaln",
    "gammainc",
    "gammaincc",
    "gammaincinv",
    "digamma",
    "polygamma",
    "beta",
    "betaln",
    "betainc",
    "betaincinv",
    "factorial",
    "factorial2",
    "comb",
    "perm",
]
