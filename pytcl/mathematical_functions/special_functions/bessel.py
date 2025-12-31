"""
Bessel functions and related special functions.

This module provides Bessel functions commonly used in signal processing,
antenna theory, and scattering problems in tracking applications.
"""

from typing import Union

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


def besselj(
    n: Union[int, float, ArrayLike],
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Bessel function of the first kind.

    Computes J_n(x), the Bessel function of the first kind of order n.

    Parameters
    ----------
    n : int, float, or array_like
        Order of the Bessel function.
    x : array_like
        Argument of the Bessel function.

    Returns
    -------
    J : ndarray
        Values of J_n(x).

    Examples
    --------
    >>> besselj(0, 0)
    1.0
    >>> besselj(1, np.array([0, 1, 2]))
    array([0.        , 0.44005059, 0.57672481])

    See Also
    --------
    scipy.special.jv : Bessel function of first kind of real order.
    """
    return np.asarray(sp.jv(n, x), dtype=np.float64)


def bessely(
    n: Union[int, float, ArrayLike],
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Bessel function of the second kind (Neumann function).

    Computes Y_n(x), the Bessel function of the second kind of order n.

    Parameters
    ----------
    n : int, float, or array_like
        Order of the Bessel function.
    x : array_like
        Argument of the Bessel function. Must be positive.

    Returns
    -------
    Y : ndarray
        Values of Y_n(x).

    Notes
    -----
    Y_n(x) is singular at x = 0.

    See Also
    --------
    scipy.special.yv : Bessel function of second kind of real order.
    """
    return np.asarray(sp.yv(n, x), dtype=np.float64)


def besseli(
    n: Union[int, float, ArrayLike],
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Modified Bessel function of the first kind.

    Computes I_n(x), the modified Bessel function of the first kind.

    Parameters
    ----------
    n : int, float, or array_like
        Order of the Bessel function.
    x : array_like
        Argument of the Bessel function.

    Returns
    -------
    I : ndarray
        Values of I_n(x).

    Examples
    --------
    >>> besseli(0, 0)
    1.0

    See Also
    --------
    scipy.special.iv : Modified Bessel function of first kind.
    """
    return np.asarray(sp.iv(n, x), dtype=np.float64)


def besselk(
    n: Union[int, float, ArrayLike],
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Modified Bessel function of the second kind.

    Computes K_n(x), the modified Bessel function of the second kind.

    Parameters
    ----------
    n : int, float, or array_like
        Order of the Bessel function.
    x : array_like
        Argument of the Bessel function. Must be positive.

    Returns
    -------
    K : ndarray
        Values of K_n(x).

    Notes
    -----
    K_n(x) is singular at x = 0.

    See Also
    --------
    scipy.special.kv : Modified Bessel function of second kind.
    """
    return np.asarray(sp.kv(n, x), dtype=np.float64)


def besselh(
    n: Union[int, float, ArrayLike],
    k: int,
    x: ArrayLike,
) -> NDArray[np.complexfloating]:
    """
    Hankel function (Bessel function of the third kind).

    Computes H^(k)_n(x), the Hankel function of the first (k=1) or
    second (k=2) kind.

    Parameters
    ----------
    n : int, float, or array_like
        Order of the Hankel function.
    k : int
        Kind of Hankel function. Must be 1 or 2.
    x : array_like
        Argument of the Hankel function.

    Returns
    -------
    H : ndarray
        Complex values of H^(k)_n(x).

    Notes
    -----
    H^(1)_n(x) = J_n(x) + i*Y_n(x)
    H^(2)_n(x) = J_n(x) - i*Y_n(x)

    See Also
    --------
    scipy.special.hankel1 : Hankel function of first kind.
    scipy.special.hankel2 : Hankel function of second kind.
    """
    if k == 1:
        return np.asarray(sp.hankel1(n, x), dtype=np.complex128)
    elif k == 2:
        return np.asarray(sp.hankel2(n, x), dtype=np.complex128)
    else:
        raise ValueError(f"k must be 1 or 2, got {k}")


def spherical_jn(
    n: int,
    x: ArrayLike,
    derivative: bool = False,
) -> NDArray[np.floating]:
    """
    Spherical Bessel function of the first kind.

    Computes j_n(x), the spherical Bessel function of the first kind.

    Parameters
    ----------
    n : int
        Order of the function (non-negative).
    x : array_like
        Argument of the function.
    derivative : bool, optional
        If True, return the derivative j_n'(x) instead. Default is False.

    Returns
    -------
    j : ndarray
        Values of j_n(x) or j_n'(x).

    Notes
    -----
    j_n(x) = sqrt(pi / (2*x)) * J_{n+1/2}(x)

    See Also
    --------
    scipy.special.spherical_jn : Spherical Bessel function of first kind.
    """
    return np.asarray(sp.spherical_jn(n, x, derivative=derivative), dtype=np.float64)


def spherical_yn(
    n: int,
    x: ArrayLike,
    derivative: bool = False,
) -> NDArray[np.floating]:
    """
    Spherical Bessel function of the second kind.

    Computes y_n(x), the spherical Bessel function of the second kind.

    Parameters
    ----------
    n : int
        Order of the function (non-negative).
    x : array_like
        Argument of the function. Must be positive.
    derivative : bool, optional
        If True, return the derivative y_n'(x) instead. Default is False.

    Returns
    -------
    y : ndarray
        Values of y_n(x) or y_n'(x).

    See Also
    --------
    scipy.special.spherical_yn : Spherical Bessel function of second kind.
    """
    return np.asarray(sp.spherical_yn(n, x, derivative=derivative), dtype=np.float64)


def spherical_in(
    n: int,
    x: ArrayLike,
    derivative: bool = False,
) -> NDArray[np.floating]:
    """
    Modified spherical Bessel function of the first kind.

    Computes i_n(x), the modified spherical Bessel function of the first kind.

    Parameters
    ----------
    n : int
        Order of the function (non-negative).
    x : array_like
        Argument of the function.
    derivative : bool, optional
        If True, return the derivative i_n'(x) instead. Default is False.

    Returns
    -------
    i : ndarray
        Values of i_n(x) or i_n'(x).

    See Also
    --------
    scipy.special.spherical_in : Modified spherical Bessel function of first kind.
    """
    return np.asarray(sp.spherical_in(n, x, derivative=derivative), dtype=np.float64)


def spherical_kn(
    n: int,
    x: ArrayLike,
    derivative: bool = False,
) -> NDArray[np.floating]:
    """
    Modified spherical Bessel function of the second kind.

    Computes k_n(x), the modified spherical Bessel function of the second kind.

    Parameters
    ----------
    n : int
        Order of the function (non-negative).
    x : array_like
        Argument of the function. Must be positive.
    derivative : bool, optional
        If True, return the derivative k_n'(x) instead. Default is False.

    Returns
    -------
    k : ndarray
        Values of k_n(x) or k_n'(x).

    See Also
    --------
    scipy.special.spherical_kn : Modified spherical Bessel function of second kind.
    """
    return np.asarray(sp.spherical_kn(n, x, derivative=derivative), dtype=np.float64)


def airy(x: ArrayLike) -> tuple:
    """
    Airy functions and their derivatives.

    Computes Ai(x), Ai'(x), Bi(x), Bi'(x).

    Parameters
    ----------
    x : array_like
        Argument of the Airy functions.

    Returns
    -------
    Ai : ndarray
        Airy function Ai(x).
    Aip : ndarray
        Derivative of Airy function Ai'(x).
    Bi : ndarray
        Airy function Bi(x).
    Bip : ndarray
        Derivative of Airy function Bi'(x).

    See Also
    --------
    scipy.special.airy : Airy functions.
    """
    result = sp.airy(x)
    return tuple(np.asarray(r, dtype=np.float64) for r in result)


__all__ = [
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "besselh",
    "spherical_jn",
    "spherical_yn",
    "spherical_in",
    "spherical_kn",
    "airy",
]
