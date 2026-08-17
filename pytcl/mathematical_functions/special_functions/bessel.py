"""
Bessel functions and related special functions.

This module provides Bessel functions commonly used in signal processing,
antenna theory, and scattering problems in tracking applications.
"""

from typing import Any, Union

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
    >>> float(besselj(0, 0))
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

    Examples
    --------
    >>> round(float(bessely(0, 1)), 6)
    0.088257

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
    >>> float(besseli(0, 0))
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

    Examples
    --------
    >>> round(float(besselk(0, 1)), 6)
    0.421024
    >>> round(float(besselk(1, 2)), 6)
    0.139866

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

    Examples
    --------
    >>> h = besselh(0, 1, 1)  # H^(1)_0(1)
    >>> round(float(h.real), 6)
    0.765198
    >>> round(float(h.imag), 6)
    0.088257

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

    Examples
    --------
    >>> round(float(spherical_jn(0, 1)), 6)  # sin(1)/1
    0.841471
    >>> round(float(spherical_jn(0, 1, derivative=True)), 6)  # Derivative
    -0.301169

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

    Examples
    --------
    >>> round(float(spherical_yn(0, 1)), 6)  # -cos(1)/1
    -0.540302

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

    Examples
    --------
    >>> round(float(spherical_in(0, 1)), 6)  # sinh(1)/1
    1.175201

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

    Examples
    --------
    >>> round(float(spherical_kn(0, 1)), 6)  # (pi/2) * exp(-1)
    0.577864

    See Also
    --------
    scipy.special.spherical_kn : Modified spherical Bessel function of second kind.
    """
    return np.asarray(sp.spherical_kn(n, x, derivative=derivative), dtype=np.float64)


def airy(
    x: ArrayLike,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
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

    Examples
    --------
    >>> Ai, Aip, Bi, Bip = airy(0)
    >>> round(float(Ai), 6)
    0.355028
    >>> round(float(Bi), 6)
    0.614927

    See Also
    --------
    scipy.special.airy : Airy functions.
    """
    ai, aip, bi, bip = sp.airy(x)
    return (
        np.asarray(ai, dtype=np.float64),
        np.asarray(aip, dtype=np.float64),
        np.asarray(bi, dtype=np.float64),
        np.asarray(bip, dtype=np.float64),
    )


def _bessel_ratio_cf(
    n: Union[int, float], x: NDArray[np.float64], sign: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Continued fraction for B_{n+1}(x)/B_n(x) via the modified Lentz method.

    Evaluates f = a_1/(b_1 + a_2/(b_2 + a_3/(b_3 + ...))) with
    b_k = 2(n+k)/x, a_1 = 1, and a_k = sign for k >= 2, where sign is -1
    for J (ordinary) and +1 for I (modified). All elements of ``x`` must be
    finite and nonzero.

    Returns the fraction values and a boolean mask of the elements that
    converged (|delta - 1| < eps reached). For kind 'j' the fraction needs
    roughly max(n, |x|) terms (Numerical Recipes 3rd ed., Sec. 6.5), so
    elements with |x| beyond the iteration cap do NOT converge here; the
    caller must handle them, never returning the unconverged value.
    """
    tiny = 1e-300
    eps = np.finfo(np.float64).eps
    f = np.full_like(x, tiny)
    c = f.copy()
    d = np.zeros_like(x)
    converged = np.zeros(x.shape, dtype=bool)
    for k in range(1, 10001):
        a = 1.0 if k == 1 else sign
        b = 2.0 * (n + k) / x
        d = b + a * d
        d[d == 0.0] = tiny
        c = b + a / c
        c[c == 0.0] = tiny
        d = 1.0 / d
        delta = c * d
        f = np.where(converged, f, f * delta)
        converged |= np.abs(delta - 1.0) < eps
        if converged.all():
            break
    return f, converged


def bessel_ratio(
    n: Union[int, float],
    x: ArrayLike,
    kind: str = "j",
) -> NDArray[np.floating]:
    """
    Ratio of Bessel functions J_{n+1}(x) / J_n(x) or I_{n+1}(x) / I_n(x).

    Parameters
    ----------
    n : int or float
        Order of the Bessel function in the denominator.
    x : array_like
        Argument of the Bessel function.
    kind : str, optional
        Type of Bessel function: 'j' for J_n, 'i' for I_n. Default is 'j'.

    Returns
    -------
    ratio : ndarray
        Values of J_{n+1}(x) / J_n(x) or I_{n+1}(x) / I_n(x).

    Notes
    -----
    Evaluated with the modified Lentz method (Thompson & Barnett, J. Comput.
    Phys. 64:490, 1986; Numerical Recipes 3rd ed., Sec. 6.5) applied to the
    continued fraction that follows from the three-term recurrence:

        J_{n+1}(x)/J_n(x) = 1/(2(n+1)/x - 1/(2(n+2)/x - 1/(2(n+3)/x - ...)))
        I_{n+1}(x)/I_n(x) = 1/(2(n+1)/x + 1/(2(n+2)/x + 1/(2(n+3)/x + ...)))

    J_{n+k}(x) and I_{n+k}(x) are the minimal solutions of their recurrences
    as k grows, so by Pincherle's theorem the fractions converge to the exact
    ratio without ever forming the numerator and denominator separately --
    the ratio therefore stays finite where a direct quotient of float64
    Bessel values underflows to 0/0 (e.g. n = 170, x = 1). Measured against
    a 50-digit mpmath reference over the grid n in {0, 1, 5, 20, 80, 170,
    400} x x in {0.5, 1, 10, 50, 100}, the worst relative error is 9.6e-15
    for kind 'j' and 1.5e-15 for kind 'i'
    (tests/validation/test_special_functions_audit.py).

    The 'j' fraction needs roughly max(n, |x|) terms to converge; where it
    misses the iteration cap of 10000 (|x| in the thousands and beyond,
    small n) the direct quotient jv(n+1, x)/jv(n, x) is used instead --
    machine-accurate there since neither value underflows at large |x|.
    Measured worst relative error 7.9e-14 at x in {9000, 15000, 30000},
    n in {0, 5}; the all-positive 'i' fraction converges within the cap
    (1.3e-15 measured at x = 30000).

    Near a zero of J_n(x) the ratio is well-defined and the fraction
    converges to it, but roundoff in evaluating the fraction is amplified
    like 1/|J_n(x)|: measured agreement with mpmath at n = 0 loosens from
    5e-15 at x = 2.404 to 1e-8 at x = 2.4048255576, i.e. 1e-10 from the
    first zero of J_0 at x ~= 2.40482555769577, and reaches O(1) relative
    error (measured 1.4) at the float64 neighbor of the zero. At x = 0 the
    limit value 0 is returned for every order (the two-sided limit of the
    ratio); the ratio is nan for non-finite x.

    Examples
    --------
    >>> round(float(bessel_ratio(0, 1)), 6)  # J_1(1) / J_0(1)
    0.575081
    """
    x = np.asarray(x, dtype=np.float64)

    kind = kind.lower()
    if kind == "j":
        sign = -1.0
    elif kind == "i":
        sign = 1.0
    else:
        raise ValueError(f"kind must be 'j' or 'i', got '{kind}'")

    flat = np.atleast_1d(x).ravel()
    out = np.full(flat.shape, np.nan)
    out[flat == 0.0] = 0.0
    regular = np.isfinite(flat) & (flat != 0.0)
    if regular.any():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            cf, converged = _bessel_ratio_cf(n, flat[regular], sign)
            if not converged.all():
                # The fraction needs ~max(n, |x|) terms, so it misses the
                # iteration cap for |x| in the thousands and beyond -- the
                # regime where neither Bessel value underflows and the
                # direct quotient is machine-accurate. Fall back to it
                # rather than return an unconverged fraction value.
                fn = sp.jv if kind == "j" else sp.iv
                xs = flat[regular][~converged]
                num = fn(n + 1, xs)
                den = fn(n, xs)
                cf[~converged] = np.where(den != 0, num / den, np.inf * np.sign(num))
        out[regular] = cf

    return out.reshape(x.shape)


def bessel_deriv(
    n: Union[int, float],
    x: ArrayLike,
    kind: str = "j",
) -> NDArray[np.floating]:
    """
    Derivative of Bessel function d/dx[B_n(x)].

    Parameters
    ----------
    n : int or float
        Order of the Bessel function.
    x : array_like
        Argument of the Bessel function.
    kind : str, optional
        Type of Bessel function: 'j', 'y', 'i', or 'k'. Default is 'j'.

    Returns
    -------
    deriv : ndarray
        Values of dB_n(x)/dx.

    Notes
    -----
    Uses the identity:
    dJ_n/dx = (J_{n-1}(x) - J_{n+1}(x)) / 2
    dY_n/dx = (Y_{n-1}(x) - Y_{n+1}(x)) / 2
    dI_n/dx = (I_{n-1}(x) + I_{n+1}(x)) / 2
    dK_n/dx = -(K_{n-1}(x) + K_{n+1}(x)) / 2

    Examples
    --------
    >>> round(float(bessel_deriv(0, 1, kind='j')), 6)  # -J_1(1)
    -0.440051
    """
    x = np.asarray(x, dtype=np.float64)

    kind = kind.lower()
    if kind == "j":
        deriv = (sp.jv(n - 1, x) - sp.jv(n + 1, x)) / 2
    elif kind == "y":
        deriv = (sp.yv(n - 1, x) - sp.yv(n + 1, x)) / 2
    elif kind == "i":
        deriv = (sp.iv(n - 1, x) + sp.iv(n + 1, x)) / 2
    elif kind == "k":
        deriv = -(sp.kv(n - 1, x) + sp.kv(n + 1, x)) / 2
    else:
        raise ValueError(f"kind must be 'j', 'y', 'i', or 'k', got '{kind}'")

    return np.asarray(deriv, dtype=np.float64)


def struve_h(
    n: Union[int, float],
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Struve function H_n(x).

    The Struve function is defined by the integral::

        H_n(x) = (2/sqrt(pi)) * (x/2)^n * integral from 0 to pi/2 of
                 sin(x*cos(t)) * sin^(2n)(t) dt

    Parameters
    ----------
    n : int or float
        Order of the Struve function.
    x : array_like
        Argument of the function.

    Returns
    -------
    H : ndarray
        Values of H_n(x).

    Notes
    -----
    Related to Bessel functions through:
    H_0(x) is the particular solution of y'' + y'/x + y = 2/(pi*x)

    Examples
    --------
    >>> round(float(struve_h(0, 1)), 6)
    0.568657
    """
    return np.asarray(sp.struve(n, x), dtype=np.float64)


def struve_l(
    n: Union[int, float],
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Modified Struve function L_n(x).

    The modified Struve function is related to the Struve function by:
    L_n(x) = -i * exp(-i*n*pi/2) * H_n(i*x)

    Parameters
    ----------
    n : int or float
        Order of the modified Struve function.
    x : array_like
        Argument of the function.

    Returns
    -------
    L : ndarray
        Values of L_n(x).

    Examples
    --------
    >>> round(float(struve_l(0, 1)), 6)
    0.710243
    """
    return np.asarray(sp.modstruve(n, x), dtype=np.float64)


def bessel_zeros(
    n: int,
    nt: int,
    kind: str = "j",
) -> NDArray[np.floating]:
    """
    Zeros of Bessel functions.

    Computes the first nt zeros of J_n(x), Y_n(x), or their derivatives.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    nt : int
        Number of zeros to compute.
    kind : str, optional
        Type: 'j' for J_n zeros, 'y' for Y_n zeros,
        'jp' for J_n' zeros, 'yp' for Y_n' zeros. Default is 'j'.

    Returns
    -------
    zeros : ndarray
        Array of zeros.

    Examples
    --------
    >>> bessel_zeros(0, 3, kind='j')  # First 3 zeros of J_0
    array([2.40482556, 5.52007811, 8.65372791])
    """
    kind = kind.lower()

    if kind == "j":
        return np.asarray(sp.jn_zeros(n, nt), dtype=np.float64)
    elif kind == "y":
        return np.asarray(sp.yn_zeros(n, nt), dtype=np.float64)
    elif kind == "jp":
        return np.asarray(sp.jnp_zeros(n, nt), dtype=np.float64)
    elif kind == "yp":
        return np.asarray(sp.ynp_zeros(n, nt), dtype=np.float64)
    else:
        raise ValueError(f"kind must be 'j', 'y', 'jp', or 'yp', got '{kind}'")


def kelvin(
    x: ArrayLike,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    """
    Kelvin functions ber, bei, ker, kei.

    Kelvin functions are the real and imaginary parts of the
    Bessel functions with argument x*exp(3*pi*i/4).

    Parameters
    ----------
    x : array_like
        Argument of the Kelvin functions.

    Returns
    -------
    ber : ndarray
        Kelvin function ber(x).
    bei : ndarray
        Kelvin function bei(x).
    ker : ndarray
        Kelvin function ker(x).
    kei : ndarray
        Kelvin function kei(x).

    Notes
    -----
    ber(x) + i*bei(x) = J_0(x * exp(3*pi*i/4))
    ker(x) + i*kei(x) = K_0(x * exp(pi*i/4))

    Examples
    --------
    >>> ber, bei, ker, kei = kelvin(1)
    >>> round(float(ber), 6)
    0.984382
    """
    x = np.asarray(x, dtype=np.float64)

    # Use the individual scipy Kelvin functions for real-valued results
    ber = np.asarray(sp.ber(x), dtype=np.float64)
    bei = np.asarray(sp.bei(x), dtype=np.float64)
    ker = np.asarray(sp.ker(x), dtype=np.float64)
    kei = np.asarray(sp.kei(x), dtype=np.float64)

    return ber, bei, ker, kei


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
    "bessel_ratio",
    "bessel_deriv",
    "struve_h",
    "struve_l",
    "bessel_zeros",
    "kelvin",
]
