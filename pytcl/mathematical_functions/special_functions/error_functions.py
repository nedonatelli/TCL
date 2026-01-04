"""
Error functions and related special functions.

This module provides error functions and their variants, commonly used
in probability theory and statistical analysis.
"""

from typing import Any

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


def erf(x: ArrayLike) -> NDArray[np.floating]:
    """
    Error function.

    Computes erf(x) = (2/√π) * ∫_0^x e^(-t²) dt.

    Parameters
    ----------
    x : array_like
        Argument of the error function.

    Returns
    -------
    y : ndarray
        Values of erf(x).

    Notes
    -----
    - erf(0) = 0
    - erf(∞) = 1
    - erf(-x) = -erf(x)

    The error function is related to the normal distribution CDF by:
    Φ(x) = (1 + erf(x/√2)) / 2

    Examples
    --------
    >>> erf(0)
    0.0
    >>> erf(1)
    0.8427007929497149

    See Also
    --------
    scipy.special.erf : Error function.
    erfc : Complementary error function.
    """
    return np.asarray(sp.erf(x), dtype=np.float64)


def erfc(x: ArrayLike) -> NDArray[np.floating]:
    """
    Complementary error function.

    Computes erfc(x) = 1 - erf(x) = (2/√π) * ∫_x^∞ e^(-t²) dt.

    Parameters
    ----------
    x : array_like
        Argument of the function.

    Returns
    -------
    y : ndarray
        Values of erfc(x).

    Notes
    -----
    This function is more accurate than computing 1 - erf(x) for large x.

    Examples
    --------
    >>> erfc(0)
    1.0
    >>> erfc(3)  # Very small
    2.2090496998585438e-05

    See Also
    --------
    scipy.special.erfc : Complementary error function.
    """
    return np.asarray(sp.erfc(x), dtype=np.float64)


def erfcx(x: ArrayLike) -> NDArray[np.floating]:
    """
    Scaled complementary error function.

    Computes erfcx(x) = exp(x²) * erfc(x).

    Parameters
    ----------
    x : array_like
        Argument of the function.

    Returns
    -------
    y : ndarray
        Values of erfcx(x).

    Notes
    -----
    This function is useful when erfc(x) underflows but the scaled
    version remains representable.

    See Also
    --------
    scipy.special.erfcx : Scaled complementary error function.
    """
    return np.asarray(sp.erfcx(x), dtype=np.float64)


def erfi(x: ArrayLike) -> NDArray[np.floating]:
    """
    Imaginary error function.

    Computes erfi(x) = -i * erf(i*x) = (2/√π) * ∫_0^x e^(t²) dt.

    Parameters
    ----------
    x : array_like
        Argument of the function.

    Returns
    -------
    y : ndarray
        Values of erfi(x).

    See Also
    --------
    scipy.special.erfi : Imaginary error function.
    """
    return np.asarray(sp.erfi(x), dtype=np.float64)


def erfinv(y: ArrayLike) -> NDArray[np.floating]:
    """
    Inverse error function.

    Finds x such that erf(x) = y.

    Parameters
    ----------
    y : array_like
        Values in the range (-1, 1).

    Returns
    -------
    x : ndarray
        Inverse error function values.

    Examples
    --------
    >>> erfinv(0)
    0.0
    >>> erf(erfinv(0.5))
    0.5

    See Also
    --------
    scipy.special.erfinv : Inverse error function.
    """
    return np.asarray(sp.erfinv(y), dtype=np.float64)


def erfcinv(y: ArrayLike) -> NDArray[np.floating]:
    """
    Inverse complementary error function.

    Finds x such that erfc(x) = y.

    Parameters
    ----------
    y : array_like
        Values in the range (0, 2).

    Returns
    -------
    x : ndarray
        Inverse complementary error function values.

    See Also
    --------
    scipy.special.erfcinv : Inverse complementary error function.
    """
    return np.asarray(sp.erfcinv(y), dtype=np.float64)


def dawsn(x: ArrayLike) -> NDArray[np.floating]:
    """
    Dawson's integral.

    Computes F(x) = exp(-x²) * ∫_0^x exp(t²) dt.

    Parameters
    ----------
    x : array_like
        Argument of Dawson's integral.

    Returns
    -------
    F : ndarray
        Values of Dawson's integral.

    Notes
    -----
    Dawson's integral is related to the imaginary error function by:
    F(x) = (√π/2) * exp(-x²) * erfi(x)

    See Also
    --------
    scipy.special.dawsn : Dawson's integral.
    """
    return np.asarray(sp.dawsn(x), dtype=np.float64)


def fresnel(x: ArrayLike) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Fresnel integrals.

    Computes the Fresnel sine and cosine integrals:
    S(x) = ∫_0^x sin(π*t²/2) dt
    C(x) = ∫_0^x cos(π*t²/2) dt

    Parameters
    ----------
    x : array_like
        Argument of the Fresnel integrals.

    Returns
    -------
    S : ndarray
        Fresnel sine integral.
    C : ndarray
        Fresnel cosine integral.

    See Also
    --------
    scipy.special.fresnel : Fresnel integrals.
    """
    S, C = sp.fresnel(x)
    return np.asarray(S, dtype=np.float64), np.asarray(C, dtype=np.float64)


def wofz(z: ArrayLike) -> NDArray[np.complexfloating]:
    """
    Faddeeva function.

    Computes w(z) = exp(-z²) * erfc(-i*z).

    Parameters
    ----------
    z : array_like
        Argument (can be complex).

    Returns
    -------
    w : ndarray
        Complex Faddeeva function values.

    Notes
    -----
    This function is useful in spectral line modeling and plasma physics.

    See Also
    --------
    scipy.special.wofz : Faddeeva function.
    """
    return np.asarray(sp.wofz(z), dtype=np.complex128)


def voigt_profile(x: ArrayLike, sigma: float, gamma: float) -> NDArray[np.floating]:
    """
    Voigt profile.

    The Voigt profile is a convolution of a Gaussian and Lorentzian profile,
    commonly used in spectroscopy and line shape analysis.

    Parameters
    ----------
    x : array_like
        Position parameter.
    sigma : float
        Standard deviation of the Gaussian component.
    gamma : float
        Half-width at half-maximum of the Lorentzian component.

    Returns
    -------
    V : ndarray
        Voigt profile values (normalized to unit area).

    See Also
    --------
    scipy.special.voigt_profile : Voigt profile.
    """
    return np.asarray(sp.voigt_profile(x, sigma, gamma), dtype=np.float64)


__all__ = [
    "erf",
    "erfc",
    "erfcx",
    "erfi",
    "erfinv",
    "erfcinv",
    "dawsn",
    "fresnel",
    "wofz",
    "voigt_profile",
]
