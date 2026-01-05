"""
Elliptic integrals and functions.

This module provides elliptic integrals used in various physical
applications including orbits, pendulums, and electromagnetic calculations.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


def ellipk(m: ArrayLike) -> NDArray[np.floating]:
    """
    Complete elliptic integral of the first kind.

    Computes K(m) = ∫_0^(π/2) (1 - m*sin²(θ))^(-1/2) dθ.

    Parameters
    ----------
    m : array_like
        Parameter m (not the modulus k). Note: m = k².
        Must be in [0, 1).

    Returns
    -------
    K : ndarray
        Values of the complete elliptic integral of the first kind.

    Notes
    -----
    As m → 1, K(m) → ∞.

    Examples
    --------
    >>> ellipk(0)  # K(0) = π/2
    1.5707963267948966
    >>> ellipk(0.5)
    1.8540746773013719

    See Also
    --------
    scipy.special.ellipk : Complete elliptic integral of first kind.
    """
    return np.asarray(sp.ellipk(m), dtype=np.float64)


def ellipkm1(p: ArrayLike) -> NDArray[np.floating]:
    """
    Complete elliptic integral of the first kind around m = 1.

    Computes K(1 - p) for small p, more accurate than ellipk(1 - p).

    Parameters
    ----------
    p : array_like
        Parameter p = 1 - m.

    Returns
    -------
    K : ndarray
        Values of K(1 - p).

    Examples
    --------
    >>> ellipkm1(0.1)  # K(0.9)
    2.578...

    See Also
    --------
    scipy.special.ellipkm1 : Elliptic integral near m = 1.
    """
    return np.asarray(sp.ellipkm1(p), dtype=np.float64)


def ellipe(m: ArrayLike) -> NDArray[np.floating]:
    """
    Complete elliptic integral of the second kind.

    Computes E(m) = ∫_0^(π/2) (1 - m*sin²(θ))^(1/2) dθ.

    Parameters
    ----------
    m : array_like
        Parameter m (not the modulus k). Note: m = k².
        Must be in [0, 1].

    Returns
    -------
    E : ndarray
        Values of the complete elliptic integral of the second kind.

    Examples
    --------
    >>> ellipe(0)  # E(0) = π/2
    1.5707963267948966
    >>> ellipe(1)  # E(1) = 1
    1.0

    See Also
    --------
    scipy.special.ellipe : Complete elliptic integral of second kind.
    """
    return np.asarray(sp.ellipe(m), dtype=np.float64)


def ellipeinc(phi: ArrayLike, m: ArrayLike) -> NDArray[np.floating]:
    """
    Incomplete elliptic integral of the second kind.

    Computes E(φ, m) = ∫_0^φ (1 - m*sin²(θ))^(1/2) dθ.

    Parameters
    ----------
    phi : array_like
        Amplitude (in radians).
    m : array_like
        Parameter m = k².

    Returns
    -------
    E : ndarray
        Values of the incomplete elliptic integral of the second kind.

    Examples
    --------
    >>> import numpy as np
    >>> ellipeinc(np.pi/2, 0)  # Same as ellipe(0) = π/2
    1.5707...

    See Also
    --------
    scipy.special.ellipeinc : Incomplete elliptic integral of second kind.
    """
    return np.asarray(sp.ellipeinc(phi, m), dtype=np.float64)


def ellipkinc(phi: ArrayLike, m: ArrayLike) -> NDArray[np.floating]:
    """
    Incomplete elliptic integral of the first kind.

    Computes F(φ, m) = ∫_0^φ (1 - m*sin²(θ))^(-1/2) dθ.

    Parameters
    ----------
    phi : array_like
        Amplitude (in radians).
    m : array_like
        Parameter m = k².

    Returns
    -------
    F : ndarray
        Values of the incomplete elliptic integral of the first kind.

    Examples
    --------
    >>> import numpy as np
    >>> ellipkinc(np.pi/2, 0)  # Same as ellipk(0) = π/2
    1.5707...

    See Also
    --------
    scipy.special.ellipkinc : Incomplete elliptic integral of first kind.
    """
    return np.asarray(sp.ellipkinc(phi, m), dtype=np.float64)


def elliprd(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.floating]:
    """
    Carlson symmetric elliptic integral R_D.

    Computes the symmetric elliptic integral:
    R_D(x, y, z) = (3/2) ∫_0^∞ [(t+x)(t+y)]^(-1/2) (t+z)^(-3/2) dt

    Parameters
    ----------
    x : array_like
        First argument (non-negative).
    y : array_like
        Second argument (non-negative).
    z : array_like
        Third argument (positive).

    Returns
    -------
    R_D : ndarray
        Values of the Carlson R_D integral.

    Examples
    --------
    >>> elliprd(1, 2, 3)
    0.297...

    See Also
    --------
    scipy.special.elliprd : Carlson R_D integral.
    """
    return np.asarray(sp.elliprd(x, y, z), dtype=np.float64)


def elliprf(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.floating]:
    """
    Carlson symmetric elliptic integral R_F.

    Computes the symmetric elliptic integral:
    R_F(x, y, z) = (1/2) ∫_0^∞ [(t+x)(t+y)(t+z)]^(-1/2) dt

    Parameters
    ----------
    x : array_like
        First argument (non-negative).
    y : array_like
        Second argument (non-negative).
    z : array_like
        Third argument (non-negative).
        At most one of x, y, z can be zero.

    Returns
    -------
    R_F : ndarray
        Values of the Carlson R_F integral.

    Notes
    -----
    The complete elliptic integral of the first kind is:
    K(m) = R_F(0, 1-m, 1)

    Examples
    --------
    >>> elliprf(1, 1, 1)  # R_F(a, a, a) = 1/sqrt(a)
    1.0

    See Also
    --------
    scipy.special.elliprf : Carlson R_F integral.
    """
    return np.asarray(sp.elliprf(x, y, z), dtype=np.float64)


def elliprg(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.floating]:
    """
    Carlson symmetric elliptic integral R_G.

    Computes the symmetric elliptic integral R_G(x, y, z).

    Parameters
    ----------
    x : array_like
        First argument (non-negative).
    y : array_like
        Second argument (non-negative).
    z : array_like
        Third argument (non-negative).

    Returns
    -------
    R_G : ndarray
        Values of the Carlson R_G integral.

    Notes
    -----
    The complete elliptic integral of the second kind is:
    E(m) = 2 * R_G(0, 1-m, 1)

    Examples
    --------
    >>> elliprg(1, 1, 1)  # R_G(a, a, a) = sqrt(a)
    1.0

    See Also
    --------
    scipy.special.elliprg : Carlson R_G integral.
    """
    return np.asarray(sp.elliprg(x, y, z), dtype=np.float64)


def elliprj(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, p: ArrayLike
) -> NDArray[np.floating]:
    """
    Carlson symmetric elliptic integral R_J.

    Computes the symmetric elliptic integral:
    R_J(x, y, z, p) = (3/2) ∫_0^∞ [(t+x)(t+y)(t+z)]^(-1/2) (t+p)^(-1) dt

    Parameters
    ----------
    x : array_like
        First argument (non-negative).
    y : array_like
        Second argument (non-negative).
    z : array_like
        Third argument (non-negative).
    p : array_like
        Fourth argument (non-zero).

    Returns
    -------
    R_J : ndarray
        Values of the Carlson R_J integral.

    Notes
    -----
    The complete elliptic integral of the third kind can be computed using R_J.

    Examples
    --------
    >>> elliprj(1, 2, 3, 4)
    0.213...

    See Also
    --------
    scipy.special.elliprj : Carlson R_J integral.
    """
    return np.asarray(sp.elliprj(x, y, z, p), dtype=np.float64)


def elliprc(x: ArrayLike, y: ArrayLike) -> NDArray[np.floating]:
    """
    Carlson degenerate elliptic integral R_C.

    Computes R_C(x, y) = R_F(x, y, y).

    Parameters
    ----------
    x : array_like
        First argument (non-negative).
    y : array_like
        Second argument (non-zero).

    Returns
    -------
    R_C : ndarray
        Values of the Carlson R_C integral.

    Notes
    -----
    - R_C(x, y) = arctan(sqrt((x-y)/y)) / sqrt(x-y) for x > y
    - R_C(x, y) = arctanh(sqrt((y-x)/y)) / sqrt(y-x) for x < y

    Examples
    --------
    >>> elliprc(1, 1)  # R_C(a, a) = 1/sqrt(a)
    1.0

    See Also
    --------
    scipy.special.elliprc : Carlson R_C integral.
    """
    return np.asarray(sp.elliprc(x, y), dtype=np.float64)


__all__ = [
    "ellipk",
    "ellipkm1",
    "ellipe",
    "ellipeinc",
    "ellipkinc",
    "elliprd",
    "elliprf",
    "elliprg",
    "elliprj",
    "elliprc",
]
