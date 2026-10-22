"""
Debye functions.

Debye functions appear in solid-state physics for computing
thermodynamic properties of solids (heat capacity, entropy).

Method
------
Three regimes, each free of cancellation. For x < 1 the Bernoulli series of
Abramowitz & Stegun 27.1.1 (Numba JIT). For 1 <= x < n + 2*sqrt(n) + 5 a
composite Gauss-Legendre rule on D_n(x) = n * int_0^1 s^(n-1) g(s*x) ds with
g(y) = y/expm1(y), whose terms are all positive. Beyond that the integral is
past the bulk of its mass, which peaks near t = n, and the tail form
n * n!/x^n * [zeta(n+1) - sum_j Q(n+1, j*x)/j^(n+1)] with Q the regularized
upper incomplete gamma function subtracts a sum below ~5% of zeta(n+1).

Accuracy
--------
Measured against a 40-digit mpmath quadrature of the Debye integrand for
every integer n from 1 to 10 and n in {12, 15, 20, 25, 30, 35, 40, 45, 50},
on 21 x values between 1e-6 and 700 plus each side of both regime
boundaries (x = 1 and x = n + 2*sqrt(n) + 5). The worst relative error on
that grid is 5.7e-14 (n = 45, x = 700), so the claimed bound is about 1e-13,
on that grid only. n = 60 measured 7.2e-14 at x = 300 but is outside the
claim. The previous large-x branch subtracted same-magnitude terms and was
wrong from n = 10 up: D_20(2) was 3.9% off and D_30(2) was 2.2e8 against
0.3267614.
"""

from typing import Any

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincc, gammaln, zeta

# B_{2k} / (2k)! for k = 1..10 (Bernoulli numbers over factorials), used in
# the small-x expansion of t/(e^t - 1) = 1 - t/2 + sum B_{2k} t^{2k}/(2k)!
_BERNOULLI_COEF = np.array(
    [
        1.0 / 12.0,
        -1.0 / 720.0,
        1.0 / 30240.0,
        -1.0 / 1209600.0,
        1.0 / 47900160.0,
        -691.0 / 1307674368000.0,
        1.0 / 74724249600.0,
        -3617.0 / 10670622842880000.0,
        43867.0 / 5109094217170944000.0,
        -174611.0 / 802857662698291200000.0,
    ]
)


@njit(cache=True)
def _debye_small_x(x: float, n: int, coef: np.ndarray[Any, Any]) -> float:
    """
    Bernoulli series expansion for x < 1 (converges for |x| < 2*pi).

    D_n(x) = 1 - n*x/(2*(n+1)) + n * sum_k B_{2k}/(2k)! * x^{2k}/(2k+n)
    """
    result = 1.0 - n * x / (2.0 * (n + 1))
    x2 = x * x
    xp = 1.0
    for k in range(len(coef)):
        xp *= x2
        result += n * coef[k] * xp / (2 * (k + 1) + n)
    return result


@njit(cache=True, parallel=True)
def _debye_small_batch(
    n: int,
    x_arr: np.ndarray[Any, Any],
    coef: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Bernoulli series for an array of x, each in [0, _X_SMALL)."""
    result = np.empty(len(x_arr), dtype=np.float64)
    for i in prange(len(x_arr)):  # ty: ignore[not-iterable]
        result[i] = _debye_small_x(x_arr[i], n, coef)
    return result


_X_SMALL = 1.0
_PANEL_T = 4.0
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _debye_quadrature(n: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Composite Gauss-Legendre on D_n(x) = n * int_0^1 s^(n-1) g(sx) ds.

    g(y) = y / expm1(y) is positive and smooth (nearest singularity at
    y = 2*pi*i), so every term is positive and nothing cancels. Panels are at
    most ``_PANEL_T`` long in t = s*x, well inside the 2*pi pole distance.
    """
    out = np.empty_like(x)
    npanel = np.ceil(x / _PANEL_T).astype(np.int64)
    for p in np.unique(npanel):
        sel = npanel == p
        xs = x[sel][:, None, None]
        lo = (np.arange(p) / p)[None, :, None]
        s = lo + (_GL_NODES[None, None, :] + 1.0) / (2.0 * p)
        y = s * xs
        integrand = s ** (n - 1) * (y / np.expm1(y))
        out[sel] = (
            n * (integrand * _GL_WEIGHTS[None, None, :]).sum(axis=(1, 2)) / (2.0 * p)
        )
    return out


def _debye_complement(n: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tail form for x past the bulk of t^n/(e^t - 1), which peaks near t = n.

    D_n(x) = n * n! / x^n * [zeta(n+1) - sum_j Q(n+1, j*x) / j^(n+1)] with Q
    the regularized upper incomplete gamma function. The subtracted sum is
    below ~5% of zeta(n+1) in the region this is used, so it loses at most
    one digit.
    """
    zeta_np1 = float(zeta(n + 1))
    tail = np.zeros_like(x)
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(1, 2000):
            term = gammaincc(n + 1, j * x) / float(j) ** (n + 1)
            tail += term
            if np.all(term <= 1e-17 * zeta_np1):
                break
    with np.errstate(divide="ignore", over="ignore"):
        scale = n * np.exp(gammaln(n + 1) - n * np.log(x))
    return scale * (zeta_np1 - tail)


def _x_complement(n: int) -> float:
    return n + 2.0 * np.sqrt(n) + 5.0


def debye(
    n: int,
    x: ArrayLike,
) -> NDArray[np.floating]:
    """
    Debye function D_n(x).

    The Debye function of order n is defined as:
    D_n(x) = (n/x^n) * integral from 0 to x of t^n / (exp(t) - 1) dt

    Parameters
    ----------
    n : int
        Order of the Debye function (positive integer). Accuracy is measured
        on a grid of orders up to 50 (see the module docstring); other
        orders are not verified.
    x : array_like
        Argument of the function, x >= 0.

    Returns
    -------
    D : ndarray
        Values of D_n(x).

    Raises
    ------
    ValueError
        If n is not a positive integer or any x is negative.

    Notes
    -----
    Special cases:
    - D_n(0) = 1
    - D_n(inf) = n! * zeta(n+1) / x^n -> 0

    The Debye function D_3(x) appears in the heat capacity
    of solids at low temperatures.

    Examples
    --------
    >>> float(debye(3, 0)[0])  # D_3(0) = 1
    1.0
    >>> round(float(debye(3, 1)[0]), 6)
    0.674416
    >>> round(float(debye(3, 10)[0]), 6)
    0.019296

    References
    ----------
    - Debye, P. (1912). "Zur Theorie der spezifischen Wärmen".
      Annalen der Physik, 344(14), 789-839.
    """
    if int(n) != n or n < 1:
        raise ValueError(f"Order n must be >= 1 and an integer, got {n}")
    n = int(n)

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if np.any(x < 0):
        raise ValueError("x must be >= 0")

    result = np.full(x.shape, np.nan)
    flat = x.ravel()
    out = result.ravel()
    x_switch = _x_complement(n)

    out[flat == 0.0] = 1.0
    small = (flat > 0.0) & (flat < _X_SMALL)
    if small.any():
        out[small] = _debye_small_batch(n, flat[small], _BERNOULLI_COEF)
    mid = (flat >= _X_SMALL) & (flat < x_switch)
    if mid.any():
        out[mid] = _debye_quadrature(n, flat[mid])
    large = (flat >= x_switch) & np.isfinite(flat)
    if large.any():
        out[large] = _debye_complement(n, flat[large])
    out[np.isposinf(flat)] = 0.0
    return out.reshape(x.shape)


def debye_1(x: ArrayLike) -> NDArray[np.floating]:
    """
    First-order Debye function D_1(x).

    Parameters
    ----------
    x : array_like
        Argument of the function, x >= 0.

    Returns
    -------
    D : ndarray
        Values of D_1(x).

    Notes
    -----
    D_1(x) = (1/x) * integral from 0 to x of t / (exp(t) - 1) dt
    """
    return debye(1, x)


def debye_2(x: ArrayLike) -> NDArray[np.floating]:
    """
    Second-order Debye function D_2(x).

    Parameters
    ----------
    x : array_like
        Argument of the function, x >= 0.

    Returns
    -------
    D : ndarray
        Values of D_2(x).

    Notes
    -----
    D_2(x) = (2/x^2) * integral from 0 to x of t^2 / (exp(t) - 1) dt
    """
    return debye(2, x)


def debye_3(x: ArrayLike) -> NDArray[np.floating]:
    """
    Third-order Debye function D_3(x).

    This is the most commonly used Debye function, appearing in
    the heat capacity of solids.

    Parameters
    ----------
    x : array_like
        Argument of the function, x >= 0.

    Returns
    -------
    D : ndarray
        Values of D_3(x).

    Notes
    -----
    D_3(x) = (3/x^3) * integral from 0 to x of t^3 / (exp(t) - 1) dt

    The heat capacity of a solid in the Debye model is:
    C_V = 9 * N * k_B * (T/Θ_D)^3 * D_3(Θ_D/T)

    where Θ_D is the Debye temperature.
    """
    return debye(3, x)


def debye_4(x: ArrayLike) -> NDArray[np.floating]:
    """
    Fourth-order Debye function D_4(x).

    Parameters
    ----------
    x : array_like
        Argument of the function, x >= 0.

    Returns
    -------
    D : ndarray
        Values of D_4(x).

    Notes
    -----
    D_4(x) = (4/x^4) * integral from 0 to x of t^4 / (exp(t) - 1) dt

    This appears in computing the entropy of solids.
    """
    return debye(4, x)


def debye_heat_capacity(
    temperature: ArrayLike,
    debye_temperature: float,
) -> NDArray[np.floating]:
    """
    Debye model heat capacity (normalized).

    Computes C_V / (3*N*k_B) using the Debye model.

    Parameters
    ----------
    temperature : array_like
        Temperature in Kelvin.
    debye_temperature : float
        Debye temperature Θ_D in Kelvin.

    Returns
    -------
    cv_normalized : ndarray
        Normalized heat capacity C_V / (3*N*k_B).
        Multiply by 3*N*k_B for actual heat capacity.

    Notes
    -----
    The Debye model heat capacity is:
    C_V / (3*N*k_B) = 4*D_3(x) - 3*x/(e^x - 1), with x = Θ_D/T

    Limits:
    - High T (T >> Θ_D): C_V -> 3*N*k_B (classical)
    - Low T (T << Θ_D): C_V ~ (4*π^4/5) * (T/Θ_D)^3 (quantum)

    Examples
    --------
    >>> # Aluminum at room temperature (Θ_D ≈ 428 K)
    >>> cv = debye_heat_capacity(300, 428)  # ~0.91
    """
    T = np.asarray(temperature, dtype=np.float64)
    theta_D = float(debye_temperature)

    if np.any(T <= 0):
        raise ValueError("Temperature must be positive")
    if theta_D <= 0:
        raise ValueError("Debye temperature must be positive")

    x = np.atleast_1d(theta_D / T)
    # C_V / (3*N*k_B) = 4*D_3(x) - 3*x/(e^x - 1)
    # (obtained by integrating the Debye phonon spectrum by parts)
    with np.errstate(over="ignore"):
        boltzmann_term = np.where(x > 500, 0.0, 3.0 * x / np.expm1(np.minimum(x, 700)))
    return 4.0 * debye(3, x) - boltzmann_term


def debye_entropy(
    temperature: ArrayLike,
    debye_temperature: float,
) -> NDArray[np.floating]:
    """
    Debye model entropy (normalized).

    Computes S / (3*N*k_B) using the Debye model.

    Parameters
    ----------
    temperature : array_like
        Temperature in Kelvin.
    debye_temperature : float
        Debye temperature Θ_D in Kelvin.

    Returns
    -------
    s_normalized : ndarray
        Normalized entropy S / (3*N*k_B).

    Notes
    -----
    The entropy in the Debye model is:
    S / (3*N*k_B) = (4/3)*D_3(Θ_D/T) - ln(1 - exp(-Θ_D/T))
    """
    T = np.asarray(temperature, dtype=np.float64)
    theta_D = float(debye_temperature)

    if np.any(T <= 0):
        raise ValueError("Temperature must be positive")
    if theta_D <= 0:
        raise ValueError("Debye temperature must be positive")

    x = np.atleast_1d(theta_D / T)

    # log1p(-e^{-x}) is accurate for all x > 0, including large x where
    # e^{-x} underflows harmlessly to 0
    log_term = np.log1p(-np.exp(-np.minimum(x, 700)))

    return (4.0 / 3.0) * debye(3, x) - log_term


__all__ = [
    "debye",
    "debye_1",
    "debye_2",
    "debye_3",
    "debye_4",
    "debye_heat_capacity",
    "debye_entropy",
]
