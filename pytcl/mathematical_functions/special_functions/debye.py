"""
Debye functions.

Debye functions appear in solid-state physics for computing
thermodynamic properties of solids (heat capacity, entropy).

Performance
-----------
This module uses Numba JIT compilation with rapidly convergent series
expansions (Abramowitz & Stegun 27.1.1-27.1.3), providing high accuracy
(~1e-14 relative) and ~10-50x speedup for batch computations compared to
scipy.integrate.quad.
"""

from typing import Any

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy.special import zeta

# Pre-compute zeta values for common orders (n=1 to 10)
_ZETA_VALUES = np.array([zeta(k + 1) for k in range(11)])

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


@njit(cache=True)
def _debye_large_x(x: float, n: int, n_fact: float, zeta_n_plus_1: float) -> float:
    """
    Complement series for x >= 1 (A&S 27.1.2-27.1.3).

    D_n(x) = (n/x^n) * [n! * zeta(n+1)
             - sum_{j>=1} e^{-jx} * (n!/j^{n+1}) * sum_{i=0}^{n} (jx)^i/i!]
    """
    total = n_fact * zeta_n_plus_1
    for j in range(1, 500):
        jx = j * x
        if jx > 700.0:
            break
        # Partial exponential sum: sum_{i=0}^{n} (jx)^i / i!
        s = 1.0
        term = 1.0
        for i in range(1, n + 1):
            term *= jx / i
            s += term
        contrib = np.exp(-jx) * n_fact / float(j) ** (n + 1) * s
        total -= contrib
        if contrib < 1e-17 * total:
            break
    return n / x**n * total


@njit(cache=True, parallel=True)
def _debye_batch(
    n: int,
    x_arr: np.ndarray[Any, Any],
    zeta_n_plus_1: float,
    coef: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """
    Batch computation of Debye function for array input.

    Parameters
    ----------
    n : int
        Order of the Debye function.
    x_arr : ndarray
        Array of x values.
    zeta_n_plus_1 : float
        Pre-computed zeta(n+1) value.
    coef : ndarray
        Bernoulli coefficients B_{2k}/(2k)! for the small-x series.

    Returns
    -------
    ndarray
        Debye function values.
    """
    result = np.empty(len(x_arr), dtype=np.float64)
    n_fact = 1.0
    for k in range(1, n + 1):
        n_fact *= k

    for i in prange(len(x_arr)):
        xi = x_arr[i]
        if xi == 0.0:
            result[i] = 1.0
        elif xi < 1.0:
            result[i] = _debye_small_x(xi, n, coef)
        else:
            result[i] = _debye_large_x(xi, n, n_fact, zeta_n_plus_1)

    return result


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
        Order of the Debye function (positive integer).
    x : array_like
        Argument of the function, x >= 0.

    Returns
    -------
    D : ndarray
        Values of D_n(x).

    Notes
    -----
    Special cases:
    - D_n(0) = 1
    - D_n(inf) = n! * zeta(n+1) / x^n -> 0

    The Debye function D_3(x) appears in the heat capacity
    of solids at low temperatures.

    This implementation uses Numba JIT compilation for performance,
    achieving ~10-50x speedup compared to scipy.integrate.quad for
    batch computations.

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
    if n < 1:
        raise ValueError(f"Order n must be >= 1, got {n}")

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))

    # Get pre-computed zeta value if available, otherwise compute
    if n < len(_ZETA_VALUES):
        zeta_n_plus_1 = _ZETA_VALUES[n]
    else:
        zeta_n_plus_1 = zeta(n + 1)

    return _debye_batch(n, x, zeta_n_plus_1, _BERNOULLI_COEF)


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
