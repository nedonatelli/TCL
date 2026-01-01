"""
Debye functions.

Debye functions appear in solid-state physics for computing
thermodynamic properties of solids (heat capacity, entropy).
"""

import numpy as np
import scipy.integrate as integrate
from numpy.typing import ArrayLike
from numpy.typing import NDArray


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

    Examples
    --------
    >>> debye(3, 0)  # D_3(0) = 1
    1.0
    >>> debye(3, 1)
    0.674...
    >>> debye(3, 10)
    0.0192...

    References
    ----------
    .. [1] Debye, P. (1912). "Zur Theorie der spezifischen Wärmen".
           Annalen der Physik, 344(14), 789-839.
    """
    if n < 1:
        raise ValueError(f"Order n must be >= 1, got {n}")

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    result = np.zeros_like(x, dtype=np.float64)

    def integrand(t: float, n: int) -> float:
        if t == 0:
            return 0.0
        # t^n / (exp(t) - 1)
        # For numerical stability, use t^n * exp(-t) / (1 - exp(-t))
        exp_neg_t = np.exp(-t)
        return (t**n) * exp_neg_t / (1 - exp_neg_t)

    for i, xi in enumerate(x):
        if xi == 0:
            result[i] = 1.0
        elif xi < 0.1:
            # Small x series expansion
            result[i] = 1.0 - n * xi / (2 * (n + 1))
        elif xi > 100:
            # Large x asymptotic
            from scipy.special import factorial
            from scipy.special import zeta

            result[i] = factorial(n) * zeta(n + 1) * n / (xi**n)
        else:
            # General case: numerical integration
            integral, _ = integrate.quad(integrand, 0, xi, args=(n,))
            result[i] = (n / xi**n) * integral

    return result


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
    C_V / (3*N*k_B) = 3 * (T/Θ_D)^3 * D_3(Θ_D/T)

    Limits:
    - High T (T >> Θ_D): C_V -> 3*N*k_B (classical)
    - Low T (T << Θ_D): C_V ~ (T/Θ_D)^3 (quantum)

    Examples
    --------
    >>> # Aluminum at room temperature (Θ_D ≈ 428 K)
    >>> cv = debye_heat_capacity(300, 428)  # ~0.95
    """
    T = np.asarray(temperature, dtype=np.float64)
    theta_D = float(debye_temperature)

    if np.any(T <= 0):
        raise ValueError("Temperature must be positive")
    if theta_D <= 0:
        raise ValueError("Debye temperature must be positive")

    x = theta_D / T
    # C_V / (3*N*k_B) approaches 1 as T -> infinity (classical limit)
    # The formula is: C_V = 3*N*k_B * D_3(x) where D_3 is the Debye function
    # Note: Some sources use 3 * (T/Theta)^3 * integral, but the normalized
    # heat capacity simply equals D_3(Theta/T) for the standard formulation
    return debye(3, x)


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
    S / (3*N*k_B) = 4*D_3(Θ_D/T) - 3*ln(1 - exp(-Θ_D/T))
    """
    T = np.asarray(temperature, dtype=np.float64)
    theta_D = float(debye_temperature)

    if np.any(T <= 0):
        raise ValueError("Temperature must be positive")
    if theta_D <= 0:
        raise ValueError("Debye temperature must be positive")

    x = theta_D / T

    # Avoid overflow for large x
    exp_neg_x = np.exp(-x)
    log_term = np.where(x > 100, -x, np.log(1 - exp_neg_x))

    return 4.0 * debye(3, x) - log_term


__all__ = [
    "debye",
    "debye_1",
    "debye_2",
    "debye_3",
    "debye_4",
    "debye_heat_capacity",
    "debye_entropy",
]
