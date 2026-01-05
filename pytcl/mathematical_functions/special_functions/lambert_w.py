"""
Lambert W function and related functions.

The Lambert W function appears in diverse applications including
delay differential equations, combinatorics, and physics.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


def lambert_w(
    z: ArrayLike,
    k: int = 0,
    tol: float = 1e-10,
) -> NDArray[np.complexfloating]:
    """
    Lambert W function W_k(z).

    The Lambert W function is defined as the inverse of f(w) = w * exp(w),
    satisfying W(z) * exp(W(z)) = z.

    Parameters
    ----------
    z : array_like
        Argument of the function. Can be complex.
    k : int, optional
        Branch index. Default is 0 (principal branch).
        - k = 0: Principal branch, real for z >= -1/e
        - k = -1: Lower branch, real for -1/e <= z < 0
        - Other k: Complex branches
    tol : float, optional
        Tolerance for convergence (used in edge cases). Default is 1e-10.

    Returns
    -------
    W : ndarray
        Values of W_k(z).

    Notes
    -----
    The principal branch W_0(z) satisfies:
    - W_0(0) = 0
    - W_0(e) = 1
    - W_0(-1/e) = -1

    The function has a branch point at z = -1/e ≈ -0.3679.

    Examples
    --------
    >>> lambert_w(0)  # W(0) = 0
    0.0
    >>> lambert_w(np.e)  # W(e) = 1
    1.0
    >>> lambert_w(-np.exp(-1))  # W(-1/e) = -1
    -1.0

    References
    ----------
    .. [1] Corless, R.M., et al. (1996). "On the Lambert W Function".
           Advances in Computational Mathematics, 5, 329-359.
    """
    z = np.asarray(z)
    return np.asarray(sp.lambertw(z, k=k, tol=tol), dtype=np.complex128)


def lambert_w_real(
    x: ArrayLike,
    branch: int = 0,
) -> NDArray[np.floating]:
    """
    Real-valued Lambert W function.

    Returns only the real part of the Lambert W function for
    real inputs.

    Parameters
    ----------
    x : array_like
        Real argument. For branch 0: x >= -1/e. For branch -1: -1/e <= x < 0.
    branch : int, optional
        Branch index: 0 (principal) or -1 (lower). Default is 0.

    Returns
    -------
    W : ndarray
        Real values of W(x).

    Raises
    ------
    ValueError
        If x is outside the valid range for real-valued output.

    Examples
    --------
    >>> lambert_w_real(1)
    0.5671432904097838
    >>> lambert_w_real(-0.2, branch=-1)
    -2.5426413577735264
    """
    x = np.asarray(x, dtype=np.float64)

    branch_point = -np.exp(-1)

    if branch == 0:
        if np.any(x < branch_point):
            raise ValueError(f"For branch 0, x must be >= -1/e ≈ {branch_point:.6f}")
    elif branch == -1:
        if np.any((x < branch_point) | (x >= 0)):
            raise ValueError(
                f"For branch -1, x must be in [-1/e, 0) ≈ [{branch_point:.6f}, 0)"
            )
    else:
        raise ValueError(f"branch must be 0 or -1, got {branch}")

    result = sp.lambertw(x, k=branch)
    return np.real(result).astype(np.float64)


def omega_constant() -> float:
    """
    Omega constant (principal value of W(1)).

    The omega constant Ω is the unique real solution to
    Ω * exp(Ω) = 1, satisfying Ω = W_0(1).

    Returns
    -------
    omega : float
        Ω ≈ 0.5671432904097838729999686622...

    Notes
    -----
    The omega constant appears in:
    - Growth of the iterated logarithm
    - Stirling's approximation refinements
    - Analysis of tree structures

    Examples
    --------
    >>> omega = omega_constant()
    >>> omega * np.exp(omega)  # Should equal 1
    1.0
    """
    return float(np.real(sp.lambertw(1.0)))


def wright_omega(z: ArrayLike) -> NDArray[np.complexfloating]:
    """
    Wright omega function ω(z).

    The Wright omega function is defined as ω(z) = W_k(e^z) for
    the appropriate branch k.

    Parameters
    ----------
    z : array_like
        Argument of the function. Can be complex.

    Returns
    -------
    omega : ndarray
        Values of the Wright omega function.

    Notes
    -----
    The Wright omega function satisfies:
    ω(z) + log(ω(z)) = z

    It is entire (analytic everywhere) unlike the Lambert W function.

    Examples
    --------
    >>> wright_omega(0)  # Omega constant
    (0.5671...+0j)

    References
    ----------
    .. [1] Wright, E.M. (1959). "Solution of the equation z*exp(z) = a".
           Bull. Amer. Math. Soc., 65, 89-93.
    """
    z = np.asarray(z, dtype=np.complex128)

    # Wright omega is W(exp(z)) for appropriate branch
    # For most z, use principal branch
    return np.asarray(sp.lambertw(np.exp(z)), dtype=np.complex128)


def solve_exponential_equation(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
) -> NDArray[np.complexfloating]:
    """
    Solve a*x*exp(b*x) = c using Lambert W.

    Finds x such that a*x*exp(b*x) = c.

    Parameters
    ----------
    a : array_like
        Coefficient of x.
    b : array_like
        Coefficient in the exponential.
    c : array_like
        Right-hand side constant.

    Returns
    -------
    x : ndarray
        Solution(s) to the equation.

    Notes
    -----
    The solution is: x = W(b*c/a) / b

    Examples
    --------
    >>> x = solve_exponential_equation(1, 1, np.e)  # x*exp(x) = e
    >>> x  # Should be 1
    1.0
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)

    # a*x*exp(b*x) = c
    # Let u = b*x, then (a/b)*u*exp(u) = c
    # u*exp(u) = b*c/a
    # u = W(b*c/a)
    # x = u/b = W(b*c/a)/b

    arg = b * c / a
    w = sp.lambertw(arg)
    x = w / b

    return np.asarray(x, dtype=np.complex128)


def time_delay_equation(
    a: ArrayLike,
    tau: ArrayLike,
) -> NDArray[np.complexfloating]:
    """
    Solve characteristic equation for first-order delay system.

    Finds s such that s + a*exp(-s*tau) = 0, which appears
    in delay differential equations.

    Parameters
    ----------
    a : array_like
        Coefficient in the characteristic equation.
    tau : array_like
        Time delay.

    Returns
    -------
    s : ndarray
        Root(s) of the characteristic equation.

    Notes
    -----
    The solution is: s = W(a*tau)/tau

    This is the dominant eigenvalue for the delay system:
    dx/dt = -a * x(t - tau)

    Examples
    --------
    >>> s = time_delay_equation(1, 1)  # s + exp(-s) = 0
    >>> s + np.exp(-s)  # Should be approximately 0
    ~0
    """
    a = np.asarray(a, dtype=np.complex128)
    tau = np.asarray(tau, dtype=np.complex128)

    # s + a*exp(-s*tau) = 0
    # s = -a*exp(-s*tau)
    # s*exp(s*tau) = -a
    # Let u = s*tau: u*exp(u) = -a*tau
    # u = W(-a*tau)
    # s = u/tau = W(-a*tau)/tau

    w = sp.lambertw(-a * tau)
    s = w / tau

    return np.asarray(s, dtype=np.complex128)


__all__ = [
    "lambert_w",
    "lambert_w_real",
    "omega_constant",
    "wright_omega",
    "solve_exponential_equation",
    "time_delay_equation",
]
