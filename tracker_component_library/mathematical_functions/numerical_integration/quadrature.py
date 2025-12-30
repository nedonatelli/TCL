"""
Numerical integration (quadrature) methods.

This module provides Gaussian quadrature rules and numerical integration
functions commonly used in state estimation and filtering.
"""

from typing import Callable, Tuple, Optional, Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.integrate as integrate


def gauss_legendre(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gauss-Legendre quadrature points and weights.

    For integrating f(x) over [-1, 1]:
    ∫_{-1}^{1} f(x) dx ≈ Σ w_i * f(x_i)

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    x : ndarray
        Quadrature points of shape (n,).
    w : ndarray
        Quadrature weights of shape (n,).

    Examples
    --------
    >>> x, w = gauss_legendre(5)
    >>> # Integrate x^2 from -1 to 1 (exact = 2/3)
    >>> np.sum(w * x**2)
    0.6666666666666666

    See Also
    --------
    numpy.polynomial.legendre.leggauss : Equivalent function.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def gauss_hermite(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gauss-Hermite quadrature points and weights.

    For integrating f(x) * exp(-x^2) over (-∞, ∞):
    ∫_{-∞}^{∞} f(x) * exp(-x²) dx ≈ Σ w_i * f(x_i)

    Useful for expectations over Gaussian distributions.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    x : ndarray
        Quadrature points of shape (n,).
    w : ndarray
        Quadrature weights of shape (n,).

    Notes
    -----
    For computing E[f(X)] where X ~ N(μ, σ²):
    E[f(X)] = (1/√π) * Σ w_i * f(μ + √2 * σ * x_i)

    See Also
    --------
    numpy.polynomial.hermite.hermgauss : Equivalent function.
    """
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def gauss_laguerre(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gauss-Laguerre quadrature points and weights.

    For integrating f(x) * exp(-x) over [0, ∞):
    ∫_0^∞ f(x) * exp(-x) dx ≈ Σ w_i * f(x_i)

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    x : ndarray
        Quadrature points of shape (n,).
    w : ndarray
        Quadrature weights of shape (n,).

    See Also
    --------
    numpy.polynomial.laguerre.laggauss : Equivalent function.
    """
    x, w = np.polynomial.laguerre.laggauss(n)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def gauss_chebyshev(
    n: int,
    kind: Literal[1, 2] = 1,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gauss-Chebyshev quadrature points and weights.

    For kind=1, integrates f(x) / sqrt(1-x²) over [-1, 1].
    For kind=2, integrates f(x) * sqrt(1-x²) over [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points.
    kind : {1, 2}, optional
        Type of Chebyshev polynomial. Default is 1.

    Returns
    -------
    x : ndarray
        Quadrature points of shape (n,).
    w : ndarray
        Quadrature weights of shape (n,).

    See Also
    --------
    numpy.polynomial.chebyshev.chebgauss : Type 1 Chebyshev.
    """
    if kind == 1:
        x, w = np.polynomial.chebyshev.chebgauss(n)
    elif kind == 2:
        # Chebyshev type 2
        k = np.arange(1, n + 1)
        x = np.cos(k * np.pi / (n + 1))
        w = np.pi / (n + 1) * np.sin(k * np.pi / (n + 1)) ** 2
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")

    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    **kwargs,
) -> Tuple[float, float]:
    """
    Adaptive quadrature integration.

    Computes ∫_a^b f(x) dx using adaptive Gaussian quadrature.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit.
    b : float
        Upper limit.
    **kwargs
        Additional arguments passed to scipy.integrate.quad.

    Returns
    -------
    result : float
        Estimated integral value.
    error : float
        Estimate of the absolute error.

    Examples
    --------
    >>> result, error = quad(lambda x: x**2, 0, 1)
    >>> result
    0.33333333333333337

    See Also
    --------
    scipy.integrate.quad : Underlying implementation.
    """
    result, error = integrate.quad(f, a, b, **kwargs)
    return float(result), float(error)


def dblquad(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    gfun: Callable[[float], float],
    hfun: Callable[[float], float],
    **kwargs,
) -> Tuple[float, float]:
    """
    Double integration.

    Computes ∫_a^b ∫_{g(x)}^{h(x)} f(y, x) dy dx.

    Parameters
    ----------
    f : callable
        Function f(y, x) to integrate.
    a : float
        Lower limit of x.
    b : float
        Upper limit of x.
    gfun : callable
        Lower limit of y as function of x.
    hfun : callable
        Upper limit of y as function of x.
    **kwargs
        Additional arguments passed to scipy.integrate.dblquad.

    Returns
    -------
    result : float
        Estimated integral value.
    error : float
        Estimate of the absolute error.

    See Also
    --------
    scipy.integrate.dblquad : Underlying implementation.
    """
    result, error = integrate.dblquad(f, a, b, gfun, hfun, **kwargs)
    return float(result), float(error)


def tplquad(
    f: Callable[[float, float, float], float],
    a: float,
    b: float,
    gfun: Callable[[float], float],
    hfun: Callable[[float], float],
    qfun: Callable[[float, float], float],
    rfun: Callable[[float, float], float],
    **kwargs,
) -> Tuple[float, float]:
    """
    Triple integration.

    Computes ∫_a^b ∫_{g(x)}^{h(x)} ∫_{q(x,y)}^{r(x,y)} f(z, y, x) dz dy dx.

    Parameters
    ----------
    f : callable
        Function f(z, y, x) to integrate.
    a : float
        Lower limit of x.
    b : float
        Upper limit of x.
    gfun : callable
        Lower limit of y as function of x.
    hfun : callable
        Upper limit of y as function of x.
    qfun : callable
        Lower limit of z as function of x, y.
    rfun : callable
        Upper limit of z as function of x, y.
    **kwargs
        Additional arguments passed to scipy.integrate.tplquad.

    Returns
    -------
    result : float
        Estimated integral value.
    error : float
        Estimate of the absolute error.

    See Also
    --------
    scipy.integrate.tplquad : Underlying implementation.
    """
    result, error = integrate.tplquad(f, a, b, gfun, hfun, qfun, rfun, **kwargs)
    return float(result), float(error)


def fixed_quad(
    f: Callable[[NDArray], NDArray],
    a: float,
    b: float,
    n: int = 5,
) -> Tuple[float, None]:
    """
    Fixed-order Gaussian quadrature.

    Computes ∫_a^b f(x) dx using n-point Gauss-Legendre quadrature.

    Parameters
    ----------
    f : callable
        Function to integrate. Should accept and return arrays.
    a : float
        Lower limit.
    b : float
        Upper limit.
    n : int, optional
        Number of quadrature points. Default is 5.

    Returns
    -------
    result : float
        Estimated integral value.
    None
        Placeholder for compatibility (no error estimate).

    Examples
    --------
    >>> result, _ = fixed_quad(lambda x: x**2, 0, 1, n=5)
    >>> result
    0.3333333333333333

    See Also
    --------
    scipy.integrate.fixed_quad : Underlying implementation.
    """
    result, _ = integrate.fixed_quad(f, a, b, n=n)
    return float(result), None


def romberg(
    f: Callable[[float], float],
    a: float,
    b: float,
    **kwargs,
) -> float:
    """
    Romberg integration.

    Uses Richardson extrapolation to accelerate the trapezoidal rule.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit.
    b : float
        Upper limit.
    **kwargs
        Additional arguments passed to scipy.integrate.romberg.

    Returns
    -------
    result : float
        Estimated integral value.

    See Also
    --------
    scipy.integrate.romberg : Underlying implementation.
    """
    return float(integrate.romberg(f, a, b, **kwargs))


def simpson(
    y: ArrayLike,
    x: Optional[ArrayLike] = None,
    dx: float = 1.0,
) -> float:
    """
    Simpson's rule integration from samples.

    Parameters
    ----------
    y : array_like
        Array of function values.
    x : array_like, optional
        Sample points. If None, uses uniform spacing dx.
    dx : float, optional
        Spacing between samples if x is None. Default is 1.

    Returns
    -------
    result : float
        Estimated integral.

    See Also
    --------
    scipy.integrate.simpson : Underlying implementation.
    """
    return float(integrate.simpson(y, x=x, dx=dx))


def trapezoid(
    y: ArrayLike,
    x: Optional[ArrayLike] = None,
    dx: float = 1.0,
) -> float:
    """
    Trapezoidal rule integration from samples.

    Parameters
    ----------
    y : array_like
        Array of function values.
    x : array_like, optional
        Sample points. If None, uses uniform spacing dx.
    dx : float, optional
        Spacing between samples if x is None. Default is 1.

    Returns
    -------
    result : float
        Estimated integral.

    See Also
    --------
    scipy.integrate.trapezoid : Underlying implementation.
    """
    return float(integrate.trapezoid(y, x=x, dx=dx))


def cubature_gauss_hermite(
    n_dim: int,
    n_points_per_dim: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Tensor product Gauss-Hermite cubature rule.

    Creates a multi-dimensional quadrature rule for integrating over
    a multivariate Gaussian distribution.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    n_points_per_dim : int
        Number of quadrature points per dimension.

    Returns
    -------
    points : ndarray
        Cubature points of shape (n_points_per_dim^n_dim, n_dim).
    weights : ndarray
        Cubature weights of shape (n_points_per_dim^n_dim,).

    Notes
    -----
    The number of points grows exponentially with dimension.
    For high dimensions, consider using sparse grid methods.

    Examples
    --------
    >>> points, weights = cubature_gauss_hermite(2, 3)
    >>> points.shape
    (9, 2)
    """
    x1d, w1d = gauss_hermite(n_points_per_dim)

    # Create tensor product grid
    grids = np.meshgrid(*[x1d] * n_dim, indexing="ij")
    points = np.column_stack([g.ravel() for g in grids])

    # Tensor product of weights
    weight_grids = np.meshgrid(*[w1d] * n_dim, indexing="ij")
    weights = np.prod(np.column_stack([g.ravel() for g in weight_grids]), axis=1)

    return points, weights


def spherical_cubature(
    n_dim: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Spherical cubature rule for Gaussian integrals.

    A 2n-point cubature rule that is exact for polynomials up to degree 3.
    This is the rule used in the Cubature Kalman Filter (CKF).

    Parameters
    ----------
    n_dim : int
        Number of dimensions.

    Returns
    -------
    points : ndarray
        Cubature points of shape (2*n_dim, n_dim).
    weights : ndarray
        Cubature weights of shape (2*n_dim,).

    Notes
    -----
    Points are at ±√n along each axis, scaled for use with standard
    normal distributions.

    For computing E[f(X)] where X ~ N(μ, P):
    - Transform points: x_i = μ + chol(P) @ points[i]
    - E[f(X)] ≈ Σ weights[i] * f(x_i)

    References
    ----------
    Arasaratnam & Haykin, "Cubature Kalman Filters", IEEE TAC, 2009.
    """
    # Points at ±√n along each axis
    sqrt_n = np.sqrt(n_dim)

    points = np.zeros((2 * n_dim, n_dim))
    for i in range(n_dim):
        points[2 * i, i] = sqrt_n
        points[2 * i + 1, i] = -sqrt_n

    # Equal weights
    weights = np.ones(2 * n_dim) / (2 * n_dim)

    return points, weights


def unscented_transform_points(
    n_dim: int,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: Optional[float] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate sigma points and weights for unscented transform.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    alpha : float, optional
        Spread of sigma points. Default is 1e-3.
    beta : float, optional
        Prior knowledge parameter (2 is optimal for Gaussian). Default is 2.
    kappa : float, optional
        Secondary scaling parameter. Default is 3 - n_dim.

    Returns
    -------
    sigma_points : ndarray
        Relative sigma point positions of shape (2*n_dim + 1, n_dim).
        Center point is at index 0, followed by ±directions.
    wm : ndarray
        Weights for computing mean, shape (2*n_dim + 1,).
    wc : ndarray
        Weights for computing covariance, shape (2*n_dim + 1,).

    Notes
    -----
    For a random variable X ~ N(μ, P), the sigma points are:
    - χ_0 = μ
    - χ_i = μ + (√((n+λ)P))_i for i = 1..n
    - χ_{n+i} = μ - (√((n+λ)P))_i for i = 1..n

    where (√A)_i is the i-th column of the matrix square root.

    References
    ----------
    Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation",
    Proc. IEEE, 2004.
    """
    if kappa is None:
        kappa = 3.0 - n_dim

    lambda_ = alpha**2 * (n_dim + kappa) - n_dim
    scale = np.sqrt(n_dim + lambda_)

    # Sigma points (relative to mean)
    sigma_points = np.zeros((2 * n_dim + 1, n_dim))
    # sigma_points[0] = 0 (center point)
    for i in range(n_dim):
        sigma_points[1 + i, i] = scale
        sigma_points[1 + n_dim + i, i] = -scale

    # Weights for mean
    wm = np.zeros(2 * n_dim + 1)
    wm[0] = lambda_ / (n_dim + lambda_)
    wm[1:] = 1.0 / (2 * (n_dim + lambda_))

    # Weights for covariance
    wc = wm.copy()
    wc[0] = wm[0] + (1 - alpha**2 + beta)

    return sigma_points, wm, wc


__all__ = [
    # 1D Quadrature rules
    "gauss_legendre",
    "gauss_hermite",
    "gauss_laguerre",
    "gauss_chebyshev",
    # Integration functions
    "quad",
    "dblquad",
    "tplquad",
    "fixed_quad",
    "romberg",
    "simpson",
    "trapezoid",
    # Multi-dimensional cubature
    "cubature_gauss_hermite",
    "spherical_cubature",
    "unscented_transform_points",
]
