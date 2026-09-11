"""
Numerical integration (quadrature) methods.

This module provides Gaussian quadrature rules and numerical integration
functions commonly used in state estimation and filtering.
"""

from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np
import scipy.integrate as integrate
from numpy.typing import ArrayLike, NDArray


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
    >>> round(float(np.sum(w * x**2)), 6)
    0.666667

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

    Examples
    --------
    >>> x, w = gauss_hermite(5)
    >>> # Compute E[X^2] for X ~ N(0, 1) (exact = 1)
    >>> result = np.sum(w * (np.sqrt(2) * x)**2) / np.sqrt(np.pi)
    >>> abs(result - 1.0) < 1e-10
    True

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

    Examples
    --------
    >>> x, w = gauss_laguerre(5)
    >>> # Integrate x * exp(-x) from 0 to inf (exact = 1)
    >>> round(float(np.sum(w * x)), 10)
    1.0

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

    Examples
    --------
    >>> x, w = gauss_chebyshev(5, kind=1)
    >>> x.shape
    (5,)

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
    **kwargs: Any,
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
    >>> round(result, 6)
    0.333333

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
    **kwargs: Any,
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

    Examples
    --------
    >>> # Integrate x*y over unit square
    >>> result, error = dblquad(lambda y, x: x*y, 0, 1, lambda x: 0, lambda x: 1)
    >>> round(result, 10)  # Should be 0.25
    0.25

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
    **kwargs: Any,
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

    Examples
    --------
    >>> # Integrate x*y*z over unit cube
    >>> result, error = tplquad(
    ...     lambda z, y, x: x*y*z,
    ...     0, 1,
    ...     lambda x: 0, lambda x: 1,
    ...     lambda x, y: 0, lambda x, y: 1
    ... )
    >>> abs(result - 0.125) < 1e-6  # 1/8
    True

    See Also
    --------
    scipy.integrate.tplquad : Underlying implementation.
    """
    result, error = integrate.tplquad(f, a, b, gfun, hfun, qfun, rfun, **kwargs)
    return float(result), float(error)


def fixed_quad(
    f: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    a: float,
    b: float,
    n: int = 5,
) -> tuple[float, None]:
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
    >>> round(result, 6)
    0.333333

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
    tol: float = 1e-8,
    max_steps: int = 20,
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
    tol : float, optional
        Desired tolerance. Default is 1e-8.
    max_steps : int, optional
        Maximum number of extrapolation steps. Default is 20.

    Returns
    -------
    result : float
        Estimated integral value.

    Examples
    --------
    >>> # Integrate x^2 from 0 to 1 (exact = 1/3)
    >>> result = romberg(lambda x: x**2, 0, 1)
    >>> abs(result - 1/3) < 1e-8
    True

    Notes
    -----
    This is a native implementation that does not depend on scipy.integrate.romberg,
    which was deprecated in scipy 1.12 and removed in scipy 1.15.
    """
    # Romberg table
    R = np.zeros((max_steps, max_steps), dtype=np.float64)

    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, max_steps):
        h = h / 2.0

        # Composite trapezoidal rule with 2^i intervals
        n_new = 2 ** (i - 1)
        total = 0.0
        for k in range(1, n_new + 1):
            total += f(a + (2 * k - 1) * h)
        R[i, 0] = 0.5 * R[i - 1, 0] + h * total

        # Richardson extrapolation
        for j in range(1, i + 1):
            factor = 4**j
            R[i, j] = (factor * R[i, j - 1] - R[i - 1, j - 1]) / (factor - 1)

        # Check convergence
        if i > 0 and abs(R[i, i] - R[i - 1, i - 1]) < tol:
            return float(R[i, i])

    return float(R[max_steps - 1, max_steps - 1])


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

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, np.pi, 101)
    >>> y = np.sin(x)
    >>> result = simpson(y, x)  # Should be ~2.0
    >>> abs(result - 2.0) < 1e-5
    True

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

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 11)
    >>> y = x**2  # Integrate x^2 from 0 to 1
    >>> result = trapezoid(y, x)
    >>> abs(result - 1/3) < 0.01  # Approximation
    True

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

    Examples
    --------
    >>> points, weights = spherical_cubature(3)
    >>> points.shape  # 2*n = 6 points in 3D
    (6, 3)
    >>> weights.shape
    (6,)
    >>> round(float(np.sum(weights)), 6)  # Weights sum to 1
    1.0
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

    See Also
    --------
    pytcl.mathematical_functions.numerical_integration.cubature_points.second_order_cubature_points :
        A structurally different (n+2)-point rule from the same paper's
        Appendix III (spherical simplex sigma points), exact through degree
        2 only (this function's 2n+1 points are exact through degree 3).
        Not an alternate parameterization of this rule -- see its Notes
        section for the reconciliation.

    Examples
    --------
    >>> sigma_points, wm, wc = unscented_transform_points(3)
    >>> sigma_points.shape  # 2*n+1 = 7 points in 3D
    (7, 3)
    >>> wm.shape
    (7,)
    >>> np.abs(np.sum(wm) - 1.0) < 1e-10  # Mean weights sum to 1
    True
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


def clenshaw_curtis_points_1d(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Clenshaw-Curtis quadrature points and weights on [-1, 1].

    Port of ``ClenshawCurtisPoints1D``. The n+1 points are the Chebyshev
    extrema ``cos(k*pi/n)``, k = 0..n (descending from 1 to -1), and the
    weights are built with Waldvogel's FFT method. The rule integrates
    polynomials up to order n exactly against the unit weight, so the
    weights sum to 2 (the length of the interval) -- unlike the
    probability-normalized cubature generators, whose weights sum to 1.

    Parameters
    ----------
    n : int
        Polynomial order of the rule; must be >= 2. Returns n+1 points.

    Returns
    -------
    xi : ndarray
        Quadrature points of shape (n+1,), descending from 1 to -1.
    w : ndarray
        Quadrature weights of shape (n+1,), all positive, summing to 2.

    References
    ----------
    - J. Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
      quadrature rules," BIT Numerical Mathematics 46(1):195-202, 2006.

    Examples
    --------
    >>> xi, w = clenshaw_curtis_points_1d(8)
    >>> round(float(np.sum(w * xi**2)), 12)  # integral of x^2 over [-1,1]
    0.666666666667
    >>> round(float(np.sum(w)), 12)
    2.0
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    # Points cos(k*pi/n) via the Chebyshev three-term recurrence, as in
    # the MATLAB original (bit-compatible with its tables).
    xi = np.zeros(n + 1)
    xi[0] = 1.0
    cos_theta = np.cos(np.pi / n)
    xi[1] = cos_theta
    xi[2] = 2 * cos_theta**2 - 1
    for k in range(3, n):
        xi[k] = 2 * cos_theta * xi[k - 1] - xi[k - 2]
    xi[n] = -1.0

    # Weights: Waldvogel Eqs. 2.6, 3.10, 4.2.
    n2 = n // 2
    v = np.zeros(n)
    k = np.arange(n2)
    v[:n2] = 2.0 / (1 - 4 * k**2)
    v[n2] = (n - 3) / (2 * n2 - 1) - 1
    start = n2 + (n % 2)
    v[n2 + 1 :] = v[start - 1 : 0 : -1]

    w0cc = 1.0 / (n**2 - 1 + n % 2)
    g = np.zeros(n)
    g[: n2 + 1] = -w0cc
    g[n2] = w0cc * ((2 - n % 2) * n - 1)
    g[n2 + 1 :] = g[start - 1 : 0 : -1]

    w = np.zeros(n + 1)
    w[:n] = np.real(np.fft.ifft(v + g))
    w[n] = w0cc
    return xi, w


def fejer_points_1d(
    n: int,
    rule: int = 1,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Fejer quadrature points and weights on [-1, 1].

    Port of ``FejerPoints1D`` (Waldvogel's FFT construction). Rule 1
    uses the Chebyshev roots ``cos((k - 1/2) pi / n)`` (n points, order
    n-1); rule 2 uses the interior Chebyshev extrema (n-1 points, order
    n-2). Weights sum to 2, the length of the interval.

    Parameters
    ----------
    n : int
        Number-of-points parameter; rule 1 returns n points, rule 2
        returns n-1.
    rule : int, optional
        1 for Fejer's first rule (default), 2 for the second.

    Returns
    -------
    xi : ndarray
        Quadrature points, descending.
    w : ndarray
        Quadrature weights, summing to 2.

    References
    ----------
    - J. Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
      quadrature rules," BIT Numerical Mathematics 46(1):195-202, 2006.

    Examples
    --------
    >>> xi, w = fejer_points_1d(30)
    >>> round(float(np.sum(w * xi**4)), 12)  # integral of x^4 over [-1,1]
    0.4
    """
    if rule == 1:
        # Waldvogel Eq. 4.4.
        v = np.zeros(n, dtype=complex)
        nm = (n - 1) // 2
        k = np.arange(nm + 1)
        v[: nm + 1] = (2.0 / (1 - 4 * k**2)) * np.exp(1j * k * np.pi / n)
        start = nm + 2 + (n % 2 == 0)
        v[start - 1 :] = np.conj(v[nm:0:-1])
        w = np.real(np.fft.ifft(v))

        # Points cos((k - 1/2) pi / n) via the angle-addition recurrence.
        xi = np.zeros(n)
        xi[0] = np.cos(0.5 * np.pi / n)
        sin_cur = np.sin(0.5 * np.pi / n)
        cos_t = np.cos(np.pi / n)
        sin_t = np.sin(np.pi / n)
        for k in range(1, n):
            xi[k] = cos_t * xi[k - 1] - sin_t * sin_cur
            sin_cur = sin_t * xi[k - 1] + sin_cur * cos_t
        return xi, w

    if rule == 2:
        # Shares Waldvogel Eq. 3.10 with Clenshaw-Curtis, then drops the
        # first (always-zero) weight and its endpoint.
        n2 = n // 2
        v = np.zeros(n)
        k = np.arange(n2)
        v[:n2] = 2.0 / (1 - 4 * k**2)
        v[n2] = (n - 3) / (2 * n2 - 1) - 1
        start = n2 + (n % 2)
        v[n2 + 1 :] = v[start - 1 : 0 : -1]
        w = np.real(np.fft.ifft(v))[1:]

        xi = np.zeros(n - 1)
        cos_theta = np.cos(np.pi / n)
        xi[0] = cos_theta
        xi[1] = 2 * cos_theta**2 - 1
        for k in range(2, n - 1):
            xi[k] = 2 * cos_theta * xi[k - 1] - xi[k - 2]
        return xi, w

    raise ValueError("rule must be 1 or 2")


def conform_map_quad_pts_1d(
    n: int,
    mapping: Optional[int] = None,
    point_type: int = 0,
    param: Optional[float] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Conformally mapped quadrature points and weights on [-1, 1].

    Port of ``conformMapQuadPts1D`` (Hale & Trefethen 2008): a base
    Gauss-Legendre or Clenshaw-Curtis rule is transplanted through a
    conformal map that spreads the clustered endpoint nodes toward
    uniform spacing, raising the effective resolution per point for
    analytic integrands.

    Parameters
    ----------
    n : int
        Number-of-points parameter of the base rule.
    mapping : int, optional
        0: degree-``param`` Taylor expansion of arcsine (``param``
        default 9, should be odd); 1: Kosloff-Tal-Ezer map with
        ``alpha = 2/(param + 1/param)``; 2: strip map (default when
        ``point_type=0``; requires Gauss-Legendre base points);
        3: the appendix approximate strip map. ``param`` defaults to
        1.4 for mappings 1-3.
    point_type : int, optional
        0 (default): Gauss-Legendre base points; 1: Clenshaw-Curtis.
    param : float, optional
        Mapping parameter; see ``mapping``.

    Returns
    -------
    xi : ndarray
        Mapped quadrature points.
    w : ndarray
        Transplanted weights (Hale & Trefethen Eq. 2.6).

    References
    ----------
    - N. Hale and L. N. Trefethen, "New quadrature formulas from
      conformal maps," SIAM J. Numer. Anal. 46(2):930-948, 2008.

    Examples
    --------
    >>> xi, w = conform_map_quad_pts_1d(20)
    >>> round(float(np.sum(w * np.exp(xi))), 4)  # integral of e^x over [-1,1]
    2.3504
    """
    if point_type == 0:
        xi, w = gauss_legendre(n)
    elif point_type == 1:
        xi, w = clenshaw_curtis_points_1d(n)
    else:
        raise ValueError("point_type must be 0 or 1")

    if mapping is None:
        mapping = 2 if point_type == 0 else 0
    if param is None:
        param = 9 if mapping == 0 else 1.4
    rho = param

    if mapping == 0:
        # Degree-d arcsine Taylor expansion, normalized to fix g(1) = 1.
        d = int(rho)
        coeffs = np.zeros(d + 1)
        odd = np.arange(1, d + 1, 2, dtype=np.float64)
        series = np.concatenate(
            ([1.0], np.cumprod(np.arange(1, d - 1, 2)) / np.cumprod(np.arange(2, d, 2)))
        )
        coeffs[d - 1 :: -2][: len(odd)] = (1.0 / odd) * series
        coeffs = coeffs / np.sum(coeffs)
        dg = np.polyval(np.polyder(coeffs), xi)
        xi = np.polyval(coeffs, xi)
        w = w * dg
    elif mapping == 1:
        alpha = 2.0 / (rho + 1.0 / rho)
        dg = alpha / (np.sqrt(1 - xi**2 * alpha**2) * np.arcsin(alpha))
        xi = np.arcsin(alpha * xi) / np.arcsin(alpha)
        w = w * dg
    elif mapping == 2:
        if point_type == 1:
            raise ValueError(
                "the strip mapping (mapping=2) cannot be used with "
                "Clenshaw-Curtis points"
            )
        from pytcl.mathematical_functions.special_functions.elliptic import (
            ellipkinc,
            jacobi_elliptic,
        )

        # Hale & Trefethen Eq. 3.1: the modulus from rho.
        num = 0.0
        den = 0.0
        for j in range(1, round(0.5 + np.sqrt(10.0 / np.log(rho))) + 1):
            num += rho ** (-4 * (j - 0.5) ** 2)
            den += rho ** (-4 * j**2)
        m4 = 2 * num / (1 + 2 * den)
        m = m4**4

        K = float(ellipkinc(np.pi / 2, m))
        u = np.arcsin(xi)
        omega = 2 * K * u / np.pi
        sn, cn, dn = jacobi_elliptic(omega, m)

        # Eqs. 3.3 and 3.2.
        dg = (
            (2 * K * m4 / (np.pi * np.sqrt(1 - xi**2)))
            * (cn * dn / (1 - m4**2 * sn**2))
            / np.arctanh(m4)
        )
        xi = np.arctanh(m4 * sn) / np.arctanh(m4)
        w = w * dg
    elif mapping == 3:
        u = np.arcsin(xi)
        tau = np.pi / np.log(rho)
        d = 0.5 + 1.0 / (np.exp(tau * np.pi) + 1)
        pd2pu = np.pi / 2 + u
        pd2mu = np.pi / 2 - u
        C = 1.0 / (np.log(1 + np.exp(-tau * np.pi)) - np.log(2) + (np.pi / 2) * tau * d)

        # Eq. A.2, with the removable singularity at +-1 replaced by
        # Eq. A.3. That substitution is only correct for nodes AT +-1,
        # i.e. the Clenshaw-Curtis endpoints; the MATLAB original
        # applies it to whatever sits in the first and last positions
        # of its point array regardless of base, which for its
        # Gauss-Legendre storage order corrupts one extreme and one
        # INTERIOR weight. This port applies it only to the
        # Clenshaw-Curtis endpoints (a loud fix of the upstream
        # defect; the oracle test excludes MATLAB's two corrupted
        # nodes for the Gauss-Legendre case).
        with np.errstate(divide="ignore", invalid="ignore"):
            dg = (
                -C
                * (tau / np.sqrt(1 - xi**2))
                * (
                    1.0 / (np.exp(tau * pd2pu) + 1)
                    + 1.0 / (np.exp(tau * pd2mu) + 1)
                    - d
                )
            )
        if point_type == 1:
            dg[0] = dg[-1] = (C * tau**2 / 4) * np.tanh((np.pi / 2) * tau) ** 2
        w = w * dg
        xi = C * (
            np.log(1 + np.exp(-tau * pd2pu))
            - np.log(1 + np.exp(-tau * pd2mu))
            + d * tau * u
        )
    else:
        raise ValueError("mapping must be 0, 1, 2, or 3")

    return xi, w


__all__ = [
    # 1D Quadrature rules
    "gauss_legendre",
    "gauss_hermite",
    "gauss_laguerre",
    "gauss_chebyshev",
    "clenshaw_curtis_points_1d",
    "fejer_points_1d",
    "conform_map_quad_pts_1d",
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
