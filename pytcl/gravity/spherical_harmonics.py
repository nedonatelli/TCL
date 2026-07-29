"""
Spherical harmonic functions for geophysical models.

Spherical harmonics are used to represent gravitational and magnetic
fields on a sphere. This module provides functions for evaluating
associated Legendre polynomials and spherical harmonic expansions.

References
----------
- W. A. Heiskanen and H. Moritz, "Physical Geodesy," W. H. Freeman, 1967.
- O. Montenbruck and E. Gill, "Satellite Orbits," Springer, 2000.
"""

import logging
from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Module logger
_logger = logging.getLogger("pytcl.gravity.spherical_harmonics")

# Cache configuration for Legendre polynomials.
# Quantization must be fine enough that the induced error (~n^2 * 0.5*10^-d
# in relative terms) stays negligible at high degree.
_LEGENDRE_CACHE_DECIMALS = 12  # Precision for x quantization
_LEGENDRE_CACHE_MAXSIZE = 64  # Max cached (n_max, m_max, x) combinations


def _quantize_x(x: float) -> float:
    """Quantize x value for cache key compatibility."""
    return round(x, _LEGENDRE_CACHE_DECIMALS)


@lru_cache(maxsize=_LEGENDRE_CACHE_MAXSIZE)
def _associated_legendre_cached(
    n_max: int,
    m_max: int,
    x_quantized: float,
    normalized: bool,
) -> tuple[tuple[np.ndarray[Any, Any], ...], ...]:
    """Cached Legendre polynomial computation (internal).

    Returns tuple of tuples for hashability.
    """
    P = np.zeros((n_max + 1, m_max + 1))
    u = np.sqrt(1 - x_quantized * x_quantized)

    P[0, 0] = 1.0

    for m in range(1, m_max + 1):
        if normalized:
            # Geodesy full normalization includes sqrt(2 - delta_0m); the
            # sectoral recursion factor is sqrt(3) for m = 1 and
            # sqrt((2m+1)/(2m)) for m > 1 (Holmes & Featherstone 2002).
            factor = np.sqrt(3.0) if m == 1 else np.sqrt((2 * m + 1) / (2 * m))
            P[m, m] = u * factor * P[m - 1, m - 1]
        else:
            P[m, m] = (2 * m - 1) * u * P[m - 1, m - 1]

    for m in range(m_max):
        if m + 1 <= n_max:
            if normalized:
                P[m + 1, m] = x_quantized * np.sqrt(2 * m + 3) * P[m, m]
            else:
                P[m + 1, m] = x_quantized * (2 * m + 1) * P[m, m]

    for m in range(m_max + 1):
        for n in range(m + 2, n_max + 1):
            if normalized:
                a_nm = np.sqrt((4 * n * n - 1) / (n * n - m * m))
                b_nm = np.sqrt(((n - 1) ** 2 - m * m) / (4 * (n - 1) ** 2 - 1))
                P[n, m] = a_nm * (x_quantized * P[n - 1, m] - b_nm * P[n - 2, m])
            else:
                P[n, m] = (
                    (2 * n - 1) * x_quantized * P[n - 1, m] - (n + m - 1) * P[n - 2, m]
                ) / (n - m)

    # Convert to tuple of tuples for hashability
    return tuple(tuple(row) for row in P)


def associated_legendre(
    n_max: int,
    m_max: int,
    x: float,
    normalized: bool = True,
) -> NDArray[np.floating]:
    """
    Compute associated Legendre polynomials P_n^m(x).

    Uses the recursive algorithm for numerical stability.

    Parameters
    ----------
    n_max : int
        Maximum degree.
    m_max : int
        Maximum order (must be <= n_max).
    x : float
        Argument, typically cos(colatitude). Must be in [-1, 1].
    normalized : bool, optional
        If True, return fully normalized (geodetic) coefficients.
        Default True.

    Returns
    -------
    P : ndarray
        Array of shape (n_max+1, m_max+1) containing P_n^m(x).

    Notes
    -----
    The fully normalized (geodesy convention) associated Legendre
    functions include the factor sqrt((2 - delta_0m)(2n+1)(n-m)!/(n+m)!)
    and satisfy:

    .. math::

        \\int_{-1}^{1} [\\bar{P}_n^m(x)]^2 dx = 2(2 - \\delta_{0m})

    Results are cached for repeated queries with the same parameters.
    Cache key quantizes x to 12 decimal places (~1e-12 precision).

    Examples
    --------
    >>> P = associated_legendre(2, 2, 0.5)
    >>> float(round(P[2, 0], 6))  # Fully normalized: sqrt(5) * (3x^2 - 1)/2
    -0.279508
    """
    if m_max > n_max:
        raise ValueError("m_max must be <= n_max")
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1, 1]")

    # Use cached computation
    x_q = _quantize_x(x)
    cached = _associated_legendre_cached(n_max, m_max, x_q, normalized)
    return np.array(cached)


def associated_legendre_derivative(
    n_max: int,
    m_max: int,
    x: float,
    P: Optional[NDArray[np.floating]] = None,
    normalized: bool = True,
) -> NDArray[np.floating]:
    """
    Compute derivatives of associated Legendre polynomials dP_n^m/dx.

    Parameters
    ----------
    n_max : int
        Maximum degree.
    m_max : int
        Maximum order.
    x : float
        Argument in [-1, 1].
    P : ndarray, optional
        Precomputed P_n^m values. If None, computed internally.
    normalized : bool, optional
        If True, use fully normalized functions. Default True.

    Returns
    -------
    dP : ndarray
        Array of shape (n_max+1, m_max+1) containing dP_n^m/dx.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.cos(np.radians(45))  # cos of 45 degrees
    >>> dP = associated_legendre_derivative(2, 2, x)
    >>> dP.shape
    (3, 3)
    >>> abs(dP[0, 0]) < 1e-10  # dP_0^0/dx = 0
    True
    """
    if P is None:
        P = associated_legendre(n_max, m_max, x, normalized)

    dP = np.zeros((n_max + 1, m_max + 1))

    # Handle x = ±1 specially (poles)
    if abs(abs(x) - 1) < 1e-14:
        # At poles, derivatives need special handling
        # For now, return zeros (valid for m > 0)
        return dP

    u2 = 1 - x * x  # sin^2(theta)

    for n in range(n_max + 1):
        for m in range(min(n, m_max) + 1):
            if n == 0:
                dP[n, m] = 0.0
            elif normalized:
                # (1-x^2) dP_nm/dx = f_nm * P_(n-1)m - n*x*P_nm with
                # f_nm = sqrt((n^2-m^2)(2n+1)/(2n-1)); f_nn = 0 covers m == n
                f_nm = np.sqrt((n * n - m * m) * (2 * n + 1) / (2 * n - 1))
                P_prev = P[n - 1, m] if m <= n - 1 else 0.0
                dP[n, m] = (f_nm * P_prev - n * x * P[n, m]) / u2
            else:
                # (1-x^2) dP_nm/dx = (n+m) P_(n-1)m - n*x*P_nm
                P_prev = P[n - 1, m] if m <= n - 1 else 0.0
                dP[n, m] = ((n + m) * P_prev - n * x * P[n, m]) / u2

    return dP


def spherical_harmonic_sum(
    lat: float,
    lon: float,
    r: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate spherical harmonic expansion for a scalar field.

    Computes the value and gradient of a field represented by
    spherical harmonic coefficients C_nm and S_nm.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from center of mass.
    C : ndarray
        Cosine coefficients C_nm, shape (n_max+1, n_max+1).
    S : ndarray
        Sine coefficients S_nm, shape (n_max+1, n_max+1).
    R : float
        Reference radius (e.g., Earth's equatorial radius).
    GM : float
        Gravitational parameter (G * M).
    n_max : int, optional
        Maximum degree to use. Default uses full coefficient array.

    Returns
    -------
    V : float
        Potential value.
    dV_r : float
        Radial derivative dV/dr.
    dV_lat : float
        Latitudinal derivative (1/r) * dV/dlat.

    Notes
    -----
    .. warning::
       This routine forms ``Pbar_nm`` directly, so the ``sin(theta)**m`` factor
       underflows at very high order. It is reliable through ``n_max = 1600``
       but becomes unstable at EGM2008's degree 2190 -- at colatitude 30
       degrees on the reference sphere it returns a value twelve orders of
       magnitude too large. Use
       :func:`spherical_harmonic_sum_high_degree` for models above about
       degree 1600; it is also roughly four times faster there.

    The spherical harmonic expansion of the gravitational potential is:

    .. math::

        V = \\frac{GM}{r} \\sum_{n=0}^{N} \\left(\\frac{R}{r}\\right)^n
            \\sum_{m=0}^{n} \\bar{P}_n^m(\\sin\\phi)
            (C_{nm}\\cos m\\lambda + S_{nm}\\sin m\\lambda)

    Examples
    --------
    >>> import numpy as np
    >>> # Simple monopole (degree 0 only)
    >>> C = np.array([[1.0]])
    >>> S = np.array([[0.0]])
    >>> R = 6.378e6  # meters
    >>> GM = 3.986e14  # m^3/s^2
    >>> V, dV_r, dV_lat = spherical_harmonic_sum(0, 0, R, C, S, R, GM, n_max=0)
    >>> abs(V - GM/R) / (GM/R) < 1e-10  # V = GM/r for degree 0
    True
    """
    if n_max is None:
        n_max = C.shape[0] - 1

    _logger.debug(
        "spherical_harmonic_sum: lat=%.4f, lon=%.4f, r=%.1f, n_max=%d",
        lat,
        lon,
        r,
        n_max,
    )

    # Colatitude for Legendre polynomials
    colat = np.pi / 2 - lat
    cos_colat = np.cos(colat)
    sin_colat = np.sin(colat)

    # Compute Legendre polynomials and derivatives
    P = associated_legendre(n_max, n_max, cos_colat, normalized=True)
    dP = associated_legendre_derivative(n_max, n_max, cos_colat, P, normalized=True)

    # Initialize sums
    V = 0.0
    dV_r = 0.0
    dV_colat = 0.0
    dV_lon = 0.0

    # Compute (R/r)^n factors
    r_ratio = R / r
    r_power = 1.0  # (R/r)^0

    for n in range(n_max + 1):
        r_power_n = r_power  # (R/r)^n

        for m in range(n + 1):
            cos_m_lon = np.cos(m * lon)
            sin_m_lon = np.sin(m * lon)

            # Coefficient combination
            Cnm = C[n, m] if n < C.shape[0] and m < C.shape[1] else 0.0
            Snm = S[n, m] if n < S.shape[0] and m < S.shape[1] else 0.0

            coeff = Cnm * cos_m_lon + Snm * sin_m_lon
            coeff_lon = m * (-Cnm * sin_m_lon + Snm * cos_m_lon)

            # Potential contribution
            V += r_power_n * P[n, m] * coeff

            # Radial derivative contribution:
            # d/dr [(R/r)^n / r] = -(n+1) (R/r)^n / r^2
            dV_r += -(n + 1) * r_power_n * P[n, m] * coeff

            # Colatitude derivative contribution
            # dP/d(colat) = -sin(colat) * dP/d(cos(colat))
            dV_colat += r_power_n * (-sin_colat) * dP[n, m] * coeff

            # Longitude derivative contribution
            dV_lon += r_power_n * P[n, m] * coeff_lon

        r_power *= r_ratio  # Update for next n

    # Scale by GM/r
    scale = GM / r
    V *= scale
    dV_r *= GM / (r * r)

    # Convert colatitude derivative to latitude derivative
    dV_lat = -dV_colat * scale / r  # (1/r) dV/dlat

    return V, dV_r, dV_lat


def gravity_acceleration(
    lat: float,
    lon: float,
    h: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute gravity acceleration vector from spherical harmonics.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float
        Height above reference ellipsoid in meters.
    C : ndarray
        Cosine coefficients.
    S : ndarray
        Sine coefficients.
    R : float
        Reference radius.
    GM : float
        Gravitational parameter.
    n_max : int, optional
        Maximum degree.

    Returns
    -------
    g_r : float
        Radial component of gravity (positive outward).
    g_lat : float
        Northward component of gravity.
    g_lon : float
        Eastward component of gravity.

    Examples
    --------
    >>> import numpy as np
    >>> # Simple monopole field
    >>> C = np.array([[1.0]])
    >>> S = np.array([[0.0]])
    >>> R = 6.378e6
    >>> GM = 3.986e14
    >>> g_r, g_lat, g_lon = gravity_acceleration(0, 0, 0, C, S, R, GM, n_max=0)
    >>> g_r < 0  # Gravity points inward (negative radial)
    True
    """
    # Approximate radial distance (simplified, ignoring ellipsoid flattening)
    r = R + h

    V, dV_r, dV_lat = spherical_harmonic_sum(lat, lon, r, C, S, R, GM, n_max)

    # spherical_harmonic_sum uses the geodesy-positive potential (V = +GM/r),
    # for which gravity is g = +grad(V); dV_r = -GM/r^2 already points inward
    g_r = dV_r
    g_lat = dV_lat

    # Longitude component (for non-zonal terms)
    # This would require additional computation for full accuracy
    g_lon = 0.0  # Simplified

    return g_r, g_lat, g_lon


@lru_cache(maxsize=64)
def _legendre_scaling_factors_cached(n_max: int) -> Tuple[float, ...]:
    """Cached computation of Legendre scaling factors.

    Returns tuple for hashability.
    """
    if n_max <= 150:
        return tuple([1.0] * (n_max + 1))

    scale = []
    for n in range(n_max + 1):
        exponent = -280.0 * n / n_max
        scale.append(10.0**exponent)
    return tuple(scale)


def legendre_scaling_factors(n_max: int) -> NDArray[np.floating]:
    """Precompute scaling factors to prevent overflow in Legendre recursion.

    For degrees > ~150, standard Legendre recursion can overflow in
    double precision. These scaling factors keep intermediate values
    in a representable range.

    The scaling follows the approach of Holmes & Featherstone (2002),
    using a factor that grows with degree to counteract the natural
    growth of the Legendre functions.

    Parameters
    ----------
    n_max : int
        Maximum degree.

    Returns
    -------
    scale : ndarray
        Scaling factors of shape (n_max+1,). The factor for degree n
        is 10^(-280 * n / n_max) for n_max > 150, else 1.0.

    References
    ----------
    - Holmes, S.A. and Featherstone, W.E. "A unified approach to the
      Clenshaw summation and the recursive computation of very high
      degree and order normalised associated Legendre functions."
      Journal of Geodesy 76.5 (2002): 279-299.

    Examples
    --------
    >>> scale = legendre_scaling_factors(100)
    >>> len(scale)
    101
    >>> scale[0]  # No scaling for low degrees
    1.0

    >>> scale_high = legendre_scaling_factors(200)
    >>> scale_high[200] < scale_high[0]  # Higher degrees scaled down
    True
    """
    return np.array(_legendre_scaling_factors_cached(n_max))


def associated_legendre_scaled(
    n_max: int,
    m_max: int,
    x: float,
    scale: Optional[NDArray[np.floating]] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute scaled associated Legendre functions for ultra-high degrees.

    Implements the Holmes & Featherstone (2002) order-wise scaling. The
    quantity that underflows in the ordinary recursion is ``u**m`` (with
    ``u = sin(theta)``), not anything degree-dependent: at degree 2190 and
    colatitude 20 degrees the ordinary functions flush to zero for every
    ``m > 696``, discarding roughly a quarter of the addition-theorem norm.

    This routine therefore recurses on ``Pbar_nm / u**m``, in which ``u``
    cancels from every recursion relation, and applies a global ``1e-280``
    factor so the peak near ``m ~ n * u`` stays inside double range.

    Parameters
    ----------
    n_max : int
        Maximum degree.
    m_max : int
        Maximum order (must be <= n_max).
    x : float
        Argument in [-1, 1], typically cos(colatitude).
    scale : ndarray, optional
        Unused; retained for backward compatibility.

    Returns
    -------
    P_scaled : ndarray
        Scaled values, shape (n_max+1, m_max+1), equal to
        ``Pbar_nm / u**m * 1e-280``.
    scale_exp : ndarray
        Base-10 log of the per-order reconstruction factor, shape
        (m_max+1,), so that ``Pbar_nm == P_scaled[n, m] * 10**scale_exp[m]``.

    Notes
    -----
    The reconstruction factor is per **order**, not per degree. Reconstructing
    individual values is only meaningful where ``Pbar_nm`` is representable;
    the point of the scaling is to feed a summation that applies ``u**m``
    progressively (Horner's scheme), which never forms the underflowing factor
    explicitly. See :func:`spherical_harmonic_sum_high_degree`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.cos(np.radians(45))
    >>> P_scaled, scale_exp = associated_legendre_scaled(10, 10, x)
    >>> P_scaled.shape
    (11, 11)
    >>> # Reconstruct and compare against the direct computation
    >>> P_direct = associated_legendre(10, 10, x, normalized=True)
    >>> recon = P_scaled[10, 3] * 10.0 ** scale_exp[3]
    >>> bool(abs(recon - P_direct[10, 3]) < 1e-10)
    True

    References
    ----------
    - Holmes, S.A. and Featherstone, W.E. "A unified approach to the
      Clenshaw summation and the recursive computation of very high
      degree and order normalised associated Legendre functions."
      Journal of Geodesy 76.5 (2002): 279-299.
    """
    if m_max > n_max:
        raise ValueError("m_max must be <= n_max")
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1, 1]")

    u = np.sqrt(max(0.0, 1.0 - x * x))

    P_scaled = np.zeros((n_max + 1, m_max + 1))

    # Global scale keeping the peak of Pbar/u**m inside double range.
    global_scale = 1e-280
    log_global = -280.0

    # Sectoral seeds. In Pbar_mm / u**m the u factor cancels entirely, so the
    # seeds grow only like sqrt(m) instead of collapsing like u**m.
    P_scaled[0, 0] = global_scale
    for m in range(1, m_max + 1):
        factor = np.sqrt(3.0) if m == 1 else np.sqrt((2 * m + 1) / (2 * m))
        P_scaled[m, m] = factor * P_scaled[m - 1, m - 1]

    # First off-diagonal.
    for m in range(m_max + 1):
        if m + 1 <= n_max:
            P_scaled[m + 1, m] = x * np.sqrt(2 * m + 3) * P_scaled[m, m]

    # Standard degree recursion; identical in form because u**m divides out.
    for m in range(m_max + 1):
        for n in range(m + 2, n_max + 1):
            a_nm = np.sqrt((4 * n * n - 1) / (n * n - m * m))
            b_nm = np.sqrt(((n - 1) ** 2 - m * m) / (4 * (n - 1) ** 2 - 1))
            P_scaled[n, m] = a_nm * (x * P_scaled[n - 1, m] - b_nm * P_scaled[n - 2, m])

    # Per-order reconstruction exponent: Pbar_nm = P_scaled[n,m] * 10**exp[m].
    orders = np.arange(m_max + 1, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_u = np.log10(u) if u > 0.0 else -np.inf
    scale_exp = orders * log_u - log_global

    return P_scaled, scale_exp


def spherical_harmonic_sum_high_degree(
    lat: float,
    lon: float,
    r: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Harmonic synthesis for ultra-high degree models (e.g. EGM2008).

    Uses the Holmes & Featherstone (2002) order-wise scaling together with
    Horner's scheme in ``u = sin(theta)``. Because the ``u**m`` factor is
    applied progressively rather than formed explicitly, no intermediate
    underflows -- which is what limits :func:`spherical_harmonic_sum` at high
    degree, where ``u**m`` flushes to zero and silently discards the
    corresponding orders.

    Parameters
    ----------
    lat : float
        Geocentric latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance in metres.
    C, S : ndarray
        Fully normalized cosine and sine coefficients, shape (n_max+1, n_max+1).
    R : float
        Reference radius in metres.
    GM : float
        Gravitational parameter in m^3/s^2.
    n_max : int, optional
        Maximum degree to evaluate. Defaults to the coefficient array size.

    Returns
    -------
    V : float
        Potential (geodesy-positive convention, V = +GM/r for a point mass).
    dV_r : float
        Radial derivative dV/dr.
    dV_lat : float
        Latitudinal derivative (1/r) dV/dlat.

    Notes
    -----
    Validated against :func:`spherical_harmonic_sum` at degrees where both are
    reliable, and used beyond that. At degree 2190 (EGM2008) the ordinary
    routine loses every order above ``m ~ 700`` at mid colatitudes; this one
    preserves the addition-theorem norm to ten significant figures.

    References
    ----------
    - Holmes, S.A. and Featherstone, W.E. "A unified approach to the
      Clenshaw summation and the recursive computation of very high
      degree and order normalised associated Legendre functions."
      Journal of Geodesy 76.5 (2002): 279-299.
    """
    if n_max is None:
        n_max = C.shape[0] - 1

    colat = np.pi / 2 - lat
    x = np.cos(colat)
    u = np.sin(colat)

    P_scaled, _ = associated_legendre_scaled(n_max, n_max, x)
    unscale = 1e280

    r_ratio = R / r
    r_pow = r_ratio ** np.arange(n_max + 1)

    orders = np.arange(n_max + 1)
    cos_m = np.cos(orders * lon)
    sin_m = np.sin(orders * lon)

    # Horner accumulators over decreasing order: after the loop each holds
    # sum_m inner_m * u**m without ever forming u**m directly.
    acc_v = 0.0
    acc_r = 0.0
    acc_t = 0.0

    for m in range(n_max, -1, -1):
        degrees = np.arange(m, n_max + 1)
        coeff = C[m:, m] * cos_m[m] + S[m:, m] * sin_m[m]
        weight = r_pow[m:] * coeff

        column = P_scaled[m:, m]
        inner_v = float(np.sum(weight * column))
        inner_r = float(np.sum(weight * (degrees + 1.0) * column))

        # dPbar_nm/dtheta = u**(m-1) * (n x Ptilde_nm - f_nm Ptilde_{n-1,m}),
        # so this accumulator carries one fewer power of u; the factor is
        # restored after the Horner loop.
        f_nm = np.sqrt(
            (2.0 * degrees + 1.0)
            / np.maximum(2.0 * degrees - 1.0, 1.0)
            * (degrees**2 - m**2)
        )
        shifted = np.zeros_like(column)
        shifted[1:] = P_scaled[m:n_max, m]
        inner_t = float(np.sum(weight * (degrees * x * column - f_nm * shifted)))

        acc_v = acc_v * u + inner_v
        acc_r = acc_r * u + inner_r
        acc_t = acc_t * u + inner_t

    V = GM / r * acc_v * unscale
    dV_r = -GM / (r * r) * acc_r * unscale
    if u > 1e-15:
        dV_theta = GM / r * (acc_t / u) * unscale
    else:
        dV_theta = 0.0
    dV_lat = -dV_theta / r

    return float(V), float(dV_r), float(dV_lat)


def clear_legendre_cache() -> None:
    """Clear cached Legendre polynomial results.

    Call this function to clear the cached associated Legendre
    polynomial arrays. Useful when memory is constrained or after
    processing a batch with different colatitude values.

    Examples
    --------
    >>> _ = associated_legendre(10, 10, 0.5)  # Populate cache
    >>> info = get_legendre_cache_info()
    >>> clear_legendre_cache()
    >>> info_after = get_legendre_cache_info()
    >>> info_after.currsize
    0
    """
    _associated_legendre_cached.cache_clear()
    _logger.debug("Legendre polynomial cache cleared")


def get_legendre_cache_info() -> Any:
    """Get cache statistics for Legendre polynomials.

    Returns
    -------
    CacheInfo
        Named tuple with hits, misses, maxsize, currsize.

    Examples
    --------
    >>> clear_legendre_cache()  # Start fresh
    >>> _ = associated_legendre(5, 5, 0.5)
    >>> info = get_legendre_cache_info()
    >>> info.currsize >= 1  # At least one entry cached
    True
    """
    return _associated_legendre_cached.cache_info()


__all__ = [
    "associated_legendre",
    "associated_legendre_derivative",
    "spherical_harmonic_sum",
    "spherical_harmonic_sum_high_degree",
    "gravity_acceleration",
    "legendre_scaling_factors",
    "associated_legendre_scaled",
    "clear_legendre_cache",
    "get_legendre_cache_info",
]
