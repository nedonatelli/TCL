"""
Clenshaw summation for efficient spherical harmonic evaluation.

The Clenshaw algorithm evaluates spherical harmonic series via backward
recursion over degree for each fixed order, avoiding explicit storage of
the full associated Legendre matrix.

At high degree and small ``u = sin(colatitude)`` two double-precision
hazards appear in the plain algorithm: the backward-recursion partial
sums grow like ``1/u**m`` (overflow to inf), while the sectoral seed
``Pbar_mm ~ u**m`` underflows to zero -- their product (the finite,
physical partial sum) then evaluates as ``inf * 0 = NaN``. Following the
extended-range treatment of Holmes & Featherstone (2002, Sec. 6) and
Wittwer et al. (2008), simplified to a single power-of-ten exponent, this
implementation:

1. dynamically rescales the backward-recursion state by ``1e-140``
   whenever it exceeds ``1e140``, accumulating the shed decades in an
   integer exponent (coefficients injected after a rescale are scaled
   identically, which is faithful: their true weight in the final sum is
   below double precision by construction);
2. seeds the recursion result with the sectoral ratio ``Pbar_mm / u**m``
   (shared with :func:`pytcl.gravity.spherical_harmonics.\
associated_legendre_scaled`), recombining the ``u**m`` envelope and the
   shed exponent in log10 space only when the direct power would
   under- or overflow.

Stability is a measured claim, not an asymptotic one. Verified finite and
in agreement with ``spherical_harmonic_sum_high_degree`` to relative
tolerance 1e-10 (worst observed deviation ~1.3e-12) on random
fully-normalized coefficients
(tests/validation/test_gravity_audit.py::TestClenshawHighDegree):

- potential: ``n_max`` in {50, 500} over colatitudes 0.1-179.9 deg;
  ``n_max`` in {2050, 2190} at colatitudes 15 and 30 deg (4 seeds); and
  ``n_max = 2190`` over colatitudes 0.1-90 deg;
- gradients: ``n_max = 500`` over colatitudes 0.1-150 deg and
  ``n_max = 2190`` at colatitude 30 deg.

``n_max = 2190`` is the EGM2008 maximum. Behavior beyond this grid
(higher degrees, other coefficient statistics) is untested.

References
----------
- Holmes, S.A. and Featherstone, W.E. "A unified approach to the
  Clenshaw summation and the recursive computation of very high
  degree and order normalized associated Legendre functions."
  Journal of Geodesy 76.5 (2002): 279-299.
- Wittwer, T., et al. "Ultra-high degree spherical harmonic analysis
  and synthesis using extended-range arithmetic."
  Journal of Geodesy 82.4-5 (2008): 223-229.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.gravity.spherical_harmonics import _sectoral_ratio

# Rescale the backward-recursion state by 1e-140 whenever it exceeds 1e140.
# One recursion step multiplies by at most ~2*sqrt(2*n_max) ~ 1.4e2 at the
# degrees supported here, so the state can never jump from below the
# threshold past the double-precision overflow limit between checks.
_RESCALE_THRESHOLD = 1e140
_RESCALE_FACTOR = 1e-140
_RESCALE_DECADES = 140


def _recursion_coefficients(m: int, n_max: int) -> Tuple[List[float], List[float]]:
    """Coefficients a_{n+1,m}, b_{n+2,m} for n = m..n_max, as Python lists.

    All integer products fit exactly in float64 for the supported degrees,
    so the vectorized evaluation is bit-identical to per-term evaluation.
    """
    n = np.arange(m, n_max + 1, dtype=np.float64)
    np1 = n + 1.0
    np2 = n + 2.0
    a = np.sqrt((2.0 * np1 + 1.0) * (2.0 * np1 - 1.0) / ((np1 - m) * (np1 + m)))
    b = np.sqrt(
        (2.0 * np2 + 1.0)
        * (np2 + m - 1.0)
        * (np2 - m - 1.0)
        / ((np2 - m) * (np2 + m) * (2.0 * np2 - 3.0))
    )
    return a.tolist(), b.tolist()


def _scaled_power(u: float, p: int, shed_decades: int) -> float:
    """Compute ``u**p * 10**shed_decades`` without spurious under/overflow.

    When no rescaling occurred and the direct power is representable, the
    exact ``u**p`` is used (this is the healthy low-degree path and matches
    the plain algorithm to machine precision). Otherwise the exponents are
    combined in log10 space; the result is the finite, physical magnitude
    even though neither factor is representable on its own.
    """
    if p == 0:
        return 10.0**shed_decades if shed_decades else 1.0
    au = abs(u)
    if au == 0.0:
        return 0.0
    sign = -1.0 if (u < 0.0 and p % 2) else 1.0
    log10_total = p * math.log10(au) + shed_decades
    if shed_decades == 0 and log10_total > -300.0:
        return sign * au**p
    if log10_total < -320.0:
        return 0.0
    return sign * 10.0**log10_total


def clenshaw_sum_order(
    m: int,
    cos_theta: float,
    sin_theta: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    n_max: int,
) -> Tuple[float, float]:
    """Clenshaw summation for fixed order m, summing over degrees n=m to n_max.

    Evaluates the partial sums:
        sum_C = sum_{n=m}^{n_max} C[n,m] * P_n^m(cos_theta)
        sum_S = sum_{n=m}^{n_max} S[n,m] * P_n^m(cos_theta)

    using backward recursion from n_max down to m.

    Parameters
    ----------
    m : int
        Order (fixed for this summation).
    cos_theta : float
        Cosine of colatitude.
    sin_theta : float
        Sine of colatitude.
    C : ndarray
        Cosine coefficients array, shape (n_max+1, n_max+1).
    S : ndarray
        Sine coefficients array, shape (n_max+1, n_max+1).
    n_max : int
        Maximum degree.

    Returns
    -------
    sum_C : float
        Sum of C terms weighted by Legendre functions.
    sum_S : float
        Sum of S terms weighted by Legendre functions.

    Examples
    --------
    >>> import numpy as np
    >>> C = np.zeros((5, 5))
    >>> S = np.zeros((5, 5))
    >>> C[2, 0] = 1.0  # Only C20 term
    >>> cos_theta, sin_theta = np.cos(np.pi/4), np.sin(np.pi/4)
    >>> sum_C, sum_S = clenshaw_sum_order(0, cos_theta, sin_theta, C, S, 4)
    >>> isinstance(sum_C, float)
    True

    Notes
    -----
    Stabilized per Holmes & Featherstone (2002) / Wittwer et al. (2008):
    the backward-recursion state is rescaled by 1e-140 whenever it exceeds
    1e140 (shed decades tracked in an integer exponent), and the sectoral
    envelope ``Pbar_mm = (Pbar_mm / u**m) * u**m`` is recombined with the
    shed exponent in log10 space when the direct powers would under- or
    overflow. Measured stable for ``n_max <= 2190`` -- see the module
    docstring for the exact tested grid.
    """
    # Handle edge case
    if m > n_max:
        return 0.0, 0.0

    x = float(cos_theta)
    u = float(sin_theta)
    a_list, b_list = _recursion_coefficients(m, n_max)
    ax_list = [a * x for a in a_list]
    c_col = C[m : n_max + 1, m].tolist()
    s_col = S[m : n_max + 1, m].tolist()

    # Initialize backward recursion variables
    # s_{n_max+2} = 0, s_{n_max+1} = 0
    s_c_np2 = 0.0  # s^C_{n+2}
    s_c_np1 = 0.0  # s^C_{n+1}
    s_s_np2 = 0.0  # s^S_{n+2}
    s_s_np1 = 0.0  # s^S_{n+1}

    coeff_scale = 1.0  # running 10**(-shed) applied to injected coefficients
    shed = 0  # decades shed from the recursion state so far

    # Backward recursion from n = n_max down to n = m (j = n - m)
    for j in range(n_max - m, -1, -1):
        # Recursion: s_n = a_{n+1,m} * cos_theta * s_{n+1} - b_{n+2,m} * s_{n+2} + c_n
        ax = ax_list[j]
        b = b_list[j]

        s_c_n = ax * s_c_np1 - b * s_c_np2 + c_col[j] * coeff_scale
        s_s_n = ax * s_s_np1 - b * s_s_np2 + s_col[j] * coeff_scale

        if abs(s_c_n) > _RESCALE_THRESHOLD or abs(s_s_n) > _RESCALE_THRESHOLD:
            s_c_n *= _RESCALE_FACTOR
            s_s_n *= _RESCALE_FACTOR
            s_c_np1 *= _RESCALE_FACTOR
            s_s_np1 *= _RESCALE_FACTOR
            coeff_scale *= _RESCALE_FACTOR
            shed += _RESCALE_DECADES

        # Shift for next iteration
        s_c_np2 = s_c_np1
        s_c_np1 = s_c_n
        s_s_np2 = s_s_np1
        s_s_np1 = s_s_n

    # After the loop, s_c_np1 = s_m scaled by 10**(-shed). Multiply by the
    # effective sectoral envelope Pbar_mm * 10**shed to get the actual sum.
    P_eff = _sectoral_ratio(m) * _scaled_power(u, m, shed)

    return P_eff * s_c_np1, P_eff * s_s_np1


def clenshaw_sum_order_derivative(
    m: int,
    cos_theta: float,
    sin_theta: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    n_max: int,
) -> Tuple[float, float, float, float]:
    """Clenshaw summation with derivative for fixed order m.

    Evaluates both the partial sums and their derivatives with respect
    to colatitude.

    Parameters
    ----------
    m : int
        Order.
    cos_theta : float
        Cosine of colatitude.
    sin_theta : float
        Sine of colatitude.
    C : ndarray
        Cosine coefficients.
    S : ndarray
        Sine coefficients.
    n_max : int
        Maximum degree.

    Returns
    -------
    sum_C : float
        Sum of C terms.
    sum_S : float
        Sum of S terms.
    dsum_C : float
        Derivative of sum_C with respect to theta.
    dsum_S : float
        Derivative of sum_S with respect to theta.

    Examples
    --------
    >>> import numpy as np
    >>> C = np.zeros((5, 5))
    >>> S = np.zeros((5, 5))
    >>> C[2, 0] = -0.0005  # J2-like term
    >>> cos_theta, sin_theta = np.cos(np.pi/4), np.sin(np.pi/4)
    >>> sum_C, sum_S, dsum_C, dsum_S = clenshaw_sum_order_derivative(
    ...     0, cos_theta, sin_theta, C, S, 4)
    >>> len([sum_C, sum_S, dsum_C, dsum_S])
    4

    Notes
    -----
    Uses the same Holmes & Featherstone (2002) rescaling as
    :func:`clenshaw_sum_order` (all eight recursion states share one shed
    exponent), with the sectoral derivative in closed form:
    ``dPbar_mm/dtheta = m * cos_theta * u**(m-1) * (Pbar_mm / u**m)``.
    Measured stable for ``n_max <= 2190`` -- see the module docstring for
    the exact tested grid.
    """
    if m > n_max:
        return 0.0, 0.0, 0.0, 0.0

    x = float(cos_theta)
    u = float(sin_theta)
    a_list, b_list = _recursion_coefficients(m, n_max)
    ax_list = [a * x for a in a_list]
    au_list = [a * u for a in a_list]
    c_col = C[m : n_max + 1, m].tolist()
    s_col = S[m : n_max + 1, m].tolist()

    # Backward recursion for both value and derivative
    s_c_np2 = 0.0
    s_c_np1 = 0.0
    s_s_np2 = 0.0
    s_s_np1 = 0.0

    # Also need recursion for derivatives
    ds_c_np2 = 0.0
    ds_c_np1 = 0.0
    ds_s_np2 = 0.0
    ds_s_np1 = 0.0

    coeff_scale = 1.0
    shed = 0

    for j in range(n_max - m, -1, -1):
        ax = ax_list[j]
        au = au_list[j]
        b = b_list[j]

        # Value recursion
        s_c_n = ax * s_c_np1 - b * s_c_np2 + c_col[j] * coeff_scale
        s_s_n = ax * s_s_np1 - b * s_s_np2 + s_col[j] * coeff_scale

        # Derivative recursion (d/d_theta)
        # d(s_n)/d_theta = a * (-sin_theta * s_{n+1} + cos_theta * ds_{n+1}/d_theta)
        #                  - b * ds_{n+2}/d_theta
        ds_c_n = -au * s_c_np1 + ax * ds_c_np1 - b * ds_c_np2
        ds_s_n = -au * s_s_np1 + ax * ds_s_np1 - b * ds_s_np2

        if (
            abs(s_c_n) > _RESCALE_THRESHOLD
            or abs(s_s_n) > _RESCALE_THRESHOLD
            or abs(ds_c_n) > _RESCALE_THRESHOLD
            or abs(ds_s_n) > _RESCALE_THRESHOLD
        ):
            s_c_n *= _RESCALE_FACTOR
            s_s_n *= _RESCALE_FACTOR
            ds_c_n *= _RESCALE_FACTOR
            ds_s_n *= _RESCALE_FACTOR
            s_c_np1 *= _RESCALE_FACTOR
            s_s_np1 *= _RESCALE_FACTOR
            ds_c_np1 *= _RESCALE_FACTOR
            ds_s_np1 *= _RESCALE_FACTOR
            coeff_scale *= _RESCALE_FACTOR
            shed += _RESCALE_DECADES

        # Shift
        s_c_np2, s_c_np1 = s_c_np1, s_c_n
        s_s_np2, s_s_np1 = s_s_np1, s_s_n
        ds_c_np2, ds_c_np1 = ds_c_np1, ds_c_n
        ds_s_np2, ds_s_np1 = ds_s_np1, ds_s_n

    # Effective sectoral envelope and its theta-derivative, carrying the
    # shed decades: Pbar_mm = ratio * u**m, dPbar_mm/dtheta = m*x*ratio*u**(m-1)
    ratio = _sectoral_ratio(m)
    P_eff = ratio * _scaled_power(u, m, shed)
    dP_eff = m * x * ratio * _scaled_power(u, m - 1, shed) if m > 0 else 0.0

    # Final results using product rule
    # d(P_mm * s_m)/d_theta = dP_mm * s_m + P_mm * ds_m
    sum_C = P_eff * s_c_np1
    sum_S = P_eff * s_s_np1
    dsum_C = dP_eff * s_c_np1 + P_eff * ds_c_np1
    dsum_S = dP_eff * s_s_np1 + P_eff * ds_s_np1

    return sum_C, sum_S, dsum_C, dsum_S


def clenshaw_geoid(
    lat: float,
    lon: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    gamma: float,
    n_max: Optional[int] = None,
) -> float:
    """Compute geoid height using Clenshaw summation.

    The geoid height N is the height of the geoid above the reference
    ellipsoid, computed from the disturbing potential T:

        N = T / gamma

    where gamma is the normal gravity on the ellipsoid.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    C : ndarray
        Cosine coefficients (fully normalized), shape (n_max+1, n_max+1).
    S : ndarray
        Sine coefficients (fully normalized), shape (n_max+1, n_max+1).
    R : float
        Reference radius in meters.
    GM : float
        Gravitational parameter in m^3/s^2.
    gamma : float
        Normal gravity at the evaluation point in m/s^2.
    n_max : int, optional
        Maximum degree to use. Default uses full coefficient array.

    Returns
    -------
    float
        Geoid height in meters.

    Notes
    -----
    The geoid height is computed as:

    .. math::

        N = \\frac{GM}{r \\gamma} \\sum_{n=2}^{n_{max}} \\left(\\frac{R}{r}\\right)^n
            \\sum_{m=0}^{n} P_n^m(\\sin\\phi) (C_{nm}\\cos m\\lambda + S_{nm}\\sin m\\lambda)

    The n=0 and n=1 terms are excluded as they represent the reference field.

    Examples
    --------
    >>> import numpy as np
    >>> C = np.zeros((5, 5))
    >>> S = np.zeros((5, 5))
    >>> C[0, 0] = 1.0
    >>> R = 6.378e6
    >>> GM = 3.986e14
    >>> gamma = 9.81
    >>> N = clenshaw_geoid(0, 0, C, S, R, GM, gamma)
    >>> N  # n=0,1 terms are excluded, so a pure central term gives 0
    0.0
    """
    if n_max is None:
        n_max = C.shape[0] - 1

    # Colatitude
    colat = np.pi / 2 - lat
    cos_theta = np.cos(colat)
    sin_theta = np.sin(colat)

    # Exclude the n=0,1 terms (reference field), as documented
    C_dist = np.array(C, dtype=float, copy=True)
    S_dist = np.array(S, dtype=float, copy=True)
    nz = min(2, C_dist.shape[0])
    C_dist[:nz, :] = 0.0
    S_dist[:nz, :] = 0.0

    # On the reference ellipsoid, r ≈ R (simplified), so (R/r)^n = 1
    r = R

    # Sum over all orders m
    V = 0.0
    for m in range(n_max + 1):
        # Get the Clenshaw sum for this order
        sum_C, sum_S = clenshaw_sum_order(
            m, cos_theta, sin_theta, C_dist, S_dist, n_max
        )

        cos_m_lon = np.cos(m * lon)
        sin_m_lon = np.sin(m * lon)

        V += sum_C * cos_m_lon + sum_S * sin_m_lon

    # Bruns' formula: N = T / gamma with T = GM/r * V
    N = GM / (r * gamma) * V

    return N


def clenshaw_potential(
    lat: float,
    lon: float,
    r: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> float:
    """Compute gravitational potential using Clenshaw summation.

    Evaluates the spherical harmonic expansion of the gravitational potential
    efficiently using Clenshaw's algorithm.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from Earth center in meters.
    C : ndarray
        Cosine coefficients (fully normalized).
    S : ndarray
        Sine coefficients (fully normalized).
    R : float
        Reference radius in meters.
    GM : float
        Gravitational parameter in m^3/s^2.
    n_max : int, optional
        Maximum degree.

    Returns
    -------
    float
        Gravitational potential in m^2/s^2.

    Examples
    --------
    >>> import numpy as np
    >>> C = np.zeros((5, 5))
    >>> S = np.zeros((5, 5))
    >>> C[0, 0] = 1.0  # Central term only
    >>> R = 6.378e6
    >>> GM = 3.986e14
    >>> V = clenshaw_potential(0, 0, R, C, S, R, GM)
    >>> abs(V - GM/R) / (GM/R) < 0.01  # ~GM/r for central term
    True
    """
    if n_max is None:
        n_max = C.shape[0] - 1

    # Colatitude
    colat = np.pi / 2 - lat
    cos_theta = np.cos(colat)
    sin_theta = np.sin(colat)

    r_ratio = R / r

    # For proper r^n weighting, we modify the algorithm
    # Create scaled coefficients: C_scaled[n,m] = C[n,m] * (R/r)^n
    C_scaled = np.zeros_like(C)
    S_scaled = np.zeros_like(S)

    r_power = 1.0
    for n in range(n_max + 1):
        C_scaled[n, : n + 1] = C[n, : n + 1] * r_power
        S_scaled[n, : n + 1] = S[n, : n + 1] * r_power
        r_power *= r_ratio

    # Sum over all orders
    V = 0.0

    for m in range(n_max + 1):
        sum_C, sum_S = clenshaw_sum_order(
            m, cos_theta, sin_theta, C_scaled, S_scaled, n_max
        )

        cos_m_lon = np.cos(m * lon)
        sin_m_lon = np.sin(m * lon)

        V += sum_C * cos_m_lon + sum_S * sin_m_lon

    # Scale by GM/r
    V *= GM / r

    return V


def clenshaw_gravity(
    lat: float,
    lon: float,
    r: float,
    C: NDArray[np.floating],
    S: NDArray[np.floating],
    R: float,
    GM: float,
    n_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute gravity disturbance vector using Clenshaw summation.

    Evaluates both the potential and its gradient efficiently using
    Clenshaw's algorithm with derivative recursions.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from Earth center in meters.
    C : ndarray
        Cosine coefficients (fully normalized).
    S : ndarray
        Sine coefficients (fully normalized).
    R : float
        Reference radius in meters.
    GM : float
        Gravitational parameter in m^3/s^2.
    n_max : int, optional
        Maximum degree.

    Returns
    -------
    g_r : float
        Radial component of gravity disturbance (positive outward) in m/s^2.
    g_lat : float
        Northward component of gravity disturbance in m/s^2.
    g_lon : float
        Eastward component of gravity disturbance in m/s^2.

    Examples
    --------
    >>> import numpy as np
    >>> C = np.zeros((5, 5))
    >>> S = np.zeros((5, 5))
    >>> C[0, 0] = 1.0
    >>> R = 6.378e6
    >>> GM = 3.986e14
    >>> g_r, g_lat, g_lon = clenshaw_gravity(0, 0, R, C, S, R, GM)
    >>> g_r < 0  # Gravity points inward
    True
    """
    if n_max is None:
        n_max = C.shape[0] - 1

    # Colatitude
    colat = np.pi / 2 - lat
    cos_theta = np.cos(colat)
    sin_theta = np.sin(colat)

    r_ratio = R / r

    # Create scaled coefficients with r^n and (n+1)*r^n for radial derivative
    C_scaled = np.zeros_like(C)
    S_scaled = np.zeros_like(S)
    C_r_scaled = np.zeros_like(C)  # For radial derivative
    S_r_scaled = np.zeros_like(S)

    r_power = 1.0
    for n in range(n_max + 1):
        C_scaled[n, : n + 1] = C[n, : n + 1] * r_power
        S_scaled[n, : n + 1] = S[n, : n + 1] * r_power
        # Radial derivative coefficient: -(n+1)/r * (R/r)^n
        C_r_scaled[n, : n + 1] = -(n + 1) * C[n, : n + 1] * r_power / r
        S_r_scaled[n, : n + 1] = -(n + 1) * S[n, : n + 1] * r_power / r
        r_power *= r_ratio

    # Initialize gradient sums
    V = 0.0
    dV_r = 0.0
    dV_theta = 0.0
    dV_lon = 0.0

    for m in range(n_max + 1):
        # Value sum
        sum_C, sum_S = clenshaw_sum_order(
            m, cos_theta, sin_theta, C_scaled, S_scaled, n_max
        )

        # Radial derivative sum
        sum_C_r, sum_S_r = clenshaw_sum_order(
            m, cos_theta, sin_theta, C_r_scaled, S_r_scaled, n_max
        )

        # Theta derivative (colatitude)
        _, _, dsum_C, dsum_S = clenshaw_sum_order_derivative(
            m, cos_theta, sin_theta, C_scaled, S_scaled, n_max
        )

        cos_m_lon = np.cos(m * lon)
        sin_m_lon = np.sin(m * lon)

        # Potential
        V += sum_C * cos_m_lon + sum_S * sin_m_lon

        # Radial derivative
        dV_r += sum_C_r * cos_m_lon + sum_S_r * sin_m_lon

        # Colatitude derivative
        dV_theta += dsum_C * cos_m_lon + dsum_S * sin_m_lon

        # Longitude derivative (using d(cos(m*lon))/d_lon = -m*sin(m*lon))
        dV_lon += m * (-sum_C * sin_m_lon + sum_S * cos_m_lon)

    # Scale by GM/r
    scale = GM / r
    dV_r = dV_r * scale  # C_r_scaled already carries one 1/r; total GM/r^2
    dV_theta *= scale / r  # (1/r) * dV/d_theta
    dV_lon *= scale / (r * sin_theta)  # (1/(r*sin_theta)) * dV/d_lon

    # Geodesy-positive potential (V = +GM/r * sum): gravity is g = +grad(V),
    # so the radial component dV_r = -GM/r^2 already points inward
    g_r = dV_r

    # g_lat = (1/r) * dV/d_lat = -(1/r) * dV/d_colat
    g_lat = -dV_theta  # Points north (toward decreasing colatitude)

    # g_lon = (1/(r*sin_theta)) * dV/d_lon
    g_lon = dV_lon

    return g_r, g_lat, g_lon


__all__ = [
    "clenshaw_sum_order",
    "clenshaw_sum_order_derivative",
    "clenshaw_geoid",
    "clenshaw_potential",
    "clenshaw_gravity",
]
