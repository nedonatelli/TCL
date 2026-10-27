"""
Marcum Q function and related functions.

The Marcum Q function is crucial in radar and communications for
analyzing detection probabilities and signal statistics.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray

from pytcl.core.exceptions import ConvergenceError


def marcum_q(
    a: ArrayLike,
    b: ArrayLike,
    m: int = 1,
) -> NDArray[np.floating]:
    """
    Generalized Marcum Q function Q_m(a, b).

    The Marcum Q function is the complementary cumulative distribution
    function of the noncentral chi-squared distribution and appears
    in radar detection theory.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.
    m : int, optional
        Order of the Marcum Q function (positive integer). Default is 1.

    Returns
    -------
    Q : ndarray
        Values of Q_m(a, b).

    Notes
    -----
    For m = 1, this is the standard Marcum Q function:
    Q_1(a, b) = integral from b to inf of x * exp(-(x^2 + a^2)/2) * I_0(a*x) dx

    The function is related to the noncentral chi-squared distribution:
    Q_m(a, b) = P(X > b^2) where X ~ chi^2(2m, a^2)

    Special cases:
    - Q_m(0, b) = 1 - gammainc(m, b^2/2) = gammaincc(m, b^2/2)
    - Q_m(a, 0) = 1

    Where ``a > 0``, ``b > 0`` and ``a*b > 2e4``, all finite, the value is
    ``exp`` of the log-domain evaluation :func:`log_marcum_q` uses there: a
    quadrature of the Marcum integrand where ``(m-1)^2 <= 4ab``, otherwise
    the Poisson-mixture series. Everywhere else it is
    ``scipy.stats.ncx2.sf``. Before v2.12.0 every element went through
    ``ncx2.sf``, which at large arguments returned 4.7e-3 for ``Q = 0.5``
    (``a = 1e7``) and 8.6e-179 for 2.8e-89 (``a = 1e8``, ``m = 2``,
    ``b - a = 20``), with Boost "Series did not converge" warnings.

    Relative error against the mpmath oracle in
    ``tests/fixtures/marcum_q/oracle.csv``, at ``b = sqrt(a^2 + 2m) + c``
    for ``c`` in {-40, -10, -3, 0, 3, 10, 20}:

    ================  =====================  =========
    ``a``             ``m``                  max error
    ================  =====================  =========
    10                1, 2, 100, 1e4         2.6e-13
    1e3               1, 2, 100              2.9e-14
    1e5               1, 2, 100, 1e4         2.9e-14
    1e7, 1e8          1, 2, 100, 1e4, 1e7    2.9e-14
    ================  =====================  =========

    Where ``a*b > 2e4`` and ``(m-1)^2 > 4ab`` (the series corner), on
    ``a = 1e3``, ``m`` in {3000, 1e4}, ``c`` in {-3, 0, 7, 20} and
    ``a = 10``, ``m = 1e7``, ``c`` in {-40, ..., 20}: 3.6e-15 at ``a = 1e3``
    and 1.4e-14 at ``a = 10``.

    Examples
    --------
    >>> float(marcum_q(0, 0))  # Q_1(0, 0) = 1
    1.0
    >>> round(float(marcum_q(3, 4)), 6)  # Standard Marcum Q
    0.196512

    References
    ----------
    - Marcum, J.I. (1950). "Table of Q Functions".
    - Shnidman, D.A. (1989). "The Calculation of the Probability of
      Detection and the Generalized Marcum Q-Function". IEEE Trans.
      on Information Theory, 35(2), 389-400.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")

    # Broadcast so boolean masking works for any scalar/array combination
    a, b = (np.array(v, dtype=np.float64) for v in np.broadcast_arrays(a, b))

    # Handle edge cases
    result = np.ones(a.shape, dtype=np.float64)

    # Where b == 0, Q_m(a, 0) = 1
    b_zero = b == 0

    # Where a == 0, use incomplete gamma function
    a_zero = (a == 0) & (~b_zero)
    if np.any(a_zero):
        result[a_zero] = sp.gammaincc(m, 0.5 * b[a_zero] ** 2)

    large = _large_product(a, b)
    for idx in map(tuple, np.argwhere(large)):
        result[idx] = np.exp(_log_marcum_q_tail(a[idx], b[idx], m))

    # Q_m(a, b) = P(X > b^2) where X ~ chi^2(2m, a^2)
    general = (~a_zero) & (~b_zero) & (~large)
    if np.any(general):
        from scipy.stats import ncx2

        result[general] = ncx2.sf(b[general] ** 2, 2 * m, a[general] ** 2)

    return result


def _large_product(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray:
    """Elements ``ncx2.sf`` gets wrong: finite, positive, ``a*b > 2e4``."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a * b > 2e4)


def marcum_q1(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """
    Standard Marcum Q function Q_1(a, b).

    Convenience function for the first-order Marcum Q function.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.

    Returns
    -------
    Q : ndarray
        Values of Q_1(a, b).

    Examples
    --------
    >>> round(float(marcum_q1(2, 2)), 6)
    0.603501

    See Also
    --------
    marcum_q : Generalized Marcum Q function.
    """
    return marcum_q(a, b, m=1)


def log_marcum_q(
    a: ArrayLike,
    b: ArrayLike,
    m: int = 1,
) -> NDArray[np.floating]:
    """
    Natural logarithm of the Marcum Q function.

    Computes log(Q_m(a, b)) with better numerical precision for small
    values of Q.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.
    m : int, optional
        Order of the Marcum Q function. Default is 1.

    Returns
    -------
    log_Q : ndarray
        Values of log(Q_m(a, b)).

    Notes
    -----
    Where ``a > 0``, ``b > 0`` and ``a*b > 2e4`` (all finite), where
    ``Q < 1e-10``, and where ``marcum_q`` returns NaN for finite input
    (``a^2`` or ``b^2`` overflowing), the value is computed in the log
    domain. When ``a^2/2`` and ``b^2/2`` are both finite and either
    ``a*b <= 2e4`` or ``(m-1)^2 > 4ab``, it is a log-sum-exp of the
    Poisson-mixture series ``sum_k Pois(k; a^2/2) Q(m + k, b^2/2)``, or of
    its complement ``sum_k Pois(k; a^2/2) P(m + k, b^2/2)`` where
    ``Q > 0.5``.
    Otherwise it is a quadrature of the exact integrand
    ``x (x/a)^(m-1) ive(m-1, a x) exp(-(x-a)^2/2)`` with ``exp(-(b-a)^2/2)``
    factored out, which needs neither squared argument. Elsewhere it is
    ``log Q`` from ``marcum_q`` where ``Q <= 0.5`` and ``log1p(-P)`` above,
    with ``P = 1 - Q`` from ``scipy.stats.ncx2.cdf`` (``gammainc`` at
    ``a = 0``), so ``log Q`` keeps its relative accuracy as ``Q`` nears 1.

    Relative error of ``log Q`` against the mpmath oracle in
    ``tests/fixtures/marcum_q/oracle.csv``, at ``b = sqrt(a^2 + 2m) + c``
    for ``c`` in {-40, -10, -3, 0, 3, 10, 20}:

    ================  =====================  ======================
    ``a``             ``m``                  max error
    ================  =====================  ======================
    10                1, 2, 100              4.2e-15
    10                1e4                    3.0e-14; 1.7e-12 at
                                             ``c = -10``
    1e3               1, 2, 100              7.7e-16
    1e5, 1e7, 1e8     1, 2, 100, 1e4         6.4e-15
    1e7, 1e8          1e7                    1.6e-15
    ================  =====================  ======================

    Where ``a*b > 2e4`` but ``(m-1)^2 > 4ab`` the series is used. Its
    incomplete gammas are Poisson tail sums of log pmf terms, or Temme's
    uniform expansion for orders above 1e4, with ``a^2/2`` and ``b^2/2``
    carried to twice working precision; only the tail sum (``Q`` or
    ``1 - Q``) is formed, and its terms are sampled at a stride below a
    quarter of their width, so memory and time stay bounded for any input
    (at most 2^16 terms; past that it raises ``ConvergenceError``).
    ``log Q`` is within 2.1e-15 on ``a = 1e3``, ``m`` in {3000, 1e4}, ``c``
    in {-3, 0, 7, 20}, and within 3.8e-15 on ``a = 10``, ``m = 1e7``, ``c``
    in {-40, -10, -3, 0, 3, 10, 20}. Called directly on all 159 oracle rows
    (``a`` up to 1e8, where the terms peak at ``k = 5e15``) the series is
    within 5.6e-15. On ``a``, ``b`` in 12 log-spaced values over
    [1e-3, 1e8] and ``m`` in {1, 2, 30, 3000, 1e7}, the traced memory of
    all 1440 calls together peaks at 0.25 MiB.
    v2.11.1 also measured ``c`` in {7, 30, 200, 1000} for ``a`` in
    {0.5, 5, 30, 140}, ``m`` in {1, 20, 200} (below 2.2e-14), and ``c`` in
    {7, 30, 200} for ``a`` in {1e3, 1e5, 1e8, 1e12}, ``m`` in {1, 100}
    (below 5e-13); those sets were not remeasured. Where ``b^2/2``
    overflows and ``b - a`` is large, ``Q`` underflows and the result is
    ``-inf``. For ``a >= 1e100`` it matches ``log Phi(a - b)`` to 2e-13.

    Before v2.11.1 that branch called ``scipy.stats.ncx2.logsf``, which
    underflows to ``-inf`` exactly where this function is needed:
    ``log_marcum_q(1, 40)`` returned ``-inf`` instead of -763.235.

    Examples
    --------
    >>> round(float(log_marcum_q(1, 5)), 6)  # log(Q_1(1, 5))
    -9.506564
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")

    out_shape = np.broadcast_shapes(a.shape, b.shape)
    a_b = np.broadcast_to(a, out_shape).ravel()
    b_b = np.broadcast_to(b, out_shape).ravel()

    large = _large_product(a_b, b_b)
    with np.errstate(over="ignore", invalid="ignore"):
        q_val = np.where(large, 1.0, marcum_q(np.where(large, 0.0, a_b), b_b, m))

    with np.errstate(divide="ignore"):
        result = np.log(q_val)

    # Near Q = 1, log(Q) keeps only the absolute accuracy of Q; log1p of the
    # directly computed 1 - Q keeps the relative accuracy of log Q.
    near_one = ~large & (q_val > 0.5)
    if np.any(near_one):
        from scipy.stats import ncx2

        an, bn = a_b[near_one], b_b[near_one]
        with np.errstate(over="ignore", invalid="ignore"):
            p_val = np.where(
                an == 0.0,
                sp.gammainc(m, 0.5 * bn**2),
                ncx2.cdf(bn**2, 2 * m, an**2),
            )
        result[near_one] = np.log1p(-p_val)

    # marcum_q returns NaN (or a spurious 0) once a^2 or b^2 overflows, so
    # NaN results from finite inputs are recomputed here too.
    finite = np.isfinite(a_b) & np.isfinite(b_b)
    redo = large | (finite & ((q_val < 1e-10) | np.isnan(q_val)))
    for i in np.flatnonzero(redo):
        result[i] = _log_marcum_q_tail(a_b[i], b_b[i], m)
    result[np.isposinf(a_b) & np.isfinite(b_b)] = 0.0

    return result.reshape(out_shape)


def _log_marcum_q_tail(a: float, b: float, m: int) -> float:
    """Route log Q_m(a, b) to the series or to the quadrature.

    The series needs a^2/2 and b^2/2 finite. Past a*b = 2e4 the quadrature,
    measured there first, is kept wherever it applies: it needs
    ``(m-1)^2 <= 4ab`` so the Bessel factor neither underflows nor cancels
    in its Hankel expansion.
    """
    nu2 = (m - 1.0) ** 2
    with np.errstate(over="ignore"):
        lam = 0.5 * a * a
        x = 0.5 * b * b
        ab = a * b
        if a == 0.0:
            return -np.inf if np.isinf(x) else _log_marcum_q_series(a, b, m)
        if np.isfinite(lam) and np.isfinite(x) and (ab <= 2e4 or nu2 > 4.0 * ab):
            return _log_marcum_q_series(a, b, m)
        return _log_marcum_q_quad(a, b, m)


_LAGUERRE = np.polynomial.laguerre.laggauss(60)
_LEGENDRE = np.polynomial.legendre.leggauss(16)


def _log_ive(nu: float, a: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """log of the scaled Bessel ``ive(nu, a*x)``, safe where ``a*x`` overflows.

    scipy's ``ive`` returns NaN above about z = 2e9, so from z = 1e9 the
    Hankel expansion is summed instead. The quadrature only calls this with
    ``nu^2 / (2 z)`` at most 2, where the expansion converges without
    significant cancellation.
    """
    z = a * x
    out = np.empty(x.shape, dtype=np.float64)
    low = z < 1e9
    with np.errstate(divide="ignore"):
        out[low] = np.log(sp.ive(nu, z[low]))
    high = ~low
    if np.any(high):
        xh = x[high]
        total = np.ones(xh.shape)
        term = np.ones(xh.shape)
        mu = 4.0 * nu * nu
        for j in range(1, 60):
            term = -term * ((mu - (2 * j - 1) ** 2) / (8.0 * j * a)) / xh
            total += term
            if np.all(np.abs(term) < 1e-17 * np.abs(total)):
                break
        out[high] = -0.5 * (np.log(2 * np.pi) + np.log(a) + np.log(xh)) + np.log(total)
    return out


def _log_marcum_q_quad(a: float, b: float, m: int) -> float:
    """log Q_m(a, b) by quadrature of the exact Marcum integrand.

    ``Q_m(a, b) = int_b^inf x (x/a)^(m-1) ive(m-1, a x) exp(-(x-a)^2/2) dx``.
    For ``d = b - a >= 8`` the Gaussian tail is factored out as
    ``exp(-d^2/2)`` and the rest integrated by 60-point Gauss-Laguerre in
    ``d*(x - b)``. For ``0 <= d < 8``, ``t = x - a`` is integrated over
    ``[d, 50]`` with 16-point Gauss-Legendre on unit panels; for ``d < 0``
    the mass over ``[max(-a, -45), d]`` is integrated and subtracted from 1.
    """
    nu = m - 1.0
    d = b - a

    def _log_h(t: NDArray[np.floating]) -> NDArray[np.floating]:
        # log(x/a) as log1p(t/a): log(x) - log(a) cancels to an absolute
        # 4e-15, which the factor nu turns into 4e-8 at nu = 1e7.
        x = a + t
        return np.log(x) + nu * np.log1p(t / a) + _log_ive(nu, a, x)

    if d >= 8.0:
        lead = -0.5 * d * d
        if np.isinf(lead):
            return -np.inf
        v, w = _LAGUERRE
        u = v / d
        terms = _log_h(d + u) - 0.5 * u * u + np.log(w)
        top = terms.max()
        return float(lead + top + np.log(np.sum(np.exp(terms - top))) - np.log(d))
    nodes, weights = _LEGENDRE

    def _log_integral(lo: float, hi: float) -> float:
        edges = np.linspace(lo, hi, int(np.ceil(hi - lo)) + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
        half = 0.5 * (edges[1:] - edges[:-1])[:, None]
        t = (mid + half * nodes).ravel()
        terms = _log_h(t) - 0.5 * t * t + np.log((half * weights).ravel())
        top = terms.max()
        return float(top + np.log(np.sum(np.exp(terms - top))))

    if d >= 0.0:
        return _log_integral(d, 50.0)
    # For b < a, Q is 1 minus the mass below b; integrating the upper part
    # directly would let quadrature error push log Q above 0.
    lo = max(-a, -45.0)
    if lo >= d:
        return -0.0
    return float(np.log1p(-np.exp(_log_integral(lo, d))))


_LOG_2PI = float(np.log(2.0 * np.pi))


def _stirlerr(j: NDArray[np.floating]) -> NDArray[np.floating]:
    """``log(j!) - log(sqrt(2 pi j) (j/e)^j)`` for ``j >= 16``."""
    jj = j * j
    return (
        1 / 12 - (1 / 360 - (1 / 1260 - (1 / 1680 - 1 / (1188 * jj)) / jj) / jj) / jj
    ) / j


def _bd0(j: NDArray[np.floating], mu: float) -> NDArray[np.floating]:
    """``j log(j/mu) + mu - j``, without its cancellation near ``j = mu``."""
    with np.errstate(divide="ignore"):
        out = j * np.log(j / mu) + mu - j
    near = np.abs(j - mu) < 0.1 * (j + mu)
    if np.any(near):
        jn = j[near]
        v = (jn - mu) / (jn + mu)
        v2 = v * v
        term = v * v2
        odd = term / 3.0
        for k in range(2, 12):
            term = term * v2
            odd = odd + term / (2 * k + 1)
        out[near] = (jn - mu) * v + 2.0 * jn * odd
    return out


def _half_square(v: float) -> tuple[float, float]:
    """v*v/2 as an unevaluated sum hi + lo (Dekker's exact product)."""
    hi = v * v
    if not np.isfinite(hi):
        return 0.5 * hi, 0.0
    t = 134217729.0 * v
    v1 = t - (t - v)
    v2 = v - v1
    lo = ((v1 * v1 - hi) + 2.0 * v1 * v2) + v2 * v2
    return 0.5 * hi, 0.5 * lo


def _log_poisson_pmf(
    j: NDArray[np.floating], mu: float, mu_lo: float = 0.0
) -> NDArray[np.floating]:
    """``log(mu^j exp(-mu) / j!)`` to an absolute 1e-15 at any ``mu``.

    The direct ``j log(mu) - mu - gammaln(j + 1)`` cancels terms of size
    ``mu log(mu)``: an absolute 1e-9 at ``mu = 5e5``. From ``j = 16`` the
    saddle-point form ``-stirlerr(j) - bd0(j, mu) - log(2 pi j)/2``
    (Loader 2000) is used instead.
    """
    out = sp.xlogy(j, mu) - mu - sp.gammaln(j + 1.0)
    big = j >= 16.0
    if np.any(big):
        jb = j[big]
        out[big] = -_stirlerr(jb) - _bd0(jb, mu) - 0.5 * (_LOG_2PI + np.log(jb))
    if mu_lo:
        out += mu_lo * (j / mu - 1.0)
    return out


def _poisson_reach(j: float, x: float) -> int:
    """Steps from ``j`` (away from the mode of Poisson(x)) for the pmf to fall
    by more than ``e^-45``."""
    ratio = abs(np.log((j + 1.0) / x)) if x > 0 else np.inf
    steps = min(45.0 / ratio if ratio > 0 else np.inf, np.sqrt(90.0 * (j + 1.0)))
    return int(np.ceil(steps)) + 50


def _log_poisson_tails(
    n: NDArray[np.int64], x: float, x_lo: float = 0.0
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """``log P(N <= n - 1)`` and ``log P(N >= n)`` for ``N ~ Poisson(x)``.

    For integer ``n`` these are the regularized incomplete gammas
    ``Q(n, x)`` and ``P(n, x)``. Each is a cumulative ``logaddexp`` of
    log pmf terms, all positive, so both keep their relative accuracy in
    the tail. The pmf window runs to where the terms are ``e^-45`` below
    those it is read at; past the bulk (``mode +- sqrt(90 x) + 50``) a tail
    is read as ``log 1 = 0``.
    """
    mode = np.floor(x)
    bulk = int(np.ceil(np.sqrt(90.0 * x))) + 50
    lo_cdf = int(n.min()) - 1
    hi_cdf = min(int(n.max()) - 1, int(mode) + bulk)
    log_cdf = np.zeros(n.shape)
    if lo_cdf <= hi_cdf:
        start = min(lo_cdf, int(mode) - bulk)
        if lo_cdf < mode - bulk:
            start = lo_cdf - _poisson_reach(lo_cdf, x)
        j = np.arange(max(start, 0), hi_cdf + 1, dtype=np.float64)
        acc = np.logaddexp.accumulate(_log_poisson_pmf(j, x, x_lo))
        inside = n - 1 <= hi_cdf
        log_cdf[inside] = acc[n[inside] - 1 - int(j[0])]
    lo_sf = max(int(n.min()), int(mode) - bulk)
    hi_sf = int(n.max())
    log_sf = np.zeros(n.shape)
    if lo_sf <= hi_sf:
        stop = max(hi_sf, int(mode) + bulk)
        if hi_sf > mode + bulk:
            stop = hi_sf + _poisson_reach(hi_sf, x)
        j = np.arange(max(lo_sf, 0), stop + 1, dtype=np.float64)
        acc = np.logaddexp.accumulate(_log_poisson_pmf(j, x, x_lo)[::-1])[::-1]
        inside = n >= lo_sf
        log_sf[inside] = acc[n[inside] - int(j[0])]
    return log_cdf, log_sf


# Temme's coefficients c_k(eta), k = 0..3, as Taylor series in eta (DLMF
# 8.12.10-11), from exact rational series arithmetic;
# tests/validation/test_marcum_q.py regenerates them.
_TEMME_C = (
    (
        -0.3333333333333333,
        0.08333333333333333,
        -0.014814814814814815,
        0.0011574074074074073,
        0.0003527336860670194,
        -0.0001787551440329218,
        3.919263178522438e-05,
        -2.185448510679992e-06,
        -1.85406221071516e-06,
        8.296711340953087e-07,
        -1.7665952736826078e-07,
        6.707853543401498e-09,
        1.0261809784240309e-08,
        -4.382036018453353e-09,
        9.14769958223679e-10,
        -2.5514193994946248e-11,
        -5.830772132550426e-11,
        2.4361948020667415e-11,
        -5.0276692801141755e-12,
        1.1004392031956135e-13,
        3.371763262400985e-13,
        -1.392388722418162e-13,
        2.8534893807047445e-14,
        -5.139111834242572e-16,
        -1.9752288294349442e-15,
        8.099521156704561e-16,
        -1.6522531216398162e-16,
        2.5305430097478883e-18,
        1.1686939738559576e-17,
        -4.770037049820485e-18,
    ),
    (
        -0.001851851851851852,
        -0.003472222222222222,
        0.0026455026455026454,
        -0.0009902263374485596,
        0.00020576131687242798,
        -4.018775720164609e-07,
        -1.8098550334489977e-05,
        7.64916091608111e-06,
        -1.6120900894563446e-06,
        4.647127802807434e-09,
        1.378633446915721e-07,
        -5.752545603517705e-08,
        1.1951628599778148e-08,
        -1.7543241719747647e-11,
        -1.0091543710600413e-09,
        4.162792991842583e-10,
        -8.56390702649298e-11,
        6.067215101604758e-14,
        7.1624989648114856e-12,
        -2.933186643771437e-12,
        5.996696365683689e-13,
        -2.1671786527323313e-16,
        -4.978339972369262e-14,
        2.0291628823713425e-14,
        -4.13125571381061e-15,
        8.286516239883097e-19,
        3.4100308869333327e-16,
        -1.3854195302893971e-16,
        2.812346653228875e-17,
        -3.406444194143029e-21,
    ),
    (
        0.004133597883597883,
        -0.0026813271604938273,
        0.0007716049382716049,
        2.0093878600823047e-06,
        -0.0001073665322636516,
        5.2923448829120125e-05,
        -1.2760635188618728e-05,
        3.423578734096138e-08,
        1.3721957309062934e-06,
        -6.298992138380055e-07,
        1.4280614206064242e-07,
        -2.0477098421990866e-10,
        -1.409252991086752e-08,
        6.228974084922022e-09,
        -1.3670488396617114e-09,
        9.428356159014678e-13,
        1.2872252400089318e-10,
        -5.5645956134363323e-11,
        1.197593554636698e-11,
        -4.1689782251838634e-15,
        -1.0940640427884595e-12,
        4.662239946390136e-13,
        -9.905105763906907e-14,
        1.8931876768373515e-17,
        8.859221872591127e-15,
        -3.737820398046405e-15,
        7.868833639035156e-16,
        -9.000027395741211e-20,
        -6.928881229347671e-17,
        2.9020384270164786e-17,
    ),
    (
        0.0006494341563786008,
        0.00022947209362139917,
        -0.0004691894943952557,
        0.00026772063206283885,
        -7.561801671883977e-05,
        -2.396505113867297e-07,
        1.1082654115347302e-05,
        -5.6749528269915965e-06,
        1.4230900732435883e-06,
        -2.7861080291528143e-11,
        -1.6958404091930278e-07,
        8.099464905388083e-08,
        -1.9111168485973655e-08,
        2.3928620439808118e-12,
        2.0620131815488797e-09,
        -9.460496661855133e-10,
        2.1541049775774907e-10,
        -1.388823336813903e-14,
        -2.1894761681963938e-11,
        9.790998951171684e-12,
        -2.178219188018096e-12,
        6.208819573407901e-17,
        2.126978363279737e-13,
        -9.344688791517433e-14,
        2.045367122678285e-14,
        -2.58260790403495e-19,
        -1.9405297673344544e-15,
        8.415979290484816e-16,
        -1.8200430439538226e-16,
        1.0735443641247309e-21,
    ),
)

_TEMME_POLY = tuple(np.array(ck[::-1]) for ck in _TEMME_C)
_TEMME_MIN_ORDER = 1e4
_SERIES_MAX_POINTS = 1 << 16


def _log_poisson_q_p_temme(
    n: NDArray[np.floating], x: float
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """``(log Q(n, x), log P(n, x))`` by Temme's uniform expansion.

    ``Q(n, x) = erfc(eta sqrt(n/2))/2 + exp(-n eta^2/2)
    sum_k c_k(eta) n^-k / sqrt(2 pi n)`` with ``n eta^2 / 2 = bd0(n, x)``.
    The smaller of ``Q`` and ``P`` is formed as ``exp(-bd0)`` times a
    bracket that does not cancel for ``|x/n - 1| <= 1/2``, and the other is
    ``log1p`` of minus it. With ``n >= 1e4`` the omitted ``c_4/n^4`` is
    below 1e-19.
    """
    mu = (x - n) / n
    bd = _bd0(n, x)
    sign = np.where(mu >= 0.0, 1.0, -1.0)
    eta = sign * np.sqrt(2.0 * bd / n)
    s = sign * np.sqrt(bd)
    total = np.zeros(n.shape)
    for ck in _TEMME_POLY[::-1]:
        total = total / n + np.polyval(ck, eta)
    corr = total / np.sqrt(2.0 * np.pi * n)
    upper = mu >= 0.0
    bracket = np.where(upper, 0.5 * sp.erfcx(s) + corr, 0.5 * sp.erfcx(-s) - corr)
    log_small = np.log(bracket) - bd
    log_large = np.log1p(-np.exp(log_small))
    return np.where(upper, log_small, log_large), np.where(upper, log_large, log_small)


def _log_poisson_q_p_geometric(
    n: NDArray[np.floating], x: float
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """``(log Q(n, x), log P(n, x))`` where ``|x/n - 1| > 1/2``.

    For ``x > n``, ``Q = pmf(n-1) sum_i prod_{l<=i} (n-l)/x``; for
    ``x < n``, ``P = pmf(n) sum_i prod_{l<=i} x/(n+l)``. Successive ratios
    are below 2/3, so 120 terms reach 1e-21.
    """
    log_q = np.empty(n.shape)
    log_p = np.empty(n.shape)
    up = x > n
    if np.any(up):
        nu = n[up]
        term = np.ones(nu.shape)
        total = np.ones(nu.shape)
        for i in range(1, 121):
            term = term * np.maximum(nu - i, 0.0) / x
            total += term
        small = _log_poisson_pmf(nu - 1.0, x) + np.log(total)
        log_q[up] = small
        log_p[up] = np.log1p(-np.exp(small))
    down = ~up
    if np.any(down):
        nd = n[down]
        term = np.ones(nd.shape)
        total = np.ones(nd.shape)
        for i in range(1, 121):
            term = term * x / (nd + i)
            total += term
        with np.errstate(divide="ignore"):
            small = _log_poisson_pmf(nd, x) + np.log(total)
        log_p[down] = small
        log_q[down] = np.log1p(-np.exp(small))
    return log_q, log_p


def _log_poisson_q_p(
    n: NDArray[np.floating], x: float, x_lo: float = 0.0
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """``(log Q(n, x + x_lo), log P(n, x + x_lo))`` at independent orders ``n``.

    ``Q(n, x) = P(Poisson(x) <= n - 1)`` and ``P = 1 - Q``. Each element
    costs O(1) except integer ``n < 1e4`` within ``|x/n - 1| <= 1/2``,
    which are read from :func:`_log_poisson_tails` over a window of at most
    about 2e4 terms.
    """
    log_q = np.empty(n.shape)
    log_p = np.empty(n.shape)
    near = np.abs(x - n) <= 0.5 * n
    far = ~near
    if np.any(far):
        log_q[far], log_p[far] = _log_poisson_q_p_geometric(n[far], x)
    temme = near & (n >= _TEMME_MIN_ORDER)
    if np.any(temme):
        log_q[temme], log_p[temme] = _log_poisson_q_p_temme(n[temme], x)
    small = near & ~temme
    if np.any(small):
        log_q[small], log_p[small] = _log_poisson_tails(
            n[small].astype(np.int64), x, x_lo
        )
    shift = ~small & (x_lo != 0.0)
    if np.any(shift):
        # dQ/dx = -pmf(n-1; x) = -dP/dx: carry the rounding of x = b^2/2.
        log_pmf = _log_poisson_pmf(n[shift] - 1.0, x)
        log_q[shift] -= x_lo * np.exp(log_pmf - log_q[shift])
        log_p[shift] += x_lo * np.exp(log_pmf - log_p[shift])
    return log_q, log_p


def _log_marcum_q_series(a: float, b: float, m: int) -> float:
    """log Q_m(a, b) from its Poisson-mixture series, in O(1) memory.

    ``Q_m(a, b) = sum_k Pois(k; a^2/2) Q(m + k, b^2/2)`` and
    ``1 - Q_m(a, b) = sum_k Pois(k; a^2/2) P(m + k, b^2/2)``. Only the tail
    sum is formed: ``Q`` where ``b^2 >= a^2 + 2m``, ``1 - Q`` otherwise,
    then ``log1p``. Both sums have log-concave terms (a Poisson pmf times a
    Poisson cdf or survival function in ``k``), so the peak ``k*`` is found
    by ternary search inside a bracket that provably holds it:
    ``[lam - 1, k_u + 1]`` for ``Q``, where ``k_u`` solves
    ``(k+1)(m+k) = lam (m+k+x)``, and ``[0, lam + 1]`` for ``1 - Q``. The
    sum then runs over ``k* +- (12 sigma + 20)``, ``sigma`` from the terms'
    curvature at the peak, widened until both edges are ``e^-40`` below the
    peak. Where ``sigma >= 8`` it samples every ``h = floor(sigma/4)``-th
    term and multiplies by ``h``: the terms are a smooth peak of width
    ``sigma``, so the error is of order ``exp(-2 pi^2 (sigma/h)^2) < 1e-130``;
    the sum is repeated on the grid offset by ``h/2`` and must agree to
    1e-14, else ``h`` is halved. At most ``2^16`` terms are evaluated per
    sum; more raises :class:`~pytcl.core.exceptions.ConvergenceError`.
    """
    lam, lam_lo = _half_square(a)
    x, x_lo = _half_square(b)
    upper = (b - a) * (b + a) >= 2.0 * m
    if lam == 0.0:
        log_q, log_p = _log_poisson_q_p(np.array([float(m)]), x, x_lo)
        return float(log_q[0])

    def _terms(k: NDArray[np.floating]) -> NDArray[np.floating]:
        log_q, log_p = _log_poisson_q_p(m + k, x, x_lo)
        return _log_poisson_pmf(k, lam, lam_lo) + (log_q if upper else log_p)

    if upper:
        lo = max(0.0, np.floor(lam) - 1.0)
        c1 = m + 1.0 - lam
        c0 = m - lam * (m + x)
        k_u = 0.5 * (-c1 + np.sqrt(c1 * c1 - 4.0 * c0)) if c1 * c1 > 4.0 * c0 else lo
        hi = max(lo + 1.0, np.ceil(k_u) + 1.0)
    else:
        lo, hi = 0.0, np.ceil(lam) + 1.0
    while hi - lo > 2.0:
        third = np.floor((hi - lo) / 3.0)
        t1, t2 = _terms(np.array([lo + third, hi - third]))
        if t1 < t2:
            lo = lo + third + 1.0
        else:
            hi = hi - third
    cand = np.arange(lo, hi + 1.0)
    k_star = float(cand[np.argmax(_terms(cand))])
    top = float(_terms(np.array([k_star]))[0])

    d = max(1.0, np.floor(0.5 * np.sqrt(k_star)))
    if k_star >= d:
        t_lo, t_hi = _terms(np.array([k_star - d, k_star + d]))
        curv = (t_lo + t_hi - 2.0 * top) / (d * d)
        sigma = 1.0 / np.sqrt(-curv) if curv < 0.0 else np.sqrt(k_star + 1.0)
    else:
        sigma = 1.0
    h = max(1.0, np.floor(sigma / 4.0))
    reach = 12.0 * sigma + 20.0

    def _window_sum(h: float, offset: float) -> float:
        span = reach
        while True:
            j = np.ceil(span / h)
            k = k_star + offset + h * np.arange(-j, j + 1.0)
            k = k[k >= 0.0]
            if k.size > _SERIES_MAX_POINTS:
                raise ConvergenceError(
                    f"Marcum Q series needs more than {_SERIES_MAX_POINTS} "
                    f"terms at a={a!r}, b={b!r}, m={m}"
                )
            t = _terms(k)
            peak = t.max()
            if (k[0] < h or t[0] < peak - 40.0) and t[-1] < peak - 40.0:
                return float(np.log(h) + peak + np.log(np.sum(np.exp(t - peak))))
            span *= 2.0

    if k_star - reach < 0.0:
        h = 1.0
    log_tail = _window_sum(h, 0.0)
    while h > 1.0:
        other = _window_sum(h, np.floor(h / 2.0))
        if abs(np.expm1(other - log_tail)) <= 1e-14:
            break
        h = max(1.0, np.floor(h / 2.0))
        log_tail = _window_sum(h, 0.0)
    if upper:
        return log_tail
    return float(np.log1p(-np.exp(log_tail)))


def marcum_q_inv(
    a: ArrayLike,
    q: ArrayLike,
    m: int = 1,
) -> NDArray[np.floating]:
    """
    Inverse Marcum Q function.

    Finds b such that Q_m(a, b) = q.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    q : array_like
        Target probability value, 0 < q < 1.
    m : int, optional
        Order of the Marcum Q function. Default is 1.

    Returns
    -------
    b : ndarray
        Values such that Q_m(a, b) = q.

    Notes
    -----
    Closed form via the noncentral chi-squared inverse survival function:
    ``b = sqrt(ncx2.isf(q, 2m, a**2))``. No iteration is involved -- the
    previous ``tol``/``max_iter`` parameters were accepted and never read,
    alongside a Notes claim of Newton-Raphson iteration that no code
    performed.

    Examples
    --------
    >>> b = marcum_q_inv(3, 0.5)  # Find b where Q_1(3, b) = 0.5
    >>> round(float(marcum_q(3, b)), 6)  # Verify
    0.5
    """
    a = np.asarray(a, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("q must be in (0, 1)")

    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")

    from scipy.stats import ncx2

    # Q_m(a, b) = ncx2.sf(b^2, 2m, a^2)
    # So we need b^2 = ncx2.isf(q, 2m, a^2)
    # b = sqrt(ncx2.isf(q, 2m, a^2))

    b_squared = ncx2.isf(q, 2 * m, a**2)
    b = np.sqrt(np.maximum(b_squared, 0))

    return b


def rician_cdf(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """
    Rician cumulative distribution function, ``1 - Q_1(a, b)``.

    Parameters
    ----------
    a : array_like
        Non-centrality parameter, a >= 0.
    b : array_like
        Threshold, b >= 0.

    Returns
    -------
    P : ndarray
        Values of ``1 - Q_1(a, b)``.

    Notes
    -----
    This is the probability ``P(X <= b^2)`` for ``X ~ chi^2(2, a^2)``.

    Formerly exported as ``nuttall_q``, which was a misnomer: the Nuttall Q
    function ``Q_{m,n}(a, b)`` is a different integral, a generalization of the
    Marcum Q with an extra power of the integration variable. This routine
    computes neither -- it is the complementary Marcum Q, which is exactly the
    Rician CDF, and it always did so correctly (gh-20). Only the name was
    wrong. The deprecated ``nuttall_q`` alias was removed in v2.8.0.

    Examples
    --------
    >>> round(float(rician_cdf(2, 2)), 6)  # 1 - Q_1(2, 2)
    0.396499

    See Also
    --------
    marcum_q : Marcum Q function.
    """
    return 1.0 - marcum_q(a, b, m=1)


def swerling_detection_probability(
    snr: ArrayLike,
    pfa: float,
    n_pulses: int = 1,
    swerling_case: int = 1,
) -> NDArray[np.floating]:
    """
    Detection probability for Swerling target models.

    Computes probability of detection for different Swerling cases
    using the Marcum Q function.

    Parameters
    ----------
    snr : array_like
        Signal-to-noise ratio (linear, not dB).
    pfa : float
        Probability of false alarm (0 < pfa < 1).
    n_pulses : int, optional
        Number of integrated pulses. Default is 1.
    swerling_case : int, optional
        Swerling case (0, 1, 2, 3, or 4). Default is 1.
        - 0: Non-fluctuating (Marcum)
        - 1: Slow fluctuation, Rayleigh PDF
        - 2: Fast fluctuation, Rayleigh PDF
        - 3: Slow fluctuation, one dominant + Rayleigh
        - 4: Fast fluctuation, one dominant + Rayleigh

    Returns
    -------
    Pd : ndarray
        Probability of detection.

    Notes
    -----
    The detection threshold T is set from the false alarm probability via
    pfa = gammaincc(n, T/2) (square-law detector, n integrated pulses).

    For Swerling 0 (non-fluctuating):
        P_d = Q_n(sqrt(2*n*SNR), sqrt(T))

    Swerling 1 and 2 use the exact closed forms for chi-squared (2 DOF)
    target fluctuation with scan-to-scan (1) or pulse-to-pulse (2)
    decorrelation. Swerling 3 uses the DiFranco-Rubin closed form for
    chi-squared (4 DOF) scan-to-scan fluctuation, and Swerling 4 the exact
    finite-sum for pulse-to-pulse chi-squared (4 DOF) fluctuation.

    Before v2.11.1 the Swerling 3 branch evaluated only the single-term form
    that is exact for ``n <= 2``, so ``n > 2`` was wrong: -8.9% at
    ``pfa=1e-6, n=6, snr=0.1`` and -21% at ``pfa=1e-8, n=30, snr=0.05``.
    For ``n > 2`` the general-``n`` closed form is used when the mean
    scan SNR ``theta = n*snr/2`` exceeds ``max(1, sqrt(n))``; below that
    its terms cancel (2.6e-11 relative at ``n=1000, theta=1.5, pfa=1e-6``;
    ``-inf`` at ``n=200, theta=0.01``), so the exact negative-binomial
    mixture of gamma tails is summed instead. Against that series in mpmath
    at 40 digits the relative error is below 2e-13 for ``pfa`` 1e-10 to
    1e-2, ``n`` 3 to 5000 and ``theta`` 1e-3 to 300, measured on both sides
    of each switch.

    ``snr = 0`` returns ``pfa`` for every case, elementwise. Cases 1 (for
    ``n > 1``) and 3 used to return NaN or ``-inf`` there.

    Examples
    --------
    >>> pd = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=0)
    >>> pd > 0.9  # High probability of detection with 10 dB SNR
    True

    References
    ----------
    - Swerling, P. (1960). "Probability of Detection for Fluctuating
      Targets". IRE Trans. on Information Theory, IT-6, 269-308.
    """
    snr = np.asarray(snr, dtype=np.float64)
    n = n_pulses

    # Detection threshold from false alarm probability
    # For chi-squared with 2*n_pulses DOF: P(X > T) = Q(n, T/2) = pfa
    threshold = 2 * sp.gammainccinv(n, pfa)
    vt = threshold / 2.0  # normalized threshold

    if swerling_case == 0:
        # Non-fluctuating (Marcum case)
        a = np.sqrt(2 * n * snr)
        b = np.sqrt(threshold)
        return marcum_q(a, b, m=n)

    elif swerling_case == 1:
        # Scan-to-scan Rayleigh fluctuation (exact, DiFranco & Rubin)
        if n == 1:
            return np.exp(-vt / (1 + snr))
        # snr = 0 makes c infinite and the sum inf * 0 = NaN; its limit is pfa.
        zero = snr == 0
        s = np.where(zero, 1.0, snr)
        c = 1 + 1 / (n * s)
        pd = (
            1
            - sp.gammainc(n - 1, vt)
            + c ** (n - 1) * sp.gammainc(n - 1, vt / c) * np.exp(-vt / (1 + n * s))
        )
        return np.asarray(np.where(zero, pfa, pd), dtype=np.float64)

    elif swerling_case == 2:
        # Pulse-to-pulse Rayleigh fluctuation (exact): the integrated sum is
        # gamma-distributed with shape n and per-pulse scale (1 + snr)
        return np.asarray(sp.gammaincc(n, vt / (1 + snr)), dtype=np.float64)

    elif swerling_case == 3:
        # Scan-to-scan chi-squared 4 DOF fluctuation (exact, DiFranco & Rubin).
        # The single-term form K0 is exact only for n <= 2; n > 2 needs the
        # two gamma terms as well (dropping them was -8.9% at n=6).
        zero = snr == 0
        s = np.where(zero, 1.0, snr)
        k = 1 + n * s / 2
        if n <= 2:
            pd = (
                (1 + 2 / (n * s)) ** (n - 2)
                * (1 + vt / k - 2 * (n - 2) / (n * s))
                * np.exp(-vt / k)
            )
            return np.asarray(np.where(zero, pfa, pd), dtype=np.float64)
        # The closed form's terms cancel for mean scan SNR n*snr/2 up to about
        # sqrt(n) (2.6e-11 at n=1000, theta=1.5; -inf at n=200, theta=0.01), so
        # there the exact negative-binomial mixture of gamma tails is summed.
        # Its ratio q = theta/(1+theta) leaves a tail of about J q^J, below
        # 1e-65 for J = 160 (1 + theta).
        theta = n * s / 2
        series = theta <= max(1.0, np.sqrt(n))
        pd = np.empty(s.shape, dtype=np.float64)
        if np.any(series):
            th = theta[series]
            j = np.arange(np.ceil(160.0 * (1.0 + th.max())))
            tails = sp.gammaincc(n + j, vt)
            mix = np.empty(th.shape, dtype=np.float64)
            block = max(1, 2_000_000 // j.size)
            for lo in range(0, th.size, block):
                t = th[lo : lo + block, None]
                log_w = np.log1p(j) + j * np.log(t / (1 + t)) - 2 * np.log1p(t)
                mix[lo : lo + block] = np.exp(log_w) @ tails
            pd[series] = mix
        closed = ~series
        if np.any(closed):
            sc = s[closed]
            kc = k[closed]
            # (1 + 2/(n snr))^(n-2) overflows for n above ~1000; the product
            # with the gamma CDF is formed in log space.
            log_k0 = (
                (n - 2) * np.log1p(2 / (n * sc))
                - vt / kc
                + np.log(sp.gammainc(n - 1, vt / (1 + 2 / (n * sc))))
            )
            pd[closed] = (
                (1 + vt / kc - 2 * (n - 2) / (n * sc)) * np.exp(log_k0)
                + sp.gammaincc(n - 1, vt)
                + np.exp((n - 1) * np.log(vt) - vt - sp.gammaln(n - 1)) / kc
            )
        return np.asarray(np.where(zero, pfa, pd), dtype=np.float64)

    elif swerling_case == 4:
        # Pulse-to-pulse chi-squared 4 DOF fluctuation (exact finite sum).
        # Per-pulse MGF is (1-2s)/(1-(2+snr)s)^2, so the n-pulse sum expands
        # into a finite mixture of gamma tails.
        beta = 2 + snr
        ratio = snr / beta
        pd = np.zeros(snr.shape, dtype=np.float64)
        for k in range(n + 1):
            for j in range(k + 1):
                pd += (
                    sp.comb(n, k, exact=True)
                    * sp.comb(k, j, exact=True)
                    * (-1.0) ** j
                    * ratio**k
                    * sp.gammaincc(n + k - j, threshold / beta)
                )
        return pd

    else:
        raise ValueError(f"swerling_case must be 0-4, got {swerling_case}")


__all__ = [
    "marcum_q",
    "marcum_q1",
    "log_marcum_q",
    "marcum_q_inv",
    "rician_cdf",
    "swerling_detection_probability",
]
