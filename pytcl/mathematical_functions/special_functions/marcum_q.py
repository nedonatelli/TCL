"""
Marcum Q function and related functions.

The Marcum Q function is crucial in radar and communications for
analyzing detection probabilities and signal statistics.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


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
    ``a = 10``, ``m = 1e7``, ``c`` in {-40, ..., 20}: 7e-15 at ``a = 1e3``
    and 2.3e-13 at ``a = 10`` (at ``Q = 5e-176``; ``exp`` scales the error
    of ``log Q`` by ``-log Q``).

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
    incomplete gammas are Poisson tail sums of log pmf terms, with ``a^2/2``
    and ``b^2/2`` carried to twice working precision. ``log Q`` is within
    2.0e-15 on ``a = 1e3``, ``m`` in {3000, 1e4}, ``c`` in {-3, 0, 7, 20},
    and within 8.1e-14 on ``a = 10``, ``m = 1e7``, ``c`` in
    {-40, -10, -3, 0, 3, 10, 20} (the maximum at ``c = -10``, where
    ``log Q = -9.7e-46``).
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

    The series needs a^2/2 and b^2/2 finite and its windows grow like
    sqrt(a*b) and sqrt(b^2/2), so past a*b/2 = 1e4 the quadrature is used.
    The quadrature needs ``(m-1)^2 <= 4ab`` so the Bessel factor neither
    underflows nor cancels in its Hankel expansion.
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


def _log_marcum_q_series(a: float, b: float, m: int) -> float:
    """log Q_m(a, b) as a log-sum-exp of its Poisson-mixture series.

    ``Q_m(a, b) = sum_k Pois(k; a^2/2) Q(m + k, b^2/2)`` and
    ``1 - Q_m(a, b) = sum_k Pois(k; a^2/2) P(m + k, b^2/2)``. The terms are
    unimodal in k with the peak near ``a*b/2``, so only a window around it is
    summed; the window doubles until both edges are below exp(-40) of the
    peak. Above ``Q = 1/2`` the result is ``log1p`` of minus the complement.
    """
    lam, lam_lo = _half_square(a)
    x, x_lo = _half_square(b)
    k_peak = max(0.0, 0.5 * (np.sqrt((m - 1.0) ** 2 + 4.0 * lam * x) - (m + 1.0)))
    half = 50.0 + 10.0 * np.sqrt(k_peak + m)
    while True:
        k = np.arange(max(0.0, np.floor(k_peak - half)), np.ceil(k_peak + half) + 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_w = _log_poisson_pmf(k, lam, lam_lo)
        log_cdf, log_sf = _log_poisson_tails(m + k.astype(np.int64), x, x_lo)
        q_terms = log_w + log_cdf
        p_terms = log_w + log_sf
        if all(
            (k[0] == 0 or t[0] < t.max() - 40.0) and t[-1] < t.max() - 40.0
            for t in (q_terms, p_terms)
        ):
            break
        k_peak = float(k[np.argmax(q_terms)])
        half *= 2.0
    top = q_terms.max()
    log_q = float(top + np.log(np.sum(np.exp(q_terms - top))))
    if log_q < -np.log(2.0):
        return log_q
    top = p_terms.max()
    log_p = top + np.log(np.sum(np.exp(p_terms - top)))
    return float(np.log1p(-np.exp(log_p)))


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
