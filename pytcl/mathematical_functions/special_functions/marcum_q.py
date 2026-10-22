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

    # General case: use ncx2 survival function
    # Q_m(a, b) = P(X > b^2) where X ~ chi^2(2m, a^2)
    general = (~a_zero) & (~b_zero)
    if np.any(general):
        from scipy.stats import ncx2

        # Degrees of freedom = 2m, non-centrality = a^2
        result[general] = ncx2.sf(b[general] ** 2, 2 * m, a[general] ** 2)

    return result


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
    Where ``Q < 1e-10`` the value is a log-sum-exp of the Poisson-mixture
    series ``sum_k Pois(k; a^2/2) Q(m + k, b^2/2)``, with the log upper
    incomplete gamma taken from a continued fraction once ``gammaincc``
    underflows. Against that series evaluated in mpmath at 40 digits the
    relative error of ``log Q`` is below 2e-14 for ``a`` in [0, 60],
    ``b - a`` in [9, 93] and ``m`` in [1, 40] (log Q down to about -5000).

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

    q_val = np.atleast_1d(marcum_q(a_b, b_b, m))

    with np.errstate(divide="ignore"):
        result = np.log(q_val)

    small_q = (q_val < 1e-10) & np.isfinite(b_b)
    for i in np.flatnonzero(small_q):
        result[i] = _log_marcum_q_series(a_b[i], b_b[i], m)

    return result.reshape(out_shape)


def _log_gammaincc(s: NDArray[np.floating], x: float) -> NDArray[np.floating]:
    """log of the regularized upper incomplete gamma Q(s, x), past underflow."""
    q = sp.gammaincc(s, x)
    out = np.empty(s.shape, dtype=np.float64)
    ok = q > 1e-280
    out[ok] = np.log(q[ok])
    tail = ~ok
    if np.any(tail):
        # Lentz continued fraction for Gamma(s, x) / (exp(-x) x^s). It only
        # runs where Q underflows, which needs x far above s, so it converges
        # in a few dozen steps.
        st = s[tail]
        tiny = 1e-300
        bk = x + 1.0 - st
        c = np.full(st.shape, 1.0 / tiny)
        d = 1.0 / bk
        h = d.copy()
        for i in range(1, 1000):
            an = -i * (i - st)
            bk = bk + 2.0
            d = an * d + bk
            d = np.where(np.abs(d) < tiny, tiny, d)
            c = bk + an / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            delta = d * c
            h = h * delta
            if np.all(np.abs(delta - 1.0) < 1e-16):
                break
        out[tail] = -x + st * np.log(x) - sp.gammaln(st) + np.log(h)
    return out


def _log_marcum_q_series(a: float, b: float, m: int) -> float:
    """log Q_m(a, b) as a log-sum-exp of its Poisson-mixture series.

    ``Q_m(a, b) = sum_k Pois(k; a^2/2) Q(m + k, b^2/2)``. The terms are
    unimodal in k with the peak near ``a*b/2``, so only a window around it is
    summed; the window doubles until both edges are below exp(-40) of the
    peak.
    """
    lam = 0.5 * a * a
    x = 0.5 * b * b
    k_peak = max(0.0, 0.5 * (np.sqrt((m - 1.0) ** 2 + 4.0 * lam * x) - (m + 1.0)))
    half = 50.0 + 10.0 * np.sqrt(k_peak + m)
    while True:
        k = np.arange(max(0.0, np.floor(k_peak - half)), np.ceil(k_peak + half) + 1.0)
        terms = sp.xlogy(k, lam) - lam - sp.gammaln(k + 1.0) + _log_gammaincc(m + k, x)
        top = terms.max()
        if np.isnan(top) or top == np.inf:
            return float("nan")
        if (k[0] == 0 or terms[0] < top - 40.0) and terms[-1] < top - 40.0:
            return float(top + np.log(np.sum(np.exp(terms - top))))
        k_peak = float(k[np.argmax(terms)])
        half *= 2.0


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
    its terms cancel (2e-11 relative at ``n=1000, theta=1.5``; ``-inf`` at
    ``n=200, theta=0.01``), so the exact negative-binomial mixture of gamma
    tails is summed instead. Against that series in mpmath at 40 digits the
    relative error is below 2e-13 for ``pfa`` 1e-10 to 1e-2, ``n`` 3 to
    5000 and ``theta`` 1e-3 to 300, measured on both sides of each switch.

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
        # sqrt(n) (2e-11 at n=1000, theta=1; -inf at n=200, theta=0.01), so
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
