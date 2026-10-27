"""
Hypergeometric functions.

This module provides hypergeometric functions commonly used in
mathematical physics, probability theory, and special function evaluation.

Performance
-----------
The generalized hypergeometric function uses Numba JIT compilation for
the series summation loop, providing significant speedup for the general
case (p > 2 or q > 1).
"""

import math
import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import scipy.special as sp
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit(cache=True)
def _hypergeometric_series(
    a: np.ndarray[Any, Any],
    b: np.ndarray[Any, Any],
    z: np.ndarray[Any, Any],
    max_terms: int,
    tol: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
    """
    Numba-optimized series summation for generalized hypergeometric function.

    Parameters
    ----------
    a : ndarray
        Numerator parameters (1D array).
    b : ndarray
        Denominator parameters (1D array).
    z : ndarray
        Argument values (1D array).
    max_terms : int
        Maximum number of series terms.
    tol : float
        Convergence tolerance.

    Returns
    -------
    result : ndarray
        Computed pFq values for each z.
    max_term : ndarray
        Largest term magnitude seen for each z; the rounding error of
        ``result`` scales with it, not with ``result``.
    converged : bool
        False when max_terms was exhausted before the tolerance held.
    """
    n_z = len(z)
    p = len(a)
    q = len(b)

    result = np.ones(n_z, dtype=np.float64)
    term = np.ones(n_z, dtype=np.float64)
    max_term = np.ones(n_z, dtype=np.float64)

    for k in range(1, max_terms):
        # Compute numerator product: prod(a_i + k - 1)
        num_factor = 1.0
        for i in range(p):
            num_factor *= a[i] + k - 1

        # Compute denominator product: prod(b_i + k - 1) * k
        den_factor = float(k)
        for i in range(q):
            den_factor *= b[i] + k - 1

        # Update term and result for each z value
        ratio = num_factor / den_factor
        converged = True
        for j in range(n_z):
            term[j] = term[j] * z[j] * ratio
            result[j] += term[j]

            # Relative to the largest term, not the partial sum: an
            # alternating series cancels the sum far below its terms.
            if np.abs(term[j]) > max_term[j]:
                max_term[j] = np.abs(term[j])
            if np.abs(term[j]) >= tol * max_term[j]:
                converged = False

        if converged:
            break

    return result, max_term, converged


# scipy's hyp1f1 takes time proportional to |z| for non-terminating 1F1 at
# z > 0 (14 s at 1e13, over 60 s at 1e14; scipy 1.17.1) and at z < 0 when
# b - a is a non-positive integer (4 s at -1e12, 36 s at -1e13). From 1e9 on
# it is not called: non-terminating 1F1 uses the DLMF 13.7.2 expansion. From
# 1e2 on the expansion is also used where it is at least as accurate as
# scipy: always for z > 0 (where it only ever returns an overflow), and for
# z < 0 when b > 0 and b - a > 0.
_HYP1F1_ASYMPTOTIC_MIN_ABS_Z = 1e2
_HYP1F1_SCIPY_MAX_ABS_Z = 1e9
_LOG_DOUBLE_MAX = math.log(np.finfo(np.float64).max)
_LOG_DOUBLE_TINY = math.log(np.finfo(np.float64).smallest_subnormal)
_LOG_EPS = math.log(np.finfo(np.float64).eps)
_LN2 = math.log(2.0)
_CERTIFIED_SUM_MAX_TERMS = 1_000_000
_ASYMPTOTIC_MAX_TERMS = 200
# scipy's hyp1f1 at 1e2 <= -z < 1e9 loses up to eps * abs(b - a) / d of its
# relative accuracy when the exact b - a is a distance d from a pole of
# Gamma(b - a) (a non-positive integer), b - a not being carried exactly
# inside it. Inside this distance the expansion is used where it certifies.
_POLE_DISTANCE = 1e-6


def _asymptotic_sum(p: float, q: float, x: float) -> float | None:
    """Sum (p)_s (q)_s / (s! x^s) to double precision, or None.

    None when the terms start to grow before they fall below half an ulp
    of the sum: the expansion is then not accurate at this ``x``.
    """
    total, term = 1.0, 1.0
    for s in range(_ASYMPTOTIC_MAX_TERMS):
        nxt = term * (p + s) * (q + s) / ((s + 1) * x)
        if nxt == 0.0:
            return total
        if abs(nxt) >= abs(term):
            return None
        total += nxt
        if abs(nxt) <= 0.5 * np.finfo(np.float64).eps * abs(total):
            return total
        term = nxt
    return None


def _xmul(*factors: float) -> tuple[float, int]:
    """Product as (mantissa, exponent) so intermediates cannot overflow."""
    m, e = 1.0, 0
    for f in factors:
        fm, fe = math.frexp(f)
        m, de = math.frexp(m * fm)
        e += fe + de
    return m, e


def _xpow(x: float, p: float) -> tuple[float, int]:
    """x**p as (mantissa, exponent); p is halved exactly until x**p fits."""
    k = 0
    while abs(p * math.log2(x)) / 2.0**k > 1000.0:
        k += 1
    m, e = math.frexp(x ** (p / 2.0**k))
    for _ in range(k):
        m, de = math.frexp(m * m)
        e = 2 * e + de
    return m, e


def _sinpi(x: float) -> float:
    """sin(pi x) with the argument reduced exactly."""
    n = round(x)
    r = math.sin(math.pi * (x - n))
    return -r if n % 2 else r


def _sinpi_diff(b: float, a: float) -> float:
    """sin(pi (b - a)) without rounding ``b - a`` first.

    Near a pole of Gamma(b - a) the rounding of ``b - a`` is amplified by
    1/sin; here both arguments are reduced exactly and their fractional
    difference is carried with its rounding error (TwoSum).
    """
    nb, na = round(b), round(a)
    rb, ra = b - nb, a - na
    d = rb - ra
    t = d - rb
    err = (rb - (d - t)) + (-ra - t)
    n = round(d)
    r = math.sin(math.pi * ((d - n) + err))
    return -r if (nb - na + n) % 2 else r


def _two_diff(b: Any, a: Any) -> tuple[Any, Any]:
    """``b - a`` as the rounded difference and its exact error (TwoSum)."""
    c = b - a
    t = c - b
    return c, (b - (c - t)) + (-a - t)


def _pole_distance(b: Any, a: Any) -> Any:
    """Distance from the exact ``b - a`` to the nearest non-positive integer.

    Zero exactly when the exact difference is a non-positive integer, a
    pole of Gamma(b - a): the rounded difference alone says -300 for
    -400.3 - (-100.3), whose exact value is -300 - 1.4e-14. Exact for
    abs(b - a) < 2**53; beyond that the TwoSum error can itself be an
    integer and the distance is only approximate (never a false zero).
    Scalars and arrays alike.
    """
    c, err = _two_diff(b, a)
    n = np.minimum(0.0, np.round(c))
    return np.abs((c - n) + err)


def _lgamma_diff(b: float, a: float) -> tuple[float, float]:
    """log(abs(Gamma(b - a))) and the sign of Gamma(b - a), ``b - a`` exact.

    Below 1/2 the reflection formula is used with sin(pi (b - a)) from the
    exactly reduced difference, so a near-pole value is right; at a pole
    the result is (inf, 0.0).
    """
    c = b - a
    if c >= 0.5:
        return float(sp.gammaln(c)), 1.0
    s = _sinpi_diff(b, a)
    if s == 0.0:
        return math.inf, 0.0
    log_abs = math.log(math.pi) - math.log(abs(s)) - float(sp.gammaln(1.0 - c))
    return log_abs, math.copysign(1.0, s)


_STIRLING_COEFFS = tuple(
    bn / (2 * k * (2 * k - 1))
    for k, bn in enumerate(
        (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510),
        start=1,
    )
)


def _gamma_ratio_stirling(b: float, a: float) -> tuple[float, int]:
    """Gamma(b) / Gamma(b - a) for b, b - a >= 20, as (mantissa, exponent).

    The difference of Stirling series, b^a exp(R) with
    R = -(b - a - 1/2) log1p(-a/b) - a + sum_k c_k (b^(1-2k) - (b-a)^(1-2k)),
    never feeds the rounded ``b - a`` to a Gamma function, so the error
    stays near eps * abs(a) instead of growing with b.
    """
    c = b - a
    r = -(c - 0.5) * math.log1p(-a / b) - a
    r += sum(
        ck * (b ** (1 - 2 * k) - c ** (1 - 2 * k))
        for k, ck in enumerate(_STIRLING_COEFFS, start=1)
    )
    m, e = _xpow(b, a)
    k = math.floor(r / _LN2)
    m, de = math.frexp(m * math.exp(r - k * _LN2))
    return m, e + k + de


def _gamma_ratio(b: float, a: float) -> tuple[float, int] | None:
    """Gamma(b) / Gamma(b - a) as (mantissa, exponent), or None.

    Positive arguments are shifted down to about 10 (near 170 an ulp of
    ``b - a`` is 3e-14, which digamma turns into 1e-13 in Gamma, and
    scipy's ``poch`` is no better); negative ones are reflected with an
    exactly reduced sin(pi x). None when no shift fits in 10,000 steps.
    """
    c = b - a
    if b > 0 and c > 0 and min(b, c) >= 20.0:
        return _gamma_ratio_stirling(b, a)
    if b > 0 and c > 0:
        hi, lo = max(b, c), min(b, c)
        n = max(0, math.ceil(hi - 170.0), math.floor(lo - 10.0))
        if lo - n <= 0 or n > 10_000:
            return None
        m, e = _xmul(sp.gamma(b - n), sp.rgamma(b - n - a))
        for k in range(1, n + 1):
            bk = b - k
            fm, fe = math.frexp(bk / (bk - a))
            m, de = math.frexp(m * fm)
            e += fe + de
        return m, e
    if b < 0 and c < 0:
        inner = _gamma_ratio(1.0 - c, a)
        if inner is None:
            return None
        m, e = _xmul(inner[0], _sinpi_diff(b, a) / _sinpi(b))
        return m, e + inner[1]
    if max(b, 1.0 - b) > 170.0 or max(c, 1.0 - c) > 170.0:
        return None
    if b > 0:
        return _xmul(sp.gamma(b), _sinpi_diff(b, a) / math.pi, sp.gamma(1.0 - c))
    return _xmul(math.pi / _sinpi(b), sp.rgamma(1.0 - b), sp.rgamma(c))


def _xvalue(m: float, e: int) -> float:
    with np.errstate(over="ignore", under="ignore"):
        return float(np.ldexp(m, e))


def _proven_overflow(log_mag: float, scale: float) -> int:
    """+1 if ``log_mag`` exceeds the double range beyond its rounding error
    (about ``eps * scale``), -1 if it is below the overflow threshold by
    as much, 0 if undecided."""
    margin = 1e-12 * scale + 1e-9
    if log_mag > _LOG_DOUBLE_MAX + margin:
        return 1
    if log_mag < _LOG_DOUBLE_MAX - margin:
        return -1
    return 0


def _overflow_from_tail(a: float, b: float, z: float) -> float:
    """``+-inf`` when the 1F1 series at z > 0 provably overflows, else NaN.

    From k0 = max(0, ceil(-a), ceil(-b)) on, every term has the sign of
    Gamma(b)/Gamma(a), so the tail is at least its largest term in
    magnitude. The overflow is proven when that term less a bound on the
    sum of the terms before k0 (k0 times the largest of them, which must
    stay under half the tail term) exceeds the double range.
    """
    k0 = max(0, math.ceil(-a), math.ceil(-b))
    if k0 > 1_000_000:
        return np.nan
    log_z = math.log(z)
    log_head_bound = -math.inf
    if k0:
        k = np.arange(k0, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_ratio = np.log(np.abs((a + k) / (b + k))) + log_z - np.log(k + 1)
        log_head = np.concatenate(([0.0], np.cumsum(log_ratio)[:-1]))
        log_head_bound = math.log(k0) + float(np.max(log_head))
        if log_head_bound > _LOG_DOUBLE_MAX - 10.0:
            return np.nan

    def _log_term(k: float) -> float:
        return (
            sp.gammaln(a + k)
            - sp.gammaln(a)
            - sp.gammaln(b + k)
            + sp.gammaln(b)
            + k * log_z
            - sp.gammaln(k + 1.0)
        )

    # The terms peak where (a + k) z = (b + k)(k + 1).
    q = b + 1.0 - z
    disc = q * q - 4.0 * (b - a * z)
    peak = (-q + math.sqrt(disc)) / 2.0 if disc >= 0 else float(k0)
    candidates = {float(k0)} | {
        float(k) for k in (math.floor(peak), math.ceil(peak)) if k >= k0
    }
    best = max(_log_term(k) for k in candidates)
    scale = abs(best) + max(candidates) * (abs(log_z) + 50.0)
    if log_head_bound > best - _LN2:
        return np.nan
    log_lower = best + math.log1p(-math.exp(log_head_bound - best))
    if _proven_overflow(log_lower, scale) == 1:
        return sp.gammasgn(b) * sp.gammasgn(a) * np.inf
    return np.nan


def _certified_series(
    a: float, b: float, z: float, a_err: float = 0.0
) -> tuple[float, float] | None:
    """Sign of 1F1(a; b; z) and the log of a lower bound on its magnitude
    from its series, or None.

    Every term up to N = 2 max(abs(a), abs(b)) + 6 abs(z) + 100 is formed
    by the term-ratio recurrence in mantissa/exponent form (relative error
    under 8 k eps in term k), scaled by the largest term and summed
    exactly (``math.fsum``). Past N each ratio is at most
    r = (N + abs(a)) abs(z) / ((N - abs(b)) (N + 1)) < 1, so the rest is at
    most abs(t_N) r / (1 - r). With E the rounding bound plus that
    remainder, the sign is certified only when abs(sum) > 2 E, and the
    magnitude is then at least abs(sum) - E, which is what is returned
    (the sum itself is known only to within a factor of about 1.5). None
    otherwise, or when N exceeds ``_CERTIFIED_SUM_MAX_TERMS``.

    ``a_err`` is the magnitude of the rounding error in ``a`` when it is a
    rounded difference (Kummer's form): term k of the exact series is term
    k here times prod_{j<k} (1 + err / (a + j)), within a factor
    exp(+-P) of it with P = a_err sum_{j<N} 1 / abs(a + j). P is added to
    the rounding bound (e^P - 1 < 2 P for P <= 1e-2) and is None beyond
    that, which includes every ``a`` within ``a_err`` of a non-positive
    integer, where the exact and the rounded series differ in kind.
    """
    n = math.ceil(2.0 * max(abs(a), abs(b)) + 6.0 * abs(z) + 100.0)
    if n > _CERTIFIED_SUM_MAX_TERMS:
        return None
    k = np.arange(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (a + k) * z / ((b + k) * (k + 1.0))
    if not np.all(np.isfinite(ratio)):
        return None
    perturbation = 0.0
    if a_err:
        with np.errstate(divide="ignore"):
            perturbation = a_err * float(np.sum(1.0 / np.abs(a + k)))
        if not perturbation <= 1e-2:
            return None
    rm, re = np.frexp(ratio)
    mant = np.empty(n + 1)
    expo = np.empty(n + 1, dtype=np.int64)
    mant[0], expo[0] = 0.5, 1
    carry_m, carry_e = 0.5, 1
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        block_m = carry_m * np.cumprod(rm[start:stop])
        block_e = carry_e + np.cumsum(re[start:stop])
        bm, be = np.frexp(block_m)
        mant[start + 1 : stop + 1] = bm
        expo[start + 1 : stop + 1] = block_e + be
        carry_m, carry_e = float(bm[-1]), int(block_e[-1] + be[-1])
    nonzero = mant != 0.0
    top = int(np.max(expo[nonzero]))
    with np.errstate(under="ignore"):
        scaled = np.ldexp(mant, (expo - top).astype(np.int64))
    total = math.fsum(scaled.tolist())
    magnitude = math.fsum(np.abs(scaled).tolist())
    r = (n + abs(a)) * abs(z) / ((n - abs(b)) * (n + 1.0))
    if not r < 1.0:
        return None
    remainder = abs(scaled[-1]) * r / (1.0 - r)
    eps = np.finfo(np.float64).eps
    rounding = (8.0 * (n + 2) * eps + 2.0 * perturbation) * magnitude
    error = rounding + remainder
    if abs(total) <= 2.0 * error:
        return None
    return math.copysign(1.0, total), math.log(abs(total) - error) + top * _LN2


def _certified_overflow(a: float, b: float, z: float) -> float:
    """``+-inf`` when a certified series sum proves 1F1 overflows, else NaN.

    The overflow is proven when the certified lower bound on the magnitude
    exceeds the double range. For z < 0 Kummer's form
    e^z 1F1(b - a; b; -z) is tried first (with b > 0 and b - a > 0 its
    terms are all positive); the rounding error of ``b - a`` is passed on
    to the certification, which accounts for it or declines.
    """
    forms = [(a, b, z, 0.0, 0.0)]
    if z < 0:
        c, err = _two_diff(b, a)
        forms.insert(0, (c, b, -z, z, abs(err)))
    for p, q, w, shift, p_err in forms:
        certified = _certified_series(p, q, w, p_err)
        if certified is None:
            continue
        sign, log_lower = certified
        if _proven_overflow(log_lower + shift, abs(log_lower) + abs(shift)) == 1:
            return sign * np.inf
        return np.nan
    return np.nan


def _hyp1f1_large_z(a: float, b: float, z: float, overflowed: bool = False) -> float:
    """Non-terminating 1F1(a; b; z), b not a pole, at large ``abs(z)``.

    DLMF 13.7.2 is the sum of an exponential term Gamma(b)/Gamma(a)
    e^z z^(a-b) S_e and an algebraic term Gamma(b)/Gamma(b-a) z^(-a) S_a
    (moduli; the phase factor has modulus 1). Each series must reach half
    an ulp before its terms grow, and the other term must be below an ulp
    of the dominant one; otherwise NaN. For z > 0 only a proven overflow is
    returned (as ``+-inf``): a finite value from the logarithms would lose
    about ``z * eps``. ``overflowed`` marks a z where scipy returned
    ``+-inf``; that is passed on only once proven, here or from the series
    tail (``_overflow_from_tail``).
    """
    if z > 0:
        sign = sp.gammasgn(b) * sp.gammasgn(a)
        if np.isinf(z):
            return sign * np.inf
        fallback = _overflow_from_tail(a, b, z) if overflowed else np.nan
        series = _asymptotic_sum(b - a, 1.0 - a, z)
        if series is None:
            return fallback
        log_z = math.log(z)
        lg_a, lg_b = sp.gammaln(a), sp.gammaln(b)
        log_mag = z + (a - b) * log_z + lg_b - lg_a + math.log(abs(series))
        if _pole_distance(b, a) > 0.0:
            alg_series = _asymptotic_sum(a, a - b + 1.0, z)
            if alg_series is None:
                return fallback
            lg_c = _lgamma_diff(b, a)[0]
            log_alg = lg_b - lg_c - a * log_z + math.log(abs(alg_series))
            if log_alg >= log_mag + _LOG_EPS - 1.0:
                return fallback
        scale = z + abs(a - b) * abs(log_z) + abs(lg_a) + abs(lg_b)
        if _proven_overflow(log_mag, scale) == 1:
            return math.copysign(np.inf, sign * series)
        return np.nan

    x = -z
    # The exact b - a decides whether the algebraic term vanishes: the
    # rounded difference is an integer for (-100.3, -400.3), where the term
    # is 4.4e734 at z = -1e10 (fl(b - a) = -300, exact -300 - 1.4e-14).
    algebraic_zero = _pole_distance(b, a) == 0.0
    if np.isinf(x):
        if algebraic_zero or a > 0:
            return 0.0
        return sp.gammasgn(b) * _lgamma_diff(b, a)[1] * np.inf
    if algebraic_zero:
        value, log_alg = 0.0, -np.inf
    else:
        series = _asymptotic_sum(a, a - b + 1.0, x)
        if series is None:
            return np.nan
        coef = _gamma_ratio(b, a)
        if coef is None:
            # Only a result provably out of the double range is returned.
            lg_b = sp.gammaln(b)
            lg_c, sign_c = _lgamma_diff(b, a)
            log_alg = lg_b - lg_c - a * math.log(x) + math.log(abs(series))
            scale = abs(lg_b) + abs(lg_c) + abs(a * math.log(x))
            sign = sp.gammasgn(b) * sign_c * math.copysign(1.0, series)
            if _proven_overflow(log_alg, scale) == 1:
                value = sign * np.inf
            elif log_alg < _LOG_DOUBLE_TINY - 1e-12 * scale - 1.0:
                value = sign * 0.0
            else:
                return np.nan
        else:
            pm, pe = _xpow(x, -a)
            m, e = _xmul(coef[0], pm, series)
            e += coef[1] + pe
            value = _xvalue(m, e)
            log_alg = math.log(abs(m)) + e * math.log(2.0)
    floor = max(_LOG_DOUBLE_TINY, log_alg + _LOG_EPS - 1.0)
    log_exp = -x + (a - b) * math.log(x) + sp.gammaln(b) - sp.gammaln(a)
    exp_series = _asymptotic_sum(b - a, 1.0 - a, x)
    if exp_series is None or log_exp + math.log(abs(exp_series)) >= floor:
        return np.nan
    return value


def _hyp1f1(a: ArrayLike, b: ArrayLike, z: ArrayLike) -> NDArray[np.floating]:
    """scipy's hyp1f1, routed away from where it hangs or overflows wrongly."""
    a, b, z = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    finite_params = np.isfinite(a) & np.isfinite(b)
    terminating = finite_params & (a <= 0) & (a == np.round(a))
    general = finite_params & ~terminating & ~((b <= 0) & (b == np.round(b)))
    beyond = np.abs(z) >= _HYP1F1_SCIPY_MAX_ABS_Z
    large = general & (np.abs(z) >= _HYP1F1_ASYMPTOTIC_MIN_ABS_Z)
    # At a pole itself (b - a exactly a non-positive integer) the algebraic
    # term vanishes and scipy is right; it is the near miss that it rounds.
    # Where the expansion does not certify, scipy's value is kept there as
    # everywhere else: its near-pole error cannot be estimated pointwise
    # (see the docstring).
    pole_distance = np.full(z.shape, np.inf)
    pole_distance[finite_params] = _pole_distance(b[finite_params], a[finite_params])
    near_pole = (
        large & (z < 0) & (pole_distance > 0.0) & (pole_distance < _POLE_DISTANCE)
    )
    far = large & (beyond | (z > 0) | ((b > 0) & (b - a > 0)) | near_pole)
    result = np.full(z.shape, np.nan)
    for i in np.flatnonzero(far):
        result.flat[i] = _hyp1f1_large_z(a.flat[i], b.flat[i], z.flat[i])
    polynomial_limit = terminating & np.isinf(z)
    for i in np.flatnonzero(polynomial_limit):
        result.flat[i] = _pfq_limit_at_infinity(
            a.flat[i : i + 1], b.flat[i : i + 1], z.flat[i]
        )
    uncertified = far & np.isnan(result)
    unanswered = uncertified & beyond
    use_scipy = (~far & ~polynomial_limit) | (uncertified & ~beyond)
    result[use_scipy] = sp.hyp1f1(a[use_scipy], b[use_scipy], z[use_scipy])
    scipy_inf = use_scipy & np.isinf(result) & np.isfinite(z)
    overflowed = scipy_inf & general
    for i in np.flatnonzero(overflowed):
        ai, bi, zi = a.flat[i], b.flat[i], z.flat[i]
        value = _hyp1f1_large_z(ai, bi, zi, overflowed=True)
        if np.isnan(value):
            value = _certified_overflow(ai, bi, zi)
        result.flat[i] = value
    for i in np.flatnonzero(scipy_inf & terminating):
        result.flat[i] = _exact_polynomial_value(
            a.flat[i : i + 1], b.flat[i : i + 1], z.flat[i]
        )
    unproven = scipy_inf & (general | terminating) & np.isnan(result)
    if np.any(unanswered) or np.any(unproven):
        warnings.warn(
            "hyp1f1: returning NaN. Either the large-|z| asymptotic "
            "expansion (DLMF 13.7.2) does not reach double precision here "
            "and scipy's hyp1f1 is not used at |z| >= 1e9 (its run time "
            "grows with |z|), or scipy overflowed and the overflow could "
            "not be proven.",
            RuntimeWarning,
            stacklevel=3,
        )
    return result


def hyp0f1(
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Confluent hypergeometric limit function 0F1(b; z).

    The function 0F1(b; z) is defined by the series:
    0F1(b; z) = sum_{k=0}^inf z^k / ((b)_k * k!)

    where (b)_k is the Pochhammer symbol (rising factorial).

    Parameters
    ----------
    b : array_like
        Numerator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function.

    Returns
    -------
    F : ndarray
        Values of 0F1(b; z).

    Notes
    -----
    Related to Bessel functions:
    J_n(x) = (x/2)^n / Gamma(n+1) * 0F1(n+1; -x^2/4)
    I_n(x) = (x/2)^n / Gamma(n+1) * 0F1(n+1; x^2/4)

    Examples
    --------
    >>> float(hyp0f1(1, 0))  # 0F1(1; 0) = 1
    1.0
    >>> round(float(hyp0f1(1, 1)), 6)
    2.279585

    References
    ----------
    - NIST Digital Library of Mathematical Functions, Chapter 16.
    """
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyp0f1(b, z), dtype=np.float64)


def hyp1f1(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Confluent hypergeometric function 1F1(a; b; z) (Kummer's function M).

    The function 1F1(a; b; z) is defined by the series:
    1F1(a; b; z) = sum_{k=0}^inf (a)_k * z^k / ((b)_k * k!)

    Parameters
    ----------
    a : array_like
        Numerator parameter.
    b : array_like
        Denominator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function.

    Returns
    -------
    F : ndarray
        Values of 1F1(a; b; z).

    Notes
    -----
    Also known as Kummer's function M(a, b, z).

    scipy's ``hyp1f1`` takes time proportional to ``abs(z)`` (about 14 s at
    ``z = 1e13``, over 60 s beyond) and is not called at ``abs(z) >= 1e9``:
    non-terminating 1F1 there uses the DLMF 13.7.2 asymptotic expansion.
    From ``abs(z) = 1e2`` the expansion also replaces scipy at ``z > 0``
    (where it only returns proven overflows) and at ``z < 0`` when
    ``b > 0`` and ``b - a > 0``. Measured against mpmath (an exact series
    sum where ``b < 0``), ``a`` in [-60, 60], ``abs(b)`` up to 1e4; every
    figure is the maximum measured over the stated random sample, not a
    bound:

    - expansion at ``1e2 <= -z < 1e9``: 3.1e-14 relative over 939 points,
      where scipy reached 2.6e-11; scipy was better at 34 of them, by
      about 1e-14;
    - expansion at ``-z >= 1e9``, either sign of ``b``: 1.4e-14 over two
      samples of 600 and 1,000 points up to 1e13;
    - across ``z = +-1e2`` and ``+-1e9`` the value moved 2.4e-14 more
      than mpmath's over 20 parameter sets;
    - near a pole of Gamma(b - a), that is when the exact ``b - a`` is
      within 1e-6 of a non-positive integer but not at one, at
      ``1e2 <= -z < 1e9``: the expansion where it certifies (1.1e-14 over
      150 random points at ``-z`` to 2e4; ``hyp1f1(25.6, 0.6, -1000)`` is
      1.047e-66 where scipy gives -0.0), else scipy's value. scipy does
      not carry ``b - a`` exactly there and its relative error reaches
      ``eps * abs(b - a) / d`` with ``d`` the distance to the pole (the
      maximum over 150 random points at ``-z`` to 2e4: 2.3e-7 at ``d`` in
      [1e-8, 1e-6), 2.0e-1 below 1e-12), but it is often accurate: on
      2,182 random points with ``abs(a) <= 150``, ``-z`` from 1e2 to 3e3
      and the expansion not certifying, scipy was within 1e-12 at 1,440
      and worse at 742 (its error is not predictable from the inputs: a
      one-ulp shift of ``a`` often leaves it unchanged, and the bound
      above is exceeded at every one of the 1,440 too). Unfixed scipy
      values of this kind: ``hyp1f1(-100.3, -400.3, -5000)`` is 0.0 (exact
      9.366e99; ``fl(b - a) = -300``, exact -300 - 1.4e-14),
      ``hyp1f1(12.3, 2.3, -100)`` is off by 45%, ``hyp1f1(20.1, 0.1, -150)``
      and ``hyp1f1(30.1, 10.1, -300)`` have the wrong sign. The expansion's
      own integer tests use the exact difference (TwoSum), so
      ``hyp1f1(-100.3, -400.3, -1e10)`` is ``+inf`` (4.4e734) where it was
      0.0.
    - elsewhere the value is scipy's, unchanged, including its errors: up
      to 3.4e-11 at ``z < 0`` (b near 6e3) and 1.4e-11 at finite ``z > 0``
      in the samples above; its near-pole loss at ``-z < 1e2`` (5.0e-3 at
      ``d`` below 1e-12 over 150 points) and at ``z > 0`` (3.4e-13 at
      ``hyp1f1(-100.3, -400.3, 500)``); and its silent NaN for large
      parameters (``max(abs(a), abs(b))`` above about 150) at ``abs(z)``
      from 1e-300 to 1e4: 14 of the 150,000 points above (all with
      ``max(abs(a), abs(b)) >= 277`` and ``abs(z)`` from 10 to 1e4),
      ``hyp1f1(116.18, -146.25, 11.66)``, ``hyp1f1(-1401.302,
      -3092.199, 1950.047)`` (an overflow), ``hyp1f1(1, 9999.5, -1e-300)``,
      ``hyp1f1(9999.5, 0.5, -1)`` and ``hyp1f1(-1e4, 1, 1)`` return NaN
      without a warning; the values of the last three are 1, 0.29107 and
      -0.025079.

    A ``+-inf`` is returned only when a lower bound on the magnitude is
    proven to exceed the double range: from the expansion's logarithm,
    from the series tail (past ``k = max(-a, -b)`` every term has one
    sign, so the tail is at least its largest term, less a bound on the
    terms before), or from the whole series summed with a certified sign
    and error bound (term magnitudes in mantissa/exponent form, directly
    or in Kummer's form e^z 1F1(b - a; b; -z) for ``z < 0``, whose rounded
    ``b - a`` is accounted for), which needs at most 1e6 terms
    (``2 max(abs(a), abs(b)) + 6 abs(z) + 100``); a terminating series that
    scipy overflows is summed exactly. A value below the double range is a
    signed 0.0. Otherwise the result is NaN with a ``RuntimeWarning``: on
    150,000 random points (``abs(a)`` to 200, ``abs(b)`` to 1e4,
    ``abs(z)`` to 1e6) that happened at 127 of 29,630 scipy overflows, all
    beyond the 1e6-term cap (``abs(z)`` above 1.6e5; none in the band
    between the certified lower bound and the sum itself), and it happens
    at ``abs(z) >= 1e9`` when the expansion does not reach double
    precision. scipy returns ``+inf`` for every overflow whatever its
    sign. At ``z = +-inf`` the limit is returned. A certified sum near the
    cap costs about 0.2 s per element (0.4 s at ``z < 0``, where both
    forms may be tried).

    Special cases:
    - 1F1(0; b; z) = 1
    - 1F1(a; a; z) = exp(z)
    - 1F1(1; 2; 2z) = sinh(z) * exp(z) / z

    Related to incomplete gamma:
    gammainc(a, z) = z^a * exp(-z) * 1F1(1; 1+a; z) / (a * Gamma(a))

    Examples
    --------
    >>> round(float(hyp1f1(1, 1, 1)), 6)  # exp(1)
    2.718282
    >>> round(float(hyp1f1(0.5, 1.5, -1)), 6)  # erf(1) * sqrt(pi) / 2
    0.746824

    References
    ----------
    - Abramowitz & Stegun, "Handbook of Mathematical Functions", Ch. 13.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return _hyp1f1(a, b, z)


def hyp2f1(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Gauss hypergeometric function 2F1(a, b; c; z).

    The function 2F1(a, b; c; z) is defined by the series::

        2F1(a, b; c; z) = sum_{k=0}^inf (a)_k * (b)_k * z^k / ((c)_k * k!)

    converging for ``|z| < 1``.

    Parameters
    ----------
    a : array_like
        First numerator parameter.
    b : array_like
        Second numerator parameter.
    c : array_like
        Denominator parameter. Must not be a non-positive integer.
    z : array_like
        Argument of the function. For ``|z| >= 1``, analytic continuation
        is used.

    Returns
    -------
    F : ndarray
        Values of 2F1(a, b; c; z).

    Notes
    -----
    Many elementary and special functions are special cases:
    - (1-z)^(-a) = 2F1(a, b; b; z)
    - log(1+z)/z = 2F1(1, 1; 2; -z)
    - arcsin(z)/z = 2F1(1/2, 1/2; 3/2; z^2)
    - Complete elliptic integrals K(k) and E(k)

    Examples
    --------
    >>> round(float(hyp2f1(1, 1, 2, 0.5)), 6)  # -log(1-0.5)/0.5 = 2*log(2)
    1.386294
    >>> round(float(hyp2f1(0.5, 0.5, 1.5, 0.25)), 6)  # arcsin(0.5)/0.5 = pi/3
    1.047198

    References
    ----------
    - NIST DLMF, Chapter 15.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyp2f1(a, b, c, z), dtype=np.float64)


def hyperu(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Confluent hypergeometric function U(a, b, z) (Tricomi function).

    The function U(a, b, z) is defined as::

        U(a, b, z) = Gamma(1-b)/Gamma(a-b+1) * 1F1(a; b; z)
                     + Gamma(b-1)/Gamma(a) * z^(1-b) * 1F1(a-b+1; 2-b; z)

    Parameters
    ----------
    a : array_like
        First parameter.
    b : array_like
        Second parameter.
    z : array_like
        Argument of the function (must be positive for real result).

    Returns
    -------
    U : ndarray
        Values of U(a, b, z).

    Notes
    -----
    Also known as Tricomi's function or Kummer's function of the second kind.

    Asymptotic behavior for large z:
    U(a, b, z) ~ z^(-a) as z -> inf

    Examples
    --------
    >>> round(float(hyperu(1, 1, 1)), 6)
    0.596347
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return np.asarray(sp.hyperu(a, b, z), dtype=np.float64)


def hyp1f1_regularized(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.floating]:
    """
    Regularized confluent hypergeometric function 1F1(a; b; z) / Gamma(b).

    This is useful when b may be near a non-positive integer.

    Parameters
    ----------
    a : array_like
        Numerator parameter.
    b : array_like
        Denominator parameter.
    z : array_like
        Argument of the function.

    Returns
    -------
    F : ndarray
        Values of 1F1(a; b; z) / Gamma(b).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.mathematical_functions.special_functions import hyp1f1_regularized
    >>> # Regularized form avoids overflow for problematic b values
    >>> a, b, z = 0.5, 1.5, 1.0
    >>> f_reg = hyp1f1_regularized(a, b, z)
    >>> # Should give finite, non-overflowing result
    >>> np.isfinite(f_reg)
    True
    >>> # Compare to regular hypergeometric computation
    >>> from pytcl.mathematical_functions.special_functions import hyp1f1
    >>> import scipy.special as sp
    >>> f_normal = hyp1f1(a, b, z) / sp.gamma(b)
    >>> np.allclose(f_reg, f_normal)
    True

    Notes
    -----
    This function remains finite even when b is a non-positive integer,
    unlike the standard 1F1. For b = -m (m = 0, 1, ...) the limit is

        (a)_{m+1} z^{m+1} / (m+1)! * 1F1(a+m+1; m+2; z)

    so that ``hyp1f1_regularized(1, 0, 1)`` is e (it was nan, from
    ``hyp1f1/gamma`` evaluating inf/inf). Large ``abs(z)`` is evaluated as
    in :func:`hyp1f1`. The shifted parameter ``a + m + 1`` is rounded. At
    ``z < 0`` the value has a part proportional to sin(pi a), which is
    sensitive to that rounding when ``1 - a`` is within 1e-6 of a
    non-positive integer; the sensitivity is measured with a one-ulp
    shift of the parameter, and when the rounding moves the value by more
    than 1e-12 NaN with a ``RuntimeWarning`` is returned
    (``hyp1f1_regularized(1.0000000000000002, -3, -1e10)``, whose value is
    -5.33e-25, was 0.0; at ``b = -3`` and ``a = 1 + 2 ulp`` the refusal
    starts at ``z = -25``, where the rounded value was off by 5e-11, and
    ``z = -10`` keeps its 2.4e-15; at ``z >= 0`` and at small ``abs(z)``
    nothing changes, ``hyp1f1_regularized(1.0000000000000002, -3, 1)`` is
    e to 3.9e-16). The prefactor ``(a)_{m+1} z^{m+1} / (m+1)!`` is
    evaluated in double precision and overflows for large ``m`` or
    ``abs(z)``, giving NaN or ``+-inf`` for a finite value
    (``hyp1f1_regularized(32.000000000000007, -33, -1e9)`` is NaN,
    5.9e-212; ``hyp1f1_regularized(0.5, -150, 3)`` is ``inf``, 1.01e72).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    a, b, z = np.broadcast_arrays(a, b, z)
    pole = (b <= 0.0) & (b == np.round(b))
    result = np.empty(z.shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result[~pole] = _hyp1f1(a[~pole], b[~pole], z[~pole]) / sp.gamma(b[~pole])
        a, m1, z = a[pole], 1.0 - b[pole], z[pole]
        shifted, shift_err = _two_diff(a, -m1)
        prefactor = sp.poch(a, m1) * z**m1 * sp.rgamma(m1 + 1.0)
        at_pole = prefactor * _hyp1f1(shifted, m1 + 1.0, z)
        # The shifted parameter a + m + 1 is rounded. At z < 0 the value has
        # a part proportional to sin(pi a), which 1 - a near a pole of
        # Gamma(1 - a) makes sensitive to that rounding: the sensitivity is
        # measured by a one-ulp shift and the value refused when it moves
        # by more than 1e-12 (from z = -25 on for a = 1 + 2 ulp, b = -3).
        check = np.isfinite(a) & (shift_err != 0.0) & (z < 0.0)
        check[check] &= _pole_distance(1.0, a[check]) < _POLE_DISTANCE
        unsafe = np.zeros(z.shape, dtype=bool)
        for i in np.flatnonzero(check):
            v0, step = float(at_pole.flat[i]), float(np.spacing(shifted.flat[i]))
            v1 = prefactor.flat[i] * float(
                _hyp1f1(shifted.flat[i] + step, m1.flat[i] + 1.0, z.flat[i])
            )
            if v0 == 0.0 or not np.isfinite(v1):
                unsafe.flat[i] = True
                continue
            shift = abs(v1 - v0) / abs(v0) * abs(float(shift_err.flat[i])) / step
            unsafe.flat[i] = shift > 1e-12
        result[pole] = np.where(unsafe, np.nan, at_pole)
    if np.any(unsafe):
        warnings.warn(
            "hyp1f1_regularized: returning NaN at a non-positive integer b "
            "where a + 1 - b is rounded, 1 - a is within 1e-6 of a "
            "non-positive integer, and the rounding moves the value by more "
            "than 1e-12 (its sin(pi a) part is lost in it).",
            RuntimeWarning,
            stacklevel=2,
        )
    return result[()]


def pochhammer(
    a: ArrayLike,
    n: ArrayLike,
) -> NDArray[np.floating]:
    """
    Pochhammer symbol (rising factorial) (a)_n.

    The Pochhammer symbol is defined as:
    (a)_n = a * (a+1) * (a+2) * ... * (a+n-1) = Gamma(a+n) / Gamma(a)

    Parameters
    ----------
    a : array_like
        Base value.
    n : array_like
        Number of terms (can be non-integer for generalization).

    Returns
    -------
    p : ndarray
        Values of (a)_n.

    Notes
    -----
    Special cases:
    - (a)_0 = 1
    - (1)_n = n!
    - (a)_1 = a

    Examples
    --------
    >>> float(pochhammer(1, 5))  # 5!
    120.0
    >>> float(pochhammer(3, 4))  # 3*4*5*6
    360.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    return np.asarray(sp.poch(a, n), dtype=np.float64)


def falling_factorial(
    a: ArrayLike,
    n: ArrayLike,
) -> NDArray[np.floating]:
    """
    Falling factorial (a)_n (Pochhammer symbol variant).

    The falling factorial is defined as:
    (a)_n = a * (a-1) * (a-2) * ... * (a-n+1)

    Parameters
    ----------
    a : array_like
        Base value.
    n : array_like
        Number of terms.

    Returns
    -------
    f : ndarray
        Values of the falling factorial.

    Notes
    -----
    Related to rising factorial:
    (a)_n (falling) = (-1)^n * (-a)_n (rising)

    Examples
    --------
    >>> float(falling_factorial(5, 3))  # 5*4*3
    60.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    # (a)_n falling = a!/(a-n)! = Gamma(a+1)/Gamma(a-n+1)
    return np.asarray(sp.poch(a - n + 1, n), dtype=np.float64)


_EPS = float(np.finfo(np.float64).eps)
_CANCELLATION_WARN = 1e-8
_EXACT_MAX_TERMS = 500


def _exact_polynomial_value(a: Any, b: Any, z: float) -> float:
    """Evaluate a terminating pFq in exact rational arithmetic.

    Floats are exact rationals, so the only rounding is the final
    conversion, which gives ``+-inf`` past the double range. Returns NaN
    when an input is not finite, a denominator
    parameter hits a pole, or the series has more than
    ``_EXACT_MAX_TERMS`` terms (the rational arithmetic is superlinear in
    the term count: 0.1 s at 500 terms, 4.6 s at 2000).
    """
    if not (np.isfinite(z) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    n = int(min(-ai for ai in a if ai <= 0 and float(ai).is_integer()))
    if n > _EXACT_MAX_TERMS:
        return float("nan")
    zf = Fraction(float(z))
    term = Fraction(1)
    total = Fraction(1)
    for k in range(1, n + 1):
        den = Fraction(k)
        for ai in a:
            term *= Fraction(float(ai)) + (k - 1)
        for bi in b:
            den *= Fraction(float(bi)) + (k - 1)
        if den == 0:
            return float("nan")
        term = term * zf / den
        total += term
    try:
        return float(total)
    except OverflowError:
        return math.inf if total > 0 else -math.inf


def _flag_cancellation(
    result: NDArray[np.floating],
    max_term: NDArray[np.floating],
    a: Any,
    b: Any,
    z: NDArray[np.floating],
    terminates: bool,
) -> NDArray[np.floating]:
    """Blank out sums whose rounding error swamps the value.

    A terminating series with no surviving digits is re-evaluated in exact
    arithmetic instead, so an exact zero (``1 - z`` at ``z = 1``) stays 0
    while a non-terminating sum that merely rounds to 0.0 still becomes NaN.

    Summing terms as large as ``max_term`` leaves an absolute error of about
    ``eps * max_term``. When that reaches ``abs(result)`` no significant
    digit survives, so the entry becomes NaN; when it exceeds
    ``_CANCELLATION_WARN`` of the result, the value is kept with a warning.
    Entries with a non-finite ``z`` or parameter are NaN without a warning:
    their NaN comes from the input, not from cancellation.
    """
    bad = ~np.isfinite(z)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        bad[:] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(bad, 0.0, _EPS * max_term / np.abs(result))
    result = np.where(bad, np.nan, result)
    lost = ~(loss < 1.0)
    if terminates and np.any(lost):
        result = result.copy()
        for j in np.flatnonzero(lost):
            result[j] = _exact_polynomial_value(a, b, z[j])
        recovered = lost & ~np.isnan(result)
        lost = lost & ~recovered
        loss = np.where(recovered, 0.0, loss)
    degraded = ~lost & (loss > _CANCELLATION_WARN)
    if np.any(lost):
        warnings.warn(
            "generalized_hypergeometric: catastrophic cancellation; the "
            "largest series term exceeds the sum by more than 1/eps, so no "
            "significant digits remain. Returning NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
        result = np.where(lost, np.nan, result)
    if np.any(degraded):
        warnings.warn(
            "generalized_hypergeometric: cancellation in the series sum "
            "has cost more than half of the available digits.",
            RuntimeWarning,
            stacklevel=3,
        )
    return result


def _negative_factors(x: float, n: int) -> int:
    """Number of negative factors in the Pochhammer symbol (x)_n."""
    return 0 if x >= 0 else min(n, math.ceil(-x))


def _pfq_limit_at_infinity(
    a: NDArray[np.floating], b: NDArray[np.floating], z: float
) -> float:
    """Limit of pFq(a; b; z) as z -> +inf or -inf (DLMF 16.11).

    The caller has already rejected the divergent non-terminating cases
    (p = q + 1 other than 2F1, and p > q + 1).
    """
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    p, q = len(a), len(b)
    degrees = [int(-ai) for ai in a if ai <= 0 and float(ai).is_integer()]
    if degrees:
        n = min(degrees)
        if any(-n < bj <= 0 and float(bj).is_integer() for bj in b):
            return np.nan
        if n == 0:
            return 1.0
        negatives = sum(_negative_factors(x, n) for x in (*a, *b))
        if z < 0:
            negatives += n
        return -np.inf if negatives % 2 else np.inf
    if not (np.all(a > 0) and np.all(b > 0)):
        return np.nan
    if (p, q) == (2, 1):
        return np.nan if z > 0 else 0.0
    if z > 0:
        return np.inf
    if p == q:
        return 0.0
    if p == q - 1 and a.sum() - b.sum() + 0.5 < 0:
        return 0.0
    return np.nan


def generalized_hypergeometric(
    a: ArrayLike,
    b: ArrayLike,
    z: ArrayLike,
    max_terms: int = 500,
    tol: float = 1e-15,
) -> NDArray[np.floating]:
    """
    Generalized hypergeometric function pFq(a; b; z).

    Computes the generalized hypergeometric function with p numerator
    and q denominator parameters.

    Parameters
    ----------
    a : array_like
        Numerator parameters (1D array of length p).
    b : array_like
        Denominator parameters (1D array of length q).
    z : array_like
        Argument of the function.
    max_terms : int, optional
        Maximum number of series terms. Default is 500.
    tol : float, optional
        Tolerance for series convergence. Default is 1e-15.

    Returns
    -------
    F : ndarray
        Values of pFq(a; b; z).

    Notes
    -----
    The series converges for::

        - p <= q: all z
        - p = q + 1: |z| < 1
        - p > q + 1: diverges except for polynomial cases

    Uses Numba JIT compilation for the general case (p > 2 or q > 1),
    providing 5-10x speedup over pure Python loops.

    Outside the convergence domain (p = q + 1 with ``|z| >= 1``, or
    p > q + 1 with ``z != 0``) a ``ValueError`` is raised unless the
    series terminates (some ``a[i]`` a non-positive integer). A
    ``RuntimeWarning`` is emitted when ``max_terms`` is exhausted
    before the tolerance holds.

    Convergence is judged against the largest term seen, not the partial
    sum. An alternating series (negative ``z``) sums to a value far below
    its terms, and the old test against the partial sum stopped on a term
    still enormous relative to the true value (``0F0(-40)`` returned
    -2.23 against ``exp(-40) = 4.2e-18``; ``3F3([1,2,3],[4,5,6],-100)``
    returned 3.2e12 against 0.1993). The rounding error of the sum is about
    ``eps`` times the largest term: when that reaches the magnitude of the
    sum (largest term more than ``1/eps`` times the sum) the result is NaN
    with a ``RuntimeWarning``, and a ``RuntimeWarning`` is emitted with the
    value kept once more than half the digits are lost. A terminating
    series (some ``a[i]`` a non-positive integer) that loses every digit is
    re-evaluated exactly in rational arithmetic instead (Laguerre
    ``L_80(20)`` gives -1986.4165), so the NaN applies only to
    non-terminating series and to terminating ones longer than 500 terms.
    A NaN ``z`` or a non-finite parameter gives NaN without a cancellation
    warning. ``0F0(z)`` is evaluated as ``exp(z)``.

    At ``z = +inf`` or ``z = -inf`` the limit is returned where one exists
    (DLMF 16.11). Parameters are real: they are cast to float64, so a
    complex parameter raises ``TypeError``. 1F1 (``p == q == 1``) is
    :func:`hyp1f1` at every ``z``, so its infinite-``z`` value is the true
    limit from DLMF 13.7.2 (for example ``-inf`` for 1F1(-0.5; 1; +inf)); a
    non-positive integer ``b`` follows scipy's ``inf``. For every other
    ``(p, q)``, with ``n`` the smallest ``abs(a[i])`` over the non-positive
    integer ``a[i]`` and ``c_n = prod (a_i)_n / (prod (b_j)_n * n!)`` the
    leading coefficient of the terminating polynomial:

    ======================================  ===================  ===========================================
    Case                                    ``z = +inf``         ``z = -inf``
    ======================================  ===================  ===========================================
    terminating                             ``sign(c_n) * inf``  ``sign(c_n) * (-1)**n * inf``
    terminating, ``n = 0``                  1.0                  1.0
    non-terminating, ``p == q``             ``+inf``             0.0
    non-terminating, ``p == q - 1``         ``+inf``             0.0 if ``sum(a) - sum(b) + 1/2 < 0``, else NaN (oscillates)
    non-terminating, ``p <= q - 2``         ``+inf``             NaN (oscillating growth)
    non-terminating, ``(p, q) == (2, 1)``   NaN (branch cut)     0.0
    non-terminating, other ``p >= q + 1``   ``ValueError``       ``ValueError``
    anything else                           NaN                  NaN
    ======================================  ===================  ===========================================

    The non-terminating rows other than the ``ValueError`` one require
    every ``a[i] > 0`` and every ``b[j] > 0``; a negative non-integer
    ``a[i]``, a non-positive ``b[j]``, or a terminating series with a pole
    (``(b_j)_n = 0``) is "anything else".

    Examples
    --------
    >>> round(float(generalized_hypergeometric([1], [2], 1)), 6)  # 1F1(1; 2; 1) = e - 1
    1.718282
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    p = len(a)
    q = len(b)

    # The series terminates (is a polynomial) when any numerator
    # parameter is a non-positive integer; only then does it converge
    # outside the |z| < 1 (p = q+1) / z = 0 (p > q+1) domains. 2F1 is
    # analytically continued by scipy instead.
    z_arr = np.atleast_1d(z)
    terminates = any(ai <= 0 and float(ai).is_integer() for ai in a)
    if not terminates and (p, q) != (2, 1):
        if p == q + 1 and np.any(np.abs(z_arr) >= 1.0):
            raise ValueError(
                "pFq with p = q + 1 diverges for |z| >= 1 (the series "
                "is not a polynomial); no value exists to return."
            )
        if p > q + 1 and np.any(z_arr != 0.0):
            raise ValueError(
                "pFq with p > q + 1 diverges for z != 0 (the series "
                "is not a polynomial); no value exists to return."
            )

    if p == 1 and q == 1:
        return hyp1f1(a[0], b[0], z)

    infinite = np.isinf(z)
    if np.any(infinite):
        out = np.empty(z.shape)
        if not np.all(infinite):
            out[...] = generalized_hypergeometric(
                a, b, np.where(infinite, 0.0, z), max_terms, tol
            )
        out[infinite] = [_pfq_limit_at_infinity(a, b, zi) for zi in z[infinite]]
        return out if out.ndim else out[()]

    # For special cases, use scipy's optimized functions
    if p == 0 and q == 0:
        return np.exp(z)
    if p == 0 and q == 1:
        return hyp0f1(b[0], z)
    elif p == 2 and q == 1:
        return hyp2f1(a[0], a[1], b[0], z)

    # General case: use Numba-optimized series summation
    result, max_term, converged = _hypergeometric_series(a, b, z_arr, max_terms, tol)
    if not converged:
        warnings.warn(
            f"generalized_hypergeometric did not reach tol={tol:g} "
            f"within max_terms={max_terms}; the returned partial sum "
            "may be inaccurate. Increase max_terms.",
            RuntimeWarning,
            stacklevel=2,
        )

    result = _flag_cancellation(result, max_term, a, b, z_arr, terminates)
    return result if result.size > 1 else result[0]


__all__ = [
    "hyp0f1",
    "hyp1f1",
    "hyp2f1",
    "hyperu",
    "hyp1f1_regularized",
    "pochhammer",
    "falling_factorial",
    "generalized_hypergeometric",
]
