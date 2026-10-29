"""
Tests for Marcum Q function and related functions.

Tests cover:
- Generalized Marcum Q function
- Standard Marcum Q (Q_1)
- Log Marcum Q
- Inverse Marcum Q
- Nuttall Q function
- Swerling detection probability

``nuttall_q`` was renamed ``rician_cdf`` in v2.0.0 (the computation was
always correct; only the name was wrong) and the deprecated alias was
removed in v2.8.0.
"""

import functools
import importlib
import math
import tracemalloc
import warnings
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import mpmath
import numpy as np
import pytest
from scipy.stats import ncx2

from pytcl.core.exceptions import ConvergenceError
from pytcl.mathematical_functions.special_functions.marcum_q import (
    log_marcum_q,
    marcum_q,
    marcum_q1,
    marcum_q_inv,
    rician_cdf,
    swerling_detection_probability,
)

_mq = importlib.import_module("pytcl.mathematical_functions.special_functions.marcum_q")
_ORACLE = Path(__file__).resolve().parent.parent / "fixtures/marcum_q/oracle.csv"


class _OracleRow(NamedTuple):
    a: float
    b: float
    m: int
    log_q: str


def _is_series_corner(row: _OracleRow) -> bool:
    ab = row.a * row.b
    return ab > 2e4 and (row.m - 1.0) ** 2 > 4.0 * ab


def _oracle_rows(corner: bool = False) -> list[_OracleRow]:
    """Rows of ``scripts/marcum_q_reference.py``'s output, in or out of the corner."""
    lines = [ln for ln in _ORACLE.read_text().splitlines() if not ln.startswith("#")]
    rows = []
    for line in lines[1:]:
        a, b, m, log_q = line.split(",")
        row = _OracleRow(float(a), float(b), int(m), log_q)
        if _is_series_corner(row) == corner:
            rows.append(row)
    return rows


def _row_id(row: _OracleRow) -> str:
    c = row.b - math.sqrt(row.a**2 + 2.0 * row.m)
    return f"a={row.a:g}-m={row.m}-c={c:+.0f}"


def mpmath_exp_as_float(log_q: str) -> float:
    """exp of a decimal string at 40 digits; 0.0 below the double range."""
    with mpmath.workdps(40):
        return float(mpmath.exp(mpmath.mpf(log_q)))


# Moderate-argument oracles, independent of pytcl and scipy. All at 30 digits.
# Q_m(a, b) = 1 - int_0^b x (x/a)^(m-1) exp(-(x^2 + a^2)/2) I_{m-1}(a x) dx,
# the definition; for a = 0 the closed form exp(-y) sum_{k<m} y^k / k!, y = b^2/2.
# Swerling: Q(n + k, vt) is the regularised upper incomplete gamma function.


@functools.lru_cache(maxsize=None)
def _mp_rice_cdf_m(a: float, b: float, m: int) -> "mpmath.mpf":
    """P_m(a, b) = 1 - Q_m(a, b) by direct quadrature of the density."""
    with mpmath.workdps(30):
        aa, bb = mpmath.mpf(a), mpmath.mpf(b)
        pts = [bb * i / 8 for i in range(9)]
        return mpmath.quad(
            lambda x: (
                x
                * (x / aa) ** (m - 1)
                * mpmath.exp(-(x * x + aa * aa) / 2)
                * mpmath.besseli(m - 1, aa * x)
            ),
            pts,
            method="gauss-legendre",
        )


@functools.lru_cache(maxsize=None)
def _mp_marcum_q(a: float, b: float, m: int) -> "mpmath.mpf":
    with mpmath.workdps(30):
        if a == 0:
            y = mpmath.mpf(b) ** 2 / 2
            return mpmath.exp(-y) * sum(y**k / mpmath.factorial(k) for k in range(m))
        return 1 - _mp_rice_cdf_m(a, b, m)


def _mp_q_gamma(shape: int, x: "mpmath.mpf") -> "mpmath.mpf":
    return mpmath.gammainc(shape, x, mpmath.inf, regularized=True)


def _mp_vt(pfa: float, n: int) -> "mpmath.mpf":
    """Normalised threshold vt with Q(n, vt) = pfa (n pulses, square law)."""
    with mpmath.workdps(30):
        lo, hi = mpmath.mpf(0), mpmath.mpf(200)
        for _ in range(110):  # bisection: Q(n, v) - pfa is decreasing in v
            mid = (lo + hi) / 2
            if _mp_q_gamma(n, mid) > pfa:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


@functools.lru_cache(maxsize=None)
def _mp_swerling(snr: float, pfa: float, n: int, case: int) -> "mpmath.mpf":
    """Pd as a mixture of gamma tails over the noncentrality of the sum.

    Given integrated noncentrality lam, the n-pulse square-law statistic has
    Pd = sum_k Pois(k; lam) Q(n + k, vt). Averaging over the target model:
    case 0, lam = n snr fixed; otherwise lam ~ Gamma(r, theta) gives the
    negative binomial weight C(k + r - 1, k) (theta/(1+theta))^k (1+theta)^-r,
    with (r, theta) = (1, n snr) for Swerling 1 (one exponential power per
    scan), (n, snr) for 2 (n independent exponentials), (2, n snr / 2) for 3
    and (2n, snr / 2) for 4 (chi-squared with 4 DOF, per scan / per pulse).
    """
    with mpmath.workdps(30):
        vt = _mp_vt(pfa, n)
        snr_m = mpmath.mpf(snr)
        total = mpmath.mpf(0)
        if case == 0:
            lam = n * snr_m
            k = 0
            while True:
                term = mpmath.exp(-lam) * lam**k / mpmath.factorial(k)
                total += term * _mp_q_gamma(n + k, vt)
                if k > lam and term < mpmath.mpf(10) ** -34:
                    return total
                k += 1
        r, theta = {
            1: (1, n * snr_m),
            2: (n, snr_m),
            3: (2, n * snr_m / 2),
            4: (2 * n, snr_m / 2),
        }[case]
        ratio = theta / (1 + theta)
        k = 0
        while True:
            weight = mpmath.binomial(k + r - 1, k) * ratio**k / (1 + theta) ** r
            total += weight * _mp_q_gamma(n + k, vt)
            if weight < mpmath.mpf(10) ** -34 and k > r * ratio / (1 - ratio):
                return total
            k += 1


# =============================================================================
# Tests for Marcum Q function
# =============================================================================


class TestMarcumQ:
    """Tests for generalized Marcum Q function."""

    def test_marcum_q_zero_zero(self):
        """Q_1(0, 0) = 1."""
        assert marcum_q(0, 0) == pytest.approx(1.0, rel=1e-14, abs=0)

    @pytest.mark.parametrize("m", [1, 2, 3, 5])
    @pytest.mark.parametrize("b", [0.5, 2.0, 4.0])
    def test_marcum_q_a_zero_is_the_gamma_tail(self, m, b):
        """Q_m(0, b) = exp(-y) sum_{k<m} y^k / k!, y = b^2/2 (Q_1 = exp(-b^2/2))."""
        y = b * b / 2.0
        expected = math.exp(-y) * sum(y**k / math.factorial(k) for k in range(m))
        assert marcum_q(0, b, m=m) == pytest.approx(expected, rel=1e-12, abs=0)

    def test_marcum_q1_a_zero_is_exp(self):
        assert marcum_q(0, 2, m=1) == pytest.approx(math.exp(-2.0), rel=1e-13, abs=0)

    @pytest.mark.parametrize("m", [1, 2, 4])
    @pytest.mark.parametrize("a", [0, 1, 5, 10])
    def test_marcum_q_b_zero_is_one(self, a, m):
        """Q_m(a, 0) = 1 for every a and m."""
        assert marcum_q(a, 0, m=m) == pytest.approx(1.0, rel=1e-14, abs=0)

    @pytest.mark.parametrize(
        ("a", "b", "m"),
        [
            (3.0, 4.0, 1),
            (2.0, 2.0, 1),
            (2.0, 3.0, 2),
            (1.0, 5.0, 1),
            (5.0, 3.0, 3),
            (0.5, 0.2, 1),
            (4.0, 8.0, 4),
            (0.1, 1.5, 2),
        ],
    )
    def test_marcum_q_matches_quadrature_of_the_definition(self, a, b, m):
        assert marcum_q(a, b, m=m) == pytest.approx(
            float(_mp_marcum_q(a, b, m)), rel=1e-12, abs=0
        )

    def test_marcum_q_array_matches_quadrature_elementwise(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        got = marcum_q(a, b)
        assert got.shape == (3,)
        want = [float(_mp_marcum_q(ai, bi, 1)) for ai, bi in zip(a, b, strict=True)]
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)

    def test_marcum_q_invalid_m(self):
        """Test invalid m raises ValueError."""
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            marcum_q(1, 2, m=0)


class TestMarcumQLargeArguments:
    """``a*b > 2e4`` against the mpmath oracle in tests/fixtures/marcum_q.

    Before v2.12.0 these went through ``scipy.stats.ncx2.sf``: 4.7e-3
    against 0.5 at ``(1e7, sqrt(1e14 + 2), m=1)`` and 8.6e-179 against
    2.8e-89 at ``(1e8, sqrt(1e16 + 4) + 20, m=2)``.
    """

    @pytest.mark.parametrize("row", _oracle_rows(), ids=_row_id)
    def test_marcum_q_matches_the_oracle(self, row):
        got = marcum_q(row.a, row.b, row.m)
        want = mpmath_exp_as_float(row.log_q)
        assert got == pytest.approx(want, rel=1e-12, abs=0)

    @pytest.mark.parametrize("row", _oracle_rows(), ids=_row_id)
    def test_log_marcum_q_matches_the_oracle(self, row):
        got = log_marcum_q(row.a, row.b, row.m)
        assert got == pytest.approx(float(row.log_q), rel=1e-12, abs=0)

    def test_no_boost_warnings_at_large_arguments(self, recwarn):
        marcum_q(1e8, 1e8 + 20.0, 10_000_000)
        assert not recwarn

    def test_mixed_regime_array_matches_scalar_calls(self):
        a = np.array([0.0, 10.0, 1e3, 1e7, 1e8, 1e3])
        b = np.array([3.0, 0.0, 1010.0, 1e7 + 3.0, 1e8 + 20.0, 1009.9955134495867])
        m = 2
        got = marcum_q(a, b, m)
        want = [marcum_q(ai, bi, m) for ai, bi in zip(a, b, strict=True)]
        assert np.array_equal(got, want)

    def test_broadcast_scalar_matches_scalar_calls(self):
        a = np.array([[0.0, 5.0], [1e3, 1e8]])
        b = 1e3 + 3.0
        got = marcum_q(a, b, 3)
        assert got.shape == (2, 2)
        for idx in np.ndindex(a.shape):
            assert got[idx] == marcum_q(a[idx], b, 3)


class TestMarcumQ1:
    """Tests for standard Marcum Q function."""

    @pytest.mark.parametrize(("a", "b"), [(2.0, 2.0), (2.0, 3.0), (3.0, 4.0)])
    def test_marcum_q1_matches_quadrature(self, a, b):
        assert marcum_q1(a, b) == pytest.approx(
            float(_mp_marcum_q(a, b, 1)), rel=1e-12, abs=0
        )

    def test_marcum_q1_a_zero_is_exp(self):
        assert marcum_q1(0.0, 3.0) == pytest.approx(math.exp(-4.5), rel=1e-13, abs=0)


# =============================================================================
# Tests for log Marcum Q
# =============================================================================


class TestLogMarcumQ:
    """Tests for log Marcum Q function."""

    def test_log_marcum_q_zero_zero(self):
        """Test log(Q_1(0, 0)) = log(1) = 0."""
        result = log_marcum_q(0, 0)
        assert result == 0.0

    @pytest.mark.parametrize(
        ("a", "b", "m"),
        [(2.0, 3.0, 1), (1.0, 5.0, 1), (5.0, 3.0, 3), (4.0, 8.0, 4)],
    )
    def test_log_marcum_q_matches_quadrature(self, a, b, m):
        want = float(mpmath.log(_mp_marcum_q(a, b, m)))
        assert log_marcum_q(a, b, m=m) == pytest.approx(want, rel=1e-12, abs=0)

    def test_log_marcum_q_a_zero(self):
        """log Q_1(0, b) = -b^2 / 2."""
        assert log_marcum_q(0.0, 3.0) == pytest.approx(-4.5, rel=1e-13, abs=0)

    def test_log_marcum_q_invalid_m(self):
        """Test invalid m raises ValueError."""
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            log_marcum_q(1, 2, m=0)

    @pytest.mark.parametrize("row", _oracle_rows(corner=True), ids=_row_id)
    def test_series_corner_matches_the_oracle(self, row):
        """``a*b > 2e4``, ``(m-1)^2 > 4ab``: the Poisson-mixture series.

        Its incomplete gammas came from ``scipy.special.gammaincc`` before
        v2.12.0: -27.462231887380316 against -27.46223188673473 (2.35e-11)
        at ``(1e3, 1009.9955134495867, m=3000)``.
        """
        got = log_marcum_q(row.a, row.b, row.m)
        assert got == pytest.approx(float(row.log_q), rel=1e-13, abs=0)

    def test_series_corner_rows_are_present(self):
        corner = {(row.a, row.m) for row in _oracle_rows(corner=True)}
        assert {(1e3, 3000), (1e3, 10_000), (10.0, 10_000_000)} <= corner

    @staticmethod
    def _mpmath_log_q(a: float, b: float, m: int, dps: int = 30) -> float:
        """``log sum_k Pois(k; a^2/2) Q(m + k, b^2/2)`` with mpmath's gammainc."""
        with mpmath.workdps(dps):
            lam = mpmath.mpf(a) ** 2 / 2
            x = mpmath.mpf(b) ** 2 / 2
            total = mpmath.mpf(0)
            k = 0
            while True:
                w = mpmath.exp(-lam) * lam**k / mpmath.factorial(k)
                if w == 0 or (k > lam and w < mpmath.mpf(10) ** (5 - dps)):
                    break
                q = mpmath.gammainc(m + k, x, mpmath.inf, regularized=True)
                total += w * q
                k += 1
            return float(mpmath.log(total))

    @pytest.mark.parametrize("m", [1000, 3000, 9999])
    @pytest.mark.parametrize("a", [0.0, 1.0])
    def test_b_sweep_across_the_poisson_bulk_edge(self, a, m):
        """``b = sqrt(a^2 + 2m) + c``, ``c`` in [0, 20] step 0.25.

        The Poisson tail window once started at the bulk edge when the order
        read sat just inside it, dropping the mass beyond: up to 4.3e-2 at
        ``(0, 148.637..., 9999)``, where v2.11.1 was correct.
        """
        mu = math.sqrt(a * a + 2.0 * m)
        for c in np.linspace(0.0, 20.0, 81):
            b = float(mu + c)
            want = self._mpmath_log_q(a, b, m)
            assert log_marcum_q(a, b, m) == pytest.approx(want, rel=1e-13, abs=0)

    @pytest.mark.parametrize(
        ("b", "m", "want"),
        [
            (85.09406559801923, 3000, -59.855966560818529236),
            (148.6371420607918, 9999, -54.591974316565443494),
        ],
    )
    def test_bulk_edge_pins(self, b, m, want):
        """Orders read exactly at the bulk edge; values from mpmath."""
        assert log_marcum_q(0.0, b, m) == pytest.approx(want, rel=1e-13, abs=0)

    @pytest.mark.parametrize("z", [3, 5, 10, 14])
    @pytest.mark.parametrize("m", [100, 10_000, 1_000_000, 10_000_000])
    @pytest.mark.parametrize("a", [0.0, 1.0])
    def test_near_one_is_relatively_accurate(self, a, m, z):
        """``log Q`` near 0 at ``b = sqrt(a^2 + 2m) - z/sqrt(2)``.

        ``log1p(-P)`` took ``P`` from scipy: at ``a = 0`` its ``gammainc``
        was 3.1e-2 off at ``m = 1e7``, and ``ncx2.cdf`` at ``a = 1`` 6.7e-10.
        """
        b = math.sqrt(a * a + 2.0 * m) - z / math.sqrt(2.0)
        want = self._mpmath_log_q(a, b, m, dps=90)
        assert log_marcum_q(a, b, m) == pytest.approx(want, rel=1e-13, abs=0)

    @pytest.mark.parametrize(
        ("a", "m", "z"),
        [
            (0.0, 10_000_000, 5),
            (0.0, 10_000_000, 10),
            (1.0, 10_000_000, 10),
            (1.0, 10_000_000, 14),
            (10.0, 1_000_000, 10),
            (10.0, 1_000_000, 14),
        ],
    )
    def test_near_one_review_pins(self, a, m, z):
        """``b = sqrt(2 (m - z sqrt(m)))``: scipy's ``1 - Q`` was 3.1e-2 off
        (``a = 0``, ``m = 1e7``, ``z = 5``) and 1.2e-9 (``a = 1``, ``m = 1e7``).
        """
        b = math.sqrt(2.0 * (m - z * math.sqrt(m)))
        want = self._mpmath_log_q(a, b, m, dps=90)
        assert log_marcum_q(a, b, m) == pytest.approx(want, rel=1e-13, abs=0)

    @staticmethod
    def _mpmath_log_q_large_b(a: float, b: float, m: int) -> tuple[float, float]:
        """``log Q`` where ``b^2/2`` dwarfs every order ``m + k`` that counts.

        ``Q(n, x) = pmf(n-1; x) S`` with ``1 <= S <= 1/(1 - n/x)``, and
        ``sum_k Pois(k; a^2/2) pmf(m+k-1; b^2/2)`` is
        ``exp(-(a^2+b^2)/2) (b/a)^(m-1) I_{m-1}(a b)``. Returns that log
        (Debye expansion of the Bessel function, ``nu = m - 1 >= 1e5``)
        and the bracket width ``-log(1 - n_max/x)``.
        """
        with mpmath.workdps(50):
            A, B = mpmath.mpf(a), mpmath.mpf(b)
            nu = mpmath.mpf(m - 1)
            t = A * B / nu
            r = mpmath.sqrt(1 + t * t)
            p = 1 / r
            u1 = (3 * p - 5 * p**3) / 24
            u2 = (81 * p**2 - 462 * p**4 + 385 * p**6) / 1152
            u3 = (30375 * p**3 - 369603 * p**5 + 765765 * p**7 - 425425 * p**9) / 414720
            log_i = (
                nu * (r + mpmath.log(t / (1 + r)))
                - mpmath.log(2 * mpmath.pi * nu) / 2
                - mpmath.log(r) / 2
                + mpmath.log(1 + u1 / nu + u2 / nu**2 + u3 / nu**3)
            )
            log_q = (m - 1) * mpmath.log(B / A) - (A * A + B * B) / 2 + log_i
            lam, x = A * A / 2, B * B / 2
            k = (-m + mpmath.sqrt(m * m + 4 * lam * x)) / 2
            n_max = m + k + 40 * mpmath.sqrt(k + 1) + 100
            return float(log_q), float(-mpmath.log(1 - n_max / x))

    @pytest.mark.parametrize("m", [100_000, 10_000_000])
    @pytest.mark.parametrize("a", [1e-3, 0.5, 10.0])
    def test_huge_b_in_the_series_corner(self, a, m):
        """``b`` from 1e8 to 1e14: terms near ``-b^2/2`` once broke the
        adaptive window tests and reached the ``ConvergenceError`` backstop.
        """
        checked = 0
        for b in np.logspace(8.0, 14.0, 13):
            if (m - 1.0) ** 2 <= 4.0 * a * b:
                continue
            want, slack = self._mpmath_log_q_large_b(a, b, m)
            assert marcum_q(a, b, m) == 0.0
            if slack <= abs(want) * 1e-17:
                got = log_marcum_q(a, b, m)
                assert got == pytest.approx(want, rel=1e-15, abs=0)
                checked += 1
        assert checked >= 1

    @pytest.mark.parametrize(
        ("a", "b", "want"),
        [
            (0.001, 7.0353064238e10, -2.474776823837688771337e21),
            (10.0, 2.4999995e12, -3.124998749975124742553e24),
        ],
    )
    def test_huge_b_pins(self, a, b, want):
        """These raised ConvergenceError; v2.11.1 returned 0.0 for ``Q``.

        ``want`` is ``-(a^2 + b^2)/2 + (m-1) log(b/a) + log I_{m-1}(a b)``
        in mpmath (Debye expansion), which brackets ``log Q`` to 2e-11
        absolute here because ``m + k << b^2/2`` for every term that counts.
        """
        assert marcum_q(a, b, 10_000_000) == 0.0
        got = log_marcum_q(a, b, 10_000_000)
        assert got == pytest.approx(want, rel=1e-15, abs=0)


def _temme_coefficients(n_series: int = 40, n_terms: int = 4) -> list[list[Fraction]]:
    """Temme's ``c_k(eta)`` Taylor coefficients in exact rational arithmetic.

    ``mu = lambda - 1`` is reverted from ``eta^2/2 = mu - log(1 + mu)`` by
    Lagrange inversion; ``c_0 = 1/mu - 1/eta`` and ``c_k = c_{k-1}'/eta +
    (-1)^k g_k / mu``, with ``g_k`` fixed by cancelling the pole at
    ``eta = 0`` (it comes out as the Stirling coefficients 1/12, 1/288, ...).
    """
    n = n_series

    def mul(p, q):
        r = [Fraction(0)] * n
        for i, pi in enumerate(p):
            if pi:
                for j in range(n - i):
                    r[i + j] += pi * q[j]
        return r

    def inv(p):
        r = [Fraction(0)] * n
        r[0] = 1 / p[0]
        for k in range(1, n):
            r[k] = -sum(p[i] * r[k - i] for i in range(1, k + 1)) / p[0]
        return r

    def sqrt1(p):
        r = [Fraction(1)] + [Fraction(0)] * (n - 1)
        for k in range(1, n):
            r[k] = (p[k] - sum(r[i] * r[k - i] for i in range(1, k))) / 2
        return r

    g2 = [Fraction(2 * (-1) ** i, i + 2) for i in range(n)]
    mu_over_eta = inv(sqrt1(g2))
    mu = [Fraction(0)] * n
    power = [Fraction(1)] + [Fraction(0)] * (n - 1)
    for k in range(1, n):
        power = mul(power, mu_over_eta)
        mu[k] = power[k - 1] / k
    eta_over_mu = inv(mu[1:] + [Fraction(0)])
    coeffs = [[eta_over_mu[j + 1] for j in range(n - 1)]]
    for _ in range(1, n_terms):
        prev = coeffs[-1]
        coeffs.append(
            [
                (j + 2) * prev[j + 2] - prev[1] * eta_over_mu[j + 1]
                for j in range(len(prev) - 2)
            ]
        )
    return coeffs


class TestMarcumQSeriesBounded:
    """The Poisson-mixture series runs in bounded memory for any input.

    Before this fix, ``(a=1e5, b=1, m=3000)`` (``b << a``, so the terms of
    ``Q`` peak near ``k = a^2/2 = 5e9``, far from the seed) doubled its
    window without bound and was killed at 49 GB; v2.11.1 returned 1.0.
    """

    _GRID = np.logspace(-3, 8, 12)
    _ORDERS = (1, 2, 30, 3000, 10_000_000)

    def test_reproducer_is_bounded_and_matches_v2_11_1(self):
        tracemalloc.start()
        try:
            q = marcum_q(1e5, 1.0, 3000)
            log_q = log_marcum_q(1e5, 1.0, 3000)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert q == 1.0
        assert log_q == 0.0
        assert peak < 16 * 2**20

    def test_grid_completes_in_bounded_memory(self):
        """A memory bound and a smoke check; it carries no accuracy weight."""
        tracemalloc.start()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for m in self._ORDERS:
                    for a in self._GRID:
                        for b in self._GRID:
                            q = marcum_q(a, b, m)
                            log_q = log_marcum_q(a, b, m)
                            assert 0.0 <= q <= 1.0
                            assert log_q <= 0.0
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 16 * 2**20

    def test_b_much_less_than_a_matches_v2_11_1(self):
        """On ``b <= a/10`` v2.11.1 was ``ncx2.sf`` and ``log`` of it."""
        checked = 0
        for m in self._ORDERS:
            for a in self._GRID:
                for b in self._GRID[self._GRID <= a / 10]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        old = ncx2.sf(b * b, 2 * m, a * a)
                    assert marcum_q(a, b, m) == old
                    if old >= 1e-10:
                        # v2.11.1's log Q carried Q's absolute error only.
                        assert abs(log_marcum_q(a, b, m) - np.log(old)) <= 2.3e-16
                    checked += 1
        assert checked > 300

    @pytest.mark.parametrize(
        "row", _oracle_rows() + _oracle_rows(corner=True), ids=_row_id
    )
    def test_series_alone_matches_every_oracle_row(self, row):
        """Including ``a = 1e8``, where the peak is at ``k = 5e15``."""
        got = _mq._log_marcum_q_series(row.a, row.b, row.m)
        assert got == pytest.approx(float(row.log_q), rel=1e-14, abs=0)

    def test_window_backstop_raises(self, monkeypatch):
        monkeypatch.setattr(_mq, "_SERIES_MAX_POINTS", 50)
        with pytest.raises(ConvergenceError, match="more than 50 terms"):
            _mq._log_marcum_q_series(1e3, 1009.9955134495867, 3000)

    def test_temme_table_is_the_exact_series(self):
        exact = _temme_coefficients()
        for table, series in zip(_mq._TEMME_C, exact, strict=True):
            assert table == tuple(float(c) for c in series[: len(table)])

    @pytest.mark.parametrize("z", [-10, -3, 0, 3, 10])
    @pytest.mark.parametrize("x", [3e4, 1e7])
    def test_temme_matches_mpmath(self, x, z):
        """The smaller of ``Q(n, x)``, ``P(n, x)`` at ``n = x + z sqrt(x)``."""
        n = float(round(x + z * math.sqrt(x)))
        got_q, got_p = _mq._log_poisson_q_p_temme(np.array([n]), x)
        with mpmath.workdps(80):
            q = mpmath.gammainc(n, x, mpmath.inf, regularized=True)
            want_q, want_p = float(mpmath.log(q)), float(mpmath.log1p(-q))
        got, want = (got_q[0], want_q) if want_q < want_p else (got_p[0], want_p)
        assert got == pytest.approx(want, rel=1e-15, abs=0)

    def test_ternary_search_terminates_past_2_53(self):
        """``a^2/2 >= 2^53``: the float bracket of 2.12.0-dev never shrank.

        No accuracy is claimed here (no oracle reaches ``m = 3e8``); the
        assertion is that both calls return.
        """
        a, m = 1.4e8, 300_000_000
        b = math.sqrt(a * a + 2.0 * m) + 5.0
        assert 0.0 < marcum_q(a, b, m) < 1.0
        assert log_marcum_q(a, b, m) < 0.0

    @pytest.mark.parametrize("m", [1, 2, 30, 3000, 10_000_000])
    @pytest.mark.parametrize("func", [marcum_q, log_marcum_q])
    def test_arrays_with_special_values_match_scalar_calls(self, func, m):
        v = np.array([0.0, 1e-3, 1.0, 30.0, 1e3, 1e5, 1e8, 1e200, np.inf, np.nan])
        a, b = np.meshgrid(v, v, indexing="ij")
        with warnings.catch_warnings():
            # b**2 overflows at 1e200 inside scipy's ncx2 path, as in v2.11.1.
            warnings.simplefilter("ignore", RuntimeWarning)
            got = func(a, b, m)
            want = np.array([[func(ai, bi, m) for bi in v] for ai in v])
        assert np.array_equal(got, want, equal_nan=True)


# =============================================================================
# Tests for inverse Marcum Q
# =============================================================================


class TestMarcumQInv:
    """Tests for inverse Marcum Q function."""

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    def test_a_zero_m1_closed_form(self, q):
        """Q_1(0, b) = exp(-b^2/2)  =>  b = sqrt(-2 ln q)."""
        assert marcum_q_inv(0.0, q) == pytest.approx(
            math.sqrt(-2.0 * math.log(q)), rel=1e-12, abs=0
        )

    @pytest.mark.parametrize(
        ("a", "q", "m"),
        [
            (2.0, 0.5, 1),
            (3.0, 0.5, 1),
            (2.0, 0.1, 1),
            (2.0, 0.9, 1),
            (2.0, 0.5, 2),
            (2.0, 0.5, 3),
        ],
    )
    def test_returned_b_has_oracle_q_equal_to_q(self, a, q, m):
        """The oracle's Q_m(a, b) at the returned b equals the requested q."""
        b = float(marcum_q_inv(a, q, m=m))
        assert float(_mp_marcum_q(a, b, m)) == pytest.approx(q, rel=1e-11, abs=0)

    def test_invalid_q(self):
        with pytest.raises(ValueError, match="q must be in"):
            marcum_q_inv(1, 0)
        with pytest.raises(ValueError, match="q must be in"):
            marcum_q_inv(1, 1)

    def test_invalid_m(self):
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            marcum_q_inv(1, 0.5, m=0)


# =============================================================================
# Tests for the Rician CDF (complementary Marcum Q)
# =============================================================================


class TestRicianCdf:
    """Tests for the Rician CDF P(a, b) = 1 - Q_1(a, b)."""

    @pytest.mark.parametrize(("a", "b"), [(2.0, 3.0), (1.0, 5.0), (0.5, 0.2)])
    def test_matches_quadrature_of_the_rice_density(self, a, b):
        assert rician_cdf(a, b) == pytest.approx(
            float(_mp_rice_cdf_m(a, b, 1)), rel=1e-12, abs=0
        )

    def test_b_zero(self):
        assert rician_cdf(2, 0) == 0.0

    def test_a_zero_is_the_rayleigh_cdf(self):
        """P(0, b) = 1 - exp(-b^2/2)."""
        assert rician_cdf(0.0, 2.0) == pytest.approx(
            -math.expm1(-2.0), rel=1e-13, abs=0
        )


# =============================================================================
# Tests for Swerling detection probability
# =============================================================================


class TestSwerlingDetection:
    """Swerling Pd against gamma-tail mixtures summed in mpmath (see _mp_swerling)."""

    @pytest.mark.parametrize("case", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n", [1, 4])
    @pytest.mark.parametrize("snr", [0.5, 3.0, 8.0])
    def test_matches_mixture_oracle(self, case, n, snr):
        got = swerling_detection_probability(snr, 1e-6, n_pulses=n, swerling_case=case)
        assert float(got) == pytest.approx(
            float(_mp_swerling(snr, 1e-6, n, case)), rel=1e-11, abs=0
        )

    @pytest.mark.parametrize("snr", [0.5, 3.0, 8.0])
    def test_single_pulse_rayleigh_closed_form(self, snr):
        """Swerling 1 and 2 with one pulse: Pd = pfa^(1 / (1 + snr))."""
        pfa = 1e-6
        want = pfa ** (1.0 / (1.0 + snr))
        for case in (1, 2):
            got = swerling_detection_probability(snr, pfa, 1, case)
            assert float(got) == pytest.approx(want, rel=1e-13, abs=0)

    @pytest.mark.parametrize("case", [0, 1, 2, 3, 4])
    def test_zero_snr_returns_pfa(self, case):
        got = swerling_detection_probability(0.0, 1e-6, n_pulses=4, swerling_case=case)
        assert float(got) == pytest.approx(1e-6, rel=1e-11, abs=0)

    def test_array_snr_matches_oracle_elementwise(self):
        snr = np.array([0.5, 3.0, 8.0])
        got = swerling_detection_probability(snr, 1e-6, n_pulses=4, swerling_case=1)
        assert got.shape == (3,)
        want = [float(_mp_swerling(float(s), 1e-6, 4, 1)) for s in snr]
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=0)

    def test_invalid_case(self):
        with pytest.raises(ValueError, match="swerling_case must be 0-4"):
            swerling_detection_probability(10, 1e-6, swerling_case=5)


class TestSeriesCorners:
    def test_bracket_overflow_raises(self):
        # lam (m + x) overflows the quadratic for the upper bracket.
        with pytest.raises(ConvergenceError, match="bracket overflows"):
            log_marcum_q(1e77, 1e78, 10**78)

    def test_fixed_grid_with_the_peak_near_zero(self):
        # noise above 1 nat (b^2/2 = 5e15) with k* = 50 inside 12 sigma + 20:
        # the fixed grid runs at unit stride from k = 0. Oracle: with
        # Q(n, x) = pmf(n-1; x) S, 1 <= S <= 1/(1 - n/x), the sum over k of
        # Pois(k; a^2/2) pmf(k; b^2/2) is exp(-(a^2+b^2)/2) I_0(a b), and
        # S - 1 is below 1e-13 for every order that counts.
        a, b = 1e-6, 1e8
        with mpmath.workdps(50):
            A, B = mpmath.mpf(a), mpmath.mpf(b)
            want = float(-(A * A + B * B) / 2 + mpmath.log(mpmath.besseli(0, A * B)))
        assert marcum_q(a, b, 1) == 0.0
        assert log_marcum_q(a, b, 1) == pytest.approx(want, rel=1e-15, abs=0)
