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


def _log_oracle_params() -> list:
    """Oracle rows, with the one measured miss marked strict-xfail.

    At ``a = 10, m = 1e4, c = -10`` (outside the ``a*b > 2e4`` domain),
    ``log1p(-ncx2.cdf)`` is off by 1.7e-12 relative on ``log Q = -1.5e-46``.
    """
    params = []
    for row in _oracle_rows():
        c = row.b - math.sqrt(row.a**2 + 2.0 * row.m)
        if (row.a, row.m, round(c)) == (10.0, 10_000, -10):
            mark = pytest.mark.xfail(reason="ncx2.cdf: 1.7e-12 relative", strict=True)
            params.append(pytest.param(row, marks=mark))
        else:
            params.append(row)
    return params


def mpmath_exp_as_float(log_q: str) -> float:
    """exp of a decimal string at 40 digits; 0.0 below the double range."""
    with mpmath.workdps(40):
        return float(mpmath.exp(mpmath.mpf(log_q)))


# =============================================================================
# Tests for Marcum Q function
# =============================================================================


class TestMarcumQ:
    """Tests for generalized Marcum Q function."""

    def test_marcum_q_zero_zero(self):
        """Test Q_1(0, 0) = 1."""
        result = marcum_q(0, 0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_marcum_q_a_zero(self):
        """Test Q_m(0, b) uses incomplete gamma."""
        result = marcum_q(0, 2, m=1)
        # Q_1(0, b) = exp(-b^2/2)
        expected = np.exp(-2)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_marcum_q_b_zero(self):
        """Test Q_m(a, 0) = 1 for any a."""
        a_values = np.array([0, 1, 5, 10])
        for a in a_values:
            result = marcum_q(a, 0)
            assert result == pytest.approx(1.0, rel=1e-10)

    def test_marcum_q_bounds(self):
        """Test Q is bounded by 0 and 1."""
        a = np.linspace(0, 5, 20)
        b = np.linspace(0, 5, 20)
        for ai in a:
            for bi in b:
                result = marcum_q(ai, bi)
                assert 0.0 <= result <= 1.0

    def test_marcum_q_decreasing_in_b(self):
        """Test Q_m(a, b) is decreasing in b."""
        a = np.full(20, 2.0)
        b = np.linspace(0.1, 5, 20)
        result = marcum_q(a, b)
        assert np.all(np.diff(result) <= 0)

    def test_marcum_q_known_value(self):
        """Test Q_1(3, 4) has known value."""
        result = marcum_q(3, 4)
        # Value from scipy.stats.ncx2.sf
        assert 0.1 < result < 0.3

    def test_marcum_q_order_2(self):
        """Test higher order Marcum Q."""
        result = marcum_q(2, 3, m=2)
        assert 0 < result < 1

    def test_marcum_q_array(self):
        """Test Marcum Q with array inputs."""
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        result = marcum_q(a, b)
        assert result.shape == (3,)

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

    @pytest.mark.parametrize("row", _log_oracle_params(), ids=_row_id)
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

    def test_marcum_q1_equals_m1(self):
        """Test Q_1 equals marcum_q with m=1."""
        a, b = 2, 3
        result1 = marcum_q1(a, b)
        result2 = marcum_q(a, b, m=1)
        assert result1 == pytest.approx(result2, rel=1e-10)

    def test_marcum_q1_known_value(self):
        """Test Q_1(2, 2) has known value."""
        result = marcum_q1(2, 2)
        assert 0.6 < result < 0.8


# =============================================================================
# Tests for log Marcum Q
# =============================================================================


class TestLogMarcumQ:
    """Tests for log Marcum Q function."""

    def test_log_marcum_q_zero_zero(self):
        """Test log(Q_1(0, 0)) = log(1) = 0."""
        result = log_marcum_q(0, 0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_log_marcum_q_relation(self):
        """Test log_marcum_q equals log(marcum_q) for moderate values."""
        a, b = 2, 3
        result = log_marcum_q(a, b)
        expected = np.log(marcum_q(a, b))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_log_marcum_q_negative_for_small_q(self):
        """Test log is negative when Q < 1."""
        a, b = 1, 5
        result = log_marcum_q(a, b)
        assert result < 0

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

    @pytest.mark.parametrize("x", [3e4, 5.1e5, 1e7])
    def test_temme_matches_poisson_sums(self, x):
        """The smaller of ``Q(n, x)``, ``P(n, x)`` against summed pmf terms."""
        offsets = np.linspace(-40.0, 40.0, 161) * np.sqrt(x)
        n = np.unique(np.round(x + offsets)).astype(np.int64)
        want_q, want_p = _mq._log_poisson_tails(n, x)
        got_q, got_p = _mq._log_poisson_q_p_temme(n.astype(np.float64), x)
        for got, want in ((got_q, want_q), (got_p, want_p)):
            small = want <= -np.log(2.0)
            assert got[small] == pytest.approx(want[small], rel=1e-14, abs=0)

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

    def test_marcum_q_inv_roundtrip(self):
        """Test Q(a, Q_inv(a, q)) = q."""
        a, q = 2, 0.5
        b = marcum_q_inv(a, q)
        result = marcum_q(a, b)
        assert result == pytest.approx(q, rel=1e-6)

    def test_marcum_q_inv_q_half(self):
        """Test inverse for q = 0.5."""
        a = 3.0
        b = marcum_q_inv(a, 0.5)
        assert b > 0
        assert marcum_q(a, b) == pytest.approx(0.5, rel=1e-6)

    def test_marcum_q_inv_small_q(self):
        """Test inverse for small q."""
        a, q = 2, 0.1
        b = marcum_q_inv(a, q)
        assert b > 0
        assert marcum_q(a, b) == pytest.approx(q, rel=0.01)

    def test_marcum_q_inv_large_q(self):
        """Test inverse for large q."""
        a, q = 2, 0.9
        b = marcum_q_inv(a, q)
        assert b >= 0
        assert marcum_q(a, b) == pytest.approx(q, rel=0.01)

    def test_marcum_q_inv_invalid_q(self):
        """Test invalid q raises ValueError."""
        with pytest.raises(ValueError, match="q must be in"):
            marcum_q_inv(1, 0)
        with pytest.raises(ValueError, match="q must be in"):
            marcum_q_inv(1, 1)

    def test_marcum_q_inv_invalid_m(self):
        """Test invalid m raises ValueError."""
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            marcum_q_inv(1, 0.5, m=0)


# =============================================================================
# Tests for the Rician CDF (complementary Marcum Q)
# =============================================================================


class TestRicianCdf:
    """Tests for the Rician CDF (complementary Marcum Q)."""

    def test_rician_cdf_relation(self):
        """Test P(a, b) = 1 - Q_1(a, b)."""
        a, b = 2, 3
        result = rician_cdf(a, b)
        expected = 1 - marcum_q(a, b, m=1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_rician_cdf_b_zero(self):
        """Test P(a, 0) = 0."""
        result = rician_cdf(2, 0)
        assert result == pytest.approx(0.0, rel=1e-10)

    def test_rician_cdf_bounds(self):
        """Test the CDF is bounded by 0 and 1."""
        a = np.linspace(0, 5, 20)
        b = np.linspace(0, 5, 20)
        for ai in a:
            for bi in b:
                result = rician_cdf(ai, bi)
                assert 0.0 <= result <= 1.0

    def test_rician_cdf_increasing_in_b(self):
        """Test P(a, b) is increasing in b."""
        a = np.full(20, 2.0)
        b = np.linspace(0.1, 5, 20)
        result = rician_cdf(a, b)
        assert np.all(np.diff(result) >= 0)


# =============================================================================
# Tests for Swerling detection probability
# =============================================================================


class TestSwerlingDetection:
    """Tests for Swerling detection probability."""

    def test_swerling_case_0(self):
        """Test Swerling case 0 (non-fluctuating)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=0)
        assert 0 < result < 1

    def test_swerling_case_1(self):
        """Test Swerling case 1 (slow Rayleigh)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=1)
        assert 0 < result < 1

    def test_swerling_case_2(self):
        """Test Swerling case 2 (fast Rayleigh)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=2)
        assert 0 < result < 1

    def test_swerling_case_3(self):
        """Test Swerling case 3 (slow chi-squared)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=3)
        assert 0 < result < 1

    def test_swerling_case_4(self):
        """Test Swerling case 4 (fast chi-squared)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=4)
        assert 0 < result < 1

    def test_swerling_increasing_snr(self):
        """Test detection probability increases with SNR."""
        snr_values = np.array([1, 5, 10, 20])
        results = []
        for snr in snr_values:
            results.append(
                swerling_detection_probability(snr, 1e-6, n_pulses=10, swerling_case=0)
            )
        results = np.array(results)
        assert np.all(np.diff(results) > 0)

    def test_swerling_more_pulses_better(self):
        """Test more integrated pulses gives higher detection."""
        # With higher SNR to ensure detection probability is reasonable
        result_1 = swerling_detection_probability(20, 1e-6, n_pulses=1, swerling_case=0)
        result_10 = swerling_detection_probability(
            20, 1e-6, n_pulses=10, swerling_case=0
        )
        # Both should be high with SNR=20
        assert result_10 > 0.5
        assert result_1 > 0.5

    def test_swerling_invalid_case(self):
        """Test invalid Swerling case raises ValueError."""
        with pytest.raises(ValueError, match="swerling_case must be 0-4"):
            swerling_detection_probability(10, 1e-6, swerling_case=5)

    def test_swerling_high_snr(self):
        """Test high SNR gives high detection probability."""
        result = swerling_detection_probability(100, 1e-6, n_pulses=10, swerling_case=0)
        assert result > 0.99

    def test_swerling_array_snr(self):
        """Test with array SNR input."""
        snr = np.array([1, 5, 10])
        result = swerling_detection_probability(snr, 1e-6, n_pulses=5, swerling_case=1)
        assert result.shape == (3,)


# =============================================================================
# Integration tests
# =============================================================================


class TestMarcumQIntegration:
    """Integration tests for Marcum Q functions."""

    def test_marcum_rician_sum_one(self):
        """Test Q + P = 1."""
        a, b = 2, 3
        q = marcum_q(a, b)
        p = rician_cdf(a, b)
        assert q + p == pytest.approx(1.0, rel=1e-10)

    def test_marcum_q_inv_consistent(self):
        """Test marcum_q_inv is consistent across orders."""
        a, q = 2, 0.5
        for m in [1, 2, 3]:
            b = marcum_q_inv(a, q, m=m)
            result = marcum_q(a, b, m=m)
            assert result == pytest.approx(q, rel=0.01)

    def test_log_exp_roundtrip(self):
        """Test exp(log_marcum_q) = marcum_q for moderate values."""
        a, b = 2, 3
        log_q = log_marcum_q(a, b)
        result = np.exp(log_q)
        expected = marcum_q(a, b)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_swerling_case_0_vs_marcum(self):
        """Test Swerling 0 uses Marcum Q correctly."""
        snr = 10
        pfa = 1e-6
        n_pulses = 1
        pd = swerling_detection_probability(snr, pfa, n_pulses, swerling_case=0)
        # Direct computation
        threshold = -2 * n_pulses * np.log(pfa)
        a = np.sqrt(2 * n_pulses * snr)
        b = np.sqrt(threshold)
        expected = marcum_q(a, b, m=n_pulses)
        assert pd == pytest.approx(expected, rel=1e-6)
