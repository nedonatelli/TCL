"""Reference tests for special-function edge cases, one class per family.

Every expected value comes from an oracle independent of the implementation
(mpmath at high precision), never from pytcl itself.
"""

import math
import warnings

import mpmath
import numpy as np
import pytest
import scipy.special as sp

from pytcl.mathematical_functions.special_functions.marcum_q import (
    log_marcum_q,
    swerling_detection_probability,
)


def _poisson_pmf(k, lam):
    """Poisson pmf; ``Q(s+1, x) = Q(s, x) + pmf(s; x)`` steps the gamma tail."""
    if lam == 0:
        return mpmath.mpf(1) if k == 0 else mpmath.mpf(0)
    return mpmath.exp(k * mpmath.log(lam) - lam - mpmath.loggamma(k + 1))


def _nb_mixture_pd(snr, pfa, n, case):
    """Exact Pd as a Poisson mixture of gamma tails, at 40 digits.

    Conditional on total signal-to-noise ``s`` the square-law statistic is
    ``chi^2(2n, 2s)``, so ``Pd = sum_k P(K=k) Q(n+k, vt)`` with ``K`` Poisson
    in ``s``. Averaging over each Swerling fluctuation law makes ``K``
    Poisson (case 0) or negative binomial with ``r`` shape and ``theta`` mean
    per unit shape (cases 1-4).
    """
    with mpmath.workdps(40):
        vt = mpmath.mpf(float(sp.gammainccinv(n, pfa)))
        snr = mpmath.mpf(snr)
        if case == 0:
            lam = n * snr

            def weight(k):
                return mpmath.exp(-lam) * lam**k / mpmath.factorial(k)

        else:
            r, theta = {
                1: (1, n * snr),
                2: (n, snr),
                3: (2, n * snr / 2),
                4: (2 * n, snr / 2),
            }[case]
            p = 1 / (1 + theta)
            q = theta / (1 + theta)

            def weight(k):
                return mpmath.binomial(r + k - 1, k) * p**r * q**k

        total = mpmath.mpf(0)
        upper = mpmath.gammainc(n, vt, regularized=True)
        k = 0
        while True:
            term = weight(k) * upper
            total += term
            upper += _poisson_pmf(n + k, vt)
            k += 1
            if k > n + vt and term < total * mpmath.mpf(10) ** -25:
                return float(total)


def _swerling3_quad_pd(snr, pfa, n):
    """Pd for case 3 by quadrature over the 4-DOF (Gamma(2, 1/2)) RCS law.

    The density beyond x = 40 is below exp(-79) and is dropped.
    """
    with mpmath.workdps(30):
        vt = mpmath.mpf(float(sp.gammainccinv(n, pfa)))
        mean = n * mpmath.mpf(snr)

        def conditional(s):
            terms = []
            upper = mpmath.gammainc(n, vt, regularized=True)
            k = 0
            while True:
                t = _poisson_pmf(k, s) * upper
                terms.append(t)
                upper += _poisson_pmf(n + k, vt)
                k += 1
                if k > s + n + vt and t < mpmath.mpf(10) ** -25:
                    return mpmath.fsum(terms)

        def integrand(x):
            return 4 * x * mpmath.exp(-2 * x) * conditional(mean * x)

        return float(mpmath.quad(integrand, [0, 0.5, 1, 2, 5, 15, 40]))


def _log_marcum_q_series(a, b, m):
    """log Q_m(a, b) from the Poisson mixture ``sum_k Pois(k; a^2/2) Q(m+k, b^2/2)``."""
    with mpmath.workdps(40):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        lam = a**2 / 2
        x = b**2 / 2
        k_max = int(a * b / 2 + m + 60 * mpmath.sqrt(a * b / 2 + m + 1) + 100)
        terms = []
        upper = mpmath.gammainc(m, x, regularized=True)
        for k in range(k_max):
            terms.append(_poisson_pmf(k, lam) * upper)
            upper += _poisson_pmf(m + k, x)
        return float(mpmath.log(mpmath.fsum(terms)))


def _log_marcum_q_quad(a, b, m):
    """log Q_m(a, b) by mpmath quadrature of the Marcum integrand (a > 0).

    Integrates over ``t = x - a`` with the Bessel factor scaled by
    ``exp(-a x)``; working precision grows with ``log10(a)`` so ``a + t``
    keeps its digits. For ``b < a`` the mass below ``b`` is subtracted from 1.
    """
    dps = 40 + int(max(0.0, math.log10(a)))
    with mpmath.workdps(dps):
        a = mpmath.mpf(a)
        d = mpmath.mpf(b) - a
        nu = m - 1

        def integrand(t, shift):
            x = a + t
            return (
                x
                * (x / a) ** nu
                * mpmath.besseli(nu, a * x)
                * mpmath.exp(-a * x - t * t / 2 + shift)
            )

        if d >= 0:
            # exp(-d^2/2) is factored out so the integrand is O(1) at t = d.
            s = 1 / max(d, 1)
            pts = [d + s * c for c in (0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)]
            total = mpmath.quad(
                lambda t: integrand(t, d * d / 2), pts + [d + 60], maxdegree=10
            )
            return float(mpmath.log(total) - d * d / 2)
        lo = max(-a, mpmath.mpf(-60))
        pts = [lo] + [p for p in (-30, -20, -10, -5, -2) if lo < p < d] + [d]
        below = mpmath.quad(lambda t: integrand(t, 0), pts, maxdegree=10)
        return float(mpmath.log1p(-below))


class TestMarcumSwerling:
    """Swerling Pd and log Marcum Q against the exact Poisson-mixture series."""

    # Cases 0, 1, 2 and 4 were already exact; n <= 2 was exact for case 3.
    WORKING = [
        (case, pfa, n, snr)
        for case in (0, 1, 2, 4)
        for pfa, n, snr in [
            (1e-6, 1, 3.0),
            (1e-6, 4, 0.5),
            (1e-4, 10, 0.2),
            (1e-8, 6, 1.5),
        ]
    ] + [(3, 1e-6, 1, 3.0), (3, 1e-6, 2, 1.0), (3, 1e-4, 2, 0.2)]

    @pytest.mark.parametrize("case,pfa,n,snr", WORKING)
    def test_working_paths_match_series(self, case, pfa, n, snr):
        want = _nb_mixture_pd(snr, pfa, n, case)
        got = float(swerling_detection_probability(snr, pfa, n, case))
        assert got == pytest.approx(want, rel=1e-13, abs=0)

    # Measured before the fix: 8.23856e-6 against 9.04805e-6 (-8.9%) at
    # n=6; rel 1.98e-05 at n=3, 3.27e-04 at n=4. n*snr/2 = 1 is the switch
    # between the summed mixture and the closed form; n=200 overflowed.
    CASE3 = [
        (1e-6, 6, 0.1),
        (1e-6, 3, 0.5),
        (1e-6, 4, 0.3),
        (1e-4, 10, 2.0),
        (1e-8, 30, 0.05),
        (1e-2, 100, 0.01),
        (1e-6, 200, 1e-4),
        (1e-10, 1000, 2e-3),
        (1e-6, 1000, 0.1),
        (1e-6, 3, 0.6),
        (1e-6, 3, 0.7),
        # Review round 1: just above theta = 1 the closed form overflowed for
        # n > ~1025 (n=1100 gave -inf, n=2000 NaN) and cancelled to 2e-11 at
        # n=1000. The switch is now theta = sqrt(n); both sides are pinned.
        (1e-6, 1100, 2 * 1.000001 / 1100),
        (1e-6, 2000, 2 * 1.5 / 2000),
        (1e-6, 1000, 2 * 1.5 / 1000),
        (1e-6, 1000, 2 * 1.000001 / 1000),
        (1e-6, 1000, 2 * np.sqrt(1000) * (1 - 1e-6) / 1000),
        (1e-6, 1000, 2 * np.sqrt(1000) * (1 + 1e-6) / 1000),
        (1e-10, 2000, 2 * 2 * np.sqrt(2000) / 2000),
        (1e-2, 5000, 2 * 300 / 5000),
    ]

    @pytest.mark.parametrize("pfa,n,snr", CASE3)
    def test_case3_general_n_matches_series(self, pfa, n, snr):
        want = _nb_mixture_pd(snr, pfa, n, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = float(swerling_detection_probability(snr, pfa, n, 3))
        assert got == pytest.approx(want, rel=1e-12, abs=0)

    @pytest.mark.parametrize("pfa,n,snr", [(1e-6, 6, 0.1), (1e-6, 3, 0.5)])
    def test_case3_series_oracle_agrees_with_quadrature(self, pfa, n, snr):
        assert _nb_mixture_pd(snr, pfa, n, 3) == pytest.approx(
            _swerling3_quad_pd(snr, pfa, n), rel=1e-12, abs=0
        )

    @pytest.mark.parametrize("case", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_zero_snr_gives_pfa_elementwise(self, case, n):
        pfa = 1e-6
        snr = np.array([0.0, 1.0, 0.0, 0.3])
        got = swerling_detection_probability(snr, pfa, n, case)
        assert got[0] == pytest.approx(pfa, rel=1e-12, abs=0)
        assert got[2] == pytest.approx(pfa, rel=1e-12, abs=0)
        assert got[1] == pytest.approx(
            _nb_mixture_pd(1.0, pfa, n, case), rel=1e-11, abs=0
        )
        assert got[3] == pytest.approx(
            _nb_mixture_pd(0.3, pfa, n, case), rel=1e-11, abs=0
        )
        scalar = swerling_detection_probability(0.0, pfa, n, case)
        assert float(scalar) == pytest.approx(pfa, rel=1e-12, abs=0)

    # Measured before the fix: log_marcum_q(1, 40) = -inf (true -763.235);
    # ncx2.logsf underflows for (m=5,a=10,b=100), (1,1,40), (20,0,100).
    LOG_Q = [
        (1.0, 40.0, 1),
        (10.0, 100.0, 5),
        (0.0, 100.0, 20),
        (3.0, 30.0, 2),
        (1e-3, 50.0, 3),
        (30.0, 60.0, 1),
        (50.0, 80.0, 10),
        (100.0, 140.0, 2),
        (1.0, 5.0, 1),
    ]

    @pytest.mark.parametrize("a,b,m", LOG_Q)
    def test_log_marcum_q_matches_series(self, a, b, m):
        want = _log_marcum_q_series(a, b, m)
        got = float(log_marcum_q(a, b, m))
        assert got == pytest.approx(want, rel=1e-13, abs=0)

    @pytest.mark.parametrize("a,b,m", [(1.0, 40.0, 1), (10.0, 100.0, 5)])
    def test_log_marcum_q_series_oracle_agrees_with_quadrature(self, a, b, m):
        assert _log_marcum_q_series(a, b, m) == pytest.approx(
            _log_marcum_q_quad(a, b, m), rel=1e-11, abs=0
        )

    def test_case3_array_spanning_the_switch(self):
        n, pfa = 2000, 1e-6
        theta = np.array([0.01, 1.0, 44.0, 45.0, 200.0])
        got = swerling_detection_probability(2 * theta / n, pfa, n, 3)
        for g, th in zip(got, theta):
            assert g == pytest.approx(
                _nb_mixture_pd(2 * th / n, pfa, n, 3), rel=1e-12, abs=0
            )

    def test_log_marcum_q_infinite_threshold_is_minus_inf(self):
        # Review round 1: b = inf sent NaN terms into the window-growth loop,
        # which doubled until np.arange raised "Maximum allowed size exceeded".
        got = log_marcum_q(np.array([1.0, 0.0, 3.0]), np.array([np.inf, np.inf, 40.0]))
        assert got[0] == -np.inf
        assert got[1] == -np.inf
        assert got[2] == pytest.approx(
            _log_marcum_q_series(3.0, 40.0, 1), rel=1e-13, abs=0
        )

    # Review round 2: b^2/2 or a*b overflowing sent inf into the series
    # window, and np.arange raised "Maximum allowed size exceeded";
    # (0, 1e155) returned NaN. Q underflows to 0 there, so log Q is -inf.
    @pytest.mark.parametrize(
        "a,b,m",
        [
            (1.0, 1e160, 1),
            (1.0, 1e200, 2),
            (1e160, 1e161, 1),
            (0.0, 1e155, 1),
            (1e-308, 1.7e308, 1),
        ],
    )
    def test_log_marcum_q_overflowing_threshold_is_minus_inf(self, a, b, m):
        assert log_marcum_q(a, b, m) == -np.inf

    # For a >= 1e100 the Marcum integrand is the Gaussian density to within
    # O(m/a), so log Q = log Phi(a - b) to double precision.
    @pytest.mark.parametrize(
        "a,b,m",
        [
            (1e160, 1.0, 1),
            (1e160, 9e159, 1),
            (1e154, 1e-300, 3),
            (1e300, 1e300, 5),
            (1.7e308, 1.7e308, 1),
            (1e160, 1e160 + 1e150, 1),
            (1e100, 1e100 + 3e84, 1),
        ],
    )
    def test_log_marcum_q_huge_arguments_match_gaussian_limit(self, a, b, m):
        want = sp.log_ndtr(a - b)
        got = float(log_marcum_q(a, b, m))
        if want == 0.0:
            assert got == 0.0
        else:
            assert got == pytest.approx(want, rel=2e-13, abs=0)

    # Past a*b/2 = max(1e6, (m-1)^2) the series window is replaced by
    # quadrature of the exact integrand.
    @pytest.mark.parametrize(
        "a,b,m",
        [
            (1e5, 1e5 + 30.0, 5),
            (1e5, 1e5 + 8.0, 1),
            (3e3, 3e3 + 20.0, 200),
            (1e8, 1e8 + 1000.0, 1),
            (1e12, 1e12 + 12.0, 3),
        ],
    )
    def test_log_marcum_q_large_product_matches_quadrature(self, a, b, m):
        want = _log_marcum_q_quad(a, b, m)
        got = float(log_marcum_q(a, b, m))
        assert got == pytest.approx(want, rel=1e-13, abs=0)

    def test_log_marcum_q_nan_input_is_nan(self):
        got = log_marcum_q(np.array([np.nan, 1.0]), np.array([1.0, np.nan]))
        assert np.isnan(got).all()

    def test_log_marcum_q_array_mixes_both_regimes(self):
        a = np.array([[1.0, 2.0], [10.0, 0.0]])
        b = np.array([[40.0, 2.0], [100.0, 100.0]])
        got = log_marcum_q(a, b, m=5)
        assert got.shape == (2, 2)
        for idx in np.ndindex(a.shape):
            want = _log_marcum_q_series(a[idx], b[idx], 5)
            assert got[idx] == pytest.approx(want, rel=1e-13, abs=0)
