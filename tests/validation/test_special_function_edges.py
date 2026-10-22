"""Reference tests for special-function edge cases, one class per family.

Every expected value comes from an oracle independent of the implementation
(mpmath at high precision), never from pytcl itself.
"""

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
    """log Q_m(a, b) by quadrature of the Marcum integrand (a > 0)."""
    with mpmath.workdps(40):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        scale = 1 / (b - a)

        def integrand(x):
            return (
                x**m
                / a ** (m - 1)
                * mpmath.exp(-(x**2 + a**2) / 2)
                * mpmath.besseli(m - 1, a * x)
            )

        pts = [b + scale * c for c in (0, 0.1, 0.3, 1, 3, 10, 30, 100)]
        return float(mpmath.log(mpmath.quad(integrand, pts + [mpmath.inf])))


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
        assert got == pytest.approx(want, rel=1e-13)

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
    ]

    @pytest.mark.parametrize("pfa,n,snr", CASE3)
    def test_case3_general_n_matches_series(self, pfa, n, snr):
        want = _nb_mixture_pd(snr, pfa, n, 3)
        got = float(swerling_detection_probability(snr, pfa, n, 3))
        assert got == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("pfa,n,snr", [(1e-6, 6, 0.1), (1e-6, 3, 0.5)])
    def test_case3_series_oracle_agrees_with_quadrature(self, pfa, n, snr):
        assert _nb_mixture_pd(snr, pfa, n, 3) == pytest.approx(
            _swerling3_quad_pd(snr, pfa, n), rel=1e-12
        )

    @pytest.mark.parametrize("case", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_zero_snr_gives_pfa_elementwise(self, case, n):
        pfa = 1e-6
        snr = np.array([0.0, 1.0, 0.0, 0.3])
        got = swerling_detection_probability(snr, pfa, n, case)
        assert got[0] == pytest.approx(pfa, rel=1e-12)
        assert got[2] == pytest.approx(pfa, rel=1e-12)
        assert got[1] == pytest.approx(_nb_mixture_pd(1.0, pfa, n, case), rel=1e-11)
        assert got[3] == pytest.approx(_nb_mixture_pd(0.3, pfa, n, case), rel=1e-11)
        scalar = swerling_detection_probability(0.0, pfa, n, case)
        assert float(scalar) == pytest.approx(pfa, rel=1e-12)

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
        assert got == pytest.approx(want, rel=1e-13)

    @pytest.mark.parametrize("a,b,m", [(1.0, 40.0, 1), (10.0, 100.0, 5)])
    def test_log_marcum_q_series_oracle_agrees_with_quadrature(self, a, b, m):
        assert _log_marcum_q_series(a, b, m) == pytest.approx(
            _log_marcum_q_quad(a, b, m), rel=1e-11
        )

    def test_log_marcum_q_array_mixes_both_regimes(self):
        a = np.array([[1.0, 2.0], [10.0, 0.0]])
        b = np.array([[40.0, 2.0], [100.0, 100.0]])
        got = log_marcum_q(a, b, m=5)
        assert got.shape == (2, 2)
        for idx in np.ndindex(a.shape):
            want = _log_marcum_q_series(a[idx], b[idx], 5)
            assert got[idx] == pytest.approx(want, rel=1e-13)
