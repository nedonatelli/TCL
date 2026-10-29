"""Reference coverage of the CFAR threshold and detection-probability branches.

Oracles are independent of ``detection.py``: the CA closed form, Monte-Carlo
simulation of the detectors on exponential (square-law) noise, and numerical
integration over the reference-cell gamma density with ``scipy.integrate.quad``.
Monte-Carlo tolerances are the binomial 5-sigma bound, not an empirical fudge.
"""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.optimize import brentq

from pytcl.mathematical_functions.signal_processing import detection as det

N_TRIALS = 2_000_000
CHUNK = 200_000


def _binomial_5sigma(p: float, n: int = N_TRIALS) -> float:
    return 5.0 * float(np.sqrt(p * (1.0 - p) / n))


def _mc_exceedance(stat_fn, alpha: float, mean_cut: float = 1.0, seed: int = 20261028):
    """Fraction of trials with cut > alpha * stat, cut ~ Exp(mean_cut)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_TRIALS // CHUNK):
        cut = rng.exponential(mean_cut, CHUNK)
        hits += int(np.count_nonzero(cut > alpha * stat_fn(rng)))
    return hits / N_TRIALS


class TestThresholdFactorReference:
    @pytest.mark.parametrize("n_ref", [1, 4, 16, 64])
    @pytest.mark.parametrize("pfa", [1e-2, 1e-4, 1e-6])
    def test_ca_closed_form(self, n_ref, pfa):
        expected = n_ref * (pfa ** (-1.0 / n_ref) - 1.0)
        assert det.threshold_factor(pfa, n_ref, "ca") == pytest.approx(
            expected, rel=1e-13, abs=0
        )

    def test_ca_monte_carlo_pfa(self):
        n = 16
        pfa = 0.05
        alpha = det.threshold_factor(pfa, n, "ca")
        got = _mc_exceedance(
            lambda r: r.exponential(1.0, (CHUNK, n)).mean(axis=1), alpha
        )
        assert abs(got - pfa) < _binomial_5sigma(pfa)

    @pytest.mark.parametrize("method", ["go", "so"])
    def test_go_so_monte_carlo_pfa(self, method):
        """GO/SO Pfa by simulation. Range: n_ref=16 (8 per half), pfa=0.05 only.

        Monte-Carlo cannot resolve small Pfa at 2e6 trials, so the deep-Pfa
        tail of the Gandhi-Kassam expressions is not covered by this oracle.
        """
        n_ref = 16
        pfa = 0.05
        alpha = det.threshold_factor(pfa, n_ref, method)
        pick = np.maximum if method == "go" else np.minimum

        def stat(r):
            x = r.exponential(1.0, (CHUNK, n_ref))
            return pick(x[:, :8].mean(axis=1), x[:, 8:].mean(axis=1))

        got = _mc_exceedance(stat, alpha)
        assert abs(got - pfa) < _binomial_5sigma(pfa)

    def test_odd_n_ref_uses_floor_half_window(self):
        assert det.threshold_factor(1e-3, 17, "so") == pytest.approx(
            det.threshold_factor(1e-3, 16, "so"), rel=1e-14, abs=0
        )

    def test_os_monte_carlo_pfa_explicit_k(self):
        """OS Pfa by simulation. Range: n_ref=16, k=12, pfa=0.05 only (see GO/SO)."""
        n_ref, k, pfa = 16, 12, 0.05
        alpha = det.threshold_factor(pfa, n_ref, "os", k=k)
        got = _mc_exceedance(
            lambda r: np.partition(r.exponential(1.0, (CHUNK, n_ref)), k - 1, axis=1)[
                :, k - 1
            ],
            alpha,
        )
        assert abs(got - pfa) < _binomial_5sigma(pfa)

    def test_os_default_k_is_three_quarters(self):
        assert det.threshold_factor(1e-4, 32, "os") == pytest.approx(
            det.threshold_factor(1e-4, 32, "os", k=24), rel=1e-14, abs=0
        )

    @pytest.mark.parametrize("k_given, k_used", [(0, 1), (-5, 1), (99, 8)])
    def test_os_k_is_clamped_to_one_through_n(self, k_given, k_used):
        assert det.threshold_factor(1e-3, 8, "os", k=k_given) == pytest.approx(
            det.threshold_factor(1e-3, 8, "os", k=k_used), rel=1e-14, abs=0
        )

    @pytest.mark.parametrize("pfa", [0.0, 1.0, -0.1, 1.5])
    def test_pfa_out_of_range(self, pfa):
        with pytest.raises(ValueError, match="pfa must be between 0 and 1"):
            det.threshold_factor(pfa, 8)

    def test_n_ref_below_one(self):
        with pytest.raises(ValueError, match="n_ref must be at least 1"):
            det.threshold_factor(1e-3, 0)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method: median"):
            det.threshold_factor(1e-3, 8, "median")

    def test_unreachable_pfa_exhausts_bracket(self):
        # n_half = 1 gives Pfa(alpha) ~ 2/alpha, so Pfa = 1e-15 would need
        # alpha ~ 2e15, beyond the 1e12 search bracket.
        with pytest.raises(
            ValueError,
            match=r"pfa=1e-15 is below what the threshold search reaches.*"
            r"method='so', n_ref=2",
        ):
            det.threshold_factor(1e-15, 2, "so")

    def test_unreachable_pfa_names_os_order_statistic(self):
        # OS with k=1 of 2 cells: Pfa(alpha) ~ 2/alpha as well.
        with pytest.raises(ValueError, match=r"method='os', n_ref=2, k=1"):
            det.threshold_factor(1e-15, 2, "os", k=1)


def _pd_by_quadrature(snr: float, alpha: float, n: int) -> float:
    """P(cut > alpha * mean of n noise cells), cut ~ Exp(1+snr), by quadrature."""

    def integrand(g):
        return stats.gamma.pdf(g, a=n, scale=1.0 / n) * np.exp(-alpha * g / (1 + snr))

    return float(integrate.quad(integrand, 0.0, np.inf, epsabs=0, epsrel=1e-12)[0])


class TestDetectionProbabilityReference:
    @pytest.mark.parametrize("n_ref", [4, 16, 32])
    @pytest.mark.parametrize("snr", [1.0, 10.0, 100.0])
    def test_matches_quadrature_over_reference_cells(self, n_ref, snr):
        pfa = 1e-4
        alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1.0)
        assert det.detection_probability(snr, pfa, n_ref) == pytest.approx(
            _pd_by_quadrature(snr, alpha, n_ref), rel=1e-8, abs=0
        )

    def test_monte_carlo_pd(self):
        n, snr, pfa = 16, 10.0, 1e-2
        alpha = det.threshold_factor(pfa, n)
        got = _mc_exceedance(
            lambda r: r.exponential(1.0, (CHUNK, n)).mean(axis=1),
            alpha,
            mean_cut=1.0 + snr,
            seed=7,
        )
        pd = det.detection_probability(snr, pfa, n)
        assert abs(got - pd) < _binomial_5sigma(pd)

    def test_zero_snr_reduces_to_pfa(self):
        assert det.detection_probability(0.0, 1e-5, 24) == pytest.approx(
            1e-5, rel=1e-12, abs=0
        )

    def test_unknown_method_propagates(self):
        with pytest.raises(ValueError, match="Unknown method: bogus"):
            det.detection_probability(10.0, 1e-3, 8, method="bogus")


class TestSnrLossReference:
    @staticmethod
    def _required_snr_ca(pfa, pd, n):
        alpha = n * (pfa ** (-1.0 / n) - 1.0)
        return brentq(
            lambda s: _pd_by_quadrature(s, alpha, n) - pd, 1e-6, 1e9, xtol=1e-12
        )

    @pytest.mark.parametrize("n_ref", [8, 16, 64])
    def test_matches_quadrature_inversion(self, n_ref):
        pfa, pd = 1e-6, 0.5
        s_ideal = float(np.log(pfa) / np.log(pd) - 1.0)
        want = 10 * np.log10(
            (1 + self._required_snr_ca(pfa, pd, n_ref)) / (1 + s_ideal)
        )
        assert det.snr_loss(n_ref, pfa, pd) == pytest.approx(want, rel=1e-7, abs=0)

    @pytest.mark.parametrize(
        "kwargs, text",
        [
            ({"pfa": 0.0}, "pfa must be in"),
            ({"pfa": 1.0}, "pfa must be in"),
            ({"pd": 0.0}, "pd must be in"),
            ({"pd": 1.0}, "pd must be in"),
            ({"n_ref": 0}, "n_ref must be at least 1"),
        ],
    )
    def test_validation(self, kwargs, text):
        args = {"n_ref": 16, "pfa": 1e-6, "pd": 0.5} | kwargs
        with pytest.raises(ValueError, match=text):
            det.snr_loss(**args)

    @pytest.mark.parametrize("method", ["go", "so", "os"])
    def test_non_ca_not_implemented(self, method):
        with pytest.raises(NotImplementedError, match="only implemented for"):
            det.snr_loss(16, 1e-6, method=method)
