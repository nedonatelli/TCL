"""Gaussian mixture reduction vs MATLAB ``RunnalsGaussMixRed`` / ``WestGaussReduction``.

Fixtures from ``scripts/matlab_capture/capture_mixture_reduction.m`` (TCL
commit a9acd8f): the two functions' own docstring examples plus
independent 2-D, 3-D and 4-D mixtures (the 3-D and 4-D cases exercise
MATLAB's explicit-determinant and general branches). Fixture tests skip
when a CSV is absent unless ``PYTCL_REQUIRE_MATLAB_FIXTURES=1``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pytcl.clustering.gaussian_mixture import (
    GaussianComponent,
    reduce_mixture_runnalls,
    reduce_mixture_west,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "matlab"
REQUIRE = os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1"


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        if REQUIRE:
            pytest.fail(f"required MATLAB fixture missing: {name}")
        pytest.skip(f"MATLAB fixture not captured: {name}")
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _mixture(stem):
    w = _load(f"mixred_{stem}_w.csv").ravel()
    mu = _load(f"mixred_{stem}_mu.csv")
    d = mu.shape[1]
    P = _load(f"mixred_{stem}_P.csv").reshape(-1, d, d)
    return w, mu, P


def _components(stem):
    w, mu, P = _mixture(f"{stem}_in")
    return [GaussianComponent(float(w[i]), mu[i], P[i]) for i in range(len(w))]


def _arrays(result):
    w = np.array([c.weight for c in result.components])
    mu = np.array([c.mean for c in result.components])
    P = np.array([c.covariance for c in result.components])
    return w, mu, P


def _assert_same_mixture(result, stem, ordered):
    w_ref, mu_ref, P_ref = _mixture(stem)
    w, mu, P = _arrays(result)
    assert len(w) == len(w_ref)
    if not ordered:
        order = np.lexsort(mu.T[::-1])
        order_ref = np.lexsort(mu_ref.T[::-1])
        w, mu, P = w[order], mu[order], P[order]
        w_ref, mu_ref, P_ref = w_ref[order_ref], mu_ref[order_ref], P_ref[order_ref]
    np.testing.assert_allclose(w, w_ref, rtol=0, atol=1e-14)
    np.testing.assert_allclose(mu, mu_ref, rtol=0, atol=1e-13)
    np.testing.assert_allclose(P, P_ref, rtol=0, atol=1e-13)


RUNNALLS_CASES = [("r1", 6), ("w1", 6), ("mv2", 4), ("mv3", 3), ("mv4", 3)]
WEST_CASES = [("w1", 6), ("mv2", 4), ("mv3", 3), ("mv4", 3)]


class TestRunnallsMatlab:
    @pytest.mark.parametrize("stem,k", RUNNALLS_CASES)
    def test_matches_matlab(self, stem, k):
        result = reduce_mixture_runnalls(_components(stem), k, weight_threshold=0.0)
        # pytcl appends merged components; MATLAB keeps them in the first
        # partner's slot, so compare order-independently.
        _assert_same_mixture(result, f"{stem}_runnalls", ordered=False)


class TestWestMatlab:
    @pytest.mark.parametrize("stem,k", WEST_CASES)
    def test_kl_matches_matlab(self, stem, k):
        result = reduce_mixture_west(_components(stem), k, weight_threshold=0.0)
        # The port mirrors WestGaussReduction's slot bookkeeping, so the
        # surviving components come out in MATLAB's order.
        _assert_same_mixture(result, f"{stem}_west_kl", ordered=True)

    def test_enhanced_west_matches_matlab(self):
        result = reduce_mixture_west(
            _components("w1"), 6, enhanced=True, weight_threshold=0.0
        )
        _assert_same_mixture(result, "w1_west_kl_enh", ordered=True)

    def test_ise_matches_matlab(self):
        result = reduce_mixture_west(
            _components("w1"), 6, distance="ise", weight_threshold=0.0
        )
        _assert_same_mixture(result, "w1_west_ise", ordered=True)

    @pytest.mark.parametrize("stem", ["mv3", "mv4"])
    def test_ise_refreshes_merged_self_term(self, stem):
        # Deliberate deviation: unmodified WestGaussReduction.m assigns a
        # merged component's refreshed ISE self-term to a local scalar and
        # keeps using the stale one. These fixtures come from MATLAB with
        # that assignment corrected (see tests/fixtures/matlab/README.md);
        # dropping the refresh in the port makes them fail.
        result = reduce_mixture_west(
            _components(stem), 2, distance="ise", weight_threshold=0.0
        )
        _assert_same_mixture(result, f"{stem}_west_ise_patched", ordered=True)


class TestWestSemantics:
    """Behaviour that needs no fixture: the lightest-first rule and the
    ``gamma``/``k_max`` early stop of WestGaussReduction."""

    def _three(self):
        # The two heavy components are the closest pair; West must NOT merge
        # them -- it merges the lightest component with its nearest neighbour.
        return [
            GaussianComponent(0.45, np.array([0.0]), np.array([[1.0]])),
            GaussianComponent(0.45, np.array([0.5]), np.array([[1.0]])),
            GaussianComponent(0.10, np.array([4.0]), np.array([[1.0]])),
        ]

    def test_merges_lightest_component_first(self):
        result = reduce_mixture_west(self._three(), 2, weight_threshold=0.0)
        weights = sorted(c.weight for c in result.components)
        assert weights == pytest.approx([0.45, 0.55])

    def test_gamma_stops_early_when_nearest_is_too_far(self):
        result = reduce_mixture_west(self._three(), 1, gamma=1e-3, weight_threshold=0.0)
        assert len(result.components) == 3

    def test_k_max_overrides_gamma(self):
        result = reduce_mixture_west(
            self._three(), 1, gamma=1e-3, k_max=2, weight_threshold=0.0
        )
        assert len(result.components) == 2

    def test_rejects_unknown_distance(self):
        with pytest.raises(ValueError):
            reduce_mixture_west(self._three(), 2, distance="mahalanobis")
