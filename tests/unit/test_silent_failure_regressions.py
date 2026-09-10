"""Regression tests for the v2.11.0 tier-0 silent-failure fixes.

Each test pins a path that previously returned wrong or meaningless
results with no signal: the fix is only real if the loud behavior is
exercised.
"""

import sys

import numpy as np
import pytest

from pytcl.assignment_algorithms.data_association import (
    compute_association_cost,
)
from pytcl.core.exceptions import ConvergenceError
from pytcl.dynamic_estimation.kalman.linear import kf_update
from pytcl.dynamic_estimation.smoothers import two_filter_smoother
from pytcl.mathematical_functions.special_functions.hypergeometric import (
    generalized_hypergeometric,
)
from pytcl.navigation.geodesy import inverse_geodetic


class TestVincentyAntipodal:
    ANTIPODAL = (0.0, 0.0, np.radians(0.5), np.radians(179.7))

    def test_falls_back_to_geographiclib(self):
        geographiclib = pytest.importorskip("geographiclib.geodesic")
        dist, az1, _ = inverse_geodetic(*self.ANTIPODAL)
        ref = geographiclib.Geodesic.WGS84.Inverse(0.0, 0.0, 0.5, 179.7)
        # The unconverged Vincenty result was 3.7 km / 22 degrees off;
        # the Karney fallback matches geographiclib to the cache's
        # 1e-10 rad input quantization (~sub-millimeter).
        np.testing.assert_allclose(dist, ref["s12"], atol=1e-2)
        np.testing.assert_allclose(np.degrees(az1), ref["azi1"], atol=1e-6)

    def test_raises_without_geographiclib(self, monkeypatch):
        import pytcl.navigation.geodesy as geo

        # The lru_cache would replay the fallback result; call the
        # uncached implementation with geographiclib made unimportable.
        monkeypatch.setitem(sys.modules, "geographiclib", None)
        monkeypatch.setitem(sys.modules, "geographiclib.geodesic", None)
        with pytest.raises(ConvergenceError, match="did not converge"):
            geo._inverse_geodetic_cached.__wrapped__(
                0.0,
                0.0,
                np.radians(0.5),
                np.radians(179.7),
                6378137.0,
                1.0 / 298.257223563,
            )


class TestAssociationNoise:
    def test_per_track_noise_matrices(self):
        preds = np.array([[0.0, 1.0], [5.0, -1.0]])
        covs = np.array([np.eye(2), 2 * np.eye(2)])
        Z = np.array([[0.1], [4.9]])
        H = np.array([[1.0, 0.0]])
        R_stack = np.array([[[0.5]], [[2.0]]])  # (n_tracks, m, m)
        C = compute_association_cost(preds, covs, Z, H, measurement_noise=R_stack)
        for i in range(2):
            S = H @ covs[i] @ H.T + R_stack[i]
            for j in range(2):
                nu = Z[j] - H @ preds[i]
                np.testing.assert_allclose(
                    C[i, j], float(nu @ np.linalg.inv(S) @ nu), rtol=1e-9
                )


class TestKalmanCholeskyWarning:
    def test_non_psd_innovation_warns_and_zeroes_likelihood(self):
        x = np.zeros(2)
        P = np.eye(2)
        H = np.array([[1.0, 0.0]])
        R = np.array([[-2.0]])  # drives S negative definite
        with pytest.warns(RuntimeWarning, match="not positive definite"):
            upd = kf_update(x, P, np.array([0.5]), H, R)
        assert upd.likelihood == 0.0


class TestTwoFilterSmootherWarnings:
    def test_singular_transition_warns(self):
        F = np.array([[1.0, 0.0], [1.0, 0.0]])  # singular
        H = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = 0.1 * np.eye(2)
        zs = [np.array([1.0, 0.0]), np.array([1.1, 0.1]), np.array([1.2, 0.2])]
        with pytest.warns(RuntimeWarning, match="singular"):
            two_filter_smoother(
                np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), zs, F, Q, H, R
            )

    def test_singular_fusion_covariances_warn_per_site(self):
        # Rank-deficient initials with identity dynamics, zero process
        # noise, and no measurement keep both filters' covariances
        # singular, so the forward, backward, and fused inversions all
        # fall back to the pseudo-inverse -- each with its own warning.
        P_sing = np.diag([1.0, 0.0])
        with pytest.warns(RuntimeWarning) as record:
            two_filter_smoother(
                np.zeros(2),
                P_sing,
                np.zeros(2),
                P_sing,
                [None],
                np.eye(2),
                np.zeros((2, 2)),
                np.eye(2),
                0.1 * np.eye(2),
            )
        messages = [str(w.message) for w in record]
        for which in ("forward", "backward", "fused"):
            assert any(f"singular {which}" in m for m in messages), which


class TestHypergeometricGuards:
    def test_divergent_p_eq_q_plus_1_raises(self):
        with pytest.raises(ValueError, match="diverges"):
            generalized_hypergeometric([1, 1, 1], [2, 2], 1.5)

    def test_divergent_p_gt_q_plus_1_raises(self):
        with pytest.raises(ValueError, match="diverges"):
            generalized_hypergeometric([1, 1, 1, 1], [2, 2], 0.5)

    def test_terminating_series_allowed_outside_domain(self):
        # a = -2 terminates the series: 3F2(-2, 1, 1; 2, 2; 2) is a
        # polynomial, evaluated exactly. Sum: 1 - 2*(2)/4 + ... = 4/9.
        val = float(generalized_hypergeometric([-2, 1, 1], [2, 2], 2.0))
        np.testing.assert_allclose(val, 4.0 / 9.0, rtol=1e-12)

    def test_exhaustion_warns(self):
        with pytest.warns(RuntimeWarning, match="did not reach"):
            generalized_hypergeometric([1, 1, 1], [2, 2], 0.999, max_terms=50)
