"""Validation tests for the Bernoulli filter.

Oracle: the closed-form equations of Ristic, Vo, Vo & Farina, "A
Tutorial on Bernoulli Filters" (IEEE TSP 2013), computed independently
here with scipy densities, plus the filter's exact limiting behaviors.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pytcl.trackers.bernoulli import (
    BernoulliConfig,
    BernoulliState,
    bernoulli_predict,
    bernoulli_update,
)

H = np.eye(2)
R = 0.1 * np.eye(2)
F = np.array([[1.0, 0.5], [0.0, 1.0]])
Q = 0.01 * np.eye(2)
BIRTH_MEAN = np.zeros(2)
BIRTH_COV = 4.0 * np.eye(2)


class TestPredictOracle:
    def test_existence_prediction_formula(self):
        cfg = BernoulliConfig(birth_prob=0.03, survival_prob=0.97)
        state = BernoulliState(r=0.4, x=np.array([1.0, 0.2]), P=np.eye(2))
        pred = bernoulli_predict(state, F, Q, BIRTH_MEAN, BIRTH_COV, cfg)
        expected_r = 0.03 * (1 - 0.4) + 0.97 * 0.4
        np.testing.assert_allclose(pred.r, expected_r, rtol=1e-12)

    def test_moment_matched_mixture(self):
        cfg = BernoulliConfig(birth_prob=0.2, survival_prob=0.8)
        state = BernoulliState(r=0.5, x=np.array([1.0, 0.2]), P=np.eye(2))
        pred = bernoulli_predict(state, F, Q, BIRTH_MEAN, BIRTH_COV, cfg)

        r_pred = 0.2 * 0.5 + 0.8 * 0.5
        w_b = 0.2 * 0.5 / r_pred
        w_s = 0.8 * 0.5 / r_pred
        x_s = F @ state.x
        P_s = F @ state.P @ F.T + Q
        x_exp = w_b * BIRTH_MEAN + w_s * x_s
        P_exp = w_b * (
            BIRTH_COV + np.outer(BIRTH_MEAN - x_exp, BIRTH_MEAN - x_exp)
        ) + w_s * (P_s + np.outer(x_s - x_exp, x_s - x_exp))
        np.testing.assert_allclose(pred.x, x_exp, rtol=1e-12)
        np.testing.assert_allclose(pred.P, P_exp, rtol=1e-12)

    def test_impossible_existence_keeps_birth_density(self):
        cfg = BernoulliConfig(birth_prob=0.0, survival_prob=0.99)
        state = BernoulliState(r=0.0, x=np.array([1.0, 0.2]), P=np.eye(2))
        pred = bernoulli_predict(state, F, Q, BIRTH_MEAN, BIRTH_COV, cfg)
        assert pred.r == 0.0
        np.testing.assert_allclose(pred.x, BIRTH_MEAN)
        np.testing.assert_allclose(pred.P, BIRTH_COV)

    def test_certain_existence_reduces_to_kalman_prediction(self):
        cfg = BernoulliConfig(birth_prob=0.1, survival_prob=1.0)
        state = BernoulliState(r=1.0, x=np.array([1.0, 0.2]), P=np.eye(2))
        pred = bernoulli_predict(state, F, Q, BIRTH_MEAN, BIRTH_COV, cfg)
        np.testing.assert_allclose(pred.r, 1.0)
        np.testing.assert_allclose(pred.x, F @ state.x, rtol=1e-12)
        np.testing.assert_allclose(pred.P, F @ state.P @ F.T + Q, rtol=1e-12)


class TestUpdateOracle:
    def test_existence_update_formula_single_measurement(self):
        cfg = BernoulliConfig(detection_prob=0.9, clutter_intensity=0.01)
        state = BernoulliState(r=0.5, x=np.array([1.0, 0.2]), P=np.eye(2))
        z = np.array([1.2, 0.1])
        upd = bernoulli_update(state, [z], H, R, cfg)

        S = H @ state.P @ H.T + R
        q = multivariate_normal.pdf(z, mean=H @ state.x, cov=S)
        delta = 0.9 * (1.0 - q / 0.01)
        r_exp = state.r * (1.0 - delta) / (1.0 - state.r * delta)
        np.testing.assert_allclose(upd.r, r_exp, rtol=1e-10)

    def test_spatial_update_moment_match(self):
        cfg = BernoulliConfig(detection_prob=0.8, clutter_intensity=0.05)
        state = BernoulliState(r=0.5, x=np.array([1.0, 0.2]), P=np.eye(2))
        z0 = np.array([1.3, 0.0])
        z1 = np.array([0.5, 0.7])
        upd = bernoulli_update(state, [z0, z1], H, R, cfg)

        S = H @ state.P @ H.T + R
        K = state.P @ H.T @ np.linalg.inv(S)
        P_k = state.P - K @ S @ K.T
        weights = [1.0 - 0.8]
        means = [state.x]
        covs = [state.P]
        for z in (z0, z1):
            q = multivariate_normal.pdf(z, mean=H @ state.x, cov=S)
            weights.append(0.8 * q / 0.05)
            means.append(state.x + K @ (z - H @ state.x))
            covs.append(P_k)
        w = np.array(weights) / sum(weights)
        x_exp = sum(wi * mi for wi, mi in zip(w, means))
        P_exp = sum(
            wi * (ci + np.outer(mi - x_exp, mi - x_exp))
            for wi, mi, ci in zip(w, means, covs)
        )
        np.testing.assert_allclose(upd.x, x_exp, rtol=1e-10)
        np.testing.assert_allclose(upd.P, P_exp, rtol=1e-10)

    def test_no_detection_decays_existence(self):
        cfg = BernoulliConfig(detection_prob=0.9)
        state = BernoulliState(r=0.6, x=np.zeros(2), P=np.eye(2))
        upd = bernoulli_update(state, [], H, R, cfg)
        # delta = Pd; r = r (1-Pd) / (1 - r Pd)
        r_exp = 0.6 * 0.1 / (1.0 - 0.6 * 0.9)
        np.testing.assert_allclose(upd.r, r_exp, rtol=1e-12)
        np.testing.assert_allclose(upd.x, state.x)
        np.testing.assert_allclose(upd.P, state.P)

    def test_zero_detection_prob_is_uninformative(self):
        cfg = BernoulliConfig(detection_prob=0.0, clutter_intensity=0.01)
        state = BernoulliState(r=0.6, x=np.zeros(2), P=np.eye(2))
        upd = bernoulli_update(state, [np.array([5.0, 5.0])], H, R, cfg)
        np.testing.assert_allclose(upd.r, state.r, rtol=1e-12)
        np.testing.assert_allclose(upd.x, state.x)
        np.testing.assert_allclose(upd.P, state.P)

    def test_nonpositive_clutter_intensity_raises(self):
        cfg = BernoulliConfig(clutter_intensity=0.0)
        state = BernoulliState(r=0.5, x=np.zeros(2), P=np.eye(2))
        with pytest.raises(ValueError, match="clutter_intensity"):
            bernoulli_update(state, [], H, R, cfg)


class TestScenarios:
    CFG = BernoulliConfig(
        birth_prob=0.05,
        survival_prob=0.99,
        detection_prob=0.9,
        clutter_intensity=0.01,
    )

    def test_present_target_confirms_and_tracks(self):
        state = BernoulliState(r=0.0, x=BIRTH_MEAN, P=BIRTH_COV)
        for k in range(1, 21):
            state = bernoulli_predict(state, F, Q, BIRTH_MEAN, BIRTH_COV, self.CFG)
            truth = np.array([0.5 * k, 1.0])  # x advances 0.5/scan, v = 1
            z = np.array([truth[0], truth[1]])
            state = bernoulli_update(state, [z], H, R, self.CFG)
        assert state.r > 0.95, f"existence should confirm, got r={state.r:.3f}"
        assert abs(state.x[0] - 10.0) < 0.5

    def test_absent_target_stays_unconfirmed(self):
        rng = np.random.default_rng(7)
        state = BernoulliState(r=0.0, x=BIRTH_MEAN, P=BIRTH_COV)
        for _ in range(20):
            state = bernoulli_predict(state, F, Q, BIRTH_MEAN, BIRTH_COV, self.CFG)
            # Sparse far-away clutter, inconsistent scan to scan.
            clutter = [rng.uniform(20.0, 60.0, size=2)]
            state = bernoulli_update(state, clutter, H, R, self.CFG)
        assert state.r < 0.3, (
            f"clutter-only data should not confirm, got r={state.r:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
