"""Monte-Carlo ("QMC") Kalman family validation.

The deterministic parts (update-with-prediction, gain) are validated
against MATLAB fixtures captured via
scripts/matlab_capture/capture_qmc_parts.m with a hand-built
measurement-prediction struct (mirrored verbatim here). The sampling
functions draw from MATLAB's global RNG with no injection point, so
they are validated statistically: on a linear-Gaussian problem their
moments must converge to the exact Kalman filter's.
"""

from pathlib import Path

import numpy as np

from pytcl.dynamic_estimation.kalman import (
    QMCMeasPredInfo,
    calc_qmc_kalman_gain,
    qmc_kf_meas_pred,
    qmc_kf_predict,
    qmc_kf_update,
    qmc_kf_update_with_pred,
)
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Literal transcriptions of the deterministic parts: measured max
# disagreement 3.1e-15.
ATOL = 1e-12


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


# The hand-built two-component measurement-prediction scene from the
# capture script.
X_PRED = np.array([[1.0, -0.6], [-0.4, 0.9], [0.3, 1.2]])
P_PRED = np.stack(
    [
        np.array([[0.5, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.3]]),
        np.array([[0.9, -0.2, 0.1], [-0.2, 0.6, 0.0], [0.1, 0.0, 0.4]]),
    ],
    axis=2,
)
Z_PRED = np.array([[0.95, -0.55], [-0.35, 0.85]])
PZ_PRED = np.stack(
    [
        np.array([[0.45, 0.08], [0.08, 0.7]]),
        np.array([[0.85, -0.15], [-0.15, 0.55]]),
    ],
    axis=2,
)
PXZ = np.stack(
    [
        np.array([[0.4, 0.05], [0.12, 0.65], [0.02, 0.08]]),
        np.array([[0.8, -0.1], [-0.18, 0.5], [0.09, 0.03]]),
    ],
    axis=2,
)
INFO = QMCMeasPredInfo(
    Z_PRED, PZ_PRED, PXZ, X_PRED, P_PRED, lambda a, b: a - b, lambda x: x
)
Z = np.array([1.15, -0.25])
R = 0.2 * np.eye(2)


class TestDeterministicPartsAgainstMatlab:
    def test_update_with_pred_matches_matlab(self):
        up = qmc_kf_update_with_pred(Z, R, INFO)
        np.testing.assert_allclose(up.x, _load("qmc_upwp_x.csv"), atol=ATOL)
        np.testing.assert_allclose(
            np.vstack([up.P[:, :, 0], up.P[:, :, 1]]),
            _load("qmc_upwp_P.csv"),
            atol=ATOL,
        )
        np.testing.assert_allclose(up.innov, _load("qmc_upwp_innov.csv"), atol=ATOL)
        np.testing.assert_allclose(
            np.vstack([up.pzz[:, :, 0], up.pzz[:, :, 1]]),
            _load("qmc_upwp_Pzz.csv"),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            np.vstack([up.gain[:, :, 0], up.gain[:, :, 1]]),
            _load("qmc_upwp_W.csv"),
            atol=ATOL,
        )

    def test_gain_matches_matlab(self):
        g = calc_qmc_kalman_gain(R, PZ_PRED[:, :, 0], INFO)
        np.testing.assert_allclose(g, _load("qmc_gain.csv"), atol=ATOL)


class TestSamplingFunctionsConvergeToKalman:
    """The sampling steps must reproduce the exact KF moments on a
    linear-Gaussian problem as the sample count grows."""

    F = np.array([[1.0, 0.5, 0.125], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = 0.01 * np.eye(3)
    R2 = 0.04 * np.eye(2)
    X0 = np.array([1.0, -0.5, 0.25])
    P0 = np.diag([0.4, 0.9, 0.2])
    Z2 = np.array([1.6, -0.7])

    def test_predict_tracks_kf(self):
        rng = np.random.default_rng(42)
        kf = kf_predict(self.X0, self.P0, self.F, self.Q)
        mc = qmc_kf_predict(
            self.X0, self.P0, lambda x: self.F @ x, self.Q, 40000, rng=rng
        )
        # Monte-Carlo error ~1/sqrt(n); bounds sit ~5x above observed.
        np.testing.assert_allclose(mc.x, kf.x, atol=0.03)
        np.testing.assert_allclose(mc.P, kf.P, atol=0.03)

    def test_update_tracks_kf(self):
        rng = np.random.default_rng(43)
        kf_p = kf_predict(self.X0, self.P0, self.F, self.Q)
        kf_u = kf_update(kf_p.x, kf_p.P, self.Z2, self.H, self.R2)
        mc = qmc_kf_update(
            kf_p.x, kf_p.P, self.Z2, self.R2, lambda x: self.H @ x, 40000, rng=rng
        )
        np.testing.assert_allclose(mc.x, kf_u.x, atol=0.03)
        np.testing.assert_allclose(mc.P, kf_u.P, atol=0.03)

    def test_meas_pred_plus_update_matches_direct_update_statistically(self):
        rng = np.random.default_rng(44)
        info = qmc_kf_meas_pred(
            self.X0, self.P0, 2, lambda x: self.H @ x, 40000, rng=rng
        )
        up = qmc_kf_update_with_pred(self.Z2, self.R2, info)
        kf_u = kf_update(self.X0, self.P0, self.Z2, self.H, self.R2)
        np.testing.assert_allclose(up.x[:, 0], kf_u.x, atol=0.03)
        np.testing.assert_allclose(up.P[:, :, 0], kf_u.P, atol=0.03)

    def test_multi_component_bank(self):
        rng = np.random.default_rng(45)
        x_bank = np.column_stack([self.X0, self.X0 + 0.5])
        p_bank = np.stack([self.P0, 2.0 * self.P0], axis=2)
        info = qmc_kf_meas_pred(x_bank, p_bank, 2, lambda x: self.H @ x, 20000, rng=rng)
        up = qmc_kf_update_with_pred(self.Z2, self.R2, info)
        assert up.x.shape == (3, 2)
        for c, (x0, p0) in enumerate(
            [(self.X0, self.P0), (self.X0 + 0.5, 2.0 * self.P0)]
        ):
            kf_u = kf_update(x0, p0, self.Z2, self.H, self.R2)
            np.testing.assert_allclose(up.x[:, c], kf_u.x, atol=0.05)

    def test_reproducible_with_seeded_generator(self):
        a = qmc_kf_predict(
            self.X0,
            self.P0,
            lambda x: self.F @ x,
            self.Q,
            500,
            rng=np.random.default_rng(7),
        )
        b = qmc_kf_predict(
            self.X0,
            self.P0,
            lambda x: self.F @ x,
            self.Q,
            500,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.P, b.P)
