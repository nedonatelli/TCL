"""
Tests for smoothers and information filter modules.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.batch_smoothers import kalman_batch_smoother
from pytcl.dynamic_estimation.information_filter import (
    InformationFilterResult,
    InformationState,
    SRIFResult,
    fuse_information,
    information_filter,
    information_to_state,
    srif_filter,
    srif_predict,
    srif_update,
    state_to_information,
)
from pytcl.dynamic_estimation.smoothers import (
    FixedLagResult,
    RTSResult,
    SmoothedState,
    fixed_interval_smoother,
    fixed_lag_smoother,
    rts_smoother,
    rts_smoother_single_step,
    two_filter_smoother,
)

RTOL = 1e-10

# 2-state constant-velocity model shared by the oracle-backed tests.
CV_F = np.array([[1.0, 1.0], [0.0, 1.0]])
CV_Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.1
CV_H = np.array([[1.0, 0.0]])
CV_R = np.array([[1.0]])


def _cv_measurements(n, seed, noise=0.5):
    rng = np.random.default_rng(seed)
    return [np.array([float(k) + rng.normal() * noise]) for k in range(n)]


def _rts_oracle(x0, P0, measurements, F, Q, H, R, F_list=None, Q_list=None):
    """Closed-form RTS smoother via the MATLAB-validated batch smoother.

    ``tests/validation/test_batch_smoothers.py`` checks
    ``kalman_batch_smoother(use_fp=False)`` against KalmanBatchSmoother.m.
    ``rts_smoother`` predicts before its first update, so the batch
    smoother's prior is the predicted ``(F0 x0, F0 P0 F0' + Q0)``. A missing
    measurement is a zero row of H, which carries no information.
    Returns ``(x, P)`` with shapes (N, n) and (N, n, n).
    """
    n_steps = len(measurements)
    F_all = [np.asarray(F_list[k] if F_list else F, float) for k in range(n_steps)]
    Q_all = [np.asarray(Q_list[k] if Q_list else Q, float) for k in range(n_steps)]
    H = np.asarray(H, float)
    z_dim = H.shape[0]
    z = np.zeros((z_dim, n_steps))
    h_stack = np.zeros((z_dim, H.shape[1], n_steps))
    for k, m in enumerate(measurements):
        if m is not None:
            z[:, k] = m
            h_stack[:, :, k] = H
    r_stack = np.repeat(np.asarray(R, float)[:, :, None], n_steps, axis=2)
    x_init = F_all[0] @ np.asarray(x0, float)
    p_init = F_all[0] @ np.asarray(P0, float) @ F_all[0].T + Q_all[0]
    res = kalman_batch_smoother(
        x_init,
        p_init,
        z,
        None,
        h_stack,
        np.stack(F_all[1:], axis=2) if n_steps > 1 else F_all[0][:, :, None][:, :, :0],
        r_stack,
        np.stack(Q_all[1:], axis=2) if n_steps > 1 else Q_all[0][:, :, None][:, :, :0],
        use_fp=False,
    )
    return res.x.T, np.moveaxis(res.P, 2, 0)


def _filtered_oracle(x0, P0, measurements, F, Q, H, R):
    """x_{k|k}, P_{k|k}: the last smoothed estimate of each prefix."""
    xs, Ps = [], []
    for k in range(1, len(measurements) + 1):
        x, P = _rts_oracle(x0, P0, measurements[:k], F, Q, H, R)
        xs.append(x[-1])
        Ps.append(P[-1])
    return np.array(xs), np.array(Ps)


class TestRTSSmoother:
    """Tests for RTS smoother."""

    def test_basic_smoothing(self):
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 10.0
        measurements = _cv_measurements(10, seed=11)

        result = rts_smoother(x0, P0, measurements, CV_F, CV_Q, CV_H, CV_R)

        assert isinstance(result, RTSResult)
        x_ref, P_ref = _rts_oracle(x0, P0, measurements, CV_F, CV_Q, CV_H, CV_R)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)
        xf_ref, Pf_ref = _filtered_oracle(x0, P0, measurements, CV_F, CV_Q, CV_H, CV_R)
        np.testing.assert_allclose(result.x_filt, xf_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_filt, Pf_ref, rtol=RTOL, atol=0)

    def test_time_varying_matrices(self):
        n = 2
        x0 = np.zeros(n)
        P0 = np.eye(n)
        F = np.eye(n)
        Q = np.eye(n) * 0.1
        H = np.eye(n)
        R = np.eye(n) * 0.5
        Q_list = [np.eye(n) * (0.1 + 0.01 * k) for k in range(5)]
        rng = np.random.default_rng(12)
        measurements = [
            np.array([1.0, 2.0]) + rng.normal(size=2) * 0.1 for _ in range(5)
        ]

        result = rts_smoother(x0, P0, measurements, F, Q, H, R, Q_list=Q_list)

        x_ref, P_ref = _rts_oracle(x0, P0, measurements, F, Q, H, R, Q_list=Q_list)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)

    def test_missing_measurements(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.array([[0.5]])
        measurements = [
            np.array([1.0]),
            None,
            np.array([3.1]),
            None,
            np.array([5.0]),
        ]

        result = rts_smoother(x0, P0, measurements, CV_F, Q, CV_H, R)

        x_ref, P_ref = _rts_oracle(x0, P0, measurements, CV_F, Q, CV_H, R)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)

    def test_single_measurement(self):
        # Closed-form Kalman update of the predicted prior with H=[1, 0]:
        # predicted P = F P0 F' + Q, K = P H' / (H P H' + R).
        x0 = np.zeros(2)
        P0 = np.eye(2)
        F = np.eye(2)
        Q = np.eye(2) * 0.1
        measurements = [np.array([1.0])]

        result = rts_smoother(x0, P0, measurements, F, Q, CV_H, CV_R)

        P_pred = 1.1 * np.eye(2)
        gain = np.array([1.1, 0.0]) / (1.1 + 1.0)
        np.testing.assert_allclose(result.x_smooth[0], gain * 1.0, rtol=RTOL, atol=0)
        P_expected = P_pred - np.outer(gain, [1.1, 0.0])
        np.testing.assert_allclose(result.P_smooth[0], P_expected, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.x_filt[0], gain * 1.0, rtol=RTOL, atol=0)


class TestFixedLagSmoother:
    """Tests for fixed-lag smoother."""

    @staticmethod
    def _expected(x0, P0, measurements, F, Q, H, R, lag):
        # Entry k holds x_{k-lag|k}; until k reaches lag it is the filtered x_{k|k}.
        xs, Ps = [], []
        for k in range(len(measurements)):
            x, P = _rts_oracle(x0, P0, measurements[: k + 1], F, Q, H, R)
            idx = k - lag if k >= lag else k
            xs.append(x[idx])
            Ps.append(P[idx])
        return np.array(xs), np.array(Ps)

    def test_basic_fixed_lag(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2) * 5.0
        Q = np.eye(2) * 0.1
        R = np.array([[0.5]])
        measurements = _cv_measurements(20, seed=13, noise=0.3)
        lag = 5

        result = fixed_lag_smoother(x0, P0, measurements, CV_F, Q, CV_H, R, lag=lag)

        assert isinstance(result, FixedLagResult)
        assert result.lag == lag
        x_ref, P_ref = self._expected(x0, P0, measurements, CV_F, Q, CV_H, R, lag)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)

    def test_lag_larger_than_data(self):
        x0 = np.zeros(2)
        P0 = np.eye(2)
        F = np.eye(2)
        Q = np.eye(2) * 0.1
        H = np.eye(2)
        R = np.eye(2)
        measurements = [np.array([1.0, 2.0]) for _ in range(3)]

        result = fixed_lag_smoother(x0, P0, measurements, F, Q, H, R, lag=10)

        assert result.lag == 3
        x_ref, P_ref = self._expected(x0, P0, measurements, F, Q, H, R, 3)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)


class TestTwoFilterSmoother:
    """Tests for two-filter (Fraser-Potter) smoother."""

    def test_basic_two_filter(self):
        # The backward filter treats (x0_bwd, P0_bwd) as a direct observation
        # of the state one step past the last measurement. That is exactly an
        # RTS smoother on the batch extended by that observation (H = I,
        # R = P0_bwd) with the real measurements' padding row carrying no
        # information, so the oracle is the batch smoother on the extension.
        n = 2
        x0_fwd = np.zeros(n)
        P0_fwd = np.eye(n)
        x0_bwd = np.zeros(n)
        P0_bwd = np.eye(n) * 100.0
        Q = np.eye(n) * 0.1
        R = np.array([[0.5]])
        measurements = [np.array([float(k)]) for k in range(10)]

        result = two_filter_smoother(
            x0_fwd, P0_fwd, x0_bwd, P0_bwd, measurements, CV_F, Q, CV_H, R
        )

        assert isinstance(result, RTSResult)
        n_meas = len(measurements)
        z = np.zeros((n, n_meas + 1))
        h = np.zeros((n, n, n_meas + 1))
        r = np.zeros((n, n, n_meas + 1))
        for k, m in enumerate(measurements):
            z[0, k] = m[0]
            h[0, :, k] = CV_H[0]
            r[0, 0, k] = R[0, 0]
            r[1, 1, k] = 1.0
        z[:, n_meas] = x0_bwd
        h[:, :, n_meas] = np.eye(n)
        r[:, :, n_meas] = P0_bwd
        ref = kalman_batch_smoother(
            CV_F @ x0_fwd,
            CV_F @ P0_fwd @ CV_F.T + Q,
            z,
            None,
            h,
            CV_F,
            r,
            Q,
            use_fp=False,
        )
        np.testing.assert_allclose(result.x_smooth, ref.x.T[:n_meas], rtol=RTOL, atol=0)
        np.testing.assert_allclose(
            result.P_smooth, np.moveaxis(ref.P, 2, 0)[:n_meas], rtol=RTOL, atol=0
        )


class TestFixedIntervalSmoother:
    """Tests for fixed-interval smoother."""

    def test_fixed_interval_alias(self):
        x0 = np.zeros(2)
        P0 = np.eye(2)
        F = np.eye(2)
        Q = np.eye(2) * 0.1
        H = np.eye(2)
        R = np.eye(2)
        measurements = [np.array([1.0, 2.0]) for _ in range(5)]

        result = fixed_interval_smoother(x0, P0, measurements, F, Q, H, R)

        x_ref, P_ref = _rts_oracle(x0, P0, measurements, F, Q, H, R)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)


class TestRTSSmootherSingleStep:
    """Tests for single-step RTS smoother."""

    def test_single_step_matches_rts_formula(self):
        # With F = I and P_filt = I the gain is G = 1 / 1.1 exactly:
        # x = x_filt + G (x_s - x_p), P = P_filt + G (P_s - P_p) G.
        x_filt = np.array([1.0, 2.0])
        P_filt = np.eye(2)
        x_pred_next = np.array([1.5, 2.0])
        P_pred_next = np.eye(2) * 1.1
        x_smooth_next = np.array([1.4, 2.1])
        P_smooth_next = np.eye(2) * 0.9
        F = np.eye(2)

        result = rts_smoother_single_step(
            x_filt, P_filt, x_pred_next, P_pred_next, x_smooth_next, P_smooth_next, F
        )

        assert isinstance(result, SmoothedState)
        gain = 1.0 / 1.1
        np.testing.assert_allclose(
            result.x,
            [1.0 + gain * (1.4 - 1.5), 2.0 + gain * (2.1 - 2.0)],
            rtol=1e-14,
            atol=0,
        )
        np.testing.assert_allclose(
            result.P, (1.0 + gain * gain * (0.9 - 1.1)) * np.eye(2), rtol=1e-14, atol=0
        )


class TestInformationFilter:
    """Tests for information filter."""

    def test_state_information_conversion(self):
        # Y = inv(P) via the 2x2 adjugate, y = Y x.
        x = np.array([1.0, 2.0])
        P = np.array([[1.0, 0.2], [0.2, 0.5]])

        y, Y = state_to_information(x, P)

        Y_expected = np.array([[0.5, -0.2], [-0.2, 1.0]]) / (1.0 * 0.5 - 0.2 * 0.2)
        np.testing.assert_allclose(Y, Y_expected, rtol=1e-14, atol=0)
        np.testing.assert_allclose(y, Y_expected @ x, rtol=1e-14, atol=0)
        x_back, P_back = information_to_state(Y_expected @ x, Y_expected)
        np.testing.assert_allclose(x_back, x, rtol=1e-13, atol=0)
        np.testing.assert_allclose(P_back, P, rtol=1e-13, atol=0)

    def test_basic_information_filter(self):
        n = 2
        x0 = np.zeros(n)
        P0 = np.eye(n) * 10.0
        y0, Y0 = state_to_information(x0, P0)
        Q = np.eye(n) * 0.1
        R = np.array([[1.0]])
        measurements = [np.array([float(k)]) for k in range(5)]

        result = information_filter(y0, Y0, measurements, CV_F, Q, CV_H, R)

        assert isinstance(result, InformationFilterResult)
        x_ref, P_ref = _filtered_oracle(x0, P0, measurements, CV_F, Q, CV_H, R)
        Y_ref = np.linalg.inv(P_ref)
        np.testing.assert_allclose(result.x_filt, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_filt, P_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.Y_filt, Y_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(
            result.y_filt,
            np.einsum("kij,kj->ki", Y_ref, x_ref),
            rtol=RTOL,
            atol=1e-14,  # exact zeros in y round to ~1e-16
        )

    def test_unknown_initial_state(self):
        # Zero information prior, F = H = I, Q = 0.01 I, R = 0.5 I: every
        # component follows the scalar recursion
        #   p_pred = p + q, K = p_pred / (p_pred + r), x = x_pred + K (z - x_pred)
        # started from x_1 = z, p_1 = r (the first update has no prior).
        n = 2
        y0 = np.zeros(n)
        Y0 = np.zeros((n, n))
        F = np.eye(n)
        Q = np.eye(n) * 0.01
        H = np.eye(n)
        R = np.eye(n) * 0.5
        z = np.array([1.0, 2.0])
        measurements = [z.copy() for _ in range(5)]

        result = information_filter(y0, Y0, measurements, F, Q, H, R)

        p, q, r = 0.5, 0.01, 0.5
        expected_p = [p]
        for _ in range(4):
            p_pred = p + q
            p = p_pred - p_pred * p_pred / (p_pred + r)
            expected_p.append(p)
        np.testing.assert_allclose(
            result.P_filt,
            [pk * np.eye(n) for pk in expected_p],
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            result.Y_filt,
            [np.eye(n) / pk for pk in expected_p],
            rtol=1e-12,
            atol=0,
        )
        # z is constant, so the estimate is z at every step.
        np.testing.assert_allclose(result.x_filt, [z] * 5, rtol=1e-12, atol=0)


class TestSRIF:
    """Tests for Square-Root Information Filter."""

    def test_srif_predict(self):
        # P0 = I so R0 = I and r0 = x0. F = I, Q = 0.1 I give P = 1.1 I:
        # R'R = I / 1.1 and R^{-1} r = x0.
        n = 2
        x0 = np.array([1.0, 0.0])
        R0 = np.eye(n)
        r0 = R0 @ x0

        r_pred, R_pred = srif_predict(r0, R0, np.eye(n), np.eye(n) * 0.1)

        np.testing.assert_allclose(
            R_pred.T @ R_pred, np.eye(n) / 1.1, rtol=1e-14, atol=0
        )
        np.testing.assert_allclose(
            np.linalg.solve(R_pred, r_pred), x0, rtol=1e-14, atol=0
        )

    def test_srif_update(self):
        # Information form: Y = R'R + H' Rm^-1 H, y = R'r + H' Rm^-1 z.
        n = 2
        R = np.eye(n) * 0.5
        r = np.array([0.5, 0.0])
        z = np.array([1.0])
        H = np.array([[1.0, 0.0]])
        R_meas = np.array([[1.0]])

        r_upd, R_upd = srif_update(r, R, z, H, R_meas)

        Y_expected = np.array([[1.25, 0.0], [0.0, 0.25]])
        y_expected = np.array([1.25, 0.0])
        np.testing.assert_allclose(R_upd.T @ R_upd, Y_expected, rtol=1e-14, atol=0)
        np.testing.assert_allclose(R_upd.T @ r_upd, y_expected, rtol=1e-14, atol=1e-16)

    def test_basic_srif_filter(self):
        n = 2
        x0 = np.zeros(n)
        P0 = np.eye(n) * 10.0
        R0 = np.linalg.inv(np.linalg.cholesky(P0)).T
        r0 = R0 @ x0
        Q = np.eye(n) * 0.1
        R_meas = np.array([[1.0]])
        measurements = [np.array([float(k)]) for k in range(5)]

        result = srif_filter(r0, R0, measurements, CV_F, Q, CV_H, R_meas)

        assert isinstance(result, SRIFResult)
        x_ref, P_ref = _filtered_oracle(x0, P0, measurements, CV_F, Q, CV_H, R_meas)
        np.testing.assert_allclose(result.x_filt, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_filt, P_ref, rtol=RTOL, atol=0)
        for k in range(5):
            np.testing.assert_allclose(
                result.R_filt[k].T @ result.R_filt[k],
                np.linalg.inv(P_ref[k]),
                rtol=1e-9,
                atol=0,
            )
            np.testing.assert_allclose(
                result.r_filt[k], result.R_filt[k] @ x_ref[k], rtol=1e-9, atol=1e-12
            )


class TestInformationFusion:
    """Tests for information fusion."""

    def test_fuse_two_sensors(self):
        """Test fusing information from two sensors."""
        state1 = InformationState(
            y=np.array([1.0, 0.5]),
            Y=np.array([[0.5, 0], [0, 0.1]]),
        )
        state2 = InformationState(
            y=np.array([0.8, 0.6]),
            Y=np.array([[0.3, 0], [0, 0.2]]),
        )

        fused = fuse_information([state1, state2])

        # Fusion is additive
        np.testing.assert_allclose(fused.y, state1.y + state2.y)
        np.testing.assert_allclose(fused.Y, state1.Y + state2.Y)

    def test_fuse_multiple_sensors(self):
        """Test fusing information from multiple sensors."""
        n = 3
        states = [
            InformationState(
                y=np.ones(n) * i,
                Y=np.eye(n) * (0.1 + 0.1 * i),
            )
            for i in range(1, 5)
        ]

        fused = fuse_information(states)

        expected_y = sum(s.y for s in states)
        expected_Y = sum(s.Y for s in states)

        np.testing.assert_allclose(fused.y, expected_y)
        np.testing.assert_allclose(fused.Y, expected_Y)

    def test_fuse_empty_raises(self):
        """Test that fusing empty list raises error."""
        with pytest.raises(ValueError):
            fuse_information([])


class TestSmootherConsistency:
    """Tests for consistency between smoothers."""

    def test_rts_matches_oracle_over_long_run(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2) * 10.0
        measurements = _cv_measurements(20, seed=14)

        result = rts_smoother(x0, P0, measurements, CV_F, CV_Q, CV_H, CV_R)

        x_ref, P_ref = _rts_oracle(x0, P0, measurements, CV_F, CV_Q, CV_H, CV_R)
        np.testing.assert_allclose(result.x_smooth, x_ref, rtol=RTOL, atol=0)
        np.testing.assert_allclose(result.P_smooth, P_ref, rtol=RTOL, atol=0)
