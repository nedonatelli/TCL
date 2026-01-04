"""
Tests for smoothers and information filter modules.
"""

import numpy as np
import pytest

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


class TestRTSSmoother:
    """Tests for RTS smoother."""

    def test_basic_smoothing(self):
        """Test basic RTS smoother on 1D tracking."""
        # Simple CV model
        x0 = np.array([0.0, 0.0])  # [position, velocity]
        P0 = np.eye(2) * 10.0

        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        Q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[1.0]])

        # Generate measurements with constant velocity (vel=1.0)
        measurements = [
            np.array([float(k) + np.random.randn() * 0.5]) for k in range(10)
        ]

        result = rts_smoother(x0, P0, measurements, F, Q, H, R)

        assert isinstance(result, RTSResult)
        assert len(result.x_smooth) == 10
        assert len(result.P_smooth) == 10
        assert len(result.x_filt) == 10
        assert len(result.P_filt) == 10

        # Smoothed covariances should be smaller than filtered
        for k in range(len(result.x_smooth) - 1):
            # Compare trace of covariances
            trace_smooth = np.trace(result.P_smooth[k])
            trace_filt = np.trace(result.P_filt[k])
            assert trace_smooth <= trace_filt + 1e-10

    def test_time_varying_matrices(self):
        """Test RTS smoother with time-varying system matrices."""
        n = 2
        x0 = np.zeros(n)
        P0 = np.eye(n)

        F = np.eye(n)
        Q = np.eye(n) * 0.1
        H = np.eye(n)
        R = np.eye(n) * 0.5

        # Time-varying Q
        Q_list = [np.eye(n) * (0.1 + 0.01 * k) for k in range(5)]

        measurements = [
            np.array([1.0, 2.0]) + np.random.randn(2) * 0.1 for _ in range(5)
        ]

        result = rts_smoother(x0, P0, measurements, F, Q, H, R, Q_list=Q_list)

        assert len(result.x_smooth) == 5

    def test_missing_measurements(self):
        """Test RTS smoother with missing measurements."""
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        # Some measurements are None
        measurements = [
            np.array([1.0]),
            None,
            np.array([3.1]),
            None,
            np.array([5.0]),
        ]

        result = rts_smoother(x0, P0, measurements, F, Q, H, R)

        assert len(result.x_smooth) == 5
        # Filter should still work with missing measurements

    def test_single_measurement(self):
        """Test RTS smoother with single measurement."""
        x0 = np.zeros(2)
        P0 = np.eye(2)

        F = np.eye(2)
        Q = np.eye(2) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[1.0]])

        measurements = [np.array([1.0])]

        result = rts_smoother(x0, P0, measurements, F, Q, H, R)

        assert len(result.x_smooth) == 1
        # With single measurement, smooth == filtered
        np.testing.assert_allclose(result.x_smooth[0], result.x_filt[0])


class TestFixedLagSmoother:
    """Tests for fixed-lag smoother."""

    def test_basic_fixed_lag(self):
        """Test basic fixed-lag smoother."""
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2) * 5.0

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        measurements = [np.array([float(k)]) for k in range(20)]

        lag = 5
        result = fixed_lag_smoother(x0, P0, measurements, F, Q, H, R, lag=lag)

        assert isinstance(result, FixedLagResult)
        assert len(result.x_smooth) == 20
        assert result.lag == lag

    def test_lag_larger_than_data(self):
        """Test with lag larger than number of measurements."""
        x0 = np.zeros(2)
        P0 = np.eye(2)

        F = np.eye(2)
        Q = np.eye(2) * 0.1
        H = np.eye(2)
        R = np.eye(2)

        measurements = [np.array([1.0, 2.0]) for _ in range(3)]

        result = fixed_lag_smoother(x0, P0, measurements, F, Q, H, R, lag=10)

        assert len(result.x_smooth) == 3
        assert result.lag == 3  # Should be clamped to data length


class TestTwoFilterSmoother:
    """Tests for two-filter (Fraser-Potter) smoother."""

    def test_basic_two_filter(self):
        """Test basic two-filter smoother."""
        n = 2
        x0_fwd = np.zeros(n)
        P0_fwd = np.eye(n)
        x0_bwd = np.zeros(n)
        P0_bwd = np.eye(n) * 100.0  # Diffuse backward prior

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(n) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        measurements = [np.array([float(k)]) for k in range(10)]

        result = two_filter_smoother(
            x0_fwd, P0_fwd, x0_bwd, P0_bwd, measurements, F, Q, H, R
        )

        assert isinstance(result, RTSResult)
        assert len(result.x_smooth) == 10


class TestFixedIntervalSmoother:
    """Tests for fixed-interval smoother."""

    def test_fixed_interval_alias(self):
        """Test that fixed_interval_smoother is alias for rts_smoother."""
        x0 = np.zeros(2)
        P0 = np.eye(2)

        F = np.eye(2)
        Q = np.eye(2) * 0.1
        H = np.eye(2)
        R = np.eye(2)

        measurements = [np.array([1.0, 2.0]) for _ in range(5)]

        result1 = rts_smoother(x0, P0, measurements, F, Q, H, R)
        result2 = fixed_interval_smoother(x0, P0, measurements, F, Q, H, R)

        for k in range(5):
            np.testing.assert_allclose(result1.x_smooth[k], result2.x_smooth[k])
            np.testing.assert_allclose(result1.P_smooth[k], result2.P_smooth[k])


class TestRTSSmootherSingleStep:
    """Tests for single-step RTS smoother."""

    def test_single_step_returns_tuple(self):
        """Test single step wrapper."""
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
        assert len(result.x) == 2
        assert result.P.shape == (2, 2)


class TestInformationFilter:
    """Tests for information filter."""

    def test_state_information_conversion(self):
        """Test conversion between state and information form."""
        x = np.array([1.0, 2.0])
        P = np.array([[1.0, 0.2], [0.2, 0.5]])

        y, Y = state_to_information(x, P)
        x_back, P_back = information_to_state(y, Y)

        np.testing.assert_allclose(x, x_back)
        np.testing.assert_allclose(P, P_back)

    def test_basic_information_filter(self):
        """Test basic information filter run."""
        n = 2
        x0 = np.zeros(n)
        P0 = np.eye(n) * 10.0
        y0, Y0 = state_to_information(x0, P0)

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(n) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[1.0]])

        measurements = [np.array([float(k)]) for k in range(5)]

        result = information_filter(y0, Y0, measurements, F, Q, H, R)

        assert isinstance(result, InformationFilterResult)
        assert len(result.x_filt) == 5
        assert len(result.y_filt) == 5
        assert len(result.Y_filt) == 5
        assert len(result.P_filt) == 5

    def test_unknown_initial_state(self):
        """Test information filter with unknown initial state."""
        n = 2
        # Zero information = unknown state
        y0 = np.zeros(n)
        Y0 = np.zeros((n, n))

        F = np.eye(n)
        Q = np.eye(n) * 0.01
        H = np.eye(n)
        R = np.eye(n) * 0.5

        measurements = [np.array([1.0, 2.0]) for _ in range(5)]

        result = information_filter(y0, Y0, measurements, F, Q, H, R)

        # After several measurements, should have information
        assert np.linalg.matrix_rank(result.Y_filt[-1]) == n


class TestSRIF:
    """Tests for Square-Root Information Filter."""

    def test_srif_predict(self):
        """Test SRIF prediction step."""
        n = 2
        x0 = np.array([1.0, 0.0])
        P0 = np.eye(n)

        # Get initial SRIF state
        R0 = np.linalg.inv(np.linalg.cholesky(P0)).T
        r0 = R0 @ x0

        F = np.eye(n)
        Q = np.eye(n) * 0.1

        r_pred, R_pred = srif_predict(r0, R0, F, Q)

        assert len(r_pred) == n
        assert R_pred.shape == (n, n)

    def test_srif_update(self):
        """Test SRIF update step."""
        n = 2
        R = np.eye(n) * 0.5
        r = np.array([0.5, 0.0])

        z = np.array([1.0])
        H = np.array([[1, 0]])
        R_meas = np.array([[1.0]])

        r_upd, R_upd = srif_update(r, R, z, H, R_meas)

        assert len(r_upd) == n
        assert R_upd.shape == (n, n)

    def test_basic_srif_filter(self):
        """Test full SRIF filter run."""
        n = 2
        x0 = np.zeros(n)
        P0 = np.eye(n) * 10.0

        R0 = np.linalg.inv(np.linalg.cholesky(P0)).T
        r0 = R0 @ x0

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(n) * 0.1
        H = np.array([[1, 0]])
        R_meas = np.array([[1.0]])

        measurements = [np.array([float(k)]) for k in range(5)]

        result = srif_filter(r0, R0, measurements, F, Q, H, R_meas)

        assert isinstance(result, SRIFResult)
        assert len(result.x_filt) == 5
        assert len(result.r_filt) == 5
        assert len(result.R_filt) == 5
        assert len(result.P_filt) == 5


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

    def test_rts_covariance_reduction(self):
        """Test that RTS smoother reduces covariance compared to filter."""
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2) * 10.0

        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.1
        H = np.array([[1, 0]])
        R = np.array([[1.0]])

        # Generate measurements following the CV model
        n_steps = 20
        measurements = [
            np.array([float(k) + np.random.randn() * 0.5]) for k in range(n_steps)
        ]

        result = rts_smoother(x0, P0, measurements, F, Q, H, R)

        # Smoothed covariances should generally be smaller (except at endpoints)
        # Check average trace reduction for interior points
        avg_trace_filt = np.mean([np.trace(result.P_filt[k]) for k in range(5, 15)])
        avg_trace_smooth = np.mean([np.trace(result.P_smooth[k]) for k in range(5, 15)])

        # Smoothing should reduce average covariance
        assert avg_trace_smooth < avg_trace_filt
