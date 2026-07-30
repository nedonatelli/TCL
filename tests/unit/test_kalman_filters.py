"""
Tests for Kalman filter implementations.

Tests cover:
- Linear Kalman filter (predict, update, smooth)
- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- Cubature Kalman filter (CKF)
- Information filter
- Sigma point generation
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman import (  # Linear KF; EKF; UKF; CKF
    KalmanPrediction,
    KalmanState,
    KalmanUpdate,
    SigmaPoints,
    ckf_predict,
    ckf_spherical_cubature_points,
    ckf_update,
    ekf_predict,
    ekf_predict_auto,
    ekf_update,
    ekf_update_auto,
    information_filter_predict,
    information_filter_update,
    iterated_ekf_update,
    kf_predict,
    kf_predict_update,
    kf_smooth,
    kf_update,
    numerical_jacobian,
    sigma_points_julier,
    sigma_points_merwe,
    ukf_predict,
    ukf_update,
    unscented_transform,
)


class TestLinearKalmanPredict:
    """Tests for linear Kalman filter prediction."""

    def test_kf_predict_basic(self):
        """Test basic KF prediction."""
        x = np.array([0.0, 1.0])  # position=0, velocity=1
        P = np.eye(2) * 0.1
        F = np.array([[1, 1], [0, 1]])  # CV model, T=1
        Q = np.array([[0.25, 0.5], [0.5, 1.0]])

        pred = kf_predict(x, P, F, Q)

        # State should propagate
        np.testing.assert_allclose(pred.x, [1.0, 1.0])
        # Covariance should be valid
        assert pred.P.shape == (2, 2)
        np.testing.assert_allclose(pred.P, pred.P.T)  # Symmetric

    def test_kf_predict_with_control(self):
        """Test KF prediction with control input."""
        x = np.array([0.0, 0.0])
        P = np.eye(2) * 0.1
        F = np.eye(2)
        Q = np.eye(2) * 0.01
        B = np.array([[0.5], [1.0]])  # Control input matrix
        u = np.array([2.0])  # Control input

        pred = kf_predict(x, P, F, Q, B, u)

        # Control should affect state
        np.testing.assert_allclose(pred.x, [1.0, 2.0])

    def test_kf_predict_preserves_positive_definite(self):
        """Test that prediction preserves positive definite covariance."""
        x = np.array([0.0, 1.0, 2.0])
        P = np.diag([1.0, 2.0, 3.0])
        F = np.eye(3)
        Q = np.eye(3) * 0.1

        pred = kf_predict(x, P, F, Q)

        eigenvalues = np.linalg.eigvalsh(pred.P)
        assert np.all(eigenvalues > 0)


class TestLinearKalmanUpdate:
    """Tests for linear Kalman filter update."""

    def test_kf_update_basic(self):
        """Test basic KF update."""
        x = np.array([1.0, 1.0])
        P = np.array([[0.35, 0.5], [0.5, 1.1]])
        z = np.array([1.2])  # position measurement
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        upd = kf_update(x, P, z, H, R)

        # State should be updated
        assert upd.x is not None
        # Innovation should be z - H@x
        expected_innovation = z[0] - x[0]
        np.testing.assert_allclose(upd.y, [expected_innovation], atol=1e-10)
        # Likelihood should be positive
        assert upd.likelihood > 0

    def test_kf_update_reduces_uncertainty(self):
        """Test that update reduces covariance (trace)."""
        x = np.array([0.0, 0.0])
        P = np.eye(2) * 10.0
        z = np.array([0.1])
        H = np.array([[1, 0]])
        R = np.array([[1.0]])

        upd = kf_update(x, P, z, H, R)

        # Updated trace should be smaller
        assert np.trace(upd.P) < np.trace(P)

    def test_kf_update_preserves_symmetry(self):
        """Test that update preserves covariance symmetry."""
        x = np.array([1.0, 2.0, 3.0])
        P = np.diag([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0])
        H = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.eye(2) * 0.1

        upd = kf_update(x, P, z, H, R)

        np.testing.assert_allclose(upd.P, upd.P.T, atol=1e-10)


class TestKalmanPredictUpdate:
    """Tests for combined predict-update."""

    def test_kf_predict_update_equivalence(self):
        """Test combined function equals separate calls."""
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        z = np.array([0.5])
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        # Combined
        result = kf_predict_update(x, P, z, F, Q, H, R)

        # Separate
        pred = kf_predict(x, P, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)

        np.testing.assert_allclose(result.x, upd.x)
        np.testing.assert_allclose(result.P, upd.P)


class TestKalmanSmooth:
    """Tests for RTS smoother."""

    def test_kf_smooth_basic(self):
        """Test basic RTS smoothing step."""
        x_filt = np.array([0.0, 1.0])
        P_filt = np.eye(2)
        x_pred = np.array([1.0, 1.0])
        P_pred = np.eye(2) * 1.1
        x_smooth_next = np.array([0.9, 1.0])
        P_smooth_next = np.eye(2) * 0.8
        F = np.array([[1, 1], [0, 1]])

        x_smooth, P_smooth = kf_smooth(
            x_filt, P_filt, x_pred, P_pred, x_smooth_next, P_smooth_next, F
        )

        # Smoothed state should be different from filtered
        assert x_smooth is not None
        assert P_smooth is not None
        # Smoothed covariance should be symmetric
        np.testing.assert_allclose(P_smooth, P_smooth.T, atol=1e-10)


class TestInformationFilter:
    """Tests for information filter."""

    def test_information_filter_predict(self):
        """Test information filter prediction."""
        # Start with state space representation
        x = np.array([1.0, 2.0])
        P = np.eye(2)

        # Convert to information form
        Y = np.linalg.inv(P)
        y = Y @ x

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.1

        y_pred, Y_pred = information_filter_predict(y, Y, F, Q)

        # Convert back to state space for comparison
        P_pred = np.linalg.inv(Y_pred)
        x_pred = P_pred @ y_pred

        # Compare with standard KF predict
        pred = kf_predict(x, P, F, Q)

        np.testing.assert_allclose(x_pred, pred.x, rtol=1e-6)
        np.testing.assert_allclose(P_pred, pred.P, rtol=1e-6)

    def test_information_filter_update(self):
        """Test information filter update is additive."""
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        Y = np.linalg.inv(P)
        y = Y @ x

        z = np.array([1.1])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        y_upd, Y_upd = information_filter_update(y, Y, z, H, R)

        # Information should increase
        assert np.trace(Y_upd) > np.trace(Y)


class TestExtendedKalmanFilter:
    """Tests for Extended Kalman Filter."""

    def test_ekf_predict_linear_case(self):
        """Test EKF on linear system matches standard KF."""
        x = np.array([0.0, 1.0])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01

        # Linear dynamics function
        def f(x):
            return F @ x

        # EKF predict
        ekf_pred = ekf_predict(x, P, f, F, Q)

        # Standard KF predict
        kf_pred = kf_predict(x, P, F, Q)

        np.testing.assert_allclose(ekf_pred.x, kf_pred.x)
        np.testing.assert_allclose(ekf_pred.P, kf_pred.P)

    def test_ekf_update_linear_case(self):
        """Test EKF update on linear system matches standard KF."""
        x = np.array([1.0, 1.0])
        P = np.eye(2)
        z = np.array([1.1])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        # Linear measurement function
        def h(x):
            return H @ x

        # EKF update
        ekf_upd = ekf_update(x, P, z, h, H, R)

        # Standard KF update
        kf_upd = kf_update(x, P, z, H, R)

        np.testing.assert_allclose(ekf_upd.x, kf_upd.x)
        np.testing.assert_allclose(ekf_upd.P, kf_upd.P, atol=1e-10)


class TestNumericalJacobian:
    """Tests for numerical Jacobian computation."""

    def test_numerical_jacobian_linear(self):
        """Test numerical Jacobian on linear function."""
        A = np.array([[1, 2], [3, 4]])

        def f(x):
            return A @ x

        x = np.array([1.0, 2.0])
        J = numerical_jacobian(f, x)

        np.testing.assert_allclose(J, A, rtol=1e-5)

    def test_numerical_jacobian_nonlinear(self):
        """Test numerical Jacobian on nonlinear function."""

        def f(x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        x = np.array([2.0, 3.0])
        J = numerical_jacobian(f, x)

        # Analytical Jacobian: [[2*x[0], 0], [x[1], x[0]]]
        expected = np.array([[4.0, 0.0], [3.0, 2.0]])

        np.testing.assert_allclose(J, expected, rtol=1e-5)


class TestEKFAuto:
    """Tests for automatic Jacobian EKF functions."""

    def test_ekf_predict_auto(self):
        """Test EKF prediction with automatic Jacobian."""
        x = np.array([0.0, 1.0])
        P = np.eye(2)
        Q = np.eye(2) * 0.01

        # Simple linear function
        def f(x):
            return np.array([x[0] + x[1], x[1]])

        pred = ekf_predict_auto(x, P, f, Q)

        np.testing.assert_allclose(pred.x, [1.0, 1.0])

    def test_ekf_update_auto(self):
        """Test EKF update with automatic Jacobian."""
        x = np.array([1.0, 1.0])
        P = np.eye(2)
        z = np.array([1.0])
        R = np.array([[0.1]])

        def h(x):
            return np.array([x[0]])

        upd = ekf_update_auto(x, P, z, h, R)

        assert upd.x is not None
        assert upd.likelihood > 0


class TestIteratedEKF:
    """Tests for iterated EKF."""

    def test_iterated_ekf_converges(self):
        """Test IEKF converges on linear system."""
        x = np.array([1.0, 1.0])
        P = np.eye(2)
        z = np.array([1.1])
        R = np.array([[0.1]])

        def h(x):
            return np.array([x[0]])

        def H_func(x):
            return np.array([[1.0, 0.0]])

        upd = iterated_ekf_update(x, P, z, h, H_func, R, max_iter=10)

        # Should match standard EKF for linear case
        ekf_upd = ekf_update(x, P, z, h, np.array([[1.0, 0.0]]), R)

        np.testing.assert_allclose(upd.x, ekf_upd.x, atol=1e-5)


class TestSigmaPoints:
    """Tests for sigma point generation."""

    def test_sigma_points_merwe_shape(self):
        """Test sigma point shape."""
        x = np.array([0.0, 0.0])
        P = np.eye(2)
        sp = sigma_points_merwe(x, P)

        # 2n+1 sigma points for n=2
        assert sp.points.shape == (5, 2)
        assert len(sp.Wm) == 5
        assert len(sp.Wc) == 5

    def test_sigma_points_weights_sum(self):
        """Test sigma point weights sum to 1."""
        x = np.array([0.0, 0.0, 0.0])
        P = np.eye(3)
        sp = sigma_points_merwe(x, P)

        np.testing.assert_allclose(np.sum(sp.Wm), 1.0, rtol=1e-10)

    def test_sigma_points_julier_shape(self):
        """Test Julier sigma point shape."""
        x = np.array([0.0, 0.0])
        P = np.eye(2)
        sp = sigma_points_julier(x, P, kappa=1.0)

        assert sp.points.shape == (5, 2)

    def test_sigma_points_preserve_mean(self):
        """Test sigma points preserve mean."""
        x = np.array([1.0, 2.0])
        P = np.diag([0.5, 1.0])
        sp = sigma_points_merwe(x, P)

        # Weighted mean should equal x
        mean = np.sum(sp.Wm[:, np.newaxis] * sp.points, axis=0)
        np.testing.assert_allclose(mean, x, rtol=1e-10)


class TestUnscentedTransform:
    """Tests for unscented transform."""

    def test_unscented_transform_identity(self):
        """Test UT with identity function."""
        x = np.array([1.0, 2.0])
        P = np.diag([0.1, 0.2])
        sp = sigma_points_merwe(x, P)

        mean, cov = unscented_transform(sp.points, sp.Wm, sp.Wc)

        np.testing.assert_allclose(mean, x, rtol=1e-10)
        # Use atol for near-zero off-diagonal elements
        np.testing.assert_allclose(cov, P, rtol=1e-6, atol=1e-15)


class TestUnscentedKalmanFilter:
    """Tests for Unscented Kalman Filter."""

    def test_ukf_predict_linear(self):
        """Test UKF prediction on linear system."""
        x = np.array([0.0, 1.0])
        P = np.eye(2)
        Q = np.eye(2) * 0.01

        def f(x):
            return np.array([x[0] + x[1], x[1]])

        pred = ukf_predict(x, P, f, Q)

        np.testing.assert_allclose(pred.x, [1.0, 1.0], atol=1e-6)

    def test_ukf_update_linear(self):
        """Test UKF update on linear system."""
        x = np.array([1.0, 1.0])
        P = np.eye(2)
        z = np.array([1.0])
        R = np.array([[0.1]])

        def h(x):
            return np.array([x[0]])

        upd = ukf_update(x, P, z, h, R)

        # Should be close to standard KF for linear case
        H = np.array([[1, 0]])
        kf_upd = kf_update(x, P, z, H, R)

        np.testing.assert_allclose(upd.x, kf_upd.x, atol=0.1)

    def test_ukf_preserves_positive_definite(self):
        """Test UKF preserves positive definite covariance."""
        x = np.array([0.0, 1.0, 2.0])
        P = np.diag([1.0, 2.0, 3.0])
        Q = np.eye(3) * 0.01

        def f(x):
            return x.copy()

        pred = ukf_predict(x, P, f, Q)

        eigenvalues = np.linalg.eigvalsh(pred.P)
        assert np.all(eigenvalues > 0)


class TestCubatureKalmanFilter:
    """Tests for Cubature Kalman Filter."""

    def test_ckf_cubature_points_shape(self):
        """Test cubature point generation."""
        n = 3
        points, weights = ckf_spherical_cubature_points(n)

        assert points.shape == (2 * n, n)
        assert len(weights) == 2 * n
        np.testing.assert_allclose(np.sum(weights), 1.0)

    def test_ckf_predict_linear(self):
        """Test CKF prediction on linear system."""
        x = np.array([0.0, 1.0])
        P = np.eye(2)
        Q = np.eye(2) * 0.01

        def f(x):
            return np.array([x[0] + x[1], x[1]])

        pred = ckf_predict(x, P, f, Q)

        np.testing.assert_allclose(pred.x, [1.0, 1.0], atol=1e-6)

    def test_ckf_update_linear(self):
        """Test CKF update on linear system."""
        x = np.array([1.0, 1.0])
        P = np.eye(2)
        z = np.array([1.0])
        R = np.array([[0.1]])

        def h(x):
            return np.array([x[0]])

        upd = ckf_update(x, P, z, h, R)

        assert upd.x is not None
        assert upd.likelihood > 0


class TestFilterComparison:
    """Compare different filter implementations on same problem."""

    @pytest.fixture
    def linear_system(self):
        """Create a simple linear tracking system."""
        return {
            "x": np.array([0.0, 1.0]),
            "P": np.eye(2) * 0.1,
            "F": np.array([[1, 0.1], [0, 1]]),
            "Q": np.eye(2) * 0.01,
            "H": np.array([[1, 0]]),
            "R": np.array([[0.1]]),
            "z": np.array([0.15]),
        }

    def test_filters_agree_on_linear_system(self, linear_system):
        """Test that all filters agree on linear system."""
        x = linear_system["x"]
        P = linear_system["P"]
        F = linear_system["F"]
        Q = linear_system["Q"]
        H = linear_system["H"]
        R = linear_system["R"]
        z = linear_system["z"]

        # Linear functions
        def f(x):
            return F @ x

        def h(x):
            return H @ x

        # Standard KF
        kf_pred = kf_predict(x, P, F, Q)
        kf_upd = kf_update(kf_pred.x, kf_pred.P, z, H, R)

        # EKF
        ekf_pred = ekf_predict(x, P, f, F, Q)
        ekf_upd = ekf_update(ekf_pred.x, ekf_pred.P, z, h, H, R)

        # UKF
        ukf_pred = ukf_predict(x, P, f, Q)
        ukf_upd = ukf_update(ukf_pred.x, ukf_pred.P, z, h, R)

        # CKF
        ckf_pred = ckf_predict(x, P, f, Q)
        ckf_upd = ckf_update(ckf_pred.x, ckf_pred.P, z, h, R)

        # All should be close for linear system
        np.testing.assert_allclose(kf_upd.x, ekf_upd.x, atol=1e-6)
        np.testing.assert_allclose(kf_upd.x, ukf_upd.x, atol=0.1)
        np.testing.assert_allclose(kf_upd.x, ckf_upd.x, atol=0.1)


class TestNamedTuples:
    """Tests for NamedTuple classes."""

    def test_kalman_state(self):
        """Test KalmanState NamedTuple."""
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        state = KalmanState(x=x, P=P)

        np.testing.assert_array_equal(state.x, x)
        np.testing.assert_array_equal(state.P, P)

    def test_kalman_prediction(self):
        """Test KalmanPrediction NamedTuple."""
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        pred = KalmanPrediction(x=x, P=P)

        assert pred.x is not None
        assert pred.P is not None

    def test_kalman_update(self):
        """Test KalmanUpdate NamedTuple."""
        upd = KalmanUpdate(
            x=np.array([1.0]),
            P=np.eye(1),
            y=np.array([0.1]),
            S=np.eye(1),
            K=np.array([[0.5]]),
            likelihood=0.5,
        )

        assert upd.likelihood == 0.5
        assert len(upd.y) == 1

    def test_sigma_points(self):
        """Test SigmaPoints NamedTuple."""
        sp = SigmaPoints(
            points=np.zeros((5, 2)),
            Wm=np.ones(5) / 5,
            Wc=np.ones(5) / 5,
        )

        assert sp.points.shape == (5, 2)
        np.testing.assert_allclose(np.sum(sp.Wm), 1.0)
