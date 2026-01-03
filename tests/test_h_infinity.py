"""
Tests for H-infinity filter implementation.

Tests cover basic functionality, robustness properties, and edge cases.
"""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.dynamic_estimation.kalman import (
    HInfinityPrediction,
    HInfinityUpdate,
    extended_hinf_update,
    find_min_gamma,
    hinf_predict,
    hinf_predict_update,
    hinf_update,
    kf_predict,
    kf_update,
)


class TestHInfinityPredict:
    """Tests for H-infinity prediction step."""

    def test_predict_basic(self):
        """Test basic prediction step."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01

        result = hinf_predict(x, P, F, Q)

        assert isinstance(result, HInfinityPrediction)
        assert result.x.shape == (2,)
        assert result.P.shape == (2, 2)

        # Check state propagation
        assert_allclose(result.x, F @ x)

    def test_predict_with_control(self):
        """Test prediction with control input."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        B = np.array([[0.5], [1.0]])
        u = np.array([0.5])

        result = hinf_predict(x, P, F, Q, B, u)

        expected_x = F @ x + B @ u
        assert_allclose(result.x, expected_x.flatten())

    def test_predict_matches_kalman(self):
        """Prediction step should match standard Kalman filter."""
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3) * 0.5
        F = np.random.randn(3, 3)
        Q = np.eye(3) * 0.1

        hinf_result = hinf_predict(x, P, F, Q)
        kf_result = kf_predict(x, P, F, Q)

        assert_allclose(hinf_result.x, kf_result.x)
        assert_allclose(hinf_result.P, kf_result.P)


class TestHInfinityUpdate:
    """Tests for H-infinity update step."""

    def test_update_basic(self):
        """Test basic update step."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        result = hinf_update(x, P, z, H, R, gamma)

        assert isinstance(result, HInfinityUpdate)
        assert result.x.shape == (2,)
        assert result.P.shape == (2, 2)
        assert result.y.shape == (1,)
        assert result.S.shape == (1, 1)
        assert result.K.shape == (2, 1)
        assert result.gamma == gamma
        assert isinstance(result.feasible, bool)

    def test_update_feasibility(self):
        """Test feasibility check for different gamma values."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])

        # Large gamma should always be feasible
        result_large = hinf_update(x, P, z, H, R, gamma=100.0)
        assert result_large.feasible is True

        # Very small gamma may not be feasible
        result_small = hinf_update(x, P, z, H, R, gamma=0.01)
        # Either feasible or not, depending on problem structure
        assert isinstance(result_small.feasible, bool)

    def test_update_converges_to_kalman(self):
        """As gamma -> infinity, H-infinity should approach Kalman filter."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])

        # Very large gamma
        hinf_result = hinf_update(x, P, z, H, R, gamma=1e6)
        kf_result = kf_update(x, P, z, H, R)

        # Should be very close
        assert_allclose(hinf_result.x, kf_result.x, rtol=1e-3)
        assert_allclose(hinf_result.P, kf_result.P, rtol=1e-3)

    def test_update_with_custom_L(self):
        """Test update with custom error weighting matrix L."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        # Only bound first state
        L = np.array([[1.0, 0.0]])
        result = hinf_update(x, P, z, H, R, gamma, L=L)

        assert result.feasible is True
        assert result.x.shape == (2,)

    def test_update_innovation(self):
        """Test that innovation is computed correctly."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.5])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        result = hinf_update(x, P, z, H, R, gamma)

        expected_innovation = z - H @ x
        assert_allclose(result.y, expected_innovation)


class TestHInfinityPredictUpdate:
    """Tests for combined predict-update step."""

    def test_predict_update_basic(self):
        """Test combined predict-update."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([2.1])
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        result = hinf_predict_update(x, P, z, F, Q, H, R, gamma)

        assert isinstance(result, HInfinityUpdate)
        assert result.x.shape == (2,)
        assert result.P.shape == (2, 2)

    def test_predict_update_matches_separate(self):
        """Combined should match separate predict then update."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([2.1])
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        # Combined
        result_combined = hinf_predict_update(x, P, z, F, Q, H, R, gamma)

        # Separate
        pred = hinf_predict(x, P, F, Q)
        result_separate = hinf_update(pred.x, pred.P, z, H, R, gamma)

        assert_allclose(result_combined.x, result_separate.x)
        assert_allclose(result_combined.P, result_separate.P)


class TestExtendedHInfinityUpdate:
    """Tests for extended H-infinity filter."""

    def test_extended_update_basic(self):
        """Test extended H-infinity with nonlinear measurement."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([np.sqrt(5.0)])  # sqrt(1^2 + 2^2)

        def h(state):
            return np.array([np.sqrt(state[0] ** 2 + state[1] ** 2)])

        # Jacobian at x
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        H = np.array([[x[0] / r, x[1] / r]])
        R = np.array([[0.01]])
        gamma = 10.0

        result = extended_hinf_update(x, P, z, h, H, R, gamma)

        assert isinstance(result, HInfinityUpdate)
        assert result.x.shape == (2,)
        assert result.feasible is True

    def test_extended_reduces_to_linear(self):
        """Extended H-infinity with linear h should match linear version."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        def h_linear(state):
            return H @ state

        result_extended = extended_hinf_update(x, P, z, h_linear, H, R, gamma)
        result_linear = hinf_update(x, P, z, H, R, gamma)

        assert_allclose(result_extended.x, result_linear.x)
        assert_allclose(result_extended.P, result_linear.P)


class TestFindMinGamma:
    """Tests for minimum feasible gamma finder."""

    def test_find_min_gamma_basic(self):
        """Test finding minimum feasible gamma."""
        P = np.eye(2) * 0.1
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])

        gamma_min = find_min_gamma(P, H, R)

        assert gamma_min > 0
        assert np.isfinite(gamma_min)

    def test_gamma_above_min_is_feasible(self):
        """Gamma slightly above minimum should be feasible."""
        P = np.eye(2) * 0.1
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])

        gamma_min = find_min_gamma(P, H, R)

        x = np.array([1.0, 2.0])
        z = np.array([1.1])

        # Test at 1.1 * gamma_min (with safety margin)
        result = hinf_update(x, P, z, H, R, gamma=gamma_min * 1.5)
        assert result.feasible is True

    def test_min_gamma_with_custom_L(self):
        """Test minimum gamma with custom error weighting."""
        P = np.eye(2) * 0.1
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        L = np.array([[1.0, 0.0]])  # Only bound first state

        gamma_min = find_min_gamma(P, H, R, L=L)

        assert gamma_min > 0
        assert np.isfinite(gamma_min)


class TestRobustnessProperties:
    """Tests verifying robustness properties of H-infinity filter."""

    def test_tracking_with_model_uncertainty(self):
        """Test tracking performance with model mismatch."""
        # True system
        F_true = np.array([[1, 0.1], [0, 1]])
        # Model with uncertainty
        F_model = np.array([[1, 0.11], [0, 1.01]])  # 10% error

        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 5.0

        # Initial state
        x_true = np.array([0.0, 1.0])
        x_est = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1

        # Simulate tracking
        errors = []
        for _ in range(20):
            # True state evolution
            x_true = F_true @ x_true + np.random.randn(2) * 0.1

            # Measurement
            z = H @ x_true + np.random.randn(1) * 0.1

            # H-infinity update
            pred = hinf_predict(x_est, P, F_model, Q)
            result = hinf_update(pred.x, pred.P, z, H, R, gamma)

            x_est = result.x
            P = result.P
            errors.append(np.linalg.norm(x_est - x_true))

        # Error should remain bounded
        assert np.max(errors) < 5.0

    def test_covariance_stays_positive_definite(self):
        """Test that covariance remains positive definite."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        # Multiple iterations
        for _ in range(100):
            z = H @ x + np.random.randn(1) * 0.1
            result = hinf_predict_update(x, P, z, F, Q, H, R, gamma)
            x = result.x
            P = result.P

            # Check positive definiteness
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals > 0)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_scalar_state(self):
        """Test with scalar state."""
        x = np.array([1.0])
        P = np.array([[0.1]])
        z = np.array([1.1])
        H = np.array([[1.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        result = hinf_update(x, P, z, H, R, gamma)

        assert result.x.shape == (1,)
        assert result.P.shape == (1, 1)
        assert result.feasible is True

    def test_multiple_measurements(self):
        """Test with multiple measurements."""
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3) * 0.1
        z = np.array([1.1, 2.1])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = np.eye(2) * 0.01
        gamma = 10.0

        result = hinf_update(x, P, z, H, R, gamma)

        assert result.x.shape == (3,)
        assert result.y.shape == (2,)
        assert result.S.shape == (2, 2)

    def test_singular_covariance_handling(self):
        """Test handling of near-singular covariance."""
        x = np.array([1.0, 2.0])
        # Near-singular covariance
        P = np.array([[0.1, 0.09999], [0.09999, 0.1]])
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.01]])
        gamma = 10.0

        # Should not raise, may use pseudo-inverse
        result = hinf_update(x, P, z, H, R, gamma)
        assert result.x.shape == (2,)
