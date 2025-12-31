"""
Tests for v0.3.0 features: Square-root filters, IMM, and JPDA.
"""

import numpy as np
from numpy.testing import assert_allclose

# =============================================================================
# Square-Root Kalman Filter Tests
# =============================================================================


class TestSquareRootKalmanFilter:
    """Tests for square-root Kalman filter implementations."""

    def test_srkf_predict_basic(self):
        """Test basic SRKF prediction."""
        from pytcl.dynamic_estimation.kalman import srkf_predict

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.25, 0.25], [0.25, 1.0]])  # Must be positive definite
        S_Q = np.linalg.cholesky(Q)

        pred = srkf_predict(x, S, F, S_Q)

        assert pred.x.shape == (2,)
        assert pred.S.shape == (2, 2)
        # Predicted state should be F @ x
        assert_allclose(pred.x, F @ x)
        # Reconstructed covariance should be positive definite
        P_pred = pred.S @ pred.S.T
        eigvals = np.linalg.eigvalsh(P_pred)
        assert np.all(eigvals > 0)

    def test_srkf_update_basic(self):
        """Test basic SRKF update."""
        from pytcl.dynamic_estimation.kalman import srkf_update

        x = np.array([1.0, 1.0])
        P = np.array([[0.35, 0.5], [0.5, 1.1]])
        S = np.linalg.cholesky(P)
        z = np.array([1.2])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        S_R = np.linalg.cholesky(R)

        upd = srkf_update(x, S, z, H, S_R)

        assert upd.x.shape == (2,)
        assert upd.S.shape == (2, 2)
        assert upd.y.shape == (1,)
        assert upd.likelihood >= 0

    def test_srkf_matches_kf(self):
        """Test that SRKF gives same results as standard KF."""
        from pytcl.dynamic_estimation.kalman import (
            kf_predict,
            kf_update,
            srkf_predict,
            srkf_update,
        )

        np.random.seed(42)
        x = np.random.randn(4)
        P = np.random.randn(4, 4)
        P = P @ P.T + np.eye(4)  # Ensure positive definite
        S = np.linalg.cholesky(P)

        F = np.eye(4)
        F[0, 1] = F[2, 3] = 1.0
        Q = np.eye(4) * 0.1
        S_Q = np.linalg.cholesky(Q)

        # Predict
        pred_kf = kf_predict(x, P, F, Q)
        pred_sr = srkf_predict(x, S, F, S_Q)

        assert_allclose(pred_sr.x, pred_kf.x, rtol=1e-10)
        P_sr = pred_sr.S @ pred_sr.S.T
        assert_allclose(P_sr, pred_kf.P, rtol=1e-6)

        # Update
        z = np.array([1.0, 2.0])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.5
        S_R = np.linalg.cholesky(R)

        upd_kf = kf_update(pred_kf.x, pred_kf.P, z, H, R)
        upd_sr = srkf_update(pred_sr.x, pred_sr.S, z, H, S_R)

        assert_allclose(upd_sr.x, upd_kf.x, rtol=1e-6)
        P_upd_sr = upd_sr.S @ upd_sr.S.T
        assert_allclose(P_upd_sr, upd_kf.P, rtol=1e-5)


class TestUDFactorization:
    """Tests for U-D factorization filter."""

    def test_ud_factorize_basic(self):
        """Test U-D factorization."""
        from pytcl.dynamic_estimation.kalman import ud_factorize, ud_reconstruct

        P = np.array([[4.0, 2.0], [2.0, 3.0]])
        U, D = ud_factorize(P)

        # U should be unit upper triangular
        assert_allclose(np.diag(U), np.ones(2))
        assert_allclose(U[1, 0], 0.0)

        # D should be positive
        assert np.all(D > 0)

        # Reconstruction should match
        P_rec = ud_reconstruct(U, D)
        assert_allclose(P_rec, P, rtol=1e-10)

    def test_ud_update_scalar(self):
        """Test scalar U-D update."""
        from pytcl.dynamic_estimation.kalman import (
            ud_factorize,
            ud_reconstruct,
            ud_update_scalar,
        )

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.5
        U, D = ud_factorize(P)

        z = 0.1
        h = np.array([1.0, 0.0])
        r = 0.1

        x_upd, U_upd, D_upd = ud_update_scalar(x, U, D, z, h, r)

        assert x_upd.shape == (2,)
        P_upd = ud_reconstruct(U_upd, D_upd)
        # Updated covariance should be smaller
        assert np.trace(P_upd) < np.trace(P)

    def test_ud_update_vector(self):
        """Test vector U-D update."""
        from pytcl.dynamic_estimation.kalman import (
            kf_update,
            ud_factorize,
            ud_update,
        )

        x = np.array([0.0, 1.0, 0.0, -0.5])
        P = np.diag([0.5, 1.0, 0.5, 1.0])
        U, D = ud_factorize(P)

        z = np.array([0.1, 0.2])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1

        x_upd, U_upd, D_upd, y, likelihood = ud_update(x, U, D, z, H, R)

        # Compare with standard KF
        upd_kf = kf_update(x, P, z, H, R)
        assert_allclose(x_upd, upd_kf.x, rtol=1e-6)


# =============================================================================
# Square-Root UKF Tests
# =============================================================================


class TestSquareRootUKF:
    """Tests for square-root UKF."""

    def test_sr_ukf_predict(self):
        """Test SR-UKF prediction."""
        from pytcl.dynamic_estimation.kalman import sr_ukf_predict

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        Q = np.eye(2) * 0.01
        S_Q = np.linalg.cholesky(Q)

        def f(x):
            return np.array([x[0] + x[1], x[1]])

        pred = sr_ukf_predict(x, S, f, S_Q)

        assert pred.x.shape == (2,)
        assert pred.S.shape == (2, 2)
        # Should be lower triangular
        assert_allclose(pred.S, np.tril(pred.S))

    def test_sr_ukf_update(self):
        """Test SR-UKF update."""
        from pytcl.dynamic_estimation.kalman import sr_ukf_update

        x = np.array([1.0, 1.0])
        P = np.eye(2) * 0.2
        S = np.linalg.cholesky(P)
        R = np.array([[0.1]])
        S_R = np.linalg.cholesky(R)
        z = np.array([1.1])

        def h(x):
            return np.array([x[0]])

        upd = sr_ukf_update(x, S, z, h, S_R)

        assert upd.x.shape == (2,)
        assert upd.S.shape == (2, 2)
        assert upd.likelihood > 0


# =============================================================================
# IMM Estimator Tests
# =============================================================================


class TestIMMEstimator:
    """Tests for Interacting Multiple Model estimator."""

    def test_imm_predict_basic(self):
        """Test basic IMM prediction."""
        from pytcl.dynamic_estimation.imm import imm_predict

        x1 = np.array([0.0, 1.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0, 0.0])
        P = np.eye(4) * 0.1

        mu = np.array([0.9, 0.1])
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])

        F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01

        pred = imm_predict([x1, x2], [P, P], mu, Pi, [F, F], [Q, Q])

        assert pred.x.shape == (4,)
        assert pred.P.shape == (4, 4)
        assert len(pred.mode_states) == 2
        assert len(pred.mode_probs) == 2
        assert_allclose(np.sum(pred.mode_probs), 1.0)

    def test_imm_update_basic(self):
        """Test basic IMM update."""
        from pytcl.dynamic_estimation.imm import imm_update

        x1 = np.array([1.0, 1.0, 0.0, 0.0])
        x2 = np.array([1.0, 1.0, 0.0, 0.0])
        P = np.eye(4) * 0.2

        mu = np.array([0.9, 0.1])
        z = np.array([1.1, 0.1])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1

        upd = imm_update([x1, x2], [P, P], mu, z, [H, H], [R, R])

        assert upd.x.shape == (4,)
        assert upd.P.shape == (4, 4)
        assert len(upd.mode_probs) == 2
        assert_allclose(np.sum(upd.mode_probs), 1.0)

    def test_imm_estimator_class(self):
        """Test IMMEstimator class."""
        from pytcl.dynamic_estimation.imm import IMMEstimator

        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        x0 = np.array([0.0, 1.0, 0.0, 0.0])
        P0 = np.eye(4) * 0.1
        imm.initialize(x0, P0)

        F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        imm.set_mode_model(0, F, Q)
        imm.set_mode_model(1, F, Q)

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1
        imm.set_measurement_model(H, R)

        # Predict and update
        pred = imm.predict()
        assert pred.x.shape == (4,)

        z = np.array([1.1, 0.1])
        upd = imm.update(z)
        assert upd.x.shape == (4,)


# =============================================================================
# JPDA Tests
# =============================================================================


class TestJPDA:
    """Tests for Joint Probabilistic Data Association."""

    def test_jpda_probabilities_single_track(self):
        """Test JPDA with single track."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        # Single track, two measurements
        likelihood = np.array([[0.8, 0.1]])
        gated = np.array([[True, True]])

        beta = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.01
        )

        assert beta.shape == (1, 3)  # 1 track, 2 meas + 1 for no-meas
        # Probabilities should sum to 1 for each track
        assert_allclose(np.sum(beta[0, :]), 1.0, rtol=1e-6)
        # Higher likelihood measurement should have higher probability
        assert beta[0, 0] > beta[0, 1]

    def test_jpda_update_basic(self):
        """Test basic JPDA update."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x1 = np.array([0.0, 1.0])
        x2 = np.array([5.0, -1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([[0.1], [5.2], [10.0]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x1, x2], [P, P], measurements, H, R)

        assert len(result.states) == 2
        assert len(result.covariances) == 2
        assert result.association_probs.shape == (2, 4)  # 2 tracks, 3 meas + no-meas

    def test_jpda_no_measurements(self):
        """Test JPDA with no measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([]).reshape(0, 1)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x], [P], measurements, H, R)

        # With no measurements, state should be unchanged
        assert_allclose(result.states[0], x)

    def test_jpda_result_convenience(self):
        """Test JPDA convenience function."""
        from pytcl.assignment_algorithms import jpda

        x1 = np.array([0.0, 1.0])
        x2 = np.array([5.0, -1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([[0.1], [5.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda([x1, x2], [P, P], measurements, H, R)

        assert result.association_probs.shape == (2, 3)
        assert len(result.marginal_probs) == 2
        assert result.likelihood_matrix.shape == (2, 2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining new features."""

    def test_srkf_tracking_sequence(self):
        """Test SRKF over multiple time steps."""
        from pytcl.dynamic_estimation.kalman import srkf_predict, srkf_update

        # Initial state
        x = np.array([0.0, 1.0, 0.0, 0.5])
        P = np.eye(4) * 0.1
        S = np.linalg.cholesky(P)

        # Model
        T = 0.1
        F = np.array(
            [
                [1, T, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, T],
                [0, 0, 0, 1],
            ]
        )
        Q = np.eye(4) * 0.01
        S_Q = np.linalg.cholesky(Q)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1
        S_R = np.linalg.cholesky(R)

        # Generate measurements
        np.random.seed(42)
        true_state = x.copy()
        for _ in range(10):
            true_state = F @ true_state
            z = H @ true_state + np.random.randn(2) * 0.1

            pred = srkf_predict(x, S, F, S_Q)
            upd = srkf_update(pred.x, pred.S, z, H, S_R)
            x, S = upd.x, upd.S

        # Filter should track reasonably well
        assert np.linalg.norm(x[:2] - true_state[:2]) < 1.0

    def test_imm_mode_switching(self):
        """Test IMM can detect mode switching."""
        from pytcl.dynamic_estimation.imm import IMMEstimator

        Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
        imm = IMMEstimator(n_modes=2, state_dim=2, transition_matrix=Pi)

        # Mode 1: slow, Mode 2: fast
        F1 = np.array([[1, 0.1], [0, 1]])
        F2 = np.array([[1, 1.0], [0, 1]])
        Q = np.eye(2) * 0.01

        imm.initialize(np.array([0.0, 0.5]), np.eye(2) * 0.1)
        imm.set_mode_model(0, F1, Q)
        imm.set_mode_model(1, F2, Q)

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        imm.set_measurement_model(H, R)

        # Simulate fast movement (consistent with mode 2)
        np.random.seed(42)
        for i in range(20):
            z = np.array([i * 1.0 + np.random.randn() * 0.1])
            imm.predict_update(z)

        # Mode 2 should have higher probability after consistent fast movement
        assert imm.mode_probs[1] > imm.mode_probs[0]
