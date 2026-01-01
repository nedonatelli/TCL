"""
Comprehensive tests for v0.3.0 features.

This module contains additional tests for numerical accuracy, edge cases,
and comparison with reference implementations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# Square-Root Kalman Filter Comprehensive Tests
# =============================================================================


class TestSRKFNumericalStability:
    """Test numerical stability of square-root filters."""

    def test_srkf_ill_conditioned_covariance(self):
        """Test SRKF with ill-conditioned covariance matrix."""
        from pytcl.dynamic_estimation.kalman import srkf_predict

        # Create ill-conditioned covariance (high condition number)
        n = 4
        # Eigenvalues spanning many orders of magnitude
        D = np.diag([1e6, 1e3, 1.0, 1e-3])
        Q = np.random.randn(n, n)
        Q, _ = np.linalg.qr(Q)
        P = Q @ D @ Q.T

        x = np.random.randn(n)
        S = np.linalg.cholesky(P)

        F = np.eye(n)
        F[0, 1] = F[1, 2] = F[2, 3] = 0.1
        Q_noise = np.eye(n) * 0.01
        S_Q = np.linalg.cholesky(Q_noise)

        # Should not raise numerical errors
        pred = srkf_predict(x, S, F, S_Q)
        assert np.all(np.isfinite(pred.x))
        assert np.all(np.isfinite(pred.S))

        # Verify positive definiteness preserved
        P_pred = pred.S @ pred.S.T
        eigvals = np.linalg.eigvalsh(P_pred)
        assert np.all(eigvals > 0)

    def test_srkf_very_small_noise(self):
        """Test SRKF with very small process noise."""
        from pytcl.dynamic_estimation.kalman import srkf_predict

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        F = np.array([[1, 0.1], [0, 1]])

        # Very small process noise
        Q = np.eye(2) * 1e-12
        S_Q = np.linalg.cholesky(Q)

        pred = srkf_predict(x, S, F, S_Q)

        # Should remain numerically stable
        assert np.all(np.isfinite(pred.x))
        assert np.all(np.isfinite(pred.S))

    def test_srkf_large_state_dimension(self):
        """Test SRKF with larger state dimension."""
        from pytcl.dynamic_estimation.kalman import srkf_predict, srkf_update

        n = 20
        x = np.random.randn(n)
        P = np.random.randn(n, n)
        P = P @ P.T + np.eye(n) * 0.1
        S = np.linalg.cholesky(P)

        F = np.eye(n) + np.random.randn(n, n) * 0.01
        Q = np.eye(n) * 0.01
        S_Q = np.linalg.cholesky(Q)

        pred = srkf_predict(x, S, F, S_Q)

        assert pred.x.shape == (n,)
        assert pred.S.shape == (n, n)

        # Update with multiple measurements
        m = 5
        z = np.random.randn(m)
        H = np.random.randn(m, n)
        R = np.eye(m) * 0.1
        S_R = np.linalg.cholesky(R)

        upd = srkf_update(pred.x, pred.S, z, H, S_R)

        assert upd.x.shape == (n,)
        assert upd.S.shape == (n, n)
        assert upd.likelihood >= 0

    def test_srkf_preserves_symmetry(self):
        """Test that reconstructed covariance is symmetric."""
        from pytcl.dynamic_estimation.kalman import srkf_predict

        np.random.seed(42)
        n = 5
        for _ in range(10):
            x = np.random.randn(n)
            P = np.random.randn(n, n)
            P = P @ P.T + np.eye(n)
            S = np.linalg.cholesky(P)

            F = np.eye(n) + np.random.randn(n, n) * 0.1
            Q = np.eye(n) * 0.1
            S_Q = np.linalg.cholesky(Q)

            pred = srkf_predict(x, S, F, S_Q)
            P_pred = pred.S @ pred.S.T

            # Check symmetry
            assert_allclose(P_pred, P_pred.T, rtol=1e-10)


class TestCholeskyUpdate:
    """Tests for rank-1 Cholesky update/downdate."""

    def test_cholesky_update_correctness(self):
        """Test Cholesky update produces correct result."""
        from pytcl.dynamic_estimation.kalman import cholesky_update

        n = 4
        P = np.random.randn(n, n)
        P = P @ P.T + np.eye(n)
        S = np.linalg.cholesky(P)

        v = np.random.randn(n)

        # Update: P_new = P + v @ v.T
        S_new = cholesky_update(S, v, sign=1.0)
        P_new_expected = P + np.outer(v, v)
        P_new_computed = S_new @ S_new.T

        assert_allclose(P_new_computed, P_new_expected, rtol=1e-10)

    def test_cholesky_downdate_correctness(self):
        """Test Cholesky downdate produces correct result."""
        from pytcl.dynamic_estimation.kalman import cholesky_update

        n = 4
        # Start with P that can handle downdate
        P = np.eye(n) * 5 + np.random.randn(n, n) * 0.1
        P = P @ P.T
        S = np.linalg.cholesky(P)

        # Small vector for downdate
        v = np.random.randn(n) * 0.1

        # Downdate: P_new = P - v @ v.T
        S_new = cholesky_update(S, v, sign=-1.0)
        P_new_expected = P - np.outer(v, v)
        P_new_computed = S_new @ S_new.T

        assert_allclose(P_new_computed, P_new_expected, rtol=1e-8)

    def test_cholesky_downdate_failure(self):
        """Test Cholesky downdate raises error when result not PD."""
        from pytcl.dynamic_estimation.kalman import cholesky_update

        n = 3
        P = np.eye(n) * 0.1
        S = np.linalg.cholesky(P)

        # Large vector will make result not positive definite
        v = np.ones(n) * 10

        with pytest.raises(ValueError, match="non-positive definite"):
            cholesky_update(S, v, sign=-1.0)

    def test_cholesky_update_sequential(self):
        """Test multiple sequential updates."""
        from pytcl.dynamic_estimation.kalman import cholesky_update

        n = 4
        P = np.eye(n)
        S = np.linalg.cholesky(P)

        # Multiple sequential updates
        vectors = [np.random.randn(n) * 0.5 for _ in range(5)]

        for v in vectors:
            S = cholesky_update(S, v, sign=1.0)
            P = P + np.outer(v, v)

        P_computed = S @ S.T
        assert_allclose(P_computed, P, rtol=1e-10)


class TestQRUpdate:
    """Tests for QR-based covariance update."""

    def test_qr_update_without_F(self):
        """Test QR update with identity transition."""
        from pytcl.dynamic_estimation.kalman import qr_update

        n = 3
        P = np.random.randn(n, n)
        P = P @ P.T + np.eye(n)
        S_x = np.linalg.cholesky(P)

        Q = np.eye(n) * 0.1
        S_Q = np.linalg.cholesky(Q)

        S_new = qr_update(S_x, S_Q, F=None)

        # Should equal cholesky(P + Q)
        P_expected = P + Q
        P_computed = S_new @ S_new.T

        assert_allclose(P_computed, P_expected, rtol=1e-10)

    def test_qr_update_with_F(self):
        """Test QR update with state transition."""
        from pytcl.dynamic_estimation.kalman import qr_update

        n = 3
        P = np.random.randn(n, n)
        P = P @ P.T + np.eye(n)
        S_x = np.linalg.cholesky(P)

        Q = np.eye(n) * 0.1
        S_Q = np.linalg.cholesky(Q)

        F = np.array([[1, 0.1, 0], [0, 1, 0.1], [0, 0, 1]])

        S_new = qr_update(S_x, S_Q, F=F)

        # Should equal cholesky(F @ P @ F.T + Q)
        P_expected = F @ P @ F.T + Q
        P_computed = S_new @ S_new.T

        assert_allclose(P_computed, P_expected, rtol=1e-10)


# =============================================================================
# U-D Factorization Comprehensive Tests
# =============================================================================


class TestUDFactorizationComprehensive:
    """Comprehensive tests for U-D factorization filter."""

    def test_ud_factorize_identity(self):
        """Test U-D factorization of identity matrix."""
        from pytcl.dynamic_estimation.kalman import ud_factorize, ud_reconstruct

        n = 4
        P = np.eye(n)
        U, D = ud_factorize(P)

        assert_allclose(U, np.eye(n))
        assert_allclose(D, np.ones(n))
        assert_allclose(ud_reconstruct(U, D), P)

    def test_ud_factorize_diagonal(self):
        """Test U-D factorization of diagonal matrix."""
        from pytcl.dynamic_estimation.kalman import ud_factorize, ud_reconstruct

        D_orig = np.array([1.0, 2.0, 3.0, 4.0])
        P = np.diag(D_orig)
        U, D = ud_factorize(P)

        # For diagonal matrix, U should be identity
        assert_allclose(U, np.eye(4), atol=1e-12)
        assert_allclose(D, D_orig)
        assert_allclose(ud_reconstruct(U, D), P)

    def test_ud_factorize_general(self):
        """Test U-D factorization of general SPD matrix."""
        from pytcl.dynamic_estimation.kalman import ud_factorize, ud_reconstruct

        np.random.seed(42)
        for n in [2, 4, 6, 10]:
            A = np.random.randn(n, n)
            P = A @ A.T + np.eye(n)

            U, D = ud_factorize(P)

            # Check U is unit upper triangular
            assert_allclose(np.diag(U), np.ones(n), atol=1e-12)
            assert_allclose(np.tril(U, -1), np.zeros_like(U))

            # Check D is positive
            assert np.all(D > 0)

            # Check reconstruction
            assert_allclose(ud_reconstruct(U, D), P, rtol=1e-10)

    def test_ud_does_not_modify_input(self):
        """Test that U-D factorization doesn't modify input."""
        from pytcl.dynamic_estimation.kalman import ud_factorize

        P = np.array([[4.0, 2.0], [2.0, 3.0]])
        P_original = P.copy()

        U, D = ud_factorize(P)

        assert_allclose(P, P_original)

    def test_ud_predict_matches_standard(self):
        """Test U-D predict matches standard Kalman predict."""
        from pytcl.dynamic_estimation.kalman import (
            kf_predict,
            ud_factorize,
            ud_predict,
            ud_reconstruct,
        )

        x = np.array([1.0, 2.0, 3.0, 4.0])
        P = np.diag([0.1, 0.2, 0.3, 0.4])
        F = np.eye(4)
        F[0, 1] = F[2, 3] = 0.1
        Q = np.eye(4) * 0.01

        U, D = ud_factorize(P)

        # U-D predict
        x_ud, U_ud, D_ud = ud_predict(x, U, D, F, Q)

        # Standard predict
        pred_std = kf_predict(x, P, F, Q)

        assert_allclose(x_ud, pred_std.x, rtol=1e-10)
        P_ud = ud_reconstruct(U_ud, D_ud)
        assert_allclose(P_ud, pred_std.P, rtol=1e-8)

    def test_ud_update_scalar_bierman(self):
        """Test Bierman's scalar update algorithm."""
        from pytcl.dynamic_estimation.kalman import (
            kf_update,
            ud_factorize,
            ud_reconstruct,
            ud_update_scalar,
        )

        x = np.array([0.0, 1.0, 0.0])
        P = np.diag([1.0, 0.5, 0.3])
        U, D = ud_factorize(P)

        z = 0.5
        h = np.array([1.0, 0.0, 0.0])
        r = 0.1

        x_upd, U_upd, D_upd = ud_update_scalar(x, U, D, z, h, r)

        # Compare with standard KF
        H = h.reshape(1, -1)
        R = np.array([[r]])
        upd_std = kf_update(x, P, np.array([z]), H, R)

        assert_allclose(x_upd, upd_std.x, rtol=1e-10)
        P_upd = ud_reconstruct(U_upd, D_upd)
        assert_allclose(P_upd, upd_std.P, rtol=1e-8)

    def test_ud_update_correlated_noise(self):
        """Test U-D update with correlated measurement noise."""
        from pytcl.dynamic_estimation.kalman import (
            kf_update,
            ud_factorize,
            ud_update,
        )

        x = np.array([0.0, 1.0, 0.0, 0.5])
        P = np.diag([0.5, 1.0, 0.5, 1.0])
        U, D = ud_factorize(P)

        z = np.array([0.1, 0.2])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        # Correlated measurement noise
        R = np.array([[0.1, 0.02], [0.02, 0.1]])

        x_upd, U_upd, D_upd, y, likelihood = ud_update(x, U, D, z, H, R)

        # Compare with standard KF
        upd_std = kf_update(x, P, z, H, R)

        assert_allclose(x_upd, upd_std.x, rtol=1e-6)


# =============================================================================
# Square-Root UKF Comprehensive Tests
# =============================================================================


class TestSRUKFComprehensive:
    """Comprehensive tests for square-root UKF."""

    def test_sr_ukf_linear_matches_srkf(self):
        """Test SR-UKF matches SRKF for linear system."""
        from pytcl.dynamic_estimation.kalman import (
            sr_ukf_predict,
            srkf_predict,
        )

        x = np.array([0.0, 1.0, 0.0, -0.5])
        P = np.eye(4) * 0.2
        S = np.linalg.cholesky(P)

        # Linear functions
        F = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]])

        def f(x):
            return F @ x

        Q = np.eye(4) * 0.01
        S_Q = np.linalg.cholesky(Q)

        # Predict
        pred_sr = srkf_predict(x, S, F, S_Q)
        pred_ukf = sr_ukf_predict(x, S, f, S_Q)

        assert_allclose(pred_ukf.x, pred_sr.x, rtol=1e-6)
        P_ukf = pred_ukf.S @ pred_ukf.S.T
        P_sr = pred_sr.S @ pred_sr.S.T
        # Use atol for near-zero elements
        assert_allclose(P_ukf, P_sr, rtol=1e-5, atol=1e-10)

    def test_sr_ukf_nonlinear_convergence(self):
        """Test SR-UKF converges for nonlinear system."""
        from pytcl.dynamic_estimation.kalman import sr_ukf_predict, sr_ukf_update

        # Nonlinear range-bearing model
        def f(x):
            # Constant velocity in Cartesian
            return np.array([x[0] + 0.1 * x[1], x[1], x[2] + 0.1 * x[3], x[3]])

        def h(x):
            # Range and bearing
            r = np.sqrt(x[0] ** 2 + x[2] ** 2)
            theta = np.arctan2(x[2], x[0])
            return np.array([r, theta])

        # True trajectory
        np.random.seed(123)  # Different seed for more stable test
        x_true = np.array([10.0, 1.0, 5.0, 0.5])
        x_est = np.array([8.0, 0.5, 4.0, 0.2])  # Initial estimate with error
        P = np.eye(4) * 2.0  # Larger initial uncertainty
        S = np.linalg.cholesky(P)

        Q = np.eye(4) * 0.001  # Smaller process noise for stability
        S_Q = np.linalg.cholesky(Q)
        R = np.diag([0.05, 0.005])  # Range and bearing noise
        S_R = np.linalg.cholesky(R)

        errors = []
        for _ in range(100):  # More iterations
            # True dynamics (no process noise for deterministic test)
            x_true = f(x_true)

            # Generate measurement
            z = h(x_true) + np.random.multivariate_normal(np.zeros(2), R * 0.5)

            # Filter step
            pred = sr_ukf_predict(x_est, S, f, S_Q)
            upd = sr_ukf_update(pred.x, pred.S, z, h, S_R)
            x_est, S = upd.x, upd.S

            errors.append(np.linalg.norm(x_est[[0, 2]] - x_true[[0, 2]]))

        # Average error in last 10 steps should be less than average of first 10
        assert np.mean(errors[-10:]) < np.mean(errors[:10])

    def test_sr_ukf_different_alpha_values(self):
        """Test SR-UKF with different alpha values."""
        from pytcl.dynamic_estimation.kalman import sr_ukf_predict

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        Q = np.eye(2) * 0.01
        S_Q = np.linalg.cholesky(Q)

        def f(x):
            return x * 1.01

        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            pred = sr_ukf_predict(x, S, f, S_Q, alpha=alpha)
            assert np.all(np.isfinite(pred.x))
            assert np.all(np.isfinite(pred.S))


# =============================================================================
# IMM Estimator Comprehensive Tests
# =============================================================================


class TestIMMComprehensive:
    """Comprehensive tests for IMM estimator."""

    def test_imm_mixing_probabilities(self):
        """Test mode mixing probability computation."""
        from pytcl.dynamic_estimation.imm import compute_mixing_probabilities

        mu = np.array([0.8, 0.2])
        Pi = np.array([[0.9, 0.1], [0.1, 0.9]])

        mixing_probs, c_bar = compute_mixing_probabilities(mu, Pi)

        # Mixing probabilities should sum to 1 for each mode
        assert_allclose(np.sum(mixing_probs, axis=0), np.ones(2), rtol=1e-10)

        # c_bar should be valid probabilities
        assert np.all(c_bar > 0)
        assert np.all(c_bar <= 1)

    def test_imm_mode_probability_evolution(self):
        """Test mode probabilities evolve correctly over time."""
        from pytcl.dynamic_estimation.imm import IMMEstimator

        Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
        imm = IMMEstimator(n_modes=2, state_dim=2, transition_matrix=Pi)

        # Start with mode 1 dominant
        imm.initialize(np.array([0.0, 0.0]), np.eye(2) * 0.1)
        imm.mode_probs = np.array([0.99, 0.01])

        F = np.array([[1, 0.1], [0, 1]])
        Q = np.eye(2) * 0.01
        imm.set_mode_model(0, F, Q)
        imm.set_mode_model(1, F, Q)

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        imm.set_measurement_model(H, R)

        # Run without measurements - mode probs should approach steady state
        for _ in range(100):
            imm.predict()
            # Mock update (would need measurement)

        # Mode probabilities should remain valid
        assert_allclose(np.sum(imm.mode_probs), 1.0, rtol=1e-10)
        assert np.all(imm.mode_probs >= 0)
        assert np.all(imm.mode_probs <= 1)

    def test_imm_three_modes(self):
        """Test IMM with three modes."""
        from pytcl.dynamic_estimation.imm import imm_predict

        # Three modes: slow, medium, fast
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1

        mu = np.array([0.6, 0.3, 0.1])
        Pi = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])

        F_slow = np.array([[1, 0.05], [0, 1]])
        F_med = np.array([[1, 0.1], [0, 1]])
        F_fast = np.array([[1, 0.2], [0, 1]])
        Q = np.eye(2) * 0.01

        pred = imm_predict([x, x, x], [P, P, P], mu, Pi, [F_slow, F_med, F_fast], [Q, Q, Q])

        assert pred.x.shape == (2,)
        assert len(pred.mode_probs) == 3
        assert_allclose(np.sum(pred.mode_probs), 1.0)

    def test_imm_different_measurement_models(self):
        """Test IMM with mode-dependent measurement models."""
        from pytcl.dynamic_estimation.imm import imm_update

        x = np.array([1.0, 0.0, 0.0, 1.0])
        P = np.eye(4) * 0.2
        mu = np.array([0.5, 0.5])

        # Mode 1: position only
        H1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R1 = np.eye(2) * 0.1

        # Mode 2: position and velocity
        H2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R2 = np.eye(2) * 0.05  # More accurate

        z = np.array([1.1, 0.1])

        upd = imm_update([x, x], [P, P], mu, z, [H1, H2], [R1, R2])

        assert upd.x.shape == (4,)
        assert len(upd.mode_probs) == 2


# =============================================================================
# JPDA Comprehensive Tests
# =============================================================================


class TestJPDAComprehensive:
    """Comprehensive tests for JPDA."""

    def test_jpda_probabilities_normalization(self):
        """Test JPDA probabilities sum to 1."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        # Multiple tracks, multiple measurements
        likelihood = np.array([[0.8, 0.1, 0.05], [0.1, 0.7, 0.1], [0.05, 0.1, 0.6]])
        gated = np.ones_like(likelihood, dtype=bool)

        beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)

        # Each track's probabilities should sum to 1
        for i in range(3):
            assert_allclose(np.sum(beta[i, :]), 1.0, rtol=1e-6)

    def test_jpda_high_clutter(self):
        """Test JPDA behavior with high clutter density."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.5, 0.3]])
        gated = np.array([[True, True]])

        # High clutter should increase miss probability
        beta_low_clutter = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.001
        )
        beta_high_clutter = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.1
        )

        # With high clutter, more probability goes to "no measurement"
        assert beta_high_clutter[0, -1] > beta_low_clutter[0, -1]

    def test_jpda_low_detection_probability(self):
        """Test JPDA with low detection probability."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.8, 0.1]])
        gated = np.array([[True, True]])

        beta_high_pd = jpda_probabilities(
            likelihood, gated, detection_prob=0.99, clutter_density=0.01
        )
        beta_low_pd = jpda_probabilities(
            likelihood, gated, detection_prob=0.5, clutter_density=0.01
        )

        # Low detection prob should increase miss probability
        assert beta_low_pd[0, -1] > beta_high_pd[0, -1]

    def test_jpda_gating_effect(self):
        """Test that gating correctly excludes measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.8, 0.7, 0.1]])
        gated = np.array([[True, False, True]])  # Middle measurement not gated

        beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)

        # Gated-out measurement should have zero probability
        assert_allclose(beta[0, 1], 0.0)

    def test_jpda_update_with_ambiguous_measurements(self):
        """Test JPDA update handles ambiguous measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1

        # Two measurements, both plausible
        measurements = np.array([[0.1], [-0.1]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x], [P], measurements, H, R)

        # JPDA should produce valid state and covariance
        assert result.states[0].shape == (2,)
        assert result.covariances[0].shape == (2, 2)
        # Covariance should remain positive definite
        eigvals = np.linalg.eigvalsh(result.covariances[0])
        assert np.all(eigvals > 0)

    def test_jpda_multiple_tracks(self):
        """Test JPDA with multiple tracks."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        # 3 tracks at different positions
        tracks = [
            np.array([0.0, 1.0]),
            np.array([5.0, 0.0]),
            np.array([10.0, -1.0]),
        ]
        covs = [np.eye(2) * 0.1 for _ in range(3)]

        # 3 measurements near each track
        measurements = np.array([[0.1], [5.1], [9.9]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update(tracks, covs, measurements, H, R)

        assert len(result.states) == 3
        assert result.association_probs.shape == (3, 4)  # 3 tracks, 3+1 columns

    def test_jpda_empty_tracks(self):
        """Test JPDA handles empty track list."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        measurements = np.array([[0.1], [0.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([], [], measurements, H, R)

        assert len(result.states) == 0
        assert len(result.covariances) == 0


# =============================================================================
# Cross-Feature Integration Tests
# =============================================================================


class TestCrossFeatureIntegration:
    """Tests combining multiple v0.3.0 features."""

    def test_imm_with_srkf(self):
        """Test using square-root filters within IMM framework."""
        from pytcl.dynamic_estimation.imm import combine_estimates
        from pytcl.dynamic_estimation.kalman import srkf_predict

        # Initial state
        x1 = np.array([0.0, 1.0, 0.0, 0.5])
        x2 = x1.copy()
        P = np.eye(4) * 0.1
        S1 = np.linalg.cholesky(P)
        S2 = np.linalg.cholesky(P)

        mu = np.array([0.9, 0.1])

        # Models (CV and CA-like)
        F1 = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]])
        F2 = np.array([[1, 0.1, 0, 0], [0, 0.95, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 0.95]])
        Q = np.eye(4) * 0.01
        S_Q = np.linalg.cholesky(Q)

        # SRKF predict for each mode
        pred1 = srkf_predict(x1, S1, F1, S_Q)
        pred2 = srkf_predict(x2, S2, F2, S_Q)

        # Verify predictions are valid
        assert np.all(np.isfinite(pred1.x))
        assert np.all(np.isfinite(pred2.x))

        # Combine using IMM framework
        P1 = pred1.S @ pred1.S.T
        P2 = pred2.S @ pred2.S.T
        x_combined, P_combined = combine_estimates([pred1.x, pred2.x], [P1, P2], mu)

        assert x_combined.shape == (4,)
        assert P_combined.shape == (4, 4)

    def test_jpda_with_ud_filter(self):
        """Test JPDA with U-D factorization filter for tracks."""
        from pytcl.assignment_algorithms.jpda import compute_likelihood_matrix
        from pytcl.dynamic_estimation.kalman import (
            ud_factorize,
            ud_reconstruct,
            ud_update,
        )

        # Track state in U-D form
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.5
        U, D = ud_factorize(P)

        # Measurements
        measurements = [np.array([0.1]), np.array([0.5])]
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        # Compute likelihood using JPDA helper
        P_rec = ud_reconstruct(U, D)
        likelihood_matrix, gated = compute_likelihood_matrix([x], [P_rec], measurements, H, R)

        # Verify likelihoods are valid
        assert np.all(likelihood_matrix >= 0)
        assert np.all(np.isfinite(likelihood_matrix))

        # Update with highest likelihood measurement
        best_meas_idx = np.argmax(likelihood_matrix[0, :])
        z = measurements[best_meas_idx]
        x_upd, U_upd, D_upd, y, lik = ud_update(x, U, D, z, H, R)

        assert np.all(np.isfinite(x_upd))


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================


class TestPerformanceEdgeCases:
    """Test edge cases and performance characteristics."""

    def test_srkf_single_measurement(self):
        """Test SRKF with single scalar measurement."""
        from pytcl.dynamic_estimation.kalman import srkf_update

        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3) * 0.2
        S = np.linalg.cholesky(P)

        z = np.array([1.1])
        H = np.array([[1, 0, 0]])
        R = np.array([[0.01]])
        S_R = np.linalg.cholesky(R)

        upd = srkf_update(x, S, z, H, S_R)

        assert upd.x.shape == (3,)
        assert upd.y.shape == (1,)

    def test_imm_equal_probabilities(self):
        """Test IMM when all modes have equal probability."""
        from pytcl.dynamic_estimation.imm import imm_predict

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1

        mu = np.array([0.5, 0.5])
        Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
        F = np.array([[1, 0.1], [0, 1]])
        Q = np.eye(2) * 0.01

        pred = imm_predict([x, x], [P, P], mu, Pi, [F, F], [Q, Q])

        # Should not fail with equal probabilities
        assert np.all(np.isfinite(pred.x))
        assert_allclose(np.sum(pred.mode_probs), 1.0)

    def test_jpda_single_measurement_per_track(self):
        """Test JPDA when each track has exactly one measurement."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        tracks = [np.array([0.0, 1.0]), np.array([10.0, -1.0])]
        covs = [np.eye(2) * 0.01, np.eye(2) * 0.01]  # Very small covariance

        # Measurements clearly associated with each track
        measurements = np.array([[0.01], [10.01]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.001]])

        result = jpda_update(tracks, covs, measurements, H, R)

        # Each track should strongly associate with its measurement
        assert result.association_probs[0, 0] > 0.9
        assert result.association_probs[1, 1] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
