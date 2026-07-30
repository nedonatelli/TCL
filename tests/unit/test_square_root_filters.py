"""
Tests for square-root Kalman filter implementations.

Tests cover:
- Square-root UKF (sr_ukf.py)
- Kalman matrix utilities (matrix_utils.py)
- U-D factorization filter (ud_filter.py)
- Square-root linear KF (square_root.py)
- IMM estimator (imm.py)
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.imm import (
    IMMEstimator,
    combine_estimates,
    compute_mixing_probabilities,
    imm_predict,
    imm_predict_update,
    imm_update,
    mix_states,
)
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_estimation.kalman.matrix_utils import (
    cholesky_update,
    compute_innovation_likelihood,
    compute_mahalanobis_distance,
    compute_matrix_sqrt,
    compute_merwe_weights,
    ensure_symmetric,
    qr_update,
)
from pytcl.dynamic_estimation.kalman.square_root import (
    srkf_predict,
    srkf_predict_update,
    srkf_update,
)
from pytcl.dynamic_estimation.kalman.sr_ukf import sr_ukf_predict, sr_ukf_update
from pytcl.dynamic_estimation.kalman.ud_filter import (
    ud_factorize,
    ud_predict,
    ud_reconstruct,
    ud_update,
    ud_update_scalar,
)

# =============================================================================
# Tests for Cholesky Update (matrix_utils.py)
# =============================================================================


class TestCholeskyUpdate:
    """Tests for Cholesky rank-1 update/downdate."""

    def test_cholesky_update_basic(self):
        """Test basic rank-1 update."""
        P = np.eye(2)
        S = np.linalg.cholesky(P)
        v = np.array([0.5, 0.5])

        S_updated = cholesky_update(S, v, sign=1.0)
        P_updated = S_updated @ S_updated.T

        expected = P + np.outer(v, v)
        np.testing.assert_allclose(P_updated, expected, rtol=1e-10)

    def test_cholesky_update_larger_matrix(self):
        """Test rank-1 update on larger matrix."""
        n = 4
        A = np.random.randn(n, n)
        P = A @ A.T + np.eye(n)  # Ensure positive definite
        S = np.linalg.cholesky(P)
        v = np.random.randn(n)

        S_updated = cholesky_update(S, v, sign=1.0)
        P_updated = S_updated @ S_updated.T

        expected = P + np.outer(v, v)
        np.testing.assert_allclose(P_updated, expected, rtol=1e-10)

    def test_cholesky_downdate_basic(self):
        """Test basic rank-1 downdate."""
        v = np.array([0.3, 0.3])
        P = np.eye(2) + 2 * np.outer(v, v)  # Ensure P - vv' is positive definite
        S = np.linalg.cholesky(P)

        S_downdated = cholesky_update(S, v, sign=-1.0)
        P_downdated = S_downdated @ S_downdated.T

        expected = P - np.outer(v, v)
        np.testing.assert_allclose(P_downdated, expected, rtol=1e-10)

    def test_cholesky_downdate_fails_non_positive_definite(self):
        """Test that downdate fails for non-positive definite result."""
        P = np.eye(2)
        S = np.linalg.cholesky(P)
        v = np.array([2.0, 0.0])  # This would make P - vv' not positive definite

        with pytest.raises(ValueError, match="non-positive definite"):
            cholesky_update(S, v, sign=-1.0)

    def test_cholesky_update_preserves_lower_triangular(self):
        """Test that update preserves lower triangular form."""
        P = np.array([[4.0, 2.0], [2.0, 3.0]])
        S = np.linalg.cholesky(P)
        v = np.array([0.5, 0.5])

        S_updated = cholesky_update(S, v, sign=1.0)

        # Check lower triangular
        np.testing.assert_allclose(S_updated, np.tril(S_updated))


class TestQRUpdate:
    """Tests for QR-based covariance propagation."""

    def test_qr_update_identity_F(self):
        """Test QR update with identity transition matrix."""
        P = np.diag([1.0, 2.0])
        S_x = np.linalg.cholesky(P)
        Q = np.eye(2) * 0.1
        S_noise = np.linalg.cholesky(Q)

        S_new = qr_update(S_x, S_noise, F=None)
        P_new = S_new @ S_new.T

        expected = P + Q
        np.testing.assert_allclose(P_new, expected, rtol=1e-10)

    def test_qr_update_with_transition(self):
        """Test QR update with state transition matrix."""
        P = np.eye(2) * 0.1
        S_x = np.linalg.cholesky(P)
        Q = np.eye(2) * 0.01
        S_noise = np.linalg.cholesky(Q)
        F = np.array([[1, 1], [0, 1]])

        S_new = qr_update(S_x, S_noise, F)
        P_new = S_new @ S_new.T

        expected = F @ P @ F.T + Q
        np.testing.assert_allclose(P_new, expected, rtol=1e-6)

    def test_qr_update_preserves_positive_definite(self):
        """Test QR update preserves positive definiteness."""
        P = np.diag([0.1, 0.2, 0.3])
        S_x = np.linalg.cholesky(P)
        Q = np.eye(3) * 0.01
        S_noise = np.linalg.cholesky(Q)
        F = np.eye(3)

        S_new = qr_update(S_x, S_noise, F)

        eigenvalues = np.linalg.eigvalsh(S_new @ S_new.T)
        assert np.all(eigenvalues > 0)


class TestEnsureSymmetric:
    """Tests for symmetry enforcement."""

    def test_ensure_symmetric_basic(self):
        """Test symmetry enforcement."""
        P = np.array([[1.0, 0.5 + 1e-15], [0.5, 1.0]])
        P_sym = ensure_symmetric(P)

        np.testing.assert_allclose(P_sym, P_sym.T)

    def test_ensure_symmetric_already_symmetric(self):
        """Test with already symmetric matrix."""
        P = np.array([[1.0, 0.5], [0.5, 2.0]])
        P_sym = ensure_symmetric(P)

        np.testing.assert_allclose(P_sym, P)


class TestComputeMatrixSqrt:
    """Tests for matrix square root computation."""

    def test_compute_matrix_sqrt_basic(self):
        """Test basic matrix square root."""
        P = np.array([[4.0, 2.0], [2.0, 3.0]])
        sqrt_P = compute_matrix_sqrt(P)

        np.testing.assert_allclose(sqrt_P @ sqrt_P.T, P, rtol=1e-10)

    def test_compute_matrix_sqrt_with_scale(self):
        """Test matrix square root with scaling."""
        P = np.array([[4.0, 2.0], [2.0, 3.0]])
        scale = 2.0
        sqrt_P = compute_matrix_sqrt(P, scale=scale)

        np.testing.assert_allclose(sqrt_P @ sqrt_P.T, scale * P, rtol=1e-10)

    def test_compute_matrix_sqrt_near_singular(self):
        """Test matrix square root with near-singular matrix."""
        # Create a nearly singular positive semi-definite matrix
        P = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        sqrt_P = compute_matrix_sqrt(P, use_eigh_fallback=True)

        # Result should be approximately valid
        reconstructed = sqrt_P @ sqrt_P.T
        np.testing.assert_allclose(reconstructed, P, rtol=1e-3)

    def test_compute_matrix_sqrt_no_fallback_raises(self):
        """Test that non-positive definite matrix raises without fallback."""
        # Create a non-positive definite matrix
        P = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3

        with pytest.raises(np.linalg.LinAlgError):
            compute_matrix_sqrt(P, use_eigh_fallback=False)


class TestComputeInnovationLikelihood:
    """Tests for innovation likelihood computation."""

    def test_compute_likelihood_zero_innovation(self):
        """Test likelihood with zero innovation."""
        y = np.array([0.0, 0.0])
        S = np.eye(2)

        likelihood = compute_innovation_likelihood(y, S)

        # Maximum likelihood at zero innovation
        expected = 1.0 / (2 * np.pi)  # For 2D standard normal
        np.testing.assert_allclose(likelihood, expected, rtol=1e-10)

    def test_compute_likelihood_with_cholesky(self):
        """Test likelihood using Cholesky factor."""
        y = np.array([0.5, -0.3])
        P = np.array([[0.5, 0.1], [0.1, 0.4]])
        S = np.linalg.cholesky(P)

        likelihood_full = compute_innovation_likelihood(y, P, S_is_cholesky=False)
        likelihood_chol = compute_innovation_likelihood(y, S, S_is_cholesky=True)

        np.testing.assert_allclose(likelihood_full, likelihood_chol, rtol=1e-10)

    def test_compute_likelihood_positive(self):
        """Test that likelihood is always positive."""
        y = np.array([1.0, -2.0, 0.5])
        S = np.eye(3) * 0.5

        likelihood = compute_innovation_likelihood(y, S)
        assert likelihood > 0


class TestComputeMahalanobisDistance:
    """Tests for Mahalanobis distance computation."""

    def test_mahalanobis_identity_covariance(self):
        """Test Mahalanobis distance with identity covariance."""
        y = np.array([1.0, 0.0])
        S = np.eye(2)

        dist = compute_mahalanobis_distance(y, S)

        # Should equal Euclidean norm
        np.testing.assert_allclose(dist, 1.0)

    def test_mahalanobis_scaled_covariance(self):
        """Test Mahalanobis distance with scaled covariance."""
        y = np.array([2.0, 0.0])
        S = np.eye(2) * 4.0  # Variance of 4 in each dimension

        dist = compute_mahalanobis_distance(y, S)

        # y normalized by sqrt(variance) = 2/2 = 1
        np.testing.assert_allclose(dist, 1.0)

    def test_mahalanobis_with_cholesky(self):
        """Test Mahalanobis distance using Cholesky factor."""
        y = np.array([0.5, -0.3])
        P = np.array([[0.5, 0.1], [0.1, 0.4]])
        S = np.linalg.cholesky(P)

        dist_full = compute_mahalanobis_distance(y, P, S_is_cholesky=False)
        dist_chol = compute_mahalanobis_distance(y, S, S_is_cholesky=True)

        np.testing.assert_allclose(dist_full, dist_chol, rtol=1e-10)


class TestComputeMerweWeights:
    """Tests for sigma point weight computation."""

    def test_merwe_weights_sum_to_one(self):
        """Test that mean weights sum to 1."""
        n = 4
        W_m, W_c = compute_merwe_weights(n)

        np.testing.assert_allclose(np.sum(W_m), 1.0, rtol=1e-10)

    def test_merwe_weights_shape(self):
        """Test weight array shapes."""
        n = 3
        W_m, W_c = compute_merwe_weights(n)

        assert len(W_m) == 2 * n + 1
        assert len(W_c) == 2 * n + 1

    def test_merwe_weights_custom_params(self):
        """Test weights with custom parameters."""
        n = 2
        alpha = 0.01
        beta = 2.0
        kappa = 1.0

        W_m, W_c = compute_merwe_weights(n, alpha=alpha, beta=beta, kappa=kappa)

        np.testing.assert_allclose(np.sum(W_m), 1.0, rtol=1e-10)


# =============================================================================
# Tests for Square-Root UKF (sr_ukf.py)
# =============================================================================


class TestSRUKFPredict:
    """Tests for Square-Root UKF prediction."""

    def test_sr_ukf_predict_basic(self):
        """Test basic SR-UKF prediction."""

        def f(x):
            return np.array([x[0] + x[1], x[1]])

        x = np.array([1.0, 0.5])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        Q = np.eye(2) * 0.01
        S_Q = np.linalg.cholesky(Q)

        pred = sr_ukf_predict(x, S, f, S_Q)

        np.testing.assert_allclose(pred.x, [1.5, 0.5], rtol=1e-6)
        assert pred.S.shape == (2, 2)

    def test_sr_ukf_predict_preserves_cholesky(self):
        """Test that prediction returns valid Cholesky factor."""

        def f(x):
            return x

        x = np.array([0.0, 1.0])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        S_Q = np.linalg.cholesky(np.eye(2) * 0.01)

        pred = sr_ukf_predict(x, S, f, S_Q)

        # Check lower triangular
        np.testing.assert_allclose(pred.S, np.tril(pred.S), atol=1e-10)
        # Check positive diagonal
        assert np.all(np.diag(pred.S) > 0)

    def test_sr_ukf_predict_nonlinear(self):
        """Test SR-UKF prediction with nonlinear dynamics."""

        def f(x):
            return np.array([np.sin(x[0]) + x[1], np.cos(x[1])])

        x = np.array([0.0, 0.5])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        S_Q = np.linalg.cholesky(np.eye(2) * 0.01)

        pred = sr_ukf_predict(x, S, f, S_Q)

        # Check that covariance is positive definite
        P_pred = pred.S @ pred.S.T
        eigenvalues = np.linalg.eigvalsh(P_pred)
        assert np.all(eigenvalues > 0)

    def test_sr_ukf_predict_custom_params(self):
        """Test SR-UKF prediction with custom sigma point parameters."""

        def f(x):
            return x

        x = np.array([1.0, 2.0])
        S = np.linalg.cholesky(np.eye(2))
        S_Q = np.linalg.cholesky(np.eye(2) * 0.1)

        pred = sr_ukf_predict(x, S, f, S_Q, alpha=0.1, beta=2.0, kappa=0.0)

        assert pred.x is not None
        assert pred.S is not None


class TestSRUKFUpdate:
    """Tests for Square-Root UKF update."""

    def test_sr_ukf_update_basic(self):
        """Test basic SR-UKF update."""

        def h(x):
            return np.array([x[0]])  # Measure first state

        x = np.array([1.0, 0.5])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        z = np.array([1.1])
        S_R = np.linalg.cholesky(np.array([[0.05]]))

        upd = sr_ukf_update(x, S, z, h, S_R)

        assert upd.x is not None
        assert upd.S is not None
        assert upd.y is not None
        assert upd.likelihood > 0

    def test_sr_ukf_update_reduces_uncertainty(self):
        """Test that update reduces uncertainty (trace of covariance)."""

        def h(x):
            return np.array([x[0], x[1]])  # Measure both states

        x = np.array([1.0, 0.5])
        P = np.eye(2) * 10.0
        S = np.linalg.cholesky(P)
        z = np.array([1.0, 0.5])
        R = np.eye(2) * 0.1
        S_R = np.linalg.cholesky(R)

        upd = sr_ukf_update(x, S, z, h, S_R)

        P_upd = upd.S @ upd.S.T
        assert np.trace(P_upd) < np.trace(P)

    def test_sr_ukf_update_nonlinear(self):
        """Test SR-UKF update with nonlinear measurement."""

        def h(x):
            # Range-bearing measurement
            return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2)])

        x = np.array([3.0, 4.0])  # Range should be 5
        S = np.linalg.cholesky(np.eye(2) * 0.5)
        z = np.array([5.1])  # Slightly noisy measurement
        S_R = np.linalg.cholesky(np.array([[0.1]]))

        upd = sr_ukf_update(x, S, z, h, S_R)

        assert upd.likelihood > 0
        # Check that covariance is valid
        P_upd = upd.S @ upd.S.T
        eigenvalues = np.linalg.eigvalsh(P_upd)
        assert np.all(eigenvalues > 0)


class TestSRUKFConsistency:
    """Tests for SR-UKF consistency with standard UKF."""

    def test_sr_ukf_linear_matches_kf(self):
        """Test SR-UKF matches linear KF on linear system."""
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        Q = np.eye(2) * 0.01
        S_Q = np.linalg.cholesky(Q)
        R = np.array([[0.1]])
        S_R = np.linalg.cholesky(R)
        z = np.array([0.5])

        # SR-UKF
        sr_pred = sr_ukf_predict(x, S, f, S_Q)
        sr_upd = sr_ukf_update(sr_pred.x, sr_pred.S, z, h, S_R)

        # Standard KF
        kf_pred = kf_predict(x, P, F, Q)
        kf_upd = kf_update(kf_pred.x, kf_pred.P, z, H, R)

        # Results should be similar (not exact due to sigma point approximation)
        np.testing.assert_allclose(sr_upd.x, kf_upd.x, atol=0.1)


# =============================================================================
# Tests for U-D Factorization Filter (ud_filter.py)
# =============================================================================


class TestUDFactorize:
    """Tests for U-D factorization."""

    def test_ud_factorize_basic(self):
        """Test basic U-D factorization."""
        P = np.array([[4.0, 2.0], [2.0, 3.0]])
        U, D = ud_factorize(P)

        # Reconstruct and compare
        P_reconstructed = U @ np.diag(D) @ U.T
        np.testing.assert_allclose(P_reconstructed, P, rtol=1e-10)

    def test_ud_factorize_unit_upper_triangular(self):
        """Test that U is unit upper triangular."""
        P = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 2.0]])
        U, D = ud_factorize(P)

        # Check unit upper triangular
        np.testing.assert_allclose(np.diag(U), np.ones(3))
        np.testing.assert_allclose(np.tril(U, -1), np.zeros((3, 3)))

    def test_ud_factorize_diagonal(self):
        """Test U-D factorization of diagonal matrix."""
        P = np.diag([1.0, 2.0, 3.0])
        U, D = ud_factorize(P)

        np.testing.assert_allclose(U, np.eye(3))
        np.testing.assert_allclose(D, [1.0, 2.0, 3.0])


class TestUDReconstruct:
    """Tests for U-D reconstruction."""

    def test_ud_reconstruct_basic(self):
        """Test basic U-D reconstruction."""
        U = np.array([[1.0, 0.5], [0.0, 1.0]])
        D = np.array([2.0, 1.0])

        P = ud_reconstruct(U, D)

        expected = U @ np.diag(D) @ U.T
        np.testing.assert_allclose(P, expected)

    def test_ud_factorize_reconstruct_roundtrip(self):
        """Test factorize then reconstruct gives original matrix."""
        P_original = np.array([[4.0, 2.0], [2.0, 3.0]])
        U, D = ud_factorize(P_original)
        P_reconstructed = ud_reconstruct(U, D)

        np.testing.assert_allclose(P_reconstructed, P_original, rtol=1e-10)


class TestUDPredict:
    """Tests for U-D filter prediction."""

    def test_ud_predict_basic(self):
        """Test basic U-D prediction."""
        x = np.array([1.0, 0.0])
        P = np.eye(2) * 0.1
        U, D = ud_factorize(P)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01

        x_pred, U_pred, D_pred = ud_predict(x, U, D, F, Q)

        # Check state prediction
        np.testing.assert_allclose(x_pred, [1.0, 0.0])

        # Check covariance prediction
        P_pred = ud_reconstruct(U_pred, D_pred)
        expected_P = F @ P @ F.T + Q
        np.testing.assert_allclose(P_pred, expected_P, rtol=1e-6)

    def test_ud_predict_matches_kf(self):
        """Test U-D prediction matches standard KF."""
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.5
        U, D = ud_factorize(P)
        F = np.array([[1, 0.1], [0, 1]])
        Q = np.eye(2) * 0.01

        # U-D prediction
        x_ud, U_ud, D_ud = ud_predict(x, U, D, F, Q)
        P_ud = ud_reconstruct(U_ud, D_ud)

        # Standard KF prediction
        kf_pred = kf_predict(x, P, F, Q)

        np.testing.assert_allclose(x_ud, kf_pred.x, rtol=1e-10)
        np.testing.assert_allclose(P_ud, kf_pred.P, rtol=1e-6)


class TestUDUpdateScalar:
    """Tests for U-D scalar measurement update."""

    def test_ud_update_scalar_basic(self):
        """Test basic scalar update."""
        x = np.array([1.0, 0.5])
        U = np.eye(2)
        D = np.array([0.2, 0.1])
        z = 1.1
        h = np.array([1.0, 0.0])
        r = 0.1

        x_upd, U_upd, D_upd = ud_update_scalar(x, U, D, z, h, r)

        # Check that state was updated
        assert not np.allclose(x_upd, x)
        # Check that U remains unit upper triangular
        np.testing.assert_allclose(np.diag(U_upd), np.ones(2), rtol=1e-10)

    def test_ud_update_scalar_reduces_uncertainty(self):
        """Test that scalar update reduces uncertainty."""
        x = np.array([1.0, 0.5])
        P = np.eye(2) * 10.0
        U, D = ud_factorize(P)
        z = 1.0
        h = np.array([1.0, 0.0])
        r = 0.1

        x_upd, U_upd, D_upd = ud_update_scalar(x, U, D, z, h, r)
        P_upd = ud_reconstruct(U_upd, D_upd)

        assert np.trace(P_upd) < np.trace(P)


class TestUDUpdate:
    """Tests for U-D vector measurement update."""

    def test_ud_update_basic(self):
        """Test basic vector update."""
        x = np.array([1.0, 0.5])
        P = np.eye(2) * 0.2
        U, D = ud_factorize(P)
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        x_upd, U_upd, D_upd, y, likelihood = ud_update(x, U, D, z, H, R)

        assert likelihood > 0
        assert len(y) == 1

    def test_ud_update_matches_kf(self):
        """Test U-D update matches standard KF."""
        x = np.array([1.0, 0.5])
        P = np.eye(2) * 0.5
        U, D = ud_factorize(P)
        z = np.array([1.1])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        # U-D update
        x_ud, U_ud, D_ud, _, _ = ud_update(x, U, D, z, H, R)
        P_ud = ud_reconstruct(U_ud, D_ud)

        # Standard KF update
        kf_upd = kf_update(x, P, z, H, R)

        np.testing.assert_allclose(x_ud, kf_upd.x, rtol=1e-6)
        np.testing.assert_allclose(P_ud, kf_upd.P, rtol=1e-5)

    def test_ud_update_multiple_measurements(self):
        """Test U-D update with multiple measurements."""
        x = np.array([1.0, 0.5, 0.0])
        P = np.eye(3) * 0.5
        U, D = ud_factorize(P)
        z = np.array([1.1, 0.6])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = np.eye(2) * 0.1

        x_upd, U_upd, D_upd, y, likelihood = ud_update(x, U, D, z, H, R)

        assert likelihood > 0
        P_upd = ud_reconstruct(U_upd, D_upd)
        assert np.trace(P_upd) < np.trace(P)


# =============================================================================
# Tests for Square-Root Linear KF (square_root.py)
# =============================================================================


class TestSRKFPredict:
    """Tests for square-root linear KF prediction."""

    def test_srkf_predict_basic(self):
        """Test basic SR-KF prediction."""
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.333, 0.5], [0.5, 1.0]])  # Positive definite
        S_Q = np.linalg.cholesky(Q)

        pred = srkf_predict(x, S, F, S_Q)

        np.testing.assert_allclose(pred.x, [1.0, 1.0])
        assert pred.S.shape == (2, 2)

    def test_srkf_predict_with_control(self):
        """Test SR-KF prediction with control input."""
        x = np.array([0.0, 0.0])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        F = np.eye(2)
        S_Q = np.linalg.cholesky(np.eye(2) * 0.01)
        B = np.array([[0.5], [1.0]])
        u = np.array([2.0])

        pred = srkf_predict(x, S, F, S_Q, B, u)

        np.testing.assert_allclose(pred.x, [1.0, 2.0])

    def test_srkf_predict_matches_kf(self):
        """Test SR-KF prediction matches standard KF."""
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        S = np.linalg.cholesky(P)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        S_Q = np.linalg.cholesky(Q)

        # SR-KF
        sr_pred = srkf_predict(x, S, F, S_Q)
        P_sr = sr_pred.S @ sr_pred.S.T

        # Standard KF
        kf_pred = kf_predict(x, P, F, Q)

        np.testing.assert_allclose(sr_pred.x, kf_pred.x)
        np.testing.assert_allclose(P_sr, kf_pred.P, rtol=1e-6)


class TestSRKFUpdate:
    """Tests for square-root linear KF update."""

    def test_srkf_update_basic(self):
        """Test basic SR-KF update."""
        x = np.array([1.0, 1.0])
        P = np.array([[0.35, 0.5], [0.5, 1.1]])
        S = np.linalg.cholesky(P)
        z = np.array([1.2])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])
        S_R = np.linalg.cholesky(R)

        upd = srkf_update(x, S, z, H, S_R)

        assert upd.x is not None
        assert upd.likelihood > 0

    def test_srkf_update_matches_kf(self):
        """Test SR-KF update matches standard KF."""
        x = np.array([1.0, 1.0])
        P = np.eye(2) * 0.5
        S = np.linalg.cholesky(P)
        z = np.array([1.1])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])
        S_R = np.linalg.cholesky(R)

        # SR-KF
        sr_upd = srkf_update(x, S, z, H, S_R)
        P_sr = sr_upd.S @ sr_upd.S.T

        # Standard KF
        kf_upd = kf_update(x, P, z, H, R)

        np.testing.assert_allclose(sr_upd.x, kf_upd.x, rtol=1e-6)
        np.testing.assert_allclose(P_sr, kf_upd.P, rtol=1e-5)


class TestSRKFPredictUpdate:
    """Tests for combined SR-KF predict-update."""

    def test_srkf_predict_update_basic(self):
        """Test combined SR-KF predict-update."""
        x = np.array([0.0, 1.0])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        F = np.array([[1, 1], [0, 1]])
        S_Q = np.linalg.cholesky(np.eye(2) * 0.01)
        H = np.array([[1, 0]])
        S_R = np.linalg.cholesky(np.array([[0.1]]))
        z = np.array([1.05])

        result = srkf_predict_update(x, S, z, F, S_Q, H, S_R)

        assert result.x is not None
        assert result.likelihood > 0

    def test_srkf_predict_update_equivalence(self):
        """Test combined equals separate calls."""
        x = np.array([0.0, 1.0])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        F = np.array([[1, 1], [0, 1]])
        S_Q = np.linalg.cholesky(np.eye(2) * 0.01)
        H = np.array([[1, 0]])
        S_R = np.linalg.cholesky(np.array([[0.1]]))
        z = np.array([1.05])

        # Combined
        result = srkf_predict_update(x, S, z, F, S_Q, H, S_R)

        # Separate
        pred = srkf_predict(x, S, F, S_Q)
        upd = srkf_update(pred.x, pred.S, z, H, S_R)

        np.testing.assert_allclose(result.x, upd.x)
        np.testing.assert_allclose(result.S, upd.S)


# =============================================================================
# Tests for IMM Estimator (imm.py)
# =============================================================================


class TestComputeMixingProbabilities:
    """Tests for mixing probability computation."""

    def test_mixing_probs_uniform(self):
        """Test mixing probabilities with uniform mode probs."""
        mode_probs = np.array([0.5, 0.5])
        Pi = np.array([[0.9, 0.1], [0.1, 0.9]])

        mixing_probs, c_bar = compute_mixing_probabilities(mode_probs, Pi)

        # c_bar should be uniform too
        np.testing.assert_allclose(c_bar, [0.5, 0.5])
        # mixing_probs columns should sum to 1
        np.testing.assert_allclose(np.sum(mixing_probs, axis=0), [1.0, 1.0])

    def test_mixing_probs_biased(self):
        """Test mixing probabilities with biased mode probs."""
        mode_probs = np.array([0.9, 0.1])
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])

        mixing_probs, c_bar = compute_mixing_probabilities(mode_probs, Pi)

        # c_bar should reflect the bias
        assert c_bar[0] > c_bar[1]
        # mixing_probs columns should sum to 1
        np.testing.assert_allclose(np.sum(mixing_probs, axis=0), [1.0, 1.0], rtol=1e-10)


class TestMixStates:
    """Tests for state mixing."""

    def test_mix_states_identical(self):
        """Test mixing identical states."""
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        mode_states = [x.copy(), x.copy()]
        mode_covs = [P.copy(), P.copy()]
        mixing_probs = np.array([[0.9, 0.1], [0.1, 0.9]])

        mixed_states, mixed_covs = mix_states(mode_states, mode_covs, mixing_probs)

        # Mixed states should be close to original
        for ms in mixed_states:
            np.testing.assert_allclose(ms, x, rtol=1e-10)

    def test_mix_states_different(self):
        """Test mixing different states."""
        mode_states = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        mode_covs = [np.eye(2) * 0.1, np.eye(2) * 0.1]
        # Equal mixing
        mixing_probs = np.array([[0.5, 0.5], [0.5, 0.5]])

        mixed_states, mixed_covs = mix_states(mode_states, mode_covs, mixing_probs)

        # Mixed states should be averages
        expected = np.array([0.5, 0.5])
        for ms in mixed_states:
            np.testing.assert_allclose(ms, expected)


class TestCombineEstimates:
    """Tests for estimate combination."""

    def test_combine_estimates_single_mode(self):
        """Test combining with single dominant mode."""
        mode_states = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        mode_covs = [np.eye(2) * 0.1, np.eye(2) * 0.1]
        mode_probs = np.array([1.0, 0.0])  # Only first mode

        x, P = combine_estimates(mode_states, mode_covs, mode_probs)

        np.testing.assert_allclose(x, [1.0, 0.0])
        np.testing.assert_allclose(P, np.eye(2) * 0.1)

    def test_combine_estimates_equal_weights(self):
        """Test combining with equal weights."""
        mode_states = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        mode_covs = [np.eye(2) * 0.1, np.eye(2) * 0.1]
        mode_probs = np.array([0.5, 0.5])

        x, P = combine_estimates(mode_states, mode_covs, mode_probs)

        np.testing.assert_allclose(x, [0.5, 0.5])
        # Covariance should be larger due to spread
        assert np.trace(P) > np.trace(mode_covs[0])


class TestIMMPredict:
    """Tests for IMM prediction."""

    def test_imm_predict_basic(self):
        """Test basic IMM prediction."""
        x1 = np.array([0.0, 1.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0, 0.0])
        P1 = np.eye(4) * 0.1
        P2 = np.eye(4) * 0.1
        mu = np.array([0.9, 0.1])
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        F1 = np.eye(4)
        F2 = np.eye(4)
        Q1 = np.eye(4) * 0.01
        Q2 = np.eye(4) * 0.01

        pred = imm_predict([x1, x2], [P1, P2], mu, Pi, [F1, F2], [Q1, Q2])

        assert pred.x is not None
        assert len(pred.mode_states) == 2
        assert len(pred.mode_probs) == 2

    def test_imm_predict_mode_probs_sum_to_one(self):
        """Test that predicted mode probs sum to 1."""
        states = [np.zeros(2), np.zeros(2)]
        covs = [np.eye(2), np.eye(2)]
        mu = np.array([0.7, 0.3])
        Pi = np.array([[0.9, 0.1], [0.2, 0.8]])
        F = [np.eye(2), np.eye(2)]
        Q = [np.eye(2) * 0.01, np.eye(2) * 0.01]

        pred = imm_predict(states, covs, mu, Pi, F, Q)

        np.testing.assert_allclose(np.sum(pred.mode_probs), 1.0, rtol=1e-10)


class TestIMMUpdate:
    """Tests for IMM update."""

    def test_imm_update_basic(self):
        """Test basic IMM update."""
        x1 = np.array([1.0, 1.0, 0.0, 0.0])
        x2 = np.array([1.0, 1.0, 0.0, 0.0])
        P1 = np.eye(4) * 0.2
        P2 = np.eye(4) * 0.2
        mu = np.array([0.9, 0.1])
        z = np.array([1.1, 0.1])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1

        upd = imm_update([x1, x2], [P1, P2], mu, z, [H, H], [R, R])

        assert upd.x is not None
        assert len(upd.mode_likelihoods) == 2
        np.testing.assert_allclose(np.sum(upd.mode_probs), 1.0, rtol=1e-10)

    def test_imm_update_mode_prob_shift(self):
        """Test that mode probabilities shift based on likelihoods."""
        # Mode 1: state matches measurement
        # Mode 2: state doesn't match measurement
        x1 = np.array([1.0, 0.0])
        x2 = np.array([5.0, 0.0])  # Far from measurement
        P1 = np.eye(2) * 0.1
        P2 = np.eye(2) * 0.1
        mu = np.array([0.5, 0.5])  # Equal initial probs
        z = np.array([1.0])  # Matches mode 1
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        upd = imm_update([x1, x2], [P1, P2], mu, z, [H, H], [R, R])

        # Mode 1 should have higher probability after update
        assert upd.mode_probs[0] > upd.mode_probs[1]


class TestIMMPredictUpdate:
    """Tests for combined IMM predict-update."""

    def test_imm_predict_update_basic(self):
        """Test combined IMM predict-update."""
        states = [np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])]
        covs = [np.eye(4) * 0.1, np.eye(4) * 0.1]
        probs = np.array([0.9, 0.1])
        trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        F_cv = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        Q = np.eye(4) * 0.01
        R = np.eye(2) * 0.1
        z = np.array([1.0, 1.0])

        result = imm_predict_update(
            states, covs, probs, trans, z, [F_cv, F_cv], [Q, Q], [H, H], [R, R]
        )

        assert len(result.mode_probs) == 2
        np.testing.assert_allclose(np.sum(result.mode_probs), 1.0, rtol=1e-10)


class TestIMMEstimator:
    """Tests for IMMEstimator class."""

    def test_imm_estimator_init(self):
        """Test IMMEstimator initialization."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        assert imm.n_modes == 2
        assert imm.state_dim == 4
        assert len(imm.mode_states) == 2
        assert len(imm.mode_covs) == 2

    def test_imm_estimator_initialize(self):
        """Test IMMEstimator initialization with state."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        x0 = np.array([0.0, 1.0, 0.0, 0.0])
        P0 = np.eye(4) * 0.1
        imm.initialize(x0, P0)

        np.testing.assert_allclose(imm.x, x0)
        np.testing.assert_allclose(imm.P, P0)
        for ms in imm.mode_states:
            np.testing.assert_allclose(ms, x0)

    def test_imm_estimator_set_mode_model(self):
        """Test setting mode-specific models."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        F1 = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        Q1 = np.eye(4) * 0.01
        imm.set_mode_model(0, F1, Q1)

        np.testing.assert_allclose(imm.F_list[0], F1)
        np.testing.assert_allclose(imm.Q_list[0], Q1)

    def test_imm_estimator_set_measurement_model(self):
        """Test setting measurement model."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1
        imm.set_measurement_model(H, R)

        assert len(imm.H_list) == 2
        assert len(imm.R_list) == 2

    def test_imm_estimator_predict(self):
        """Test IMMEstimator prediction."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        x0 = np.array([0.0, 1.0, 0.0, 0.0])
        P0 = np.eye(4) * 0.1
        imm.initialize(x0, P0)

        F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        imm.set_mode_model(0, F, Q)
        imm.set_mode_model(1, F, Q)

        result = imm.predict()

        assert result.x is not None
        np.testing.assert_allclose(np.sum(result.mode_probs), 1.0, rtol=1e-10)

    def test_imm_estimator_update(self):
        """Test IMMEstimator update."""
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

        imm.predict()
        z = np.array([1.0, 0.0])
        result = imm.update(z)

        assert result.x is not None
        assert len(result.mode_likelihoods) == 2

    def test_imm_estimator_predict_update(self):
        """Test IMMEstimator combined predict-update."""
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

        z = np.array([1.0, 0.0])
        result = imm.predict_update(z)

        assert result.x is not None

    def test_imm_estimator_get_state(self):
        """Test IMMEstimator get_state method."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        x0 = np.array([0.0, 1.0, 0.0, 0.0])
        P0 = np.eye(4) * 0.1
        imm.initialize(x0, P0)

        state = imm.get_state()

        np.testing.assert_allclose(state.x, x0)
        np.testing.assert_allclose(state.P, P0)
        assert len(state.mode_states) == 2
        assert len(state.mode_covs) == 2

    def test_imm_estimator_update_without_measurement_model_raises(self):
        """Test that update without measurement model raises error."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        x0 = np.array([0.0, 1.0, 0.0, 0.0])
        P0 = np.eye(4) * 0.1
        imm.initialize(x0, P0)

        # No measurement model set
        z = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Measurement model not set"):
            imm.update(z)

    def test_imm_estimator_full_sequence(self):
        """Test IMMEstimator over multiple time steps."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        x0 = np.array([0.0, 1.0, 0.0, 0.5])
        P0 = np.eye(4) * 0.1
        imm.initialize(x0, P0, mode_probs=np.array([0.9, 0.1]))

        F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        imm.set_mode_model(0, F, Q)
        imm.set_mode_model(1, F, Q)

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 0.1
        imm.set_measurement_model(H, R)

        # Simulate multiple steps
        for k in range(5):
            z = np.array([float(k), 0.5 * k])
            imm.predict_update(z)

        # Check that mode probs are valid
        np.testing.assert_allclose(np.sum(imm.mode_probs), 1.0, rtol=1e-10)
        assert np.all(imm.mode_probs >= 0)
