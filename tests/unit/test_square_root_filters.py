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
from scipy.stats import multivariate_normal

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
from pytcl.dynamic_estimation.kalman.unscented import ukf_predict, ukf_update

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
        np.testing.assert_allclose(P_new, expected, rtol=1e-12, atol=0)

    def test_qr_update_preserves_positive_definite(self):
        """Test QR update preserves positive definiteness."""
        P = np.diag([0.1, 0.2, 0.3])
        S_x = np.linalg.cholesky(P)
        Q = np.eye(3) * 0.01
        S_noise = np.linalg.cholesky(Q)
        F = np.eye(3)

        S_new = qr_update(S_x, S_noise, F)

        np.testing.assert_allclose(
            S_new @ S_new.T, np.diag([0.11, 0.21, 0.31]), rtol=1e-12, atol=0
        )


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

        expected = multivariate_normal(mean=np.zeros(3), cov=S).pdf(y)
        np.testing.assert_allclose(likelihood, expected, rtol=1e-12, atol=0)


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

        # Defaults alpha=1e-3, beta=2, kappa=0: lambda = alpha^2 (n + kappa) - n.
        lam = 1e-3**2 * n - n
        w_rest = 1.0 / (2 * (n + lam))
        W_m_expected = np.full(2 * n + 1, w_rest)
        W_c_expected = np.full(2 * n + 1, w_rest)
        W_m_expected[0] = lam / (n + lam)
        W_c_expected[0] = lam / (n + lam) + (1 - 1e-3**2 + 2.0)
        np.testing.assert_allclose(W_m, W_m_expected, rtol=1e-12, atol=0)
        np.testing.assert_allclose(W_c, W_c_expected, rtol=1e-12, atol=0)

    def test_merwe_weights_custom_params(self):
        """Test weights with custom parameters."""
        n = 2
        alpha = 0.01
        beta = 2.0
        kappa = 1.0

        W_m, W_c = compute_merwe_weights(n, alpha=alpha, beta=beta, kappa=kappa)

        lam = alpha**2 * (n + kappa) - n
        w_rest = 1.0 / (2 * (n + lam))
        W_m_expected = np.full(2 * n + 1, w_rest)
        W_c_expected = np.full(2 * n + 1, w_rest)
        W_m_expected[0] = lam / (n + lam)
        W_c_expected[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        np.testing.assert_allclose(W_m, W_m_expected, rtol=1e-12, atol=0)
        np.testing.assert_allclose(W_c, W_c_expected, rtol=1e-12, atol=0)
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

        # f is linear, so the unscented transform is exact: the covariance is
        # the Kalman prediction F P F' + Q with F = [[1, 1], [0, 1]].
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf = kf_predict(x, P, F, Q)
        # rtol 1e-9, not 1e-12: the default alpha=1e-3 makes the sigma weights
        # ~1e6 with opposite signs, so the sums cancel to ~1e-10 relative error
        # (alpha=1 reaches 1e-12; see test_sr_ukf_linear_matches_kf_at_unit_alpha).
        np.testing.assert_allclose(pred.x, [1.5, 0.5], rtol=1e-9, atol=0)
        np.testing.assert_allclose(pred.S @ pred.S.T, kf.P, rtol=1e-9, atol=0)

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
        # A lower-triangular factor with positive diagonal is unique: the
        # Cholesky factor of P + Q = 0.11 I.
        np.testing.assert_allclose(
            pred.S, np.linalg.cholesky(np.eye(2) * 0.11), rtol=1e-9, atol=1e-12
        )

    def test_sr_ukf_predict_nonlinear(self):
        """Test SR-UKF prediction with nonlinear dynamics."""

        def f(x):
            return np.array([np.sin(x[0]) + x[1], np.cos(x[1])])

        x = np.array([0.0, 0.5])
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        S_Q = np.linalg.cholesky(np.eye(2) * 0.01)

        pred = sr_ukf_predict(x, S, f, S_Q)

        # Oracle: the covariance-form UKF with the same sigma-point scaling.
        ref = ukf_predict(x, np.eye(2) * 0.1, f, np.eye(2) * 0.01)
        np.testing.assert_allclose(pred.x, ref.x, rtol=1e-9, atol=0)
        np.testing.assert_allclose(pred.S @ pred.S.T, ref.P, rtol=1e-9, atol=0)

    def test_sr_ukf_predict_custom_params(self):
        """Test SR-UKF prediction with custom sigma point parameters."""

        def f(x):
            return x

        x = np.array([1.0, 2.0])
        S = np.linalg.cholesky(np.eye(2))
        S_Q = np.linalg.cholesky(np.eye(2) * 0.1)

        pred = sr_ukf_predict(x, S, f, S_Q, alpha=0.1, beta=2.0, kappa=0.0)

        # Identity dynamics: the mean is unchanged and P = I + 0.1 I.
        np.testing.assert_allclose(pred.x, x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            pred.S @ pred.S.T, np.eye(2) * 1.1, rtol=1e-9, atol=1e-12
        )


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

        # h is linear, so this is the Kalman update (innovation covariance
        # 0.1 + 0.05, normal density of the innovation for the likelihood).
        kf = kf_update(
            x, np.eye(2) * 0.1, z, np.array([[1.0, 0.0]]), np.array([[0.05]])
        )
        np.testing.assert_allclose(upd.x, kf.x, rtol=1e-9, atol=0)
        np.testing.assert_allclose(upd.S @ upd.S.T, kf.P, rtol=1e-9, atol=0)
        np.testing.assert_allclose(upd.y, [0.1], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            upd.likelihood,
            multivariate_normal(mean=[0.0], cov=[[0.15]]).pdf([0.1]),
            rtol=1e-9,
            atol=0,
        )

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

        kf = kf_update(x, P, z, np.eye(2), R)
        np.testing.assert_allclose(upd.S @ upd.S.T, kf.P, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(upd.x, kf.x, rtol=1e-9, atol=0)

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

        # Oracle: the covariance-form UKF update with the same scaling.
        ref = ukf_update(x, np.eye(2) * 0.5, z, h, np.array([[0.1]]))
        np.testing.assert_allclose(upd.x, ref.x, rtol=1e-9, atol=0)
        np.testing.assert_allclose(upd.S @ upd.S.T, ref.P, rtol=1e-9, atol=0)
        np.testing.assert_allclose(upd.likelihood, ref.likelihood, rtol=1e-9, atol=0)


class TestSRUKFConsistency:
    """Tests for SR-UKF consistency with standard UKF."""

    def test_sr_ukf_linear_matches_kf_at_unit_alpha(self):
        """With alpha=1, kappa=0 every sigma weight is O(1)."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        z = np.array([0.5])
        kw = {"alpha": 1.0, "beta": 2.0, "kappa": 0.0}

        pred = sr_ukf_predict(
            x, np.linalg.cholesky(P), lambda v: F @ v, np.linalg.cholesky(Q), **kw
        )
        upd = sr_ukf_update(
            pred.x, pred.S, z, lambda v: H @ v, np.linalg.cholesky(R), **kw
        )

        kf_pred = kf_predict(x, P, F, Q)
        kf_upd = kf_update(kf_pred.x, kf_pred.P, z, H, R)
        np.testing.assert_allclose(pred.S @ pred.S.T, kf_pred.P, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(upd.x, kf_upd.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(upd.S @ upd.S.T, kf_upd.P, rtol=1e-12, atol=1e-15)

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

        # Both maps are linear, so the unscented transform is exact.
        np.testing.assert_allclose(sr_upd.x, kf_upd.x, rtol=1e-9, atol=0)
        np.testing.assert_allclose(sr_upd.S @ sr_upd.S.T, kf_upd.P, rtol=1e-9, atol=0)


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

        kf = kf_update(x, U @ np.diag(D) @ U.T, np.array([z]), h[None, :], [[r]])
        np.testing.assert_allclose(x_upd, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            ud_reconstruct(U_upd, D_upd), kf.P, rtol=1e-12, atol=0
        )
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

        kf = kf_update(x, P, np.array([z]), h[None, :], [[r]])
        np.testing.assert_allclose(P_upd, kf.P, rtol=1e-12, atol=0)
        np.testing.assert_allclose(x_upd, kf.x, rtol=1e-12, atol=0)


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

        kf = kf_update(x, P, z, H, R)
        np.testing.assert_allclose(y, [0.1], rtol=1e-12, atol=0)
        np.testing.assert_allclose(x_upd, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            ud_reconstruct(U_upd, D_upd), kf.P, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            likelihood,
            multivariate_normal(mean=[0.0], cov=[[0.3]]).pdf([0.1]),
            rtol=1e-12,
            atol=0,
        )

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

        kf = kf_update(x, P, z, H, R)
        np.testing.assert_allclose(x_upd, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            ud_reconstruct(U_upd, D_upd), kf.P, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            likelihood,
            multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * 0.6).pdf(z - x[:2]),
            rtol=1e-12,
            atol=0,
        )


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
        np.testing.assert_allclose(
            pred.S @ pred.S.T, F @ P @ F.T + Q, rtol=1e-12, atol=0
        )

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

        kf = kf_update(x, P, z, H, R)
        np.testing.assert_allclose(upd.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(upd.S @ upd.S.T, kf.P, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            upd.likelihood,
            multivariate_normal(mean=[0.0], cov=[[0.45]]).pdf([0.2]),
            rtol=1e-12,
            atol=0,
        )

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

        pred = kf_predict(x, np.eye(2) * 0.1, F, np.eye(2) * 0.01)
        kf = kf_update(pred.x, pred.P, z, H, np.array([[0.1]]))
        np.testing.assert_allclose(result.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.S @ result.S.T, kf.P, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            result.likelihood,
            multivariate_normal(mean=[0.0], cov=kf.S).pdf(kf.y),
            rtol=1e-12,
            atol=0,
        )

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

        # c_j = sum_i Pi[i, j] mu_i; mu_{i|j} = Pi[i, j] mu_i / c_j.
        np.testing.assert_allclose(c_bar, [0.86, 0.14], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            mixing_probs,
            [[0.855 / 0.86, 0.045 / 0.14], [0.005 / 0.86, 0.095 / 0.14]],
            rtol=1e-12,
            atol=0,
        )


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
        # Mean of the mode covariances plus the spread of the means:
        # sum_i mu_i (x_i - x)(x_i - x)' = 0.25 [[1, -1], [-1, 1]].
        np.testing.assert_allclose(
            P,
            np.eye(2) * 0.1 + 0.25 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
            rtol=1e-12,
            atol=0,
        )


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

        # Identical modes with F = I: mixing changes nothing, every mode
        # covariance becomes P + Q, and mu_pred = Pi' mu = [0.86, 0.14].
        np.testing.assert_allclose(pred.x, x1, rtol=1e-12, atol=0)
        np.testing.assert_allclose(pred.P, np.eye(4) * 0.11, rtol=1e-12, atol=0)
        for ms in pred.mode_states:
            np.testing.assert_allclose(ms, x1, rtol=1e-12, atol=0)
        np.testing.assert_allclose(pred.mode_probs, [0.86, 0.14], rtol=1e-12, atol=0)

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

        # Both modes share x, P, H, R: innovation [0.1, 0.1] with covariance
        # H P H' + R = 0.3 I, equal likelihoods, so the posterior mode
        # probabilities are the prior and x is the single Kalman update.
        like = multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * 0.3).pdf(
            [0.1, 0.1]
        )
        np.testing.assert_allclose(
            upd.mode_likelihoods, [like, like], rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(upd.mode_probs, mu, rtol=1e-12, atol=0)
        kf = kf_update(x1, P1, z, H, R)
        np.testing.assert_allclose(upd.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(upd.P, kf.P, rtol=1e-12, atol=0)

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

        # Bayes: mu_j proportional to mu_j N(z - H x_j; 0, H P H' + R).
        like = np.array(
            [
                multivariate_normal(mean=[0.0], cov=[[0.2]]).pdf([0.0]),
                multivariate_normal(mean=[0.0], cov=[[0.2]]).pdf([-4.0]),
            ]
        )
        np.testing.assert_allclose(upd.mode_likelihoods, like, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            upd.mode_probs, like / like.sum(), rtol=1e-12, atol=0
        )


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

        # Identical modes: equal likelihoods, so mu = Pi' mu_prior = [0.82, 0.18]
        # and x is the single-model Kalman predict-update.
        np.testing.assert_allclose(result.mode_probs, [0.82, 0.18], rtol=1e-12, atol=0)
        pred = kf_predict(states[0].astype(float), covs[0], F_cv, Q)
        kf = kf_update(pred.x, pred.P, z, H, R)
        np.testing.assert_allclose(result.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.P, kf.P, rtol=1e-12, atol=0)


class TestIMMEstimator:
    """Tests for IMMEstimator class."""

    def test_imm_estimator_init(self):
        """Test IMMEstimator initialization."""
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

        assert imm.n_modes == 2
        assert imm.state_dim == 4
        np.testing.assert_array_equal(imm.mode_states, np.zeros((2, 4)))
        np.testing.assert_array_equal(imm.mode_covs, [np.eye(4), np.eye(4)])
        np.testing.assert_array_equal(imm.mode_probs, [0.5, 0.5])

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

        for k in range(2):
            np.testing.assert_array_equal(imm.H_list[k], H)
            np.testing.assert_array_equal(imm.R_list[k], R)

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

        kf = kf_predict(x0, P0, F, Q)
        np.testing.assert_allclose(result.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.P, kf.P, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.mode_probs, [0.5, 0.5], rtol=1e-12, atol=0)

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

        pred = kf_predict(x0, P0, F, Q)
        kf = kf_update(pred.x, pred.P, z, H, R)
        like = multivariate_normal(mean=np.zeros(2), cov=kf.S).pdf(kf.y)
        np.testing.assert_allclose(result.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.P, kf.P, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            result.mode_likelihoods, [like, like], rtol=1e-12, atol=0
        )

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

        pred = kf_predict(x0, P0, F, Q)
        kf = kf_update(pred.x, pred.P, z, H, R)
        np.testing.assert_allclose(result.x, kf.x, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.P, kf.P, rtol=1e-12, atol=0)

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
        for k in range(2):
            np.testing.assert_allclose(state.mode_states[k], x0, rtol=1e-14, atol=0)
            np.testing.assert_allclose(state.mode_covs[k], P0, rtol=1e-14, atol=0)

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

        # All modes share one model, so the likelihoods are equal and the
        # IMM reduces to a single Kalman filter, with mode probabilities
        # following the Markov chain mu <- Pi' mu.
        x_ref, P_ref = x0, P0
        mu_ref = np.array([0.9, 0.1])
        for k in range(5):
            pred = kf_predict(x_ref, P_ref, F, Q)
            kf = kf_update(pred.x, pred.P, np.array([float(k), 0.5 * k]), H, R)
            x_ref, P_ref = kf.x, kf.P
            mu_ref = Pi.T @ mu_ref
        np.testing.assert_allclose(imm.x, x_ref, rtol=1e-10, atol=0)
        np.testing.assert_allclose(imm.P, P_ref, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(imm.mode_probs, mu_ref, rtol=1e-10, atol=0)


class TestMatrixUtilsFallbackPaths:
    """The numerical-fallback branches nothing executed (JIT-disabled measure).

    Each is the path taken when the well-conditioned route fails: Cholesky
    raising on a non-PD matrix, or the product-of-diagonal determinant
    underflowing to zero on a legitimately PD but tiny covariance. Silent
    wrong turns in these branches are invisible precisely because they only
    run on degenerate inputs.
    """

    def test_matrix_sqrt_eigh_fallback_on_non_pd_input(self):
        """Lines ~307-331: cholesky raises; eigh fallback rejects an eigenvalue
        this negative (-0.5, relative to a max of 1.0) as genuinely non-PSD
        rather than clamping it -- see test_matrix_sqrt_guards.py for the
        roundoff-scale case this must still clamp silently."""
        from pytcl.dynamic_estimation.kalman.matrix_utils import compute_matrix_sqrt

        P = np.array([[1.0, 0.0], [0.0, -0.5]])  # indefinite: cholesky raises
        with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
            compute_matrix_sqrt(P, use_eigh_fallback=True)

        with pytest.raises(np.linalg.LinAlgError):
            compute_matrix_sqrt(P, use_eigh_fallback=False)

    def test_likelihood_underflowed_determinant_returns_zero(self):
        """Line 373: cholesky SUCCEEDS but prod(diag)**2 underflows to 0.0.

        diag(1e-200, 1e-200) is genuinely PD; the determinant computed as a
        product of Cholesky diagonals underflows float64, and the function's
        contract for that is 0.0 rather than a division by zero.
        """
        from pytcl.dynamic_estimation.kalman.matrix_utils import (
            compute_innovation_likelihood,
        )

        S = np.diag([1e-200, 1e-200])
        assert compute_innovation_likelihood(np.array([0.1, 0.1]), S) == 0.0

        # same guard on the S_is_cholesky=True path (line 361): the caller
        # supplies the factor, whose diagonal product underflows identically
        S_chol = np.diag([1e-200, 1e-200])
        assert (
            compute_innovation_likelihood(
                np.array([0.1, 0.1]), S_chol, S_is_cholesky=True
            )
            == 0.0
        )

    def test_likelihood_indefinite_det_negative_warns_and_returns_zero(self):
        from pytcl.dynamic_estimation.kalman.matrix_utils import (
            compute_innovation_likelihood,
        )

        S = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite, det = -3
        with pytest.warns(RuntimeWarning, match="not positive definite"):
            assert compute_innovation_likelihood(np.array([0.1, 0.2]), S) == 0.0

    @pytest.mark.parametrize(
        "innovation, S",
        [
            (np.full(2, 1e-2), np.diag([-1e-3, -1e-3])),
            (np.full(2, 1e-2), -np.eye(2)),
            (np.full(3, 1e-2), np.diag([1.0, -1.0, -1.0])),
        ],
    )
    def test_likelihood_even_negative_eigenvalues_warns_and_returns_zero(
        self, innovation, S
    ):
        """Defect test: det(S) > 0 despite two negative variances. Measured
        before the fix: 175.893, 0.15917, 0.063497 with zero warnings."""
        from pytcl.dynamic_estimation.kalman.matrix_utils import (
            compute_innovation_likelihood,
        )

        with pytest.warns(RuntimeWarning, match="not positive definite") as rec:
            assert compute_innovation_likelihood(innovation, S) == 0.0
        assert str(rec[0].message) == (
            "compute_innovation_likelihood: innovation covariance is not "
            "positive definite; likelihood set to 0.0 (numerical failure, "
            "not evidence). Check R and the covariance conditioning."
        )

    def test_mahalanobis_negative_definite_warns_and_returns_nan(self):
        """Defect test: sqrt of a negative quadratic form gave nan with only
        numpy's generic warning."""
        from pytcl.dynamic_estimation.kalman.matrix_utils import (
            compute_mahalanobis_distance,
        )

        with pytest.warns(RuntimeWarning, match="not positive definite"):
            d = compute_mahalanobis_distance(np.full(2, 1e-2), -np.eye(2))
        assert np.isnan(d)

    def test_mahalanobis_indefinite_does_not_return_a_finite_distance(self):
        """Defect test: S with eigenvalues 3 and -1 and an innovation along
        the +3 eigenvector gave a finite distance from the fallback solve, as
        if S were a covariance."""
        from pytcl.dynamic_estimation.kalman.matrix_utils import (
            compute_mahalanobis_distance,
        )

        S = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.warns(RuntimeWarning, match="not positive definite"):
            d = compute_mahalanobis_distance(np.array([1.0, 1.0]), S)
        assert np.isnan(d)

    @pytest.mark.parametrize("scale", [1e-9, 1e-12, 1e-15])
    def test_matrix_sqrt_floor_does_not_inflate_tiny_covariance(self, scale):
        """Defect test: an absolute 1e-10 floor turned diag(1e-12, 0) into
        diag(1e-10, 1e-10) (100x inflated)."""
        from pytcl.dynamic_estimation.kalman.matrix_utils import compute_matrix_sqrt

        S = compute_matrix_sqrt(np.diag([scale, 0.0]))
        recon = S @ S.T
        assert recon[0, 0] == pytest.approx(scale, rel=1e-12, abs=0)
        assert recon[1, 1] <= 1e-10 * scale * 1.0000001

    @pytest.mark.parametrize("predict", ["ukf", "ckf"])
    def test_sigma_point_floor_does_not_inflate_tiny_covariance(self, predict):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict, ukf_predict

        fn = ukf_predict if predict == "ukf" else ckf_predict
        P = np.diag([1e-12, 0.0])
        pred = fn(np.zeros(2), P, lambda x: x, np.zeros((2, 2)))
        assert pred.P[0, 0] == pytest.approx(1e-12, rel=1e-9, abs=0)
        assert pred.P[1, 1] < 1e-20
