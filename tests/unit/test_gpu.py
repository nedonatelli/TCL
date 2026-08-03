"""Tests for GPU acceleration module.

These tests verify the GPU-accelerated implementations produce results
consistent with CPU implementations. They run on whichever GPU compute
backend is installed -- CuPy (NVIDIA, float64) or MLX (Apple Silicon,
float32) -- and are skipped only when neither is available.

Tolerances are backend-dependent: CuPy keeps the original float64
tolerances, MLX relaxes to float32-appropriate values.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.gpu._backend import compute_backend_available, get_compute_backend

# Skip all tests if no GPU compute backend is available
pytestmark = pytest.mark.skipif(
    not compute_backend_available(),
    reason="no GPU compute backend (cupy or mlx) available",
)

_FLOAT64 = compute_backend_available() and get_compute_backend().supports_float64
# float32 backends resolve ~1e-7; keep float64 backends at their original
# strict tolerances so CuPy validation is not weakened.
RTOL_TIGHT = 1e-10 if _FLOAT64 else 1e-5
RTOL_LOOSE = 1e-9 if _FLOAT64 else 1e-5
RTOL_MED = 1e-8 if _FLOAT64 else 1e-4
# An absolute floor is required alongside rtol: these fixtures use unseeded
# random data, so state components land arbitrarily close to zero, where a
# pure relative tolerance can never be met in float32 (1e-7 absolute noise
# on a 1e-3 element is 1e-4 relative).
ATOL = 0.0 if _FLOAT64 else 1e-6


class TestGPUUtils:
    """Tests for GPU utility functions."""

    def test_is_gpu_available(self):
        """Test GPU availability check."""
        from pytcl.gpu.utils import is_gpu_available

        # Should not raise
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_to_gpu_and_back(self):
        """Test array transfer to and from GPU."""
        from pytcl.gpu.utils import is_gpu_available, to_cpu, to_gpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        x = np.array([1.0, 2.0, 3.0])
        x_gpu = to_gpu(x)
        x_back = to_cpu(x_gpu)

        assert_allclose(x_back, x)

    def test_get_array_module(self):
        """Test array module detection."""
        from pytcl.gpu.utils import get_array_module, is_gpu_available, to_gpu

        # Numpy array
        x_np = np.array([1, 2, 3])
        xp = get_array_module(x_np)
        assert xp is np

        if is_gpu_available():
            # The module depends on which backend is active
            x_gpu = to_gpu(x_np)
            xp = get_array_module(x_gpu)
            assert xp is get_compute_backend().xp


class TestBatchKalmanFilter:
    """Tests for GPU-accelerated batch Kalman filter."""

    def test_batch_kf_predict_shapes(self):
        """Test batch KF prediction output shapes."""
        from pytcl.gpu.kalman import batch_kf_predict
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        n_tracks = 10
        state_dim = 4

        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim), (n_tracks, 1, 1))
        F = np.eye(state_dim)
        Q = np.eye(state_dim) * 0.1

        result = batch_kf_predict(x, P, F, Q)

        assert to_cpu(result.x).shape == (n_tracks, state_dim)
        assert to_cpu(result.P).shape == (n_tracks, state_dim, state_dim)

    def test_batch_kf_predict_values(self):
        """Test batch KF prediction produces correct values."""
        from pytcl.dynamic_estimation.kalman.linear import kf_predict
        from pytcl.gpu.kalman import batch_kf_predict
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        n_tracks = 5
        state_dim = 4

        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim) * 0.5, (n_tracks, 1, 1))
        F = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float64
        )
        Q = np.eye(state_dim) * 0.1

        # GPU batch prediction
        gpu_result = batch_kf_predict(x, P, F, Q)
        x_pred_gpu = to_cpu(gpu_result.x)
        P_pred_gpu = to_cpu(gpu_result.P)

        # CPU predictions
        for i in range(n_tracks):
            cpu_result = kf_predict(x[i], P[i], F, Q)
            assert_allclose(x_pred_gpu[i], cpu_result.x, rtol=RTOL_TIGHT, atol=ATOL)
            assert_allclose(P_pred_gpu[i], cpu_result.P, rtol=RTOL_TIGHT, atol=ATOL)

    def test_batch_kf_update_shapes(self):
        """Test batch KF update output shapes."""
        from pytcl.gpu.kalman import batch_kf_update
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        n_tracks = 10
        state_dim = 4
        meas_dim = 2

        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim), (n_tracks, 1, 1))
        z = np.random.randn(n_tracks, meas_dim)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        R = np.eye(meas_dim) * 0.5

        result = batch_kf_update(x, P, z, H, R)

        assert to_cpu(result.x).shape == (n_tracks, state_dim)
        assert to_cpu(result.P).shape == (n_tracks, state_dim, state_dim)
        assert to_cpu(result.y).shape == (n_tracks, meas_dim)
        assert to_cpu(result.S).shape == (n_tracks, meas_dim, meas_dim)
        assert to_cpu(result.K).shape == (n_tracks, state_dim, meas_dim)
        assert to_cpu(result.likelihood).shape == (n_tracks,)

    def test_batch_kf_update_values(self):
        """Test batch KF update produces correct values."""
        from pytcl.dynamic_estimation.kalman.linear import kf_update
        from pytcl.gpu.kalman import batch_kf_update
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        n_tracks = 5
        state_dim = 4
        meas_dim = 2

        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim) * 0.5, (n_tracks, 1, 1))
        z = np.random.randn(n_tracks, meas_dim)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        R = np.eye(meas_dim) * 0.5

        # GPU batch update
        gpu_result = batch_kf_update(x, P, z, H, R)
        x_upd_gpu = to_cpu(gpu_result.x)
        P_upd_gpu = to_cpu(gpu_result.P)

        # CPU updates
        for i in range(n_tracks):
            cpu_result = kf_update(x[i], P[i], z[i], H, R)
            assert_allclose(x_upd_gpu[i], cpu_result.x, rtol=RTOL_LOOSE, atol=ATOL)
            assert_allclose(P_upd_gpu[i], cpu_result.P, rtol=RTOL_LOOSE, atol=ATOL)

    def test_cupy_kalman_filter_class(self):
        """Test CuPyKalmanFilter class interface."""
        from pytcl.gpu.kalman import CuPyKalmanFilter
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        state_dim = 4
        meas_dim = 2

        kf = CuPyKalmanFilter(
            state_dim=state_dim,
            meas_dim=meas_dim,
            F=np.array(
                [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                dtype=np.float64,
            ),
            H=np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64),
            Q=np.eye(state_dim) * 0.1,
            R=np.eye(meas_dim) * 0.5,
        )

        n_tracks = 100
        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim), (n_tracks, 1, 1))
        z = np.random.randn(n_tracks, meas_dim)

        # Predict
        x_pred, P_pred = kf.predict(x, P)
        assert to_cpu(x_pred).shape == (n_tracks, state_dim)

        # Update
        result = kf.update(x_pred, P_pred, z)
        assert to_cpu(result.x).shape == (n_tracks, state_dim)


class TestGPUParticleFilter:
    """Tests for GPU-accelerated particle filter."""

    def test_gpu_resample_systematic(self):
        """Test GPU systematic resampling."""
        from pytcl.gpu.particle_filter import gpu_resample_systematic
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        weights = np.array([0.1, 0.3, 0.4, 0.2])
        indices = gpu_resample_systematic(weights)
        indices_cpu = to_cpu(indices)

        assert len(indices_cpu) == len(weights)
        assert all(0 <= idx < len(weights) for idx in indices_cpu)

    def test_gpu_resample_multinomial(self):
        """Test GPU multinomial resampling."""
        from pytcl.gpu.particle_filter import gpu_resample_multinomial
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        weights = np.array([0.1, 0.3, 0.4, 0.2])
        indices = gpu_resample_multinomial(weights)
        indices_cpu = to_cpu(indices)

        assert len(indices_cpu) == len(weights)
        assert all(0 <= idx < len(weights) for idx in indices_cpu)

    def test_gpu_effective_sample_size(self):
        """Test GPU ESS computation."""
        from pytcl.gpu.particle_filter import gpu_effective_sample_size
        from pytcl.gpu.utils import is_gpu_available

        if not is_gpu_available():
            pytest.skip("No GPU available")

        # Uniform weights should give ESS = N
        n = 100
        weights = np.ones(n) / n
        ess = gpu_effective_sample_size(weights)
        assert_allclose(ess, n, rtol=RTOL_TIGHT, atol=ATOL)

        # Single dominant particle should give ESS ≈ 1
        weights = np.zeros(n)
        weights[0] = 1.0
        ess = gpu_effective_sample_size(weights)
        assert_allclose(ess, 1.0, rtol=RTOL_TIGHT, atol=ATOL)

    def test_cupy_particle_filter_class(self):
        """Test CuPyParticleFilter class interface."""
        from pytcl.gpu.particle_filter import CuPyParticleFilter
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        n_particles = 1000
        state_dim = 2

        pf = CuPyParticleFilter(n_particles=n_particles, state_dim=state_dim)

        # Initialize
        pf.initialize(mean=np.zeros(state_dim), cov=np.eye(state_dim))

        particles = pf.get_particles_cpu()
        assert particles.shape == (n_particles, state_dim)

        # Estimate should be close to mean with uniform weights
        estimate = to_cpu(pf.get_estimate())
        assert estimate.shape == (state_dim,)


class TestGPUMatrixUtils:
    """Tests for GPU matrix utilities."""

    def test_gpu_cholesky(self):
        """Test GPU Cholesky decomposition."""
        from pytcl.gpu.matrix_utils import gpu_cholesky
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        # Create positive definite matrix
        A = np.array([[4, 2, 1], [2, 5, 2], [1, 2, 6]], dtype=np.float64)

        L = gpu_cholesky(A)
        L_cpu = to_cpu(L)

        # Verify L @ L.T = A
        assert_allclose(L_cpu @ L_cpu.T, A, rtol=RTOL_TIGHT, atol=ATOL)

    def test_gpu_qr(self):
        """Test GPU QR decomposition."""
        from pytcl.gpu.matrix_utils import gpu_qr
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        A = np.random.randn(4, 3)
        Q, R = gpu_qr(A)
        Q_cpu = to_cpu(Q)
        R_cpu = to_cpu(R)

        # Verify Q @ R = A
        assert_allclose(Q_cpu @ R_cpu, A, rtol=RTOL_TIGHT, atol=ATOL)

        # Verify Q is orthogonal
        # Orthogonality: compare against identity with an ABSOLUTE tolerance.
        # The off-diagonal targets are exactly 0, which no relative tolerance
        # can satisfy.
        assert_allclose(Q_cpu.T @ Q_cpu, np.eye(3), atol=1e-10 if _FLOAT64 else 1e-5)

    def test_gpu_solve(self):
        """Test GPU linear system solve."""
        from pytcl.gpu.matrix_utils import gpu_solve
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        A = np.array([[3, 1], [1, 2]], dtype=np.float64)
        b = np.array([9, 8], dtype=np.float64)

        x = gpu_solve(A, b)
        x_cpu = to_cpu(x)

        # Verify A @ x = b
        assert_allclose(A @ x_cpu, b, rtol=RTOL_TIGHT, atol=ATOL)

    def test_gpu_inv(self):
        """Test GPU matrix inversion."""
        from pytcl.gpu.matrix_utils import gpu_inv
        from pytcl.gpu.utils import is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        A_inv = gpu_inv(A)
        A_inv_cpu = to_cpu(A_inv)

        # Verify A @ A_inv = I
        assert_allclose(A @ A_inv_cpu, np.eye(2), rtol=RTOL_TIGHT, atol=ATOL)

    def test_memory_pool(self):
        """Test memory pool manager."""
        from pytcl.gpu.matrix_utils import get_memory_pool
        from pytcl.gpu.utils import is_gpu_available

        if not is_gpu_available():
            pytest.skip("No GPU available")

        pool = get_memory_pool()
        stats = pool.get_stats()

        assert "used" in stats
        assert "total" in stats
        assert "free" in stats


class TestGPUEKF:
    """Tests for GPU Extended Kalman Filter."""

    def test_batch_ekf_predict(self):
        """Test batch EKF prediction."""
        from pytcl.gpu.ekf import batch_ekf_predict
        from pytcl.gpu.utils import get_array_module, is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        def f(x):
            xp = get_array_module(x)
            return xp.stack([x[:, 0] + x[:, 1], x[:, 1] * 0.99], axis=1)

        def F_jac(x):
            xp = get_array_module(x)
            return xp.broadcast_to(
                xp.array([[1.0, 1.0], [0.0, 0.99]]), (x.shape[0], 2, 2)
            )

        n_tracks = 5
        state_dim = 2

        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim), (n_tracks, 1, 1))
        Q = np.eye(state_dim) * 0.01

        result = batch_ekf_predict(x, P, f, F_jac, Q)

        assert to_cpu(result.x).shape == (n_tracks, state_dim)
        assert to_cpu(result.P).shape == (n_tracks, state_dim, state_dim)


class TestGPUUKF:
    """Tests for GPU Unscented Kalman Filter."""

    def test_batch_ukf_predict(self):
        """Test batch UKF prediction."""
        from pytcl.gpu.ukf import batch_ukf_predict
        from pytcl.gpu.utils import get_array_module, is_gpu_available, to_cpu

        if not is_gpu_available():
            pytest.skip("No GPU available")

        def f(x):
            xp = get_array_module(x)
            return xp.stack([x[:, 0] + x[:, 1], x[:, 1] * 0.99], axis=1)

        n_tracks = 5
        state_dim = 2

        x = np.random.randn(n_tracks, state_dim)
        P = np.tile(np.eye(state_dim), (n_tracks, 1, 1))
        Q = np.eye(state_dim) * 0.01

        result = batch_ukf_predict(x, P, f, Q)

        assert to_cpu(result.x).shape == (n_tracks, state_dim)
        assert to_cpu(result.P).shape == (n_tracks, state_dim, state_dim)
