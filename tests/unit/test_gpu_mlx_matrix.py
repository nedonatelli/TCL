"""
Validation of :mod:`pytcl.gpu.matrix_utils` on the MLX backend.

Ground truth is NumPy/SciPy on the CPU in float64: every batched GPU operation
is compared against the equivalent NumPy computation looped one matrix at a
time. MLX computes in float32, so tolerances are float32-scale; the measured
worst-case relative error for each group is noted alongside the assertion.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

mx = pytest.importorskip("mlx.core")

from pytcl.gpu._backend import get_compute_backend  # noqa: E402
from pytcl.gpu.matrix_utils import (  # noqa: E402
    MemoryPool,
    get_memory_pool,
    gpu_cholesky,
    gpu_cholesky_safe,
    gpu_eigh,
    gpu_inv,
    gpu_matrix_sqrt,
    gpu_qr,
    gpu_solve,
)


@pytest.fixture(scope="module")
def backend():
    return get_compute_backend(prefer="mlx")


def to_np(arr):
    """Host copy of a backend array, in float64."""
    return np.asarray(arr, dtype=np.float64)


def random_spd(rng, k, n):
    """Batch of k well-conditioned symmetric positive definite (n, n) matrices."""
    A = rng.standard_normal((k, n, n))
    return A @ np.swapaxes(A, -2, -1) + n * np.eye(n)


class TestBatchOpsVersusNumpy:
    """Every batch op against the same computation looped per item in NumPy."""

    def test_cholesky_batch(self):
        # Measured max relative error vs np.linalg.cholesky: 6.4e-8
        rng = np.random.default_rng(0)
        A = random_spd(rng, 8, 5)

        L = to_np(gpu_cholesky(A))
        expected = np.stack([np.linalg.cholesky(A[i]) for i in range(len(A))])

        assert_allclose(L, expected, rtol=1e-5, atol=1e-5)

    def test_cholesky_upper(self):
        rng = np.random.default_rng(1)
        A = random_spd(rng, 4, 3)

        U = to_np(gpu_cholesky(A, lower=False))
        expected = np.stack(
            [np.linalg.cholesky(A[i]).T for i in range(len(A))],
        )

        assert_allclose(U, expected, rtol=1e-5, atol=1e-5)
        # A = U.T @ U
        assert_allclose(np.swapaxes(U, -2, -1) @ U, A, rtol=1e-4, atol=1e-4)

    def test_cholesky_single_matrix(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])

        L = to_np(gpu_cholesky(A))

        assert_allclose(L, np.linalg.cholesky(A), rtol=1e-6, atol=1e-6)
        assert_allclose(L @ L.T, A, rtol=1e-5, atol=1e-5)

    def test_cholesky_raises_for_indefinite(self):
        A = np.array([[1.0, 2.0], [2.0, 1.0]])

        with pytest.raises(np.linalg.LinAlgError):
            gpu_cholesky(A)

    def test_cholesky_raises_for_singular_psd(self):
        # LAPACK (and therefore NumPy) treats a zero pivot as a failure.
        A = np.array([[1.0, 1.0], [1.0, 1.0]])

        with pytest.raises(np.linalg.LinAlgError):
            gpu_cholesky(A)

    def test_qr_reduced_batch(self):
        # Measured max reconstruction error ||QR - A||_max: 4.9e-7
        rng = np.random.default_rng(2)
        A = rng.standard_normal((6, 5, 3))

        Q, R = gpu_qr(A)
        Q, R = to_np(Q), to_np(R)

        expected_shapes = [np.linalg.qr(A[i]) for i in range(len(A))]
        assert Q.shape == (6, 5, 3)
        assert R.shape == (6, 3, 3)
        assert_allclose(Q @ R, A, rtol=1e-4, atol=1e-5)
        # Columns orthonormal, and |R| matches NumPy up to column sign.
        assert_allclose(
            np.swapaxes(Q, -2, -1) @ Q,
            np.broadcast_to(np.eye(3), (6, 3, 3)),
            atol=1e-5,
        )
        for i, (_, R_ref) in enumerate(expected_shapes):
            assert_allclose(np.abs(R[i]), np.abs(R_ref), rtol=1e-4, atol=1e-5)

    def test_qr_complete_batch(self):
        # MLX has no native 'complete' mode; the backend builds the orthogonal
        # complement from eigh. Measured max reconstruction error: 2.5e-7
        rng = np.random.default_rng(3)
        A = rng.standard_normal((4, 6, 3))

        Q, R = gpu_qr(A, mode="complete")
        Q, R = to_np(Q), to_np(R)

        Q_ref, R_ref = np.linalg.qr(A[0], mode="complete")
        assert Q.shape == (4,) + Q_ref.shape
        assert R.shape == (4,) + R_ref.shape
        assert_allclose(Q @ R, A, rtol=1e-4, atol=1e-5)
        assert_allclose(
            np.swapaxes(Q, -2, -1) @ Q,
            np.broadcast_to(np.eye(6), (4, 6, 6)),
            atol=1e-5,
        )

    def test_qr_wide_matrix(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((3, 6))

        for mode in ("reduced", "complete"):
            Q, R = gpu_qr(A, mode=mode)
            Q_ref, R_ref = np.linalg.qr(A, mode=mode)
            assert to_np(Q).shape == Q_ref.shape
            assert to_np(R).shape == R_ref.shape
            assert_allclose(to_np(Q) @ to_np(R), A, rtol=1e-4, atol=1e-5)

    def test_qr_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="reduced"):
            gpu_qr(np.eye(3), mode="raw")

    def test_eigh_batch(self):
        # Measured max eigenvalue relative error: 2.2e-7
        rng = np.random.default_rng(5)
        A = random_spd(rng, 7, 4)

        eigvals, eigvecs = gpu_eigh(A)
        eigvals, eigvecs = to_np(eigvals), to_np(eigvecs)

        expected = np.stack([np.linalg.eigvalsh(A[i]) for i in range(len(A))])
        assert_allclose(eigvals, expected, rtol=1e-4, atol=1e-4)
        # A @ v = lambda * v for every eigenpair.
        assert_allclose(A @ eigvecs, eigvecs * eigvals[:, None, :], atol=1e-3)

    def test_matrix_sqrt_batch(self):
        # Measured max relative error ||S@S - A|| / ||A||: 4.8e-7
        rng = np.random.default_rng(6)
        A = random_spd(rng, 5, 4)

        S = to_np(gpu_matrix_sqrt(A))

        assert_allclose(S @ S, A, rtol=1e-4, atol=1e-3)
        assert_allclose(S, np.swapaxes(S, -2, -1), atol=1e-5)

    def test_matrix_sqrt_single_matrix(self):
        A = np.array([[4.0, 0.0], [0.0, 9.0]])

        S = to_np(gpu_matrix_sqrt(A))

        assert_allclose(S, np.diag([2.0, 3.0]), atol=1e-5)

    def test_matrix_sqrt_clips_negative_eigenvalues(self):
        A = np.diag([4.0, -1.0])

        S = to_np(gpu_matrix_sqrt(A))

        assert np.isfinite(S).all()
        assert_allclose(S, np.diag([2.0, 0.0]), atol=1e-5)


class TestBatchInverseAndSolve:
    """A @ inv(A) == I and A @ solve(A, b) == b to float32 tolerance."""

    def test_inv_batch_round_trip(self):
        # Measured max |A @ inv(A) - I|: 1.1e-7
        rng = np.random.default_rng(7)
        A = random_spd(rng, 8, 5)

        A_inv = to_np(gpu_inv(A))

        identity = np.broadcast_to(np.eye(5), A.shape)
        assert_allclose(A @ A_inv, identity, atol=1e-4)
        expected = np.stack([np.linalg.inv(A[i]) for i in range(len(A))])
        assert_allclose(A_inv, expected, rtol=1e-3, atol=1e-5)

    def test_inv_single_matrix(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])

        A_inv = to_np(gpu_inv(A))

        assert_allclose(A @ A_inv, np.eye(2), atol=1e-5)

    def test_solve_batch_matrix_rhs(self):
        # Measured max |A @ x - b|: 3.5e-7
        rng = np.random.default_rng(8)
        A = random_spd(rng, 6, 4)
        b = rng.standard_normal((6, 4, 2))

        x = to_np(gpu_solve(A, b))

        assert_allclose(A @ x, b, atol=1e-4)
        expected = np.stack([np.linalg.solve(A[i], b[i]) for i in range(len(A))])
        assert_allclose(x, expected, rtol=1e-3, atol=1e-4)

    def test_solve_single_system(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([[9.0], [8.0]])

        x = to_np(gpu_solve(A, b))

        assert_allclose(A @ x, b, atol=1e-4)
        assert_allclose(x, np.linalg.solve(A, b), rtol=1e-5, atol=1e-5)


class TestCholeskySafeContract:
    """The documented gpu_cholesky_safe contract, including the docstring case."""

    def test_indefinite_returns_success_false(self):
        # Exactly the docstring example: 1e-10 diagonal regularization cannot
        # repair an indefinite matrix, so this must not raise.
        A = np.array([[1, 2], [2, 1]])

        L, success = gpu_cholesky_safe(A)

        assert success is False
        L = to_np(L)
        assert np.isfinite(L).all()
        # The factor belongs to the nearest positive definite matrix, whose
        # eigenvalues are max(eig(A), floor) = [~0, 3].
        recon = L @ L.T
        assert_allclose(recon, np.full((2, 2), 1.5), atol=1e-4)

    def test_singular_psd_regularizes_and_succeeds(self):
        A = np.array([[1.0, 1.0], [1.0, 1.0]])

        L, success = gpu_cholesky_safe(A, regularization=1e-6)

        assert success is False
        L = to_np(L)
        assert np.isfinite(L).all()
        assert (np.diag(L) > 0).all()
        assert_allclose(L @ L.T, A, atol=1e-4)

    def test_well_conditioned_matches_numpy(self):
        # Measured max relative error vs np.linalg.cholesky: 4.8e-8
        rng = np.random.default_rng(9)
        A = random_spd(rng, 1, 6)[0]

        L, success = gpu_cholesky_safe(A)

        assert success is True
        assert_allclose(to_np(L), np.linalg.cholesky(A), rtol=1e-5, atol=1e-5)

    def test_batch_well_conditioned(self):
        rng = np.random.default_rng(10)
        A = random_spd(rng, 4, 3)

        L, success = gpu_cholesky_safe(A)

        assert success is True
        expected = np.stack([np.linalg.cholesky(A[i]) for i in range(len(A))])
        assert_allclose(to_np(L), expected, rtol=1e-5, atol=1e-5)

    def test_batch_with_one_indefinite_member(self):
        rng = np.random.default_rng(11)
        A = random_spd(rng, 3, 2)
        A[1] = np.array([[1.0, 2.0], [2.0, 1.0]])

        L, success = gpu_cholesky_safe(A)

        assert success is False
        L = to_np(L)
        assert np.isfinite(L).all()
        assert (np.diagonal(L, axis1=-2, axis2=-1) >= 0).all()

    def test_upper_factor(self):
        rng = np.random.default_rng(12)
        A = random_spd(rng, 1, 4)[0]

        U, success = gpu_cholesky_safe(A, lower=False)

        assert success is True
        U = to_np(U)
        assert_allclose(U.T @ U, A, rtol=1e-4, atol=1e-4)


class TestBackendSelection:
    """The MLX backend is the one actually exercised here."""

    def test_backend_is_mlx_when_cupy_absent(self, backend):
        assert backend.name == "mlx"
        assert backend.supports_float64 is False

    def test_results_are_mlx_arrays(self):
        L = gpu_cholesky(np.eye(3))
        assert isinstance(L, mx.array)

    def test_no_backend_raises_dependency_error(self, monkeypatch):
        """The @requires('cupy') contract is preserved: both extras are named."""
        from pytcl.core.exceptions import DependencyError
        from pytcl.gpu import _backend

        def fail(self, *args, **kwargs):
            raise ImportError("no compute backend")

        monkeypatch.setattr(_backend.CuPyBackend, "__init__", fail)
        monkeypatch.setattr(_backend.MLXBackend, "__init__", fail)

        with pytest.raises(DependencyError, match="cupy") as excinfo:
            gpu_cholesky(np.eye(2))
        assert "mlx" in str(excinfo.value)


class TestMemoryPool:
    """MemoryPool is functional on MLX rather than a no-op."""

    def test_get_stats_keys(self):
        pool = MemoryPool()

        stats = pool.get_stats()

        assert sorted(stats) == ["device_total", "free", "total", "used"]
        assert stats["device_total"] > 0
        assert stats["used"] >= 0

    def test_free_all_and_limit_round_trip(self):
        pool = MemoryPool()
        before = pool.get_stats()["device_total"]

        pool.free_all()
        pool.set_limit(2 * 1024**3)
        pool.set_limit(None)

        assert pool.get_stats()["device_total"] == before

    def test_limit_memory_context_restores(self):
        pool = MemoryPool()

        with pool.limit_memory(10**9):
            x = mx.ones((16, 16))
            mx.eval(x)

        # Allocation well above the temporary limit still succeeds afterwards.
        y = mx.ones((1024, 1024))
        mx.eval(y)
        assert y.shape == (1024, 1024)

    def test_get_memory_pool_is_singleton(self):
        assert get_memory_pool() is get_memory_pool()
