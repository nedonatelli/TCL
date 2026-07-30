"""Correctness audit for pytcl.gpu (v2 verification).

Two independent layers are verified:

1. MLX layer (runs for real on this Apple Silicon machine): backend
   detection, array transfer (``to_gpu``/``to_cpu``/``ensure_gpu_array``),
   ``get_array_module``, and the memory/sync utilities. Float32 roundtrips
   are exact; float64 inputs are downcast to float32 by MLX (documented
   lossy, relative error bounded by float32 eps / 2 ~= 6e-8).

2. CuPy-gated algorithm layer: every batch filter, particle filter, and
   matrix utility is decorated ``@requires("cupy")`` and cannot execute
   here (no NVIDIA GPU). The *algorithms* are validated by substituting
   numpy (whose API the code consumes identically) for cupy and bypassing
   device transfer, then comparing against the reference-validated CPU
   implementations in ``pytcl.dynamic_estimation``. Device execution on
   real CuPy hardware remains hardware-gated.

Observed numerical behavior (float64 shim): batch KF/EKF match the CPU
reference to ~1e-15 (bit-level up to summation order). Batch UKF matches
to ~3e-9 at the default alpha=1e-3 because Merwe weights are O(1/alpha^2)
~= 1e6, amplifying float64 eps; the discrepancy shrinks to 0 as alpha -> 1
(verified), i.e. it is precision-limited, not an algorithmic bias.
"""

import sys
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pytcl.gpu.ekf as gpu_ekf
import pytcl.gpu.kalman as gpu_kalman
import pytcl.gpu.matrix_utils as gpu_matrix_utils
import pytcl.gpu.particle_filter as gpu_pf
import pytcl.gpu.ukf as gpu_ukf
from pytcl.core import optional_deps
from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import is_available
from pytcl.dynamic_estimation.kalman.extended import ekf_predict, ekf_update
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_estimation.kalman.unscented import ukf_predict, ukf_update
from pytcl.gpu import utils as gpu_utils

mx = pytest.importorskip("mlx.core", reason="MLX required for GPU audit")

_HAS_REAL_CUPY = is_available("cupy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clear_detection_caches() -> None:
    gpu_utils.is_apple_silicon.cache_clear()
    gpu_utils.is_mlx_available.cache_clear()
    gpu_utils.is_cupy_available.cache_clear()
    gpu_utils.get_backend.cache_clear()
    gpu_utils.is_gpu_available.cache_clear()


@pytest.fixture
def detection_caches():
    """Clear lru_cache'd detection results before and after a test."""
    _clear_detection_caches()
    yield
    _clear_detection_caches()


class _CupyShim(types.ModuleType):
    """numpy masquerading as cupy (their APIs coincide for the ops used)."""

    def __getattr__(self, name):
        if name == "asnumpy":
            # CuPy's device->host transfer; a no-op for numpy arrays.
            return np.asarray
        return getattr(np, name)


@pytest.fixture
def numpy_cupy(monkeypatch):
    """Run the CuPy-gated code paths on numpy, bypassing device transfer.

    This validates the batch algorithms themselves; device execution is
    hardware-gated and untestable on this machine.
    """
    shim = _CupyShim("cupy")
    monkeypatch.setitem(sys.modules, "cupy", shim)
    monkeypatch.setitem(optional_deps._availability_cache, "cupy", True)

    def ensure_np(arr, dtype=np.float64, backend=None):
        return np.asarray(arr, dtype=dtype)

    def to_cpu_np(arr):
        return np.asarray(arr)

    # Modules ported to the backend-dispatch layer no longer import
    # ensure_gpu_array; there the shim above is picked up by CuPyBackend, which
    # imports cupy (i.e. numpy) directly.
    for mod in (gpu_kalman, gpu_ekf, gpu_ukf, gpu_pf, gpu_matrix_utils):
        monkeypatch.setattr(mod, "ensure_gpu_array", ensure_np, raising=False)
        if hasattr(mod, "to_cpu"):
            monkeypatch.setattr(mod, "to_cpu", to_cpu_np)
    yield shim


def _raise_import_error(self, *args, **kwargs):
    raise ImportError("no compute backend")


def _random_spd(rng, n, scale=1.0, jitter=1.0):
    a = rng.normal(size=(n, n)) * scale
    return a @ a.T + jitter * np.eye(n)


# ---------------------------------------------------------------------------
# Backend detection (real MLX machine)
# ---------------------------------------------------------------------------


class TestBackendDetectionTruthful:
    def test_real_machine_detection(self, detection_caches):
        """On this arm64 Mac with MLX installed, detection must be truthful."""
        assert gpu_utils.is_apple_silicon() is True
        assert gpu_utils.is_mlx_available() is True
        assert gpu_utils.is_gpu_available() is True
        assert gpu_utils.get_backend() == "mlx"

    def test_cupy_detection_matches_reality(self, detection_caches):
        assert gpu_utils.is_cupy_available() is _HAS_REAL_CUPY

    def test_detection_without_mlx(self, detection_caches, monkeypatch):
        """Simulate MLX not being importable: everything degrades to numpy."""
        monkeypatch.setitem(optional_deps._availability_cache, "mlx", False)
        monkeypatch.setitem(optional_deps._availability_cache, "cupy", False)
        assert gpu_utils.is_mlx_available() is False
        assert gpu_utils.get_backend() == "numpy"
        assert gpu_utils.is_gpu_available() is False
        with pytest.raises(RuntimeError, match="No GPU available"):
            gpu_utils.to_gpu(np.zeros(3))

    def test_detection_on_non_apple_platform(self, detection_caches, monkeypatch):
        """MLX must not be reported available off Apple Silicon."""
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        assert gpu_utils.is_apple_silicon() is False
        assert gpu_utils.is_mlx_available() is False

    def test_lazy_module_exports(self):
        import pytcl.gpu as gpu

        assert callable(gpu.batch_kf_predict)
        assert callable(gpu.batch_ekf_update)
        assert callable(gpu.batch_ukf_predict)
        assert callable(gpu.gpu_resample_systematic)
        assert gpu.MemoryPool is gpu_matrix_utils.MemoryPool
        with pytest.raises(AttributeError):
            gpu.no_such_symbol


# ---------------------------------------------------------------------------
# Array transfer on the real MLX backend
# ---------------------------------------------------------------------------


class TestMLXTransfer:
    def test_float32_roundtrip_exact(self):
        x = np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32)
        g = gpu_utils.to_gpu(x)
        assert isinstance(g, mx.array)
        assert g.dtype == mx.float32
        back = gpu_utils.to_cpu(g)
        assert back.dtype == np.float32
        assert np.array_equal(back, x)

    def test_float64_roundtrip_documented_lossy(self):
        """MLX silently downcasts float64 -> float32; error <= eps32/2 rel."""
        x = np.random.default_rng(1).normal(size=(100,))
        g = gpu_utils.to_gpu(x)
        assert g.dtype == mx.float32
        back = gpu_utils.to_cpu(g)
        rel = np.abs(back - x) / np.maximum(np.abs(x), 1e-30)
        assert rel.max() < 1.2e-7  # float32 rounding, not algorithmic loss
        # And the loss is exactly the float32 cast, no more
        assert np.array_equal(back, x.astype(np.float32))

    def test_to_gpu_explicit_dtype(self):
        x = np.arange(4, dtype=np.float64)
        g = gpu_utils.to_gpu(x, dtype=np.float32)
        assert g.dtype == mx.float32
        # requesting float64 on MLX maps to float32 (documented)
        g64 = gpu_utils.to_gpu(g, dtype=np.float64)
        assert g64.dtype == mx.float32

    def test_to_gpu_passthrough_for_mlx_input(self):
        g = mx.array([1.0, 2.0])
        assert gpu_utils.to_gpu(g) is g

    def test_int_and_bool_roundtrip(self):
        i = np.array([1, 2, 3])
        gi = gpu_utils.to_gpu(i)
        assert np.array_equal(gpu_utils.to_cpu(gi), i)
        b = np.array([True, False, True])
        gb = gpu_utils.to_gpu(b)
        assert np.array_equal(gpu_utils.to_cpu(gb), b)

    def test_ensure_gpu_array(self):
        e = gpu_utils.ensure_gpu_array(np.arange(6).reshape(2, 3))
        assert isinstance(e, mx.array)
        assert e.dtype == mx.float32  # float64 request maps to float32 on MLX
        e2 = gpu_utils.ensure_gpu_array(e)
        assert isinstance(e2, mx.array)
        assert_allclose(gpu_utils.to_cpu(e2), np.arange(6).reshape(2, 3))

    def test_get_array_module(self):
        assert gpu_utils.get_array_module(np.zeros(2)) is np
        assert gpu_utils.get_array_module(mx.array([1.0])) is mx

    def test_to_cpu_numpy_passthrough(self):
        x = np.zeros(3)
        assert gpu_utils.to_cpu(x) is x

    def test_sync_and_memory_utilities(self):
        # Force some real GPU work, then the utilities must not error and
        # must report the mlx backend.
        a = mx.random.normal((256, 256))
        b = a @ a
        mx.eval(b)
        gpu_utils.sync_gpu()
        info = gpu_utils.get_gpu_memory_info()
        assert info["backend"] == "mlx"
        gpu_utils.clear_gpu_memory()

    def test_memory_pool_uses_mlx_without_cupy(self):
        """Issue #12: MemoryPool reports real MLX numbers instead of no-op zeros."""
        if _HAS_REAL_CUPY:
            pytest.skip("CuPy present; MLX path not reachable")
        pool = gpu_matrix_utils.MemoryPool()
        stats = pool.get_stats()
        assert sorted(stats) == ["device_total", "free", "total", "used"]
        assert stats["device_total"] > 0
        assert stats["used"] >= 0
        pool.free_all()
        with pool.limit_memory(1 << 30):
            pass
        pool.set_limit(None)
        assert isinstance(
            gpu_matrix_utils.get_memory_pool(), gpu_matrix_utils.MemoryPool
        )


# ---------------------------------------------------------------------------
# The advertised-but-absent MLX batch backend (documents the gap)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_REAL_CUPY, reason="only meaningful without CuPy")
class TestBatchOpsRunOnMLX:
    """The gap this class used to document is closed: the batch filters, the
    particle filter, and the matrix utilities are backend-dispatched and run on
    MLX here. Without a compute backend at all they still raise
    DependencyError, naming both extras."""

    def test_particle_filter_runs_on_mlx(self):
        """Issue #12: the PF is backend-dispatched, no CuPy required."""
        idx = np.asarray(gpu_pf.gpu_resample_systematic(np.full(4, 0.25)))
        assert idx.shape == (4,)
        assert idx.min() >= 0 and idx.max() < 4

    def test_no_backend_raises_dependency_error(self, monkeypatch):
        from pytcl.gpu import _backend

        monkeypatch.setattr(
            _backend.CuPyBackend, "__init__", _raise_import_error, raising=True
        )
        monkeypatch.setattr(
            _backend.MLXBackend, "__init__", _raise_import_error, raising=True
        )
        with pytest.raises(DependencyError, match="cupy"):
            gpu_pf.gpu_resample_systematic(np.full(4, 0.25))


# ---------------------------------------------------------------------------
# Batch KF algorithm vs CPU reference (numpy-as-cupy shim)
# ---------------------------------------------------------------------------


ATOL = 1e-10  # float64 shim; observed max deviation ~3e-15


class TestBatchKalmanMath:
    @pytest.fixture(autouse=True)
    def _shim(self, numpy_cupy):
        pass

    def _problem(self, seed=42, n_tracks=25, n=4, m=2):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n_tracks, n))
        P = np.stack([_random_spd(rng, n) for _ in range(n_tracks)])
        F = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
        )
        Q = np.eye(n) * 0.1
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
        R = np.eye(m) * 0.5
        z = rng.normal(size=(n_tracks, m))
        return x, P, F, Q, H, R, z

    def test_predict_matches_cpu_loop(self):
        x, P, F, Q, _, _, _ = self._problem()
        pred = gpu_kalman.batch_kf_predict(x, P, F, Q)
        for i in range(len(x)):
            ref = kf_predict(x[i], P[i], F, Q)
            assert_allclose(pred.x[i], ref.x, atol=ATOL)
            assert_allclose(pred.P[i], ref.P, atol=ATOL)

    def test_predict_with_control_input(self):
        x, P, F, Q, _, _, _ = self._problem()
        rng = np.random.default_rng(0)
        B = rng.normal(size=(4, 2))
        u = rng.normal(size=(len(x), 2))
        pred = gpu_kalman.batch_kf_predict(x, P, F, Q, B=B, u=u)
        for i in range(len(x)):
            ref = kf_predict(x[i], P[i], F, Q, B=B, u=u[i])
            assert_allclose(pred.x[i], ref.x, atol=ATOL)

    def test_predict_per_track_F_and_Q(self):
        x, P, F, Q, _, _, _ = self._problem(n_tracks=8)
        rng = np.random.default_rng(5)
        F_batch = np.stack([F + rng.normal(size=F.shape) * 0.01 for _ in range(8)])
        Q_batch = np.stack([_random_spd(rng, 4, 0.1, 0.01) for _ in range(8)])
        pred = gpu_kalman.batch_kf_predict(x, P, F_batch, Q_batch)
        for i in range(8):
            ref = kf_predict(x[i], P[i], F_batch[i], Q_batch[i])
            assert_allclose(pred.x[i], ref.x, atol=ATOL)
            assert_allclose(pred.P[i], ref.P, atol=ATOL)

    def test_update_matches_cpu_loop(self):
        x, P, F, Q, H, R, z = self._problem()
        upd = gpu_kalman.batch_kf_update(x, P, z, H, R)
        for i in range(len(x)):
            ref = kf_update(x[i], P[i], z[i], H, R)
            assert_allclose(upd.x[i], ref.x, atol=ATOL)
            assert_allclose(upd.P[i], ref.P, atol=ATOL)
            assert_allclose(upd.y[i], ref.y, atol=ATOL)
            assert_allclose(upd.S[i], ref.S, atol=ATOL)
            assert_allclose(upd.K[i], ref.K, atol=ATOL)
            assert_allclose(upd.likelihood[i], ref.likelihood, atol=1e-12)

    def test_predict_update_composition(self):
        x, P, F, Q, H, R, z = self._problem()
        combined = gpu_kalman.batch_kf_predict_update(x, P, z, F, Q, H, R)
        pred = gpu_kalman.batch_kf_predict(x, P, F, Q)
        upd = gpu_kalman.batch_kf_update(pred.x, pred.P, z, H, R)
        assert_allclose(combined.x, upd.x, atol=ATOL)
        assert_allclose(combined.P, upd.P, atol=ATOL)

    def test_kalman_filter_class(self):
        x, P, F, Q, H, R, z = self._problem(n_tracks=10)
        kf = gpu_kalman.CuPyKalmanFilter(state_dim=4, meas_dim=2, F=F, H=H, Q=Q, R=R)
        x_pred, P_pred = kf.predict(x, P)
        result = kf.update(x_pred, P_pred, z)
        for i in range(10):
            p = kf_predict(x[i], P[i], F, Q)
            u = kf_update(p.x, p.P, z[i], H, R)
            assert_allclose(result.x[i], u.x, atol=ATOL)
            assert_allclose(result.P[i], u.P, atol=ATOL)

    def test_updated_covariance_is_symmetric_psd(self):
        x, P, F, Q, H, R, z = self._problem()
        upd = gpu_kalman.batch_kf_update(x, P, z, H, R)
        P_upd = np.asarray(upd.P)
        assert_allclose(P_upd, np.swapaxes(P_upd, -2, -1), atol=ATOL)
        eigs = np.linalg.eigvalsh(P_upd)
        assert eigs.min() > 0


# ---------------------------------------------------------------------------
# Batch EKF algorithm vs CPU reference
# ---------------------------------------------------------------------------


def _f_ct(x):
    w = 0.05
    return np.array(
        [x[0] + np.cos(w) * x[2], x[1] + np.sin(w) * x[3], 0.99 * x[2], 0.99 * x[3]]
    )


def _F_ct(x):
    w = 0.05
    return np.array(
        [
            [1, 0, np.cos(w), 0],
            [0, 1, 0, np.sin(w)],
            [0, 0, 0.99, 0],
            [0, 0, 0, 0.99],
        ]
    )


def _h_polar(x):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def _H_polar(x):
    r2 = x[0] ** 2 + x[1] ** 2
    r = np.sqrt(r2)
    return np.array([[x[0] / r, x[1] / r, 0, 0], [-x[1] / r2, x[0] / r2, 0, 0]])


def _ekf_problem(seed=7, n_tracks=15):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_tracks, 4)) + np.array([10.0, 10.0, 1.0, 1.0])
    P = np.stack([_random_spd(rng, 4, 0.3, 0.5) for _ in range(n_tracks)])
    Q = np.eye(4) * 0.01
    R = np.diag([0.1, 0.01])
    z = np.array([_h_polar(xi) for xi in x])
    z += rng.normal(size=z.shape) * [0.3, 0.03]
    return x, P, Q, R, z


class TestBatchEKFMath:
    @pytest.fixture(autouse=True)
    def _shim(self, numpy_cupy):
        pass

    def test_predict_matches_cpu_loop(self):
        x, P, Q, _, _ = _ekf_problem()
        pred = gpu_ekf.batch_ekf_predict(x, P, _f_ct, _F_ct, Q)
        for i in range(len(x)):
            ref = ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q)
            assert_allclose(pred.x[i], ref.x, atol=ATOL)
            assert_allclose(pred.P[i], ref.P, atol=ATOL)

    def test_numerical_jacobian_close_to_analytic(self):
        x, P, Q, _, _ = _ekf_problem()
        analytic = gpu_ekf.batch_ekf_predict(x, P, _f_ct, _F_ct, Q)
        numeric = gpu_ekf.batch_ekf_predict(x, P, _f_ct, None, Q)
        assert_allclose(numeric.P, analytic.P, atol=1e-6)

    def test_update_matches_cpu_loop(self):
        x, P, Q, R, z = _ekf_problem()
        upd = gpu_ekf.batch_ekf_update(x, P, z, _h_polar, _H_polar, R)
        for i in range(len(x)):
            ref = ekf_update(x[i], P[i], z[i], _h_polar, _H_polar(x[i]), R)
            assert_allclose(upd.x[i], ref.x, atol=ATOL)
            assert_allclose(upd.P[i], ref.P, atol=ATOL)
            assert_allclose(upd.likelihood[i], ref.likelihood, atol=1e-12)

    def test_ekf_class_predict_update(self):
        x, P, Q, R, z = _ekf_problem(n_tracks=6)
        ekf = gpu_ekf.CuPyExtendedKalmanFilter(
            state_dim=4,
            meas_dim=2,
            f=_f_ct,
            h=_h_polar,
            F_jacobian=_F_ct,
            H_jacobian=_H_polar,
            Q=Q,
            R=R,
        )
        result = ekf.predict_update(x, P, z)
        for i in range(6):
            p = ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q)
            u = ekf_update(p.x, p.P, z[i], _h_polar, _H_polar(p.x), R)
            assert_allclose(result.x[i], u.x, atol=ATOL)
            assert_allclose(result.P[i], u.P, atol=ATOL)


# ---------------------------------------------------------------------------
# Batch UKF algorithm vs CPU reference
# ---------------------------------------------------------------------------


class TestBatchUKFMath:
    @pytest.fixture(autouse=True)
    def _shim(self, numpy_cupy):
        pass

    # Default alpha=1e-3 gives Merwe weights of magnitude ~1e6, which
    # amplifies float64 rounding to ~1e-8 absolute; alpha=0.5 is well
    # conditioned and matches tightly. Both compare against the CPU UKF.
    @pytest.mark.parametrize("alpha,tol", [(1e-3, 1e-6), (0.5, 1e-10)])
    def test_predict_matches_cpu_loop(self, alpha, tol):
        x, P, Q, _, _ = _ekf_problem(seed=11)
        pred = gpu_ukf.batch_ukf_predict(x, P, _f_ct, Q, alpha=alpha)
        for i in range(len(x)):
            ref = ukf_predict(x[i], P[i], _f_ct, Q, alpha=alpha)
            assert_allclose(pred.x[i], ref.x, atol=tol)
            assert_allclose(pred.P[i], ref.P, atol=tol)

    @pytest.mark.parametrize("alpha,tol", [(1e-3, 1e-6), (0.5, 1e-10)])
    def test_update_matches_cpu_loop(self, alpha, tol):
        x, P, Q, R, z = _ekf_problem(seed=13)
        upd = gpu_ukf.batch_ukf_update(x, P, z, _h_polar, R, alpha=alpha)
        for i in range(len(x)):
            ref = ukf_update(x[i], P[i], z[i], _h_polar, R, alpha=alpha)
            assert_allclose(upd.x[i], ref.x, atol=tol)
            assert_allclose(upd.P[i], ref.P, atol=tol)
            assert_allclose(upd.y[i], ref.y, atol=tol)
            assert_allclose(upd.S[i], ref.S, atol=tol)
            assert_allclose(upd.likelihood[i], ref.likelihood, atol=1e-6)

    def test_discrepancy_shrinks_with_weight_conditioning(self):
        """Error must be precision-limited (falls as alpha grows), not bias."""
        x, P, Q, _, _ = _ekf_problem(seed=17)
        errs = []
        for alpha in (1e-3, 1e-1, 1.0):
            pred = gpu_ukf.batch_ukf_predict(x, P, _f_ct, Q, alpha=alpha)
            err = max(
                np.abs(
                    pred.x[i] - ukf_predict(x[i], P[i], _f_ct, Q, alpha=alpha).x
                ).max()
                for i in range(len(x))
            )
            errs.append(err)
        assert errs[1] < errs[0]
        assert errs[2] <= errs[1]

    def test_sigma_point_eigh_fallback_batch(self):
        """Regression: the non-PD fallback used cp.diag on batched eigvals
        (shape error) and a transposed factor that was not a square root."""
        n_tracks, n = 3, 4
        P_sing = np.tile(np.diag([1.0, 1.0, 1.0, 0.0]), (n_tracks, 1, 1))
        x = np.zeros((n_tracks, n))
        alpha, kappa = 0.5, 0.0
        sigma = np.asarray(gpu_ukf._generate_sigma_points(x, P_sing, alpha, kappa))
        assert sigma.shape == (n_tracks, 2 * n + 1, n)
        # Recovered covariance from sigma points must equal the clamped P
        lambda_ = alpha**2 * (n + kappa) - n
        spread = sigma[:, 1 : n + 1, :] - x[:, None, :]  # +gamma * L columns
        recovered = np.einsum("nji,njk->nik", spread, spread) / (n + lambda_)
        P_clamped = np.tile(np.diag([1.0, 1.0, 1.0, 1e-10]), (n_tracks, 1, 1))
        assert_allclose(recovered, P_clamped, atol=1e-8)

    def test_ukf_class_matches_functions(self):
        x, P, Q, R, z = _ekf_problem(seed=19, n_tracks=5)
        ukf = gpu_ukf.CuPyUnscentedKalmanFilter(
            state_dim=4, meas_dim=2, f=_f_ct, h=_h_polar, Q=Q, R=R, alpha=0.5
        )
        result = ukf.predict_update(x, P, z)
        pred = gpu_ukf.batch_ukf_predict(x, P, _f_ct, Q, alpha=0.5)
        upd = gpu_ukf.batch_ukf_update(pred.x, pred.P, z, _h_polar, R, alpha=0.5)
        assert_allclose(result.x, upd.x, atol=ATOL)
        assert_allclose(result.P, upd.P, atol=ATOL)


# ---------------------------------------------------------------------------
# Particle filter algorithms
# ---------------------------------------------------------------------------


class TestParticleFilterMath:
    @pytest.fixture(autouse=True)
    def _shim(self, numpy_cupy):
        pass

    def test_normalize_weights_vs_direct(self):
        rng = np.random.default_rng(2)
        log_w = rng.normal(size=500) * 10  # wide dynamic range
        weights, log_lik = gpu_pf.gpu_normalize_weights(log_w)
        weights = np.asarray(weights)
        # Direct (stable) reference
        ref = np.exp(log_w - log_w.max())
        ref /= ref.sum()
        assert_allclose(weights, ref, atol=1e-14)
        assert_allclose(weights.sum(), 1.0, atol=1e-12)
        ref_ll = log_w.max() + np.log(np.exp(log_w - log_w.max()).sum())
        assert_allclose(log_lik, ref_ll, atol=1e-10)

    def test_effective_sample_size(self):
        w = np.full(100, 0.01)
        assert_allclose(gpu_pf.gpu_effective_sample_size(w), 100.0)
        w2 = np.zeros(100)
        w2[0] = 1.0
        assert_allclose(gpu_pf.gpu_effective_sample_size(w2), 1.0)

    def test_systematic_resampling_low_variance_property(self):
        """Systematic resampling guarantees |count_i - n*w_i| < 1."""
        rng = np.random.default_rng(3)
        w = rng.random(50)
        w /= w.sum()
        np.random.seed(1)
        for _ in range(20):
            idx = np.asarray(gpu_pf.gpu_resample_systematic(w))
            assert idx.shape == (50,)
            assert idx.min() >= 0 and idx.max() < 50
            counts = np.bincount(idx, minlength=50)
            assert np.abs(counts - 50 * w).max() < 1.0 + 1e-9

    def test_multinomial_resampling_chi_squared(self):
        """Aggregated multinomial counts vs expectation via chi-squared."""
        scipy_stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(4)
        w = rng.random(20)
        w /= w.sum()
        np.random.seed(2)
        trials, n = 500, 20
        counts = np.zeros(n)
        for _ in range(trials):
            idx = np.asarray(gpu_pf.gpu_resample_multinomial(w))
            counts += np.bincount(idx, minlength=n)
        total = trials * n
        _, p_value = scipy_stats.chisquare(counts, f_exp=w * total)
        assert p_value > 1e-3  # seeded; fails only if distribution is wrong

    def test_stratified_resampling_bounded_counts(self):
        rng = np.random.default_rng(5)
        w = rng.random(40)
        w /= w.sum()
        np.random.seed(3)
        for _ in range(20):
            idx = np.asarray(gpu_pf.gpu_resample_stratified(w))
            counts = np.bincount(idx, minlength=40)
            # stratified: |count_i - n*w_i| < 2
            assert np.abs(counts - 40 * w).max() < 2.0

    @pytest.mark.parametrize(
        "resample",
        ["gpu_resample_systematic", "gpu_resample_multinomial"],
    )
    def test_resampling_preserves_weighted_mean(self, resample):
        """E[mean of resampled values] equals the weighted mean (seeded CLT)."""
        fn = getattr(gpu_pf, resample)
        rng = np.random.default_rng(6)
        n = 200
        values = rng.normal(size=n) * 3.0
        w = rng.random(n)
        w /= w.sum()
        target = np.dot(w, values)
        np.random.seed(4)
        trials = 300
        means = [values[np.asarray(fn(w))].mean() for _ in range(trials)]
        est = np.mean(means)
        # Multinomial std of a single-trial mean bounds systematic too
        sigma_trial = np.sqrt(np.dot(w, (values - target) ** 2) / n)
        assert abs(est - target) < 4 * sigma_trial / np.sqrt(trials)

    def test_particle_filter_class_weights_match_scipy(self):
        """gpu PF weight update vs per-particle scipy Gaussian likelihood."""
        scipy_stats = pytest.importorskip("scipy.stats")
        n, dim = 400, 2
        pf = gpu_pf.CuPyParticleFilter(n_particles=n, state_dim=dim)
        np.random.seed(5)
        pf.initialize(np.zeros(dim), np.eye(dim))
        particles = np.asarray(pf.particles).copy()
        z = np.array([0.3, -0.2])
        R = np.diag([0.5, 0.8])

        def likelihood(p, meas):
            d = np.asarray(p) - np.asarray(meas)
            inv = np.linalg.inv(R)
            quad = np.einsum("ni,ij,nj->n", d, inv, d)
            norm = 1.0 / np.sqrt((2 * np.pi) ** dim * np.linalg.det(R))
            return norm * np.exp(-0.5 * quad)

        pf.resample_threshold = 0.0  # keep weights (no resample) for comparison
        pf.update(z, likelihood)
        ref = scipy_stats.multivariate_normal(mean=z, cov=R).pdf(particles)
        ref /= ref.sum()
        assert_allclose(np.asarray(pf.weights), ref, atol=1e-12)

        est = pf.get_estimate()
        assert_allclose(np.asarray(est), ref @ particles, atol=1e-10)
        cov = np.asarray(pf.get_covariance())
        diff = particles - ref @ particles
        assert_allclose(cov, np.einsum("n,ni,nj->ij", ref, diff, diff), atol=1e-10)

    def test_particle_filter_resampling_resets_weights(self):
        pf = gpu_pf.CuPyParticleFilter(n_particles=100, state_dim=1)
        np.random.seed(6)
        pf.initialize(np.zeros(1), np.eye(1))

        def sharp_likelihood(p, z):
            return np.exp(-50.0 * (np.asarray(p)[:, 0] - 5.0) ** 2)

        pf.update(np.array([5.0]), sharp_likelihood)  # ESS collapses -> resample
        w = np.asarray(pf.weights)
        assert_allclose(w, np.full(100, 0.01), atol=1e-12)

    def test_batch_particle_filter_update(self):
        rng = np.random.default_rng(9)
        n_filters, n_particles, dim = 4, 100, 2
        particles = rng.normal(size=(n_filters, n_particles, dim))
        weights = np.full((n_filters, n_particles), 1.0 / n_particles)
        measurements = rng.normal(size=(n_filters, dim))

        def likelihood(p, z):
            return np.exp(-0.5 * np.sum((np.asarray(p) - np.asarray(z)) ** 2, axis=1))

        w_upd, log_liks, ess = gpu_pf.batch_particle_filter_update(
            particles, weights, measurements, likelihood
        )
        for i in range(n_filters):
            lik = likelihood(particles[i], measurements[i])
            ref = weights[i] * (lik + 1e-300)
            ref_norm = ref / ref.sum()
            assert_allclose(np.asarray(w_upd[i]), ref_norm, atol=1e-12)
            assert_allclose(np.asarray(ess[i]), 1.0 / np.sum(ref_norm**2), atol=1e-8)


# ---------------------------------------------------------------------------
# Matrix utilities (numpy-as-cupy shim)
# ---------------------------------------------------------------------------


class TestMatrixUtilsMath:
    @pytest.fixture(autouse=True)
    def _shim(self, numpy_cupy):
        pass

    def test_cholesky_single_and_batch(self):
        rng = np.random.default_rng(10)
        A = _random_spd(rng, 4)
        L = gpu_matrix_utils.gpu_cholesky(A)
        assert_allclose(L @ L.T, A, atol=1e-10)
        U = gpu_matrix_utils.gpu_cholesky(A, lower=False)
        assert_allclose(U.T @ U, A, atol=1e-10)
        batch = np.stack([_random_spd(rng, 3) for _ in range(5)])
        Lb = gpu_matrix_utils.gpu_cholesky(batch)
        assert_allclose(np.einsum("nij,nkj->nik", Lb, Lb), batch, atol=1e-10)

    def test_cholesky_safe_regularizes_singular_psd(self):
        A = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular PSD
        L, success = gpu_matrix_utils.gpu_cholesky_safe(A)
        assert success is False
        assert_allclose(L @ L.T, A, atol=1e-8)

    def test_qr(self):
        rng = np.random.default_rng(11)
        A = rng.normal(size=(6, 4))
        Q, R = gpu_matrix_utils.gpu_qr(A)
        assert_allclose(Q @ R, A, atol=1e-10)
        assert_allclose(Q.T @ Q, np.eye(4), atol=1e-10)

    def test_solve_and_inv(self):
        rng = np.random.default_rng(12)
        A = _random_spd(rng, 5)
        b = rng.normal(size=5)
        x = gpu_matrix_utils.gpu_solve(A, b)
        assert_allclose(A @ x, b, atol=1e-9)
        A_inv = gpu_matrix_utils.gpu_inv(A)
        assert_allclose(A @ A_inv, np.eye(5), atol=1e-9)

    def test_eigh(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        vals, vecs = gpu_matrix_utils.gpu_eigh(A)
        assert_allclose(np.asarray(vals), [1.0, 3.0], atol=1e-12)
        assert_allclose(vecs @ np.diag(vals) @ vecs.T, A, atol=1e-12)

    def test_matrix_sqrt_single_and_batch(self):
        rng = np.random.default_rng(13)
        A = _random_spd(rng, 4)
        S = gpu_matrix_utils.gpu_matrix_sqrt(A)
        assert_allclose(S @ S, A, atol=1e-9)
        batch = np.stack([_random_spd(rng, 3) for _ in range(4)])
        Sb = gpu_matrix_utils.gpu_matrix_sqrt(batch)
        assert_allclose(np.einsum("nij,njk->nik", Sb, Sb), batch, atol=1e-9)
