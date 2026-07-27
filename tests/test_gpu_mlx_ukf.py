"""Batch UKF on the MLX compute backend (Apple Silicon).

These tests execute :mod:`pytcl.gpu.ukf` for real on the GPU — MLX is the
only compute backend available on this machine — and validate it against the
reference-validated CPU implementations in :mod:`pytcl.dynamic_estimation`.

Precision context: MLX computes in float32 and the Merwe sigma-point weights
scale as ``O(1/alpha**2)``, so accuracy depends strongly on ``alpha``. The
measured maximum relative error against the float64 CPU UKF on a linear
problem is 5.8e+01 at ``alpha=1e-3``, 4.6e-03 at 1e-2, 1.9e-05 at 1e-1 and
1.9e-06 at 1.0. Every comparison below therefore runs at ``alpha=0.5``, where
float32 is adequate (errors ~1e-5 relative), except the tests that
deliberately probe the alpha scaling.
"""

import warnings

import numpy as np
import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_estimation.kalman.unscented import (
    sigma_points_merwe,
    ukf_predict,
    ukf_update,
)
from pytcl.gpu import ukf as gpu_ukf
from pytcl.gpu._backend import get_compute_backend

pytest.importorskip("mlx.core", reason="MLX required for the GPU backend")

# alpha at which float32 sigma-point weights are well conditioned.
ALPHA = 0.5

# Relative tolerance for float32 compute at ALPHA (measured ~2e-5 worst case).
RTOL32 = 2e-4

DT = 1.0
F_CV = np.array(
    [
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
H_POS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def _f_linear(x):
    return F_CV @ np.asarray(x, dtype=np.float64)


def _h_linear(x):
    return H_POS @ np.asarray(x, dtype=np.float64)


def _f_ct(x):
    """Mildly nonlinear (drag-like) dynamics."""
    x = np.asarray(x, dtype=np.float64)
    speed = np.hypot(x[2], x[3])
    damp = 1.0 - 1e-3 * speed
    return np.array([x[0] + DT * x[2], x[1] + DT * x[3], damp * x[2], damp * x[3]])


def _h_polar(x):
    x = np.asarray(x, dtype=np.float64)
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def _problem(seed=5, n_tracks=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_tracks, 4)) * np.array([100.0, 100.0, 10.0, 10.0])
    P = np.empty((n_tracks, 4, 4))
    for i in range(n_tracks):
        a = rng.normal(size=(4, 4))
        P[i] = a @ a.T + 4.0 * np.eye(4)
    Q = np.diag([1.0, 1.0, 0.1, 0.1])
    R = np.diag([4.0, 4.0])
    z = x @ H_POS.T + rng.normal(size=(n_tracks, 2))
    return x, P, Q, R, z


def _max_rel(actual, reference):
    """Max absolute deviation normalised by the reference's own scale."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.abs(reference).max()), 1e-12)
    return float(np.abs(actual - reference).max() / scale)


@pytest.fixture(scope="module")
def backend():
    return get_compute_backend()


class TestBackendIsMLX:
    def test_backend_is_float32_mlx(self, backend):
        """Guard: these tolerances assume the float32 MLX backend."""
        assert backend.name == "mlx"
        assert backend.supports_float64 is False


class TestBatchVsCPULoop:
    """Batch GPU results must equal looping the CPU UKF track by track."""

    def test_predict_matches_cpu_loop(self):
        x, P, Q, _, _ = _problem(seed=11)
        pred = gpu_ukf.batch_ukf_predict(x, P, _f_ct, Q, alpha=ALPHA)
        px, pP = np.asarray(pred.x), np.asarray(pred.P)
        assert px.shape == x.shape
        assert pP.shape == P.shape
        for i in range(len(x)):
            ref = ukf_predict(x[i], P[i], _f_ct, Q, alpha=ALPHA)
            assert _max_rel(px[i], ref.x) < RTOL32
            assert _max_rel(pP[i], ref.P) < RTOL32

    def test_update_matches_cpu_loop(self):
        x, P, Q, R, z = _problem(seed=13)
        upd = gpu_ukf.batch_ukf_update(x, P, z, _h_polar, R, alpha=ALPHA)
        for i in range(len(x)):
            ref = ukf_update(x[i], P[i], z[i], _h_polar, R, alpha=ALPHA)
            assert _max_rel(np.asarray(upd.x)[i], ref.x) < RTOL32
            assert _max_rel(np.asarray(upd.P)[i], ref.P) < RTOL32
            assert _max_rel(np.asarray(upd.y)[i], ref.y) < RTOL32
            assert _max_rel(np.asarray(upd.S)[i], ref.S) < RTOL32
            assert (
                _max_rel(np.asarray(upd.likelihood)[i], np.asarray(ref.likelihood))
                < RTOL32
            )

    def test_filter_class_matches_functions(self):
        x, P, Q, R, z = _problem(seed=19, n_tracks=5)
        ukf = gpu_ukf.CuPyUnscentedKalmanFilter(
            state_dim=4, meas_dim=2, f=_f_ct, h=_h_polar, Q=Q, R=R, alpha=ALPHA
        )
        result = ukf.predict_update(x, P, z)
        pred = gpu_ukf.batch_ukf_predict(x, P, _f_ct, Q, alpha=ALPHA)
        upd = gpu_ukf.batch_ukf_update(pred.x, pred.P, z, _h_polar, R, alpha=ALPHA)
        assert _max_rel(np.asarray(result.x), np.asarray(upd.x)) < 1e-6
        assert _max_rel(np.asarray(result.P), np.asarray(upd.P)) < 1e-6


class TestReducesToLinearKF:
    """On linear f and h the UKF is algebraically the linear Kalman filter."""

    def test_predict_equals_kf_predict(self):
        x, P, Q, _, _ = _problem(seed=23)
        pred = gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=ALPHA)
        for i in range(len(x)):
            ref = kf_predict(x[i], P[i], F_CV, Q)
            assert _max_rel(np.asarray(pred.x)[i], ref.x) < RTOL32
            assert _max_rel(np.asarray(pred.P)[i], ref.P) < RTOL32

    def test_update_equals_kf_update(self):
        x, P, _, R, z = _problem(seed=29)
        upd = gpu_ukf.batch_ukf_update(x, P, z, _h_linear, R, alpha=ALPHA)
        for i in range(len(x)):
            ref = kf_update(x[i], P[i], z[i], H_POS, R)
            assert _max_rel(np.asarray(upd.x)[i], ref.x) < RTOL32
            assert _max_rel(np.asarray(upd.P)[i], ref.P) < RTOL32
            assert _max_rel(np.asarray(upd.S)[i], ref.S) < RTOL32


class TestSigmaPoints:
    def test_shape_and_first_point_is_the_mean(self):
        x, P, _, _, _ = _problem(seed=31)
        n = x.shape[1]
        sigma = np.asarray(gpu_ukf._generate_sigma_points(x, P, ALPHA, 0.0))
        assert sigma.shape == (len(x), 2 * n + 1, n)
        assert _max_rel(sigma[:, 0, :], x) < 1e-6

    def test_weighted_moments_recover_mean_and_covariance(self):
        """sum(Wm*X) == x and sum(Wc*(X-x)(X-x)^T) == P."""
        x, P, _, _, _ = _problem(seed=37)
        n = x.shape[1]
        sigma = np.asarray(
            gpu_ukf._generate_sigma_points(x, P, ALPHA, 0.0), dtype=np.float64
        )
        Wm, Wc = gpu_ukf._compute_sigma_weights(n, ALPHA, 2.0, 0.0)
        mean = np.einsum("j,njk->nk", Wm, sigma)
        diff = sigma - mean[:, None, :]
        cov = np.einsum("j,nji,njk->nik", Wc, diff, diff)
        assert _max_rel(mean, x) < RTOL32
        assert _max_rel(cov, P) < RTOL32

    def test_matches_cpu_sigma_points(self):
        x, P, _, _, _ = _problem(seed=41, n_tracks=4)
        sigma = np.asarray(gpu_ukf._generate_sigma_points(x, P, ALPHA, 0.0))
        for i in range(len(x)):
            ref = sigma_points_merwe(x[i], P[i], alpha=ALPHA, kappa=0.0)
            assert _max_rel(sigma[i], ref.points) < RTOL32


class TestNonPositiveDefiniteFallback:
    """The eigh fallback must clamp eigenvalues and stay a true square root.

    MLX does not raise on a non-PD Cholesky (it returns a factor with a
    non-positive diagonal), so the fallback is selected from the factor's
    diagonal rather than from an exception.
    """

    @staticmethod
    def _recovered_cov(sigma, x, n, alpha, kappa=0.0):
        lambda_ = alpha**2 * (n + kappa) - n
        spread = sigma[:, 1 : n + 1, :] - x[:, None, :]  # +gamma * L columns
        return np.einsum("nji,njk->nik", spread, spread) / (n + lambda_)

    def test_singular_covariance_is_eigenvalue_clamped(self):
        n_tracks, n = 3, 4
        P_sing = np.tile(np.diag([1.0, 1.0, 1.0, 0.0]), (n_tracks, 1, 1))
        x = np.zeros((n_tracks, n))
        sigma = np.asarray(
            gpu_ukf._generate_sigma_points(x, P_sing, ALPHA, 0.0), dtype=np.float64
        )
        assert sigma.shape == (n_tracks, 2 * n + 1, n)
        recovered = self._recovered_cov(sigma, x, n, ALPHA)
        P_clamped = np.tile(np.diag([1.0, 1.0, 1.0, 1e-10]), (n_tracks, 1, 1))
        assert np.abs(recovered - P_clamped).max() < 1e-5

    def test_indefinite_covariance_is_eigenvalue_clamped(self):
        """A negative eigenvalue must come back clamped, not negated."""
        n_tracks, n = 2, 3
        P_bad = np.tile(np.diag([2.0, 1.0, -0.5]), (n_tracks, 1, 1))
        x = np.zeros((n_tracks, n))
        sigma = np.asarray(
            gpu_ukf._generate_sigma_points(x, P_bad, ALPHA, 0.0), dtype=np.float64
        )
        recovered = self._recovered_cov(sigma, x, n, ALPHA)
        P_clamped = np.tile(np.diag([2.0, 1.0, 1e-10]), (n_tracks, 1, 1))
        assert np.abs(recovered - P_clamped).max() < 1e-5

    def test_matrix_sqrt_reconstructs_clamped_covariance(self, backend):
        """L @ L.T must equal the clamped input covariance."""
        n = 4
        P_sing = np.tile(np.diag([3.0, 2.0, 1.0, 0.0]), (2, 1, 1))
        L = gpu_ukf._matrix_sqrt(backend, backend.asarray(P_sing), n)
        L_np = np.asarray(backend.to_numpy(L), dtype=np.float64)
        recon = L_np @ np.swapaxes(L_np, -2, -1)
        P_clamped = np.tile(np.diag([3.0, 2.0, 1.0, 1e-10]), (2, 1, 1))
        assert np.abs(recon - P_clamped).max() < 1e-5

    def test_positive_definite_uses_cholesky(self, backend):
        """The PD path must return the lower-triangular Cholesky factor."""
        rng = np.random.default_rng(3)
        a = rng.normal(size=(2, 4, 4))
        P = a @ np.swapaxes(a, -2, -1) + 4.0 * np.eye(4)
        L = np.asarray(
            backend.to_numpy(gpu_ukf._matrix_sqrt(backend, backend.asarray(P), 4)),
            dtype=np.float64,
        )
        assert np.abs(np.triu(L, 1)).max() == 0.0  # lower triangular
        assert _max_rel(L @ np.swapaxes(L, -2, -1), P) < 1e-5


class TestFloat32AlphaWarning:
    """A float32 backend must warn (not silently rescale) for tiny alpha."""

    def test_warns_below_threshold(self):
        x, P, Q, _, _ = _problem(seed=43, n_tracks=2)
        with pytest.warns(RuntimeWarning, match="alpha"):
            gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=1e-3)

    def test_update_warns_below_threshold(self):
        x, P, _, R, z = _problem(seed=43, n_tracks=2)
        with pytest.warns(RuntimeWarning, match="float32"):
            gpu_ukf.batch_ukf_update(x, P, z, _h_linear, R, alpha=1e-3)

    def test_no_warning_at_recommended_alpha(self):
        x, P, Q, _, _ = _problem(seed=43, n_tracks=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=ALPHA)

    def test_alpha_is_not_modified(self):
        """The warning must not change the caller's alpha behind their back."""
        x, P, Q, _, _ = _problem(seed=43, n_tracks=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            small = gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=1e-3)
        large = gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=ALPHA)
        # If alpha had been silently raised these would agree bit for bit.
        assert not np.allclose(np.asarray(small.P), np.asarray(large.P))


class TestNoBackendErrorContract:
    """With no compute backend at all the module must still raise
    DependencyError naming both extras (the pre-port ``@requires`` contract)."""

    def test_dependency_error_names_both_extras(self, monkeypatch):
        from pytcl.gpu import _backend

        def _no_backend(self, *args, **kwargs):
            raise ImportError("no compute backend")

        monkeypatch.setattr(_backend.CuPyBackend, "__init__", _no_backend)
        monkeypatch.setattr(_backend.MLXBackend, "__init__", _no_backend)
        x, P, Q, _, _ = _problem(seed=47, n_tracks=2)
        with pytest.raises(DependencyError) as exc:
            gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=ALPHA)
        message = str(exc.value)
        assert "cupy" in message
        assert "mlx" in message


class TestAlphaPrecisionScaling:
    """The documented float32 error table: error falls as alpha grows."""

    def test_error_decreases_with_alpha(self):
        x, P, Q, _, _ = _problem(seed=5)
        errs = []
        for alpha in (1e-3, 1e-2, 1e-1, 1.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pred = gpu_ukf.batch_ukf_predict(x, P, _f_linear, Q, alpha=alpha)
            errs.append(
                max(
                    _max_rel(np.asarray(pred.x)[i], kf_predict(x[i], P[i], F_CV, Q).x)
                    for i in range(len(x))
                )
            )
        assert errs == sorted(errs, reverse=True)
        # Documented magnitudes: meaningless at 1e-3, ~1e-5 or better at >= 0.1
        assert errs[0] > 1e-2
        assert errs[2] < 1e-4
        assert errs[3] < 1e-4
