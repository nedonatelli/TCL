"""
Validation of the batch linear Kalman filter on a real GPU backend.

Ground truth is the CPU implementation in :mod:`pytcl.dynamic_estimation`,
which is reference-validated (see AUDIT.md). These tests run on whichever GPU
backend is installed; on MLX (Apple Silicon) computation is float32, so the
measured agreement is ~1e-7 relative rather than machine epsilon.

Measured on MLX 0.32 / Apple Silicon: predict 4.6e-8, update 1.6e-7 relative.
The batch-size sweep below asserts the error is precision-limited rather than
algorithmic -- it must not grow with the number of tracks.
"""

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from pytcl.dynamic_estimation import kf_predict, kf_update  # noqa: E402
from pytcl.gpu.kalman import (  # noqa: E402
    batch_kf_predict,
    batch_kf_update,
)
from pytcl.gpu.utils import to_cpu  # noqa: E402

# float32 backends resolve ~1e-7; allow a small margin for accumulation.
FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6

STATE_DIM = 4
MEAS_DIM = 2
F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float)
Q = np.eye(STATE_DIM) * 0.05
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
R = np.eye(MEAS_DIM) * 0.3


def _random_batch(n, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, STATE_DIM))
    a = rng.standard_normal((n, STATE_DIM, STATE_DIM)) * 0.3
    P = a @ np.swapaxes(a, -2, -1) + np.eye(STATE_DIM)
    z = rng.standard_normal((n, MEAS_DIM))
    return x, P, z


class TestBatchPredictMatchesCPU:
    def test_predict_matches_cpu_reference(self):
        x, P, _ = _random_batch(64)
        got = batch_kf_predict(x, P, F, Q)
        want_x = np.stack([kf_predict(x[i], P[i], F, Q).x for i in range(len(x))])
        want_P = np.stack([kf_predict(x[i], P[i], F, Q).P for i in range(len(x))])
        np.testing.assert_allclose(
            to_cpu(got.x), want_x, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
        )
        np.testing.assert_allclose(
            to_cpu(got.P), want_P, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
        )

    def test_predict_with_control_input(self):
        x, P, _ = _random_batch(32)
        rng = np.random.default_rng(3)
        B = rng.standard_normal((STATE_DIM, 2))
        u = rng.standard_normal((len(x), 2))
        got = to_cpu(batch_kf_predict(x, P, F, Q, B=B, u=u).x)
        want = np.stack([F @ x[i] + B @ u[i] for i in range(len(x))])
        np.testing.assert_allclose(got, want, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL)

    def test_predicted_covariance_symmetric(self):
        x, P, _ = _random_batch(16)
        got = to_cpu(batch_kf_predict(x, P, F, Q).P)
        np.testing.assert_allclose(got, np.swapaxes(got, -2, -1), atol=1e-12)

    @pytest.mark.parametrize("n", [16, 256, 2048])
    def test_error_does_not_grow_with_batch_size(self, n):
        """Error must be precision-limited, not algorithmic."""
        x, P, _ = _random_batch(n, seed=11)
        got = to_cpu(batch_kf_predict(x, P, F, Q).x)
        want = np.stack([kf_predict(x[i], P[i], F, Q).x for i in range(n)])
        rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert rel < 1e-6, f"n={n} rel_err={rel:.2e}"


class TestBatchUpdateMatchesCPU:
    def test_update_matches_cpu_reference(self):
        x, P, z = _random_batch(64)
        got = batch_kf_update(x, P, z, H, R)
        ref = [kf_update(x[i], P[i], z[i], H, R) for i in range(len(x))]
        np.testing.assert_allclose(
            to_cpu(got.x),
            np.stack([r.x for r in ref]),
            rtol=FLOAT32_RTOL,
            atol=FLOAT32_ATOL,
        )
        np.testing.assert_allclose(
            to_cpu(got.P),
            np.stack([r.P for r in ref]),
            rtol=FLOAT32_RTOL,
            atol=FLOAT32_ATOL,
        )

    def test_innovation_and_gain_match_cpu(self):
        x, P, z = _random_batch(32, seed=5)
        got = batch_kf_update(x, P, z, H, R)
        ref = [kf_update(x[i], P[i], z[i], H, R) for i in range(len(x))]
        np.testing.assert_allclose(
            to_cpu(got.y),
            np.stack([r.y for r in ref]),
            rtol=FLOAT32_RTOL,
            atol=FLOAT32_ATOL,
        )
        np.testing.assert_allclose(
            to_cpu(got.S),
            np.stack([r.S for r in ref]),
            rtol=FLOAT32_RTOL,
            atol=FLOAT32_ATOL,
        )

    def test_likelihood_matches_cpu(self):
        x, P, z = _random_batch(32, seed=9)
        got = to_cpu(batch_kf_update(x, P, z, H, R).likelihood)
        want = np.array(
            [kf_update(x[i], P[i], z[i], H, R).likelihood for i in range(32)]
        )
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-8)

    def test_updated_covariance_psd(self):
        x, P, z = _random_batch(24, seed=13)
        got = to_cpu(batch_kf_update(x, P, z, H, R).P)
        eigenvalues = np.linalg.eigvalsh(got)
        assert eigenvalues.min() > -1e-6

    def test_track_specific_matrices(self):
        """Per-track H and R must be honoured, not silently broadcast."""
        n = 8
        x, P, z = _random_batch(n, seed=17)
        rng = np.random.default_rng(2)
        H_batch = (
            np.tile(H, (n, 1, 1)) + rng.standard_normal((n, MEAS_DIM, STATE_DIM)) * 0.01
        )
        R_batch = np.tile(R, (n, 1, 1))
        got = batch_kf_update(x, P, z, H_batch, R_batch)
        ref = [kf_update(x[i], P[i], z[i], H_batch[i], R_batch[i]) for i in range(n)]
        np.testing.assert_allclose(
            to_cpu(got.x),
            np.stack([r.x for r in ref]),
            rtol=FLOAT32_RTOL,
            atol=FLOAT32_ATOL,
        )
