"""Validation of the backend-dispatched batch EKF on the MLX backend.

``pytcl.gpu.ekf`` used to be hardwired to CuPy, so on Apple Silicon every
entry point raised ``DependencyError`` (issue #12). It is now written against
:mod:`pytcl.gpu._backend`, so these tests execute the real device code path on
this machine.

Ground truth is the CPU implementation in
:mod:`pytcl.dynamic_estimation.kalman.extended`, which is reference-validated.

Measured accuracy (MLX float32, coordinated-turn dynamics with polar
measurements, seeds 0-5 x batch sizes 4/32/256/1024, error taken as
``max|gpu - cpu| / max|cpu|`` over each output array):

===========  =========  =============
output       max error  float32 ulps
===========  =========  =============
predicted x  4.7e-08    0.4
predicted P  1.6e-07    1.3
updated x    1.6e-07    1.4
updated P    1.1e-07    0.9
gain K       2.4e-07    2.0
innov cov S  1.4e-07    1.1
likelihood   8.4e-07    7.0
===========  =========  =============

float32 eps is 1.19e-07, so every output lands within a handful of rounding
steps of the double-precision CPU answer (the likelihood is largest because
it exponentiates an already-rounded log-likelihood sum). The discrepancy is
therefore precision-limited, not algorithmic. Two tests pin that down rather
than assuming it: ``test_error_does_not_grow_with_batch_size`` shows the error
is flat from 4 to 1024 tracks, and
``test_per_track_result_independent_of_batch_size`` shows a single track's
answer is unchanged by who else is in the batch -- both would fail on
cross-track contamination from bad broadcasting or reshaping.

Element-wise tolerances below are ~100x float32 eps, plus an absolute floor,
because state vectors contain velocity components that pass through zero.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("mlx.core", reason="MLX required (Apple Silicon GPU backend)")

from pytcl.dynamic_estimation.kalman.extended import (  # noqa: E402
    ekf_predict,
    ekf_update,
)
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update  # noqa: E402
from pytcl.gpu.ekf import (  # noqa: E402
    CuPyExtendedKalmanFilter,
    batch_ekf_predict,
    batch_ekf_update,
)
from pytcl.gpu.utils import to_cpu  # noqa: E402

# ~100x float32 eps (1.19e-07). See module docstring for measured errors.
RTOL = 1e-5
ATOL = 1e-6

OMEGA = 0.05
T = 1.0


# ---------------------------------------------------------------------------
# Nonlinear test problem: coordinated turn dynamics, polar measurement
# ---------------------------------------------------------------------------


def _f_ct(x):
    px, py, vx, vy = x
    s, c = np.sin(OMEGA * T), np.cos(OMEGA * T)
    return np.array(
        [
            px + s / OMEGA * vx - (1 - c) / OMEGA * vy,
            py + (1 - c) / OMEGA * vx + s / OMEGA * vy,
            c * vx - s * vy,
            s * vx + c * vy,
        ]
    )


def _F_ct(x):
    s, c = np.sin(OMEGA * T), np.cos(OMEGA * T)
    return np.array(
        [
            [1, 0, s / OMEGA, -(1 - c) / OMEGA],
            [0, 1, (1 - c) / OMEGA, s / OMEGA],
            [0, 0, c, -s],
            [0, 0, s, c],
        ]
    )


def _h_polar(x):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def _H_polar(x):
    r2 = x[0] ** 2 + x[1] ** 2
    r = np.sqrt(r2)
    return np.array([[x[0] / r, x[1] / r, 0, 0], [-x[1] / r2, x[0] / r2, 0, 0]])


def _random_spd(rng, n, scale=0.3, jitter=0.5):
    a = rng.normal(size=(n, n)) * scale
    return a @ a.T + jitter * np.eye(n)


def _problem(seed=0, n_tracks=15):
    """Well-conditioned batch of EKF problems (states away from the origin,
    so the polar Jacobian is benign)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_tracks, 4)) + np.array([10.0, 10.0, 1.0, 1.0])
    P = np.stack([_random_spd(rng, 4) for _ in range(n_tracks)])
    Q = np.eye(4) * 0.01
    R = np.diag([0.1, 0.01])
    z = np.array([_h_polar(xi) for xi in x])
    z = z + rng.normal(size=z.shape) * [0.3, 0.03]
    return x, P, Q, R, z


SEEDS = [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Batch EKF vs looped CPU EKF
# ---------------------------------------------------------------------------


class TestBatchEKFvsCPU:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_predict_matches_cpu_loop(self, seed):
        x, P, Q, _, _ = _problem(seed)
        pred = batch_ekf_predict(x, P, _f_ct, _F_ct, Q)
        gx, gP = to_cpu(pred.x), to_cpu(pred.P)
        assert gx.shape == x.shape
        assert gP.shape == P.shape
        for i in range(len(x)):
            ref = ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q)
            assert_allclose(gx[i], ref.x, rtol=RTOL, atol=ATOL)
            assert_allclose(gP[i], ref.P, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_update_matches_cpu_loop(self, seed):
        x, P, Q, R, z = _problem(seed)
        upd = batch_ekf_update(x, P, z, _h_polar, _H_polar, R)
        gx, gP = to_cpu(upd.x), to_cpu(upd.P)
        gy, gS, gK = to_cpu(upd.y), to_cpu(upd.S), to_cpu(upd.K)
        glik = to_cpu(upd.likelihood)
        for i in range(len(x)):
            ref = ekf_update(x[i], P[i], z[i], _h_polar, _H_polar(x[i]), R)
            assert_allclose(gx[i], ref.x, rtol=RTOL, atol=ATOL)
            assert_allclose(gP[i], ref.P, rtol=RTOL, atol=ATOL)
            assert_allclose(gy[i], ref.y, rtol=RTOL, atol=ATOL)
            assert_allclose(gS[i], ref.S, rtol=RTOL, atol=ATOL)
            assert_allclose(gK[i], ref.K, rtol=1e-4, atol=ATOL)
            assert_allclose(glik[i], ref.likelihood, rtol=RTOL, atol=ATOL)

    def test_predicted_covariance_symmetric_and_pd(self):
        x, P, Q, _, _ = _problem(3)
        gP = to_cpu(batch_ekf_predict(x, P, _f_ct, _F_ct, Q).P)
        assert_allclose(gP, np.swapaxes(gP, -2, -1), rtol=0, atol=0)
        assert np.all(np.linalg.eigvalsh(gP) > 0)

    def test_numerical_jacobian_matches_analytic(self):
        """F_jacobian=None must fall back to central differences."""
        x, P, Q, _, _ = _problem(4)
        analytic = to_cpu(batch_ekf_predict(x, P, _f_ct, _F_ct, Q).P)
        numeric = to_cpu(batch_ekf_predict(x, P, _f_ct, None, Q).P)
        assert_allclose(numeric, analytic, rtol=1e-4, atol=1e-5)

    def test_per_track_Q_and_R(self):
        """3-D Q/R (per-track noise) must not be broadcast over."""
        x, P, _, _, z = _problem(5)
        n = len(x)
        rng = np.random.default_rng(99)
        Q = np.stack([_random_spd(rng, 4, 0.05, 0.01) for _ in range(n)])
        R = np.stack([_random_spd(rng, 2, 0.05, 0.05) for _ in range(n)])
        pred = batch_ekf_predict(x, P, _f_ct, _F_ct, Q)
        upd = batch_ekf_update(x, P, z, _h_polar, _H_polar, R)
        gpP, guP, gux = to_cpu(pred.P), to_cpu(upd.P), to_cpu(upd.x)
        for i in range(n):
            rp = ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q[i])
            ru = ekf_update(x[i], P[i], z[i], _h_polar, _H_polar(x[i]), R[i])
            assert_allclose(gpP[i], rp.P, rtol=RTOL, atol=ATOL)
            assert_allclose(guP[i], ru.P, rtol=RTOL, atol=ATOL)
            assert_allclose(gux[i], ru.x, rtol=RTOL, atol=ATOL)

    def test_class_predict_update_matches_cpu(self):
        x, P, Q, R, z = _problem(2, n_tracks=8)
        ekf = CuPyExtendedKalmanFilter(
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
        gx, gP = to_cpu(result.x), to_cpu(result.P)
        for i in range(len(x)):
            p = ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q)
            u = ekf_update(p.x, p.P, z[i], _h_polar, _H_polar(p.x), R)
            assert_allclose(gx[i], u.x, rtol=RTOL, atol=ATOL)
            assert_allclose(gP[i], u.P, rtol=RTOL, atol=ATOL)

    def test_class_defaults_construct_on_device(self):
        """Q=None/R=None defaults must be built with backend ops, not CuPy."""
        ekf = CuPyExtendedKalmanFilter(state_dim=4, meas_dim=2, f=_f_ct, h=_h_polar)
        assert_allclose(to_cpu(ekf.Q), np.eye(4) * 0.01, rtol=0, atol=1e-9)
        assert_allclose(to_cpu(ekf.R), np.eye(2), rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Precision is float32-limited, not algorithmic
# ---------------------------------------------------------------------------


def _rel(actual, ref):
    """Error relative to the scale of the reference array.

    Deliberately *not* element-wise ``|a-r|/|r|``: state vectors contain
    velocity components that pass through zero, and an element-wise ratio
    against a near-zero denominator measures the denominator, not the solver.
    """
    return float(np.abs(actual - ref).max() / np.abs(ref).max())


class TestErrorIsPrecisionLimited:
    def test_error_does_not_grow_with_batch_size(self):
        """Tracks are independent, so error must be flat in ``n_tracks``.

        A batch-size-dependent error would mean cross-track contamination
        (a broadcasting or reshape bug), not float32 rounding.
        """
        errors = {}
        for n in (4, 32, 256, 1024):
            x, P, Q, R, z = _problem(seed=11, n_tracks=n)
            gx = to_cpu(batch_ekf_predict(x, P, _f_ct, _F_ct, Q).x)
            gu = to_cpu(batch_ekf_update(x, P, z, _h_polar, _H_polar, R).x)
            ref_p = np.stack(
                [ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q).x for i in range(n)]
            )
            ref_u = np.stack(
                [
                    ekf_update(x[i], P[i], z[i], _h_polar, _H_polar(x[i]), R).x
                    for i in range(n)
                ]
            )
            errors[n] = max(_rel(gx, ref_p), _rel(gu, ref_u))

        eps32 = float(np.finfo(np.float32).eps)
        # Every batch size sits within a few float32 ulps ...
        assert max(errors.values()) < 20 * eps32, errors
        # ... and 256x more tracks buys no more than a 4x error.
        assert errors[1024] <= 4 * max(errors[4], eps32), errors

    def test_per_track_result_independent_of_batch_size(self):
        """The sharpest form of the same claim: a track's answer must not
        depend on who else is in the batch."""
        x, P, Q, R, z = _problem(seed=11, n_tracks=1024)
        big_p = to_cpu(batch_ekf_predict(x, P, _f_ct, _F_ct, Q).x)
        big_u = to_cpu(batch_ekf_update(x, P, z, _h_polar, _H_polar, R).x)
        for i in (0, 7, 511, 1023):
            sl = slice(i, i + 1)
            solo_p = to_cpu(batch_ekf_predict(x[sl], P[sl], _f_ct, _F_ct, Q).x)
            solo_u = to_cpu(
                batch_ekf_update(x[sl], P[sl], z[sl], _h_polar, _H_polar, R).x
            )
            assert_allclose(big_p[i], solo_p[0], rtol=0, atol=0)
            assert_allclose(big_u[i], solo_u[0], rtol=1e-6, atol=1e-6)

    def test_error_is_near_float32_eps(self):
        """Absolute proof of scale: error is a few float32 ulps of the data."""
        eps32 = float(np.finfo(np.float32).eps)
        x, P, Q, R, z = _problem(seed=12, n_tracks=64)
        gu = to_cpu(batch_ekf_update(x, P, z, _h_polar, _H_polar, R).x)
        ref_u = np.stack(
            [
                ekf_update(x[i], P[i], z[i], _h_polar, _H_polar(x[i]), R).x
                for i in range(len(x))
            ]
        )
        gp = to_cpu(batch_ekf_predict(x, P, _f_ct, _F_ct, Q).P)
        ref_p = np.stack(
            [ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q).P for i in range(len(x))]
        )
        assert _rel(gu, ref_u) < 20 * eps32, _rel(gu, ref_u)
        assert _rel(gp, ref_p) < 20 * eps32, _rel(gp, ref_p)


# ---------------------------------------------------------------------------
# Linear reduction: EKF with linear f/h must equal the linear KF
# ---------------------------------------------------------------------------


def _linear_problem(seed=21, n_tracks=12):
    rng = np.random.default_rng(seed)
    F = np.array(
        [[1.0, 0.0, T, 0.0], [0.0, 1.0, 0.0, T], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    )
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    x = rng.normal(size=(n_tracks, 4)) * 5.0 + np.array([20.0, 20.0, 1.0, 1.0])
    P = np.stack([_random_spd(rng, 4) for _ in range(n_tracks)])
    Q = np.eye(4) * 0.05
    R = np.diag([0.2, 0.2])
    z = x @ H.T + rng.normal(size=(n_tracks, 2)) * 0.4
    return x, P, F, H, Q, R, z


class TestLinearReduction:
    """With linear f/h the EKF is the linear KF; verify to float32 tolerance."""

    def test_predict_reduces_to_linear_kf(self):
        x, P, F, _, Q, _, _ = _linear_problem()
        pred = batch_ekf_predict(x, P, lambda xi: F @ xi, lambda xi: F, Q)
        gx, gP = to_cpu(pred.x), to_cpu(pred.P)
        for i in range(len(x)):
            ref = kf_predict(x[i], P[i], F, Q)
            assert_allclose(gx[i], ref.x, rtol=RTOL, atol=ATOL)
            assert_allclose(gP[i], ref.P, rtol=RTOL, atol=ATOL)

    def test_update_reduces_to_linear_kf(self):
        x, P, _, H, _, R, z = _linear_problem()
        upd = batch_ekf_update(x, P, z, lambda xi: H @ xi, lambda xi: H, R)
        gx, gP, gK, gS = (
            to_cpu(upd.x),
            to_cpu(upd.P),
            to_cpu(upd.K),
            to_cpu(upd.S),
        )
        glik = to_cpu(upd.likelihood)
        for i in range(len(x)):
            ref = kf_update(x[i], P[i], z[i], H, R)
            assert_allclose(gx[i], ref.x, rtol=RTOL, atol=ATOL)
            assert_allclose(gS[i], ref.S, rtol=RTOL, atol=ATOL)
            assert_allclose(gK[i], ref.K, rtol=1e-4, atol=ATOL)
            assert_allclose(glik[i], ref.likelihood, rtol=RTOL, atol=ATOL)
            # kf_update uses the short form (I-KH)P; batch_ekf_update uses
            # the Joseph form. Algebraically identical at the optimal gain.
            assert_allclose(gP[i], ref.P, rtol=1e-4, atol=1e-5)

    def test_matches_batch_linear_kf_when_available(self):
        """Cross-check against the batch linear KF on the same backend."""
        from pytcl.core.exceptions import DependencyError

        try:
            from pytcl.gpu.kalman import batch_kf_predict
        except ImportError:  # pragma: no cover - module always importable
            pytest.skip("pytcl.gpu.kalman unavailable")

        x, P, F, _, Q, _, _ = _linear_problem()
        try:
            ref = batch_kf_predict(x, P, F, Q)
        except DependencyError:
            pytest.skip("batch_kf_predict is still CuPy-only on this machine")
        ekf = batch_ekf_predict(x, P, lambda xi: F @ xi, lambda xi: F, Q)
        assert_allclose(to_cpu(ekf.x), to_cpu(ref.x), rtol=RTOL, atol=ATOL)
        assert_allclose(to_cpu(ekf.P), to_cpu(ref.P), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Input handling regressions found during the port
# ---------------------------------------------------------------------------


class TestInputHandling:
    def test_accepts_device_arrays(self):
        """Regression: ``np.asarray`` on a device array is illegal on CuPy;
        the CPU-side callable path must go through ``to_cpu``."""
        import mlx.core as mx

        x, P, Q, R, z = _problem(6, n_tracks=5)
        pred = batch_ekf_predict(
            mx.array(x.astype(np.float32)),
            mx.array(P.astype(np.float32)),
            _f_ct,
            _F_ct,
            Q,
        )
        upd = batch_ekf_update(
            pred.x, pred.P, mx.array(z.astype(np.float32)), _h_polar, _H_polar, R
        )
        gx = to_cpu(upd.x)
        assert np.all(np.isfinite(gx))
        for i in range(len(x)):
            p = ekf_predict(x[i], P[i], _f_ct, _F_ct(x[i]), Q)
            u = ekf_update(p.x, p.P, z[i], _h_polar, _H_polar(p.x), R)
            assert_allclose(gx[i], u.x, rtol=RTOL, atol=ATOL)

    def test_integer_state_input_is_not_truncated(self):
        """Regression: ``np.zeros_like(x)`` inherited an integer dtype and
        truncated the propagated state."""
        x_int = np.array([[10, 10, 1, 1], [12, 8, 2, 1]])
        P = np.tile(np.eye(4) * 0.5, (2, 1, 1))
        Q = np.eye(4) * 0.01
        gx = to_cpu(batch_ekf_predict(x_int, P, _f_ct, _F_ct, Q).x)
        for i in range(2):
            ref = ekf_predict(x_int[i], P[i], _f_ct, _F_ct(x_int[i]), Q)
            assert_allclose(gx[i], ref.x, rtol=RTOL, atol=ATOL)
        # Would be exactly integral if truncation had occurred.
        assert not np.allclose(gx, np.round(gx))
