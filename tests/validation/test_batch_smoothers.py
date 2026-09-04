"""Batch and interval smoothers against MATLAB TCL fixtures.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_smoothers.m; inputs
mirrored verbatim (MATLAB's 1-based kD is the port's k_d + 1). All
smoothers are deterministic given the inputs, so state fixtures are
machine-precision; square-root factors are compared through S @ S.T
because QR sign conventions differ between LAPACK builds.

Two ports deliberately diverge from upstream and are validated
differently:

- sqrt_info_batch_smoother: the MATLAB backward pass reads the stored
  Rw/Rwx factors one index below where its forward pass stores them,
  so its first smoothed step consumes never-written zeros. The port
  corrects the index and is validated against the RTS optimum at
  machine precision instead of against MATLAB output.
- ekalman_batch_smoother: the MATLAB original cannot run as shipped
  (it calls DiscEKFPred while the library ships discEKFPred.m); its
  fixtures were captured with a case shim.
"""

from pathlib import Path

import numpy as np

from pytcl.dynamic_estimation import (
    ekalman_batch_smoother,
    fp_info_batch_smoother,
    fp_info_interval_smoother,
    kalman_batch_smoother,
    kalman_fir_smoother,
    kalman_fir_smoother_coeffs,
    kalman_interval_smoother,
    sqrt_cub_kal_batch_smoother,
    sqrt_info_batch_smoother,
)
from pytcl.dynamic_estimation.batch_smoothers import (
    _info_filter_update,
)
from pytcl.dynamic_estimation.kalman.sqrt_cubature import (
    sqrt_ckf_predict,
    sqrt_ckf_update,
)
from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    fifth_order_cubature_points,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Deterministic transcriptions: measured max disagreement 4.3e-13.
ATOL = 1e-10

F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
R = np.array([[0.01]])
Q = np.array([[0.02, 0.01], [0.01, 0.03]])
Z = np.array([[0.05, 1.1, 1.95, 3.02]])
X_INIT = np.array([0.0, 1.0])
P_INIT = np.array([[1.0, 0.2], [0.2, 0.5]])
U = np.array([[0.1, -0.05, 0.02], [0.05, 0.1, -0.1]])
N = 4


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


def _load_stack(name, x_dim=2):
    flat = _load(name)
    n = flat.shape[0] // x_dim
    return np.stack([flat[k * x_dim : (k + 1) * x_dim, :] for k in range(n)], axis=2)


class TestKalmanBatchSmoother:
    def test_fp_delegation_matches_matlab(self):
        res = kalman_batch_smoother(X_INIT, P_INIT, Z, U, H, F, R, Q)
        np.testing.assert_allclose(res.x, _load("sm_kbs_fp_x.csv"), atol=ATOL)
        np.testing.assert_allclose(res.P, _load_stack("sm_kbs_fp_P.csv"), atol=ATOL)

    def test_rts_form_matches_matlab(self):
        res = kalman_batch_smoother(X_INIT, P_INIT, Z, U, H, F, R, Q, use_fp=False)
        np.testing.assert_allclose(res.x, _load("sm_kbs_rts_x.csv"), atol=ATOL)
        np.testing.assert_allclose(res.P, _load_stack("sm_kbs_rts_P.csv"), atol=ATOL)

    def test_single_step_matches_matlab(self):
        # MATLAB kD=2 is the port's k_d=1.
        res = kalman_batch_smoother(
            X_INIT, P_INIT, Z, U, H, F, R, Q, k_d=1, use_fp=False
        )
        np.testing.assert_allclose(res.x, _load("sm_kbs_rts_x2.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(res.P, _load("sm_kbs_rts_P2.csv"), atol=ATOL)

    def test_fp_and_rts_forms_agree(self):
        fp = kalman_batch_smoother(X_INIT, P_INIT, Z, U, H, F, R, Q)
        rts = kalman_batch_smoother(X_INIT, P_INIT, Z, U, H, F, R, Q, use_fp=False)
        np.testing.assert_allclose(fp.x, rts.x, atol=1e-9)
        np.testing.assert_allclose(fp.P, rts.P, atol=1e-9)


class TestFPInfoBatchSmoother:
    def test_uninformative_prior_matches_matlab(self):
        res = fp_info_batch_smoother(None, None, Z, U, H, F, R, Q)
        np.testing.assert_allclose(res.y, _load("sm_fp_y.csv"), atol=ATOL)
        np.testing.assert_allclose(res.p_inv, _load_stack("sm_fp_Pinv.csv"), atol=ATOL)
        np.testing.assert_allclose(res.x, _load("sm_fp_x.csv"), atol=ATOL)
        np.testing.assert_allclose(res.P, _load_stack("sm_fp_P.csv"), atol=ATOL)

    def test_single_step_slices_the_batch(self):
        full = fp_info_batch_smoother(None, None, Z, U, H, F, R, Q)
        one = fp_info_batch_smoother(None, None, Z, U, H, F, R, Q, k_d=2)
        np.testing.assert_allclose(one.x, full.x[:, 2], atol=1e-14)
        np.testing.assert_allclose(one.P, full.P[:, :, 2], atol=1e-14)


class TestEKalmanBatchSmoother:
    Z_N = np.array([[1.05, 4.2, 8.85, 16.4]])
    X_INIT_E = np.array([1.0, 1.0])
    R_E = np.array([[0.04]])

    @staticmethod
    def _h(x):
        return np.array([x[0] ** 2])

    @staticmethod
    def _hj(x):
        return np.array([[2.0 * x[0], 0.0]])

    def test_matches_matlab(self):
        res = ekalman_batch_smoother(
            self.X_INIT_E,
            P_INIT,
            self.Z_N,
            self._h,
            self._hj,
            lambda x: F @ x,
            lambda x: F,
            self.R_E,
            Q,
        )
        np.testing.assert_allclose(res.x, _load("sm_ekbs_x.csv"), atol=ATOL)
        np.testing.assert_allclose(res.P, _load_stack("sm_ekbs_P.csv"), atol=ATOL)
        np.testing.assert_allclose(res.x_upd, _load("sm_ekbs_xupd.csv"), atol=ATOL)

    def test_iterated_matches_matlab(self):
        res = ekalman_batch_smoother(
            self.X_INIT_E,
            P_INIT,
            self.Z_N,
            self._h,
            self._hj,
            lambda x: F @ x,
            lambda x: F,
            self.R_E,
            Q,
            num_iter=2,
        )
        np.testing.assert_allclose(res.x, _load("sm_ekbs_it_x.csv"), atol=ATOL)

    def test_numerical_jacobians_agree_with_exact(self):
        exact = ekalman_batch_smoother(
            self.X_INIT_E,
            P_INIT,
            self.Z_N,
            self._h,
            self._hj,
            lambda x: F @ x,
            lambda x: F,
            self.R_E,
            Q,
        )
        numeric = ekalman_batch_smoother(
            self.X_INIT_E,
            P_INIT,
            self.Z_N,
            self._h,
            None,
            lambda x: F @ x,
            None,
            self.R_E,
            Q,
        )
        np.testing.assert_allclose(numeric.x, exact.x, rtol=1e-6)


class TestSqrtCubatureSteps:
    S_INIT = np.linalg.cholesky(P_INIT)
    S_Q = np.linalg.cholesky(Q)
    S_R = np.array([[0.1]])

    @staticmethod
    def _h(x):
        return np.array([x[0] + 0.05 * x[1] ** 2])

    @staticmethod
    def _f(x):
        return np.array([x[0] + x[1] + 0.01 * x[0] ** 2, x[1] - 0.02 * x[0]])

    def test_predict_matches_matlab(self):
        xi, w = fifth_order_cubature_points(2)
        pred = sqrt_ckf_predict(X_INIT, self.S_INIT, self._f, self.S_Q, xi, w)
        np.testing.assert_allclose(
            pred.x, _load("sm_sckf_xpred.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(
            pred.S @ pred.S.T, _load("sm_sckf_Ppred.csv"), atol=ATOL
        )

    def test_update_matches_matlab(self):
        xi, w = fifth_order_cubature_points(2)
        pred = sqrt_ckf_predict(X_INIT, self.S_INIT, self._f, self.S_Q, xi, w)
        up = sqrt_ckf_update(pred.x, pred.S, Z[:, 1], self.S_R, self._h, xi, w)
        np.testing.assert_allclose(up.x, _load("sm_sckf_xupd.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(up.S @ up.S.T, _load("sm_sckf_Pupd.csv"), atol=ATOL)

    def test_batch_smoother_matches_matlab(self):
        xi, w = fifth_order_cubature_points(2)
        res = sqrt_cub_kal_batch_smoother(
            X_INIT, self.S_INIT, Z, self._h, self._f, self.S_R, self.S_Q, xi, w
        )
        np.testing.assert_allclose(res.x, _load("sm_scks_x.csv"), atol=ATOL)
        np.testing.assert_allclose(res.x_upd, _load("sm_scks_xupd.csv"), atol=ATOL)
        p_expected = _load_stack("sm_scks_P.csv")
        for k in range(N):
            np.testing.assert_allclose(
                res.S[:, :, k] @ res.S[:, :, k].T, p_expected[:, :, k], atol=ATOL
            )

    def test_default_points_run(self):
        res = sqrt_cub_kal_batch_smoother(
            X_INIT, self.S_INIT, Z, self._h, self._f, self.S_R, self.S_Q
        )
        assert res.x.shape == (2, N)


class TestSqrtInfoBatchSmoother:
    def test_matches_rts_optimum_exactly(self):
        # The upstream backward pass reads Rw/Rwx one index below where
        # the forward pass stores them (its first smoothed step consumes
        # never-written zeros); the port corrects the index, and the
        # exactness oracle is agreement with the RTS optimum on the same
        # linear-Gaussian problem.
        s_r = np.array([[0.1]])
        s_q = np.linalg.cholesky(Q)
        res = sqrt_info_batch_smoother(
            np.linalg.solve(np.linalg.cholesky(P_INIT), X_INIT),
            np.linalg.inv(np.linalg.cholesky(P_INIT)),
            Z,
            U,
            H,
            F,
            s_r,
            s_q,
        )
        ref = kalman_batch_smoother(X_INIT, P_INIT, Z, U, H, F, R, Q, use_fp=False)
        for k in range(N):
            x_k = np.linalg.solve(res.p_inv_sqrt[:, :, k], res.y_sqrt[:, k])
            p_k = np.linalg.inv(res.p_inv_sqrt[:, :, k].T @ res.p_inv_sqrt[:, :, k])
            np.testing.assert_allclose(x_k, ref.x[:, k], atol=1e-9)
            np.testing.assert_allclose(p_k, ref.P[:, :, k], atol=1e-9)

    def test_single_step_slices_the_batch(self):
        s_r = np.array([[0.1]])
        s_q = np.linalg.cholesky(Q)
        y0 = np.linalg.solve(np.linalg.cholesky(P_INIT), X_INIT)
        p0 = np.linalg.inv(np.linalg.cholesky(P_INIT))
        full = sqrt_info_batch_smoother(y0, p0, Z, U, H, F, s_r, s_q)
        one = sqrt_info_batch_smoother(y0, p0, Z, U, H, F, s_r, s_q, k_d=1)
        np.testing.assert_allclose(one.y_sqrt, full.y_sqrt[:, 1], atol=1e-14)


class TestKalmanIntervalSmoother:
    def test_grow_then_slide_matches_matlab(self):
        x_fwd_pred = None
        p_fwd_pred = None
        x_fwd_post = X_INIT[:, np.newaxis]
        p_fwd_post = P_INIT[:, :, np.newaxis]
        for step in (1, 2, 3):  # z columns 1..3 = MATLAB curStep 2..4
            res = kalman_interval_smoother(
                x_fwd_pred,
                p_fwd_pred,
                x_fwd_post,
                p_fwd_post,
                3,
                Z[:, step],
                R,
                H,
                F,
                Q,
            )
            x_fwd_pred, p_fwd_pred = res.x_fwd_pred, res.p_fwd_pred
            x_fwd_post, p_fwd_post = res.x_fwd_post, res.p_fwd_post
            np.testing.assert_allclose(
                res.x, _load(f"sm_kis_x{step + 1}.csv"), atol=ATOL
            )
            np.testing.assert_allclose(
                res.P, _load_stack(f"sm_kis_P{step + 1}.csv"), atol=ATOL
            )

    def test_newest_estimate_equals_plain_kalman_filter(self):
        # The MATLAB example's consistency check: smoothing never alters
        # the newest step, so the end of the interval must equal the
        # plain Kalman filter posterior fed the same measurements.
        from pytcl.dynamic_estimation import kf_predict, kf_update

        res = None
        x_fwd_pred = p_fwd_pred = None
        x_fwd_post = X_INIT[:, np.newaxis]
        p_fwd_post = P_INIT[:, :, np.newaxis]
        x_kf, p_kf = X_INIT, P_INIT
        for step in (1, 2, 3):
            res = kalman_interval_smoother(
                x_fwd_pred,
                p_fwd_pred,
                x_fwd_post,
                p_fwd_post,
                4,
                Z[:, step],
                R,
                H,
                F,
                Q,
            )
            x_fwd_pred, p_fwd_pred = res.x_fwd_pred, res.p_fwd_pred
            x_fwd_post, p_fwd_post = res.x_fwd_post, res.p_fwd_post
            pred = kf_predict(x_kf, p_kf, F, Q)
            upd = kf_update(pred.x, pred.P, Z[:, step], H, R)
            x_kf, p_kf = upd.x, upd.P
            np.testing.assert_allclose(res.x[:, -1], x_kf, atol=1e-12)
            np.testing.assert_allclose(res.P[:, :, -1], p_kf, atol=1e-12)


class TestFPInfoIntervalSmoother:
    def test_grow_then_slide_matches_matlab(self):
        y_prev, p_inv_prev = _info_filter_update(
            np.zeros(2), np.zeros((2, 2)), Z[:, 0], R, H
        )
        y_fwd_pred = np.zeros((2, 1))
        p_inv_fwd_pred = np.zeros((2, 2, 1))
        for step in (1, 2, 3):  # MATLAB curStep 2..4
            z_win = Z[:, max(0, step - 2) : step + 1]
            res = fp_info_interval_smoother(
                y_fwd_pred,
                p_inv_fwd_pred,
                y_prev,
                p_inv_prev,
                3,
                z_win,
                R,
                H,
                F,
                Q,
            )
            y_fwd_pred, p_inv_fwd_pred = res.y_fwd_pred, res.p_inv_fwd_pred
            y_prev, p_inv_prev = res.y_fwd_end, res.p_inv_fwd_end
            np.testing.assert_allclose(
                res.y, _load(f"sm_fpis_y{step + 1}.csv"), atol=ATOL
            )
            np.testing.assert_allclose(
                res.p_inv, _load_stack(f"sm_fpis_Pinv{step + 1}.csv"), atol=ATOL
            )


class TestKalmanFIRSmoother:
    H_STACK = np.repeat(H[:, :, np.newaxis], N, axis=2)
    F_STACK = np.repeat(F[:, :, np.newaxis], N - 1, axis=2)
    R_STACK = np.repeat(R[:, :, np.newaxis], N, axis=2)
    Q_STACK = np.repeat(Q[:, :, np.newaxis], N - 1, axis=2)

    def test_matches_matlab_at_each_kd(self):
        for kd_matlab in (1, 2, 4):
            res = kalman_fir_smoother(
                Z,
                U,
                self.H_STACK,
                self.F_STACK,
                self.R_STACK,
                self.Q_STACK,
                kd_matlab - 1,
            )
            np.testing.assert_allclose(
                res.x, _load(f"sm_fir_x{kd_matlab}.csv").ravel(), atol=ATOL
            )
            np.testing.assert_allclose(
                res.P, _load(f"sm_fir_P{kd_matlab}.csv"), atol=ATOL
            )

    def test_agrees_with_fp_smoother_without_prior(self):
        # The FIR smoother is the no-prior optimal smoother, so it must
        # agree with the Fraser-Potter smoother run with no prior.
        ref = fp_info_batch_smoother(None, None, Z, U, H, F, R, Q)
        res = kalman_fir_smoother(
            Z, U, self.H_STACK, self.F_STACK, self.R_STACK, self.Q_STACK, 2
        )
        np.testing.assert_allclose(res.x, ref.x[:, 2], atol=1e-9)
        np.testing.assert_allclose(res.P, ref.P[:, :, 2], atol=1e-9)

    def test_coefficients_reconstruct_the_smoothed_estimate(self):
        coeffs = kalman_fir_smoother_coeffs(
            self.H_STACK, self.F_STACK, self.R_STACK, self.Q_STACK, 1
        )
        x_est = np.zeros(2)
        for k in range(N - 1):
            x_est = x_est + coeffs.a[:, :, k] @ Z[:, k] + coeffs.b[:, :, k] @ U[:, k]
        x_est = x_est + coeffs.a[:, :, N - 1] @ Z[:, N - 1]
        res = kalman_fir_smoother(
            Z, U, self.H_STACK, self.F_STACK, self.R_STACK, self.Q_STACK, 1
        )
        np.testing.assert_allclose(x_est, res.x, atol=1e-13)
        np.testing.assert_allclose(coeffs.p_kn, res.P, atol=1e-13)
