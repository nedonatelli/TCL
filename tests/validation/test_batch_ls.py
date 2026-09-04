"""Batch least-squares estimators against MATLAB TCL fixtures.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_batch_ls.m; inputs
mirrored verbatim (MATLAB's 1-based kD is the port's k_d + 1). The
Gauss-Newton and closed-form estimators are deterministic; the LM
variants replace LSEstLMarquardt with SciPy's LM, so their optima are
compared at optimizer tolerance.
"""

from pathlib import Path

import numpy as np

from pytcl.dynamic_estimation import (
    batch_ls_lin_meas_lin_dyn,
    batch_ls_nonlin_meas_lin_dyn,
    batch_ls_nonlin_meas_lin_dyn_lm,
    batch_ls_nonlin_meas_nonlin_dyn,
    batch_ls_nonlin_meas_nonlin_dyn_lm,
    two_point_diff_init,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Deterministic transcriptions: measured max disagreement 4.9e-15.
ATOL = 1e-12
# LM optima across different damping schedules: measured 8.6e-9.
ATOL_LM = 1e-6


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
R = np.array([[0.01]])
Q = np.array([[0.02, 0.01], [0.01, 0.03]])
Z_LIN = np.array([[0.1, 1.05, 2.02, 2.95]])
Z_NONLIN = np.array([[4.1, 6.2, 9.1, 12.2]])
X_INIT = np.array([1.8, 0.4])


def _h(x):
    return np.array([x[0] ** 2])


def _hj(x):
    return np.array([[2.0 * x[0], 0.0]])


def _folded():
    f_pows = [np.linalg.matrix_power(F, k) for k in range(4)]
    hs = [(lambda fk: lambda x: np.array([(fk[0, :] @ x) ** 2]))(fk) for fk in f_pows]
    hjs = [
        (lambda fk: lambda x: np.atleast_2d(2.0 * (fk[0, :] @ x) * fk[0, :]))(fk)
        for fk in f_pows
    ]
    return hs, hjs


class TestLinearBatch:
    def test_matches_matlab(self):
        res = batch_ls_lin_meas_lin_dyn(Z_LIN, H, F, R, 1, Q)
        np.testing.assert_allclose(res.x, _load("batchls_lin_x.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(res.P, _load("batchls_lin_P.csv"), atol=ATOL)

    def test_covariance_only_mode(self):
        full = batch_ls_lin_meas_lin_dyn(Z_LIN, H, F, R, 1, Q)
        cov = batch_ls_lin_meas_lin_dyn(None, H, F, R, 1, Q, num_meas=4)
        assert cov.x is None
        np.testing.assert_allclose(cov.P, full.P, atol=1e-14)

    def test_exact_recovery_without_noise(self):
        # Noise-free constant-velocity data recovers the truth exactly.
        z = np.array([[0.0, 1.0, 2.0, 3.0]])
        res = batch_ls_lin_meas_lin_dyn(z, H, F, R, 0)
        np.testing.assert_allclose(res.x, [0.0, 1.0], atol=1e-9)


class TestGaussNewtonBatches:
    def test_nonlin_meas_lin_dyn_matches_matlab(self):
        res = batch_ls_nonlin_meas_lin_dyn(X_INIT, Z_NONLIN, _h, F, R, 0, _hj, 10)
        np.testing.assert_allclose(res.x, _load("batchls_nlm_x.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(res.P, _load("batchls_nlm_P.csv"), atol=ATOL)

    def test_nonlin_meas_nonlin_dyn_matches_matlab(self):
        hs, hjs = _folded()
        res = batch_ls_nonlin_meas_nonlin_dyn(X_INIT, Z_NONLIN, hs, R, hjs, 10)
        np.testing.assert_allclose(res.x, _load("batchls_nnl_x.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(res.P, _load("batchls_nnl_P.csv"), atol=ATOL)

    def test_numerical_jacobian_default_agrees(self):
        exact = batch_ls_nonlin_meas_lin_dyn(X_INIT, Z_NONLIN, _h, F, R, 0, _hj, 10)
        numeric = batch_ls_nonlin_meas_lin_dyn(X_INIT, Z_NONLIN, _h, F, R, 0, None, 10)
        np.testing.assert_allclose(numeric.x, exact.x, rtol=1e-6)


class TestLevenbergMarquardtBatches:
    def test_lm_matches_matlab_optimum(self):
        # MATLAB's two-output form of this function crashes upstream
        # (its covariance block reads a variable only built for three
        # outputs); the fixture used the three-output call.
        res = batch_ls_nonlin_meas_lin_dyn_lm(X_INIT, Z_NONLIN, _h, F, R, 0, None, _hj)
        assert res.success
        np.testing.assert_allclose(
            res.x, _load("batchls_lm_x.csv").ravel(), atol=ATOL_LM
        )
        np.testing.assert_allclose(res.P, _load("batchls_lm_P.csv"), atol=ATOL_LM)

    def test_lm_trajectory_mode_matches_matlab(self):
        res = batch_ls_nonlin_meas_lin_dyn_lm(X_INIT, Z_NONLIN, _h, F, R, 0, Q, _hj)
        assert res.success
        np.testing.assert_allclose(
            res.x, _load("batchls_lmq_x.csv").ravel(), atol=ATOL_LM
        )
        np.testing.assert_allclose(
            res.x_batch, _load("batchls_lmq_xbatch.csv"), atol=ATOL_LM
        )

    def test_nonlin_dyn_lm_matches_matlab_optimum(self):
        hs, hjs = _folded()
        res = batch_ls_nonlin_meas_nonlin_dyn_lm(X_INIT, Z_NONLIN, hs, R, 0, hjs)
        assert res.success
        np.testing.assert_allclose(
            res.x, _load("batchls_nnlm_x.csv").ravel(), atol=ATOL_LM
        )

    def test_nonlin_dyn_lm_covariance_matches_gauss_newton_sibling(self):
        # The upstream covariance inverts the stacked Cholesky factors
        # instead of R (disagreeing with its own non-LM sibling); the
        # port computes the R-inverse form, so it must agree with the
        # Gauss-Newton estimator's covariance at the same optimum.
        hs, hjs = _folded()
        lm = batch_ls_nonlin_meas_nonlin_dyn_lm(X_INIT, Z_NONLIN, hs, R, 0, hjs)
        gn = batch_ls_nonlin_meas_nonlin_dyn(X_INIT, Z_NONLIN, hs, R, hjs, 15)
        np.testing.assert_allclose(lm.P, gn.P, rtol=1e-5)


class TestTwoPointDiffInit:
    Z_PAIRS = np.stack(
        [
            np.array([[0.0, 2.0], [1.0, 0.0]]),
            np.array([[1.0, 1.5], [-1.0, -0.5]]),
        ],
        axis=2,
    )
    R_TP = np.array([[0.04, 0.01], [0.01, 0.09]])

    def test_matches_matlab(self):
        res = two_point_diff_init(2.0, self.Z_PAIRS, self.R_TP, 0.5)
        np.testing.assert_allclose(res.x, _load("twopoint_x.csv"), atol=ATOL)
        np.testing.assert_allclose(
            np.vstack([res.P[:, :, 0], res.P[:, :, 1]]),
            _load("twopoint_P.csv"),
            atol=ATOL,
        )

    def test_single_pair_form(self):
        res = two_point_diff_init(2.0, self.Z_PAIRS[:, :, 0], self.R_TP, 0.5)
        np.testing.assert_allclose(res.x, _load("twopoint_x.csv")[:, 0], atol=ATOL)

    def test_velocity_is_the_difference_quotient(self):
        res = two_point_diff_init(0.5, np.array([[1.0, 3.0]]), np.eye(1))
        np.testing.assert_allclose(res.x, [3.0, 4.0], atol=1e-12)
