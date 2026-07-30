"""Correctness audit tests for pytcl.dynamic_estimation.

Every filter variant is checked against independently-computed references:

- Linear-Gaussian problems: closed-form Kalman algebra computed inline with
  numpy. All algebraically-equivalent variants (standard KF, square-root KF,
  U-D, information filter, SRIF) must produce identical state/covariance.
- Nonlinear filters (EKF/UKF/CKF/SR-UKF): must reduce to the linear KF on
  linear problems; NEES chi-squared consistency on a nonlinear problem.
- Particle filters: bootstrap PF must approach the KF posterior on a
  linear-Gaussian problem; resampling and moment utilities checked
  analytically.
- Smoothers: RTS vs an independent inline backward recursion, Loewner
  ordering, fixed-lag vs truncated RTS, two-filter vs RTS.
- IMM: reduction to a single KF for identical models; mode-probability
  tracking through a maneuver.
- H-infinity: reduction to the KF as gamma -> infinity.
"""

import numpy as np
import pytest
from scipy.stats import chi2, multivariate_normal

from pytcl.dynamic_estimation.gaussian_sum_filter import (
    GaussianComponent,
    GaussianSumFilter,
    gaussian_sum_filter_predict,
    gaussian_sum_filter_update,
)
from pytcl.dynamic_estimation.imm import (
    IMMEstimator,
    combine_estimates,
    compute_mixing_probabilities,
    imm_predict_update,
    mix_states,
)
from pytcl.dynamic_estimation.information_filter import (
    InformationState,
    fuse_information,
    information_filter,
    information_to_state,
    srif_filter,
    srif_predict,
    srif_update,
    state_to_information,
)
from pytcl.dynamic_estimation.kalman.constrained import (
    ConstrainedEKF,
    ConstraintFunction,
    constrained_ekf_update,
)
from pytcl.dynamic_estimation.kalman.extended import (
    ekf_predict,
    ekf_predict_auto,
    ekf_update,
    ekf_update_auto,
    iterated_ekf_update,
    numerical_jacobian,
)
from pytcl.dynamic_estimation.kalman.h_infinity import (
    extended_hinf_update,
    find_min_gamma,
    hinf_predict,
    hinf_predict_update,
    hinf_update,
)
from pytcl.dynamic_estimation.kalman.linear import (
    information_filter_predict,
    information_filter_update,
    kf_predict,
    kf_predict_update,
    kf_update,
)
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
from pytcl.dynamic_estimation.kalman.unscented import (
    ckf_predict,
    ckf_spherical_cubature_points,
    ckf_update,
    sigma_points_julier,
    sigma_points_merwe,
    ukf_predict,
    ukf_update,
    unscented_transform,
)
from pytcl.dynamic_estimation.particle_filters import (
    bootstrap_pf_step,
    bootstrap_pf_update,
    effective_sample_size,
    gaussian_likelihood,
    initialize_particles,
    particle_covariance,
    particle_mean,
    resample_multinomial,
    resample_residual,
    resample_systematic,
)
from pytcl.dynamic_estimation.rbpf import RBPFFilter, rbpf_predict, rbpf_update
from pytcl.dynamic_estimation.smoothers import (
    fixed_interval_smoother,
    fixed_lag_smoother,
    rts_smoother,
    rts_smoother_single_step,
    two_filter_smoother,
)

# ---------------------------------------------------------------------------
# Shared linear-Gaussian tracking scenario (CV model, position measurement)
# ---------------------------------------------------------------------------

F = np.array([[1.0, 1.0], [0.0, 1.0]])
# Strictly positive definite: the pure gain-vector form [[0.25, 0.5], [0.5, 1.0]]
# has det == 0 exactly, and Cholesky on it is BLAS-roundoff luck (passes on
# Accelerate, raises on OpenBLAS). The jitter keeps every filter variant on
# the same well-posed problem.
Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.1 + np.eye(2) * 1e-6
H = np.array([[1.0, 0.0]])
R = np.array([[0.5]])
X0 = np.array([0.0, 1.0])
P0 = np.array([[2.0, 0.3], [0.3, 1.0]])
N_STEPS = 10


def _simulate(seed=7, nsteps=N_STEPS):
    rng = np.random.default_rng(seed)
    xt = X0.copy()
    zs = []
    for _ in range(nsteps):
        xt = F @ xt + rng.multivariate_normal(np.zeros(2), Q)
        zs.append(H @ xt + rng.multivariate_normal(np.zeros(1), R))
    return zs


ZS = _simulate()


def _closed_form_filter(zs=ZS):
    """Inline closed-form Kalman recursion (independent reference)."""
    x, P = X0.copy(), P0.copy()
    filt, pred = [], []
    for z in zs:
        x = F @ x
        P = F @ P @ F.T + Q
        pred.append((x.copy(), P.copy()))
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(2) - K @ H) @ P
        filt.append((x.copy(), P.copy()))
    return filt, pred


TRUTH_FILT, TRUTH_PRED = _closed_form_filter()


def _assert_matches_truth(states, atol=1e-9):
    for (xa, Pa), (xb, Pb) in zip(TRUTH_FILT, states):
        np.testing.assert_allclose(xb, xa, atol=atol)
        np.testing.assert_allclose(Pb, Pa, atol=atol)


def f_lin(x):
    return F @ x


def h_lin(x):
    return H @ x


# ---------------------------------------------------------------------------
# Linear-Gaussian equivalence: every variant must agree with closed form
# ---------------------------------------------------------------------------


class TestLinearEquivalence:
    def test_kf(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = kf_predict(x, P, F, Q)
            upd = kf_update(pred.x, pred.P, z, H, R)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out)

    def test_kf_predict_update(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            upd = kf_predict_update(x, P, z, F, Q, H, R)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out)

    def test_kf_control_input(self):
        B = np.array([[0.5], [1.0]])
        u = np.array([0.3])
        pred = kf_predict(X0, P0, F, Q, B=B, u=u)
        np.testing.assert_allclose(pred.x, F @ X0 + B @ u)
        np.testing.assert_allclose(pred.P, F @ P0 @ F.T + Q)

    def test_kf_likelihood(self):
        pred = kf_predict(X0, P0, F, Q)
        upd = kf_update(pred.x, pred.P, ZS[0], H, R)
        S = H @ pred.P @ H.T + R
        ref = multivariate_normal.pdf(ZS[0], mean=H @ pred.x, cov=S)
        assert np.isclose(upd.likelihood, ref)

    def test_srkf(self):
        x = X0.copy()
        S = np.linalg.cholesky(P0)
        SQ = np.linalg.cholesky(Q)
        SR = np.linalg.cholesky(R)
        out = []
        for z in ZS:
            pred = srkf_predict(x, S, F, SQ)
            upd = srkf_update(pred.x, pred.S, z, H, SR)
            x, S = upd.x, upd.S
            out.append((x, S @ S.T))
        _assert_matches_truth(out)

    def test_srkf_predict_update(self):
        S = np.linalg.cholesky(P0)
        SQ = np.linalg.cholesky(Q)
        SR = np.linalg.cholesky(R)
        upd = srkf_predict_update(X0, S, ZS[0], F, SQ, H, SR)
        xa, Pa = TRUTH_FILT[0]
        np.testing.assert_allclose(upd.x, xa, atol=1e-9)
        np.testing.assert_allclose(upd.S @ upd.S.T, Pa, atol=1e-9)

    def test_ud_filter(self):
        x = X0.copy()
        U, D = ud_factorize(P0)
        out = []
        for z in ZS:
            x, U, D = ud_predict(x, U, D, F, Q)
            x, U, D, _, _ = ud_update(x, U, D, z, H, R)
            out.append((x, ud_reconstruct(U, D)))
        _assert_matches_truth(out)

    def test_information_filter_steps(self):
        y, Y = state_to_information(X0, P0)
        out = []
        for z in ZS:
            y, Y = information_filter_predict(y, Y, F, Q)
            y, Y = information_filter_update(y, Y, z, H, R)
            out.append(information_to_state(y, Y))
        _assert_matches_truth(out)

    def test_information_filter_driver(self):
        y0, Y0 = state_to_information(X0, P0)
        res = information_filter(y0, Y0, ZS, F, Q, H, R)
        _assert_matches_truth(list(zip(res.x_filt, res.P_filt)))

    def test_srif_steps(self):
        Y0 = np.linalg.inv(P0)
        R0 = np.linalg.cholesky(Y0).T  # upper triangular, R0.T @ R0 = Y0
        r, Rm = R0 @ X0, R0.copy()
        out = []
        for z in ZS:
            r, Rm = srif_predict(r, Rm, F, Q)
            r, Rm = srif_update(r, Rm, z, H, R)
            Y = Rm.T @ Rm
            out.append((np.linalg.solve(Y, Rm.T @ r), np.linalg.inv(Y)))
        _assert_matches_truth(out, atol=1e-8)

    def test_srif_driver(self):
        Y0 = np.linalg.inv(P0)
        R0 = np.linalg.cholesky(Y0).T
        res = srif_filter(R0 @ X0, R0, ZS, F, Q, H, R)
        _assert_matches_truth(list(zip(res.x_filt, res.P_filt)), atol=1e-8)

    def test_hinf_large_gamma_equals_kf(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = hinf_predict(x, P, F, Q)
            upd = hinf_update(pred.x, pred.P, z, H, R, gamma=1e8)
            assert upd.feasible
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out, atol=1e-6)

    def test_ekf_reduces_to_kf(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = ekf_predict(x, P, f_lin, F, Q)
            upd = ekf_update(pred.x, pred.P, z, h_lin, H, R)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out)

    def test_ekf_auto_reduces_to_kf(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = ekf_predict_auto(x, P, f_lin, Q)
            upd = ekf_update_auto(pred.x, pred.P, z, h_lin, R)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out, atol=1e-5)

    def test_iterated_ekf_reduces_to_kf(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = kf_predict(x, P, F, Q)
            upd = iterated_ekf_update(pred.x, pred.P, z, h_lin, lambda x: H, R)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out)

    @pytest.mark.parametrize("alpha,atol", [(1.0, 1e-8), (1e-3, 1e-6)])
    def test_ukf_reduces_to_kf(self, alpha, atol):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = ukf_predict(x, P, f_lin, Q, alpha=alpha)
            upd = ukf_update(pred.x, pred.P, z, h_lin, R, alpha=alpha)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out, atol=atol)

    def test_ckf_reduces_to_kf(self):
        x, P = X0.copy(), P0.copy()
        out = []
        for z in ZS:
            pred = ckf_predict(x, P, f_lin, Q)
            upd = ckf_update(pred.x, pred.P, z, h_lin, R)
            x, P = upd.x, upd.P
            out.append((x, P))
        _assert_matches_truth(out, atol=1e-8)

    @pytest.mark.parametrize("alpha,atol", [(1.0, 1e-7), (1e-3, 1e-6)])
    def test_sr_ukf_reduces_to_kf(self, alpha, atol):
        x = X0.copy()
        S = np.linalg.cholesky(P0)
        SQ = np.linalg.cholesky(Q)
        SR = np.linalg.cholesky(R)
        out = []
        for z in ZS:
            pred = sr_ukf_predict(x, S, f_lin, SQ, alpha=alpha)
            upd = sr_ukf_update(pred.x, pred.S, z, h_lin, SR, alpha=alpha)
            x, S = upd.x, upd.S
            out.append((x, S @ S.T))
        _assert_matches_truth(out, atol=atol)


# ---------------------------------------------------------------------------
# U-D filter internals
# ---------------------------------------------------------------------------


class TestUDFilter:
    P3 = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])

    def test_factorize_reconstruct_roundtrip(self):
        U, D = ud_factorize(self.P3)
        assert np.allclose(np.diag(U), 1.0)
        assert np.allclose(U, np.triu(U))
        np.testing.assert_allclose(ud_reconstruct(U, D), self.P3, atol=1e-12)

    def test_scalar_update_vs_joseph(self):
        x = np.array([1.0, -2.0, 0.5])
        h = np.array([1.0, 2.0, -1.0])
        r, z = 0.3, 1.7
        U, D = ud_factorize(self.P3)
        x_u, U_u, D_u = ud_update_scalar(x, U, D, z, h, r)
        S = h @ self.P3 @ h + r
        K = self.P3 @ h / S
        np.testing.assert_allclose(x_u, x + K * (z - h @ x), atol=1e-12)
        P_ref = self.P3 - np.outer(K, h @ self.P3)
        np.testing.assert_allclose(ud_reconstruct(U_u, D_u), P_ref, atol=1e-12)

    def test_vector_update_correlated_noise(self):
        x = np.array([1.0, -2.0, 0.5])
        H2 = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, -0.5]])
        R2 = np.array([[0.5, 0.1], [0.1, 0.4]])
        z2 = np.array([1.0, 2.0])
        U, D = ud_factorize(self.P3)
        x_u, U_u, D_u, y_u, lik = ud_update(x, U, D, z2, H2, R2)
        S = H2 @ self.P3 @ H2.T + R2
        K = self.P3 @ H2.T @ np.linalg.inv(S)
        np.testing.assert_allclose(x_u, x + K @ (z2 - H2 @ x), atol=1e-10)
        P_ref = (np.eye(3) - K @ H2) @ self.P3
        np.testing.assert_allclose(ud_reconstruct(U_u, D_u), P_ref, atol=1e-10)
        np.testing.assert_allclose(y_u, z2 - H2 @ x)
        ref_lik = multivariate_normal.pdf(z2, mean=H2 @ x, cov=S)
        assert np.isclose(lik, ref_lik)


# ---------------------------------------------------------------------------
# Sigma points and unscented transform
# ---------------------------------------------------------------------------


class TestSigmaPoints:
    x = np.array([1.0, -2.0, 0.5])
    P = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])

    @pytest.mark.parametrize(
        "sp_kwargs",
        [dict(alpha=1e-3), dict(alpha=0.5, kappa=1.0), dict(alpha=1.0, beta=0.0)],
    )
    def test_merwe_moments(self, sp_kwargs):
        sp = sigma_points_merwe(self.x, self.P, **sp_kwargs)
        mean = np.sum(sp.Wm[:, None] * sp.points, axis=0)
        d = sp.points - mean
        cov = sum(sp.Wc[i] * np.outer(d[i], d[i]) for i in range(len(sp.Wm)))
        np.testing.assert_allclose(mean, self.x, atol=1e-9)
        np.testing.assert_allclose(cov, self.P, atol=1e-6)
        assert np.isclose(sp.Wm.sum(), 1.0)

    def test_julier_moments(self):
        sp = sigma_points_julier(self.x, self.P, kappa=1.0)
        mean = np.sum(sp.Wm[:, None] * sp.points, axis=0)
        d = sp.points - mean
        cov = sum(sp.Wc[i] * np.outer(d[i], d[i]) for i in range(len(sp.Wm)))
        np.testing.assert_allclose(mean, self.x, atol=1e-12)
        np.testing.assert_allclose(cov, self.P, atol=1e-9)

    @pytest.mark.parametrize("alpha", [0.9, 1e-3])
    def test_unscented_transform_identity(self, alpha):
        # Identity transform must reproduce input moments, including the
        # negative-central-weight regime (small alpha).
        sp = sigma_points_merwe(self.x, self.P, alpha=alpha)
        mean, cov = unscented_transform(sp.points, sp.Wm, sp.Wc)
        np.testing.assert_allclose(mean, self.x, atol=1e-9)
        np.testing.assert_allclose(cov, self.P, atol=1e-6)

    def test_unscented_transform_noise_cov(self):
        sp = sigma_points_merwe(self.x, self.P, alpha=1.0)
        noise = np.diag([0.1, 0.2, 0.3])
        _, cov = unscented_transform(sp.points, sp.Wm, sp.Wc, noise)
        np.testing.assert_allclose(cov, self.P + noise, atol=1e-9)

    def test_ckf_cubature_points(self):
        pts, wts = ckf_spherical_cubature_points(3)
        assert pts.shape == (6, 3)
        assert np.isclose(wts.sum(), 1.0)
        mean = np.sum(wts[:, None] * pts, axis=0)
        cov = sum(wts[i] * np.outer(pts[i], pts[i]) for i in range(6))
        np.testing.assert_allclose(mean, 0.0, atol=1e-12)
        np.testing.assert_allclose(cov, np.eye(3), atol=1e-12)

    def test_compute_merwe_weights_matches_sigma_points(self):
        Wm, Wc = compute_merwe_weights(3, alpha=0.5, beta=2.0, kappa=1.0)
        sp = sigma_points_merwe(self.x, self.P, alpha=0.5, beta=2.0, kappa=1.0)
        np.testing.assert_allclose(Wm, sp.Wm)
        np.testing.assert_allclose(Wc, sp.Wc)


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------


class TestMatrixUtils:
    P = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])

    def test_cholesky_update_downdate(self):
        S = np.linalg.cholesky(self.P)
        v = np.array([0.3, -0.2, 0.5])
        Su = cholesky_update(S, v, sign=1.0)
        np.testing.assert_allclose(Su @ Su.T, self.P + np.outer(v, v), atol=1e-12)
        Sd = cholesky_update(S, 0.5 * v, sign=-1.0)
        np.testing.assert_allclose(
            Sd @ Sd.T, self.P - 0.25 * np.outer(v, v), atol=1e-12
        )

    def test_cholesky_downdate_indefinite_raises(self):
        S = np.linalg.cholesky(np.eye(2))
        with pytest.raises(ValueError):
            cholesky_update(S, np.array([2.0, 0.0]), sign=-1.0)

    def test_qr_update(self):
        S = np.linalg.cholesky(self.P)
        Fm = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]])
        Qs = np.diag([0.1, 0.2, 0.3])
        Sn = qr_update(S, np.linalg.cholesky(Qs), Fm)
        np.testing.assert_allclose(Sn @ Sn.T, Fm @ self.P @ Fm.T + Qs, atol=1e-12)
        # F=None means identity
        Sn2 = qr_update(S, np.linalg.cholesky(Qs))
        np.testing.assert_allclose(Sn2 @ Sn2.T, self.P + Qs, atol=1e-12)

    def test_ensure_symmetric(self):
        M = np.array([[1.0, 2.0], [0.0, 1.0]])
        np.testing.assert_allclose(
            ensure_symmetric(M), np.array([[1.0, 1.0], [1.0, 1.0]])
        )

    def test_compute_matrix_sqrt(self):
        S = compute_matrix_sqrt(self.P, scale=2.0)
        np.testing.assert_allclose(S @ S.T, 2.0 * self.P, atol=1e-12)

    def test_innovation_likelihood_and_mahalanobis(self):
        y = np.array([0.3, -0.4, 0.1])
        ref = multivariate_normal.pdf(y, mean=np.zeros(3), cov=self.P)
        assert np.isclose(compute_innovation_likelihood(y, self.P), ref)
        S_chol = np.linalg.cholesky(self.P)
        assert np.isclose(
            compute_innovation_likelihood(y, S_chol, S_is_cholesky=True), ref
        )
        d_ref = np.sqrt(y @ np.linalg.solve(self.P, y))
        assert np.isclose(compute_mahalanobis_distance(y, self.P), d_ref)

    def test_numerical_jacobian(self):
        def f(x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        J = numerical_jacobian(f, np.array([2.0, 3.0]))
        np.testing.assert_allclose(J, [[4.0, 0.0], [3.0, 2.0]], atol=1e-5)


# ---------------------------------------------------------------------------
# Nonlinear consistency (NEES) and SR-UKF vs UKF agreement
# ---------------------------------------------------------------------------


def _h_rb(x):
    return np.array([np.hypot(x[0], x[2]), np.arctan2(x[2], x[0])])


def _H_rb(x):
    r2 = x[0] ** 2 + x[2] ** 2
    r = np.sqrt(r2)
    return np.array([[x[0] / r, 0.0, x[2] / r, 0.0], [-x[2] / r2, 0.0, x[0] / r2, 0.0]])


class TestNonlinearConsistency:
    def test_nees_chi2_consistency(self):
        """UKF and EKF NEES must be chi-squared consistent on a mildly
        nonlinear range-bearing tracking problem (seeded Monte Carlo)."""
        F4 = np.eye(4)
        F4[0, 1] = F4[2, 3] = 1.0
        Q4 = np.kron(np.eye(2), np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])) * 0.01
        Rrb = np.diag([0.5, 1e-4])
        n_mc, n_k = 80, 8
        rng = np.random.default_rng(99)
        nees_ukf, nees_ekf = [], []
        for _ in range(n_mc):
            x_true = np.array([50.0, 1.0, 30.0, -0.5]) + rng.multivariate_normal(
                np.zeros(4), np.eye(4) * 0.5
            )
            x_u = np.array([50.0, 1.0, 30.0, -0.5])
            P_u = np.eye(4) * 0.5
            x_e, P_e = x_u.copy(), P_u.copy()
            for _ in range(n_k):
                x_true = F4 @ x_true + rng.multivariate_normal(np.zeros(4), Q4)
                z = _h_rb(x_true) + rng.multivariate_normal(np.zeros(2), Rrb)
                pu = ukf_predict(x_u, P_u, lambda x: F4 @ x, Q4, alpha=1.0)
                uu = ukf_update(pu.x, pu.P, z, _h_rb, Rrb, alpha=1.0)
                x_u, P_u = uu.x, uu.P
                pe = ekf_predict(x_e, P_e, lambda x: F4 @ x, F4, Q4)
                ue = ekf_update(pe.x, pe.P, z, _h_rb, _H_rb(pe.x), Rrb)
                x_e, P_e = ue.x, ue.P
            e_u = x_u - x_true
            e_e = x_e - x_true
            nees_ukf.append(e_u @ np.linalg.solve(P_u, e_u))
            nees_ekf.append(e_e @ np.linalg.solve(P_e, e_e))
        lo = chi2.ppf(0.0005, 4 * n_mc) / n_mc
        hi = chi2.ppf(0.9995, 4 * n_mc) / n_mc
        assert lo < np.mean(nees_ukf) < hi
        assert lo < np.mean(nees_ekf) < hi

    @pytest.mark.parametrize("alpha", [1.0, 1e-3])
    def test_sr_ukf_matches_ukf_nonlinear(self, alpha):
        def h(x):
            return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])

        x = np.array([100.0, 50.0])
        P = np.array([[10.0, 2.0], [2.0, 8.0]])
        z = np.array([112.0, 0.47])
        Rm = np.diag([1.0, 1e-3])
        u1 = ukf_update(x, P, z, h, Rm, alpha=alpha)
        u2 = sr_ukf_update(
            x, np.linalg.cholesky(P), z, h, np.linalg.cholesky(Rm), alpha=alpha
        )
        np.testing.assert_allclose(u2.x, u1.x, atol=1e-8)
        np.testing.assert_allclose(u2.S @ u2.S.T, u1.P, atol=1e-8)
        np.testing.assert_allclose(u2.S_y @ u2.S_y.T, u1.S, atol=1e-8)

    def test_ckf_close_to_ukf_nonlinear(self):
        def f(x):
            return np.array([x[0] + 0.1 * np.sin(x[1]), 0.95 * x[1]])

        x = np.array([1.0, 0.5])
        P = np.eye(2) * 0.2
        Qm = np.eye(2) * 0.01
        p_ckf = ckf_predict(x, P, f, Qm)
        p_ukf = ukf_predict(x, P, f, Qm, alpha=1.0, kappa=0.0)
        # CKF is UKF with kappa=0, alpha=1 minus the (zero-weight) mean point
        np.testing.assert_allclose(p_ckf.x, p_ukf.x, atol=1e-3)
        np.testing.assert_allclose(p_ckf.P, p_ukf.P, atol=1e-3)


# ---------------------------------------------------------------------------
# Smoothers
# ---------------------------------------------------------------------------


def _reference_rts(zs=ZS):
    """Independent inline RTS reference from the closed-form filter."""
    filt, pred = _closed_form_filter(zs)
    n = len(zs)
    xs = [None] * n
    Ps = [None] * n
    xs[-1], Ps[-1] = filt[-1]
    for k in range(n - 2, -1, -1):
        x_f, P_f = filt[k]
        x_p, P_p = pred[k + 1]
        G = P_f @ F.T @ np.linalg.inv(P_p)
        xs[k] = x_f + G @ (xs[k + 1] - x_p)
        Ps[k] = P_f + G @ (Ps[k + 1] - P_p) @ G.T
    return xs, Ps


class TestSmoothers:
    def test_rts_vs_inline_reference(self):
        xs_ref, Ps_ref = _reference_rts()
        res = rts_smoother(X0, P0, ZS, F, Q, H, R)
        for k in range(N_STEPS):
            np.testing.assert_allclose(res.x_smooth[k], xs_ref[k], atol=1e-9)
            np.testing.assert_allclose(res.P_smooth[k], Ps_ref[k], atol=1e-9)

    def test_smoothed_cov_le_filtered_cov(self):
        res = rts_smoother(X0, P0, ZS, F, Q, H, R)
        for k in range(N_STEPS):
            diff = res.P_filt[k] - res.P_smooth[k]
            assert np.min(np.linalg.eigvalsh(diff)) > -1e-10

    def test_fixed_interval_equals_rts(self):
        res1 = rts_smoother(X0, P0, ZS, F, Q, H, R)
        res2 = fixed_interval_smoother(X0, P0, ZS, F, Q, H, R)
        for k in range(N_STEPS):
            np.testing.assert_allclose(res2.x_smooth[k], res1.x_smooth[k])
            np.testing.assert_allclose(res2.P_smooth[k], res1.P_smooth[k])

    def test_fixed_lag_equals_truncated_rts(self):
        lag = 3
        fl = fixed_lag_smoother(X0, P0, ZS, F, Q, H, R, lag=lag)
        assert fl.lag == lag
        for k in range(lag, N_STEPS):
            ref = rts_smoother(X0, P0, ZS[: k + 1], F, Q, H, R)
            np.testing.assert_allclose(fl.x_smooth[k], ref.x_smooth[k - lag], atol=1e-8)
            np.testing.assert_allclose(fl.P_smooth[k], ref.P_smooth[k - lag], atol=1e-8)

    def test_two_filter_equals_rts(self):
        """With a diffuse backward prior the two-filter smoother must agree
        with RTS (measurements must not be double-counted)."""
        res = rts_smoother(X0, P0, ZS, F, Q, H, R)
        tf = two_filter_smoother(X0, P0, np.zeros(2), np.eye(2) * 1e8, ZS, F, Q, H, R)
        for k in range(N_STEPS):
            np.testing.assert_allclose(tf.x_smooth[k], res.x_smooth[k], atol=1e-6)
            np.testing.assert_allclose(tf.P_smooth[k], res.P_smooth[k], atol=1e-6)

    def test_rts_single_step(self):
        res = rts_smoother(X0, P0, ZS, F, Q, H, R)
        pred = kf_predict(res.x_filt[3], res.P_filt[3], F, Q)
        ss = rts_smoother_single_step(
            res.x_filt[3],
            res.P_filt[3],
            pred.x,
            pred.P,
            res.x_smooth[4],
            res.P_smooth[4],
            F,
        )
        np.testing.assert_allclose(ss.x, res.x_smooth[3], atol=1e-10)
        np.testing.assert_allclose(ss.P, res.P_smooth[3], atol=1e-10)


# ---------------------------------------------------------------------------
# Information filter specifics
# ---------------------------------------------------------------------------


class TestInformationFilter:
    def test_state_information_roundtrip(self):
        y, Y = state_to_information(X0, P0)
        np.testing.assert_allclose(Y, np.linalg.inv(P0))
        x, P = information_to_state(y, Y)
        np.testing.assert_allclose(x, X0, atol=1e-12)
        np.testing.assert_allclose(P, P0, atol=1e-12)

    def test_fuse_information_additive(self):
        s1 = InformationState(y=np.array([1.0, 0.5]), Y=np.diag([0.5, 0.1]))
        s2 = InformationState(y=np.array([0.8, 0.6]), Y=np.diag([0.3, 0.2]))
        fused = fuse_information([s1, s2])
        np.testing.assert_allclose(fused.y, [1.8, 1.1])
        np.testing.assert_allclose(fused.Y, np.diag([0.8, 0.3]))

    def test_fusion_equals_stacked_kf_update(self):
        P = np.eye(2) * 2.0
        x = np.array([1.0, -0.5])
        H1 = np.array([[1.0, 0.0]])
        H2 = np.array([[0.0, 1.0]])
        R1 = np.array([[0.5]])
        R2 = np.array([[0.3]])
        z1, z2 = np.array([1.2]), np.array([-0.3])
        Y0 = np.linalg.inv(P)
        prior = InformationState(y=Y0 @ x, Y=Y0)
        c1 = InformationState(
            y=H1.T @ np.linalg.inv(R1) @ z1, Y=H1.T @ np.linalg.inv(R1) @ H1
        )
        c2 = InformationState(
            y=H2.T @ np.linalg.inv(R2) @ z2, Y=H2.T @ np.linalg.inv(R2) @ H2
        )
        fused = fuse_information([prior, c1, c2])
        P_f = np.linalg.inv(fused.Y)
        x_f = P_f @ fused.y
        upd = kf_update(
            x, P, np.concatenate([z1, z2]), np.vstack([H1, H2]), np.diag([0.5, 0.3])
        )
        np.testing.assert_allclose(x_f, upd.x, atol=1e-10)
        np.testing.assert_allclose(P_f, upd.P, atol=1e-10)

    def test_diffuse_start_matches_diffuse_kf(self):
        """Y0 = 0 (fully unknown state) must converge to the same estimate
        as a Kalman filter with a very diffuse prior."""
        res = information_filter(np.zeros(2), np.zeros((2, 2)), ZS, F, Q, H, R)
        x, P = np.zeros(2), np.eye(2) * 1e10
        for z in ZS:
            pred = kf_predict(x, P, F, Q)
            upd = kf_update(pred.x, pred.P, z, H, R)
            x, P = upd.x, upd.P
        np.testing.assert_allclose(res.x_filt[-1], x, atol=1e-4)
        np.testing.assert_allclose(res.P_filt[-1], P, atol=1e-4)


# ---------------------------------------------------------------------------
# IMM
# ---------------------------------------------------------------------------


class TestIMM:
    Pi = np.array([[0.95, 0.05], [0.05, 0.95]])

    def test_identical_models_reduce_to_kf(self):
        mode_states = [X0.copy(), X0.copy()]
        mode_covs = [P0.copy(), P0.copy()]
        mode_probs = np.array([0.6, 0.4])
        x_kf, P_kf = X0.copy(), P0.copy()
        for z in ZS:
            upd = imm_predict_update(
                mode_states,
                mode_covs,
                mode_probs,
                self.Pi,
                z,
                [F, F],
                [Q, Q],
                [H, H],
                [R, R],
            )
            mode_states = upd.mode_states
            mode_covs = upd.mode_covs
            mode_probs = upd.mode_probs
            pred = kf_predict(x_kf, P_kf, F, Q)
            u = kf_update(pred.x, pred.P, z, H, R)
            x_kf, P_kf = u.x, u.P
            np.testing.assert_allclose(upd.x, x_kf, atol=1e-9)
            np.testing.assert_allclose(upd.P, P_kf, atol=1e-9)

    def test_mixing_probabilities_analytic(self):
        mu = np.array([0.6, 0.4])
        mp, c_bar = compute_mixing_probabilities(mu, self.Pi)
        np.testing.assert_allclose(c_bar, self.Pi.T @ mu)
        np.testing.assert_allclose(mp, (self.Pi * mu[:, None]) / (self.Pi.T @ mu))
        np.testing.assert_allclose(mp.sum(axis=0), [1.0, 1.0])

    def test_combine_and_mix_moments(self):
        s1, s2 = np.array([1.0, 0.0]), np.array([3.0, 1.0])
        c1, c2 = np.eye(2), 2 * np.eye(2)
        w = np.array([0.3, 0.7])
        xc, Pc = combine_estimates([s1, s2], [c1, c2], w)
        x_ref = 0.3 * s1 + 0.7 * s2
        P_ref = 0.3 * (c1 + np.outer(s1 - x_ref, s1 - x_ref)) + 0.7 * (
            c2 + np.outer(s2 - x_ref, s2 - x_ref)
        )
        np.testing.assert_allclose(xc, x_ref)
        np.testing.assert_allclose(Pc, P_ref)
        mp, _ = compute_mixing_probabilities(w, self.Pi)
        ms, mc = mix_states([s1, s2], [c1, c2], mp)
        for j in range(2):
            wj = mp[:, j]
            xr = wj[0] * s1 + wj[1] * s2
            Pr = wj[0] * (c1 + np.outer(s1 - xr, s1 - xr)) + wj[1] * (
                c2 + np.outer(s2 - xr, s2 - xr)
            )
            np.testing.assert_allclose(ms[j], xr)
            np.testing.assert_allclose(mc[j], Pr)

    def test_mode_probabilities_track_maneuver(self):
        rng = np.random.default_rng(11)
        Q_low = np.eye(2) * 1e-4
        Q_high = np.eye(2) * 0.5
        imm = IMMEstimator(2, 2, self.Pi)
        imm.initialize(np.array([0.0, 1.0]), np.eye(2) * 0.1)
        imm.set_mode_model(0, F, Q_low)
        imm.set_mode_model(1, F, Q_high)
        imm.set_measurement_model(H, np.array([[0.05]]))
        xt = np.array([0.0, 1.0])
        probs = []
        for k in range(30):
            if k < 15:
                xt = F @ xt
            else:
                xt = F @ xt + rng.multivariate_normal(np.zeros(2), Q_high)
            z = H @ xt + rng.normal(0, np.sqrt(0.05), 1)
            upd = imm.predict_update(z)
            probs.append(upd.mode_probs.copy())
        # Quiescent phase favors the low-noise mode; maneuver shifts weight
        assert np.mean([p[0] for p in probs[5:14]]) > 0.8
        assert np.mean([p[1] for p in probs[20:]]) > 0.5
        state = imm.get_state()
        np.testing.assert_allclose(state.mode_probs, probs[-1])


# ---------------------------------------------------------------------------
# H-infinity
# ---------------------------------------------------------------------------


class TestHInfinity:
    def test_predict_update_wrapper(self):
        upd = hinf_predict_update(X0, P0, ZS[0], F, Q, H, R, gamma=1e8)
        pred = kf_predict(X0, P0, F, Q)
        ref = kf_update(pred.x, pred.P, ZS[0], H, R)
        np.testing.assert_allclose(upd.x, ref.x, atol=1e-6)
        np.testing.assert_allclose(upd.P, ref.P, atol=1e-6)

    def test_extended_equals_linear(self):
        u1 = hinf_update(X0, P0, ZS[0], H, R, gamma=10.0)
        u2 = extended_hinf_update(X0, P0, ZS[0], h_lin, H, R, gamma=10.0)
        np.testing.assert_allclose(u2.x, u1.x)
        np.testing.assert_allclose(u2.P, u1.P)

    def test_find_min_gamma_boundary(self):
        gmin = find_min_gamma(P0, H, R)

        def min_eig(g):
            M = (
                np.linalg.inv(P0)
                - (1.0 / g**2) * np.eye(2)
                + H.T @ np.linalg.inv(R) @ H
            )
            return np.min(np.linalg.eigvalsh(M))

        assert min_eig(gmin * 1.01) > 0
        assert min_eig(gmin * 0.9) < 0
        assert hinf_update(X0, P0, ZS[0], H, R, gamma=gmin * 1.05).feasible
        assert not hinf_update(X0, P0, ZS[0], H, R, gamma=gmin * 0.5).feasible

    def test_smaller_gamma_larger_covariance(self):
        gmin = find_min_gamma(P0, H, R)
        u_robust = hinf_update(X0, P0, ZS[0], H, R, gamma=gmin * 1.1)
        u_kf = hinf_update(X0, P0, ZS[0], H, R, gamma=1e8)
        diff = u_robust.P - u_kf.P
        assert np.min(np.linalg.eigvalsh(diff)) > -1e-12


# ---------------------------------------------------------------------------
# Constrained EKF
# ---------------------------------------------------------------------------


class TestConstrainedEKF:
    def test_no_constraints_equals_ekf(self):
        cekf = ConstrainedEKF()
        r1 = cekf.update(X0, P0, ZS[0], h_lin, H, R)
        r2 = ekf_update(X0, P0, ZS[0], h_lin, H, R)
        np.testing.assert_allclose(r1.x, r2.x)
        np.testing.assert_allclose(r1.P, r2.P)
        pred = cekf.predict(X0, P0, f_lin, F, Q)
        ref = ekf_predict(X0, P0, f_lin, F, Q)
        np.testing.assert_allclose(pred.x, ref.x)
        np.testing.assert_allclose(pred.P, ref.P)

    def test_equality_constraint_projection(self):
        A = np.array([[1.0, -1.0]])
        con = ConstraintFunction(
            g=lambda x: A @ x, G=lambda x: A, constraint_type="equality"
        )
        res = constrained_ekf_update(
            X0, P0, np.array([0.6]), h_lin, H, R, constraints=[con]
        )
        # Constraint satisfied
        assert abs((A @ res.x)[0]) < 1e-5
        # Matches analytic minimum-variance projection of unconstrained update
        u = ekf_update(X0, P0, np.array([0.6]), h_lin, H, R)
        lam = np.linalg.solve(A @ u.P @ A.T, A @ u.x)
        x_ref = u.x - u.P @ A.T @ lam
        np.testing.assert_allclose(res.x, x_ref, atol=1e-5)

    def test_constraint_function_numeric_jacobian(self):
        con = ConstraintFunction(g=lambda x: np.array([x[0] ** 2 + x[1] - 1.0]))
        J = con.jacobian(np.array([2.0, 3.0]))
        np.testing.assert_allclose(J, [[4.0, 1.0]], atol=1e-4)
        assert con.is_satisfied(np.array([0.0, 0.5]))
        assert not con.is_satisfied(np.array([2.0, 3.0]))


# ---------------------------------------------------------------------------
# Particle filters
# ---------------------------------------------------------------------------


class TestParticleFilters:
    def test_effective_sample_size(self):
        assert np.isclose(
            effective_sample_size(np.array([0.25, 0.25, 0.25, 0.25])), 4.0
        )
        w = np.array([0.7, 0.1, 0.1, 0.1])
        assert np.isclose(effective_sample_size(w), 1.0 / np.sum(w**2))

    @pytest.mark.parametrize(
        "resample_fn", [resample_multinomial, resample_systematic, resample_residual]
    )
    def test_resampling_preserves_weighted_mean(self, resample_fn):
        rng = np.random.default_rng(1)
        parts = rng.normal(0, 1, (20000, 2))
        w = rng.uniform(0, 1, 20000)
        w /= w.sum()
        target = particle_mean(parts, w)
        rs = resample_fn(parts, w, np.random.default_rng(2))
        assert rs.shape == parts.shape
        np.testing.assert_allclose(rs.mean(axis=0), target, atol=0.05)

    def test_residual_resampling_floor_counts(self):
        parts = np.arange(5, dtype=float).reshape(-1, 1)
        w = np.array([0.1, 0.3, 0.05, 0.35, 0.2])
        rs = resample_residual(parts, w, np.random.default_rng(0))
        counts = np.array([np.sum(rs == v) for v in range(5)])
        assert np.all(counts >= np.floor(5 * w))

    def test_gaussian_likelihood_vs_scipy(self):
        z = np.array([1.0, -0.5])
        zp = np.array([0.7, 0.1])
        Rm = np.array([[0.5, 0.1], [0.1, 0.4]])
        ref = multivariate_normal.pdf(z, mean=zp, cov=Rm)
        assert np.isclose(gaussian_likelihood(z, zp, Rm), ref)

    def test_particle_moments_analytic(self):
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        w = np.array([0.1, 0.2, 0.3, 0.4])
        m_ref = np.sum(w[:, None] * p, axis=0)
        C_ref = sum(w[i] * np.outer(p[i] - m_ref, p[i] - m_ref) for i in range(4))
        np.testing.assert_allclose(particle_mean(p, w), m_ref)
        np.testing.assert_allclose(particle_covariance(p, w), C_ref)

    def test_initialize_particles_moments(self):
        rng = np.random.default_rng(42)
        st = initialize_particles(
            np.array([1.0, -1.0]), np.diag([0.5, 2.0]), 100000, rng
        )
        assert np.allclose(st.weights, 1.0 / 100000)
        np.testing.assert_allclose(st.particles.mean(axis=0), [1.0, -1.0], atol=0.03)
        np.testing.assert_allclose(
            np.cov(st.particles.T), np.diag([0.5, 2.0]), atol=0.05
        )

    def test_update_weights_and_log_likelihood(self):
        """Weight update is Bayes' rule; the returned log-likelihood must
        match the analytic KF marginal p(z) = N(z; 0, P0 + R)."""
        rng = np.random.default_rng(3)
        Np = 200000
        Rm = np.array([[0.4]])
        parts = rng.normal(0.0, 1.0, (Np, 1))  # exact prior N(0, 1)
        w0 = np.ones(Np) / Np
        z = np.array([0.5])
        w_new, loglik = bootstrap_pf_update(
            parts, w0, z, lambda z, x: gaussian_likelihood(z, x, Rm)
        )
        assert np.isclose(w_new.sum(), 1.0)
        ref = multivariate_normal.logpdf(z, mean=[0.0], cov=np.array([[1.0]]) + Rm)
        assert abs(loglik - ref) < 0.02
        # Posterior mean matches the KF update
        u = kf_update(np.array([0.0]), np.array([[1.0]]), z, np.array([[1.0]]), Rm)
        np.testing.assert_allclose(particle_mean(parts, w_new), u.x, atol=0.01)

    def test_bootstrap_pf_matches_kf(self):
        rng = np.random.default_rng(5)
        Qp = np.eye(2) * 0.05
        LQ = np.linalg.cholesky(Qp)
        st = initialize_particles(X0, np.eye(2) * 0.5, 20000, rng)
        particles, weights = st.particles, st.weights
        x_kf, P_kf = X0.copy(), np.eye(2) * 0.5
        xt = X0.copy()
        for _ in range(10):
            xt = F @ xt + rng.multivariate_normal(np.zeros(2), Qp)
            z = H @ xt + rng.normal(0, np.sqrt(0.5), 1)
            stp = bootstrap_pf_step(
                particles,
                weights,
                z,
                f_lin,
                h_lin,
                lambda n, r: r.normal(0, 1, (n, 2)) @ LQ.T,
                R,
                rng=rng,
            )
            particles, weights = stp.particles, stp.weights
            pred = kf_predict(x_kf, P_kf, F, Qp)
            u = kf_update(pred.x, pred.P, z, H, R)
            x_kf, P_kf = u.x, u.P
            np.testing.assert_allclose(
                particle_mean(particles, weights), x_kf, atol=0.06
            )
        np.testing.assert_allclose(
            particle_covariance(particles, weights), P_kf, atol=0.05
        )


# ---------------------------------------------------------------------------
# Rao-Blackwellized particle filter
# ---------------------------------------------------------------------------


class TestRBPF:
    def test_trivial_nonlinear_part_equals_kf(self):
        """With a constant, noiseless nonlinear component the RBPF's linear
        part must reduce exactly to a Kalman filter."""
        np.random.seed(0)
        rb = RBPFFilter(max_particles=40)
        rb.initialize(np.array([0.0]), X0, np.eye(2) * 0.5, num_particles=40)
        Qy = np.zeros((1, 1))
        x_kf, P_kf = X0.copy(), np.eye(2) * 0.5
        for z in ZS[:6]:
            rb.predict(lambda y: y, np.eye(1), Qy, lambda x, y: F @ x, F, Q)
            rb.update(z, lambda x, y: H @ x, H, R)
            pred = kf_predict(x_kf, P_kf, F, Q)
            u = kf_update(pred.x, pred.P, z, H, R)
            x_kf, P_kf = u.x, u.P
            _, x_est, P_est = rb.estimate()
            np.testing.assert_allclose(x_est, x_kf, atol=1e-9)
            np.testing.assert_allclose(P_est, P_kf, atol=1e-9)

    def test_functional_interface_weights_normalized(self):
        np.random.seed(0)
        from pytcl.dynamic_estimation.rbpf import RBPFParticle

        particles = [
            RBPFParticle(y=np.array([0.0]), x=X0.copy(), P=P0.copy(), w=0.1)
            for _ in range(10)
        ]
        particles = rbpf_predict(
            particles,
            lambda y: y,
            np.eye(1),
            np.zeros((1, 1)),
            lambda x, y: F @ x,
            F,
            Q,
        )
        particles = rbpf_update(particles, ZS[0], lambda x, y: H @ x, H, R)
        assert np.isclose(sum(p.w for p in particles), 1.0)
        pred = kf_predict(X0, P0, F, Q)
        u = kf_update(pred.x, pred.P, ZS[0], H, R)
        x_mean = sum(p.w * p.x for p in particles)
        np.testing.assert_allclose(x_mean, u.x, atol=1e-10)


# ---------------------------------------------------------------------------
# Gaussian sum filter
# ---------------------------------------------------------------------------


class TestGaussianSum:
    def test_single_component_equals_kf(self):
        gsf = GaussianSumFilter()
        gsf.initialize(X0, P0, num_components=1)
        x_kf, P_kf = X0.copy(), P0.copy()
        for z in ZS[:6]:
            gsf.predict(f_lin, F, Q)
            gsf.update(z, h_lin, H, R)
            pred = kf_predict(x_kf, P_kf, F, Q)
            u = kf_update(pred.x, pred.P, z, H, R)
            x_kf, P_kf = u.x, u.P
            xg, Pg = gsf.estimate()
            np.testing.assert_allclose(xg, x_kf, atol=1e-9)
            np.testing.assert_allclose(Pg, P_kf, atol=1e-9)

    def test_functional_interface_equals_kf(self):
        comps = [GaussianComponent(x=X0.copy(), P=P0.copy(), w=1.0)]
        x_kf, P_kf = X0.copy(), P0.copy()
        for z in ZS[:6]:
            comps = gaussian_sum_filter_predict(comps, f_lin, F, Q)
            comps = gaussian_sum_filter_update(comps, z, h_lin, H, R)
            pred = kf_predict(x_kf, P_kf, F, Q)
            u = kf_update(pred.x, pred.P, z, H, R)
            x_kf, P_kf = u.x, u.P
            np.testing.assert_allclose(comps[0].x, x_kf, atol=1e-9)

    def test_weight_adaptation_bayes(self):
        Pc = np.eye(2) * 0.5
        c1 = GaussianComponent(x=np.array([1.0, 0.5]), P=Pc, w=0.5)
        c2 = GaussianComponent(x=np.array([3.0, 0.5]), P=Pc, w=0.5)
        z = np.array([1.1])
        updated = gaussian_sum_filter_update([c1, c2], z, h_lin, H, R)
        S = H @ Pc @ H.T + R
        l1 = multivariate_normal.pdf(z, mean=H @ c1.x, cov=S)
        l2 = multivariate_normal.pdf(z, mean=H @ c2.x, cov=S)
        assert np.isclose(updated[0].w, 0.5 * l1 / (0.5 * l1 + 0.5 * l2))
        assert np.isclose(updated[0].w + updated[1].w, 1.0)

    def test_estimate_mixture_moments(self):
        c1 = GaussianComponent(x=np.array([1.0, 0.0]), P=np.eye(2), w=0.3)
        c2 = GaussianComponent(x=np.array([3.0, 1.0]), P=2 * np.eye(2), w=0.7)
        gsf = GaussianSumFilter()
        gsf.components = [c1, c2]
        x, P = gsf.estimate()
        x_ref = 0.3 * c1.x + 0.7 * c2.x
        P_ref = 0.3 * (c1.P + np.outer(c1.x - x_ref, c1.x - x_ref)) + 0.7 * (
            c2.P + np.outer(c2.x - x_ref, c2.x - x_ref)
        )
        np.testing.assert_allclose(x, x_ref)
        np.testing.assert_allclose(P, P_ref)
        assert gsf.get_num_components() == 2
        assert len(gsf.get_components()) == 2
