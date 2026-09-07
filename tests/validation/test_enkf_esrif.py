"""EnKF and ESRIF against MATLAB TCL fixtures and KF ground truth.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_enkf_esrif.m. The
EnKF fixtures pass explicit noise-sample arrays, which makes both steps
fully deterministic despite the stochastic algorithm; the
internally-drawn path is validated statistically against the exact
Kalman posterior instead (MATLAB's RNG stream cannot be reproduced
from numpy). ESRIF fixtures store QR-sign-invariant quantities: the
recovered state and the information matrix R.T @ R.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.dynamic_estimation import esrif_predict, esrif_update
from pytcl.dynamic_estimation.kalman import enkf_predict, enkf_update
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Literal transcriptions: measured max disagreement 4.3e-14.
ATOL = 1e-11


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


X_ENS = np.array(
    [
        [1.0, 1.2, 0.8, 1.1, 0.9, 1.05],
        [-0.5, -0.4, -0.6, -0.45, -0.55, -0.5],
        [2.0, 2.1, 1.9, 2.05, 1.95, 2.0],
    ]
)
V_SAMP = np.array(
    [
        [0.05, -0.03, 0.02, -0.04, 0.01, -0.01],
        [-0.02, 0.04, -0.01, 0.03, -0.03, -0.01],
        [0.01, 0.02, -0.02, -0.01, 0.03, -0.03],
    ]
)
W_SAMP = np.array(
    [
        [0.06, -0.02, 0.03, -0.05, 0.02, -0.04],
        [-0.03, 0.05, -0.02, 0.01, -0.04, 0.03],
    ]
)
F = np.array([[1.0, 0.5, 0.125], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
SQ = 0.1 * np.eye(3)
SR = 0.2 * np.eye(2)
Z = np.array([1.6, -0.7])


class TestEnKFAgainstMatlab:
    def _predict(self):
        return enkf_predict(X_ENS, lambda x: F @ x, SQ, 0, None, V_SAMP)

    def test_predict_matches_matlab(self):
        pred = self._predict()
        np.testing.assert_allclose(
            pred.x_ensemble, _load("enkf_pred_ensemble.csv"), atol=ATOL
        )
        # The multi-output outputs are derived here because the MATLAB
        # original's multi-output form calls an undefined function (the
        # stateAvgFun bug the port fixes).
        np.testing.assert_allclose(
            pred.x_pred, np.mean(_load("enkf_pred_ensemble.csv"), axis=1), atol=ATOL
        )

    @pytest.mark.parametrize("filter_type", [0, 2])
    def test_update_matches_matlab(self, filter_type):
        pred = self._predict()
        up = enkf_update(
            pred.x_ensemble, Z, SR, lambda x: H @ x, filter_type, w_samp=W_SAMP
        )
        np.testing.assert_allclose(
            up.x_ensemble, _load(f"enkf_up_ensemble_ft{filter_type}.csv"), atol=ATOL
        )
        np.testing.assert_allclose(
            up.x_update, _load(f"enkf_up_x_ft{filter_type}.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(
            up.p_update, _load(f"enkf_up_P_ft{filter_type}.csv"), atol=ATOL
        )
        np.testing.assert_allclose(
            up.pzz, _load(f"enkf_up_Pzz_ft{filter_type}.csv"), atol=ATOL
        )
        np.testing.assert_allclose(
            up.gain, _load(f"enkf_up_W_ft{filter_type}.csv"), atol=ATOL
        )

    def test_invalid_filter_type_raises(self):
        with pytest.raises(ValueError, match="filter type"):
            enkf_predict(X_ENS, lambda x: x, SQ, 5, None, V_SAMP)
        with pytest.raises(ValueError, match="filter type"):
            enkf_update(X_ENS, Z, SR, lambda x: H @ x, 5, w_samp=W_SAMP)

    def test_nonadditive_type_receives_noise_argument(self):
        pred = enkf_predict(X_ENS, lambda x, v: F @ x + 2.0 * v, SQ, 1, None, V_SAMP)
        expected = F @ X_ENS + 2.0 * V_SAMP
        np.testing.assert_allclose(pred.x_ensemble, expected, atol=1e-12)


class TestEnKFConvergesToKalman:
    """The drawn-noise path cannot match MATLAB's RNG; instead the
    ensemble statistics must converge to the exact Kalman posterior on
    a linear-Gaussian problem."""

    def test_large_ensemble_tracks_kf_posterior(self):
        rng = np.random.default_rng(20260904)
        n = 20000
        x0 = np.array([1.0, -0.5, 0.25])
        p0 = np.diag([0.4, 0.9, 0.2])
        q = SQ @ SQ.T
        r = SR @ SR.T

        kf_pred = kf_predict(x0, p0, F, q)
        kf_up = kf_update(kf_pred.x, kf_pred.P, Z, H, r)

        ens = x0[:, None] + np.linalg.cholesky(p0) @ rng.standard_normal((3, n))
        pred = enkf_predict(ens, lambda x: F @ x, SQ, rng=rng)
        up = enkf_update(pred.x_ensemble, Z, SR, lambda x: H @ x, rng=rng)

        # Monte-Carlo error scales as 1/sqrt(n); these bounds sit ~5x
        # above the observed error at this seed and ensemble size.
        np.testing.assert_allclose(pred.x_pred, kf_pred.x, atol=0.05)
        np.testing.assert_allclose(pred.p_pred, kf_pred.P, atol=0.05)
        np.testing.assert_allclose(up.x_update, kf_up.x, atol=0.05)
        np.testing.assert_allclose(up.p_update, kf_up.P, atol=0.05)


# The ESRIF scene from the capture script.
X0 = np.array([1.0, -0.4, 0.3])
P0 = np.diag([0.5, 0.8, 0.3])
U = np.array([0.1, -0.2, 0.05])
GAMMA = np.diag([1.0, 1.0, 0.5])


class TestESRIFAgainstMatlab:
    def _r_info(self):
        return np.linalg.cholesky(np.linalg.inv(P0)).T

    def test_predict_matches_matlab_invariants(self):
        r_info = self._r_info()
        pred = esrif_predict(
            r_info @ X0, r_info, lambda x: F @ x, lambda x: F, SQ, U, GAMMA
        )
        ref = _load("esrif_pred.csv")
        np.testing.assert_allclose(
            np.linalg.solve(pred.R, pred.y_sqrt), ref[0], atol=ATOL
        )
        np.testing.assert_allclose(pred.R.T @ pred.R, ref[1:], atol=ATOL)

    def test_update_matches_matlab_invariants(self):
        r_info = self._r_info()
        pred = esrif_predict(
            r_info @ X0, r_info, lambda x: F @ x, lambda x: F, SQ, U, GAMMA
        )
        up = esrif_update(pred.y_sqrt, pred.R, Z, SR, lambda x: H @ x, lambda x: H)
        ref = _load("esrif_update.csv")
        np.testing.assert_allclose(np.linalg.solve(up.R, up.y_sqrt), ref[0], atol=ATOL)
        np.testing.assert_allclose(up.R.T @ up.R, ref[1:], atol=ATOL)

    def test_linear_case_agrees_with_kalman_filter(self):
        # On a linear problem the ESRIF is algebraically the KF in
        # square-root information form.
        r_info = self._r_info()
        q = SQ @ SQ.T
        r = SR @ SR.T
        kf_pred = kf_predict(X0, P0, F, q, B=GAMMA @ np.eye(3), u=np.zeros(3))
        # ESRIF's u enters through SQ\\u in the stacked system; compare
        # against the KF with the same effective control input.
        pred = esrif_predict(
            r_info @ X0, r_info, lambda x: F @ x, lambda x: F, SQ, np.zeros(3), GAMMA
        )
        x_pred = np.linalg.solve(pred.R, pred.y_sqrt)
        np.testing.assert_allclose(x_pred, kf_pred.x, atol=1e-10)
        p_pred = np.linalg.inv(pred.R.T @ pred.R)
        np.testing.assert_allclose(
            p_pred, F @ P0 @ F.T + GAMMA @ q @ GAMMA.T, atol=1e-10
        )

        up = esrif_update(pred.y_sqrt, pred.R, Z, SR, lambda x: H @ x, lambda x: H)
        kf_up = kf_update(x_pred, p_pred, Z, H, r)
        np.testing.assert_allclose(
            np.linalg.solve(up.R, up.y_sqrt), kf_up.x, atol=1e-10
        )
        np.testing.assert_allclose(np.linalg.inv(up.R.T @ up.R), kf_up.P, atol=1e-10)

    def test_numerical_jacobian_default(self):
        r_info = self._r_info()
        pred_exact = esrif_predict(
            r_info @ X0, r_info, lambda x: F @ x, lambda x: F, SQ, U, GAMMA
        )
        pred_num = esrif_predict(
            r_info @ X0, r_info, lambda x: F @ x, None, SQ, U, GAMMA
        )
        np.testing.assert_allclose(
            np.linalg.solve(pred_num.R, pred_num.y_sqrt),
            np.linalg.solve(pred_exact.R, pred_exact.y_sqrt),
            rtol=1e-6,
        )


class TestDefaultArgumentPaths:
    """The optional-argument branches: default u/gamma/rng and the
    noise-through-the-measurement-function filter type."""

    def _r_info(self):
        p0 = np.diag([0.5, 0.4, 0.3])
        return np.linalg.inv(np.linalg.cholesky(p0)).T

    def test_esrif_predict_defaults_equal_explicit_zeros_and_identity(self):
        r_info = self._r_info()
        explicit = esrif_predict(
            r_info @ X0,
            r_info,
            lambda x: F @ x,
            lambda x: F,
            SQ,
            np.zeros(3),
            np.eye(3),
        )
        defaulted = esrif_predict(r_info @ X0, r_info, lambda x: F @ x, lambda x: F, SQ)
        np.testing.assert_allclose(defaulted.y_sqrt, explicit.y_sqrt, atol=1e-13)
        np.testing.assert_allclose(defaulted.R, explicit.R, atol=1e-13)

    def test_esrif_update_numerical_jacobian_default(self):
        r_info = self._r_info()
        pred = esrif_predict(r_info @ X0, r_info, lambda x: F @ x, lambda x: F, SQ)
        exact = esrif_update(pred.y_sqrt, pred.R, Z, SR, lambda x: H @ x, lambda x: H)
        numeric = esrif_update(pred.y_sqrt, pred.R, Z, SR, lambda x: H @ x, None)
        np.testing.assert_allclose(
            np.linalg.solve(numeric.R, numeric.y_sqrt),
            np.linalg.solve(exact.R, exact.y_sqrt),
            rtol=1e-6,
        )

    def test_enkf_update_noise_through_h_equals_additive(self):
        rng = np.random.default_rng(7)
        x_ens = X0[:, np.newaxis] + 0.1 * rng.standard_normal((3, 40))
        w_samp = SR @ rng.standard_normal((2, 40))
        additive = enkf_update(x_ens.copy(), Z, SR, lambda x: H @ x, 0, w_samp=w_samp)
        through_h = enkf_update(
            x_ens.copy(), Z, SR, lambda x, w: H @ x + w, 1, w_samp=w_samp
        )
        np.testing.assert_allclose(through_h.x_update, additive.x_update, atol=1e-12)
        np.testing.assert_allclose(through_h.p_update, additive.p_update, atol=1e-12)

    def test_default_rng_draws_are_centered(self):
        # Without explicit noise samples or an rng, the internal draws
        # are still recentered to zero mean, so an identity dynamic
        # preserves the ensemble mean exactly.
        rng = np.random.default_rng(11)
        x_ens = X0[:, np.newaxis] + 0.1 * rng.standard_normal((3, 30))
        pred = enkf_predict(x_ens.copy(), lambda x: x, SQ)
        np.testing.assert_allclose(
            np.mean(pred.x_ensemble, axis=1), np.mean(x_ens, axis=1), atol=1e-12
        )
