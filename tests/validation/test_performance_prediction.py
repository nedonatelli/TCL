"""Performance prediction against MATLAB TCL fixtures.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via
scripts/matlab_capture/capture_performance_prediction.m; inputs
mirrored verbatim. All functions are deterministic, but the
Riccati/FIM functions seed from a DARE solve (MATLAB: qz-based
RiccatiSolveD; port: SciPy's solve_discrete_are) whose unique
stabilizing solution the two solvers reach with different roundoff,
so those fixtures are compared at 1e-8 rather than machine precision.

Two ports deliberately diverge from upstream:

- track_purity_lin_approx: the original's initial update parses as
  (H'*R)\\H, a dimension error whenever z_dim != x_dim; the square
  H=eye fixture (where the two parses coincide) comes from the
  unmodified original, and the rectangular fixture from a
  precedence-fixed MATLAB shim.
- pcrlb_pred_add: the original discards user-supplied cubature points
  (an off-by-one nargin test); the port honors them, and the cubature
  fixtures use the default fifth-order points both sides agree on.
"""

from pathlib import Path

import numpy as np

from pytcl.dynamic_estimation import (
    correct_assoc_prob_approx,
    disc_prior_p_model,
    fim_post_no_clutter,
    fim_pred_no_clutter,
    lin_target_is_untrackable,
    pcrlb_pred_add,
    pcrlb_update_add_no_clutter,
    riccati_post_no_clutter,
    riccati_pred_no_clutter,
    track_purity_lin_approx,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# DARE-seeded results: SciPy and MATLAB reach the same stabilizing
# solution (and the PD<1 fixed point contracts away seed roundoff);
# measured max disagreement 4.3e-14 locally, with headroom left for
# other LAPACK builds.
ATOL_DARE = 1e-8
# Pure transcriptions with no DARE involved: measured 2.0e-13.
ATOL = 1e-12

# The 6-state NCV model from the MATLAB docstring examples,
# position-major ordering as built by FPolyKal/QPolyKal via kron.
F6 = np.kron(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(3))
Q6 = np.kron(np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]), np.eye(3))
R3 = np.diag([10.0, 10.0, 10.0])
H3 = np.hstack([np.eye(3), np.zeros((3, 3))])

F2 = np.array([[1.0, 1.0], [0.0, 1.0]])
H2 = np.array([[1.0, 0.0]])
R2 = np.array([[4.0]])
Q2 = np.array([[0.02, 0.01], [0.01, 0.03]])


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


class TestRiccati:
    def test_pred_pd1_matches_matlab(self):
        res = riccati_pred_no_clutter(H3, F6, R3, Q6, 1.0)
        assert res.converged
        np.testing.assert_allclose(res.P, _load("pp_ricpred_pd1.csv"), atol=ATOL_DARE)

    def test_pred_pd05_matches_matlab(self):
        res = riccati_pred_no_clutter(H3, F6, R3, Q6, 0.5)
        assert res.converged
        np.testing.assert_allclose(res.P, _load("pp_ricpred_pd05.csv"), atol=ATOL_DARE)

    def test_post_pd1_matches_matlab(self):
        res = riccati_post_no_clutter(H3, F6, R3, Q6, 1.0)
        np.testing.assert_allclose(res.P, _load("pp_ricpost_pd1.csv"), atol=ATOL_DARE)

    def test_post_pd05_matches_matlab(self):
        res = riccati_post_no_clutter(H3, F6, R3, Q6, 0.5)
        np.testing.assert_allclose(res.P, _load("pp_ricpost_pd05.csv"), atol=ATOL_DARE)

    def test_pred_solution_satisfies_its_equation(self):
        p = riccati_pred_no_clutter(H3, F6, R3, Q6, 0.5).P
        rhs = (
            F6 @ p @ F6.T
            - 0.5 * F6 @ p @ H3.T @ np.linalg.solve(H3 @ p @ H3.T + R3, H3 @ p @ F6.T)
            + Q6
        )
        np.testing.assert_allclose(p, rhs, atol=1e-8)

    def test_post_is_pred_after_update(self):
        # One Kalman update maps the asymptotic prediction onto the
        # asymptotic posterior (PD=1 fixed-point consistency).
        pred = riccati_pred_no_clutter(H3, F6, R3, Q6, 1.0).P
        gain = pred @ H3.T @ np.linalg.inv(H3 @ pred @ H3.T + R3)
        post = (np.eye(6) - gain @ H3) @ pred
        np.testing.assert_allclose(
            post, riccati_post_no_clutter(H3, F6, R3, Q6, 1.0).P, atol=1e-7
        )


class TestFIM:
    def test_post_pd1_matches_matlab(self):
        j = fim_post_no_clutter(H3, F6, R3, Q6, 1.0)
        np.testing.assert_allclose(j, _load("pp_fimpost_pd1.csv"), atol=ATOL_DARE)

    def test_post_pd05_matches_matlab(self):
        j = fim_post_no_clutter(H3, F6, R3, Q6, 0.5)
        np.testing.assert_allclose(j, _load("pp_fimpost_pd05.csv"), atol=ATOL_DARE)

    def test_pred_pd05_matches_matlab(self):
        j = fim_pred_no_clutter(H3, F6, R3, Q6, 0.5)
        np.testing.assert_allclose(j, _load("pp_fimpred_pd05.csv"), atol=ATOL_DARE)

    def test_singular_q_iterative_branch_matches_matlab(self):
        q_sing = 0.5 * np.array([[1 / 4, 1 / 2], [1 / 2, 1.0]])
        j = fim_post_no_clutter(H2, F2, R2, q_sing, 0.9)
        np.testing.assert_allclose(j, _load("pp_fimpost_singq.csv"), atol=1e-6)

    def test_pd1_inverse_fim_is_riccati_covariance(self):
        j = fim_post_no_clutter(H3, F6, R3, Q6, 1.0)
        p = riccati_post_no_clutter(H3, F6, R3, Q6, 1.0).P
        np.testing.assert_allclose(np.linalg.inv(j), p, rtol=1e-7)


class TestPCRLBRecursion:
    def test_constant_matrices_match_matlab(self):
        j = np.zeros((2, 2))
        for _ in range(3):
            j = pcrlb_pred_add(j, None, None, Q2, F2)
            j = pcrlb_update_add_no_clutter(j, None, None, R2, 0.8, H2)
        np.testing.assert_allclose(j, _load("pp_pcrlb_const.csv"), atol=ATOL)

    X_PRIOR = np.array([1.0, 0.5])
    P_PRIOR = np.array([[0.09, 0.02], [0.02, 0.06]])

    @staticmethod
    def _fj(x):
        return np.array([[1.0, 1.0], [-0.1 * x[0], 1.0]])

    @staticmethod
    def _hj(x):
        return np.array([[2.0 * x[0], 0.2]])

    def test_cubature_averaged_prediction_matches_matlab(self):
        j = pcrlb_pred_add(0.5 * np.eye(2), self.X_PRIOR, self.P_PRIOR, Q2, self._fj)
        np.testing.assert_allclose(j, _load("pp_pcrlb_cubpred.csv"), atol=ATOL)

    def test_cubature_averaged_update_matches_matlab(self):
        j = pcrlb_update_add_no_clutter(
            0.5 * np.eye(2), self.X_PRIOR, self.P_PRIOR, R2, 0.8, self._hj
        )
        np.testing.assert_allclose(j, _load("pp_pcrlb_cubupd.csv"), atol=ATOL)

    def test_jacobian_at_mean_matches_matlab(self):
        j = pcrlb_update_add_no_clutter(
            0.5 * np.eye(2), self.X_PRIOR, np.zeros((2, 2)), R2, 0.8, self._hj
        )
        np.testing.assert_allclose(j, _load("pp_pcrlb_jacmean.csv"), atol=ATOL)

    def test_explicit_points_are_honored(self):
        # The upstream original always discards user-supplied points
        # (nargin<8 in a seven-argument function); the port must use
        # them. Passing the mean as a single unit-weight point must
        # reproduce the Jacobian-at-mean result, not the default
        # cubature average.
        xi = np.zeros((1, 2))
        w = np.array([1.0])
        j_pts = pcrlb_pred_add(
            0.5 * np.eye(2), self.X_PRIOR, self.P_PRIOR, Q2, self._fj, xi, w
        )
        f_mat = self._fj(self.X_PRIOR)
        j_ref = pcrlb_pred_add(0.5 * np.eye(2), self.X_PRIOR, None, Q2, f_mat)
        np.testing.assert_allclose(j_pts, j_ref, atol=1e-12)


class TestAssociationMetrics:
    def test_correct_assoc_prob_matches_matlab(self):
        pc = correct_assoc_prob_approx(3, 1e-4, float(np.linalg.det(R3)))
        np.testing.assert_allclose(pc, _load("pp_assoc_prob.csv")[0, 0], atol=ATOL)

    def test_track_purity_square_matches_unmodified_matlab(self):
        # With H = eye the original's precedence bug is invisible
        # ((H'*R)\H equals H'*(R\H)), so this fixture comes from the
        # unmodified original.
        res = track_purity_lin_approx(
            2e-3, np.eye(2), F2, np.diag([4.0, 9.0]), Q2, 10, np.eye(2)
        )
        np.testing.assert_allclose(
            res.pc, _load("pp_purity_sq_pc.csv")[0, 0], atol=ATOL
        )
        np.testing.assert_allclose(res.p_inv, _load("pp_purity_sq_pinv.csv"), atol=1e-9)

    def test_track_purity_rectangular_matches_fixed_matlab(self):
        # The rectangular case crashes upstream; its fixture comes
        # from a MATLAB shim carrying the one-character precedence fix.
        res = track_purity_lin_approx(2e-3, H2, F2, R2, Q2, 10, np.eye(2))
        np.testing.assert_allclose(
            res.pc, _load("pp_purity_rect_pc.csv")[0, 0], atol=ATOL
        )
        np.testing.assert_allclose(
            res.p_inv, _load("pp_purity_rect_pinv.csv"), atol=1e-9
        )

    def test_untrackability_matches_matlab(self):
        expected = _load("pp_untrackable.csv").ravel().astype(bool)
        s = np.diag([100.0, 100.0])
        assert lin_target_is_untrackable(s, 0.5, 1e-8) == expected[0]
        assert lin_target_is_untrackable(s, 0.5, 10.0) == expected[1]
        assert lin_target_is_untrackable((H2, F2, R2, Q2), 0.5, 1e-6) == expected[2]


class TestDiscPriorPModel:
    def test_generic_form_matches_matlab(self):
        res = disc_prior_p_model(3, [0.0, 1.0], f=F2, q=Q2)
        np.testing.assert_allclose(
            res.x, _load("pp_prior_gen_x.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(res.P, _load("pp_prior_gen_P.csv"), atol=ATOL)

    def test_ncv_form_matches_matlab(self):
        res = disc_prior_p_model(4, np.arange(1.0, 7.0), T=0.5, q0=2.0)
        np.testing.assert_allclose(
            res.x, _load("pp_prior_ncv_x.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(res.P, _load("pp_prior_ncv_P.csv"), atol=ATOL)

    def test_nca_form_matches_matlab(self):
        res = disc_prior_p_model(2, np.arange(1.0, 10.0), T=0.5, q0=2.0)
        np.testing.assert_allclose(
            res.x, _load("pp_prior_nca_x.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(res.P, _load("pp_prior_nca_P.csv"), atol=ATOL)

    def test_position_only_form_matches_matlab(self):
        res = disc_prior_p_model(5, [1.0, 2.0, 3.0], T=0.5, q0=2.0)
        np.testing.assert_allclose(
            res.x, _load("pp_prior_pos_x.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(res.P, _load("pp_prior_pos_P.csv"), atol=ATOL)


class TestEdgeAndGuardPaths:
    """Non-convergence returns, the singular-Q FIM prediction branch,
    Jacobian-at-mean and explicit-point paths, and the input guards."""

    def test_riccati_iteration_caps_report_nonconvergence(self):
        pred = riccati_pred_no_clutter(H3, F6, R3, Q6, 0.5, max_iter=1)
        assert not pred.converged
        post = riccati_post_no_clutter(H3, F6, R3, Q6, 0.5, max_iter=1)
        assert not post.converged

    def test_fim_iterative_branch_warns_when_divergent(self):
        import pytest

        # No velocity process noise: velocity information grows without
        # bound, so the singular-Q recursion cannot converge.
        q_div = np.array([[0.5, 0.0], [0.0, 0.0]])
        with pytest.warns(UserWarning, match="without convergence"):
            fim_post_no_clutter(H2, F2, R2, q_div, 0.9)

    def test_fim_pred_singular_q_branch(self):
        q_sing = 0.5 * np.array([[1 / 4, 1 / 2], [1 / 2, 1.0]])
        j = fim_pred_no_clutter(H2, F2, R2, q_sing, 1.0)
        p = riccati_pred_no_clutter(H2, F2, R2, q_sing, 1.0).P
        np.testing.assert_allclose(np.linalg.inv(j), p, rtol=1e-6)

    def test_pcrlb_pred_jacobian_at_mean(self):
        x_prior = np.array([1.0, 0.5])
        fj = lambda x: np.array([[1.0, 1.0], [-0.1 * x[0], 1.0]])  # noqa: E731
        via_callable = pcrlb_pred_add(0.5 * np.eye(2), x_prior, None, Q2, fj)
        via_matrix = pcrlb_pred_add(0.5 * np.eye(2), None, None, Q2, fj(x_prior))
        np.testing.assert_allclose(via_callable, via_matrix, atol=1e-13)

    def test_pcrlb_update_explicit_points_are_honored(self):
        x_cur = np.array([1.0, 0.5])
        p_cur = np.array([[0.09, 0.02], [0.02, 0.06]])
        hj = lambda x: np.array([[2.0 * x[0], 0.2]])  # noqa: E731
        xi = np.zeros((1, 2))
        w = np.array([1.0])
        with_points = pcrlb_update_add_no_clutter(
            0.5 * np.eye(2), x_cur, p_cur, R2, 0.8, hj, xi, w
        )
        at_mean = pcrlb_update_add_no_clutter(0.5 * np.eye(2), x_cur, None, R2, 0.8, hj)
        np.testing.assert_allclose(with_points, at_mean, atol=1e-13)

    def test_disc_prior_input_guards(self):
        import pytest

        with pytest.raises(ValueError):
            disc_prior_p_model(2, [0.0, 1.0], f=F2)  # q missing
        with pytest.raises(ValueError):
            disc_prior_p_model(2, [0.0, 1.0])  # neither form
        with pytest.raises(ValueError):
            disc_prior_p_model(2, np.zeros(4), T=0.5, q0=1.0)  # bad length

    def test_rcond_zero_matrix_guard(self):
        from pytcl.dynamic_estimation.performance_prediction import _rcond

        assert _rcond(np.zeros((2, 2))) == 0.0
