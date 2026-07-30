"""
Correctness audit tests for dynamic models, performance metrics, and trackers.

Reference strategy:

- F matrices are checked against scipy.linalg.expm(A*T), which is exact for
  these linear time-invariant models.
- Q matrices are checked against the Van Loan method (matrix exponential of
  the augmented matrix), the canonical discretization of continuous-time
  process noise.
- OSPA and MOT metrics are checked against hand-computed values (the hand
  calculation is documented in comments) and against the metric axioms.
- NEES/NIS chi-squared consistency is checked on a correctly-tuned Kalman
  filter with a seeded Monte Carlo simulation.
- Trackers are checked on synthetic scenarios with well-separated targets,
  and hypothesis enumeration counts against combinatorial hand counts.
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from pytcl.assignment_algorithms import gnn_association
from pytcl.dynamic_models import (
    continuous_to_discrete,
    diffusion_constant_acceleration,
    diffusion_constant_velocity,
    diffusion_singer,
    discretize_lti,
    drift_constant_acceleration,
    drift_constant_velocity,
    drift_coordinated_turn_2d,
    drift_singer,
    f_constant_acceleration,
    f_constant_velocity,
    f_coord_turn_2d,
    f_coord_turn_3d,
    f_coord_turn_polar,
    f_discrete_white_noise_accel,
    f_piecewise_white_noise_jerk,
    f_poly_kal,
    f_singer,
    f_singer_2d,
    f_singer_3d,
    q_constant_acceleration,
    q_constant_velocity,
    q_continuous_white_noise,
    q_coord_turn_2d,
    q_coord_turn_3d,
    q_coord_turn_polar,
    q_discrete_white_noise,
    q_poly_kal,
    q_singer,
    q_singer_2d,
    q_singer_3d,
    state_jacobian_ca,
    state_jacobian_cv,
    state_jacobian_singer,
)
from pytcl.performance_evaluation import (
    average_nees,
    consistency_test,
    credibility_interval,
    estimation_error_bounds,
    identity_switches,
    monte_carlo_rmse,
    mot_metrics,
    nees,
    nees_sequence,
    nis,
    nis_sequence,
    ospa,
    ospa_over_time,
    position_rmse,
    rmse,
    track_fragmentation,
    track_purity,
    velocity_rmse,
)
from pytcl.trackers import (
    Hypothesis,
    HypothesisTree,
    MHTConfig,
    MHTTrack,
    MHTTracker,
    MHTTrackStatus,
    MultiTargetTracker,
    SingleTargetTracker,
    TrackStatus,
    compute_association_likelihood,
    generate_joint_associations,
    n_scan_prune,
    prune_hypotheses_by_probability,
)


def van_loan_reference(A, G, Qc, T):
    """Canonical Van Loan discretization: reference for F and Q_d."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = G @ Qc @ G.T
    M[n:, n:] = A.T
    E = expm(M * T)
    F = E[n:, n:].T
    Qd = F @ E[:n, n:]
    return F, (Qd + Qd.T) / 2


def a_cv():
    return np.array([[0.0, 1.0], [0.0, 0.0]])


def a_ca():
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])


def a_singer(tau):
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]])


def a_ct2d(omega):
    # State [x, vx, y, vy]: dvx/dt = -omega*vy, dvy/dt = omega*vx
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -omega],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, omega, 0.0, 0.0],
        ]
    )


class TestStateTransitionVsExpm:
    """F matrices must equal expm(A*T) exactly for these LTI models."""

    @pytest.mark.parametrize("T", [0.05, 0.5, 1.0, 3.0])
    def test_f_poly_kal_cv_ca(self, T):
        np.testing.assert_allclose(f_poly_kal(1, T), expm(a_cv() * T), atol=1e-12)
        np.testing.assert_allclose(f_poly_kal(2, T), expm(a_ca() * T), atol=1e-12)

    def test_f_poly_kal_order3(self):
        T = 0.7
        A = np.diag(np.ones(3), k=1)
        np.testing.assert_allclose(f_poly_kal(3, T), expm(A * T), atol=1e-12)

    def test_cv_ca_wrappers_and_aliases(self):
        T = 0.3
        F1 = f_constant_velocity(T, num_dims=2)
        assert F1.shape == (4, 4)
        np.testing.assert_allclose(F1[:2, :2], expm(a_cv() * T))
        np.testing.assert_allclose(F1[2:, 2:], expm(a_cv() * T))
        np.testing.assert_allclose(F1[:2, 2:], 0.0, atol=0)

        F2 = f_constant_acceleration(T, num_dims=1)
        np.testing.assert_allclose(F2, expm(a_ca() * T), atol=1e-12)

        np.testing.assert_array_equal(
            f_discrete_white_noise_accel(T, 2), f_constant_velocity(T, 2)
        )
        np.testing.assert_array_equal(
            f_piecewise_white_noise_jerk(T, 2), f_constant_acceleration(T, 2)
        )

    @pytest.mark.parametrize("T", [0.1, 1.0, 2.0])
    @pytest.mark.parametrize("tau", [0.5, 5.0, 20.0])
    def test_f_singer_vs_expm(self, T, tau):
        np.testing.assert_allclose(
            f_singer(T, tau), expm(a_singer(tau) * T), atol=1e-12
        )

    def test_f_singer_multidim_block_structure(self):
        T, tau = 0.5, 8.0
        F1 = f_singer(T, tau)
        F2 = f_singer_2d(T, tau)
        F3 = f_singer_3d(T, tau)
        assert F2.shape == (6, 6) and F3.shape == (9, 9)
        for Fk, nd in ((F2, 2), (F3, 3)):
            for d in range(nd):
                s = 3 * d
                np.testing.assert_allclose(Fk[s : s + 3, s : s + 3], F1)

    def test_f_singer_limits(self):
        # tau -> inf: Singer reduces to constant acceleration.
        T = 1.0
        np.testing.assert_allclose(
            f_singer(T, tau=1e4), f_constant_acceleration(T, num_dims=1), rtol=1e-3
        )
        # tau -> 0: acceleration decorrelates (white); coupling terms vanish.
        F = f_singer(T, tau=1e-4)
        assert abs(F[2, 2]) < 1e-12
        assert abs(F[1, 2] - 1e-4) < 1e-8
        np.testing.assert_allclose(F[:2, :2], [[1.0, T], [0.0, 1.0]])

    @pytest.mark.parametrize("T", [0.1, 1.0, 2.0])
    @pytest.mark.parametrize("omega", [0.05, 0.5, -1.2])
    def test_f_coord_turn_2d_vs_expm(self, T, omega):
        np.testing.assert_allclose(
            f_coord_turn_2d(T, omega), expm(a_ct2d(omega) * T), atol=1e-12
        )

    def test_f_coord_turn_2d_zero_omega_is_cv(self):
        T = 1.5
        np.testing.assert_allclose(
            f_coord_turn_2d(T, 0.0), f_constant_velocity(T, num_dims=2)
        )

    def test_f_coord_turn_2d_analytic_circle(self):
        # Exact circular-arc solution: starting at origin with speed v along
        # +x and turn rate omega, after time T the target is at
        # x = (v/omega)*sin(omega*T), y = (v/omega)*(1 - cos(omega*T)).
        v, omega, T = 100.0, 0.2, 2.0
        state = np.array([0.0, v, 0.0, 0.0])
        out = f_coord_turn_2d(T, omega) @ state
        np.testing.assert_allclose(out[0], v / omega * np.sin(omega * T))
        np.testing.assert_allclose(out[2], v / omega * (1 - np.cos(omega * T)))
        np.testing.assert_allclose(out[1], v * np.cos(omega * T))
        np.testing.assert_allclose(out[3], v * np.sin(omega * T))

    def test_f_coord_turn_2d_matches_drift_integration(self):
        # Integrating the continuous-time drift must reproduce F @ x0.
        omega, T = 0.3, 1.7
        x0 = np.array([1.0, 40.0, -2.0, 25.0])
        sol = solve_ivp(
            lambda t, s: drift_coordinated_turn_2d(np.append(s, omega))[:4],
            (0, T),
            x0,
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            f_coord_turn_2d(T, omega) @ x0, sol.y[:, -1], rtol=1e-8
        )

    def test_f_coord_turn_2d_omega_state_variant(self):
        T, omega = 0.5, 0.4
        F = f_coord_turn_2d(T, omega, state_type="position_velocity_omega")
        assert F.shape == (5, 5)
        np.testing.assert_allclose(F[:4, :4], f_coord_turn_2d(T, omega))
        assert F[4, 4] == 1.0

    @pytest.mark.parametrize("omega", [0.0, 0.25])
    def test_f_coord_turn_3d_vs_expm(self, omega):
        T = 0.8
        A = np.zeros((6, 6))
        A[:4, :4] = a_ct2d(omega)
        A[4, 5] = 1.0  # z constant velocity
        np.testing.assert_allclose(f_coord_turn_3d(T, omega), expm(A * T), atol=1e-12)

    def test_f_coord_turn_polar_zero_omega(self):
        # Zero-turn-rate branch: straight-line motion at heading psi=0.
        # NOTE: the nonzero-omega branch is NOT the Jacobian of the nonlinear
        # polar propagation (see audit report); only the omega=0 branch is
        # verified here.
        T = 1.0
        F = f_coord_turn_polar(T, omega=0.0, speed=100.0)
        assert F.shape == (5, 5)
        assert F[0, 3] == T  # dx/dspeed
        assert F[2, 4] == T  # dpsi/domega


class TestProcessNoiseVsVanLoan:
    """Q matrices vs the Van Loan reference (canonical for every model)."""

    @pytest.mark.parametrize("T", [0.1, 1.0, 2.5])
    @pytest.mark.parametrize("q", [0.5, 3.0])
    def test_q_poly_kal_cv_ca(self, T, q):
        G2 = np.array([[0.0], [1.0]])
        _, Qref = van_loan_reference(a_cv(), G2, np.array([[q]]), T)
        np.testing.assert_allclose(q_poly_kal(1, T, q), Qref, rtol=1e-10, atol=1e-14)

        G3 = np.array([[0.0], [0.0], [1.0]])
        _, Qref3 = van_loan_reference(a_ca(), G3, np.array([[q]]), T)
        np.testing.assert_allclose(q_poly_kal(2, T, q), Qref3, rtol=1e-10, atol=1e-14)

    def test_q_poly_kal_closed_form(self):
        # Hand formula for continuous white-noise acceleration:
        # Q = q*[[T^3/3, T^2/2], [T^2/2, T]]
        T, q = 2.0, 1.5
        expected = q * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
        np.testing.assert_allclose(q_poly_kal(1, T, q), expected)

    def test_q_continuous_white_noise_is_continuous_model(self):
        # Regression test for the fixed bug: this function previously
        # returned the discrete white-noise matrix q*[[T^4/4, ...]] instead
        # of the continuous-model integral q*[[T^3/3, ...]].
        T, q = 1.0, 1.0
        G2 = np.array([[0.0], [1.0]])
        _, Qref = van_loan_reference(a_cv(), G2, np.array([[q]]), T)
        np.testing.assert_allclose(q_continuous_white_noise(2, T, q), Qref, rtol=1e-10)
        G3 = np.array([[0.0], [0.0], [1.0]])
        _, Qref3 = van_loan_reference(a_ca(), G3, np.array([[q]]), 0.5)
        np.testing.assert_allclose(
            q_continuous_white_noise(3, 0.5, q), Qref3, rtol=1e-10
        )

    @pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
    def test_q_discrete_white_noise_gain_model(self, T):
        # Discrete model: Q = var * G G' with G the noise gain vector.
        var = 1.3
        G2 = np.array([T**2 / 2, T])
        np.testing.assert_allclose(
            q_discrete_white_noise(2, T, var), var * np.outer(G2, G2)
        )
        G3 = np.array([T**3 / 6, T**2 / 2, T])
        np.testing.assert_allclose(
            q_discrete_white_noise(3, T, var), var * np.outer(G3, G3)
        )
        G4 = np.array([T**4 / 24, T**3 / 6, T**2 / 2, T])
        np.testing.assert_allclose(
            q_discrete_white_noise(4, T, var), var * np.outer(G4, G4)
        )

    def test_q_cv_ca_wrappers(self):
        T, sigma = 0.5, 2.0
        Q = q_constant_velocity(T, sigma, num_dims=2)
        assert Q.shape == (4, 4)
        np.testing.assert_allclose(Q[:2, :2], q_discrete_white_noise(2, T, sigma**2))
        np.testing.assert_allclose(Q[:2, 2:], 0.0, atol=0)

        Qa = q_constant_acceleration(T, sigma, num_dims=1)
        np.testing.assert_allclose(Qa, q_discrete_white_noise(3, T, sigma**2))

    @pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("tau", [1.0, 5.0, 20.0])
    def test_q_singer_vs_van_loan(self, T, tau):
        # Van Loan on the Singer model da/dt = -a/tau + w,
        # spectral density q_c = 2*sigma_m^2/tau: the canonical reference.
        sigma_m = 2.0
        q_c = 2 * sigma_m**2 / tau
        G = np.array([[0.0], [0.0], [1.0]])
        _, Qref = van_loan_reference(a_singer(tau), G, np.array([[q_c]]), T)
        np.testing.assert_allclose(
            q_singer(T, tau, sigma_m), Qref, rtol=1e-7, atol=1e-12
        )

    def test_q_singer_psd_and_symmetric(self):
        Q = q_singer(1.0, 5.0, 3.0)
        np.testing.assert_allclose(Q, Q.T)
        assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)

    def test_q_singer_large_tau_limit(self):
        # tau -> inf: Singer Q tends to the continuous white-jerk CA model
        # with the same spectral density q_c.
        T, tau, sigma_m = 0.5, 1e3, 2.0
        q_c = 2 * sigma_m**2 / tau
        np.testing.assert_allclose(
            q_singer(T, tau, sigma_m), q_poly_kal(2, T, q_c), rtol=1e-2
        )

    def test_q_singer_multidim_blocks(self):
        T, tau, sm = 0.5, 8.0, 1.5
        Q1 = q_singer(T, tau, sm)
        Q2 = q_singer_2d(T, tau, sm)
        Q3 = q_singer_3d(T, tau, sm)
        assert Q2.shape == (6, 6) and Q3.shape == (9, 9)
        for Qk, nd in ((Q2, 2), (Q3, 3)):
            for d in range(nd):
                s = 3 * d
                np.testing.assert_allclose(Qk[s : s + 3, s : s + 3], Q1)
            np.testing.assert_allclose(Qk[:3, 3:6], 0.0, atol=0)

    def test_q_coord_turn_2d_3d_blocks(self):
        # Per-axis DWNA blocks; omega variance sigma_omega^2 * T^2.
        T, sa, so = 0.5, 2.0, 0.1
        block = sa**2 * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
        Q = q_coord_turn_2d(T, sa)
        np.testing.assert_allclose(Q[:2, :2], block)
        np.testing.assert_allclose(Q[2:, 2:], block)
        np.testing.assert_allclose(Q[:2, 2:], 0.0, atol=0)

        Q5 = q_coord_turn_2d(T, sa, so, state_type="position_velocity_omega")
        assert Q5.shape == (5, 5)
        np.testing.assert_allclose(Q5[4, 4], so**2 * T**2)

        Q6 = q_coord_turn_3d(T, sa)
        assert Q6.shape == (6, 6)
        np.testing.assert_allclose(Q6[4:6, 4:6], block)
        Q7 = q_coord_turn_3d(T, sa, so, state_type="position_velocity_omega")
        np.testing.assert_allclose(Q7[6, 6], so**2 * T**2)

    def test_q_coord_turn_polar_hand_entries(self):
        T, sa, sod = 0.5, 2.0, 0.1
        Q = q_coord_turn_polar(T, sa, sod)
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T)
        np.testing.assert_allclose(Q[3, 3], sa**2 * T**2)
        np.testing.assert_allclose(Q[4, 4], sod**2 * T**2)
        np.testing.assert_allclose(Q[2, 2], sod**2 * T**4 / 4)
        assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)


class TestContinuousTimeDynamics:
    """Drift/diffusion functions and discretization utilities."""

    def test_drift_cv_ca_singer_linear_forms(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            drift_constant_velocity(x, num_dims=2), state_jacobian_cv(x, 2) @ x
        )
        x3 = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            drift_constant_acceleration(x3, num_dims=1), state_jacobian_ca(x3, 1) @ x3
        )
        np.testing.assert_allclose(
            drift_singer(x3, tau=5.0, num_dims=1),
            state_jacobian_singer(x3, tau=5.0, num_dims=1) @ x3,
        )

    def test_drift_coordinated_turn_2d_hand(self):
        # dvx/dt = -omega*vy, dvy/dt = omega*vx
        x = np.array([0.0, 100.0, 0.0, 50.0, 0.1])
        np.testing.assert_allclose(
            drift_coordinated_turn_2d(x), [100.0, -5.0, 50.0, 10.0, 0.0]
        )

    @pytest.mark.parametrize(
        "jac_fn,drift_fn,kwargs,dim",
        [
            (state_jacobian_cv, drift_constant_velocity, {"num_dims": 2}, 4),
            (state_jacobian_ca, drift_constant_acceleration, {"num_dims": 1}, 3),
            (state_jacobian_singer, drift_singer, {"tau": 4.0, "num_dims": 1}, 3),
        ],
    )
    def test_jacobians_vs_numerical_differentiation(
        self, jac_fn, drift_fn, kwargs, dim
    ):
        rng = np.random.default_rng(7)
        x0 = rng.normal(size=dim)
        eps = 1e-6
        J_num = np.zeros((dim, dim))
        for k in range(dim):
            dp = x0.copy()
            dm = x0.copy()
            dp[k] += eps
            dm[k] -= eps
            J_num[:, k] = (drift_fn(dp, **kwargs) - drift_fn(dm, **kwargs)) / (2 * eps)
        np.testing.assert_allclose(jac_fn(x0, **kwargs), J_num, atol=1e-8)

    def test_diffusion_matrices_structure(self):
        D = diffusion_constant_velocity(np.zeros(4), sigma_a=0.5, num_dims=2)
        expected = np.zeros((4, 2))
        expected[1, 0] = 0.5
        expected[3, 1] = 0.5
        np.testing.assert_allclose(D, expected)

        D3 = diffusion_constant_acceleration(np.zeros(3), sigma_j=0.1, num_dims=1)
        np.testing.assert_allclose(D3, [[0.0], [0.0], [0.1]])

        Ds = diffusion_singer(np.zeros(3), sigma_m=1.0, tau=10.0, num_dims=1)
        np.testing.assert_allclose(Ds[2, 0], math.sqrt(2 * 1.0 / 10.0))

    def test_diffusion_consistent_with_q_matrices(self):
        # Van Loan with (A, D from diffusion, Qc=I) must equal the analytic
        # Q of the corresponding model.
        T = 0.7
        D = diffusion_singer(np.zeros(3), sigma_m=2.0, tau=5.0, num_dims=1)
        _, Qref = van_loan_reference(a_singer(5.0), D, np.eye(1), T)
        np.testing.assert_allclose(q_singer(T, 5.0, 2.0), Qref, rtol=1e-8)

        Dc = diffusion_constant_velocity(np.zeros(2), sigma_a=1.5, num_dims=1)
        _, Qcv = van_loan_reference(a_cv(), Dc, np.eye(1), T)
        np.testing.assert_allclose(q_poly_kal(1, T, 1.5**2), Qcv, rtol=1e-10)

    def test_continuous_to_discrete_cv_closed_form(self):
        T = 0.5
        F, Qd = continuous_to_discrete(
            a_cv(), np.array([[0.0], [1.0]]), np.array([[1.0]]), T
        )
        np.testing.assert_allclose(F, [[1.0, T], [0.0, 1.0]])
        np.testing.assert_allclose(
            Qd, [[T**3 / 3, T**2 / 2], [T**2 / 2, T]], rtol=1e-10
        )

    def test_continuous_to_discrete_matches_singer(self):
        T, tau, sm = 1.0, 5.0, 2.0
        q_c = 2 * sm**2 / tau
        F, Qd = continuous_to_discrete(
            a_singer(tau), np.array([[0.0], [0.0], [1.0]]), np.array([[q_c]]), T
        )
        np.testing.assert_allclose(F, f_singer(T, tau), rtol=1e-10)
        np.testing.assert_allclose(Qd, q_singer(T, tau, sm), rtol=1e-7)

    def test_discretize_lti_cv_input(self):
        T = 0.1
        F, G = discretize_lti(a_cv(), np.array([[0.0], [1.0]]), T)
        np.testing.assert_allclose(F, [[1.0, T], [0.0, 1.0]])
        # G = int_0^T expm(A s) B ds = [T^2/2, T]'
        np.testing.assert_allclose(G, [[T**2 / 2], [T]], rtol=1e-10)

    def test_discretize_lti_no_input(self):
        F, G = discretize_lti(a_ca(), None, 0.4)
        assert G is None
        np.testing.assert_allclose(F, expm(a_ca() * 0.4))


class TestOSPA:
    """OSPA vs hand-computed values and metric axioms."""

    def test_both_empty(self):
        r = ospa([], [])
        assert r.ospa == 0.0 and r.localization == 0.0 and r.cardinality == 0.0

    def test_one_empty_is_cutoff(self):
        X = [np.array([0.0, 0.0])]
        assert ospa(X, [], c=100.0).ospa == 100.0
        assert ospa([], X, c=7.5).ospa == 7.5

    def test_localization_only_hand(self):
        # Distances after optimal assignment: 1 and 1.
        # OSPA (p=2, equal cardinality) = ((1^2 + 1^2)/2)^(1/2) = 1.
        X = [np.array([0.0, 0.0]), np.array([10.0, 10.0])]
        Y = [np.array([1.0, 0.0]), np.array([10.0, 11.0])]
        r = ospa(X, Y, c=100.0, p=2.0)
        np.testing.assert_allclose(r.ospa, 1.0)
        np.testing.assert_allclose(r.localization, 1.0)
        assert r.cardinality == 0.0

    def test_cardinality_only_hand(self):
        # X = {(0,0)}, Y = {(0,0), (5,0)}, c=10, p=2.
        # Optimal assignment cost 0; penalty (n-m)*c^p = 100.
        # OSPA = (100/2)^(1/2) = sqrt(50).
        X = [np.array([0.0, 0.0])]
        Y = [np.array([0.0, 0.0]), np.array([5.0, 0.0])]
        r = ospa(X, Y, c=10.0, p=2.0)
        np.testing.assert_allclose(r.ospa, math.sqrt(50.0))
        np.testing.assert_allclose(r.cardinality, math.sqrt(50.0))
        assert r.localization == 0.0

    def test_mixed_hand(self):
        # X = {(0,0)}, Y = {(3,4), (100,100)}, c=10, p=2.
        # Best assignment: (0,0)->(3,4), d=5 -> 25. Unassigned meas: c^2=100.
        # OSPA = ((25 + 100)/2)^(1/2) = sqrt(62.5).
        X = [np.array([0.0, 0.0])]
        Y = [np.array([3.0, 4.0]), np.array([100.0, 100.0])]
        r = ospa(X, Y, c=10.0, p=2.0)
        np.testing.assert_allclose(r.ospa, math.sqrt(62.5))
        np.testing.assert_allclose(r.localization, math.sqrt(12.5))
        np.testing.assert_allclose(r.cardinality, math.sqrt(50.0))

    def test_cutoff_saturation(self):
        # Distance 50 > c=10 saturates to c: OSPA = c.
        X = [np.array([0.0, 0.0])]
        Y = [np.array([50.0, 0.0])]
        np.testing.assert_allclose(ospa(X, Y, c=10.0).ospa, 10.0)

    def test_identity_axiom(self):
        rng = np.random.default_rng(0)
        X = [rng.normal(size=2) for _ in range(4)]
        assert ospa(X, [x.copy() for x in X]).ospa == 0.0

    def test_symmetry_axiom(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            X = [rng.uniform(0, 10, 2) for _ in range(rng.integers(0, 5))]
            Y = [rng.uniform(0, 10, 2) for _ in range(rng.integers(0, 5))]
            np.testing.assert_allclose(ospa(X, Y, c=5.0).ospa, ospa(Y, X, c=5.0).ospa)

    def test_triangle_inequality(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            X = [rng.uniform(0, 10, 2) for _ in range(rng.integers(1, 5))]
            Y = [rng.uniform(0, 10, 2) for _ in range(rng.integers(1, 5))]
            Z = [rng.uniform(0, 10, 2) for _ in range(rng.integers(1, 5))]
            dxy = ospa(X, Y, c=5.0).ospa
            dyz = ospa(Y, Z, c=5.0).ospa
            dxz = ospa(X, Z, c=5.0).ospa
            assert dxz <= dxy + dyz + 1e-10

    def test_ospa_over_time_matches_per_step(self):
        X_seq = [
            [np.array([0.0, 0.0]), np.array([10.0, 10.0])],
            [np.array([1.0, 0.0])],
        ]
        Y_seq = [
            [np.array([1.0, 0.0]), np.array([10.0, 11.0])],
            [np.array([1.0, 0.0]), np.array([5.0, 5.0])],
        ]
        vals = ospa_over_time(X_seq, Y_seq, c=10.0, p=2.0)
        assert len(vals) == 2
        for k in range(2):
            np.testing.assert_allclose(
                vals[k], ospa(X_seq[k], Y_seq[k], c=10.0, p=2.0).ospa
            )
        with pytest.raises(ValueError):
            ospa_over_time(X_seq, Y_seq[:1])


class TestLabelMetrics:
    def test_track_purity_hand(self):
        # est track 0 covers true labels {0,0}: 2 pure.
        # est track 1 covers {0,1,1,1}: majority 3.
        # purity = (2+3)/6.
        t = np.array([0, 0, 0, 1, 1, 1])
        e = np.array([0, 0, 1, 1, 1, 1])
        np.testing.assert_allclose(track_purity(t, e), 5.0 / 6.0)
        assert track_purity(t, t) == 1.0
        assert track_purity(np.array([]), np.array([])) == 1.0
        with pytest.raises(ValueError):
            track_purity(np.array([0]), np.array([0, 1]))

    def test_track_fragmentation_hand(self):
        assert track_fragmentation(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1])) == 1
        # Alternating tracks: 4 transitions.
        assert track_fragmentation(np.array([0] * 5), np.array([0, 1, 0, 1, 0])) == 4
        assert track_fragmentation(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 0

    def test_track_fragmentation_time_ordering(self):
        # Out-of-order input: sorted by time it is [0, 0, 1, 1] -> 1 frag.
        t = np.array([0, 0, 0, 0])
        e = np.array([1, 1, 0, 0])
        times = np.array([3, 2, 0, 1])
        assert track_fragmentation(t, e, time_indices=times) == 1

    def test_identity_switches_hand(self):
        # Estimated track 0 follows true target 0 then true target 1: 1 switch.
        assert identity_switches(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])) == 1
        # No switches when each est track sticks to one target.
        assert identity_switches(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 0
        # Two est tracks each switching once: 2 total.
        t = np.array([0, 1, 0, 1])
        e = np.array([0, 1, 1, 0])
        # est 0 sees true [0, 1] -> 1 switch; est 1 sees true [1, 0] -> 1.
        assert identity_switches(t, e) == 2

    def test_identity_switches_time_ordering(self):
        t = np.array([1, 1, 0, 0])
        e = np.array([0, 0, 0, 0])
        times = np.array([2, 3, 0, 1])
        # Sorted: true [0, 0, 1, 1] -> single switch.
        assert identity_switches(t, e, time_indices=times) == 1


class TestMOTMetrics:
    def test_perfect_tracking(self):
        gt = [
            [np.array([0.0, 0.0]), np.array([10.0, 10.0])],
            [np.array([1.0, 0.0]), np.array([11.0, 10.0])],
        ]
        r = mot_metrics(gt, [list(f) for f in gt], threshold=5.0)
        assert r.mota == 1.0
        assert r.motp == 0.0
        assert r.num_switches == 0
        assert r.num_false_positives == 0
        assert r.num_misses == 0

    def test_miss_and_false_positive(self):
        # Frame 1: 1 GT, no est -> 1 miss. Frame 2: no GT, 1 est -> 1 FP.
        # MOTA = 1 - (1 + 1 + 0)/1 = -1.
        gt = [[np.array([0.0, 0.0])], []]
        est: list = [[], [np.array([5.0, 5.0])]]
        r = mot_metrics(gt, est, threshold=5.0)
        assert r.num_misses == 1
        assert r.num_false_positives == 1
        assert r.mota == -1.0

    def test_identity_switch_counted(self):
        # Two stationary well-separated targets; estimates swap between
        # frames so both GT indices change their matched estimate: 2 switches.
        # MOTA = 1 - 2/4 = 0.5. MOTP = 0 (exact matches).
        a, b = np.array([0.0, 0.0]), np.array([100.0, 100.0])
        gt = [[a, b], [a, b]]
        est = [[a.copy(), b.copy()], [b.copy(), a.copy()]]
        r = mot_metrics(gt, est, threshold=5.0)
        assert r.num_switches == 2
        assert r.mota == 0.5
        assert r.motp == 0.0

    def test_motp_is_mean_matched_distance(self):
        # Single target, offsets 1.0 and 3.0 -> MOTP = 2.0.
        gt = [[np.array([0.0, 0.0])], [np.array([0.0, 0.0])]]
        est = [[np.array([1.0, 0.0])], [np.array([3.0, 0.0])]]
        r = mot_metrics(gt, est, threshold=5.0)
        np.testing.assert_allclose(r.motp, 2.0)
        assert r.mota == 1.0


class TestEstimationMetrics:
    def test_rmse_vs_numpy(self):
        rng = np.random.default_rng(3)
        t = rng.normal(size=(20, 4))
        e = rng.normal(size=(20, 4))
        np.testing.assert_allclose(rmse(t, e), np.sqrt(np.mean((t - e) ** 2)))
        np.testing.assert_allclose(
            rmse(t, e, axis=0), np.sqrt(np.mean((t - e) ** 2, axis=0))
        )
        np.testing.assert_allclose(
            rmse(t, e, axis=1), np.sqrt(np.mean((t - e) ** 2, axis=1))
        )

    def test_position_velocity_rmse(self):
        rng = np.random.default_rng(4)
        t = rng.normal(size=(10, 4))
        e = rng.normal(size=(10, 4))
        np.testing.assert_allclose(
            position_rmse(t, e, [0, 2]),
            np.sqrt(np.mean((t[:, [0, 2]] - e[:, [0, 2]]) ** 2)),
        )
        np.testing.assert_allclose(
            velocity_rmse(t, e, [1, 3]),
            np.sqrt(np.mean((t[:, [1, 3]] - e[:, [1, 3]]) ** 2)),
        )

    def test_monte_carlo_rmse(self):
        rng = np.random.default_rng(5)
        errs = rng.normal(size=(30, 5, 2))
        np.testing.assert_allclose(
            monte_carlo_rmse(errs, axis=0), np.sqrt(np.mean(errs**2, axis=0))
        )

    def test_nees_hand(self):
        # e = [1, 1], P = [[2, 1], [1, 2]], P^-1 = (1/3)[[2, -1], [-1, 2]].
        # NEES = (1/3)(2 - 1 - 1 + 2) = 2/3.
        P = np.array([[2.0, 1.0], [1.0, 2.0]])
        val = nees(np.array([1.0, 1.0]), np.array([0.0, 0.0]), P)
        np.testing.assert_allclose(val, 2.0 / 3.0)

    def test_nis_hand(self):
        # nu = [0.5, -0.3], S = 0.25*I -> NIS = (0.25 + 0.09)/0.25 = 1.36.
        np.testing.assert_allclose(nis(np.array([0.5, -0.3]), np.eye(2) * 0.25), 1.36)

    def test_sequences_match_scalar_calls(self):
        rng = np.random.default_rng(6)
        t = rng.normal(size=(5, 2))
        e = rng.normal(size=(5, 2))
        P = np.array([np.eye(2) * (0.5 + i) for i in range(5)])
        seq = nees_sequence(t, e, P)
        for k in range(5):
            np.testing.assert_allclose(seq[k], nees(t[k], e[k], P[k]))
        np.testing.assert_allclose(average_nees(t, e, P), seq.mean())

        innovs = rng.normal(size=(5, 2))
        nis_seq = nis_sequence(innovs, P)
        for k in range(5):
            np.testing.assert_allclose(nis_seq[k], nis(innovs[k], P[k]))

    def test_consistency_test_bounds_formula(self):
        vals = np.full(100, 2.0)
        r = consistency_test(vals, df=2, confidence=0.95)
        np.testing.assert_allclose(r.lower_bound, chi2.ppf(0.025, 200) / 100)
        np.testing.assert_allclose(r.upper_bound, chi2.ppf(0.975, 200) / 100)
        assert r.is_consistent  # mean exactly at df

    def test_consistency_test_detects_inconsistency(self):
        rng = np.random.default_rng(8)
        good = rng.chisquare(df=4, size=400)
        assert consistency_test(good, df=4).is_consistent
        assert not consistency_test(3.0 * good, df=4).is_consistent
        assert not consistency_test(good / 3.0, df=4).is_consistent

    def test_credibility_interval_matched_covariance(self):
        rng = np.random.default_rng(9)
        n, d = 2000, 2
        L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.5]]))
        errors = rng.standard_normal((n, d)) @ L.T
        P = np.array([L @ L.T] * n)
        frac = credibility_interval(errors, P, interval=0.95)
        assert 0.93 < frac < 0.97
        # Overconfident covariance: far fewer errors inside.
        frac_bad = credibility_interval(errors, P / 10.0, interval=0.95)
        assert frac_bad < 0.7

    def test_estimation_error_bounds_hand(self):
        P = np.array([[[1.0, 0.0], [0.0, 4.0]], [[0.25, 0.0], [0.0, 1.0]]])
        b = estimation_error_bounds(P, sigma=2.0)
        np.testing.assert_allclose(b, [[2.0, 4.0], [1.0, 2.0]])


class TestKalmanFilterConsistency:
    """NEES/NIS chi-squared consistency on a correctly-tuned filter."""

    def test_single_target_tracker_nees_nis_consistent(self):
        rng = np.random.default_rng(42)
        T = 1.0
        F = np.array([[1.0, T], [0.0, 1.0]])
        Q = q_poly_kal(1, T, 0.1)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.5]])

        tracker = SingleTargetTracker(2, 1, F, H, Q, R)
        x_true = np.array([0.0, 1.0])
        P0 = np.eye(2)
        x0 = x_true + rng.multivariate_normal(np.zeros(2), P0)
        tracker.initialize(x0, P0)

        Lq = np.linalg.cholesky(Q)
        nees_vals = []
        nis_vals = []
        n_steps = 400
        for _ in range(n_steps):
            x_true = F @ x_true + Lq @ rng.standard_normal(2)
            z = H @ x_true + math.sqrt(R[0, 0]) * rng.standard_normal(1)
            tracker.predict(T)
            z_pred, S = tracker.predict_measurement()
            nis_vals.append(nis(z - z_pred, S))
            state, _ = tracker.update(z)
            nees_vals.append(nees(x_true, state.state, state.covariance))

        # For a consistent filter, mean NEES ~ state_dim and mean NIS ~ meas_dim.
        mean_nees = np.mean(nees_vals)
        mean_nis = np.mean(nis_vals)
        assert 1.6 < mean_nees < 2.5, mean_nees
        assert 0.8 < mean_nis < 1.25, mean_nis

    def test_single_target_tracker_matches_hand_kf(self):
        # One predict/update cycle vs hand-computed Kalman equations.
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = 0.1 * np.eye(2)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.5]])
        tracker = SingleTargetTracker(2, 1, F, H, Q, R)
        x0 = np.array([1.0, 2.0])
        P0 = np.array([[1.0, 0.2], [0.2, 2.0]])
        tracker.initialize(x0, P0)
        tracker.predict(1.0)

        x_pred = F @ x0
        P_pred = F @ P0 @ F.T + Q
        z = np.array([3.4])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ (z - H @ x_pred)
        P_upd = (np.eye(2) - K @ H) @ P_pred

        state, d2 = tracker.update(z)
        np.testing.assert_allclose(state.state, x_upd)
        np.testing.assert_allclose(state.covariance, P_upd)
        nu = z - H @ x_pred
        np.testing.assert_allclose(d2, float(nu @ np.linalg.inv(S) @ nu))

    def test_single_target_tracker_gating(self):
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        tracker = SingleTargetTracker(
            2, 1, F, H, 0.01 * np.eye(2), np.array([[0.1]]), gate_threshold=9.0
        )
        tracker.initialize(np.zeros(2), np.eye(2) * 0.1)
        tracker.predict(1.0)
        before = tracker.state.state.copy()
        state, d2 = tracker.update(np.array([100.0]))  # far outside gate
        assert d2 > 9.0
        np.testing.assert_array_equal(state.state, before)


class TestMultiTargetTracker:
    def _make_tracker(self, **kwargs):
        T = 1.0
        F = np.array(
            [[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]], dtype=float
        )
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
        Q = 0.01 * np.eye(4)
        R = 0.1 * np.eye(2)
        return MultiTargetTracker(4, 2, F, H, Q, R, **kwargs), F, H

    def test_recovers_well_separated_targets(self):
        rng = np.random.default_rng(11)
        tracker, _, _ = self._make_tracker()
        truths = [
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([100.0, -1.0, 100.0, 0.0]),
        ]
        T = 1.0
        F = np.array(
            [[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]], dtype=float
        )
        for _ in range(8):
            truths = [F @ x for x in truths]
            meas = [x[[0, 2]] + 0.05 * rng.standard_normal(2) for x in truths]
            tracker.process(meas, dt=T)

        confirmed = tracker.confirmed_tracks
        assert len(confirmed) == 2
        est_pos = sorted([tuple(t.state[[0, 2]]) for t in confirmed])
        true_pos = sorted([tuple(x[[0, 2]]) for x in truths])
        for ep, tp in zip(est_pos, true_pos):
            np.testing.assert_allclose(ep, tp, atol=1.0)

    def test_association_matches_linear_sum_assignment(self):
        # GNN on an ungated square problem must match scipy's optimum.
        rng = np.random.default_rng(12)
        for _ in range(20):
            C = rng.uniform(0, 10, size=(4, 4))
            result = gnn_association(C)
            rows, cols = linear_sum_assignment(C)
            np.testing.assert_allclose(result.total_cost, C[rows, cols].sum())

    def test_track_lifecycle(self):
        tracker, _, _ = self._make_tracker(confirm_hits=3, max_misses=2)
        z = np.array([5.0, 5.0])
        tracker.process([z], dt=1.0)
        assert tracker.tracks[0].status == TrackStatus.TENTATIVE
        tracker.process([z], dt=1.0)
        tracker.process([z], dt=1.0)
        assert tracker.tracks[0].status == TrackStatus.CONFIRMED
        # Miss enough scans -> deleted.
        tracker.process([], dt=1.0)
        tracker.process([], dt=1.0)
        assert len(tracker.tracks) == 0


class TestHypothesisManagement:
    def test_joint_association_count_2x2(self):
        # 2 tracks, 2 measurements, all gated. Count by hand:
        # k assignments: k=0 -> 1; k=1 -> C(2,1)*2 = 4; k=2 -> 2! = 2. Total 7.
        gated = np.ones((2, 2), dtype=bool)
        assocs = generate_joint_associations(gated, 2, 2)
        assert len(assocs) == 7
        # All must be valid: measurement used at most once.
        for a in assocs:
            used = [m for m in a.values() if m >= 0]
            assert len(used) == len(set(used))

    def test_joint_association_count_2x3(self):
        # k=0: 1; k=1: 2 tracks * 3 meas = 6; k=2: C(2,2)*P(3,2) = 6. Total 13.
        gated = np.ones((2, 3), dtype=bool)
        assert len(generate_joint_associations(gated, 2, 3)) == 13

    def test_joint_association_partial_gating(self):
        # gated = [[T, F], [T, T]]: track0 in {-1, 0}, track1 in {-1, 0, 1},
        # measurement 0 shared. Valid: (-1,-1), (-1,0), (-1,1), (0,-1), (0,1).
        gated = np.array([[True, False], [True, True]])
        assocs = generate_joint_associations(gated, 2, 2)
        expected = [
            {0: -1, 1: -1},
            {0: -1, 1: 0},
            {0: -1, 1: 1},
            {0: 0, 1: -1},
            {0: 0, 1: 1},
        ]
        assert len(assocs) == 5
        for e in expected:
            assert e in assocs

    def test_no_tracks(self):
        assert generate_joint_associations(np.zeros((0, 2), dtype=bool), 0, 2) == [{}]

    def test_association_likelihood_hand(self):
        L = np.array([[0.9, 0.1], [0.1, 0.8]])
        pd, lam = 0.9, 1e-3
        # Full assignment: pd*L00 * pd*L11, no clutter.
        lik = compute_association_likelihood({0: 0, 1: 1}, L, pd, lam, n_meas=2)
        np.testing.assert_allclose(lik, pd * 0.9 * pd * 0.8)
        # Track 1 missed: pd*L00 * (1-pd) * lam^1 (meas 1 unexplained).
        lik_miss = compute_association_likelihood({0: 0, 1: -1}, L, pd, lam, n_meas=2)
        np.testing.assert_allclose(lik_miss, pd * 0.9 * (1 - pd) * lam)
        # All missed: (1-pd)^2 * lam^2.
        lik_none = compute_association_likelihood({0: -1, 1: -1}, L, pd, lam, n_meas=2)
        np.testing.assert_allclose(lik_none, (1 - pd) ** 2 * lam**2)

    def _mk_track(self, tid, scan_created, score=1.0):
        return MHTTrack(
            id=tid,
            state=np.zeros(2),
            covariance=np.eye(2),
            score=score,
            status=MHTTrackStatus.CONFIRMED,
            history=[0],
            parent_id=-1,
            scan_created=scan_created,
            n_hits=3,
            n_misses=0,
        )

    def test_n_scan_prune_survivors(self):
        # Tracks A, B created at scan 0; C at scan 2. Cutoff = 3 - 2 = 1.
        # Best hyp h1 contains {A} at cutoff. h2 {B} disagrees -> pruned.
        # h3 {A, C}: at cutoff only {A} -> agrees -> kept.
        tracks = {
            0: self._mk_track(0, 0),
            1: self._mk_track(1, 0),
            2: self._mk_track(2, 2),
        }
        h1 = Hypothesis(
            id=0, probability=0.6, track_ids=[0], scan_created=0, parent_id=-1
        )
        h2 = Hypothesis(
            id=1, probability=0.3, track_ids=[1], scan_created=0, parent_id=-1
        )
        h3 = Hypothesis(
            id=2, probability=0.1, track_ids=[0, 2], scan_created=2, parent_id=0
        )
        pruned, committed = n_scan_prune([h1, h2, h3], tracks, n_scan=2, current_scan=3)
        assert committed == {0}
        ids = {h.id for h in pruned}
        assert ids == {0, 2}
        # Renormalized: 0.6/0.7 and 0.1/0.7.
        probs = {h.id: h.probability for h in pruned}
        np.testing.assert_allclose(probs[0], 0.6 / 0.7)
        np.testing.assert_allclose(probs[2], 0.1 / 0.7)

    def test_n_scan_prune_disabled(self):
        h = Hypothesis(
            id=0, probability=1.0, track_ids=[], scan_created=0, parent_id=-1
        )
        pruned, committed = n_scan_prune([h], {}, n_scan=0, current_scan=5)
        assert pruned == [h] and committed == set()

    def test_prune_by_probability(self):
        hyps = [
            Hypothesis(id=i, probability=p, track_ids=[i], scan_created=0, parent_id=-1)
            for i, p in enumerate([0.5, 0.3, 0.1, 0.05, 1e-9])
        ]
        pruned = prune_hypotheses_by_probability(hyps, max_hypotheses=3)
        assert [h.id for h in pruned] == [0, 1, 2]
        np.testing.assert_allclose(sum(h.probability for h in pruned), 1.0)
        np.testing.assert_allclose(pruned[0].probability, 0.5 / 0.9)
        # All below threshold: keeps single best.
        tiny = [
            Hypothesis(id=i, probability=p, track_ids=[], scan_created=0, parent_id=-1)
            for i, p in enumerate([1e-9, 5e-9])
        ]
        kept = prune_hypotheses_by_probability(tiny, max_hypotheses=5)
        assert len(kept) == 1 and kept[0].id == 1

    def test_hypothesis_tree_initialize_and_add(self):
        tree = HypothesisTree(max_hypotheses=10, n_scan=2)
        tree.initialize()
        assert len(tree.hypotheses) == 1
        assert tree.hypotheses[0].probability == 1.0
        tid = tree.add_track(self._mk_track(-1, 0))
        assert tid in tree.tracks
        best = tree.get_best_hypothesis()
        assert best is not None and best.probability == 1.0


class TestMHTTracker:
    def _make_tracker(self, **cfg_kwargs):
        T = 1.0
        F = np.array(
            [[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]], dtype=float
        )
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
        Q = 0.01 * np.eye(4)
        R = 0.1 * np.eye(2)
        config = MHTConfig(**cfg_kwargs) if cfg_kwargs else MHTConfig()
        return MHTTracker(4, 2, F, H, Q, R, config=config)

    def test_measurement_likelihood_is_gaussian(self):
        # The per-pair likelihood must be Pd * N(innovation; 0, S).
        tracker = self._make_tracker()
        track = MHTTrack(
            id=0,
            state=np.array([0.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(4),
            score=0.0,
            status=MHTTrackStatus.TENTATIVE,
            history=[],
            parent_id=-1,
            scan_created=0,
            n_hits=1,
            n_misses=0,
        )
        Z = np.array([[0.5, -0.5]])
        gated, lik = tracker._compute_gating_and_likelihoods({0: track}, Z)
        assert (0, 0) in gated
        H = tracker.H
        S = H @ np.eye(4) @ H.T + tracker.R
        nu = Z[0] - H @ track.state
        expected = (
            tracker.config.detection_prob
            * np.exp(-0.5 * nu @ np.linalg.inv(S) @ nu)
            / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(S))
        )
        np.testing.assert_allclose(lik[(0, 0)], expected, rtol=1e-12)

    def test_single_target_confirmation_and_state(self):
        tracker = self._make_tracker()
        rng = np.random.default_rng(21)
        x = np.array([0.0, 1.0, 0.0, 0.5])
        T = 1.0
        F = np.array(
            [[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]], dtype=float
        )
        result = None
        for _ in range(6):
            x = F @ x
            z = x[[0, 2]] + 0.05 * rng.standard_normal(2)
            result = tracker.process([z], dt=T)
        assert len(result.confirmed_tracks) == 1
        est = result.confirmed_tracks[0].state
        np.testing.assert_allclose(est[[0, 2]], x[[0, 2]], atol=1.0)
        assert 1 <= result.n_hypotheses <= tracker.config.max_hypotheses

    def test_two_separated_targets(self):
        tracker = self._make_tracker(n_scan=2, max_hypotheses=50)
        T = 1.0
        F = np.array(
            [[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]], dtype=float
        )
        truths = [
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([200.0, -1.0, 200.0, 0.0]),
        ]
        result = None
        for _ in range(6):
            truths = [F @ x for x in truths]
            result = tracker.process([x[[0, 2]] for x in truths], dt=T)
        assert len(result.confirmed_tracks) == 2
        est_pos = sorted(tuple(t.state[[0, 2]]) for t in result.confirmed_tracks)
        true_pos = sorted(tuple(x[[0, 2]]) for x in truths)
        for ep, tp in zip(est_pos, true_pos):
            np.testing.assert_allclose(ep, tp, atol=1.0)

    def test_hypothesis_count_bounded(self):
        tracker = self._make_tracker(max_hypotheses=20)
        rng = np.random.default_rng(22)
        for _ in range(5):
            meas = [rng.uniform(0, 50, 2) for _ in range(3)]
            tracker.process(meas, dt=1.0)
        assert tracker.n_hypotheses <= 20

    def test_miss_score_is_log_one_minus_pd(self):
        tracker = self._make_tracker()
        track = MHTTrack(
            id=0,
            state=np.zeros(4),
            covariance=np.eye(4),
            score=1.5,
            status=MHTTrackStatus.CONFIRMED,
            history=[0],
            parent_id=-1,
            scan_created=0,
            n_hits=3,
            n_misses=0,
        )
        missed = tracker._miss_track(track)
        np.testing.assert_allclose(
            missed.score, 1.5 + np.log(1 - tracker.config.detection_prob)
        )
        assert missed.n_misses == 1
        assert missed.history[-1] == -1
