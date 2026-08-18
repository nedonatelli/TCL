"""Check-0 validation for the Gaussian LCD objective and analytic gradients.

This is the spec's check 0 (docs/superpowers/specs/2026-08-16-lcd-samples-design.md,
Section 5): finite-difference verification of every transcribed gradient routine
against its objective piece, with no optimizer in the loop. A subtly wrong
gradient can still descend into a plausible basin and pass optimizer-level
checks, so these tests gate everything downstream.

No MATLAB fixtures are used here by design: raw-coordinate comparison is
provably invalid for this problem (flat optimum manifold under O(n); spec
Section 3). Only gradient check-0 and structural invariants are tested.

Measured baselines (this grid, 10 seeded points per (n, num_points) case,
Apple Silicon macOS, 2026-08-18): worst central-difference relative
discrepancy was 1.9e-7 for the four gradient pieces at h=1e-5, and 9.3e-7
for the assembled objective at h=1e-4. The objective needs the larger step
because its value is O(2500) while -2*D2' + D3' can nearly cancel (gradient
norms down to ~0.04 on this grid), so the eps*|f|/h subtraction-roundoff
floor dominates at h=1e-5 (measured 8.3e-6 there, shrinking 10x per 10x
step increase -- pure FD noise, verified not a transcription error).
Assertions allow 1e-5 (margin >= 10x); anything worse than 1e-4 would
indicate a transcription error per the campaign brief.
"""

import numpy as np
import pytest
from scipy.special import exp1

from pytcl.mathematical_functions.numerical_integration.lcd_samples import (
    _compute_d1,
    _compute_de2,
    _compute_de2_grad,
    _compute_de3,
    _compute_de3_grad,
    _compute_do2_cont_term,
    _compute_do2_grad,
    _compute_do2_simp,
    _compute_do3,
    _compute_do3_grad,
    _ei,
    _lcd_objective,
)

# The campaign grid: both parities of n (1, 3 odd; 2, 4 even) and of
# num_points (10, 20 even; 5, 15 odd).
GRID = [(1, 5), (2, 10), (2, 20), (3, 15), (4, 20)]

B_MAX = 70.0
ABS_TOL = 1e-14
REL_TOL = 1e-14

# Central-difference steps, calibrated per target (see module docstring):
# pieces have O(1) gradient norms so h=1e-5 suffices; the assembled
# objective's near-cancelling gradient needs h=1e-4 to stay above the
# eps*|f|/h roundoff floor.
FD_STEP = 1e-5
FD_STEP_OBJECTIVE = 1e-4

# Measured worst was 1.9e-7 (pieces) / 9.3e-7 (objective) on this grid;
# 1e-5 leaves >=10x margin, far below the 1e-4 investigate-threshold.
CHECK_GRAD_TOL = 1e-5

NUM_RANDOM_POINTS = 10

# The four analytic gradient routines, each paired with the objective piece
# it differentiates. Names match the MATLAB subfunctions they transcribe.
PIECES = {
    "De2": (
        lambda s, b: _compute_de2(s, b, ABS_TOL, REL_TOL),
        lambda s, b: _compute_de2_grad(s, b, ABS_TOL, REL_TOL),
    ),
    "De3": (
        lambda s, b: _compute_de3(s, b),
        lambda s, b: _compute_de3_grad(s, b),
    ),
    "Do2": (
        lambda s, b: _compute_do2_simp(s, b, ABS_TOL, REL_TOL),
        lambda s, b: _compute_do2_grad(s, b, ABS_TOL, REL_TOL),
    ),
    "Do3": (
        lambda s, b: _compute_do3(s, b),
        lambda s, b: _compute_do3_grad(s, b),
    ),
}


def _half_shape(n, num_points):
    return (n, num_points // 2)


def _random_s(n, num_points, k):
    rng = np.random.default_rng((n, num_points, k))
    return rng.standard_normal(_half_shape(n, num_points))


def _central_diff(f, x0, h=FD_STEP):
    """Central-difference gradient of scalar f at flat vector x0."""
    g = np.empty_like(x0)
    for k in range(x0.size):
        xp = x0.copy()
        xp[k] += h
        xm = x0.copy()
        xm[k] -= h
        g[k] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def _rel_discrepancy(g_analytic, g_numeric):
    return np.linalg.norm(g_analytic - g_numeric) / np.linalg.norm(g_analytic)


class TestCheckGradPieces:
    """Check-0 for each of the four gradient routines individually."""

    @pytest.mark.parametrize("piece", sorted(PIECES))
    @pytest.mark.parametrize("n,num_points", GRID)
    def test_piece_gradient_matches_central_differences(self, piece, n, num_points):
        value_fn, grad_fn = PIECES[piece]
        shape = _half_shape(n, num_points)
        worst = 0.0
        for k in range(NUM_RANDOM_POINTS):
            s = _random_s(n, num_points, k)

            def f(x):
                return value_fn(x.reshape(shape, order="F"), B_MAX)

            g_ana = grad_fn(s, B_MAX).flatten(order="F")
            g_num = _central_diff(f, s.flatten(order="F"))
            worst = max(worst, _rel_discrepancy(g_ana, g_num))
        assert worst < CHECK_GRAD_TOL, (
            f"{piece} gradient check-0 failed at ({n},{num_points}): "
            f"worst relative discrepancy {worst:.3e}"
        )

    @pytest.mark.parametrize("piece", ["De3", "Do3"])
    @pytest.mark.parametrize("n,num_points", [(2, 10), (3, 15)])
    def test_piece_gradient_small_b_max(self, piece, n, num_points):
        """Repeat check-0 at b_max=5 so a mis-scaled b_max term cannot hide
        behind b_max=70's near-flat exp/Ei regime."""
        value_fn, grad_fn = PIECES[piece]
        shape = _half_shape(n, num_points)
        for k in range(NUM_RANDOM_POINTS):
            s = _random_s(n, num_points, k)

            def f(x):
                return value_fn(x.reshape(shape, order="F"), 5.0)

            g_ana = grad_fn(s, 5.0).flatten(order="F")
            g_num = _central_diff(f, s.flatten(order="F"))
            assert _rel_discrepancy(g_ana, g_num) < CHECK_GRAD_TOL


class TestCheckGradObjective:
    """Check-0 for the assembled objective/gradient pair (both parity
    branches of modCvMDist/modCvMDistGrad are exercised by the grid)."""

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_objective_gradient_matches_central_differences(self, n, num_points):
        worst = 0.0
        for k in range(NUM_RANDOM_POINTS):
            s_flat = _random_s(n, num_points, k).flatten(order="F")

            def f(x):
                return _lcd_objective(x, n, num_points)[0]

            g_ana = _lcd_objective(s_flat, n, num_points)[1]
            g_num = _central_diff(f, s_flat, h=FD_STEP_OBJECTIVE)
            worst = max(worst, _rel_discrepancy(g_ana, g_num))
        assert worst < CHECK_GRAD_TOL, (
            f"objective check-0 failed at ({n},{num_points}): "
            f"worst relative discrepancy {worst:.3e}"
        )


class TestOrthogonalInvariance:
    """cost(R @ s) == cost(s) for orthogonal R (spec Section 3 proof)."""

    @pytest.mark.parametrize("n,num_points", [(2, 10), (2, 20), (3, 15), (4, 20)])
    def test_cost_invariant_under_orthogonal_transform(self, n, num_points):
        for k in range(5):
            s = _random_s(n, num_points, k)
            rng = np.random.default_rng((7, n, num_points, k))
            q, r = np.linalg.qr(rng.standard_normal((n, n)))
            q = q * np.sign(np.diag(r))  # Haar-ish; det sign irrelevant

            cost = _lcd_objective(s.flatten(order="F"), n, num_points)[0]
            cost_rot = _lcd_objective((q @ s).flatten(order="F"), n, num_points)[0]
            assert cost_rot == pytest.approx(cost, rel=1e-12, abs=1e-12)


class TestObjectiveSanity:
    @pytest.mark.parametrize("n,num_points", [(2, 10), (3, 15)])
    def test_cost_invariant_under_column_permutation(self, n, num_points):
        s = _random_s(n, num_points, 0)
        perm = np.random.default_rng(3).permutation(s.shape[1])
        cost = _lcd_objective(s.flatten(order="F"), n, num_points)[0]
        cost_perm = _lcd_objective(s[:, perm].flatten(order="F"), n, num_points)[0]
        assert cost_perm == pytest.approx(cost, rel=1e-12)

    @pytest.mark.parametrize("n,num_points", [(2, 10), (3, 15)])
    def test_cost_invariant_under_single_column_negation(self, n, num_points):
        """Negating one column swaps the diff/sum roles in D3 and leaves
        column norms (D2) unchanged, so the cost is invariant."""
        s = _random_s(n, num_points, 1)
        s_neg = s.copy()
        s_neg[:, 0] *= -1.0
        cost = _lcd_objective(s.flatten(order="F"), n, num_points)[0]
        cost_neg = _lcd_objective(s_neg.flatten(order="F"), n, num_points)[0]
        assert cost_neg == pytest.approx(cost, rel=1e-12)

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_objective_returns_float_and_matching_gradient_shape(self, n, num_points):
        s_flat = _random_s(n, num_points, 2).flatten(order="F")
        value, grad = _lcd_objective(s_flat, n, num_points)
        assert isinstance(value, float)
        assert np.isfinite(value)
        assert grad.shape == s_flat.shape
        assert np.all(np.isfinite(grad))

    def test_odd_d2_is_rescaled_even_d2(self):
        """computeDo2Simp is (2L/(2L+1)) * computeDe2 (MATLAB lines 360-368)."""
        s = _random_s(3, 15, 3)
        length = s.shape[1]
        de2 = _compute_de2(s, B_MAX, ABS_TOL, REL_TOL)
        do2 = _compute_do2_simp(s, B_MAX, ABS_TOL, REL_TOL)
        assert do2 == pytest.approx(2 * length / (2 * length + 1) * de2, rel=1e-13)

    def test_constant_terms_positive_and_finite(self):
        """D1 and the odd-count D2 constant are positive integrals of
        positive integrands; they never touch the gradient."""
        for n in (1, 2, 3, 4):
            assert _compute_d1(B_MAX, n, ABS_TOL, REL_TOL) > 0.0
        assert _compute_do2_cont_term(3, 7, B_MAX, ABS_TOL, REL_TOL) > 0.0


class TestEi:
    def test_zero_maps_to_zero(self):
        """MATLAB redefines Ei(0)=0 to kill the 0*Inf diagonal terms."""
        assert _ei(np.array(0.0)) == 0.0
        assert np.all(_ei(np.array([0.0, 0.0])) == 0.0)

    def test_negative_arguments_match_exp1(self):
        """For x<0, Ei(x) = -real(expint(-x)) = -E1(-x)."""
        x = np.array([-2.0, -0.5, -1e-8])
        np.testing.assert_allclose(_ei(x), -exp1(-x), rtol=1e-14)

    def test_preserves_input_shape(self):
        x = -np.ones((3, 4))
        assert _ei(x).shape == (3, 4)
