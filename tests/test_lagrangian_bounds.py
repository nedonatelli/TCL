"""
Validity of the Lagrangian bounds in the N-D and 3-D assignment solvers.

Regression tests for gh-14. The relaxation solvers previously computed their
"lower bound" by solving the relaxed inner problem *greedily*. Greedy is not
the relaxed minimiser, so the quantity was not a bound at all: it could exceed
the true optimum, making ``gap`` negative and certifying suboptimal answers as
optimal. Measured before the fix: 22 of 30 random 3x3x3 instances reported
``converged=True`` with ``gap=0.0`` while being up to 0.30 suboptimal.

The inner problem is now solved exactly (it reduces to a 2-D assignment), which
is what makes L(lambda) a genuine lower bound.

Two invariants are asserted throughout, against brute-force enumeration:

1. **Bound validity** -- the reported lower bound (``cost - gap``) never
   exceeds the true optimum.
2. **Certificate honesty** -- ``converged=True`` implies the returned
   assignment really is optimal.
"""

import itertools

import numpy as np
import pytest

from pytcl.assignment_algorithms.nd_assignment import relaxation_assignment_nd
from pytcl.assignment_algorithms.three_dimensional.assignment import (
    assign3d_lagrangian,
)


def brute_force_optimal(cost: np.ndarray) -> float:
    """Exact optimum by enumerating all assignments (small tensors only)."""
    n = cost.shape[0]
    n_dims = cost.ndim
    best = np.inf
    for perms in itertools.product(
        *[itertools.permutations(range(n)) for _ in range(n_dims - 1)]
    ):
        total = sum(cost[(i,) + tuple(p[i] for p in perms)] for i in range(n))
        best = min(best, total)
    return float(best)


def assert_feasible(assignments: np.ndarray, n_dims: int) -> None:
    """No index may repeat within any dimension."""
    for d in range(n_dims):
        column = assignments[:, d].tolist()
        assert len(set(column)) == len(column), f"index reused in dimension {d}"


class TestNDBoundValidity:
    @pytest.mark.parametrize("n_dims", [3, 4])
    def test_lower_bound_never_exceeds_optimum(self, n_dims):
        rng = np.random.default_rng(20 + n_dims)
        for _ in range(25):
            n = 3
            cost = rng.random((n,) * n_dims)
            res = relaxation_assignment_nd(cost, max_iterations=60)
            lower_bound = res.cost - res.gap
            optimum = brute_force_optimal(cost)
            assert lower_bound <= optimum + 1e-9, (
                f"invalid bound: LB={lower_bound:.6f} > optimum={optimum:.6f}"
            )

    @pytest.mark.parametrize("n_dims", [3, 4])
    def test_convergence_certificate_is_honest(self, n_dims):
        """converged=True must imply the answer is optimal."""
        rng = np.random.default_rng(40 + n_dims)
        certified = 0
        for _ in range(25):
            cost = rng.random((3,) * n_dims)
            res = relaxation_assignment_nd(cost, max_iterations=60)
            optimum = brute_force_optimal(cost)
            if res.converged:
                certified += 1
                assert res.cost <= optimum + 1e-6, (
                    f"false certificate: cost={res.cost:.6f} vs "
                    f"optimum={optimum:.6f}, gap={res.gap:.2e}"
                )
        assert certified > 0, "nothing certified; the test would be vacuous"

    def test_gap_is_non_negative(self):
        """A negative gap would mean the bounds crossed, i.e. one is invalid."""
        rng = np.random.default_rng(7)
        for _ in range(30):
            n = int(rng.integers(2, 5))
            cost = rng.random((n, n, n))
            res = relaxation_assignment_nd(cost, max_iterations=50)
            assert res.gap >= -1e-9, f"negative gap {res.gap:.2e}"

    def test_solutions_are_feasible(self):
        rng = np.random.default_rng(8)
        for n_dims in (3, 4):
            for _ in range(15):
                cost = rng.random((3,) * n_dims)
                res = relaxation_assignment_nd(cost, max_iterations=40)
                assert_feasible(res.assignments, n_dims)

    def test_two_dimensional_input_is_solved_exactly(self):
        """A 2-D problem needs no relaxation and must be exactly optimal."""
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(9)
        for _ in range(20):
            n = int(rng.integers(2, 7))
            cost = rng.random((n, n))
            res = relaxation_assignment_nd(cost)
            rows, cols = linear_sum_assignment(cost)
            assert res.converged
            assert res.gap == pytest.approx(0.0, abs=1e-12)
            assert res.cost == pytest.approx(float(cost[rows, cols].sum()))

    def test_reported_cost_matches_assignments(self):
        rng = np.random.default_rng(10)
        for _ in range(20):
            cost = rng.random((4, 4, 4))
            res = relaxation_assignment_nd(cost, max_iterations=40)
            recomputed = float(cost[tuple(res.assignments.T)].sum())
            assert res.cost == pytest.approx(recomputed, abs=1e-9)


class TestAssign3DLagrangianBoundValidity:
    def test_lower_bound_never_exceeds_optimum(self):
        rng = np.random.default_rng(31)
        for _ in range(40):
            n = int(rng.integers(2, 5))
            cost = rng.random((n, n, n))
            res = assign3d_lagrangian(cost, max_iter=60)
            lower_bound = res.cost - res.gap
            optimum = brute_force_optimal(cost)
            assert lower_bound <= optimum + 1e-9

    def test_convergence_certificate_is_honest(self):
        rng = np.random.default_rng(32)
        certified = 0
        for _ in range(40):
            n = int(rng.integers(2, 5))
            cost = rng.random((n, n, n))
            res = assign3d_lagrangian(cost, max_iter=60)
            optimum = brute_force_optimal(cost)
            if res.converged:
                certified += 1
                assert res.cost <= optimum + 1e-6, "false optimality certificate"
        assert certified > 0

    def test_maximize_certificate_is_honest(self):
        rng = np.random.default_rng(33)
        certified = 0
        for _ in range(30):
            n = int(rng.integers(2, 5))
            cost = rng.random((n, n, n))
            res = assign3d_lagrangian(cost, max_iter=60, maximize=True)
            best = -brute_force_optimal(-cost)
            if res.converged:
                certified += 1
                assert res.cost >= best - 1e-6, "false optimality certificate"
        assert certified > 0

    def test_gap_non_negative_and_solutions_feasible(self):
        rng = np.random.default_rng(34)
        for maximize in (False, True):
            for _ in range(20):
                n = int(rng.integers(2, 5))
                cost = rng.random((n, n, n))
                res = assign3d_lagrangian(cost, max_iter=50, maximize=maximize)
                assert res.gap >= -1e-9
                assert_feasible(res.tuples, 3)

    def test_obvious_diagonal_is_certified_optimal(self):
        cost = np.full((4, 4, 4), 100.0)
        for i in range(4):
            cost[i, i, i] = 1.0
        res = assign3d_lagrangian(cost, max_iter=100)
        assert res.cost == pytest.approx(4.0)
        assert res.converged
        assert res.gap == pytest.approx(0.0, abs=1e-6)

    def test_max_iter_zero_does_not_raise(self):
        """gap was NameError-prone when the loop body never ran."""
        cost = np.random.default_rng(0).random((3, 3, 3))
        res = assign3d_lagrangian(cost, max_iter=0)
        assert res.tuples.shape[1] == 3
        assert np.isfinite(res.gap) or np.isinf(res.gap)
