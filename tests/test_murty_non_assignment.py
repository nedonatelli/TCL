"""
k-best enumeration with a finite cost of non-assignment.

Regression tests for gh-15. ``murty`` and ``kbest_assign2d`` partitioned over
the raw cost matrix, which can only represent *complete* matchings. With a
finite ``cost_of_non_assignment`` every solution that leaves something
unassigned was therefore unreachable, and the ranked list was silently
truncated: asking for k=7 on a 2x2 problem returned 2 solutions, omitting the
five that involve non-assignment.

Both functions now enumerate over an augmented rectangular problem in which
each row may take a private zero-cost dummy column. That encoding is
*bijective* -- one representation per real solution -- unlike the square
(n+m)x(n+m) form used by ``assign2d``, whose zero-cost dummy-to-dummy block
gives a solution with r pairs r! equivalent representations.

Ground truth throughout is exhaustive enumeration of every partial matching,
scored with the same cost model as ``assign2d``: pairs plus the penalty for
each unassigned row *and* each unassigned column.
"""

import itertools

import numpy as np
import pytest

from pytcl.assignment_algorithms.two_dimensional.assignment import assign2d
from pytcl.assignment_algorithms.two_dimensional.kbest import (
    kbest_assign2d,
    murty,
)


def enumerate_all(cost, cna, maximize=False):
    """All partial matchings, ranked, under the assign2d cost model."""
    n, m = cost.shape
    out = []
    for r in range(min(n, m) + 1):
        for rows in itertools.combinations(range(n), r):
            for cols in itertools.permutations(range(m), r):
                base = sum(cost[i, j] for i, j in zip(rows, cols))
                penalty = cna * ((n - r) + (m - r))
                total = base - penalty if maximize else base + penalty
                out.append(round(float(total), 9))
    return sorted(out, reverse=maximize)


class TestRankedListIsComplete:
    def test_documented_failure_case(self):
        """The exact case from gh-15: k=7 returned only 2 solutions."""
        cost = np.array([[3.0, 7.0], [8.0, 4.0]])
        result = murty(cost, k=7, cost_of_non_assignment=10.0)
        expected = [7.0, 15.0, 23.0, 24.0, 27.0, 28.0, 40.0]
        assert result.n_found == 7
        np.testing.assert_allclose(result.costs, expected)

    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    @pytest.mark.parametrize("maximize", [False, True])
    def test_matches_exhaustive_enumeration(self, solver, maximize):
        rng = np.random.default_rng(5)
        for _ in range(40):
            n = int(rng.integers(1, 4))
            m = int(rng.integers(1, 4))
            cost = np.round(rng.uniform(0, 10, size=(n, m)), 3)
            cna = float(np.round(rng.uniform(1, 12), 3))
            truth = enumerate_all(cost, cna, maximize)
            k = min(8, len(truth))
            res = solver(cost, k=k, cost_of_non_assignment=cna, maximize=maximize)
            assert res.n_found == k, f"asked for {k}, got {res.n_found}"
            np.testing.assert_allclose(res.costs, truth[:k], atol=1e-9)

    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    def test_costs_are_monotonic(self, solver):
        rng = np.random.default_rng(6)
        for _ in range(20):
            cost = rng.uniform(0, 10, size=(3, 3))
            res = solver(cost, k=10, cost_of_non_assignment=6.0)
            assert np.all(np.diff(res.costs) >= -1e-9), "ranked order violated"

    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    def test_returned_assignments_are_distinct(self, solver):
        """The augmented encoding must not yield duplicate real solutions."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            cost = rng.uniform(0, 10, size=(3, 3))
            res = solver(cost, k=12, cost_of_non_assignment=5.0)
            seen = {
                tuple(sorted(zip(a.row_indices.tolist(), a.col_indices.tolist())))
                for a in res.assignments
            }
            assert len(seen) == res.n_found, "duplicate assignments returned"


class TestReportedCostsAreConsistent:
    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    def test_cost_matches_recomputed_value(self, solver):
        """Each reported cost must equal pairs + penalty, recomputed."""
        rng = np.random.default_rng(8)
        cna = 7.0
        for _ in range(20):
            n, m = 3, 4
            cost = rng.uniform(0, 10, size=(n, m))
            res = solver(cost, k=6, cost_of_non_assignment=cna)
            for a, reported in zip(res.assignments, res.costs):
                pairs = float(cost[a.row_indices, a.col_indices].sum())
                r = len(a.row_indices)
                expected = pairs + cna * ((n - r) + (m - r))
                assert reported == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    def test_unassigned_fields_are_correct(self, solver):
        rng = np.random.default_rng(9)
        for _ in range(15):
            n, m = 3, 3
            cost = rng.uniform(0, 10, size=(n, m))
            res = solver(cost, k=8, cost_of_non_assignment=4.0)
            for a in res.assignments:
                assert set(a.unassigned_rows.tolist()) == set(range(n)) - set(
                    a.row_indices.tolist()
                )
                assert set(a.unassigned_cols.tolist()) == set(range(m)) - set(
                    a.col_indices.tolist()
                )

    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    def test_first_solution_matches_assign2d(self, solver):
        """The best of the k-best must be exactly what assign2d returns."""
        rng = np.random.default_rng(10)
        for _ in range(20):
            cost = rng.uniform(0, 10, size=(4, 3))
            cna = 6.0
            best = assign2d(cost, cost_of_non_assignment=cna)
            res = solver(cost, k=3, cost_of_non_assignment=cna)
            assert res.costs[0] == pytest.approx(best.cost, abs=1e-9)


class TestCompleteMatchingUnaffected:
    """Guard the infinite-cost path, which was already correct."""

    @pytest.mark.parametrize("solver", [murty, kbest_assign2d])
    def test_matches_permutation_enumeration(self, solver):
        rng = np.random.default_rng(11)
        for _ in range(20):
            n = int(rng.integers(2, 5))
            cost = np.round(rng.uniform(0, 10, size=(n, n)), 3)
            truth = sorted(
                round(float(sum(cost[i, p[i]] for i in range(n))), 9)
                for p in itertools.permutations(range(n))
            )
            k = min(6, len(truth))
            res = solver(cost, k=k)
            np.testing.assert_allclose(res.costs, truth[:k], atol=1e-9)

    def test_k_larger_than_solution_count(self):
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])
        res = murty(cost, k=99)
        assert res.n_found == 2  # only 2 permutations exist
