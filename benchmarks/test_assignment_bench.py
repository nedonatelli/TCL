"""
Benchmarks for 2D assignment algorithms.

These are full benchmarks that run on main branch merges and nightly builds.
"""

import pytest
from scipy.optimize import linear_sum_assignment as scipy_lsa

from pytcl.assignment_algorithms.two_dimensional.assignment import assign2d, hungarian


class TestHungarianBenchmarks:
    """Benchmark the Hungarian (linear sum assignment) solver."""

    @pytest.mark.full
    def test_hungarian_dense_500x500(self, benchmark, dense_cost_matrix_500):
        """Benchmark hungarian() on a dense 500x500 cost matrix.

        Reproduces the ROADMAP.md "Hungarian Assignment (500x500)"
        performance target. pytcl's hungarian() is a thin wrapper around
        scipy.optimize.linear_sum_assignment (see
        test_hungarian_scipy_floor_500x500 below for the reference it
        delegates to).
        """
        cost = dense_cost_matrix_500

        row_ind, col_ind, total_cost = benchmark(hungarian, cost)

        assert len(row_ind) == 500
        assert len(col_ind) == 500

    @pytest.mark.full
    def test_hungarian_scipy_floor_500x500(self, benchmark, dense_cost_matrix_500):
        """Benchmark scipy.optimize.linear_sum_assignment directly.

        Same 500x500 matrix as test_hungarian_dense_500x500, run through
        scipy directly rather than pytcl's hungarian() wrapper. This is
        the floor reference: pytcl's own implementation cannot beat this
        without replacing the delegation to scipy entirely.
        """
        cost = dense_cost_matrix_500

        row_ind, col_ind = benchmark(scipy_lsa, cost)

        assert len(row_ind) == 500
        assert len(col_ind) == 500


class TestAssign2DBenchmarks:
    """Benchmark assign2d's finite-cost_of_non_assignment augmented path."""

    @pytest.mark.full
    def test_assign2d_augmented_500x500(self, benchmark, dense_cost_matrix_500):
        """Benchmark assign2d() with a finite cost_of_non_assignment.

        A finite `cost_of_non_assignment` makes `assign2d`
        (pytcl/assignment_algorithms/two_dimensional/assignment.py:277)
        build an (n+m)x(n+m) augmented matrix before delegating to scipy,
        so the same 500x500 dense cost matrix as
        `test_hungarian_dense_500x500` becomes a 1000x1000 internal solve
        here -- this is the doubled-dimension path the v2.5.0 campaign
        parked without measurement.
        """
        cost = dense_cost_matrix_500

        result = benchmark(assign2d, cost, cost_of_non_assignment=50.0)

        assert len(result.row_indices) + len(result.unassigned_rows) == 500
        assert len(result.col_indices) + len(result.unassigned_cols) == 500
