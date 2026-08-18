"""
Benchmarks for 2D assignment algorithms.

These are full benchmarks that run on main branch merges and nightly builds.
"""

import pytest
from scipy.optimize import linear_sum_assignment as scipy_lsa

from pytcl.assignment_algorithms.two_dimensional.assignment import hungarian


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
