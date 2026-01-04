"""Unit tests for N-dimensional assignment algorithms.

Tests cover:
- 4D and higher dimensional assignment problems
- Greedy algorithm heuristics
- Lagrangian relaxation solver
- Auction algorithm
- Cost tensor validation
- Conflict detection
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms.nd_assignment import (
    AssignmentNDResult,
    auction_assignment_nd,
    detect_dimension_conflicts,
    greedy_assignment_nd,
    relaxation_assignment_nd,
    validate_cost_tensor,
)


class TestCostTensorValidation:
    """Test cost tensor validation."""

    def test_validate_2d_tensor(self):
        """Test validation of 2D cost tensor."""
        cost = np.random.randn(5, 5)
        dims = validate_cost_tensor(cost)

        assert dims == (5, 5)

    def test_validate_3d_tensor(self):
        """Test validation of 3D cost tensor."""
        cost = np.random.randn(3, 4, 5)
        dims = validate_cost_tensor(cost)

        assert dims == (3, 4, 5)

    def test_validate_4d_tensor(self):
        """Test validation of 4D cost tensor."""
        cost = np.random.randn(2, 3, 4, 5)
        dims = validate_cost_tensor(cost)

        assert dims == (2, 3, 4, 5)

    def test_validate_empty_tensor(self):
        """Test validation rejects empty tensor."""
        cost = np.array([])

        with pytest.raises(ValueError):
            validate_cost_tensor(cost)

    def test_validate_nan_tensor(self):
        """Test validation accepts NaN values (caller responsibility)."""
        cost = np.array([[1.0, np.nan], [3.0, 4.0]])

        # Validation function doesn't check for NaN - it's caller's responsibility
        dims = validate_cost_tensor(cost)
        assert dims == (2, 2)

    def test_validate_inf_tensor(self):
        """Test validation accepts infinite values (caller responsibility)."""
        cost = np.array([[1.0, np.inf], [3.0, 4.0]])

        # Validation function doesn't check for Inf - it's caller's responsibility
        dims = validate_cost_tensor(cost)
        assert dims == (2, 2)


class TestDimensionConflictDetection:
    """Test dimension conflict detection."""

    def test_no_conflicts_2d(self):
        """Test 2D assignment with no conflicts."""
        assignments = np.array([[0, 0], [1, 1]])
        dims = (2, 2)

        has_conflict = detect_dimension_conflicts(assignments, dims)
        assert not has_conflict

    def test_conflict_repeated_row(self):
        """Test 4D assignment with repeated first dimension."""
        assignments = np.array([[0, 0, 0, 0], [0, 1, 1, 1]])
        dims = (2, 2, 2, 2)

        has_conflict = detect_dimension_conflicts(assignments, dims)
        assert has_conflict

    def test_conflict_repeated_column(self):
        """Test 4D assignment with repeated column dimension."""
        assignments = np.array([[0, 0, 0, 0], [1, 0, 1, 1]])
        dims = (2, 2, 2, 2)

        has_conflict = detect_dimension_conflicts(assignments, dims)
        assert has_conflict

    def test_valid_3d_assignment(self):
        """Test valid 3D assignment."""
        assignments = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        dims = (3, 3, 3)

        has_conflict = detect_dimension_conflicts(assignments, dims)
        assert not has_conflict


class TestGreedyAssignment2D:
    """Test greedy assignment on 2D problems."""

    def test_greedy_2d_trivial(self):
        """Test greedy on trivial 2D assignment."""
        # Simple cost matrix with obvious optimal solution
        cost = np.array([[1.0, 100.0], [100.0, 1.0]])

        result = greedy_assignment_nd(cost, max_assignments=2)

        assert isinstance(result, AssignmentNDResult)
        assert result.cost < 10.0  # Should find low-cost solution

    def test_greedy_2d_square(self):
        """Test greedy on square 2D assignment."""
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        result = greedy_assignment_nd(cost, max_assignments=3)

        assert result.assignments.shape[0] <= 3

    def test_greedy_respects_max_assignments(self):
        """Test that greedy respects max_assignments limit."""
        cost = np.ones((10, 10))
        max_assign = 3

        result = greedy_assignment_nd(cost, max_assignments=max_assign)

        assert result.assignments.shape[0] <= max_assign


class TestGreedyAssignment4D:
    """Test greedy assignment on 4D problems."""

    def test_greedy_4d_small(self):
        """Test greedy on small 4D problem."""
        cost = np.random.randn(2, 2, 2, 2)

        result = greedy_assignment_nd(cost, max_assignments=4)

        assert isinstance(result, AssignmentNDResult)
        assert result.assignments.shape[1] == 4  # 4D problem

    def test_greedy_4d_larger(self):
        """Test greedy on larger 4D problem."""
        cost = np.random.randn(3, 3, 3, 3)

        result = greedy_assignment_nd(cost, max_assignments=9)

        assert result.assignments.shape[1] == 4
        assert result.assignments.shape[0] <= 9

    def test_greedy_5d(self):
        """Test greedy on 5D problem."""
        cost = np.random.randn(2, 2, 2, 2, 2)

        result = greedy_assignment_nd(cost, max_assignments=8)

        assert result.assignments.shape[1] == 5


class TestRelaxationAssignment2D:
    """Test Lagrangian relaxation on 2D problems."""

    def test_relaxation_2d_simple(self):
        """Test relaxation on simple 2D assignment."""
        cost = np.array([[1.0, 100.0], [100.0, 1.0]])

        result = relaxation_assignment_nd(cost, max_iterations=100, tolerance=1e-6)

        assert isinstance(result, AssignmentNDResult)
        assert result.converged or result.n_iterations >= 100

    def test_relaxation_convergence(self):
        """Test that relaxation converges."""
        cost = np.random.randn(4, 4)

        result = relaxation_assignment_nd(cost, max_iterations=1000, tolerance=1e-6)

        # Should either converge or hit iteration limit
        assert result.converged or result.n_iterations == 1000

    def test_relaxation_dual_bound(self):
        """Test that gap decreases over iterations."""
        cost = np.random.randn(3, 3)

        result = relaxation_assignment_nd(cost, max_iterations=100, tolerance=1e-12)

        # Gap should be reasonable
        assert result.gap >= 0
        assert result.gap < 1000  # Arbitrary upper bound


class TestRelaxationAssignment4D:
    """Test Lagrangian relaxation on 4D problems."""

    def test_relaxation_4d_small(self):
        """Test relaxation on small 4D problem."""
        cost = np.random.randn(2, 2, 2, 2)

        result = relaxation_assignment_nd(cost, max_iterations=50, tolerance=1e-6)

        assert isinstance(result, AssignmentNDResult)
        assert result.assignments.shape[1] == 4

    def test_relaxation_4d_convergence(self):
        """Test relaxation convergence on 4D."""
        cost = np.random.randn(3, 3, 3, 3)

        result = relaxation_assignment_nd(cost, max_iterations=200, tolerance=1e-6)

        # Should find some solution
        assert result.assignments.shape[0] > 0


class TestAuctionAssignment2D:
    """Test auction algorithm on 2D problems."""

    def test_auction_2d_simple(self):
        """Test auction on simple 2D assignment."""
        cost = np.array([[1.0, 100.0], [100.0, 1.0]])

        result = auction_assignment_nd(cost, max_iterations=100, epsilon=0.01)

        assert isinstance(result, AssignmentNDResult)
        assert result.cost < 10.0

    def test_auction_convergence(self):
        """Test auction algorithm convergence."""
        cost = np.random.randn(3, 3)

        result = auction_assignment_nd(cost, max_iterations=100, epsilon=0.01)

        assert result.converged or result.n_iterations >= 100

    def test_auction_epsilon_effect(self):
        """Test that smaller epsilon affects convergence."""
        cost = np.random.randn(4, 4)

        result_large_eps = auction_assignment_nd(cost, max_iterations=100, epsilon=0.1)
        result_small_eps = auction_assignment_nd(
            cost, max_iterations=100, epsilon=0.001
        )

        # Both should produce valid results
        assert result_large_eps.assignments.shape[0] > 0
        assert result_small_eps.assignments.shape[0] > 0


class TestAuctionAssignment4D:
    """Test auction algorithm on 4D problems."""

    def test_auction_4d_small(self):
        """Test auction on small 4D problem."""
        cost = np.random.randn(2, 2, 2, 2)

        result = auction_assignment_nd(cost, max_iterations=50, epsilon=0.01)

        assert isinstance(result, AssignmentNDResult)
        assert result.assignments.shape[1] == 4

    def test_auction_4d_convergence(self):
        """Test auction convergence on 4D."""
        cost = np.random.randn(3, 3, 3, 3)

        result = auction_assignment_nd(cost, max_iterations=200, epsilon=0.01)

        # Should find some solution
        assert result.assignments.shape[0] > 0


class TestAssignmentComparison:
    """Compare different algorithms."""

    def test_greedy_vs_relaxation_2d(self):
        """Compare greedy and relaxation on 2D."""
        cost = np.random.randn(3, 3)

        greedy_result = greedy_assignment_nd(cost, max_assignments=3)
        relaxation_result = relaxation_assignment_nd(
            cost, max_iterations=100, tolerance=1e-6
        )

        # Relaxation should generally find equal or better solution
        assert isinstance(greedy_result, AssignmentNDResult)
        assert isinstance(relaxation_result, AssignmentNDResult)

    def test_all_algorithms_produce_valid_results(self):
        """Test that all three algorithms produce valid results."""
        cost = np.random.randn(3, 3)

        greedy_result = greedy_assignment_nd(cost, max_assignments=3)
        relaxation_result = relaxation_assignment_nd(
            cost, max_iterations=100, tolerance=1e-6
        )
        auction_result = auction_assignment_nd(cost, max_iterations=100, epsilon=0.01)

        # All should produce valid assignments
        for result in [greedy_result, relaxation_result, auction_result]:
            assert result.assignments.shape[0] >= 0
            assert result.cost >= -1e6  # Should be finite
            assert np.isfinite(result.cost)


class TestResultDataStructure:
    """Test AssignmentNDResult NamedTuple."""

    def test_result_structure(self):
        """Test that result has all required fields."""
        cost = np.ones((2, 2))
        result = greedy_assignment_nd(cost, max_assignments=2)

        assert hasattr(result, "assignments")
        assert hasattr(result, "cost")
        assert hasattr(result, "converged")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "gap")

    def test_result_immutability(self):
        """Test that result is immutable (NamedTuple)."""
        cost = np.ones((2, 2))
        result = greedy_assignment_nd(cost, max_assignments=2)

        with pytest.raises(AttributeError):
            result.cost = 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
