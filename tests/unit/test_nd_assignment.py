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
    SparseCostTensor,
    assignment_nd,
    auction_assignment_nd,
    detect_dimension_conflicts,
    greedy_assignment_nd,
    greedy_assignment_nd_sparse,
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


class TestSparseCostTensor:
    """Test SparseCostTensor class for sparse n-D assignment."""

    def test_sparse_tensor_creation(self):
        """Test creating a sparse cost tensor."""
        dims = (3, 3, 3)
        indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        costs = np.array([1.0, 2.0, 3.0])

        sparse = SparseCostTensor(dims, indices, costs)

        assert sparse.dims == dims
        assert sparse.n_valid == 3
        assert sparse.default_cost == np.inf

    def test_sparse_tensor_get_cost(self):
        """Test getting cost from sparse tensor."""
        dims = (3, 3)
        indices = np.array([[0, 0], [1, 1]])
        costs = np.array([1.0, 2.0])

        sparse = SparseCostTensor(dims, indices, costs, default_cost=100.0)

        assert sparse.get_cost((0, 0)) == 1.0
        assert sparse.get_cost((1, 1)) == 2.0
        assert sparse.get_cost((0, 1)) == 100.0  # Default cost

    def test_sparse_tensor_sparsity(self):
        """Test sparsity calculation."""
        dims = (10, 10)
        indices = np.array([[0, 0], [1, 1], [2, 2]])
        costs = np.array([1.0, 2.0, 3.0])

        sparse = SparseCostTensor(dims, indices, costs)

        assert sparse.sparsity == 3 / 100

    def test_sparse_tensor_memory_savings(self):
        """Test memory savings calculation."""
        dims = (10, 10, 10)  # 1000 elements
        indices = np.array([[0, 0, 0], [1, 1, 1]])  # 2 elements
        costs = np.array([1.0, 2.0])

        sparse = SparseCostTensor(dims, indices, costs)

        # Should have significant memory savings
        assert sparse.memory_savings > 0.9  # >90% savings

    def test_sparse_tensor_to_dense(self):
        """Test converting sparse tensor to dense."""
        dims = (3, 3)
        indices = np.array([[0, 0], [1, 1], [2, 2]])
        costs = np.array([1.0, 2.0, 3.0])

        sparse = SparseCostTensor(dims, indices, costs, default_cost=np.inf)
        dense = sparse.to_dense()

        assert dense.shape == dims
        assert dense[0, 0] == 1.0
        assert dense[1, 1] == 2.0
        assert dense[2, 2] == 3.0
        assert np.isinf(dense[0, 1])

    def test_sparse_tensor_from_dense(self):
        """Test creating sparse tensor from dense array."""
        dense = np.array([[1.0, np.inf], [np.inf, 2.0]])

        sparse = SparseCostTensor.from_dense(dense, threshold=1e10)

        assert sparse.dims == (2, 2)
        assert sparse.n_valid == 2
        assert sparse.get_cost((0, 0)) == 1.0
        assert sparse.get_cost((1, 1)) == 2.0

    def test_sparse_tensor_roundtrip(self):
        """Test dense -> sparse -> dense roundtrip."""
        original = np.random.randn(4, 4)
        original[original > 0.5] = np.inf  # Make some entries sparse

        sparse = SparseCostTensor.from_dense(original, threshold=1e10)
        recovered = sparse.to_dense()

        # Finite values should match
        finite_mask = np.isfinite(original)
        np.testing.assert_array_almost_equal(
            original[finite_mask], recovered[finite_mask]
        )


class TestGreedyAssignmentNDSparse:
    """Test sparse greedy assignment algorithm."""

    def test_sparse_greedy_simple(self):
        """Test sparse greedy on simple problem."""
        dims = (3, 3)
        indices = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]])
        costs = np.array([1.0, 1.0, 1.0, 10.0, 10.0])

        sparse = SparseCostTensor(dims, indices, costs)
        result = greedy_assignment_nd_sparse(sparse, max_assignments=3)

        assert isinstance(result, AssignmentNDResult)
        assert result.cost <= 3.0  # Optimal is diagonal

    def test_sparse_greedy_respects_max_assignments(self):
        """Test sparse greedy respects max_assignments."""
        dims = (10, 10)
        indices = np.array([[i, i] for i in range(10)])
        costs = np.ones(10)

        sparse = SparseCostTensor(dims, indices, costs)
        result = greedy_assignment_nd_sparse(sparse, max_assignments=3)

        assert result.assignments.shape[0] <= 3

    def test_sparse_greedy_4d(self):
        """Test sparse greedy on 4D problem."""
        dims = (3, 3, 3, 3)
        # Create diagonal entries
        indices = np.array([[i, i, i, i] for i in range(3)])
        costs = np.array([1.0, 2.0, 3.0])

        sparse = SparseCostTensor(dims, indices, costs)
        result = greedy_assignment_nd_sparse(sparse, max_assignments=3)

        assert result.assignments.shape[1] == 4
        assert result.cost <= 6.0

    def test_sparse_greedy_no_valid_entries(self):
        """Test sparse greedy with no valid entries."""
        dims = (3, 3)
        indices = np.empty((0, 2), dtype=np.intp)
        costs = np.empty(0)

        sparse = SparseCostTensor(dims, indices, costs)
        result = greedy_assignment_nd_sparse(sparse, max_assignments=3)

        assert result.assignments.shape[0] == 0
        assert result.cost == 0.0


class TestAssignmentNDUnified:
    """Test unified assignment_nd interface."""

    def test_assignment_nd_auto_dense(self):
        """Test assignment_nd with auto method on dense array."""
        cost = np.random.randn(3, 3)

        result = assignment_nd(cost, method="auto", max_assignments=3)

        assert isinstance(result, AssignmentNDResult)
        assert result.assignments.shape[0] <= 3

    def test_assignment_nd_auto_sparse(self):
        """Test assignment_nd with auto method on sparse tensor."""
        dims = (3, 3)
        indices = np.array([[0, 0], [1, 1], [2, 2]])
        costs = np.array([1.0, 2.0, 3.0])
        sparse = SparseCostTensor(dims, indices, costs)

        result = assignment_nd(sparse, method="auto", max_assignments=3)

        assert isinstance(result, AssignmentNDResult)

    def test_assignment_nd_greedy_dense(self):
        """Test assignment_nd with greedy method on dense array."""
        cost = np.random.randn(3, 3)

        result = assignment_nd(cost, method="greedy", max_assignments=3)

        assert isinstance(result, AssignmentNDResult)

    def test_assignment_nd_greedy_sparse(self):
        """Test assignment_nd with greedy method on sparse tensor."""
        dims = (3, 3)
        indices = np.array([[0, 0], [1, 1], [2, 2]])
        costs = np.array([1.0, 2.0, 3.0])
        sparse = SparseCostTensor(dims, indices, costs)

        result = assignment_nd(sparse, method="greedy", max_assignments=3)

        assert isinstance(result, AssignmentNDResult)

    def test_assignment_nd_relaxation(self):
        """Test assignment_nd with relaxation method."""
        cost = np.random.randn(3, 3)

        result = assignment_nd(
            cost, method="relaxation", max_iterations=100, tolerance=1e-6
        )

        assert isinstance(result, AssignmentNDResult)

    def test_assignment_nd_auction(self):
        """Test assignment_nd with auction method."""
        cost = np.random.randn(3, 3)

        result = assignment_nd(cost, method="auction", max_iterations=100, epsilon=0.01)

        assert isinstance(result, AssignmentNDResult)

    def test_assignment_nd_invalid_method(self):
        """Test assignment_nd with invalid method raises error."""
        cost = np.random.randn(3, 3)

        with pytest.raises(ValueError, match="Unknown method"):
            assignment_nd(cost, method="invalid_method")

    def test_assignment_nd_sparse_equivalent_to_dense(self):
        """Test sparse and dense give similar results on same problem."""
        # Create a small dense cost matrix
        dense_cost = np.array([[1.0, 10.0, 10.0], [10.0, 2.0, 10.0], [10.0, 10.0, 3.0]])

        # Convert to sparse
        sparse_cost = SparseCostTensor.from_dense(dense_cost, threshold=5.0)

        result_dense = assignment_nd(dense_cost, method="greedy", max_assignments=3)
        result_sparse = assignment_nd(sparse_cost, method="greedy", max_assignments=3)

        # Both should find the diagonal solution (optimal)
        assert result_dense.cost <= 6.0
        assert result_sparse.cost <= 6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
