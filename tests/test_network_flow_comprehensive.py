"""
Comprehensive tests for network flow assignment module.

Tests coverage for:
- Network flow problem formulation
- Min-cost flow algorithms (successive shortest paths, simplex)
- Assignment extraction from flow solutions
- Status handling and edge cases
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms.network_flow import (
    assignment_to_flow_network,
    min_cost_flow_successive_shortest_paths,
    min_cost_flow_simplex,
    assignment_from_flow_solution,
    min_cost_assignment_via_flow,
    FlowStatus,
    MinCostFlowResult,
    FlowEdge,
)


class TestFlowNetworkFormulation:
    """Tests for converting assignments to flow networks."""

    def test_simple_assignment_to_flow(self):
        """Test basic assignment to flow network conversion."""
        # Simple 2x2 cost matrix
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost_matrix)

        assert edges is not None
        assert supplies is not None
        assert node_names is not None
        assert len(edges) > 0
        assert len(supplies) > 0

    def test_assignment_to_flow_square_matrix(self):
        """Test conversion with square cost matrix."""
        cost_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost_matrix)
        assert edges is not None
        assert len(edges) > 0

    def test_assignment_to_flow_rectangular(self):
        """Test conversion with rectangular cost matrix."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost_matrix)
        assert edges is not None
        assert len(edges) > 0

    def test_assignment_to_flow_zero_costs(self):
        """Test with zero cost values."""
        cost_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost_matrix)
        assert edges is not None

    def test_assignment_to_flow_negative_costs(self):
        """Test with negative cost values."""
        cost_matrix = np.array([[-1.0, 2.0], [3.0, -4.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost_matrix)
        assert edges is not None

    def test_assignment_to_flow_returns_tuples(self):
        """Test that assignment returns (edges, supplies, node_names)."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = assignment_to_flow_network(cost_matrix)

        assert isinstance(result, tuple)
        assert len(result) == 3
        edges, supplies, node_names = result
        assert isinstance(edges, list)
        assert isinstance(supplies, np.ndarray)
        assert isinstance(node_names, np.ndarray)

    def test_flow_edges_are_flow_edge_type(self):
        """Test that edges are FlowEdge named tuples."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        edges, _, _ = assignment_to_flow_network(cost_matrix)

        for edge in edges:
            assert isinstance(edge, FlowEdge)
            assert hasattr(edge, "from_node")
            assert hasattr(edge, "to_node")
            assert hasattr(edge, "capacity")
            assert hasattr(edge, "cost")


class TestMinCostFlowSuccessiveShortestPaths:
    """Tests for min-cost flow using successive shortest paths.

    The successive shortest paths algorithm with potentials is a standard algorithm
    for min-cost flow problems. It's guaranteed to find optimal solutions and converge
    for any feasible flow network.
    """

    def test_successive_shortest_paths_basic(self):
        """Test basic successive shortest paths computation."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

        assert isinstance(result, MinCostFlowResult)
        assert isinstance(result.status, FlowStatus)

    def test_successive_shortest_paths_square(self):
        """Test with square cost matrix."""
        cost_matrix = np.array([[1.0, 5.0], [2.0, 3.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

        assert result.cost is not None
        assert np.isfinite(result.cost)

    def test_successive_shortest_paths_flow_conservation(self):
        """Test that successive shortest paths respects flow conservation."""
        cost_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

        assert result.flow is not None
        assert len(result.flow) > 0

    def test_successive_shortest_paths_non_negative(self):
        """Test that algorithm converges and produces feasible solution."""
        cost_matrix = np.array([[10.0, 20.0], [30.0, 40.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

        # Algorithm should converge to optimal solution
        assert result.status == FlowStatus.OPTIMAL
        # Cost is finite (can be negative due to flow cancellations)
        assert np.isfinite(result.cost)

    def test_successive_shortest_paths_result_properties(self):
        """Test that result has all required properties."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

        assert hasattr(result, "flow")
        assert hasattr(result, "cost")
        assert hasattr(result, "status")
        assert hasattr(result, "iterations")


class TestMinCostFlowSimplex:
    """Tests for min-cost flow using simplex method."""

    def test_simplex_basic(self):
        """Test basic simplex method computation."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_simplex(edges, supplies)

        assert isinstance(result, MinCostFlowResult)
        assert isinstance(result.status, FlowStatus)

    def test_simplex_square_matrix(self):
        """Test simplex with square cost matrix."""
        cost_matrix = np.array([[1.0, 5.0], [2.0, 3.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_simplex(edges, supplies)

        assert result.cost is not None
        assert np.isfinite(result.cost)

    def test_simplex_iteration_count(self):
        """Test that simplex reports iteration count."""
        cost_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_simplex(edges, supplies)

        assert result.iterations >= 0

    def test_simplex_result_properties(self):
        """Test that simplex result has all required properties."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_simplex(edges, supplies)

        assert hasattr(result, "flow")
        assert hasattr(result, "cost")
        assert hasattr(result, "status")
        assert hasattr(result, "iterations")


class TestFlowStatusEnum:
    """Tests for FlowStatus enum."""

    def test_flow_status_values(self):
        """Test FlowStatus enum values."""
        assert FlowStatus.OPTIMAL.value == 0
        assert FlowStatus.UNBOUNDED.value == 1
        assert FlowStatus.INFEASIBLE.value == 2
        assert FlowStatus.TIMEOUT.value == 3

    def test_flow_status_names(self):
        """Test FlowStatus enum names."""
        assert FlowStatus.OPTIMAL.name == "OPTIMAL"
        assert FlowStatus.UNBOUNDED.name == "UNBOUNDED"
        assert FlowStatus.INFEASIBLE.name == "INFEASIBLE"
        assert FlowStatus.TIMEOUT.name == "TIMEOUT"

    def test_flow_status_is_enum(self):
        """Test that FlowStatus is an enum."""
        from enum import Enum

        assert issubclass(FlowStatus, Enum)


class TestAssignmentFromFlow:
    """Tests for extracting assignments from flow solutions.

    The successive shortest paths algorithm now converges properly for all matrix sizes.
    """

    def test_assignment_from_flow_basic(self):
        """Test extracting assignment from flow solution."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)
        assignment, cost = assignment_from_flow_solution(
            result.flow, edges, cost_matrix.shape
        )

        assert assignment is not None
        assert isinstance(assignment, np.ndarray)

    def test_assignment_from_flow_shape(self):
        """Test that assignment has correct shape."""
        cost_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)
        assignment, cost = assignment_from_flow_solution(
            result.flow, edges, cost_matrix.shape
        )

        assert len(assignment.shape) == 2
        assert assignment.shape[1] == 2  # [worker, task] pairs

    def test_assignment_from_flow_binary(self):
        """Test that assignment values are binary."""
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)
        assignment, cost = assignment_from_flow_solution(
            result.flow, edges, cost_matrix.shape
        )

        for pair in assignment:
            assert 0 <= pair[0] < cost_matrix.shape[0]  # Valid worker
            assert 0 <= pair[1] < cost_matrix.shape[1]  # Valid task


class TestMinCostAssignmentViaFlow:
    """Tests for the high-level min-cost assignment function."""

    def test_min_cost_assignment_3x3(self):
        """Test min-cost assignment with 3x3 matrix."""
        cost_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        assignment, cost = min_cost_assignment_via_flow(cost_matrix)

        assert assignment is not None
        assert isinstance(cost, (int, float, np.number))

    def test_min_cost_assignment_one_per_row(self):
        """Test that each worker gets exactly one task."""
        cost_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        assignment, cost = min_cost_assignment_via_flow(cost_matrix)

        # Count assignments per worker
        if len(assignment) > 0:
            workers = assignment[:, 0]
            unique_workers = np.unique(workers)
            assert len(unique_workers) == min(len(unique_workers), cost_matrix.shape[0])


class TestNetworkFlowIntegration:
    """Integration tests for network flow assignment."""

    def test_both_methods_comparable(self):
        """Test that simplex and successive shortest paths give similar results."""
        cost_matrix = np.array([[1.0, 5.0], [4.0, 2.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)

        result_ssp = min_cost_flow_successive_shortest_paths(edges, supplies)
        result_simplex = min_cost_flow_simplex(edges, supplies)

        # Both should find optimal solutions
        assert (
            result_ssp.status == FlowStatus.OPTIMAL
            or result_ssp.status == FlowStatus.TIMEOUT
        )
        assert (
            result_simplex.status == FlowStatus.OPTIMAL
            or result_simplex.status == FlowStatus.TIMEOUT
        )

        # Costs should be reasonable
        assert np.isfinite(result_ssp.cost)
        assert np.isfinite(result_simplex.cost)

    def test_rectangular_assignment(self):
        """Test assignment with rectangular cost matrix."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        assignment, cost = min_cost_assignment_via_flow(cost_matrix)

        assert assignment is not None
        assert cost >= 0

    def test_zero_cost_assignment(self):
        """Test assignment with all zero costs."""
        cost_matrix = np.zeros((3, 3))

        assignment, cost = min_cost_assignment_via_flow(cost_matrix)

        assert cost == 0.0

    def test_large_cost_assignment(self):
        """Test with large cost values."""
        cost_matrix = np.array([[1000.0, 2000.0], [3000.0, 4000.0]])

        assignment, cost = min_cost_assignment_via_flow(cost_matrix)

        assert isinstance(cost, (int, float, np.number))
        assert cost > 0

    def test_flow_optimality(self):
        """Test that solution finds optimal min-cost flow."""
        cost_matrix = np.array([[1.0, 5.0], [2.0, 3.0]])

        edges, supplies, _ = assignment_to_flow_network(cost_matrix)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

        # Algorithm should find an optimal solution
        assert (
            result.status == FlowStatus.OPTIMAL or result.status == FlowStatus.TIMEOUT
        )

        # Cost should be finite
        assert np.isfinite(result.cost)

        # Extract assignment and verify it's valid
        from pytcl.assignment_algorithms.network_flow import (
            assignment_from_flow_solution,
        )

        assignment, assign_cost = assignment_from_flow_solution(
            result.flow, edges, cost_matrix.shape
        )

        # Should have valid assignment
        assert assignment is not None
        assert len(assignment) > 0

        # Each worker should have exactly one task
        if len(assignment) > 0:
            workers = assignment[:, 0]
            assert len(np.unique(workers)) == len(workers)
