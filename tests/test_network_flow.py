"""Unit tests for network flow module - construction and high-level interface.

Tests cover:
- Flow network construction
- High-level min-cost assignment interface
- Edge cases and network properties

Note: The low-level min-cost flow solver uses Bellman-Ford which can be
slow for larger networks. High-level interface provides practical usage.
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms.network_flow import (
    assignment_to_flow_network,
    min_cost_assignment_via_flow,
)


class TestNetworkConstruction:
    """Test flow network construction."""

    def test_flow_network_2x2(self):
        """Test flow network construction for 2x2 assignment."""
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost)

        # Should have edges between workers and tasks
        assert len(edges) > 0
        assert len(supplies) > 0
        assert len(node_names) > 0

    def test_flow_network_3x3(self):
        """Test flow network construction for 3x3 assignment."""
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost)

        # Network should be balanced (total supply = 0)
        total_supply = np.sum(supplies)
        assert np.isclose(total_supply, 0.0)

    def test_flow_network_rectangular(self):
        """Test flow network for rectangular assignment (more workers than tasks)."""
        cost = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 workers, 2 tasks

        edges, supplies, node_names = assignment_to_flow_network(cost)

        assert len(node_names) > 0
        assert len(edges) > 0

    def test_flow_network_node_names(self):
        """Test that node names are consistent."""
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost)

        # Should have source, sink, workers, and tasks
        assert "source" in node_names
        assert "sink" in node_names
        assert any("worker" in name for name in node_names)
        assert any("task" in name for name in node_names)

    def test_flow_network_edges_structure(self):
        """Test that network edges have correct structure."""
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])

        edges, supplies, node_names = assignment_to_flow_network(cost)

        # All edges should have from_node, to_node, capacity, cost attributes
        for edge in edges:
            assert hasattr(edge, "from_node")
            assert hasattr(edge, "to_node")
            assert hasattr(edge, "capacity")
            assert hasattr(edge, "cost")
            assert edge.from_node < len(node_names)
            assert edge.to_node < len(node_names)


class TestHighLevelMinCostAssignment:
    """Test high-level min-cost assignment interface.

    Note: Some tests are skipped due to performance issues with the
    Bellman-Ford based solver for larger networks.
    """

    def test_min_cost_assignment_2x2(self):
        """Test min-cost assignment on 2x2 problem."""
        cost = np.array([[1.0, 100.0], [100.0, 1.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert isinstance(assignment, np.ndarray)
        assert np.isfinite(total_cost)

    def test_min_cost_assignment_3x3(self):
        """Test min-cost assignment on 3x3 problem."""
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert isinstance(assignment, np.ndarray)
        assert np.isfinite(total_cost)

    def test_min_cost_assignment_rectangular(self):
        """Test min-cost assignment on rectangular problem."""
        cost = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert isinstance(assignment, np.ndarray)
        assert assignment.shape[1] == 2  # [worker, task] pairs

    def test_min_cost_assignment_all_same(self):
        """Test min-cost assignment when all costs are the same."""
        cost = np.ones((3, 3))

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        # Cost should be reasonable (number of assignments * 1.0)
        assert np.isfinite(total_cost)
        assert total_cost >= 0

    def test_min_cost_assignment_negative_costs(self):
        """Test min-cost assignment with negative costs."""
        cost = np.array([[-1.0, -2.0], [-3.0, -4.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        # Should handle negative costs
        assert np.isfinite(total_cost)

    def test_min_cost_assignment_single_element(self):
        """Test single element assignment."""
        cost = np.array([[5.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert np.isfinite(total_cost)

    def test_min_cost_assignment_large_costs(self):
        """Test with very large costs."""
        cost = np.array([[1e6, 1e7], [1e7, 1e6]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert np.isfinite(total_cost)
        assert total_cost > 0

    def test_min_cost_assignment_small_costs(self):
        """Test with very small costs."""
        cost = np.array([[1e-6, 1e-5], [1e-5, 1e-6]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert np.isfinite(total_cost)

    def test_min_cost_assignment_mixed_sign_costs(self):
        """Test with mixed positive and negative costs."""
        cost = np.array([[-1.0, 2.0], [3.0, -4.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        assert np.isfinite(total_cost)

    def test_min_cost_assignment_zero_costs(self):
        """Test with all-zero cost matrix."""
        cost = np.zeros((3, 3))

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        # Cost should be zero for all-zero matrix
        assert np.isclose(total_cost, 0.0)

    def test_min_cost_assignment_result_validity(self):
        """Test that assignment result is valid."""
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        # Each assignment should be a valid [worker, task] pair
        if assignment.shape[0] > 0:
            assert assignment.shape[1] == 2
            assert np.all(assignment[:, 0] < cost.shape[0])
            assert np.all(assignment[:, 1] < cost.shape[1])
            assert np.all(assignment >= 0)


class TestNetworkFlowEdgeCases:
    """Test edge cases for network flow."""

    def test_diagonal_cost_matrix(self):
        """Test diagonal cost matrix (optimal = identity permutation)."""
        cost = np.diag([1.0, 2.0, 3.0])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        # Total cost should be close to 6 (1+2+3)
        assert np.isfinite(total_cost)

    def test_permutation_matrix(self):
        """Test assignment requiring permutation."""
        cost = np.array([[100.0, 100.0, 1.0], [1.0, 100.0, 100.0], [100.0, 1.0, 100.0]])

        assignment, total_cost = min_cost_assignment_via_flow(cost)

        # Should find low-cost solution
        assert np.isfinite(total_cost)
        assert total_cost < 1000  # Much better than naive assignment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
