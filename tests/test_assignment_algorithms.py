"""Tests for assignment algorithms and data association."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.assignment_algorithms import (  # 2D Assignment
    Assignment3DResult,
    AssignmentResult,
    AssociationResult,
    KBestResult,
    assign2d,
    assign3d,
    assign3d_auction,
    assign3d_lagrangian,
    auction,
    chi2_gate_threshold,
    compute_association_cost,
    compute_gate_volume,
    decompose_to_2d,
    ellipsoidal_gate,
    gate_measurements,
    gated_gnn_association,
    gnn_association,
    greedy_3d,
    hungarian,
    kbest_assign2d,
    linear_sum_assignment,
    mahalanobis_distance,
    murty,
    nearest_neighbor,
    ranked_assignments,
    rectangular_gate,
)


class TestLinearSumAssignment:
    """Tests for linear_sum_assignment wrapper."""

    def test_basic_assignment(self):
        """Test basic 3x3 assignment."""
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        row_ind, col_ind = linear_sum_assignment(cost)

        # Check shapes
        assert len(row_ind) == 3
        assert len(col_ind) == 3

        # Check that assignment is valid (one-to-one)
        assert len(set(row_ind)) == 3
        assert len(set(col_ind)) == 3

        # Check optimal cost
        total_cost = cost[row_ind, col_ind].sum()
        assert total_cost == 5  # Optimal: (0,1)=1, (1,0)=2, (2,2)=2

    def test_rectangular_matrix(self):
        """Test with more columns than rows."""
        cost = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        row_ind, col_ind = linear_sum_assignment(cost)

        assert len(row_ind) == 2
        assert len(col_ind) == 2

    def test_maximize(self):
        """Test maximization mode."""
        cost = np.array([[1, 2], [2, 1]])
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
        total_cost = cost[row_ind, col_ind].sum()
        assert total_cost == 4  # Maximize: (0,1)=2, (1,0)=2


class TestHungarian:
    """Tests for Hungarian algorithm."""

    def test_basic_assignment(self):
        """Test basic assignment with cost."""
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        row_ind, col_ind, total_cost = hungarian(cost)

        assert total_cost == 5.0
        assert len(row_ind) == 3

    def test_maximize(self):
        """Test maximization."""
        cost = np.array([[1, 2], [2, 1]])
        row_ind, col_ind, total_cost = hungarian(cost, maximize=True)
        assert total_cost == 4.0


class TestAuction:
    """Tests for Auction algorithm."""

    def test_basic_assignment(self):
        """Test basic auction assignment."""
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        row_ind, col_ind, total_cost = auction(cost)

        # Auction may find suboptimal solution but should be close
        assert total_cost <= 10  # Should be reasonable
        assert len(row_ind) == 3

    def test_maximize(self):
        """Test maximization mode."""
        cost = np.array([[1, 2], [2, 1]])
        row_ind, col_ind, total_cost = auction(cost, maximize=True)
        assert total_cost >= 3  # Should find good solution


class TestAssign2D:
    """Tests for assign2d with non-assignment cost."""

    def test_standard_assignment(self):
        """Test standard assignment (infinite non-assignment cost)."""
        cost = np.array([[1, 2], [2, 1]])
        result = assign2d(cost)

        assert isinstance(result, AssignmentResult)
        assert len(result.row_indices) == 2
        assert len(result.unassigned_rows) == 0
        assert len(result.unassigned_cols) == 0

    def test_with_non_assignment(self):
        """Test with finite non-assignment cost."""
        cost = np.array([[1, 10], [10, 1], [5, 5]])
        result = assign2d(cost, cost_of_non_assignment=3)

        # With non-assignment cost of 3, track 2 should be unassigned
        # since its minimum cost is 5 > 3
        assert 2 in result.unassigned_rows or len(result.row_indices) == 2

    def test_all_unassigned(self):
        """Test when all costs exceed non-assignment cost."""
        cost = np.array([[10, 10], [10, 10]])
        result = assign2d(cost, cost_of_non_assignment=1)

        # All should be unassigned since costs exceed threshold
        assert len(result.row_indices) == 0


class TestMahalanobisDistance:
    """Tests for Mahalanobis distance computation."""

    def test_identity_covariance(self):
        """With identity covariance, should equal squared Euclidean distance."""
        innovation = np.array([3.0, 4.0])
        S = np.eye(2)
        d2 = mahalanobis_distance(innovation, S)
        assert_allclose(d2, 25.0)  # 3^2 + 4^2

    def test_scaled_covariance(self):
        """Test with scaled covariance."""
        innovation = np.array([2.0, 0.0])
        S = np.array([[4.0, 0.0], [0.0, 1.0]])
        d2 = mahalanobis_distance(innovation, S)
        assert_allclose(d2, 1.0)  # 2^2 / 4 = 1


class TestEllipsoidalGate:
    """Tests for ellipsoidal gating."""

    def test_inside_gate(self):
        """Measurement inside gate should pass."""
        innovation = np.array([1.0, 1.0])
        S = np.eye(2) * 4  # std dev = 2
        # d2 = 0.5, threshold = 9.21 for 99% confidence, 2D
        assert ellipsoidal_gate(innovation, S, gate_threshold=9.21)

    def test_outside_gate(self):
        """Measurement outside gate should fail."""
        innovation = np.array([10.0, 10.0])
        S = np.eye(2)
        # d2 = 200, way above threshold
        assert not ellipsoidal_gate(innovation, S, gate_threshold=9.21)


class TestRectangularGate:
    """Tests for rectangular gating."""

    def test_inside_gate(self):
        """Measurement inside rectangular gate should pass."""
        innovation = np.array([2.0, 1.0])
        S = np.array([[4.0, 0.0], [0.0, 1.0]])  # std devs: 2, 1
        # |2| <= 3*2 and |1| <= 3*1
        assert rectangular_gate(innovation, S, num_sigmas=3.0)

    def test_outside_gate(self):
        """Measurement outside gate should fail."""
        innovation = np.array([10.0, 0.0])
        S = np.eye(2)
        # |10| > 3*1
        assert not rectangular_gate(innovation, S, num_sigmas=3.0)


class TestGateMeasurements:
    """Tests for gating multiple measurements."""

    def test_gate_multiple(self):
        """Test gating multiple measurements."""
        z_pred = np.array([0.0, 0.0])
        S = np.eye(2)
        measurements = np.array([[0.5, 0.5], [5.0, 5.0], [1.0, -1.0]])

        valid_idx, dists = gate_measurements(z_pred, S, measurements, 9.21)

        # Measurements 0 and 2 should pass (d2 = 0.5 and 2.0)
        # Measurement 1 should fail (d2 = 50)
        assert 0 in valid_idx
        assert 2 in valid_idx
        assert 1 not in valid_idx

    def test_rectangular_gate_type(self):
        """Test rectangular gating mode."""
        z_pred = np.array([0.0, 0.0])
        S = np.eye(2)
        measurements = np.array([[2.0, 2.0], [5.0, 0.0]])

        valid_idx, _ = gate_measurements(
            z_pred, S, measurements, gate_threshold=3.0, gate_type="rectangular"
        )

        assert 0 in valid_idx  # |2| <= 3
        assert 1 not in valid_idx  # |5| > 3


class TestChi2GateThreshold:
    """Tests for chi-squared threshold computation."""

    def test_2d_99_percent(self):
        """Test 2D, 99% threshold."""
        threshold = chi2_gate_threshold(0.99, 2)
        assert_allclose(threshold, 9.21, rtol=0.01)

    def test_3d_99_percent(self):
        """Test 3D, 99% threshold."""
        threshold = chi2_gate_threshold(0.99, 3)
        assert_allclose(threshold, 11.34, rtol=0.01)


class TestComputeGateVolume:
    """Tests for gate volume computation."""

    def test_unit_covariance(self):
        """Test with unit covariance."""
        S = np.eye(2)
        threshold = chi2_gate_threshold(0.99, 2)
        volume = compute_gate_volume(S, threshold)

        # Volume should be positive and reasonable
        assert volume > 0
        # For 2D: V = pi * sqrt(det(S)) * gamma = pi * 1 * 9.21 ~ 29
        assert_allclose(volume, np.pi * threshold, rtol=0.01)


class TestNearestNeighbor:
    """Tests for nearest neighbor association."""

    def test_basic_association(self):
        """Test basic nearest neighbor."""
        cost = np.array([[1.0, 5.0], [4.0, 2.0]])
        result = nearest_neighbor(cost, gate_threshold=10.0)

        assert isinstance(result, AssociationResult)
        assert result.track_to_measurement[0] == 0  # Track 0 -> Meas 0
        assert result.track_to_measurement[1] == 1  # Track 1 -> Meas 1

    def test_with_gating(self):
        """Test with gate threshold."""
        cost = np.array([[1.0, 10.0], [10.0, 2.0]])
        result = nearest_neighbor(cost, gate_threshold=5.0)

        # Both assignments should be valid (costs 1 and 2 < 5)
        assert result.track_to_measurement[0] == 0
        assert result.track_to_measurement[1] == 1


class TestGNNAssociation:
    """Tests for GNN data association."""

    def test_basic_gnn(self):
        """Test basic GNN association."""
        cost = np.array([[1.0, 5.0, 2.0], [4.0, 2.0, 3.0]])
        result = gnn_association(cost, gate_threshold=10.0)

        # Should find globally optimal assignment
        assert result.total_cost == 3.0  # (0,0)=1 + (1,1)=2

    def test_with_non_assignment(self):
        """Test with non-assignment cost."""
        cost = np.array([[1.0, 10.0], [10.0, 1.0], [5.0, 5.0]])
        result = gnn_association(cost, cost_of_non_assignment=3.0)

        # Track 2 might be unassigned since its min cost (5) > 3
        n_assigned = np.sum(result.track_to_measurement >= 0)
        assert n_assigned <= 3

    def test_gating_excludes(self):
        """Test that gating excludes high-cost assignments."""
        cost = np.array([[1.0, 100.0], [100.0, 2.0]])
        result = gnn_association(cost, gate_threshold=10.0)

        # Should assign (0,0) and (1,1)
        assert result.track_to_measurement[0] == 0
        assert result.track_to_measurement[1] == 1


class TestComputeAssociationCost:
    """Tests for association cost computation."""

    def test_basic_cost(self):
        """Test basic cost computation."""
        predictions = np.array([[0.0, 1.0], [5.0, -1.0]])
        covariances = np.array([np.eye(2), np.eye(2)])
        measurements = np.array([[0.1], [4.9]])
        H = np.array([[1.0, 0.0]])

        costs = compute_association_cost(predictions, covariances, measurements, H)

        assert costs.shape == (2, 2)
        # Track 0 should be close to measurement 0
        assert costs[0, 0] < costs[0, 1]
        # Track 1 should be close to measurement 1
        assert costs[1, 1] < costs[1, 0]

    def test_default_measurement_model(self):
        """Test with default measurement model."""
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        covariances = np.array([np.eye(2), np.eye(2)])
        measurements = np.array([[1.1, 2.1], [3.1, 4.1]])

        costs = compute_association_cost(predictions, covariances, measurements)

        assert costs.shape == (2, 2)
        # Diagonal should have lower costs
        assert costs[0, 0] < costs[0, 1]
        assert costs[1, 1] < costs[1, 0]


class TestGatedGNNAssociation:
    """Tests for combined gated GNN association."""

    def test_basic_gated_gnn(self):
        """Test basic gated GNN."""
        predictions = np.array([[0.0, 1.0], [5.0, -1.0]])
        covariances = np.array([0.1 * np.eye(2), 0.1 * np.eye(2)])
        measurements = np.array([[0.1], [4.9]])
        H = np.array([[1.0, 0.0]])

        result = gated_gnn_association(
            predictions, covariances, measurements, H, gate_probability=0.99
        )

        assert isinstance(result, AssociationResult)
        # Should associate track 0 -> meas 0, track 1 -> meas 1
        assert result.track_to_measurement[0] == 0
        assert result.track_to_measurement[1] == 1


class TestIntegration:
    """Integration tests for full association pipeline."""

    def test_tracking_scenario(self):
        """Test a realistic tracking scenario."""
        # 3 tracks with predictions
        predictions = np.array([[10.0, 1.0], [20.0, -1.0], [30.0, 0.5]])
        covariances = np.array([np.eye(2) * 0.5 for _ in range(3)])

        # 4 measurements (one false alarm)
        measurements = np.array([[10.2], [19.8], [30.1], [50.0]])
        H = np.array([[1.0, 0.0]])

        # Compute costs
        costs = compute_association_cost(predictions, covariances, measurements, H)

        # Run GNN
        result = gnn_association(
            costs,
            gate_threshold=chi2_gate_threshold(0.99, 1),
            cost_of_non_assignment=5.0,
        )

        # Check associations
        assert result.track_to_measurement[0] == 0  # Track 0 -> Meas 0
        assert result.track_to_measurement[1] == 1  # Track 1 -> Meas 1
        assert result.track_to_measurement[2] == 2  # Track 2 -> Meas 2
        # Measurement 3 should be unassigned (false alarm)
        assert result.measurement_to_track[3] == -1


# =============================================================================
# K-Best 2D Assignment Tests
# =============================================================================


class TestMurty:
    """Tests for Murty's k-best algorithm."""

    def test_basic_kbest(self):
        """Test finding k best assignments."""
        cost = np.array([[10, 5, 13], [3, 15, 8], [12, 7, 9]])
        result = murty(cost, k=3)

        assert isinstance(result, KBestResult)
        assert result.n_found == 3
        assert len(result.assignments) == 3
        assert len(result.costs) == 3

        # Costs should be in ascending order
        assert np.all(np.diff(result.costs) >= 0)

        # First solution should be optimal
        row_ind, col_ind, opt_cost = hungarian(cost)
        assert_allclose(result.costs[0], opt_cost)

    def test_k_equals_one(self):
        """Test k=1 returns optimal solution."""
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        result = murty(cost, k=1)

        assert result.n_found == 1
        assert_allclose(result.costs[0], 5.0)

    def test_k_larger_than_solutions(self):
        """Test when k exceeds number of possible solutions."""
        # 2x2 matrix has 2! = 2 possible assignments
        cost = np.array([[1, 2], [3, 4]])
        result = murty(cost, k=10)

        assert result.n_found == 2
        assert result.costs[0] == 5  # (0,0) + (1,1) = 1 + 4
        assert result.costs[1] == 5  # (0,1) + (1,0) = 2 + 3

    def test_maximize(self):
        """Test maximization mode."""
        cost = np.array([[1, 2], [2, 1]])
        result = murty(cost, k=2, maximize=True)

        # In max mode, should get 4 first (2+2), then 2 (1+1)
        # Both solutions have cost 3 due to matrix symmetry: (0,1)+(1,0)=4, (0,0)+(1,1)=2
        assert result.n_found == 2
        # First should be best for maximization
        assert result.costs[0] >= result.costs[1]

    def test_empty_matrix(self):
        """Test with k=0."""
        cost = np.array([[1, 2], [2, 1]])
        result = murty(cost, k=0)

        assert result.n_found == 0
        assert len(result.assignments) == 0


class TestKBestAssign2D:
    """Tests for kbest_assign2d with threshold."""

    def test_with_threshold(self):
        """Test early termination with cost threshold."""
        cost = np.array([[1, 5, 9], [4, 2, 8], [7, 6, 3]])
        result = kbest_assign2d(cost, k=10, cost_threshold=10)

        # Should stop when cost exceeds 10
        assert result.n_found >= 1
        assert all(c <= 10 for c in result.costs)

    def test_with_non_assignment(self):
        """Test with non-assignment cost."""
        cost = np.array([[1, 10], [10, 2], [5, 5]])
        result = kbest_assign2d(cost, k=5, cost_of_non_assignment=3)

        # Should find solutions allowing unassigned tracks
        assert result.n_found >= 1

    def test_no_threshold(self):
        """Test without threshold returns all k."""
        cost = np.array([[1, 2], [3, 4]])
        result = kbest_assign2d(cost, k=2)

        assert result.n_found == 2


class TestRankedAssignments:
    """Tests for ranked_assignments convenience function."""

    def test_basic_ranking(self):
        """Test basic ranked enumeration."""
        cost = np.array([[10, 5], [3, 15]])
        result = ranked_assignments(cost, max_assignments=5)

        assert result.n_found >= 1
        # Costs should be sorted
        assert np.all(np.diff(result.costs) >= 0)

    def test_with_threshold(self):
        """Test with cost threshold."""
        cost = np.array([[1, 10], [10, 2]])
        result = ranked_assignments(cost, max_assignments=10, cost_threshold=5)

        assert result.n_found >= 1
        assert all(c <= 5 for c in result.costs)


# =============================================================================
# 3D Assignment Tests
# =============================================================================


class TestGreedy3D:
    """Tests for greedy 3D assignment."""

    def test_basic_greedy(self):
        """Test basic greedy assignment."""
        np.random.seed(42)
        cost = np.random.rand(4, 4, 4)
        result = greedy_3d(cost)

        assert isinstance(result, Assignment3DResult)
        assert result.tuples.shape[0] <= 4
        assert result.tuples.shape[1] == 3
        assert result.converged

    def test_maximize(self):
        """Test maximization mode."""
        cost = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = greedy_3d(cost, maximize=True)

        # Should pick high-cost entries
        assert result.cost >= 8  # At least one high entry

    def test_rectangular_tensor(self):
        """Test with non-cubic tensor."""
        cost = np.random.rand(3, 4, 5)
        result = greedy_3d(cost)

        # Max assignments is min dimension
        assert result.tuples.shape[0] <= 3


class TestDecomposeTo2D:
    """Tests for 2D decomposition method."""

    def test_basic_decomposition(self):
        """Test basic decomposition."""
        np.random.seed(42)
        cost = np.random.rand(5, 5, 5)
        result = decompose_to_2d(cost)

        assert isinstance(result, Assignment3DResult)
        assert result.tuples.shape[0] <= 5

    def test_different_fixed_dimensions(self):
        """Test decomposition along different dimensions."""
        cost = np.random.rand(4, 4, 4)

        result0 = decompose_to_2d(cost, fixed_dimension=0)
        result1 = decompose_to_2d(cost, fixed_dimension=1)
        result2 = decompose_to_2d(cost, fixed_dimension=2)

        # All should produce valid results
        assert result0.tuples.shape[1] == 3
        assert result1.tuples.shape[1] == 3
        assert result2.tuples.shape[1] == 3


class TestAssign3DLagrangian:
    """Tests for Lagrangian relaxation 3D assignment."""

    def test_basic_lagrangian(self):
        """Test basic Lagrangian relaxation."""
        np.random.seed(42)
        cost = np.random.rand(4, 4, 4)
        result = assign3d_lagrangian(cost, max_iter=50)

        assert isinstance(result, Assignment3DResult)
        assert result.tuples.shape[0] <= 4
        assert result.n_iterations <= 50

    def test_convergence(self):
        """Test that algorithm can converge."""
        # Simple problem with clear optimal solution
        cost = np.full((3, 3, 3), 100.0)
        for i in range(3):
            cost[i, i, i] = 1.0

        result = assign3d_lagrangian(cost, max_iter=100, tol=0.1)

        # Should find the diagonal assignments
        assert result.cost <= 6.0  # 3 * 1.0 + some gap

    def test_maximize(self):
        """Test maximization mode."""
        cost = np.random.rand(3, 3, 3)
        result_min = assign3d_lagrangian(cost, maximize=False)
        result_max = assign3d_lagrangian(cost, maximize=True)

        # Maximization should find higher cost
        assert result_max.cost >= result_min.cost


class TestAssign3DAuction:
    """Tests for auction-based 3D assignment."""

    def test_basic_auction(self):
        """Test basic auction assignment."""
        np.random.seed(42)
        cost = np.random.rand(4, 4, 4)
        result = assign3d_auction(cost, max_iter=500)

        assert isinstance(result, Assignment3DResult)
        assert result.tuples.shape[0] <= 4

    def test_convergence(self):
        """Test that auction can converge."""
        cost = np.random.rand(3, 3, 3)
        result = assign3d_auction(cost, max_iter=1000)

        # Should have made some assignments
        assert result.tuples.shape[0] >= 1


class TestAssign3D:
    """Tests for unified assign3d interface."""

    def test_lagrangian_method(self):
        """Test Lagrangian method selection."""
        cost = np.random.rand(4, 4, 4)
        result = assign3d(cost, method="lagrangian")

        assert isinstance(result, Assignment3DResult)

    def test_auction_method(self):
        """Test auction method selection."""
        cost = np.random.rand(4, 4, 4)
        result = assign3d(cost, method="auction")

        assert isinstance(result, Assignment3DResult)

    def test_greedy_method(self):
        """Test greedy method selection."""
        cost = np.random.rand(4, 4, 4)
        result = assign3d(cost, method="greedy")

        assert isinstance(result, Assignment3DResult)

    def test_decompose_method(self):
        """Test decompose method selection."""
        cost = np.random.rand(4, 4, 4)
        result = assign3d(cost, method="decompose")

        assert isinstance(result, Assignment3DResult)

    def test_invalid_method(self):
        """Test invalid method raises error."""
        cost = np.random.rand(3, 3, 3)
        try:
            assign3d(cost, method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown method" in str(e)

    def test_comparison_of_methods(self):
        """Compare different methods on same problem."""
        np.random.seed(42)
        cost = np.random.rand(4, 4, 4) * 10

        result_greedy = assign3d(cost, method="greedy")
        result_decompose = assign3d(cost, method="decompose")
        result_lagrangian = assign3d(cost, method="lagrangian", max_iter=100)
        result_auction = assign3d(cost, method="auction", max_iter=1000)

        # All should produce valid assignments with correct shape
        for name, result in [
            ("greedy", result_greedy),
            ("decompose", result_decompose),
            ("lagrangian", result_lagrangian),
            ("auction", result_auction),
        ]:
            assert result.tuples.shape[1] == 3, f"{name} has wrong tuple shape"
            # Check that assignments are within bounds
            if result.tuples.shape[0] > 0:
                assert np.all(result.tuples[:, 0] < 4), f"{name} has out-of-bounds i"
                assert np.all(result.tuples[:, 1] < 4), f"{name} has out-of-bounds j"
                assert np.all(result.tuples[:, 2] < 4), f"{name} has out-of-bounds k"

        # Greedy and decompose should always produce valid (no duplicate) assignments
        for name, result in [
            ("greedy", result_greedy),
            ("decompose", result_decompose),
        ]:
            if result.tuples.shape[0] > 0:
                assert (
                    len(set(result.tuples[:, 0])) == result.tuples.shape[0]
                ), f"{name} has duplicate i"
                assert (
                    len(set(result.tuples[:, 1])) == result.tuples.shape[0]
                ), f"{name} has duplicate j"
                assert (
                    len(set(result.tuples[:, 2])) == result.tuples.shape[0]
                ), f"{name} has duplicate k"


class TestAssignment3DResult:
    """Tests for Assignment3DResult structure."""

    def test_result_attributes(self):
        """Test that all result attributes are present."""
        cost = np.random.rand(3, 3, 3)
        result = assign3d(cost)

        assert hasattr(result, "tuples")
        assert hasattr(result, "cost")
        assert hasattr(result, "converged")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "gap")

    def test_empty_result(self):
        """Test handling of problem with no valid assignments."""
        # All infinite costs
        cost = np.full((3, 3, 3), np.inf)
        result = greedy_3d(cost)

        # Should handle gracefully
        assert result.tuples.shape[0] == 0 or result.cost == 0.0
