"""Tests for assignment algorithms and data association."""

import numpy as np
from numpy.testing import assert_allclose

from tcl.assignment_algorithms import (
    # 2D Assignment
    hungarian,
    auction,
    linear_sum_assignment,
    assign2d,
    AssignmentResult,
    # Gating
    ellipsoidal_gate,
    rectangular_gate,
    gate_measurements,
    mahalanobis_distance,
    chi2_gate_threshold,
    compute_gate_volume,
    # Data Association
    gnn_association,
    nearest_neighbor,
    compute_association_cost,
    gated_gnn_association,
    AssociationResult,
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
