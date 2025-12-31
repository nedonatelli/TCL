"""Tests for performance evaluation module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.performance_evaluation import (  # Track metrics; Estimation metrics
    ConsistencyResult,
    MOTMetrics,
    OSPAResult,
    average_nees,
    consistency_test,
    credibility_interval,
    estimation_error_bounds,
    identity_switches,
    monte_carlo_rmse,
    mot_metrics,
    nees,
    nees_sequence,
    nis,
    nis_sequence,
    ospa,
    ospa_over_time,
    position_rmse,
    rmse,
    track_fragmentation,
    track_purity,
    velocity_rmse,
)


class TestOSPA:
    """Tests for OSPA metric."""

    def test_ospa_empty_sets(self):
        """Test OSPA with empty sets."""
        result = ospa([], [])
        assert result.ospa == 0.0
        assert result.localization == 0.0
        assert result.cardinality == 0.0

    def test_ospa_one_empty(self):
        """Test OSPA with one empty set."""
        X = [np.array([0, 0])]
        result = ospa(X, [], c=100)
        assert result.ospa == 100.0  # Full cardinality error

        result = ospa([], X, c=100)
        assert result.ospa == 100.0

    def test_ospa_identical_sets(self):
        """Test OSPA with identical sets."""
        X = [np.array([0, 0]), np.array([10, 10])]
        result = ospa(X, X, c=100, p=2)
        assert result.ospa == 0.0
        assert result.localization == 0.0

    def test_ospa_close_points(self):
        """Test OSPA with close points."""
        X = [np.array([0, 0])]
        Y = [np.array([1, 0])]  # Distance = 1
        result = ospa(X, Y, c=100, p=2)
        assert_allclose(result.ospa, 1.0, atol=1e-10)

    def test_ospa_cardinality_mismatch(self):
        """Test OSPA with cardinality mismatch."""
        X = [np.array([0, 0])]
        Y = [np.array([0, 0]), np.array([10, 10])]  # Extra point
        result = ospa(X, Y, c=10, p=2)
        # Should have cardinality component
        assert result.cardinality > 0

    def test_ospa_cutoff(self):
        """Test OSPA cutoff parameter."""
        X = [np.array([0, 0])]
        Y = [np.array([100, 0])]  # Distance = 100
        result_low_c = ospa(X, Y, c=10, p=2)
        result_high_c = ospa(X, Y, c=200, p=2)

        # Low cutoff should cap the distance
        assert_allclose(result_low_c.ospa, 10.0, atol=1e-10)
        assert_allclose(result_high_c.ospa, 100.0, atol=1e-10)

    def test_ospa_symmetry(self):
        """Test OSPA is symmetric."""
        X = [np.array([0, 0]), np.array([5, 5])]
        Y = [np.array([1, 0]), np.array([4, 6])]
        result1 = ospa(X, Y, c=100, p=2)
        result2 = ospa(Y, X, c=100, p=2)
        assert_allclose(result1.ospa, result2.ospa, atol=1e-10)


class TestOSPAOverTime:
    """Tests for OSPA over time."""

    def test_ospa_over_time_basic(self):
        """Test OSPA over time sequence."""
        X_seq = [[np.array([0, 0])], [np.array([1, 1])]]
        Y_seq = [[np.array([0.1, 0])], [np.array([1.1, 1])]]

        ospa_vals = ospa_over_time(X_seq, Y_seq, c=100, p=2)
        assert len(ospa_vals) == 2
        assert_allclose(ospa_vals[0], 0.1, atol=1e-10)
        assert_allclose(ospa_vals[1], 0.1, atol=1e-10)

    def test_ospa_over_time_length_mismatch(self):
        """Test OSPA over time with mismatched lengths."""
        X_seq = [[np.array([0, 0])], [np.array([1, 1])]]
        Y_seq = [[np.array([0, 0])]]

        with pytest.raises(ValueError):
            ospa_over_time(X_seq, Y_seq)


class TestTrackPurity:
    """Tests for track purity metric."""

    def test_perfect_purity(self):
        """Test perfect track purity."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        est_labels = np.array([0, 0, 0, 1, 1, 1])
        assert track_purity(true_labels, est_labels) == 1.0

    def test_mixed_tracks(self):
        """Test mixed tracks."""
        true_labels = np.array([0, 0, 1, 1])
        est_labels = np.array([0, 0, 0, 0])  # All in one track
        # Track 0 has 2 from target 0, 2 from target 1 → max is 2
        assert track_purity(true_labels, est_labels) == 0.5

    def test_empty_labels(self):
        """Test with empty labels."""
        assert track_purity(np.array([]), np.array([])) == 1.0

    def test_mismatched_lengths(self):
        """Test with mismatched lengths."""
        with pytest.raises(ValueError):
            track_purity(np.array([0, 1]), np.array([0]))


class TestTrackFragmentation:
    """Tests for track fragmentation."""

    def test_no_fragmentation(self):
        """Test with no fragmentation."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        est_labels = np.array([0, 0, 0, 1, 1, 1])
        assert track_fragmentation(true_labels, est_labels) == 0

    def test_single_fragmentation(self):
        """Test single fragmentation."""
        true_labels = np.array([0, 0, 0, 0])
        est_labels = np.array([0, 0, 1, 1])  # One switch
        assert track_fragmentation(true_labels, est_labels) == 1

    def test_multiple_fragmentations(self):
        """Test multiple fragmentations."""
        true_labels = np.array([0, 0, 0, 0, 0])
        est_labels = np.array([0, 1, 0, 1, 0])  # Four switches
        assert track_fragmentation(true_labels, est_labels) == 4


class TestIdentitySwitches:
    """Tests for identity switches."""

    def test_no_switches(self):
        """Test with no identity switches."""
        true_labels = np.array([0, 0, 1, 1])
        est_labels = np.array([0, 0, 1, 1])
        assert identity_switches(true_labels, est_labels) == 0

    def test_single_switch(self):
        """Test single identity switch."""
        true_labels = np.array([0, 0, 1, 1])
        est_labels = np.array([0, 0, 0, 0])  # Track 0 switches from target 0 to 1
        assert identity_switches(true_labels, est_labels) == 1


class TestMOTMetrics:
    """Tests for MOT metrics."""

    def test_perfect_tracking(self):
        """Test perfect tracking scenario."""
        gt = [[np.array([0, 0])], [np.array([1, 1])]]
        est = [[np.array([0, 0])], [np.array([1, 1])]]
        result = mot_metrics(gt, est, threshold=1.0)

        assert result.mota == 1.0
        assert result.motp == 0.0
        assert result.num_misses == 0
        assert result.num_false_positives == 0

    def test_all_misses(self):
        """Test all misses scenario."""
        gt = [[np.array([0, 0])], [np.array([1, 1])]]
        est = [[], []]
        result = mot_metrics(gt, est, threshold=1.0)

        # MOTA = 1 - (misses + fp + switches) / total_gt = 1 - 2/2 = 0
        assert result.mota == 0.0
        assert result.num_misses == 2

    def test_all_false_positives(self):
        """Test all false positives scenario."""
        gt = [[], []]
        est = [[np.array([0, 0])], [np.array([1, 1])]]
        result = mot_metrics(gt, est, threshold=1.0)

        assert result.num_false_positives == 2


class TestRMSE:
    """Tests for RMSE metric."""

    def test_rmse_zero_error(self):
        """Test RMSE with zero error."""
        true = np.array([[0, 0], [1, 1], [2, 2]])
        result = rmse(true, true)
        assert result == 0.0

    def test_rmse_known_value(self):
        """Test RMSE with known value."""
        true = np.array([[0], [0], [0], [0]])
        est = np.array([[1], [1], [1], [1]])
        assert rmse(true, est) == 1.0

    def test_rmse_per_component(self):
        """Test RMSE per component."""
        true = np.array([[0, 0], [0, 0]])
        est = np.array([[1, 2], [1, 2]])
        result = rmse(true, est, axis=0)
        assert_allclose(result, [1.0, 2.0])

    def test_position_rmse(self):
        """Test position RMSE."""
        # State = [x, vx, y, vy]
        true = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
        est = np.array([[1, 1, 0, 1], [1, 1, 0, 1]])  # x error = 1 at both times
        result = position_rmse(true, est, [0, 2])
        # Errors: x=[1,1], y=[0,0], RMSE = sqrt(mean([1,0,1,0])) = sqrt(0.5)
        assert_allclose(result, np.sqrt(0.5), atol=1e-10)

    def test_velocity_rmse(self):
        """Test velocity RMSE."""
        # State = [x, vx, y, vy]
        true = np.array([[0, 1, 0, 2], [0, 1, 0, 2]])
        est = np.array([[0, 2, 0, 3], [0, 2, 0, 3]])  # vx error=1, vy error=1
        result = velocity_rmse(true, est, [1, 3])
        assert_allclose(result, 1.0, atol=1e-10)


class TestNEES:
    """Tests for NEES metric."""

    def test_nees_zero_error(self):
        """Test NEES with zero error."""
        true = np.array([1.0, 2.0])
        P = np.eye(2)
        result = nees(true, true, P)
        assert result == 0.0

    def test_nees_known_value(self):
        """Test NEES with known value."""
        true = np.array([0.0, 0.0])
        est = np.array([1.0, 0.0])
        P = np.eye(2) * 0.5  # Variance = 0.5
        # NEES = 1^2 / 0.5 = 2.0
        result = nees(true, est, P)
        assert_allclose(result, 2.0, atol=1e-10)

    def test_nees_sequence(self):
        """Test NEES sequence."""
        true = np.array([[0, 0], [0, 0]])
        est = np.array([[1, 0], [0, 1]])
        P = np.array([np.eye(2), np.eye(2)])

        result = nees_sequence(true, est, P)
        assert_allclose(result, [1.0, 1.0], atol=1e-10)

    def test_average_nees(self):
        """Test average NEES."""
        true = np.array([[0, 0], [0, 0]])
        est = np.array([[1, 0], [0, 1]])
        P = np.array([np.eye(2), np.eye(2)])

        result = average_nees(true, est, P)
        assert_allclose(result, 1.0, atol=1e-10)


class TestNIS:
    """Tests for NIS metric."""

    def test_nis_zero_innovation(self):
        """Test NIS with zero innovation."""
        innovation = np.array([0.0, 0.0])
        S = np.eye(2)
        result = nis(innovation, S)
        assert result == 0.0

    def test_nis_known_value(self):
        """Test NIS with known value."""
        innovation = np.array([1.0, 0.0])
        S = np.eye(2) * 2.0  # Variance = 2
        # NIS = 1^2 / 2 = 0.5
        result = nis(innovation, S)
        assert_allclose(result, 0.5, atol=1e-10)

    def test_nis_sequence(self):
        """Test NIS sequence."""
        innovations = np.array([[1, 0], [0, 1]])
        S = np.array([np.eye(2), np.eye(2)])

        result = nis_sequence(innovations, S)
        assert_allclose(result, [1.0, 1.0], atol=1e-10)


class TestConsistencyTest:
    """Tests for consistency test."""

    def test_consistent_filter(self):
        """Test with consistent filter (chi-squared samples)."""
        np.random.seed(42)
        # Generate chi-squared samples with df=4
        nees_vals = np.random.chisquare(df=4, size=100)
        result = consistency_test(nees_vals, df=4, confidence=0.95)

        # Should be consistent most of the time
        assert isinstance(result, ConsistencyResult)
        assert result.lower_bound < result.upper_bound

    def test_inconsistent_filter_high_nees(self):
        """Test with inconsistent filter (high NEES)."""
        # All NEES values much higher than expected
        nees_vals = np.ones(100) * 100  # Way too high for df=4
        result = consistency_test(nees_vals, df=4, confidence=0.95)

        assert not result.is_consistent
        assert result.mean_value > result.upper_bound


class TestCredibilityInterval:
    """Tests for credibility interval."""

    def test_perfect_consistency(self):
        """Test with errors well within bounds."""
        errors = np.array([[0.1, 0.1], [0.1, 0.1]])
        P = np.array([np.eye(2) * 10, np.eye(2) * 10])  # Large covariance

        fraction = credibility_interval(errors, P, interval=0.95)
        assert fraction == 1.0  # All within bounds


class TestMonteCarloRMSE:
    """Tests for Monte Carlo RMSE."""

    def test_monte_carlo_rmse_basic(self):
        """Test Monte Carlo RMSE."""
        # 10 runs, 5 time steps, 2 state dims
        errors = np.random.randn(10, 5, 2)
        result = monte_carlo_rmse(errors, axis=0)
        assert result.shape == (5, 2)


class TestEstimationErrorBounds:
    """Tests for estimation error bounds."""

    def test_error_bounds_basic(self):
        """Test error bounds computation."""
        # 3 time steps, 2x2 covariance
        P = np.array([np.eye(2) * 1, np.eye(2) * 4, np.eye(2) * 9])
        bounds = estimation_error_bounds(P, sigma=1.0)

        assert bounds.shape == (3, 2)
        assert_allclose(bounds[0], [1, 1])
        assert_allclose(bounds[1], [2, 2])
        assert_allclose(bounds[2], [3, 3])

    def test_error_bounds_two_sigma(self):
        """Test 2-sigma error bounds."""
        P = np.array([np.eye(2)])
        bounds = estimation_error_bounds(P, sigma=2.0)
        assert_allclose(bounds[0], [2, 2])


class TestNamedTuples:
    """Tests for named tuple structures."""

    def test_ospa_result(self):
        """Test OSPAResult creation."""
        result = OSPAResult(ospa=1.0, localization=0.5, cardinality=0.5)
        assert result.ospa == 1.0
        assert result.localization == 0.5
        assert result.cardinality == 0.5

    def test_mot_metrics(self):
        """Test MOTMetrics creation."""
        result = MOTMetrics(
            mota=0.9,
            motp=0.5,
            num_switches=2,
            num_fragmentations=1,
            num_false_positives=5,
            num_misses=3,
        )
        assert result.mota == 0.9
        assert result.motp == 0.5

    def test_consistency_result(self):
        """Test ConsistencyResult creation."""
        result = ConsistencyResult(
            is_consistent=True,
            statistic=4.0,
            lower_bound=2.0,
            upper_bound=6.0,
            mean_value=4.0,
        )
        assert result.is_consistent
        assert result.mean_value == 4.0
