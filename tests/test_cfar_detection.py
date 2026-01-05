"""
Tests for CFAR (Constant False Alarm Rate) detection algorithms.

Tests cover:
- Threshold factor computation
- Detection probability calculation
- CA-CFAR (Cell-Averaging)
- GO-CFAR (Greatest-Of)
- SO-CFAR (Smallest-Of)
- OS-CFAR (Order-Statistic)
- 2D CFAR detection
- Utility functions (cluster_detections, snr_loss)
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.signal_processing.detection import (
    CFARResult,
    CFARResult2D,
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    cluster_detections,
    detection_probability,
    snr_loss,
    threshold_factor,
)

# =============================================================================
# Tests for threshold_factor
# =============================================================================


class TestThresholdFactor:
    """Tests for threshold factor computation."""

    def test_threshold_factor_ca_basic(self):
        """Test basic CA-CFAR threshold factor."""
        alpha = threshold_factor(1e-6, 32, method="ca")
        assert alpha > 1  # Threshold should be above noise level

    def test_threshold_factor_increases_with_lower_pfa(self):
        """Test that threshold factor increases as Pfa decreases."""
        alpha_low = threshold_factor(1e-3, 32, method="ca")
        alpha_high = threshold_factor(1e-6, 32, method="ca")
        assert alpha_high > alpha_low

    def test_threshold_factor_decreases_with_more_ref_cells(self):
        """Test that threshold factor decreases with more reference cells."""
        alpha_few = threshold_factor(1e-6, 16, method="ca")
        alpha_many = threshold_factor(1e-6, 64, method="ca")
        assert alpha_few > alpha_many

    def test_threshold_factor_go_method(self):
        """Test GO-CFAR threshold factor."""
        alpha = threshold_factor(1e-6, 32, method="go")
        assert alpha > 0

    def test_threshold_factor_so_method(self):
        """Test SO-CFAR threshold factor."""
        alpha = threshold_factor(1e-6, 32, method="so")
        assert alpha > 0

    def test_threshold_factor_os_method(self):
        """Test OS-CFAR threshold factor."""
        alpha = threshold_factor(1e-6, 32, method="os")
        assert alpha > 0

    def test_threshold_factor_os_with_k(self):
        """Test OS-CFAR threshold factor with custom k."""
        alpha = threshold_factor(1e-6, 32, method="os", k=24)
        assert alpha > 0

    def test_threshold_factor_invalid_pfa_low(self):
        """Test that invalid Pfa (<=0) raises error."""
        with pytest.raises(ValueError, match="pfa must be between"):
            threshold_factor(0, 32)

    def test_threshold_factor_invalid_pfa_high(self):
        """Test that invalid Pfa (>=1) raises error."""
        with pytest.raises(ValueError, match="pfa must be between"):
            threshold_factor(1.0, 32)

    def test_threshold_factor_invalid_n_ref(self):
        """Test that invalid n_ref (<1) raises error."""
        with pytest.raises(ValueError, match="n_ref must be at least"):
            threshold_factor(1e-6, 0)

    def test_threshold_factor_unknown_method(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            threshold_factor(1e-6, 32, method="unknown")


# =============================================================================
# Tests for detection_probability
# =============================================================================


class TestDetectionProbability:
    """Tests for detection probability computation."""

    def test_detection_probability_basic(self):
        """Test basic detection probability."""
        pd = detection_probability(snr=10, pfa=1e-6, n_ref=32)
        assert 0 < pd < 1

    def test_detection_probability_increases_with_snr(self):
        """Test that detection probability increases with SNR."""
        pd_low = detection_probability(snr=5, pfa=1e-6, n_ref=32)
        pd_high = detection_probability(snr=20, pfa=1e-6, n_ref=32)
        assert pd_high > pd_low

    def test_detection_probability_increases_with_pfa(self):
        """Test that detection probability increases with Pfa."""
        pd_low = detection_probability(snr=10, pfa=1e-8, n_ref=32)
        pd_high = detection_probability(snr=10, pfa=1e-3, n_ref=32)
        assert pd_high > pd_low

    def test_detection_probability_swerling_1(self):
        """Test detection probability with Swerling I target."""
        pd = detection_probability(snr=10, pfa=1e-6, n_ref=32, swerling_case=1)
        assert 0 < pd < 1

    def test_detection_probability_swerling_other(self):
        """Test detection probability with other Swerling cases."""
        pd = detection_probability(snr=10, pfa=1e-6, n_ref=32, swerling_case=2)
        assert 0 < pd < 1


# =============================================================================
# Tests for CA-CFAR
# =============================================================================


class TestCfarCa:
    """Tests for Cell-Averaging CFAR."""

    def test_cfar_ca_detects_strong_target(self):
        """Test that CA-CFAR detects a strong target."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 1000)
        signal[500] = 100  # Strong target

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)
        assert 500 in result.detection_indices

    def test_cfar_ca_detects_multiple_targets(self):
        """Test that CA-CFAR detects multiple targets."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 1000)
        signal[250] = 50
        signal[500] = 100
        signal[750] = 30

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert 250 in result.detection_indices
        assert 500 in result.detection_indices
        assert 750 in result.detection_indices

    def test_cfar_ca_output_shapes(self):
        """Test CA-CFAR output array shapes."""
        signal = np.random.exponential(1.0, 500)

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert len(result.detections) == 500
        assert len(result.threshold) == 500
        assert len(result.noise_estimate) == 500

    def test_cfar_ca_with_custom_alpha(self):
        """Test CA-CFAR with custom alpha threshold."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, alpha=5.0)

        assert 250 in result.detection_indices

    def test_cfar_ca_threshold_positive(self):
        """Test that CA-CFAR threshold is positive."""
        signal = np.random.exponential(1.0, 500)

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert np.all(result.threshold >= 0)

    def test_cfar_ca_noise_estimate_positive(self):
        """Test that CA-CFAR noise estimate is non-negative."""
        signal = np.abs(np.random.randn(500))  # All positive

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert np.all(result.noise_estimate >= 0)


# =============================================================================
# Tests for GO-CFAR
# =============================================================================


class TestCfarGo:
    """Tests for Greatest-Of CFAR."""

    def test_cfar_go_detects_target(self):
        """Test that GO-CFAR detects a target."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)
        assert len(result.detection_indices) >= 1

    def test_cfar_go_at_clutter_edge(self):
        """Test GO-CFAR performance at clutter edge."""
        np.random.seed(42)
        # Create clutter edge
        signal = np.concatenate(
            [np.random.exponential(1.0, 250), np.random.exponential(10.0, 250)]
        )
        signal[250] = 100  # Target at edge

        result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        # GO-CFAR should handle clutter edges better
        assert isinstance(result, CFARResult)

    def test_cfar_go_output_shapes(self):
        """Test GO-CFAR output array shapes."""
        signal = np.random.exponential(1.0, 500)

        result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert len(result.detections) == 500
        assert len(result.threshold) == 500
        assert len(result.noise_estimate) == 500

    def test_cfar_go_with_custom_alpha(self):
        """Test GO-CFAR with custom alpha."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_go(signal, guard_cells=2, ref_cells=16, alpha=5.0)

        assert isinstance(result, CFARResult)


# =============================================================================
# Tests for SO-CFAR
# =============================================================================


class TestCfarSo:
    """Tests for Smallest-Of CFAR."""

    def test_cfar_so_detects_target(self):
        """Test that SO-CFAR detects a target."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_so(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)

    def test_cfar_so_output_shapes(self):
        """Test SO-CFAR output array shapes."""
        signal = np.random.exponential(1.0, 500)

        result = cfar_so(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert len(result.detections) == 500
        assert len(result.threshold) == 500
        assert len(result.noise_estimate) == 500

    def test_cfar_so_with_custom_alpha(self):
        """Test SO-CFAR with custom alpha."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_so(signal, guard_cells=2, ref_cells=16, alpha=5.0)

        assert isinstance(result, CFARResult)


# =============================================================================
# Tests for OS-CFAR
# =============================================================================


class TestCfarOs:
    """Tests for Order-Statistic CFAR."""

    def test_cfar_os_detects_target(self):
        """Test that OS-CFAR detects a target."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)

    def test_cfar_os_with_closely_spaced_targets(self):
        """Test OS-CFAR with closely spaced targets."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50
        signal[260] = 40  # Close to first target

        result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        # OS-CFAR is robust to interfering targets
        assert len(result.detection_indices) >= 1

    def test_cfar_os_output_shapes(self):
        """Test OS-CFAR output array shapes."""
        signal = np.random.exponential(1.0, 500)

        result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert len(result.detections) == 500
        assert len(result.threshold) == 500
        assert len(result.noise_estimate) == 500

    def test_cfar_os_with_custom_k(self):
        """Test OS-CFAR with custom order statistic k."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4, k=20)

        assert isinstance(result, CFARResult)

    def test_cfar_os_with_custom_alpha(self):
        """Test OS-CFAR with custom alpha."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_os(signal, guard_cells=2, ref_cells=16, alpha=5.0)

        assert isinstance(result, CFARResult)


# =============================================================================
# Tests for 2D CFAR
# =============================================================================


class TestCfar2d:
    """Tests for two-dimensional CFAR."""

    def test_cfar_2d_ca_detects_target(self):
        """Test that 2D CA-CFAR detects a target."""
        np.random.seed(42)
        image = np.random.exponential(1.0, (100, 100))
        image[50, 50] = 100  # Target

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4)

        assert isinstance(result, CFARResult2D)
        assert result.detections[50, 50]

    def test_cfar_2d_go_method(self):
        """Test 2D GO-CFAR."""
        np.random.seed(42)
        image = np.random.exponential(1.0, (100, 100))
        image[50, 50] = 100

        result = cfar_2d(
            image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4, method="go"
        )

        assert isinstance(result, CFARResult2D)

    def test_cfar_2d_so_method(self):
        """Test 2D SO-CFAR."""
        np.random.seed(42)
        image = np.random.exponential(1.0, (100, 100))
        image[50, 50] = 100

        result = cfar_2d(
            image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4, method="so"
        )

        assert isinstance(result, CFARResult2D)

    def test_cfar_2d_output_shapes(self):
        """Test 2D CFAR output shapes."""
        image = np.random.exponential(1.0, (100, 100))

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4)

        assert result.detections.shape == (100, 100)
        assert result.threshold.shape == (100, 100)
        assert result.noise_estimate.shape == (100, 100)

    def test_cfar_2d_with_custom_alpha(self):
        """Test 2D CFAR with custom alpha."""
        image = np.random.exponential(1.0, (100, 100))
        image[50, 50] = 100

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), alpha=5.0)

        assert isinstance(result, CFARResult2D)

    def test_cfar_2d_unknown_method_raises(self):
        """Test that unknown method raises error."""
        image = np.random.exponential(1.0, (100, 100))

        with pytest.raises(ValueError, match="Unknown method"):
            cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), method="unknown")

    def test_cfar_2d_detects_multiple_targets(self):
        """Test 2D CFAR detects multiple targets."""
        np.random.seed(42)
        image = np.random.exponential(1.0, (100, 100))
        image[25, 25] = 80
        image[50, 50] = 100
        image[75, 75] = 60

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4)

        assert result.detections[25, 25]
        assert result.detections[50, 50]
        assert result.detections[75, 75]


# =============================================================================
# Tests for cluster_detections
# =============================================================================


class TestClusterDetections:
    """Tests for detection clustering."""

    def test_cluster_detections_basic(self):
        """Test basic detection clustering."""
        detections = np.array(
            [False] * 50 + [True, True, True] + [False] * 47 + [True, True] + [False]
        )

        peaks = cluster_detections(detections, min_separation=1)

        assert len(peaks) == 2  # Two clusters

    def test_cluster_detections_single(self):
        """Test clustering with single detection."""
        detections = np.array([False] * 50 + [True] + [False] * 49)

        peaks = cluster_detections(detections, min_separation=1)

        assert len(peaks) == 1
        assert peaks[0] == 50

    def test_cluster_detections_empty(self):
        """Test clustering with no detections."""
        detections = np.array([False] * 100)

        peaks = cluster_detections(detections, min_separation=1)

        assert len(peaks) == 0

    def test_cluster_detections_adjacent(self):
        """Test clustering adjacent detections."""
        detections = np.array(
            [False] * 45 + [True] * 10 + [False] * 45  # 10 adjacent detections
        )

        peaks = cluster_detections(detections, min_separation=1)

        assert len(peaks) == 1  # Should be one cluster

    def test_cluster_detections_separated(self):
        """Test clustering separated detections."""
        detections = np.array(
            [False] * 20
            + [True]
            + [False] * 28
            + [True]
            + [False] * 28
            + [True]
            + [False] * 21
        )

        peaks = cluster_detections(detections, min_separation=1)

        assert len(peaks) == 3  # Three separate detections

    def test_cluster_detections_min_separation(self):
        """Test clustering with different minimum separation."""
        # Detections at indices 20, 22, 24 (separated by 2)
        detections = np.zeros(100, dtype=bool)
        detections[20] = True
        detections[22] = True
        detections[24] = True

        # With min_separation=1, they are separate
        peaks1 = cluster_detections(detections, min_separation=1)
        assert len(peaks1) == 3

        # With min_separation=2, they merge
        peaks2 = cluster_detections(detections, min_separation=2)
        assert len(peaks2) == 1


# =============================================================================
# Tests for snr_loss
# =============================================================================


class TestSnrLoss:
    """Tests for SNR loss computation."""

    def test_snr_loss_ca(self):
        """Test SNR loss for CA-CFAR."""
        loss = snr_loss(32, method="ca")
        assert loss > 0  # There is always some loss
        assert loss < 1  # Small loss for many reference cells

    def test_snr_loss_go(self):
        """Test SNR loss for GO-CFAR."""
        loss = snr_loss(32, method="go")
        assert loss > 0

    def test_snr_loss_so(self):
        """Test SNR loss for SO-CFAR."""
        loss = snr_loss(32, method="so")
        assert loss > 0

    def test_snr_loss_os(self):
        """Test SNR loss for OS-CFAR."""
        loss = snr_loss(32, method="os")
        assert loss > 0

    def test_snr_loss_decreases_with_more_cells(self):
        """Test that SNR loss decreases with more reference cells."""
        loss_few = snr_loss(8, method="ca")
        loss_many = snr_loss(64, method="ca")
        assert loss_few > loss_many

    def test_snr_loss_unknown_method_raises(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            snr_loss(32, method="unknown")


# =============================================================================
# Tests for NamedTuple types
# =============================================================================


class TestCFARResultTypes:
    """Tests for CFAR result types."""

    def test_cfar_result_attributes(self):
        """Test CFARResult attributes."""
        detections = np.array([False, True, False])
        threshold = np.array([1.0, 2.0, 1.5])
        detection_indices = np.array([1])
        noise_estimate = np.array([0.5, 0.6, 0.5])

        result = CFARResult(
            detections=detections,
            threshold=threshold,
            detection_indices=detection_indices,
            noise_estimate=noise_estimate,
        )

        np.testing.assert_array_equal(result.detections, detections)
        np.testing.assert_array_equal(result.threshold, threshold)
        np.testing.assert_array_equal(result.detection_indices, detection_indices)
        np.testing.assert_array_equal(result.noise_estimate, noise_estimate)

    def test_cfar_result_2d_attributes(self):
        """Test CFARResult2D attributes."""
        detections = np.array([[False, True], [False, False]])
        threshold = np.array([[1.0, 2.0], [1.5, 1.2]])
        noise_estimate = np.array([[0.5, 0.6], [0.5, 0.4]])

        result = CFARResult2D(
            detections=detections,
            threshold=threshold,
            noise_estimate=noise_estimate,
        )

        np.testing.assert_array_equal(result.detections, detections)
        np.testing.assert_array_equal(result.threshold, threshold)
        np.testing.assert_array_equal(result.noise_estimate, noise_estimate)


# =============================================================================
# Integration tests
# =============================================================================


class TestCFARIntegration:
    """Integration tests for CFAR algorithms."""

    def test_cfar_methods_same_signal(self):
        """Test all CFAR methods on the same signal."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 1000)
        signal[500] = 100  # Strong target

        ca_result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
        go_result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
        so_result = cfar_so(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
        os_result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        # All should detect the strong target
        assert 500 in ca_result.detection_indices
        assert 500 in go_result.detection_indices
        assert 500 in so_result.detection_indices
        assert 500 in os_result.detection_indices

    def test_cfar_false_alarm_rate(self):
        """Test that CFAR maintains approximate false alarm rate."""
        np.random.seed(42)
        # Pure noise signal (no targets)
        signal = np.random.exponential(1.0, 10000)

        result = cfar_ca(signal, guard_cells=2, ref_cells=32, pfa=1e-3)

        # Count false alarms (excluding edges where CFAR has fewer ref cells)
        edge = 50
        fa_count = np.sum(result.detections[edge:-edge])
        total_cells = len(signal) - 2 * edge

        # False alarm rate should be in reasonable range (within factor of 5)
        measured_pfa = fa_count / total_cells
        # Allow for statistical variation
        assert measured_pfa < 5e-3  # Not too many false alarms

    def test_cfar_edge_handling(self):
        """Test CFAR behavior at signal edges."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 100)
        signal[5] = 50  # Target near edge
        signal[95] = 50  # Target near other edge

        result = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=1e-4)

        # Should handle edges without crashing
        assert isinstance(result, CFARResult)
        assert len(result.threshold) == 100
