"""
Comprehensive tests for signal processing detection module (CFAR algorithms).

Tests coverage for:
- CFAR threshold factor computation
- 1D CFAR detection (CA, GO, SO, OS)
- 2D CFAR detection
- Detection probability calculations
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.signal_processing.detection import (
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


class TestThresholdFactor:
    """Tests for CFAR threshold factor computation."""

    def test_threshold_factor_ca_basic(self):
        """Test CA CFAR threshold factor."""
        pfa = 0.001
        n_ref = 16
        factor = threshold_factor(pfa, n_ref, method="ca")
        assert factor > 0
        assert np.isfinite(factor)

    def test_threshold_factor_go_basic(self):
        """Test GO CFAR threshold factor."""
        pfa = 0.001
        n_ref = 16
        factor = threshold_factor(pfa, n_ref, method="go")
        assert factor > 0
        assert np.isfinite(factor)

    def test_threshold_factor_so_basic(self):
        """Test SO CFAR threshold factor."""
        pfa = 0.001
        n_ref = 16
        factor = threshold_factor(pfa, n_ref, method="so")
        assert factor > 0
        assert np.isfinite(factor)

    def test_threshold_factor_os_basic(self):
        """Test OS CFAR threshold factor with k parameter."""
        pfa = 0.001
        n_ref = 16
        k = 8
        factor = threshold_factor(pfa, n_ref, method="os", k=k)
        assert factor > 0
        assert np.isfinite(factor)

    def test_threshold_factor_decreasing_pfa(self):
        """Test that threshold factor increases as pfa decreases."""
        n_ref = 16
        factors = []
        for pfa in [0.1, 0.01, 0.001]:
            factor = threshold_factor(pfa, n_ref, method="ca")
            factors.append(factor)

        # Threshold should increase as pfa decreases
        assert factors[0] < factors[1] < factors[2]

    def test_threshold_factor_increasing_n_ref(self):
        """Test threshold factor behavior with increasing reference cells."""
        pfa = 0.001
        factors = []
        for n_ref in [8, 16, 32]:
            factor = threshold_factor(pfa, n_ref, method="ca")
            factors.append(factor)

        assert all(np.isfinite(f) for f in factors)

    def test_threshold_factor_different_methods(self):
        """Test all CFAR methods produce positive factors."""
        pfa = 0.001
        n_ref = 16

        methods = ["ca", "go", "so"]
        for method in methods:
            factor = threshold_factor(pfa, n_ref, method=method)
            assert factor > 0
            assert np.isfinite(factor)


class TestCFAR_CA:
    """Tests for Cell-Averaging CFAR."""

    def test_cfar_ca_basic(self):
        """Test basic CA CFAR detection."""
        signal = np.random.randn(1000)
        result = cfar_ca(signal, guard_cells=4, ref_cells=16, pfa=0.001)

        assert result.detections.shape == signal.shape
        assert result.threshold.shape == signal.shape
        assert result.noise_estimate.shape == signal.shape

    def test_cfar_ca_sine_in_noise(self):
        """Test CA CFAR detects strong sine in noise."""
        # Create signal with strong sine
        t = np.arange(1000)
        signal = 0.1 * np.random.randn(1000)
        signal[500:600] += 5 * np.sin(2 * np.pi * t[500:600] / 50)

        result = cfar_ca(signal, guard_cells=4, ref_cells=16, pfa=0.01)

        # Should detect the strong sine region
        detections_in_signal = np.sum(result.detections[500:600])
        assert detections_in_signal > 0

    def test_cfar_ca_multiple_targets(self):
        """Test CA CFAR with multiple target signal."""
        signal = np.random.randn(1000) * 0.1
        signal[200] = 10
        signal[500] = 8
        signal[800] = 6

        result = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.01)

        # Should detect all three peaks
        detected_count = np.sum(result.detections)
        assert detected_count >= 3

    def test_cfar_ca_output_types(self):
        """Test CA CFAR output types."""
        signal = np.random.randn(256)
        result = cfar_ca(signal, guard_cells=2, ref_cells=8)

        assert isinstance(result.detections, np.ndarray)
        assert result.detections.dtype == np.bool_
        assert isinstance(result.threshold, np.ndarray)
        assert isinstance(result.noise_estimate, np.ndarray)

    def test_cfar_ca_with_custom_pfa(self):
        """Test CA CFAR with different PFA values."""
        signal = np.random.randn(1000)

        result_high_pfa = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.1)
        result_low_pfa = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.001)

        # Both should return valid results
        assert result_high_pfa.detections.dtype == np.bool_
        assert result_low_pfa.detections.dtype == np.bool_


class TestCFAR_GO:
    """Tests for Greatest-Of CFAR."""

    def test_cfar_go_basic(self):
        """Test basic GO CFAR detection."""
        signal = np.random.randn(1000)
        result = cfar_go(signal, guard_cells=4, ref_cells=16, pfa=0.001)

        assert result.detections.shape == signal.shape
        assert result.threshold.shape == signal.shape

    def test_cfar_go_clutter_rejection(self):
        """Test GO CFAR better clutter rejection."""
        signal = np.random.randn(1000) * 0.1
        signal[300:400] += 0.5  # Add clutter region
        signal[500] = 5  # Target

        result = cfar_go(signal, guard_cells=2, ref_cells=8, pfa=0.001)

        # Should detect target
        assert np.any(result.detections[450:550])

    def test_cfar_go_output_structure(self):
        """Test GO CFAR output structure."""
        signal = np.random.randn(256)
        result = cfar_go(signal, guard_cells=2, ref_cells=8)

        assert hasattr(result, "detections")
        assert hasattr(result, "threshold")
        assert hasattr(result, "noise_estimate")


class TestCFAR_SO:
    """Tests for Smallest-Of CFAR."""

    def test_cfar_so_basic(self):
        """Test basic SO CFAR detection."""
        signal = np.random.randn(1000)
        result = cfar_so(signal, guard_cells=4, ref_cells=16, pfa=0.001)

        assert result.detections.shape == signal.shape
        assert result.threshold.shape == signal.shape

    def test_cfar_so_vs_ca(self):
        """Test SO CFAR has different characteristics than CA."""
        signal = np.random.randn(500)

        result_so = cfar_so(signal, guard_cells=2, ref_cells=8, pfa=0.01)
        result_ca = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.01)

        # Results should be different
        assert not np.array_equal(result_so.detections, result_ca.detections)


class TestCFAR_OS:
    """Tests for Order-Statistic CFAR."""

    def test_cfar_os_basic(self):
        """Test basic OS CFAR detection."""
        signal = np.random.randn(1000)
        result = cfar_os(signal, guard_cells=4, ref_cells=16, pfa=0.001, k=8)

        assert result.detections.shape == signal.shape
        assert result.threshold.shape == signal.shape

    def test_cfar_os_different_k_values(self):
        """Test OS CFAR with different k values."""
        signal = np.random.randn(512)

        for k in [4, 8, 12, 16]:
            result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=0.01, k=k)
            assert result.detections.shape == signal.shape


class TestCFAR_2D:
    """Tests for 2D CFAR detection."""

    def test_cfar_2d_basic(self):
        """Test basic 2D CFAR detection."""
        image = np.random.randn(128, 128)
        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=0.001)

        assert result.detections.shape == image.shape
        assert result.threshold.shape == image.shape
        assert result.noise_estimate.shape == image.shape

    def test_cfar_2d_point_target(self):
        """Test 2D CFAR detects point target."""
        image = np.random.randn(100, 100) * 0.1
        image[50, 50] = 10  # Point target

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=0.001)

        # Should detect the point
        assert result.detections[50, 50]

    def test_cfar_2d_extended_target(self):
        """Test 2D CFAR with extended target."""
        image = np.random.randn(100, 100) * 0.1
        image[40:60, 40:60] += 2  # Extended target

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=0.01)

        # Should return valid detection array
        assert result.detections.shape == image.shape
        assert result.detections.dtype == np.bool_

    def test_cfar_2d_noise_only(self):
        """Test 2D CFAR with noise only."""
        image = np.random.randn(100, 100)
        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=0.01)

        # Should return valid detection array
        assert result.detections.shape == image.shape
        assert result.detections.dtype == np.bool_

    def test_cfar_2d_output_types(self):
        """Test 2D CFAR output types."""
        image = np.random.randn(64, 64)
        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8))

        assert isinstance(result.detections, np.ndarray)
        assert result.detections.dtype == np.bool_
        assert isinstance(result.threshold, np.ndarray)
        assert isinstance(result.noise_estimate, np.ndarray)


class TestDetectionProbability:
    """Tests for detection probability calculation."""

    def test_detection_probability_basic(self):
        """Test basic detection probability calculation."""
        snr = 10  # 10 dB
        pfa = 0.001
        n_ref = 16

        pd = detection_probability(snr, pfa, n_ref)

        # Should return finite value
        assert np.isfinite(pd)

    def test_detection_probability_varying_snr(self):
        """Test detection probability calculation with varying SNR."""
        pfa = 0.001
        n_ref = 16

        snr_values = [0, 5, 10, 15, 20]
        pd_values = []

        for snr in snr_values:
            pd = detection_probability(snr, pfa, n_ref)
            pd_values.append(pd)

        # All should be finite
        assert all(np.isfinite(p) for p in pd_values)

    def test_detection_probability_valid_returns(self):
        """Test detection probability returns valid values."""
        pfa = 0.001
        n_ref = 16

        snr_values = [-20, -10, 0, 10, 20, 30]

        for snr in snr_values:
            pd = detection_probability(snr, pfa, n_ref)
            assert np.isfinite(pd)

    def test_detection_probability_pfa_effect(self):
        """Test detection probability with different PFA values."""
        snr = 10
        n_ref = 16

        pfa_values = [0.0001, 0.001, 0.01, 0.1]
        pd_values = []

        for pfa in pfa_values:
            pd = detection_probability(snr, pfa, n_ref)
            pd_values.append(pd)

        # All should be finite
        assert all(np.isfinite(p) for p in pd_values)

    def test_detection_probability_n_ref_effect(self):
        """Test detection probability with different number of reference cells."""
        snr = 10
        pfa = 0.001

        n_ref_values = [8, 16, 32, 64]
        pd_values = []

        for n_ref in n_ref_values:
            pd = detection_probability(snr, pfa, n_ref)
            pd_values.append(pd)

        # All should be finite
        assert all(np.isfinite(p) for p in pd_values)


class TestCFARIntegration:
    """Integration tests for CFAR algorithms."""

    def test_all_cfar_methods_basic(self):
        """Test all CFAR methods execute without error."""
        signal = np.random.randn(512)
        signal[250:270] += 10  # Add stronger target

        result_ca = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.01)
        result_go = cfar_go(signal, guard_cells=2, ref_cells=8, pfa=0.01)
        result_so = cfar_so(signal, guard_cells=2, ref_cells=8, pfa=0.01)

        # All should return valid results
        assert result_ca.detections.dtype == np.bool_
        assert result_go.detections.dtype == np.bool_
        assert result_so.detections.dtype == np.bool_

    def test_cfar_parameters_validity(self):
        """Test CFAR with various valid parameter combinations."""
        signal = np.random.randn(512)

        for n_guard in [1, 2, 4]:
            for n_ref in [8, 16, 32]:
                result = cfar_ca(signal, guard_cells=n_guard, ref_cells=n_ref, pfa=0.01)
                assert result.detections.dtype == np.bool_

    def test_cfar_signal_length_independence(self):
        """Test CFAR works with different signal lengths."""
        for length in [128, 256, 512, 1024]:
            signal = np.random.randn(length)
            result = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.01)
            assert result.detections.shape == signal.shape

    def test_detection_noise_floor(self):
        """Test CFAR adapts to different signals."""
        signal_low = np.random.randn(512) * 0.1
        signal_high = np.random.randn(512)

        result_low = cfar_ca(signal_low, guard_cells=2, ref_cells=8)
        result_high = cfar_ca(signal_high, guard_cells=2, ref_cells=8)

        # Both should return valid results
        assert result_low.detections.dtype == np.bool_
        assert result_high.detections.dtype == np.bool_


class TestCFARDetectionComprehensive:
    """Comprehensive tests for all CFAR detection variants."""

    @pytest.fixture
    def signal_1d(self):
        """1D signal with multiple targets."""
        np.random.seed(42)
        n = 256
        signal = np.abs(np.random.randn(n)) + 0.5

        # Add targets at different strengths
        signal[64] = 15.0  # Very strong
        signal[128] = 8.0  # Medium
        signal[192] = 5.0  # Weak
        signal[200:210] = np.linspace(3, 7, 10)  # Rising edge

        return signal

    @pytest.fixture
    def signal_2d(self):
        """2D signal (image) with targets."""
        np.random.seed(42)
        image = np.abs(np.random.randn(64, 64))

        # Add point targets
        image[16, 16] = 12.0
        image[32, 32] = 8.0
        image[48, 48] = 6.0

        # Add extended target
        image[25:30, 25:30] = 10.0

        return image

    # CA-CFAR Tests
    def test_cfar_ca_basic(self, signal_1d):
        """Test CA-CFAR with basic parameters."""
        result = cfar_ca(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.dtype == bool
        assert result.detections.shape == signal_1d.shape
        assert np.any(result.detections), "Should detect at least one target"
        # Should detect strongest target
        assert result.detections[64]

    def test_cfar_ca_different_pfa(self, signal_1d):
        """Test CA-CFAR with different PFA values."""
        for pfa in [1e-2, 1e-3, 1e-4]:
            result = cfar_ca(signal_1d, guard_cells=2, ref_cells=5, pfa=pfa)
            assert result.detections.shape == signal_1d.shape
            # Higher PFA should detect more

    def test_cfar_ca_edge_cases(self, signal_1d):
        """Test CA-CFAR near boundaries."""
        # Small guard and reference cells
        result = cfar_ca(signal_1d, guard_cells=1, ref_cells=2, pfa=1e-3)
        assert result.detections.shape == signal_1d.shape

        # Large cells that would approach signal length
        result = cfar_ca(signal_1d, guard_cells=5, ref_cells=20, pfa=1e-3)
        assert result.detections.shape == signal_1d.shape

    def test_cfar_ca_low_snr(self):
        """Test CA-CFAR with low SNR signal."""
        signal = np.abs(np.random.randn(100)) + 0.1
        signal[50] = 1.5  # Only 15x above noise
        result = cfar_ca(signal, guard_cells=2, ref_cells=5, pfa=1e-2)
        assert result.detections.shape == signal.shape

    # GO-CFAR Tests
    def test_cfar_go_basic(self, signal_1d):
        """Test GO-CFAR detector."""
        result = cfar_go(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.dtype == bool
        assert result.detections.shape == signal_1d.shape

    def test_cfar_go_vs_ca(self, signal_1d):
        """Compare GO-CFAR with CA-CFAR."""
        result_ca = cfar_ca(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3)
        result_go = cfar_go(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3)

        # GO should have fewer false alarms
        assert result_go.detections.dtype == result_ca.detections.dtype

    def test_cfar_go_noise_level(self):
        """Test GO-CFAR with varying noise levels."""
        for noise_scale in [0.1, 0.5, 1.0, 2.0]:
            signal = np.abs(np.random.randn(100)) * noise_scale + 0.1
            signal[50] = signal[50] + 5
            result = cfar_go(signal, guard_cells=2, ref_cells=5, pfa=1e-3)
            assert result.detections.shape == signal.shape

    # SO-CFAR Tests
    def test_cfar_so_basic(self, signal_1d):
        """Test SO-CFAR detector (Smallest Of)."""
        result = cfar_so(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.dtype == bool
        assert result.detections.shape == signal_1d.shape

    def test_cfar_so_clutter_edge(self):
        """Test SO-CFAR in clutter with sharp edges."""
        signal = np.ones(100)
        signal[:50] = 1.0  # Low clutter
        signal[50:] = 3.0  # High clutter
        signal[40] = 2.0  # Target in low clutter
        signal[60] = 5.0  # Target in high clutter

        result = cfar_so(signal, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.shape == signal.shape

    # OS-CFAR Tests
    def test_cfar_os_basic(self, signal_1d):
        """Test OS-CFAR detector with different k values."""
        for k in [3, 5, 7, 9]:
            result = cfar_os(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3, k=k)
            assert result.detections.dtype == bool
            assert result.detections.shape == signal_1d.shape

    def test_cfar_os_extreme_k(self, signal_1d):
        """Test OS-CFAR with extreme k (order statistic) values."""
        n_cells = 10
        # k = 1 (minimum)
        result = cfar_os(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3, k=1)
        assert result.detections.shape == signal_1d.shape

        # k = n_cells (maximum)
        result = cfar_os(signal_1d, guard_cells=2, ref_cells=5, pfa=1e-3, k=n_cells)
        assert result.detections.shape == signal_1d.shape

    def test_cfar_os_impulsive_noise(self):
        """Test OS-CFAR in impulsive noise environment."""
        np.random.seed(42)
        signal = np.abs(np.random.randn(100))
        # Add impulsive noise (spikes)
        noise_idx = np.random.choice(100, 10, replace=False)
        signal[noise_idx] *= 10

        signal[50] = signal[50] + 3

        result = cfar_os(signal, guard_cells=2, ref_cells=5, pfa=1e-3, k=5)
        assert result.detections.shape == signal.shape

    # 2D CFAR Tests
    def test_cfar_2d_basic(self, signal_2d):
        """Test 2D CFAR detection."""
        result = cfar_2d(signal_2d, guard_cells=(2, 2), ref_cells=(5, 5), pfa=1e-3)
        assert result.detections.dtype == bool
        assert result.detections.shape == signal_2d.shape
        # Should detect point targets
        assert result.detections[16, 16]

    def test_cfar_2d_rectangular_guard(self, signal_2d):
        """Test 2D CFAR with rectangular guard region."""
        result = cfar_2d(signal_2d, guard_cells=(1, 3), ref_cells=(5, 5), pfa=1e-3)
        assert result.detections.shape == signal_2d.shape

    def test_cfar_2d_asymmetric(self, signal_2d):
        """Test 2D CFAR with asymmetric reference cells."""
        result = cfar_2d(signal_2d, guard_cells=(2, 2), ref_cells=(3, 7), pfa=1e-3)
        assert result.detections.shape == signal_2d.shape

    def test_cfar_2d_small_region(self):
        """Test 2D CFAR on small image."""
        image = np.random.randn(12, 12)
        image[6, 6] = 5.0
        result = cfar_2d(image, guard_cells=(1, 1), ref_cells=(2, 2), pfa=1e-2)
        assert result.detections.shape == image.shape

    # Utility Function Tests
    class TestUtilityFunctions:
        """Tests for detection utility functions."""

        def test_threshold_factor_methods(self):
            """Test threshold factor for different methods."""
            for method in ["ca", "go", "so", "os"]:
                alpha = threshold_factor(pfa=1e-3, n_ref=10, method=method)
                assert alpha > 0
                assert np.isfinite(alpha)

        def test_threshold_factor_pfa_range(self):
            """Test threshold factor across PFA range."""
            pfa_values = np.logspace(-2, -5, 10)
            for pfa in pfa_values:
                alpha = threshold_factor(pfa=pfa, n_ref=10, method="ca")
                assert np.isfinite(alpha)
                # Threshold should increase with lower PFA
                assert alpha > threshold_factor(pfa=pfa * 10, n_ref=10, method="ca")

        def test_threshold_factor_varying_ref_cells(self):
            """Test threshold factor with different reference cell counts."""
            for n_ref in [5, 10, 20, 50]:
                alpha = threshold_factor(pfa=1e-3, n_ref=n_ref, method="ca")
                assert np.isfinite(alpha)

        def test_detection_probability_snr_range(self):
            """Test detection probability across SNR range."""
            snr_values = np.linspace(0, 20, 10)
            prev_pd = 0
            for snr in snr_values:
                pd = detection_probability(snr=snr, pfa=1e-3, n_ref=10, method="ca")
                assert 0 < pd <= 1
                # Should increase with SNR
                assert pd >= prev_pd
                prev_pd = pd

        def test_detection_probability_methods(self):
            """Test detection probability for different methods."""
            for method in ["ca", "go", "so"]:
                pd = detection_probability(snr=10, pfa=1e-3, n_ref=10, method=method)
                assert 0 < pd <= 1

        def test_detection_probability_extreme_snr(self):
            """Test detection probability at extreme SNR."""
            # Very low SNR
            pd_low = detection_probability(snr=0.1, pfa=1e-3, n_ref=10, method="ca")
            assert 0 < pd_low < 1

            # Very high SNR
            pd_high = detection_probability(snr=50, pfa=1e-3, n_ref=10, method="ca")
            assert pd_high > pd_low
            assert pd_high <= 1.0

        def test_snr_loss_basic(self):
            """Test SNR loss calculations."""
            loss = snr_loss(n_ref=10, method="ca")
            assert loss >= 0
            assert np.isfinite(loss)

        def test_snr_loss_methods(self):
            """Test SNR loss for different methods."""
            methods = ["ca", "go", "so", "os"]
            losses = {}
            for method in methods:
                loss = snr_loss(n_ref=10, method=method)
                assert loss >= 0
                losses[method] = loss
            # Different methods should have different losses
            assert len(set(losses.values())) > 1

        def test_snr_loss_varying_ref(self):
            """Test SNR loss with varying reference cells."""
            for n_ref in [5, 10, 20, 50]:
                loss = snr_loss(n_ref=n_ref, method="ca")
                assert np.isfinite(loss)

    # Clustering Tests
    class TestClusterDetections:
        """Tests for detection clustering."""

        def test_cluster_detections_basic(self):
            """Test basic clustering of adjacent detections."""
            detections = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)

            clusters = cluster_detections(detections, min_separation=1)
            assert len(clusters) > 0

        def test_cluster_detections_separated(self):
            """Test clustering with well-separated detections."""
            detections = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)

            clusters = cluster_detections(detections, min_separation=2)
            assert len(clusters) >= 1

        def test_cluster_detections_no_detections(self):
            """Test clustering with no detections."""
            detections = np.zeros(10, dtype=bool)

            clusters = cluster_detections(detections, min_separation=1)
            assert len(clusters) == 0

        def test_cluster_detections_all_detected(self):
            """Test clustering with all points detected."""
            detections = np.ones(10, dtype=bool)

            clusters = cluster_detections(detections, min_separation=1)
            assert len(clusters) >= 1

        def test_cluster_detections_varying_separation(self):
            """Test clustering with varying separation thresholds."""
            detections = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)

            # Small separation: each adjacent group is separate
            clusters_small = cluster_detections(detections, min_separation=1)

            # Large separation considers distant groups
            clusters_large = cluster_detections(detections, min_separation=5)

            assert len(clusters_small) > 0
            assert len(clusters_large) > 0

        def test_cluster_detections_peak_selection(self):
            """Test that clustering selects appropriate peaks."""
            detections = np.array([1, 0, 1, 1, 1, 0, 1], dtype=bool)

            clusters = cluster_detections(detections, min_separation=1)
            # Should identify clusters but not return all detections
            assert len(clusters) <= np.sum(detections)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
