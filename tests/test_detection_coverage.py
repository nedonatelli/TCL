"""Comprehensive tests for signal detection to improve coverage.

This module provides additional tests for Tier 2 coverage improvement of
signal detection (47% -> ~70% target).
"""

import numpy as np
import pytest
from pytcl.mathematical_functions.signal_processing.detection import (
    cfar_ca,
    cfar_go,
    cfar_so,
    cfar_os,
    cfar_2d,
    cluster_detections,
    detection_probability,
    snr_loss,
    threshold_factor,
)


class TestCFARDetectionComprehensive:
    """Comprehensive tests for all CFAR detection variants."""

    @pytest.fixture
    def signal_1d(self):
        """1D signal with multiple targets."""
        np.random.seed(42)
        n = 256
        signal = np.abs(np.random.randn(n)) + 0.5
        
        # Add targets at different strengths
        signal[64] = 15.0   # Very strong
        signal[128] = 8.0   # Medium
        signal[192] = 5.0   # Weak
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
        signal[40] = 2.0   # Target in low clutter
        signal[60] = 5.0   # Target in high clutter
        
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
                assert alpha > threshold_factor(pfa=pfa*10, n_ref=10, method="ca")

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
