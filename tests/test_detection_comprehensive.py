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
    threshold_factor,
    cfar_ca,
    cfar_go,
    cfar_so,
    cfar_os,
    cfar_2d,
    detection_probability,
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
