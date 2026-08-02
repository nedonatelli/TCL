"""Tests for CFAR (Constant False Alarm Rate) detection algorithms.

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

    def test_threshold_factor_all_methods_positive(self):
        """Test all CFAR methods produce positive factors."""
        for method in ["ca", "go", "so", "os"]:
            factor = threshold_factor(1e-3, 16, method=method)
            assert factor > 0
            assert np.isfinite(factor)

    def test_threshold_factor_pfa_range(self):
        """Test threshold factor across PFA range."""
        pfa_values = np.logspace(-2, -5, 10)
        for pfa in pfa_values:
            alpha = threshold_factor(pfa=pfa, n_ref=10, method="ca")
            assert np.isfinite(alpha)
            assert alpha > threshold_factor(pfa=pfa * 10, n_ref=10, method="ca")

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

    def test_detection_probability_varying_snr(self):
        """Test detection probability monotonically increases with SNR."""
        snr_values = np.linspace(0, 20, 10)
        prev_pd = 0
        for snr in snr_values:
            pd = detection_probability(snr=snr, pfa=1e-3, n_ref=10, method="ca")
            assert 0 < pd <= 1
            assert pd >= prev_pd
            prev_pd = pd

    def test_detection_probability_valid_returns(self):
        """Test detection probability returns valid values across wide SNR range."""
        for snr in [-20, -10, 0, 10, 20, 30]:
            pd = detection_probability(snr, pfa=1e-3, n_ref=16)
            assert np.isfinite(pd)

    def test_detection_probability_pfa_effect(self):
        """Test detection probability with different PFA values."""
        for pfa in [0.0001, 0.001, 0.01, 0.1]:
            pd = detection_probability(snr=10, pfa=pfa, n_ref=16)
            assert np.isfinite(pd)

    def test_detection_probability_n_ref_effect(self):
        """Test detection probability with different reference cell counts."""
        for n_ref in [8, 16, 32, 64]:
            pd = detection_probability(snr=10, pfa=1e-3, n_ref=n_ref)
            assert np.isfinite(pd)

    def test_detection_probability_methods(self):
        """Test detection probability for different methods."""
        for method in ["ca", "go", "so"]:
            pd = detection_probability(snr=10, pfa=1e-3, n_ref=10, method=method)
            assert 0 < pd <= 1

    def test_detection_probability_extreme_snr(self):
        """Test detection probability at extreme SNR."""
        pd_low = detection_probability(snr=0.1, pfa=1e-3, n_ref=10, method="ca")
        pd_high = detection_probability(snr=50, pfa=1e-3, n_ref=10, method="ca")
        assert 0 < pd_low < 1
        assert pd_high > pd_low
        assert pd_high <= 1.0


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

    def test_cfar_ca_output_types(self):
        """Test CA-CFAR output types."""
        signal = np.random.randn(256)
        result = cfar_ca(signal, guard_cells=2, ref_cells=8)

        assert isinstance(result.detections, np.ndarray)
        assert result.detections.dtype == np.bool_
        assert isinstance(result.threshold, np.ndarray)
        assert isinstance(result.noise_estimate, np.ndarray)

    def test_cfar_ca_with_custom_alpha(self):
        """Test CA-CFAR with custom alpha threshold."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, alpha=5.0)

        assert 250 in result.detection_indices

    def test_cfar_ca_with_custom_pfa(self):
        """Test CA-CFAR with different PFA values."""
        signal = np.random.randn(1000)

        result_high_pfa = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.1)
        result_low_pfa = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.001)

        assert result_high_pfa.detections.dtype == np.bool_
        assert result_low_pfa.detections.dtype == np.bool_

    def test_cfar_ca_threshold_positive(self):
        """Test that CA-CFAR threshold is positive."""
        signal = np.random.exponential(1.0, 500)

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert np.all(result.threshold >= 0)

    def test_cfar_ca_noise_estimate_positive(self):
        """Test that CA-CFAR noise estimate is non-negative."""
        signal = np.abs(np.random.randn(500))

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert np.all(result.noise_estimate >= 0)

    def test_cfar_ca_sine_in_noise(self):
        """Test CA-CFAR detects strong sine in noise."""
        t = np.arange(1000)
        signal = 0.1 * np.random.randn(1000)
        signal[500:600] += 5 * np.sin(2 * np.pi * t[500:600] / 50)

        result = cfar_ca(signal, guard_cells=4, ref_cells=16, pfa=0.01)

        detections_in_signal = np.sum(result.detections[500:600])
        assert detections_in_signal > 0

    def test_cfar_ca_low_snr(self):
        """Test CA-CFAR with low SNR signal."""
        signal = np.abs(np.random.randn(100)) + 0.1
        signal[50] = 1.5
        result = cfar_ca(signal, guard_cells=2, ref_cells=5, pfa=1e-2)
        assert result.detections.shape == signal.shape


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
        signal = np.concatenate(
            [np.random.exponential(1.0, 250), np.random.exponential(10.0, 250)]
        )
        signal[250] = 100

        result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

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

    def test_cfar_go_clutter_rejection(self):
        """Test GO-CFAR clutter rejection."""
        signal = np.random.randn(1000) * 0.1
        signal[300:400] += 0.5  # Clutter region
        signal[500] = 5  # Target

        result = cfar_go(signal, guard_cells=2, ref_cells=8, pfa=0.001)

        assert np.any(result.detections[450:550])

    def test_cfar_go_noise_level(self):
        """Test GO-CFAR with varying noise levels."""
        for noise_scale in [0.1, 0.5, 1.0, 2.0]:
            signal = np.abs(np.random.randn(100)) * noise_scale + 0.1
            signal[50] = signal[50] + 5
            result = cfar_go(signal, guard_cells=2, ref_cells=5, pfa=1e-3)
            assert result.detections.shape == signal.shape


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

    def test_cfar_so_vs_ca(self):
        """Test SO-CFAR has different characteristics than CA-CFAR."""
        signal = np.random.randn(500)

        result_so = cfar_so(signal, guard_cells=2, ref_cells=8, pfa=0.01)
        result_ca = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=0.01)

        assert not np.array_equal(result_so.detections, result_ca.detections)

    def test_cfar_so_clutter_edge(self):
        """Test SO-CFAR in clutter with sharp edges."""
        signal = np.ones(100)
        signal[:50] = 1.0
        signal[50:] = 3.0
        signal[40] = 2.0
        signal[60] = 5.0

        result = cfar_so(signal, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.shape == signal.shape


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
        signal[260] = 40

        result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

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

    def test_cfar_os_different_k_values(self):
        """Test OS-CFAR with different k values."""
        signal = np.random.randn(512)

        for k in [4, 8, 12, 16]:
            result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=0.01, k=k)
            assert result.detections.shape == signal.shape

    def test_cfar_os_with_custom_alpha(self):
        """Test OS-CFAR with custom alpha."""
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_os(signal, guard_cells=2, ref_cells=16, alpha=5.0)

        assert isinstance(result, CFARResult)

    def test_cfar_os_extreme_k(self):
        """Test OS-CFAR with extreme k (order statistic) values."""
        np.random.seed(42)
        signal = np.abs(np.random.randn(256)) + 0.5
        signal[64] = 15.0

        # k = 1 (minimum)
        result = cfar_os(signal, guard_cells=2, ref_cells=5, pfa=1e-3, k=1)
        assert result.detections.shape == signal.shape

        # k = n_cells (maximum)
        result = cfar_os(signal, guard_cells=2, ref_cells=5, pfa=1e-3, k=10)
        assert result.detections.shape == signal.shape

    def test_cfar_os_impulsive_noise(self):
        """Test OS-CFAR in impulsive noise environment."""
        np.random.seed(42)
        signal = np.abs(np.random.randn(100))
        noise_idx = np.random.choice(100, 10, replace=False)
        signal[noise_idx] *= 10
        signal[50] = signal[50] + 3

        result = cfar_os(signal, guard_cells=2, ref_cells=5, pfa=1e-3, k=5)
        assert result.detections.shape == signal.shape


# =============================================================================
# Tests for 2D CFAR
# =============================================================================


class TestCfar2d:
    """Tests for two-dimensional CFAR."""

    def test_cfar_2d_ca_detects_target(self):
        """Test that 2D CA-CFAR detects a target."""
        np.random.seed(42)
        image = np.random.exponential(1.0, (100, 100))
        image[50, 50] = 100

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

    def test_cfar_2d_output_types(self):
        """Test 2D CFAR output types."""
        image = np.random.randn(64, 64)
        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8))

        assert isinstance(result.detections, np.ndarray)
        assert result.detections.dtype == np.bool_
        assert isinstance(result.threshold, np.ndarray)
        assert isinstance(result.noise_estimate, np.ndarray)

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

    def test_cfar_2d_extended_target(self):
        """Test 2D CFAR with extended target."""
        image = np.random.randn(100, 100) * 0.1
        image[40:60, 40:60] += 2

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=0.01)

        assert result.detections.shape == image.shape
        assert result.detections.dtype == np.bool_

    def test_cfar_2d_noise_only(self):
        """Test 2D CFAR with noise only."""
        image = np.random.randn(100, 100)
        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=0.01)

        assert result.detections.shape == image.shape
        assert result.detections.dtype == np.bool_

    def test_cfar_2d_rectangular_guard(self):
        """Test 2D CFAR with rectangular guard region."""
        np.random.seed(42)
        image = np.abs(np.random.randn(64, 64))
        image[16, 16] = 12.0

        result = cfar_2d(image, guard_cells=(1, 3), ref_cells=(5, 5), pfa=1e-3)
        assert result.detections.shape == image.shape

    def test_cfar_2d_asymmetric_ref_cells(self):
        """Test 2D CFAR with asymmetric reference cells."""
        np.random.seed(42)
        image = np.abs(np.random.randn(64, 64))
        image[32, 32] = 8.0

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(3, 7), pfa=1e-3)
        assert result.detections.shape == image.shape

    def test_cfar_2d_small_region(self):
        """Test 2D CFAR on small image."""
        image = np.random.randn(12, 12)
        image[6, 6] = 5.0
        result = cfar_2d(image, guard_cells=(1, 1), ref_cells=(2, 2), pfa=1e-2)
        assert result.detections.shape == image.shape


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
        detections = np.array([False] * 45 + [True] * 10 + [False] * 45)

        peaks = cluster_detections(detections, min_separation=1)

        assert len(peaks) == 1

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

        assert len(peaks) == 3

    def test_cluster_detections_min_separation(self):
        """Test clustering with different minimum separation."""
        detections = np.zeros(100, dtype=bool)
        detections[20] = True
        detections[22] = True
        detections[24] = True

        peaks1 = cluster_detections(detections, min_separation=1)
        assert len(peaks1) == 3

        peaks2 = cluster_detections(detections, min_separation=2)
        assert len(peaks2) == 1

    def test_cluster_detections_all_detected(self):
        """Test clustering with all points detected."""
        detections = np.ones(10, dtype=bool)

        clusters = cluster_detections(detections, min_separation=1)
        assert len(clusters) >= 1

    def test_cluster_detections_peak_selection(self):
        """Test that clustering selects appropriate peaks."""
        detections = np.array([1, 0, 1, 1, 1, 0, 1], dtype=bool)

        clusters = cluster_detections(detections, min_separation=1)
        assert len(clusters) <= np.sum(detections)


# =============================================================================
# Tests for snr_loss
# =============================================================================


class TestSnrLoss:
    """Tests for SNR loss computation."""

    def test_snr_loss_ca(self):
        """Test SNR loss for CA-CFAR."""
        loss = snr_loss(32, method="ca")
        assert loss > 0
        assert loss < 1

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

    def test_snr_loss_different_methods(self):
        """Test SNR loss differs across methods."""
        methods = ["ca", "go", "so", "os"]
        losses = {}
        for method in methods:
            loss = snr_loss(n_ref=10, method=method)
            assert loss >= 0
            losses[method] = loss
        assert len(set(losses.values())) > 1

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
        signal[500] = 100

        ca_result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
        go_result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
        so_result = cfar_so(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
        os_result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert 500 in ca_result.detection_indices
        assert 500 in go_result.detection_indices
        assert 500 in so_result.detection_indices
        assert 500 in os_result.detection_indices

    def test_cfar_false_alarm_rate(self):
        """Test that CFAR maintains approximate false alarm rate."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 10000)

        result = cfar_ca(signal, guard_cells=2, ref_cells=32, pfa=1e-3)

        edge = 50
        fa_count = np.sum(result.detections[edge:-edge])
        total_cells = len(signal) - 2 * edge
        measured_pfa = fa_count / total_cells
        assert measured_pfa < 5e-3

    def test_cfar_edge_handling(self):
        """Test CFAR behavior at signal edges."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 100)
        signal[5] = 50
        signal[95] = 50

        result = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=1e-4)

        assert isinstance(result, CFARResult)
        assert len(result.threshold) == 100

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

    def test_cfar_adapts_to_noise_floor(self):
        """Test CFAR adapts to different noise levels."""
        signal_low = np.random.randn(512) * 0.1
        signal_high = np.random.randn(512)

        result_low = cfar_ca(signal_low, guard_cells=2, ref_cells=8)
        result_high = cfar_ca(signal_high, guard_cells=2, ref_cells=8)

        assert result_low.detections.dtype == np.bool_
        assert result_high.detections.dtype == np.bool_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
