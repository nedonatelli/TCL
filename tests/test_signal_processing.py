"""Tests for signal processing module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.signal_processing import (
    CFARResult,
    CFARResult2D,
    FilterCoefficients,
    FrequencyResponse,
    MatchedFilterResult,
    PulseCompressionResult,
    apply_filter,
    bessel_design,
    butter_design,
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    cheby1_design,
    cheby2_design,
    cluster_detections,
    detection_probability,
    ellip_design,
    filter_order,
    filtfilt,
    fir_design,
    fir_design_remez,
    frequency_response,
    generate_lfm_chirp,
    generate_nlfm_chirp,
    group_delay,
    matched_filter,
    matched_filter_frequency,
    optimal_filter,
    pulse_compression,
    snr_loss,
    threshold_factor,
)


class TestFilterDesign:
    """Tests for digital filter design functions."""

    def test_butter_lowpass(self):
        """Test Butterworth lowpass filter design."""
        fs = 1000
        coeffs = butter_design(4, 100, fs, btype="low")

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None
        assert len(coeffs.b) > 0
        assert len(coeffs.a) > 0

    def test_butter_highpass(self):
        """Test Butterworth highpass filter design."""
        fs = 1000
        coeffs = butter_design(4, 100, fs, btype="high")

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None

    def test_butter_bandpass(self):
        """Test Butterworth bandpass filter design."""
        fs = 1000
        coeffs = butter_design(4, (50, 150), fs, btype="band")

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None

    def test_cheby1_design(self):
        """Test Chebyshev Type I filter design."""
        fs = 1000
        coeffs = cheby1_design(4, 0.5, 100, fs)

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None

    def test_cheby2_design(self):
        """Test Chebyshev Type II filter design."""
        fs = 1000
        coeffs = cheby2_design(4, 40, 100, fs)

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None

    def test_ellip_design(self):
        """Test elliptic filter design."""
        fs = 1000
        coeffs = ellip_design(4, 0.5, 40, 100, fs)

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None

    def test_bessel_design(self):
        """Test Bessel filter design."""
        fs = 1000
        coeffs = bessel_design(4, 100, fs)

        assert isinstance(coeffs, FilterCoefficients)
        assert coeffs.sos is not None

    def test_fir_design(self):
        """Test FIR filter design."""
        fs = 1000
        h = fir_design(101, 100, fs)

        assert len(h) == 101
        # FIR filter should be symmetric for linear phase
        assert_allclose(h, h[::-1], atol=1e-10)

    def test_fir_design_remez(self):
        """Test Remez FIR filter design."""
        fs = 1000
        # Stopband must end before Nyquist (500 Hz)
        h = fir_design_remez(101, [0, 100, 150, 450], [1, 0], fs)

        assert len(h) == 101

    def test_frequency_response(self):
        """Test frequency response computation."""
        fs = 1000
        coeffs = butter_design(4, 100, fs)
        response = frequency_response(coeffs, fs)

        assert isinstance(response, FrequencyResponse)
        assert len(response.frequencies) == 512
        assert len(response.magnitude) == 512
        assert len(response.phase) == 512
        # DC gain should be approximately 1 for lowpass
        assert_allclose(response.magnitude[0], 1.0, atol=0.01)

    def test_group_delay_fir(self):
        """Test group delay for symmetric FIR filter."""
        fs = 1000
        h = fir_design(51, 100, fs)
        freqs, gd = group_delay(h, fs)

        # Symmetric FIR should have constant group delay = (N-1)/2
        assert_allclose(gd, 25, atol=0.1)

    def test_filter_order(self):
        """Test filter order estimation."""
        order = filter_order(100, 150, 0.5, 40, 1000, "butter")
        assert order > 0

    def test_apply_filter(self):
        """Test filter application."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        coeffs = butter_design(4, 100, fs)
        y = apply_filter(coeffs, x)

        assert len(y) == len(x)

    def test_filtfilt_zero_phase(self):
        """Test zero-phase filtering."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        coeffs = butter_design(4, 100, fs)
        y = filtfilt(coeffs, x)

        assert len(y) == len(x)
        # filtfilt should not introduce phase shift for passband signal
        # Find peak of original and filtered - should be at same location
        peak_x = np.argmax(x[:100])
        peak_y = np.argmax(y[:100])
        assert abs(peak_x - peak_y) <= 1


class TestMatchedFilter:
    """Tests for matched filtering functions."""

    def test_matched_filter_pulse(self):
        """Test matched filter with simple pulse."""
        template = np.ones(10)
        signal = np.zeros(100)
        signal[50:60] = 1.0

        result = matched_filter(signal, template)

        assert isinstance(result, MatchedFilterResult)
        assert 50 <= result.peak_index <= 60
        assert result.snr_gain > 0

    def test_matched_filter_chirp(self):
        """Test matched filter with chirp signal."""
        fs = 1000
        chirp = generate_lfm_chirp(0.01, 50, 200, fs)
        signal = np.zeros(500)
        signal[200 : 200 + len(chirp)] = chirp

        result = matched_filter(signal, chirp)

        assert 200 <= result.peak_index <= 200 + len(chirp)

    def test_matched_filter_frequency(self):
        """Test frequency-domain matched filter."""
        template = np.sin(2 * np.pi * 0.1 * np.arange(50))
        signal = np.zeros(200)
        signal[100:150] = template

        result = matched_filter_frequency(signal, template)

        assert isinstance(result, MatchedFilterResult)
        assert 100 <= result.peak_index <= 150

    def test_optimal_filter(self):
        """Test optimal filter with white noise."""
        signal = np.random.randn(256)
        template = np.ones(16)
        noise_psd = np.ones(256)  # White noise

        output = optimal_filter(signal, template, noise_psd)

        assert len(output) == len(signal)

    def test_pulse_compression(self):
        """Test pulse compression."""
        fs = 1000
        chirp = generate_lfm_chirp(0.05, 50, 200, fs)
        signal = np.zeros(1000)
        signal[300 : 300 + len(chirp)] = chirp

        result = pulse_compression(signal, chirp)

        assert isinstance(result, PulseCompressionResult)
        assert result.compression_ratio > 1

    def test_lfm_chirp_generation(self):
        """Test LFM chirp generation."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)

        assert len(chirp) == 44
        assert chirp[0] == pytest.approx(1.0, abs=0.01)

    def test_nlfm_chirp_generation(self):
        """Test NLFM chirp generation."""
        chirp = generate_nlfm_chirp(0.001, 1000, 5000, 44100, beta=2.0)

        assert len(chirp) == 44


class TestCFAR:
    """Tests for CFAR detection algorithms."""

    def test_cfar_ca_detection(self):
        """Test CA-CFAR detection."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 1000)
        signal[500] = 100  # Strong target

        result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)
        assert 500 in result.detection_indices

    def test_cfar_go_detection(self):
        """Test GO-CFAR detection."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)
        assert len(result.detection_indices) >= 1

    def test_cfar_so_detection(self):
        """Test SO-CFAR detection."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50

        result = cfar_so(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)

    def test_cfar_os_detection(self):
        """Test OS-CFAR detection."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 500)
        signal[250] = 50
        signal[260] = 40  # Closely spaced targets

        result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)

        assert isinstance(result, CFARResult)

    def test_cfar_2d(self):
        """Test 2D CFAR detection."""
        np.random.seed(42)
        image = np.random.exponential(1.0, (100, 100))
        image[50, 50] = 100

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4)

        assert isinstance(result, CFARResult2D)
        assert result.detections[50, 50]

    def test_threshold_factor(self):
        """Test CFAR threshold factor computation."""
        alpha = threshold_factor(1e-6, 32, method="ca")

        assert alpha > 1

    def test_detection_probability(self):
        """Test detection probability computation."""
        pd = detection_probability(snr=10, pfa=1e-6, n_ref=32)

        assert 0 < pd < 1

    def test_cluster_detections(self):
        """Test detection clustering."""
        detections = np.array([False] * 100)
        detections[50:55] = True  # Cluster of 5

        peaks = cluster_detections(detections, min_separation=2)

        assert len(peaks) == 1
        assert 50 <= peaks[0] <= 54

    def test_snr_loss(self):
        """Test CFAR SNR loss computation."""
        loss = snr_loss(32, method="ca")

        assert 0 < loss < 1  # Small loss for many cells


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_filter_with_fir(self):
        """Test applying FIR filter."""
        h = np.ones(10) / 10  # Moving average
        x = np.random.randn(100)
        y = apply_filter(h, x)

        assert len(y) == len(x)

    def test_cfar_short_signal(self):
        """Test CFAR with short signal."""
        signal = np.array([1, 2, 100, 2, 1])

        result = cfar_ca(signal, guard_cells=0, ref_cells=1, pfa=1e-2)

        assert len(result.detections) == 5

    def test_matched_filter_identical_signals(self):
        """Test matched filter with identical signal and template."""
        template = np.random.randn(50)
        signal = template.copy()

        result = matched_filter(signal, template, normalize=True)

        # Peak should be near 1 for normalized auto-correlation
        assert result.peak_value > 0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_filter_and_detect(self):
        """Test filtering followed by CFAR detection."""
        np.random.seed(42)
        fs = 1000

        # Create signal with noise and targets
        signal = np.random.randn(1000)
        signal[300:320] = 10  # Target 1
        signal[700:720] = 8  # Target 2

        # Filter
        coeffs = butter_design(4, 100, fs)
        filtered = apply_filter(coeffs, np.abs(signal) ** 2)

        # Detect
        result = cfar_ca(filtered, guard_cells=5, ref_cells=20, pfa=1e-4)

        # Should detect both targets
        assert len(result.detection_indices) >= 2

    def test_chirp_compression_detection(self):
        """Test chirp generation, compression, and detection."""
        fs = 10000
        chirp = generate_lfm_chirp(0.001, 500, 2000, fs)

        # Create signal with chirp in noise
        signal = 0.5 * np.random.randn(5000)
        signal[2000 : 2000 + len(chirp)] += chirp

        # Compress
        result = pulse_compression(signal, chirp)

        # Peak should be near chirp location
        assert 1990 <= result.peak_index <= 2010 + len(chirp)
