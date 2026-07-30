"""
Tests for digital filter design functions.

Tests cover:
- Butterworth, Chebyshev, elliptic, Bessel filter design
- FIR filter design (windowed and Remez)
- Frequency response, group delay, filter order estimation
- Filter application and zero-phase filtering
- Integration tests (filter + detect pipeline)
"""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.signal_processing import (
    FilterCoefficients,
    FrequencyResponse,
    apply_filter,
    bessel_design,
    butter_design,
    cfar_ca,
    cheby1_design,
    cheby2_design,
    ellip_design,
    filter_order,
    filtfilt,
    fir_design,
    fir_design_remez,
    frequency_response,
    generate_lfm_chirp,
    group_delay,
    pulse_compression,
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


class TestEdgeCases:
    """Test edge cases for filter design."""

    def test_filter_with_fir(self):
        """Test applying FIR filter."""
        h = np.ones(10) / 10  # Moving average
        x = np.random.randn(100)
        y = apply_filter(h, x)

        assert len(y) == len(x)


class TestIntegration:
    """Integration tests combining filtering and detection."""

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
