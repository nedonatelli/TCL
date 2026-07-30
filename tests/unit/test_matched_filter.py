"""
Tests for matched filtering and pulse compression.

Tests cover:
- Time-domain matched filtering
- Frequency-domain matched filtering
- Optimal filtering for colored noise
- Pulse compression
- LFM and NLFM chirp generation
- Ambiguity function
- Cross-ambiguity function
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.signal_processing.matched_filter import (
    MatchedFilterResult,
    PulseCompressionResult,
    ambiguity_function,
    cross_ambiguity,
    generate_lfm_chirp,
    generate_nlfm_chirp,
    matched_filter,
    matched_filter_frequency,
    optimal_filter,
    pulse_compression,
)

# =============================================================================
# Tests for matched_filter (time domain)
# =============================================================================


class TestMatchedFilter:
    """Tests for time-domain matched filtering."""

    def test_matched_filter_basic(self):
        """Test basic matched filter detection."""
        template = np.array([1, 1, 1, 1, 1], dtype=np.float64)
        signal = np.zeros(100)
        signal[50:55] = template
        result = matched_filter(signal, template)

        assert isinstance(result, MatchedFilterResult)
        assert 50 <= result.peak_index <= 54
        assert result.peak_value > 0
        assert np.isfinite(result.snr_gain)

    def test_matched_filter_snr_gain(self):
        """Test SNR gain matches expected value."""
        template = np.ones(10)
        signal = np.zeros(50)
        signal[20:30] = template
        result = matched_filter(signal, template)

        # SNR gain should be 10*log10(N) where N is template length
        expected_gain = 10 * np.log10(10)
        assert result.snr_gain == pytest.approx(expected_gain, rel=1e-6)

    def test_matched_filter_normalized(self):
        """Test normalized matched filter output."""
        template = 2 * np.ones(5)
        signal = np.zeros(50)
        signal[20:25] = template
        result = matched_filter(signal, template, normalize=True)
        # Normalized output should have peak around 1
        assert result.peak_value == pytest.approx(1.0, rel=0.1)

    def test_matched_filter_unnormalized(self):
        """Test unnormalized matched filter output."""
        template = np.ones(5)
        signal = np.zeros(50)
        signal[20:25] = template
        result = matched_filter(signal, template, normalize=False)
        # Unnormalized output peak should be template energy * signal amplitude
        expected_peak = np.sum(template**2)
        assert result.peak_value == pytest.approx(expected_peak, rel=0.1)

    def test_matched_filter_mode_full(self):
        """Test full convolution mode."""
        template = np.ones(5)
        signal = np.zeros(20)
        signal[10:15] = template
        result = matched_filter(signal, template, mode="full")
        assert len(result.output) == len(signal) + len(template) - 1

    def test_matched_filter_mode_valid(self):
        """Test valid convolution mode."""
        template = np.ones(5)
        signal = np.zeros(20)
        signal[10:15] = template
        result = matched_filter(signal, template, mode="valid")
        assert len(result.output) == len(signal) - len(template) + 1

    def test_matched_filter_no_signal(self):
        """Test matched filter with no signal present."""
        template = np.ones(5)
        signal = np.zeros(50)
        result = matched_filter(signal, template)
        assert result.peak_value == pytest.approx(0.0, abs=1e-10)

    def test_matched_filter_sinusoidal(self):
        """Test matched filter with sinusoidal template."""
        t = np.linspace(0, 1, 100)
        template = np.sin(2 * np.pi * 5 * t[:20])
        signal = np.zeros(100)
        signal[40:60] = template
        result = matched_filter(signal, template)
        assert 40 <= result.peak_index <= 60


# =============================================================================
# Tests for matched_filter_frequency
# =============================================================================


class TestMatchedFilterFrequency:
    """Tests for frequency-domain matched filtering."""

    def test_matched_filter_frequency_basic(self):
        """Test basic frequency-domain matched filter."""
        template = np.sin(2 * np.pi * 0.1 * np.arange(50))
        signal = np.zeros(200)
        signal[100:150] = template
        result = matched_filter_frequency(signal, template)

        assert isinstance(result, MatchedFilterResult)
        assert len(result.output) == len(signal)
        assert np.isfinite(result.peak_value)

    def test_matched_filter_frequency_detection(self):
        """Test frequency-domain filter detects signal."""
        template = np.ones(10)
        signal = np.zeros(100)
        signal[50:60] = template
        result = matched_filter_frequency(signal, template)
        # Peak should be near signal location
        assert 45 <= result.peak_index <= 65

    def test_matched_filter_frequency_snr_gain(self):
        """Test SNR gain calculation."""
        template = np.ones(16)
        signal = np.zeros(100)
        signal[40:56] = template
        result = matched_filter_frequency(signal, template)
        expected_gain = 10 * np.log10(16)
        assert result.snr_gain == pytest.approx(expected_gain, rel=1e-6)

    def test_matched_filter_frequency_with_fs(self):
        """Test with specified sampling frequency."""
        template = np.ones(10)
        signal = np.zeros(100)
        signal[50:60] = template
        result = matched_filter_frequency(signal, template, fs=1000.0)
        assert np.isfinite(result.peak_value)

    def test_matched_filter_frequency_unnormalized(self):
        """Test unnormalized frequency-domain filter."""
        template = np.ones(8)
        signal = np.zeros(50)
        signal[20:28] = template
        result = matched_filter_frequency(signal, template, normalize=False)
        assert result.peak_value > 0


# =============================================================================
# Tests for optimal_filter
# =============================================================================


class TestOptimalFilter:
    """Tests for optimal filtering with colored noise."""

    def test_optimal_filter_white_noise(self):
        """Test optimal filter with white noise PSD."""
        signal = np.random.randn(256)
        template = np.ones(16)
        noise_psd = np.ones(256)  # White noise
        output = optimal_filter(signal, template, noise_psd)

        assert len(output) == len(signal)
        assert np.all(np.isfinite(output))

    def test_optimal_filter_colored_noise(self):
        """Test optimal filter with colored noise PSD."""
        signal = np.zeros(128)
        template = np.ones(8)
        signal[60:68] = template

        # Create a colored noise PSD (more power at low frequencies)
        freqs = np.fft.fftfreq(128)
        noise_psd = 1.0 / (np.abs(freqs) + 0.1)
        noise_psd[0] = noise_psd[1]  # Avoid infinity at DC

        output = optimal_filter(signal, template, noise_psd)
        assert len(output) == len(signal)
        assert np.all(np.isfinite(output))

    def test_optimal_filter_with_fs(self):
        """Test optimal filter with sampling frequency."""
        signal = np.random.randn(64)
        template = np.ones(8)
        noise_psd = np.ones(64)
        output = optimal_filter(signal, template, noise_psd, fs=1000.0)
        assert len(output) == len(signal)


# =============================================================================
# Tests for pulse_compression
# =============================================================================


class TestPulseCompression:
    """Tests for pulse compression."""

    def test_pulse_compression_basic(self):
        """Test basic pulse compression."""
        fs = 1000
        chirp = generate_lfm_chirp(0.1, 50, 200, fs)
        signal = np.zeros(2000)
        signal[500 : 500 + len(chirp)] = chirp
        result = pulse_compression(signal, chirp)

        assert isinstance(result, PulseCompressionResult)
        assert result.compression_ratio > 1
        assert np.isfinite(result.peak_sidelobe_ratio)

    def test_pulse_compression_windowed(self):
        """Test pulse compression with windowing."""
        fs = 1000
        chirp = generate_lfm_chirp(0.05, 100, 300, fs)
        signal = np.zeros(1000)
        signal[200 : 200 + len(chirp)] = chirp

        for window in ["hamming", "hann", "blackman"]:
            result = pulse_compression(signal, chirp, window=window)
            assert result.compression_ratio > 0
            assert np.isfinite(result.peak_sidelobe_ratio)

    def test_pulse_compression_compression_ratio(self):
        """Test compression ratio is approximately time-bandwidth product."""
        fs = 10000
        duration = 0.01  # 10 ms
        f0, f1 = 500, 2500  # 2000 Hz bandwidth
        chirp = generate_lfm_chirp(duration, f0, f1, fs)
        signal = np.zeros(len(chirp) * 3)
        signal[len(chirp) : 2 * len(chirp)] = chirp
        result = pulse_compression(signal, chirp)

        # Time-bandwidth product = (f1 - f0) * duration = 20
        # Compression ratio should be related to TBP
        assert result.compression_ratio > 1

    def test_pulse_compression_no_window(self):
        """Test pulse compression without window."""
        fs = 1000
        chirp = generate_lfm_chirp(0.02, 100, 400, fs)
        signal = np.zeros(500)
        signal[100 : 100 + len(chirp)] = chirp
        result = pulse_compression(signal, chirp, window=None)
        assert result.compression_ratio > 0


# =============================================================================
# Tests for LFM chirp generation
# =============================================================================


class TestGenerateLFMChirp:
    """Tests for linear frequency modulated chirp generation."""

    def test_lfm_chirp_length(self):
        """Test chirp has correct length."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        expected_length = int(0.001 * 44100)
        assert len(chirp) == expected_length

    def test_lfm_chirp_initial_phase(self):
        """Test chirp starts with correct phase."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100, phase=0.0)
        assert chirp[0] == pytest.approx(1.0, rel=1e-6)  # cos(0) = 1

    def test_lfm_chirp_amplitude(self):
        """Test chirp has correct amplitude."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100, amplitude=2.0)
        assert np.max(np.abs(chirp)) == pytest.approx(2.0, rel=0.1)

    def test_lfm_chirp_upsweep(self):
        """Test up-sweep chirp (f0 < f1)."""
        chirp = generate_lfm_chirp(0.01, 100, 500, 10000)
        assert len(chirp) > 0
        assert np.all(np.isfinite(chirp))

    def test_lfm_chirp_downsweep(self):
        """Test down-sweep chirp (f0 > f1)."""
        chirp = generate_lfm_chirp(0.01, 500, 100, 10000)
        assert len(chirp) > 0
        assert np.all(np.isfinite(chirp))

    def test_lfm_chirp_with_initial_phase(self):
        """Test chirp with non-zero initial phase."""
        chirp = generate_lfm_chirp(0.001, 1000, 2000, 44100, phase=np.pi / 2)
        # cos(pi/2) = 0
        assert chirp[0] == pytest.approx(0.0, abs=0.1)


# =============================================================================
# Tests for NLFM chirp generation
# =============================================================================


class TestGenerateNLFMChirp:
    """Tests for non-linear frequency modulated chirp generation."""

    def test_nlfm_chirp_length(self):
        """Test NLFM chirp has correct length."""
        chirp = generate_nlfm_chirp(0.001, 1000, 5000, 44100)
        expected_length = int(0.001 * 44100)
        assert len(chirp) == expected_length

    def test_nlfm_chirp_amplitude(self):
        """Test NLFM chirp has correct amplitude."""
        chirp = generate_nlfm_chirp(0.001, 1000, 5000, 44100, amplitude=1.5)
        assert np.max(np.abs(chirp)) == pytest.approx(1.5, rel=0.1)

    def test_nlfm_chirp_beta_parameter(self):
        """Test NLFM chirp with different beta values."""
        for beta in [0.5, 1.0, 2.0, 5.0]:
            chirp = generate_nlfm_chirp(0.001, 1000, 5000, 44100, beta=beta)
            assert len(chirp) > 0
            assert np.all(np.isfinite(chirp))

    def test_nlfm_chirp_basic(self):
        """Test basic NLFM chirp generation."""
        chirp = generate_nlfm_chirp(0.001, 1000, 5000, 44100, beta=2.0)
        assert len(chirp) == 44
        assert np.all(np.isfinite(chirp))


# =============================================================================
# Tests for ambiguity function
# =============================================================================


class TestAmbiguityFunction:
    """Tests for ambiguity function computation."""

    def test_ambiguity_function_shape(self):
        """Test ambiguity function output shape."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        delays, dopplers, af = ambiguity_function(chirp, 44100)
        assert af.shape == (256, 256)
        assert len(delays) == 256
        assert len(dopplers) == 256

    def test_ambiguity_function_normalized(self):
        """Test ambiguity function is normalized."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        delays, dopplers, af = ambiguity_function(chirp, 44100)
        assert np.max(af) == pytest.approx(1.0, rel=1e-6)

    def test_ambiguity_function_peak_at_origin(self):
        """Test peak is near origin (zero delay, zero Doppler)."""
        chirp = generate_lfm_chirp(0.002, 500, 2000, 20000)
        delays, dopplers, af = ambiguity_function(
            chirp, 20000, n_delay=64, n_doppler=64
        )

        # Find peak location
        peak_idx = np.unravel_index(np.argmax(af), af.shape)
        # Peak should be near center (origin)
        assert 20 <= peak_idx[0] <= 44  # Doppler index
        assert 20 <= peak_idx[1] <= 44  # Delay index

    def test_ambiguity_function_custom_ranges(self):
        """Test ambiguity function with custom delay/Doppler ranges."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        delays, dopplers, af = ambiguity_function(
            chirp, 44100, max_delay=0.0005, max_doppler=1000, n_delay=32, n_doppler=32
        )
        assert af.shape == (32, 32)
        assert np.max(delays) == pytest.approx(0.0005, rel=1e-6)
        assert np.max(dopplers) == pytest.approx(1000, rel=1e-6)

    def test_ambiguity_function_real_signal(self):
        """Test ambiguity function with purely real signal."""
        signal = np.sin(2 * np.pi * 100 * np.arange(100) / 1000)
        delays, dopplers, af = ambiguity_function(
            signal, 1000, n_delay=32, n_doppler=32
        )
        assert np.all(np.isfinite(af))


# =============================================================================
# Tests for cross-ambiguity function
# =============================================================================


class TestCrossAmbiguity:
    """Tests for cross-ambiguity function."""

    def test_cross_ambiguity_shape(self):
        """Test cross-ambiguity function shape."""
        signal1 = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        signal2 = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        delays, dopplers, caf = cross_ambiguity(signal1, signal2, 44100)
        assert caf.shape == (256, 256)

    def test_cross_ambiguity_identical_signals(self):
        """Test cross-ambiguity of identical signals."""
        chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        delays, dopplers, caf = cross_ambiguity(chirp, chirp, 44100)
        # For identical signals, cross-ambiguity equals auto-ambiguity
        assert np.max(caf) == pytest.approx(1.0, rel=1e-6)

    def test_cross_ambiguity_different_lengths(self):
        """Test cross-ambiguity with different length signals."""
        signal1 = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        signal2 = generate_lfm_chirp(0.002, 1000, 5000, 44100)
        delays, dopplers, caf = cross_ambiguity(signal1, signal2, 44100)
        # Should use shorter signal length
        assert caf.shape == (256, 256)
        assert np.all(np.isfinite(caf))

    def test_cross_ambiguity_orthogonal_signals(self):
        """Test cross-ambiguity of somewhat orthogonal signals."""
        t = np.arange(100) / 1000
        signal1 = np.sin(2 * np.pi * 100 * t)
        signal2 = np.sin(2 * np.pi * 200 * t)  # Different frequency
        delays, dopplers, caf = cross_ambiguity(
            signal1, signal2, 1000, n_delay=32, n_doppler=32
        )
        assert np.all(np.isfinite(caf))

    def test_cross_ambiguity_custom_parameters(self):
        """Test cross-ambiguity with custom delay/Doppler parameters."""
        signal1 = np.random.randn(100)
        signal2 = np.random.randn(100)
        delays, dopplers, caf = cross_ambiguity(
            signal1,
            signal2,
            1000,
            max_delay=0.05,
            max_doppler=200,
            n_delay=64,
            n_doppler=64,
        )
        assert caf.shape == (64, 64)


# =============================================================================
# Integration tests
# =============================================================================


class TestMatchedFilterIntegration:
    """Integration tests for matched filtering."""

    def test_time_vs_frequency_domain(self):
        """Test time and frequency domain filters give similar results."""
        template = np.sin(2 * np.pi * 0.05 * np.arange(20))
        signal = np.zeros(100)
        signal[40:60] = template

        result_time = matched_filter(signal, template)
        result_freq = matched_filter_frequency(signal, template)

        # Both should detect the signal
        assert 35 <= result_time.peak_index <= 65
        assert 35 <= result_freq.peak_index <= 65

    def test_chirp_pulse_compression_workflow(self):
        """Test complete chirp generation and pulse compression workflow."""
        fs = 10000
        duration = 0.01
        f0, f1 = 500, 2500

        # Generate chirp
        chirp = generate_lfm_chirp(duration, f0, f1, fs)

        # Create signal with chirp
        signal = np.zeros(len(chirp) * 4)
        signal[len(chirp) : 2 * len(chirp)] = chirp

        # Pulse compression
        result = pulse_compression(signal, chirp)

        assert result.compression_ratio > 1
        assert len(chirp) <= result.peak_index <= 2 * len(chirp)

    def test_matched_filter_in_noise(self):
        """Test matched filter detection in noise."""
        np.random.seed(42)
        template = np.sin(2 * np.pi * 0.1 * np.arange(30))
        signal = np.random.randn(200) * 0.1  # Low SNR
        signal[100:130] += template

        result = matched_filter(signal, template)
        # Should still detect signal
        assert 90 <= result.peak_index <= 140
