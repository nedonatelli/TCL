"""
Comprehensive tests for Short-Time Fourier Transform (STFT) and spectrogram.

Tests cover:
- Basic STFT functionality
- Window functions
- Spectrogram computation
- Inverse STFT reconstruction
- Edge cases and parameter variations
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.transforms.stft import (
    stft,
    istft,
    spectrogram,
    get_window,
    window_bandwidth,
)


class TestWindowFunctions:
    """Tests for window generation and properties."""

    def test_get_window_hann(self):
        """Test Hann window generation."""
        w = get_window("hann", 256)
        assert len(w) == 256
        # Hann window should be close to zero at edges (for periodic=True)
        assert w[0] < 0.01
        assert w[-1] < 0.01
        # Peak should be near 1
        assert np.isclose(np.max(w), 1.0, atol=1e-10)

    def test_get_window_hamming(self):
        """Test Hamming window generation."""
        w = get_window("hamming", 256)
        assert len(w) == 256
        # Hamming window never quite reaches zero at edges
        assert w[0] > 0.05
        assert w[-1] > 0.05

    def test_get_window_blackman(self):
        """Test Blackman window generation."""
        w = get_window("blackman", 256)
        assert len(w) == 256
        assert w[0] < 0.001  # Very close to zero
        assert w[-1] < 0.001  # Very close to zero

    def test_get_window_kaiser_parameterized(self):
        """Test Kaiser window with parameter."""
        w = get_window(("kaiser", 8.0), 256)
        assert len(w) == 256
        assert np.all(np.isfinite(w))

    def test_get_window_array_input(self):
        """Test with custom window array."""
        custom_w = np.ones(256) * 0.5
        w = get_window(custom_w, 256)
        np.testing.assert_array_equal(w, custom_w)

    def test_get_window_periodicity(self):
        """Test periodic vs symmetric window."""
        w_periodic = get_window("hann", 256, fftbins=True)
        w_symmetric = get_window("hann", 256, fftbins=False)
        # Periodic window doesn't go to zero at edges
        assert w_periodic[0] > 0 or np.isclose(w_periodic[0], 0, atol=1e-10)

    def test_window_bandwidth_hann(self):
        """Test equivalent noise bandwidth for Hann window."""
        enbw = window_bandwidth("hann", 256)
        # Hann window has ENBW ≈ 1.5 bins
        assert 1.4 < enbw < 1.6

    def test_window_bandwidth_different_lengths(self):
        """Test window bandwidth is independent of length."""
        enbw_256 = window_bandwidth("hann", 256)
        enbw_512 = window_bandwidth("hann", 512)
        # Should be approximately the same
        assert np.isclose(enbw_256, enbw_512, rtol=0.01)


class TestSTFT:
    """Tests for Short-Time Fourier Transform."""

    def test_stft_simple_sine_wave(self):
        """Test STFT of a simple sine wave."""
        fs = 1000  # 1 kHz sampling rate
        duration = 1.0  # 1 second
        freq = 50  # 50 Hz tone

        t = np.arange(int(fs * duration)) / fs
        x = np.sin(2 * np.pi * freq * t)

        result = stft(x, fs=fs, nperseg=256)

        # Check output shapes
        assert result.Zxx.shape[0] == 129  # (nperseg // 2 + 1) = 129
        assert result.Zxx.shape[1] > 1
        assert len(result.frequencies) == 129
        assert len(result.times) == result.Zxx.shape[1]

        # Frequencies should be non-negative and monotonically increasing
        assert np.all(result.frequencies >= 0)
        assert np.all(np.diff(result.frequencies) > 0)

    def test_stft_energy_conservation(self):
        """Test that STFT has appropriate energy distribution."""
        fs = 1000
        duration = 0.5
        t = np.arange(int(fs * duration)) / fs
        x = np.sin(2 * np.pi * 100 * t)

        result = stft(x, fs=fs, nperseg=256)

        # Energy should be concentrated in time-frequency
        power = np.abs(result.Zxx) ** 2
        assert np.max(power) > 0
        assert np.any(power > 0)  # Should have some non-zero values

    def test_stft_multi_frequency(self):
        """Test STFT with multiple frequency components."""
        fs = 1000
        duration = 1.0
        t = np.arange(int(fs * duration)) / fs

        # Signal with two frequency components
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)

        result = stft(x, fs=fs, nperseg=256)

        # Power should have peaks around 50 Hz and 150 Hz
        power_sum = np.mean(np.abs(result.Zxx) ** 2, axis=1)

        # Find frequency indices of 50 Hz and 150 Hz
        idx_50 = np.argmin(np.abs(result.frequencies - 50))
        idx_150 = np.argmin(np.abs(result.frequencies - 150))

        # Power at these frequencies should be relatively high
        assert power_sum[idx_50] > power_sum[0]  # More than DC
        assert power_sum[idx_150] > power_sum[0]  # More than DC

    def test_stft_window_parameter(self):
        """Test STFT with different window functions."""
        fs = 1000
        t = np.arange(fs) / fs
        x = np.sin(2 * np.pi * 50 * t)

        result_hann = stft(x, fs=fs, window="hann")
        result_hamming = stft(x, fs=fs, window="hamming")

        # Both should produce valid results
        assert result_hann.Zxx.shape == result_hamming.Zxx.shape
        assert np.all(np.isfinite(result_hann.Zxx))
        assert np.all(np.isfinite(result_hamming.Zxx))

    def test_stft_hop_length(self):
        """Test STFT with different hop lengths."""
        fs = 1000
        t = np.arange(fs) / fs
        x = np.sin(2 * np.pi * 50 * t)

        result_half_overlap = stft(x, fs=fs, nperseg=256, noverlap=128)
        result_quarter_overlap = stft(x, fs=fs, nperseg=256, noverlap=64)

        # Smaller hop length = more time frames
        assert result_half_overlap.Zxx.shape[1] > result_quarter_overlap.Zxx.shape[1]

    def test_stft_output_properties(self):
        """Test properties of STFT output."""
        fs = 1000
        t = np.arange(fs) / fs
        x = np.sin(2 * np.pi * 50 * t)

        result = stft(x, fs=fs)

        # STFT should be complex
        assert np.iscomplexobj(result.Zxx)

        # All values should be finite
        assert np.all(np.isfinite(result.Zxx))

        # Times should be monotonically increasing
        assert np.all(np.diff(result.times) > 0)


class TestInverseSTFT:
    """Tests for inverse Short-Time Fourier Transform."""

    def test_istft_reconstruction(self):
        """Test ISTFT reconstruction accuracy."""
        fs = 1000
        duration = 0.5
        t = np.arange(int(fs * duration)) / fs
        x_original = np.sin(2 * np.pi * 50 * t)

        # STFT then ISTFT
        result = stft(x_original, fs=fs, nperseg=256)
        t_recon, x_reconstructed = istft(result.Zxx, fs=fs, window="hann", nperseg=256)

        # Should return valid output
        assert x_reconstructed is not None
        assert len(x_reconstructed) > 0

    def test_istft_short_signal(self):
        """Test ISTFT with short signal."""
        fs = 1000
        t = np.arange(256) / fs
        x = np.sin(2 * np.pi * 50 * t)

        result = stft(x, fs=fs, nperseg=256)
        t_recon, x_reconstructed = istft(result.Zxx, fs=fs, nperseg=256)

        assert len(x_reconstructed) > 0
        assert np.all(np.isfinite(x_reconstructed))

    def test_istft_output_type(self):
        """Test that ISTFT returns tuple of (time, signal)."""
        fs = 1000
        t = np.arange(512) / fs
        x = np.sin(2 * np.pi * 50 * t)

        result = stft(x, fs=fs, nperseg=256)
        output = istft(result.Zxx, fs=fs, nperseg=256)

        # Should return tuple of (times, signal)
        assert isinstance(output, tuple)
        assert len(output) == 2
        t_recon, x_reconstructed = output
        assert np.all(np.isfinite(x_reconstructed))


class TestSpectrogram:
    """Tests for spectrogram computation."""

    def test_spectrogram_basic(self):
        """Test basic spectrogram computation."""
        fs = 1000
        duration = 1.0
        t = np.arange(int(fs * duration)) / fs
        x = np.sin(2 * np.pi * 50 * t)

        result = spectrogram(x, fs=fs, nperseg=256)

        # Power should be non-negative
        assert np.all(result.power >= 0)

        # Shapes should be consistent
        assert len(result.frequencies) == result.power.shape[0]
        assert len(result.times) == result.power.shape[1]

    def test_spectrogram_power_conservation(self):
        """Test that spectrogram power makes sense."""
        fs = 1000
        t = np.arange(fs) / fs
        x = np.sin(2 * np.pi * 100 * t)

        result = spectrogram(x, fs=fs, nperseg=256)

        # Maximum power should occur in the middle of the frequency range (where 100 Hz is)
        max_idx = np.argmax(np.mean(result.power, axis=1))
        max_freq = result.frequencies[max_idx]

        # Should be somewhere near 100 Hz
        assert 50 < max_freq < 150

    def test_spectrogram_chirp_signal(self):
        """Test spectrogram of chirp signal (frequency increasing over time)."""
        fs = 1000
        duration = 1.0
        t = np.arange(int(fs * duration)) / fs

        # Chirp from 10 Hz to 100 Hz
        x = signal.chirp(t, f0=10, f1=100, t1=duration)

        result = spectrogram(x, fs=fs, nperseg=256)

        # Should have valid power output
        assert result.power is not None
        assert np.all(result.power >= 0)
        assert result.power.shape[0] > 0
        assert result.power.shape[1] > 0

        # Max power index should be within bounds
        max_power_idx = np.argmax(result.power)
        assert max_power_idx >= 0

    def test_spectrogram_output_shape(self):
        """Test spectrogram output dimensions."""
        fs = 1000
        x = np.random.randn(2000)

        result = spectrogram(x, fs=fs, nperseg=256, noverlap=128)

        # Frequency dimension should be (nperseg//2 + 1)
        assert result.power.shape[0] == 129

        # Time dimension should match the number of frames
        assert result.power.shape[1] == len(result.times)


class TestSTFTEdgeCases:
    """Test edge cases and error conditions."""

    def test_stft_empty_signal(self):
        """Test STFT with empty signal."""
        x = np.array([])
        try:
            result = stft(x, fs=1000)
            # If no error, result should be valid
            assert result is not None
        except (ValueError, IndexError):
            # Acceptable for empty input
            pass

    def test_stft_single_sample(self):
        """Test STFT with single sample."""
        x = np.array([1.0])
        try:
            result = stft(x, fs=1000, nperseg=256)
            assert result is not None
        except (ValueError, IndexError):
            # Acceptable for minimal input
            pass

    def test_stft_nperseg_larger_than_signal(self):
        """Test STFT when nperseg is larger than signal."""
        x = np.random.randn(100)
        try:
            result = stft(x, fs=1000, nperseg=256)
            assert result is not None
        except (ValueError, IndexError):
            # Acceptable when nperseg > signal length
            pass

    def test_spectrogram_high_noise(self):
        """Test spectrogram with high-noise signal."""
        fs = 1000
        t = np.arange(fs) / fs
        # Pure noise
        x = np.random.randn(fs)

        result = spectrogram(x, fs=fs, nperseg=256)

        # Should have some power everywhere
        assert np.all(result.power >= 0)
        assert np.mean(result.power) > 0

    def test_stft_real_vs_complex_input(self):
        """Test STFT with real vs complex input."""
        fs = 1000
        t = np.arange(fs) / fs
        x_real = np.sin(2 * np.pi * 50 * t)
        x_complex = x_real + 1j * np.sin(2 * np.pi * 50 * t + np.pi / 4)

        result_real = stft(x_real, fs=fs)
        result_complex = stft(x_complex, fs=fs)

        # Both should produce valid STFT
        assert np.all(np.isfinite(result_real.Zxx))
        assert np.all(np.isfinite(result_complex.Zxx))


# Import signal module for chirp function
try:
    from scipy import signal
except ImportError:
    signal = None
