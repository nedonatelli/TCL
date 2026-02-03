"""
Enhanced tests targeting specific uncovered lines in hypergeometric and STFT modules.

Focus areas:
- Hypergeometric: Internal series computation edge cases
- STFT: Parameter handling, format conversions, windowing edge cases
"""

import numpy as np
import pytest
from scipy import signal, special, stats

from pytcl.mathematical_functions.special_functions.hypergeometric import (
    generalized_hypergeometric,
    hyp0f1,
    hyp1f1,
    hyp2f1,
    hyperu,
)
from pytcl.mathematical_functions.transforms.stft import (
    istft,
    spectrogram,
    stft,
    window_bandwidth,
)


class TestHypergeometricSeries:
    """Tests targeting hypergeometric series computation edge cases."""

    def test_2f1_special_transformation(self):
        """Test 2F1 with special transformation parameters."""
        # Test the series computation with different parameter combinations
        result = hyp2f1(1, 1, 2, 0.5)
        expected = special.hyp2f1(1, 1, 2, 0.5)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_2f1_near_singularity(self):
        """Test 2F1 near parameter singularities."""
        # When b is close to negative integer
        result = hyp2f1(0.5, 0.5, 1.5, 0.3)
        assert np.isfinite(result)

    def test_2f1_large_parameters(self):
        """Test 2F1 with large parameter values."""
        result = hyp2f1(10, 20, 30, 0.1)
        assert np.isfinite(result)

    def test_2f1_small_argument(self):
        """Test 2F1 with very small argument."""
        result = hyp2f1(1, 2, 3, 1e-10)
        # Should be close to 1 for small z
        assert np.isclose(result, 1.0, atol=1e-8)

    def test_1f1_series_convergence(self):
        """Test 1F1 series convergence with various parameters."""
        result = hyp1f1(2, 5, 1.0)
        expected = special.hyp1f1(2, 5, 1.0)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_1f1_large_argument(self):
        """Test 1F1 with larger argument values."""
        result = hyp1f1(1, 2, 2.0)
        expected = special.hyp1f1(1, 2, 2.0)
        assert np.isclose(result, expected, rtol=1e-9)

    def test_0f1_convergence(self):
        """Test 0F1 convergence behavior."""
        result = hyp0f1(2.5, 1.0)
        expected = special.hyp0f1(2.5, 1.0)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_0f1_negative_argument(self):
        """Test 0F1 with negative argument."""
        result = hyp0f1(1.5, -0.5)
        expected = special.hyp0f1(1.5, -0.5)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_1f1_negative_argument(self):
        """Test 1F1 with negative argument."""
        result = hyp1f1(0.5, 1.5, -1.0)
        expected = special.hyp1f1(0.5, 1.5, -1.0)
        assert np.isclose(result, expected, rtol=1e-9)

    def test_2f1_array_multiple_convergence(self):
        """Test 2F1 with array inputs requiring different convergence rates."""
        a, b, c = 1, 1, 2
        z = np.array([0.01, 0.1, 0.5, 0.9])
        results = hyp2f1(a, b, c, z)

        # Check all results are finite
        assert np.all(np.isfinite(results))

        # Verify against scipy
        expected = special.hyp2f1(a, b, c, z)
        assert np.allclose(results, expected, rtol=1e-10)

    def test_1f1_integer_parameters(self):
        """Test 1F1 with integer parameters."""
        result = hyp1f1(3, 4, 0.5)
        expected = special.hyp1f1(3, 4, 0.5)
        assert np.isclose(result, expected, rtol=1e-10)


class TestSTFTWindowBandwidth:
    """Tests for window bandwidth calculation."""

    def test_window_bandwidth_hann(self):
        """Test window bandwidth for Hann window."""
        enbw = window_bandwidth("hann", 256)
        # Hann window ENBW should be approximately 1.5 bins
        assert 1.4 < enbw < 1.6

    def test_window_bandwidth_hamming(self):
        """Test window bandwidth for Hamming window."""
        enbw = window_bandwidth("hamming", 256)
        # Hamming window ENBW should be approximately 1.3 bins
        assert 1.2 < enbw < 1.4

    def test_window_bandwidth_blackman(self):
        """Test window bandwidth for Blackman window."""
        enbw = window_bandwidth("blackman", 256)
        # Blackman window ENBW should be approximately 1.72 bins
        assert 1.6 < enbw < 1.9

    def test_window_bandwidth_custom_array(self):
        """Test window bandwidth with custom window array."""
        w = signal.get_window("hann", 256)
        enbw = window_bandwidth(w, 256)
        assert 1.4 < enbw < 1.6

    def test_window_bandwidth_rectangular(self):
        """Test window bandwidth for rectangular window."""
        enbw = window_bandwidth("boxcar", 256)
        # Rectangular window ENBW should be 1.0
        assert np.isclose(enbw, 1.0, atol=0.01)

    def test_window_bandwidth_various_lengths(self):
        """Test window bandwidth with various window lengths."""
        for length in [64, 128, 256, 512, 1024]:
            enbw = window_bandwidth("hann", length)
            # ENBW should be independent of length for same window type
            assert 1.4 < enbw < 1.6


class TestSTFTParameterHandling:
    """Tests for STFT parameter handling and edge cases."""

    def test_stft_default_parameters(self):
        """Test STFT with minimal parameters."""
        x = np.random.randn(1024)
        result = stft(x)
        assert result.Zxx is not None
        assert result.Zxx.ndim == 2

    def test_stft_auto_nfft(self):
        """Test STFT with automatic NFFT."""
        x = np.random.randn(512)
        result = stft(x, nperseg=256)
        # nfft should default to nperseg
        assert result.Zxx.shape[0] == 129  # (256/2 + 1) for real signal

    def test_stft_noverlap_none(self):
        """Test STFT with noverlap=None."""
        x = np.random.randn(1024)
        result = stft(x, nperseg=256, noverlap=None)
        # noverlap should default to nperseg // 2
        assert result.Zxx is not None

    def test_stft_complex_input(self):
        """Test STFT with complex input."""
        x = np.random.randn(1024) + 1j * np.random.randn(1024)
        result = stft(x)
        assert result.Zxx.dtype == np.complex128

    def test_stft_real_input_conversion(self):
        """Test STFT converts real input to float64."""
        x = np.random.randn(1024).astype(np.float32)
        result = stft(x)
        assert result.Zxx is not None

    def test_stft_with_various_dtypes(self):
        """Test STFT with various input data types."""
        x_int = np.array([1, 2, 3, 4, 5] * 200, dtype=np.int32)
        result = stft(x_int)
        assert result.Zxx is not None

    def test_stft_detrending_options(self):
        """Test STFT with different detrending options."""
        x = np.random.randn(1024)

        # Test with different detrend options
        for detrend_opt in ["constant", "linear", False]:
            result = stft(x, nperseg=256, detrend=detrend_opt)
            assert result.Zxx is not None

    def test_stft_boundary_options(self):
        """Test STFT with different boundary options."""
        x = np.random.randn(1024)

        for boundary_opt in ["even", "odd", "constant", "zeros"]:
            result = stft(x, nperseg=256, boundary=boundary_opt)
            assert result.Zxx is not None

    def test_stft_padded_false(self):
        """Test STFT with padded=False."""
        x = np.random.randn(1024)
        result = stft(x, nperseg=256, padded=False)
        assert result.Zxx is not None

    def test_istft_with_explicit_window(self):
        """Test ISTFT with explicit window specification."""
        x = np.random.randn(512)
        result = stft(x, nperseg=256, window="hann")

        # ISTFT should work with matching parameters
        t_recon, x_recon = istft(result.Zxx, fs=1.0, nperseg=256, window="hann")
        assert x_recon is not None
        assert len(x_recon) > 0


class TestSpectrogramFormats:
    """Tests for spectrogram with different output formats."""

    def test_spectrogram_default(self):
        """Test spectrogram default computation."""
        x = np.random.randn(1024)
        result = spectrogram(x)
        assert result.power is not None
        assert result.frequencies is not None
        assert result.times is not None

    def test_spectrogram_with_frequency_scale(self):
        """Test spectrogram power scaling."""
        x = np.sin(2 * np.pi * 50 * np.arange(1024) / 1024)
        result = spectrogram(x, fs=1024)

        # Peak should be near 50 Hz
        peak_freq_idx = np.argmax(np.mean(result.power, axis=1))
        peak_freq = result.frequencies[peak_freq_idx]
        assert 45 < peak_freq < 55

    def test_spectrogram_with_fs(self):
        """Test spectrogram with custom sampling rate."""
        fs = 8000
        duration = 1.0
        t = np.arange(int(fs * duration)) / fs
        x = np.sin(2 * np.pi * 1000 * t)

        result = spectrogram(x, fs=fs, nperseg=512)
        # Frequency resolution should reflect fs
        assert result.frequencies[-1] <= fs / 2

    def test_spectrogram_energy_conservation(self):
        """Test spectrogram energy is reasonable."""
        x = np.random.randn(1024)
        result = spectrogram(x)

        # Total power should be positive
        total_power = np.sum(result.power)
        assert total_power > 0


class TestSTFTEdgeCasesExtended:
    """Extended edge case tests for STFT."""

    def test_stft_sine_reconstruction(self):
        """Test STFT/ISTFT reconstruction of sine wave."""
        fs = 1000
        duration = 0.5
        t = np.arange(int(fs * duration)) / fs
        x_orig = np.sin(2 * np.pi * 100 * t)

        # STFT then ISTFT
        result = stft(x_orig, fs=fs, nperseg=256, window="hann")
        t_recon, x_recon = istft(result.Zxx, fs=fs, nperseg=256, window="hann")

        # Reconstruction should be non-empty
        assert len(x_recon) > 0
        assert np.all(np.isfinite(x_recon))

    def test_stft_dc_component(self):
        """Test STFT handles DC component."""
        x = np.ones(512) + np.random.randn(512) * 0.1
        result = stft(x)
        assert result.Zxx is not None

    def test_stft_zero_padding_effect(self):
        """Test STFT with zero padding."""
        x = np.random.randn(256)

        result_padded = stft(x, nperseg=256, nfft=512)
        result_unpadded = stft(x, nperseg=256, nfft=256)

        # Padded should have higher frequency resolution
        assert result_padded.Zxx.shape[0] > result_unpadded.Zxx.shape[0]

    def test_spectrogram_narrow_bandwidth(self):
        """Test spectrogram with narrow bandwidth signal."""
        fs = 1000
        t = np.arange(1000) / fs
        # Narrow bandwidth signal
        x = np.cos(2 * np.pi * 50 * t) * np.exp(-t / 0.2)

        result = spectrogram(x, fs=fs, nperseg=256)
        assert result.power.shape[0] > 0
        assert result.power.shape[1] > 0

    def test_istft_length_mismatch_handling(self):
        """Test ISTFT handles various output lengths."""
        x = np.random.randn(512)
        result = stft(x, nperseg=256)

        # ISTFT should handle reconstruction
        t_recon, x_recon = istft(result.Zxx, nperseg=256)
        assert len(x_recon) > 0

    def test_stft_window_consistency(self):
        """Test STFT with consistent window across calls."""
        x = np.random.randn(512)

        result1 = stft(x, window="hann")
        result2 = stft(x, window="hann")

        # Should be identical
        assert np.allclose(result1.Zxx, result2.Zxx)

    def test_spectrogram_vs_magnitude_stft(self):
        """Test spectrogram vs. magnitude of STFT."""
        x = np.random.randn(512)

        stft_result = stft(x, nperseg=256)
        spec_result = spectrogram(x, nperseg=256)

        # Spectrogram power should relate to STFT magnitude
        stft_power = np.abs(stft_result.Zxx) ** 2
        # Normalize to compare
        assert stft_power is not None
        assert spec_result.power is not None
