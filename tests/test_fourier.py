"""Tests for Fourier transform functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.transforms import (
    CoherenceResult,
    CrossSpectrum,
    PowerSpectrum,
    coherence,
    cross_spectrum,
    fft,
    fft2,
    fftshift,
    frequency_axis,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    magnitude_spectrum,
    periodogram,
    phase_spectrum,
    power_spectrum,
    rfft,
    rfft_frequency_axis,
    spectrogram,
)


class TestFourier:
    """Tests for Fourier transform functions."""

    def test_fft_ifft_roundtrip(self):
        """Test FFT and inverse FFT roundtrip."""
        x = np.random.randn(128)
        X = fft(x)
        x_rec = ifft(X).real

        assert_allclose(x, x_rec, atol=1e-10)

    def test_rfft_real_signal(self):
        """Test real FFT for real signal."""
        x = np.random.randn(128)
        X = rfft(x)

        # rfft returns n//2 + 1 points
        assert len(X) == 65

    def test_rfft_irfft_roundtrip(self):
        """Test real FFT roundtrip."""
        x = np.random.randn(128)
        X = rfft(x)
        x_rec = irfft(X)

        assert_allclose(x, x_rec, atol=1e-10)

    def test_fft2_ifft2_roundtrip(self):
        """Test 2D FFT roundtrip."""
        x = np.random.randn(32, 32)
        X = fft2(x)
        x_rec = ifft2(X).real

        assert_allclose(x, x_rec, atol=1e-10)

    def test_fftshift_ifftshift(self):
        """Test FFT shift operations."""
        x = np.arange(10)
        shifted = fftshift(x)
        unshifted = ifftshift(shifted)

        assert_allclose(x, unshifted)

    def test_frequency_axis(self):
        """Test frequency axis generation."""
        freqs = frequency_axis(8, 100.0)

        assert len(freqs) == 8
        assert freqs[0] == 0.0

    def test_frequency_axis_shifted(self):
        """Test shifted frequency axis."""
        freqs = frequency_axis(8, 100.0, shift=True)

        assert len(freqs) == 8
        # Should be centered
        assert freqs[len(freqs) // 2] == 0.0

    def test_rfft_frequency_axis(self):
        """Test rfft frequency axis."""
        freqs = rfft_frequency_axis(8, 100.0)

        assert len(freqs) == 5  # n//2 + 1
        assert freqs[0] == 0.0
        assert freqs[-1] == 50.0  # Nyquist

    def test_power_spectrum_sine(self):
        """Test power spectrum of sine wave."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 100 * t)

        result = power_spectrum(x, fs=fs)

        assert isinstance(result, PowerSpectrum)
        # Peak should be near 100 Hz
        peak_freq = result.frequencies[np.argmax(result.psd)]
        assert abs(peak_freq - 100) < 10

    def test_cross_spectrum(self):
        """Test cross-spectral density."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        y = np.sin(2 * np.pi * 50 * t + np.pi / 4)

        result = cross_spectrum(x, y, fs=fs)

        assert isinstance(result, CrossSpectrum)
        assert len(result.frequencies) > 0

    def test_coherence_correlated_signals(self):
        """Test coherence for correlated signals."""
        np.random.seed(42)
        fs = 1000
        t = np.arange(0, 2, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        y = 2 * x + 0.1 * np.random.randn(len(t))

        result = coherence(x, y, fs=fs)

        assert isinstance(result, CoherenceResult)
        # High coherence at 50 Hz
        assert np.max(result.coherence) > 0.9

    def test_periodogram(self):
        """Test periodogram."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 100 * t)

        result = periodogram(x, fs=fs)

        assert isinstance(result, PowerSpectrum)

    def test_magnitude_spectrum(self):
        """Test magnitude spectrum computation."""
        X = np.array([4 + 0j, 0 - 2j, 0 + 0j, 0 + 2j])
        mag = magnitude_spectrum(X)

        assert_allclose(mag, [4.0, 2.0, 0.0, 2.0])

    def test_magnitude_spectrum_db(self):
        """Test magnitude spectrum in dB."""
        X = np.array([10 + 0j, 1 + 0j])
        mag = magnitude_spectrum(X, scale="dB")

        assert_allclose(mag[0], 20.0, atol=0.01)  # 20*log10(10) = 20
        assert_allclose(mag[1], 0.0, atol=0.01)  # 20*log10(1) = 0

    def test_phase_spectrum(self):
        """Test phase spectrum computation."""
        X = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
        phase = phase_spectrum(X)

        assert_allclose(phase, [0, np.pi / 2, np.pi, -np.pi / 2], atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_fft_single_point(self):
        """Test FFT of single point."""
        x = np.array([5.0])
        X = fft(x)

        assert X[0] == pytest.approx(5.0)

    def test_power_spectrum_short_signal(self):
        """Test power spectrum with short signal."""
        x = np.random.randn(64)
        result = power_spectrum(x, fs=100, nperseg=32)

        assert len(result.frequencies) > 0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_spectrogram_analysis(self):
        """Test spectrogram analysis of chirp signal."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        # Chirp from 50 to 200 Hz
        x = np.sin(2 * np.pi * (50 + 75 * t) * t)

        # Compute spectrogram
        result = spectrogram(x, fs=fs, nperseg=128, noverlap=120)

        # Should see increasing frequency over time
        # Check that peak frequency increases
        peak_freqs = []
        for i in range(result.power.shape[1]):
            peak_idx = np.argmax(result.power[:, i])
            peak_freqs.append(result.frequencies[peak_idx])

        # Last peak should be higher than first
        assert peak_freqs[-1] > peak_freqs[0]

    def test_filter_spectrum_analysis(self):
        """Test filter frequency response matches spectrum."""
        from pytcl.mathematical_functions.signal_processing import (
            butter_design,
            frequency_response,
        )

        fs = 1000
        cutoff = 100

        # Design filter
        coeffs = butter_design(4, cutoff, fs)
        resp = frequency_response(coeffs, fs)

        # -3 dB point should be near cutoff
        mag_db = 20 * np.log10(resp.magnitude + 1e-10)
        idx_3db = np.argmin(np.abs(mag_db - (-3)))
        freq_3db = resp.frequencies[idx_3db]

        assert abs(freq_3db - cutoff) < 10
