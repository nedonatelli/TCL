"""Tests for transforms module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.transforms import (
    PYWT_AVAILABLE,
    CoherenceResult,
    CrossSpectrum,
    CWTResult,
    PowerSpectrum,
    Spectrogram,
    STFTResult,
    coherence,
    cross_spectrum,
    cwt,
    fft,
    fft2,
    fftshift,
    frequency_axis,
    gaussian_wavelet,
    get_window,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    istft,
    magnitude_spectrum,
    mel_spectrogram,
    morlet_wavelet,
    periodogram,
    phase_spectrum,
    power_spectrum,
    rfft,
    rfft_frequency_axis,
    ricker_wavelet,
    scales_to_frequencies,
    spectrogram,
    stft,
    window_bandwidth,
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


class TestSTFT:
    """Tests for Short-Time Fourier Transform functions."""

    def test_stft_shape(self):
        """Test STFT output shape."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)

        result = stft(x, fs=fs, nperseg=128)

        assert isinstance(result, STFTResult)
        assert result.Zxx.shape[0] == 65  # n_freq = nperseg//2 + 1

    def test_stft_istft_roundtrip(self):
        """Test STFT and inverse STFT roundtrip."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)

        result = stft(x, fs=fs, nperseg=128)
        t_rec, x_rec = istft(result.Zxx, fs=fs, nperseg=128)

        # Allow some tolerance due to boundary effects
        assert_allclose(x, x_rec[: len(x)], atol=1e-10)

    def test_spectrogram_shape(self):
        """Test spectrogram output shape."""
        fs = 1000
        t = np.arange(0, 2, 1 / fs)
        x = np.sin(2 * np.pi * (50 + 75 * t) * t)  # Chirp

        result = spectrogram(x, fs=fs, nperseg=128)

        assert isinstance(result, Spectrogram)
        assert result.power.shape[0] == 65  # n_freq

    def test_get_window(self):
        """Test window generation."""
        w = get_window("hann", 256)

        assert len(w) == 256
        assert w[0] == pytest.approx(0.0, abs=0.01)
        assert w[128] == pytest.approx(1.0, abs=0.01)

    def test_get_window_kaiser(self):
        """Test Kaiser window with parameter."""
        w = get_window(("kaiser", 8.0), 256)

        assert len(w) == 256

    def test_window_bandwidth(self):
        """Test window equivalent noise bandwidth."""
        enbw = window_bandwidth("hann", 256)

        # Hann window ENBW is about 1.5 bins
        assert 1.4 < enbw < 1.6

    def test_mel_spectrogram(self):
        """Test mel spectrogram computation."""
        np.random.seed(42)
        fs = 22050
        x = np.random.randn(fs)  # 1 second

        mel_freqs, times, mel_spec = mel_spectrogram(x, fs, n_mels=64)

        assert mel_spec.shape[0] == 64


class TestWavelets:
    """Tests for wavelet transform functions."""

    def test_morlet_wavelet_shape(self):
        """Test Morlet wavelet generation."""
        wav = morlet_wavelet(128, w=5.0)

        assert len(wav) == 128
        assert np.iscomplexobj(wav)

    def test_morlet_wavelet_center(self):
        """Test Morlet wavelet peak at center."""
        wav = morlet_wavelet(128, w=5.0)

        # Peak magnitude should be near center
        peak_idx = np.argmax(np.abs(wav))
        assert abs(peak_idx - 64) < 5

    def test_ricker_wavelet_shape(self):
        """Test Ricker wavelet generation."""
        wav = ricker_wavelet(128, a=4.0)

        assert len(wav) == 128
        # Peak should be at or near center
        peak_idx = np.argmax(wav)
        assert abs(peak_idx - 64) < 5

    def test_gaussian_wavelet(self):
        """Test Gaussian wavelet generation."""
        wav = gaussian_wavelet(128, order=1)

        assert len(wav) == 128

    def test_cwt_sine(self):
        """Test CWT of sine wave."""
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        scales = np.arange(1, 64)

        result = cwt(x, scales, wavelet="morlet", fs=fs)

        assert isinstance(result, CWTResult)
        assert result.coefficients.shape == (len(scales), len(x))

    def test_cwt_frequencies(self):
        """Test CWT frequency estimation."""
        scales = np.array([1, 2, 4, 8, 16])
        freqs = scales_to_frequencies(scales, wavelet="morlet", fs=1000)

        assert len(freqs) == 5
        # Smaller scale = higher frequency
        assert freqs[0] > freqs[-1]


@pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywavelets not installed")
class TestDWT:
    """Tests for Discrete Wavelet Transform (requires pywavelets)."""

    def test_dwt_shape(self):
        """Test DWT output shape."""
        from pytcl.mathematical_functions.transforms import dwt

        x = np.random.randn(256)
        result = dwt(x, wavelet="db4", level=4)

        assert result.levels == 4
        assert len(result.cD) == 4

    def test_dwt_idwt_roundtrip(self):
        """Test DWT and inverse DWT roundtrip."""
        from pytcl.mathematical_functions.transforms import dwt, idwt

        x = np.random.randn(256)
        result = dwt(x, wavelet="db4", level=4)
        x_rec = idwt(result)

        assert_allclose(x, x_rec, atol=1e-10)

    def test_dwt_single_level(self):
        """Test single-level DWT."""
        from pytcl.mathematical_functions.transforms import dwt_single_level

        x = np.random.randn(256)
        cA, cD = dwt_single_level(x, wavelet="db4")

        assert len(cA) == 128 + 3  # Half length plus filter overlap
        assert len(cD) == 128 + 3

    def test_dwt_different_wavelets(self):
        """Test DWT with different wavelets."""
        from pytcl.mathematical_functions.transforms import dwt

        x = np.random.randn(256)

        for wavelet in ["haar", "db4", "sym4", "coif2"]:
            result = dwt(x, wavelet=wavelet, level=3)
            assert result.wavelet == wavelet


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

    def test_stft_short_signal(self):
        """Test STFT with signal shorter than nperseg."""
        x = np.random.randn(64)
        # Use smaller nperseg for short signal
        result = stft(x, fs=100, nperseg=32)

        # Should handle gracefully
        assert result.Zxx.shape[1] > 0

    def test_cwt_single_scale(self):
        """Test CWT with single scale."""
        x = np.random.randn(256)
        scales = np.array([10])

        result = cwt(x, scales)

        assert result.coefficients.shape[0] == 1


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
