"""
Benchmarks for signal processing and transform functions.

Covers FFT, STFT, wavelets, matched filtering, and pulse compression.
"""

import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def signal_test_data():
    """Pre-computed test signals for benchmarks."""
    np.random.seed(42)

    # Generate test signals of various lengths
    fs = 44100  # Audio sampling rate
    return {
        # Short signals
        "signal_256": np.random.randn(256),
        "signal_1024": np.random.randn(1024),
        # Medium signals
        "signal_4096": np.random.randn(4096),
        "signal_16384": np.random.randn(16384),
        # Long signals
        "signal_65536": np.random.randn(65536),
        # Complex signal
        "complex_4096": np.random.randn(4096) + 1j * np.random.randn(4096),
        # Sampling frequency
        "fs": fs,
        # 2D signal for 2D FFT
        "signal_2d_64": np.random.randn(64, 64),
        "signal_2d_256": np.random.randn(256, 256),
    }


@pytest.fixture(scope="session")
def chirp_test_data():
    """Pre-computed chirp signals for matched filter benchmarks."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        generate_lfm_chirp,
    )

    fs = 10000
    return {
        "chirp_short": generate_lfm_chirp(0.01, 500, 2000, fs),  # 10ms chirp
        "chirp_medium": generate_lfm_chirp(0.05, 500, 2000, fs),  # 50ms chirp
        "chirp_long": generate_lfm_chirp(0.1, 500, 2000, fs),  # 100ms chirp
        "fs": fs,
    }


# =============================================================================
# FFT Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("n", [256, 1024, 4096])
def test_fft_1d(benchmark, signal_test_data, n):
    """Benchmark 1D FFT for various signal lengths."""
    from pytcl.mathematical_functions.transforms.fourier import fft

    signal = signal_test_data[f"signal_{n}"]

    result = benchmark(fft, signal)
    assert len(result) == n


@pytest.mark.benchmark
@pytest.mark.full
@pytest.mark.parametrize("n", [16384, 65536])
def test_fft_1d_large(benchmark, signal_test_data, n):
    """Benchmark 1D FFT for large signals."""
    from pytcl.mathematical_functions.transforms.fourier import fft

    signal = signal_test_data[f"signal_{n}"]

    result = benchmark(fft, signal)
    assert len(result) == n


@pytest.mark.benchmark
@pytest.mark.light
def test_rfft_1d(benchmark, signal_test_data):
    """Benchmark real FFT (optimized for real signals)."""
    from pytcl.mathematical_functions.transforms.fourier import rfft

    signal = signal_test_data["signal_4096"]

    result = benchmark(rfft, signal)
    assert len(result) == 2049  # n//2 + 1


@pytest.mark.benchmark
@pytest.mark.light
def test_fft_2d_small(benchmark, signal_test_data):
    """Benchmark 2D FFT for small images."""
    from pytcl.mathematical_functions.transforms.fourier import fft2

    signal = signal_test_data["signal_2d_64"]

    result = benchmark(fft2, signal)
    assert result.shape == (64, 64)


@pytest.mark.benchmark
@pytest.mark.full
def test_fft_2d_large(benchmark, signal_test_data):
    """Benchmark 2D FFT for larger images."""
    from pytcl.mathematical_functions.transforms.fourier import fft2

    signal = signal_test_data["signal_2d_256"]

    result = benchmark(fft2, signal)
    assert result.shape == (256, 256)


# =============================================================================
# Power Spectrum Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_power_spectrum_welch(benchmark, signal_test_data):
    """Benchmark Welch power spectrum estimation."""
    from pytcl.mathematical_functions.transforms.fourier import power_spectrum

    signal = signal_test_data["signal_4096"]
    fs = signal_test_data["fs"]

    result = benchmark(power_spectrum, signal, fs=fs)
    assert len(result.frequencies) > 0
    assert len(result.psd) == len(result.frequencies)


@pytest.mark.benchmark
@pytest.mark.light
def test_periodogram(benchmark, signal_test_data):
    """Benchmark periodogram computation."""
    from pytcl.mathematical_functions.transforms.fourier import periodogram

    signal = signal_test_data["signal_4096"]
    fs = signal_test_data["fs"]

    result = benchmark(periodogram, signal, fs=fs)
    assert len(result.psd) > 0


@pytest.mark.benchmark
@pytest.mark.full
def test_coherence(benchmark, signal_test_data):
    """Benchmark coherence estimation between two signals."""
    from pytcl.mathematical_functions.transforms.fourier import coherence

    signal1 = signal_test_data["signal_4096"]
    signal2 = signal_test_data["signal_4096"] * 0.8 + np.random.randn(4096) * 0.2
    fs = signal_test_data["fs"]

    result = benchmark(coherence, signal1, signal2, fs=fs)
    assert np.all(result.coherence >= 0)
    assert np.all(result.coherence <= 1)


# =============================================================================
# STFT Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_stft_small(benchmark, signal_test_data):
    """Benchmark STFT for moderate signal length."""
    from pytcl.mathematical_functions.transforms.stft import stft

    signal = signal_test_data["signal_4096"]
    fs = signal_test_data["fs"]

    result = benchmark(stft, signal, fs=fs, nperseg=256)
    assert len(result.frequencies) > 0
    assert len(result.times) > 0


@pytest.mark.benchmark
@pytest.mark.full
def test_stft_large(benchmark, signal_test_data):
    """Benchmark STFT for larger signal."""
    from pytcl.mathematical_functions.transforms.stft import stft

    signal = signal_test_data["signal_16384"]
    fs = signal_test_data["fs"]

    result = benchmark(stft, signal, fs=fs, nperseg=512)
    assert result.Zxx.shape[1] > 0


@pytest.mark.benchmark
@pytest.mark.light
def test_spectrogram(benchmark, signal_test_data):
    """Benchmark spectrogram computation."""
    from pytcl.mathematical_functions.transforms.stft import spectrogram

    signal = signal_test_data["signal_4096"]
    fs = signal_test_data["fs"]

    result = benchmark(spectrogram, signal, fs=fs, nperseg=256)
    assert np.all(result.power >= 0)


@pytest.mark.benchmark
@pytest.mark.full
def test_istft_roundtrip(benchmark, signal_test_data):
    """Benchmark inverse STFT (roundtrip)."""
    from pytcl.mathematical_functions.transforms.stft import istft, stft

    signal = signal_test_data["signal_4096"]
    fs = signal_test_data["fs"]

    # Pre-compute STFT
    stft_result = stft(signal, fs=fs, nperseg=256)

    result = benchmark(istft, stft_result.Zxx, fs=fs, nperseg=256)
    assert len(result) > 0


# =============================================================================
# Wavelet Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_cwt_morlet(benchmark, signal_test_data):
    """Benchmark Continuous Wavelet Transform with Morlet wavelet."""
    from pytcl.mathematical_functions.transforms.wavelets import cwt

    signal = signal_test_data["signal_1024"]
    scales = np.arange(1, 64)

    result = benchmark(cwt, signal, scales, wavelet="morlet")
    assert result.coefficients.shape == (len(scales), len(signal))


@pytest.mark.benchmark
@pytest.mark.full
def test_cwt_large(benchmark, signal_test_data):
    """Benchmark CWT for larger signal."""
    from pytcl.mathematical_functions.transforms.wavelets import cwt

    signal = signal_test_data["signal_4096"]
    scales = np.arange(1, 128)

    result = benchmark(cwt, signal, scales, wavelet="morlet")
    assert result.coefficients.shape[0] == len(scales)


@pytest.mark.benchmark
@pytest.mark.light
def test_morlet_wavelet_generation(benchmark):
    """Benchmark Morlet wavelet generation."""
    from pytcl.mathematical_functions.transforms.wavelets import morlet_wavelet

    result = benchmark(morlet_wavelet, 256, w=5.0)
    assert len(result) == 256


@pytest.mark.benchmark
@pytest.mark.light
def test_ricker_wavelet_generation(benchmark):
    """Benchmark Ricker wavelet generation."""
    from pytcl.mathematical_functions.transforms.wavelets import ricker_wavelet

    result = benchmark(ricker_wavelet, 256, 10.0)
    assert len(result) == 256


# =============================================================================
# Matched Filter Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_matched_filter_time(benchmark, chirp_test_data):
    """Benchmark time-domain matched filtering."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        matched_filter,
    )

    chirp = chirp_test_data["chirp_short"]
    # Create signal with embedded chirp
    signal = np.zeros(len(chirp) * 3)
    signal[len(chirp) : 2 * len(chirp)] = chirp

    result = benchmark(matched_filter, signal, chirp)
    assert result.peak_index > 0


@pytest.mark.benchmark
@pytest.mark.light
def test_matched_filter_frequency(benchmark, chirp_test_data):
    """Benchmark frequency-domain matched filtering."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        matched_filter_frequency,
    )

    chirp = chirp_test_data["chirp_medium"]
    signal = np.zeros(len(chirp) * 3)
    signal[len(chirp) : 2 * len(chirp)] = chirp

    result = benchmark(
        matched_filter_frequency, signal, chirp, fs=chirp_test_data["fs"]
    )
    assert result.peak_index > 0


@pytest.mark.benchmark
@pytest.mark.full
def test_pulse_compression(benchmark, chirp_test_data):
    """Benchmark pulse compression."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        pulse_compression,
    )

    chirp = chirp_test_data["chirp_medium"]
    signal = np.zeros(len(chirp) * 3)
    signal[len(chirp) : 2 * len(chirp)] = chirp

    result = benchmark(pulse_compression, signal, chirp)
    assert result.compression_ratio > 1


@pytest.mark.benchmark
@pytest.mark.full
def test_ambiguity_function(benchmark, chirp_test_data):
    """Benchmark ambiguity function computation (Numba-accelerated)."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        ambiguity_function,
    )

    chirp = chirp_test_data["chirp_short"]
    fs = chirp_test_data["fs"]

    # Warm up JIT
    _ = ambiguity_function(chirp, fs, n_delay=32, n_doppler=32)

    result = benchmark(ambiguity_function, chirp, fs, n_delay=64, n_doppler=64)
    delays, dopplers, af = result
    assert af.shape == (64, 64)


# =============================================================================
# Chirp Generation Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_generate_lfm_chirp(benchmark):
    """Benchmark LFM chirp generation."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        generate_lfm_chirp,
    )

    result = benchmark(generate_lfm_chirp, 0.1, 500, 5000, 44100)
    assert len(result) == 4410


@pytest.mark.benchmark
@pytest.mark.light
def test_generate_nlfm_chirp(benchmark):
    """Benchmark NLFM chirp generation."""
    from pytcl.mathematical_functions.signal_processing.matched_filter import (
        generate_nlfm_chirp,
    )

    result = benchmark(generate_nlfm_chirp, 0.1, 500, 5000, 44100, beta=2.0)
    assert len(result) == 4410
