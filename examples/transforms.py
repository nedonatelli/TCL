"""
Transforms Example.

This example demonstrates:
1. FFT and power spectrum analysis
2. Short-time Fourier Transform (STFT)
3. Spectrogram visualization
4. Wavelet transforms (CWT, DWT)

Run with: python examples/transforms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.mathematical_functions.transforms import (  # noqa: E402
    coherence,
    cross_spectrum,
    cwt,
    dwt,
    fft,
    idwt,
    ifft,
    istft,
    morlet_wavelet,
    power_spectrum,
    rfft,
    ricker_wavelet,
    spectrogram,
    stft,
)


def fft_demo() -> None:
    """Demonstrate FFT operations."""
    print("=" * 60)
    print("1. FFT AND SPECTRAL ANALYSIS")
    print("=" * 60)

    np.random.seed(42)

    # Create a test signal with known frequency content
    fs = 1000.0  # Sample rate
    t = np.linspace(0, 1, int(fs), endpoint=False)

    # Multi-tone signal
    f1, f2, f3 = 50, 120, 200  # Hz
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    signal += 0.25 * np.sin(2 * np.pi * f3 * t)

    print(f"\nTest signal: sum of sinusoids at {f1}, {f2}, {f3} Hz")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: 1 second ({len(t)} samples)")

    # Compute FFT
    X = fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)

    print("\nFFT Results:")
    print(f"  FFT length: {len(X)}")
    print(f"  Frequency resolution: {fs / len(signal):.2f} Hz")

    # Find peaks in positive frequencies
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_mag = np.abs(X[pos_mask])

    # Find significant peaks
    peak_threshold = 0.1 * np.max(pos_mag)
    print("\nSignificant frequency peaks:")
    for i in range(1, len(pos_mag) - 1):
        if pos_mag[i] > pos_mag[i - 1] and pos_mag[i] > pos_mag[i + 1]:
            if pos_mag[i] > peak_threshold:
                print(f"  {pos_freqs[i]:.1f} Hz: magnitude = {pos_mag[i]:.2f}")

    # Verify inverse FFT
    signal_recovered = ifft(X).real
    reconstruction_error = np.max(np.abs(signal - signal_recovered))
    print(f"\nIFFT reconstruction error: {reconstruction_error:.2e}")

    # Real FFT (more efficient for real signals)
    print("\nReal FFT (rfft):")
    X_real = rfft(signal)
    print(f"  rfft length: {len(X_real)} (vs {len(X)} for full fft)")
    print(f"  Memory savings: {100 * (1 - len(X_real) / len(X)):.1f}%")


def power_spectrum_demo() -> None:
    """Demonstrate power spectrum estimation."""
    print("\n" + "=" * 60)
    print("2. POWER SPECTRUM ESTIMATION")
    print("=" * 60)

    np.random.seed(42)

    # Create a noisy signal with known spectrum
    fs = 1000.0
    t = np.linspace(0, 2, int(2 * fs), endpoint=False)

    # Signal with two frequency components plus noise
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    signal += 0.5 * np.random.randn(len(t))

    print("\nSignal: 50 Hz + 120 Hz sinusoids in noise")
    print(f"  SNR (approx): {10 * np.log10(1.25 / 0.25):.1f} dB")

    # Welch's method power spectrum
    ps = power_spectrum(signal, fs, window="hann", nperseg=256)

    print("\nPower Spectrum (Welch's method):")
    print(f"  Number of frequency bins: {len(ps.frequencies)}")
    print(f"  Frequency resolution: {ps.frequencies[1] - ps.frequencies[0]:.2f} Hz")

    # Find peaks
    psd_db = 10 * np.log10(np.maximum(ps.psd, 1e-10))
    print("\nPeak frequencies:")
    for i in range(1, len(psd_db) - 1):
        if psd_db[i] > psd_db[i - 1] and psd_db[i] > psd_db[i + 1]:
            if psd_db[i] > np.max(psd_db) - 20:
                print(f"  {ps.frequencies[i]:.1f} Hz: {psd_db[i]:.1f} dB")


def cross_spectrum_demo() -> None:
    """Demonstrate cross-spectrum and coherence."""
    print("\n" + "=" * 60)
    print("3. CROSS-SPECTRUM AND COHERENCE")
    print("=" * 60)

    np.random.seed(42)

    # Create two related signals
    fs = 1000.0
    t = np.linspace(0, 2, int(2 * fs), endpoint=False)

    # Common component
    common = np.sin(2 * np.pi * 50 * t)

    # Independent noise
    noise1 = 0.5 * np.random.randn(len(t))
    noise2 = 0.5 * np.random.randn(len(t))

    # Signal 1: common + independent noise
    x = common + noise1
    # Signal 2: common (delayed) + independent noise
    delay_samples = 10
    y = np.roll(common, delay_samples) + noise2

    print("\nTwo signals with common 50 Hz component:")
    print("  Signal x: 50 Hz + noise")
    print(f"  Signal y: 50 Hz (delayed {delay_samples} samples) + noise")

    # Cross-spectrum
    csd = cross_spectrum(x, y, fs, window="hann")
    print("\nCross-spectrum:")
    print(f"  Frequency bins: {len(csd.frequencies)}")

    # Find phase at 50 Hz
    idx_50hz = np.argmin(np.abs(csd.frequencies - 50))
    phase_50hz = np.angle(csd.csd[idx_50hz])
    expected_phase = -2 * np.pi * 50 * delay_samples / fs
    print("\nPhase at 50 Hz:")
    print(f"  Measured: {np.degrees(phase_50hz):.1f} deg")
    print(f"  Expected: {np.degrees(expected_phase):.1f} deg")

    # Coherence
    coh = coherence(x, y, fs, nperseg=256)
    print("\nCoherence at 50 Hz:")
    idx_coh = np.argmin(np.abs(coh.frequencies - 50))
    print(f"  Coherence: {coh.coherence[idx_coh]:.3f}")
    print("  (1.0 = perfectly correlated, 0.0 = uncorrelated)")


def stft_demo() -> None:
    """Demonstrate Short-Time Fourier Transform."""
    print("\n" + "=" * 60)
    print("4. SHORT-TIME FOURIER TRANSFORM (STFT)")
    print("=" * 60)

    np.random.seed(42)

    # Create a chirp signal (frequency varies with time)
    fs = 1000.0
    duration = 2.0
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Linear chirp from 50 Hz to 200 Hz
    f0, f1 = 50, 200
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) / (2 * duration) * t) * t)

    print(f"\nChirp signal: frequency sweeps from {f0} to {f1} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {fs} Hz")

    # Compute STFT
    nperseg = 128
    noverlap = nperseg // 2
    result = stft(chirp, fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    print("\nSTFT Results:")
    print(f"  Segment length: {nperseg} samples")
    print(f"  Overlap: {noverlap} samples")
    print(f"  Number of time frames: {len(result.times)}")
    print(f"  Number of frequency bins: {len(result.frequencies)}")
    print(f"  Time resolution: {result.times[1] - result.times[0]:.3f} s")
    print(f"  Frequency resolution: {result.frequencies[1] - result.frequencies[0]:.2f} Hz")

    # Verify reconstruction with inverse STFT
    t_rec, reconstructed = istft(result.Zxx, fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    min_len = min(len(chirp), len(reconstructed))
    recon_error = np.sqrt(np.mean((chirp[:min_len] - reconstructed[:min_len]) ** 2))
    print(f"\nReconstruction RMS error: {recon_error:.6f}")

    # Track instantaneous frequency over time
    print("\nInstantaneous frequency tracking:")
    for i in range(0, len(result.times), len(result.times) // 5):
        # Find peak frequency at this time
        mag = np.abs(result.Zxx[:, i])
        peak_idx = np.argmax(mag)
        peak_freq = result.frequencies[peak_idx]
        expected_freq = f0 + (f1 - f0) * result.times[i] / duration
        print(
            f"  t={result.times[i]:.2f}s: measured={peak_freq:.1f} Hz, "
            f"expected={expected_freq:.1f} Hz"
        )


def spectrogram_demo() -> None:
    """Demonstrate spectrogram computation."""
    print("\n" + "=" * 60)
    print("5. SPECTROGRAM")
    print("=" * 60)

    np.random.seed(42)

    # Create a signal with time-varying frequency content
    fs = 1000.0
    duration = 3.0
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Three segments with different frequencies
    signal = np.zeros_like(t)
    signal[t < 1] = np.sin(2 * np.pi * 50 * t[t < 1])  # 50 Hz
    signal[(t >= 1) & (t < 2)] = np.sin(2 * np.pi * 100 * t[(t >= 1) & (t < 2)])  # 100 Hz
    signal[t >= 2] = np.sin(2 * np.pi * 150 * t[t >= 2])  # 150 Hz

    print("\nTime-varying signal:")
    print("  0-1 s: 50 Hz")
    print("  1-2 s: 100 Hz")
    print("  2-3 s: 150 Hz")

    # Compute spectrogram
    spec = spectrogram(signal, fs, window="hann", nperseg=256, noverlap=128)

    print("\nSpectrogram dimensions:")
    print(f"  Time bins: {len(spec.times)}")
    print(f"  Frequency bins: {len(spec.frequencies)}")

    # Find dominant frequency in each time segment
    print("\nDominant frequencies by segment:")
    time_points = [0.5, 1.5, 2.5]  # Center of each segment
    for tp in time_points:
        idx = np.argmin(np.abs(spec.times - tp))
        power_slice = spec.power[:, idx]
        peak_idx = np.argmax(power_slice)
        print(f"  t={tp}s: {spec.frequencies[peak_idx]:.1f} Hz")


def wavelet_demo() -> None:
    """Demonstrate wavelet transforms."""
    print("\n" + "=" * 60)
    print("6. WAVELET TRANSFORMS")
    print("=" * 60)

    np.random.seed(42)

    # Create a signal with a transient event
    fs = 1000.0
    t = np.linspace(0, 1, int(fs), endpoint=False)

    # Background oscillation + transient pulse
    signal = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz background
    # Add transient at t=0.5s
    transient_center = 0.5
    transient_width = 0.02
    transient = np.exp(-((t - transient_center) ** 2) / (2 * transient_width**2))
    signal += transient

    print("\nTest signal: 10 Hz oscillation + Gaussian pulse at t=0.5s")

    # Generate wavelet shapes
    print("\nWavelet shapes:")
    morlet = morlet_wavelet(64, w=5.0)
    ricker = ricker_wavelet(64, a=8.0)
    print(f"  Morlet wavelet: {len(morlet)} points")
    print(f"  Ricker (Mexican hat) wavelet: {len(ricker)} points")

    # Continuous Wavelet Transform
    scales = np.arange(1, 64)
    cwt_result = cwt(signal, scales, wavelet="morlet", fs=fs)

    print("\nContinuous Wavelet Transform (CWT):")
    print(f"  Number of scales: {len(cwt_result.scales)}")
    print(
        f"  Frequency range: {cwt_result.frequencies[-1]:.1f} to {cwt_result.frequencies[0]:.1f} Hz"
    )

    # Find the transient in CWT
    cwt_mag = np.abs(cwt_result.coefficients)
    max_idx = np.unravel_index(np.argmax(cwt_mag), cwt_mag.shape)
    peak_time = t[max_idx[1]]
    peak_freq = cwt_result.frequencies[max_idx[0]]
    print("\nTransient detection:")
    print(f"  Peak at time: {peak_time:.3f} s (true: {transient_center} s)")
    print(f"  Peak frequency: {peak_freq:.1f} Hz")

    # Discrete Wavelet Transform
    print("\nDiscrete Wavelet Transform (DWT):")
    dwt_result = dwt(signal, wavelet="db4", level=4)
    print("  Wavelet: Daubechies 4 (db4)")
    print(f"  Decomposition levels: {dwt_result.levels}")
    print(f"  Approximation coefficients: {len(dwt_result.cA)} samples")
    for i, cD in enumerate(dwt_result.cD):
        print(f"  Detail level {i + 1}: {len(cD)} samples")

    # Verify reconstruction
    reconstructed = idwt(dwt_result)
    min_len = min(len(signal), len(reconstructed))
    recon_error = np.sqrt(np.mean((signal[:min_len] - reconstructed[:min_len]) ** 2))
    print(f"\nDWT reconstruction RMS error: {recon_error:.6f}")


def main() -> None:
    """Run transform demonstrations."""
    print("\nTransforms Examples")
    print("=" * 60)
    print("Demonstrating pytcl transform capabilities")

    fft_demo()
    power_spectrum_demo()
    cross_spectrum_demo()
    stft_demo()
    spectrogram_demo()
    wavelet_demo()

    # Visualization
    visualize_fft_analysis()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def visualize_fft_analysis() -> None:
    """Visualize FFT analysis of multi-frequency signal."""
    print("\nGenerating FFT analysis visualization...")

    # Create test signal
    fs = 1000.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    f1, f2, f3 = 50, 120, 200

    signal = (
        np.sin(2 * np.pi * f1 * t)
        + 0.5 * np.sin(2 * np.pi * f2 * t)
        + 0.25 * np.sin(2 * np.pi * f3 * t)
    )
    signal += 0.1 * np.random.randn(len(t))

    # Compute FFT
    X = fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)

    # Only positive frequencies
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_mag = np.abs(X[pos_mask])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Time Domain Signal", "Frequency Domain (FFT)"),
    )

    # Time domain
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            mode="lines",
            name="Signal",
            line=dict(color="blue", width=1),
        ),
        row=1,
        col=1,
    )

    # Frequency domain
    fig.add_trace(
        go.Scatter(
            x=pos_freqs[:500],
            y=pos_mag[:500],
            mode="lines",
            name="Magnitude",
            line=dict(color="red", width=1),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)

    fig.update_layout(
        title="FFT Analysis: Multi-Frequency Signal",
        height=500,
        width=1000,
        showlegend=False,
    )

    fig.show()


if __name__ == "__main__":
    main()
