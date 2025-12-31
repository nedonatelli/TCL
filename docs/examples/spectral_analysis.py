#!/usr/bin/env python3
"""
Spectral Analysis Example
=========================

Demonstrate power spectrum and spectrogram computation.
"""

import numpy as np

from pytcl.mathematical_functions.transforms import (
    istft,
    power_spectrum,
    spectrogram,
    stft,
)


def main():
    np.random.seed(42)

    print("Spectral Analysis Examples")
    print("=" * 50)

    # Sample rate
    fs = 1000  # Hz
    duration = 2.0
    t = np.arange(0, duration, 1 / fs)

    # Signal with two tones plus noise
    f1, f2 = 50, 120  # Hz
    signal = (
        np.sin(2 * np.pi * f1 * t)
        + 0.5 * np.sin(2 * np.pi * f2 * t)
        + 0.2 * np.random.randn(len(t))
    )

    # Power spectrum
    psd = power_spectrum(signal, fs, window="hann", nperseg=256)

    # Find peaks
    peak_indices = np.argsort(psd.psd)[-5:]
    peak_freqs = psd.frequencies[peak_indices]

    print("\nPower Spectrum Analysis")
    print("-" * 40)
    print(f"True frequencies: {f1} Hz, {f2} Hz")
    print(f"Detected peaks at: {peak_freqs[-2:]:.1f} Hz")

    # Chirp signal for spectrogram
    f0, f1_chirp = 10, 200
    chirp = np.sin(2 * np.pi * (f0 + (f1_chirp - f0) * t / (2 * duration)) * t)

    # Spectrogram
    spec = spectrogram(chirp, fs, nperseg=128, noverlap=120)

    print("\nSpectrogram Analysis (Chirp Signal)")
    print("-" * 40)
    print(f"Time bins: {len(spec.times)}")
    print(f"Frequency bins: {len(spec.frequencies)}")
    print(f"Max frequency: {spec.frequencies.max():.1f} Hz")

    # Find instantaneous frequency at each time
    inst_freq = []
    for i in range(spec.power.shape[1]):
        max_idx = np.argmax(spec.power[:, i])
        inst_freq.append(spec.frequencies[max_idx])

    print(f"Frequency sweep: {inst_freq[0]:.1f} -> {inst_freq[-1]:.1f} Hz")

    # STFT round-trip
    stft_result = stft(signal, fs, nperseg=256, noverlap=128)
    reconstructed = istft(stft_result.Zxx, fs, nperseg=256, noverlap=128)

    # Check reconstruction error
    min_len = min(len(signal), len(reconstructed))
    recon_error = np.mean(np.abs(signal[:min_len] - reconstructed[:min_len]))

    print("\nSTFT Round-Trip Test")
    print("-" * 40)
    print(f"Reconstruction error: {recon_error:.2e}")


if __name__ == "__main__":
    main()
