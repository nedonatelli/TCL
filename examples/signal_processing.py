"""
Signal Processing Example.

This example demonstrates:
1. Digital filter design (Butterworth, Chebyshev, FIR)
2. Matched filtering for pulse detection
3. CFAR (Constant False Alarm Rate) detection
4. Power spectrum analysis

Run with: python examples/signal_processing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.mathematical_functions.signal_processing import (  # noqa: E402
    apply_filter,
    butter_design,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    cheby1_design,
    detection_probability,
    filtfilt,
    fir_design,
    frequency_response,
    matched_filter,
    pulse_compression,
    threshold_factor,
)


def filter_design_demo() -> None:
    """Demonstrate digital filter design."""
    print("=" * 60)
    print("1. DIGITAL FILTER DESIGN")
    print("=" * 60)

    # Sample rate
    fs = 1000.0  # Hz

    # Design Butterworth lowpass filter
    print("\nButterworth Lowpass Filter (order=4, cutoff=50 Hz):")
    butter_filt = butter_design(order=4, cutoff=50.0, fs=fs, btype="low")
    print(f"  Numerator coefficients (b): {butter_filt.b[:3]}...")
    print(f"  Denominator coefficients (a): {butter_filt.a[:3]}...")

    # Design Chebyshev Type I filter
    print("\nChebyshev Type I Lowpass (order=4, ripple=1dB, cutoff=50 Hz):")
    cheby_filt = cheby1_design(order=4, ripple=1.0, cutoff=50.0, fs=fs, btype="low")
    print(f"  Numerator coefficients (b): {cheby_filt.b[:3]}...")

    # Design FIR filter
    print("\nFIR Lowpass Filter (numtaps=51, cutoff=50 Hz):")
    fir_coeffs = fir_design(numtaps=51, cutoff=50.0, fs=fs, window="hamming")
    print(f"  Number of taps: {len(fir_coeffs)}")
    print(f"  Center tap value: {fir_coeffs[25]:.6f}")

    # Compare frequency responses
    print("\nFrequency Response Comparison:")
    butter_resp = frequency_response(butter_filt.b, butter_filt.a, fs)
    cheby_resp = frequency_response(cheby_filt.b, cheby_filt.a, fs)

    # Find -3dB point for Butterworth
    butter_mag_db = 20 * np.log10(np.maximum(butter_resp.magnitude, 1e-10))
    idx_3db = np.argmin(np.abs(butter_mag_db + 3))
    print(f"  Butterworth -3dB frequency: {butter_resp.frequencies[idx_3db]:.1f} Hz")

    # Find -3dB point for Chebyshev
    cheby_mag_db = 20 * np.log10(np.maximum(cheby_resp.magnitude, 1e-10))
    idx_3db = np.argmin(np.abs(cheby_mag_db + 3))
    print(f"  Chebyshev -3dB frequency: {cheby_resp.frequencies[idx_3db]:.1f} Hz")


def filtering_demo() -> None:
    """Demonstrate signal filtering."""
    print("\n" + "=" * 60)
    print("2. SIGNAL FILTERING")
    print("=" * 60)

    np.random.seed(42)

    # Create a test signal: 10 Hz sine + 60 Hz noise
    fs = 1000.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    signal_clean = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
    noise = 0.5 * np.sin(2 * np.pi * 60 * t)  # 60 Hz noise
    signal_noisy = signal_clean + noise + 0.1 * np.random.randn(len(t))

    print("\nTest signal: 10 Hz sine wave + 60 Hz interference + white noise")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: 1 second ({len(t)} samples)")

    # Design 30 Hz lowpass filter to remove 60 Hz noise
    filt = butter_design(order=4, cutoff=30.0, fs=fs, btype="low")

    # Apply filter using different methods
    print("\nFiltering methods:")

    # Standard filter (has phase shift)
    _filtered_standard = apply_filter(filt, signal_noisy)  # noqa: F841
    print("  Standard filtering: introduces phase shift")

    # Zero-phase filter (no phase shift)
    filtered_zerophase = filtfilt(filt, signal_noisy)
    print("  Zero-phase filtering: no phase shift (filtfilt)")

    # Compute SNR improvement
    noise_power_before = np.var(signal_noisy - signal_clean)
    noise_power_after = np.var(filtered_zerophase - signal_clean)
    snr_improvement = 10 * np.log10(noise_power_before / noise_power_after)
    print(f"\n  SNR improvement: {snr_improvement:.1f} dB")

    # RMS error
    rms_before = np.sqrt(np.mean((signal_noisy - signal_clean) ** 2))
    rms_after = np.sqrt(np.mean((filtered_zerophase - signal_clean) ** 2))
    print(f"  RMS error before filtering: {rms_before:.4f}")
    print(f"  RMS error after filtering:  {rms_after:.4f}")


def matched_filter_demo() -> None:
    """Demonstrate matched filtering for pulse detection."""
    print("\n" + "=" * 60)
    print("3. MATCHED FILTERING")
    print("=" * 60)

    np.random.seed(42)

    # Create a chirp pulse
    fs = 10000.0  # Sample rate
    T = 0.01  # Pulse duration (10 ms)
    f0 = 500  # Start frequency
    f1 = 2000  # End frequency

    t_pulse = np.linspace(0, T, int(T * fs), endpoint=False)
    # Linear frequency sweep (chirp)
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) / (2 * T) * t_pulse) * t_pulse)

    print("\nChirp pulse parameters:")
    print(f"  Duration: {T * 1000:.1f} ms")
    print(f"  Frequency sweep: {f0} Hz to {f1} Hz")
    print(f"  Bandwidth: {f1 - f0} Hz")
    print(f"  Time-bandwidth product: {(f1 - f0) * T:.1f}")

    # Create received signal with delayed pulse in noise
    n_samples = int(0.1 * fs)  # 100 ms of data
    received = 0.5 * np.random.randn(n_samples)  # Noise

    # Add pulse at known location with attenuation
    pulse_location = 500
    pulse_amplitude = 0.3
    received[pulse_location : pulse_location + len(chirp)] += pulse_amplitude * chirp

    # Matched filter
    result = matched_filter(received, chirp, normalize=True)

    print("\nMatched filter results:")
    print(f"  True pulse location: sample {pulse_location}")
    print(f"  Detected peak location: sample {result.peak_index}")
    print(f"  Detection error: {abs(result.peak_index - pulse_location)} samples")
    print(f"  Peak correlation value: {result.peak_value:.4f}")
    print(f"  SNR gain: {result.snr_gain:.1f} dB")

    # Pulse compression
    pc_result = pulse_compression(received, chirp)
    compressed_peak = np.argmax(np.abs(pc_result.output))
    print("\nPulse compression:")
    print(f"  Compressed pulse peak: sample {compressed_peak}")
    print(f"  Compression ratio: {pc_result.compression_ratio:.0f}:1")


def cfar_detection_demo() -> None:
    """Demonstrate CFAR detection algorithms."""
    print("\n" + "=" * 60)
    print("4. CFAR DETECTION")
    print("=" * 60)

    np.random.seed(42)

    # Create a range profile with targets
    n_cells = 200
    noise_power = 1.0
    noise = np.sqrt(noise_power) * np.abs(np.random.randn(n_cells))

    # Add targets at known locations
    targets = [
        (50, 15.0),  # Location, amplitude (strong target)
        (100, 8.0),  # Medium target
        (150, 5.0),  # Weak target
    ]

    signal = noise.copy()
    for loc, amp in targets:
        signal[loc] = amp

    print("\nSimulated range profile:")
    print(f"  {n_cells} range cells")
    print(f"  Noise power: {noise_power:.1f}")
    print(f"  Targets at cells: {[t[0] for t in targets]}")
    print(f"  Target amplitudes: {[t[1] for t in targets]}")

    # CFAR parameters
    guard_cells = 2
    ref_cells = 8
    pfa = 1e-4  # Probability of false alarm

    print("\nCFAR parameters:")
    print(f"  Guard cells: {guard_cells}")
    print(f"  Reference cells: {ref_cells}")
    print(f"  Pfa: {pfa}")

    # Compute threshold factor
    alpha = threshold_factor(pfa, ref_cells, method="ca")
    print(f"  Threshold factor (CA-CFAR): {alpha:.2f}")

    # Run different CFAR algorithms
    print("\nCFAR Detection Results:")
    print("-" * 60)

    # Cell-Averaging CFAR
    ca_result = cfar_ca(signal, guard_cells, ref_cells, pfa)
    ca_detections = ca_result.detection_indices
    print("\nCA-CFAR (Cell-Averaging):")
    print(f"  Detections: {ca_detections.tolist()}")
    print(
        f"  Targets detected: {len(set(ca_detections) & {t[0] for t in targets})}/{len(targets)}"
    )

    # Greatest-Of CFAR (good at clutter edges)
    go_result = cfar_go(signal, guard_cells, ref_cells, pfa)
    go_detections = go_result.detection_indices
    print("\nGO-CFAR (Greatest-Of):")
    print(f"  Detections: {go_detections.tolist()}")

    # Smallest-Of CFAR (good in clutter)
    so_result = cfar_so(signal, guard_cells, ref_cells, pfa)
    so_detections = so_result.detection_indices
    print("\nSO-CFAR (Smallest-Of):")
    print(f"  Detections: {so_detections.tolist()}")

    # Order-Statistic CFAR
    k = int(0.75 * ref_cells)  # Use 75th percentile
    os_result = cfar_os(signal, guard_cells, ref_cells, pfa, k)
    os_detections = os_result.detection_indices
    print(f"\nOS-CFAR (Order-Statistic, k={k}):")
    print(f"  Detections: {os_detections.tolist()}")

    # Detection probability analysis
    print("\nDetection Probability vs SNR:")
    print("-" * 40)
    snr_values = [5, 10, 15, 20]
    for snr in snr_values:
        pd = detection_probability(snr, pfa, ref_cells, method="ca")
        print(f"  SNR = {snr:2d} dB: Pd = {pd:.4f}")


def spectrum_analysis_demo() -> None:
    """Demonstrate power spectrum analysis."""
    print("\n" + "=" * 60)
    print("5. SPECTRUM ANALYSIS")
    print("=" * 60)

    np.random.seed(42)

    # Create a multi-tone signal
    fs = 1000.0
    t = np.linspace(0, 1, int(fs), endpoint=False)

    # Signal with known frequency components
    frequencies = [50, 120, 200]  # Hz
    amplitudes = [1.0, 0.5, 0.3]

    signal = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)

    # Add noise
    signal += 0.2 * np.random.randn(len(t))

    print("\nTest signal:")
    print(f"  Sample rate: {fs} Hz")
    print("  Duration: 1 second")
    print(f"  Frequency components: {frequencies} Hz")
    print(f"  Amplitudes: {amplitudes}")

    # Compute power spectrum using FFT
    from pytcl.mathematical_functions.transforms import power_spectrum

    ps = power_spectrum(signal, fs, window="hann", nperseg=256)

    print("\nPower spectrum analysis:")
    print(f"  Frequency resolution: {fs / 256:.2f} Hz")

    # Find peaks in spectrum
    psd_db = 10 * np.log10(np.maximum(ps.psd, 1e-10))
    peak_threshold = np.max(psd_db) - 20  # Peaks within 20 dB of max

    print(f"\nDetected frequency peaks (>{peak_threshold:.1f} dB):")
    for i in range(1, len(psd_db) - 1):
        if psd_db[i] > psd_db[i - 1] and psd_db[i] > psd_db[i + 1]:
            if psd_db[i] > peak_threshold:
                print(f"  {ps.frequencies[i]:.1f} Hz: {psd_db[i]:.1f} dB")


def main() -> None:
    """Run signal processing demonstrations."""
    print("\nSignal Processing Examples")
    print("=" * 60)
    print("Demonstrating pytcl signal processing capabilities")

    filter_design_demo()
    filtering_demo()
    matched_filter_demo()
    cfar_detection_demo()
    spectrum_analysis_demo()

    # Visualization
    visualize_filter_response()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def visualize_filter_response() -> None:
    """Visualize digital filter frequency response."""
    print("\nGenerating filter response visualization...")

    fs = 1000.0
    butter_filt = butter_design(order=4, cutoff=50.0, fs=fs, btype="low")
    fir_filt = fir_design(order=64, cutoff=50.0, fs=fs, btype="low")

    # Generate frequency response
    freq = np.linspace(0, 500, 1000)
    w = 2 * np.pi * freq / fs

    # Compute magnitude responses
    butter_mag = np.abs(
        frequency_response(butter_filt.b, butter_filt.a, w)
    )
    fir_mag = np.abs(frequency_response(fir_filt.b, [1.0], w))

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=20 * np.log10(np.maximum(butter_mag, 1e-10)),
            mode="lines",
            name="Butterworth (4th order)",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=20 * np.log10(np.maximum(fir_mag, 1e-10)),
            mode="lines",
            name="FIR (order 64)",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Digital Filter Frequency Response Comparison",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        height=500,
        width=800,
        hovermode="x unified",
    )

    fig.show()


if __name__ == "__main__":
    main()
