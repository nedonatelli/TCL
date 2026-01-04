"""Signal Processing Tutorial with Interactive Visualizations.

This tutorial demonstrates the use of digital signal processing techniques
including filtering, frequency analysis, and time-frequency decomposition.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, lfilter, spectrogram, stft

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def signal_processing_tutorial():
    """Run complete signal processing tutorial with visualizations."""

    print("=" * 70)
    print("SIGNAL PROCESSING TUTORIAL")
    print("=" * 70)

    # Step 1: Generate Synthetic Signal
    print("\nStep 1: Generate Synthetic Signal")
    print("-" * 70)

    fs = 1000  # Sampling frequency (Hz)
    duration = 5  # Duration (seconds)
    t = np.arange(0, duration, 1 / fs)

    # Create signal with multiple frequency components
    # Clean signal: 50 Hz sinusoid
    s1 = 1.0 * np.sin(2 * np.pi * 50 * t)

    # Noise: 150 Hz sinusoid
    s2 = 0.5 * np.sin(2 * np.pi * 150 * t)

    # Additional noise: 200 Hz sinusoid
    s3 = 0.3 * np.sin(2 * np.pi * 200 * t)

    # Gaussian white noise
    noise = 0.2 * np.random.randn(len(t))

    # Combined signal
    signal = s1 + s2 + s3 + noise

    print(f"Signal created: {len(t)} samples at {fs} Hz")
    print(
        f"Signal components: 50 Hz (1.0), 150 Hz (0.5), 200 Hz (0.3), Gaussian noise (0.2)"
    )

    # Step 2: Frequency Domain Analysis
    print("\nStep 2: Frequency Domain Analysis")
    print("-" * 70)

    # Compute FFT
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    magnitude = np.abs(fft_signal) / len(signal)

    # Only positive frequencies
    positive_idx = freqs > 0
    freqs_pos = freqs[positive_idx]
    magnitude_pos = magnitude[positive_idx]

    print(f"FFT computed: {len(freqs)} frequency points")

    # Step 3: Design and Apply Butterworth Filter
    print("\nStep 3: Design and Apply Butterworth Low-Pass Filter")
    print("-" * 70)

    # Design low-pass filter (cutoff at 100 Hz)
    cutoff = 100
    order = 5
    normalized_cutoff = cutoff / (fs / 2)

    b, a = butter(order, normalized_cutoff, btype="low")

    # Apply filter
    filtered_signal = lfilter(b, a, signal)

    print(f"Filter designed: Order {order}, Cutoff {cutoff} Hz")
    print(f"Expected: Remove 150 Hz and 200 Hz components")

    # Analyze filtered signal
    fft_filtered = np.fft.fft(filtered_signal)
    magnitude_filtered = np.abs(fft_filtered) / len(filtered_signal)
    magnitude_filtered_pos = magnitude_filtered[positive_idx]

    # Step 4: Time-Frequency Analysis
    print("\nStep 4: Time-Frequency Analysis (Spectrogram)")
    print("-" * 70)

    # Compute spectrogram
    f, t_spec, Sxx = spectrogram(signal, fs, nperseg=256, noverlap=200)

    print(
        f"Spectrogram computed: {f.shape[0]} frequency bins, {Sxx.shape[1]} time bins"
    )

    # Compute spectrogram of filtered signal
    f_filt, t_spec_filt, Sxx_filt = spectrogram(
        filtered_signal, fs, nperseg=256, noverlap=200
    )

    # Step 5: Visualize Results
    print("\nStep 5: Create Visualizations")
    print("-" * 70)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Time Domain Signal",
            "Frequency Spectrum (Original)",
            "Filtered Signal (Time Domain)",
            "Frequency Spectrum (Filtered)",
            "Spectrogram (Original Signal)",
            "Spectrogram (Filtered Signal)",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Row 1: Time domain
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            name="Noisy Signal",
            line=dict(color="lightblue", width=0.5),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Row 1: Frequency spectrum
    fig.add_trace(
        go.Scatter(
            x=freqs_pos,
            y=magnitude_pos,
            name="Original Magnitude",
            line=dict(color="blue", width=1),
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Add vertical lines at expected frequencies
    for freq, name in [(50, "50 Hz"), (150, "150 Hz"), (200, "200 Hz")]:
        fig.add_vline(
            x=freq,
            line_dash="dash",
            line_color="gray",
            row=1,
            col=2,
            annotation_text=name,
            annotation_position="top",
        )

    # Row 2: Filtered time domain
    fig.add_trace(
        go.Scatter(
            x=t,
            y=filtered_signal,
            name="Filtered Signal",
            line=dict(color="green", width=1),
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    # Row 2: Filtered frequency spectrum
    fig.add_trace(
        go.Scatter(
            x=freqs_pos,
            y=magnitude_filtered_pos,
            name="Filtered Magnitude",
            line=dict(color="darkgreen", width=1),
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    for freq, name in [(50, "50 Hz"), (150, "150 Hz"), (200, "200 Hz")]:
        fig.add_vline(
            x=freq,
            line_dash="dash",
            line_color="gray",
            row=2,
            col=2,
            annotation_text="",
            annotation_position="top",
        )

    # Row 3: Spectrograms
    fig.add_trace(
        go.Heatmap(
            x=t_spec,
            y=f,
            z=10 * np.log10(Sxx + 1e-10),
            colorscale="Viridis",
            showscale=False,
            name="Original",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=t_spec_filt,
            y=f_filt,
            z=10 * np.log10(Sxx_filt + 1e-10),
            colorscale="Viridis",
            showscale=True,
            name="Filtered",
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)

    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=2)

    fig.update_layout(
        title_text="Signal Processing Tutorial - Filtering and Frequency Analysis",
        height=1000,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "signal_processing.html"))

    print("✓ Signal processing visualization complete")
    print("=" * 70)


if __name__ == "__main__":
    signal_processing_tutorial()
