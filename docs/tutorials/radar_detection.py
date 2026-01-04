"""Radar Detection Tutorial with Interactive Visualizations.

This tutorial demonstrates CFAR (Constant False Alarm Rate) detection for
identifying targets in radar signals. We'll use synthetic radar data to show
how CFAR adapts to noise levels.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def radar_detection_tutorial():
    """Run complete radar detection tutorial with visualizations."""

    print("=" * 70)
    print("RADAR DETECTION TUTORIAL")
    print("=" * 70)

    # Step 1: Create Synthetic Radar Data
    print("\nStep 1: Create Synthetic Radar Data")
    print("-" * 70)

    n_range = 256  # Number of range bins
    n_doppler = 128  # Number of Doppler bins

    # Create radar cube with background noise
    np.random.seed(42)
    background_power = 1.0
    noise_variance = 0.5

    radar_data = background_power * np.random.randn(
        n_doppler, n_range
    ) + 1j * background_power * np.random.randn(n_doppler, n_range)

    # Add a few targets
    targets = [
        {"range_bin": 50, "doppler_bin": 30, "snr": 20},  # Strong target
        {"range_bin": 100, "doppler_bin": 70, "snr": 15},  # Medium target
        {"range_bin": 150, "doppler_bin": 20, "snr": 10},  # Weak target
    ]

    for target in targets:
        r, d = target["range_bin"], target["doppler_bin"]
        snr = target["snr"]
        # Insert target with specified SNR
        radius = 5  # Spread targets over nearby cells
        for dr in range(-radius, radius + 1):
            for dd in range(-radius, radius + 1):
                if 0 <= r + dr < n_range and 0 <= d + dd < n_doppler:
                    target_power = 10 ** (snr / 10)
                    distance = np.sqrt(dr**2 + dd**2)
                    weight = np.exp(-(distance**2) / (2 * (radius / 2) ** 2))
                    radar_data[d + dd, r + dr] += (
                        weight * np.sqrt(target_power) * np.exp(1j * np.random.rand())
                    )

    # Convert to power
    power_data = np.abs(radar_data) ** 2
    power_db = 10 * np.log10(power_data + 1e-10)

    print(f"Radar data created: {n_doppler} Doppler × {n_range} Range")
    print(f"Inserted {len(targets)} targets with varying SNR")

    # Step 2: Implement OS-CFAR (Order Statistic CFAR)
    print("\nStep 2: Implement OS-CFAR Detection")
    print("-" * 70)

    guard_cells = 3  # Cells around target
    train_cells = 8  # Training cells on each side
    pfa = 1e-3  # Probability of false alarm

    # Number of training samples
    n_train = 2 * train_cells

    # For OS-CFAR, we use order statistic (median is common)
    k_order = int(n_train / 2)  # Use median

    # Threshold multiplier (approximate for Rayleigh distribution)
    threshold_multiplier = pfa ** (-1 / n_train) - 1

    # Apply OS-CFAR
    detection_map = np.zeros_like(power_data, dtype=bool)
    threshold_map = np.zeros_like(power_data)

    for d in range(guard_cells + train_cells, n_doppler - guard_cells - train_cells):
        for r in range(guard_cells + train_cells, n_range - guard_cells - train_cells):
            # Extract training cells (excluding guard and test cell)
            train_window = []

            # Left training cells
            for i in range(r - train_cells - guard_cells, r - guard_cells):
                train_window.append(power_data[d, i])

            # Right training cells
            for i in range(r + guard_cells + 1, r + guard_cells + 1 + train_cells):
                train_window.append(power_data[d, i])

            train_window = np.array(train_window)

            # OS-CFAR: use order statistic
            sorted_train = np.sort(train_window)
            order_stat = sorted_train[k_order]

            # Calculate threshold
            threshold = threshold_multiplier * order_stat
            threshold_map[d, r] = threshold

            # Detection
            detection_map[d, r] = power_data[d, r] > threshold

    num_detections = np.sum(detection_map)
    print(f"OS-CFAR parameters:")
    print(f"  Guard cells: {guard_cells}")
    print(f"  Training cells: {train_cells}")
    print(f"  Target PFA: {pfa}")
    print(f"Detections: {num_detections}")

    # Step 3: Analyze Range Profile
    print("\nStep 3: Analyze Range Profile")
    print("-" * 70)

    # Integrate over Doppler dimension
    range_profile = np.sum(power_data, axis=0)
    doppler_profile = np.sum(power_data, axis=1)

    range_profile_db = 10 * np.log10(range_profile + 1e-10)
    doppler_profile_db = 10 * np.log10(doppler_profile + 1e-10)

    print(f"Range profile: {len(range_profile)} bins")
    print(f"Doppler profile: {len(doppler_profile)} bins")

    # Step 4: Visualize Results
    print("\nStep 4: Create Visualizations")
    print("-" * 70)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Range-Doppler Map (dB)",
            "OS-CFAR Detection Map",
            "Range Profile",
            "Doppler Profile",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # Row 1, Col 1: Range-Doppler map
    fig.add_trace(
        go.Heatmap(
            x=np.arange(n_range),
            y=np.arange(n_doppler),
            z=power_db,
            colorscale="Viridis",
            name="Power (dB)",
            showscale=True,
            colorbar=dict(x=0.46, len=0.4),
        ),
        row=1,
        col=1,
    )

    # Overlay target positions
    for target in targets:
        r, d = target["range_bin"], target["doppler_bin"]
        fig.add_trace(
            go.Scatter(
                x=[r],
                y=[d],
                mode="markers",
                marker=dict(size=8, color="red", symbol="x"),
                name=f"Target SNR={target['snr']} dB",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Row 1, Col 2: Detection map
    detection_db = np.full_like(power_db, np.nan)
    detection_db[detection_map] = power_db[detection_map]

    fig.add_trace(
        go.Heatmap(
            x=np.arange(n_range),
            y=np.arange(n_doppler),
            z=detection_db,
            colorscale="Reds",
            name="Detections",
            showscale=True,
            colorbar=dict(x=1.02, len=0.4),
        ),
        row=1,
        col=2,
    )

    # Row 2, Col 1: Range profile
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_range),
            y=range_profile_db,
            name="Range Profile",
            line=dict(color="blue"),
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )

    # Mark detected ranges
    for target in targets:
        r = target["range_bin"]
        fig.add_vline(x=r, line_dash="dash", line_color="red", row=2, col=1)

    # Row 2, Col 2: Doppler profile
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_doppler),
            y=doppler_profile_db,
            name="Doppler Profile",
            line=dict(color="green"),
            fill="tozeroy",
        ),
        row=2,
        col=2,
    )

    # Mark detected Doppler
    for target in targets:
        d = target["doppler_bin"]
        fig.add_vline(x=d, line_dash="dash", line_color="red", row=2, col=2)

    # Update layout
    fig.update_xaxes(title_text="Range Bin", row=1, col=1)
    fig.update_yaxes(title_text="Doppler Bin", row=1, col=1)

    fig.update_xaxes(title_text="Range Bin", row=1, col=2)
    fig.update_yaxes(title_text="Doppler Bin", row=1, col=2)

    fig.update_xaxes(title_text="Range Bin", row=2, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=2, col=1)

    fig.update_xaxes(title_text="Doppler Bin", row=2, col=2)
    fig.update_yaxes(title_text="Power (dB)", row=2, col=2)

    fig.update_layout(
        title_text="Radar Detection Tutorial - OS-CFAR Detection",
        height=800,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "radar_detection.html"))

    print("✓ Radar detection visualization complete")
    print("=" * 70)


if __name__ == "__main__":
    radar_detection_tutorial()
