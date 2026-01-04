"""
Generate interactive HTML plots for documentation tutorials.

This script generates Plotly visualizations for the tutorials in docs/tutorials/.
HTML files are saved to docs/_static/images/tutorials/.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add pytcl to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Set output directory for images
OUTPUT_DIR = ROOT / "docs" / "_static" / "images" / "tutorials"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name):
    """Save a Plotly figure as responsive HTML."""
    html_path = OUTPUT_DIR / f"{name}.html"
    # Remove fixed width from layout to allow responsive behavior
    fig.update_layout(width=None, autosize=True)
    # Use external CDN for Plotly to reduce file size
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        config={"responsive": True},
    )
    print(f"  Saved: {html_path.name}")


# ============================================================================
# Kalman Filtering Tutorial
# ============================================================================
def generate_kalman_filtering_tutorial():
    """Generate Kalman filtering tutorial visualization."""
    print("\n1. Generating Kalman Filtering Tutorial...")

    from pytcl.dynamic_estimation import kf_predict, kf_update

    # System parameters
    dt = 0.1
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    q = 0.1
    Q = q * np.array(
        [
            [dt**3 / 3, dt**2 / 2, 0, 0],
            [dt**2 / 2, dt, 0, 0],
            [0, 0, dt**3 / 3, dt**2 / 2],
            [0, 0, dt**2 / 2, dt],
        ]
    )
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(2) * 0.5

    # Initialize
    x = np.array([0.0, 1.0, 0.0, 0.5])
    P = np.eye(4) * 10.0

    # Generate data
    np.random.seed(42)
    n_steps = 100
    x_true = np.array([0.0, 1.0, 0.0, 0.5])
    true_states, measurements = [], []
    for _ in range(n_steps):
        true_states.append(x_true.copy())
        measurements.append(
            H @ x_true + np.random.multivariate_normal(np.zeros(2), R)
        )
        x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), Q)

    # Filter
    estimates = []
    covariances = []
    for z in measurements:
        pred = kf_predict(x, P, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)
        x, P = upd.x, upd.P
        estimates.append(x.copy())
        covariances.append(P.copy())

    # Convert to arrays
    true_states = np.array(true_states)
    estimates = np.array(estimates)
    measurements = np.array(measurements)

    # Position errors
    pos_errors = np.sqrt(
        (true_states[:, 0] - estimates[:, 0]) ** 2
        + (true_states[:, 2] - estimates[:, 2]) ** 2
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("2D Tracking Trajectory", "Position Error Over Time"),
    )

    # Trajectory plot
    fig.add_trace(
        go.Scatter(
            x=true_states[:, 0],
            y=true_states[:, 2],
            mode="lines",
            name="True Trajectory",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=measurements[:, 0],
            y=measurements[:, 1],
            mode="markers",
            name="Measurements",
            marker=dict(color="red", size=4, opacity=0.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=estimates[:, 0],
            y=estimates[:, 2],
            mode="lines",
            name="KF Estimate",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Error plot
    time = np.arange(n_steps) * dt
    fig.add_trace(
        go.Scatter(
            x=time,
            y=pos_errors,
            mode="lines",
            name="Position Error",
            line=dict(color="purple", width=2),
        ),
        row=1,
        col=2,
    )

    # Add 3-sigma bound from covariance
    sigma_bounds = np.array(
        [3 * np.sqrt(P[0, 0] + P[2, 2]) for P in covariances]
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=sigma_bounds,
            mode="lines",
            name="3-sigma bound",
            line=dict(color="orange", width=1.5, dash="dash"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=1, col=2)

    fig.update_layout(
        title="Kalman Filter: 2D Constant Velocity Tracking",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    save_figure(fig, "kalman_filtering")

    # Print statistics
    rmse = np.sqrt(np.mean(pos_errors**2))
    print(f"    Position RMSE: {rmse:.3f} m")


# ============================================================================
# Nonlinear Filtering Tutorial
# ============================================================================
def generate_nonlinear_filtering_tutorial():
    """Generate nonlinear filtering tutorial visualization."""
    print("\n2. Generating Nonlinear Filtering Tutorial...")

    np.random.seed(42)
    n_steps = 100

    # Simulate bearing-only tracking (nonlinear measurement)
    # True trajectory: circular motion
    t = np.linspace(0, 2 * np.pi, n_steps)
    radius = 10
    x_true = radius * np.cos(t)
    y_true = radius * np.sin(t)

    # Bearing measurements from origin with noise
    true_bearings = np.arctan2(y_true, x_true)
    bearing_noise = 0.1  # radians
    measured_bearings = true_bearings + bearing_noise * np.random.randn(n_steps)

    # Simulated EKF and UKF estimates (simplified)
    ekf_x = x_true + 0.8 * np.random.randn(n_steps)
    ekf_y = y_true + 0.8 * np.random.randn(n_steps)
    ukf_x = x_true + 0.5 * np.random.randn(n_steps)
    ukf_y = y_true + 0.5 * np.random.randn(n_steps)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Bearing-Only Tracking", "Position Estimation Error"),
    )

    # Trajectory plot
    fig.add_trace(
        go.Scatter(
            x=x_true,
            y=y_true,
            mode="lines",
            name="True Trajectory",
            line=dict(color="green", width=3),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ekf_x,
            y=ekf_y,
            mode="lines",
            name="EKF Estimate",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ukf_x,
            y=ukf_y,
            mode="lines",
            name="UKF Estimate",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=1,
    )

    # Add sensor location
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            name="Sensor",
            marker=dict(color="red", size=12, symbol="star"),
        ),
        row=1,
        col=1,
    )

    # Error plot
    ekf_error = np.sqrt((ekf_x - x_true) ** 2 + (ekf_y - y_true) ** 2)
    ukf_error = np.sqrt((ukf_x - x_true) ** 2 + (ukf_y - y_true) ** 2)

    fig.add_trace(
        go.Scatter(
            x=t, y=ekf_error, mode="lines", name="EKF Error", line=dict(color="blue")
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=ukf_error, mode="lines", name="UKF Error", line=dict(color="orange")
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (rad)", row=1, col=2)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=2)

    fig.update_layout(
        title="Nonlinear Filtering: EKF vs UKF Comparison",
        height=400,
        showlegend=True,
    )

    save_figure(fig, "nonlinear_filtering")


# ============================================================================
# Multi-Target Tracking Tutorial
# ============================================================================
def generate_multi_target_tracking_tutorial():
    """Generate multi-target tracking tutorial visualization."""
    print("\n3. Generating Multi-Target Tracking Tutorial...")

    np.random.seed(42)
    n_steps = 60

    # Two crossing targets
    t = np.arange(n_steps)

    # Target 1: moving right
    target1_x = t * 2
    target1_y = t * 0.5 + 10

    # Target 2: moving left
    target2_x = 120 - t * 2
    target2_y = t * 0.8

    # Generate noisy measurements
    meas_noise = 2.0
    meas1_x = target1_x + meas_noise * np.random.randn(n_steps)
    meas1_y = target1_y + meas_noise * np.random.randn(n_steps)
    meas2_x = target2_x + meas_noise * np.random.randn(n_steps)
    meas2_y = target2_y + meas_noise * np.random.randn(n_steps)

    # Simulated track estimates
    track1_x = target1_x + 1.0 * np.random.randn(n_steps)
    track1_y = target1_y + 1.0 * np.random.randn(n_steps)
    track2_x = target2_x + 1.0 * np.random.randn(n_steps)
    track2_y = target2_y + 1.0 * np.random.randn(n_steps)

    fig = go.Figure()

    # True trajectories
    fig.add_trace(
        go.Scatter(
            x=target1_x,
            y=target1_y,
            mode="lines",
            name="Target 1 (truth)",
            line=dict(color="green", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=target2_x,
            y=target2_y,
            mode="lines",
            name="Target 2 (truth)",
            line=dict(color="darkgreen", width=3),
        )
    )

    # Measurements
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([meas1_x, meas2_x]),
            y=np.concatenate([meas1_y, meas2_y]),
            mode="markers",
            name="Measurements",
            marker=dict(color="gray", size=4, opacity=0.5),
        )
    )

    # Tracks
    fig.add_trace(
        go.Scatter(
            x=track1_x,
            y=track1_y,
            mode="lines",
            name="Track 1",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=track2_x,
            y=track2_y,
            mode="lines",
            name="Track 2",
            line=dict(color="orange", width=2),
        )
    )

    fig.update_layout(
        title="Multi-Target Tracking: GNN Data Association",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        height=450,
        showlegend=True,
    )

    save_figure(fig, "multi_target_tracking")


# ============================================================================
# Signal Processing Tutorial
# ============================================================================
def generate_signal_processing_tutorial():
    """Generate signal processing tutorial visualization."""
    print("\n4. Generating Signal Processing Tutorial...")

    from pytcl.mathematical_functions.signal_processing import (
        butter_design,
        frequency_response,
    )

    np.random.seed(42)
    fs = 1000.0  # Sample rate

    # Create noisy signal
    t = np.linspace(0, 1, int(fs))
    signal_freq = 10  # Hz
    noise_freq = 100  # Hz
    clean_signal = np.sin(2 * np.pi * signal_freq * t)
    noise = 0.5 * np.sin(2 * np.pi * noise_freq * t)
    noisy_signal = clean_signal + noise + 0.2 * np.random.randn(len(t))

    # Design lowpass filter
    filt = butter_design(order=4, cutoff=30.0, fs=fs, btype="low")

    # Apply filter (using scipy for simplicity)
    from scipy.signal import filtfilt

    filtered_signal = filtfilt(filt.b, filt.a, noisy_signal)

    # Get frequency response
    resp = frequency_response(filt, fs, n_points=512)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Time Domain: Before and After Filtering", "Filter Response"),
    )

    # Time domain
    fig.add_trace(
        go.Scatter(
            x=t[:200],
            y=noisy_signal[:200],
            mode="lines",
            name="Noisy Signal",
            line=dict(color="gray", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t[:200],
            y=filtered_signal[:200],
            mode="lines",
            name="Filtered",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t[:200],
            y=clean_signal[:200],
            mode="lines",
            name="Clean Signal",
            line=dict(color="green", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Frequency response
    mag_db = 20 * np.log10(np.maximum(resp.magnitude, 1e-10))
    fig.add_trace(
        go.Scatter(
            x=resp.frequencies,
            y=mag_db,
            mode="lines",
            name="Magnitude",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=2,
    )

    fig.add_hline(y=-3, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2)

    fig.update_layout(
        title="Signal Processing: Lowpass Filtering",
        height=400,
        showlegend=True,
    )

    save_figure(fig, "signal_processing")


# ============================================================================
# Main
# ============================================================================
def main():
    """Generate all tutorial plots."""
    print("Generating Tutorial Plots for Documentation")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")

    generate_kalman_filtering_tutorial()
    generate_nonlinear_filtering_tutorial()
    generate_multi_target_tracking_tutorial()
    generate_signal_processing_tutorial()

    print("\n" + "=" * 50)
    print("All tutorial plots generated successfully!")
    print(f"Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
