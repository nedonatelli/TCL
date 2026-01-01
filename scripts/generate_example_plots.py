"""
Generate static PNG images from example scripts for documentation.

This script runs each example and saves PNG outputs to docs/_static/images/examples/.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Add pytcl to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Set output directory for images
OUTPUT_DIR = ROOT / "docs" / "_static" / "images" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Disable interactive display
pio.renderers.default = None


def save_figure(fig, name, width=1000, height=600):
    """Save a Plotly figure as PNG."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.write_image(str(path), width=width, height=height, scale=2)
    print(f"  Saved: {path.name}")
    return path


# ============================================================================
# Kalman Filter Comparison
# ============================================================================
def generate_kalman_filter_comparison():
    """Generate Kalman filter comparison plot."""
    print("\n1. Generating Kalman Filter Comparison...")

    from pytcl.dynamic_estimation import (
        kf_predict,
        kf_update,
    )
    from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

    np.random.seed(42)
    n_steps = 100
    dt = 1.0

    # Generate trajectory
    x0 = np.array([100.0, 2.0, 50.0, 1.0])
    F = f_constant_velocity(dt, 2)
    true_states = np.zeros((n_steps, 4))
    true_states[0] = x0
    for k in range(1, n_steps):
        true_states[k] = F @ true_states[k - 1]

    # Generate measurements
    R_linear = np.diag([5.0**2, 5.0**2])
    linear_meas = np.zeros((n_steps, 2))
    for k in range(n_steps):
        x, y = true_states[k, 0], true_states[k, 2]
        linear_meas[k] = np.array([x, y]) + np.random.multivariate_normal(
            [0, 0], R_linear
        )

    # Run KF
    Q = q_constant_velocity(dt, 0.1, 2)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.diag([5.0**2, 5.0**2])

    x = np.array([linear_meas[0, 0], 0.0, linear_meas[0, 1], 0.0])
    P = np.diag([25.0, 10.0, 25.0, 10.0])
    kf_estimates = np.zeros((n_steps, 4))
    kf_estimates[0] = x

    for k in range(1, n_steps):
        x, P = kf_predict(x, P, F, Q)
        result = kf_update(x, P, linear_meas[k], H, R)
        x, P = result.x, result.P
        kf_estimates[k] = x

    # Create plot
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Trajectory Comparison", "Position Error Over Time"),
    )

    fig.add_trace(
        go.Scatter(
            x=true_states[:, 0],
            y=true_states[:, 2],
            mode="lines",
            name="True",
            line=dict(color="black", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=linear_meas[:, 0],
            y=linear_meas[:, 1],
            mode="markers",
            name="Measurements",
            marker=dict(color="gray", size=3, opacity=0.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=kf_estimates[:, 0],
            y=kf_estimates[:, 2],
            mode="lines",
            name="KF Estimate",
            line=dict(color="blue", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Position error
    time = np.arange(n_steps)
    pos_err = np.sqrt(
        (kf_estimates[:, 0] - true_states[:, 0]) ** 2
        + (kf_estimates[:, 2] - true_states[:, 2]) ** 2
    )
    fig.add_trace(
        go.Scatter(x=time, y=pos_err, name="Position Error", line=dict(color="blue")),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Kalman Filter Tracking Example", height=400, width=900, showlegend=True
    )
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=2)

    save_figure(fig, "kalman_filter_comparison", width=900, height=400)


# ============================================================================
# Multi-Target Tracking
# ============================================================================
def generate_multi_target_tracking():
    """Generate multi-target tracking plot."""
    print("\n2. Generating Multi-Target Tracking...")

    from pytcl.trackers import MultiTargetTracker, TrackStatus

    np.random.seed(42)

    # Simulate two crossing targets
    n_steps = 50
    dt = 1.0
    true_states = []
    measurements = []

    for k in range(n_steps):
        t = k * dt
        x1, y1 = 0.0 + 2.0 * t, 0.0 + 1.0 * t
        x2, y2 = 100.0 - 2.0 * t, 0.0 + 1.5 * t
        true_states.append(np.array([x1, y1, x2, y2]))

        meas = []
        R = np.eye(2) * 2.0
        if np.random.rand() < 0.95:
            meas.append(np.array([x1, y1]) + np.random.multivariate_normal([0, 0], R))
        if np.random.rand() < 0.95:
            meas.append(np.array([x2, y2]) + np.random.multivariate_normal([0, 0], R))
        measurements.append(meas)

    # Run tracker
    def F(dt):
        return np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float64
        )

    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

    def Q(dt):
        q = 0.5
        return (
            np.array(
                [
                    [dt**4 / 4, dt**3 / 2, 0, 0],
                    [dt**3 / 2, dt**2, 0, 0],
                    [0, 0, dt**4 / 4, dt**3 / 2],
                    [0, 0, dt**3 / 2, dt**2],
                ]
            )
            * q**2
        )

    R = np.eye(2) * 2.0
    P0 = np.diag([10.0, 5.0, 10.0, 5.0])

    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=Q,
        R=R,
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=P0,
    )

    track_history = []
    for meas in measurements:
        tracks = tracker.process(meas, dt)
        track_history.append(tracks)

    # Create plot
    fig = go.Figure()

    true_arr = np.array(true_states)
    fig.add_trace(
        go.Scatter(
            x=true_arr[:, 0],
            y=true_arr[:, 1],
            mode="lines",
            line=dict(color="green", width=2),
            name="Target 1 (truth)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=true_arr[:, 2],
            y=true_arr[:, 3],
            mode="lines",
            line=dict(color="blue", width=2),
            name="Target 2 (truth)",
        )
    )

    meas_x, meas_y = [], []
    for meas in measurements:
        for z in meas:
            meas_x.append(z[0])
            meas_y.append(z[1])
    fig.add_trace(
        go.Scatter(
            x=meas_x,
            y=meas_y,
            mode="markers",
            marker=dict(color="black", size=3, opacity=0.5),
            name="Measurements",
        )
    )

    # Plot tracks
    track_positions = {}
    for tracks in track_history:
        for track in tracks:
            if track.status == TrackStatus.CONFIRMED:
                if track.id not in track_positions:
                    track_positions[track.id] = []
                track_positions[track.id].append((track.state[0], track.state[2]))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (track_id, positions) in enumerate(track_positions.items()):
        if len(positions) > 1:
            pos_arr = np.array(positions)
            fig.add_trace(
                go.Scatter(
                    x=pos_arr[:, 0],
                    y=pos_arr[:, 1],
                    mode="lines+markers",
                    line=dict(color=colors[i % 4], width=1.5),
                    marker=dict(color=colors[i % 4], size=4),
                    name=f"Track {track_id}",
                )
            )

    fig.update_layout(
        title="Multi-Target Tracking with GNN Data Association",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=800,
        height=500,
        showlegend=True,
    )

    save_figure(fig, "multi_target_tracking", width=800, height=500)


# ============================================================================
# Particle Filter
# ============================================================================
def generate_particle_filter():
    """Generate particle filter plot."""
    print("\n3. Generating Particle Filter...")

    from pytcl.dynamic_estimation.particle_filters import (
        effective_sample_size,
        resample_multinomial,
        resample_residual,
        resample_systematic,
    )

    np.random.seed(42)
    n_particles = 1000

    # Create particles with non-uniform weights
    particles = np.random.randn(n_particles, 2)
    weights = np.exp(-np.sum(particles**2, axis=1) / 4)
    weights /= weights.sum()

    # Resample
    particles_multi = resample_multinomial(particles, weights)
    particles_sys = resample_systematic(particles, weights)
    particles_res = resample_residual(particles, weights)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Original (ESS={effective_sample_size(weights):.0f})",
            "Multinomial Resampling",
            "Systematic Resampling",
            "Residual Resampling",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=particles[:, 0],
            y=particles[:, 1],
            mode="markers",
            marker=dict(size=3, color=weights, colorscale="Viridis", opacity=0.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=particles_multi[:, 0],
            y=particles_multi[:, 1],
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.6),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=particles_sys[:, 0],
            y=particles_sys[:, 1],
            mode="markers",
            marker=dict(size=3, color="green", opacity=0.6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=particles_res[:, 0],
            y=particles_res[:, 1],
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.6),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600,
        width=800,
        title_text="Particle Filter Resampling Methods",
        showlegend=False,
    )

    save_figure(fig, "particle_filter_resampling", width=800, height=600)


# ============================================================================
# Clustering
# ============================================================================
def generate_clustering():
    """Generate clustering comparison plot."""
    print("\n4. Generating Clustering Examples...")

    from pytcl.clustering import dbscan, kmeans

    np.random.seed(42)

    # Generate clustered data
    cluster1 = np.random.randn(50, 2) * 0.5 + [0, 0]
    cluster2 = np.random.randn(50, 2) * 0.5 + [4, 4]
    noise = np.random.uniform(-2, 8, (20, 2))
    data = np.vstack([cluster1, cluster2, noise])

    # K-means
    kmeans_result = kmeans(data, n_clusters=2)

    # DBSCAN
    dbscan_result = dbscan(data, eps=0.8, min_samples=5)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "K-Means (k=2)",
            f"DBSCAN ({dbscan_result.n_clusters} clusters)",
        ),
    )

    # K-means result
    colors_km = ["blue", "green"]
    for i in range(2):
        mask = kmeans_result.labels == i
        fig.add_trace(
            go.Scatter(
                x=data[mask, 0],
                y=data[mask, 1],
                mode="markers",
                marker=dict(color=colors_km[i], size=6, opacity=0.6),
                name=f"K-means Cluster {i}",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=kmeans_result.centers[:, 0],
            y=kmeans_result.centers[:, 1],
            mode="markers",
            marker=dict(color="black", size=12, symbol="x", line=dict(width=2)),
            name="Centroids",
        ),
        row=1,
        col=1,
    )

    # DBSCAN result
    colors_db = ["blue", "green", "purple", "orange"]
    unique_labels = np.unique(dbscan_result.labels)
    for label in unique_labels:
        mask = dbscan_result.labels == label
        if label == -1:
            fig.add_trace(
                go.Scatter(
                    x=data[mask, 0],
                    y=data[mask, 1],
                    mode="markers",
                    marker=dict(color="red", size=6, opacity=0.6, symbol="x"),
                    name="Noise",
                ),
                row=1,
                col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data[mask, 0],
                    y=data[mask, 1],
                    mode="markers",
                    marker=dict(color=colors_db[label % 4], size=6, opacity=0.6),
                    name=f"DBSCAN Cluster {label}",
                ),
                row=1,
                col=2,
            )

    fig.update_layout(height=400, width=900, showlegend=True)
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")

    save_figure(fig, "clustering_comparison", width=900, height=400)


# ============================================================================
# Coordinate Systems
# ============================================================================
def generate_coordinate_systems():
    """Generate coordinate system visualization."""
    print("\n5. Generating Coordinate Systems...")

    from pytcl.coordinate_systems import rotx, roty, rotz

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Rotation about X (45deg)",
            "Rotation about Y (45deg)",
            "Rotation about Z (45deg)",
        ],
    )

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    angle = np.pi / 4

    def add_axes(fig, R, col):
        for axis, color, name in [
            (x_axis, "red", "X"),
            (y_axis, "green", "Y"),
            (z_axis, "blue", "Z"),
        ]:
            fig.add_trace(
                go.Scatter3d(
                    x=[0, axis[0]],
                    y=[0, axis[1]],
                    z=[0, axis[2]],
                    mode="lines",
                    line=dict(color=color, width=3),
                    opacity=0.3,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            rotated = R @ axis
            fig.add_trace(
                go.Scatter3d(
                    x=[0, rotated[0]],
                    y=[0, rotated[1]],
                    z=[0, rotated[2]],
                    mode="lines+markers",
                    line=dict(color=color, width=5),
                    marker=dict(size=4),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    add_axes(fig, rotx(angle), col=1)
    add_axes(fig, roty(angle), col=2)
    add_axes(fig, rotz(angle), col=3)

    for i in range(1, 4):
        fig.update_scenes(
            dict(
                xaxis=dict(range=[-1.5, 1.5], title="X"),
                yaxis=dict(range=[-1.5, 1.5], title="Y"),
                zaxis=dict(range=[-1.5, 1.5], title="Z"),
                aspectmode="cube",
            ),
            row=1,
            col=i,
        )

    fig.update_layout(title="Rotation Matrix Visualization", width=1200, height=450)

    save_figure(fig, "coordinate_rotations", width=1200, height=450)


# ============================================================================
# Signal Processing
# ============================================================================
def generate_signal_processing():
    """Generate signal processing plots."""
    print("\n6. Generating Signal Processing...")

    from pytcl.mathematical_functions.signal_processing import (
        butter_design,
        cfar_ca,
        frequency_response,
    )

    np.random.seed(42)
    fs = 1000.0

    # Filter design
    filt = butter_design(order=4, cutoff=50.0, fs=fs, btype="low")
    resp = frequency_response(filt.b, filt.a, fs)

    # CFAR detection
    n_cells = 200
    noise = np.abs(np.random.randn(n_cells))
    signal = noise.copy()
    targets = [(50, 15.0), (100, 8.0), (150, 5.0)]
    for loc, amp in targets:
        signal[loc] = amp

    cfar_result = cfar_ca(signal, guard_cells=2, ref_cells=8, pfa=1e-4)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Butterworth Filter Response", "CFAR Detection")
    )

    # Filter response
    mag_db = 20 * np.log10(np.maximum(resp.magnitude, 1e-10))
    fig.add_trace(
        go.Scatter(
            x=resp.frequencies,
            y=mag_db,
            mode="lines",
            name="Magnitude",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=-3, line_dash="dash", line_color="red", row=1, col=1)

    # CFAR
    cells = np.arange(n_cells)
    fig.add_trace(
        go.Scatter(
            x=cells, y=signal, mode="lines", name="Signal", line=dict(color="blue")
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=cells,
            y=cfar_result.threshold,
            mode="lines",
            name="Threshold",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=2,
    )
    det_idx = cfar_result.detection_indices
    fig.add_trace(
        go.Scatter(
            x=det_idx,
            y=signal[det_idx],
            mode="markers",
            name="Detections",
            marker=dict(color="green", size=10, symbol="circle"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_xaxes(title_text="Range Cell", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)
    fig.update_layout(height=400, width=900, showlegend=True)

    save_figure(fig, "signal_processing", width=900, height=400)


# ============================================================================
# Performance Evaluation
# ============================================================================
def generate_performance_evaluation():
    """Generate performance evaluation plot."""
    print("\n7. Generating Performance Evaluation...")

    from pytcl.performance_evaluation import ospa

    np.random.seed(42)

    # Simulate tracking scenario over time
    n_steps = 30
    ospa_values = []
    time_steps = []

    true_targets = [np.array([0.0, 0.0]), np.array([10.0, 10.0])]

    for k in range(n_steps):
        t = k / n_steps
        # Estimates improve over time
        noise = (1 - t) * 2.0
        estimates = [
            true_targets[0] + np.random.randn(2) * noise,
            true_targets[1] + np.random.randn(2) * noise,
        ]

        result = ospa(true_targets, estimates, c=5.0, p=2)
        ospa_values.append(result.ospa)
        time_steps.append(k)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=ospa_values,
            mode="lines+markers",
            name="OSPA",
            line=dict(color="blue", width=2),
        )
    )
    fig.update_layout(
        title="OSPA Distance Over Time (Lower is Better)",
        xaxis_title="Time Step",
        yaxis_title="OSPA Distance",
        height=400,
        width=700,
    )

    save_figure(fig, "performance_evaluation", width=700, height=400)


# ============================================================================
# Main
# ============================================================================
def main():
    """Generate all example plots."""
    print("Generating Example Plots for Documentation")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")

    generate_kalman_filter_comparison()
    generate_multi_target_tracking()
    generate_particle_filter()
    generate_clustering()
    generate_coordinate_systems()
    generate_signal_processing()
    generate_performance_evaluation()

    print("\n" + "=" * 50)
    print("All plots generated successfully!")
    print(f"Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
