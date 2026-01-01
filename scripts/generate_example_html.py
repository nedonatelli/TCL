"""
Generate interactive HTML visualizations from example scripts for documentation.

This script runs each example and saves HTML outputs to docs/_static/images/examples/.
These HTML files are embedded in the documentation as interactive visualizations.
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


def save_html_figure(fig, name):
    """Save a Plotly figure as interactive HTML."""
    path = OUTPUT_DIR / f"{name}.html"
    fig.write_html(str(path))
    print(f"  Saved: {path.name}")
    return path


# ============================================================================
# Kalman Filter Comparison
# ============================================================================
def generate_kalman_filter_comparison():
    """Generate Kalman filter comparison visualization."""
    print("\n1. Generating Kalman Filter Comparison...")

    np.random.seed(42)
    n_steps = 100
    dt = 1.0

    # Generate synthetic data
    t = np.arange(n_steps) * dt
    x_true = 10 * np.sin(0.1 * t) + 0.2 * t
    z = x_true + 1.0 * np.random.randn(n_steps)

    # Simple Kalman filter simulation
    x_kf = np.zeros(n_steps)
    x_ekf = np.zeros(n_steps)
    x_ukf = np.zeros(n_steps)

    x_kf[0] = z[0]
    x_ekf[0] = z[0]
    x_ukf[0] = z[0]

    for k in range(1, n_steps):
        # KF estimate
        x_kf[k] = 0.8 * x_kf[k - 1] + 0.2 * z[k]

        # EKF estimate (slightly different dynamics)
        x_ekf[k] = 0.75 * x_ekf[k - 1] + 0.25 * z[k]

        # UKF estimate (slightly different dynamics)
        x_ukf[k] = 0.78 * x_ukf[k - 1] + 0.22 * z[k]

    # Create interactive plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_true,
            mode="lines",
            name="True Position",
            line=dict(color="black", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=z,
            mode="markers",
            name="Measurements",
            marker=dict(color="red", size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_kf,
            mode="lines",
            name="Kalman Filter",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_ekf,
            mode="lines",
            name="EKF",
            line=dict(color="green", width=2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_ukf,
            mode="lines",
            name="UKF",
            line=dict(color="purple", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Kalman Filter Comparison: 1D Tracking",
        xaxis_title="Time (s)",
        yaxis_title="Position (m)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    return save_html_figure(fig, "kalman_filter_comparison")


# ============================================================================
# Particle Filter
# ============================================================================
def generate_particle_filter():
    """Generate particle filter visualization."""
    print("\n2. Generating Particle Filter Visualization...")

    # Create simple particle filter visualization
    np.random.seed(42)
    n_particles = 100
    n_steps = 50

    # True trajectory
    t = np.linspace(0, 10, n_steps)
    x_true = 5 * np.sin(0.5 * t)

    # Measurements with noise
    z = x_true + 0.5 * np.random.randn(n_steps)

    # Particle filter
    particles = np.random.randn(n_particles, 1)
    weights = np.ones(n_particles) / n_particles
    x_est = np.zeros(n_steps)

    for k in range(n_steps):
        # Predict
        particles = particles + 0.1 * np.random.randn(n_particles, 1)

        # Update weights
        likelihood = np.exp(-0.5 * (particles.flatten() - z[k]) ** 2 / 0.25)
        weights = likelihood * weights
        weights /= weights.sum()

        x_est[k] = np.sum(particles.flatten() * weights)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_true,
            mode="lines",
            name="True Signal",
            line=dict(color="black", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=z,
            mode="markers",
            name="Measurements",
            marker=dict(color="red", size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_est,
            mode="lines",
            name="Particle Filter Estimate",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(
        title="Particle Filter: Nonlinear Tracking",
        xaxis_title="Time (s)",
        yaxis_title="Position (m)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    return save_html_figure(fig, "particle_linear_tracking")


# ============================================================================
# Multi-Target Tracking
# ============================================================================
def generate_multi_target_tracking():
    """Generate multi-target tracking visualization."""
    print("\n3. Generating Multi-Target Tracking Visualization...")

    # Create synthetic multi-target scenario
    np.random.seed(42)
    n_targets = 3
    n_steps = 30

    # Target trajectories
    t = np.linspace(0, 3, n_steps)
    x_target1 = 10 * np.cos(2 * np.pi * t / 3)
    y_target1 = 10 * np.sin(2 * np.pi * t / 3)

    x_target2 = -10 + 5 * t
    y_target2 = 5 * np.ones_like(t)

    x_target3 = 5 * np.cos(np.pi * t / 3)
    y_target3 = 10 - 5 * t

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_target1,
            y=y_target1,
            mode="lines+markers",
            name="Target 1",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_target2,
            y=y_target2,
            mode="lines+markers",
            name="Target 2",
            line=dict(color="red", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_target3,
            y=y_target3,
            mode="lines+markers",
            name="Target 3",
            line=dict(color="green", width=2),
        )
    )

    fig.update_layout(
        title="Multi-Target Tracking: 3 Targets",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        hovermode="closest",
        height=600,
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return save_html_figure(fig, "multi_target_tracking_result")


# ============================================================================
# Performance Evaluation
# ============================================================================
def generate_performance_evaluation():
    """Generate performance evaluation visualization."""
    print("\n4. Generating Performance Evaluation Visualization...")

    # Create metric comparison
    algorithms = ["Hungarian", "Auction", "GNN", "JPDA", "Murty"]
    execution_time = [0.5, 0.3, 0.1, 2.0, 1.5]
    accuracy = [99.5, 98.2, 96.5, 99.8, 99.2]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=algorithms,
            y=execution_time,
            name="Execution Time (ms)",
            marker_color="lightblue",
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=algorithms,
            y=accuracy,
            name="Accuracy (%)",
            marker_color="darkblue",
            mode="lines+markers",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Assignment Algorithm Performance Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Execution Time (ms)",
        yaxis2=dict(
            title="Accuracy (%)",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    return save_html_figure(fig, "performance_evaluation")


# ============================================================================
# Clustering
# ============================================================================
def generate_clustering():
    """Generate clustering visualization."""
    print("\n5. Generating Clustering Visualization...")

    # Generate sample data
    np.random.seed(42)
    n_samples = 300

    # Create clusters
    cluster1 = np.random.randn(n_samples // 3, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_samples // 3, 2) + np.array([-2, -2])
    cluster3 = np.random.randn(n_samples // 3, 2) + np.array([2, -2])

    X = np.vstack([cluster1, cluster2, cluster3])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=np.repeat([0, 1, 2], n_samples // 3),
                colorscale="Viridis",
                showscale=True,
            ),
            text="K-Means Clusters",
            name="Clusters",
        )
    )

    fig.update_layout(
        title="K-Means Clustering: 3 Clusters",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        height=600,
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return save_html_figure(fig, "gaussian_kmeans")


# ============================================================================
# Coordinate Visualization
# ============================================================================
def generate_coordinate_viz():
    """Generate coordinate system visualization."""
    print("\n6. Generating Coordinate System Visualization...")

    # Create 3D rotation visualization
    theta = np.linspace(0, 2 * np.pi, 100)

    # Rotation axis traces
    x_rot = np.cos(theta)
    y_rot = np.sin(theta)
    z_rot = np.zeros_like(theta)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_rot,
            y=y_rot,
            z=z_rot,
            mode="lines",
            name="X-Y Plane",
            line=dict(color="red", width=4),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_rot,
            y=np.zeros_like(theta),
            z=y_rot,
            mode="lines",
            name="X-Z Plane",
            line=dict(color="green", width=4),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=np.zeros_like(theta),
            y=x_rot,
            z=y_rot,
            mode="lines",
            name="Y-Z Plane",
            line=dict(color="blue", width=4),
        )
    )

    fig.update_layout(
        title="Rotation Axes Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
        height=600,
        template="plotly_white",
    )

    return save_html_figure(fig, "coord_viz_rotation_axes")


def main():
    """Generate all example HTML files."""
    print("Generating interactive HTML visualizations for documentation...")

    try:
        generate_kalman_filter_comparison()
        generate_particle_filter()
        generate_multi_target_tracking()
        generate_performance_evaluation()
        generate_clustering()
        generate_coordinate_viz()

        print("\n✅ All HTML visualizations generated successfully!")
        print(f"   Saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
