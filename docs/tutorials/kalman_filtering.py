"""Kalman Filtering Tutorial with Interactive Visualizations.

This tutorial demonstrates how to implement a Kalman filter for tracking
a moving object using noisy position measurements.

Problem Setup:
We will track a 2D object moving with constant velocity. The state vector
contains position and velocity in both x and y dimensions:

x = [x, ẋ, y, ẏ]ᵀ
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def kalman_filtering_tutorial():
    """Run complete Kalman filtering tutorial with visualizations."""

    print("=" * 70)
    print("KALMAN FILTERING TUTORIAL")
    print("=" * 70)

    # Step 1: Define the System Model
    print("\nStep 1: Define the System Model")
    print("-" * 70)

    dt = 0.1  # Time step

    # State transition matrix (constant velocity model)
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    # Process noise covariance
    q = 0.1  # Process noise intensity
    Q = q * np.array(
        [
            [dt**3 / 3, dt**2 / 2, 0, 0],
            [dt**2 / 2, dt, 0, 0],
            [0, 0, dt**3 / 3, dt**2 / 2],
            [0, 0, dt**2 / 2, dt],
        ]
    )

    # Measurement matrix (we observe position only)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    # Measurement noise covariance
    R = np.eye(2) * 0.5

    print(f"State transition matrix F shape: {F.shape}")
    print(f"Measurement matrix H shape: {H.shape}")
    print(f"Process noise intensity q: {q}")

    # Step 2: Initialize the Filter
    print("\nStep 2: Initialize the Filter")
    print("-" * 70)

    # Initial state estimate
    x = np.array([0.0, 1.0, 0.0, 0.5])

    # Initial covariance (high uncertainty)
    P = np.eye(4) * 10.0

    print(f"Initial state: {x}")
    print(f"Initial covariance trace: {np.trace(P):.2f}")

    # Step 3: Generate Simulated Data
    print("\nStep 3: Generate Simulated Data")
    print("-" * 70)

    np.random.seed(42)
    n_steps = 100

    # True trajectory
    true_states = []
    x_true = np.array([0.0, 1.0, 0.0, 0.5])
    for _ in range(n_steps):
        true_states.append(x_true[:2].copy())  # Store position only
        x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), Q)

    true_states = np.array(true_states)

    # Noisy measurements
    measurements = []
    for state in true_states:
        state_full = np.concatenate([state, [0, 0]])  # Pad with velocity
        measurement = H @ state_full + np.random.multivariate_normal(np.zeros(2), R)
        measurements.append(measurement)

    measurements = np.array(measurements)

    print(f"Generated {n_steps} steps of trajectory")
    print(
        f"True trajectory range: x=[{true_states[:, 0].min():.2f}, {true_states[:, 0].max():.2f}], "
        f"y=[{true_states[:, 1].min():.2f}, {true_states[:, 1].max():.2f}]"
    )

    # Step 4: Run the Kalman Filter
    print("\nStep 4: Run the Kalman Filter")
    print("-" * 70)

    x = np.array([0.0, 1.0, 0.0, 0.5])
    P = np.eye(4) * 10.0

    estimates = []

    for i, z in enumerate(measurements):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        y = z - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        estimates.append(x[:2].copy())

    estimates = np.array(estimates)

    print(
        f"Filter complete. RMSE: {np.sqrt(np.mean((estimates - true_states)**2)):.4f}"
    )

    # Step 5: Visualize Results
    print("\nStep 5: Create Visualizations")
    print("-" * 70)

    time = np.arange(n_steps) * dt

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "2D Trajectory Comparison",
            "X Position Over Time",
            "Y Position Over Time",
            "Estimation Error Over Time",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # Subplot 1: 2D Trajectory
    fig.add_trace(
        go.Scatter(
            x=true_states[:, 0],
            y=true_states[:, 1],
            mode="lines",
            name="True Trajectory",
            line=dict(color="black", width=2, dash="dash"),
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
            marker=dict(color="red", size=4, opacity=0.6),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=estimates[:, 0],
            y=estimates[:, 1],
            mode="lines",
            name="Kalman Filter Estimate",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Subplot 2: X Position
    fig.add_trace(
        go.Scatter(
            x=time,
            y=true_states[:, 0],
            name="True X",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements[:, 0],
            name="Measured X",
            mode="markers",
            marker=dict(color="red", size=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=estimates[:, 0],
            name="Estimated X",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Y Position
    fig.add_trace(
        go.Scatter(
            x=time,
            y=true_states[:, 1],
            name="True Y",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements[:, 1],
            name="Measured Y",
            mode="markers",
            marker=dict(color="red", size=3),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=estimates[:, 1],
            name="Estimated Y",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Subplot 4: Error
    error = np.linalg.norm(estimates - true_states, axis=1)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=error,
            name="Estimation Error (L2 norm)",
            line=dict(color="green", width=2),
            fill="tozeroy",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="X Position", row=1, col=1)
    fig.update_yaxes(title_text="Y Position", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="X Position", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Y Position", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Error (m)", row=2, col=2)

    fig.update_layout(
        title_text="Kalman Filtering Tutorial - 2D Tracking",
        height=800,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "kalman_filtering.html"))

    print("✓ Kalman filtering visualization complete")
    print("=" * 70)


if __name__ == "__main__":
    kalman_filtering_tutorial()
