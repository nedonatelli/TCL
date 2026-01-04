"""
Particle Filtering Tutorial
===========================

This tutorial demonstrates Sequential Monte Carlo (particle filter) methods
for nonlinear, non-Gaussian state estimation problems.

Topics covered:
  - Bootstrap particle filter (BPF)
  - Particle resampling strategies
  - Weight degeneracy problem
  - Comparison with Kalman filters for nonlinear systems
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def particle_filters_tutorial():
    """Run complete particle filtering tutorial with visualizations."""

    print("\n" + "=" * 70)
    print("PARTICLE FILTERING TUTORIAL")
    print("=" * 70)

    # Step 1: Define nonlinear system
    print("\nStep 1: Define Nonlinear System Model")
    print("-" * 70)

    def process_model(x, dt):
        """Nonlinear state transition: x_k = x_{k-1} + dt*v + 0.5*sin(x_1)*dt^2"""
        return np.array(
            [
                x[0] + x[1] * dt + 0.5 * np.sin(x[0]) * dt**2,  # position
                x[1] + np.sin(x[0]) * dt,  # velocity
            ]
        )

    def measurement_model(x):
        """Nonlinear measurement: z = x_1^2"""
        return np.array([x[0] ** 2])

    print("System: Nonlinear oscillating dynamics")
    print("  x_k = x_{k-1} + dt*v + 0.5*sin(x_1)*dt^2")
    print("  z_k = x_1^2 + noise")

    # Step 2: Generate trajectory
    print("\nStep 2: Generate True Trajectory and Measurements")
    print("-" * 70)

    np.random.seed(42)
    dt = 0.1
    n_steps = 150

    # True trajectory
    x_true = np.zeros((n_steps, 2))
    x_true[0] = [0.0, 1.0]

    for k in range(1, n_steps):
        x_true[k] = process_model(x_true[k - 1], dt)

    # Measurements
    z_all = np.zeros((n_steps, 1))
    z_noise_std = 0.5
    for k in range(n_steps):
        z_all[k] = measurement_model(x_true[k]) + np.random.randn() * z_noise_std

    print(f"Generated {n_steps} steps of nonlinear trajectory")
    print(f"Measurement noise std: {z_noise_std}")

    # Step 3: Bootstrap Particle Filter
    print("\nStep 3: Run Bootstrap Particle Filter (BPF)")
    print("-" * 70)

    n_particles = 100
    q_std = 0.05  # process noise
    r_std = z_noise_std  # measurement noise

    # Initialize particles
    particles = np.random.randn(n_particles, 2) * 0.1
    particles[:, 1] = 1.0  # velocity ~ 1.0

    weights = np.ones(n_particles) / n_particles
    x_est_bpf = np.zeros((n_steps, 2))
    particles_history = []

    for k in range(n_steps):
        # Prediction
        for i in range(n_particles):
            particles[i] = process_model(particles[i], dt)
            particles[i] += np.random.randn(2) * q_std

        # Update weights based on measurement likelihood
        for i in range(n_particles):
            z_pred = measurement_model(particles[i])
            residual = z_all[k] - z_pred
            weights[i] *= np.exp(-0.5 * residual**2 / r_std**2)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Estimate
        x_est_bpf[k] = np.average(particles, axis=0, weights=weights)
        particles_history.append(particles.copy())

        # Resampling (systematic)
        if 1.0 / np.sum(weights**2) < n_particles / 2:  # ESS criterion
            indices = np.argsort(np.random.rand(n_particles) - np.cumsum(weights))[
                :n_particles
            ]
            particles = particles[indices]
            weights = np.ones(n_particles) / n_particles

    # Calculate RMSE
    rmse_bpf = np.sqrt(np.mean((x_est_bpf - x_true) ** 2))
    print(f"Bootstrap PF RMSE: {rmse_bpf:.4f}")

    # Step 4: Extended Kalman Filter for comparison
    print("\nStep 4: Compare with Extended Kalman Filter")
    print("-" * 70)

    # EKF implementation
    x_ekf = np.zeros((n_steps, 2))
    x_ekf[0] = [0.0, 1.0]
    P = np.eye(2) * 1.0
    Q = np.eye(2) * q_std**2
    R = np.array([[r_std**2]])

    for k in range(1, n_steps):
        # Prediction
        x_pred = process_model(x_ekf[k - 1], dt)

        # Jacobian of process model
        x = x_ekf[k - 1]
        F = np.array([[1.0, dt], [np.cos(x[0]) * dt, 1.0]])
        P = F @ P @ F.T + Q

        # Update
        z_pred = measurement_model(x_pred)
        residual = z_all[k] - z_pred

        # Jacobian of measurement model
        H = np.array([[2 * x_pred[0], 0.0]])
        S = H @ P @ H.T + R
        K = P @ H.T / S[0, 0]

        x_ekf[k] = x_pred + K.flatten() * residual[0]
        P = (np.eye(2) - K @ H) @ P

    rmse_ekf = np.sqrt(np.mean((x_ekf - x_true) ** 2))
    print(f"Extended KF RMSE: {rmse_ekf:.4f}")

    # Step 5: Create visualizations
    print("\nStep 5: Create Visualizations")
    print("-" * 70)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "True vs Estimated Trajectory (Position)",
            "Velocity Estimation",
            "Position Error Over Time",
            "Measurement vs Filter Estimates",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    # Plot 1: Position
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_true[:, 0],
            name="True",
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_est_bpf[:, 0],
            name="Particle Filter",
            mode="lines",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_ekf[:, 0],
            name="Extended KF",
            mode="lines",
            line=dict(color="red", width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Plot 2: Velocity
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_true[:, 1],
            name="True",
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_est_bpf[:, 1],
            name="Particle Filter",
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_ekf[:, 1],
            name="Extended KF",
            mode="lines",
            line=dict(color="red", width=2, dash="dot"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Plot 3: Position error
    error_bpf = np.sqrt((x_est_bpf - x_true) ** 2).sum(axis=1)
    error_ekf = np.sqrt((x_ekf - x_true) ** 2).sum(axis=1)

    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=error_bpf,
            name="BPF Error",
            mode="lines",
            line=dict(color="blue", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=error_ekf,
            name="EKF Error",
            mode="lines",
            line=dict(color="red", width=2),
        ),
        row=2,
        col=1,
    )

    # Plot 4: Measurement vs estimates
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=z_all.flatten(),
            name="Measurements",
            mode="markers",
            marker=dict(color="gray", size=4, opacity=0.6),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_est_bpf[:, 0] ** 2,
            name="BPF Prediction",
            mode="lines",
            line=dict(color="blue", width=2),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_steps),
            y=x_ekf[:, 0] ** 2,
            name="EKF Prediction",
            mode="lines",
            line=dict(color="red", width=2, dash="dot"),
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=2)

    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig.update_yaxes(title_text="Measurement (m²)", row=2, col=2)

    fig.update_layout(
        title="Particle Filtering Tutorial - Bootstrap PF vs EKF",
        height=700,
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "particle_filters.html"))

    print("✓ Particle filters visualization complete")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    particle_filters_tutorial()
