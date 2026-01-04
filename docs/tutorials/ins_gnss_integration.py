"""INS-GNSS Integration Tutorial with Interactive Visualizations.

This tutorial demonstrates the integration of Inertial Navigation System (INS)
with GNSS (Global Navigation Satellite System) for robust position estimation.
We'll show how these complementary sensors can be fused for continuous navigation.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def ins_gnss_tutorial():
    """Run complete INS-GNSS integration tutorial with visualizations."""
    
    print("=" * 70)
    print("INS-GNSS INTEGRATION TUTORIAL")
    print("=" * 70)
    
    # Step 1: Define Vehicle Trajectory
    print("\nStep 1: Define Vehicle Trajectory")
    print("-" * 70)
    
    dt = 0.1  # 10 Hz update rate
    duration = 60  # 60 seconds
    n_steps = int(duration / dt)
    
    time = np.arange(n_steps) * dt
    
    # Create a realistic vehicle trajectory (circular path)
    speed = 10  # m/s
    radius = 50  # meters
    angular_velocity = speed / radius  # rad/s
    
    # True trajectory
    true_position = np.zeros((n_steps, 2))
    true_velocity = np.zeros((n_steps, 2))
    true_heading = np.zeros(n_steps)
    
    for i in range(n_steps):
        theta = angular_velocity * time[i]
        true_position[i, 0] = radius * np.sin(theta)
        true_position[i, 1] = radius * (1 - np.cos(theta))
        
        true_heading[i] = theta + np.pi / 2
        true_velocity[i, 0] = speed * np.cos(true_heading[i])
        true_velocity[i, 1] = speed * np.sin(true_heading[i])
    
    print(f"Trajectory: {n_steps} steps at {1/dt} Hz")
    print(f"Vehicle path: Circular with radius {radius} m, speed {speed} m/s")
    
    # Step 2: Simulate INS
    print("\nStep 2: Simulate Inertial Navigation System (INS)")
    print("-" * 70)
    
    # INS parameters
    accel_bias_stability = 0.001  # m/s²
    accel_noise = 0.01  # m/s²
    gyro_bias_stability = 0.0001  # rad/s
    gyro_noise = 0.001  # rad/s
    
    # Simulated INS (with drift)
    ins_position = true_position.copy()
    ins_heading = true_heading.copy()
    
    # Add drift
    accel_bias = np.random.randn(2) * accel_bias_stability
    gyro_bias = np.random.randn(1) * gyro_bias_stability
    
    for i in range(1, n_steps):
        # Simulated acceleration measurement (with drift)
        accel_true = (true_velocity[i] - true_velocity[i-1]) / dt
        accel_meas = accel_true + accel_bias + np.random.randn(2) * accel_noise
        
        # Integrate position
        ins_position[i] = ins_position[i-1] + true_velocity[i-1] * dt + 0.5 * accel_meas * dt**2
        
        # Simulated heading measurement (with drift)
        heading_rate_meas = angular_velocity + gyro_bias[0] + np.random.randn(1) * gyro_noise
        ins_heading[i] = ins_heading[i-1] + heading_rate_meas[0] * dt
    
    # Add position drift
    position_drift = np.cumsum(np.random.randn(n_steps, 2) * 0.05, axis=0)
    ins_position = ins_position + position_drift
    
    print(f"INS drift simulated with realistic noise and bias")
    print(f"Heading error accumulation: {np.abs(ins_heading[-1] - true_heading[-1]):.2f} rad")
    
    # Step 3: Simulate GNSS Measurements
    print("\nStep 3: Simulate GNSS Measurements")
    print("-" * 70)
    
    # GNSS update rate (5 Hz, less frequent than INS)
    gnss_rate = 5  # Hz
    gnss_interval = int(1 / (gnss_rate * dt))
    
    # GNSS measurement noise
    gnss_noise = 5.0  # meters (realistic for civilian GPS)
    
    gnss_position = np.full((n_steps, 2), np.nan)
    gnss_times = []
    
    for i in range(0, n_steps, gnss_interval):
        if i < n_steps:
            gnss_position[i] = true_position[i] + np.random.randn(2) * gnss_noise
            gnss_times.append(time[i])
    
    n_gnss = len(gnss_times)
    print(f"GNSS measurements: {n_gnss} updates at {gnss_rate} Hz")
    print(f"GNSS measurement noise: {gnss_noise} m")
    
    # Step 4: Simple Kalman Filter for INS-GNSS Fusion
    print("\nStep 4: Implement INS-GNSS Fusion (Kalman Filter)")
    print("-" * 70)
    
    # State: [x, y, vx, vy]
    x_est = np.zeros((n_steps, 4))
    x_est[0, :2] = ins_position[0]
    x_est[0, 2:4] = true_velocity[0]
    
    P = np.eye(4) * 100  # Initial covariance
    
    # Process noise
    q = 0.1
    Q = q * np.eye(4)
    
    # Measurement noise
    R_gnss = gnss_noise**2 * np.eye(2)
    
    # State transition
    F_kf = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Measurement matrix (position only)
    H_kf = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    fused_position = x_est[:, :2].copy()
    
    for i in range(1, n_steps):
        # Predict
        x_pred = F_kf @ x_est[i-1]
        P_pred = F_kf @ P @ F_kf.T + Q
        
        # Update with GNSS if available
        if not np.isnan(gnss_position[i, 0]):
            z = gnss_position[i]
            
            # Kalman update
            y = z - H_kf @ x_pred  # Innovation
            S = H_kf @ P_pred @ H_kf.T + R_gnss
            K = P_pred @ H_kf.T @ np.linalg.inv(S)
            
            x_est[i] = x_pred + K @ y
            P = (np.eye(4) - K @ H_kf) @ P_pred
        else:
            x_est[i] = x_pred
            P = P_pred
        
        fused_position[i] = x_est[i, :2]
    
    # Calculate errors
    ins_error = np.linalg.norm(ins_position - true_position, axis=1)
    gnss_error = np.full(n_steps, np.nan)
    for i in range(n_steps):
        if not np.isnan(gnss_position[i, 0]):
            gnss_error[i] = np.linalg.norm(gnss_position[i] - true_position[i])
    
    fused_error = np.linalg.norm(fused_position - true_position, axis=1)
    
    print(f"Fusion complete:")
    print(f"  INS RMSE: {np.sqrt(np.mean(ins_error**2)):.2f} m")
    print(f"  GNSS RMSE: {np.nanmean(gnss_error):.2f} m (sparse measurements)")
    print(f"  Fused RMSE: {np.sqrt(np.mean(fused_error**2)):.2f} m")
    
    # Step 5: Visualize Results
    print("\nStep 5: Create Visualizations")
    print("-" * 70)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "2D Trajectory Comparison",
            "North Position Error Over Time",
            "East Position Error Over Time",
            "Horizontal Position Error (RMS)"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Row 1, Col 1: 2D Trajectory
    fig.add_trace(
        go.Scatter(
            x=true_position[:, 0],
            y=true_position[:, 1],
            mode="lines",
            name="True Trajectory",
            line=dict(color="black", width=3, dash="dash"),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=ins_position[:, 0],
            y=ins_position[:, 1],
            mode="lines",
            name="INS Only",
            line=dict(color="blue", width=2),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=gnss_position[:, 0],
            y=gnss_position[:, 1],
            mode="markers",
            name="GNSS Measurements",
            marker=dict(color="red", size=6, symbol="x"),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=fused_position[:, 0],
            y=fused_position[:, 1],
            mode="lines",
            name="INS-GNSS Fused",
            line=dict(color="green", width=2),
        ),
        row=1, col=1
    )
    
    # Row 1, Col 2: North Error
    fig.add_trace(
        go.Scatter(
            x=time, y=ins_position[:, 0] - true_position[:, 0],
            name="INS Error",
            line=dict(color="blue"),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=fused_position[:, 0] - true_position[:, 0],
            name="Fused Error",
            line=dict(color="green"),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Row 2, Col 1: East Error
    fig.add_trace(
        go.Scatter(
            x=time, y=ins_position[:, 1] - true_position[:, 1],
            name="INS Error",
            line=dict(color="blue"),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=fused_position[:, 1] - true_position[:, 1],
            name="Fused Error",
            line=dict(color="green"),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Row 2, Col 2: RMS Error
    fig.add_trace(
        go.Scatter(
            x=time, y=ins_error,
            name=f"INS (RMSE: {np.sqrt(np.mean(ins_error**2)):.2f} m)",
            line=dict(color="blue"),
            fill="tozeroy",
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=fused_error,
            name=f"Fused (RMSE: {np.sqrt(np.mean(fused_error**2)):.2f} m)",
            line=dict(color="green"),
            fill="tozeroy",
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="East (m)", row=1, col=1)
    fig.update_yaxes(title_text="North (m)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="North Error (m)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="East Error (m)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Horizontal Error (m)", row=2, col=2)
    
    fig.update_layout(
        title_text="INS-GNSS Integration Tutorial - Sensor Fusion for Navigation",
        height=800,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "ins_gnss_integration.html"))
    
    print("✓ INS-GNSS integration visualization complete")
    print("=" * 70)


if __name__ == "__main__":
    ins_gnss_tutorial()
