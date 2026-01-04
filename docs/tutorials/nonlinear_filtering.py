"""Nonlinear Filtering Tutorial with Interactive Visualizations.

This tutorial demonstrates Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
for tracking an object undergoing nonlinear motion with bearing and range measurements.

Problem Setup:
We track an object in Cartesian coordinates (x, y, ẋ, ẏ) but receive measurements
in polar coordinates (range, bearing) from a radar at the origin.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def nonlinear_filtering_tutorial():
    """Run complete nonlinear filtering tutorial with visualizations."""
    
    print("=" * 70)
    print("NONLINEAR FILTERING TUTORIAL")
    print("=" * 70)
    
    # Step 1: Define the System Model
    print("\nStep 1: Define the System Model")
    print("-" * 70)
    
    dt = 0.1
    
    # State transition matrix (constant velocity in Cartesian)
    F = np.array([
        [1, dt, 0, 0],
        [0, 1,  0, 0],
        [0, 0,  1, dt],
        [0, 0,  0, 1]
    ])
    
    # Process noise
    q = 0.05
    Q = q * np.eye(4)
    
    # Measurement function: h(x) = [sqrt(x^2 + y^2), arctan(y/x)]
    def h(x):
        """Convert Cartesian to polar coordinates."""
        r = np.sqrt(x[0]**2 + x[2]**2)
        bearing = np.arctan2(x[2], x[0])
        return np.array([r, bearing])
    
    # Measurement Jacobian
    def H(x):
        """Jacobian of measurement function."""
        r = np.sqrt(x[0]**2 + x[2]**2)
        denom = x[0]**2 + x[2]**2
        return np.array([
            [x[0]/r, 0, x[2]/r, 0],
            [-x[2]/denom, 0, x[0]/denom, 0]
        ])
    
    # Measurement noise
    R = np.eye(2) * np.diag([0.1, 0.05])
    
    print("System: Constant velocity (Cartesian state space)")
    print("Measurements: Polar coordinates (range, bearing)")
    
    # Step 2: Generate True Trajectory
    print("\nStep 2: Generate True Trajectory and Measurements")
    print("-" * 70)
    
    np.random.seed(42)
    n_steps = 150
    
    # True trajectory
    x_true = np.array([5.0, 1.0, 5.0, 0.8])
    true_states = []
    
    for _ in range(n_steps):
        true_states.append(x_true[:2].copy())
        w = np.random.multivariate_normal(np.zeros(4), Q)
        x_true = F @ x_true + w
    
    true_states = np.array(true_states)
    
    # Generate measurements in polar coordinates
    measurements = []
    for state in true_states:
        state_full = np.concatenate([state, [0, 0]])
        z = h(state_full) + np.random.multivariate_normal(np.zeros(2), R)
        measurements.append(z)
    
    measurements = np.array(measurements)
    
    # Convert measurements back to Cartesian for plotting
    meas_cart = np.array([[m[0]*np.cos(m[1]), m[0]*np.sin(m[1])] for m in measurements])
    
    print(f"Generated {n_steps} steps of nonlinear trajectory")
    
    # Step 3: Run Extended Kalman Filter
    print("\nStep 3: Run Extended Kalman Filter (EKF)")
    print("-" * 70)
    
    x = np.array([4.0, 1.0, 4.0, 0.8])  # Initial state (slightly off)
    P = np.eye(4) * 2.0
    
    ekf_estimates = []
    ekf_covariances = []
    
    for z in measurements:
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update
        h_pred = h(x_pred)
        H_jac = H(x_pred)
        
        y = z - h_pred  # Innovation
        S = H_jac @ P_pred @ H_jac.T + R  # Innovation covariance
        K = P_pred @ H_jac.T @ np.linalg.inv(S)  # Kalman gain
        
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H_jac) @ P_pred
        
        ekf_estimates.append(x[:2].copy())
        ekf_covariances.append(np.trace(P[:2, :2]))
    
    ekf_estimates = np.array(ekf_estimates)
    ekf_covariances = np.array(ekf_covariances)
    
    ekf_error = np.linalg.norm(ekf_estimates - true_states, axis=1)
    ekf_rmse = np.sqrt(np.mean(ekf_error**2))
    
    print(f"EKF RMSE: {ekf_rmse:.4f}")
    
    # Step 4: Simple UKF Implementation
    print("\nStep 4: Run Unscented Kalman Filter (UKF)")
    print("-" * 70)
    
    # UKF parameters
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0
    
    n = 4
    lambda_ = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lambda_)
    
    x = np.array([4.0, 1.0, 4.0, 0.8])
    P = np.eye(4) * 2.0
    
    ukf_estimates = []
    ukf_covariances = []
    
    for z in measurements:
        # Generate sigma points
        L = np.linalg.cholesky(P)
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i+1] = x + gamma * L[:, i]
            sigma_points[n+i+1] = x - gamma * L[:, i]
        
        # Predict
        sigma_pred = np.array([F @ sp for sp in sigma_points])
        x_pred = sigma_pred.mean(axis=0)
        P_pred = np.cov(sigma_pred.T) + Q
        
        # Update
        h_sigma = np.array([h(sp) for sp in sigma_pred])
        z_pred = h_sigma.mean(axis=0)
        
        # Correct the cross-covariance calculation
        Pzz = np.eye(2) * 1e-10
        Pxz = np.zeros((4, 2))
        for sp, hs in zip(sigma_pred, h_sigma):
            Pzz += np.outer(hs - z_pred, hs - z_pred)
            Pxz += np.outer(sp - x_pred, hs - z_pred)
        Pzz = Pzz / len(sigma_pred) + R
        Pxz = Pxz / len(sigma_pred)
        
        K = Pxz @ np.linalg.inv(Pzz)
        
        x = x_pred + K @ (z - z_pred)
        P = P_pred - K @ Pzz @ K.T
        
        ukf_estimates.append(x[:2].copy())
        ukf_covariances.append(np.trace(P[:2, :2]))
    
    ukf_estimates = np.array(ukf_estimates)
    ukf_covariances = np.array(ukf_covariances)
    
    ukf_error = np.linalg.norm(ukf_estimates - true_states, axis=1)
    ukf_rmse = np.sqrt(np.mean(ukf_error**2))
    
    print(f"UKF RMSE: {ukf_rmse:.4f}")
    
    # Step 5: Visualize Results
    print("\nStep 5: Create Visualizations")
    print("-" * 70)
    
    time = np.arange(n_steps) * dt
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "2D Trajectory: EKF vs UKF",
            "X Position Over Time",
            "Y Position Over Time",
            "Filter Comparison (RMSE)"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Subplot 1: 2D Trajectory
    fig.add_trace(
        go.Scatter(
            x=true_states[:, 0],
            y=true_states[:, 1],
            mode="lines",
            name="True Trajectory",
            line=dict(color="black", width=3, dash="dash"),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=meas_cart[:, 0],
            y=meas_cart[:, 1],
            mode="markers",
            name="Measurements (Polar)",
            marker=dict(color="gray", size=4, opacity=0.5),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=ekf_estimates[:, 0],
            y=ekf_estimates[:, 1],
            mode="lines",
            name="EKF Estimate",
            line=dict(color="blue", width=2),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=ukf_estimates[:, 0],
            y=ukf_estimates[:, 1],
            mode="lines",
            name="UKF Estimate",
            line=dict(color="red", width=2),
        ),
        row=1, col=1
    )
    
    # Subplot 2: X Position
    fig.add_trace(
        go.Scatter(x=time, y=true_states[:, 0], name="True",
                   line=dict(color="black", dash="dash"), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=time, y=ekf_estimates[:, 0], name="EKF",
                   line=dict(color="blue"), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=time, y=ukf_estimates[:, 0], name="UKF",
                   line=dict(color="red"), showlegend=False),
        row=1, col=2
    )
    
    # Subplot 3: Y Position
    fig.add_trace(
        go.Scatter(x=time, y=true_states[:, 1], name="True",
                   line=dict(color="black", dash="dash"), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=ekf_estimates[:, 1], name="EKF",
                   line=dict(color="blue"), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=ukf_estimates[:, 1], name="UKF",
                   line=dict(color="red"), showlegend=False),
        row=2, col=1
    )
    
    # Subplot 4: Comparison
    fig.add_trace(
        go.Scatter(
            x=time, y=ekf_error,
            name=f"EKF Error (RMSE: {ekf_rmse:.4f})",
            line=dict(color="blue"),
            fill="tozeroy",
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=ukf_error,
            name=f"UKF Error (RMSE: {ukf_rmse:.4f})",
            line=dict(color="red"),
            fill="tozeroy",
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="X Position (m)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Error (m)", row=2, col=2)
    
    fig.update_layout(
        title_text="Nonlinear Filtering Tutorial - EKF vs UKF",
        height=800,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "nonlinear_filtering.html"))
    
    print("✓ Nonlinear filtering visualization complete")
    print("=" * 70)


if __name__ == "__main__":
    nonlinear_filtering_tutorial()
