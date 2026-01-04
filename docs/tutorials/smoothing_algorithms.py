"""
Smoothing Algorithms Tutorial
=============================

This tutorial demonstrates state smoothing algorithms that use future
information to improve state estimates.

Topics covered:
  - Forward filtering
  - Backward smoothing pass
  - Rauch-Tung-Striebel (RTS) smoother
  - Comparison with filter-only estimates
"""

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def smoothing_tutorial():
    """Run complete smoothing algorithm tutorial with visualizations."""
    
    print("\n" + "="*70)
    print("SMOOTHING ALGORITHMS TUTORIAL")
    print("="*70)
    
    # Step 1: Define system
    print("\nStep 1: Define Linear System Model")
    print("-" * 70)
    
    np.random.seed(42)
    dt = 0.1
    n_steps = 100
    
    # System matrices
    F = np.array([[1, dt], [0, 1]])  # state transition
    H = np.array([[1, 0]])  # measurement matrix
    Q = np.eye(2) * 0.01  # process noise
    R = np.array([[0.1]])  # measurement noise
    
    print("System: Constant velocity model")
    print(f"  State: [position, velocity]")
    print(f"  Process noise cov Q: {Q[0,0]:.3f}")
    print(f"  Measurement noise std: {np.sqrt(R[0,0]):.3f}")
    
    # Step 2: Generate true trajectory
    print("\nStep 2: Generate True Trajectory and Measurements")
    print("-" * 70)
    
    x_true = np.zeros((n_steps, 2))
    x_true[0] = [0.0, 1.0]  # initial position, velocity
    
    # Generate trajectory with acceleration
    for k in range(1, n_steps):
        x_true[k] = F @ x_true[k-1]
        if 30 < k < 70:  # add acceleration phase
            x_true[k] += np.array([0, 0.1])
    
    # Generate measurements
    z_all = np.zeros((n_steps, 1))
    for k in range(n_steps):
        z_all[k] = H @ x_true[k] + np.random.randn() * np.sqrt(R[0, 0])
    
    print(f"Generated {n_steps} steps of trajectory with acceleration phase")
    
    # Step 3: Forward Kalman Filter
    print("\nStep 3: Run Forward Kalman Filter")
    print("-" * 70)
    
    x_filt = np.zeros((n_steps, 2))
    P_filt = np.zeros((n_steps, 2, 2))
    x_filt[0] = [0.0, 1.0]
    P_filt[0] = np.eye(2) * 1.0
    
    for k in range(1, n_steps):
        # Prediction
        x_pred = F @ x_filt[k-1]
        P_pred = F @ P_filt[k-1] @ F.T + Q
        
        # Update
        y = z_all[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S[0, 0]
        
        x_filt[k] = x_pred + K.flatten() * y[0]
        P_filt[k] = (np.eye(2) - K @ H) @ P_pred
    
    rmse_filt = np.sqrt(np.mean((x_filt - x_true)**2))
    print(f"Filter RMSE: {rmse_filt:.4f}")
    
    # Step 4: RTS Backward Smoother
    print("\nStep 4: Run Rauch-Tung-Striebel (RTS) Smoother")
    print("-" * 70)
    
    x_smooth = np.zeros((n_steps, 2))
    x_smooth[-1] = x_filt[-1]
    
    P_smooth = np.zeros((n_steps, 2, 2))
    P_smooth[-1] = P_filt[-1]
    
    # Backward pass
    for k in range(n_steps - 2, -1, -1):
        # Predicted covariance for smoothing
        x_pred_next = F @ x_filt[k]
        P_pred_next = F @ P_filt[k] @ F.T + Q
        
        # Smoother gain
        A = P_filt[k] @ F.T @ np.linalg.inv(P_pred_next)
        
        # Smoothed estimate
        x_smooth[k] = x_filt[k] + A @ (x_smooth[k+1] - x_pred_next)
        P_smooth[k] = P_filt[k] + A @ (P_smooth[k+1] - P_pred_next) @ A.T
    
    rmse_smooth = np.sqrt(np.mean((x_smooth - x_true)**2))
    print(f"RTS Smoother RMSE: {rmse_smooth:.4f}")
    print(f"Improvement: {(rmse_filt - rmse_smooth) / rmse_filt * 100:.1f}%")
    
    # Step 5: Create visualizations
    print("\nStep 5: Create Visualizations")
    print("-" * 70)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Position Estimation: Filter vs Smoother",
            "Velocity Estimation",
            "Position Error Distribution",
            "Estimation Uncertainty (1-sigma)"
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Plot 1: Position
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=z_all.flatten(),
                   name="Measurements", mode="markers",
                   marker=dict(color="gray", size=4, opacity=0.4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=x_true[:, 0],
                   name="True", mode="lines",
                   line=dict(color="black", width=3, dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=x_filt[:, 0],
                   name="Kalman Filter", mode="lines",
                   line=dict(color="blue", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=x_smooth[:, 0],
                   name="RTS Smoother", mode="lines",
                   line=dict(color="red", width=2, dash="dot")),
        row=1, col=1
    )
    
    # Plot 2: Velocity
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=x_true[:, 1],
                   name="True", mode="lines",
                   line=dict(color="black", width=3, dash="dash"),
                   showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=x_filt[:, 1],
                   name="Kalman Filter", mode="lines",
                   line=dict(color="blue", width=2), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=x_smooth[:, 1],
                   name="RTS Smoother", mode="lines",
                   line=dict(color="red", width=2, dash="dot"), showlegend=False),
        row=1, col=2
    )
    
    # Plot 3: Position error
    error_filt = x_filt[:, 0] - x_true[:, 0]
    error_smooth = x_smooth[:, 0] - x_true[:, 0]
    
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=error_filt,
                   name="Filter Error", mode="lines",
                   line=dict(color="blue", width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=error_smooth,
                   name="Smoother Error", mode="lines",
                   line=dict(color="red", width=2)),
        row=2, col=1
    )
    
    # Plot 4: Uncertainty
    std_filt = np.sqrt([P_filt[k, 0, 0] for k in range(n_steps)])
    std_smooth = np.sqrt([P_smooth[k, 0, 0] for k in range(n_steps)])
    
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=std_filt,
                   name="Filter Uncertainty", mode="lines",
                   line=dict(color="blue", width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=std_smooth,
                   name="Smoother Uncertainty", mode="lines",
                   line=dict(color="red", width=2)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=2)
    
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig.update_yaxes(title_text="1-sigma (m)", row=2, col=2)
    
    fig.update_layout(
        title="Smoothing Algorithms Tutorial - Kalman Filter vs RTS Smoother",
        height=700,
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)"
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "smoothing_algorithms.html"))
    
    print("✓ Smoothing algorithms visualization complete")
    print("\n" + "="*70)


if __name__ == "__main__":
    smoothing_tutorial()
