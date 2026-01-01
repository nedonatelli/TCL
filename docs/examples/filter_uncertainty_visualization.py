"""
Filter Uncertainty and Covariance Visualization.

This example demonstrates:
1. Plotting covariance ellipses for Kalman filter estimates
2. Visualizing how uncertainty evolves over time
3. Comparing filter predictions with ground truth
4. Animated tracking with uncertainty bounds

Run with: python examples/filter_uncertainty_visualization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.dynamic_estimation.kalman import (  # noqa: E402
    kf_predict,
    kf_update,
)
from pytcl.dynamic_models import (  # noqa: E402
    f_constant_velocity,
    q_constant_velocity,
)


def covariance_ellipse(
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 2.0,
    n_points: int = 100,
) -> tuple:
    """
    Generate points for a 2D covariance ellipse.

    Parameters
    ----------
    mean : ndarray
        Center of the ellipse (x, y).
    cov : ndarray
        2x2 covariance matrix.
    n_std : float
        Number of standard deviations for ellipse size.
    n_points : int
        Number of points to generate.

    Returns
    -------
    x, y : ndarray
        Ellipse coordinates.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute angle
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, n_points)
    a = n_std * np.sqrt(eigenvalues[0])
    b = n_std * np.sqrt(eigenvalues[1])

    # Ellipse in standard position
    x_std = a * np.cos(t)
    y_std = b * np.sin(t)

    # Rotate and translate
    x = mean[0] + x_std * np.cos(angle) - y_std * np.sin(angle)
    y = mean[1] + x_std * np.sin(angle) + y_std * np.cos(angle)

    return x, y


def simulate_tracking_scenario(n_steps: int = 50, dt: float = 1.0) -> dict:
    """
    Simulate a target tracking scenario with Kalman filter.

    Returns
    -------
    dict
        Contains true states, measurements, estimates, and covariances.
    """
    # True initial state [x, vx, y, vy]
    x_true = np.array([0.0, 2.0, 0.0, 1.5])

    # Process and measurement noise
    sigma_a = 0.3  # Acceleration noise
    sigma_z = 2.0  # Measurement noise

    # State transition and process noise
    F = f_constant_velocity(T=dt, num_dims=2)
    Q = q_constant_velocity(T=dt, sigma_a=sigma_a, num_dims=2)

    # Measurement matrix (measure position only)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(2) * sigma_z**2

    # Initial estimate
    x_est = np.array([0.0, 0.0, 0.0, 0.0])
    P_est = np.diag([10.0, 5.0, 10.0, 5.0])

    # Storage
    true_states = [x_true.copy()]
    measurements = []
    estimates = [x_est.copy()]
    covariances = [P_est.copy()]
    predicted_states = []
    predicted_covariances = []

    np.random.seed(42)

    for k in range(n_steps):
        # Propagate true state with some process noise
        process_noise = np.random.multivariate_normal(np.zeros(4), Q)
        x_true = F @ x_true + process_noise

        # Generate measurement
        z = H @ x_true + np.random.multivariate_normal(np.zeros(2), R)

        # Kalman filter predict
        pred = kf_predict(x_est, P_est, F, Q)
        predicted_states.append(pred.x.copy())
        predicted_covariances.append(pred.P.copy())

        # Kalman filter update
        upd = kf_update(pred.x, pred.P, z, H, R)
        x_est = upd.x
        P_est = upd.P

        # Store
        true_states.append(x_true.copy())
        measurements.append(z.copy())
        estimates.append(x_est.copy())
        covariances.append(P_est.copy())

    return {
        "true_states": np.array(true_states),
        "measurements": np.array(measurements),
        "estimates": np.array(estimates),
        "covariances": covariances,
        "predicted_states": np.array(predicted_states),
        "predicted_covariances": predicted_covariances,
        "dt": dt,
    }


def plot_tracking_with_ellipses(data: dict) -> go.Figure:
    """
    Plot tracking results with covariance ellipses.
    """
    fig = go.Figure()

    true_states = data["true_states"]
    measurements = data["measurements"]
    estimates = data["estimates"]
    covariances = data["covariances"]

    # True trajectory
    fig.add_trace(
        go.Scatter(
            x=true_states[:, 0],
            y=true_states[:, 2],
            mode="lines",
            line=dict(color="green", width=3),
            name="True trajectory",
        )
    )

    # Measurements
    fig.add_trace(
        go.Scatter(
            x=measurements[:, 0],
            y=measurements[:, 1],
            mode="markers",
            marker=dict(color="black", size=6, symbol="x"),
            name="Measurements",
        )
    )

    # Estimated trajectory
    fig.add_trace(
        go.Scatter(
            x=estimates[:, 0],
            y=estimates[:, 2],
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            name="Kalman estimate",
        )
    )

    # Covariance ellipses (every 5 steps)
    for i in range(0, len(estimates), 5):
        pos_mean = np.array([estimates[i, 0], estimates[i, 2]])
        pos_cov = np.array(
            [
                [covariances[i][0, 0], covariances[i][0, 2]],
                [covariances[i][2, 0], covariances[i][2, 2]],
            ]
        )

        # 2-sigma ellipse
        ex, ey = covariance_ellipse(pos_mean, pos_cov, n_std=2.0)
        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(color="rgba(0, 100, 255, 0.3)", width=1),
                fill="toself",
                fillcolor="rgba(0, 100, 255, 0.1)",
                name="2σ covariance" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

    fig.update_layout(
        title="Kalman Filter Tracking with Covariance Ellipses",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=1000,
        height=800,
    )

    return fig


def plot_uncertainty_evolution(data: dict) -> go.Figure:
    """
    Plot how position and velocity uncertainties evolve over time.
    """
    covariances = data["covariances"]
    dt = data["dt"]
    n_steps = len(covariances)
    time = np.arange(n_steps) * dt

    # Extract position and velocity standard deviations
    pos_x_std = np.array([np.sqrt(P[0, 0]) for P in covariances])
    pos_y_std = np.array([np.sqrt(P[2, 2]) for P in covariances])
    vel_x_std = np.array([np.sqrt(P[1, 1]) for P in covariances])
    vel_y_std = np.array([np.sqrt(P[3, 3]) for P in covariances])

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Position Uncertainty (1σ)", "Velocity Uncertainty (1σ)"],
        shared_xaxes=True,
    )

    # Position uncertainties
    fig.add_trace(
        go.Scatter(x=time, y=pos_x_std, mode="lines", name="σ_x", line=dict(color="blue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time, y=pos_y_std, mode="lines", name="σ_y", line=dict(color="red")),
        row=1,
        col=1,
    )

    # Velocity uncertainties
    fig.add_trace(
        go.Scatter(
            x=time,
            y=vel_x_std,
            mode="lines",
            name="σ_vx",
            line=dict(color="blue", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=vel_y_std,
            mode="lines",
            name="σ_vy",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position std (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity std (m/s)", row=2, col=1)

    fig.update_layout(
        title="Filter Uncertainty Evolution Over Time",
        width=1000,
        height=600,
    )

    return fig


def plot_estimation_errors(data: dict) -> go.Figure:
    """
    Plot estimation errors with uncertainty bounds.
    """
    true_states = data["true_states"]
    estimates = data["estimates"]
    covariances = data["covariances"]
    dt = data["dt"]

    # Compute errors
    errors = estimates - true_states
    n_steps = len(errors)
    time = np.arange(n_steps) * dt

    # Extract 2-sigma bounds
    pos_x_2sigma = 2 * np.array([np.sqrt(P[0, 0]) for P in covariances])
    pos_y_2sigma = 2 * np.array([np.sqrt(P[2, 2]) for P in covariances])

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["X Position Error", "Y Position Error"],
        shared_xaxes=True,
    )

    # X error with bounds
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([pos_x_2sigma, -pos_x_2sigma[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±2σ bound",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=errors[:, 0],
            mode="lines",
            name="X error",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.zeros(n_steps),
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Y error with bounds
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([pos_y_2sigma, -pos_y_2sigma[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 100, 0, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time, y=errors[:, 2], mode="lines", name="Y error", line=dict(color="red")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.zeros(n_steps),
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)

    fig.update_layout(
        title="Estimation Errors with 2σ Confidence Bounds",
        width=1000,
        height=600,
    )

    return fig


def plot_animated_tracking(data: dict) -> go.Figure:
    """
    Create an animated visualization of the tracking process.
    """
    true_states = data["true_states"]
    measurements = data["measurements"]
    estimates = data["estimates"]
    covariances = data["covariances"]
    n_steps = len(measurements)

    # Create frames for animation
    frames = []

    for k in range(1, n_steps + 1):
        # True trajectory up to current time
        true_trace = go.Scatter(
            x=true_states[: k + 1, 0],
            y=true_states[: k + 1, 2],
            mode="lines",
            line=dict(color="green", width=3),
            name="True",
        )

        # Measurements up to current time
        meas_trace = go.Scatter(
            x=measurements[:k, 0],
            y=measurements[:k, 1],
            mode="markers",
            marker=dict(color="black", size=6, symbol="x"),
            name="Measurements",
        )

        # Estimates up to current time
        est_trace = go.Scatter(
            x=estimates[: k + 1, 0],
            y=estimates[: k + 1, 2],
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            name="Estimate",
        )

        # Current covariance ellipse
        pos_mean = np.array([estimates[k, 0], estimates[k, 2]])
        pos_cov = np.array(
            [
                [covariances[k][0, 0], covariances[k][0, 2]],
                [covariances[k][2, 0], covariances[k][2, 2]],
            ]
        )
        ex, ey = covariance_ellipse(pos_mean, pos_cov, n_std=2.0)

        ellipse_trace = go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(color="rgba(0, 100, 255, 0.5)", width=2),
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.2)",
            name="2σ covariance",
        )

        frames.append(
            go.Frame(
                data=[true_trace, meas_trace, est_trace, ellipse_trace],
                name=str(k),
            )
        )

    # Initial frame
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )

    # Add animation controls
    fig.update_layout(
        title="Animated Kalman Filter Tracking",
        xaxis=dict(
            range=[
                min(true_states[:, 0].min(), estimates[:, 0].min()) - 10,
                max(true_states[:, 0].max(), estimates[:, 0].max()) + 10,
            ],
            title="X Position",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[
                min(true_states[:, 2].min(), estimates[:, 2].min()) - 10,
                max(true_states[:, 2].max(), estimates[:, 2].max()) + 10,
            ],
            title="Y Position",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [str(k)],
                            dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                        ],
                        label=str(k),
                        method="animate",
                    )
                    for k in range(1, n_steps + 1)
                ],
                x=0.1,
                len=0.8,
                xanchor="left",
                y=0,
                yanchor="top",
                currentvalue=dict(
                    prefix="Time step: ",
                    visible=True,
                    xanchor="center",
                ),
            )
        ],
        width=1000,
        height=800,
    )

    return fig


def main():
    """Run filter uncertainty visualization examples."""
    print("Filter Uncertainty Visualization Examples")
    print("=" * 50)

    # Simulate tracking scenario
    print("\nSimulating tracking scenario...")
    data = simulate_tracking_scenario(n_steps=50, dt=1.0)
    print(f"  Generated {len(data['measurements'])} time steps")

    # 1. Tracking with ellipses
    print("\n1. Generating tracking with covariance ellipses...")
    fig1 = plot_tracking_with_ellipses(data)
    fig1.write_html("filter_viz_tracking_ellipses.html")
    print("   Saved to filter_viz_tracking_ellipses.html")

    # 2. Uncertainty evolution
    print("\n2. Generating uncertainty evolution plot...")
    fig2 = plot_uncertainty_evolution(data)
    fig2.write_html("filter_viz_uncertainty_evolution.html")
    print("   Saved to filter_viz_uncertainty_evolution.html")

    # 3. Estimation errors
    print("\n3. Generating estimation error plot...")
    fig3 = plot_estimation_errors(data)
    fig3.write_html("filter_viz_estimation_errors.html")
    print("   Saved to filter_viz_estimation_errors.html")

    # 4. Animated tracking
    print("\n4. Generating animated tracking visualization...")
    fig4 = plot_animated_tracking(data)
    fig4.write_html("filter_viz_animated.html")
    print("   Saved to filter_viz_animated.html")

    # Show all figures
    print("\nOpening visualizations in browser...")
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
