"""
Kalman Filter Comparison Example.

This example demonstrates:
1. Linear Kalman Filter for constant velocity tracking
2. Extended Kalman Filter (EKF) for nonlinear measurements
3. Unscented Kalman Filter (UKF) for highly nonlinear systems
4. Filter consistency checking with NEES/NIS
5. Comparison of filter performance

Run with: python examples/kalman_filter_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.dynamic_estimation import (  # noqa: E402; Linear Kalman Filter; Unscented Kalman Filter
    kf_predict,
    kf_update,
    sigma_points_merwe,
    ukf_predict,
    ukf_update,
)
from pytcl.dynamic_models import (  # noqa: E402
    f_constant_velocity,
    q_constant_velocity,
)


def generate_trajectory(
    n_steps: int = 100,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D constant velocity trajectory with nonlinear measurements.

    Returns
    -------
    true_states : ndarray, shape (n_steps, 4)
        True states [x, vx, y, vy] at each time step.
    linear_measurements : ndarray, shape (n_steps, 2)
        Linear measurements [x, y] with noise.
    nonlinear_measurements : ndarray, shape (n_steps, 2)
        Nonlinear measurements [range, bearing] with noise.
    """
    # Initial state: position (100, 50), velocity (2, 1) m/s
    x0 = np.array([100.0, 2.0, 50.0, 1.0])

    # State transition
    F = f_constant_velocity(dt, 2)

    # Generate true trajectory
    true_states = np.zeros((n_steps, 4))
    true_states[0] = x0

    for k in range(1, n_steps):
        true_states[k] = F @ true_states[k - 1]

    # Measurement noise
    R_linear = np.diag([5.0**2, 5.0**2])  # Position noise std = 5m
    R_nonlinear = np.diag([10.0**2, np.radians(2.0) ** 2])  # Range 10m, bearing 2 deg

    # Generate measurements
    linear_measurements = np.zeros((n_steps, 2))
    nonlinear_measurements = np.zeros((n_steps, 2))

    for k in range(n_steps):
        # True position
        x, y = true_states[k, 0], true_states[k, 2]

        # Linear measurement: direct position
        linear_measurements[k] = np.array([x, y]) + np.random.multivariate_normal(
            [0, 0], R_linear
        )

        # Nonlinear measurement: range and bearing from origin
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        nonlinear_measurements[k] = np.array(
            [r, theta]
        ) + np.random.multivariate_normal([0, 0], R_nonlinear)

    return true_states, linear_measurements, nonlinear_measurements


def run_linear_kf(
    measurements: np.ndarray,
    dt: float = 1.0,
    process_noise: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """
    Run linear Kalman filter on position measurements.

    Returns state estimates, covariances, NEES values, and NIS values.
    """
    n_steps = len(measurements)

    # System matrices
    F = f_constant_velocity(dt, 2)
    Q = q_constant_velocity(dt, process_noise, 2)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # Measure x, y
    R = np.diag([5.0**2, 5.0**2])

    # Initial state and covariance
    x = np.array([measurements[0, 0], 0.0, measurements[0, 1], 0.0])
    P = np.diag([25.0, 10.0, 25.0, 10.0])

    # Storage
    estimates = np.zeros((n_steps, 4))
    covariances = np.zeros((n_steps, 4, 4))
    nis_values = []

    estimates[0] = x
    covariances[0] = P

    for k in range(1, n_steps):
        # Predict
        x, P = kf_predict(x, P, F, Q)

        # Update
        z = measurements[k]
        result = kf_update(x, P, z, H, R)
        x, P = result.x, result.P

        estimates[k] = x
        covariances[k] = P

        # NIS: innovation squared normalized by innovation covariance
        nis_val = float(result.y.T @ np.linalg.solve(result.S, result.y))
        nis_values.append(nis_val)

    return estimates, covariances, nis_values


def nonlinear_measurement(x: np.ndarray) -> np.ndarray:
    """Nonlinear measurement function: h(x) = [range, bearing]."""
    px, py = x[0], x[2]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return np.array([r, theta])


def measurement_jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian of nonlinear measurement function."""
    px, py = x[0], x[2]
    r = np.sqrt(px**2 + py**2)
    r2 = r**2

    # H = dh/dx
    H = np.zeros((2, 4))
    H[0, 0] = px / r  # dr/dx
    H[0, 2] = py / r  # dr/dy
    H[1, 0] = -py / r2  # dtheta/dx
    H[1, 2] = px / r2  # dtheta/dy

    return H


def run_ekf(
    measurements: np.ndarray,
    dt: float = 1.0,
    process_noise: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run Extended Kalman Filter on range-bearing measurements.

    Returns state estimates, covariances, and NIS values.
    """
    n_steps = len(measurements)

    # System matrices (linear dynamics, nonlinear measurements)
    F = f_constant_velocity(dt, 2)
    Q = q_constant_velocity(dt, process_noise, 2)
    R = np.diag([10.0**2, np.radians(2.0) ** 2])

    # Initialize from first measurement
    r0, theta0 = measurements[0]
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)
    x = np.array([x0, 0.0, y0, 0.0])
    P = np.diag([100.0, 10.0, 100.0, 10.0])

    # Storage
    estimates = np.zeros((n_steps, 4))
    covariances = np.zeros((n_steps, 4, 4))
    nis_values = []

    estimates[0] = x
    covariances[0] = P

    for k in range(1, n_steps):
        # Predict (linear dynamics - use standard KF predict)
        x, P = kf_predict(x, P, F, Q)

        # Update with nonlinear measurement (manual EKF update)
        z = measurements[k]
        H = measurement_jacobian(x)
        z_pred = nonlinear_measurement(x)

        # Angle wrapping for bearing innovation
        y = z - z_pred
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # Wrap to [-pi, pi]

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)

        # Update
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

        estimates[k] = x
        covariances[k] = P

        # NIS
        nis_val = float(y.T @ np.linalg.solve(S, y))
        nis_values.append(nis_val)

    return estimates, covariances, nis_values


def run_ukf(
    measurements: np.ndarray,
    dt: float = 1.0,
    process_noise: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run Unscented Kalman Filter on range-bearing measurements.

    Returns state estimates, covariances, and NIS values.
    """
    n_steps = len(measurements)

    # System matrices
    F = f_constant_velocity(dt, 2)
    Q = q_constant_velocity(dt, process_noise, 2)
    R = np.diag([10.0**2, np.radians(2.0) ** 2])

    # UKF parameters
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0

    # Initialize from first measurement
    r0, theta0 = measurements[0]
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)
    x = np.array([x0, 0.0, y0, 0.0])
    P = np.diag([100.0, 10.0, 100.0, 10.0])

    # Storage
    estimates = np.zeros((n_steps, 4))
    covariances = np.zeros((n_steps, 4, 4))
    nis_values = []

    estimates[0] = x
    covariances[0] = P

    # State transition function (linear, but UKF uses it as a function)
    def f(x):
        return F @ x

    for k in range(1, n_steps):
        # Generate sigma points
        sigma_pts, Wm, Wc = sigma_points_merwe(x, P, alpha, beta, kappa)

        # Predict step using UKF
        x, P = ukf_predict(x, P, f, Q, alpha, beta, kappa)

        # Update with nonlinear measurement
        z = measurements[k]
        result = ukf_update(
            x,
            P,
            z,
            nonlinear_measurement,
            R,
            alpha,
            beta,
            kappa,
        )
        x, P = result.x, result.P

        estimates[k] = x
        covariances[k] = P

        # NIS
        nis_val = float(result.y.T @ np.linalg.solve(result.S, result.y))
        nis_values.append(nis_val)

    return estimates, covariances, nis_values


def compute_metrics(
    true_states: np.ndarray,
    estimates: np.ndarray,
    covariances: np.ndarray,
) -> dict:
    """Compute performance metrics."""
    # Position RMSE
    pos_errors = estimates[:, [0, 2]] - true_states[:, [0, 2]]
    pos_rmse = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))

    # Velocity RMSE
    vel_errors = estimates[:, [1, 3]] - true_states[:, [1, 3]]
    vel_rmse = np.sqrt(np.mean(np.sum(vel_errors**2, axis=1)))

    # NEES (Normalized Estimation Error Squared)
    nees_values = []
    for k in range(len(true_states)):
        err = estimates[k] - true_states[k]
        P = covariances[k]
        nees_val = float(err.T @ np.linalg.solve(P, err))
        nees_values.append(nees_val)

    avg_nees = np.mean(nees_values)

    return {
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        "avg_nees": avg_nees,
        "nees_values": nees_values,
    }


def plot_results(
    true_states: np.ndarray,
    linear_meas: np.ndarray,
    kf_est: np.ndarray,
    ekf_est: np.ndarray,
    ukf_est: np.ndarray,
    kf_metrics: dict,
    ekf_metrics: dict,
    ukf_metrics: dict,
) -> None:
    """Create comparison plots."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Trajectory Comparison",
            "Position Error Over Time",
            "NEES Comparison",
            "Filter Performance Summary",
        ),
    )

    # Trajectory plot
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
            x=kf_est[:, 0],
            y=kf_est[:, 2],
            mode="lines",
            name="KF",
            line=dict(color="blue", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ekf_est[:, 0],
            y=ekf_est[:, 2],
            mode="lines",
            name="EKF",
            line=dict(color="red", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ukf_est[:, 0],
            y=ukf_est[:, 2],
            mode="lines",
            name="UKF",
            line=dict(color="green", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Position error over time
    time = np.arange(len(true_states))
    kf_pos_err = np.sqrt(
        (kf_est[:, 0] - true_states[:, 0]) ** 2
        + (kf_est[:, 2] - true_states[:, 2]) ** 2
    )
    ekf_pos_err = np.sqrt(
        (ekf_est[:, 0] - true_states[:, 0]) ** 2
        + (ekf_est[:, 2] - true_states[:, 2]) ** 2
    )
    ukf_pos_err = np.sqrt(
        (ukf_est[:, 0] - true_states[:, 0]) ** 2
        + (ukf_est[:, 2] - true_states[:, 2]) ** 2
    )

    fig.add_trace(
        go.Scatter(x=time, y=kf_pos_err, name="KF Error", line=dict(color="blue")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=time, y=ekf_pos_err, name="EKF Error", line=dict(color="red")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=time, y=ukf_pos_err, name="UKF Error", line=dict(color="green")),
        row=1,
        col=2,
    )

    # NEES comparison
    fig.add_trace(
        go.Scatter(
            x=time,
            y=kf_metrics["nees_values"],
            name="KF NEES",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ekf_metrics["nees_values"],
            name="EKF NEES",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ukf_metrics["nees_values"],
            name="UKF NEES",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )
    # NEES expected value line (state dimension = 4)
    fig.add_trace(
        go.Scatter(
            x=[0, len(time)],
            y=[4, 4],
            name="Expected NEES",
            line=dict(color="black", dash="dash"),
        ),
        row=2,
        col=1,
    )

    # Summary bar chart
    filters = ["KF (linear)", "EKF", "UKF"]
    pos_rmses = [
        kf_metrics["pos_rmse"],
        ekf_metrics["pos_rmse"],
        ukf_metrics["pos_rmse"],
    ]

    fig.add_trace(
        go.Bar(
            x=filters,
            y=pos_rmses,
            name="Position RMSE (m)",
            marker_color=["blue", "red", "green"],
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Kalman Filter Comparison: KF vs EKF vs UKF",
        height=800,
        width=1200,
        showlegend=True,
    )

    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=2)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="NEES", row=2, col=1)
    fig.update_xaxes(title_text="Filter Type", row=2, col=2)
    fig.update_yaxes(title_text="RMSE (m)", row=2, col=2)

    fig.write_html("kalman_filter_comparison.html")
    print("\nInteractive plot saved to kalman_filter_comparison.html")
    fig.show()


def main() -> None:
    """Run Kalman filter comparison."""
    print("Kalman Filter Comparison Example")
    print("=" * 60)

    np.random.seed(42)

    # Generate trajectory and measurements
    print("\nGenerating trajectory and measurements...")
    n_steps = 100
    dt = 1.0
    true_states, linear_meas, nonlinear_meas = generate_trajectory(n_steps, dt)

    print(f"  {n_steps} time steps, dt = {dt}s")
    print(f"  Initial position: ({true_states[0, 0]:.1f}, {true_states[0, 2]:.1f}) m")
    print(f"  Final position: ({true_states[-1, 0]:.1f}, {true_states[-1, 2]:.1f}) m")

    # Run filters
    print("\nRunning filters...")

    print("  Linear Kalman Filter (position measurements)...")
    kf_est, kf_cov, kf_nis = run_linear_kf(linear_meas, dt)

    print("  Extended Kalman Filter (range-bearing measurements)...")
    ekf_est, ekf_cov, ekf_nis = run_ekf(nonlinear_meas, dt)

    print("  Unscented Kalman Filter (range-bearing measurements)...")
    ukf_est, ukf_cov, ukf_nis = run_ukf(nonlinear_meas, dt)

    # Compute metrics
    print("\nComputing performance metrics...")
    kf_metrics = compute_metrics(true_states, kf_est, kf_cov)
    ekf_metrics = compute_metrics(true_states, ekf_est, ekf_cov)
    ukf_metrics = compute_metrics(true_states, ukf_est, ukf_cov)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nLinear Kalman Filter (with position measurements):")
    print(f"  Position RMSE: {kf_metrics['pos_rmse']:.2f} m")
    print(f"  Velocity RMSE: {kf_metrics['vel_rmse']:.2f} m/s")
    print(f"  Average NEES:  {kf_metrics['avg_nees']:.2f} (expected: 4.0)")

    print("\nExtended Kalman Filter (with range-bearing measurements):")
    print(f"  Position RMSE: {ekf_metrics['pos_rmse']:.2f} m")
    print(f"  Velocity RMSE: {ekf_metrics['vel_rmse']:.2f} m/s")
    print(f"  Average NEES:  {ekf_metrics['avg_nees']:.2f} (expected: 4.0)")

    print("\nUnscented Kalman Filter (with range-bearing measurements):")
    print(f"  Position RMSE: {ukf_metrics['pos_rmse']:.2f} m")
    print(f"  Velocity RMSE: {ukf_metrics['vel_rmse']:.2f} m/s")
    print(f"  Average NEES:  {ukf_metrics['avg_nees']:.2f} (expected: 4.0)")

    print("\n" + "-" * 60)
    print("Note: KF uses linear (x,y) measurements, while EKF/UKF use")
    print("nonlinear (range, bearing) measurements. EKF and UKF should")
    print("perform similarly for this mildly nonlinear problem.")
    print("-" * 60)

    # Plot results
    try:
        plot_results(
            true_states,
            linear_meas,
            kf_est,
            ekf_est,
            ukf_est,
            kf_metrics,
            ekf_metrics,
            ukf_metrics,
        )
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
