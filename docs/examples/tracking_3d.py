"""
3D Tracking Example
===================

This example demonstrates target tracking with 3D position measurements
(x, y, z coordinates). It covers:

State Estimation:
- 3D constant-velocity Kalman filter
- Extended Kalman filter for nonlinear measurements
- RTS smoother for batch processing

Measurement Types:
- Direct Cartesian (x, y, z) measurements
- Spherical (range, azimuth, elevation) measurements
- Multi-sensor fusion in 3D

Applications:
- Aircraft tracking
- Spacecraft tracking
- UAV/drone tracking
- Marine vessel tracking (with depth)

3D tracking extends 2D tracking by adding the z (altitude/depth)
dimension, which is essential for air and space applications.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global flag to control plotting
SHOW_PLOTS = True


from pytcl.dynamic_estimation import (
    RTSResult,
    kf_predict,
    kf_update,
    rts_smoother,
)


def generate_3d_cv_trajectory(
    n_steps: int = 100,
    dt: float = 1.0,
    process_noise: float = 0.1,
    measurement_noise: float = 2.0,
    seed: int = 42,
):
    """Generate a 3D constant-velocity trajectory with measurements.

    The state vector is [x, vx, y, vy, z, vz] (6 states).
    Measurements are [x, y, z] (3D position).

    Returns:
        true_states: (n_steps, 6) array of states
        measurements: list of (3,) measurement arrays [x, y, z]
        F: state transition matrix (6x6)
        Q: process noise covariance (6x6)
        H: measurement matrix (3x6)
        R: measurement noise covariance (3x3)
    """
    rng = np.random.default_rng(seed)

    # State: [x, vx, y, vy, z, vz]
    # Constant velocity model in 3D
    F = np.array(
        [
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    # Process noise (discrete white noise acceleration in each axis)
    q = process_noise
    Q_1d = (
        np.array(
            [
                [dt**3 / 3, dt**2 / 2],
                [dt**2 / 2, dt],
            ]
        )
        * q
    )

    # Build full 6x6 Q matrix
    Q = np.zeros((6, 6))
    Q[0:2, 0:2] = Q_1d  # x, vx
    Q[2:4, 2:4] = Q_1d  # y, vy
    Q[4:6, 4:6] = Q_1d  # z, vz

    # Measurement: observe 3D position [x, y, z]
    H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )

    R = np.eye(3) * measurement_noise**2

    # Generate true trajectory - climbing turn
    true_states = np.zeros((n_steps, 6))
    # Start at position (0, 0, 1000) with velocity (50, 30, 5) m/s
    true_states[0] = [0, 50, 0, 30, 1000, 5]

    for k in range(1, n_steps):
        # Propagate with process noise
        process_noise_sample = rng.multivariate_normal(np.zeros(6), Q)
        true_states[k] = F @ true_states[k - 1] + process_noise_sample

    # Generate measurements
    measurements = []
    for k in range(n_steps):
        meas_noise = rng.multivariate_normal(np.zeros(3), R)
        z = H @ true_states[k] + meas_noise
        measurements.append(z)

    return true_states, measurements, F, Q, H, R


def demo_3d_kalman_filter():
    """Demonstrate Kalman filter for 3D tracking."""
    print("=" * 70)
    print("3D Kalman Filter Demo")
    print("=" * 70)

    # Generate 3D trajectory
    n_steps = 100
    dt = 1.0
    true_states, measurements, F, Q, H, R = generate_3d_cv_trajectory(n_steps=n_steps, dt=dt)

    print(f"\nSimulating {n_steps} time steps of 3D tracking")
    print("State vector: [x, vx, y, vy, z, vz]")
    print("Measurements: [x, y, z] (3D position)")

    # Initial state estimate
    x = np.array([0, 0, 0, 0, 1000, 0])  # Unknown velocities
    P = np.diag([100, 50, 100, 50, 100, 50])  # High initial uncertainty

    # Run Kalman filter
    estimates = []
    covariances = []

    for k in range(n_steps):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        z = measurements[k]
        y = z - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        x = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred

        estimates.append(x.copy())
        covariances.append(P.copy())

    estimates = np.array(estimates)

    # Compute errors
    pos_errors = np.sqrt(
        (estimates[:, 0] - true_states[:, 0]) ** 2
        + (estimates[:, 2] - true_states[:, 2]) ** 2
        + (estimates[:, 4] - true_states[:, 4]) ** 2
    )

    vel_errors = np.sqrt(
        (estimates[:, 1] - true_states[:, 1]) ** 2
        + (estimates[:, 3] - true_states[:, 3]) ** 2
        + (estimates[:, 5] - true_states[:, 5]) ** 2
    )

    print(f"\nPosition RMSE: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
    print(f"Velocity RMSE: {np.sqrt(np.mean(vel_errors**2)):.2f} m/s")

    # Show trajectory snapshots
    print("\nTrajectory snapshots:")
    print("-" * 70)
    print(f"{'Time':>6} {'True X':>10} {'True Y':>10} {'True Z':>10} {'3D Error':>10}")
    print("-" * 70)
    for t in [0, 25, 50, 75, 99]:
        true_pos = true_states[t, [0, 2, 4]]
        print(
            f"{t:>6} {true_pos[0]:>10.1f} {true_pos[1]:>10.1f} "
            f"{true_pos[2]:>10.1f} {pos_errors[t]:>10.2f}"
        )

    # Plot 3D trajectory
    if SHOW_PLOTS:
        measurements_arr = np.array(measurements)
        time = np.arange(n_steps) * dt

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "3D Trajectory",
                "Position Error Over Time",
                "XY Projection (Top View)",
            ),
        )

        # 3D trajectory plot
        fig.add_trace(
            go.Scatter3d(
                x=true_states[:, 0],
                y=true_states[:, 2],
                z=true_states[:, 4],
                mode="lines",
                name="True",
                line=dict(color="blue", width=4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=estimates[:, 0],
                y=estimates[:, 2],
                z=estimates[:, 4],
                mode="lines",
                name="Estimate",
                line=dict(color="red", width=3, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=measurements_arr[::5, 0],
                y=measurements_arr[::5, 1],
                z=measurements_arr[::5, 2],
                mode="markers",
                name="Measurements",
                marker=dict(color="gray", size=3, opacity=0.5),
            ),
            row=1,
            col=1,
        )

        # Position error over time
        fig.add_trace(
            go.Scatter(
                x=time,
                y=pos_errors,
                mode="lines",
                name="Position Error",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[time[0], time[-1]],
                y=[np.mean(pos_errors), np.mean(pos_errors)],
                mode="lines",
                name=f"Mean={np.mean(pos_errors):.2f}",
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=1,
            col=2,
        )

        # XY projection
        fig.add_trace(
            go.Scatter(
                x=true_states[:, 0],
                y=true_states[:, 2],
                mode="lines",
                name="True",
                line=dict(color="blue", width=3),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=estimates[:, 0],
                y=estimates[:, 2],
                mode="lines",
                name="Estimate",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=measurements_arr[::5, 0],
                y=measurements_arr[::5, 1],
                mode="markers",
                name="Measurements",
                marker=dict(color="gray", size=4, opacity=0.5),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            title="3D Kalman Filter Tracking",
            height=500,
            width=1400,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="3D Position Error (m)", row=1, col=2)
        fig.update_xaxes(title_text="X (m)", row=1, col=3)
        fig.update_yaxes(title_text="Y (m)", row=1, col=3)

        fig.write_html("tracking_3d_kalman.html")
        print("\n  [Plot saved to tracking_3d_kalman.html]")

    return estimates, true_states, measurements


def demo_3d_rts_smoother():
    """Demonstrate RTS smoother for 3D tracking."""
    print("\n" + "=" * 70)
    print("3D RTS Smoother Demo")
    print("=" * 70)

    # Generate 3D trajectory
    n_steps = 100
    true_states, measurements, F, Q, H, R = generate_3d_cv_trajectory(n_steps=n_steps)

    # Initial state
    x0 = np.array([0, 0, 0, 0, 1000, 0])
    P0 = np.diag([100, 50, 100, 50, 100, 50])

    # Run RTS smoother
    result = rts_smoother(x0, P0, measurements, F, Q, H, R)

    # Compute errors for filter vs smoother
    filter_pos_errors = []
    smooth_pos_errors = []

    for k in range(n_steps):
        true = true_states[k, [0, 2, 4]]  # x, y, z

        filt_pos = result.x_filt[k][[0, 2, 4]]
        smooth_pos = result.x_smooth[k][[0, 2, 4]]

        filter_pos_errors.append(np.linalg.norm(filt_pos - true))
        smooth_pos_errors.append(np.linalg.norm(smooth_pos - true))

    print("\n3D Position RMSE comparison:")
    print(f"  Filter:   {np.sqrt(np.mean(np.array(filter_pos_errors)**2)):.2f} m")
    print(f"  Smoother: {np.sqrt(np.mean(np.array(smooth_pos_errors)**2)):.2f} m")

    improvement = (1 - np.mean(smooth_pos_errors) / np.mean(filter_pos_errors)) * 100
    print(f"  Improvement: {improvement:.1f}%")

    # Velocity comparison
    filter_vel_errors = []
    smooth_vel_errors = []

    for k in range(n_steps):
        true = true_states[k, [1, 3, 5]]  # vx, vy, vz

        filt_vel = result.x_filt[k][[1, 3, 5]]
        smooth_vel = result.x_smooth[k][[1, 3, 5]]

        filter_vel_errors.append(np.linalg.norm(filt_vel - true))
        smooth_vel_errors.append(np.linalg.norm(smooth_vel - true))

    print("\n3D Velocity RMSE comparison:")
    print(f"  Filter:   {np.sqrt(np.mean(np.array(filter_vel_errors)**2)):.2f} m/s")
    print(f"  Smoother: {np.sqrt(np.mean(np.array(smooth_vel_errors)**2)):.2f} m/s")

    vel_improvement = (1 - np.mean(smooth_vel_errors) / np.mean(filter_vel_errors)) * 100
    print(f"  Improvement: {vel_improvement:.1f}%")

    # Plot comparison
    if SHOW_PLOTS:
        smooth_states = np.array(result.x_smooth)
        time = np.arange(n_steps)

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "3D Smoothed Trajectory",
                "Filter vs Smoother Error",
                "Uncertainty Comparison",
            ),
        )

        # 3D trajectory comparison
        fig.add_trace(
            go.Scatter3d(
                x=true_states[:, 0],
                y=true_states[:, 2],
                z=true_states[:, 4],
                mode="lines",
                name="True",
                line=dict(color="blue", width=4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=smooth_states[:, 0],
                y=smooth_states[:, 2],
                z=smooth_states[:, 4],
                mode="lines",
                name="Smoothed",
                line=dict(color="red", width=3, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Position error comparison
        fig.add_trace(
            go.Scatter(
                x=time,
                y=filter_pos_errors,
                mode="lines",
                name="Filter",
                line=dict(color="blue", width=2),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=smooth_pos_errors,
                mode="lines",
                name="Smoother",
                line=dict(color="red", width=2),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

        # Uncertainty comparison (trace of P)
        filter_trace = [np.trace(result.P_filt[k]) for k in range(n_steps)]
        smooth_trace = [np.trace(result.P_smooth[k]) for k in range(n_steps)]
        fig.add_trace(
            go.Scatter(
                x=time,
                y=filter_trace,
                mode="lines",
                name="Filter",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=smooth_trace,
                mode="lines",
                name="Smoother",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            title="RTS Smoother Comparison",
            height=500,
            width=1400,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time step", row=1, col=2)
        fig.update_yaxes(title_text="3D Position Error (m)", row=1, col=2)
        fig.update_xaxes(title_text="Time step", row=1, col=3)
        fig.update_yaxes(title_text="Covariance Trace", row=1, col=3)

        fig.write_html("tracking_3d_smoother.html")
        print("\n  [Plot saved to tracking_3d_smoother.html]")


def demo_spherical_measurements():
    """Demonstrate 3D tracking with spherical (radar) measurements."""
    print("\n" + "=" * 70)
    print("Spherical Measurements (Radar) Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate true 3D trajectory (aircraft flight path)
    n_steps = 80
    dt = 1.0

    # State: [x, vx, y, vy, z, vz]
    F = np.array(
        [
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    q = 0.5
    Q_1d = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
    Q = np.zeros((6, 6))
    Q[0:2, 0:2] = Q_1d
    Q[2:4, 2:4] = Q_1d
    Q[4:6, 4:6] = Q_1d

    # True trajectory - aircraft at 10km altitude, approaching
    true_states = np.zeros((n_steps, 6))
    true_states[0] = [50000, -200, 30000, -100, 10000, 0]  # 50km away, approaching

    for k in range(1, n_steps):
        process_noise = np.random.multivariate_normal(np.zeros(6), Q * 0.1)
        true_states[k] = F @ true_states[k - 1] + process_noise

    print("\nScenario: Aircraft tracking with radar")
    print(
        f"  Initial position: ({true_states[0, 0]/1000:.1f}, "
        f"{true_states[0, 2]/1000:.1f}, {true_states[0, 4]/1000:.1f}) km"
    )
    print(
        f"  Final position: ({true_states[-1, 0]/1000:.1f}, "
        f"{true_states[-1, 2]/1000:.1f}, {true_states[-1, 4]/1000:.1f}) km"
    )

    # Radar measurement function: Cartesian -> (range, azimuth, elevation)
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x**2 + y**2))
        return np.array([r, az, el])

    def spherical_to_cartesian(r, az, el):
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return np.array([x, y, z])

    # Measurement noise (typical radar)
    sigma_r = 50.0  # 50m range noise
    sigma_az = np.radians(0.5)  # 0.5 degree azimuth noise
    sigma_el = np.radians(0.5)  # 0.5 degree elevation noise
    R_spherical = np.diag([sigma_r**2, sigma_az**2, sigma_el**2])

    # Generate spherical measurements
    spherical_measurements = []
    for k in range(n_steps):
        x, y, z = true_states[k, 0], true_states[k, 2], true_states[k, 4]
        z_true = cartesian_to_spherical(x, y, z)
        noise = np.array(
            [
                np.random.randn() * sigma_r,
                np.random.randn() * sigma_az,
                np.random.randn() * sigma_el,
            ]
        )
        spherical_measurements.append(z_true + noise)

    print(f"\nMeasurement noise:")
    print(f"  Range: {sigma_r} m")
    print(f"  Azimuth: {np.degrees(sigma_az):.2f} deg")
    print(f"  Elevation: {np.degrees(sigma_el):.2f} deg")

    # Convert measurements to Cartesian for simple Kalman filter
    cartesian_measurements = []
    for z_sph in spherical_measurements:
        r, az, el = z_sph
        cart = spherical_to_cartesian(r, az, el)
        cartesian_measurements.append(cart)

    # Measurement matrix for Cartesian measurements
    H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )

    # Approximate Cartesian measurement noise at mid-range
    avg_range = np.mean([z[0] for z in spherical_measurements])
    R_cart = np.diag(
        [
            sigma_r**2 + (avg_range * sigma_az) ** 2,
            sigma_r**2 + (avg_range * sigma_az) ** 2,
            sigma_r**2 + (avg_range * sigma_el) ** 2,
        ]
    )

    # Run Kalman filter
    x = np.array([50000, 0, 30000, 0, 10000, 0])  # Initial guess
    P = np.diag([10000, 500, 10000, 500, 5000, 100])

    estimates = []
    for k in range(n_steps):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update with Cartesian measurement
        z = cartesian_measurements[k]
        y = z - H @ x
        S = H @ P @ H.T + R_cart
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

        estimates.append(x.copy())

    estimates = np.array(estimates)

    # Compute 3D position errors
    pos_errors = np.sqrt(
        (estimates[:, 0] - true_states[:, 0]) ** 2
        + (estimates[:, 2] - true_states[:, 2]) ** 2
        + (estimates[:, 4] - true_states[:, 4]) ** 2
    )

    print(f"\n3D Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.1f} m")

    # Show range over time
    ranges = [np.sqrt(s[0] ** 2 + s[2] ** 2 + s[4] ** 2) for s in true_states]
    print(f"\nRange: {ranges[0]/1000:.1f} km -> {ranges[-1]/1000:.1f} km")

    # Plot
    if SHOW_PLOTS:
        sph_arr = np.array(spherical_measurements)
        ranges_km = np.array(ranges) / 1000

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "Radar Tracking (3D)",
                "Error vs Range",
                "Radar Measurements (Spherical)",
            ),
        )

        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=true_states[:, 0] / 1000,
                y=true_states[:, 2] / 1000,
                z=true_states[:, 4] / 1000,
                mode="lines",
                name="True",
                line=dict(color="blue", width=4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=estimates[:, 0] / 1000,
                y=estimates[:, 2] / 1000,
                z=estimates[:, 4] / 1000,
                mode="lines",
                name="Estimate",
                line=dict(color="red", width=3, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                name="Radar",
                marker=dict(color="green", size=8, symbol="diamond"),
            ),
            row=1,
            col=1,
        )

        # Position error vs range
        fig.add_trace(
            go.Scatter(
                x=ranges_km,
                y=pos_errors,
                mode="markers",
                name="Error",
                marker=dict(color="blue", size=6, opacity=0.6),
            ),
            row=1,
            col=2,
        )

        # Spherical measurements (Az vs El, colored by range)
        fig.add_trace(
            go.Scatter(
                x=np.degrees(sph_arr[:, 1]),
                y=np.degrees(sph_arr[:, 2]),
                mode="markers",
                name="Measurements",
                marker=dict(
                    color=sph_arr[:, 0] / 1000,
                    colorscale="Viridis",
                    size=8,
                    colorbar=dict(title="Range (km)", x=1.02),
                ),
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            title="Spherical (Radar) Measurements Tracking",
            height=500,
            width=1400,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Range (km)", row=1, col=2)
        fig.update_yaxes(title_text="3D Position Error (m)", row=1, col=2)
        fig.update_xaxes(title_text="Azimuth (deg)", row=1, col=3)
        fig.update_yaxes(title_text="Elevation (deg)", row=1, col=3)

        fig.write_html("tracking_3d_radar.html")
        print("\n  [Plot saved to tracking_3d_radar.html]")


def demo_multi_sensor_3d():
    """Demonstrate 3D tracking with multiple sensors."""
    print("\n" + "=" * 70)
    print("Multi-Sensor 3D Fusion Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate 3D trajectory
    n_steps = 60
    dt = 1.0

    F = np.array(
        [
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    q = 0.2
    Q_1d = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
    Q = np.zeros((6, 6))
    Q[0:2, 0:2] = Q_1d
    Q[2:4, 2:4] = Q_1d
    Q[4:6, 4:6] = Q_1d

    # True trajectory
    true_states = np.zeros((n_steps, 6))
    true_states[0] = [0, 10, 0, 5, 100, 2]

    for k in range(1, n_steps):
        process_noise = np.random.multivariate_normal(np.zeros(6), Q * 0.1)
        true_states[k] = F @ true_states[k - 1] + process_noise

    # Define sensors with different noise characteristics
    sensors = [
        {"name": "GPS", "noise_std": [3.0, 3.0, 5.0]},  # x, y, z noise in meters
        {"name": "Radar", "noise_std": [5.0, 5.0, 8.0]},
        {"name": "Lidar", "noise_std": [0.5, 0.5, 0.8]},  # Most accurate
    ]

    print("\nSensors:")
    for s in sensors:
        print(f"  {s['name']}: noise std = {s['noise_std']}")

    # Generate measurements from each sensor
    sensor_measurements = {s["name"]: [] for s in sensors}

    for k in range(n_steps):
        true_pos = true_states[k, [0, 2, 4]]  # x, y, z
        for sensor in sensors:
            noise = np.array(
                [
                    np.random.randn() * sensor["noise_std"][0],
                    np.random.randn() * sensor["noise_std"][1],
                    np.random.randn() * sensor["noise_std"][2],
                ]
            )
            sensor_measurements[sensor["name"]].append(true_pos + noise)

    H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )

    # Track using individual sensors and fused
    results = {}

    for sensor in sensors:
        R = np.diag([s**2 for s in sensor["noise_std"]])
        x = np.array([0, 0, 0, 0, 100, 0])
        P = np.diag([100, 50, 100, 50, 100, 50])

        estimates = []
        for k in range(n_steps):
            x = F @ x
            P = F @ P @ F.T + Q

            z = sensor_measurements[sensor["name"]][k]
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K @ H) @ P

            estimates.append(x.copy())

        results[sensor["name"]] = np.array(estimates)

    # Fused tracking (combine all sensor measurements)
    R_fused = np.zeros((3, 3))
    for sensor in sensors:
        R_inv = np.diag([1 / s**2 for s in sensor["noise_std"]])
        R_fused += R_inv
    R_fused = np.linalg.inv(R_fused)

    x = np.array([0, 0, 0, 0, 100, 0])
    P = np.diag([100, 50, 100, 50, 100, 50])

    fused_estimates = []
    for k in range(n_steps):
        x = F @ x
        P = F @ P @ F.T + Q

        # Fuse measurements using information form
        z_fused = np.zeros(3)
        for sensor in sensors:
            R_inv = np.diag([1 / s**2 for s in sensor["noise_std"]])
            z_fused += R_inv @ sensor_measurements[sensor["name"]][k]
        z_fused = R_fused @ z_fused

        y = z_fused - H @ x
        S = H @ P @ H.T + R_fused
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

        fused_estimates.append(x.copy())

    results["Fused"] = np.array(fused_estimates)

    # Compute RMSE for each
    print("\n3D Position RMSE:")
    print("-" * 40)
    for name, estimates in results.items():
        errors = np.sqrt(
            (estimates[:, 0] - true_states[:, 0]) ** 2
            + (estimates[:, 2] - true_states[:, 2]) ** 2
            + (estimates[:, 4] - true_states[:, 4]) ** 2
        )
        rmse = np.sqrt(np.mean(errors**2))
        print(f"  {name:10s}: {rmse:.3f} m")

    # Plot
    if SHOW_PLOTS:
        time = np.arange(n_steps)
        colors = {"GPS": "blue", "Radar": "orange", "Lidar": "green", "Fused": "red"}

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "Multi-Sensor 3D Tracking",
                "Position Error Comparison",
                "XZ Projection (Side View)",
            ),
        )

        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=true_states[:, 0],
                y=true_states[:, 2],
                z=true_states[:, 4],
                mode="lines",
                name="True",
                line=dict(color="black", width=4),
            ),
            row=1,
            col=1,
        )
        for name in ["GPS", "Fused"]:
            est = results[name]
            fig.add_trace(
                go.Scatter3d(
                    x=est[:, 0],
                    y=est[:, 2],
                    z=est[:, 4],
                    mode="lines",
                    name=name,
                    line=dict(
                        color=colors[name],
                        width=3,
                        dash="dash" if name != "Fused" else "solid",
                    ),
                ),
                row=1,
                col=1,
            )

        # Error comparison
        for name, estimates in results.items():
            errors = np.sqrt(
                (estimates[:, 0] - true_states[:, 0]) ** 2
                + (estimates[:, 2] - true_states[:, 2]) ** 2
                + (estimates[:, 4] - true_states[:, 4]) ** 2
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=errors,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[name], width=2),
                    opacity=0.8,
                ),
                row=1,
                col=2,
            )

        # XZ projection (side view)
        fig.add_trace(
            go.Scatter(
                x=true_states[:, 0],
                y=true_states[:, 4],
                mode="lines",
                name="True",
                line=dict(color="black", width=3),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=results["Fused"][:, 0],
                y=results["Fused"][:, 4],
                mode="lines",
                name="Fused",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            title="Multi-Sensor 3D Fusion",
            height=500,
            width=1400,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time step", row=1, col=2)
        fig.update_yaxes(title_text="3D Position Error (m)", row=1, col=2)
        fig.update_xaxes(title_text="X (m)", row=1, col=3)
        fig.update_yaxes(title_text="Z (m)", row=1, col=3)

        fig.write_html("tracking_3d_multisensor.html")
        print("\n  [Plot saved to tracking_3d_multisensor.html]")


def demo_3d_maneuvering_target():
    """Demonstrate tracking a maneuvering target in 3D."""
    print("\n" + "=" * 70)
    print("Maneuvering Target Demo")
    print("=" * 70)

    np.random.seed(42)

    n_steps = 120
    dt = 1.0

    # Generate maneuvering trajectory (includes turns and altitude changes)
    true_states = np.zeros((n_steps, 6))
    true_states[0] = [0, 50, 0, 0, 1000, 0]

    print("\nScenario: Maneuvering aircraft")
    print("  Phase 1 (t=0-40): Straight flight")
    print("  Phase 2 (t=40-80): Climbing turn")
    print("  Phase 3 (t=80-120): Descending turn")

    for k in range(1, n_steps):
        x, vx, y, vy, z, vz = true_states[k - 1]

        if k < 40:
            # Straight flight
            vx_new, vy_new, vz_new = 50, 0, 0
        elif k < 80:
            # Climbing turn (increase vy, increase vz)
            omega = 0.02  # Turn rate
            vx_new = 50 * np.cos(omega * (k - 40))
            vy_new = 50 * np.sin(omega * (k - 40))
            vz_new = 5  # Climbing
        else:
            # Descending turn
            omega = -0.03
            vx_new = 40 * np.cos(omega * (k - 80))
            vy_new = 40 * np.sin(omega * (k - 80)) + 30
            vz_new = -3  # Descending

        # Add process noise
        noise = np.random.randn(6) * np.array([1, 0.5, 1, 0.5, 1, 0.5])

        true_states[k] = [
            x + vx * dt + noise[0],
            vx_new + noise[1],
            y + vy * dt + noise[2],
            vy_new + noise[3],
            z + vz * dt + noise[4],
            vz_new + noise[5],
        ]

    # Generate noisy measurements
    R = np.diag([4.0, 4.0, 9.0])  # x, y, z measurement noise
    H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )

    measurements = []
    for k in range(n_steps):
        true_pos = true_states[k, [0, 2, 4]]
        noise = np.random.multivariate_normal(np.zeros(3), R)
        measurements.append(true_pos + noise)

    # Constant velocity model (will struggle with maneuvers)
    F_cv = np.array(
        [
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    # Low process noise (assumes constant velocity)
    q_low = 0.1
    Q_low = np.zeros((6, 6))
    for i in [0, 2, 4]:
        Q_low[i : i + 2, i : i + 2] = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q_low

    # High process noise (handles maneuvers better)
    q_high = 5.0
    Q_high = np.zeros((6, 6))
    for i in [0, 2, 4]:
        Q_high[i : i + 2, i : i + 2] = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q_high

    # Track with both process noise levels
    def run_filter(Q):
        x = np.array([0, 50, 0, 0, 1000, 0])
        P = np.diag([100, 50, 100, 50, 100, 50])
        estimates = []

        for k in range(n_steps):
            x = F_cv @ x
            P = F_cv @ P @ F_cv.T + Q

            z = measurements[k]
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K @ H) @ P

            estimates.append(x.copy())

        return np.array(estimates)

    est_low = run_filter(Q_low)
    est_high = run_filter(Q_high)

    # Compute errors
    def compute_errors(estimates):
        return np.sqrt(
            (estimates[:, 0] - true_states[:, 0]) ** 2
            + (estimates[:, 2] - true_states[:, 2]) ** 2
            + (estimates[:, 4] - true_states[:, 4]) ** 2
        )

    err_low = compute_errors(est_low)
    err_high = compute_errors(est_high)

    print("\n3D Position RMSE:")
    print(f"  Low process noise (q={q_low}):  {np.sqrt(np.mean(err_low**2)):.2f} m")
    print(f"  High process noise (q={q_high}): {np.sqrt(np.mean(err_high**2)):.2f} m")

    # Error during maneuver phases
    print("\nRMSE by phase:")
    phases = [
        (0, 40, "Straight"),
        (40, 80, "Climbing turn"),
        (80, 120, "Descending turn"),
    ]
    for start, end, name in phases:
        rmse_low = np.sqrt(np.mean(err_low[start:end] ** 2))
        rmse_high = np.sqrt(np.mean(err_high[start:end] ** 2))
        print(f"  {name:18s}: Low Q = {rmse_low:.2f} m, High Q = {rmse_high:.2f} m")

    # Plot
    if SHOW_PLOTS:
        time = np.arange(n_steps)

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "Maneuvering Target Tracking",
                "Error Comparison",
                "Altitude Profile",
            ),
        )

        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=true_states[:, 0],
                y=true_states[:, 2],
                z=true_states[:, 4],
                mode="lines",
                name="True",
                line=dict(color="black", width=4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=est_low[:, 0],
                y=est_low[:, 2],
                z=est_low[:, 4],
                mode="lines",
                name=f"Low Q ({q_low})",
                line=dict(color="blue", width=2, dash="dash"),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=est_high[:, 0],
                y=est_high[:, 2],
                z=est_high[:, 4],
                mode="lines",
                name=f"High Q ({q_high})",
                line=dict(color="red", width=2, dash="dash"),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Error over time
        fig.add_trace(
            go.Scatter(
                x=time,
                y=err_low,
                mode="lines",
                name=f"Low Q ({q_low})",
                line=dict(color="blue", width=2),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=err_high,
                mode="lines",
                name=f"High Q ({q_high})",
                line=dict(color="red", width=2),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )
        # Add phase markers
        for boundary in [40, 80]:
            fig.add_vline(x=boundary, line_dash="dash", line_color="gray", row=1, col=2)

        # Altitude profile
        fig.add_trace(
            go.Scatter(
                x=time,
                y=true_states[:, 4],
                mode="lines",
                name="True",
                line=dict(color="black", width=3),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=est_low[:, 4],
                mode="lines",
                name="Low Q",
                line=dict(color="blue", width=2, dash="dash"),
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=est_high[:, 4],
                mode="lines",
                name="High Q",
                line=dict(color="red", width=2, dash="dash"),
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        for boundary in [40, 80]:
            fig.add_vline(x=boundary, line_dash="dash", line_color="gray", row=1, col=3)

        fig.update_layout(
            title="Maneuvering Target Tracking",
            height=500,
            width=1400,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time step", row=1, col=2)
        fig.update_yaxes(title_text="3D Position Error (m)", row=1, col=2)
        fig.update_xaxes(title_text="Time step", row=1, col=3)
        fig.update_yaxes(title_text="Altitude Z (m)", row=1, col=3)

        fig.write_html("tracking_3d_maneuver.html")
        print("\n  [Plot saved to tracking_3d_maneuver.html]")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL 3D Tracking Example")
    print("#" * 70)

    # Basic 3D tracking
    demo_3d_kalman_filter()
    demo_3d_rts_smoother()

    # Advanced scenarios
    demo_spherical_measurements()
    demo_multi_sensor_3d()
    demo_3d_maneuvering_target()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: tracking_3d_kalman.html, tracking_3d_smoother.html,")
        print("             tracking_3d_radar.html, tracking_3d_multisensor.html,")
        print("             tracking_3d_maneuver.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
