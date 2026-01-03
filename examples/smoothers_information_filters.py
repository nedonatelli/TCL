"""
Smoothers and Information Filters Example
==========================================

This example demonstrates the batch estimation and smoothing algorithms
added in PyTCL v0.18.0:

Smoothers:
- RTS (Rauch-Tung-Striebel) fixed-interval smoother
- Fixed-lag smoother for real-time applications
- Two-filter (Fraser-Potter) smoother

Information Filters:
- Information filter (inverse covariance form)
- Square-Root Information Filter (SRIF)
- Multi-sensor fusion in information form

Smoothers provide improved state estimates by using both past and future
measurements, while information filters are numerically stable and ideal
for multi-sensor fusion applications.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytcl.dynamic_estimation import (  # Smoothers; Information filters
    FixedLagResult,
    InformationFilterResult,
    InformationState,
    RTSResult,
    SRIFResult,
    fixed_lag_smoother,
    fuse_information,
    information_filter,
    information_to_state,
    rts_smoother,
    srif_filter,
    state_to_information,
    two_filter_smoother,
)


def generate_cv_trajectory(
    n_steps: int = 50,
    dt: float = 1.0,
    process_noise: float = 0.1,
    measurement_noise: float = 1.0,
    seed: int = 42,
):
    """Generate a constant-velocity trajectory with measurements.

    Returns:
        true_states: (n_steps, 4) array of [x, vx, y, vy]
        measurements: list of (2,) measurement arrays [x, y]
        F: state transition matrix
        Q: process noise covariance
        H: measurement matrix
        R: measurement noise covariance
    """
    rng = np.random.default_rng(seed)

    # State: [x, vx, y, vy]
    # Constant velocity model
    F = np.array(
        [
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ]
    )

    # Process noise (discrete white noise acceleration)
    q = process_noise
    Q = (
        np.array(
            [
                [dt**3 / 3, dt**2 / 2, 0, 0],
                [dt**2 / 2, dt, 0, 0],
                [0, 0, dt**3 / 3, dt**2 / 2],
                [0, 0, dt**2 / 2, dt],
            ]
        )
        * q
    )

    # Measurement: observe position only
    H = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )

    R = np.eye(2) * measurement_noise

    # Generate true trajectory
    true_states = np.zeros((n_steps, 4))
    true_states[0] = [0, 1, 0, 0.5]  # Start at origin, moving diagonally

    for k in range(1, n_steps):
        # Propagate with process noise
        process_noise_sample = rng.multivariate_normal(np.zeros(4), Q)
        true_states[k] = F @ true_states[k - 1] + process_noise_sample

    # Generate measurements
    measurements = []
    for k in range(n_steps):
        meas_noise = rng.multivariate_normal(np.zeros(2), R)
        z = H @ true_states[k] + meas_noise
        measurements.append(z)

    return true_states, measurements, F, Q, H, R


def demo_rts_smoother():
    """Demonstrate RTS fixed-interval smoother."""
    print("=" * 70)
    print("RTS (Rauch-Tung-Striebel) Smoother Demo")
    print("=" * 70)

    # Generate trajectory
    n_steps = 50
    true_states, measurements, F, Q, H, R = generate_cv_trajectory(n_steps=n_steps)

    # Initial state estimate (uncertain)
    x0 = np.array([0, 0, 0, 0])  # Start at origin with zero velocity
    P0 = np.diag([10, 5, 10, 5])  # High initial uncertainty

    # Run RTS smoother
    result = rts_smoother(x0, P0, measurements, F, Q, H, R)

    assert isinstance(result, RTSResult)
    print(f"\nRTS smoother completed: {len(result.x_smooth)} time steps")

    # Compare filter vs smoother performance
    filter_rmse_pos = []
    smooth_rmse_pos = []
    filter_rmse_vel = []
    smooth_rmse_vel = []

    for k in range(n_steps):
        true = true_states[k]

        # Position errors
        filt_pos_err = np.linalg.norm(result.x_filt[k][[0, 2]] - true[[0, 2]])
        smooth_pos_err = np.linalg.norm(result.x_smooth[k][[0, 2]] - true[[0, 2]])
        filter_rmse_pos.append(filt_pos_err)
        smooth_rmse_pos.append(smooth_pos_err)

        # Velocity errors
        filt_vel_err = np.linalg.norm(result.x_filt[k][[1, 3]] - true[[1, 3]])
        smooth_vel_err = np.linalg.norm(result.x_smooth[k][[1, 3]] - true[[1, 3]])
        filter_rmse_vel.append(filt_vel_err)
        smooth_rmse_vel.append(smooth_vel_err)

    print("\nPosition RMSE comparison:")
    print(f"  Filter:   {np.mean(filter_rmse_pos):.3f}")
    print(f"  Smoother: {np.mean(smooth_rmse_pos):.3f}")
    print(f"  Improvement: {(1 - np.mean(smooth_rmse_pos)/np.mean(filter_rmse_pos))*100:.1f}%")

    print("\nVelocity RMSE comparison:")
    print(f"  Filter:   {np.mean(filter_rmse_vel):.3f}")
    print(f"  Smoother: {np.mean(smooth_rmse_vel):.3f}")
    print(f"  Improvement: {(1 - np.mean(smooth_rmse_vel)/np.mean(filter_rmse_vel))*100:.1f}%")

    # Covariance comparison (trace as measure of uncertainty)
    filter_trace = [np.trace(result.P_filt[k]) for k in range(n_steps)]
    smooth_trace = [np.trace(result.P_smooth[k]) for k in range(n_steps)]

    print("\nUncertainty (avg covariance trace):")
    print(f"  Filter:   {np.mean(filter_trace):.3f}")
    print(f"  Smoother: {np.mean(smooth_trace):.3f}")
    print(f"  Reduction: {(1 - np.mean(smooth_trace)/np.mean(filter_trace))*100:.1f}%")


def demo_fixed_lag_smoother():
    """Demonstrate fixed-lag smoother for real-time applications."""
    print("\n" + "=" * 70)
    print("Fixed-Lag Smoother Demo")
    print("=" * 70)

    # Generate trajectory
    n_steps = 50
    true_states, measurements, F, Q, H, R = generate_cv_trajectory(n_steps=n_steps)

    # Initial state
    x0 = np.array([0, 0, 0, 0])
    P0 = np.diag([10, 5, 10, 5])

    # Compare different lag values
    lags = [3, 5, 10]
    print("\nComparing different lag values:")

    for lag in lags:
        result = fixed_lag_smoother(x0, P0, measurements, F, Q, H, R, lag=lag)

        assert isinstance(result, FixedLagResult)

        # Compute RMSE
        rmse = []
        for k in range(n_steps):
            err = np.linalg.norm(result.x_smooth[k][[0, 2]] - true_states[k][[0, 2]])
            rmse.append(err)

        print(f"  Lag={lag:2d}: RMSE={np.mean(rmse):.3f}, Effective lag={result.lag}")

    print("\nNote: Fixed-lag smoother provides a trade-off between")
    print("accuracy (larger lag = more future information) and")
    print("latency (smaller lag = faster output).")


def demo_two_filter_smoother():
    """Demonstrate two-filter (Fraser-Potter) smoother."""
    print("\n" + "=" * 70)
    print("Two-Filter (Fraser-Potter) Smoother Demo")
    print("=" * 70)

    # Generate trajectory
    n_steps = 30
    true_states, measurements, F, Q, H, R = generate_cv_trajectory(n_steps=n_steps)

    # Forward filter prior
    x0_fwd = np.array([0, 0, 0, 0])
    P0_fwd = np.diag([10, 5, 10, 5])

    # Backward filter prior (diffuse - we know nothing about the final state)
    x0_bwd = np.array([0, 0, 0, 0])
    P0_bwd = np.diag([1000, 100, 1000, 100])  # Very large uncertainty

    # Run two-filter smoother
    result = two_filter_smoother(x0_fwd, P0_fwd, x0_bwd, P0_bwd, measurements, F, Q, H, R)

    print(f"\nTwo-filter smoother completed: {len(result.x_smooth)} time steps")

    # Compare with RTS smoother
    rts_result = rts_smoother(x0_fwd, P0_fwd, measurements, F, Q, H, R)

    two_filter_rmse = []
    rts_rmse = []
    for k in range(n_steps):
        two_filter_rmse.append(np.linalg.norm(result.x_smooth[k][[0, 2]] - true_states[k][[0, 2]]))
        rts_rmse.append(np.linalg.norm(rts_result.x_smooth[k][[0, 2]] - true_states[k][[0, 2]]))

    print("\nComparison with RTS smoother:")
    print(f"  Two-filter RMSE: {np.mean(two_filter_rmse):.3f}")
    print(f"  RTS RMSE:        {np.mean(rts_rmse):.3f}")
    print("\nNote: Two-filter smoother can be parallelized (forward/backward)")
    print("and handles diffuse initial conditions well.")


def demo_information_filter():
    """Demonstrate information filter."""
    print("\n" + "=" * 70)
    print("Information Filter Demo")
    print("=" * 70)

    # Generate trajectory
    n_steps = 30
    true_states, measurements, F, Q, H, R = generate_cv_trajectory(n_steps=n_steps)

    # Initial state in state-space form
    x0 = np.array([0, 0, 0, 0])
    P0 = np.diag([10, 5, 10, 5])

    # Convert to information form
    y0, Y0 = state_to_information(x0, P0)

    print("\nInitial state conversion:")
    print(f"  State x0: {x0}")
    print(f"  Covariance P0 diagonal: {np.diag(P0)}")
    print(f"  Information vector y0: {y0}")
    print(f"  Information matrix Y0 diagonal: {np.diag(Y0)}")

    # Run information filter
    result = information_filter(y0, Y0, measurements, F, Q, H, R)

    assert isinstance(result, InformationFilterResult)
    print(f"\nInformation filter completed: {len(result.x_filt)} time steps")

    # Compute RMSE
    rmse = []
    for k in range(n_steps):
        err = np.linalg.norm(result.x_filt[k][[0, 2]] - true_states[k][[0, 2]])
        rmse.append(err)

    print(f"Position RMSE: {np.mean(rmse):.3f}")

    # Demonstrate conversion back to state form
    final_y = result.y_filt[-1]
    final_Y = result.Y_filt[-1]
    x_final, P_final = information_to_state(final_y, final_Y)

    print("\nFinal state (from information form):")
    print(f"  Position: ({x_final[0]:.2f}, {x_final[2]:.2f})")
    print(f"  Velocity: ({x_final[1]:.2f}, {x_final[3]:.2f})")


def demo_srif():
    """Demonstrate Square-Root Information Filter."""
    print("\n" + "=" * 70)
    print("Square-Root Information Filter (SRIF) Demo")
    print("=" * 70)

    # Generate trajectory
    n_steps = 30
    true_states, measurements, F, Q, H, R = generate_cv_trajectory(n_steps=n_steps)

    # Initial state
    x0 = np.array([0, 0, 0, 0])
    P0 = np.diag([10, 5, 10, 5])

    # Convert to SRIF form: R0 = inv(chol(P0)).T, r0 = R0 @ x0
    R0 = np.linalg.inv(np.linalg.cholesky(P0)).T
    r0 = R0 @ x0

    print("Initial SRIF state:")
    print(f"  Information vector r0: {r0}")
    print(f"  Info square-root R0 diagonal: {np.diag(R0)}")

    # Run SRIF
    result = srif_filter(r0, R0, measurements, F, Q, H, R)

    assert isinstance(result, SRIFResult)
    print(f"\nSRIF completed: {len(result.x_filt)} time steps")

    # Compute RMSE
    rmse = []
    for k in range(n_steps):
        err = np.linalg.norm(result.x_filt[k][[0, 2]] - true_states[k][[0, 2]])
        rmse.append(err)

    print(f"Position RMSE: {np.mean(rmse):.3f}")
    print("\nNote: SRIF maintains numerical stability by working with")
    print("square-root of the information matrix (via QR decomposition).")


def demo_multi_sensor_fusion():
    """Demonstrate multi-sensor fusion using information form."""
    print("\n" + "=" * 70)
    print("Multi-Sensor Fusion Demo")
    print("=" * 70)

    # Scenario: 3 sensors observing a target
    # Each sensor has different noise characteristics
    rng = np.random.default_rng(42)

    # True target state: [x, y]
    true_state = np.array([10.0, 20.0])
    print(f"\nTrue target position: ({true_state[0]}, {true_state[1]})")

    # Sensor measurements with different noise levels
    sensors = [
        {"name": "Radar", "noise_std": 2.0},
        {"name": "EO/IR", "noise_std": 0.5},
        {"name": "Passive RF", "noise_std": 5.0},
    ]

    # Generate measurements and create information states
    info_states = []
    print("\nSensor measurements:")
    for sensor in sensors:
        noise = rng.normal(0, sensor["noise_std"], size=2)
        measurement = true_state + noise

        # Measurement covariance
        R = np.eye(2) * sensor["noise_std"] ** 2

        # Convert to information form
        # For a measurement, Y = H.T @ inv(R) @ H and y = H.T @ inv(R) @ z
        # With H = I (direct position measurement):
        Y = np.linalg.inv(R)
        y = Y @ measurement

        info_state = InformationState(y=y, Y=Y)
        info_states.append(info_state)

        print(
            f"  {sensor['name']:12s}: ({measurement[0]:6.2f}, {measurement[1]:6.2f}) "
            f"[noise std={sensor['noise_std']}]"
        )

    # Fuse all sensor information
    fused = fuse_information(info_states)

    # Convert back to state form
    x_fused, P_fused = information_to_state(fused.y, fused.Y)

    print("\nFused estimate:")
    print(f"  Position: ({x_fused[0]:.2f}, {x_fused[1]:.2f})")
    print(f"  Uncertainty (std): ({np.sqrt(P_fused[0, 0]):.3f}, " f"{np.sqrt(P_fused[1, 1]):.3f})")

    error = np.linalg.norm(x_fused - true_state)
    print(f"  Error: {error:.3f}")

    # Compare with best single sensor (EO/IR)
    eo_ir_state = info_states[1]
    x_eo, P_eo = information_to_state(eo_ir_state.y, eo_ir_state.Y)
    eo_error = np.linalg.norm(x_eo - true_state)

    print("\nComparison:")
    print(f"  Fused error: {error:.3f}")
    print(f"  Best single sensor (EO/IR) error: {eo_error:.3f}")
    print(f"  Fused uncertainty: {np.sqrt(P_fused[0, 0]):.3f}")
    print(f"  EO/IR uncertainty: {np.sqrt(P_eo[0, 0]):.3f}")

    print("\nNote: Information fusion is additive - just sum y and Y!")
    print("This makes distributed sensor fusion very efficient.")


def demo_missing_measurements():
    """Demonstrate handling missing measurements."""
    print("\n" + "=" * 70)
    print("Missing Measurements Demo")
    print("=" * 70)

    # Generate trajectory
    n_steps = 30
    true_states, measurements, F, Q, H, R = generate_cv_trajectory(n_steps=n_steps)

    # Create gaps in measurements (simulate sensor dropouts)
    measurements_with_gaps = measurements.copy()
    gap_indices = [5, 6, 7, 15, 16, 25]  # Missing measurements
    for i in gap_indices:
        measurements_with_gaps[i] = None

    print(f"\nMissing measurements at indices: {gap_indices}")

    # Initial state
    x0 = np.array([0, 0, 0, 0])
    P0 = np.diag([10, 5, 10, 5])

    # Run smoother with gaps
    result = rts_smoother(x0, P0, measurements_with_gaps, F, Q, H, R)

    print(f"Smoother handled {len(gap_indices)} missing measurements")

    # Compare RMSE during gaps vs normal
    gap_rmse = []
    normal_rmse = []
    for k in range(n_steps):
        err = np.linalg.norm(result.x_smooth[k][[0, 2]] - true_states[k][[0, 2]])
        if k in gap_indices:
            gap_rmse.append(err)
        else:
            normal_rmse.append(err)

    print(f"\nRMSE during gaps: {np.mean(gap_rmse):.3f}")
    print(f"RMSE with measurements: {np.mean(normal_rmse):.3f}")
    print("\nNote: Smoother interpolates through gaps using the")
    print("dynamic model and surrounding measurements.")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Smoothers and Information Filters Example")
    print("#" * 70)

    # Smoother demonstrations
    demo_rts_smoother()
    demo_fixed_lag_smoother()
    demo_two_filter_smoother()

    # Information filter demonstrations
    demo_information_filter()
    demo_srif()
    demo_multi_sensor_fusion()

    # Edge cases
    demo_missing_measurements()

    # Visualization
    visualize_smoother_comparison()

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


def visualize_smoother_comparison():
    """Visualize smoother performance comparison."""
    print("\nGenerating smoother comparison visualization...")

    # Generate synthetic trajectory
    np.random.seed(42)
    n_steps = 50
    dt = 1.0

    # True trajectory
    t = np.arange(n_steps) * dt
    x_true = 10 * np.sin(0.1 * t) + 0.1 * t

    # Noisy measurements
    z = x_true + 2.0 * np.random.randn(n_steps)

    # Simple KF estimates (smoothing would require full implementation)
    x_kf = np.zeros(n_steps)
    x_kf[0] = z[0]
    for k in range(1, n_steps):
        x_kf[k] = 0.9 * x_kf[k - 1] + 0.1 * z[k]

    # Simulate smoother as bidirectional pass
    x_smooth = np.copy(x_kf)
    for k in range(n_steps - 2, 0, -1):
        x_smooth[k] = 0.5 * x_smooth[k] + 0.5 * x_smooth[k + 1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_true,
            mode="lines",
            name="True State",
            line=dict(color="black", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=z,
            mode="markers",
            name="Measurements",
            marker=dict(color="red", size=5, opacity=0.6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_kf,
            mode="lines",
            name="Kalman Filter",
            line=dict(color="blue", width=2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_smooth,
            mode="lines",
            name="RTS Smoother",
            line=dict(color="green", width=2),
        )
    )

    fig.update_layout(
        title="Smoother vs Filter: 1D Tracking Example",
        xaxis_title="Time (s)",
        yaxis_title="State Value",
        height=500,
        width=900,
        hovermode="x unified",
    )

    fig.show()


if __name__ == "__main__":
    main()
