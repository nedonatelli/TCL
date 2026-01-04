"""
Demonstration of dynamic models and state transition matrices.

This example shows:
- Continuous and discrete-time system models
- State transition matrices (Phi matrices)
- Process noise covariance (Q matrices)
- Kalman filter compatibility
"""

import numpy as np
import plotly.graph_objects as go

from pytcl.dynamic_models.continuous_time import (
    diffusion_constant_velocity,
)
from pytcl.dynamic_models.discrete_time import f_constant_velocity

SHOW_PLOTS = True


def demo_state_transition_matrix() -> None:
    """Demonstrate state transition matrix properties."""
    print("\n" + "=" * 60)
    print("State Transition Matrices")
    print("=" * 60)

    dt = 0.1  # Time step

    # Get state transition matrix for constant velocity model
    # Returns 6x6 matrix for 3D position/velocity
    phi = f_constant_velocity(dt)

    print(f"\nTime step: {dt}s")
    print(f"State transition matrix (Phi) shape: {phi.shape}")
    print("Matrix (first 3x3 block):")
    print(phi[:3, :3])

    # The matrix has a block structure
    # [I  dt*I]
    # [0   I  ]
    # where I is 3x3 for position and velocity in 3D


def demo_process_noise() -> None:
    """Demonstrate process noise matrices."""
    print("\n" + "=" * 60)
    print("Process Noise Covariance Matrices")
    print("=" * 60)

    dt = 0.1
    sigma_v = 1.0  # Process noise standard deviation

    # Get process noise covariance matrix
    q_matrix = diffusion_constant_velocity(dt, sigma_v)

    print(f"\nTime step: {dt}s")
    print(f"Process noise std: {sigma_v} m/s")
    print(f"Process noise matrix (Q) shape: {q_matrix.shape}")

    # Properties
    print(f"\nProcess noise norm: {np.linalg.norm(q_matrix):.6f}")


def demo_drift_matrix() -> None:
    """Demonstrate drift (mean) functions."""
    print("\n" + "=" * 60)
    print("Drift Functions")
    print("=" * 60)

    # Drift function takes a state vector and returns rate of change
    # For constant velocity, position changes at velocity rate
    state = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])  # 3D pos/vel

    from pytcl.dynamic_models.continuous_time.dynamics import (
        drift_constant_velocity as drift_cv,
    )

    drift_result = drift_cv(state)

    print(f"\nExample state (3D position + velocity):")
    print(f"  Position: {state[0::2]}")
    print(f"  Velocity: {state[1::2]}")
    print(f"\nDrift (rate of change) result:")
    print(f"  {drift_result}")

    # Expected: [vel_1, 0, vel_2, 0, vel_3, 0]
    # showing that position changes at velocity rate and velocity doesn't change


def demo_continuous_to_discrete_conversion() -> None:
    """Demonstrate continuous to discrete time conversion principle."""
    print("\n" + "=" * 60)
    print("Continuous to Discrete Conversion")
    print("=" * 60)

    # For a constant velocity model in 1D:
    # Continuous: dx/dt = v, dv/dt = 0
    # Or in matrix form: [dx/dt; dv/dt] = [0 1; 0 0] * [x; v]

    # Discrete approximation: state[k+1] = Phi * state[k]
    # where Phi = exp(F * dt) ≈ I + F*dt for small dt

    F = np.array([[0.0, 1.0], [0.0, 0.0]])  # Continuous F matrix
    dts = [0.05, 0.1, 0.2, 0.5]

    print("\nContinuous F matrix (1D constant velocity):")
    print(F)

    print("\nDiscrete Phi matrices (Phi ≈ I + F*dt):")
    print("Time Step | Phi[0,1] Value")
    print("-" * 30)

    phi_values = []
    for dt_val in dts:
        # For constant velocity: Phi = [[1, dt], [0, 1]]
        phi_approx = np.eye(2) + F * dt_val
        phi_values.append(phi_approx[0, 1])
        print(f"{dt_val:>8} | {phi_approx[0, 1]:>14.6f}")

    # Plot
    if SHOW_PLOTS:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dts,
                y=phi_values,
                mode="lines+markers",
                name="Phi[0,1]",
                line=dict(color="purple", width=2),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Discrete State Transition Element vs Time Step",
            xaxis_title="Time Step (s)",
            yaxis_title="Phi[0,1] Value",
            height=400,
        )

        if SHOW_PLOTS:
            fig.show()


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Dynamic Models Demonstration")
    print("=" * 60)

    demo_state_transition_matrix()
    demo_process_noise()
    demo_drift_matrix()
    demo_continuous_to_discrete_conversion()

    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
