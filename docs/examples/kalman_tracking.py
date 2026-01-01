#!/usr/bin/env python3
"""
Kalman Filter Tracking Example
==============================

Track a 2D object using a linear Kalman filter with position measurements.
"""

import numpy as np

from pytcl.dynamic_estimation import kf_predict, kf_update


def main():
    # System parameters
    dt = 0.1  # Time step (seconds)
    n_steps = 100

    # State transition matrix (constant velocity model)
    # State: [x, vx, y, vy]
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

    # Measurement matrix (position only)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    # Measurement noise covariance
    R = np.eye(2) * 0.5

    # Initial state and covariance
    x = np.array([0.0, 1.0, 0.0, 0.5])
    P = np.eye(4) * 10.0

    # Generate ground truth
    np.random.seed(42)
    x_true = np.array([0.0, 1.0, 0.0, 0.5])
    true_states = []
    measurements = []

    for _ in range(n_steps):
        true_states.append(x_true.copy())
        z = H @ x_true + np.random.multivariate_normal(np.zeros(2), R)
        measurements.append(z)
        x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), Q)

    # Run Kalman filter
    estimates = []
    for z in measurements:
        # Predict
        pred = kf_predict(x, P, F, Q)
        x, P = pred.x, pred.P

        # Update
        upd = kf_update(x, P, z, H, R)
        x, P = upd.x, upd.P

        estimates.append(x.copy())

    # Compute errors
    true_states = np.array(true_states)
    estimates = np.array(estimates)

    pos_errors = np.sqrt(
        (true_states[:, 0] - estimates[:, 0]) ** 2 + (true_states[:, 2] - estimates[:, 2]) ** 2
    )

    print("Kalman Filter Tracking Results")
    print("=" * 40)
    print(f"Mean position error: {np.mean(pos_errors):.3f} m")
    print(f"Max position error:  {np.max(pos_errors):.3f} m")
    print(f"Final position error: {pos_errors[-1]:.3f} m")
    print(f"Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.3f} m")


if __name__ == "__main__":
    main()
