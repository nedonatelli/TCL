#!/usr/bin/env python3
"""
Unscented Kalman Filter Example
================================

Track an object using range-bearing measurements with the UKF.
"""

import numpy as np

from pytcl.dynamic_estimation import ukf_predict, ukf_update


def main():
    # Parameters
    dt = 0.1
    n_steps = 100
    np.random.seed(42)

    # State transition (constant velocity)
    def f(x):
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        return F @ x

    # Nonlinear measurement (range, bearing)
    def h(x):
        r = np.sqrt(x[0] ** 2 + x[2] ** 2)
        theta = np.arctan2(x[2], x[0])
        return np.array([r, theta])

    # Noise covariances
    Q = np.diag([0.01, 0.1, 0.01, 0.1])
    R = np.diag([1.0, 0.02])  # 1m range, ~1 deg bearing

    # Initial state
    x = np.array([100.0, -5.0, 50.0, 2.0])
    P = np.diag([10.0, 1.0, 10.0, 1.0])

    # Generate truth and measurements
    x_true = np.array([100.0, -5.0, 50.0, 2.0])
    true_states = []
    measurements = []

    for _ in range(n_steps):
        true_states.append(x_true.copy())
        z_true = h(x_true)
        z = z_true + np.random.multivariate_normal(np.zeros(2), R)
        measurements.append(z)
        x_true = f(x_true) + np.random.multivariate_normal(np.zeros(4), Q)

    # Run UKF
    estimates = []
    for z in measurements:
        pred = ukf_predict(x, P, f, Q)
        upd = ukf_update(pred.x, pred.P, z, h, R)
        x, P = upd.x, upd.P
        estimates.append(x.copy())

    # Compute errors
    true_states = np.array(true_states)
    estimates = np.array(estimates)

    pos_errors = np.sqrt(
        (true_states[:, 0] - estimates[:, 0]) ** 2
        + (true_states[:, 2] - estimates[:, 2]) ** 2
    )

    print("UKF Range-Bearing Tracking Results")
    print("=" * 40)
    print(f"Mean position error: {np.mean(pos_errors):.3f} m")
    print(f"Max position error:  {np.max(pos_errors):.3f} m")
    print(f"Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.3f} m")


if __name__ == "__main__":
    main()
