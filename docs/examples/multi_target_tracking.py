#!/usr/bin/env python3
"""
Multi-Target Tracking Example
=============================

Track multiple crossing targets with the GNN-based multi-target tracker.
"""

import numpy as np

from pytcl.trackers import MultiTargetTracker
from pytcl.performance_evaluation import ospa_distance


def main():
    np.random.seed(42)

    # System parameters
    dt = 0.1
    n_steps = 100

    # State transition matrix (constant velocity)
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    Q = np.eye(4) * 0.01
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(2) * 1.0

    # Create tracker
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        gate_threshold=9.21,
        min_hits=3,
        max_misses=5,
        filter_type="kf",
    )
    tracker.set_dynamics(F, Q)
    tracker.set_measurement_model(H, R)

    # Three crossing targets
    targets = [
        np.array([0.0, 1.0, 50.0, 0.0]),  # Moving right
        np.array([100.0, -1.0, 50.0, 0.0]),  # Moving left
        np.array([50.0, 0.0, 0.0, 1.0]),  # Moving up
    ]

    detection_prob = 0.9
    clutter_rate = 0.5  # Expected false alarms per scan

    ospa_values = []
    n_confirmed = []

    print("Multi-Target Tracking Simulation")
    print("=" * 50)
    print(f"Targets: 3 crossing targets")
    print(f"Detection probability: {detection_prob}")
    print(f"Clutter rate: {clutter_rate}")
    print()

    for t in range(n_steps):
        # Propagate true states
        truth_positions = []
        measurements = []

        for i, x in enumerate(targets):
            targets[i] = F @ x
            truth_positions.append([targets[i][0], targets[i][2]])

            # Detect with probability
            if np.random.rand() < detection_prob:
                z = H @ targets[i] + np.random.multivariate_normal(np.zeros(2), R)
                measurements.append(z)

        # Add clutter
        n_clutter = np.random.poisson(clutter_rate)
        for _ in range(n_clutter):
            measurements.append(np.random.rand(2) * 100)

        # Update tracker
        if measurements:
            tracks = tracker.update(np.array(measurements))
        else:
            tracks = tracker.update(np.empty((0, 2)))

        n_confirmed.append(len(tracks))

        # Compute OSPA
        if tracks:
            estimates = np.array([[tr.state[0], tr.state[2]] for tr in tracks])
        else:
            estimates = np.empty((0, 2))

        ospa = ospa_distance(np.array(truth_positions), estimates, p=2, c=50.0)
        ospa_values.append(ospa.distance)

    # Summary
    print("Results")
    print("-" * 50)
    print(f"Mean OSPA distance: {np.mean(ospa_values):.2f}")
    print(f"Mean confirmed tracks: {np.mean(n_confirmed):.2f}")
    print(f"Final confirmed tracks: {n_confirmed[-1]}")

    # Convergence time (time to confirm all 3 tracks)
    for t, n in enumerate(n_confirmed):
        if n == 3:
            print(f"All targets confirmed at step: {t}")
            break


if __name__ == "__main__":
    main()
