"""
Assignment Algorithms Example
=============================

This example demonstrates the assignment algorithms in PyTCL:

2D Assignment:
- Hungarian algorithm (optimal)
- Auction algorithm
- Linear sum assignment wrapper

K-Best 2D Assignment (v0.17.0):
- Murty's algorithm for finding k-best assignments
- Ranked assignment enumeration with cost thresholds

3D Assignment (v0.17.0):
- Lagrangian relaxation
- Auction-based 3D assignment
- Greedy assignment
- 2D decomposition method

Data Association:
- Global Nearest Neighbor (GNN)
- Gating (ellipsoidal and rectangular)
- JPDA (Joint Probabilistic Data Association)

These algorithms are fundamental for multi-target tracking, where
measurements must be assigned to tracks optimally.
"""

import numpy as np

from pytcl.assignment_algorithms import (  # 2D Assignment; K-Best 2D Assignment; 3D Assignment; Gating; Data Association; JPDA
    assign2d,
    assign3d,
    assign3d_auction,
    assign3d_lagrangian,
    auction,
    chi2_gate_threshold,
    compute_association_cost,
    decompose_to_2d,
    ellipsoidal_gate,
    gated_gnn_association,
    gnn_association,
    greedy_3d,
    hungarian,
    jpda,
    kbest_assign2d,
    mahalanobis_distance,
    murty,
    ranked_assignments,
    rectangular_gate,
)


def demo_2d_assignment():
    """Demonstrate basic 2D assignment algorithms."""
    print("=" * 70)
    print("2D Assignment Algorithms Demo")
    print("=" * 70)

    # Create a cost matrix: 4 tracks, 5 measurements
    # Lower cost = better match
    np.random.seed(42)
    cost = np.array(
        [
            [10, 5, 13, 4, 8],  # Track 0: best match is measurement 3
            [3, 15, 8, 12, 6],  # Track 1: best match is measurement 0
            [12, 7, 9, 5, 11],  # Track 2: best match is measurement 3 or 1
            [8, 6, 4, 10, 3],  # Track 3: best match is measurement 4
        ],
        dtype=float,
    )

    print("\nCost matrix (4 tracks x 5 measurements):")
    print(cost)

    # Hungarian algorithm (optimal)
    row_h, col_h, cost_h = hungarian(cost)
    print("\n--- Hungarian Algorithm (Optimal) ---")
    print(f"Assignments: {list(zip(row_h, col_h))}")
    print(f"Total cost: {cost_h}")

    # Auction algorithm
    row_a, col_a, cost_a = auction(cost)
    print("\n--- Auction Algorithm ---")
    print(f"Assignments: {list(zip(row_a, col_a))}")
    print(f"Total cost: {cost_a}")

    # Using assign2d unified interface
    result = assign2d(cost)
    print("\n--- assign2d() Interface ---")
    print(f"Row indices: {result.row_indices}")
    print(f"Col indices: {result.col_indices}")
    print(f"Total cost: {result.cost}")

    # Rectangular (non-square) assignment
    print("\n--- Rectangular Assignment ---")
    rect_cost = np.random.rand(3, 6) * 10  # 3 tracks, 6 measurements
    row_rect, col_rect, cost_rect = hungarian(rect_cost)
    print("3 tracks assigned to 6 measurements")
    print(f"Assignments: {list(zip(row_rect, col_rect))}")
    assigned_cols = set(col_rect)
    unassigned_cols = [i for i in range(6) if i not in assigned_cols]
    print(f"Unassigned measurements: {unassigned_cols}")

    # With cost of non-assignment (allows skipping bad matches)
    print("\n--- With Cost of Non-Assignment ---")
    cost_with_skip = np.array(
        [
            [10, 100, 100],
            [100, 5, 100],
            [100, 100, 100],  # Track 2 has no good matches
        ],
        dtype=float,
    )
    result_skip = assign2d(cost_with_skip, cost_of_non_assignment=20)
    print("Cost matrix with one track having no good matches:")
    print(cost_with_skip)
    print(f"Assignments: {list(zip(result_skip.row_indices, result_skip.col_indices))}")
    print(f"Unassigned rows: {list(result_skip.unassigned_rows)}")


def demo_kbest_assignment():
    """Demonstrate k-best 2D assignment (Murty's algorithm)."""
    print("\n" + "=" * 70)
    print("K-Best 2D Assignment Demo (Murty's Algorithm)")
    print("=" * 70)

    # Cost matrix
    cost = np.array(
        [
            [10, 5, 13],
            [3, 15, 8],
            [12, 7, 9],
        ],
        dtype=float,
    )

    print("\nCost matrix (3x3):")
    print(cost)

    # Find k best assignments
    k = 5
    result = murty(cost, k=k)

    print(f"\n--- Finding {k} Best Assignments ---")
    print(f"Found: {result.n_found} assignments")

    for i, (assignment, cost_val) in enumerate(zip(result.assignments, result.costs)):
        row_ind = assignment.row_indices
        col_ind = assignment.col_indices
        print(f"\n  Solution {i + 1} (cost={cost_val:.1f}):")
        print(f"    Assignments: {list(zip(row_ind, col_ind))}")

    # With cost threshold
    print("\n--- With Cost Threshold ---")
    result_thresh = kbest_assign2d(cost, k=10, cost_threshold=20)
    print(f"Assignments with cost <= 20: {result_thresh.n_found}")
    for i, c in enumerate(result_thresh.costs):
        print(f"  Solution {i + 1}: cost={c:.1f}")

    # Ranked assignments (convenience function)
    print("\n--- Ranked Assignment Enumeration ---")
    ranked = ranked_assignments(cost, max_assignments=6)
    print(f"Enumerated {ranked.n_found} assignments in order of increasing cost")
    print(f"Cost range: [{ranked.costs[0]:.1f}, {ranked.costs[-1]:.1f}]")


def demo_3d_assignment():
    """Demonstrate 3D assignment algorithms."""
    print("\n" + "=" * 70)
    print("3D Assignment Algorithms Demo")
    print("=" * 70)

    # 3D assignment: associate measurements across 3 scans
    # Cost tensor: cost[i, j, k] = cost of associating
    #   measurement i from scan 1, j from scan 2, k from scan 3
    np.random.seed(42)
    n = 5  # 5 measurements per scan
    cost = np.random.rand(n, n, n) * 10

    # Add some low-cost "true" associations on the diagonal
    for i in range(n):
        cost[i, i, i] = np.random.rand() * 0.5

    print(f"\nCost tensor shape: {cost.shape}")
    print("(5 measurements per scan, 3 scans)")

    # Greedy algorithm (fast but suboptimal)
    print("\n--- Greedy Algorithm ---")
    result_greedy = greedy_3d(cost)
    print(f"Assignments found: {result_greedy.tuples.shape[0]}")
    print(f"Total cost: {result_greedy.cost:.3f}")

    # 2D decomposition method
    print("\n--- 2D Decomposition Method ---")
    result_decomp = decompose_to_2d(cost)
    print(f"Assignments found: {result_decomp.tuples.shape[0]}")
    print(f"Total cost: {result_decomp.cost:.3f}")

    # Lagrangian relaxation (better quality)
    print("\n--- Lagrangian Relaxation ---")
    result_lagr = assign3d_lagrangian(cost, max_iter=100, tol=0.01)
    print(f"Assignments found: {result_lagr.tuples.shape[0]}")
    print(f"Total cost: {result_lagr.cost:.3f}")
    print(f"Iterations: {result_lagr.n_iterations}")
    print(f"Duality gap: {result_lagr.gap:.4f}")

    # Auction algorithm
    print("\n--- Auction Algorithm ---")
    result_auct = assign3d_auction(cost, max_iter=500)
    print(f"Assignments found: {result_auct.tuples.shape[0]}")
    print(f"Total cost: {result_auct.cost:.3f}")
    print(f"Iterations: {result_auct.n_iterations}")

    # Unified interface
    print("\n--- Unified assign3d() Interface ---")
    for method in ["greedy", "decompose", "lagrangian"]:
        result = assign3d(cost, method=method)
        print(
            f"  {method:12s}: cost={result.cost:.3f}, "
            f"assignments={result.tuples.shape[0]}"
        )

    # Show the actual assignments from the best method
    print("\n--- Best Solution Details ---")
    best = result_lagr
    print("Assignment tuples (scan1, scan2, scan3):")
    for row in best.tuples:
        i, j, k = row
        print(f"  ({i}, {j}, {k}) -> cost={cost[i, j, k]:.3f}")


def demo_gating():
    """Demonstrate measurement gating."""
    print("\n" + "=" * 70)
    print("Measurement Gating Demo")
    print("=" * 70)

    # Track predicted state and covariance
    track_pred = np.array([10.0, 20.0])  # 2D position
    innovation_cov = np.array([[4.0, 0.5], [0.5, 2.0]])  # Innovation covariance

    # Measurements (some close, some far)
    measurements = np.array(
        [
            [10.5, 20.2],  # Very close
            [11.0, 19.5],  # Close
            [12.5, 22.0],  # Moderate
            [8.0, 18.0],  # Moderate
            [20.0, 30.0],  # Far
            [5.0, 25.0],  # Far
        ]
    )

    print(f"\nTrack predicted position: {track_pred}")
    print(f"Innovation covariance:\n{innovation_cov}")
    print(f"\nMeasurements:\n{measurements}")

    # Compute Mahalanobis distances
    print("\n--- Mahalanobis Distances ---")
    for i, meas in enumerate(measurements):
        innovation = meas - track_pred
        dist = mahalanobis_distance(innovation, innovation_cov)
        print(f"  Measurement {i}: distance = {dist:.3f}")

    # Chi-squared gate threshold
    n_dims = 2
    gate_prob = 0.99
    threshold = chi2_gate_threshold(gate_prob, n_dims)
    print(f"\nGate threshold (99% probability, 2D): {threshold:.3f}")

    # Ellipsoidal gating
    print("\n--- Ellipsoidal Gating ---")
    gated_indices = []
    for i, meas in enumerate(measurements):
        innovation = meas - track_pred
        in_gate = ellipsoidal_gate(innovation, innovation_cov, threshold)
        status = "PASS" if in_gate else "FAIL"
        if in_gate:
            gated_indices.append(i)
        print(f"  Measurement {i}: {status}")
    print(f"Gated measurement indices: {gated_indices}")

    # Rectangular gating (simpler, less accurate)
    print("\n--- Rectangular Gating ---")
    for i, meas in enumerate(measurements):
        innovation = meas - track_pred
        in_gate = rectangular_gate(innovation, innovation_cov, num_sigmas=3.0)
        status = "PASS" if in_gate else "FAIL"
        print(f"  Measurement {i}: {status}")


def demo_gnn_association():
    """Demonstrate Global Nearest Neighbor association."""
    print("\n" + "=" * 70)
    print("Global Nearest Neighbor (GNN) Association Demo")
    print("=" * 70)

    # Scenario: 3 tracks, 4 measurements
    np.random.seed(42)

    # Track predicted positions (2D)
    track_preds = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0],
        ]
    )

    # Track covariances (stacked)
    track_covs = np.array([np.eye(2) * 4.0 for _ in range(3)])

    # Measurements (3 close to tracks, 1 false alarm)
    measurements = np.array(
        [
            [10.5, 19.8],  # Close to track 0
            [30.2, 40.5],  # Close to track 1
            [49.5, 60.2],  # Close to track 2
            [100.0, 100.0],  # False alarm (far from all tracks)
        ]
    )

    print("Track positions:")
    for i, pred in enumerate(track_preds):
        print(f"  Track {i}: {pred}")

    print("\nMeasurements:")
    for i, meas in enumerate(measurements):
        print(f"  Measurement {i}: {meas}")

    # Compute cost matrix
    print("\n--- Computing Cost Matrix ---")
    cost_matrix = compute_association_cost(
        track_preds,
        track_covs,
        measurements,
    )
    print("Cost matrix (tracks x measurements):")
    print(np.array2string(cost_matrix, precision=2, suppress_small=True))

    # GNN association using the cost matrix
    print("\n--- GNN Association ---")
    gate_threshold = chi2_gate_threshold(0.99, 2)
    result = gnn_association(cost_matrix, gate_threshold=gate_threshold)

    print(f"Track -> Measurement mapping: {list(result.track_to_measurement)}")
    print(f"Measurement -> Track mapping: {list(result.measurement_to_track)}")
    print(f"Total cost: {result.total_cost:.3f}")

    # Interpret results
    print("\nInterpretation:")
    for track_idx, meas_idx in enumerate(result.track_to_measurement):
        if meas_idx >= 0:
            print(f"  Track {track_idx} <- Measurement {meas_idx}")
        else:
            print(f"  Track {track_idx} <- (no measurement)")

    unassigned_meas = [i for i, t in enumerate(result.measurement_to_track) if t < 0]
    if unassigned_meas:
        print(f"  Unassigned measurements (false alarms): {unassigned_meas}")

    # Gated GNN (combined gating + association)
    print("\n--- Gated GNN Association ---")
    result_gated = gated_gnn_association(
        track_preds,
        track_covs,
        measurements,
        gate_probability=0.99,
    )
    print(f"Track -> Measurement: {list(result_gated.track_to_measurement)}")
    print(f"Total cost: {result_gated.total_cost:.3f}")


def demo_jpda():
    """Demonstrate Joint Probabilistic Data Association."""
    print("\n" + "=" * 70)
    print("JPDA (Joint Probabilistic Data Association) Demo")
    print("=" * 70)

    # Scenario: 2 closely-spaced tracks with ambiguous measurements
    track_states = [
        np.array([10.0, 0.0, 20.0, 0.0]),  # [x, vx, y, vy]
        np.array([12.0, 0.0, 21.0, 0.0]),  # Close to track 0
    ]

    track_covs = [np.diag([2.0, 0.1, 2.0, 0.1]) for _ in range(2)]

    # Measurements in the ambiguous region
    measurements = np.array(
        [
            [10.5, 20.2],  # Could belong to track 0 or 1
            [11.5, 20.8],  # Could belong to track 0 or 1
            [50.0, 50.0],  # False alarm
        ]
    )

    # Measurement model: H extracts [x, y] from state
    H = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    R = np.eye(2) * 1.0

    print("Track positions (closely spaced):")
    for i, state in enumerate(track_states):
        print(f"  Track {i}: position=({state[0]:.1f}, {state[2]:.1f})")

    print("\nMeasurements:")
    for i, meas in enumerate(measurements):
        print(f"  Measurement {i}: {meas}")

    # JPDA computes association probabilities
    print("\n--- JPDA Association Probabilities ---")
    result = jpda(
        track_states,
        track_covs,
        measurements,
        H=H,
        R=R,
        detection_prob=0.9,
        clutter_density=1e-6,
        gate_probability=0.99,
    )

    print("Association probability matrix (tracks x [measurements..., no-detect]):")
    print("  Rows: tracks, Columns: [meas 0, meas 1, meas 2, no-detection]")
    print(np.array2string(result.association_probs, precision=3, suppress_small=True))

    print("\nInterpretation:")
    n_tracks, n_cols = result.association_probs.shape
    n_meas = n_cols - 1  # Last column is no-detection probability
    for i in range(n_tracks):
        print(f"  Track {i}:")
        for j in range(n_meas):
            prob = result.association_probs[i, j]
            if prob > 0.01:  # Only show significant probabilities
                print(f"    P(measurement {j}) = {prob:.3f}")
        print(f"    P(no detection) = {result.association_probs[i, -1]:.3f}")


def demo_tracking_scenario():
    """Demonstrate a complete tracking scenario."""
    print("\n" + "=" * 70)
    print("Complete Tracking Scenario Demo")
    print("=" * 70)

    np.random.seed(42)

    # Simulation: 3 targets, 10 time steps
    n_targets = 3
    n_steps = 10

    # Initial target positions
    targets = np.array(
        [
            [0.0, 0.0],
            [50.0, 0.0],
            [25.0, 43.3],  # Equilateral triangle
        ]
    )

    # Target velocities
    velocities = np.array(
        [
            [2.0, 1.0],
            [-1.0, 2.0],
            [0.0, -1.5],
        ]
    )

    print(f"Simulating {n_targets} targets over {n_steps} time steps")
    print("Initial positions:", targets.tolist())

    # Track states (initially equal to true positions)
    track_states = targets.copy()
    track_covs = np.array([np.eye(2) * 10.0 for _ in range(n_targets)])

    # Measurement noise
    meas_std = 2.0

    # Run simulation
    assignment_history = []
    for t in range(n_steps):
        # Move targets
        targets = targets + velocities

        # Generate measurements (with some noise and false alarms)
        measurements = targets + np.random.randn(n_targets, 2) * meas_std

        # Add false alarms
        n_false = np.random.poisson(1)  # Average 1 false alarm per scan
        if n_false > 0:
            false_alarms = np.random.rand(n_false, 2) * 100 - 25
            measurements = np.vstack([measurements, false_alarms])

        # Use gated_gnn_association which handles gating internally
        result = gated_gnn_association(
            track_states,
            track_covs,
            measurements,
            gate_probability=0.99,
        )

        # Use track_to_measurement directly
        track_to_meas = list(result.track_to_measurement)
        assignment_history.append(track_to_meas)

        # Update track states (simple: just use measurement if assigned)
        for i, meas_idx in enumerate(track_to_meas):
            if meas_idx >= 0:
                # Blend prediction with measurement
                alpha = 0.7  # Measurement weight
                track_states[i] = alpha * measurements[meas_idx] + (1 - alpha) * (
                    track_states[i] + velocities[i]
                )
            else:
                # No measurement, just predict
                track_states[i] = track_states[i] + velocities[i]

    print("\nAssignment history (track -> measurement index):")
    for t, assignments in enumerate(assignment_history):
        print(f"  Step {t}: {assignments}")

    print("\nFinal track positions:")
    for i, state in enumerate(track_states):
        true_pos = targets[i]
        error = np.linalg.norm(state - true_pos)
        print(f"  Track {i}: {state} (error: {error:.2f})")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Assignment Algorithms Example")
    print("#" * 70)

    # Basic 2D assignment
    demo_2d_assignment()

    # K-best assignment (Murty)
    demo_kbest_assignment()

    # 3D assignment
    demo_3d_assignment()

    # Gating
    demo_gating()

    # Data association
    demo_gnn_association()

    # JPDA
    demo_jpda()

    # Complete scenario
    demo_tracking_scenario()

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
