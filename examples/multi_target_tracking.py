"""
Multi-target tracking example.

This example demonstrates:
1. Simulating multiple crossing targets
2. Using the MultiTargetTracker for GNN-based tracking
3. Track initiation, confirmation, and deletion

Run with: python examples/multi_target_tracking.py
"""

# Add parent directory to path for development
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from tcl.trackers import (  # noqa: E402
    MultiTargetTracker,
    TrackStatus,
)


def simulate_targets(
    n_steps: int = 50,
    dt: float = 1.0,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Simulate two crossing targets with position measurements.

    Returns
    -------
    true_states : list of ndarray
        Ground truth states [x1, y1, x2, y2] at each step.
    measurements : list of list of ndarray
        Noisy position measurements at each step.
    """
    # Target 1: Moving right and up
    x1_0, y1_0 = 0.0, 0.0
    vx1, vy1 = 2.0, 1.0

    # Target 2: Moving left and up
    x2_0, y2_0 = 100.0, 0.0
    vx2, vy2 = -2.0, 1.5

    true_states = []
    measurements = []
    R = np.eye(2) * 2.0  # Measurement noise covariance

    for k in range(n_steps):
        t = k * dt

        # True positions
        x1 = x1_0 + vx1 * t
        y1 = y1_0 + vy1 * t
        x2 = x2_0 + vx2 * t
        y2 = y2_0 + vy2 * t

        true_states.append(np.array([x1, y1, x2, y2]))

        # Generate noisy measurements
        meas = []

        # Detection probability
        pd = 0.95

        if np.random.rand() < pd:
            z1 = np.array([x1, y1]) + np.random.multivariate_normal([0, 0], R)
            meas.append(z1)

        if np.random.rand() < pd:
            z2 = np.array([x2, y2]) + np.random.multivariate_normal([0, 0], R)
            meas.append(z2)

        # Add occasional false alarms
        if np.random.rand() < 0.1:
            # Random false alarm in scene
            fa = np.array([np.random.uniform(-10, 110), np.random.uniform(-10, 60)])
            meas.append(fa)

        measurements.append(meas)

    return true_states, measurements


def run_tracker(
    measurements: List[List[np.ndarray]],
    dt: float = 1.0,
) -> List[List]:
    """
    Run multi-target tracker on measurements.

    Returns list of track histories at each step.
    """

    # Constant velocity model: state = [x, vx, y, vy]
    def F(dt):
        return np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float64
        )

    # Measurement model: measure x and y
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

    # Process noise (acceleration noise)
    def Q(dt):
        q = 0.5  # Acceleration noise std
        return (
            np.array(
                [
                    [dt**4 / 4, dt**3 / 2, 0, 0],
                    [dt**3 / 2, dt**2, 0, 0],
                    [0, 0, dt**4 / 4, dt**3 / 2],
                    [0, 0, dt**3 / 2, dt**2],
                ]
            )
            * q**2
        )

    # Measurement noise
    R = np.eye(2) * 2.0

    # Initial covariance for new tracks
    P0 = np.diag([10.0, 5.0, 10.0, 5.0])

    # Create tracker
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=Q,
        R=R,
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=P0,
    )

    # Process all measurements
    track_history = []

    for meas in measurements:
        tracks = tracker.process(meas, dt)
        track_history.append(tracks)

    return track_history


def plot_results(
    true_states: List[np.ndarray],
    measurements: List[List[np.ndarray]],
    track_history: List[List],
) -> None:
    """Plot tracking results."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot true trajectories
    true_arr = np.array(true_states)
    ax.plot(true_arr[:, 0], true_arr[:, 1], "g-", linewidth=2, label="Target 1 (truth)")
    ax.plot(true_arr[:, 2], true_arr[:, 3], "b-", linewidth=2, label="Target 2 (truth)")

    # Plot measurements
    for k, meas in enumerate(measurements):
        for z in meas:
            ax.plot(z[0], z[1], "k.", markersize=3, alpha=0.5)

    # Plot tracks
    # Collect track positions by track ID
    track_positions = {}
    for k, tracks in enumerate(track_history):
        for track in tracks:
            if track.status == TrackStatus.CONFIRMED:
                if track.id not in track_positions:
                    track_positions[track.id] = []
                track_positions[track.id].append(
                    (track.state[0], track.state[2])
                )  # x, y

    # Plot each track
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (track_id, positions) in enumerate(track_positions.items()):
        if len(positions) > 1:
            positions = np.array(positions)
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                "o-",
                color=colors[i % 10],
                markersize=4,
                linewidth=1.5,
                label=f"Track {track_id}",
            )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Multi-Target Tracking with GNN Data Association")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("multi_target_tracking_result.png", dpi=150)
    print("Plot saved to multi_target_tracking_result.png")
    plt.show()


def main():
    """Run multi-target tracking example."""
    print("Multi-Target Tracking Example")
    print("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Simulate targets
    print("Simulating two crossing targets...")
    true_states, measurements = simulate_targets(n_steps=50, dt=1.0)
    print(f"  Generated {len(true_states)} time steps")
    print(f"  Total measurements: {sum(len(m) for m in measurements)}")

    # Run tracker
    print("\nRunning multi-target tracker...")
    track_history = run_tracker(measurements, dt=1.0)

    # Count tracks
    all_tracks = set()
    confirmed_tracks = set()
    for tracks in track_history:
        for track in tracks:
            all_tracks.add(track.id)
            if track.status == TrackStatus.CONFIRMED:
                confirmed_tracks.add(track.id)

    print(f"  Total tracks initiated: {len(all_tracks)}")
    print(f"  Confirmed tracks: {len(confirmed_tracks)}")

    # Final track summary
    final_tracks = track_history[-1]
    print(f"\nFinal active tracks: {len(final_tracks)}")
    for track in final_tracks:
        pos = (track.state[0], track.state[2])
        vel = (track.state[1], track.state[3])
        print(
            f"  Track {track.id}: pos=({pos[0]:.1f}, {pos[1]:.1f}), "
            f"vel=({vel[0]:.1f}, {vel[1]:.1f}), status={track.status.value}"
        )

    # Plot if matplotlib is available
    try:
        plot_results(true_states, measurements, track_history)
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
