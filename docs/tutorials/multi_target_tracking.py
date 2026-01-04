"""Multi-Target Tracking Tutorial with Interactive Visualizations.

This tutorial demonstrates the Global Nearest Neighbor (GNN) algorithm
for associating measurements to tracks in multi-target tracking scenarios.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def multi_target_tracking_tutorial():
    """Run complete multi-target tracking tutorial with visualizations."""
    
    print("=" * 70)
    print("MULTI-TARGET TRACKING TUTORIAL")
    print("=" * 70)
    
    # Step 1: Define Target Trajectories
    print("\nStep 1: Define Target Trajectories")
    print("-" * 70)
    
    dt = 0.1  # Time step
    duration = 20  # seconds
    n_steps = int(duration / dt)
    time = np.arange(n_steps) * dt
    
    # Define 4 targets with different trajectories
    targets = [
        {
            "id": 1,
            "start_pos": np.array([0.0, 0.0]),
            "velocity": np.array([5.0, 1.0]),
            "color": "blue"
        },
        {
            "id": 2,
            "start_pos": np.array([10.0, 5.0]),
            "velocity": np.array([1.0, 3.0]),
            "color": "green"
        },
        {
            "id": 3,
            "start_pos": np.array([20.0, 10.0]),
            "velocity": np.array([-2.0, -1.0]),
            "color": "red"
        },
        {
            "id": 4,
            "start_pos": np.array([5.0, 15.0]),
            "velocity": np.array([2.0, -3.0]),
            "color": "purple"
        }
    ]
    
    # Generate true positions
    true_positions = {}
    for target in targets:
        positions = []
        for i in range(n_steps):
            pos = target["start_pos"] + target["velocity"] * time[i]
            positions.append(pos)
        true_positions[target["id"]] = np.array(positions)
    
    print(f"Created {len(targets)} targets with linear trajectories")
    for target in targets:
        print(f"  Target {target['id']}: start={target['start_pos']}, "
              f"velocity={target['velocity']}")
    
    # Step 2: Generate Measurements
    print("\nStep 2: Generate Noisy Measurements")
    print("-" * 70)
    
    measurement_noise = 0.5  # Standard deviation
    detection_probability = 0.95  # Probability of detection
    false_alarm_rate = 0.01  # Probability of false alarm
    
    np.random.seed(42)
    
    all_measurements = []
    measurement_association = []  # Ground truth: which measurement came from which target
    
    for i in range(n_steps):
        step_measurements = []
        step_association = []
        
        # Generate measurements from true targets
        for target in targets:
            if np.random.rand() < detection_probability:
                true_pos = true_positions[target["id"]][i]
                meas = true_pos + np.random.randn(2) * measurement_noise
                step_measurements.append(meas)
                step_association.append(target["id"])
        
        # Generate false alarms
        n_false_alarms = np.random.poisson(false_alarm_rate)
        for _ in range(n_false_alarms):
            # False alarms in a 30x20 region
            false_meas = np.array([
                np.random.uniform(0, 30),
                np.random.uniform(0, 20)
            ])
            step_measurements.append(false_meas)
            step_association.append(0)  # 0 indicates false alarm
        
        all_measurements.append(step_measurements)
        measurement_association.append(step_association)
    
    print(f"Measurement statistics:")
    print(f"  Detection probability: {detection_probability}")
    print(f"  False alarm rate: {false_alarm_rate}")
    print(f"  Measurement noise: {measurement_noise}")
    
    # Step 3: Simple GNN Association
    print("\nStep 3: Run Global Nearest Neighbor (GNN) Tracking")
    print("-" * 70)
    
    # Track management
    class Track:
        def __init__(self, initial_meas, track_id):
            self.id = track_id
            self.position = initial_meas
            self.velocity = np.array([0.0, 0.0])
            self.covariance = np.eye(2) * measurement_noise**2
            self.age = 1
            self.position_history = [initial_meas]
    
    tracks = []
    track_counter = 0
    gating_threshold = 10.0  # Mahalanobis distance threshold
    
    estimated_positions = {}
    
    for step_idx, (measurements, true_assoc) in enumerate(zip(all_measurements, measurement_association)):
        # Predict track positions
        for track in tracks:
            track.position = track.position + track.velocity * dt
        
        measurements = np.array(measurements) if measurements else np.empty((0, 2))
        
        if len(tracks) == 0 or len(measurements) == 0:
            # Initialize new tracks from unassociated measurements
            for i, meas in enumerate(measurements):
                track_counter += 1
                tracks.append(Track(meas, track_counter))
                estimated_positions.setdefault(track_counter, []).append(meas)
        else:
            # Association using nearest neighbor
            distances = np.zeros((len(tracks), len(measurements)))
            for i, track in enumerate(tracks):
                for j, meas in enumerate(measurements):
                    distances[i, j] = np.linalg.norm(track.position - meas)
            
            # Greedy nearest neighbor
            used_measurements = set()
            used_tracks = set()
            
            # Sort by distance
            for i in range(min(len(tracks), len(measurements))):
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                track_idx, meas_idx = min_idx
                
                if distances[track_idx, meas_idx] > gating_threshold:
                    break
                
                if track_idx not in used_tracks and meas_idx not in used_measurements:
                    # Association
                    track = tracks[track_idx]
                    meas = measurements[meas_idx]
                    
                    # Update track
                    innovation = meas - track.position
                    track.velocity = 0.95 * track.velocity + 0.05 * innovation / dt
                    track.position = meas
                    track.age += 1
                    track.position_history.append(meas.copy())
                    estimated_positions.setdefault(track.id, []).append(meas)
                    
                    used_measurements.add(meas_idx)
                    used_tracks.add(track_idx)
                    
                    distances[track_idx, :] = np.inf
                    distances[:, meas_idx] = np.inf
            
            # Create new tracks from unassociated measurements
            for j, meas in enumerate(measurements):
                if j not in used_measurements:
                    track_counter += 1
                    tracks.append(Track(meas, track_counter))
                    estimated_positions.setdefault(track_counter, []).append(meas)
            
            # Remove old tracks
            tracks = [t for t in tracks if t.age > 2]
    
    print(f"GNN association complete")
    print(f"Tracks initiated: {track_counter}")
    
    # Step 4: Visualize Results
    print("\nStep 4: Create Visualizations")
    print("-" * 70)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Multi-Target Tracking: Truth and Measurements",
            "Tracked Trajectories (GNN Association)"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Subplot 1: Ground truth and measurements
    colors_map = {1: "blue", 2: "green", 3: "red", 4: "purple"}
    
    for target in targets:
        # True trajectory
        fig.add_trace(
            go.Scatter(
                x=true_positions[target["id"]][:, 0],
                y=true_positions[target["id"]][:, 1],
                mode="lines",
                name=f"Target {target['id']} (True)",
                line=dict(color=colors_map[target["id"]], width=3, dash="dash"),
                showlegend=True,
            ),
            row=1, col=1
        )
    
    # All measurements
    all_meas_x = []
    all_meas_y = []
    meas_colors = []
    
    for step_idx, measurements in enumerate(all_measurements):
        for meas_idx, meas in enumerate(measurements):
            all_meas_x.append(meas[0])
            all_meas_y.append(meas[1])
            target_id = measurement_association[step_idx][meas_idx]
            if target_id == 0:
                meas_colors.append("gray")
            else:
                meas_colors.append(colors_map[target_id])
    
    fig.add_trace(
        go.Scatter(
            x=all_meas_x, y=all_meas_y,
            mode="markers",
            name="Measurements",
            marker=dict(size=3, color=meas_colors, opacity=0.5),
            showlegend=True,
        ),
        row=1, col=1
    )
    
    # Subplot 2: Estimated trajectories
    for target_id in range(1, 5):
        if target_id in estimated_positions:
            positions = np.array(estimated_positions[target_id])
            fig.add_trace(
                go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode="lines",
                    name=f"Track {target_id}",
                    line=dict(color=colors_map[target_id], width=2),
                    showlegend=False,
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="X Position (m)", row=1, col=2)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=2)
    
    fig.update_layout(
        title_text="Multi-Target Tracking Tutorial - GNN Association",
        height=600,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "multi_target_tracking.html"))
    
    print("✓ Multi-target tracking visualization complete")
    print("=" * 70)


if __name__ == "__main__":
    multi_target_tracking_tutorial()
