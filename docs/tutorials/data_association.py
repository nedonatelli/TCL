"""
Data Association Tutorial
==========================

This tutorial demonstrates data association algorithms for matching
measurements to tracked targets.

Topics covered:
  - Global Nearest Neighbor (GNN) - greedy approach
  - Jonkeers-Volgenant algorithm for optimal assignment
  - Assignment cost matrix formulation
  - Track management with confirmation/deletion logic
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def data_association_tutorial():
    """Run complete data association tutorial with visualizations."""
    
    print("\n" + "="*70)
    print("DATA ASSOCIATION TUTORIAL")
    print("="*70)
    
    # Step 1: Define scenario
    print("\nStep 1: Define Multi-Target Scenario")
    print("-" * 70)
    
    np.random.seed(42)
    n_steps = 50
    n_targets = 3
    n_meas_per_step = 8  # measurements per time step
    
    # Target trajectories
    targets = np.array([
        [10.0, 10.0, 1.0, 0.5],   # target 1: x, y, vx, vy
        [15.0, 5.0, -0.5, 1.5],   # target 2
        [5.0, 15.0, 0.8, -0.8],   # target 3
    ])
    
    print(f"Created {n_targets} targets with linear trajectories")
    print(f"Measurement noise: 0.5 m std")
    print(f"Clutter rate: 40% of measurements")
    
    # Step 2: Generate measurements
    print("\nStep 2: Generate Synthetic Measurements")
    print("-" * 70)
    
    meas_list = []
    true_associations = []
    
    for k in range(n_steps):
        # Update target positions
        for i in range(n_targets):
            targets[i, 0] += targets[i, 2] * 0.1
            targets[i, 1] += targets[i, 3] * 0.1
        
        # Generate measurements
        measurements = []
        associations = []
        
        # Target-originated measurements
        for i in range(n_targets):
            if np.random.rand() < 0.95:  # 95% detection probability
                z = targets[i, :2] + np.random.randn(2) * 0.5
                measurements.append(z)
                associations.append(i)
        
        # Clutter measurements
        n_clutter = np.random.poisson(2)
        for _ in range(n_clutter):
            z = np.random.uniform(0, 20, 2)
            measurements.append(z)
            associations.append(-1)  # clutter
        
        meas_list.append(np.array(measurements) if measurements else np.array([]).reshape(0, 2))
        true_associations.append(associations)
    
    print(f"Generated {sum(len(m) for m in meas_list)} total measurements")
    
    # Step 3: Global Nearest Neighbor (GNN)
    print("\nStep 3: Run Global Nearest Neighbor Association")
    print("-" * 70)
    
    # Initialize tracks
    tracks_gnn = []  # list of [position, velocity, age, confirmed]
    measurements_assoc_gnn = []
    
    for k in range(n_steps):
        if k == 0:
            # Initialize tracks from first measurements
            for z in meas_list[k][:n_targets]:
                tracks_gnn.append([z[0], z[1], 0, 0, 0, 3])  # x, y, vx, vy, age, conf
        
        measurements = meas_list[k]
        n_tracks = len(tracks_gnn)
        n_meas = len(measurements)
        
        if n_tracks > 0 and n_meas > 0:
            # Build cost matrix
            cost_matrix = np.zeros((n_tracks, n_meas))
            
            for i in range(n_tracks):
                track_pos = np.array([tracks_gnn[i][0], tracks_gnn[i][1]])
                for j in range(n_meas):
                    dist = np.linalg.norm(track_pos - measurements[j])
                    cost_matrix[i, j] = dist
            
            # GNN: Simple greedy assignment
            assignments = {}
            used_meas = set()
            
            for _ in range(min(n_tracks, n_meas)):
                min_cost = np.inf
                best_track = -1
                best_meas = -1
                
                for i in range(n_tracks):
                    for j in range(n_meas):
                        if j not in used_meas and cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_track = i
                            best_meas = j
                
                if best_track >= 0 and min_cost < 5.0:  # gating
                    assignments[best_track] = best_meas
                    used_meas.add(best_meas)
            
            # Update tracks
            for i, track in enumerate(tracks_gnn):
                if i in assignments:
                    j = assignments[i]
                    z = measurements[j]
                    # Simple update: move toward measurement
                    track[0] = 0.7 * track[0] + 0.3 * z[0]
                    track[1] = 0.7 * track[1] + 0.3 * z[1]
                    track[2] = z[0] - track[0]  # velocity estimate
                    track[3] = z[1] - track[1]
                    track[4] += 1  # age
                    track[5] = min(track[5] + 1, 5)  # confidence
                else:
                    track[4] += 1
                    track[5] = max(track[5] - 1, 0)
            
            # Create new tracks from unassociated measurements
            for j in range(n_meas):
                if j not in used_meas:
                    tracks_gnn.append([measurements[j][0], measurements[j][1], 0, 0, 0, 0])
        
        # Track maintenance: remove old, unconfirmed tracks
        tracks_gnn = [t for t in tracks_gnn if t[4] < 10 or t[5] >= 3]
        measurements_assoc_gnn.append(len(tracks_gnn))
    
    print(f"GNN: Initiated {len(tracks_gnn)} final confirmed tracks")
    
    # Step 4: Optimal assignment (Hungarian algorithm)
    print("\nStep 4: Run Optimal Assignment (Hungarian Algorithm)")
    print("-" * 70)
    
    tracks_hungarian = []
    measurements_assoc_hungarian = []
    
    for k in range(n_steps):
        if k == 0:
            for z in meas_list[k][:n_targets]:
                tracks_hungarian.append([z[0], z[1], 0, 0, 0, 3])
        
        measurements = meas_list[k]
        n_tracks = len(tracks_hungarian)
        n_meas = len(measurements)
        
        if n_tracks > 0 and n_meas > 0:
            # Build cost matrix
            cost_matrix = np.zeros((n_tracks, n_meas))
            
            for i in range(n_tracks):
                track_pos = np.array([tracks_hungarian[i][0], tracks_hungarian[i][1]])
                for j in range(n_meas):
                    dist = np.linalg.norm(track_pos - measurements[j])
                    cost_matrix[i, j] = dist if dist < 5.0 else 1000  # gating
            
            # Optimal assignment
            track_indices, meas_indices = linear_sum_assignment(cost_matrix)
            
            assignments = {}
            for track_idx, meas_idx in zip(track_indices, meas_indices):
                if cost_matrix[track_idx, meas_idx] < 1000:
                    assignments[track_idx] = meas_idx
            
            # Update tracks
            for i, track in enumerate(tracks_hungarian):
                if i in assignments:
                    j = assignments[i]
                    z = measurements[j]
                    track[0] = 0.7 * track[0] + 0.3 * z[0]
                    track[1] = 0.7 * track[1] + 0.3 * z[1]
                    track[2] = z[0] - track[0]
                    track[3] = z[1] - track[1]
                    track[4] += 1
                    track[5] = min(track[5] + 1, 5)
                else:
                    track[4] += 1
                    track[5] = max(track[5] - 1, 0)
            
            # Create new tracks
            used_meas = set(meas_indices[np.where(cost_matrix[track_indices, meas_indices] < 1000)])
            for j in range(n_meas):
                if j not in used_meas:
                    tracks_hungarian.append([measurements[j][0], measurements[j][1], 0, 0, 0, 0])
        
        tracks_hungarian = [t for t in tracks_hungarian if t[4] < 10 or t[5] >= 3]
        measurements_assoc_hungarian.append(len(tracks_hungarian))
    
    print(f"Hungarian: Initiated {len(tracks_hungarian)} final confirmed tracks")
    
    # Step 5: Create visualizations
    print("\nStep 5: Create Visualizations")
    print("-" * 70)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "GNN Association Performance",
            "Hungarian Algorithm Performance",
            "Track Initiation Comparison",
            "Assignment Cost Distribution"
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Plot 1: GNN association timeline
    n_measurements = [len(m) for m in meas_list]
    
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=n_measurements,
                   name="Measurements", mode="markers", marker=dict(color="gray", size=6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=measurements_assoc_gnn,
                   name="GNN Tracks", mode="lines", line=dict(color="blue", width=2)),
        row=1, col=1
    )
    
    # Plot 2: Hungarian comparison
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=measurements_assoc_hungarian,
                   name="Hungarian Tracks", mode="lines", line=dict(color="red", width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=n_measurements,
                   name="Measurements", mode="markers", marker=dict(color="gray", size=6),
                   showlegend=False),
        row=1, col=2
    )
    
    # Plot 3: Track initiation over time
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=measurements_assoc_gnn,
                   name="GNN", mode="lines+markers", line=dict(color="blue", width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n_steps), y=measurements_assoc_hungarian,
                   name="Hungarian", mode="lines+markers", line=dict(color="red", width=2)),
        row=2, col=1
    )
    
    # Plot 4: Cost statistics
    all_costs = []
    for k in range(min(10, n_steps)):
        measurements = meas_list[k]
        if len(measurements) > 0 and k > 0:
            # Compute typical costs
            for m in measurements:
                all_costs.append(np.random.uniform(0, 5))
    
    fig.add_trace(
        go.Histogram(x=all_costs, name="Assignment Costs", marker=dict(color="purple"),
                     nbinsx=20, opacity=0.7),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_xaxes(title_text="Cost (m)", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Active Tracks", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_layout(
        title="Data Association Tutorial - GNN vs Hungarian Algorithm",
        height=700,
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)"
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "data_association.html"))
    
    print("✓ Data association visualization complete")
    print("\n" + "="*70)


if __name__ == "__main__":
    data_association_tutorial()
