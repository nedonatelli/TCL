"""
Track Management Workflow Examples
===================================

Demonstrates complete tracking pipelines using SQL and HDF5 storage backends.

Workflows:
1. Single-sensor RADAR tracking pipeline (SQL)
2. Multi-sensor fusion: RADAR + LiDAR (SQL)
3. Real-time to archive: SQL → HDF5
4. Track comparison: GNN vs MHT (HDF5 scenarios)
5. Scenario replay and validation (HDF5 + SQL)
6. Detection clutter handling (SQL)

Run with: python examples/track_management_workflows.py

Note: Workflows 3-5 require h5py: pip install h5py
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from pytcl.assignment_algorithms import (  # noqa: E402
    chi2_gate_threshold,
    compute_association_cost,
    gnn_association,
)
from pytcl.io import (  # noqa: E402
    TrackDatabaseManager,
    TrackDatabaseStatus,
)
from pytcl.trackers import (  # noqa: E402
    MHTConfig,
    MHTTracker,
    MultiTargetTracker,
    TrackStatus,
)

try:
    from pytcl.io import TrackHDF5Storage

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import plotly.graph_objects as go  # noqa: E402

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static" / "images" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Shared Simulation Utilities
# =============================================================================


def simulate_radar_targets(
    n_targets: int = 3,
    n_steps: int = 50,
    dt: float = 1.0,
    noise_std: float = 5.0,
    pd: float = 0.9,
    fa_rate: float = 0.2,
    scene_bounds: Tuple[float, float, float, float] = (-50, -50, 250, 150),
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """Simulate RADAR targets with constant velocity motion.

    Returns (true_states, measurements) where true_states is a list of
    (n_targets, 4) arrays [x, vx, y, vy] and measurements is a list of
    measurement lists per time step.
    """
    rng = np.random.default_rng(seed)

    # Initial states: [x, vx, y, vy]
    starts = [
        np.array([0.0, 2.0, 0.0, 1.0]),
        np.array([200.0, -1.5, 10.0, 1.5]),
        np.array([50.0, 0.5, 100.0, -1.0]),
    ]

    true_states: List[np.ndarray] = []
    measurements: List[List[np.ndarray]] = []
    R = np.eye(2) * noise_std**2

    for k in range(n_steps):
        t = k * dt
        states_k = np.zeros((n_targets, 4))
        meas_k: List[np.ndarray] = []

        for i in range(n_targets):
            s = starts[i % len(starts)]
            states_k[i] = [
                s[0] + s[1] * t,
                s[1],
                s[2] + s[3] * t,
                s[3],
            ]

            # Detection with probability pd
            if rng.random() < pd:
                pos = np.array([states_k[i, 0], states_k[i, 2]])
                z = pos + rng.multivariate_normal([0, 0], R)
                meas_k.append(z)

        # False alarms
        n_fa = rng.poisson(fa_rate)
        for _ in range(n_fa):
            fa = np.array(
                [
                    rng.uniform(scene_bounds[0], scene_bounds[2]),
                    rng.uniform(scene_bounds[1], scene_bounds[3]),
                ]
            )
            meas_k.append(fa)

        true_states.append(states_k)
        measurements.append(meas_k)

    return true_states, measurements


def simulate_lidar_detections(
    true_states: List[np.ndarray],
    target_indices: List[int],
    dt_lidar: float = 0.5,
    noise_std: float = 1.0,
    pd: float = 0.95,
    seed: int = 43,
) -> List[Tuple[float, List[np.ndarray]]]:
    """Generate LiDAR measurements at higher rate with lower noise.

    Returns list of (timestamp, measurements) tuples.
    """
    rng = np.random.default_rng(seed)
    R = np.eye(2) * noise_std**2
    n_steps = len(true_states)
    dt_radar = 1.0  # assumed radar dt

    lidar_scans: List[Tuple[float, List[np.ndarray]]] = []

    for k in range(n_steps):
        t_radar = k * dt_radar
        # Generate LiDAR scans between this and next radar scan
        n_lidar = int(dt_radar / dt_lidar)
        for j in range(n_lidar):
            t_lidar = t_radar + j * dt_lidar
            meas: List[np.ndarray] = []
            for idx in target_indices:
                if idx < true_states[k].shape[0]:
                    state = true_states[k][idx]
                    # Interpolate position within the radar step
                    frac = j * dt_lidar / dt_radar
                    x = state[0] + state[1] * frac * dt_radar
                    y = state[2] + state[3] * frac * dt_radar
                    if rng.random() < pd:
                        z = np.array([x, y]) + rng.multivariate_normal([0, 0], R)
                        meas.append(z)
            lidar_scans.append((t_lidar, meas))

    return lidar_scans


def setup_cv_tracker_params(
    dt: float = 1.0,
    q_std: float = 0.5,
    r_std: float = 5.0,
) -> Dict[str, Any]:
    """Return constant-velocity model parameters for 2D tracking.

    State: [x, vx, y, vy], Measurement: [x, y].
    """

    def F(dt_val: float) -> np.ndarray:
        return np.array(
            [
                [1, dt_val, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt_val],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

    def Q(dt_val: float) -> np.ndarray:
        return (
            np.array(
                [
                    [dt_val**4 / 4, dt_val**3 / 2, 0, 0],
                    [dt_val**3 / 2, dt_val**2, 0, 0],
                    [0, 0, dt_val**4 / 4, dt_val**3 / 2],
                    [0, 0, dt_val**3 / 2, dt_val**2],
                ]
            )
            * q_std**2
        )

    R = np.eye(2) * r_std**2
    P0 = np.diag([100.0, 25.0, 100.0, 25.0])

    return {"F": F, "H": H, "Q": Q, "R": R, "P0": P0, "state_dim": 4, "meas_dim": 2}


# =============================================================================
# Workflow 1: Single-Sensor RADAR Tracking Pipeline
# =============================================================================


def workflow_radar_tracking(
    output_dir: Path,
    n_steps: int = 50,
) -> None:
    """Single-sensor RADAR tracking with full SQL lifecycle management.

    Demonstrates:
    - Storing detections and track states in TrackDatabaseManager
    - Track lifecycle: initiation → confirmation → coasting → dead
    - Querying track histories and converting to TrackList
    - Database maintenance with pruning operations
    """
    print("\n" + "=" * 60)
    print("Workflow 1: Single-Sensor RADAR Tracking Pipeline")
    print("=" * 60)

    params = setup_cv_tracker_params()
    true_states, measurements = simulate_radar_targets(
        n_targets=3,
        n_steps=n_steps,
        seed=42,
    )

    db_path = str(output_dir / "radar_tracking.db")
    tracker = MultiTargetTracker(
        state_dim=params["state_dim"],
        meas_dim=params["meas_dim"],
        F=params["F"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=params["P0"],
    )

    db = TrackDatabaseManager(db_path)
    db.open(mode="w")

    t_start = time.perf_counter()
    total_dets = 0

    for k in range(n_steps):
        t = float(k)
        meas_k = measurements[k]

        # Store raw detections in SQL
        for j, z in enumerate(meas_k):
            det_id = f"det_{k:04d}_{j:02d}"
            db.store_detection(det_id, z, "radar", t)
            total_dets += 1

        # Run tracker
        tracks = tracker.process(meas_k, dt=1.0)

        # Persist all active tracks
        for track in tracks:
            db.store_from_track(track, timestamp=t)

    elapsed = time.perf_counter() - t_start

    # Demonstrate lifecycle queries
    all_tracks = db.retrieve_all_tracks()
    confirmed = db.retrieve_all_tracks(TrackDatabaseStatus.CONFIRMED)
    tentative = db.retrieve_all_tracks(TrackDatabaseStatus.TENTATIVE)

    # Mark some old tentative tracks as dead for demonstration
    for t_info in tentative:
        if t_info["hits"] <= 1:
            db.mark_track_dead(t_info["track_id"])

    dead = db.retrieve_all_tracks(TrackDatabaseStatus.DEAD)

    # Query track history for first confirmed track
    if confirmed:
        trk_id = confirmed[0]["track_id"]
        history = db.get_track_history(trk_id)
        print(f"\n  Track '{trk_id}' history: {len(history['timestamps'])} states")

    # Convert to TrackList
    track_list = db.tracks_to_tracklist(TrackDatabaseStatus.CONFIRMED)
    print(f"  TrackList: {len(track_list)} confirmed tracks")

    # Prune old detections
    pruned_dets = db.prune_old_detections(age_threshold=30.0)
    pruned_tracks = db.prune_dead_tracks(age_threshold=0.0)

    db_size = os.path.getsize(db_path)
    db.close()

    print(f"\n  Results:")
    print(f"    Total detections stored: {total_dets}")
    print(
        f"    Tracks: {len(all_tracks)} total, {len(confirmed)} confirmed, "
        f"{len(dead)} dead"
    )
    print(f"    Pruned: {pruned_dets} old detections, {pruned_tracks} dead tracks")
    print(
        f"    Processing time: {elapsed:.3f}s ({elapsed / n_steps * 1000:.1f} ms/scan)"
    )
    print(f"    Database size: {db_size / 1024:.1f} KB")


# =============================================================================
# Workflow 2: Multi-Sensor Fusion (RADAR + LiDAR)
# =============================================================================


def workflow_multi_sensor_fusion(
    output_dir: Path,
    n_steps: int = 50,
) -> None:
    """Multi-sensor fusion with RADAR and LiDAR using explicit Kalman filter.

    Demonstrates:
    - Storing detections from multiple sensors with different sensor_ids
    - Manual Kalman predict/update loop with GNN data association
    - Per-sensor detection queries
    - Comparing fused vs single-sensor tracking accuracy
    """
    print("\n" + "=" * 60)
    print("Workflow 2: Multi-Sensor Fusion (RADAR + LiDAR)")
    print("=" * 60)

    from pytcl.dynamic_estimation import kf_predict, kf_update

    params = setup_cv_tracker_params(r_std=5.0)
    true_states, radar_meas = simulate_radar_targets(
        n_targets=2,
        n_steps=n_steps,
        noise_std=5.0,
        pd=0.9,
        fa_rate=0.1,
        seed=42,
    )

    lidar_scans = simulate_lidar_detections(
        true_states,
        target_indices=[0, 1],
        dt_lidar=0.5,
        noise_std=1.0,
        pd=0.95,
        seed=43,
    )

    db_path = str(output_dir / "multi_sensor.db")
    db = TrackDatabaseManager(db_path)
    db.open(mode="w")

    H = params["H"]
    R_radar = np.eye(2) * 25.0
    R_lidar = np.eye(2) * 1.0

    # Initialize tracks from first measurements
    active_tracks: Dict[str, Dict[str, np.ndarray]] = {}
    next_track_id = 0

    # Collect all sensor events in time order
    events: List[Tuple[float, str, List[np.ndarray]]] = []
    for k in range(n_steps):
        events.append((float(k), "radar", radar_meas[k]))
    for t_l, meas_l in lidar_scans:
        events.append((t_l, "lidar", meas_l))
    events.sort(key=lambda e: e[0])

    last_predict_time = -1.0
    radar_only_errors: List[float] = []
    fused_errors: List[float] = []

    for t_event, sensor, meas_list in events:
        # Store detections
        for j, z in enumerate(meas_list):
            det_id = f"{sensor}_{t_event:.1f}_{j}"
            db.store_detection(det_id, z, sensor, t_event)

        if not meas_list:
            continue

        R_sensor = R_radar if sensor == "radar" else R_lidar

        # Predict all tracks to current time
        if active_tracks and t_event > last_predict_time:
            dt_pred = t_event - last_predict_time if last_predict_time >= 0 else 1.0
            F_k = params["F"](dt_pred)
            Q_k = params["Q"](dt_pred)
            for trk_id, trk in active_tracks.items():
                x_pred, P_pred = kf_predict(trk["state"], trk["cov"], F_k, Q_k)
                trk["state"] = x_pred
                trk["cov"] = P_pred
            last_predict_time = t_event

        # Data association
        Z = np.array(meas_list)
        if active_tracks:
            track_ids = list(active_tracks.keys())
            predictions = np.array([active_tracks[tid]["state"] for tid in track_ids])
            covariances = np.array([active_tracks[tid]["cov"] for tid in track_ids])

            H_batch = np.tile(H, (len(track_ids), 1, 1))
            cost_matrix = compute_association_cost(
                predictions,
                covariances,
                Z,
                H_batch,
            )
            gate = chi2_gate_threshold(0.99, 2)
            try:
                result = gnn_association(
                    cost_matrix,
                    gate_threshold=gate,
                    cost_of_non_assignment=gate,
                )
            except ValueError:
                # All costs gated — no feasible associations
                result = None

            # Update associated tracks
            associated_meas: set = set()
            if result is not None:
                for i, m_idx in enumerate(result.track_to_measurement):
                    if m_idx >= 0:
                        tid = track_ids[i]
                        trk = active_tracks[tid]
                        upd = kf_update(
                            trk["state"],
                            trk["cov"],
                            Z[m_idx],
                            H,
                            R_sensor,
                        )
                        x_upd, P_upd = upd.x, upd.P
                        trk["state"] = x_upd
                        trk["cov"] = P_upd
                        db.update_track_state(tid, x_upd, P_upd, t_event)
                        associated_meas.add(int(m_idx))

            # Initiate new tracks from unassociated measurements
            for j in range(len(Z)):
                if j not in associated_meas:
                    tid = f"fused_trk_{next_track_id:03d}"
                    next_track_id += 1
                    x0 = np.zeros(4)
                    x0[0], x0[2] = Z[j][0], Z[j][1]
                    P0 = params["P0"].copy()
                    active_tracks[tid] = {"state": x0, "cov": P0}
                    db.initiate_track(tid, x0, P0, t_event)
        else:
            # Initialize from first measurements
            for j, z in enumerate(Z):
                tid = f"fused_trk_{next_track_id:03d}"
                next_track_id += 1
                x0 = np.zeros(4)
                x0[0], x0[2] = z[0], z[1]
                P0 = params["P0"].copy()
                active_tracks[tid] = {"state": x0, "cov": P0}
                db.initiate_track(tid, x0, P0, t_event)

        # Compute errors at radar time steps against ground truth
        if sensor == "radar" and int(t_event) < len(true_states):
            k_idx = int(t_event)
            for trk in list(active_tracks.values())[:2]:
                pos_est = np.array([trk["state"][0], trk["state"][2]])
                best_err = float("inf")
                for tgt in range(true_states[k_idx].shape[0]):
                    pos_true = np.array(
                        [
                            true_states[k_idx][tgt, 0],
                            true_states[k_idx][tgt, 2],
                        ]
                    )
                    err = np.linalg.norm(pos_est - pos_true)
                    best_err = min(best_err, err)
                fused_errors.append(best_err)

    # Query per-sensor stats
    radar_dets = db.retrieve_detections(sensor_id="radar")
    lidar_dets = db.retrieve_detections(sensor_id="lidar")
    all_tracks_info = db.retrieve_all_tracks()
    db.close()

    fused_rmse = np.sqrt(np.mean(np.array(fused_errors) ** 2)) if fused_errors else 0.0

    print(f"\n  Results:")
    print(f"    RADAR detections: {len(radar_dets)}")
    print(f"    LiDAR detections: {len(lidar_dets)}")
    print(f"    Total tracks initiated: {len(all_tracks_info)}")
    print(f"    Fused position RMSE: {fused_rmse:.2f} m")
    print(f"    (Lower is better; LiDAR noise=1m supplements RADAR noise=5m)")


# =============================================================================
# Workflow 3: Real-Time to Archive (SQL → HDF5)
# =============================================================================


def workflow_realtime_to_archive(
    output_dir: Path,
    n_steps: int = 100,
) -> None:
    """Real-time SQL tracking with HDF5 archival and post-analysis queries.

    Demonstrates:
    - Phase 1: Real-time tracking with SQL persistence
    - Phase 2: Archival via import_from_sql()
    - Phase 3: Post-analysis queries on HDF5
    - Phase 4: Round-trip verification via export_to_sql()
    """
    if not HAS_H5PY:
        print("\n  [SKIP] Workflow 3 requires h5py")
        return

    print("\n" + "=" * 60)
    print("Workflow 3: Real-Time to Archive (SQL → HDF5)")
    print("=" * 60)

    params = setup_cv_tracker_params()
    true_states, measurements = simulate_radar_targets(
        n_targets=3,
        n_steps=n_steps,
        seed=42,
    )

    # Phase 1: Real-time SQL operations
    print("\n  Phase 1: Real-time tracking (SQL)...")
    sql_path = str(output_dir / "realtime.db")
    tracker = MultiTargetTracker(
        state_dim=params["state_dim"],
        meas_dim=params["meas_dim"],
        F=params["F"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=params["P0"],
    )

    db = TrackDatabaseManager(sql_path)
    db.open(mode="w")

    t_start = time.perf_counter()
    for k in range(n_steps):
        t = float(k)
        for j, z in enumerate(measurements[k]):
            db.store_detection(f"det_{k:04d}_{j:02d}", z, "radar", t)
        tracks = tracker.process(measurements[k], dt=1.0)
        for track in tracks:
            db.store_from_track(track, timestamp=t)
    sql_elapsed = time.perf_counter() - t_start

    n_sql_tracks = len(db.retrieve_all_tracks())
    n_sql_dets = len(db.retrieve_all_detections())
    print(f"    {n_sql_tracks} tracks, {n_sql_dets} detections in {sql_elapsed:.3f}s")

    # Phase 2: Archive to HDF5
    print("  Phase 2: Archiving to HDF5...")
    h5_path = str(output_dir / "mission_archive.h5")
    h5 = TrackHDF5Storage(h5_path)
    h5.open(mode="w")

    t_start = time.perf_counter()
    h5.import_from_sql(db, scenario_id="mission_001")
    archive_elapsed = time.perf_counter() - t_start

    h5_tracks = h5.list_tracks(scenario_id="mission_001")
    print(f"    Archived {len(h5_tracks)} tracks in {archive_elapsed:.3f}s")

    # Phase 3: Post-analysis queries
    print("  Phase 3: Post-analysis queries...")
    if h5_tracks:
        sample_track = h5_tracks[0]

        t_start = time.perf_counter()
        traj = h5.get_track_trajectory(
            sample_track,
            start_time=10.0,
            end_time=min(float(n_steps - 1), 50.0),
            scenario_id="mission_001",
        )
        traj_elapsed = time.perf_counter() - t_start

        t_start = time.perf_counter()
        state_at = h5.get_state_at_time(
            sample_track,
            time=25.0,
            interpolate=True,
            scenario_id="mission_001",
        )
        point_elapsed = time.perf_counter() - t_start

        t_start = time.perf_counter()
        region_tracks = h5.get_tracks_in_region(
            bbox=[0, 0, 100, 100],
            time_range=[0, float(n_steps)],
            scenario_id="mission_001",
        )
        region_elapsed = time.perf_counter() - t_start

        print(
            f"    Trajectory query ({len(traj['timestamps'])} states): "
            f"{traj_elapsed * 1000:.1f} ms"
        )
        print(f"    Point query (t=25.0): {point_elapsed * 1000:.1f} ms")
        print(
            f"    Region query: {len(region_tracks)} tracks in "
            f"{region_elapsed * 1000:.1f} ms"
        )

    # Phase 4: Round-trip verification
    print("  Phase 4: Round-trip verification...")
    sql2_path = str(output_dir / "roundtrip.db")
    db2 = TrackDatabaseManager(sql2_path)
    db2.open(mode="w")

    h5.export_to_sql(db2, scenario_id="mission_001")

    n_rt_tracks = len(db2.retrieve_all_tracks())
    match = n_rt_tracks == n_sql_tracks
    print(f"    Round-trip tracks: {n_rt_tracks} (match: {match})")

    db2.close()
    h5.close()
    db.close()

    # File sizes
    sql_size = os.path.getsize(sql_path)
    h5_size = os.path.getsize(h5_path)
    print(f"\n  File sizes:")
    print(f"    SQLite: {sql_size / 1024:.1f} KB")
    print(f"    HDF5:   {h5_size / 1024:.1f} KB")


# =============================================================================
# Workflow 4: Track Comparison (GNN vs MHT)
# =============================================================================


def workflow_track_comparison(
    output_dir: Path,
    n_steps: int = 50,
) -> None:
    """Compare GNN and MHT trackers using HDF5 scenario storage.

    Demonstrates:
    - Running two trackers on identical measurements
    - Storing each result as a separate HDF5 scenario
    - Using compare_scenarios() for quantitative comparison
    """
    if not HAS_H5PY:
        print("\n  [SKIP] Workflow 4 requires h5py")
        return

    print("\n" + "=" * 60)
    print("Workflow 4: Track Comparison (GNN vs MHT)")
    print("=" * 60)

    params = setup_cv_tracker_params()
    true_states, measurements = simulate_radar_targets(
        n_targets=2,
        n_steps=n_steps,
        noise_std=5.0,
        pd=0.9,
        fa_rate=0.1,
        seed=42,
    )

    # Create GNN tracker
    gnn_tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=params["F"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=params["P0"],
    )

    # Create MHT tracker (conservative settings for tractable computation)
    mht_config = MHTConfig(
        n_scan=2,
        max_hypotheses=15,
        detection_prob=0.9,
        clutter_density=1e-4,
        gate_probability=0.99,
        confirm_threshold=3,
        delete_threshold=5,
    )
    mht_tracker = MHTTracker(
        state_dim=4,
        meas_dim=2,
        F=params["F"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        config=mht_config,
        init_covariance=params["P0"],
    )

    # Run both trackers
    print("\n  Running GNN tracker...")
    gnn_tracks_history: Dict[int, Dict[str, List]] = {}
    for k in range(n_steps):
        tracks = gnn_tracker.process(measurements[k], dt=1.0)
        for track in tracks:
            if track.status == TrackStatus.CONFIRMED:
                if track.id not in gnn_tracks_history:
                    gnn_tracks_history[track.id] = {
                        "states": [],
                        "covariances": [],
                        "timestamps": [],
                    }
                gnn_tracks_history[track.id]["states"].append(track.state)
                gnn_tracks_history[track.id]["covariances"].append(track.covariance)
                gnn_tracks_history[track.id]["timestamps"].append(float(k))

    print(f"    GNN: {len(gnn_tracks_history)} confirmed tracks")

    print("  Running MHT tracker...")
    mht_tracks_history: Dict[int, Dict[str, List]] = {}
    for k in range(n_steps):
        result = mht_tracker.process(measurements[k], dt=1.0)
        for track in result.confirmed_tracks:
            if track.id not in mht_tracks_history:
                mht_tracks_history[track.id] = {
                    "states": [],
                    "covariances": [],
                    "timestamps": [],
                }
            mht_tracks_history[track.id]["states"].append(track.state)
            mht_tracks_history[track.id]["covariances"].append(track.covariance)
            mht_tracks_history[track.id]["timestamps"].append(float(k))

    print(f"    MHT: {len(mht_tracks_history)} confirmed tracks")

    # Store as HDF5 scenarios
    h5_path = str(output_dir / "tracker_comparison.h5")
    h5 = TrackHDF5Storage(h5_path)
    h5.open(mode="w")

    def _store_track_dict(
        storage: Any,
        tracks_dict: Dict[int, Dict[str, List]],
        scenario_id: str,
    ) -> None:
        track_data = {}
        for tid, data in tracks_dict.items():
            track_data[f"trk_{tid}"] = {
                "states": np.array(data["states"]),
                "covariances": np.array(data["covariances"]),
                "timestamps": np.array(data["timestamps"]),
            }
        storage.store_tracking_scenario(scenario_id, track_data)

    _store_track_dict(h5, gnn_tracks_history, "gnn_tracking")
    _store_track_dict(h5, mht_tracks_history, "mht_tracking")

    # Compare scenarios
    print("\n  Comparing scenarios...")
    comparison = h5.compare_scenarios("gnn_tracking", "mht_tracking")
    print(f"    Common tracks: {comparison.get('common_tracks', 'N/A')}")
    print(f"    GNN-only tracks: {comparison.get('only_in_first', 'N/A')}")
    print(f"    MHT-only tracks: {comparison.get('only_in_second', 'N/A')}")

    # Compute position RMSE against ground truth for each tracker
    def _compute_rmse(tracks_dict: Dict[int, Dict[str, List]]) -> float:
        errors = []
        for data in tracks_dict.values():
            for i, t in enumerate(data["timestamps"]):
                k_idx = int(t)
                if k_idx < len(true_states):
                    pos_est = np.array([data["states"][i][0], data["states"][i][2]])
                    best_err = float("inf")
                    for tgt in range(true_states[k_idx].shape[0]):
                        pos_true = np.array(
                            [
                                true_states[k_idx][tgt, 0],
                                true_states[k_idx][tgt, 2],
                            ]
                        )
                        best_err = min(best_err, np.linalg.norm(pos_est - pos_true))
                    errors.append(best_err)
        return float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else 0.0

    gnn_rmse = _compute_rmse(gnn_tracks_history)
    mht_rmse = _compute_rmse(mht_tracks_history)

    print(f"\n  Performance Summary:")
    print(f"    {'Tracker':<10} {'Confirmed':<12} {'RMSE (m)':<10}")
    print(f"    {'GNN':<10} {len(gnn_tracks_history):<12} {gnn_rmse:<10.2f}")
    print(f"    {'MHT':<10} {len(mht_tracks_history):<12} {mht_rmse:<10.2f}")

    h5.close()


# =============================================================================
# Workflow 5: Scenario Replay and Validation
# =============================================================================


def workflow_scenario_replay(
    output_dir: Path,
    n_steps: int = 50,
) -> None:
    """Store, retrieve, and validate tracking scenarios.

    Demonstrates:
    - Storing ground truth and tracker output as separate scenarios
    - Closing and reopening HDF5 for later analysis
    - Scenario listing and retrieval
    - Export to SQL and conversion to TrackList
    """
    if not HAS_H5PY:
        print("\n  [SKIP] Workflow 5 requires h5py")
        return

    print("\n" + "=" * 60)
    print("Workflow 5: Scenario Replay and Validation")
    print("=" * 60)

    params = setup_cv_tracker_params()
    true_states, measurements = simulate_radar_targets(
        n_targets=3,
        n_steps=n_steps,
        seed=42,
    )

    # Run tracker
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=params["F"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=params["P0"],
    )

    tracker_output: Dict[int, Dict[str, List]] = {}
    for k in range(n_steps):
        tracks = tracker.process(measurements[k], dt=1.0)
        for track in tracks:
            if track.status == TrackStatus.CONFIRMED:
                if track.id not in tracker_output:
                    tracker_output[track.id] = {
                        "states": [],
                        "covariances": [],
                        "timestamps": [],
                    }
                tracker_output[track.id]["states"].append(track.state)
                tracker_output[track.id]["covariances"].append(track.covariance)
                tracker_output[track.id]["timestamps"].append(float(k))

    # Store ground truth as scenario
    h5_path = str(output_dir / "scenario_replay.h5")
    h5 = TrackHDF5Storage(h5_path)
    h5.open(mode="w")

    # Ground truth tracks
    truth_tracks = {}
    for tgt in range(min(3, true_states[0].shape[0])):
        states = np.array([ts[tgt] for ts in true_states])
        covs = np.zeros((n_steps, 4, 4))  # perfect knowledge
        timestamps = np.arange(n_steps, dtype=np.float64)
        truth_tracks[f"truth_{tgt}"] = {
            "states": states,
            "covariances": covs,
            "timestamps": timestamps,
        }

    h5.store_tracking_scenario(
        "ground_truth",
        truth_tracks,
        metadata={"type": "truth", "n_targets": 3},
    )

    # Tracker output
    result_tracks = {}
    for tid, data in tracker_output.items():
        result_tracks[f"trk_{tid}"] = {
            "states": np.array(data["states"]),
            "covariances": np.array(data["covariances"]),
            "timestamps": np.array(data["timestamps"]),
        }

    h5.store_tracking_scenario(
        "tracker_output",
        result_tracks,
        metadata={"type": "result", "algorithm": "GNN"},
    )
    h5.close()

    # Simulate later analysis: reopen the file
    print("\n  Reopening HDF5 for analysis...")
    h5 = TrackHDF5Storage(h5_path)
    h5.open(mode="r")

    scenarios = h5.list_scenarios()
    print(f"  Available scenarios: {scenarios}")

    truth_data = h5.retrieve_tracking_scenario("ground_truth")
    result_data = h5.retrieve_tracking_scenario("tracker_output")

    truth_track_ids = list(truth_data["tracks"].keys())
    result_track_ids = list(result_data["tracks"].keys())
    print(f"  Ground truth tracks: {len(truth_track_ids)}")
    print(f"  Tracker output tracks: {len(result_track_ids)}")

    # Validate: compute per-track RMSE against ground truth
    print("\n  Validation (per-track RMSE):")
    for r_tid, r_data in result_data["tracks"].items():
        r_states = r_data["states"]
        r_times = r_data["timestamps"]
        best_rmse = float("inf")
        best_truth = ""
        for t_tid, t_data in truth_data["tracks"].items():
            t_states = t_data["states"]
            t_times = t_data["timestamps"]
            errors = []
            for i, t_val in enumerate(r_times):
                t_idx = np.searchsorted(t_times, t_val)
                if t_idx < len(t_times) and abs(t_times[t_idx] - t_val) < 0.5:
                    pos_r = np.array([r_states[i][0], r_states[i][2]])
                    pos_t = np.array([t_states[t_idx][0], t_states[t_idx][2]])
                    errors.append(np.linalg.norm(pos_r - pos_t))
            if errors:
                rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_truth = t_tid
        if best_rmse < float("inf"):
            completeness = len(r_times) / n_steps * 100
            print(
                f"    {r_tid} → {best_truth}: "
                f"RMSE={best_rmse:.2f} m, coverage={completeness:.0f}%"
            )

    h5.close()

    # Export to SQL and convert to TrackList
    print("\n  Exporting to SQL and converting to TrackList...")
    h5 = TrackHDF5Storage(h5_path)
    h5.open(mode="r")
    sql_path = str(output_dir / "replay_export.db")
    db = TrackDatabaseManager(sql_path)
    db.open(mode="w")
    h5.export_to_sql(db, scenario_id="tracker_output")

    track_list = db.tracks_to_tracklist()
    print(f"  TrackList: {len(track_list)} tracks")
    if len(track_list) > 0:
        stats = track_list.stats()
        print(
            f"  Stats: {stats.n_confirmed} confirmed, " f"{stats.n_tentative} tentative"
        )

    db.close()
    h5.close()


# =============================================================================
# Workflow 6: Detection Clutter Handling
# =============================================================================


def workflow_clutter_handling(
    output_dir: Path,
    n_steps: int = 50,
) -> None:
    """Detection management in heavy clutter scenarios.

    Demonstrates:
    - Storing all detections (real + clutter) in SQL
    - Gating and association to identify true measurements
    - Marking detections as associated vs unassociated
    - Monitoring the initiation queue
    - Pruning old detections at varying thresholds
    """
    print("\n" + "=" * 60)
    print("Workflow 6: Detection Clutter Handling")
    print("=" * 60)

    params = setup_cv_tracker_params()
    true_states, measurements = simulate_radar_targets(
        n_targets=2,
        n_steps=n_steps,
        noise_std=5.0,
        pd=0.9,
        fa_rate=5.0,  # Heavy clutter
        seed=42,
    )

    db_path = str(output_dir / "clutter.db")
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=params["F"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=params["P0"],
    )

    db = TrackDatabaseManager(db_path)
    db.open(mode="w")

    total_dets = 0
    n_associated = 0
    queue_depths: List[int] = []

    for k in range(n_steps):
        t = float(k)
        meas_k = measurements[k]

        # Store all detections
        det_ids = []
        for j, z in enumerate(meas_k):
            det_id = f"det_{k:04d}_{j:02d}"
            db.store_detection(det_id, z, "radar", t)
            det_ids.append(det_id)
            total_dets += 1

        # Run tracker to get associations
        tracks = tracker.process(meas_k, dt=1.0)

        # Associate detections with tracks based on tracker results
        # The tracker internally handles association; we record which
        # detections correspond to confirmed tracks
        for track in tracks:
            if track.status == TrackStatus.CONFIRMED:
                db.store_from_track(track, timestamp=t)
                # Find closest detection to this track's measurement
                if meas_k:
                    pos_track = np.array([track.state[0], track.state[2]])
                    dists = [np.linalg.norm(z - pos_track) for z in meas_k]
                    best_j = int(np.argmin(dists))
                    if dists[best_j] < 20.0 and best_j < len(det_ids):
                        track_id = f"trk_{track.id}"
                        try:
                            db.associate_detection(det_ids[best_j], track_id)
                            n_associated += 1
                        except (KeyError, Exception):
                            pass

        # Monitor initiation queue
        queue = db.get_initiation_queue(max_age=5.0)
        queue_depths.append(len(queue))

    # Final statistics
    all_dets = db.retrieve_all_detections()
    associated_dets = db.retrieve_detections(association_status="associated")
    unassociated_dets = db.retrieve_detections(association_status="unassociated")

    # Prune old detections
    pruned_5s = db.prune_old_detections(age_threshold=5.0)
    remaining = len(db.retrieve_all_detections())
    pruned_1s = db.prune_old_detections(age_threshold=1.0)
    final_remaining = len(db.retrieve_all_detections())

    all_tracks_info = db.retrieve_all_tracks()
    confirmed = db.retrieve_all_tracks(TrackDatabaseStatus.CONFIRMED)
    db.close()

    assoc_rate = n_associated / total_dets * 100 if total_dets > 0 else 0
    avg_queue = float(np.mean(queue_depths)) if queue_depths else 0

    print(f"\n  Results:")
    print(f"    Total detections: {total_dets}")
    print(f"    Associated: {len(associated_dets)} ({assoc_rate:.1f}%)")
    print(f"    Unassociated: {len(unassociated_dets)}")
    print(
        f"    Tracks initiated: {len(all_tracks_info)}, " f"confirmed: {len(confirmed)}"
    )
    print(f"    Avg initiation queue depth: {avg_queue:.1f}")
    print(f"\n  Pruning results:")
    print(f"    Pruned (5s threshold): {pruned_5s} → {remaining} remaining")
    print(f"    Pruned (1s threshold): {pruned_1s} → {final_remaining} remaining")


# =============================================================================
# Plot Helpers
# =============================================================================


def plot_radar_tracking(
    true_states: List[np.ndarray],
    measurements: List[List[np.ndarray]],
    db_path: str,
) -> None:
    """Plot radar tracking results from database."""
    if not HAS_PLOTLY:
        return

    fig = go.Figure()

    # Plot true trajectories
    n_targets = true_states[0].shape[0]
    colors = ["green", "blue", "red", "purple"]
    for tgt in range(n_targets):
        xs = [ts[tgt, 0] for ts in true_states]
        ys = [ts[tgt, 2] for ts in true_states]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=colors[tgt % len(colors)], width=2),
                name=f"Truth {tgt}",
            )
        )

    # Plot measurements
    meas_x, meas_y = [], []
    for meas_k in measurements:
        for z in meas_k:
            meas_x.append(z[0])
            meas_y.append(z[1])
    fig.add_trace(
        go.Scatter(
            x=meas_x,
            y=meas_y,
            mode="markers",
            marker=dict(color="gray", size=3, opacity=0.3),
            name="Detections",
        )
    )

    fig.update_layout(
        title="RADAR Tracking Pipeline",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        width=900,
        height=600,
    )
    path = OUTPUT_DIR / "track_management_radar.html"
    fig.write_html(str(path))
    print(f"  Plot saved: {path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all track management workflow examples."""
    print("Track Management Workflow Examples")
    print("=" * 60)

    work_dir = Path(tempfile.mkdtemp(prefix="pytcl_workflows_"))
    print(f"Working directory: {work_dir}")

    # Workflow 1: SQL-only
    workflow_radar_tracking(work_dir, n_steps=50)

    # Workflow 2: SQL-only
    workflow_multi_sensor_fusion(work_dir, n_steps=50)

    # Workflow 3: SQL + HDF5
    workflow_realtime_to_archive(work_dir, n_steps=100)

    # Workflow 4: HDF5 scenarios
    workflow_track_comparison(work_dir, n_steps=50)

    # Workflow 5: HDF5 + SQL
    workflow_scenario_replay(work_dir, n_steps=50)

    # Workflow 6: SQL-only
    workflow_clutter_handling(work_dir, n_steps=50)

    print("\n" + "=" * 60)
    print("All workflows complete!")
    print(f"Output files in: {work_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
