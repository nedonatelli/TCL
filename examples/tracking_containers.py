"""
Tracking Containers Example
===========================

This example demonstrates the tracking container classes in PyTCL:
- TrackList: Collection of tracks with filtering and batch operations
- MeasurementSet: Time-indexed measurements with spatial queries
- ClusterSet: Track clustering for formation detection

These containers provide efficient data management for multi-target tracking
applications with immutable design patterns and lazy spatial indexing.
"""

import numpy as np

from pytcl.containers import (
    ClusterSet,
    MeasurementSet,
    TrackList,
)
from pytcl.containers.cluster_set import cluster_tracks_dbscan, cluster_tracks_kmeans
from pytcl.containers.measurement_set import Measurement
from pytcl.containers.track_list import Track, TrackStatus


def create_sample_tracks(n_tracks: int = 10, seed: int = 42) -> TrackList:
    """Create sample tracks for demonstration."""
    rng = np.random.default_rng(seed)

    tracks = []
    for i in range(n_tracks):
        # State: [x, vx, y, vy] - 2D position and velocity
        x = rng.uniform(-100, 100)
        y = rng.uniform(-100, 100)
        vx = rng.uniform(-5, 5)
        vy = rng.uniform(-5, 5)
        state = np.array([x, vx, y, vy])

        # Covariance matrix
        pos_var = rng.uniform(1, 5)
        vel_var = rng.uniform(0.1, 0.5)
        P = np.diag([pos_var, vel_var, pos_var, vel_var])

        # Random status based on hits
        hits = rng.integers(1, 20)
        misses = rng.integers(0, 5)
        if hits >= 5:
            status = TrackStatus.CONFIRMED
        elif misses >= 3:
            status = TrackStatus.DELETED
        else:
            status = TrackStatus.TENTATIVE

        track = Track(
            id=i,
            state=state,
            covariance=P,
            status=status,
            hits=hits,
            misses=misses,
            time=10.0 + rng.uniform(0, 5),
        )
        tracks.append(track)

    return TrackList(tracks)


def create_sample_measurements(n_times: int = 5, n_per_time: int = 8, seed: int = 42):
    """Create sample measurements across multiple time steps."""
    rng = np.random.default_rng(seed)

    measurements = []
    meas_id = 0
    for t in range(n_times):
        time = float(t)
        for _ in range(n_per_time):
            # 2D position measurement
            value = rng.uniform(-50, 50, size=2)
            covariance = np.eye(2) * rng.uniform(0.5, 2.0)
            sensor_id = rng.integers(0, 3)  # 3 sensors

            meas = Measurement(
                value=value,
                time=time,
                covariance=covariance,
                sensor_id=sensor_id,
                id=meas_id,
            )
            measurements.append(meas)
            meas_id += 1

    return MeasurementSet(measurements)


def demo_track_list():
    """Demonstrate TrackList container operations."""
    print("=" * 70)
    print("TrackList Container Demo")
    print("=" * 70)

    # Create sample tracks
    tracks = create_sample_tracks(n_tracks=15)
    print(f"\nCreated TrackList with {len(tracks)} tracks")

    # Get statistics
    stats = tracks.stats()
    print("\nTrack Statistics:")
    print(f"  Total tracks: {stats.n_tracks}")
    print(f"  Confirmed: {stats.n_confirmed}")
    print(f"  Tentative: {stats.n_tentative}")
    print(f"  Deleted: {stats.n_deleted}")
    print(f"  Mean hits: {stats.mean_hits:.1f}")
    print(f"  Mean misses: {stats.mean_misses:.1f}")

    # Filter by status
    confirmed = tracks.filter_by_status(TrackStatus.CONFIRMED)
    tentative = tracks.filter_by_status(TrackStatus.TENTATIVE)
    print("\nFiltered by status:")
    print(f"  Confirmed tracks: {len(confirmed)}")
    print(f"  Tentative tracks: {len(tentative)}")

    # Shortcut properties
    print(f"  Using .confirmed property: {len(tracks.confirmed)}")
    print(f"  Using .tentative property: {len(tracks.tentative)}")

    # Filter by region (tracks near origin)
    center = np.array([0.0, 0.0])
    nearby = tracks.filter_by_region(center, radius=50.0, state_indices=(0, 2))
    print(f"\nTracks within 50 units of origin: {len(nearby)}")

    # Filter by time
    recent = tracks.filter_by_time(min_time=12.0)
    print(f"Tracks updated after t=12.0: {len(recent)}")

    # Custom predicate filter
    high_confidence = tracks.filter_by_predicate(lambda t: t.hits >= 10)
    print(f"Tracks with 10+ hits: {len(high_confidence)}")

    # Batch data extraction
    if len(confirmed) > 0:
        states = confirmed.states()
        positions = confirmed.positions(indices=(0, 2))
        print("\nBatch extraction from confirmed tracks:")
        print(f"  States shape: {states.shape}")
        print(f"  Positions shape: {positions.shape}")

    # Access by ID
    track_ids = tracks.track_ids
    if track_ids:
        track = tracks.get_by_id(track_ids[0])
        print(f"\nTrack {track.id}:")
        print(f"  Position: ({track.state[0]:.1f}, {track.state[2]:.1f})")
        print(f"  Velocity: ({track.state[1]:.1f}, {track.state[3]:.1f})")
        print(f"  Status: {track.status.name}")

    # Immutable operations
    new_track = Track(
        id=100,
        state=np.array([0, 0, 0, 0]),
        covariance=np.eye(4),
        status=TrackStatus.TENTATIVE,
        hits=1,
        misses=0,
        time=15.0,
    )
    tracks_with_new = tracks.add(new_track)
    print("\nAfter adding track:")
    print(f"  Original TrackList: {len(tracks)} tracks")
    print(f"  New TrackList: {len(tracks_with_new)} tracks")

    # Merge two track lists
    merged = confirmed.merge(tentative)
    print(f"\nMerged confirmed + tentative: {len(merged)} tracks")


def demo_measurement_set():
    """Demonstrate MeasurementSet container operations."""
    print("\n" + "=" * 70)
    print("MeasurementSet Container Demo")
    print("=" * 70)

    # Create sample measurements
    meas_set = create_sample_measurements(n_times=5, n_per_time=8)
    print(f"\nCreated MeasurementSet with {len(meas_set)} measurements")

    # Time properties
    times = meas_set.times
    time_range = meas_set.time_range
    print("\nTime information:")
    print(f"  Unique times: {times}")
    print(f"  Time range: {time_range}")

    # Query by time
    at_t2 = meas_set.at_time(2.0)
    print(f"\nMeasurements at t=2.0: {len(at_t2)}")

    # Query time window
    window = meas_set.in_time_window(1.0, 3.0)
    print(f"Measurements in window [1.0, 3.0]: {len(window)}")

    # Query by sensor
    sensors = meas_set.sensors
    print(f"\nSensors: {sensors}")
    for sensor_id in sensors:
        sensor_meas = meas_set.by_sensor(sensor_id)
        print(f"  Sensor {sensor_id}: {len(sensor_meas)} measurements")

    # Spatial queries
    center = np.array([0.0, 0.0])
    nearby = meas_set.in_region(center, radius=25.0)
    print(f"\nMeasurements within 25 units of origin: {len(nearby)}")

    # K-nearest neighbors
    query_point = np.array([10.0, 10.0])
    nearest = meas_set.nearest_to(query_point, k=3)
    print("\n3 nearest measurements to (10, 10):")
    for meas in nearest.measurements:
        dist = np.linalg.norm(meas.value - query_point)
        print(f"  ID {meas.id}: value={meas.value}, distance={dist:.2f}")

    # Batch extraction
    values = meas_set.values()
    print("\nBatch extraction:")
    print(f"  All values shape: {values.shape}")

    values_at_t1 = meas_set.values_at_time(1.0)
    print(f"  Values at t=1.0 shape: {values_at_t1.shape}")

    # Create from arrays
    new_values = np.random.randn(5, 2) * 10
    new_times = np.array([10.0, 10.0, 10.1, 10.1, 10.2])
    meas_from_arrays = MeasurementSet.from_arrays(new_values, new_times)
    print(f"\nCreated from arrays: {len(meas_from_arrays)} measurements")


def demo_cluster_set():
    """Demonstrate ClusterSet container operations."""
    print("\n" + "=" * 70)
    print("ClusterSet Container Demo")
    print("=" * 70)

    # Create tracks with some spatial clustering
    rng = np.random.default_rng(42)
    tracks = []

    # Cluster 1: tracks near (50, 50)
    for i in range(5):
        state = np.array(
            [
                50 + rng.normal(0, 3),  # x
                2 + rng.normal(0, 0.5),  # vx
                50 + rng.normal(0, 3),  # y
                1 + rng.normal(0, 0.5),  # vy
            ]
        )
        tracks.append(
            Track(
                id=i,
                state=state,
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=10,
                misses=0,
                time=0.0,
            )
        )

    # Cluster 2: tracks near (-30, -30)
    for i in range(4):
        state = np.array(
            [
                -30 + rng.normal(0, 3),
                -1 + rng.normal(0, 0.5),
                -30 + rng.normal(0, 3),
                2 + rng.normal(0, 0.5),
            ]
        )
        tracks.append(
            Track(
                id=5 + i,
                state=state,
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=8,
                misses=1,
                time=0.0,
            )
        )

    # Isolated tracks (noise)
    for i in range(3):
        state = np.array(
            [
                rng.uniform(-100, 100),
                rng.uniform(-3, 3),
                rng.uniform(-100, 100),
                rng.uniform(-3, 3),
            ]
        )
        tracks.append(
            Track(
                id=9 + i,
                state=state,
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=5,
                misses=2,
                time=0.0,
            )
        )

    track_list = TrackList(tracks)
    print(f"\nCreated {len(track_list)} tracks with 2 clusters + noise")

    # DBSCAN clustering
    print("\n--- DBSCAN Clustering ---")
    clusters_dbscan = cluster_tracks_dbscan(
        track_list,
        eps=10.0,  # Max distance between neighbors
        min_samples=3,  # Minimum cluster size
        state_indices=(0, 2),  # Use x, y positions
    )
    print(f"Found {len(clusters_dbscan)} clusters")

    for cluster in clusters_dbscan:
        print(f"\n  Cluster {cluster.id}:")
        print(f"    Track IDs: {cluster.track_ids}")
        print(f"    Centroid: ({cluster.centroid[0]:.1f}, {cluster.centroid[1]:.1f})")
        print(f"    Covariance diagonal: {np.diag(cluster.covariance)}")

    # Cluster statistics
    print("\n--- Cluster Statistics ---")
    all_stats = clusters_dbscan.all_stats(
        tracks=track_list,
        state_indices=(0, 2),
        velocity_indices=(1, 3),
    )
    for cluster_id, stats in all_stats.items():
        print(f"\n  Cluster {cluster_id}:")
        print(f"    Tracks: {stats.n_tracks}")
        print(f"    Mean separation: {stats.mean_separation:.2f}")
        print(f"    Max separation: {stats.max_separation:.2f}")
        print(f"    Velocity coherence: {stats.velocity_coherence:.2f}")

    # K-means clustering
    print("\n--- K-Means Clustering ---")
    clusters_kmeans = cluster_tracks_kmeans(
        track_list,
        n_clusters=3,
        state_indices=(0, 2),
        rng=np.random.default_rng(42),
    )
    print(f"Created {len(clusters_kmeans)} clusters")

    for cluster in clusters_kmeans:
        print(
            f"  Cluster {cluster.id}: {len(cluster.track_ids)} tracks at "
            f"({cluster.centroid[0]:.1f}, {cluster.centroid[1]:.1f})"
        )

    # Using ClusterSet.from_tracks factory
    print("\n--- Factory Method ---")
    clusters = ClusterSet.from_tracks(
        track_list,
        method="dbscan",
        eps=10.0,
        min_samples=2,
    )
    print(f"Created ClusterSet with {len(clusters)} clusters")

    # Spatial query on clusters
    center = np.array([50.0, 50.0])
    nearby_clusters = clusters.clusters_in_region(center, radius=30.0)
    print(f"\nClusters within 30 units of (50, 50): {len(nearby_clusters)}")

    # Track to cluster lookup
    if len(clusters) > 0:
        track_id = 0
        cluster = clusters.get_cluster_for_track(track_id)
        if cluster:
            print(f"Track {track_id} belongs to cluster {cluster.id}")

    # Cluster manipulation (immutable)
    if len(clusters) >= 2:
        cluster_ids = clusters.cluster_ids
        merged = clusters.merge_clusters(cluster_ids[0], cluster_ids[1])
        print(f"\nAfter merging clusters {cluster_ids[0]} and {cluster_ids[1]}:")
        print(f"  Original: {len(clusters)} clusters")
        print(f"  After merge: {len(merged)} clusters")


def demo_integration():
    """Demonstrate integration between containers."""
    print("\n" + "=" * 70)
    print("Container Integration Demo")
    print("=" * 70)

    # Create tracks and measurements
    tracks = create_sample_tracks(n_tracks=20)
    measurements = create_sample_measurements(n_times=10, n_per_time=15)

    print(f"\nDataset: {len(tracks)} tracks, {len(measurements)} measurements")

    # Filter to confirmed tracks
    confirmed = tracks.confirmed
    print(f"\nConfirmed tracks: {len(confirmed)}")

    # For each confirmed track, find nearby measurements
    print("\nMatching tracks to nearby measurements:")
    for track in list(confirmed)[:3]:  # Show first 3
        pos = track.state[[0, 2]]  # x, y position
        nearby_meas = measurements.in_region(pos, radius=20.0)
        print(
            f"  Track {track.id} at ({pos[0]:.1f}, {pos[1]:.1f}): "
            f"{len(nearby_meas)} nearby measurements"
        )

    # Cluster confirmed tracks
    if len(confirmed) >= 3:
        clusters = ClusterSet.from_tracks(
            confirmed,
            method="dbscan",
            eps=50.0,
            min_samples=2,
        )
        print(f"\nClustered confirmed tracks: {len(clusters)} formations")

        # For each cluster, find measurements near centroid
        for cluster in clusters:
            nearby = measurements.in_region(cluster.centroid, radius=30.0)
            print(
                f"  Cluster {cluster.id} ({len(cluster.track_ids)} tracks): "
                f"{len(nearby)} measurements near centroid"
            )

    # Time-synchronized analysis
    print("\n--- Time-Synchronized Analysis ---")
    for t in [0.0, 2.0, 4.0]:
        meas_at_t = measurements.at_time(t)
        tracks_at_t = tracks.filter_by_time(max_time=t + 1.0)
        print(
            f"  t={t}: {len(meas_at_t)} measurements, "
            f"{len(tracks_at_t)} tracks updated before t={t + 1}"
        )


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Tracking Containers Example")
    print("#" * 70)

    demo_track_list()
    demo_measurement_set()
    demo_cluster_set()
    demo_integration()

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
