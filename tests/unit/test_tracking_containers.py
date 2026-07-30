"""Tests for tracking container classes."""

import numpy as np
import pytest

from pytcl.containers import (
    ClusterSet,
    ClusterStats,
    Measurement,
    MeasurementQuery,
    MeasurementSet,
    TrackCluster,
    TrackList,
    TrackListStats,
    cluster_tracks_dbscan,
    cluster_tracks_kmeans,
    compute_cluster_centroid,
)
from pytcl.trackers.multi_target import Track, TrackStatus

# =============================================================================
# TrackList Tests
# =============================================================================


class TestTrackList:
    """Tests for TrackList container."""

    @pytest.fixture
    def sample_tracks(self):
        """Create sample tracks for testing."""
        return [
            Track(
                id=0,
                state=np.array([0.0, 1.0, 0.0, 1.0]),
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=5,
                misses=0,
                time=1.0,
            ),
            Track(
                id=1,
                state=np.array([5.0, 0.5, 5.0, 0.5]),
                covariance=np.eye(4) * 2,
                status=TrackStatus.TENTATIVE,
                hits=2,
                misses=1,
                time=1.0,
            ),
            Track(
                id=2,
                state=np.array([10.0, -0.5, 10.0, -0.5]),
                covariance=np.eye(4) * 0.5,
                status=TrackStatus.CONFIRMED,
                hits=8,
                misses=2,
                time=2.0,
            ),
        ]

    def test_construction_empty(self):
        """Test empty TrackList construction."""
        tl = TrackList()
        assert len(tl) == 0
        assert list(tl) == []

    def test_construction_from_list(self, sample_tracks):
        """Test TrackList construction from list."""
        tl = TrackList(sample_tracks)
        assert len(tl) == 3
        assert list(tl) == sample_tracks

    def test_getitem_index(self, sample_tracks):
        """Test indexing TrackList."""
        tl = TrackList(sample_tracks)
        assert tl[0] == sample_tracks[0]
        assert tl[1] == sample_tracks[1]
        assert tl[-1] == sample_tracks[-1]

    def test_getitem_slice(self, sample_tracks):
        """Test slicing TrackList."""
        tl = TrackList(sample_tracks)
        sliced = tl[1:]
        assert isinstance(sliced, TrackList)
        assert len(sliced) == 2

    def test_contains(self, sample_tracks):
        """Test contains operator."""
        tl = TrackList(sample_tracks)
        assert 0 in tl
        assert 1 in tl
        assert 99 not in tl

    def test_get_by_id(self, sample_tracks):
        """Test get_by_id method."""
        tl = TrackList(sample_tracks)
        assert tl.get_by_id(0) == sample_tracks[0]
        assert tl.get_by_id(1) == sample_tracks[1]
        assert tl.get_by_id(99) is None

    def test_get_by_ids(self, sample_tracks):
        """Test get_by_ids method."""
        tl = TrackList(sample_tracks)
        result = tl.get_by_ids([0, 2])
        assert len(result) == 2
        assert result[0].id == 0
        assert result[1].id == 2

    def test_filter_by_status(self, sample_tracks):
        """Test filtering by status."""
        tl = TrackList(sample_tracks)

        confirmed = tl.filter_by_status(TrackStatus.CONFIRMED)
        assert len(confirmed) == 2
        assert all(t.status == TrackStatus.CONFIRMED for t in confirmed)

        tentative = tl.filter_by_status(TrackStatus.TENTATIVE)
        assert len(tentative) == 1

    def test_filter_by_time(self, sample_tracks):
        """Test filtering by time range."""
        tl = TrackList(sample_tracks)

        result = tl.filter_by_time(min_time=1.5)
        assert len(result) == 1
        assert result[0].time == 2.0

        result = tl.filter_by_time(max_time=1.5)
        assert len(result) == 2

        result = tl.filter_by_time(min_time=1.0, max_time=1.5)
        assert len(result) == 2

    def test_filter_by_region(self, sample_tracks):
        """Test filtering by spatial region."""
        tl = TrackList(sample_tracks)

        # Find tracks near origin
        result = tl.filter_by_region(center=[0, 0], radius=2.0)
        assert len(result) == 1
        assert result[0].id == 0

        # Find tracks near (5, 5)
        result = tl.filter_by_region(center=[5, 5], radius=2.0)
        assert len(result) == 1
        assert result[0].id == 1

    def test_filter_by_predicate(self, sample_tracks):
        """Test filtering by custom predicate."""
        tl = TrackList(sample_tracks)

        # Filter by hits > 3
        result = tl.filter_by_predicate(lambda t: t.hits > 3)
        assert len(result) == 2
        assert all(t.hits > 3 for t in result)

    def test_confirmed_property(self, sample_tracks):
        """Test confirmed property."""
        tl = TrackList(sample_tracks)
        confirmed = tl.confirmed
        assert len(confirmed) == 2

    def test_tentative_property(self, sample_tracks):
        """Test tentative property."""
        tl = TrackList(sample_tracks)
        tentative = tl.tentative
        assert len(tentative) == 1

    def test_track_ids_property(self, sample_tracks):
        """Test track_ids property."""
        tl = TrackList(sample_tracks)
        assert tl.track_ids == [0, 1, 2]

    def test_states_extraction(self, sample_tracks):
        """Test batch state extraction."""
        tl = TrackList(sample_tracks)
        states = tl.states()
        assert states.shape == (3, 4)
        np.testing.assert_array_equal(states[0], sample_tracks[0].state)

    def test_covariances_extraction(self, sample_tracks):
        """Test batch covariance extraction."""
        tl = TrackList(sample_tracks)
        covs = tl.covariances()
        assert covs.shape == (3, 4, 4)

    def test_positions_extraction(self, sample_tracks):
        """Test position extraction."""
        tl = TrackList(sample_tracks)
        positions = tl.positions()
        assert positions.shape == (3, 2)
        np.testing.assert_array_equal(positions[0], [0.0, 0.0])
        np.testing.assert_array_equal(positions[1], [5.0, 5.0])

    def test_stats(self, sample_tracks):
        """Test statistics computation."""
        tl = TrackList(sample_tracks)
        stats = tl.stats()

        assert isinstance(stats, TrackListStats)
        assert stats.n_tracks == 3
        assert stats.n_confirmed == 2
        assert stats.n_tentative == 1
        assert stats.n_deleted == 0
        assert stats.mean_hits == 5.0  # (5 + 2 + 8) / 3
        assert stats.mean_misses == 1.0  # (0 + 1 + 2) / 3

    def test_stats_empty(self):
        """Test statistics on empty list."""
        tl = TrackList()
        stats = tl.stats()
        assert stats.n_tracks == 0
        assert stats.mean_hits == 0.0

    def test_add(self, sample_tracks):
        """Test adding a track."""
        tl = TrackList(sample_tracks[:2])
        new_tl = tl.add(sample_tracks[2])
        assert len(new_tl) == 3
        assert len(tl) == 2  # Original unchanged

    def test_remove(self, sample_tracks):
        """Test removing a track."""
        tl = TrackList(sample_tracks)
        new_tl = tl.remove(1)
        assert len(new_tl) == 2
        assert 1 not in new_tl

    def test_merge(self, sample_tracks):
        """Test merging track lists."""
        tl1 = TrackList(sample_tracks[:2])
        tl2 = TrackList(sample_tracks[2:])
        merged = tl1.merge(tl2)
        assert len(merged) == 3

    def test_copy(self, sample_tracks):
        """Test copying track list."""
        tl = TrackList(sample_tracks)
        copy = tl.copy()
        assert len(copy) == len(tl)
        assert copy[0] == tl[0]


# =============================================================================
# MeasurementSet Tests
# =============================================================================


class TestMeasurementSet:
    """Tests for MeasurementSet container."""

    @pytest.fixture
    def sample_measurements(self):
        """Create sample measurements for testing."""
        return [
            Measurement(
                value=np.array([1.0, 2.0]),
                time=0.0,
                covariance=np.eye(2) * 0.1,
                sensor_id=0,
                id=0,
            ),
            Measurement(
                value=np.array([3.0, 4.0]),
                time=0.0,
                covariance=np.eye(2) * 0.2,
                sensor_id=1,
                id=1,
            ),
            Measurement(
                value=np.array([5.0, 6.0]),
                time=1.0,
                covariance=np.eye(2) * 0.1,
                sensor_id=0,
                id=2,
            ),
            Measurement(
                value=np.array([7.0, 8.0]),
                time=2.0,
                sensor_id=0,
                id=3,
            ),
        ]

    def test_construction_empty(self):
        """Test empty MeasurementSet construction."""
        ms = MeasurementSet()
        assert len(ms) == 0

    def test_construction_from_list(self, sample_measurements):
        """Test MeasurementSet construction from list."""
        ms = MeasurementSet(sample_measurements)
        assert len(ms) == 4

    def test_from_arrays(self):
        """Test construction from arrays."""
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        times = np.array([0.0, 0.0, 1.0])
        covariances = np.array([np.eye(2)] * 3)
        sensor_ids = np.array([0, 1, 0])

        ms = MeasurementSet.from_arrays(values, times, covariances, sensor_ids)
        assert len(ms) == 3
        np.testing.assert_array_equal(ms[0].value, [1.0, 2.0])
        assert ms[1].sensor_id == 1

    def test_getitem_index(self, sample_measurements):
        """Test indexing MeasurementSet."""
        ms = MeasurementSet(sample_measurements)
        assert ms[0] == sample_measurements[0]

    def test_getitem_slice(self, sample_measurements):
        """Test slicing MeasurementSet."""
        ms = MeasurementSet(sample_measurements)
        sliced = ms[1:3]
        assert isinstance(sliced, MeasurementSet)
        assert len(sliced) == 2

    def test_at_time(self, sample_measurements):
        """Test filtering by exact time."""
        ms = MeasurementSet(sample_measurements)

        at_t0 = ms.at_time(0.0)
        assert len(at_t0) == 2

        at_t1 = ms.at_time(1.0)
        assert len(at_t1) == 1

        at_t3 = ms.at_time(3.0)
        assert len(at_t3) == 0

    def test_in_time_window(self, sample_measurements):
        """Test filtering by time window."""
        ms = MeasurementSet(sample_measurements)

        window = ms.in_time_window(0.0, 1.0)
        assert len(window) == 3

        window = ms.in_time_window(1.5, 2.5)
        assert len(window) == 1

    def test_in_region(self, sample_measurements):
        """Test filtering by spatial region."""
        ms = MeasurementSet(sample_measurements)

        # Find measurements near [1, 2]
        region = ms.in_region(center=[1.0, 2.0], radius=1.0)
        assert len(region) == 1
        assert region[0].id == 0

        # Find measurements near [4, 5]
        region = ms.in_region(center=[4.0, 5.0], radius=2.0)
        assert len(region) == 2  # [3,4] and [5,6]

    def test_by_sensor(self, sample_measurements):
        """Test filtering by sensor ID."""
        ms = MeasurementSet(sample_measurements)

        sensor0 = ms.by_sensor(0)
        assert len(sensor0) == 3

        sensor1 = ms.by_sensor(1)
        assert len(sensor1) == 1

    def test_nearest_to(self, sample_measurements):
        """Test nearest neighbor query."""
        ms = MeasurementSet(sample_measurements)

        result = ms.nearest_to([1.0, 2.0], k=1)
        assert isinstance(result, MeasurementQuery)
        assert len(result.measurements) == 1
        assert result.measurements[0].id == 0

        result = ms.nearest_to([4.0, 5.0], k=2)
        assert len(result.measurements) == 2

    def test_times_property(self, sample_measurements):
        """Test times property."""
        ms = MeasurementSet(sample_measurements)
        times = ms.times
        np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])

    def test_sensors_property(self, sample_measurements):
        """Test sensors property."""
        ms = MeasurementSet(sample_measurements)
        sensors = ms.sensors
        assert set(sensors) == {0, 1}

    def test_time_range_property(self, sample_measurements):
        """Test time_range property."""
        ms = MeasurementSet(sample_measurements)
        assert ms.time_range == (0.0, 2.0)

    def test_time_range_empty(self):
        """Test time_range on empty set."""
        ms = MeasurementSet()
        assert ms.time_range == (0.0, 0.0)

    def test_values(self, sample_measurements):
        """Test batch value extraction."""
        ms = MeasurementSet(sample_measurements)
        values = ms.values()
        assert values.shape == (4, 2)

    def test_values_at_time(self, sample_measurements):
        """Test values at specific time."""
        ms = MeasurementSet(sample_measurements)
        values = ms.values_at_time(0.0)
        assert values.shape == (2, 2)

    def test_add(self, sample_measurements):
        """Test adding a measurement."""
        ms = MeasurementSet(sample_measurements[:2])
        new_meas = Measurement(value=np.array([9.0, 10.0]), time=3.0, id=99)
        new_ms = ms.add(new_meas)
        assert len(new_ms) == 3
        assert len(ms) == 2  # Original unchanged

    def test_add_batch(self, sample_measurements):
        """Test adding multiple measurements."""
        ms = MeasurementSet(sample_measurements[:2])
        new_ms = ms.add_batch(sample_measurements[2:])
        assert len(new_ms) == 4

    def test_merge(self, sample_measurements):
        """Test merging measurement sets."""
        ms1 = MeasurementSet(sample_measurements[:2])
        ms2 = MeasurementSet(sample_measurements[2:])
        merged = ms1.merge(ms2)
        assert len(merged) == 4

    def test_copy(self, sample_measurements):
        """Test copying measurement set."""
        ms = MeasurementSet(sample_measurements)
        copy = ms.copy()
        assert len(copy) == len(ms)


# =============================================================================
# ClusterSet Tests
# =============================================================================


class TestClusterSet:
    """Tests for ClusterSet container."""

    @pytest.fixture
    def sample_tracks_for_clustering(self):
        """Create sample tracks in two distinct groups."""
        # Group 1: near origin
        tracks = [
            Track(
                id=0,
                state=np.array([0.0, 1.0, 0.0, 1.0]),
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=5,
                misses=0,
                time=1.0,
            ),
            Track(
                id=1,
                state=np.array([1.0, 1.0, 1.0, 1.0]),
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=5,
                misses=0,
                time=1.0,
            ),
            Track(
                id=2,
                state=np.array([0.5, 1.0, 0.5, 1.0]),
                covariance=np.eye(4),
                status=TrackStatus.CONFIRMED,
                hits=5,
                misses=0,
                time=1.0,
            ),
        ]
        # Group 2: near (10, 10)
        tracks.extend(
            [
                Track(
                    id=3,
                    state=np.array([10.0, -1.0, 10.0, -1.0]),
                    covariance=np.eye(4),
                    status=TrackStatus.CONFIRMED,
                    hits=5,
                    misses=0,
                    time=1.0,
                ),
                Track(
                    id=4,
                    state=np.array([11.0, -1.0, 11.0, -1.0]),
                    covariance=np.eye(4),
                    status=TrackStatus.CONFIRMED,
                    hits=5,
                    misses=0,
                    time=1.0,
                ),
            ]
        )
        return TrackList(tracks)

    @pytest.fixture
    def sample_clusters(self):
        """Create sample clusters for testing."""
        return [
            TrackCluster(
                id=0,
                track_ids=(0, 1, 2),
                centroid=np.array([0.5, 0.5]),
                covariance=np.eye(2) * 0.25,
                time=1.0,
            ),
            TrackCluster(
                id=1,
                track_ids=(3, 4),
                centroid=np.array([10.5, 10.5]),
                covariance=np.eye(2) * 0.25,
                time=1.0,
            ),
        ]

    def test_construction_empty(self):
        """Test empty ClusterSet construction."""
        cs = ClusterSet()
        assert len(cs) == 0

    def test_construction_from_list(self, sample_clusters):
        """Test ClusterSet construction from list."""
        cs = ClusterSet(sample_clusters)
        assert len(cs) == 2

    def test_from_tracks_dbscan(self, sample_tracks_for_clustering):
        """Test clustering tracks with DBSCAN."""
        cs = ClusterSet.from_tracks(
            sample_tracks_for_clustering, method="dbscan", eps=3.0, min_samples=2
        )
        # Should find 2 clusters
        assert len(cs) == 2

    def test_from_tracks_kmeans(self, sample_tracks_for_clustering):
        """Test clustering tracks with K-means."""
        cs = ClusterSet.from_tracks(
            sample_tracks_for_clustering,
            method="kmeans",
            n_clusters=2,
            rng=np.random.default_rng(42),
        )
        assert len(cs) == 2

    def test_cluster_tracks_dbscan(self, sample_tracks_for_clustering):
        """Test dbscan clustering function."""
        cs = cluster_tracks_dbscan(sample_tracks_for_clustering, eps=3.0, min_samples=2)
        assert len(cs) == 2

    def test_cluster_tracks_kmeans(self, sample_tracks_for_clustering):
        """Test kmeans clustering function."""
        cs = cluster_tracks_kmeans(
            sample_tracks_for_clustering,
            n_clusters=2,
            rng=np.random.default_rng(42),
        )
        assert len(cs) == 2

    def test_getitem(self, sample_clusters):
        """Test indexing ClusterSet."""
        cs = ClusterSet(sample_clusters)
        assert cs[0] == sample_clusters[0]

    def test_contains(self, sample_clusters):
        """Test contains operator."""
        cs = ClusterSet(sample_clusters)
        assert 0 in cs
        assert 1 in cs
        assert 99 not in cs

    def test_get_cluster(self, sample_clusters):
        """Test get_cluster method."""
        cs = ClusterSet(sample_clusters)
        assert cs.get_cluster(0) == sample_clusters[0]
        assert cs.get_cluster(99) is None

    def test_get_cluster_for_track(self, sample_clusters):
        """Test get_cluster_for_track method."""
        cs = ClusterSet(sample_clusters)
        cluster = cs.get_cluster_for_track(0)
        assert cluster is not None
        assert cluster.id == 0

        cluster = cs.get_cluster_for_track(3)
        assert cluster is not None
        assert cluster.id == 1

        assert cs.get_cluster_for_track(99) is None

    def test_clusters_in_region(self, sample_clusters):
        """Test finding clusters in region."""
        cs = ClusterSet(sample_clusters)

        # Near origin
        clusters = cs.clusters_in_region(center=[0, 0], radius=2.0)
        assert len(clusters) == 1
        assert clusters[0].id == 0

        # Near (10, 10)
        clusters = cs.clusters_in_region(center=[10, 10], radius=2.0)
        assert len(clusters) == 1
        assert clusters[0].id == 1

    def test_cluster_ids_property(self, sample_clusters):
        """Test cluster_ids property."""
        cs = ClusterSet(sample_clusters)
        assert cs.cluster_ids == [0, 1]

    def test_n_tracks_total(self, sample_clusters):
        """Test n_tracks_total property."""
        cs = ClusterSet(sample_clusters)
        assert cs.n_tracks_total == 5  # 3 + 2

    def test_cluster_stats(self, sample_tracks_for_clustering):
        """Test cluster statistics computation."""
        cs = cluster_tracks_dbscan(sample_tracks_for_clustering, eps=3.0, min_samples=2)

        stats = cs.cluster_stats(0, tracks=sample_tracks_for_clustering)
        assert stats is not None
        assert isinstance(stats, ClusterStats)
        assert stats.n_tracks > 0

    def test_all_stats(self, sample_tracks_for_clustering):
        """Test all_stats method."""
        cs = cluster_tracks_dbscan(sample_tracks_for_clustering, eps=3.0, min_samples=2)

        all_stats = cs.all_stats(tracks=sample_tracks_for_clustering)
        assert len(all_stats) == len(cs)

    def test_add_cluster(self, sample_clusters):
        """Test adding a cluster."""
        cs = ClusterSet(sample_clusters[:1])
        new_cluster = TrackCluster(
            id=5,
            track_ids=(10, 11),
            centroid=np.array([20.0, 20.0]),
            covariance=np.eye(2),
            time=2.0,
        )
        new_cs = cs.add_cluster(new_cluster)
        assert len(new_cs) == 2
        assert len(cs) == 1  # Original unchanged

    def test_remove_cluster(self, sample_clusters):
        """Test removing a cluster."""
        cs = ClusterSet(sample_clusters)
        new_cs = cs.remove_cluster(0)
        assert len(new_cs) == 1
        assert 0 not in new_cs

    def test_merge_clusters(self, sample_clusters):
        """Test merging two clusters."""
        cs = ClusterSet(sample_clusters)
        merged = cs.merge_clusters(0, 1, new_id=0)

        assert len(merged) == 1
        merged_cluster = merged.get_cluster(0)
        assert merged_cluster is not None
        assert len(merged_cluster.track_ids) == 5  # 3 + 2

    def test_split_cluster(self, sample_clusters, sample_tracks_for_clustering):
        """Test splitting a cluster."""
        cs = ClusterSet(sample_clusters)

        split = cs.split_cluster(
            cluster_id=0,
            track_ids_1=[0],
            track_ids_2=[1, 2],
            tracks=sample_tracks_for_clustering,
        )

        assert len(split) == 3  # Original 1 was kept, 0 became 2

    def test_copy(self, sample_clusters):
        """Test copying cluster set."""
        cs = ClusterSet(sample_clusters)
        copy = cs.copy()
        assert len(copy) == len(cs)

    def test_compute_cluster_centroid(self, sample_tracks_for_clustering):
        """Test compute_cluster_centroid function."""
        tracks = list(sample_tracks_for_clustering)[:3]
        centroid = compute_cluster_centroid(tracks)
        assert centroid.shape == (2,)
        # Average of [0,0], [1,1], [0.5,0.5]
        np.testing.assert_array_almost_equal(centroid, [0.5, 0.5])


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and empty scenarios."""

    def test_empty_track_list_operations(self):
        """Test operations on empty TrackList."""
        tl = TrackList()
        assert tl.states().shape == (0, 0)
        assert tl.covariances().shape == (0, 0, 0)
        assert tl.positions().shape == (0, 2)
        assert tl.track_ids == []
        assert tl.stats().n_tracks == 0

    def test_empty_measurement_set_operations(self):
        """Test operations on empty MeasurementSet."""
        ms = MeasurementSet()
        assert ms.values().shape == (0, 0)
        assert ms.time_range == (0.0, 0.0)
        assert len(ms.times) == 0

    def test_empty_cluster_set_operations(self):
        """Test operations on empty ClusterSet."""
        cs = ClusterSet()
        assert cs.cluster_ids == []
        assert cs.n_tracks_total == 0
        assert cs.clusters_in_region([0, 0], 1.0) == []

    def test_single_track_cluster(self):
        """Test clustering with single track."""
        track = Track(
            id=0,
            state=np.array([0.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(4),
            status=TrackStatus.CONFIRMED,
            hits=1,
            misses=0,
            time=1.0,
        )
        tl = TrackList([track])

        # DBSCAN with min_samples=2 should produce no clusters
        cs = cluster_tracks_dbscan(tl, eps=1.0, min_samples=2)
        assert len(cs) == 0

        # K-means with n_clusters=1 should produce 1 cluster
        cs = cluster_tracks_kmeans(tl, n_clusters=1)
        assert len(cs) == 1
