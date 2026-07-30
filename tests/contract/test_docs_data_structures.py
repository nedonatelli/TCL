"""Execute the examples on the data-structures page.

That page described a ``TrackSet`` class imported from
``tcl.tracking_containers``, with attributes ``track.uid``, ``track.position``,
``track.velocity``, ``track.age``, ``track.gate_size`` and ``track.track_type``.
None of it existed -- not the package name, not the module, not the class, not
one of the attributes. It survived because its import line began with ``tcl.``
rather than ``pytcl.``, so the docs import guard skipped it entirely.

These tests run what the page now claims, against the real API.
"""

import numpy as np
import pytest

from pytcl.containers import (
    ClusterSet,
    KDTree,
    Measurement,
    MeasurementSet,
    TrackList,
)
from pytcl.io import TrackHDF5Storage
from pytcl.trackers import MultiTargetTracker, Track, TrackStatus

F = np.array(
    [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


@pytest.fixture
def tracker():
    t = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=np.eye(4) * 0.01,
        R=np.eye(2) * 4.0,
        confirm_hits=2,
    )
    for k in range(6):
        t.process(
            [np.array([k * 2.0, k * 1.0]), np.array([50.0 + k, 60.0 - k * 0.5])],
            dt=1.0,
        )
    return t


@pytest.fixture
def detections():
    return MeasurementSet.from_arrays(
        values=np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        covariances=np.stack([np.eye(2)] * 4),
    )


def test_track_fields_are_what_the_page_lists():
    """The documented field list must match the NamedTuple exactly."""
    assert Track._fields == (
        "id",
        "state",
        "covariance",
        "status",
        "hits",
        "misses",
        "time",
    )


def test_track_has_no_invented_attributes():
    """The old page's attributes must stay gone, not quietly reappear."""
    absent = ["uid", "position", "velocity", "age", "gate_size", "track_type"]
    present = [name for name in absent if hasattr(Track, name)]
    assert not present, f"Track unexpectedly has {present}"


def test_tracksetclass_does_not_exist():
    """Guard the note on the page: there is no TrackSet."""
    import pytcl.containers as containers

    assert not hasattr(containers, "TrackSet")
    with pytest.raises(ModuleNotFoundError):
        __import__("tcl.tracking_containers")


def test_track_example(tracker):
    track = tracker.tracks[0]
    assert isinstance(track.id, int)
    assert track.status is TrackStatus.CONFIRMED
    assert track.hits == 6


def test_tracklist_example(tracker):
    tracks = TrackList.from_tracker(tracker)

    assert tracks.track_ids == [0, 1]

    stats = tracks.stats()
    assert stats.n_tracks == 2
    assert stats.n_confirmed == 2

    # confirmed / tentative / track_ids are properties, not calls
    assert tracks.confirmed.track_ids == [0, 1]
    assert tracks.tentative.track_ids == []

    assert tracks.states().shape == (2, 4)
    assert tracks.covariances().shape == (2, 4, 4)
    assert tracks.positions(indices=(0, 2)).shape == (2, 2)

    assert tracks.get_by_id(0).id == 0
    assert tracks.filter_by_status(TrackStatus.CONFIRMED).track_ids == [0, 1]
    assert tracks.filter_by_region([0.0, 0.0], radius=20.0).track_ids == [0]
    assert tracks.filter_by_predicate(lambda t: t.hits >= 5).track_ids == [0, 1]
    assert tracks.filter_by_time(min_time=2.0).track_ids == [0, 1]


def test_tracklist_is_immutable(tracker):
    """Every operation returns a new list; the original is untouched."""
    tracks = TrackList.from_tracker(tracker)
    before = list(tracks.track_ids)

    smaller = tracks.remove(0)

    assert smaller.track_ids == [1]
    assert tracks.track_ids == before, "remove() mutated the original"


def test_positions_uses_the_state_layout(tracker):
    """positions() takes indices because the layout is the caller's choice.

    With f_constant_velocity the state is [x, vx, y, vy], so position is
    elements 0 and 2. Asking for (0, 1) returns x and vx instead, which is why
    the page documents the argument rather than assuming a layout.
    """
    tracks = TrackList.from_tracker(tracker)
    states = tracks.states()

    np.testing.assert_array_equal(tracks.positions(indices=(0, 2)), states[:, [0, 2]])
    np.testing.assert_array_equal(tracks.positions(indices=(0, 1)), states[:, [0, 1]])


def test_measurementset_example(detections):
    np.testing.assert_array_equal(detections.times, np.array([0.0, 1.0, 2.0, 3.0]))
    assert detections.time_range == (0.0, 3.0)
    assert list(detections.sensors) == [0]

    assert detections.values().shape == (4, 2)
    assert len(detections.at_time(2.0)) == 1
    assert len(detections.in_time_window(1.0, 2.0)) == 2
    assert len(detections.in_region([1.0, 2.0], radius=3.0)) >= 1

    query = detections.nearest_to([2.0, 4.0], k=2)
    assert len(query.measurements) == 2
    np.testing.assert_allclose(query.measurements[0].value, [2.0, 4.0])


def test_measurementset_feeds_per_detection_covariance(tracker, detections):
    """The page's claim that a set drops straight into process()."""
    scan = detections.at_time(2.0)
    values = [m.value for m in scan]
    covariances = [m.covariance for m in scan]

    assert len(values) == len(covariances) == 1
    tracks = tracker.process(values, dt=1.0, measurement_covariances=covariances)
    assert len(tracks) >= 1


def test_measurement_fields():
    assert Measurement._fields == ("value", "time", "covariance", "sensor_id", "id")


def test_clusterset_example(tracker):
    tracks = TrackList.from_tracker(tracker)
    clusters = ClusterSet.from_tracks(tracks, method="dbscan", eps=30.0)
    assert isinstance(clusters.cluster_ids, list)
    clusters.all_stats()


def test_kdtree_example():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [5.0, 5.0]])
    index = KDTree(points)

    result = index.query(np.array([1.1, 1.1]), k=2)
    assert result is not None

    within = index.query_radius(np.array([0.0, 0.0]), r=2.0)
    assert within is not None


def test_persistence_example(tmp_path):
    path = tmp_path / "tracks.h5"
    states = np.zeros((10, 4))
    covariances = np.stack([np.eye(4)] * 10)
    timestamps = np.arange(10.0)

    storage = TrackHDF5Storage(str(path))
    storage.open("w")
    storage.store_track(
        "track_0",
        states=states,
        covariances=covariances,
        timestamps=timestamps,
        metadata={"status": "confirmed"},
    )
    storage.close()

    storage = TrackHDF5Storage(str(path))
    storage.open("r")
    try:
        assert storage.list_tracks() == ["track_0"]
        record = storage.retrieve_track("track_0")
        np.testing.assert_array_equal(np.asarray(record["states"]), states)
        np.testing.assert_array_equal(np.asarray(record["covariances"]), covariances)
        assert record["metadata"]["status"] == "confirmed"

        trajectory = storage.get_track_trajectory("track_0")
        np.testing.assert_array_equal(np.asarray(trajectory["states"]), states)
    finally:
        storage.close()
