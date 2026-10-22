"""A resumed MultiTargetTracker must continue M-of-N confirmation, not restart it."""

import msgspec
import numpy as np

from pytcl.io import load_session, save_session
from pytcl.io.session import SESSION_SCHEMA_VERSION, SessionEnvelope
from pytcl.trackers import MultiTargetTracker

F4 = np.eye(4)
H24 = np.eye(2, 4)
Q4 = 0.01 * np.eye(4)
R2 = 0.1 * np.eye(2)

SCAN = [np.array([0.0, 0.0]), np.array([30.0, 30.0])]


def _tracker_mid_confirmation() -> MultiTargetTracker:
    # Two scans: each track holds 2 of the 3 hits it needs, still TENTATIVE.
    t = MultiTargetTracker(
        4, 2, F4, H24, Q4, R2, confirm_hits=3, confirm_window=4, max_misses=3
    )
    for _ in range(2):
        t.process(SCAN, dt=1.0)
    return t


def _statuses(t: MultiTargetTracker) -> list:
    return [(tr.id, tr.status.name) for tr in t.tracks]


def test_confirmation_state_survives_a_round_trip():
    original = _tracker_mid_confirmation()
    assert {s for _, s in _statuses(original)} == {"TENTATIVE"}
    restored = load_session(save_session(original))
    assert _statuses(restored) == _statuses(original)

    original.process(SCAN, dt=1.0)
    restored.process(SCAN, dt=1.0)
    assert {s for _, s in _statuses(original)} == {"CONFIRMED"}
    assert _statuses(restored) == _statuses(original)


def test_restored_tracker_matches_unsaved_original_on_next_scans():
    original = _tracker_mid_confirmation()
    restored = load_session(save_session(original))
    # Confirmation, then empty scans until both tracks pass max_misses=3 and
    # are deleted.
    scans = [[SCAN[0]], [], SCAN, SCAN, [], [], [], []]
    live = []
    for scan in scans:
        a = original.process(scan, dt=1.0)
        b = restored.process(scan, dt=1.0)
        assert [(t.id, t.status, t.hits, t.misses) for t in a] == [
            (t.id, t.status, t.hits, t.misses) for t in b
        ]
        for ta, tb in zip(a, b):
            np.testing.assert_array_equal(ta.state, tb.state)
            np.testing.assert_array_equal(ta.covariance, tb.covariance)
        live.append(len(a))
    assert live == [2, 2, 2, 2, 2, 2, 0, 0]
    assert restored._tracks == [] and original._tracks == []


def test_window_is_bounded_and_ordered_after_round_trip():
    t = _tracker_mid_confirmation()
    t.process([SCAN[0]], dt=1.0)
    t.process([], dt=1.0)
    t.process([], dt=1.0)
    restored = load_session(save_session(t))
    for a, b in zip(t._tracks, restored._tracks):
        assert list(b.recent) == list(a.recent)
        assert len(b.recent) <= restored.confirm_window


def _legacy_bytes(t: MultiTargetTracker) -> bytes:
    """A v2 file rewritten to the v1 layout: ``recent`` dropped, version 1.

    Not written by v2.11.0 code, but for the tracker in this file it decodes
    equal to what v2.11.0's ``save_session`` writes (compared once, when the
    fixture was written).
    """
    raw = msgspec.msgpack.decode(save_session(t))
    raw["schema_version"] = 1
    for ts in raw["snapshot"]["tracks"]:
        del ts["recent"]
    return msgspec.msgpack.encode(raw)


def test_schema_version_is_bumped():
    assert SESSION_SCHEMA_VERSION == 2
    env = msgspec.msgpack.decode(
        save_session(_tracker_mid_confirmation()), type=SessionEnvelope
    )
    assert env.schema_version == SESSION_SCHEMA_VERSION


def test_legacy_v1_file_loads_with_an_empty_window():
    original = _tracker_mid_confirmation()
    restored = load_session(_legacy_bytes(original))
    assert _statuses(restored) == _statuses(original)
    assert all(len(tr.recent) == 0 for tr in restored._tracks)
    # Today's behaviour, unchanged for old files: confirmation restarts, so the
    # third hit does not confirm a track whose window was not stored.
    restored.process(SCAN, dt=1.0)
    assert {s for _, s in _statuses(restored)} == {"TENTATIVE"}
