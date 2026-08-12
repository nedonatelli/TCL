"""ASDF export/import: bitwise round-trips, schema hook, and the
DependencyError guard.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

asdf = pytest.importorskip("asdf")

from pytcl.io.asdf_io import (  # noqa: E402
    load_states_asdf,
    load_tracks_asdf,
    save_states_asdf,
    save_tracks_asdf,
)


def _history():
    # Same builder as tests/unit/test_io_serialize.py::_history (test helper,
    # duplicated per the Task 2/6 briefs rather than importing across test
    # modules).
    from pytcl.trackers import MultiTargetTracker

    rng = np.random.default_rng(3)
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
        ),
        H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
        Q=np.eye(4) * 0.01,
        R=np.eye(2) * 1.0,
        confirm_hits=1,
    )
    history, times = [], []
    for k in range(6):
        z = [np.array([k + rng.normal(0, 0.3), k + rng.normal(0, 0.3)])]
        history.append(tracker.process(z, dt=1.0))
        times.append(float(k))
    return history, times


class TestTracksRoundTrip:
    def test_bitwise_round_trip(self, tmp_path):
        history, times = _history()
        path = tmp_path / "tracks.asdf"
        save_tracks_asdf(path, history, times)
        times2, history2 = load_tracks_asdf(path)
        assert times2 == times
        assert len(history2) == len(history)
        for scan, scan2 in zip(history, history2):
            assert len(scan) == len(scan2)
            for tr, tr2 in zip(scan, scan2):
                assert tr2.id == tr.id
                assert tr2.status == tr.status.value
                assert_array_equal(tr2.state, np.asarray(tr.state, dtype=np.float64))
                assert_array_equal(
                    tr2.covariance, np.asarray(tr.covariance, dtype=np.float64)
                )
                assert tr2.covariance.shape == (len(tr2.state), len(tr2.state))

    def test_schema_version_present(self, tmp_path):
        history, times = _history()
        path = tmp_path / "tracks.asdf"
        save_tracks_asdf(path, history, times)
        with asdf.open(path) as af:
            assert af.tree["pytcl"]["schema_version"] == 1

    def test_mixed_dim_history_raises(self, tmp_path):
        from pytcl.io.serialize import SimpleTrack

        history = [
            [
                SimpleTrack(
                    id=1, state=np.zeros(4), covariance=np.eye(4), status="confirmed"
                )
            ],
            [
                SimpleTrack(
                    id=2, state=np.zeros(6), covariance=np.eye(6), status="confirmed"
                )
            ],
        ]
        times = [0.0, 1.0]
        path = tmp_path / "bad.asdf"
        with pytest.raises(ValueError, match="dim"):
            save_tracks_asdf(path, history, times)

    def test_dependency_error_without_asdf(self, monkeypatch, tmp_path):
        import pytcl.io.asdf_io as mod

        monkeypatch.setattr(mod, "_import_asdf", mod._raise_missing)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="asdf"):
            save_tracks_asdf(tmp_path / "does_not_matter.asdf", [], [])


class TestStatesRoundTrip:
    def test_bitwise_round_trip(self, tmp_path):
        rng = np.random.default_rng(11)
        x = rng.normal(size=6)
        P = rng.normal(size=(6, 6))
        path = tmp_path / "state.asdf"
        save_states_asdf(path, x, P)
        x2, P2 = load_states_asdf(path)
        assert_array_equal(x2, x)
        assert_array_equal(P2, P)

    def test_schema_version_present(self, tmp_path):
        x = np.zeros(3)
        P = np.eye(3)
        path = tmp_path / "state.asdf"
        save_states_asdf(path, x, P)
        with asdf.open(path) as af:
            assert af.tree["pytcl"]["schema_version"] == 1

    def test_dependency_error_without_asdf(self, monkeypatch, tmp_path):
        import pytcl.io.asdf_io as mod

        monkeypatch.setattr(mod, "_import_asdf", mod._raise_missing)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="asdf"):
            save_states_asdf(tmp_path / "does_not_matter.asdf", np.zeros(2), np.eye(2))
