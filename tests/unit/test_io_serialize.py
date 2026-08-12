"""msgspec serialization: bitwise round-trips and loud failures."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pytcl.io.serialize import (
    decode_states,
    decode_tracks,
    encode_states,
    encode_tracks,
)


def _history():
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


class TestTrackRoundTrip:
    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_bitwise_round_trip(self, fmt):
        history, times = _history()
        blob = encode_tracks(history, times, fmt=fmt)
        times2, history2 = decode_tracks(blob, fmt=fmt)
        assert times2 == times
        assert len(history2) == len(history)
        for scan, scan2 in zip(history, history2):
            for tr, tr2 in zip(scan, scan2):
                assert tr2.id == tr.id
                assert tr2.status == tr.status.value
                assert_array_equal(tr2.state, np.asarray(tr.state, dtype=np.float64))
                assert_array_equal(tr2.covariance, np.asarray(tr.covariance))
                assert tr2.covariance.shape == (len(tr2.state), len(tr2.state))

    def test_msgpack_preserves_non_finite(self):
        # tobytes() compares raw float64 bit patterns, so NaN == NaN here.
        x = np.array([1.0, np.nan, np.inf, -np.inf])
        P = np.eye(4)
        x2, P2 = decode_states(encode_states(x, P, fmt="msgpack"), fmt="msgpack")
        assert x2.tobytes() == x.tobytes()
        assert P2.tobytes() == P.tobytes()

    def test_json_raises_on_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            encode_states(np.array([np.nan]), np.eye(1), fmt="json")

    def test_json_finite_bitwise(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=8)
        P = rng.normal(size=(8, 8))
        x2, P2 = decode_states(encode_states(x, P, fmt="json"), fmt="json")
        assert x.tobytes() == x2.tobytes()
        assert P.tobytes() == P2.tobytes()

    def test_malformed_decode_fails_loudly(self):
        with pytest.raises(Exception) as excinfo:
            decode_tracks(b'{"nonsense": true}', fmt="json")
        assert (
            "nonsense" in str(excinfo.value)
            or "Object" in str(excinfo.value)
            or "missing" in str(excinfo.value)
        )

    def test_unknown_fmt_raises(self):
        with pytest.raises(ValueError, match="fmt"):
            encode_states(np.zeros(2), np.eye(2), fmt="pickle")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            encode_tracks([[]], [0.0, 1.0])

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_empty_history_round_trips(self, fmt):
        blob = encode_tracks([], [], fmt=fmt)
        times2, history2 = decode_tracks(blob, fmt=fmt)
        assert times2 == []
        assert history2 == []
