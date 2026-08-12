"""Round-trip properties for pytcl.io serialization.

The example-based tests in tests/unit/ assert bitwise fidelity for one
history at one state dimension; the docstrings claim it universally.
These tests generate the universe.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from pytcl.io.serialize import (
    decode_states,
    decode_tracks,
    encode_states,
    encode_tracks,
)
from tests.property._strategies import finite_floats, float64_arrays, track_histories


class TestMsgpackRoundTrip:
    @given(track_histories(finite_only=False))
    def test_bitwise_for_any_float(self, hist_times):
        history, times = hist_times
        times2, history2 = decode_tracks(
            encode_tracks(history, times, fmt="msgpack"), fmt="msgpack"
        )
        assert times2 == times
        assert [len(s) for s in history2] == [len(s) for s in history]
        for scan, scan2 in zip(history, history2):
            for tr, tr2 in zip(scan, scan2):
                # tobytes() compares raw bit patterns, so NaN == NaN here.
                assert tr2.state.tobytes() == tr.state.tobytes()
                assert tr2.covariance.tobytes() == tr.covariance.tobytes()
                assert tr2.covariance.shape == tr.covariance.shape


class TestJsonContract:
    @given(track_histories(finite_only=True))
    def test_bitwise_for_finite(self, hist_times):
        history, times = hist_times
        times2, history2 = decode_tracks(
            encode_tracks(history, times, fmt="json"), fmt="json"
        )
        assert times2 == times
        for scan, scan2 in zip(history, history2):
            for tr, tr2 in zip(scan, scan2):
                assert tr2.state.tobytes() == tr.state.tobytes()
                assert tr2.covariance.tobytes() == tr.covariance.tobytes()

    @given(
        st.lists(finite_floats(), min_size=1, max_size=5),
        st.integers(min_value=0),
        st.sampled_from([np.nan, np.inf, -np.inf]),
    )
    def test_non_finite_always_raises(self, values, index, bad):
        x = np.array(values, dtype=np.float64)
        x[index % len(x)] = bad
        with pytest.raises(ValueError, match="non-finite"):
            encode_states(x, np.eye(len(x)), fmt="json")


class TestStatesRoundTrip:
    @given(
        float64_arrays(min_size=1, max_size=8, finite_only=False),
        st.sampled_from(["msgpack"]),
    )
    def test_states_bitwise_msgpack(self, x, fmt):
        P = np.eye(len(x))
        x2, P2 = decode_states(encode_states(x, P, fmt=fmt), fmt=fmt)
        assert x2.tobytes() == x.tobytes()
        assert_array_equal(P2, P)


class TestAsdfRoundTrip:
    """ASDF round trips against real files on disk.

    Hypothesis reuses function-scoped pytest fixtures (like ``tmp_path``)
    across every generated example within one ``@given`` test -- the test
    function itself is invoked once by pytest, and Hypothesis loops inside
    it. A single shared ``tmp_path`` would make every example overwrite the
    same file, masking any example-dependent path/state bug. Each test here
    instead calls ``tmp_path_factory.mktemp(...)`` *inside* the test body,
    which mints a fresh, uniquely-numbered directory per example.
    """

    def setup_method(self):
        pytest.importorskip("asdf")

    @given(track_histories(finite_only=False))
    def test_tracks_bitwise(self, tmp_path_factory, hist_times):
        from pytcl.io.asdf_io import load_tracks_asdf, save_tracks_asdf

        history, times = hist_times
        path = tmp_path_factory.mktemp("asdf-tracks") / "tracks.asdf"
        save_tracks_asdf(path, history, times)
        times2, history2 = load_tracks_asdf(path)
        assert times2 == times
        assert [len(s) for s in history2] == [len(s) for s in history]
        for scan, scan2 in zip(history, history2):
            for tr, tr2 in zip(scan, scan2):
                assert tr2.state.tobytes() == tr.state.tobytes()
                assert tr2.covariance.tobytes() == tr.covariance.tobytes()
                assert tr2.covariance.shape == tr.covariance.shape

    @given(float64_arrays(min_size=1, max_size=8, finite_only=False))
    def test_states_bitwise(self, tmp_path_factory, x):
        from pytcl.io.asdf_io import load_states_asdf, save_states_asdf

        P = np.eye(len(x))
        path = tmp_path_factory.mktemp("asdf-states") / "state.asdf"
        save_states_asdf(path, x, P)
        x2, P2 = load_states_asdf(path)
        assert x2.tobytes() == x.tobytes()
        assert_array_equal(P2, P)
