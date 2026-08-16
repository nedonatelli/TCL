"""Tests for session save/restore: SingleTargetTracker and IMMEstimator."""

import numpy as np
import pytest

from pytcl.core.exceptions import ConfigurationError, FormatError
from pytcl.dynamic_estimation import IMMEstimator
from pytcl.io import load_session, load_session_file, save_session, save_session_file
from pytcl.trackers import MultiTargetTracker, SingleTargetTracker

F4 = np.eye(4)
H24 = np.eye(2, 4)
Q4 = 0.01 * np.eye(4)
R2 = 0.1 * np.eye(2)


def _tracker():
    t = SingleTargetTracker(4, 2, F4, H24, Q4, R2)
    t.initialize(np.arange(4.0), np.eye(4))
    t.predict(0.5)
    return t


class TestSingleTargetSession:
    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_roundtrip_bitwise(self, fmt):
        t = _tracker()
        back = load_session(save_session(t, fmt=fmt), fmt=fmt)
        assert isinstance(back, SingleTargetTracker)
        np.testing.assert_array_equal(back.state.state, t.state.state)
        np.testing.assert_array_equal(back.state.covariance, t.state.covariance)
        assert back.is_initialized

    def test_resume_equals_uninterrupted(self):
        z = np.array([1.0, 2.0])
        a = _tracker()
        b = load_session(save_session(_tracker()))
        for obj in (a, b):
            obj.predict(1.0)
            obj.update(z)
        np.testing.assert_array_equal(a.state.state, b.state.state)
        np.testing.assert_array_equal(a.state.covariance, b.state.covariance)

    def test_uninitialized_roundtrip(self):
        t = SingleTargetTracker(4, 2, F4, H24, Q4, R2)
        back = load_session(save_session(t))
        assert not back.is_initialized

    def test_callable_dynamics_require_rehydration(self):
        t = SingleTargetTracker(4, 2, lambda dt: F4, H24, Q4, R2)
        t.initialize(np.zeros(4), np.eye(4))
        data = save_session(t)
        with pytest.raises(ConfigurationError):
            load_session(data)  # no F given
        back = load_session(data, F=lambda dt: F4)
        assert back.is_initialized

    def test_file_roundtrip(self, tmp_path):
        p = tmp_path / "session.msgpack"
        save_session_file(_tracker(), p)
        back = load_session_file(p)
        assert back.is_initialized

    def test_malformed_fails_loudly(self):
        with pytest.raises(FormatError):
            load_session(b"not a session")

    def test_unsupported_type_fails_loudly(self):
        with pytest.raises(ConfigurationError):
            save_session(object())

    def test_msgpack_preserves_non_finite(self):
        t = SingleTargetTracker(4, 2, F4, H24, Q4, R2)
        t.initialize(np.array([1.0, np.nan, np.inf, -np.inf]), np.eye(4))
        back = load_session(save_session(t, fmt="msgpack"), fmt="msgpack")
        assert back.state.state.tobytes() == t.state.state.tobytes()

    def test_json_raises_on_non_finite(self):
        t = SingleTargetTracker(4, 2, F4, H24, Q4, R2)
        t.initialize(np.array([1.0, np.nan, 0.0, 0.0]), np.eye(4))
        with pytest.raises(ValueError, match="non-finite"):
            save_session(t, fmt="json")

    def test_matrix_config_rejects_rehydration_kwargs(self):
        data = save_session(_tracker())
        with pytest.raises(ConfigurationError):
            load_session(data, F=2 * F4)
        with pytest.raises(ConfigurationError):
            load_session(data, Q=2 * Q4)


class TestIMMSession:
    def test_roundtrip_and_resume(self):
        def build():
            e = IMMEstimator(2, 2, [[0.9, 0.1], [0.1, 0.9]])
            for i in range(2):
                e.set_mode_model(i, np.eye(2), 0.01 * np.eye(2))
            e.set_measurement_model(np.eye(2), 0.1 * np.eye(2))
            e.initialize(np.zeros(2), np.eye(2))
            return e

        a = build()
        b = load_session(save_session(build()))
        z = np.array([0.3, -0.2])
        for obj in (a, b):
            obj.predict()
            obj.update(z)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.P, b.P)
        np.testing.assert_array_equal(a.mode_probs, b.mode_probs)

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_roundtrip_bitwise_both_formats(self, fmt):
        e = IMMEstimator(2, 2, [[0.9, 0.1], [0.1, 0.9]])
        for i in range(2):
            e.set_mode_model(i, np.eye(2), 0.01 * np.eye(2))
        e.set_measurement_model(np.eye(2), 0.1 * np.eye(2))
        e.initialize(np.zeros(2), np.eye(2))
        e.predict()

        back = load_session(save_session(e, fmt=fmt), fmt=fmt)
        np.testing.assert_array_equal(back.x, e.x)
        np.testing.assert_array_equal(back.P, e.P)
        np.testing.assert_array_equal(back.mode_probs, e.mode_probs)
        for f1, f2 in zip(back.F_list, e.F_list):
            np.testing.assert_array_equal(f1, f2)

    def test_rejects_rehydration_kwargs(self):
        e = IMMEstimator(2, 2, [[0.9, 0.1], [0.1, 0.9]])
        for i in range(2):
            e.set_mode_model(i, np.eye(2), 0.01 * np.eye(2))
        e.set_measurement_model(np.eye(2), 0.1 * np.eye(2))
        e.initialize(np.zeros(2), np.eye(2))
        data = save_session(e)
        with pytest.raises(ConfigurationError):
            load_session(data, F=np.eye(2))


def _mt_tracker():
    t = MultiTargetTracker(
        4, 2, F4, H24, Q4, R2, confirm_hits=2, confirm_window=3, max_misses=2
    )
    rng = np.random.Generator(np.random.PCG64(5))
    for _ in range(4):
        z = [rng.normal(size=2), rng.normal(size=2) + 10.0]
        t.process(z, dt=1.0)
    return t


class TestMultiTargetSession:
    def test_roundtrip_track_table(self):
        t = _mt_tracker()
        back = load_session(save_session(t))
        assert len(back.tracks) == len(t.tracks)
        for ta, tb in zip(t.tracks, back.tracks):
            assert ta.id == tb.id and ta.status is tb.status
            assert (ta.hits, ta.misses) == (tb.hits, tb.misses)
            np.testing.assert_array_equal(ta.state, tb.state)
            np.testing.assert_array_equal(ta.covariance, tb.covariance)

    def test_resume_equals_uninterrupted(self):
        a = _mt_tracker()
        b = load_session(save_session(_mt_tracker()))
        z = [np.array([0.1, 0.2]), np.array([10.1, 10.2])]
        ra = a.process(z, dt=1.0)
        rb = b.process(z, dt=1.0)
        assert [t.id for t in ra] == [t.id for t in rb]
        for ta, tb in zip(ra, rb):
            np.testing.assert_array_equal(ta.state, tb.state)

    def test_next_id_preserved(self):
        # a fresh detection after restore must not reuse an existing track id
        t = _mt_tracker()
        existing = {tr.id for tr in t.tracks}
        back = load_session(save_session(t))
        back.process([np.array([50.0, 50.0])], dt=1.0)
        new_ids = {tr.id for tr in back.tracks} - existing
        assert new_ids and not (new_ids & existing)

    def test_nis_history_carried_when_present(self):
        from pytcl.diagnostics import disable_debug_logging, enable_debug_logging

        enable_debug_logging()
        try:
            t = _mt_tracker()
            has = [tr for tr in t._tracks if getattr(tr, "_nis_history", None)]
            assert has, "expected NIS windows under enabled diagnostics"
            back = load_session(save_session(t))
            back_map = {tr.id: tr for tr in back._tracks}
            for tr in has:
                assert list(getattr(back_map[tr.id], "_nis_history")) == list(
                    tr._nis_history
                )
        finally:
            disable_debug_logging()

    def test_callable_dynamics_require_rehydration(self):
        t = MultiTargetTracker(4, 2, lambda dt: F4, H24, Q4, R2)
        t.process([np.array([0.0, 0.0])], dt=1.0)
        data = save_session(t)
        with pytest.raises(ConfigurationError):
            load_session(data)  # no F given
        back = load_session(data, F=lambda dt: F4)
        assert len(back.tracks) == len(t.tracks)

    def test_matrix_config_rejects_rehydration_kwargs(self):
        data = save_session(_mt_tracker())
        with pytest.raises(ConfigurationError):
            load_session(data, F=2 * F4)
        with pytest.raises(ConfigurationError):
            load_session(data, Q=2 * Q4)
