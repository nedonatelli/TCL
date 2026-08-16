"""Tests for session save/restore: SingleTargetTracker and IMMEstimator."""

import json

import msgspec
import numpy as np
import pytest

from pytcl.core.exceptions import ConfigurationError, FormatError
from pytcl.dynamic_estimation import GaussianSumFilter, IMMEstimator, RBPFFilter
from pytcl.io import load_session, load_session_file, save_session, save_session_file
from pytcl.io.session import SessionEnvelope
from pytcl.trackers import (
    MHTConfig,
    MHTTracker,
    MultiTargetTracker,
    SingleTargetTracker,
)

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
        # tobytes(), not just assert_array_equal: IEEE -0.0 == 0.0 under
        # array_equal, so it would miss a sign-of-zero regression that a
        # true bit-exact round-trip (the documented guarantee) must not
        # have, on either wire format.
        assert back.state.state.tobytes() == t.state.state.tobytes()
        assert back.state.covariance.tobytes() == t.state.covariance.tobytes()
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


def _mht_tracker():
    t = MHTTracker(4, 2, F4, H24, Q4, R2, config=MHTConfig(n_scan=2, max_hypotheses=20))
    rng = np.random.Generator(np.random.PCG64(11))
    for _ in range(3):
        t.process([rng.normal(size=2), rng.normal(size=2) + 8.0], dt=1.0)
    return t


class TestMHTSession:
    def test_roundtrip_tree(self):
        t = _mht_tracker()
        back = load_session(save_session(t))
        assert back.hypothesis_tree.current_scan == t.hypothesis_tree.current_scan
        assert set(back.hypothesis_tree.tracks) == set(t.hypothesis_tree.tracks)
        for tid, tr in t.hypothesis_tree.tracks.items():
            btr = back.hypothesis_tree.tracks[tid]
            assert (
                btr.score,
                btr.status,
                btr.history,
                btr.parent_id,
                btr.scan_created,
            ) == (tr.score, tr.status, tr.history, tr.parent_id, tr.scan_created)
            np.testing.assert_array_equal(btr.state, tr.state)
        assert [h.id for h in back.hypothesis_tree.hypotheses] == [
            h.id for h in t.hypothesis_tree.hypotheses
        ]

    def test_json_roundtrip_tree(self):
        # nested list-of-Struct fields (tracks, hypotheses) must not trip
        # the json non-finite walker
        t = _mht_tracker()
        back = load_session(save_session(t, fmt="json"), fmt="json")
        assert set(back.hypothesis_tree.tracks) == set(t.hypothesis_tree.tracks)
        assert [h.id for h in back.hypothesis_tree.hypotheses] == [
            h.id for h in t.hypothesis_tree.hypotheses
        ]

    def test_resume_equals_uninterrupted(self):
        a = _mht_tracker()
        b = load_session(save_session(_mht_tracker()))
        z = [np.array([0.5, -0.5])]
        ra, rb = a.process(z, dt=1.0), b.process(z, dt=1.0)
        assert [t.id for t in ra.all_tracks] == [t.id for t in rb.all_tracks]
        for ta, tb in zip(ra.all_tracks, rb.all_tracks):
            np.testing.assert_array_equal(ta.state, tb.state)

    def test_callable_dynamics_require_rehydration(self):
        t = MHTTracker(4, 2, lambda dt: F4, H24, Q4, R2)
        t.process([np.array([0.0, 0.0])], dt=1.0)
        data = save_session(t)
        with pytest.raises(ConfigurationError):
            load_session(data)  # no F given
        back = load_session(data, F=lambda dt: F4)
        assert set(back.hypothesis_tree.tracks) == set(t.hypothesis_tree.tracks)

    def test_matrix_config_rejects_rehydration_kwargs(self):
        data = save_session(_mht_tracker())
        with pytest.raises(ConfigurationError):
            load_session(data, F=2 * F4)
        with pytest.raises(ConfigurationError):
            load_session(data, Q=2 * Q4)


class TestRBPFSession:
    def test_resume_bit_reproducible_with_instance_rng(self):
        def build():
            f = RBPFFilter(
                max_particles=16, rng=np.random.Generator(np.random.PCG64(42))
            )
            f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=16)
            return f

        a = build()
        b = load_session(save_session(build()))
        g_mat = np.eye(2)
        f_mat = np.eye(2)
        Qy = 0.01 * np.eye(2)
        Qx = 0.01 * np.eye(2)
        for filt in (a, b):
            filt.predict(
                lambda y: g_mat @ y, g_mat, Qy, lambda x, y: f_mat @ x, f_mat, Qx
            )
        for pa, pb in zip(a.get_particles(), b.get_particles()):
            np.testing.assert_array_equal(pa.y, pb.y)
            np.testing.assert_array_equal(pa.x, pb.x)
            np.testing.assert_array_equal(pa.P, pb.P)
            assert pa.w == pb.w

    def test_global_rng_roundtrips_with_none_state(self):
        f = RBPFFilter(max_particles=8)
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=8)
        back = load_session(save_session(f))
        assert back._rng is None
        assert len(back.get_particles()) == 8

    def test_instance_rng_state_roundtrips(self):
        f = RBPFFilter(max_particles=8, rng=np.random.Generator(np.random.PCG64(3)))
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=8)
        back = load_session(save_session(f))
        assert back._rng.bit_generator.state == f._rng.bit_generator.state

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_particles_roundtrip_both_formats(self, fmt):
        f = RBPFFilter(max_particles=5)
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=5)
        back = load_session(save_session(f, fmt=fmt), fmt=fmt)
        assert len(back.get_particles()) == len(f.get_particles())
        for pa, pb in zip(f.get_particles(), back.get_particles()):
            np.testing.assert_array_equal(pa.y, pb.y)
            np.testing.assert_array_equal(pa.x, pb.x)
            np.testing.assert_array_equal(pa.P, pb.P)
            assert pa.w == pb.w

    def test_rejects_rehydration_kwargs(self):
        f = RBPFFilter(max_particles=4)
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=4)
        data = save_session(f)
        with pytest.raises(ConfigurationError):
            load_session(data, F=np.eye(2))

    def test_rng_state_is_pcg64_json(self):
        f = RBPFFilter(max_particles=4, rng=np.random.Generator(np.random.PCG64(9)))
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=4)
        data = save_session(f, fmt="json")
        env = json.loads(data)
        rng_state = json.loads(env["snapshot"]["rng_state"])
        assert rng_state["bit_generator"] == "PCG64"

    def test_non_pcg64_rng_raises_configuration_error(self):
        # MT19937's bit-generator state holds an ndarray, which stdlib
        # json cannot serialize -- save_session must turn that into a
        # named ConfigurationError, not a raw TypeError from json.dumps.
        # Both GaussianSumFilter and RBPFFilter route through the same
        # _rng_state_to_json helper, so exercising it via RBPFFilter here
        # covers GaussianSumFilter too.
        f = RBPFFilter(max_particles=4, rng=np.random.Generator(np.random.MT19937(1)))
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=4)
        with pytest.raises(ConfigurationError, match="MT19937"):
            save_session(f)
        # PCG64 unaffected: covered by test_instance_rng_state_roundtrips
        # above, which already round-trips a PCG64-backed instance rng.


class TestGSFSession:
    def test_components_roundtrip(self):
        f = GaussianSumFilter(max_components=3)
        f.initialize(np.zeros(2), np.eye(2), num_components=2)
        back = load_session(save_session(f))
        assert back.get_num_components() == f.get_num_components()
        for ca, cb in zip(f.get_components(), back.get_components()):
            np.testing.assert_array_equal(ca.x, cb.x)
            np.testing.assert_array_equal(ca.P, cb.P)
            assert ca.w == cb.w

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_components_roundtrip_both_formats(self, fmt):
        f = GaussianSumFilter(max_components=3)
        f.initialize(np.zeros(2), np.eye(2), num_components=2)
        back = load_session(save_session(f, fmt=fmt), fmt=fmt)
        for ca, cb in zip(f.get_components(), back.get_components()):
            np.testing.assert_array_equal(ca.x, cb.x)
            np.testing.assert_array_equal(ca.P, cb.P)
            assert ca.w == cb.w

    def test_instance_rng_state_roundtrips(self):
        f = GaussianSumFilter(
            max_components=3, rng=np.random.Generator(np.random.PCG64(7))
        )
        f.initialize(np.zeros(2), np.eye(2), num_components=2)
        back = load_session(save_session(f))
        assert back._rng.bit_generator.state == f._rng.bit_generator.state

    def test_global_rng_roundtrips_with_none_state(self):
        f = GaussianSumFilter(max_components=3)
        f.initialize(np.zeros(2), np.eye(2), num_components=2)
        back = load_session(save_session(f))
        assert back._rng is None

    def test_rejects_rehydration_kwargs(self):
        f = GaussianSumFilter(max_components=3)
        f.initialize(np.zeros(2), np.eye(2), num_components=2)
        data = save_session(f)
        with pytest.raises(ConfigurationError):
            load_session(data, F=np.eye(2))

    def test_resume_equals_uninterrupted(self):
        F = np.eye(2)
        Q = 0.01 * np.eye(2)
        H = np.eye(2)
        R = 0.1 * np.eye(2)
        f_fn = lambda x: F @ x  # noqa: E731
        h_fn = lambda x: H @ x  # noqa: E731

        def build():
            # An instance rng makes the two independent `build()` calls
            # below (one for `a`, one inside `b`'s save/load) produce
            # identical initial components -- with the default global RNG,
            # the second call's perturbed means would draw from wherever
            # the first call left the global stream, so the two GSFs would
            # start from different states before resume even enters the
            # picture.
            filt = GaussianSumFilter(
                max_components=3, rng=np.random.Generator(np.random.PCG64(42))
            )
            filt.initialize(np.zeros(2), np.eye(2), num_components=3)
            filt.predict(f_fn, F, Q)
            filt.update(np.array([0.1, 0.2]), h_fn, H, R)
            return filt

        a = build()
        b = load_session(save_session(build()))

        z = np.array([0.3, -0.1])
        for filt in (a, b):
            filt.predict(f_fn, F, Q)
            filt.update(z, h_fn, H, R)

        xa, Pa = a.estimate()
        xb, Pb = b.estimate()
        assert xa.tobytes() == xb.tobytes()
        assert Pa.tobytes() == Pb.tobytes()
        assert a.get_num_components() == b.get_num_components()
        for ca, cb in zip(a.get_components(), b.get_components()):
            assert ca.x.tobytes() == cb.x.tobytes()
            assert ca.P.tobytes() == cb.P.tobytes()
            assert ca.w == cb.w


def _tamper(data: bytes, mutate) -> bytes:
    """Round-trip `data` through msgspec's msgpack codec, letting `mutate`
    edit the decoded envelope in place, then re-encode -- builds tampered
    session bytes at runtime instead of hardcoding golden bytes.
    """
    env = msgspec.msgpack.decode(data, type=SessionEnvelope)
    mutate(env)
    return msgspec.msgpack.encode(env)


class TestDecodeErrors:
    def test_bogus_multi_target_status_raises_format_error(self):
        data = save_session(_mt_tracker())
        tampered = _tamper(
            data, lambda env: setattr(env.snapshot.tracks[0], "status", "not_a_status")
        )
        with pytest.raises(FormatError):
            load_session(tampered)

    def test_bogus_mht_status_raises_format_error(self):
        data = save_session(_mht_tracker())
        tampered = _tamper(
            data, lambda env: setattr(env.snapshot.tracks[0], "status", "not_a_status")
        )
        with pytest.raises(FormatError):
            load_session(tampered)

    def test_bogus_rng_state_raises_format_error(self):
        f = RBPFFilter(max_particles=4, rng=np.random.Generator(np.random.PCG64(1)))
        f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=4)
        data = save_session(f)
        tampered = _tamper(
            data, lambda env: setattr(env.snapshot, "rng_state", "{not valid json")
        )
        with pytest.raises(FormatError):
            load_session(tampered)
