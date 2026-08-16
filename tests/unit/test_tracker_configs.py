import msgspec
import numpy as np
import pytest

from pytcl.core.exceptions import ConfigurationError
from pytcl.trackers import (
    MultiTargetConfig,
    MultiTargetTracker,
    SingleTargetConfig,
    SingleTargetTracker,
)

F = np.eye(4).tolist()
H = np.eye(2, 4).tolist()
Q = (0.01 * np.eye(4)).tolist()
R = (0.1 * np.eye(2)).tolist()


class TestSingleTargetConfig:
    def test_roundtrip(self):
        cfg = SingleTargetConfig(state_dim=4, meas_dim=2, H=H, R=R, F=F, Q=Q)
        assert (
            msgspec.json.decode(msgspec.json.encode(cfg), type=SingleTargetConfig)
            == cfg
        )

    def test_tracker_accepts_config(self):
        t = SingleTargetTracker(config=SingleTargetConfig(4, 2, H, R, F=F, Q=Q))
        assert t.state_dim == 4 and not t.is_initialized

    def test_matrix_dynamics_equivalent_to_kwargs(self):
        a = SingleTargetTracker(config=SingleTargetConfig(4, 2, H, R, F=F, Q=Q))
        b = SingleTargetTracker(
            4, 2, np.asarray(F), np.asarray(H), np.asarray(Q), np.asarray(R)
        )
        a.initialize(np.zeros(4), np.eye(4))
        b.initialize(np.zeros(4), np.eye(4))
        a.predict(1.0)
        b.predict(1.0)
        np.testing.assert_array_equal(a.state.state, b.state.state)

    def test_config_without_dynamics_rejected_by_tracker(self):
        # F/Q are Optional in the Struct (a snapshot of a callable-dynamics
        # tracker has none) but the CONSTRUCTOR needs dynamics.
        with pytest.raises(ConfigurationError):
            SingleTargetTracker(config=SingleTargetConfig(4, 2, H, R))

    def test_conflict(self):
        with pytest.raises(ConfigurationError):
            SingleTargetTracker(4, config=SingleTargetConfig(4, 2, H, R, F=F, Q=Q))

    def test_decode_validates_matrix_shape(self):
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"state_dim":4,"meas_dim":2,"H":[1.0,2.0],"R":[[0.1,0.0],[0.0,0.1]]}',
                type=SingleTargetConfig,
            )


class TestMultiTargetConfig:
    def test_tracker_accepts_config_and_retains_gate_probability(self):
        t = MultiTargetTracker(
            config=MultiTargetConfig(4, 2, H, R, F=F, Q=Q, gate_probability=0.95)
        )
        assert t.gate_probability == 0.95  # was previously discarded

    def test_kwargs_path_also_retains_gate_probability(self):
        t = MultiTargetTracker(
            4,
            2,
            np.asarray(F),
            np.asarray(H),
            np.asarray(Q),
            np.asarray(R),
            gate_probability=0.97,
        )
        assert t.gate_probability == 0.97

    def test_decode_validates_matrix_shape(self):
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"state_dim":4,"meas_dim":2,"H":[1.0,2.0],"R":[[0.1,0.0],[0.0,0.1]]}',
                type=MultiTargetConfig,
            )


class TestMHTConfigStruct:
    def test_is_struct_and_roundtrips(self):
        from pytcl.trackers import MHTConfig

        cfg = MHTConfig(n_scan=4)
        assert isinstance(cfg, msgspec.Struct)
        assert msgspec.json.decode(msgspec.json.encode(cfg), type=MHTConfig) == cfg

    def test_defaults_unchanged(self):
        from pytcl.trackers import MHTConfig

        cfg = MHTConfig()
        assert (
            cfg.n_scan,
            cfg.max_hypotheses,
            cfg.detection_prob,
            cfg.clutter_density,
            cfg.gate_probability,
            cfg.confirm_threshold,
            cfg.delete_threshold,
            cfg.min_hypothesis_prob,
            cfg.new_track_weight,
        ) == (3, 100, 0.9, 1e-6, 0.99, 3, 5, 1e-6, 0.1)
