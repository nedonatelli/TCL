import msgspec
import numpy as np
import pytest

from pytcl.core.exceptions import ConfigurationError
from pytcl.dynamic_estimation import (
    GaussianSumConfig,
    GaussianSumFilter,
    IMMConfig,
    IMMEstimator,
    RBPFConfig,
    RBPFFilter,
)

TPM = [[0.95, 0.05], [0.05, 0.95]]


class TestIMMConfig:
    def test_construct_and_roundtrip_json(self):
        cfg = IMMConfig(n_modes=2, state_dim=4, transition_matrix=TPM)
        raw = msgspec.json.encode(cfg)
        back = msgspec.json.decode(raw, type=IMMConfig)
        assert back == cfg

    def test_from_arrays_accepts_ndarray(self):
        cfg = IMMConfig.from_arrays(
            n_modes=2, state_dim=4, transition_matrix=np.asarray(TPM)
        )
        assert cfg.transition_matrix == TPM

    def test_estimator_accepts_config(self):
        est = IMMEstimator(config=IMMConfig(2, 4, TPM))
        assert est.n_modes == 2 and est.state_dim == 4

    def test_config_equivalent_to_kwargs(self):
        a = IMMEstimator(config=IMMConfig(2, 4, TPM))
        b = IMMEstimator(2, 4, TPM)
        np.testing.assert_array_equal(a.mode_probs, b.mode_probs)

    def test_config_conflicts_with_kwargs(self):
        with pytest.raises(ConfigurationError):
            IMMEstimator(2, config=IMMConfig(2, 4, TPM))

    def test_frozen(self):
        cfg = IMMConfig(2, 4, TPM)
        with pytest.raises(AttributeError):
            cfg.n_modes = 3

    def test_decode_validates_types(self):
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(b'{"n_modes": "two"}', type=IMMConfig)

    def test_decode_rejects_flat_transition_matrix(self):
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"n_modes":2,"state_dim":4,"transition_matrix":[1.0,2.0]}',
                type=IMMConfig,
            )


class TestScalarConfigs:
    def test_gsf_config(self):
        f = GaussianSumFilter(config=GaussianSumConfig(max_components=7))
        assert f.max_components == 7

    def test_gsf_conflict(self):
        with pytest.raises(ConfigurationError):
            GaussianSumFilter(max_components=3, config=GaussianSumConfig())

    def test_gsf_config_equivalent_to_kwargs(self):
        a = GaussianSumFilter(
            config=GaussianSumConfig(
                max_components=7, merge_threshold=0.02, prune_threshold=2e-3
            )
        )
        b = GaussianSumFilter(
            max_components=7, merge_threshold=0.02, prune_threshold=2e-3
        )
        assert a.max_components == b.max_components
        assert a.merge_threshold == b.merge_threshold
        assert a.prune_threshold == b.prune_threshold

    def test_rbpf_config(self):
        f = RBPFFilter(config=RBPFConfig(max_particles=64))
        assert f.max_particles == 64

    def test_rbpf_config_equivalent_to_kwargs(self):
        a = RBPFFilter(
            config=RBPFConfig(
                max_particles=64, resample_threshold=0.4, merge_threshold=0.3
            )
        )
        b = RBPFFilter(max_particles=64, resample_threshold=0.4, merge_threshold=0.3)
        assert a.max_particles == b.max_particles
        assert a.resample_threshold == b.resample_threshold
        assert a.merge_threshold == b.merge_threshold

    def test_rbpf_defaults_match_kwargs_defaults(self):
        assert RBPFConfig() == RBPFConfig(
            max_particles=100, resample_threshold=0.5, merge_threshold=0.5
        )
