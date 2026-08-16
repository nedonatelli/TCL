"""Typed msgspec configuration Structs for estimation classes."""

from typing import Optional

import msgspec
import numpy as np
from numpy.typing import ArrayLike


def _matrix(m: ArrayLike) -> list:
    return np.asarray(m, dtype=np.float64).tolist()


class IMMConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.dynamic_estimation.imm.IMMEstimator`."""

    n_modes: int
    state_dim: int
    transition_matrix: list  # (n_modes, n_modes) nested lists
    initial_mode_probs: Optional[list] = None

    @classmethod
    def from_arrays(
        cls,
        n_modes: int,
        state_dim: int,
        transition_matrix: ArrayLike,
        initial_mode_probs: Optional[ArrayLike] = None,
    ) -> "IMMConfig":
        return cls(
            n_modes=n_modes,
            state_dim=state_dim,
            transition_matrix=_matrix(transition_matrix),
            initial_mode_probs=(
                None
                if initial_mode_probs is None
                else np.asarray(initial_mode_probs, dtype=np.float64).tolist()
            ),
        )


class GaussianSumConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.dynamic_estimation.gaussian_sum_filter.GaussianSumFilter`."""

    max_components: int = 5
    merge_threshold: float = 0.01
    prune_threshold: float = 1e-3


class RBPFConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.dynamic_estimation.rbpf.RBPFFilter`."""

    max_particles: int = 100
    resample_threshold: float = 0.5
    merge_threshold: float = 0.5
