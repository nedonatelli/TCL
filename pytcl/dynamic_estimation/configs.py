"""Typed msgspec configuration Structs for estimation classes."""

from typing import Optional

import msgspec
import numpy as np
from numpy.typing import ArrayLike


def _matrix(m: ArrayLike) -> list:
    return np.asarray(m, dtype=np.float64).tolist()


class IMMConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.dynamic_estimation.imm.IMMEstimator`.

    Attributes
    ----------
    n_modes : int
        Number of filter modes.
    state_dim : int
        Dimension of the state vector.
    transition_matrix : list of list of float, shape (n_modes, n_modes)
        Mode transition probability matrix.
    initial_mode_probs : list of float or None, optional
        Initial mode probabilities; ``None`` for a uniform prior.
    """

    n_modes: int
    state_dim: int
    transition_matrix: list[list[float]]  # (n_modes, n_modes) nested lists
    initial_mode_probs: Optional[list[float]] = None

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
    """Configuration for :class:`~pytcl.dynamic_estimation.gaussian_sum_filter.GaussianSumFilter`.

    Attributes
    ----------
    max_components : int, default 5
        Maximum number of Gaussian mixture components retained after
        pruning and merging.
    merge_threshold : float, default 0.01
        KL-divergence threshold below which two components are merged.
    prune_threshold : float, default 1e-3
        Minimum component weight below which a component is discarded.
    """

    max_components: int = 5
    merge_threshold: float = 0.01
    prune_threshold: float = 1e-3


class RBPFConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.dynamic_estimation.rbpf.RBPFFilter`.

    Attributes
    ----------
    max_particles : int, default 100
        Number of particles in the filter.
    resample_threshold : float, default 0.5
        Effective-sample-size fraction (of `max_particles`) below which
        particles are resampled.
    merge_threshold : float, default 0.5
        KL-divergence threshold below which nearby particles are merged.
    """

    max_particles: int = 100
    resample_threshold: float = 0.5
    merge_threshold: float = 0.5
