"""Typed msgspec configuration Structs for tracker classes."""

from typing import Optional

import msgspec
import numpy as np
from numpy.typing import ArrayLike


def _matrix(m: ArrayLike) -> list[list[float]]:
    return np.asarray(m, dtype=np.float64).tolist()


class SingleTargetConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.trackers.single_target.SingleTargetTracker`.

    ``F``/``Q`` are ``None`` when dynamics are supplied as callables at
    construction; such configs identify the tracker in a session snapshot
    but cannot rebuild it without the callables (rehydrate pattern).
    """

    state_dim: int
    meas_dim: int
    H: list[list[float]]
    R: list[list[float]]
    F: Optional[list[list[float]]] = None
    Q: Optional[list[list[float]]] = None
    gate_threshold: Optional[float] = None

    @classmethod
    def from_arrays(
        cls,
        state_dim: int,
        meas_dim: int,
        H: ArrayLike,
        R: ArrayLike,
        F: Optional[ArrayLike] = None,
        Q: Optional[ArrayLike] = None,
        gate_threshold: Optional[float] = None,
    ) -> "SingleTargetConfig":
        return cls(
            state_dim=state_dim,
            meas_dim=meas_dim,
            H=_matrix(H),
            R=_matrix(R),
            F=None if F is None else _matrix(F),
            Q=None if Q is None else _matrix(Q),
            gate_threshold=gate_threshold,
        )


class MultiTargetConfig(msgspec.Struct, frozen=True):
    """Configuration for :class:`~pytcl.trackers.multi_target.MultiTargetTracker`.

    Same ``F``/``Q`` convention as :class:`SingleTargetConfig`.
    """

    state_dim: int
    meas_dim: int
    H: list[list[float]]
    R: list[list[float]]
    F: Optional[list[list[float]]] = None
    Q: Optional[list[list[float]]] = None
    gate_probability: float = 0.99
    confirm_hits: int = 3
    confirm_window: int = 5
    max_misses: int = 5
    init_covariance: Optional[list[list[float]]] = None

    @classmethod
    def from_arrays(
        cls,
        state_dim: int,
        meas_dim: int,
        H: ArrayLike,
        R: ArrayLike,
        F: Optional[ArrayLike] = None,
        Q: Optional[ArrayLike] = None,
        gate_probability: float = 0.99,
        confirm_hits: int = 3,
        confirm_window: int = 5,
        max_misses: int = 5,
        init_covariance: Optional[ArrayLike] = None,
    ) -> "MultiTargetConfig":
        return cls(
            state_dim=state_dim,
            meas_dim=meas_dim,
            H=_matrix(H),
            R=_matrix(R),
            F=None if F is None else _matrix(F),
            Q=None if Q is None else _matrix(Q),
            gate_probability=gate_probability,
            confirm_hits=confirm_hits,
            confirm_window=confirm_window,
            max_misses=max_misses,
            init_covariance=(
                None if init_covariance is None else _matrix(init_covariance)
            ),
        )
