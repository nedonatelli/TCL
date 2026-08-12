"""Shared Hypothesis strategies for pytcl's property tests.

Centralized here (rather than duplicated per test module) so later
property suites -- coordinate round trips, filter covariance properties,
assignment optimality, etc. -- draw from the same float/array/track
generators instead of each reinventing subtly different bounds.

Import from this module directly: ``from tests.property._strategies import
float64_arrays, track_histories``.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from pytcl.io.serialize import SimpleTrack

FINITE_FLOAT_KWARGS = dict(allow_nan=False, allow_infinity=False, width=64)


def finite_floats() -> st.SearchStrategy:
    """A strategy for finite (no NaN/inf) float64-representable floats."""
    return st.floats(**FINITE_FLOAT_KWARGS)


def any_floats() -> st.SearchStrategy:
    """A strategy for arbitrary float64-representable floats, incl. NaN/inf."""
    return st.floats(allow_nan=True, allow_infinity=True, width=64)


@st.composite
def float64_arrays(
    draw,
    *,
    min_size: int = 1,
    max_size: int = 8,
    finite_only: bool = True,
) -> np.ndarray:
    """A 1-D ``float64`` ndarray with length in ``[min_size, max_size]``."""
    element = finite_floats() if finite_only else any_floats()
    values = draw(st.lists(element, min_size=min_size, max_size=max_size))
    return np.array(values, dtype=np.float64)


@st.composite
def track_histories(draw, *, finite_only: bool):
    """Generated ``(history, times)`` with a uniform state dimension.

    ``history`` is a list of scans (each a list of `SimpleTrack`); ``times``
    is the parallel list of scan timestamps -- the shape consumed by
    ``pytcl.io.serialize.encode_tracks`` and
    ``pytcl.io.asdf_io.save_tracks_asdf``.
    """
    dim = draw(st.integers(min_value=1, max_value=6))
    n_scans = draw(st.integers(min_value=1, max_value=5))
    element = finite_floats() if finite_only else any_floats()
    history, times = [], []
    for k in range(n_scans):
        n_tracks = draw(st.integers(min_value=0, max_value=3))
        scan = []
        for tid in range(n_tracks):
            state = np.array(
                draw(st.lists(element, min_size=dim, max_size=dim)), dtype=np.float64
            )
            cov = np.array(
                draw(st.lists(element, min_size=dim * dim, max_size=dim * dim)),
                dtype=np.float64,
            ).reshape(dim, dim)
            scan.append(
                SimpleTrack(id=tid, state=state, covariance=cov, status="confirmed")
            )
        history.append(scan)
        times.append(float(k))
    return history, times
