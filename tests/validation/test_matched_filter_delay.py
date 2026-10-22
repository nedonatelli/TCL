"""
Ambiguity and cross-ambiguity delay indexing at exact-sample delays.

The delay grid is a linspace of seconds; each delay is mapped back to a sample
index by ``int(delay * fs)``. Truncation turns a delay that is exactly k samples
into k-1 whenever the floating-point product falls just below k.
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.signal_processing.matched_filter import (
    ambiguity_function,
    cross_ambiguity,
)

_N_HALF = 32


def _lfm_chirp(n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    duration = n / fs
    f0, f1 = 0.0, 2000.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2)
    return np.cos(phase)


def test_ambiguity_at_integer_sample_delays_equals_the_autocorrelation():
    fs = 1e4
    x = _lfm_chirp(n=256, fs=fs)
    acf = np.correlate(x, x, mode="full") / np.dot(x, x)
    centre = len(x) - 1
    _, _, af = ambiguity_function(
        x,
        fs=fs,
        max_delay=_N_HALF / fs,
        max_doppler=fs / 2,
        n_delay=2 * _N_HALF + 1,
        n_doppler=3,
    )
    for k in (0, 1, 5, 17):
        got = af[1, _N_HALF + k]
        assert abs(got) == pytest.approx(abs(acf[centre + k]), rel=1e-9, abs=0), (
            f"k={k}"
        )


def test_cross_ambiguity_at_integer_sample_delays_equals_the_cross_correlation():
    fs = 1e4
    x = _lfm_chirp(n=256, fs=fs)
    ccf = np.correlate(x, x, mode="full") / np.dot(x, x)
    centre = len(x) - 1
    _, _, caf = cross_ambiguity(
        x,
        x,
        fs,
        max_delay=_N_HALF / fs,
        max_doppler=fs / 2,
        n_delay=2 * _N_HALF + 1,
        n_doppler=3,
    )
    for k in (0, 1, 5, 17):
        got = caf[1, _N_HALF + k]
        assert abs(got) == pytest.approx(abs(ccf[centre + k]), rel=1e-9, abs=0), (
            f"k={k}"
        )
