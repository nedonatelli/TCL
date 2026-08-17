"""Round-trip properties for pytcl.io session save/restore.

Target: ``pytcl.io.save_session``/``load_session`` for
:class:`~pytcl.trackers.SingleTargetTracker` and
:class:`~pytcl.dynamic_estimation.IMMEstimator`. Both properties are
msgpack-only -- the JSON wire format's non-finite-value contract is already
covered by example tests in ``tests/unit/test_io_session.py``.

The IMM property always runs one predict/update cycle before the round
trip, so its invariant is "resume mid-track" -- the interesting bytes are
the ones a Kalman update actually touched. The single-target property
draws its step count from 0..4, so most cases cover the same "resume
mid-track" invariant but the steps=0 case instead round-trips a tracker
right after `initialize()`, with no predict/update applied -- covering
"restore a freshly initialized object" too.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pytcl.dynamic_estimation import IMMEstimator
from pytcl.io import load_session, save_session
from pytcl.trackers import SingleTargetTracker


def _spd_matrix(rng, k):
    a = rng.normal(size=(k, k))
    return a @ a.T + k * np.eye(k)


@st.composite
def _single_target_states(draw):
    state_dim = draw(st.integers(2, 6))
    meas_dim = draw(st.integers(1, state_dim))
    seed = draw(st.integers(0, 2**32 - 1))
    steps = draw(st.integers(0, 4))
    return state_dim, meas_dim, seed, steps


@given(_single_target_states())
def test_single_target_roundtrip_bit_exact(params):
    state_dim, meas_dim, seed, steps = params
    rng = np.random.Generator(np.random.PCG64(seed))
    t = SingleTargetTracker(
        state_dim,
        meas_dim,
        np.eye(state_dim),
        np.eye(meas_dim, state_dim),
        0.01 * np.eye(state_dim),
        _spd_matrix(rng, meas_dim),
    )
    t.initialize(rng.normal(size=state_dim), _spd_matrix(rng, state_dim))
    for _ in range(steps):
        t.predict(1.0)
        t.update(rng.normal(size=meas_dim))

    back = load_session(save_session(t))
    assert back.state.state.tobytes() == t.state.state.tobytes()
    assert back.state.covariance.tobytes() == t.state.covariance.tobytes()


@st.composite
def _imm_states(draw):
    n_modes = draw(st.integers(2, 3))
    state_dim = draw(st.integers(2, 5))
    meas_dim = draw(st.integers(1, state_dim))
    seed = draw(st.integers(0, 2**32 - 1))
    return n_modes, state_dim, meas_dim, seed


@given(_imm_states())
def test_imm_roundtrip_bit_exact(params):
    n_modes, state_dim, meas_dim, seed = params
    rng = np.random.Generator(np.random.PCG64(seed))

    # Row-stochastic by construction (each row is a Dirichlet draw), so no
    # extra normalization is needed for a valid transition matrix.
    transition = rng.dirichlet(np.ones(n_modes), size=n_modes)
    e = IMMEstimator(n_modes, state_dim, transition)
    for i in range(n_modes):
        # Identity-plus-small-perturbation keeps F well-conditioned so the
        # predict/update cycle below stays numerically sane regardless of
        # what Hypothesis draws for state_dim.
        F = np.eye(state_dim) + 0.01 * rng.normal(size=(state_dim, state_dim))
        Q = 0.01 * _spd_matrix(rng, state_dim)
        e.set_mode_model(i, F, Q)
    e.set_measurement_model(np.eye(meas_dim, state_dim), _spd_matrix(rng, meas_dim))
    e.initialize(rng.normal(size=state_dim), _spd_matrix(rng, state_dim))

    e.predict()
    e.update(rng.normal(size=meas_dim))

    back = load_session(save_session(e))
    assert back.x.tobytes() == e.x.tobytes()
    assert back.P.tobytes() == e.P.tobytes()
    assert back.mode_probs.tobytes() == e.mode_probs.tobytes()
