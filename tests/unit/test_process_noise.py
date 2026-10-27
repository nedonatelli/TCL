"""
Tests for process noise models.

Tests cover:
- Singer process noise covariance (1D, 2D, 3D)
- Coordinated turn process noise (2D, 3D, polar)
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pytcl.dynamic_models.process_noise.coordinated_turn import (
    q_coord_turn_2d,
    q_coord_turn_3d,
    q_coord_turn_polar,
)
from pytcl.dynamic_models.process_noise.singer import (
    q_singer,
    q_singer_2d,
    q_singer_3d,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# MATLAB's QCoordTurn state is derivative-major; pytcl's is interleaved.
MATLAB_TO_INTERLEAVED = {
    5: np.array([0, 2, 1, 3, 4]),
    7: np.array([0, 3, 1, 4, 2, 5, 6]),
}


def _qcoordturn_fixtures(name):
    path = FIXTURE_DIR / f"qcoordturn_{name}.csv"
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    n = round(np.sqrt(rows.shape[1] - 3))
    return [
        pytest.param(
            SimpleNamespace(
                T=r[0], sigmaV2=r[1], sigmaTurn2=r[2], Q=r[3:].reshape(n, n)
            ),
            id=f"T={r[0]}-sigmaV2={r[1]}-sigmaTurn2={r[2]}",
        )
        for r in rows
    ]


def _to_interleaved(Q):
    p = MATLAB_TO_INTERLEAVED[Q.shape[0]]
    return Q[np.ix_(p, p)]


def _to_matlab_order(Q):
    p = np.argsort(MATLAB_TO_INTERLEAVED[Q.shape[0]])
    return Q[np.ix_(p, p)]


@pytest.mark.parametrize("case", _qcoordturn_fixtures("case5"))
def test_2d_matches_qcoordturn(case):
    got = q_coord_turn_2d(
        case.T,
        np.sqrt(case.sigmaV2),
        np.sqrt(case.sigmaTurn2),
        state_type="position_velocity_omega",
    )
    np.testing.assert_allclose(got, _to_interleaved(case.Q), rtol=1e-15, atol=0)


@pytest.mark.parametrize("case", _qcoordturn_fixtures("case5"))
def test_2d_position_velocity_is_leading_block(case):
    got = q_coord_turn_2d(case.T, np.sqrt(case.sigmaV2), np.sqrt(case.sigmaTurn2))
    np.testing.assert_allclose(got, _to_interleaved(case.Q)[:4, :4], rtol=1e-15, atol=0)


@pytest.mark.parametrize("case", _qcoordturn_fixtures("case7"))
def test_3d_matches_qcoordturn(case):
    got = q_coord_turn_3d(
        case.T,
        np.sqrt(case.sigmaV2),
        np.sqrt(case.sigmaTurn2),
        state_type="position_velocity_omega",
    )
    np.testing.assert_allclose(got, _to_interleaved(case.Q), rtol=1e-15, atol=0)


@pytest.mark.parametrize("case", _qcoordturn_fixtures("case7"))
def test_3d_position_velocity_is_leading_block(case):
    got = q_coord_turn_3d(case.T, np.sqrt(case.sigmaV2), np.sqrt(case.sigmaTurn2))
    np.testing.assert_allclose(got, _to_interleaved(case.Q)[:6, :6], rtol=1e-15, atol=0)


def test_matlab_example_point():
    Q = q_coord_turn_2d(1.0, 2.0, 0.1, state_type="position_velocity_omega")
    np.testing.assert_allclose(
        _to_matlab_order(Q),
        [
            [1, 1, 2, 2, 0.1],
            [1, 1, 2, 2, 0.1],
            [2, 2, 4, 4, 0.2],
            [2, 2, 4, 4, 0.2],
            [0.1, 0.1, 0.2, 0.2, 0.01],
        ],
        rtol=1e-15,
        atol=0,
    )
    assert np.linalg.matrix_rank(Q) == 1


class TestSingerProcessNoise:
    """Tests for Singer process noise model."""

    def test_q_singer_1d(self):
        """Test Singer process noise covariance 1D."""
        T = 0.1
        tau = 10.0
        sigma_m = 5.0
        Q = q_singer(T, tau, sigma_m, num_dims=1)
        assert Q.shape == (3, 3)
        np.testing.assert_allclose(Q, Q.T)
        assert np.all(np.diag(Q) >= 0)

    def test_q_singer_2d(self):
        """Test Singer process noise covariance 2D."""
        Q = q_singer_2d(T=0.1, tau=10.0, sigma_m=5.0)
        assert Q.shape == (6, 6)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_singer_3d(self):
        """Test Singer process noise covariance 3D."""
        Q = q_singer_3d(T=0.1, tau=10.0, sigma_m=5.0)
        assert Q.shape == (9, 9)
        np.testing.assert_allclose(Q, Q.T)


class TestCoordinatedTurnProcessNoise:
    """Tests for coordinated turn process noise."""

    def test_q_coord_turn_polar(self):
        """Test polar form coordinated turn process noise."""
        Q = q_coord_turn_polar(T=0.1, sigma_a=1.0, sigma_omega_dot=0.01)
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T)
