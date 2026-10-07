"""The library's two state-vector layouts, and what happens when they meet.

``two_point_diff_init`` is a port of MATLAB's ``twoPointDiffInit.m`` and
returns the derivative-major state ``[x, y, vx, vy]``. The dynamic models
(``f_poly_kal``, ``q_poly_kal``, ``f_constant_velocity``, ...) are
block-diagonal per dimension, so their state is interleaved
``[x, vx, y, vy]``. Both are internally correct; composing them raises
nothing and propagates a wrong state. The converter is deferred to a minor
release, so these tests pin the hazard.

The MATLAB reference values below were captured from the MATLAB TCL sources
(``FPolyKal``, ``QPolyKal``, ``twoPointDiffInit``, ``QCoordTurn``) run under
MATLAB R2026a, not computed from this package.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.batch_estimation import two_point_diff_init
from pytcl.dynamic_models import f_constant_velocity, f_poly_kal, q_poly_kal
from pytcl.dynamic_models.process_noise.coordinated_turn import (
    q_coord_turn_2d,
    q_coord_turn_3d,
)

Z = np.array([[0.0, 2.0], [1.0, 0.0]])

MATLAB_F_POLY_KAL = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
MATLAB_Q_POLY_KAL = np.array(
    [
        [1 / 3, 0.0, 0.5, 0.0],
        [0.0, 1 / 3, 0.0, 0.5],
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 1.0],
    ]
)
MATLAB_Q_COORD_TURN_5 = np.array(
    [
        [1.0, 1.0, 2.0, 2.0, 0.1],
        [1.0, 1.0, 2.0, 2.0, 0.1],
        [2.0, 2.0, 4.0, 4.0, 0.2],
        [2.0, 2.0, 4.0, 4.0, 0.2],
        [0.1, 0.1, 0.2, 0.2, 0.01],
    ]
)

INTERLEAVED_TO_MAJOR_2D = np.array([0, 2, 1, 3])


def _init_state():
    res = two_point_diff_init(1.0, Z, np.eye(2))
    return np.asarray(res.x).ravel()


def test_two_point_diff_init_returns_matlab_block_order():
    assert np.allclose(_init_state(), [2.0, 0.0, 2.0, -1.0])


def test_dynamic_models_use_interleaved_order():
    F = f_poly_kal(order=1, T=1.0, num_dims=2)
    x_interleaved = np.array([2.0, 2.0, 0.0, -1.0])
    assert np.allclose(F @ x_interleaved, [4.0, 2.0, -1.0, -1.0])


def test_constant_velocity_model_is_interleaved_like_poly_kal():
    assert np.array_equal(
        f_constant_velocity(T=1.0, num_dims=2), f_poly_kal(1, 1.0, num_dims=2)
    )


def test_the_two_orderings_are_not_interchangeable():
    F = f_poly_kal(order=1, T=1.0, num_dims=2)
    x_block = _init_state()
    correct_block_order = np.array([4.0, -1.0, 2.0, -1.0])
    assert not np.allclose(F @ x_block, correct_block_order)
    assert np.allclose(F @ x_block, [2.0, 0.0, 1.0, -1.0])


def test_reordering_the_initializer_output_restores_the_propagation():
    F = f_poly_kal(order=1, T=1.0, num_dims=2)
    x_block = _init_state()
    perm = np.arange(4).reshape(2, 2).T.ravel()
    x_prop_interleaved = F @ x_block[perm]
    x_prop_block = np.empty(4)
    x_prop_block[perm] = x_prop_interleaved
    assert np.allclose(x_prop_block, [4.0, -1.0, 2.0, -1.0])


def test_f_poly_kal_differs_from_matlab_elementwise_but_not_up_to_permutation():
    F = f_poly_kal(order=1, T=1.0, num_dims=2)
    assert np.max(np.abs(F - MATLAB_F_POLY_KAL)) == pytest.approx(1.0)
    p = INTERLEAVED_TO_MAJOR_2D
    assert np.allclose(F[np.ix_(p, p)], MATLAB_F_POLY_KAL, atol=1e-15)


def test_q_poly_kal_differs_from_matlab_elementwise_but_not_up_to_permutation():
    Q = q_poly_kal(order=1, T=1.0, q=1.0, num_dims=2)
    assert np.max(np.abs(Q - MATLAB_Q_POLY_KAL)) == pytest.approx(2 / 3)
    p = INTERLEAVED_TO_MAJOR_2D
    assert np.allclose(Q[np.ix_(p, p)], MATLAB_Q_POLY_KAL, atol=1e-15)


def test_matlab_reference_literals_are_not_the_package_output():
    assert not np.allclose(f_poly_kal(1, 1.0, num_dims=2), MATLAB_F_POLY_KAL)
    assert not np.allclose(q_poly_kal(1, 1.0, 1.0, num_dims=2), MATLAB_Q_POLY_KAL)


class TestCoordTurnDivergesFromMatlab:
    T = 1.0
    SIGMA_A = 2.0
    SIGMA_OMEGA = 0.1
    TO_MATLAB_5 = np.array([0, 2, 1, 3, 4])

    def _ours_5(self):
        Q = q_coord_turn_2d(
            self.T, self.SIGMA_A, self.SIGMA_OMEGA, "position_velocity_omega"
        )
        return Q[np.ix_(self.TO_MATLAB_5, self.TO_MATLAB_5)]

    def test_matlab_reference_is_rank_one(self):
        assert np.linalg.matrix_rank(MATLAB_Q_COORD_TURN_5) == 1

    def test_ranks(self):
        q4 = q_coord_turn_2d(self.T, self.SIGMA_A, self.SIGMA_OMEGA)
        q5 = q_coord_turn_2d(
            self.T, self.SIGMA_A, self.SIGMA_OMEGA, "position_velocity_omega"
        )
        assert np.linalg.matrix_rank(q4) == 2
        assert np.linalg.matrix_rank(q5) == 3
        assert np.allclose(np.linalg.eigvalsh(q4), [0, 0, 5, 5])
        assert np.allclose(np.linalg.eigvalsh(q5), [0, 0, 0.01, 5, 5])

    def test_diagonals_agree_but_matrices_do_not(self):
        ours = self._ours_5()
        assert np.allclose(np.diag(ours), np.diag(MATLAB_Q_COORD_TURN_5))
        assert np.max(np.abs(ours - MATLAB_Q_COORD_TURN_5)) == pytest.approx(4.0)

    def test_position_velocity_block_differs_from_matlab_leading_block(self):
        q4 = q_coord_turn_2d(self.T, self.SIGMA_A)
        p = INTERLEAVED_TO_MAJOR_2D
        diff = q4[np.ix_(p, p)] - MATLAB_Q_COORD_TURN_5[:4, :4]
        assert np.max(np.abs(diff)) == pytest.approx(4.0)

    def test_has_no_cross_axis_or_omega_coupling(self):
        q5 = q_coord_turn_2d(
            self.T, self.SIGMA_A, self.SIGMA_OMEGA, "position_velocity_omega"
        )
        assert np.all(q5[0:2, 2:4] == 0.0)
        assert np.all(q5[0:4, 4] == 0.0)

    def test_3d_has_the_same_structure(self):
        q7 = q_coord_turn_3d(
            self.T, self.SIGMA_A, self.SIGMA_OMEGA, "position_velocity_omega"
        )
        assert np.linalg.matrix_rank(q7) == 4
        assert np.all(q7[0:2, 2:4] == 0.0)
