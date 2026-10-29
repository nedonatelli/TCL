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
from pytcl.dynamic_models.process_noise.polynomial import (
    q_constant_acceleration,
    q_constant_velocity,
    q_continuous_white_noise,
    q_discrete_white_noise,
    q_poly_kal,
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


def _polynomial_fixtures(name, n_params):
    rows = np.loadtxt(FIXTURE_DIR / f"{name}.csv", delimiter=",", ndmin=2)
    n = round(np.sqrt(rows.shape[1] - n_params))
    return [
        pytest.param(
            SimpleNamespace(params=r[:n_params], Q=r[n_params:].reshape(n, n)),
            id="-".join(f"{v:g}" for v in r[:n_params]),
        )
        for r in rows
    ]


def _derivative_major_to_interleaved(Q, num_dims):
    """Permute a MATLAB derivative-major Q ``[x,y,vx,vy,..]`` to pytcl's
    per-dimension blocks ``[x,vx,y,vy,..]``."""
    n = Q.shape[0] // num_dims
    p = np.array([k * num_dims + d for d in range(num_dims) for k in range(n)])
    return Q[np.ix_(p, p)]


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_q_poly_kal_matches_qpolykal(order):
    for case in [
        c.values[0] for c in _polynomial_fixtures(f"qpolykal_order{order}", 2)
    ]:
        T, q0 = case.params
        np.testing.assert_allclose(
            q_poly_kal(order, T, q0, num_dims=2),
            _derivative_major_to_interleaved(case.Q, 2),
            rtol=1e-14,
            atol=0,
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_q_continuous_white_noise_matches_qpolykal(order):
    for case in [
        c.values[0] for c in _polynomial_fixtures(f"qpolykal_order{order}", 2)
    ]:
        T, q0 = case.params
        np.testing.assert_allclose(
            q_continuous_white_noise(order + 1, T, q0, block_size=2),
            _derivative_major_to_interleaved(case.Q, 2),
            rtol=1e-14,
            atol=0,
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_q_discrete_white_noise_matches_qpolykaldirectdisc(order):
    for case in [
        c.values[0] for c in _polynomial_fixtures(f"qpolykaldirectdisc_order{order}", 2)
    ]:
        T, var = case.params
        np.testing.assert_allclose(
            q_discrete_white_noise(order + 1, T, var, block_size=2),
            _derivative_major_to_interleaved(case.Q, 2),
            rtol=1e-14,
            atol=0,
        )


def test_q_constant_velocity_matches_qpolykaldirectdisc():
    for case in [
        c.values[0] for c in _polynomial_fixtures("qpolykaldirectdisc_order1", 2)
    ]:
        T, var = case.params
        np.testing.assert_allclose(
            q_constant_velocity(T, np.sqrt(var), num_dims=2),
            _derivative_major_to_interleaved(case.Q, 2),
            rtol=1e-14,
            atol=0,
        )


def test_q_constant_acceleration_matches_qpolykaldirectdisc():
    for case in [
        c.values[0] for c in _polynomial_fixtures("qpolykaldirectdisc_order2", 2)
    ]:
        T, var = case.params
        np.testing.assert_allclose(
            q_constant_acceleration(T, np.sqrt(var), num_dims=2),
            _derivative_major_to_interleaved(case.Q, 2),
            rtol=1e-14,
            atol=0,
        )


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
    got = q_coord_turn_2d(case.T, np.sqrt(case.sigmaV2))
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
    got = q_coord_turn_3d(case.T, np.sqrt(case.sigmaV2))
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


@pytest.mark.parametrize("fn", [q_coord_turn_2d, q_coord_turn_3d])
def test_unknown_state_type_raises(fn):
    with pytest.raises(ValueError, match="position_velocity_omega"):
        fn(1.0, 2.0, state_type="position_velocity_omgea")


@pytest.mark.parametrize("fn", [q_coord_turn_2d, q_coord_turn_3d])
def test_ignored_sigma_omega_raises(fn):
    with pytest.raises(ValueError, match="state_type='position_velocity_omega'"):
        fn(1.0, 2.0, 0.1)


@pytest.mark.parametrize("fn", [q_coord_turn_2d, q_coord_turn_3d])
def test_zero_sigma_omega_without_omega_state_is_accepted(fn):
    np.testing.assert_array_equal(fn(1.0, 2.0, 0.0), fn(1.0, 2.0))


# q_singer is QGaussMarkov at order 2 with q = 2*sigma_m^2/tau. The fixtures
# hold T/tau >= 0.05 only: below that q_singer switches to a Taylor series and
# QGaussMarkov's closed form loses digits to cancellation, so it is not an
# oracle there (see the literal-value test below).
SINGER_CASES = _polynomial_fixtures("qgaussmarkov_order2", 3)


@pytest.mark.parametrize("case", SINGER_CASES)
def test_q_singer_2d_matches_qgaussmarkov(case):
    T, tau, sigma_m = case.params
    np.testing.assert_allclose(
        q_singer_2d(T, tau, sigma_m),
        _derivative_major_to_interleaved(case.Q, 2),
        rtol=1e-9,
        atol=0,
    )


@pytest.mark.parametrize("case", SINGER_CASES)
def test_q_singer_matches_qgaussmarkov_per_dimension(case):
    T, tau, sigma_m = case.params
    one_dim = case.Q[np.ix_([0, 2, 4], [0, 2, 4])]
    np.testing.assert_allclose(
        q_singer(T, tau, sigma_m, num_dims=1), one_dim, rtol=1e-9, atol=0
    )
    expected_3d = np.kron(np.eye(3), one_dim)
    np.testing.assert_allclose(
        q_singer_3d(T, tau, sigma_m), expected_3d, rtol=1e-9, atol=0
    )


@pytest.mark.parametrize(
    "T, tau, sigma_m, Q00, Q01, Q02, Q11, Q12, Q22",
    [
        (0.1, 10, 0.5, 2.4861605757240815e-8, 6.208506390361202e-7,
         8.2504565334176941e-6, 1.6542247922801873e-5, 0.00024751452104798771,
         0.0049503316733111747),
        (0.5, 100, 3, 0.00028047014313846411, 0.0014015722500207281,
         0.0037313014610963312, 0.0074719405079852716, 0.022387827423084182,
         0.089551496257487518),
    ],
)  # fmt: skip
def test_q_singer_taylor_branch_matches_integral(
    T, tau, sigma_m, Q00, Q01, Q02, Q11, Q12, Q22
):
    # T/tau < 0.05. Oracle: 50-digit mpmath quadrature of
    # (2 sigma_m^2/tau) * int_0^T g(s) g(s)' ds with the impulse response
    # g = [tau*s - tau^2*(1-e^(-s/tau)), tau*(1-e^(-s/tau)), e^(-s/tau)].
    expected = np.array([[Q00, Q01, Q02], [Q01, Q11, Q12], [Q02, Q12, Q22]])
    np.testing.assert_allclose(q_singer(T, tau, sigma_m), expected, rtol=1e-9, atol=0)


def test_q_coord_turn_polar_literal_values():
    # pytcl-only model: MATLAB's QPolarCoordTurn2D is a different model (its
    # state carries speed and heading with correlated terms), so there is no
    # MATLAB oracle. Hand-computed from the documented formulas at T=2,
    # sigma_a=3, sigma_omega_dot=0.5 (T^2=4, T^3=8, T^4=16).
    Q = q_coord_turn_polar(T=2.0, sigma_a=3.0, sigma_omega_dot=0.5)
    expected = np.zeros((5, 5))
    expected[0, 0] = expected[1, 1] = 9.0 * 16 / 4  # 36
    expected[2, 2] = 0.25 * 16 / 4  # 1
    expected[2, 4] = expected[4, 2] = 0.25 * 8 / 2  # 1
    expected[3, 3] = 9.0 * 4  # 36
    expected[4, 4] = 0.25 * 4  # 1
    np.testing.assert_allclose(Q, expected, rtol=1e-15, atol=0)
