"""Numeric oracle for ``ins_error_state_matrix``.

``ins_error_state_matrix`` claims to be the continuous-time linearization
of :func:`pytcl.navigation.ins.mechanize_ins_ned`. Before v2.11.1 no test
checked that claim -- only shape and "some entries are nonzero"
(``tests/unit/test_ins.py``). This file builds an independent oracle by
central-differencing ``mechanize_ins_ned`` itself, in the same tangent-space
coordinates the F matrix uses (lat/lon/alt, NED velocity, and attitude as a
nav-frame rotation-vector perturbation), and checks every entry of the 9x9
navigation block (rows/cols 0-8) that is non-zero in the analytic matrix,
*or* non-zero in the oracle even where the analytic matrix has it as an
implicit zero (a missing self-coupling term) -- the union of the two, not
either alone. 29 entries meet that test: 14 in ``_AGREEING_ENTRIES``, 15 in
``_KNOWN_WRONG_ENTRIES``. Of the 15, 8 are analytic-zero/oracle-non-zero
(``F[4,4]`` in the velocity block; ``F[6,0]`` and the six off-diagonal
``-[omega_in^n x] phi`` self-coupling entries in the attitude block).

Every entry that agrees is asserted at a tight tolerance. Every entry that
does not is an explicit ``xfail(strict=True)`` case carrying the analytic
value, the numeric value, and why -- so the inventory is visible rather
than silently wrong, and so a future fix flips the xfail to a failure that
demands the inventory be updated, rather than passing unnoticed.

The bias-coupling columns (``F[3:6,9:12]``, ``F[6:9,12:15]``) and rows 9-14
generally are not covered here at all: :func:`mechanize_ins_ned` takes no
bias arguments, so no finite difference of it can reach them. Their
correctness is unknown, not verified by this file's green entries.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.rotations import quat_multiply
from pytcl.navigation.geodesy import WGS84
from pytcl.navigation.ins import (
    IMUData,
    INSState,
    earth_rate_ned,
    ins_error_state_matrix,
    mechanize_ins_ned,
    normal_gravity,
    transport_rate_ned,
)

# Step sizes are chosen per component and verified (in the task-3.2 report,
# not re-run on every test invocation) to be stable across a couple of
# orders of magnitude of eps and across the range of mechanization dt this
# file actually exercises (1e-4 to 1e-3 s; see _MECHANIZATION_DT below) --
# these are not tuned to make any particular entry pass. That range does
# NOT extend down to dt=1e-5: at dt=1e-5, F[1,0]'s agreement with the
# oracle degrades from rel ~9.8e-3 to rel ~3-5e-1, an order-of-magnitude
# worse match, because the mechanization's own dt-discretization error
# stops being negligible next to the finite-difference perturbation at
# that step size. The claim above is scoped to dt=1e-4..1e-3, not to
# mechanization dt generally.
_EPS = {
    "lat": 1e-6,  # rad; ~6 mm on the ground
    "lon": 1e-6,  # rad
    "alt": 1.0,  # m
    "vel": 1e-3,  # m/s
    "phi": 1e-6,  # rad, nav-frame rotation-vector perturbation
}
_EPS_BY_COLUMN = [
    _EPS["lat"],
    _EPS["lon"],
    _EPS["alt"],
    _EPS["vel"],
    _EPS["vel"],
    _EPS["vel"],
    _EPS["phi"],
    _EPS["phi"],
    _EPS["phi"],
]
_MECHANIZATION_DT = 1e-4  # s; small enough that dt-discretization bias is <1e-9


def _rotation_vector(delta_q_axis: int, eps: float) -> np.ndarray:
    v = np.zeros(3)
    v[delta_q_axis] = eps
    angle = np.linalg.norm(v)
    axis = v / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _perturb_state(state: INSState, k: int, eps: float) -> INSState:
    position = np.array(state.position, dtype=np.float64)
    velocity = np.array(state.velocity, dtype=np.float64)
    quaternion = np.array(state.quaternion, dtype=np.float64)
    if k < 3:
        position = position.copy()
        position[k] += eps
    elif k < 6:
        velocity = velocity.copy()
        velocity[k - 3] += eps
    else:
        q_delta = _rotation_vector(k - 6, eps)
        quaternion = quat_multiply(q_delta, quaternion)
        quaternion = quaternion / np.linalg.norm(quaternion)
    return INSState(
        position=position, velocity=velocity, quaternion=quaternion, time=state.time
    )


def _state_rate(
    state: INSState,
    gyro_b: np.ndarray,
    accel_b: np.ndarray,
    ellipsoid,
    dt: float,
) -> np.ndarray:
    """9-vector [lat_dot, lon_dot, alt_dot, vN_dot, vE_dot, vD_dot, wx, wy, wz].

    ``w`` is the nav-frame rotation-vector rate between the mechanized
    quaternion and the input quaternion, recovered from ``R1 @ R0.T`` --
    the same nav-frame convention used to perturb attitude in
    ``_perturb_state``, so the resulting Jacobian lines up column-for-column
    with ``ins_error_state_matrix``'s attitude columns.
    """
    imu = IMUData(accel=accel_b, gyro=gyro_b, dt=dt)
    new_state = mechanize_ins_ned(state, imu, ellipsoid)
    lat_dot = (new_state.position[0] - state.position[0]) / dt
    lon_dot = (new_state.position[1] - state.position[1]) / dt
    alt_dot = (new_state.position[2] - state.position[2]) / dt
    v_dot = (np.array(new_state.velocity) - np.array(state.velocity)) / dt
    r0 = state.dcm
    r1 = new_state.dcm
    r_delta = r1 @ r0.T
    rotvec = 0.5 * np.array(
        [
            r_delta[2, 1] - r_delta[1, 2],
            r_delta[0, 2] - r_delta[2, 0],
            r_delta[1, 0] - r_delta[0, 1],
        ]
    )
    return np.concatenate([[lat_dot, lon_dot, alt_dot], v_dot, rotvec / dt])


def numeric_jacobian(state: INSState, ellipsoid=WGS84) -> np.ndarray:
    """Central-difference Jacobian of ``mechanize_ins_ned``'s state rate.

    The IMU driving the mechanization is fixed across every perturbation,
    computed once at ``state``:

    - ``accel_b`` is the body-frame reading for a wings-level, non-
      maneuvering specific force (``[0, 0, -g]`` in the nav frame). This
      is also what ``ins_error_state_matrix``'s own
      ``F[3,7] = -g`` / ``F[4,6] = g`` simplification assumes, so it is
      the fair condition for judging that approximation rather than a
      harsher, unrepresentative maneuver.
    - ``gyro_b`` is the body-frame reading that holds the body frame
      coincident with the (rotating) nav frame, i.e. zero body rate
      relative to nav -- the natural base point to linearize the attitude
      channel around.

    Returns
    -------
    J : ndarray
        9x9 Jacobian in the same [lat, lon, alt, vN, vE, vD, phi_N, phi_E,
        phi_D] ordering as ``ins_error_state_matrix``'s rows/columns 0-8.
    """
    lat, lon, alt = state.position
    vN, vE, vD = state.velocity
    r_n_b = state.dcm.T
    g = normal_gravity(lat, alt)
    omega_in_n = earth_rate_ned(lat) + transport_rate_ned(lat, alt, vN, vE, ellipsoid)
    gyro_b = r_n_b @ omega_in_n
    accel_b = r_n_b @ np.array([0.0, 0.0, -g])

    jacobian = np.zeros((9, 9))
    for k in range(9):
        eps = _EPS_BY_COLUMN[k]
        plus = _perturb_state(state, k, +eps)
        minus = _perturb_state(state, k, -eps)
        rate_plus = _state_rate(plus, gyro_b, accel_b, ellipsoid, _MECHANIZATION_DT)
        rate_minus = _state_rate(minus, gyro_b, accel_b, ellipsoid, _MECHANIZATION_DT)
        jacobian[:, k] = (rate_plus - rate_minus) / (2 * eps)
    return jacobian


@pytest.fixture(scope="module")
def canonical_state() -> INSState:
    return INSState(
        position=np.array([np.radians(40.0), np.radians(-105.0), 1600.0]),
        velocity=np.array([120.0, 85.0, -3.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        time=0.0,
    )


@pytest.fixture(scope="module")
def analytic_and_numeric(canonical_state):
    analytic = ins_error_state_matrix(canonical_state, WGS84)
    numeric = numeric_jacobian(canonical_state, WGS84)
    return analytic, numeric


# F[3,3] (d(vN_dot)/d(vN)): the oracle gives a small, sign-consistent but
# NOT stable value here -- 0 at eps=1e-4 (below the mechanization's own
# resolution at that step), then -4.3e-7 to -4.7e-7 across eps=1e-3..1.0
# and -7.1e-7 to -4.3e-7 across dt=1e-5..1e-3. That is two orders of
# magnitude below F[4,4]'s stable 1.53e-5 (below at every eps/dt tried,
# never the dominant signal) and consistent with central-difference
# truncation/roundoff noise on a curvature-scale term rather than a real,
# resolvable self-coupling. Decision: treated as noise, not added to
# _KNOWN_WRONG_ENTRIES -- recorded here explicitly rather than left
# silently uncounted, which is how F[4,4] itself was originally missed.

# (row, col): rtol for entries that agree with the oracle.
_AGREEING_ENTRIES = {
    (0, 3): 2e-2,
    # Agrees at rel 9.77e-3 against rtol=2e-2 -- a real ~1% disagreement
    # with only a factor-2 margin, not clean agreement. This entry lives
    # in TestVerticalChannelFixed, so if a future mechanization change
    # trips it, read the failure as "this entry's margin ran out", not as
    # "the vertical-channel fix broke".
    (1, 0): 2e-2,
    (1, 4): 2e-2,
    (2, 5): 2e-2,
    (3, 5): 2e-2,
    (3, 7): 2e-2,
    (4, 3): 2e-2,
    (4, 5): 2e-2,
    (4, 6): 2e-2,
    (5, 0): 2e-2,  # fixed this version: sign flip + missing dg/dlat
    (5, 2): 2e-2,  # fixed this version: sign flip (Schuler divergence)
    (5, 3): 2e-2,  # fixed this version: missing Coriolis/transport term
    (5, 4): 2e-2,
    (8, 0): 2e-2,
}

# (row, col): (reason, how far off) for entries known to disagree with the
# oracle. Values are informational -- the xfail itself is what's load-
# bearing (strict=True: a future fix must update this table, not silently
# start passing).
_KNOWN_WRONG_ENTRIES = {
    (3, 0): "velocity block: missing curvature correction, ~17% low",
    (3, 4): "velocity block: missing curvature correction, ~9.6% low",
    (4, 0): "velocity block: missing curvature correction, ~20% low",
    (4, 4): (
        "velocity block: d(vE_dot)/d(vE) self-coupling entirely missing "
        "(analytic 0.0, oracle +1.5277e-05, stable across eps=1e-4..1e-2 "
        "and dt=1e-5..1e-3 -- same kind and same order as the attitude "
        "self-coupling entries below, not finite-difference noise)"
    ),
    (6, 0): "attitude block: transport-rate coupling term entirely missing",
    (6, 4): "attitude block: wrong sign",
    (6, 7): "attitude block: -[omega_in^n x] phi self-coupling missing",
    (6, 8): "attitude block: -[omega_in^n x] phi self-coupling missing",
    (7, 0): "attitude block: wrong term, ~250x off and wrong sign",
    (7, 3): "attitude block: wrong sign",
    (7, 6): "attitude block: -[omega_in^n x] phi self-coupling missing",
    (7, 8): "attitude block: -[omega_in^n x] phi self-coupling missing",
    (8, 4): "attitude block: wrong sign",
    (8, 6): "attitude block: -[omega_in^n x] phi self-coupling missing",
    (8, 7): "attitude block: -[omega_in^n x] phi self-coupling missing",
}


class TestVerticalChannelFixed:
    """PROPERTY-class checks for the three entries fixed in v2.11.1.

    The invariant checked is that F equals the Jacobian of the function it
    claims to linearize (mechanize_ins_ned), not a comparison against an
    independent implementation or published values -- that makes this
    PROPERTY class per CONTRIBUTING's table, not REFERENCE.
    """

    @pytest.mark.parametrize("row,col", sorted(_AGREEING_ENTRIES))
    def test_entry_matches_oracle(self, analytic_and_numeric, row, col):
        analytic, numeric = analytic_and_numeric
        rtol = _AGREEING_ENTRIES[(row, col)]
        assert analytic[row, col] == pytest.approx(
            numeric[row, col], rel=rtol, abs=1e-9
        ), (
            f"F[{row},{col}]: analytic={analytic[row, col]!r} "
            f"numeric={numeric[row, col]!r}"
        )

    def test_vertical_channel_is_schuler_divergent_not_oscillatory(self):
        """The physical vertical channel diverges with tau ~ 570 s.

        Before this version F[5,2] had the opposite sign, making the {alt,
        vD} subsystem purely oscillatory (eigenvalues +-0.00175j) instead
        of divergent -- qualitatively, not just numerically, wrong.
        """
        state = INSState(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            time=0.0,
        )
        F = ins_error_state_matrix(state)
        sub = F[np.ix_([2, 5], [2, 5])]
        eigs = np.linalg.eigvals(sub)
        assert np.max(eigs.real) > 1e-4, f"vertical channel is not divergent: {eigs}"
        assert 400.0 < 1.0 / np.max(eigs.real) < 800.0


class TestKnownWrongEntries:
    """Enumerated, tracked-not-hidden inventory of entries that still
    disagree with the oracle after v2.11.1.

    These are deliberately left unfixed: the attitude block needs the
    transport-rate/earth-rate self-coupling derived and validated as a
    unit (its own tier of work); most of the velocity-row entries here are
    incomplete higher-order corrections, not sign errors or missing terms,
    so widening a tolerance to pass them would hide a real, if small,
    disagreement rather than document it -- the exception is ``F[4,4]``,
    an entirely missing ``d(vE_dot)/d(vE)`` self-coupling of the same kind
    and order as the attitude block's missing self-coupling, tracked here
    rather than fixed for the same patch-release-scope reason.
    """

    @pytest.mark.parametrize("row,col", sorted(_KNOWN_WRONG_ENTRIES))
    @pytest.mark.xfail(
        strict=True, reason="see _KNOWN_WRONG_ENTRIES; tracked, not fixed"
    )
    def test_entry_disagrees_with_oracle(self, analytic_and_numeric, row, col):
        analytic, numeric = analytic_and_numeric
        rtol = 2e-2
        assert analytic[row, col] == pytest.approx(
            numeric[row, col], rel=rtol, abs=1e-9
        ), (
            f"F[{row},{col}]: analytic={analytic[row, col]!r} "
            f"numeric={numeric[row, col]!r} ({_KNOWN_WRONG_ENTRIES[(row, col)]})"
        )
