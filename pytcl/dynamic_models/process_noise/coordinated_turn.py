"""
Process noise covariance matrices for coordinated turn models.

This module provides functions to construct process noise covariance matrices
(Q matrices) for coordinated turn motion models used in target tracking.
Coordinated turn models describe targets that maintain a constant turn rate,
such as aircraft executing banking maneuvers.

The process noise captures uncertainty in the target's motion, including:
- Linear acceleration uncertainty along the velocity direction
- Turn rate (angular velocity) uncertainty

Available functions:
- ``q_coord_turn_2d``: 2D coordinated turn (planar motion)
- ``q_coord_turn_3d``: 3D coordinated turn (spatial motion)
- ``q_coord_turn_polar``: Polar/heading-speed representation

These Q matrices are designed to work with the corresponding state transition
matrices in :mod:`pytcl.dynamic_models.discrete_time.coordinated_turn`.

Examples
--------
Create process noise for a 2D coordinated turn tracker:

>>> from pytcl.dynamic_models.process_noise import q_coord_turn_2d
>>> Q = q_coord_turn_2d(T=0.1, sigma_a=2.0)  # 0.1s step, 2 m/s² accel noise
>>> Q.shape
(4, 4)

With turn rate state estimation:

>>> Q = q_coord_turn_2d(T=0.1, sigma_a=2.0, sigma_omega=0.01,
...                     state_type='position_velocity_omega')
>>> Q.shape
(5, 5)

See Also
--------
pytcl.dynamic_models.discrete_time.coordinated_turn : State transition matrices
pytcl.dynamic_models.process_noise.polynomial : CV/CA process noise

References
----------
- Bar-Shalom, Y., Li, X.R., and Kirubarajan, T., "Estimation with
  Applications to Tracking and Navigation", Wiley, 2001.
- Blackman, S. and Popoli, R., "Design and Analysis of Modern
  Tracking Systems", Artech House, 1999.
"""

import numpy as np
from numpy.typing import NDArray

_STATE_TYPES = ("position_velocity", "position_velocity_omega")


def _q_coord_turn(
    T: float,
    sigma_a: float,
    sigma_omega: float | None,
    num_dims: int,
    state_type: str,
) -> NDArray[np.floating]:
    if state_type not in _STATE_TYPES:
        raise ValueError(
            f"state_type must be one of {_STATE_TYPES}, got {state_type!r}"
        )
    G = np.tile(np.array([T**2 / 2, T]) * sigma_a, num_dims)
    if state_type == "position_velocity_omega":
        G = np.append(G, T * (0.0 if sigma_omega is None else sigma_omega))
    elif sigma_omega is not None and sigma_omega != 0:
        raise ValueError(
            "sigma_omega has no effect without a turn-rate state; pass "
            "state_type='position_velocity_omega'"
        )
    return np.outer(G, G)


def q_coord_turn_2d(
    T: float,
    sigma_a: float,
    sigma_omega: float | None = None,
    state_type: str = "position_velocity",
) -> NDArray[np.floating]:
    """
    Create process noise covariance for 2D coordinated turn model.

    Port of MATLAB TCL's ``QCoordTurn`` (2-D, no linear acceleration).

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_a : float
        Standard deviation of the noise driving the velocity components
        [m/s²]. ``QCoordTurn``'s ``sigmaV2`` is ``sigma_a**2``.
    sigma_omega : float, optional
        Standard deviation of turn rate noise [rad/s²]. ``QCoordTurn``'s
        ``sigmaTurn2`` is ``sigma_omega**2``. Requires
        ``state_type='position_velocity_omega'`` unless it is 0; omitted
        there, it is 0.
    state_type : str, optional
        State vector composition:
        - 'position_velocity': [x, vx, y, vy] (default)
        - 'position_velocity_omega': [x, vx, y, vy, omega]

        Any other value raises ``ValueError``.

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix, rank 1.

    Raises
    ------
    ValueError
        If ``state_type`` is not one of the values above, or if a nonzero
        ``sigma_omega`` is given with ``state_type='position_velocity'``.

    Examples
    --------
    >>> import numpy as np
    >>> Q = q_coord_turn_2d(T=1.0, sigma_a=2.0, sigma_omega=0.1,
    ...                     state_type='position_velocity_omega')
    >>> Q[0]
    array([1. , 2. , 1. , 2. , 0.1])
    >>> int(np.linalg.matrix_rank(Q))
    1

    Notes
    -----
    ``Q = G G'`` with the single column
    ``G = [sigma_a*T**2/2, sigma_a*T, sigma_a*T**2/2, sigma_a*T, sigma_omega*T]``,
    as in ``QCoordTurn``. Every x, y, velocity and omega pair is correlated
    and ``Q`` has rank 1, so ``np.linalg.cholesky(Q)`` fails; a square-root
    filter should take ``G`` itself as the noise factor.

    ``QCoordTurn`` orders its state ``[x, y, xdot, ydot, omega]``; this
    function uses the interleaved ``[x, vx, y, vy, omega]`` of
    :func:`f_coord_turn_2d`. Permuting this result with indices
    ``[0, 2, 1, 3, 4]`` gives ``QCoordTurn``'s matrix (``QCoordTurn`` with
    ``sigmaV2 = sigma_a**2`` evaluates ``sqrt(sigmaV2)``, which is
    ``sigma_a`` only up to rounding).

    ``QCoordTurn`` rejects 4-element states, so
    ``state_type='position_velocity'`` returns the leading 4x4 block of the
    ``'position_velocity_omega'`` result. ``QCoordTurn``'s length-6 branch (adds a
    linear-acceleration state) has no counterpart here.

    Until v2.12.0 this function stacked independent per-axis
    discrete-white-noise-acceleration blocks instead (rank 2 for
    ``'position_velocity'``, 3 with omega; largest elementwise difference
    from ``QCoordTurn`` 4.0 at ``T=1, sigma_a=2, sigma_omega=0.1``).

    See Also
    --------
    f_coord_turn_2d : State transition matrix for 2D coordinated turn.
    """
    return _q_coord_turn(T, sigma_a, sigma_omega, 2, state_type)


def q_coord_turn_3d(
    T: float,
    sigma_a: float,
    sigma_omega: float | None = None,
    state_type: str = "position_velocity",
) -> NDArray[np.floating]:
    """
    Create process noise covariance for 3D coordinated turn model.

    Port of MATLAB TCL's ``QCoordTurn`` (3-D, no linear acceleration).

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_a : float
        Standard deviation of the noise driving the velocity components
        [m/s²]. ``QCoordTurn``'s ``sigmaV2`` is ``sigma_a**2``.
    sigma_omega : float, optional
        Standard deviation of turn rate noise [rad/s²]. ``QCoordTurn``'s
        ``sigmaTurn2`` is ``sigma_omega**2``. Requires
        ``state_type='position_velocity_omega'`` unless it is 0; omitted
        there, it is 0.
    state_type : str, optional
        State vector composition:
        - 'position_velocity': [x, vx, y, vy, z, vz] (default)
        - 'position_velocity_omega': [x, vx, y, vy, z, vz, omega]

        Any other value raises ``ValueError``.

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix, rank 1.

    Raises
    ------
    ValueError
        If ``state_type`` is not one of the values above, or if a nonzero
        ``sigma_omega`` is given with ``state_type='position_velocity'``.

    Notes
    -----
    ``Q = G G'`` with the single column
    ``G = [sigma_a*T**2/2, sigma_a*T] * 3 + [sigma_omega*T]`` (one
    position/velocity pair per axis, then omega), as in ``QCoordTurn``'s
    3-D branch. ``Q`` has rank 1, so ``np.linalg.cholesky(Q)`` fails; a
    square-root filter should take ``G`` itself as the noise factor.

    ``QCoordTurn`` orders its 3-D state ``[x, y, z, xdot, ydot, zdot,
    omega]``; this function uses the interleaved
    ``[x, vx, y, vy, z, vz, omega]``. Permuting this result with indices
    ``[0, 2, 4, 1, 3, 5, 6]`` gives ``QCoordTurn``'s matrix.
    ``state_type='position_velocity'`` returns the leading 6x6 block of the
    ``'position_velocity_omega'`` result (``QCoordTurn`` rejects 6-element
    states). The length-8 linear-acceleration branch has no counterpart
    here.

    Until v2.12.0 this function stacked independent per-axis blocks (rank 3
    without omega, 4 with it).

    See Also
    --------
    f_coord_turn_3d : State transition matrix for 3D coordinated turn.
    """
    return _q_coord_turn(T, sigma_a, sigma_omega, 3, state_type)


def q_coord_turn_polar(
    T: float,
    sigma_a: float,
    sigma_omega_dot: float,
) -> NDArray[np.floating]:
    """
    Create process noise covariance for coordinated turn in polar form.

    State vector is [x, y, heading, speed, turn_rate].

    Notes
    -----
    MATLAB parity has **not** been verified for this function; it is not
    ``QCoordTurn`` (whose state is Cartesian) and is not compared against
    any MATLAB reference. The module's EXPERIMENTAL maturity reflects that
    gap.

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_a : float
        Standard deviation of tangential acceleration [m/s²].
    sigma_omega_dot : float
        Standard deviation of turn rate derivative [rad/s²].

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix of shape (5, 5).
    """
    Q = np.zeros((5, 5), dtype=np.float64)

    # Position noise (from speed uncertainty)
    # This is a simplified model
    Q[0, 0] = sigma_a**2 * T**4 / 4
    Q[1, 1] = sigma_a**2 * T**4 / 4

    # Heading noise (from turn rate uncertainty)
    Q[2, 2] = sigma_omega_dot**2 * T**4 / 4
    Q[2, 4] = sigma_omega_dot**2 * T**3 / 2
    Q[4, 2] = Q[2, 4]

    # Speed noise
    Q[3, 3] = sigma_a**2 * T**2

    # Turn rate noise
    Q[4, 4] = sigma_omega_dot**2 * T**2

    return Q


__all__ = [
    "q_coord_turn_2d",
    "q_coord_turn_3d",
    "q_coord_turn_polar",
]
