"""
Process noise covariance matrices for coordinated turn models.
"""

import numpy as np
from numpy.typing import NDArray


def q_coord_turn_2d(
    T: float,
    sigma_a: float,
    sigma_omega: float = 0.0,
    state_type: str = "position_velocity",
) -> NDArray[np.floating]:
    """
    Create process noise covariance for 2D coordinated turn model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_a : float
        Standard deviation of acceleration noise [m/s²].
    sigma_omega : float, optional
        Standard deviation of turn rate noise [rad/s²].
        Only used if state_type includes omega.
    state_type : str, optional
        State vector composition:
        - 'position_velocity': [x, vx, y, vy] (default)
        - 'position_velocity_omega': [x, vx, y, vy, omega]

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    Examples
    --------
    >>> Q = q_coord_turn_2d(T=1.0, sigma_a=1.0)
    >>> Q.shape
    (4, 4)

    Notes
    -----
    The process noise accounts for uncertainty in the turn dynamics.
    For the coordinated turn, this includes uncertainty in both
    the linear acceleration and the turn rate.

    See Also
    --------
    f_coord_turn_2d : State transition matrix for 2D coordinated turn.
    """
    var_a = sigma_a**2

    # Discrete white noise acceleration model for x and y
    Q_pos_vel = var_a * np.array(
        [
            [T**4 / 4, T**3 / 2],
            [T**3 / 2, T**2],
        ],
        dtype=np.float64,
    )

    if state_type == "position_velocity_omega":
        n = 5
        Q = np.zeros((n, n), dtype=np.float64)
        # x, vx block
        Q[0:2, 0:2] = Q_pos_vel
        # y, vy block
        Q[2:4, 2:4] = Q_pos_vel
        # omega (turn rate uncertainty)
        Q[4, 4] = sigma_omega**2 * T**2
    else:
        n = 4
        Q = np.zeros((n, n), dtype=np.float64)
        # x, vx block
        Q[0:2, 0:2] = Q_pos_vel
        # y, vy block
        Q[2:4, 2:4] = Q_pos_vel

    return Q


def q_coord_turn_3d(
    T: float,
    sigma_a: float,
    sigma_omega: float = 0.0,
    state_type: str = "position_velocity",
) -> NDArray[np.floating]:
    """
    Create process noise covariance for 3D coordinated turn model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_a : float
        Standard deviation of acceleration noise [m/s²].
    sigma_omega : float, optional
        Standard deviation of turn rate noise [rad/s²].
    state_type : str, optional
        State vector composition.

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    See Also
    --------
    f_coord_turn_3d : State transition matrix for 3D coordinated turn.
    """
    var_a = sigma_a**2

    # Discrete white noise acceleration model
    Q_pos_vel = var_a * np.array(
        [
            [T**4 / 4, T**3 / 2],
            [T**3 / 2, T**2],
        ],
        dtype=np.float64,
    )

    if state_type == "position_velocity_omega":
        n = 7
        Q = np.zeros((n, n), dtype=np.float64)
        # x, vx block
        Q[0:2, 0:2] = Q_pos_vel
        # y, vy block
        Q[2:4, 2:4] = Q_pos_vel
        # z, vz block
        Q[4:6, 4:6] = Q_pos_vel
        # omega (turn rate uncertainty)
        Q[6, 6] = sigma_omega**2 * T**2
    else:
        n = 6
        Q = np.zeros((n, n), dtype=np.float64)
        # x, vx block
        Q[0:2, 0:2] = Q_pos_vel
        # y, vy block
        Q[2:4, 2:4] = Q_pos_vel
        # z, vz block
        Q[4:6, 4:6] = Q_pos_vel

    return Q


def q_coord_turn_polar(
    T: float,
    sigma_a: float,
    sigma_omega_dot: float,
) -> NDArray[np.floating]:
    """
    Create process noise covariance for coordinated turn in polar form.

    State vector is [x, y, heading, speed, turn_rate].

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
