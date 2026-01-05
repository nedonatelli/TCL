"""
Coordinated turn motion models.

This module provides state transition matrices for 2D and 3D coordinated
turn models, commonly used for tracking maneuvering aircraft.
"""

import numpy as np
from numpy.typing import NDArray


def f_coord_turn_2d(
    T: float,
    omega: float,
    state_type: str = "position_velocity",
) -> NDArray[np.floating]:
    """
    Create state transition matrix for 2D coordinated turn model.

    In a coordinated turn, the target moves in a circular arc with constant
    turn rate (angular velocity omega).

    Parameters
    ----------
    T : float
        Time step in seconds.
    omega : float
        Turn rate in radians per second.
        Positive = counterclockwise, negative = clockwise.
    state_type : str, optional
        State vector composition:
        - 'position_velocity': [x, vx, y, vy] (default)
        - 'position_velocity_omega': [x, vx, y, vy, omega]

    Returns
    -------
    F : ndarray
        State transition matrix.

    Examples
    --------
    >>> import numpy as np
    >>> # Constant velocity (omega=0) reduces to CV model
    >>> F = f_coord_turn_2d(T=1.0, omega=0.0)
    >>> F
    array([[1., 1., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 0., 1.]])

    >>> # 90 deg/s turn rate
    >>> F = f_coord_turn_2d(T=1.0, omega=np.pi/2)

    Notes
    -----
    For the 2D coordinated turn, the dynamics are:
        x' = x + (sin(omega*T)/omega)*vx - ((1-cos(omega*T))/omega)*vy
        vx' = cos(omega*T)*vx - sin(omega*T)*vy
        y' = y + ((1-cos(omega*T))/omega)*vx + (sin(omega*T)/omega)*vy
        vy' = sin(omega*T)*vx + cos(omega*T)*vy

    When omega is near zero, L'Hopital's rule gives the CV model.

    See Also
    --------
    f_coord_turn_3d : 3D coordinated turn model.
    """
    # Handle near-zero turn rate (approaches constant velocity)
    if np.abs(omega) < 1e-10:
        if state_type == "position_velocity_omega":
            F = np.array(
                [
                    [1, T, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, T, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
        else:
            F = np.array(
                [
                    [1, T, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, T],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
        return F

    # Compute trigonometric terms
    sin_wT = np.sin(omega * T)
    cos_wT = np.cos(omega * T)

    # Position coupling coefficients
    s_over_w = sin_wT / omega  # sin(wT)/w
    c_minus_1_over_w = (1 - cos_wT) / omega  # (1-cos(wT))/w

    if state_type == "position_velocity_omega":
        # State: [x, vx, y, vy, omega]
        F = np.array(
            [
                [1, s_over_w, 0, -c_minus_1_over_w, 0],
                [0, cos_wT, 0, -sin_wT, 0],
                [0, c_minus_1_over_w, 1, s_over_w, 0],
                [0, sin_wT, 0, cos_wT, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
    else:
        # State: [x, vx, y, vy]
        F = np.array(
            [
                [1, s_over_w, 0, -c_minus_1_over_w],
                [0, cos_wT, 0, -sin_wT],
                [0, c_minus_1_over_w, 1, s_over_w],
                [0, sin_wT, 0, cos_wT],
            ],
            dtype=np.float64,
        )

    return F


def f_coord_turn_3d(
    T: float,
    omega: float,
    state_type: str = "position_velocity",
) -> NDArray[np.floating]:
    """
    Create state transition matrix for 3D coordinated turn model.

    In the 3D case, the turn occurs in the x-y plane while z motion
    follows constant velocity dynamics.

    Parameters
    ----------
    T : float
        Time step in seconds.
    omega : float
        Turn rate in radians per second (in x-y plane).
    state_type : str, optional
        State vector composition:
        - 'position_velocity': [x, vx, y, vy, z, vz] (default)
        - 'position_velocity_omega': [x, vx, y, vy, z, vz, omega]

    Returns
    -------
    F : ndarray
        State transition matrix.

    Examples
    --------
    >>> F = f_coord_turn_3d(T=1.0, omega=0.1)
    >>> F.shape
    (6, 6)

    See Also
    --------
    f_coord_turn_2d : 2D coordinated turn model.
    """
    # Get 2D coordinated turn matrix
    if state_type == "position_velocity_omega":
        F_2d = f_coord_turn_2d(T, omega, "position_velocity_omega")
        # Expand to 3D: insert z, vz before omega
        n = 7
        F = np.zeros((n, n), dtype=np.float64)

        # Copy x, vx, y, vy block (4x4 from top-left of F_2d, excluding omega)
        F[:2, :2] = F_2d[:2, :2]  # x, vx
        F[:2, 2:4] = F_2d[:2, 2:4]  # coupling to y, vy
        F[2:4, :2] = F_2d[2:4, :2]  # y, vy coupling to x, vx
        F[2:4, 2:4] = F_2d[2:4, 2:4]  # y, vy

        # z dynamics (constant velocity)
        F[4, 4] = 1.0
        F[4, 5] = T
        F[5, 5] = 1.0

        # omega unchanged
        F[6, 6] = 1.0
    else:
        F_2d = f_coord_turn_2d(T, omega, "position_velocity")
        # Expand to 3D
        n = 6
        F = np.zeros((n, n), dtype=np.float64)

        # Copy 2D turn model for x, vx, y, vy
        F[:4, :4] = F_2d

        # z dynamics (constant velocity)
        F[4, 4] = 1.0
        F[4, 5] = T
        F[5, 5] = 1.0

    return F


def f_coord_turn_polar(
    T: float,
    omega: float,
    speed: float,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for coordinated turn in polar form.

    State vector is [x, y, heading, speed, turn_rate].

    Parameters
    ----------
    T : float
        Time step in seconds.
    omega : float
        Turn rate in radians per second.
    speed : float
        Speed magnitude in m/s.

    Returns
    -------
    F : ndarray
        State transition matrix of shape (5, 5).

    Notes
    -----
    This is a linearized model around the current state. For accurate
    propagation, use the nonlinear equations directly.

    The state is [x, y, psi, v, omega] where:
    - psi is heading angle
    - v is speed magnitude
    - omega is turn rate

    Examples
    --------
    >>> F = f_coord_turn_polar(T=1.0, omega=0.1, speed=100.0)
    >>> F.shape
    (5, 5)
    """
    # Handle near-zero turn rate
    if np.abs(omega) < 1e-10:
        # Straight line motion
        F = np.array(
            [
                [1, 0, 0, T, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, T],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        return F

    sin_wT = np.sin(omega * T)
    cos_wT = np.cos(omega * T)

    # Position changes due to curved path
    # This is a simplified linearization
    F = np.array(
        [
            [1, 0, 0, sin_wT / omega, speed * (cos_wT - 1) / omega**2],
            [0, 1, 0, (1 - cos_wT) / omega, speed * (sin_wT - omega * T) / omega**2],
            [0, 0, 1, 0, T],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    return F


__all__ = [
    "f_coord_turn_2d",
    "f_coord_turn_3d",
    "f_coord_turn_polar",
]
