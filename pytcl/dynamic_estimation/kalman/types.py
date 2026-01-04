"""
Type definitions for Kalman filter implementations.

This module provides shared NamedTuple types used across multiple Kalman
filter implementations. Separating types into their own module prevents
circular imports between filter implementations.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class SRKalmanState(NamedTuple):
    """State of a square-root Kalman filter.

    Attributes
    ----------
    x : ndarray
        State estimate.
    S : ndarray
        Lower triangular Cholesky factor of covariance (P = S @ S.T).
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]


class SRKalmanPrediction(NamedTuple):
    """Result of square-root Kalman filter prediction step.

    Attributes
    ----------
    x : ndarray
        Predicted state estimate.
    S : ndarray
        Lower triangular Cholesky factor of predicted covariance.
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]


class SRKalmanUpdate(NamedTuple):
    """Result of square-root Kalman filter update step.

    Attributes
    ----------
    x : ndarray
        Updated state estimate.
    S : ndarray
        Lower triangular Cholesky factor of updated covariance.
    y : ndarray
        Innovation (measurement residual).
    S_y : ndarray
        Lower triangular Cholesky factor of innovation covariance.
    K : ndarray
        Kalman gain.
    likelihood : float
        Measurement likelihood (for association).
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]
    y: NDArray[np.floating]
    S_y: NDArray[np.floating]
    K: NDArray[np.floating]
    likelihood: float


class UDState(NamedTuple):
    """State of a U-D factorization filter.

    The covariance is represented as P = U @ D @ U.T where U is
    unit upper triangular and D is diagonal.

    Attributes
    ----------
    x : ndarray
        State estimate.
    U : ndarray
        Unit upper triangular factor.
    D : ndarray
        Diagonal elements (1D array).
    """

    x: NDArray[np.floating]
    U: NDArray[np.floating]
    D: NDArray[np.floating]


__all__ = [
    "SRKalmanState",
    "SRKalmanPrediction",
    "SRKalmanUpdate",
    "UDState",
]
