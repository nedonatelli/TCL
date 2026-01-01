"""
Estimation performance metrics.

This module provides metrics for evaluating state estimation performance,
including RMSE, NEES, NIS, and consistency tests.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with
       Applications to Tracking and Navigation," Wiley, 2001.
"""

from typing import List, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class ConsistencyResult(NamedTuple):
    """
    Result of consistency test.

    Attributes
    ----------
    is_consistent : bool
        Whether the estimator is consistent.
    statistic : float
        Test statistic value.
    lower_bound : float
        Lower confidence bound.
    upper_bound : float
        Upper confidence bound.
    mean_value : float
        Mean of the test statistic.
    """

    is_consistent: bool
    statistic: float
    lower_bound: float
    upper_bound: float
    mean_value: float


def rmse(
    true_states: NDArray[np.float64],
    estimated_states: NDArray[np.float64],
    axis: Optional[int] = None,
) -> NDArray[np.float64] | float:
    """
    Compute Root Mean Square Error.

    Parameters
    ----------
    true_states : ndarray
        True state values, shape (N, state_dim) or (N,).
    estimated_states : ndarray
        Estimated state values, same shape as true_states.
    axis : int, optional
        Axis over which to compute RMSE.
        - None: RMSE over all elements (scalar result)
        - 0: RMSE for each state component (vector result)
        - 1: RMSE for each time step (vector result)

    Returns
    -------
    ndarray or float
        Root mean square error.

    Examples
    --------
    >>> true = np.array([[0, 0], [1, 1], [2, 2]])
    >>> est = np.array([[0.1, -0.1], [1.2, 0.9], [1.8, 2.1]])
    >>> rmse(true, est)  # Scalar RMSE  # doctest: +SKIP
    0.158...
    >>> rmse(true, est, axis=0)  # Per-component RMSE  # doctest: +SKIP
    array([0.152..., 0.115...])
    """
    true_states = np.asarray(true_states)
    estimated_states = np.asarray(estimated_states)

    errors = true_states - estimated_states
    mse = np.mean(errors**2, axis=axis)
    return np.sqrt(mse)


def position_rmse(
    true_states: NDArray[np.float64],
    estimated_states: NDArray[np.float64],
    position_indices: List[int],
) -> float:
    """
    Compute RMSE for position components only.

    Parameters
    ----------
    true_states : ndarray
        True state values, shape (N, state_dim).
    estimated_states : ndarray
        Estimated state values, shape (N, state_dim).
    position_indices : list of int
        Indices of position components in state vector.

    Returns
    -------
    float
        Position RMSE.

    Examples
    --------
    >>> # State = [x, vx, y, vy], positions are indices [0, 2]
    >>> true = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
    >>> est = np.array([[0.1, 1, -0.1, 1], [1.2, 1, 0.9, 1]])
    >>> position_rmse(true, est, [0, 2])  # doctest: +SKIP
    0.141...
    """
    true_pos = true_states[:, position_indices]
    est_pos = estimated_states[:, position_indices]
    return float(rmse(true_pos, est_pos))


def velocity_rmse(
    true_states: NDArray[np.float64],
    estimated_states: NDArray[np.float64],
    velocity_indices: List[int],
) -> float:
    """
    Compute RMSE for velocity components only.

    Parameters
    ----------
    true_states : ndarray
        True state values, shape (N, state_dim).
    estimated_states : ndarray
        Estimated state values, shape (N, state_dim).
    velocity_indices : list of int
        Indices of velocity components in state vector.

    Returns
    -------
    float
        Velocity RMSE.
    """
    true_vel = true_states[:, velocity_indices]
    est_vel = estimated_states[:, velocity_indices]
    return float(rmse(true_vel, est_vel))


def nees(
    true_state: NDArray[np.float64],
    estimated_state: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> float:
    """
    Compute Normalized Estimation Error Squared (NEES).

    NEES is a measure of filter consistency. For a properly tuned filter,
    the average NEES should be close to the state dimension.

    Parameters
    ----------
    true_state : ndarray
        True state vector, shape (state_dim,).
    estimated_state : ndarray
        Estimated state vector, shape (state_dim,).
    covariance : ndarray
        Estimation covariance, shape (state_dim, state_dim).

    Returns
    -------
    float
        NEES value (chi-squared distributed with df=state_dim).

    Notes
    -----
    NEES = (x_true - x_est)' * P^{-1} * (x_true - x_est)

    For a consistent filter, NEES should follow a chi-squared distribution
    with degrees of freedom equal to the state dimension.

    Examples
    --------
    >>> true = np.array([1.0, 2.0])
    >>> est = np.array([1.1, 1.9])
    >>> P = np.eye(2) * 0.1
    >>> nees(true, est, P)
    0.2
    """
    true_state = np.asarray(true_state)
    estimated_state = np.asarray(estimated_state)
    covariance = np.asarray(covariance)

    error = true_state - estimated_state
    P_inv = np.linalg.inv(covariance)
    return float(error @ P_inv @ error)


def nees_sequence(
    true_states: NDArray[np.float64],
    estimated_states: NDArray[np.float64],
    covariances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute NEES for a sequence of estimates.

    Parameters
    ----------
    true_states : ndarray
        True states, shape (N, state_dim).
    estimated_states : ndarray
        Estimated states, shape (N, state_dim).
    covariances : ndarray
        Covariances, shape (N, state_dim, state_dim).

    Returns
    -------
    ndarray
        NEES values for each time step, shape (N,).
    """
    N = true_states.shape[0]
    nees_values = np.zeros(N)

    for k in range(N):
        nees_values[k] = nees(true_states[k], estimated_states[k], covariances[k])

    return nees_values


def average_nees(
    true_states: NDArray[np.float64],
    estimated_states: NDArray[np.float64],
    covariances: NDArray[np.float64],
) -> float:
    """
    Compute average NEES over a sequence.

    Parameters
    ----------
    true_states : ndarray
        True states, shape (N, state_dim).
    estimated_states : ndarray
        Estimated states, shape (N, state_dim).
    covariances : ndarray
        Covariances, shape (N, state_dim, state_dim).

    Returns
    -------
    float
        Average NEES (should be close to state_dim for consistent filter).
    """
    return float(np.mean(nees_sequence(true_states, estimated_states, covariances)))


def nis(
    innovation: NDArray[np.float64],
    innovation_covariance: NDArray[np.float64],
) -> float:
    """
    Compute Normalized Innovation Squared (NIS).

    NIS is similar to NEES but computed in measurement space.
    Used to verify measurement model consistency.

    Parameters
    ----------
    innovation : ndarray
        Innovation (measurement residual) vector.
    innovation_covariance : ndarray
        Innovation covariance matrix S.

    Returns
    -------
    float
        NIS value (chi-squared distributed with df=meas_dim).

    Notes
    -----
    NIS = nu' * S^{-1} * nu

    where nu = z - H*x_pred is the innovation and S is the innovation
    covariance.
    """
    innovation = np.asarray(innovation)
    innovation_covariance = np.asarray(innovation_covariance)

    S_inv = np.linalg.inv(innovation_covariance)
    return float(innovation @ S_inv @ innovation)


def nis_sequence(
    innovations: NDArray[np.float64],
    innovation_covariances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute NIS for a sequence of innovations.

    Parameters
    ----------
    innovations : ndarray
        Innovation vectors, shape (N, meas_dim).
    innovation_covariances : ndarray
        Innovation covariances, shape (N, meas_dim, meas_dim).

    Returns
    -------
    ndarray
        NIS values for each time step.
    """
    N = innovations.shape[0]
    nis_values = np.zeros(N)

    for k in range(N):
        nis_values[k] = nis(innovations[k], innovation_covariances[k])

    return nis_values


def consistency_test(
    nees_or_nis_values: NDArray[np.float64],
    df: int,
    confidence: float = 0.95,
) -> ConsistencyResult:
    """
    Perform chi-squared consistency test on NEES or NIS values.

    Tests whether the average NEES/NIS falls within expected confidence
    bounds for a consistent estimator.

    Parameters
    ----------
    nees_or_nis_values : ndarray
        NEES or NIS values from multiple time steps or Monte Carlo runs.
    df : int
        Degrees of freedom (state_dim for NEES, meas_dim for NIS).
    confidence : float, optional
        Confidence level (default: 0.95).

    Returns
    -------
    ConsistencyResult
        Named tuple with test results.

    Examples
    --------
    >>> np.random.seed(42)
    >>> # Simulate NEES from chi-squared (consistent filter)
    >>> nees_vals = np.random.chisquare(df=4, size=100)
    >>> result = consistency_test(nees_vals, df=4)
    >>> result.is_consistent
    True
    """
    N = len(nees_or_nis_values)
    mean_val = np.mean(nees_or_nis_values)

    # Average NEES/NIS * N follows chi-squared with N*df degrees of freedom
    # We test if the sample mean is within confidence bounds
    alpha = 1 - confidence

    # Bounds for average value
    lower_bound = stats.chi2.ppf(alpha / 2, N * df) / N
    upper_bound = stats.chi2.ppf(1 - alpha / 2, N * df) / N

    is_consistent = lower_bound <= mean_val <= upper_bound

    return ConsistencyResult(
        is_consistent=is_consistent,
        statistic=mean_val,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        mean_value=mean_val,
    )


def credibility_interval(
    errors: NDArray[np.float64],
    covariances: NDArray[np.float64],
    interval: float = 0.95,
) -> float:
    """
    Compute fraction of errors within credibility interval.

    For a consistent estimator, approximately `interval` fraction of the
    errors should fall within the corresponding credibility region.

    Parameters
    ----------
    errors : ndarray
        Estimation errors, shape (N, state_dim).
    covariances : ndarray
        Covariances, shape (N, state_dim, state_dim).
    interval : float, optional
        Credibility interval (default: 0.95).

    Returns
    -------
    float
        Fraction of errors within the interval.
    """
    N = len(errors)
    state_dim = errors.shape[1]

    # Chi-squared threshold for the interval
    threshold = stats.chi2.ppf(interval, state_dim)

    count_within = 0
    for k in range(N):
        nees_val = nees(np.zeros(state_dim), errors[k], covariances[k])
        if nees_val <= threshold:
            count_within += 1

    return count_within / N


def monte_carlo_rmse(
    errors: NDArray[np.float64],
    axis: int = 0,
) -> NDArray[np.float64]:
    """
    Compute RMSE from Monte Carlo simulation errors.

    Parameters
    ----------
    errors : ndarray
        Estimation errors from multiple runs, shape (N_runs, N_time, state_dim)
        or (N_runs, state_dim).
    axis : int, optional
        Axis representing Monte Carlo runs (default: 0).

    Returns
    -------
    ndarray
        RMSE values.
    """
    return np.sqrt(np.mean(errors**2, axis=axis))


def estimation_error_bounds(
    covariances: NDArray[np.float64],
    sigma: float = 2.0,
) -> NDArray[np.float64]:
    """
    Compute estimation error bounds from covariances.

    Parameters
    ----------
    covariances : ndarray
        Covariance matrices, shape (N, state_dim, state_dim).
    sigma : float, optional
        Number of standard deviations for bounds (default: 2.0).

    Returns
    -------
    ndarray
        Error bounds (standard deviations) for each component,
        shape (N, state_dim).
    """
    # Extract diagonal elements (variances)
    variances = np.diagonal(covariances, axis1=1, axis2=2)
    return sigma * np.sqrt(variances)


__all__ = [
    "ConsistencyResult",
    "rmse",
    "position_rmse",
    "velocity_rmse",
    "nees",
    "nees_sequence",
    "average_nees",
    "nis",
    "nis_sequence",
    "consistency_test",
    "credibility_interval",
    "monte_carlo_rmse",
    "estimation_error_bounds",
]
