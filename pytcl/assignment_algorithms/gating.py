"""
Gating functions for data association in target tracking.

This module provides gating methods to determine which measurements
fall within a validation region around predicted track states.
"""

from typing import Any, List, Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2


@njit(cache=True, fastmath=True)
def _mahalanobis_distance_2d(
    innovation: np.ndarray[Any, Any],
    S_inv: np.ndarray[Any, Any],
) -> float:
    """JIT-compiled Mahalanobis distance for 2D innovations."""
    return innovation[0] * (
        S_inv[0, 0] * innovation[0] + S_inv[0, 1] * innovation[1]
    ) + innovation[1] * (S_inv[1, 0] * innovation[0] + S_inv[1, 1] * innovation[1])


@njit(cache=True, fastmath=True)
def _mahalanobis_distance_3d(
    innovation: np.ndarray[Any, Any],
    S_inv: np.ndarray[Any, Any],
) -> float:
    """JIT-compiled Mahalanobis distance for 3D innovations."""
    result = 0.0
    for i in range(3):
        for j in range(3):
            result += innovation[i] * S_inv[i, j] * innovation[j]
    return result


@njit(cache=True, fastmath=True)
def _mahalanobis_distance_general(
    innovation: np.ndarray[Any, Any],
    S_inv: np.ndarray[Any, Any],
) -> float:
    """JIT-compiled Mahalanobis distance for general dimension."""
    n = len(innovation)
    result = 0.0
    for i in range(n):
        for j in range(n):
            result += innovation[i] * S_inv[i, j] * innovation[j]
    return result


def mahalanobis_distance(
    innovation: ArrayLike,
    innovation_covariance: ArrayLike,
) -> float:
    """
    Compute the squared Mahalanobis distance.

    The Mahalanobis distance measures how many standard deviations
    a point is from the center of a distribution.

    Parameters
    ----------
    innovation : array_like
        Innovation (measurement residual) vector of shape (m,).
    innovation_covariance : array_like
        Innovation covariance matrix of shape (m, m).

    Returns
    -------
    float
        Squared Mahalanobis distance.

    Examples
    --------
    >>> innovation = np.array([1.0, 0.5])
    >>> S = np.array([[2.0, 0.0], [0.0, 1.0]])
    >>> d2 = mahalanobis_distance(innovation, S)
    >>> d2
    0.75

    Notes
    -----
    The squared Mahalanobis distance is defined as:
        d^2 = (z - z_pred)^T @ S^{-1} @ (z - z_pred)

    where S is the innovation covariance matrix.
    """
    nu = np.asarray(innovation, dtype=np.float64)
    S = np.asarray(innovation_covariance, dtype=np.float64)

    # Use solve instead of inverse for numerical stability
    S_inv_nu = np.linalg.solve(S, nu)
    return float(nu @ S_inv_nu)


def ellipsoidal_gate(
    innovation: ArrayLike,
    innovation_covariance: ArrayLike,
    gate_threshold: float,
) -> bool:
    """
    Test if a measurement passes an ellipsoidal gate.

    The ellipsoidal gate defines a validation region based on the
    chi-squared distribution of the squared Mahalanobis distance.

    Parameters
    ----------
    innovation : array_like
        Innovation vector of shape (m,).
    innovation_covariance : array_like
        Innovation covariance matrix of shape (m, m).
    gate_threshold : float
        Gate threshold (chi-squared value). Common values:
        - 9.21 for 99% probability with 2 measurements
        - 11.34 for 99% probability with 3 measurements
        - 16.27 for 99% probability with 4 measurements

    Returns
    -------
    bool
        True if measurement passes the gate (is inside the ellipsoid).

    Examples
    --------
    >>> innovation = np.array([1.0, 0.5])
    >>> S = np.array([[2.0, 0.0], [0.0, 1.0]])
    >>> ellipsoidal_gate(innovation, S, gate_threshold=9.21)
    True

    See Also
    --------
    chi2_gate_threshold : Compute threshold from probability.
    """
    d2 = mahalanobis_distance(innovation, innovation_covariance)
    return d2 <= gate_threshold


def chi2_gate_threshold(
    probability: float,
    num_dimensions: int,
) -> float:
    """
    Compute chi-squared gate threshold for a given probability.

    Parameters
    ----------
    probability : float
        Gate probability (e.g., 0.99 for 99% of true measurements to pass).
    num_dimensions : int
        Measurement dimension (degrees of freedom).

    Returns
    -------
    float
        Chi-squared threshold value.

    Examples
    --------
    >>> chi2_gate_threshold(0.99, 2)  # 2D measurement, 99% probability
    9.210340371976184
    >>> chi2_gate_threshold(0.99, 3)  # 3D measurement, 99% probability
    11.344866730144373
    """
    return float(chi2.ppf(probability, df=num_dimensions))


def rectangular_gate(
    innovation: ArrayLike,
    innovation_covariance: ArrayLike,
    num_sigmas: float = 3.0,
) -> bool:
    """
    Test if a measurement passes a rectangular gate.

    The rectangular gate defines a validation region as a hypercube
    based on the marginal standard deviations.

    Parameters
    ----------
    innovation : array_like
        Innovation vector of shape (m,).
    innovation_covariance : array_like
        Innovation covariance matrix of shape (m, m).
    num_sigmas : float, optional
        Number of standard deviations for gate bounds (default: 3.0).

    Returns
    -------
    bool
        True if measurement passes the gate.

    Examples
    --------
    >>> innovation = np.array([1.0, 0.5])
    >>> S = np.array([[4.0, 0.0], [0.0, 1.0]])
    >>> rectangular_gate(innovation, S, num_sigmas=3.0)
    True

    Notes
    -----
    Rectangular gating is computationally cheaper but less tight than
    ellipsoidal gating. It may pass more false measurements.
    """
    nu = np.asarray(innovation, dtype=np.float64)
    S = np.asarray(innovation_covariance, dtype=np.float64)

    # Extract marginal standard deviations
    sigmas = np.sqrt(np.diag(S))

    # Check if all components are within bounds
    return bool(np.all(np.abs(nu) <= num_sigmas * sigmas))


def gate_measurements(
    predicted_measurement: ArrayLike,
    innovation_covariance: ArrayLike,
    measurements: ArrayLike,
    gate_threshold: float,
    gate_type: str = "ellipsoidal",
) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:
    """
    Gate multiple measurements against a predicted track state.

    Parameters
    ----------
    predicted_measurement : array_like
        Predicted measurement of shape (m,).
    innovation_covariance : array_like
        Innovation covariance matrix of shape (m, m).
    measurements : array_like
        Array of measurements of shape (n_meas, m).
    gate_threshold : float
        Gate threshold. For ellipsoidal gates, this is the chi-squared value.
        For rectangular gates, this is the number of sigmas.
    gate_type : str, optional
        Type of gate: "ellipsoidal" or "rectangular" (default: "ellipsoidal").

    Returns
    -------
    valid_indices : ndarray
        Indices of measurements that pass the gate.
    distances : ndarray
        Squared Mahalanobis distances for valid measurements.

    Examples
    --------
    >>> z_pred = np.array([0.0, 0.0])
    >>> S = np.eye(2)
    >>> measurements = np.array([[0.5, 0.5], [5.0, 5.0], [1.0, -1.0]])
    >>> valid_idx, dists = gate_measurements(z_pred, S, measurements, 9.21)
    >>> valid_idx
    array([0, 2])

    Notes
    -----
    This function efficiently gates multiple measurements against a single
    track prediction, which is common in multi-target tracking.
    """
    z_pred = np.asarray(predicted_measurement, dtype=np.float64)
    S = np.asarray(innovation_covariance, dtype=np.float64)
    Z = np.asarray(measurements, dtype=np.float64)

    if Z.ndim == 1:
        Z = Z.reshape(1, -1)

    n_meas = Z.shape[0]
    valid_indices: List[int] = []
    distances: List[float] = []

    for i in range(n_meas):
        innovation = Z[i] - z_pred

        if gate_type == "ellipsoidal":
            d2 = mahalanobis_distance(innovation, S)
            if d2 <= gate_threshold:
                valid_indices.append(i)
                distances.append(d2)
        elif gate_type == "rectangular":
            if rectangular_gate(innovation, S, num_sigmas=gate_threshold):
                # For rectangular gate, still compute Mahalanobis distance for ranking
                d2 = mahalanobis_distance(innovation, S)
                valid_indices.append(i)
                distances.append(d2)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    return (
        np.array(valid_indices, dtype=np.intp),
        np.array(distances, dtype=np.float64),
    )


def compute_gate_volume(
    innovation_covariance: ArrayLike,
    gate_threshold: float,
) -> float:
    """
    Compute the volume of an ellipsoidal gate.

    Parameters
    ----------
    innovation_covariance : array_like
        Innovation covariance matrix of shape (m, m).
    gate_threshold : float
        Chi-squared gate threshold.

    Returns
    -------
    float
        Volume of the ellipsoidal gate region.

    Notes
    -----
    The gate volume is used in probabilistic data association methods
    to compute the clutter density.

    For an m-dimensional ellipsoid with threshold gamma:
        V = c_m * sqrt(det(S)) * gamma^(m/2)

    where c_m is the volume of the unit hypersphere in m dimensions.

    Examples
    --------
    Compute gate volume for a 2D measurement with 99% gate probability:

    >>> import numpy as np
    >>> from scipy.stats import chi2
    >>> S = np.array([[4.0, 0.0], [0.0, 1.0]])  # innovation covariance
    >>> gate_prob = 0.99
    >>> threshold = chi2.ppf(gate_prob, df=2)
    >>> volume = compute_gate_volume(S, threshold)
    >>> volume > 0
    True

    See Also
    --------
    ellipsoidal_gate : Test if measurement passes gate.
    mahalanobis_distance : Compute distance used in gating.
    """
    S = np.asarray(innovation_covariance, dtype=np.float64)
    m = S.shape[0]

    # Volume of unit hypersphere in m dimensions
    # c_m = pi^(m/2) / Gamma(m/2 + 1)
    from scipy.special import gamma as gamma_func

    c_m = np.pi ** (m / 2) / gamma_func(m / 2 + 1)

    # Gate volume
    det_S = np.linalg.det(S)
    volume = c_m * np.sqrt(det_S) * gate_threshold ** (m / 2)

    return float(volume)


@njit(cache=True, fastmath=True, parallel=False)
def mahalanobis_batch(
    innovations: np.ndarray[Any, Any],
    S_inv: np.ndarray[Any, Any],
    output: np.ndarray[Any, Any],
) -> None:
    """
    Compute Mahalanobis distances for a batch of innovations.

    JIT-compiled for performance. Computes squared Mahalanobis distances
    for multiple innovations against a single covariance matrix.

    Parameters
    ----------
    innovations : ndarray
        Innovations of shape (n_measurements, dim).
    S_inv : ndarray
        Inverse of innovation covariance matrix of shape (dim, dim).
    output : ndarray
        Output array of shape (n_measurements,) to store distances.
    """
    n_meas = innovations.shape[0]
    dim = innovations.shape[1]

    for i in range(n_meas):
        result = 0.0
        for j in range(dim):
            for k in range(dim):
                result += innovations[i, j] * S_inv[j, k] * innovations[i, k]
        output[i] = result


__all__ = [
    "mahalanobis_distance",
    "mahalanobis_batch",
    "ellipsoidal_gate",
    "rectangular_gate",
    "gate_measurements",
    "chi2_gate_threshold",
    "compute_gate_volume",
]
