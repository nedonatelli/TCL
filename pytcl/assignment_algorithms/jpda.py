"""
Joint Probabilistic Data Association (JPDA) algorithm.

JPDA computes association probabilities between tracks and measurements
by considering all possible joint association hypotheses, then performs
a probabilistically weighted update for each track.

This is more sophisticated than GNN which makes hard assignment decisions,
as JPDA can handle measurement origin uncertainty in cluttered environments.
"""

from typing import Any, List, NamedTuple, Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from pytcl.assignment_algorithms.gating import mahalanobis_distance


class JPDAResult(NamedTuple):
    """Result of JPDA algorithm.

    Attributes
    ----------
    association_probs : ndarray[Any]
        Association probability matrix of shape (n_tracks, n_measurements + 1).
        association_probs[i, j] is the probability that track i is associated
        with measurement j. The last column (j = n_measurements) represents
        the probability that track i has no measurement.
    marginal_probs : list of ndarray
        List of marginal association probabilities for each track.
        marginal_probs[i][j] = P(measurement j originated from track i).
    likelihood_matrix : ndarray[Any]
        Measurement likelihood matrix of shape (n_tracks, n_measurements).
    gated : ndarray[Any]
        Boolean matrix indicating which track-measurement pairs passed gating.
    """

    association_probs: NDArray[np.floating]
    marginal_probs: List[NDArray[np.floating]]
    likelihood_matrix: NDArray[np.floating]
    gated: NDArray[np.bool_]


class JPDAUpdate(NamedTuple):
    """Result of JPDA-based track update.

    Attributes
    ----------
    states : list of ndarray
        Updated state estimates for each track.
    covariances : list of ndarray
        Updated covariances for each track (includes spread of means).
    association_probs : ndarray[Any]
        Association probability matrix.
    innovations : list of ndarray
        Combined weighted innovations for each track.
    """

    states: List[NDArray[np.floating]]
    covariances: List[NDArray[np.floating]]
    association_probs: NDArray[np.floating]
    innovations: List[NDArray[np.floating]]


def compute_measurement_likelihood(
    innovation: NDArray[Any],
    innovation_cov: NDArray[Any],
    detection_prob: float = 1.0,
) -> float:
    """
    Compute measurement likelihood for a track-measurement pair.

    Parameters
    ----------
    innovation : ndarray[Any]
        Measurement innovation (residual), shape (m,).
    innovation_cov : ndarray[Any]
        Innovation covariance, shape (m, m).
    detection_prob : float
        Probability of detection (Pd).

    Returns
    -------
    likelihood : float
        Measurement likelihood.
    """
    m = len(innovation)
    det_S = np.linalg.det(innovation_cov)

    if det_S <= 0:
        return 0.0

    mahal_sq = innovation @ np.linalg.solve(innovation_cov, innovation)
    likelihood = (
        detection_prob * np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    )

    return likelihood


def compute_likelihood_matrix(
    track_states: list[NDArray[Any]],
    track_covariances: list[NDArray[Any]],
    measurements: NDArray[Any],
    H: NDArray[Any],
    R: NDArray[Any],
    detection_prob: float = 1.0,
    gate_threshold: Optional[float] = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Compute likelihood matrix for all track-measurement pairs.

    Parameters
    ----------
    track_states : list of ndarray
        State estimates for each track.
    track_covariances : list of ndarray
        Covariances for each track.
    measurements : ndarray[Any]
        Measurements, shape (n_meas, m).
    H : ndarray[Any]
        Measurement matrix, shape (m, n).
    R : ndarray[Any]
        Measurement noise covariance, shape (m, m).
    detection_prob : float
        Probability of detection.
    gate_threshold : float, optional
        Mahalanobis distance threshold for gating.

    Returns
    -------
    likelihood_matrix : ndarray[Any]
        Likelihood values, shape (n_tracks, n_meas).
    gated : ndarray[Any]
        Boolean gating matrix, shape (n_tracks, n_meas).

    Examples
    --------
    >>> import numpy as np
    >>> # Two tracks, two measurements
    >>> states = [np.array([0.0, 1.0]), np.array([5.0, 0.0])]
    >>> covs = [np.eye(2) * 0.5, np.eye(2) * 0.5]
    >>> measurements = np.array([[0.1], [5.2]])
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[0.1]])
    >>> L, gated = compute_likelihood_matrix(states, covs, measurements, H, R)
    >>> L.shape
    (2, 2)
    >>> # Track 0 should have high likelihood for measurement 0
    >>> L[0, 0] > L[0, 1]
    True
    """
    n_tracks = len(track_states)
    n_meas = len(measurements)

    likelihood_matrix = np.zeros((n_tracks, n_meas))
    gated = np.zeros((n_tracks, n_meas), dtype=bool)

    for i in range(n_tracks):
        # Predicted measurement and innovation covariance
        z_pred = H @ track_states[i]
        S = H @ track_covariances[i] @ H.T + R

        for j in range(n_meas):
            innovation = measurements[j] - z_pred
            mahal_dist = mahalanobis_distance(innovation, S)

            # Check gate
            if gate_threshold is None or mahal_dist <= gate_threshold:
                gated[i, j] = True
                likelihood_matrix[i, j] = compute_measurement_likelihood(
                    innovation, S, detection_prob
                )

    return likelihood_matrix, gated


def jpda_probabilities(
    likelihood_matrix: NDArray[Any],
    gated: NDArray[Any],
    detection_prob: float = 1.0,
    clutter_density: float = 1e-6,
) -> NDArray[Any]:
    """
    Compute JPDA association probabilities.

    Uses an efficient algorithm that avoids explicit enumeration of all
    joint hypotheses when possible.

    Parameters
    ----------
    likelihood_matrix : ndarray[Any]
        Likelihood values, shape (n_tracks, n_meas).
    gated : ndarray[Any]
        Boolean gating matrix, shape (n_tracks, n_meas).
    detection_prob : float
        Probability of detection (Pd).
    clutter_density : float
        Spatial density of clutter (lambda_c).

    Returns
    -------
    beta : ndarray[Any]
        Association probability matrix, shape (n_tracks, n_meas + 1).
        beta[i, j] = P(measurement j is from track i) for j < n_meas.
        beta[i, n_meas] = P(track i has no measurement).

    Examples
    --------
    >>> import numpy as np
    >>> # Likelihood matrix: 2 tracks, 2 measurements
    >>> # Track 0 has high likelihood for meas 0
    >>> # Track 1 has high likelihood for meas 1
    >>> likelihood = np.array([[0.9, 0.1],
    ...                        [0.1, 0.8]])
    >>> gated = np.array([[True, True],
    ...                   [True, True]])
    >>> beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)
    >>> beta.shape  # 2 tracks, 3 columns (2 meas + 1 miss)
    (2, 3)
    >>> # Track 0 most likely associated with measurement 0
    >>> np.argmax(beta[0, :2])
    0
    """
    n_tracks, n_meas = likelihood_matrix.shape

    # Initialize beta matrix (last column is for "no measurement")
    beta = np.zeros((n_tracks, n_meas + 1))

    if n_meas == 0:
        # No measurements - all tracks have no association
        beta[:, 0] = 1.0
        return beta

    if n_tracks == 0:
        return beta

    # For small problems, use exact enumeration
    # For larger problems, use approximate method
    if n_tracks <= 5 and n_meas <= 5:
        beta = _jpda_exact(likelihood_matrix, gated, detection_prob, clutter_density)
    else:
        beta = _jpda_approximate(
            likelihood_matrix, gated, detection_prob, clutter_density
        )

    return beta


def _jpda_exact(
    likelihood_matrix: NDArray[Any],
    gated: NDArray[Any],
    detection_prob: float,
    clutter_density: float,
) -> NDArray[Any]:
    """
    Exact JPDA computation via hypothesis enumeration.

    This is exponential in complexity but exact for small problems.
    """
    n_tracks, n_meas = likelihood_matrix.shape
    beta = np.zeros((n_tracks, n_meas + 1))

    # Generate all valid joint association hypotheses
    # A hypothesis is a mapping from measurements to tracks (or clutter)
    # measurement j -> track assignment[j] where -1 means clutter

    # We enumerate by considering all possible measurement-to-track assignments
    # where each measurement can go to at most one track that gates it

    def generate_hypotheses(
        meas_idx: int,
        current_assignment: List[int],
        used_tracks: set[Any],
    ) -> Any:
        """Recursively generate valid hypotheses."""
        if meas_idx == n_meas:
            yield current_assignment.copy()
            return

        # Option 1: measurement is clutter
        current_assignment.append(-1)
        yield from generate_hypotheses(meas_idx + 1, current_assignment, used_tracks)
        current_assignment.pop()

        # Option 2: measurement is from a track (if gated and track not used)
        for track_idx in range(n_tracks):
            if gated[track_idx, meas_idx] and track_idx not in used_tracks:
                current_assignment.append(track_idx)
                used_tracks.add(track_idx)
                yield from generate_hypotheses(
                    meas_idx + 1, current_assignment, used_tracks
                )
                used_tracks.remove(track_idx)
                current_assignment.pop()

    # Compute hypothesis probabilities
    hypothesis_probs = []
    hypothesis_assignments = []

    for assignment in generate_hypotheses(0, [], set[Any]()):
        # Compute probability of this hypothesis
        prob = 1.0

        detected_tracks = set[Any]()
        for j, track_idx in enumerate(assignment):
            if track_idx == -1:
                # Measurement j is clutter
                prob *= clutter_density
            else:
                # Measurement j is from track track_idx
                prob *= likelihood_matrix[track_idx, j]
                detected_tracks.add(track_idx)

        # Account for non-detected tracks
        for i in range(n_tracks):
            if i in detected_tracks:
                prob *= detection_prob
            else:
                prob *= 1.0 - detection_prob

        hypothesis_probs.append(prob)
        hypothesis_assignments.append(assignment)

    # Normalize
    total_prob = sum(hypothesis_probs)
    if total_prob > 0:
        hypothesis_probs = [p / total_prob for p in hypothesis_probs]

    # Compute marginal association probabilities
    for h_idx, (assignment, prob) in enumerate(
        zip(hypothesis_assignments, hypothesis_probs)
    ):
        detected_tracks = set[Any]()
        for j, track_idx in enumerate(assignment):
            if track_idx >= 0:
                beta[track_idx, j] += prob
                detected_tracks.add(track_idx)

        # Tracks with no measurement
        for i in range(n_tracks):
            if i not in detected_tracks:
                beta[i, n_meas] += prob

    return beta


@njit(cache=True)
def _jpda_approximate_core(
    likelihood_matrix: np.ndarray[Any, Any],
    gated: np.ndarray[Any, Any],
    detection_prob: float,
    clutter_density: float,
) -> np.ndarray[Any, Any]:
    """JIT-compiled core of approximate JPDA computation."""
    n_tracks = likelihood_matrix.shape[0]
    n_meas = likelihood_matrix.shape[1]
    beta = np.zeros((n_tracks, n_meas + 1), dtype=np.float64)

    # Likelihood ratio for each measurement given each track
    L = np.zeros((n_tracks, n_meas), dtype=np.float64)
    for i in range(n_tracks):
        for j in range(n_meas):
            if gated[i, j] and clutter_density > 0:
                L[i, j] = likelihood_matrix[i, j] / clutter_density
            elif gated[i, j]:
                L[i, j] = likelihood_matrix[i, j] * 1e10

    # Compute delta factors (accounts for other tracks)
    delta = np.ones((n_tracks, n_meas), dtype=np.float64)

    for j in range(n_meas):
        sum_L = 0.0
        for i in range(n_tracks):
            sum_L += L[i, j]
        for i in range(n_tracks):
            if sum_L > 0:
                delta[i, j] = 1.0 / (1.0 + sum_L - L[i, j])
            else:
                delta[i, j] = 1.0

    # Compute association probabilities
    for i in range(n_tracks):
        denom = 1.0 - detection_prob

        for j in range(n_meas):
            if gated[i, j]:
                beta[i, j] = detection_prob * L[i, j] * delta[i, j]
                denom += beta[i, j]

        if denom > 0:
            for j in range(n_meas):
                beta[i, j] /= denom
            beta[i, n_meas] = (1.0 - detection_prob) / denom
        else:
            beta[i, n_meas] = 1.0

    return beta


def _jpda_approximate(
    likelihood_matrix: NDArray[Any],
    gated: NDArray[Any],
    detection_prob: float,
    clutter_density: float,
) -> NDArray[Any]:
    """
    Approximate JPDA using parametric approach.

    Uses the approach from [1] which is O(n_tracks * n_meas^2).

    References
    ----------
    .. [1] Fitzgerald, R.J., "Development of Practical PDA Logic for
           Multitarget Tracking by Microprocessor", American Control
           Conference, 1986.
    """
    return _jpda_approximate_core(
        likelihood_matrix.astype(np.float64),
        gated.astype(np.bool_),
        detection_prob,
        clutter_density,
    )


def jpda_update(
    track_states: List[ArrayLike],
    track_covariances: List[ArrayLike],
    measurements: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    detection_prob: float = 0.9,
    clutter_density: float = 1e-6,
    gate_probability: float = 0.99,
) -> JPDAUpdate:
    """
    Perform JPDA-based track update.

    Parameters
    ----------
    track_states : list of array_like
        Predicted state estimates for each track.
    track_covariances : list of array_like
        Predicted covariances for each track.
    measurements : array_like
        Measurements, shape (n_meas, m).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
    detection_prob : float
        Probability of detection (Pd). Default 0.9.
    clutter_density : float
        Spatial density of clutter. Default 1e-6.
    gate_probability : float
        Probability for chi-squared gate. Default 0.99.

    Returns
    -------
    result : JPDAUpdate
        Updated states and covariances with association probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> # Two tracks, three measurements
    >>> x1 = np.array([0., 1.])  # [position, velocity]
    >>> x2 = np.array([5., -1.])
    >>> P = np.eye(2) * 0.1
    >>> measurements = np.array([[0.1], [5.2], [10.0]])
    >>> H = np.array([[1., 0.]])  # Measure position
    >>> R = np.array([[0.1]])
    >>> result = jpda_update([x1, x2], [P, P], measurements, H, R)
    >>> len(result.states)
    2

    Notes
    -----
    The JPDA update consists of:
    1. Compute measurement likelihoods and gating
    2. Compute association probabilities
    3. Compute combined innovations for each track
    4. Update each track with weighted innovation
    5. Compute covariance with spread of means term

    References
    ----------
    .. [1] Bar-Shalom, Y. and Fortmann, T.E., "Tracking and Data Association",
           Academic Press, 1988.
    """
    # Convert inputs
    track_states = [np.asarray(x, dtype=np.float64).flatten() for x in track_states]
    track_covariances = [np.asarray(P, dtype=np.float64) for P in track_covariances]
    measurements = np.asarray(measurements, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n_tracks = len(track_states)
    n_meas = len(measurements)
    n_state = len(track_states[0]) if n_tracks > 0 else 0
    m = H.shape[0]

    # Compute gate threshold
    gate_threshold = chi2.ppf(gate_probability, df=m)

    # Compute likelihood matrix and gating
    likelihood_matrix, gated = compute_likelihood_matrix(
        track_states,
        track_covariances,
        measurements,
        H,
        R,
        detection_prob,
        gate_threshold,
    )

    # Compute association probabilities
    beta = jpda_probabilities(likelihood_matrix, gated, detection_prob, clutter_density)

    # Update each track
    updated_states = []
    updated_covs = []
    innovations = []

    for i in range(n_tracks):
        x = track_states[i]
        P = track_covariances[i]

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)

        # Predicted measurement
        z_pred = H @ x

        # Compute combined innovation (weighted sum of innovations)
        combined_innovation = np.zeros(m)
        for j in range(n_meas):
            if gated[i, j]:
                innovation_j = measurements[j] - z_pred
                combined_innovation += beta[i, j] * innovation_j

        innovations.append(combined_innovation)

        # Updated state
        x_upd = x + K @ combined_innovation

        # Updated covariance with spread of means term
        # P_upd = beta_0 * P + (1 - beta_0) * P_c + P_spread
        # where P_c = P - K @ S @ K.T (covariance after update)
        # and P_spread = K @ E[y*y'] @ K.T - K @ E[y] @ E[y]' @ K.T

        beta_0 = beta[i, n_meas]  # Probability of no measurement

        # Covariance reduction from update
        P_c = P - K @ S @ K.T

        # Spread of means
        P_spread = np.zeros((n_state, n_state))
        for j in range(n_meas):
            if gated[i, j]:
                innovation_j = measurements[j] - z_pred
                y_weighted = innovation_j - combined_innovation
                P_spread += beta[i, j] * K @ np.outer(y_weighted, y_weighted) @ K.T

        # Combined covariance
        P_upd = beta_0 * P + (1 - beta_0) * P_c + P_spread

        # Ensure symmetry
        P_upd = (P_upd + P_upd.T) / 2

        updated_states.append(x_upd)
        updated_covs.append(P_upd)

    return JPDAUpdate(
        states=updated_states,
        covariances=updated_covs,
        association_probs=beta,
        innovations=innovations,
    )


def jpda(
    track_states: List[ArrayLike],
    track_covariances: List[ArrayLike],
    measurements: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    detection_prob: float = 0.9,
    clutter_density: float = 1e-6,
    gate_probability: float = 0.99,
) -> JPDAResult:
    """
    Compute JPDA association probabilities.

    This is a convenience function that computes association probabilities
    without performing the state update.

    Parameters
    ----------
    track_states : list of array_like
        Predicted state estimates for each track.
    track_covariances : list of array_like
        Predicted covariances for each track.
    measurements : array_like
        Measurements, shape (n_meas, m).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
    detection_prob : float
        Probability of detection. Default 0.9.
    clutter_density : float
        Spatial density of clutter. Default 1e-6.
    gate_probability : float
        Probability for chi-squared gate. Default 0.99.

    Returns
    -------
    result : JPDAResult
        Association probabilities and related information.

    Examples
    --------
    Compute association probabilities for 2 tracks and 3 measurements:

    >>> import numpy as np
    >>> # Two tracks with [x, vx] state
    >>> states = [np.array([0.0, 1.0]), np.array([10.0, -0.5])]
    >>> covariances = [np.eye(2) * 0.5, np.eye(2) * 0.5]
    >>> # Three position measurements
    >>> measurements = np.array([[0.1], [9.8], [5.0]])
    >>> H = np.array([[1, 0]])  # measure position only
    >>> R = np.array([[0.1]])
    >>> result = jpda(states, covariances, measurements, H, R)
    >>> result.association_probs.shape  # (2 tracks, 4 columns: 3 meas + miss)
    (2, 4)
    >>> # Track 0 should have high prob for measurement 0
    >>> result.association_probs[0, 0] > 0.5
    True

    See Also
    --------
    jpda_update : JPDA with state update.
    """
    # Convert inputs
    track_states = [np.asarray(x, dtype=np.float64).flatten() for x in track_states]
    track_covariances = [np.asarray(P, dtype=np.float64) for P in track_covariances]
    measurements = np.asarray(measurements, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    m = H.shape[0]

    # Compute gate threshold
    gate_threshold = chi2.ppf(gate_probability, df=m)

    # Compute likelihood matrix and gating
    likelihood_matrix, gated = compute_likelihood_matrix(
        track_states,
        track_covariances,
        measurements,
        H,
        R,
        detection_prob,
        gate_threshold,
    )

    # Compute association probabilities
    beta = jpda_probabilities(likelihood_matrix, gated, detection_prob, clutter_density)

    # Compute marginal probabilities for each track
    n_tracks = len(track_states)
    n_meas = len(measurements)
    marginal_probs = []
    for i in range(n_tracks):
        marginal_probs.append(beta[i, :n_meas].copy())

    return JPDAResult(
        association_probs=beta,
        marginal_probs=marginal_probs,
        likelihood_matrix=likelihood_matrix,
        gated=gated,
    )


__all__ = [
    "JPDAResult",
    "JPDAUpdate",
    "compute_measurement_likelihood",
    "compute_likelihood_matrix",
    "jpda_probabilities",
    "jpda_update",
    "jpda",
]
