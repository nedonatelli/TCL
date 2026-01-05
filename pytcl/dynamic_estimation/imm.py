"""
Interacting Multiple Model (IMM) estimator.

The IMM estimator handles targets with multiple possible motion modes
(e.g., constant velocity, coordinated turn, acceleration) by maintaining
a bank of filters and mixing their outputs based on mode probabilities.

The IMM algorithm consists of four steps:
1. Mode probability mixing (interaction)
2. Mode-matched filtering (prediction/update per mode)
3. Mode probability update
4. Output combination
"""

from typing import Any, List, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update


class IMMState(NamedTuple):
    """State of an IMM estimator.

    Attributes
    ----------
    x : ndarray
        Combined state estimate, shape (n,).
    P : ndarray
        Combined state covariance, shape (n, n).
    mode_states : list of ndarray
        State estimates for each mode, each shape (n,).
    mode_covs : list of ndarray
        Covariances for each mode, each shape (n, n).
    mode_probs : ndarray
        Mode probabilities, shape (r,) where r is number of modes.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    mode_states: List[NDArray[np.floating]]
    mode_covs: List[NDArray[np.floating]]
    mode_probs: NDArray[np.floating]


class IMMPrediction(NamedTuple):
    """Result of IMM prediction step.

    Attributes
    ----------
    x : ndarray
        Combined predicted state estimate.
    P : ndarray
        Combined predicted state covariance.
    mode_states : list of ndarray
        Predicted state estimates for each mode.
    mode_covs : list of ndarray
        Predicted covariances for each mode.
    mode_probs : ndarray
        Mode probabilities (unchanged during prediction).
    mixing_probs : ndarray
        Mixing probabilities used, shape (r, r).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    mode_states: List[NDArray[np.floating]]
    mode_covs: List[NDArray[np.floating]]
    mode_probs: NDArray[np.floating]
    mixing_probs: NDArray[np.floating]


class IMMUpdate(NamedTuple):
    """Result of IMM update step.

    Attributes
    ----------
    x : ndarray
        Combined updated state estimate.
    P : ndarray
        Combined updated state covariance.
    mode_states : list of ndarray
        Updated state estimates for each mode.
    mode_covs : list of ndarray
        Updated covariances for each mode.
    mode_probs : ndarray
        Updated mode probabilities.
    mode_likelihoods : ndarray
        Measurement likelihoods for each mode.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    mode_states: List[NDArray[np.floating]]
    mode_covs: List[NDArray[np.floating]]
    mode_probs: NDArray[np.floating]
    mode_likelihoods: NDArray[np.floating]


def compute_mixing_probabilities(
    mode_probs: ArrayLike,
    transition_matrix: ArrayLike,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Compute mixing probabilities and predicted mode probabilities.

    Parameters
    ----------
    mode_probs : array_like
        Current mode probabilities, shape (r,).
    transition_matrix : array_like
        Mode transition probability matrix, shape (r, r).
        Element [i, j] is P(mode_j at k | mode_i at k-1).

    Returns
    -------
    mixing_probs : ndarray
        Mixing probabilities, shape (r, r).
        Element [i, j] is P(mode_i at k-1 | mode_j at k).
    c_bar : ndarray
        Predicted mode probabilities, shape (r,).
    """
    mode_probs = np.asarray(mode_probs, dtype=np.float64)
    Pi = np.asarray(transition_matrix, dtype=np.float64)
    r = len(mode_probs)

    # Predicted mode probabilities: c_bar[j] = sum_i Pi[i,j] * mu[i]
    c_bar = Pi.T @ mode_probs

    # Mixing probabilities: mu[i|j] = Pi[i,j] * mu[i] / c_bar[j] (vectorized)
    # Compute numerator: Pi[i,j] * mu[i] for all i,j
    numerator = Pi * mode_probs[:, np.newaxis]
    # Divide by c_bar (with safe division for near-zero values)
    safe_c_bar = np.where(c_bar > 1e-15, c_bar, 1.0)
    mixing_probs = numerator / safe_c_bar
    # Set uniform for columns where c_bar was too small
    zero_mask = c_bar <= 1e-15
    if np.any(zero_mask):
        mixing_probs[:, zero_mask] = 1.0 / r

    return mixing_probs, c_bar


def mix_states(
    mode_states: List[NDArray[Any]],
    mode_covs: List[NDArray[Any]],
    mixing_probs: NDArray[Any],
) -> tuple[List[NDArray[Any]], List[NDArray[Any]]]:
    """
    Mix states and covariances for interaction step.

    Parameters
    ----------
    mode_states : list of ndarray
        State estimates for each mode, each shape (n,).
    mode_covs : list of ndarray
        Covariances for each mode, each shape (n, n).
    mixing_probs : ndarray
        Mixing probabilities, shape (r, r).

    Returns
    -------
    mixed_states : list of ndarray
        Mixed state estimates for each mode.
    mixed_covs : list of ndarray
        Mixed covariances for each mode.
    """
    r = len(mode_states)

    # Stack states and covariances for vectorized operations
    states_array = np.array(mode_states)  # shape (r, n)
    covs_array = np.array(mode_covs)  # shape (r, n, n)

    mixed_states = []
    mixed_covs = []

    for j in range(r):
        # Mixed state: x_0j = sum_i mu[i|j] * x_i (vectorized)
        x_mixed = mixing_probs[:, j] @ states_array

        # Mixed covariance: P_0j = sum_i mu[i|j] * (P_i + (x_i - x_0j)(x_i - x_0j)^T)
        # Compute differences for all modes at once
        diffs = states_array - x_mixed  # shape (r, n)
        # Weighted covariances + outer products (vectorized)
        weights = mixing_probs[:, j]
        # Weighted sum of covariances
        P_mixed = np.tensordot(weights, covs_array, axes=([0], [0]))
        # Add weighted outer products: sum_i w_i * outer(diff_i, diff_i)
        weighted_diffs = np.sqrt(weights)[:, np.newaxis] * diffs
        P_mixed += weighted_diffs.T @ weighted_diffs

        mixed_states.append(x_mixed)
        mixed_covs.append(P_mixed)

    return mixed_states, mixed_covs


def combine_estimates(
    mode_states: List[NDArray[Any]],
    mode_covs: List[NDArray[Any]],
    mode_probs: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Combine mode-conditioned estimates into overall estimate.

    Parameters
    ----------
    mode_states : list of ndarray
        State estimates for each mode.
    mode_covs : list of ndarray
        Covariances for each mode.
    mode_probs : ndarray
        Mode probabilities.

    Returns
    -------
    x : ndarray
        Combined state estimate.
    P : ndarray
        Combined covariance.
    """
    # Stack states and covariances for vectorized operations
    states_array = np.array(mode_states)  # shape (r, n)
    covs_array = np.array(mode_covs)  # shape (r, n, n)

    # Combined state: x = sum_j mu_j * x_j (vectorized)
    x = mode_probs @ states_array

    # Combined covariance: P = sum_j mu_j * (P_j + (x_j - x)(x_j - x)^T) (vectorized)
    diffs = states_array - x  # shape (r, n)
    # Weighted sum of covariances
    P = np.tensordot(mode_probs, covs_array, axes=([0], [0]))
    # Add weighted outer products
    weighted_diffs = np.sqrt(mode_probs)[:, np.newaxis] * diffs
    P += weighted_diffs.T @ weighted_diffs

    # Ensure symmetry
    P = (P + P.T) / 2

    return x, P


def imm_predict(
    mode_states: List[ArrayLike],
    mode_covs: List[ArrayLike],
    mode_probs: ArrayLike,
    transition_matrix: ArrayLike,
    F_list: List[ArrayLike],
    Q_list: List[ArrayLike],
) -> IMMPrediction:
    """
    IMM prediction step.

    Performs:
    1. Compute mixing probabilities
    2. Mix states and covariances
    3. Mode-matched prediction for each filter

    Parameters
    ----------
    mode_states : list of array_like
        Current state estimates for each mode, each shape (n,).
    mode_covs : list of array_like
        Current covariances for each mode, each shape (n, n).
    mode_probs : array_like
        Current mode probabilities, shape (r,).
    transition_matrix : array_like
        Mode transition probability matrix, shape (r, r).
    F_list : list of array_like
        State transition matrices for each mode.
    Q_list : list of array_like
        Process noise covariances for each mode.

    Returns
    -------
    result : IMMPrediction
        Predicted states, covariances, and mode probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> # Two modes: constant velocity and coordinated turn
    >>> x1 = np.array([0., 1., 0., 0.])  # Mode 1 state
    >>> x2 = np.array([0., 1., 0., 0.])  # Mode 2 state
    >>> P1 = np.eye(4) * 0.1
    >>> P2 = np.eye(4) * 0.1
    >>> mu = np.array([0.9, 0.1])  # Mostly CV
    >>> Pi = np.array([[0.95, 0.05], [0.05, 0.95]])  # Transition matrix
    >>> F1 = np.eye(4)  # CV transition
    >>> F2 = np.eye(4)  # CT transition
    >>> Q1 = np.eye(4) * 0.01
    >>> Q2 = np.eye(4) * 0.01
    >>> pred = imm_predict([x1, x2], [P1, P2], mu, Pi, [F1, F2], [Q1, Q2])
    """
    # Convert inputs
    mode_states = [np.asarray(x, dtype=np.float64).flatten() for x in mode_states]
    mode_covs = [np.asarray(P, dtype=np.float64) for P in mode_covs]
    mode_probs = np.asarray(mode_probs, dtype=np.float64)
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    F_list = [np.asarray(F, dtype=np.float64) for F in F_list]
    Q_list = [np.asarray(Q, dtype=np.float64) for Q in Q_list]

    r = len(mode_states)

    # Step 1: Compute mixing probabilities
    mixing_probs, c_bar = compute_mixing_probabilities(mode_probs, transition_matrix)

    # Step 2: Mix states and covariances
    mixed_states, mixed_covs = mix_states(mode_states, mode_covs, mixing_probs)

    # Step 3: Mode-matched prediction
    pred_states = []
    pred_covs = []

    for j in range(r):
        pred = kf_predict(mixed_states[j], mixed_covs[j], F_list[j], Q_list[j])
        pred_states.append(pred.x)
        pred_covs.append(pred.P)

    # Step 4: Combine estimates
    x_combined, P_combined = combine_estimates(pred_states, pred_covs, c_bar)

    return IMMPrediction(
        x=x_combined,
        P=P_combined,
        mode_states=pred_states,
        mode_covs=pred_covs,
        mode_probs=c_bar,
        mixing_probs=mixing_probs,
    )


def imm_update(
    mode_states: List[ArrayLike],
    mode_covs: List[ArrayLike],
    mode_probs: ArrayLike,
    z: ArrayLike,
    H_list: List[ArrayLike],
    R_list: List[ArrayLike],
) -> IMMUpdate:
    """
    IMM update step.

    Performs:
    1. Mode-matched measurement update for each filter
    2. Mode probability update using measurement likelihoods
    3. Output combination

    Parameters
    ----------
    mode_states : list of array_like
        Predicted state estimates for each mode.
    mode_covs : list of array_like
        Predicted covariances for each mode.
    mode_probs : array_like
        Predicted mode probabilities.
    z : array_like
        Measurement.
    H_list : list of array_like
        Measurement matrices for each mode.
    R_list : list of array_like
        Measurement noise covariances for each mode.

    Returns
    -------
    result : IMMUpdate
        Updated states, covariances, and mode probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> # After prediction
    >>> x1 = np.array([1., 1., 0., 0.])
    >>> x2 = np.array([1., 1., 0., 0.])
    >>> P1 = np.eye(4) * 0.2
    >>> P2 = np.eye(4) * 0.2
    >>> mu = np.array([0.9, 0.1])
    >>> z = np.array([1.1, 0.1])  # Position measurement
    >>> H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    >>> R = np.eye(2) * 0.1
    >>> upd = imm_update([x1, x2], [P1, P2], mu, z, [H, H], [R, R])
    """
    # Convert inputs
    mode_states = [np.asarray(x, dtype=np.float64).flatten() for x in mode_states]
    mode_covs = [np.asarray(P, dtype=np.float64) for P in mode_covs]
    mode_probs = np.asarray(mode_probs, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H_list = [np.asarray(H, dtype=np.float64) for H in H_list]
    R_list = [np.asarray(R, dtype=np.float64) for R in R_list]

    r = len(mode_states)

    # Step 1: Mode-matched update
    upd_states = []
    upd_covs = []
    likelihoods = np.zeros(r)

    for j in range(r):
        upd = kf_update(mode_states[j], mode_covs[j], z, H_list[j], R_list[j])
        upd_states.append(upd.x)
        upd_covs.append(upd.P)
        likelihoods[j] = upd.likelihood

    # Step 2: Mode probability update
    # mu_j = c_j * Lambda_j / sum_i(c_i * Lambda_i)
    weighted_likelihoods = mode_probs * likelihoods
    total_likelihood = np.sum(weighted_likelihoods)

    if total_likelihood > 1e-300:
        upd_probs = weighted_likelihoods / total_likelihood
    else:
        # Keep current probabilities if all likelihoods are zero
        upd_probs = mode_probs.copy()

    # Normalize to ensure sum = 1
    upd_probs = upd_probs / np.sum(upd_probs)

    # Step 3: Combine estimates
    x_combined, P_combined = combine_estimates(upd_states, upd_covs, upd_probs)

    return IMMUpdate(
        x=x_combined,
        P=P_combined,
        mode_states=upd_states,
        mode_covs=upd_covs,
        mode_probs=upd_probs,
        mode_likelihoods=likelihoods,
    )


def imm_predict_update(
    mode_states: List[ArrayLike],
    mode_covs: List[ArrayLike],
    mode_probs: ArrayLike,
    transition_matrix: ArrayLike,
    z: ArrayLike,
    F_list: List[ArrayLike],
    Q_list: List[ArrayLike],
    H_list: List[ArrayLike],
    R_list: List[ArrayLike],
) -> IMMUpdate:
    """
    Combined IMM prediction and update step.

    Parameters
    ----------
    mode_states : list of array_like
        Current state estimates for each mode.
    mode_covs : list of array_like
        Current covariances for each mode.
    mode_probs : array_like
        Current mode probabilities.
    transition_matrix : array_like
        Mode transition probability matrix.
    z : array_like
        Measurement.
    F_list : list of array_like
        State transition matrices for each mode.
    Q_list : list of array_like
        Process noise covariances for each mode.
    H_list : list of array_like
        Measurement matrices for each mode.
    R_list : list of array_like
        Measurement noise covariances for each mode.

    Returns
    -------
    result : IMMUpdate
        Updated states, covariances, and mode probabilities.

    Examples
    --------
    Track a target with 2 motion modes (CV and CA):

    >>> import numpy as np
    >>> # Two modes: constant velocity and constant acceleration
    >>> states = [np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])]
    >>> covs = [np.eye(4) * 0.1, np.eye(4) * 0.1]
    >>> probs = np.array([0.9, 0.1])  # likely CV mode
    >>> # Mode transition matrix (90% stay, 10% switch)
    >>> trans = np.array([[0.9, 0.1], [0.1, 0.9]])
    >>> # Dynamics and measurement matrices for each mode
    >>> F_cv = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    >>> F_ca = F_cv.copy()  # simplified
    >>> H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    >>> Q = np.eye(4) * 0.01
    >>> R = np.eye(2) * 0.1
    >>> z = np.array([1.0, 1.0])
    >>> result = imm_predict_update(states, covs, probs, trans, z,
    ...                             [F_cv, F_ca], [Q, Q], [H, H], [R, R])
    >>> len(result.mode_probs)
    2

    See Also
    --------
    imm_predict : IMM prediction step only.
    imm_update : IMM update step only.
    IMMEstimator : Object-oriented interface.
    """
    pred = imm_predict(
        mode_states, mode_covs, mode_probs, transition_matrix, F_list, Q_list
    )
    return imm_update(
        pred.mode_states, pred.mode_covs, pred.mode_probs, z, H_list, R_list
    )


class IMMEstimator:
    """
    Interacting Multiple Model (IMM) estimator class.

    Provides an object-oriented interface for IMM filtering with
    automatic state management.

    Parameters
    ----------
    n_modes : int
        Number of motion modes.
    state_dim : int
        Dimension of state vector.
    transition_matrix : array_like
        Mode transition probability matrix, shape (n_modes, n_modes).
    initial_mode_probs : array_like, optional
        Initial mode probabilities. Default is uniform.

    Attributes
    ----------
    mode_states : list of ndarray
        Current state estimates for each mode.
    mode_covs : list of ndarray
        Current covariances for each mode.
    mode_probs : ndarray
        Current mode probabilities.
    x : ndarray
        Combined state estimate.
    P : ndarray
        Combined covariance.

    Examples
    --------
    >>> import numpy as np
    >>> # 2-mode IMM (CV and CT) for 4D state [x, vx, y, vy]
    >>> Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
    >>> imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)
    >>> # Initialize
    >>> x0 = np.array([0., 1., 0., 0.])
    >>> P0 = np.eye(4) * 0.1
    >>> imm.initialize(x0, P0)
    >>> # Set models
    >>> F1 = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    >>> Q1 = np.eye(4) * 0.01
    >>> imm.set_mode_model(0, F1, Q1)
    >>> imm.set_mode_model(1, F1, Q1)  # Same F for simplicity
    """

    def __init__(
        self,
        n_modes: int,
        state_dim: int,
        transition_matrix: ArrayLike,
        initial_mode_probs: Optional[ArrayLike] = None,
    ):
        self.n_modes = n_modes
        self.state_dim = state_dim
        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)

        if initial_mode_probs is None:
            self.mode_probs = np.ones(n_modes) / n_modes
        else:
            self.mode_probs = np.asarray(initial_mode_probs, dtype=np.float64)

        # Initialize mode-conditioned estimates
        self.mode_states = [np.zeros(state_dim) for _ in range(n_modes)]
        self.mode_covs = [np.eye(state_dim) for _ in range(n_modes)]

        # Mode-specific models (must be set by user)
        self.F_list: List[NDArray[Any]] = [np.eye(state_dim) for _ in range(n_modes)]
        self.Q_list: List[NDArray[Any]] = [np.eye(state_dim) for _ in range(n_modes)]
        self.H_list: List[NDArray[Any]] = []
        self.R_list: List[NDArray[Any]] = []

        # Combined estimates
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def initialize(
        self,
        x: ArrayLike,
        P: ArrayLike,
        mode_probs: Optional[ArrayLike] = None,
    ) -> None:
        """
        Initialize all modes with the same state.

        Parameters
        ----------
        x : array_like
            Initial state estimate.
        P : array_like
            Initial covariance.
        mode_probs : array_like, optional
            Initial mode probabilities.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        P = np.asarray(P, dtype=np.float64)

        for j in range(self.n_modes):
            self.mode_states[j] = x.copy()
            self.mode_covs[j] = P.copy()

        if mode_probs is not None:
            self.mode_probs = np.asarray(mode_probs, dtype=np.float64)

        self.x = x.copy()
        self.P = P.copy()

    def set_mode_model(
        self,
        mode_idx: int,
        F: ArrayLike,
        Q: ArrayLike,
    ) -> None:
        """
        Set the dynamic model for a specific mode.

        Parameters
        ----------
        mode_idx : int
            Mode index.
        F : array_like
            State transition matrix.
        Q : array_like
            Process noise covariance.
        """
        self.F_list[mode_idx] = np.asarray(F, dtype=np.float64)
        self.Q_list[mode_idx] = np.asarray(Q, dtype=np.float64)

    def set_measurement_model(
        self,
        H: ArrayLike,
        R: ArrayLike,
        mode_specific: bool = False,
    ) -> None:
        """
        Set the measurement model.

        Parameters
        ----------
        H : array_like or list of array_like
            Measurement matrix. If mode_specific=True, should be a list.
        R : array_like or list of array_like
            Measurement noise covariance. If mode_specific=True, should be a list.
        mode_specific : bool
            If True, H and R are lists with different models per mode.
        """
        if mode_specific:
            self.H_list = [np.asarray(h, dtype=np.float64) for h in H]
            self.R_list = [np.asarray(r, dtype=np.float64) for r in R]
        else:
            H = np.asarray(H, dtype=np.float64)
            R = np.asarray(R, dtype=np.float64)
            self.H_list = [H for _ in range(self.n_modes)]
            self.R_list = [R for _ in range(self.n_modes)]

    def predict(self) -> IMMPrediction:
        """
        Perform IMM prediction step.

        Returns
        -------
        result : IMMPrediction
            Prediction result.
        """
        result = imm_predict(
            self.mode_states,
            self.mode_covs,
            self.mode_probs,
            self.transition_matrix,
            self.F_list,
            self.Q_list,
        )

        # Update internal state
        self.mode_states = result.mode_states
        self.mode_covs = result.mode_covs
        self.mode_probs = result.mode_probs
        self.x = result.x
        self.P = result.P

        return result

    def update(self, z: ArrayLike) -> IMMUpdate:
        """
        Perform IMM update step.

        Parameters
        ----------
        z : array_like
            Measurement.

        Returns
        -------
        result : IMMUpdate
            Update result.
        """
        if not self.H_list:
            raise ValueError(
                "Measurement model not set. Call set_measurement_model first."
            )

        result = imm_update(
            self.mode_states,
            self.mode_covs,
            self.mode_probs,
            z,
            self.H_list,
            self.R_list,
        )

        # Update internal state
        self.mode_states = result.mode_states
        self.mode_covs = result.mode_covs
        self.mode_probs = result.mode_probs
        self.x = result.x
        self.P = result.P

        return result

    def predict_update(self, z: ArrayLike) -> IMMUpdate:
        """
        Combined prediction and update.

        Parameters
        ----------
        z : array_like
            Measurement.

        Returns
        -------
        result : IMMUpdate
            Update result.
        """
        self.predict()
        return self.update(z)

    def get_state(self) -> IMMState:
        """
        Get current IMM state.

        Returns
        -------
        state : IMMState
            Current state.
        """
        return IMMState(
            x=self.x.copy(),
            P=self.P.copy(),
            mode_states=[s.copy() for s in self.mode_states],
            mode_covs=[p.copy() for p in self.mode_covs],
            mode_probs=self.mode_probs.copy(),
        )


__all__ = [
    "IMMState",
    "IMMPrediction",
    "IMMUpdate",
    "compute_mixing_probabilities",
    "mix_states",
    "combine_estimates",
    "imm_predict",
    "imm_update",
    "imm_predict_update",
    "IMMEstimator",
]
