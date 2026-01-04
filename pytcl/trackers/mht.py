"""
Multiple Hypothesis Tracking (MHT) implementation.

MHT maintains multiple hypotheses about measurement-to-track associations,
deferring hard decisions until more information is available. This allows
the tracker to recover from association errors.

This implementation uses track-oriented MHT with N-scan pruning.

References
----------
.. [1] S. Blackman and R. Popoli, "Design and Analysis of Modern
       Tracking Systems," Artech House, 1999.
.. [2] D. Reid, "An Algorithm for Tracking Multiple Targets,"
       IEEE Trans. Automatic Control, 1979.
"""

from typing import Callable, Dict, List, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from pytcl.assignment_algorithms.gating import mahalanobis_distance
from pytcl.trackers.hypothesis import (
    Hypothesis,
    HypothesisTree,
    MHTTrack,
    MHTTrackStatus,
    generate_joint_associations,
)


class MHTConfig(NamedTuple):
    """Configuration for MHT tracker.

    Attributes
    ----------
    n_scan : int
        Number of scans for N-scan pruning. Default 3.
    max_hypotheses : int
        Maximum number of hypotheses to maintain. Default 100.
    detection_prob : float
        Probability of detection (Pd). Default 0.9.
    clutter_density : float
        Spatial density of false alarms. Default 1e-6.
    gate_probability : float
        Gating probability for chi-squared test. Default 0.99.
    confirm_threshold : int
        Number of hits to confirm a track. Default 3.
    delete_threshold : int
        Number of consecutive misses to delete a track. Default 5.
    min_hypothesis_prob : float
        Minimum hypothesis probability. Default 1e-6.
    new_track_weight : float
        Prior weight for new track hypothesis. Default 0.1.
    """

    n_scan: int = 3
    max_hypotheses: int = 100
    detection_prob: float = 0.9
    clutter_density: float = 1e-6
    gate_probability: float = 0.99
    confirm_threshold: int = 3
    delete_threshold: int = 5
    min_hypothesis_prob: float = 1e-6
    new_track_weight: float = 0.1


class MHTResult(NamedTuple):
    """Result of MHT processing step.

    Attributes
    ----------
    confirmed_tracks : list of MHTTrack
        Tracks that are confirmed.
    tentative_tracks : list of MHTTrack
        Tracks that are tentative.
    all_tracks : list of MHTTrack
        All active tracks from best hypothesis.
    n_hypotheses : int
        Number of active hypotheses.
    best_hypothesis_prob : float
        Probability of the best hypothesis.
    """

    confirmed_tracks: List[MHTTrack]
    tentative_tracks: List[MHTTrack]
    all_tracks: List[MHTTrack]
    n_hypotheses: int
    best_hypothesis_prob: float


class MHTTracker:
    """
    Multiple Hypothesis Tracking (MHT) tracker.

    Maintains multiple hypotheses about measurement-to-track associations,
    with N-scan pruning for complexity control.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    meas_dim : int
        Dimension of measurement vector.
    F : callable or ndarray
        State transition matrix or function F(dt) -> ndarray.
    H : ndarray
        Measurement matrix.
    Q : callable or ndarray
        Process noise covariance or function Q(dt) -> ndarray.
    R : ndarray
        Measurement noise covariance.
    config : MHTConfig, optional
        Tracker configuration. Uses defaults if not provided.
    init_covariance : ndarray, optional
        Initial covariance for new tracks.

    Examples
    --------
    >>> import numpy as np
    >>> # Constant velocity model
    >>> F = lambda dt: np.array([[1, dt, 0, 0],
    ...                          [0, 1, 0, 0],
    ...                          [0, 0, 1, dt],
    ...                          [0, 0, 0, 1]])
    >>> H = np.array([[1, 0, 0, 0],
    ...               [0, 0, 1, 0]])
    >>> Q = lambda dt: 0.1 * np.eye(4)
    >>> R = np.eye(2) * 0.5
    >>> tracker = MHTTracker(4, 2, F, H, Q, R)
    >>> # Process measurements
    >>> measurements = [np.array([1, 2]), np.array([5, 6])]
    >>> result = tracker.process(measurements, dt=1.0)
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        F: Callable[[float], NDArray[np.floating]] | NDArray[np.floating],
        H: NDArray[np.floating],
        Q: Callable[[float], NDArray[np.floating]] | NDArray[np.floating],
        R: NDArray[np.floating],
        config: Optional[MHTConfig] = None,
        init_covariance: Optional[NDArray[np.floating]] = None,
    ):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        self._F = F if callable(F) else lambda dt: np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self._Q = Q if callable(Q) else lambda dt: np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

        self.config = config or MHTConfig()

        if init_covariance is not None:
            self.init_covariance = np.asarray(init_covariance, dtype=np.float64)
        else:
            self.init_covariance = np.eye(state_dim) * 100.0

        # Compute gate threshold
        self.gate_threshold = chi2.ppf(self.config.gate_probability, df=meas_dim)

        # Initialize hypothesis tree
        self.hypothesis_tree = HypothesisTree(
            max_hypotheses=self.config.max_hypotheses,
            n_scan=self.config.n_scan,
            min_probability=self.config.min_hypothesis_prob,
        )
        self.hypothesis_tree.initialize()

        self._time = 0.0
        self._scan = 0

    def process(
        self,
        measurements: List[ArrayLike],
        dt: float,
    ) -> MHTResult:
        """
        Process measurements at new time step.

        Parameters
        ----------
        measurements : list of array_like
            List of measurement vectors.
        dt : float
            Time step since last update.

        Returns
        -------
        result : MHTResult
            Tracking result with confirmed and tentative tracks.
        """
        self._time += dt
        self._scan += 1

        # Convert measurements
        if len(measurements) == 0:
            Z = np.zeros((0, self.meas_dim))
        else:
            Z = np.array([np.asarray(m, dtype=np.float64) for m in measurements])

        n_meas = len(Z)

        # Get current tracks from all hypotheses
        all_track_ids = set()
        for hyp in self.hypothesis_tree.hypotheses:
            all_track_ids.update(hyp.track_ids)

        current_tracks = {
            tid: self.hypothesis_tree.tracks[tid]
            for tid in all_track_ids
            if tid in self.hypothesis_tree.tracks
        }

        # Predict all tracks
        F = self._F(dt)
        Q = self._Q(dt)
        predicted_tracks = self._predict_tracks(current_tracks, F, Q)

        # Compute gating and likelihoods
        gated, likelihood_matrix = self._compute_gating_and_likelihoods(
            predicted_tracks, Z
        )

        # Generate associations for each hypothesis
        track_id_list = list(predicted_tracks.keys())
        n_tracks = len(track_id_list)

        if n_tracks > 0:
            # Create gating matrix indexed by position
            gated_matrix = np.zeros((n_tracks, n_meas), dtype=bool)
            likelihood_mat = np.zeros((n_tracks, n_meas))
            for i, tid in enumerate(track_id_list):
                for j in range(n_meas):
                    if (tid, j) in gated:
                        gated_matrix[i, j] = True
                        likelihood_mat[i, j] = likelihood_matrix.get((tid, j), 0.0)

            # Generate joint associations
            associations = generate_joint_associations(gated_matrix, n_tracks, n_meas)
        else:
            associations = [{}]

        # Compute likelihood for each association
        assoc_likelihoods = []
        for assoc in associations:
            # Convert position-based to track_id-based
            track_assoc = {}
            for pos_idx, meas_idx in assoc.items():
                track_id = track_id_list[pos_idx] if n_tracks > 0 else -1
                track_assoc[track_id] = meas_idx

            # Compute likelihood
            lik = self._compute_association_likelihood(
                track_assoc, predicted_tracks, Z, likelihood_matrix
            )
            assoc_likelihoods.append(lik)

        # Normalize likelihoods
        total_lik = sum(assoc_likelihoods)
        if total_lik > 0:
            assoc_likelihoods = [lik / total_lik for lik in assoc_likelihoods]
        elif len(assoc_likelihoods) > 0:
            assoc_likelihoods = [1.0 / len(assoc_likelihoods)] * len(assoc_likelihoods)

        # Update tracks based on associations
        new_tracks_per_assoc: Dict[int, List[MHTTrack]] = {}
        updated_tracks: Dict[int, Dict[int, MHTTrack]] = (
            {}
        )  # assoc_idx -> track_id -> track

        for assoc_idx, assoc in enumerate(associations):
            updated_tracks[assoc_idx] = {}
            new_tracks_per_assoc[assoc_idx] = []

            # Update existing tracks
            for pos_idx, meas_idx in assoc.items():
                if n_tracks == 0:
                    continue
                track_id = track_id_list[pos_idx]
                pred_track = predicted_tracks[track_id]

                if meas_idx >= 0:
                    # Track with measurement
                    upd_track = self._update_track(pred_track, Z[meas_idx], meas_idx)
                else:
                    # Missed detection
                    upd_track = self._miss_track(pred_track)

                updated_tracks[assoc_idx][track_id] = upd_track

            # Handle unassigned measurements -> new tracks
            assigned_meas = set(
                meas_idx for meas_idx in assoc.values() if meas_idx >= 0
            )
            for j in range(n_meas):
                if j not in assigned_meas:
                    new_track = self._initiate_track(Z[j], j)
                    new_tracks_per_assoc[assoc_idx].append(new_track)

        # Store updated tracks
        for assoc_idx, track_dict in updated_tracks.items():
            for track_id, track in track_dict.items():
                self.hypothesis_tree.tracks[track_id] = track

        # Expand hypotheses
        self._expand_hypotheses(
            associations,
            assoc_likelihoods,
            new_tracks_per_assoc,
            track_id_list,
        )

        # Build result
        return self._build_result()

    def _predict_tracks(
        self,
        tracks: Dict[int, MHTTrack],
        F: NDArray[np.floating],
        Q: NDArray[np.floating],
    ) -> Dict[int, MHTTrack]:
        """Predict all tracks forward in time."""
        predicted = {}
        for tid, track in tracks.items():
            if track.status == MHTTrackStatus.DELETED:
                continue

            x_pred = F @ track.state
            P_pred = F @ track.covariance @ F.T + Q

            predicted[tid] = MHTTrack(
                id=track.id,
                state=x_pred,
                covariance=P_pred,
                score=track.score,
                status=track.status,
                history=track.history,
                parent_id=track.parent_id,
                scan_created=track.scan_created,
                n_hits=track.n_hits,
                n_misses=track.n_misses,
            )

        return predicted

    def _compute_gating_and_likelihoods(
        self,
        tracks: Dict[int, MHTTrack],
        Z: NDArray[np.floating],
    ) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
        """Compute gating matrix and likelihood values."""
        gated = set()
        likelihood_matrix = {}

        for tid, track in tracks.items():
            z_pred = self.H @ track.state
            S = self.H @ track.covariance @ self.H.T + self.R

            for j in range(len(Z)):
                innovation = Z[j] - z_pred
                mahal_dist = mahalanobis_distance(innovation, S)

                if mahal_dist <= self.gate_threshold:
                    gated.add((tid, j))

                    # Compute likelihood
                    det_S = np.linalg.det(S)
                    if det_S > 0:
                        m = len(innovation)
                        mahal_sq = innovation @ np.linalg.solve(S, innovation)
                        likelihood = (
                            self.config.detection_prob
                            * np.exp(-0.5 * mahal_sq)
                            / np.sqrt((2 * np.pi) ** m * det_S)
                        )
                        likelihood_matrix[(tid, j)] = likelihood

        return gated, likelihood_matrix

    def _compute_association_likelihood(
        self,
        association: Dict[int, int],
        tracks: Dict[int, MHTTrack],
        Z: NDArray[np.floating],
        likelihood_matrix: dict[tuple[int, int], float],
    ) -> float:
        """Compute likelihood of a joint association."""
        likelihood = 1.0

        used_meas = set()
        for track_id, meas_idx in association.items():
            if meas_idx == -1:
                # Missed detection
                likelihood *= 1.0 - self.config.detection_prob
            else:
                # Detection
                if (track_id, meas_idx) in likelihood_matrix:
                    likelihood *= likelihood_matrix[(track_id, meas_idx)]
                else:
                    likelihood *= 1e-10  # Very small for ungated
                used_meas.add(meas_idx)

        # Clutter and new track terms for unassigned measurements
        n_unassigned = len(Z) - len(used_meas)
        likelihood *= (
            self.config.clutter_density + self.config.new_track_weight
        ) ** n_unassigned

        return likelihood

    def _update_track(
        self,
        track: MHTTrack,
        measurement: NDArray[np.floating],
        meas_idx: int,
    ) -> MHTTrack:
        """Update a track with a measurement."""
        # Innovation
        z_pred = self.H @ track.state
        innovation = measurement - z_pred
        S = self.H @ track.covariance @ self.H.T + self.R

        # Kalman gain
        K = track.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        x_upd = track.state + K @ innovation
        P_upd = (np.eye(self.state_dim) - K @ self.H) @ track.covariance

        # Update score (log-likelihood ratio)
        det_S = np.linalg.det(S)
        if det_S > 0:
            mahal_sq = innovation @ np.linalg.solve(S, innovation)
            score_update = 0.5 * (
                np.log(self.config.detection_prob)
                - np.log(self.config.clutter_density)
                - 0.5 * mahal_sq
                - 0.5 * np.log(det_S)
            )
        else:
            score_update = 0.0

        new_score = track.score + score_update

        # Update status
        n_hits = track.n_hits + 1
        n_misses = 0
        status = track.status
        if status == MHTTrackStatus.TENTATIVE:
            if n_hits >= self.config.confirm_threshold:
                status = MHTTrackStatus.CONFIRMED

        # Update history
        new_history = track.history + [meas_idx]

        return MHTTrack(
            id=track.id,
            state=x_upd,
            covariance=P_upd,
            score=new_score,
            status=status,
            history=new_history,
            parent_id=track.parent_id,
            scan_created=track.scan_created,
            n_hits=n_hits,
            n_misses=n_misses,
        )

    def _miss_track(self, track: MHTTrack) -> MHTTrack:
        """Handle missed detection for a track."""
        # Update score for missed detection
        score_update = np.log(1.0 - self.config.detection_prob)
        new_score = track.score + score_update

        # Update status
        n_misses = track.n_misses + 1
        status = track.status
        if n_misses >= self.config.delete_threshold:
            status = MHTTrackStatus.DELETED

        # Update history
        new_history = track.history + [-1]

        return MHTTrack(
            id=track.id,
            state=track.state,
            covariance=track.covariance,
            score=new_score,
            status=status,
            history=new_history,
            parent_id=track.parent_id,
            scan_created=track.scan_created,
            n_hits=track.n_hits,
            n_misses=n_misses,
        )

    def _initiate_track(
        self,
        measurement: NDArray[np.floating],
        meas_idx: int,
    ) -> MHTTrack:
        """Initiate a new track from a measurement."""
        # Initialize state from measurement
        H_pinv = np.linalg.pinv(self.H)
        state = H_pinv @ measurement

        return MHTTrack(
            id=-1,  # Will be assigned by hypothesis tree
            state=state,
            covariance=self.init_covariance.copy(),
            score=np.log(self.config.new_track_weight),
            status=MHTTrackStatus.TENTATIVE,
            history=[meas_idx],
            parent_id=-1,
            scan_created=self._scan,
            n_hits=1,
            n_misses=0,
        )

    def _expand_hypotheses(
        self,
        associations: List[Dict[int, int]],
        likelihoods: List[float],
        new_tracks: Dict[int, List[MHTTrack]],
        track_id_list: List[int],
    ) -> None:
        """Expand hypotheses with new associations."""
        new_hypotheses = []

        for hyp in self.hypothesis_tree.hypotheses:
            for assoc_idx, (assoc, likelihood) in enumerate(
                zip(associations, likelihoods)
            ):
                # Compute new hypothesis probability
                new_prob = hyp.probability * likelihood

                if new_prob < self.config.min_hypothesis_prob:
                    continue

                # Determine track IDs for new hypothesis
                new_track_ids = []

                # Keep surviving tracks from original hypothesis
                for track_id in hyp.track_ids:
                    if track_id in self.hypothesis_tree.tracks:
                        track = self.hypothesis_tree.tracks[track_id]
                        if track.status != MHTTrackStatus.DELETED:
                            new_track_ids.append(track_id)

                # Add new tracks from this association
                if assoc_idx in new_tracks:
                    for new_track in new_tracks[assoc_idx]:
                        tid = self.hypothesis_tree.add_track(new_track)
                        new_track_ids.append(tid)

                # Create new hypothesis
                new_hyp = Hypothesis(
                    id=self.hypothesis_tree._get_next_hypothesis_id(),
                    probability=new_prob,
                    track_ids=new_track_ids,
                    scan_created=self._scan,
                    parent_id=hyp.id,
                )
                new_hypotheses.append(new_hyp)

        # Update hypotheses
        if new_hypotheses:
            self.hypothesis_tree.hypotheses = new_hypotheses
        else:
            # Keep at least one hypothesis
            if self.hypothesis_tree.hypotheses:
                best = max(self.hypothesis_tree.hypotheses, key=lambda h: h.probability)
                self.hypothesis_tree.hypotheses = [best]

        # Prune
        self.hypothesis_tree.prune()

    def _build_result(self) -> MHTResult:
        """Build result from current state."""
        best_tracks = self.hypothesis_tree.get_best_tracks()
        confirmed = [t for t in best_tracks if t.status == MHTTrackStatus.CONFIRMED]
        tentative = [t for t in best_tracks if t.status == MHTTrackStatus.TENTATIVE]

        best_hyp = self.hypothesis_tree.get_best_hypothesis()
        best_prob = best_hyp.probability if best_hyp else 0.0

        return MHTResult(
            confirmed_tracks=confirmed,
            tentative_tracks=tentative,
            all_tracks=best_tracks,
            n_hypotheses=len(self.hypothesis_tree.hypotheses),
            best_hypothesis_prob=best_prob,
        )

    @property
    def tracks(self) -> List[MHTTrack]:
        """Get all tracks from best hypothesis."""
        return self.hypothesis_tree.get_best_tracks()

    @property
    def confirmed_tracks(self) -> List[MHTTrack]:
        """Get confirmed tracks from best hypothesis."""
        return self.hypothesis_tree.get_confirmed_tracks()

    @property
    def n_hypotheses(self) -> int:
        """Number of active hypotheses."""
        return len(self.hypothesis_tree.hypotheses)


__all__ = [
    "MHTConfig",
    "MHTResult",
    "MHTTracker",
]
