"""
Multiple Hypothesis Tracking (MHT) implementation.

MHT maintains multiple hypotheses about measurement-to-track associations,
deferring hard decisions until more information is available. This allows
the tracker to recover from association errors.

This implementation uses track-oriented MHT with N-scan pruning.

References
----------
- S. Blackman and R. Popoli, "Design and Analysis of Modern
  Tracking Systems," Artech House, 1999.
- D. Reid, "An Algorithm for Tracking Multiple Targets,"
  IEEE Trans. Automatic Control, 1979.
"""

from typing import Callable, Dict, List, NamedTuple, Optional

import msgspec
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from pytcl.assignment_algorithms.gating import mahalanobis_distance
from pytcl.diagnostics import diagnostics_enabled, logger
from pytcl.trackers.hypothesis import (
    Hypothesis,
    HypothesisTree,
    MHTTrack,
    MHTTrackStatus,
    generate_joint_associations,
)


class MHTConfig(msgspec.Struct, frozen=True):
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

        # Store dynamics, retaining the pre-normalization matrix form (None
        # for callables) so session snapshots can recover it -- see
        # pytcl.io.session.
        self._F_matrix = None if callable(F) else np.asarray(F, dtype=np.float64)
        self._Q_matrix = None if callable(Q) else np.asarray(Q, dtype=np.float64)
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

        # Expand hypotheses: each hypothesis enumerates joint associations
        # over its OWN tracks, and every (track, measurement-or-miss)
        # decision creates a distinct child branch keyed by parent_id, so
        # hypotheses that disagree about an association carry genuinely
        # different track states. (The previous implementation stored one
        # global track per id -- the last association overwrote all
        # others, so every hypothesis shared a single state and the
        # branching machinery was dead code.)
        branch_cache: Dict[tuple[int, int], int] = {}
        new_track_cache: Dict[int, int] = {}
        self._expand_hypotheses(
            predicted_tracks,
            Z,
            gated,
            likelihood_matrix,
            branch_cache,
            new_track_cache,
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
        """Compute likelihood of a joint association.

        Different formula from the public
        :func:`pytcl.trackers.hypothesis.compute_association_likelihood`:
        this method's unassigned-measurement term is
        ``(clutter_density + new_track_weight) ** n_unassigned``, folding in
        the possibility that an unassigned measurement starts a new track,
        rather than the plain ``clutter_density ** n_clutter`` used there.
        """
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

        # Log-likelihood ratio increment for a detection, in the standard form
        #
        #     ln Pd - ln lambda - 0.5 * (d^2 + m ln(2 pi) + ln det S)
        #
        # where d^2 is the Mahalanobis distance and m the measurement
        # dimension. This used to carry an overall factor of 0.5 and omit the
        # (2 pi)^m normalization, which made it a scaled quantity rather than a
        # log-likelihood ratio (gh-25).
        #
        # The factor mattered beyond naming: _miss_track adds a full
        # ln(1 - Pd), so hits were accumulating at half the scale of misses and
        # the running total was not a consistent quantity in either unit.
        # Nothing reads `score` -- it is reported, never used for confirm or
        # delete, which go by cumulative n_hits and consecutive n_misses
        # respectively (not M-of-N; MultiTargetTracker is the one with a
        # windowed rule) -- so this changes what is displayed, not what the
        # tracker does.
        det_S = np.linalg.det(S)
        if det_S > 0:
            mahal_sq = innovation @ np.linalg.solve(S, innovation)
            meas_dim = len(innovation)
            score_update = (
                np.log(self.config.detection_prob)
                - np.log(self.config.clutter_density)
                - 0.5 * (mahal_sq + meas_dim * np.log(2 * np.pi) + np.log(det_S))
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

    def _get_branch(
        self,
        parent: MHTTrack,
        meas_idx: int,
        Z: NDArray[np.floating],
        branch_cache: Dict[tuple[int, int], int],
    ) -> int:
        """Get (creating and memoizing) the child branch of `parent` for
        one association decision: update with measurement `meas_idx`, or a
        missed detection when `meas_idx` is -1.

        The child gets a fresh id with ``parent_id`` pointing at the
        parent branch and ``scan_created`` set to the current scan, so the
        ancestry chain records when each association decision was made --
        which is what N-scan pruning walks. Hypotheses that agree on this
        decision share the same child node via the cache.
        """
        key = (parent.id, meas_idx)
        if key in branch_cache:
            return branch_cache[key]

        if meas_idx >= 0:
            child = self._update_track(parent, Z[meas_idx], meas_idx)
        else:
            child = self._miss_track(parent)

        child = child._replace(
            id=self.hypothesis_tree._get_next_track_id(),
            parent_id=parent.id,
            scan_created=self._scan,
        )
        self.hypothesis_tree.tracks[child.id] = child
        branch_cache[key] = child.id
        return child.id

    def _get_new_track(
        self,
        meas_idx: int,
        Z: NDArray[np.floating],
        new_track_cache: Dict[int, int],
    ) -> int:
        """Get (creating and memoizing) the new-track branch for an
        unassigned measurement. One node per measurement per scan, shared
        by every hypothesis that treats the measurement as a new target.
        """
        if meas_idx in new_track_cache:
            return new_track_cache[meas_idx]
        tid = self.hypothesis_tree.add_track(
            self._initiate_track(Z[meas_idx], meas_idx)
        )
        new_track_cache[meas_idx] = tid
        return tid

    def _expand_hypotheses(
        self,
        predicted_tracks: Dict[int, MHTTrack],
        Z: NDArray[np.floating],
        gated: set[tuple[int, int]],
        likelihood_matrix: dict[tuple[int, int], float],
        branch_cache: Dict[tuple[int, int], int],
        new_track_cache: Dict[int, int],
    ) -> None:
        """Expand each hypothesis with the joint associations of its own
        tracks, creating per-association child branches."""
        n_meas = len(Z)
        tree = self.hypothesis_tree

        # (probability, child track ids, parent hypothesis id), merged by
        # identical track-id sets: different parents can produce the same
        # global interpretation, whose probabilities then add.
        merged: Dict[frozenset[int], List] = {}

        for hyp in tree.hypotheses:
            hyp_tids = [tid for tid in hyp.track_ids if tid in predicted_tracks]
            n_tracks = len(hyp_tids)

            if n_tracks > 0:
                gated_matrix = np.zeros((n_tracks, n_meas), dtype=bool)
                for i, tid in enumerate(hyp_tids):
                    for j in range(n_meas):
                        if (tid, j) in gated:
                            gated_matrix[i, j] = True
                associations = generate_joint_associations(
                    gated_matrix, n_tracks, n_meas
                )
            else:
                associations = [{}]

            for assoc in associations:
                track_assoc = {
                    hyp_tids[pos_idx]: meas_idx for pos_idx, meas_idx in assoc.items()
                }
                likelihood = self._compute_association_likelihood(
                    track_assoc, predicted_tracks, Z, likelihood_matrix
                )
                new_prob = hyp.probability * likelihood
                if new_prob <= 0.0:
                    continue

                child_ids = []
                for tid, meas_idx in track_assoc.items():
                    child_id = self._get_branch(
                        predicted_tracks[tid], meas_idx, Z, branch_cache
                    )
                    if tree.tracks[child_id].status != MHTTrackStatus.DELETED:
                        child_ids.append(child_id)

                assigned = set(m for m in track_assoc.values() if m >= 0)
                for j in range(n_meas):
                    if j not in assigned:
                        child_ids.append(self._get_new_track(j, Z, new_track_cache))

                key = frozenset(child_ids)
                if key in merged:
                    merged[key][0] += new_prob
                else:
                    merged[key] = [new_prob, child_ids, hyp.id]

        # Normalize and rebuild the hypothesis set
        total_prob = sum(entry[0] for entry in merged.values())
        new_hypotheses = []
        for prob, child_ids, parent_id in merged.values():
            new_hypotheses.append(
                Hypothesis(
                    id=tree._get_next_hypothesis_id(),
                    probability=prob / total_prob
                    if total_prob > 0
                    else 1.0 / len(merged),
                    track_ids=child_ids,
                    scan_created=self._scan,
                    parent_id=parent_id,
                )
            )

        if new_hypotheses:
            tree.hypotheses = new_hypotheses
        elif tree.hypotheses:
            # Keep at least one hypothesis
            best = max(tree.hypotheses, key=lambda h: h.probability)
            tree.hypotheses = [best]

        # Advance the tree's scan counter before pruning: N-scan pruning
        # measures decision age against it. (It previously never advanced,
        # which made N-scan pruning a permanent no-op.)
        tree.current_scan = self._scan

        # Prune
        log_pruning = diagnostics_enabled()
        pre_prune_count = len(self.hypothesis_tree.hypotheses) if log_pruning else 0
        self.hypothesis_tree.prune()

        if log_pruning:
            post_prune_count = len(self.hypothesis_tree.hypotheses)
            best_score = (
                max(h.probability for h in self.hypothesis_tree.hypotheses)
                if self.hypothesis_tree.hypotheses
                else 0.0
            )
            logger.bind(site="association", algo="mht").debug(
                "mht scan {}: {} hypotheses, pruned {}, best_score={:.6g}",
                self._scan,
                post_prune_count,
                pre_prune_count - post_prune_count,
                best_score,
            )

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
