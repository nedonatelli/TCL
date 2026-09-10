"""
Hypothesis management for Multiple Hypothesis Tracking (MHT).

This module provides data structures and algorithms for managing
hypothesis trees in track-oriented MHT implementations.

References
----------
- S. Blackman and R. Popoli, "Design and Analysis of Modern
  Tracking Systems," Artech House, 1999.
- D. Reid, "An Algorithm for Tracking Multiple Targets,"
  IEEE Trans. Automatic Control, 1979.
"""

from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


class MHTTrackStatus(Enum):
    """Track status in MHT."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


class MHTTrack(NamedTuple):
    """Track state within MHT.

    Attributes
    ----------
    id : int
        Unique track identifier.
    state : ndarray
        State estimate vector.
    covariance : ndarray
        State covariance matrix.
    score : float
        Log-likelihood ratio score.
    status : MHTTrackStatus
        Track status.
    history : list of int
        Measurement indices associated with this track branch.
        -1 indicates a missed detection.
    parent_id : int
        ID of parent track (-1 for root tracks).
    scan_created : int
        Scan number when this track branch was created.
    n_hits : int
        Number of measurement updates.
    n_misses : int
        Number of consecutive misses.
    """

    id: int
    state: NDArray[np.floating]
    covariance: NDArray[np.floating]
    score: float
    status: MHTTrackStatus
    history: List[int]
    parent_id: int
    scan_created: int
    n_hits: int
    n_misses: int


class Hypothesis(NamedTuple):
    """A global hypothesis in MHT.

    A hypothesis represents a consistent assignment of measurements
    to tracks across multiple scans. Each hypothesis maintains a
    set of track branches that are mutually compatible.

    Attributes
    ----------
    id : int
        Unique hypothesis identifier.
    probability : float
        Posterior probability of this hypothesis.
    track_ids : list of int
        IDs of tracks included in this hypothesis.
    scan_created : int
        Scan number when this hypothesis was created.
    parent_id : int
        ID of parent hypothesis (-1 for initial hypothesis).
    """

    id: int
    probability: float
    track_ids: List[int]
    scan_created: int
    parent_id: int


def generate_joint_associations(
    gated: NDArray[np.bool_],
    n_tracks: int,
    n_meas: int,
) -> List[Dict[int, int]]:
    """
    Generate all valid joint measurement-to-track associations.

    A valid association satisfies:
    - Each measurement is assigned to at most one track
    - Each track is assigned to at most one measurement
    - Only gated track-measurement pairs are considered

    Parameters
    ----------
    gated : ndarray
        Boolean gating matrix, shape (n_tracks, n_meas).
        gated[i, j] = True if track i can be associated with measurement j.
    n_tracks : int
        Number of tracks.
    n_meas : int
        Number of measurements.

    Returns
    -------
    associations : list of dict
        List of valid associations. Each dict maps track_id to meas_idx.
        meas_idx = -1 indicates missed detection.

    Examples
    --------
    >>> gated = np.array([[True, True], [True, True]])
    >>> associations = generate_joint_associations(gated, 2, 2)
    >>> len(associations)  # All valid 2-track, 2-measurement associations
    7
    """
    associations = []

    def _enumerate(
        track_idx: int,
        current: Dict[int, int],
        used_meas: Set[int],
    ) -> None:
        """Recursively enumerate associations."""
        if track_idx == n_tracks:
            associations.append(current.copy())
            return

        # Option 1: Track has no measurement (missed detection)
        current[track_idx] = -1
        _enumerate(track_idx + 1, current, used_meas)

        # Option 2: Track is associated with a gated measurement
        for j in range(n_meas):
            if gated[track_idx, j] and j not in used_meas:
                current[track_idx] = j
                used_meas.add(j)
                _enumerate(track_idx + 1, current, used_meas)
                used_meas.remove(j)

    if n_tracks > 0:
        _enumerate(0, {}, set())
    else:
        associations.append({})

    return associations


def compute_association_likelihood(
    association: Dict[int, int],
    likelihood_matrix: NDArray[np.floating],
    detection_prob: float,
    clutter_density: float,
    n_meas: int,
) -> float:
    """
    Compute likelihood of a joint association.

    Parameters
    ----------
    association : dict
        Mapping from track_id to measurement_idx (-1 for miss).
    likelihood_matrix : ndarray
        Likelihood values, shape (n_tracks, n_meas).
    detection_prob : float
        Probability of detection.
    clutter_density : float
        Spatial density of clutter (false alarms).
    n_meas : int
        Total number of measurements.

    Returns
    -------
    likelihood : float
        Joint likelihood of the association.

    Notes
    -----
    Unassigned measurements contribute a plain clutter term,
    ``clutter_density ** n_clutter``. This is a different formula from
    :meth:`pytcl.trackers.mht.MHTTracker._compute_association_likelihood`,
    whose unassigned-measurement term folds in ``new_track_weight`` as well
    (``(clutter_density + new_track_weight) ** n_unassigned``), since MHT
    must also account for the possibility that an unassigned measurement
    starts a new track rather than being clutter. :class:`MHTTracker` uses
    its own internal method, not this one.

    Examples
    --------
    >>> import numpy as np
    >>> # 2 tracks, 2 measurements
    >>> likelihood_matrix = np.array([[0.9, 0.1],
    ...                                [0.1, 0.8]])
    >>> # Association: track 0 -> meas 0, track 1 -> meas 1
    >>> association = {0: 0, 1: 1}
    >>> lik = compute_association_likelihood(
    ...     association, likelihood_matrix,
    ...     detection_prob=0.9, clutter_density=1e-6, n_meas=2
    ... )
    >>> lik > 0
    True
    >>> # Association with missed detection
    >>> assoc_miss = {0: 0, 1: -1}  # track 1 misses
    >>> lik_miss = compute_association_likelihood(
    ...     assoc_miss, likelihood_matrix,
    ...     detection_prob=0.9, clutter_density=1e-6, n_meas=2
    ... )
    >>> lik > lik_miss  # Full detection more likely
    True
    """
    likelihood = 1.0

    used_meas = set()
    for track_id, meas_idx in association.items():
        if meas_idx == -1:
            # Missed detection
            likelihood *= 1.0 - detection_prob
        else:
            # Detection
            likelihood *= detection_prob * likelihood_matrix[track_id, meas_idx]
            used_meas.add(meas_idx)

    # Clutter terms for unassigned measurements
    n_clutter = n_meas - len(used_meas)
    likelihood *= clutter_density**n_clutter

    return likelihood


def n_scan_prune(
    hypotheses: List[Hypothesis],
    tracks: Dict[int, MHTTrack],
    n_scan: int,
    current_scan: int,
) -> Tuple[List[Hypothesis], Set[int]]:
    """
    N-scan pruning of hypotheses.

    Removes hypotheses that diverged from the most likely hypothesis
    more than n_scan scans ago. This implements "deferred decision"
    pruning where associations older than N scans are committed to.

    Parameters
    ----------
    hypotheses : list of Hypothesis
        Current hypotheses.
    tracks : dict
        Mapping from track_id to MHTTrack.
    n_scan : int
        Number of scans to look back.
    current_scan : int
        Current scan number.

    Returns
    -------
    pruned_hypotheses : list of Hypothesis
        Hypotheses surviving pruning.
    committed_track_ids : set
        Track IDs that are now committed (survived N-scan).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers.hypothesis import (
    ...     Hypothesis, MHTTrack, MHTTrackStatus, n_scan_prune
    ... )
    >>> # Two hypotheses, tracks with different creation scans
    >>> track1 = MHTTrack(id=0, state=np.zeros(2), covariance=np.eye(2),
    ...                   score=1.0, status=MHTTrackStatus.CONFIRMED,
    ...                   history=[0], parent_id=-1, scan_created=0,
    ...                   n_hits=3, n_misses=0)
    >>> track2 = MHTTrack(id=1, state=np.zeros(2), covariance=np.eye(2),
    ...                   score=0.5, status=MHTTrackStatus.TENTATIVE,
    ...                   history=[1], parent_id=-1, scan_created=2,
    ...                   n_hits=1, n_misses=0)
    >>> tracks = {0: track1, 1: track2}
    >>> hyp1 = Hypothesis(id=0, probability=0.8, track_ids=[0],
    ...                   scan_created=0, parent_id=-1)
    >>> hyp2 = Hypothesis(id=1, probability=0.2, track_ids=[1],
    ...                   scan_created=2, parent_id=-1)
    >>> pruned, committed = n_scan_prune([hyp1, hyp2], tracks, n_scan=2,
    ...                                   current_scan=3)
    >>> len(pruned) >= 1
    True

    Notes
    -----
    This implementation applies a single-MAP-hypothesis rule, not
    agreement across all high-probability hypotheses::

        1. Take the single highest-probability (MAP) hypothesis and,
           for each of its track branches, walk the ``parent_id`` chain
           back to the ancestor branch whose association decision was
           made at or before scan (current_scan - n_scan). That ancestor
           set is what the MAP hypothesis has committed to.
        2. Keep a hypothesis only if its own ancestor set at that scan
           is identical to the MAP hypothesis's set. Track families born
           after the cutoff have no ancestor there and are excluded from
           the comparison on both sides.

    Hypotheses that agree with each other but disagree with the MAP
    hypothesis are pruned along with genuinely divergent ones -- the
    rule is "match the MAP hypothesis or be removed," not a vote across
    all high-probability hypotheses.
    """
    if not hypotheses or n_scan <= 0:
        return hypotheses, set()

    cutoff_scan = current_scan - n_scan

    def _ancestor_at_cutoff(track_id: int) -> Optional[int]:
        """Walk the branch's parent chain to the node whose association
        decision was made at or before the cutoff scan; None when the
        whole family is younger than the cutoff."""
        node = tracks.get(track_id)
        while node is not None and node.scan_created > cutoff_scan:
            if node.parent_id < 0:
                return None
            node = tracks.get(node.parent_id)
        return None if node is None else node.id

    def _committed_ancestors(hyp: Hypothesis) -> Set[int]:
        ancestors = set()
        for track_id in hyp.track_ids:
            ancestor = _ancestor_at_cutoff(track_id)
            if ancestor is not None:
                ancestors.add(ancestor)
        return ancestors

    # Find best hypothesis
    best_hyp = max(hypotheses, key=lambda h: h.probability)
    best_tracks_at_cutoff = _committed_ancestors(best_hyp)

    # Prune hypotheses whose committed past diverges from the MAP one
    pruned = [
        hyp for hyp in hypotheses if _committed_ancestors(hyp) == best_tracks_at_cutoff
    ]

    # Renormalize probabilities
    total_prob = sum(h.probability for h in pruned)
    if total_prob > 0:
        pruned = [
            Hypothesis(
                id=h.id,
                probability=h.probability / total_prob,
                track_ids=h.track_ids,
                scan_created=h.scan_created,
                parent_id=h.parent_id,
            )
            for h in pruned
        ]

    return pruned, best_tracks_at_cutoff


def prune_hypotheses_by_probability(
    hypotheses: List[Hypothesis],
    max_hypotheses: int,
    min_probability: float = 1e-6,
) -> List[Hypothesis]:
    """
    Prune hypotheses by probability threshold and count limit.

    Parameters
    ----------
    hypotheses : list of Hypothesis
        Current hypotheses.
    max_hypotheses : int
        Maximum number of hypotheses to retain.
    min_probability : float
        Minimum probability threshold.

    Returns
    -------
    pruned : list of Hypothesis
        Pruned and renormalized hypotheses.

    Examples
    --------
    >>> from pytcl.trackers.hypothesis import Hypothesis, prune_hypotheses_by_probability
    >>> # 5 hypotheses with varying probabilities
    >>> hyps = [
    ...     Hypothesis(id=0, probability=0.5, track_ids=[0], scan_created=0, parent_id=-1),
    ...     Hypothesis(id=1, probability=0.3, track_ids=[1], scan_created=0, parent_id=-1),
    ...     Hypothesis(id=2, probability=0.1, track_ids=[2], scan_created=0, parent_id=-1),
    ...     Hypothesis(id=3, probability=0.05, track_ids=[3], scan_created=0, parent_id=-1),
    ...     Hypothesis(id=4, probability=1e-8, track_ids=[4], scan_created=0, parent_id=-1),
    ... ]
    >>> pruned = prune_hypotheses_by_probability(hyps, max_hypotheses=3)
    >>> len(pruned)  # Only top 3 kept
    3
    >>> sum(h.probability for h in pruned)  # Renormalized to 1
    1.0
    """
    if not hypotheses:
        return []

    # Filter by minimum probability
    filtered = [h for h in hypotheses if h.probability >= min_probability]

    if not filtered:
        # Keep at least the best one
        filtered = [max(hypotheses, key=lambda h: h.probability)]

    # Sort by probability (descending)
    filtered.sort(key=lambda h: h.probability, reverse=True)

    # Keep top max_hypotheses
    filtered = filtered[:max_hypotheses]

    # Renormalize
    total_prob = sum(h.probability for h in filtered)
    if total_prob > 0:
        filtered = [
            Hypothesis(
                id=h.id,
                probability=h.probability / total_prob,
                track_ids=h.track_ids,
                scan_created=h.scan_created,
                parent_id=h.parent_id,
            )
            for h in filtered
        ]

    return filtered


class HypothesisTree:
    """
    Manages hypothesis tree for MHT.

    The hypothesis tree represents all possible interpretations of
    measurement-to-track associations across multiple scans.

    Parameters
    ----------
    max_hypotheses : int
        Maximum number of hypotheses to maintain.
    n_scan : int
        Number of scans for N-scan pruning.
    min_probability : float
        Minimum hypothesis probability threshold.

    Attributes
    ----------
    hypotheses : list of Hypothesis
        Current set of hypotheses.
    tracks : dict
        Mapping from track_id to MHTTrack.
    current_scan : int
        Current scan number.
    """

    def __init__(
        self,
        max_hypotheses: int = 100,
        n_scan: int = 3,
        min_probability: float = 1e-6,
    ):
        self.max_hypotheses = max_hypotheses
        self.n_scan = n_scan
        self.min_probability = min_probability

        self.hypotheses: List[Hypothesis] = []
        self.tracks: Dict[int, MHTTrack] = {}
        self.current_scan = 0

        self._next_hypothesis_id = 0
        self._next_track_id = 0

    def initialize(
        self,
        initial_tracks: Optional[List[MHTTrack]] = None,
    ) -> None:
        """
        Initialize the hypothesis tree.

        Parameters
        ----------
        initial_tracks : list of MHTTrack, optional
            Initial tracks to include.
        """
        self.hypotheses = []
        self.tracks = {}
        self.current_scan = 0
        self._next_hypothesis_id = 0
        self._next_track_id = 0

        if initial_tracks:
            for track in initial_tracks:
                self.tracks[track.id] = track
                self._next_track_id = max(self._next_track_id, track.id + 1)

        # Create initial hypothesis
        track_ids = list(self.tracks.keys()) if initial_tracks else []
        initial_hyp = Hypothesis(
            id=self._get_next_hypothesis_id(),
            probability=1.0,
            track_ids=track_ids,
            scan_created=0,
            parent_id=-1,
        )
        self.hypotheses = [initial_hyp]

    def _get_next_hypothesis_id(self) -> int:
        """Get next hypothesis ID."""
        id = self._next_hypothesis_id
        self._next_hypothesis_id += 1
        return id

    def _get_next_track_id(self) -> int:
        """Get next track ID."""
        id = self._next_track_id
        self._next_track_id += 1
        return id

    def add_track(self, track: MHTTrack) -> int:
        """
        Add a new track to the tree.

        Parameters
        ----------
        track : MHTTrack
            Track to add.

        Returns
        -------
        track_id : int
            ID assigned to the track.
        """
        track_id = self._get_next_track_id()
        new_track = MHTTrack(
            id=track_id,
            state=track.state,
            covariance=track.covariance,
            score=track.score,
            status=track.status,
            history=track.history,
            parent_id=track.parent_id,
            scan_created=track.scan_created,
            n_hits=track.n_hits,
            n_misses=track.n_misses,
        )
        self.tracks[track_id] = new_track
        return track_id

    def prune(self) -> None:
        """Apply all pruning strategies."""
        # Probability-based pruning
        self.hypotheses = prune_hypotheses_by_probability(
            self.hypotheses,
            self.max_hypotheses,
            self.min_probability,
        )

        # N-scan pruning
        self.hypotheses, committed = n_scan_prune(
            self.hypotheses,
            self.tracks,
            self.n_scan,
            self.current_scan,
        )

        # Remove orphaned tracks
        self._remove_orphaned_tracks()

    def _remove_orphaned_tracks(self) -> None:
        """Remove tracks not reachable from any hypothesis.

        A branch is reachable if a hypothesis references it directly or
        it is an ancestor (via ``parent_id``) of a referenced branch --
        ancestors must survive because N-scan pruning walks the parent
        chain to compare committed decisions.
        """
        referenced: Set[int] = set()
        stack = [tid for hyp in self.hypotheses for tid in hyp.track_ids]
        while stack:
            tid = stack.pop()
            if tid in referenced or tid not in self.tracks:
                continue
            referenced.add(tid)
            parent = self.tracks[tid].parent_id
            if parent >= 0:
                stack.append(parent)

        orphans = set(self.tracks.keys()) - referenced
        for track_id in orphans:
            del self.tracks[track_id]

    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        """Get the most probable hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.probability)

    def get_best_tracks(self) -> List[MHTTrack]:
        """Get tracks from the best hypothesis."""
        best = self.get_best_hypothesis()
        if best is None:
            return []
        return [self.tracks[tid] for tid in best.track_ids if tid in self.tracks]

    def get_confirmed_tracks(self) -> List[MHTTrack]:
        """Get confirmed tracks from the best hypothesis."""
        tracks = self.get_best_tracks()
        return [t for t in tracks if t.status == MHTTrackStatus.CONFIRMED]


__all__ = [
    "MHTTrackStatus",
    "MHTTrack",
    "Hypothesis",
    "generate_joint_associations",
    "compute_association_likelihood",
    "n_scan_prune",
    "prune_hypotheses_by_probability",
    "HypothesisTree",
]
