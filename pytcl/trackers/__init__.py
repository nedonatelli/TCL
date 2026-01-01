"""
End-to-end tracker implementations.

This module provides complete tracker implementations that combine
filtering, data association, and track management.
"""

from pytcl.trackers.hypothesis import Hypothesis
from pytcl.trackers.hypothesis import HypothesisAssignment
from pytcl.trackers.hypothesis import HypothesisTree
from pytcl.trackers.hypothesis import MHTTrack
from pytcl.trackers.hypothesis import MHTTrackStatus
from pytcl.trackers.hypothesis import compute_association_likelihood
from pytcl.trackers.hypothesis import generate_joint_associations
from pytcl.trackers.hypothesis import n_scan_prune
from pytcl.trackers.hypothesis import prune_hypotheses_by_probability
from pytcl.trackers.mht import MHTConfig
from pytcl.trackers.mht import MHTResult
from pytcl.trackers.mht import MHTTracker
from pytcl.trackers.multi_target import MultiTargetTracker
from pytcl.trackers.multi_target import Track
from pytcl.trackers.multi_target import TrackStatus
from pytcl.trackers.single_target import SingleTargetTracker
from pytcl.trackers.single_target import TrackState

__all__ = [
    # Single target
    "SingleTargetTracker",
    "TrackState",
    # Multi-target (GNN-based)
    "MultiTargetTracker",
    "Track",
    "TrackStatus",
    # MHT hypothesis management
    "MHTTrackStatus",
    "MHTTrack",
    "Hypothesis",
    "HypothesisAssignment",
    "HypothesisTree",
    "generate_joint_associations",
    "compute_association_likelihood",
    "n_scan_prune",
    "prune_hypotheses_by_probability",
    # MHT tracker
    "MHTConfig",
    "MHTResult",
    "MHTTracker",
]
