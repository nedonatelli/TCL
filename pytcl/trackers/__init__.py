"""
End-to-end tracker implementations.

This module provides complete tracker implementations that combine
filtering, data association, and track management.
"""

from pytcl.trackers.hypothesis import (
    Hypothesis,
    HypothesisAssignment,
    HypothesisTree,
    MHTTrack,
    MHTTrackStatus,
    compute_association_likelihood,
    generate_joint_associations,
    n_scan_prune,
    prune_hypotheses_by_probability,
)
from pytcl.trackers.mht import MHTConfig, MHTResult, MHTTracker
from pytcl.trackers.multi_target import MultiTargetTracker, Track, TrackStatus
from pytcl.trackers.single_target import SingleTargetTracker, TrackState

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
