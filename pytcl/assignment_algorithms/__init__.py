"""
Assignment algorithms for data association in target tracking.

This module provides:
- 2D assignment algorithms (Hungarian, Auction)
- K-best 2D assignment (Murty's algorithm)
- 3D assignment algorithms (Lagrangian relaxation, Auction)
- Gating methods (ellipsoidal, rectangular)
- Data association algorithms (GNN, JPDA)
"""

from pytcl.assignment_algorithms.data_association import (
    AssociationResult,
    compute_association_cost,
    gated_gnn_association,
    gnn_association,
    nearest_neighbor,
)
from pytcl.assignment_algorithms.gating import (
    chi2_gate_threshold,
    compute_gate_volume,
    ellipsoidal_gate,
    gate_measurements,
    mahalanobis_distance,
    rectangular_gate,
)
from pytcl.assignment_algorithms.jpda import (
    JPDAResult,
    JPDAUpdate,
    compute_likelihood_matrix,
    jpda,
    jpda_probabilities,
    jpda_update,
)
from pytcl.assignment_algorithms.three_dimensional import (
    Assignment3DResult,
    assign3d,
    assign3d_auction,
    assign3d_lagrangian,
    decompose_to_2d,
    greedy_3d,
)
from pytcl.assignment_algorithms.two_dimensional import (
    AssignmentResult,
    KBestResult,
    assign2d,
    auction,
    hungarian,
    kbest_assign2d,
    linear_sum_assignment,
    murty,
    ranked_assignments,
)

__all__ = [
    # 2D Assignment
    "hungarian",
    "auction",
    "linear_sum_assignment",
    "assign2d",
    "AssignmentResult",
    # K-Best 2D Assignment
    "KBestResult",
    "murty",
    "kbest_assign2d",
    "ranked_assignments",
    # 3D Assignment
    "Assignment3DResult",
    "assign3d",
    "assign3d_lagrangian",
    "assign3d_auction",
    "greedy_3d",
    "decompose_to_2d",
    # Gating
    "ellipsoidal_gate",
    "rectangular_gate",
    "gate_measurements",
    "mahalanobis_distance",
    "chi2_gate_threshold",
    "compute_gate_volume",
    # Data Association
    "gnn_association",
    "nearest_neighbor",
    "compute_association_cost",
    "gated_gnn_association",
    "AssociationResult",
    # JPDA
    "JPDAResult",
    "JPDAUpdate",
    "jpda",
    "jpda_update",
    "jpda_probabilities",
    "compute_likelihood_matrix",
]
