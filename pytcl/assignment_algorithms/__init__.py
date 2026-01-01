"""
Assignment algorithms for data association in target tracking.

This module provides:
- 2D assignment algorithms (Hungarian, Auction)
- K-best 2D assignment (Murty's algorithm)
- 3D assignment algorithms (Lagrangian relaxation, Auction)
- Gating methods (ellipsoidal, rectangular)
- Data association algorithms (GNN, JPDA)
"""

from pytcl.assignment_algorithms.data_association import AssociationResult
from pytcl.assignment_algorithms.data_association import compute_association_cost
from pytcl.assignment_algorithms.data_association import gated_gnn_association
from pytcl.assignment_algorithms.data_association import gnn_association
from pytcl.assignment_algorithms.data_association import nearest_neighbor
from pytcl.assignment_algorithms.gating import chi2_gate_threshold
from pytcl.assignment_algorithms.gating import compute_gate_volume
from pytcl.assignment_algorithms.gating import ellipsoidal_gate
from pytcl.assignment_algorithms.gating import gate_measurements
from pytcl.assignment_algorithms.gating import mahalanobis_distance
from pytcl.assignment_algorithms.gating import rectangular_gate
from pytcl.assignment_algorithms.jpda import JPDAResult
from pytcl.assignment_algorithms.jpda import JPDAUpdate
from pytcl.assignment_algorithms.jpda import compute_likelihood_matrix
from pytcl.assignment_algorithms.jpda import jpda
from pytcl.assignment_algorithms.jpda import jpda_probabilities
from pytcl.assignment_algorithms.jpda import jpda_update
from pytcl.assignment_algorithms.three_dimensional import Assignment3DResult
from pytcl.assignment_algorithms.three_dimensional import assign3d
from pytcl.assignment_algorithms.three_dimensional import assign3d_auction
from pytcl.assignment_algorithms.three_dimensional import assign3d_lagrangian
from pytcl.assignment_algorithms.three_dimensional import decompose_to_2d
from pytcl.assignment_algorithms.three_dimensional import greedy_3d
from pytcl.assignment_algorithms.two_dimensional import AssignmentResult
from pytcl.assignment_algorithms.two_dimensional import KBestResult
from pytcl.assignment_algorithms.two_dimensional import assign2d
from pytcl.assignment_algorithms.two_dimensional import auction
from pytcl.assignment_algorithms.two_dimensional import hungarian
from pytcl.assignment_algorithms.two_dimensional import kbest_assign2d
from pytcl.assignment_algorithms.two_dimensional import linear_sum_assignment
from pytcl.assignment_algorithms.two_dimensional import murty
from pytcl.assignment_algorithms.two_dimensional import ranked_assignments

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
