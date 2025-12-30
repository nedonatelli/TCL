"""
Assignment algorithms for data association in target tracking.

This module provides:
- 2D assignment algorithms (Hungarian, Auction, JVC)
- Gating methods (ellipsoidal, rectangular)
- Data association algorithms (GNN, JPDA)
"""

from pytcl.assignment_algorithms.two_dimensional import (
    hungarian,
    auction,
    linear_sum_assignment,
    assign2d,
    AssignmentResult,
)

from pytcl.assignment_algorithms.gating import (
    ellipsoidal_gate,
    rectangular_gate,
    gate_measurements,
    mahalanobis_distance,
    chi2_gate_threshold,
    compute_gate_volume,
)

from pytcl.assignment_algorithms.data_association import (
    gnn_association,
    nearest_neighbor,
    compute_association_cost,
    gated_gnn_association,
    AssociationResult,
)

from pytcl.assignment_algorithms.jpda import (
    JPDAResult,
    JPDAUpdate,
    jpda,
    jpda_update,
    jpda_probabilities,
    compute_likelihood_matrix,
)

__all__ = [
    # 2D Assignment
    "hungarian",
    "auction",
    "linear_sum_assignment",
    "assign2d",
    "AssignmentResult",
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
