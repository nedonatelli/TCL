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
from pytcl.assignment_algorithms.dijkstra_min_cost import (
    min_cost_flow_dijkstra_potentials,
)
from pytcl.assignment_algorithms.gating import (
    chi2_gate_threshold,
    compute_gate_volume,
    ellipsoidal_gate,
    gate_measurements,
    mahalanobis_batch,
    mahalanobis_distance,
    rectangular_gate,
)
from pytcl.assignment_algorithms.jpda import (
    JPDAResult,
    JPDAUpdate,
    compute_likelihood_matrix,
    compute_measurement_likelihood,
    jpda,
    jpda_probabilities,
    jpda_update,
)
from pytcl.assignment_algorithms.nd_assignment import (
    AssignmentNDResult,
    SparseCostTensor,
    assignment_nd,
    auction_assignment_nd,
    detect_dimension_conflicts,
    greedy_assignment_nd,
    greedy_assignment_nd_sparse,
    relaxation_assignment_nd,
    validate_cost_tensor,
)
from pytcl.assignment_algorithms.network_flow import (
    FlowEdge,
    FlowStatus,
    MinCostFlowResult,
    assignment_from_flow_solution,
    assignment_to_flow_network,
    min_cost_assignment_via_flow,
    min_cost_flow_simplex,
    min_cost_flow_successive_shortest_paths,
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
    # Each of these has an exported sibling -- mahalanobis_distance,
    # compute_likelihood_matrix, greedy_assignment_nd,
    # min_cost_flow_successive_shortest_paths -- so the split was arbitrary.
    "mahalanobis_batch",
    "compute_measurement_likelihood",
    "greedy_assignment_nd_sparse",
    "min_cost_flow_dijkstra_potentials",
    # Min-cost flow. The split here had no rationale: min_cost_flow_simplex was
    # private while its result type was public and its input type FlowEdge was
    # not, so the function could not be called even if imported. The migration
    # guide names it as a replacement for the removed network_simplex module.
    "FlowEdge",
    "min_cost_flow_simplex",
    "assignment_from_flow_solution",
    # N-dimensional assignment
    "SparseCostTensor",
    "assignment_nd",
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
    # N-Dimensional Assignment (4D+)
    "AssignmentNDResult",
    "validate_cost_tensor",
    "greedy_assignment_nd",
    "relaxation_assignment_nd",
    "auction_assignment_nd",
    "detect_dimension_conflicts",
    # Network Flow-Based Assignment
    "FlowStatus",
    "MinCostFlowResult",
    "assignment_to_flow_network",
    "min_cost_flow_successive_shortest_paths",
    "min_cost_assignment_via_flow",
]
