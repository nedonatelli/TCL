"""
Track performance metrics for multi-target tracking evaluation.

This module provides metrics for evaluating multi-target tracker performance,
including OSPA (Optimal Sub-Pattern Assignment), track purity, and fragmentation.

References
----------
.. [1] D. Schuhmacher, B.-T. Vo, and B.-N. Vo, "A Consistent Metric for
       Performance Evaluation of Multi-Object Filters," IEEE Trans. Signal
       Processing, vol. 56, no. 8, pp. 3447-3457, Aug. 2008.
.. [2] K. Bernardin and R. Stiefelhagen, "Evaluating Multiple Object Tracking
       Performance: The CLEAR MOT Metrics," EURASIP J. Image Video Process.,
       2008.
"""

from typing import List, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from pytcl.assignment_algorithms import hungarian


class OSPAResult(NamedTuple):
    """
    Result of OSPA metric computation.

    Attributes
    ----------
    ospa : float
        Total OSPA distance.
    localization : float
        Localization component.
    cardinality : float
        Cardinality component.
    """

    ospa: float
    localization: float
    cardinality: float


class MOTMetrics(NamedTuple):
    """
    Multiple Object Tracking (MOT) metrics.

    Attributes
    ----------
    mota : float
        Multiple Object Tracking Accuracy.
    motp : float
        Multiple Object Tracking Precision.
    num_switches : int
        Number of identity switches.
    num_fragmentations : int
        Number of track fragmentations.
    num_false_positives : int
        Number of false positive detections.
    num_misses : int
        Number of missed detections.
    """

    mota: float
    motp: float
    num_switches: int
    num_fragmentations: int
    num_false_positives: int
    num_misses: int


def ospa(
    X: List[NDArray[np.float64]],
    Y: List[NDArray[np.float64]],
    c: float = 100.0,
    p: float = 2.0,
) -> OSPAResult:
    """
    Compute Optimal Sub-Pattern Assignment (OSPA) metric.

    The OSPA metric provides a mathematically consistent measure of distance
    between two sets of points, accounting for both localization error and
    cardinality mismatch.

    Parameters
    ----------
    X : list of ndarray
        First set of points (e.g., ground truth).
    Y : list of ndarray
        Second set of points (e.g., estimated tracks).
    c : float, optional
        Cutoff parameter for localization error (default: 100.0).
    p : float, optional
        Order parameter for the metric (default: 2.0).

    Returns
    -------
    OSPAResult
        Named tuple containing:
        - ospa: Total OSPA distance
        - localization: Localization component
        - cardinality: Cardinality component

    Notes
    -----
    - If both sets are empty, OSPA is 0.
    - The metric is symmetric: ospa(X, Y) = ospa(Y, X).
    - For p=2 (default), the metric is a proper L2 distance.

    Examples
    --------
    >>> X = [np.array([0, 0]), np.array([10, 10])]
    >>> Y = [np.array([1, 0]), np.array([10, 11])]
    >>> result = ospa(X, Y, c=100, p=2)
    >>> result.ospa  # doctest: +SKIP
    1.118...
    """
    m = len(X)
    n = len(Y)

    # Handle empty sets
    if m == 0 and n == 0:
        return OSPAResult(ospa=0.0, localization=0.0, cardinality=0.0)

    if m == 0 or n == 0:
        # All cardinality error
        cardinality = c
        return OSPAResult(ospa=cardinality, localization=0.0, cardinality=cardinality)

    # Ensure m <= n (swap if needed)
    if m > n:
        X, Y = Y, X
        m, n = n, m

    # Compute pairwise distances
    cost_matrix = np.zeros((m, n))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            d = np.linalg.norm(np.asarray(x) - np.asarray(y))
            cost_matrix[i, j] = min(d, c) ** p

    # Solve assignment problem
    row_ind, col_ind, _ = hungarian(cost_matrix)

    # Compute localization component
    localization_sum = 0.0
    for i, j in zip(row_ind, col_ind):
        localization_sum += cost_matrix[i, j]

    # Cardinality penalty
    cardinality_penalty = (n - m) * (c**p)

    # OSPA distance
    total = localization_sum + cardinality_penalty
    ospa_val = (total / n) ** (1.0 / p)

    # Decompose into components
    loc_component = (localization_sum / n) ** (1.0 / p) if localization_sum > 0 else 0.0
    card_component = (cardinality_penalty / n) ** (1.0 / p) if n > m else 0.0

    return OSPAResult(
        ospa=ospa_val, localization=loc_component, cardinality=card_component
    )


def ospa_over_time(
    X_sequence: List[List[NDArray[np.float64]]],
    Y_sequence: List[List[NDArray[np.float64]]],
    c: float = 100.0,
    p: float = 2.0,
) -> NDArray[np.float64]:
    """
    Compute OSPA metric over a time sequence.

    Parameters
    ----------
    X_sequence : list of list of ndarray
        Sequence of ground truth point sets.
    Y_sequence : list of list of ndarray
        Sequence of estimated point sets.
    c : float, optional
        Cutoff parameter (default: 100.0).
    p : float, optional
        Order parameter (default: 2.0).

    Returns
    -------
    ndarray
        OSPA values at each time step.

    Raises
    ------
    ValueError
        If sequences have different lengths.
    """
    if len(X_sequence) != len(Y_sequence):
        raise ValueError("Sequences must have the same length")

    ospa_values = np.zeros(len(X_sequence))
    for k, (X, Y) in enumerate(zip(X_sequence, Y_sequence)):
        result = ospa(X, Y, c, p)
        ospa_values[k] = result.ospa

    return ospa_values


def track_purity(
    true_labels: NDArray[np.int_],
    estimated_labels: NDArray[np.int_],
) -> float:
    """
    Compute track purity metric.

    Track purity measures how well estimated tracks correspond to single
    ground truth targets. A purity of 1.0 means each estimated track contains
    observations from only one true target.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth target labels for each observation.
    estimated_labels : ndarray
        Estimated track labels for each observation.

    Returns
    -------
    float
        Track purity score in [0, 1].

    Examples
    --------
    >>> true_labels = np.array([0, 0, 0, 1, 1, 1])
    >>> estimated_labels = np.array([0, 0, 0, 1, 1, 1])  # Perfect
    >>> track_purity(true_labels, estimated_labels)
    1.0
    >>> estimated_labels = np.array([0, 0, 1, 1, 1, 1])  # Mixed
    >>> track_purity(true_labels, estimated_labels)  # doctest: +SKIP
    0.833...
    """
    true_labels = np.asarray(true_labels)
    estimated_labels = np.asarray(estimated_labels)

    if len(true_labels) != len(estimated_labels):
        raise ValueError("Label arrays must have the same length")

    if len(true_labels) == 0:
        return 1.0

    unique_est = np.unique(estimated_labels)
    total_correct = 0

    for est_label in unique_est:
        mask = estimated_labels == est_label
        true_in_track = true_labels[mask]
        # Count most frequent true label
        if len(true_in_track) > 0:
            _, counts = np.unique(true_in_track, return_counts=True)
            total_correct += counts.max()

    return total_correct / len(true_labels)


def track_fragmentation(
    true_labels: NDArray[np.int_],
    estimated_labels: NDArray[np.int_],
    time_indices: Optional[NDArray[np.int_]] = None,
) -> int:
    """
    Count number of track fragmentations.

    A fragmentation occurs when observations from a single ground truth
    target are split across multiple estimated tracks.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth target labels for each observation.
    estimated_labels : ndarray
        Estimated track labels for each observation.
    time_indices : ndarray, optional
        Time indices for each observation (for temporal ordering).

    Returns
    -------
    int
        Number of fragmentations.

    Examples
    --------
    >>> true_labels = np.array([0, 0, 0, 0])
    >>> estimated_labels = np.array([0, 0, 1, 1])  # One fragmentation
    >>> track_fragmentation(true_labels, estimated_labels)
    1
    """
    true_labels = np.asarray(true_labels)
    estimated_labels = np.asarray(estimated_labels)

    if time_indices is not None:
        # Sort by time
        sort_idx = np.argsort(time_indices)
        true_labels = true_labels[sort_idx]
        estimated_labels = estimated_labels[sort_idx]

    fragmentations = 0
    unique_true = np.unique(true_labels)

    for true_label in unique_true:
        mask = true_labels == true_label
        est_for_target = estimated_labels[mask]

        # Count transitions between different estimated tracks
        for i in range(1, len(est_for_target)):
            if est_for_target[i] != est_for_target[i - 1]:
                fragmentations += 1

    return fragmentations


def identity_switches(
    true_labels: NDArray[np.int_],
    estimated_labels: NDArray[np.int_],
    time_indices: Optional[NDArray[np.int_]] = None,
) -> int:
    """
    Count number of identity switches.

    An identity switch occurs when an estimated track changes which
    ground truth target it is associated with.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth target labels for each observation.
    estimated_labels : ndarray
        Estimated track labels for each observation.
    time_indices : ndarray, optional
        Time indices for each observation.

    Returns
    -------
    int
        Number of identity switches.
    """
    true_labels = np.asarray(true_labels)
    estimated_labels = np.asarray(estimated_labels)

    if time_indices is not None:
        sort_idx = np.argsort(time_indices)
        true_labels = true_labels[sort_idx]
        estimated_labels = estimated_labels[sort_idx]

    switches = 0
    unique_est = np.unique(estimated_labels)

    for est_label in unique_est:
        mask = estimated_labels == est_label
        true_for_track = true_labels[mask]

        # Count transitions between different true targets
        for i in range(1, len(true_for_track)):
            if true_for_track[i] != true_for_track[i - 1]:
                switches += 1

    return switches


def mot_metrics(
    ground_truth: List[List[NDArray[np.float64]]],
    estimates: List[List[NDArray[np.float64]]],
    threshold: float = 10.0,
) -> MOTMetrics:
    """
    Compute CLEAR MOT metrics.

    Parameters
    ----------
    ground_truth : list of list of ndarray
        Ground truth positions at each time step.
    estimates : list of list of ndarray
        Estimated positions at each time step.
    threshold : float, optional
        Distance threshold for valid associations (default: 10.0).

    Returns
    -------
    MOTMetrics
        Named tuple containing MOTA, MOTP, and counts.

    Notes
    -----
    MOTA (Multiple Object Tracking Accuracy) accounts for false positives,
    misses, and identity switches. MOTP (Precision) measures localization
    accuracy for correctly matched pairs.
    """
    total_gt = 0
    total_fp = 0
    total_misses = 0
    total_switches = 0
    total_frags = 0
    total_distance = 0.0
    total_matches = 0

    prev_assignment: dict[int, int] = {}  # gt_idx -> est_idx from previous frame

    for k, (gt_frame, est_frame) in enumerate(zip(ground_truth, estimates)):
        n_gt = len(gt_frame)
        n_est = len(est_frame)
        total_gt += n_gt

        if n_gt == 0 and n_est == 0:
            continue

        if n_gt == 0:
            total_fp += n_est
            continue

        if n_est == 0:
            total_misses += n_gt
            prev_assignment = {}
            continue

        # Build cost matrix
        cost_matrix = np.zeros((n_gt, n_est))
        for i, gt in enumerate(gt_frame):
            for j, est in enumerate(est_frame):
                cost_matrix[i, j] = np.linalg.norm(np.asarray(gt) - np.asarray(est))

        # Solve assignment
        row_ind, col_ind, _ = hungarian(cost_matrix)

        # Count matches within threshold
        current_assignment = {}
        frame_matches = 0
        frame_distance = 0.0

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= threshold:
                current_assignment[i] = j
                frame_matches += 1
                frame_distance += cost_matrix[i, j]

                # Check for identity switch
                if i in prev_assignment and prev_assignment[i] != j:
                    total_switches += 1

        total_matches += frame_matches
        total_distance += frame_distance
        total_misses += n_gt - frame_matches
        total_fp += n_est - frame_matches

        prev_assignment = current_assignment

    # Compute metrics
    if total_gt > 0:
        mota = 1.0 - (total_misses + total_fp + total_switches) / total_gt
    else:
        mota = 1.0

    if total_matches > 0:
        motp = total_distance / total_matches
    else:
        motp = 0.0

    return MOTMetrics(
        mota=mota,
        motp=motp,
        num_switches=total_switches,
        num_fragmentations=total_frags,
        num_false_positives=total_fp,
        num_misses=total_misses,
    )


__all__ = [
    "OSPAResult",
    "MOTMetrics",
    "ospa",
    "ospa_over_time",
    "track_purity",
    "track_fragmentation",
    "identity_switches",
    "mot_metrics",
]
