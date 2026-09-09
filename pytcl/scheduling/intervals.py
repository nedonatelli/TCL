"""
Interval scheduling algorithms.

Ports of the MATLAB TCL's ``Scheduling`` directory: the classic
greedy and dynamic-programming interval problems -- maximum-cardinality
selection, minimum resource partitioning, minimum-lateness sequencing
and maximum-weight selection. Intervals are ``[start; finish]``
columns of a (2, N) array, matching the MATLAB layout; job indices in
the results are 0-based.

References
----------
- J. Kleinberg and E. Tardos, Algorithm Design, 1st ed. Pearson,
  2005: Chapter 4.1 (interval scheduling and partitioning), 4.2
  (minimum lateness) and 6.1-6.2 (weighted interval scheduling) --
  the chapters the MATLAB implementations cite.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "schedule_intervals",
    "partition_intervals",
    "schedule_min_lateness_dense",
    "schedule_weighted_intervals",
]


def _cols(intervals: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(2, 1)
    return arr


def schedule_intervals(intervals: ArrayLike) -> NDArray[np.intp]:
    """
    Maximum-cardinality set of non-overlapping intervals.

    The earliest-finish-time greedy algorithm; an interval may start
    exactly when the previous one finishes.

    Port of ``scheduleIntervals``.

    Parameters
    ----------
    intervals : array_like
        (2, N) columns [start; finish].

    Returns
    -------
    jobs : ndarray
        Indices of the selected intervals, in the order scheduled.

    Examples
    --------
    >>> schedule_intervals([[0.0, 1.0, 3.0], [2.0, 4.0, 5.0]])
    array([0, 2])
    """
    arr = _cols(intervals)
    if arr.size == 0:
        return np.array([], dtype=np.intp)
    idx = np.argsort(arr[1, :], kind="stable")
    ordered = arr[:, idx]
    selected = [int(idx[0])]
    last_finish = ordered[1, 0]
    for k in range(1, ordered.shape[1]):
        if ordered[0, k] >= last_finish:
            selected.append(int(idx[k]))
            last_finish = ordered[1, k]
    return np.array(selected, dtype=np.intp)


def partition_intervals(intervals: ArrayLike) -> NDArray[np.intp]:
    """
    Partition intervals into a minimum number of compatible groups.

    The earliest-start greedy algorithm: each interval joins the first
    existing group whose last job has finished, or opens a new group.
    The number of groups equals the depth of the interval set.

    Port of ``partitionIntervals``.

    Parameters
    ----------
    intervals : array_like
        (2, N) columns [start; finish].

    Returns
    -------
    partition : ndarray
        For each interval, its 0-based group index.

    Examples
    --------
    >>> partition_intervals([[0.0, 1.0, 3.0], [2.0, 4.0, 5.0]])
    array([0, 1, 0])
    """
    arr = _cols(intervals)
    if arr.size == 0:
        return np.array([], dtype=np.intp)
    n = arr.shape[1]
    idx = np.argsort(arr[0, :], kind="stable")
    ordered = arr[:, idx]
    partition_sorted = np.zeros(n, dtype=np.intp)
    finish_times = [ordered[1, 0]]
    for k in range(1, n):
        found = -1
        for part, ft in enumerate(finish_times):
            if ft <= ordered[0, k]:
                found = part
                break
        if found < 0:
            finish_times.append(ordered[1, k])
            found = len(finish_times) - 1
        else:
            finish_times[found] = ordered[1, k]
        partition_sorted[k] = found
    out = np.zeros(n, dtype=np.intp)
    out[idx] = partition_sorted
    return out


def schedule_min_lateness_dense(
    deadlines: ArrayLike,
    durations: ArrayLike,
    t_start: float = 0.0,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """
    Sequence jobs back-to-back to minimize the maximum lateness.

    The earliest-deadline-first rule; jobs run densely (no idle time)
    from ``t_start``.

    Port of ``scheduleMinLatenessDense``.

    Parameters
    ----------
    deadlines : array_like
        (N,) job deadlines.
    durations : array_like
        (N,) job durations.
    t_start : float, optional
        Time the first job starts. Default 0.

    Returns
    -------
    jobs : ndarray
        Job indices in execution order.
    t_starts : ndarray
        The start time of each scheduled job, in the same order.

    Examples
    --------
    >>> jobs, starts = schedule_min_lateness_dense([6.0, 2.0], [1.0, 2.0])
    >>> jobs
    array([1, 0])
    >>> starts
    array([0., 2.])
    """
    dl = np.asarray(deadlines, dtype=np.float64).reshape(-1)
    if dl.size == 0:
        return np.array([], dtype=np.intp), np.array([])
    dur = np.asarray(durations, dtype=np.float64).reshape(-1)
    jobs = np.argsort(dl, kind="stable")
    t_starts = t_start + np.concatenate([[0.0], np.cumsum(dur[jobs])[:-1]])
    return jobs.astype(np.intp), t_starts


def schedule_weighted_intervals(
    intervals: ArrayLike, weights: ArrayLike
) -> tuple[float, NDArray[np.intp]]:
    """
    Maximum-weight set of non-overlapping intervals.

    The classic dynamic program over intervals sorted by finish time,
    with the compatible-predecessor table built by binary search.

    Port of ``scheduleWeightedIntervals``. Weights are assumed
    positive, as upstream (the MATLAB traceback loop does not
    terminate for a leading run of non-positive optima).

    Parameters
    ----------
    intervals : array_like
        (2, N) columns [start; finish].
    weights : array_like
        (N,) positive interval weights.

    Returns
    -------
    weight : float
        The total weight of the optimal selection.
    jobs : ndarray
        Indices of the selected intervals.

    Examples
    --------
    >>> w, jobs = schedule_weighted_intervals(
    ...     [[0.0, 1.0, 3.0], [2.0, 4.0, 5.0]], [1.0, 5.0, 1.0])
    >>> w
    5.0
    >>> jobs
    array([1])
    """
    arr = _cols(intervals)
    if arr.size == 0:
        return 0.0, np.array([], dtype=np.intp)
    n = arr.shape[1]
    idx = np.argsort(arr[1, :], kind="stable")
    ordered = arr[:, idx]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)[idx]

    # p[k]: 1-based index of the rightmost interval finishing no later
    # than interval k starts; 0 when none is compatible.
    finishes = ordered[1, :]
    p = np.zeros(n, dtype=np.intp)
    for k in range(1, n):
        j = int(np.searchsorted(finishes[:k], ordered[0, k], side="right"))
        p[k] = j  # 1-based by construction (0 means incompatible)

    opt = np.zeros(n + 1)
    for k in range(1, n + 1):
        take = w[k - 1] + opt[p[k - 1]]
        opt[k] = max(take, opt[k - 1])
    weight_val = float(opt[n])

    selected = []
    k = n
    while k > 0:
        if w[k - 1] + opt[p[k - 1]] > opt[k - 1]:
            selected.append(int(idx[k - 1]))
            k = int(p[k - 1])
        else:
            k -= 1
    return weight_val, np.array(selected, dtype=np.intp)
