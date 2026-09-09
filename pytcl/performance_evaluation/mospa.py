"""
MOSPA/MMOSPA metrics and estimates.

Ports of the MATLAB TCL's MOSPA family: the mean OSPA error of an
estimate over a weighted hypothesis set (``calcMOSPAError``), the
exact minimum-MOSPA estimate for two 2-D targets (``MMOSPA2Tar2D``)
and the sweep-based approximation for the general case
(``MMOSPAApprox``). Target states are columns: ``x`` has shape
(x_dim, num_targets, num_hypotheses) with per-hypothesis weights
``w``.

References
----------
- D. F. Crouse, P. Willett, and Y. Bar-Shalom, "Developing a
  real-time track display that operators do not hate," IEEE
  Transactions on Signal Processing, vol. 59, no. 7, Jul. 2011.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.assignment_algorithms.two_dimensional.assignment import assign2d

__all__ = [
    "calc_mospa_error",
    "mmospa2tar_2d",
    "mmospa_approx",
]


def _col4row(cost: NDArray[np.float64]) -> NDArray[np.intp]:
    """MATLAB assign2D semantics: for each row, its assigned column."""
    res = assign2d(cost)
    out = np.empty(cost.shape[0], dtype=np.intp)
    out[res.row_indices] = res.col_indices
    return out


def calc_mospa_error(x_est: ArrayLike, x: ArrayLike, w: ArrayLike) -> float:
    """
    Mean OSPA error of an estimate over a weighted hypothesis set.

    For each hypothesis, the targets are optimally permuted against
    the estimate (a 2-D assignment maximizing the inner product) and
    the weighted mean squared error accumulated; the result is the
    root of the per-target average.

    Port of ``calcMOSPAError``.

    Parameters
    ----------
    x_est : array_like
        (x_dim, num_targets) estimate.
    x : array_like
        (x_dim, num_targets, num_hyp) hypothesis states.
    w : array_like
        (num_hyp,) hypothesis weights (summing to one).

    Returns
    -------
    val : float
        The MOSPA error.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((2, 2, 1)); x[:, 1, 0] = [1.0, 0.0]
    >>> round(calc_mospa_error(x[:, :, 0], x, [1.0]), 12)
    0.0
    """
    est = np.asarray(x_est, dtype=np.float64)
    hyp = np.asarray(x, dtype=np.float64)
    wv = np.asarray(w, dtype=np.float64).reshape(-1)
    num_tar = hyp.shape[1]
    val = 0.0
    for k in range(hyp.shape[2]):
        cost = -(est.T @ hyp[:, :, k])
        order = _col4row(cost)
        diff = est - hyp[:, order, k]
        val += wv[k] * float(np.sum(diff * diff))
    return float(np.sqrt(val / num_tar))


def mmospa2tar_2d(particles: ArrayLike) -> NDArray[np.float64]:
    """
    Exact MMOSPA estimate for two 2-D targets from particles.

    Particles are sorted by the direction of the inter-target
    displacement (modulo pi) and the optimal split point of the
    progressive swap sweep selected exactly.

    Port of ``MMOSPA2Tar2D``.

    Parameters
    ----------
    particles : array_like
        (4, N) stacked [x1; y1; x2; y2] particle states, equally
        weighted.

    Returns
    -------
    mmospa : ndarray
        (4,) the MMOSPA estimate of the stacked states.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    >>> mmospa2tar_2d(p)
    array([0., 0., 1., 1.])
    """
    p = np.array(particles, dtype=np.float64)
    if p.shape[0] != 4:
        raise ValueError("This function only works with two 2D states.")
    n = p.shape[1]
    d = p[0:2, :] - p[2:4, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = np.arctan(d[1, :] / d[0, :])
    perm = np.argsort(direction, kind="stable")
    p = p[:, perm]
    # Canonicalize each particle: first target to the -x side.
    for j in range(n):
        if -(p[0, j] - p[2, j]) < 0:
            p[:, j] = p[[2, 3, 0, 1], j]
    mospa_temp = p.mean(axis=1)
    max_reward = float(mospa_temp @ mospa_temp)
    mmospa = mospa_temp.copy()
    for j in range(n):
        switched = p[[2, 3, 0, 1], j] / n
        temp = mospa_temp - p[:, j] / n
        mospa_temp = temp + switched
        reward = float(temp @ temp + 2.0 * temp @ switched + switched @ switched)
        if reward > max_reward:
            max_reward = reward
            mmospa = mospa_temp.copy()
    return mmospa


def mmospa_approx(
    x: ArrayLike, w: ArrayLike, num_scans: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """
    Approximate MMOSPA estimate over a weighted hypothesis set.

    A forward pass orders each hypothesis against the running estimate
    by optimal 2-D assignment; additional scans sweep backward and
    forward re-evaluating each hypothesis against the estimate with
    that hypothesis removed.

    Port of ``MMOSPAApprox``.

    Parameters
    ----------
    x : array_like
        (x_dim, num_targets, num_hyp) hypothesis states.
    w : array_like
        (num_hyp,) hypothesis weights.
    num_scans : int, optional
        Total passes; 1 is the forward pass alone. Default 1.

    Returns
    -------
    mmospa_est : ndarray
        (x_dim, num_targets) the approximate MMOSPA estimate.
    order_list : ndarray
        (num_targets, num_hyp) the target ordering applied to each
        hypothesis (0-based).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((2, 2, 2))
    >>> x[:, :, 0] = [[0.0, 1.0], [0.0, 0.0]]
    >>> x[:, :, 1] = [[1.0, 0.0], [0.0, 0.0]]  # swapped ordering
    >>> est, orders = mmospa_approx(x, [0.5, 0.5])
    >>> est[0]
    array([0., 1.])
    """
    hyp = np.asarray(x, dtype=np.float64)
    wv = np.asarray(w, dtype=np.float64).reshape(-1)
    num_tar = hyp.shape[1]
    num_hyp = hyp.shape[2]

    order_list = np.zeros((num_tar, num_hyp), dtype=np.intp)
    order_list[:, 0] = np.arange(num_tar)
    est = wv[0] * hyp[:, :, 0]
    for k in range(1, num_hyp):
        cost = -(est.T @ hyp[:, :, k])
        order_list[:, k] = _col4row(cost)
        est = est + wv[k] * hyp[:, order_list[:, k], k]

    def _update(k: int, est: NDArray[np.float64]) -> NDArray[np.float64]:
        x_opt = est - hyp[:, order_list[:, k], k] * wv[k]
        cost = -(x_opt.T @ hyp[:, :, k])
        order_list[:, k] = _col4row(cost)
        return x_opt + wv[k] * hyp[:, order_list[:, k], k]

    for _ in range(num_scans - 1):
        for k in range(num_hyp - 2, 0, -1):
            est = _update(k, est)
        for k in range(2, num_hyp):
            est = _update(k, est)
    return est, order_list
