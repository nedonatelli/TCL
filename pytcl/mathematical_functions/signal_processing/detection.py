"""
Constant False Alarm Rate (CFAR) detection algorithms.

CFAR algorithms maintain a constant probability of false alarm by adaptively
setting detection thresholds based on local noise estimates. These are
essential for radar signal processing where the noise environment varies.

Functions
---------
- cfar_ca: Cell-Averaging CFAR
- cfar_go: Greatest-Of CFAR
- cfar_so: Smallest-Of CFAR
- cfar_os: Order-Statistic CFAR
- cfar_2d: Two-dimensional CFAR
- threshold_factor: Compute CFAR threshold multiplier
- detection_probability: Compute detection probability

References
----------
- Richards, M. A. (2014). Fundamentals of Radar Signal Processing
  (2nd ed.). McGraw-Hill.
- Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple
  target situations. IEEE Transactions on Aerospace and Electronic
  Systems, 19(4), 608-621.
"""

from math import comb
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

# =============================================================================
# Result Types
# =============================================================================


class CFARResult(NamedTuple):
    """
    Result of 1D CFAR detection.

    Attributes
    ----------
    detections : ndarray
        Boolean array indicating detections.
    threshold : ndarray
        Adaptive threshold values.
    detection_indices : ndarray
        Indices of detection points.
    noise_estimate : ndarray
        Estimated noise level at each cell.
    """

    detections: NDArray[np.bool_]
    threshold: NDArray[np.floating]
    detection_indices: NDArray[np.intp]
    noise_estimate: NDArray[np.floating]


class CFARResult2D(NamedTuple):
    """
    Result of 2D CFAR detection.

    Attributes
    ----------
    detections : ndarray
        2D boolean array indicating detections.
    threshold : ndarray
        2D adaptive threshold values.
    noise_estimate : ndarray
        2D estimated noise level.
    """

    detections: NDArray[np.bool_]
    threshold: NDArray[np.floating]
    noise_estimate: NDArray[np.floating]


# =============================================================================
# Threshold Factor Computation
# =============================================================================


def _pfa_ca(alpha: float, n: int) -> float:
    """Exact Pfa for CA-CFAR with n reference cells (exponential noise)."""
    return float((1.0 + alpha / n) ** (-n))


def _pfa_so(alpha: float, n: int) -> float:
    """Exact Pfa for SO-CFAR with n cells per half window (Gandhi & Kassam 1988)."""
    r = 2.0 + alpha / n
    return float(2.0 * sum(comb(n + j - 1, j) * r ** (-(n + j)) for j in range(n)))


def _pfa_go(alpha: float, n: int) -> float:
    """Exact Pfa for GO-CFAR with n cells per half window (Gandhi & Kassam 1988)."""
    return 2.0 * _pfa_ca(alpha, n) - _pfa_so(alpha, n)


def _pfa_os(alpha: float, n: int, k: int) -> float:
    """Exact Pfa for OS-CFAR using the k-th order statistic of n cells (Rohling 1983)."""
    p = 1.0
    for i in range(k):
        p *= (n - i) / (n - i + alpha)
    return p


def _solve_alpha(pfa_func: Callable[[float], float], pfa: float) -> float:
    """Solve pfa_func(alpha) = pfa for alpha (pfa_func monotone decreasing)."""
    hi = 1.0
    while pfa_func(hi) > pfa:
        hi *= 2.0
        if hi > 1e12:
            break
    return float(brentq(lambda a: pfa_func(a) - pfa, 0.0, hi))


def threshold_factor(
    pfa: float,
    n_ref: int,
    method: str = "ca",
    k: Optional[int] = None,
) -> float:
    """
    Compute the CFAR threshold multiplier for a given probability of false alarm.

    Parameters
    ----------
    pfa : float
        Desired probability of false alarm (0 < pfa < 1).
    n_ref : int
        Total number of reference cells. For GO/SO-CFAR each half window
        contains n_ref // 2 cells.
    method : {'ca', 'go', 'so', 'os'}, optional
        CFAR method. Default is 'ca'.
    k : int, optional
        Order statistic index for OS-CFAR (1 <= k <= n_ref).

    Returns
    -------
    alpha : float
        Threshold multiplier.

    Examples
    --------
    >>> alpha = threshold_factor(1e-6, 32, method='ca')
    >>> alpha > 1
    True

    Notes
    -----
    For CA-CFAR with n_ref reference cells, the relationship between
    threshold factor alpha and Pfa is::

        Pfa = (1 + alpha/n_ref)^(-n_ref)

    Solving for alpha::

        alpha = n_ref * (Pfa^(-1/n_ref) - 1)

    For GO/SO-CFAR the exact expressions from Gandhi & Kassam (1988) are
    solved numerically. For OS-CFAR the exact Rohling (1983) relation::

        Pfa = prod_{i=0}^{k-1} (n_ref - i) / (n_ref - i + alpha)

    is solved numerically.
    """
    if pfa <= 0 or pfa >= 1:
        raise ValueError("pfa must be between 0 and 1")
    if n_ref < 1:
        raise ValueError("n_ref must be at least 1")

    if method == "ca":
        # CA-CFAR threshold factor (exact closed form)
        alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1)
    elif method == "go" or method == "so":
        # Exact GO/SO-CFAR threshold with n_ref // 2 cells per half window
        n_half = max(1, n_ref // 2)
        pfa_func = _pfa_go if method == "go" else _pfa_so
        alpha = _solve_alpha(lambda a: pfa_func(a, n_half), pfa)
    elif method == "os":
        if k is None:
            k = int(0.75 * n_ref)  # Default: 75th percentile
        k = max(1, min(int(k), n_ref))
        # Exact OS-CFAR threshold from Rohling (1983)
        alpha = _solve_alpha(lambda a: _pfa_os(a, n_ref, k), pfa)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(alpha)


def detection_probability(
    snr: float,
    pfa: float,
    n_ref: int,
    method: str = "ca",
) -> float:
    """
    Compute Swerling 1 detection probability for a given SNR and Pfa.

    This is the CA-CFAR result for an exponentially fluctuating (Swerling 1)
    target::

        Pd = (1 + alpha/(n_ref*(1+snr)))^(-n_ref)

    with ``alpha`` the threshold factor for the requested Pfa.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio (linear, not dB).
    pfa : float
        Probability of false alarm.
    n_ref : int
        Number of reference cells.
    method : {'ca'}, optional
        CFAR method. Default is 'ca'.

    Returns
    -------
    pd : float
        Probability of detection for a Swerling 1 target.

    Examples
    --------
    >>> pd = detection_probability(snr=10, pfa=1e-6, n_ref=32)
    >>> 0 < pd < 1
    True

    Notes
    -----
    Only the Swerling 1 model is implemented. This function used to accept a
    ``swerling_case`` argument covering cases 0 through 4, but all of its
    branches evaluated the same expression, so the argument selected nothing --
    a caller asking for a non-fluctuating target got the Swerling 1 answer
    (gh-20). The argument has been removed rather than left to imply a choice
    that was never offered.

    The difference is not small. At SNR 10, Pfa 1e-6 and 32 reference cells,
    this returns about 0.62, while a genuinely non-fluctuating (Swerling 0 /
    Marcum) target detects with probability about 0.90 -- the fluctuating model
    understates a steady target substantially.

    For a real choice of target model use ``swerling_detection_probability``,
    which implements cases 0 through 4 as genuinely distinct expressions built
    on the Marcum Q function. It is a pulse-integration model rather than a
    CFAR one, so it answers a slightly different question: this function
    accounts for threshold estimation from ``n_ref`` reference cells, that one
    for coherent integration over ``n_pulses``.

    See Also
    --------
    pytcl.mathematical_functions.special_functions.swerling_detection_probability :
        Detection probability for Swerling cases 0-4.
    """
    alpha = threshold_factor(pfa, n_ref, method=method)
    pd = (1 + alpha / (n_ref * (1 + snr))) ** (-n_ref)
    return float(pd)


# =============================================================================
# JIT-Compiled Kernels
# =============================================================================


@njit(cache=True, fastmath=True)
def _cfar_ca_kernel(
    signal: np.ndarray[Any, Any],
    guard_cells: int,
    ref_cells: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled CA-CFAR kernel."""
    n = len(signal)
    half_window = guard_cells + ref_cells

    for i in range(n):
        left_start = max(0, i - half_window)
        left_end = max(0, i - guard_cells)
        right_start = min(n, i + guard_cells + 1)
        right_end = min(n, i + half_window + 1)

        ref_sum = 0.0
        n_cells = 0

        for j in range(left_start, left_end):
            ref_sum += signal[j]
            n_cells += 1

        for j in range(right_start, right_end):
            ref_sum += signal[j]
            n_cells += 1

        if n_cells > 0:
            noise_estimate[i] = ref_sum / n_cells
        else:
            noise_estimate[i] = 0.0

        threshold[i] = alpha * noise_estimate[i]


@njit(cache=True, fastmath=True)
def _cfar_go_kernel(
    signal: np.ndarray[Any, Any],
    guard_cells: int,
    ref_cells: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled GO-CFAR kernel."""
    n = len(signal)
    half_window = guard_cells + ref_cells

    for i in range(n):
        left_start = max(0, i - half_window)
        left_end = max(0, i - guard_cells)
        right_start = min(n, i + guard_cells + 1)
        right_end = min(n, i + half_window + 1)

        left_sum = 0.0
        left_count = 0
        for j in range(left_start, left_end):
            left_sum += signal[j]
            left_count += 1

        right_sum = 0.0
        right_count = 0
        for j in range(right_start, right_end):
            right_sum += signal[j]
            right_count += 1

        left_avg = left_sum / left_count if left_count > 0 else 0.0
        right_avg = right_sum / right_count if right_count > 0 else 0.0

        noise_estimate[i] = max(left_avg, right_avg)
        threshold[i] = alpha * noise_estimate[i]


@njit(cache=True, fastmath=True)
def _cfar_so_kernel(
    signal: np.ndarray[Any, Any],
    guard_cells: int,
    ref_cells: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled SO-CFAR kernel."""
    n = len(signal)
    half_window = guard_cells + ref_cells

    for i in range(n):
        left_start = max(0, i - half_window)
        left_end = max(0, i - guard_cells)
        right_start = min(n, i + guard_cells + 1)
        right_end = min(n, i + half_window + 1)

        left_sum = 0.0
        left_count = 0
        for j in range(left_start, left_end):
            left_sum += signal[j]
            left_count += 1

        right_sum = 0.0
        right_count = 0
        for j in range(right_start, right_end):
            right_sum += signal[j]
            right_count += 1

        left_avg = left_sum / left_count if left_count > 0 else np.inf
        right_avg = right_sum / right_count if right_count > 0 else np.inf

        noise_est = min(left_avg, right_avg)
        if noise_est == np.inf:
            noise_est = 0.0
        noise_estimate[i] = noise_est
        threshold[i] = alpha * noise_estimate[i]


@njit(cache=True, fastmath=True)
def _cfar_os_kernel(
    signal: np.ndarray[Any, Any],
    guard_cells: int,
    ref_cells: int,
    k: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled OS-CFAR kernel."""
    n = len(signal)
    half_window = guard_cells + ref_cells
    max_ref = 2 * ref_cells + 4  # Buffer for edge cells

    for i in range(n):
        left_start = max(0, i - half_window)
        left_end = max(0, i - guard_cells)
        right_start = min(n, i + guard_cells + 1)
        right_end = min(n, i + half_window + 1)

        # Collect reference cells into temporary array
        ref_buffer = np.empty(max_ref, dtype=np.float64)
        n_cells = 0

        for j in range(left_start, left_end):
            if n_cells < max_ref:
                ref_buffer[n_cells] = signal[j]
                n_cells += 1

        for j in range(right_start, right_end):
            if n_cells < max_ref:
                ref_buffer[n_cells] = signal[j]
                n_cells += 1

        if n_cells > 0:
            # Sort the reference cells
            ref_values = ref_buffer[:n_cells]
            ref_values.sort()
            k_index = min(k - 1, n_cells - 1)
            noise_estimate[i] = ref_values[k_index]
        else:
            noise_estimate[i] = 0.0

        threshold[i] = alpha * noise_estimate[i]


@njit(cache=True, fastmath=True, parallel=True)
def _cfar_2d_ca_kernel(
    image: np.ndarray[Any, Any],
    guard_rows: int,
    guard_cols: int,
    ref_rows: int,
    ref_cols: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled 2D CA-CFAR kernel with parallel execution."""
    n_rows, n_cols = image.shape
    half_row = guard_rows + ref_rows
    half_col = guard_cols + ref_cols

    for i in prange(n_rows):
        for j in range(n_cols):
            row_min = max(0, i - half_row)
            row_max = min(n_rows, i + half_row + 1)
            col_min = max(0, j - half_col)
            col_max = min(n_cols, j + half_col + 1)

            guard_row_min = max(0, i - guard_rows)
            guard_row_max = min(n_rows, i + guard_rows + 1)
            guard_col_min = max(0, j - guard_cols)
            guard_col_max = min(n_cols, j + guard_cols + 1)

            ref_sum = 0.0
            n_cells = 0

            for ri in range(row_min, row_max):
                for ci in range(col_min, col_max):
                    if not (
                        guard_row_min <= ri < guard_row_max
                        and guard_col_min <= ci < guard_col_max
                    ):
                        ref_sum += image[ri, ci]
                        n_cells += 1

            if n_cells > 0:
                noise_estimate[i, j] = ref_sum / n_cells
            else:
                noise_estimate[i, j] = 0.0

            threshold[i, j] = alpha * noise_estimate[i, j]


@njit(cache=True, fastmath=True, parallel=True)
def _cfar_2d_go_kernel(
    image: np.ndarray[Any, Any],
    guard_rows: int,
    guard_cols: int,
    ref_rows: int,
    ref_cols: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled 2D GO-CFAR kernel with parallel execution."""
    n_rows, n_cols = image.shape
    half_row = guard_rows + ref_rows
    half_col = guard_cols + ref_cols

    for i in prange(n_rows):
        for j in range(n_cols):
            row_min = max(0, i - half_row)
            row_max = min(n_rows, i + half_row + 1)
            col_min = max(0, j - half_col)
            col_max = min(n_cols, j + half_col + 1)

            guard_row_min = max(0, i - guard_rows)
            guard_row_max = min(n_rows, i + guard_rows + 1)
            guard_col_min = max(0, j - guard_cols)
            guard_col_max = min(n_cols, j + guard_cols + 1)

            top_sum = 0.0
            top_count = 0
            bottom_sum = 0.0
            bottom_count = 0

            for ri in range(row_min, row_max):
                for ci in range(col_min, col_max):
                    if not (
                        guard_row_min <= ri < guard_row_max
                        and guard_col_min <= ci < guard_col_max
                    ):
                        if ri < i:
                            top_sum += image[ri, ci]
                            top_count += 1
                        else:
                            bottom_sum += image[ri, ci]
                            bottom_count += 1

            top_avg = top_sum / top_count if top_count > 0 else 0.0
            bottom_avg = bottom_sum / bottom_count if bottom_count > 0 else 0.0
            noise_estimate[i, j] = max(top_avg, bottom_avg)
            threshold[i, j] = alpha * noise_estimate[i, j]


@njit(cache=True, fastmath=True, parallel=True)
def _cfar_2d_so_kernel(
    image: np.ndarray[Any, Any],
    guard_rows: int,
    guard_cols: int,
    ref_rows: int,
    ref_cols: int,
    alpha: float,
    noise_estimate: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled 2D SO-CFAR kernel with parallel execution."""
    n_rows, n_cols = image.shape
    half_row = guard_rows + ref_rows
    half_col = guard_cols + ref_cols

    for i in prange(n_rows):
        for j in range(n_cols):
            row_min = max(0, i - half_row)
            row_max = min(n_rows, i + half_row + 1)
            col_min = max(0, j - half_col)
            col_max = min(n_cols, j + half_col + 1)

            guard_row_min = max(0, i - guard_rows)
            guard_row_max = min(n_rows, i + guard_rows + 1)
            guard_col_min = max(0, j - guard_cols)
            guard_col_max = min(n_cols, j + guard_cols + 1)

            top_sum = 0.0
            top_count = 0
            bottom_sum = 0.0
            bottom_count = 0

            for ri in range(row_min, row_max):
                for ci in range(col_min, col_max):
                    if not (
                        guard_row_min <= ri < guard_row_max
                        and guard_col_min <= ci < guard_col_max
                    ):
                        if ri < i:
                            top_sum += image[ri, ci]
                            top_count += 1
                        else:
                            bottom_sum += image[ri, ci]
                            bottom_count += 1

            top_avg = top_sum / top_count if top_count > 0 else np.inf
            bottom_avg = bottom_sum / bottom_count if bottom_count > 0 else np.inf
            noise_est = min(top_avg, bottom_avg)
            if noise_est == np.inf:
                noise_est = 0.0
            noise_estimate[i, j] = noise_est
            threshold[i, j] = alpha * noise_estimate[i, j]


# =============================================================================
# 1D CFAR Algorithms
# =============================================================================


def cfar_ca(
    signal: ArrayLike,
    guard_cells: int,
    ref_cells: int,
    pfa: float = 1e-6,
    alpha: Optional[float] = None,
) -> CFARResult:
    """
    Cell-Averaging CFAR detector.

    CA-CFAR estimates the noise level by averaging the cells in the reference
    window (excluding guard cells around the cell under test).

    Parameters
    ----------
    signal : array_like
        Input signal (typically power or magnitude).
    guard_cells : int
        Number of guard cells on each side of the cell under test.
    ref_cells : int
        Number of reference cells on each side.
    pfa : float, optional
        Probability of false alarm. Default is 1e-6.
    alpha : float, optional
        Threshold multiplier. If provided, overrides pfa calculation.

    Returns
    -------
    result : CFARResult
        Named tuple with detections, threshold, indices, and noise estimate.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Noise with a few targets
    >>> signal = np.random.exponential(1.0, 1000)
    >>> signal[250] = 50  # Target 1
    >>> signal[500] = 100  # Target 2
    >>> signal[750] = 30  # Target 3
    >>> result = cfar_ca(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
    >>> 250 in result.detection_indices
    True

    Notes
    -----
    The CA-CFAR is optimal for homogeneous noise (noise power constant
    across all cells). It suffers in heterogeneous environments and near
    closely-spaced targets.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if alpha is None:
        alpha = threshold_factor(pfa, 2 * ref_cells, method="ca")

    noise_estimate = np.zeros(n, dtype=np.float64)
    threshold = np.zeros(n, dtype=np.float64)

    # Use JIT-compiled kernel for performance
    _cfar_ca_kernel(signal, guard_cells, ref_cells, alpha, noise_estimate, threshold)

    detections = signal > threshold
    detection_indices = np.where(detections)[0]

    return CFARResult(
        detections=detections,
        threshold=threshold,
        detection_indices=detection_indices,
        noise_estimate=noise_estimate,
    )


def cfar_go(
    signal: ArrayLike,
    guard_cells: int,
    ref_cells: int,
    pfa: float = 1e-6,
    alpha: Optional[float] = None,
) -> CFARResult:
    """
    Greatest-Of CFAR detector.

    GO-CFAR takes the maximum of the leading and lagging reference window
    averages. This provides better performance at clutter edges but
    increased loss against distributed targets.

    Parameters
    ----------
    signal : array_like
        Input signal.
    guard_cells : int
        Number of guard cells on each side.
    ref_cells : int
        Number of reference cells on each side.
    pfa : float, optional
        Probability of false alarm. Default is 1e-6.
    alpha : float, optional
        Threshold multiplier.

    Returns
    -------
    result : CFARResult
        Named tuple with detection results.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.random.exponential(1.0, 500)
    >>> signal[250] = 50
    >>> result = cfar_go(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
    >>> len(result.detection_indices) >= 1
    True

    Notes
    -----
    GO-CFAR reduces false alarms at clutter edges (where noise level
    changes abruptly) compared to CA-CFAR, at the cost of slightly
    reduced detection probability in homogeneous noise.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if alpha is None:
        alpha = threshold_factor(pfa, 2 * ref_cells, method="go")

    noise_estimate = np.zeros(n, dtype=np.float64)
    threshold = np.zeros(n, dtype=np.float64)

    # Use JIT-compiled kernel for performance
    _cfar_go_kernel(signal, guard_cells, ref_cells, alpha, noise_estimate, threshold)

    detections = signal > threshold
    detection_indices = np.where(detections)[0]

    return CFARResult(
        detections=detections,
        threshold=threshold,
        detection_indices=detection_indices,
        noise_estimate=noise_estimate,
    )


def cfar_so(
    signal: ArrayLike,
    guard_cells: int,
    ref_cells: int,
    pfa: float = 1e-6,
    alpha: Optional[float] = None,
) -> CFARResult:
    """
    Smallest-Of CFAR detector.

    SO-CFAR takes the minimum of the leading and lagging reference window
    averages. This provides better detection near clutter edges but
    increased false alarms in some scenarios.

    Parameters
    ----------
    signal : array_like
        Input signal.
    guard_cells : int
        Number of guard cells on each side.
    ref_cells : int
        Number of reference cells on each side.
    pfa : float, optional
        Probability of false alarm. Default is 1e-6.
    alpha : float, optional
        Threshold multiplier.

    Returns
    -------
    result : CFARResult
        Named tuple with detection results.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.mathematical_functions.signal_processing import cfar_so
    >>> # Create test signal with closely spaced targets in clutter
    >>> np.random.seed(42)
    >>> signal = np.random.exponential(1.0, 500)
    >>> signal[200:205] = [30, 40, 35, 45, 38]  # Target cluster
    >>> signal[350] = 50  # Isolated target
    >>> # Detect using SO-CFAR
    >>> result = cfar_so(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
    >>> # SO-CFAR good for clutter edge detection
    >>> len(result.detection_indices) >= 2  # Should find multiple targets
    True

    Notes
    -----
    SO-CFAR is complementary to GO-CFAR. It is more sensitive near
    clutter edges but may produce more false alarms when interfering
    targets are present in the reference window.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if alpha is None:
        alpha = threshold_factor(pfa, 2 * ref_cells, method="so")

    noise_estimate = np.zeros(n, dtype=np.float64)
    threshold = np.zeros(n, dtype=np.float64)

    # Use JIT-compiled kernel for performance
    _cfar_so_kernel(signal, guard_cells, ref_cells, alpha, noise_estimate, threshold)

    detections = signal > threshold
    detection_indices = np.where(detections)[0]

    return CFARResult(
        detections=detections,
        threshold=threshold,
        detection_indices=detection_indices,
        noise_estimate=noise_estimate,
    )


def cfar_os(
    signal: ArrayLike,
    guard_cells: int,
    ref_cells: int,
    pfa: float = 1e-6,
    k: Optional[int] = None,
    alpha: Optional[float] = None,
) -> CFARResult:
    """
    Order-Statistic CFAR detector.

    OS-CFAR uses an order statistic (k-th smallest value) of the reference
    cells instead of the mean. This makes it robust to interfering targets
    in the reference window.

    Parameters
    ----------
    signal : array_like
        Input signal.
    guard_cells : int
        Number of guard cells on each side.
    ref_cells : int
        Number of reference cells on each side.
    pfa : float, optional
        Probability of false alarm. Default is 1e-6.
    k : int, optional
        Order statistic to use (1 = minimum, n_ref = maximum).
        Default is 0.75 * n_ref.
    alpha : float, optional
        Threshold multiplier.

    Returns
    -------
    result : CFARResult
        Named tuple with detection results.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> signal = np.random.exponential(1.0, 500)
    >>> signal[250] = 50
    >>> signal[260] = 40  # Closely spaced target
    >>> result = cfar_os(signal, guard_cells=2, ref_cells=16, pfa=1e-4)
    >>> len(result.detection_indices) >= 2
    True

    Notes
    -----
    OS-CFAR is robust to interfering targets in the reference window
    because the order statistic ignores outliers. The choice of k trades
    off between:
    - Low k: Robust to multiple interferers, but sensitive to noise
    - High k: Less robust to interferers, but better in homogeneous noise
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    n_total_ref = 2 * ref_cells

    if k is None:
        k = int(0.75 * n_total_ref)
    k = max(1, min(k, n_total_ref))

    if alpha is None:
        alpha = threshold_factor(pfa, n_total_ref, method="os", k=k)

    noise_estimate = np.zeros(n, dtype=np.float64)
    threshold = np.zeros(n, dtype=np.float64)

    # Use JIT-compiled kernel for performance
    _cfar_os_kernel(signal, guard_cells, ref_cells, k, alpha, noise_estimate, threshold)

    detections = signal > threshold
    detection_indices = np.where(detections)[0]

    return CFARResult(
        detections=detections,
        threshold=threshold,
        detection_indices=detection_indices,
        noise_estimate=noise_estimate,
    )


# =============================================================================
# 2D CFAR
# =============================================================================


def cfar_2d(
    image: ArrayLike,
    guard_cells: tuple[int, int],
    ref_cells: tuple[int, int],
    pfa: float = 1e-6,
    method: str = "ca",
    alpha: Optional[float] = None,
) -> CFARResult2D:
    """
    Two-dimensional CFAR detector.

    2D CFAR is used for range-Doppler maps or image detection where the
    reference window extends in both dimensions.

    Parameters
    ----------
    image : array_like
        2D input (e.g., range-Doppler map).
    guard_cells : tuple
        (guard_rows, guard_cols) - guard cells in each direction.
    ref_cells : tuple
        (ref_rows, ref_cols) - reference cells in each direction.
    pfa : float, optional
        Probability of false alarm. Default is 1e-6.
    method : {'ca', 'go', 'so'}, optional
        CFAR method. Default is 'ca'.
    alpha : float, optional
        Threshold multiplier.

    Returns
    -------
    result : CFARResult2D
        Named tuple with 2D detections, threshold, and noise estimate.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> image = np.random.exponential(1.0, (100, 100))
    >>> image[50, 50] = 100  # Target
    >>> result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(8, 8), pfa=1e-4)
    >>> result.detections[50, 50]
    True

    Notes
    -----
    The 2D reference window forms a rectangular annulus around the cell
    under test. The total number of reference cells is::

        (2*guard_rows + 2*ref_rows + 1) * (2*guard_cols + 2*ref_cols + 1)
        - (2*guard_rows + 1) * (2*guard_cols + 1)
    """
    image = np.asarray(image, dtype=np.float64)

    guard_rows, guard_cols = guard_cells
    ref_rows, ref_cols = ref_cells

    # Count reference cells
    outer_rows = 2 * (guard_rows + ref_rows) + 1
    outer_cols = 2 * (guard_cols + ref_cols) + 1
    inner_rows = 2 * guard_rows + 1
    inner_cols = 2 * guard_cols + 1
    n_ref = outer_rows * outer_cols - inner_rows * inner_cols

    if alpha is None:
        alpha = threshold_factor(pfa, n_ref, method=method)

    noise_estimate = np.zeros_like(image)
    threshold = np.zeros_like(image)

    # Use JIT-compiled kernel for performance (with parallel execution)
    if method == "ca":
        _cfar_2d_ca_kernel(
            image,
            guard_rows,
            guard_cols,
            ref_rows,
            ref_cols,
            alpha,
            noise_estimate,
            threshold,
        )
    elif method == "go":
        _cfar_2d_go_kernel(
            image,
            guard_rows,
            guard_cols,
            ref_rows,
            ref_cols,
            alpha,
            noise_estimate,
            threshold,
        )
    elif method == "so":
        _cfar_2d_so_kernel(
            image,
            guard_rows,
            guard_cols,
            ref_rows,
            ref_cols,
            alpha,
            noise_estimate,
            threshold,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    detections = image > threshold

    return CFARResult2D(
        detections=detections,
        threshold=threshold,
        noise_estimate=noise_estimate,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def cluster_detections(
    detections: ArrayLike,
    min_separation: int = 1,
) -> NDArray[np.intp]:
    """
    Cluster nearby detections and return peak indices.

    Parameters
    ----------
    detections : array_like
        Boolean detection array or signal values at detection points.
    min_separation : int, optional
        Minimum separation between distinct detections. Default is 1.

    Returns
    -------
    peak_indices : ndarray
        Indices of detection peaks after clustering.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.mathematical_functions.signal_processing import cluster_detections
    >>> # CFAR detection result with closely spaced detections
    >>> detections = np.zeros(100, dtype=bool)
    >>> detections[20:24] = True  # Cluster 1 (4 adjacent detections)
    >>> detections[60] = True      # Cluster 2 (single detection)
    >>> detections[62] = True      # Close to cluster 2
    >>> # Gaps <= min_separation merge into the same cluster
    >>> peaks = cluster_detections(detections, min_separation=2)
    >>> len(peaks)  # Should find 2 clusters
    2
    >>> int(peaks[0])  # Center of first cluster (indices 20-23)
    21
    """
    detections = np.asarray(detections)

    if detections.dtype == bool:
        det_indices = np.where(detections)[0]
    else:
        det_indices = np.arange(len(detections))

    if len(det_indices) == 0:
        return np.array([], dtype=np.intp)

    # Cluster nearby indices
    clusters = []
    current_cluster = [det_indices[0]]

    for i in range(1, len(det_indices)):
        if det_indices[i] - det_indices[i - 1] <= min_separation:
            current_cluster.append(det_indices[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [det_indices[i]]

    clusters.append(current_cluster)

    # Take center of each cluster
    peak_indices = [int(np.mean(cluster)) for cluster in clusters]

    return np.array(peak_indices, dtype=np.intp)


def snr_loss(
    n_ref: int,
    pfa: float,
    pd: float = 0.5,
    method: str = "ca",
) -> float:
    """
    CA-CFAR loss: the extra SNR needed because the threshold is estimated.

    An ideal detector knows the noise power. A CFAR detector estimates it from
    ``n_ref`` reference cells, and needs more signal to reach the same
    detection probability at the same false-alarm rate. The loss is the
    difference between the two required SNRs, for a Swerling 1 target::

        S_ideal = ln(Pfa)/ln(Pd) - 1
        S_ca    = (Pfa^(-1/n) - 1) / (Pd^(-1/n) - 1) - 1
        L_dB    = 10 log10( (1 + S_ca) / (1 + S_ideal) )

    Parameters
    ----------
    n_ref : int
        Number of reference cells.
    pfa : float
        Design probability of false alarm.
    pd : float, optional
        Detection probability at which the loss is evaluated. Default 0.5.
        CFAR loss is only defined against an operating point.
    method : {'ca'}, optional
        CFAR method. Only ``'ca'`` is implemented. Default is 'ca'.

    Returns
    -------
    loss : float
        SNR loss in dB, non-negative and decreasing in ``n_ref``.

    Raises
    ------
    NotImplementedError
        For ``'go'``, ``'so'`` and ``'os'``. See the notes.

    Examples
    --------
    >>> round(snr_loss(16, pfa=1e-6), 3)
    1.915
    >>> snr_loss(8, pfa=1e-6) > snr_loss(64, pfa=1e-6)  # fewer cells, more loss
    True

    Notes
    -----
    This used to be computed from ad-hoc expressions -- ``1 + 1/n_ref`` for CA,
    ``1 + 2/n_ref`` for GO and SO, ``1 + 3/n_ref`` for OS -- taking neither
    ``pfa`` nor ``pd`` (gh-20). CFAR loss depends on both, so a function of
    ``n_ref`` alone cannot express it. The heuristics understated it by roughly
    a factor of four: at 8 reference cells and Pfa 1e-6 they returned 0.51 dB
    against a true 4.09 dB, and returned that same figure for every operating
    point.

    Only CA is implemented, because the loss is defined through the detection
    probability and this library has a closed form for Pd under CA-CFAR alone.
    GO, SO and OS raise rather than return a number: a plausible wrong loss is
    worse than an absent one, and the earlier heuristics for those three were
    not derived from anything.

    Comparing threshold factors directly would be the obvious shortcut and does
    not work. ``threshold_factor`` returns a multiplier on each method's own
    noise statistic -- the mean for CA, the larger or smaller half-window mean
    for GO and SO, an order statistic for OS -- and those statistics have
    different expectations, so the multipliers are not comparable. Doing it
    that way gives OS-CFAR a *negative* loss at 64 reference cells.

    See Also
    --------
    detection_probability : The Swerling 1 CA-CFAR Pd this is derived from.
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError(f"pfa must be in (0, 1), got {pfa}")
    if not 0.0 < pd < 1.0:
        raise ValueError(f"pd must be in (0, 1), got {pd}")
    if n_ref < 1:
        raise ValueError(f"n_ref must be at least 1, got {n_ref}")
    if method != "ca":
        raise NotImplementedError(
            f"snr_loss is only implemented for method='ca', got {method!r}. "
            f"The loss is defined through the detection probability, and no "
            f"closed-form Pd is available here for GO, SO or OS-CFAR."
        )

    snr_ideal = np.log(pfa) / np.log(pd) - 1.0
    snr_cfar = (pfa ** (-1.0 / n_ref) - 1.0) / (pd ** (-1.0 / n_ref) - 1.0) - 1.0

    return float(10 * np.log10((1.0 + snr_cfar) / (1.0 + snr_ideal)))


__all__ = [
    # Result Types
    "CFARResult",
    "CFARResult2D",
    # Threshold and Detection Probability
    "threshold_factor",
    "detection_probability",
    # CFAR Detectors
    "cfar_ca",
    "cfar_go",
    "cfar_so",
    "cfar_os",
    "cfar_2d",
    # Utilities
    "cluster_detections",
    "snr_loss",
]
