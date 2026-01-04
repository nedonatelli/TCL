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
.. [1] Richards, M. A. (2014). Fundamentals of Radar Signal Processing
       (2nd ed.). McGraw-Hill.
.. [2] Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple
       target situations. IEEE Transactions on Aerospace and Electronic
       Systems, 19(4), 608-621.
"""

from typing import Any, NamedTuple, Optional

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

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
        Number of reference cells.
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
    threshold factor alpha and Pfa is:
        Pfa = (1 + alpha/n_ref)^(-n_ref)

    Solving for alpha:
        alpha = n_ref * (Pfa^(-1/n_ref) - 1)
    """
    if pfa <= 0 or pfa >= 1:
        raise ValueError("pfa must be between 0 and 1")
    if n_ref < 1:
        raise ValueError("n_ref must be at least 1")

    if method == "ca":
        # CA-CFAR threshold factor
        alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1)
    elif method == "go" or method == "so":
        # For GO/SO CFAR, use CA formula with half the cells as approximation
        n_half = n_ref // 2
        if n_half < 1:
            n_half = 1
        alpha = n_half * (pfa ** (-1.0 / n_half) - 1)
    elif method == "os":
        if k is None:
            k = int(0.75 * n_ref)  # Default: 75th percentile
        # OS-CFAR uses order statistics
        # Approximate formula based on Rohling (1983)
        alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1)
        # Adjustment factor for order statistic
        alpha = alpha * (n_ref - k + 1) / n_ref
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(alpha)


def detection_probability(
    snr: float,
    pfa: float,
    n_ref: int,
    method: str = "ca",
    swerling_case: int = 0,
) -> float:
    """
    Compute detection probability for a given SNR and Pfa.

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
    swerling_case : {0, 1, 2, 3, 4}, optional
        Swerling target model. 0 is non-fluctuating (Marcum).
        Default is 0.

    Returns
    -------
    pd : float
        Probability of detection.

    Examples
    --------
    >>> pd = detection_probability(snr=10, pfa=1e-6, n_ref=32)
    >>> 0 < pd < 1
    True

    Notes
    -----
    For a non-fluctuating target (Swerling 0/Marcum case) with CA-CFAR:
        Pd = (1 + alpha/(n_ref*(1+snr)))^(-n_ref)

    where alpha is the threshold factor for the given Pfa.
    """
    alpha = threshold_factor(pfa, n_ref, method=method)

    if swerling_case == 0:
        # Non-fluctuating (Marcum) target
        # Pd for CA-CFAR
        pd = (1 + alpha / (n_ref * (1 + snr))) ** (-n_ref)
    elif swerling_case == 1:
        # Swerling I: scan-to-scan decorrelation, chi-squared 2 DOF
        pd = (1 + alpha / (n_ref * (1 + snr))) ** (-n_ref)
    else:
        # For other Swerling cases, use approximate formula
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
        alpha = threshold_factor(pfa, ref_cells, method="go")

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

    Notes
    -----
    SO-CFAR is complementary to GO-CFAR. It is more sensitive near
    clutter edges but may produce more false alarms when interfering
    targets are present in the reference window.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if alpha is None:
        alpha = threshold_factor(pfa, ref_cells, method="so")

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
    under test. The total number of reference cells is:
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
    method: str = "ca",
) -> float:
    """
    Compute the SNR loss due to CFAR processing.

    CFAR detectors have an inherent SNR loss compared to an ideal detector
    with known noise level.

    Parameters
    ----------
    n_ref : int
        Number of reference cells.
    method : {'ca', 'go', 'so', 'os'}, optional
        CFAR method. Default is 'ca'.

    Returns
    -------
    loss : float
        SNR loss in dB.

    Examples
    --------
    >>> loss = snr_loss(32, method='ca')
    >>> 0 < loss < 1  # Small loss for many reference cells
    True
    """
    if method == "ca":
        # CA-CFAR loss approximately 1/sqrt(n_ref) in linear terms
        # or 10*log10(1 + 1/n_ref) in dB for large n_ref
        loss_linear = 1 + 1 / n_ref
        loss_db = 10 * np.log10(loss_linear)
    elif method == "go":
        # GO-CFAR has slightly higher loss
        loss_linear = 1 + 2 / n_ref
        loss_db = 10 * np.log10(loss_linear)
    elif method == "so":
        # SO-CFAR similar to GO
        loss_linear = 1 + 2 / n_ref
        loss_db = 10 * np.log10(loss_linear)
    elif method == "os":
        # OS-CFAR typically has highest loss
        loss_linear = 1 + 3 / n_ref
        loss_db = 10 * np.log10(loss_linear)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(loss_db)


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
