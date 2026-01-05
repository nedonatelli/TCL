"""
Marcum Q function and related functions.

The Marcum Q function is crucial in radar and communications for
analyzing detection probabilities and signal statistics.
"""

import numpy as np
import scipy.special as sp
from numpy.typing import ArrayLike, NDArray


def marcum_q(
    a: ArrayLike,
    b: ArrayLike,
    m: int = 1,
) -> NDArray[np.floating]:
    """
    Generalized Marcum Q function Q_m(a, b).

    The Marcum Q function is the complementary cumulative distribution
    function of the noncentral chi-squared distribution and appears
    in radar detection theory.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.
    m : int, optional
        Order of the Marcum Q function (positive integer). Default is 1.

    Returns
    -------
    Q : ndarray
        Values of Q_m(a, b).

    Notes
    -----
    For m = 1, this is the standard Marcum Q function:
    Q_1(a, b) = integral from b to inf of x * exp(-(x^2 + a^2)/2) * I_0(a*x) dx

    The function is related to the noncentral chi-squared distribution:
    Q_m(a, b) = P(X > b^2) where X ~ chi^2(2m, a^2)

    Special cases:
    - Q_m(0, b) = 1 - gammainc(m, b^2/2) = gammaincc(m, b^2/2)
    - Q_m(a, 0) = 1

    Examples
    --------
    >>> marcum_q(0, 0)  # Q_1(0, 0) = 1
    1.0
    >>> marcum_q(3, 4)  # Standard Marcum Q
    0.17789...

    References
    ----------
    .. [1] Marcum, J.I. (1950). "Table of Q Functions".
    .. [2] Shnidman, D.A. (1989). "The Calculation of the Probability of
           Detection and the Generalized Marcum Q-Function". IEEE Trans.
           on Information Theory, 35(2), 389-400.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")

    # Handle edge cases
    result = np.ones_like(a * b, dtype=np.float64)

    # Where b == 0, Q_m(a, 0) = 1
    b_zero = b == 0

    # Where a == 0, use incomplete gamma function
    a_zero = (a == 0) & (~b_zero)
    if np.any(a_zero):
        result[a_zero] = sp.gammaincc(m, 0.5 * b[a_zero] ** 2)

    # General case: use ncx2 survival function
    # Q_m(a, b) = P(X > b^2) where X ~ chi^2(2m, a^2)
    general = (~a_zero) & (~b_zero)
    if np.any(general):
        from scipy.stats import ncx2

        # Degrees of freedom = 2m, non-centrality = a^2
        result[general] = ncx2.sf(b[general] ** 2, 2 * m, a[general] ** 2)

    return result


def marcum_q1(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """
    Standard Marcum Q function Q_1(a, b).

    Convenience function for the first-order Marcum Q function.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.

    Returns
    -------
    Q : ndarray
        Values of Q_1(a, b).

    Examples
    --------
    >>> marcum_q1(2, 2)
    0.735...

    See Also
    --------
    marcum_q : Generalized Marcum Q function.
    """
    return marcum_q(a, b, m=1)


def log_marcum_q(
    a: ArrayLike,
    b: ArrayLike,
    m: int = 1,
) -> NDArray[np.floating]:
    """
    Natural logarithm of the Marcum Q function.

    Computes log(Q_m(a, b)) with better numerical precision for small
    values of Q.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.
    m : int, optional
        Order of the Marcum Q function. Default is 1.

    Returns
    -------
    log_Q : ndarray
        Values of log(Q_m(a, b)).

    Notes
    -----
    For small Q values (large b), this function provides better
    numerical accuracy than computing log(marcum_q(a, b)).

    Examples
    --------
    >>> log_marcum_q(1, 5)  # log(Q_1(1, 5))
    -10.96...
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")

    # For moderate Q values, compute directly
    q_val = marcum_q(a, b, m)

    # Handle edge cases
    result = np.log(q_val)

    # For very small Q values, use log survival function
    small_q = q_val < 1e-10
    if np.any(small_q):
        from scipy.stats import ncx2

        # Use logsf for better precision
        a_small = np.atleast_1d(a)[small_q] if np.atleast_1d(a).size > 1 else a
        b_small = np.atleast_1d(b)[small_q] if np.atleast_1d(b).size > 1 else b

        if np.isscalar(a) and not np.isscalar(b):
            a_small = np.full_like(b_small, a)
        if np.isscalar(b) and not np.isscalar(a):
            b_small = np.full_like(a_small, b)

        result_arr = np.atleast_1d(result)
        result_arr[small_q] = ncx2.logsf(b_small**2, 2 * m, a_small**2)
        result = result_arr[0] if result.ndim == 0 else result_arr

    return result


def marcum_q_inv(
    a: ArrayLike,
    q: ArrayLike,
    m: int = 1,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> NDArray[np.floating]:
    """
    Inverse Marcum Q function.

    Finds b such that Q_m(a, b) = q.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    q : array_like
        Target probability value, 0 < q < 1.
    m : int, optional
        Order of the Marcum Q function. Default is 1.
    tol : float, optional
        Tolerance for convergence. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    b : ndarray
        Values such that Q_m(a, b) = q.

    Notes
    -----
    Uses Newton-Raphson iteration with the noncentral chi-squared
    distribution.

    Examples
    --------
    >>> b = marcum_q_inv(3, 0.5)  # Find b where Q_1(3, b) = 0.5
    >>> marcum_q(3, b)  # Verify
    0.5
    """
    a = np.asarray(a, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("q must be in (0, 1)")

    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")

    from scipy.stats import ncx2

    # Q_m(a, b) = ncx2.sf(b^2, 2m, a^2)
    # So we need b^2 = ncx2.isf(q, 2m, a^2)
    # b = sqrt(ncx2.isf(q, 2m, a^2))

    b_squared = ncx2.isf(q, 2 * m, a**2)
    b = np.sqrt(np.maximum(b_squared, 0))

    return b


def nuttall_q(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """
    Nuttall Q function (complementary Marcum Q).

    Computes 1 - Q_1(a, b), which is the CDF of the Rician distribution.

    Parameters
    ----------
    a : array_like
        First argument (non-centrality parameter), a >= 0.
    b : array_like
        Second argument (threshold), b >= 0.

    Returns
    -------
    P : ndarray
        Values of 1 - Q_1(a, b).

    Notes
    -----
    This is the probability P(X <= b^2) for X ~ chi^2(2, a^2).

    Examples
    --------
    >>> nuttall_q(2, 2)  # 1 - Q_1(2, 2)
    0.264...

    See Also
    --------
    marcum_q : Marcum Q function.
    """
    return 1.0 - marcum_q(a, b, m=1)


def swerling_detection_probability(
    snr: ArrayLike,
    pfa: float,
    n_pulses: int = 1,
    swerling_case: int = 1,
) -> NDArray[np.floating]:
    """
    Detection probability for Swerling target models.

    Computes probability of detection for different Swerling cases
    using the Marcum Q function.

    Parameters
    ----------
    snr : array_like
        Signal-to-noise ratio (linear, not dB).
    pfa : float
        Probability of false alarm (0 < pfa < 1).
    n_pulses : int, optional
        Number of integrated pulses. Default is 1.
    swerling_case : int, optional
        Swerling case (0, 1, 2, 3, or 4). Default is 1.
        - 0: Non-fluctuating (Marcum)
        - 1: Slow fluctuation, Rayleigh PDF
        - 2: Fast fluctuation, Rayleigh PDF
        - 3: Slow fluctuation, one dominant + Rayleigh
        - 4: Fast fluctuation, one dominant + Rayleigh

    Returns
    -------
    Pd : ndarray
        Probability of detection.

    Notes
    -----
    For Swerling 0 (non-fluctuating):
        P_d = Q_n(sqrt(2*n*SNR), sqrt(threshold))

    For Swerling 1:
        P_d = exp(-threshold / (2 + 2*n*SNR)) * (1 + n*SNR/...)

    Examples
    --------
    >>> pd = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=0)
    >>> pd > 0.9  # High probability of detection with 10 dB SNR
    True

    References
    ----------
    .. [1] Swerling, P. (1960). "Probability of Detection for Fluctuating
           Targets". IRE Trans. on Information Theory, IT-6, 269-308.
    """
    snr = np.asarray(snr, dtype=np.float64)

    # Detection threshold from false alarm probability
    # For chi-squared with 2*n_pulses DOF: P(X > T) = pfa
    threshold = -2 * n_pulses * np.log(pfa)

    if swerling_case == 0:
        # Non-fluctuating (Marcum case)
        a = np.sqrt(2 * n_pulses * snr)
        b = np.sqrt(threshold)
        return marcum_q(a, b, m=n_pulses)

    elif swerling_case == 1:
        # Slow Rayleigh fluctuation
        avg_snr = snr
        return np.exp(-threshold / (2 * (1 + n_pulses * avg_snr)))

    elif swerling_case == 2:
        # Fast Rayleigh fluctuation (chi-squared fading)
        avg_snr = snr
        gamma_factor = 1 / (1 + n_pulses * avg_snr)
        return sp.gammaincc(n_pulses, threshold * gamma_factor / 2)

    elif swerling_case in (3, 4):
        # Chi-squared with 4 DOF (one dominant + Rayleigh)
        avg_snr = snr
        x = threshold / (1 + 0.5 * n_pulses * avg_snr)
        return (1 + x / 2) * np.exp(-x / 2)

    else:
        raise ValueError(f"swerling_case must be 0-4, got {swerling_case}")


__all__ = [
    "marcum_q",
    "marcum_q1",
    "log_marcum_q",
    "marcum_q_inv",
    "nuttall_q",
    "swerling_detection_probability",
]
