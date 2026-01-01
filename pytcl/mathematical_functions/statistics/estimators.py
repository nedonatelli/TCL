"""
Statistical estimators and descriptive statistics.

This module provides functions for computing sample statistics,
robust estimators, and related quantities used in tracking applications.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def weighted_mean(
    x: ArrayLike,
    weights: ArrayLike,
    axis: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute weighted mean.

    Parameters
    ----------
    x : array_like
        Input data.
    weights : array_like
        Weights for each data point.
    axis : int, optional
        Axis along which to compute. Default is None (all elements).

    Returns
    -------
    mean : ndarray
        Weighted mean.

    Examples
    --------
    >>> weighted_mean([1, 2, 3], [1, 1, 2])
    2.25
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    return np.average(x, weights=weights, axis=axis)


def weighted_var(
    x: ArrayLike,
    weights: ArrayLike,
    ddof: int = 0,
    axis: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute weighted variance.

    Parameters
    ----------
    x : array_like
        Input data.
    weights : array_like
        Weights for each data point.
    ddof : int, optional
        Delta degrees of freedom. Default is 0 (population variance).
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    var : ndarray
        Weighted variance.
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    mean = np.average(x, weights=weights, axis=axis, keepdims=True)
    variance = np.average((x - mean) ** 2, weights=weights, axis=axis)

    if ddof != 0:
        # Reliability weights correction
        sum_weights = np.sum(weights, axis=axis)
        sum_weights_sq = np.sum(weights**2, axis=axis)
        correction = sum_weights / (sum_weights - ddof * sum_weights_sq / sum_weights)
        variance = variance * correction

    return variance


def weighted_cov(
    x: ArrayLike,
    weights: ArrayLike,
    ddof: int = 0,
) -> NDArray[np.floating]:
    """
    Compute weighted covariance matrix.

    Parameters
    ----------
    x : array_like
        Data matrix of shape (n_samples, n_features).
    weights : array_like
        Weights of shape (n_samples,).
    ddof : int, optional
        Delta degrees of freedom. Default is 0.

    Returns
    -------
    cov : ndarray
        Weighted covariance matrix of shape (n_features, n_features).
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Normalize weights
    weights = weights / np.sum(weights)

    # Compute weighted mean
    mean = np.sum(weights[:, np.newaxis] * x, axis=0)

    # Compute weighted covariance
    centered = x - mean
    cov = (weights[:, np.newaxis] * centered).T @ centered

    if ddof != 0:
        sum_weights_sq = np.sum(weights**2)
        cov = cov / (1 - ddof * sum_weights_sq)

    return cov


def sample_mean(
    x: ArrayLike,
    axis: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute sample mean.

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    mean : ndarray
        Sample mean.
    """
    return np.mean(x, axis=axis, dtype=np.float64)


def sample_var(
    x: ArrayLike,
    ddof: int = 1,
    axis: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute sample variance.

    Parameters
    ----------
    x : array_like
        Input data.
    ddof : int, optional
        Delta degrees of freedom. Default is 1 (unbiased estimator).
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    var : ndarray
        Sample variance.
    """
    return np.var(x, ddof=ddof, axis=axis, dtype=np.float64)


def sample_cov(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    ddof: int = 1,
) -> NDArray[np.floating]:
    """
    Compute sample covariance matrix.

    Parameters
    ----------
    x : array_like
        Data matrix of shape (n_samples, n_features) or 1D array.
    y : array_like, optional
        Second variable for cross-covariance.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    cov : ndarray
        Covariance matrix.
    """
    x = np.asarray(x, dtype=np.float64)

    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        return np.cov(x, y, ddof=ddof)

    if x.ndim == 1:
        return np.var(x, ddof=ddof)

    return np.cov(x.T, ddof=ddof)


def sample_corr(x: ArrayLike) -> NDArray[np.floating]:
    """
    Compute sample correlation matrix.

    Parameters
    ----------
    x : array_like
        Data matrix of shape (n_samples, n_features).

    Returns
    -------
    corr : ndarray
        Correlation matrix of shape (n_features, n_features).
    """
    return np.corrcoef(np.asarray(x, dtype=np.float64).T)


def median(
    x: ArrayLike,
    axis: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute median.

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    med : ndarray
        Median value(s).
    """
    return np.median(x, axis=axis)


def mad(
    x: ArrayLike,
    axis: Optional[int] = None,
    scale: float = 1.4826,
) -> NDArray[np.floating]:
    """
    Median Absolute Deviation (MAD).

    A robust measure of statistical dispersion.

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which to compute.
    scale : float, optional
        Scale factor for consistency with standard deviation for normal
        distributions. Default is 1.4826.

    Returns
    -------
    mad : ndarray
        MAD value(s).

    Notes
    -----
    For normally distributed data, scale * MAD approximates the
    standard deviation.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=axis, keepdims=True)
    return scale * np.median(np.abs(x - med), axis=axis)


def iqr(
    x: ArrayLike,
    axis: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Interquartile range (IQR).

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    iqr : ndarray
        Interquartile range (Q3 - Q1).
    """
    x = np.asarray(x, dtype=np.float64)
    q75, q25 = np.percentile(x, [75, 25], axis=axis)
    return q75 - q25


def skewness(
    x: ArrayLike,
    axis: Optional[int] = None,
    bias: bool = True,
) -> NDArray[np.floating]:
    """
    Compute sample skewness.

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which to compute.
    bias : bool, optional
        If False, apply bias correction. Default is True.

    Returns
    -------
    skew : ndarray
        Skewness value(s).
    """
    from scipy.stats import skew as scipy_skew

    return np.asarray(scipy_skew(x, axis=axis, bias=bias), dtype=np.float64)


def kurtosis(
    x: ArrayLike,
    axis: Optional[int] = None,
    fisher: bool = True,
    bias: bool = True,
) -> NDArray[np.floating]:
    """
    Compute sample kurtosis.

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which to compute.
    fisher : bool, optional
        If True, return excess kurtosis (Fisher definition).
        If False, return Pearson kurtosis. Default is True.
    bias : bool, optional
        If False, apply bias correction. Default is True.

    Returns
    -------
    kurt : ndarray
        Kurtosis value(s).
    """
    from scipy.stats import kurtosis as scipy_kurtosis

    return np.asarray(
        scipy_kurtosis(x, axis=axis, fisher=fisher, bias=bias), dtype=np.float64
    )


def moment(
    x: ArrayLike,
    order: int,
    axis: Optional[int] = None,
    central: bool = True,
) -> NDArray[np.floating]:
    """
    Compute sample moment.

    Parameters
    ----------
    x : array_like
        Input data.
    order : int
        Order of the moment.
    axis : int, optional
        Axis along which to compute.
    central : bool, optional
        If True, compute central moment. Default is True.

    Returns
    -------
    m : ndarray
        Moment value(s).
    """
    from scipy.stats import moment as scipy_moment

    if central:
        return np.asarray(scipy_moment(x, moment=order, axis=axis), dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
        return np.mean(x**order, axis=axis)


def nees(
    error: ArrayLike,
    covariance: ArrayLike,
) -> NDArray[np.floating]:
    """
    Normalized Estimation Error Squared (NEES).

    A consistency metric for estimators. For a consistent estimator,
    NEES should be chi-squared distributed with n degrees of freedom.

    Parameters
    ----------
    error : array_like
        Estimation error vector(s) of shape (n,) or (m, n).
    covariance : array_like
        Covariance matrix of shape (n, n).

    Returns
    -------
    nees : ndarray
        NEES value(s). Scalar if error is 1D, array if error is 2D.

    Examples
    --------
    >>> error = np.array([1.0, 0.5])
    >>> cov = np.array([[1, 0], [0, 1]])
    >>> nees(error, cov)
    1.25
    """
    error = np.asarray(error, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    cov_inv = np.linalg.inv(covariance)

    if error.ndim == 1:
        return error @ cov_inv @ error
    else:
        return np.sum(error @ cov_inv * error, axis=1)


def nis(
    innovation: ArrayLike,
    innovation_covariance: ArrayLike,
) -> NDArray[np.floating]:
    """
    Normalized Innovation Squared (NIS).

    A filter consistency metric based on measurement innovations.
    For a consistent filter, NIS should be chi-squared distributed.

    Parameters
    ----------
    innovation : array_like
        Innovation (measurement residual) vector(s).
    innovation_covariance : array_like
        Innovation covariance matrix.

    Returns
    -------
    nis : ndarray
        NIS value(s).

    Notes
    -----
    This is equivalent to NEES applied to innovations.
    """
    return nees(innovation, innovation_covariance)


__all__ = [
    "weighted_mean",
    "weighted_var",
    "weighted_cov",
    "sample_mean",
    "sample_var",
    "sample_cov",
    "sample_corr",
    "median",
    "mad",
    "iqr",
    "skewness",
    "kurtosis",
    "moment",
    "nees",
    "nis",
]
