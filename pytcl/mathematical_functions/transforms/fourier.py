"""
Fourier transform utilities.

This module provides wrappers and utilities for Discrete Fourier Transform
operations, power spectral density estimation, and spectral analysis.

Functions
---------
- fft, ifft: Complex FFT and inverse
- rfft, irfft: Real-signal FFT and inverse
- fftshift, ifftshift: Shift zero-frequency to center
- power_spectrum: Power spectral density estimation
- cross_spectrum: Cross-spectral density
- coherence: Magnitude-squared coherence
- frequency_axis: Generate frequency axis for FFT

References
----------
.. [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
       Processing (3rd ed.). Prentice Hall.
.. [2] Welch, P. D. (1967). The use of fast Fourier transform for the
       estimation of power spectra. IEEE Transactions on Audio and
       Electroacoustics, 15(2), 70-73.
"""

from typing import NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal

# =============================================================================
# Result Types
# =============================================================================


class PowerSpectrum(NamedTuple):
    """
    Result of power spectrum estimation.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values in Hz.
    psd : ndarray
        Power spectral density estimate.
    """

    frequencies: NDArray[np.floating]
    psd: NDArray[np.floating]


class CrossSpectrum(NamedTuple):
    """
    Result of cross-spectrum estimation.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values in Hz.
    csd : ndarray
        Cross-spectral density estimate (complex).
    """

    frequencies: NDArray[np.floating]
    csd: NDArray[np.complexfloating]


class CoherenceResult(NamedTuple):
    """
    Result of coherence estimation.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values in Hz.
    coherence : ndarray
        Magnitude-squared coherence, values between 0 and 1.
    """

    frequencies: NDArray[np.floating]
    coherence: NDArray[np.floating]


# =============================================================================
# Core FFT Functions
# =============================================================================


def fft(
    x: ArrayLike,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[str] = None,
) -> NDArray[np.complexfloating]:
    """
    Compute the one-dimensional discrete Fourier Transform.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output. If n is smaller than
        the length of the input, the input is cropped. If it is larger,
        the input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. Default is -1.
    norm : {None, "ortho", "forward", "backward"}, optional
        Normalization mode. Default is None (backward).

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 1.0, -1.0])
    >>> X = fft(x)
    >>> np.allclose(X, [3.+0.j, 0.-2.j, 1.+0.j, 0.+2.j])
    True
    """
    x = np.asarray(x)
    return scipy_fft.fft(x, n=n, axis=axis, norm=norm)


def ifft(
    X: ArrayLike,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[str] = None,
) -> NDArray[np.complexfloating]:
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    X : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
    axis : int, optional
        Axis over which to compute the inverse FFT. Default is -1.
    norm : {None, "ortho", "forward", "backward"}, optional
        Normalization mode. Default is None (backward).

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([3.+0.j, 0.-2.j, 1.+0.j, 0.+2.j])
    >>> x = ifft(X)
    >>> np.allclose(x, [1.0, 2.0, 1.0, -1.0])
    True
    """
    X = np.asarray(X)
    return scipy_fft.ifft(X, n=n, axis=axis, norm=norm)


def rfft(
    x: ArrayLike,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[str] = None,
) -> NDArray[np.complexfloating]:
    """
    Compute the one-dimensional FFT for real input.

    This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    x : array_like
        Input array, must be real.
    n : int, optional
        Number of points to use. If n is smaller than the length of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. Default is -1.
    norm : {None, "ortho", "forward", "backward"}, optional
        Normalization mode.

    Returns
    -------
    out : ndarray
        The FFT of the input along the indicated axis. Only the non-negative
        frequency terms are returned (n//2 + 1 points).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0., 1., 0., 0.])
    >>> X = rfft(x)
    >>> len(X)  # Only positive frequencies
    3
    """
    x = np.asarray(x, dtype=np.float64)
    return scipy_fft.rfft(x, n=n, axis=axis, norm=norm)


def irfft(
    X: ArrayLike,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[str] = None,
) -> NDArray[np.floating]:
    """
    Compute the inverse FFT for real output.

    Parameters
    ----------
    X : array_like
        Input array (typically from rfft).
    n : int, optional
        Length of the output (along the transformed axis).
        If not given, n = 2*(len(X)-1).
    axis : int, optional
        Axis over which to compute the inverse FFT. Default is -1.
    norm : {None, "ortho", "forward", "backward"}, optional
        Normalization mode.

    Returns
    -------
    out : ndarray
        Real-valued inverse FFT result.

    Examples
    --------
    >>> import numpy as np
    >>> X = rfft([0., 1., 0., 0.])
    >>> x = irfft(X)
    >>> np.allclose(x, [0., 1., 0., 0.])
    True
    """
    X = np.asarray(X)
    return scipy_fft.irfft(X, n=n, axis=axis, norm=norm)


def fft2(
    x: ArrayLike,
    s: Optional[tuple[int, ...]] = None,
    axes: tuple[int, ...] = (-2, -1),
    norm: Optional[str] = None,
) -> NDArray[np.complexfloating]:
    """
    Compute the 2-dimensional discrete Fourier Transform.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : tuple of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : tuple of ints, optional
        Axes over which to compute the FFT. Default is (-2, -1).
    norm : {None, "ortho", "forward", "backward"}, optional
        Normalization mode.

    Returns
    -------
    out : ndarray
        The 2D FFT of the input.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 0], [0, 1]], dtype=float)
    >>> X = fft2(x)
    >>> X.shape
    (2, 2)
    """
    x = np.asarray(x)
    return scipy_fft.fft2(x, s=s, axes=axes, norm=norm)


def ifft2(
    X: ArrayLike,
    s: Optional[tuple[int, ...]] = None,
    axes: tuple[int, ...] = (-2, -1),
    norm: Optional[str] = None,
) -> NDArray[np.complexfloating]:
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    X : array_like
        Input array, can be complex.
    s : tuple of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : tuple of ints, optional
        Axes over which to compute the inverse FFT. Default is (-2, -1).
    norm : {None, "ortho", "forward", "backward"}, optional
        Normalization mode.

    Returns
    -------
    out : ndarray
        The 2D inverse FFT of the input.
    """
    X = np.asarray(X)
    return scipy_fft.ifft2(X, s=s, axes=axes, norm=norm)


def fftshift(
    x: ArrayLike,
    axes: Optional[Union[int, tuple[int, ...]]] = None,
) -> NDArray[np.floating]:
    """
    Shift the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or tuple of ints, optional
        Axes over which to shift. Default is all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    Examples
    --------
    >>> import numpy as np
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    >>> fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    """
    x = np.asarray(x)
    return scipy_fft.fftshift(x, axes=axes)


def ifftshift(
    x: ArrayLike,
    axes: Optional[Union[int, tuple[int, ...]]] = None,
) -> NDArray[np.floating]:
    """
    Inverse of fftshift. Shift zero-frequency back to beginning.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or tuple of ints, optional
        Axes over which to shift. Default is all axes.

    Returns
    -------
    y : ndarray
        The shifted array.
    """
    x = np.asarray(x)
    return scipy_fft.ifftshift(x, axes=axes)


# =============================================================================
# Frequency Axis Utilities
# =============================================================================


def frequency_axis(n: int, fs: float, shift: bool = False) -> NDArray[np.floating]:
    """
    Generate frequency axis values for FFT output.

    Parameters
    ----------
    n : int
        Number of FFT points.
    fs : float
        Sampling frequency in Hz.
    shift : bool, optional
        If True, return frequencies in centered (fftshift) order.
        Default is False.

    Returns
    -------
    freqs : ndarray
        Frequency values in Hz.

    Examples
    --------
    >>> frequency_axis(8, 100.0)
    array([  0. ,  12.5,  25. ,  37.5, -50. , -37.5, -25. , -12.5])
    >>> frequency_axis(8, 100.0, shift=True)
    array([-50. , -37.5, -25. , -12.5,   0. ,  12.5,  25. ,  37.5])
    """
    freqs = scipy_fft.fftfreq(n, d=1.0 / fs)
    if shift:
        freqs = fftshift(freqs)
    return freqs


def rfft_frequency_axis(n: int, fs: float) -> NDArray[np.floating]:
    """
    Generate frequency axis values for rfft output.

    Parameters
    ----------
    n : int
        Number of points in the original signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : ndarray
        Non-negative frequency values in Hz.

    Examples
    --------
    >>> rfft_frequency_axis(8, 100.0)
    array([ 0. , 12.5, 25. , 37.5, 50. ])
    """
    return scipy_fft.rfftfreq(n, d=1.0 / fs)


# =============================================================================
# Spectral Analysis
# =============================================================================


def power_spectrum(
    x: ArrayLike,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
    scaling: str = "density",
) -> PowerSpectrum:
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    x : array_like
        Time series data.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, optional
        Window function to use. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Number of points to overlap between segments. Default is nperseg//2.
    nfft : int, optional
        Length of the FFT used. Default is nperseg.
    detrend : str or bool, optional
        Detrending method: 'constant', 'linear', or False. Default is 'constant'.
    scaling : {'density', 'spectrum'}, optional
        'density' for power spectral density (V^2/Hz),
        'spectrum' for power spectrum (V^2). Default is 'density'.

    Returns
    -------
    result : PowerSpectrum
        Named tuple with frequencies and power spectral density.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000  # 1 kHz
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine
    >>> result = power_spectrum(x, fs=fs)
    >>> peak_freq = result.frequencies[np.argmax(result.psd)]
    >>> abs(peak_freq - 100) < 5  # Peak near 100 Hz
    True

    Notes
    -----
    Uses Welch's method which averages modified periodograms of overlapping
    segments to reduce variance of the spectral estimate.
    """
    x = np.asarray(x, dtype=np.float64)

    if nperseg is None:
        nperseg = min(256, len(x))

    frequencies, psd = scipy_signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
    )

    return PowerSpectrum(frequencies=frequencies, psd=psd)


def cross_spectrum(
    x: ArrayLike,
    y: ArrayLike,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
) -> CrossSpectrum:
    """
    Estimate cross-spectral density between two signals.

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Number of points to overlap between segments. Default is nperseg//2.
    nfft : int, optional
        Length of FFT. Default is nperseg.
    detrend : str or bool, optional
        Detrending method. Default is 'constant'.

    Returns
    -------
    result : CrossSpectrum
        Named tuple with frequencies and cross-spectral density.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t)
    >>> y = np.sin(2 * np.pi * 50 * t + np.pi/4)  # Same freq, phase shifted
    >>> result = cross_spectrum(x, y, fs=fs)
    >>> len(result.frequencies) > 0
    True
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if nperseg is None:
        nperseg = min(256, len(x))

    frequencies, csd = scipy_signal.csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
    )

    return CrossSpectrum(frequencies=frequencies, csd=csd)


def coherence(
    x: ArrayLike,
    y: ArrayLike,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
) -> CoherenceResult:
    """
    Estimate magnitude-squared coherence between two signals.

    The coherence measures the linear correlation between two signals as a
    function of frequency. Values range from 0 (no correlation) to 1
    (perfect correlation).

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Number of overlap points. Default is nperseg//2.
    nfft : int, optional
        Length of FFT. Default is nperseg.
    detrend : str or bool, optional
        Detrending method. Default is 'constant'.

    Returns
    -------
    result : CoherenceResult
        Named tuple with frequencies and coherence values.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t)
    >>> y = 2 * x + 0.1 * np.random.randn(len(t))  # Correlated
    >>> result = coherence(x, y, fs=fs)
    >>> np.max(result.coherence) > 0.9  # High coherence at 50 Hz
    True

    Notes
    -----
    Coherence is defined as:
        C_xy = |S_xy|^2 / (S_xx * S_yy)

    where S_xy is the cross-spectral density and S_xx, S_yy are the power
    spectral densities.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if nperseg is None:
        nperseg = min(256, len(x))

    frequencies, coh = scipy_signal.coherence(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
    )

    return CoherenceResult(frequencies=frequencies, coherence=coh)


# =============================================================================
# Utility Functions
# =============================================================================


def periodogram(
    x: ArrayLike,
    fs: float = 1.0,
    window: Optional[str] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
    scaling: str = "density",
) -> PowerSpectrum:
    """
    Estimate power spectral density using a periodogram.

    Unlike Welch's method, the periodogram uses the entire signal without
    segmentation and averaging. This gives higher frequency resolution but
    higher variance.

    Parameters
    ----------
    x : array_like
        Time series data.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, optional
        Window function. Default is None (rectangular).
    nfft : int, optional
        Length of FFT. Default is len(x).
    detrend : str or bool, optional
        Detrending method. Default is 'constant'.
    scaling : {'density', 'spectrum'}, optional
        Scaling mode. Default is 'density'.

    Returns
    -------
    result : PowerSpectrum
        Named tuple with frequencies and power spectral density.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 100 * t)
    >>> result = periodogram(x, fs=fs)
    >>> len(result.frequencies) == len(result.psd)
    True
    """
    x = np.asarray(x, dtype=np.float64)

    frequencies, psd = scipy_signal.periodogram(
        x,
        fs=fs,
        window=window,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
    )

    return PowerSpectrum(frequencies=frequencies, psd=psd)


def magnitude_spectrum(
    X: ArrayLike,
    scale: str = "linear",
) -> NDArray[np.floating]:
    """
    Compute magnitude spectrum from FFT coefficients.

    Parameters
    ----------
    X : array_like
        Complex FFT coefficients.
    scale : {'linear', 'dB'}, optional
        Output scale. Default is 'linear'.

    Returns
    -------
    mag : ndarray
        Magnitude spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([4+0j, 0-2j, 0+0j, 0+2j])
    >>> magnitude_spectrum(X)
    array([4., 2., 0., 2.])
    >>> magnitude_spectrum(X, scale='dB')  # doctest: +SKIP
    array([12.04..., 6.02..., -inf, 6.02...])
    """
    X = np.asarray(X)
    mag = np.abs(X)

    if scale == "dB":
        with np.errstate(divide="ignore"):
            mag = 20 * np.log10(mag)
    elif scale != "linear":
        raise ValueError(f"scale must be 'linear' or 'dB', got '{scale}'")

    return mag


def phase_spectrum(
    X: ArrayLike,
    unwrap: bool = False,
) -> NDArray[np.floating]:
    """
    Compute phase spectrum from FFT coefficients.

    Parameters
    ----------
    X : array_like
        Complex FFT coefficients.
    unwrap : bool, optional
        If True, unwrap the phase to remove discontinuities.
        Default is False.

    Returns
    -------
    phase : ndarray
        Phase spectrum in radians.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1+0j, 0+1j, -1+0j, 0-1j])
    >>> phase = phase_spectrum(X)
    >>> np.allclose(phase, [0, np.pi/2, np.pi, -np.pi/2])
    True
    """
    X = np.asarray(X)
    phase = np.angle(X)

    if unwrap:
        phase = np.unwrap(phase)

    return phase
