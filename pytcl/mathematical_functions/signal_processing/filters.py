"""
Digital filter design and application.

This module provides functions for designing and applying various types of
digital filters including Butterworth, Chebyshev, elliptic, Bessel, and FIR.

Functions
---------
- butter_design: Butterworth filter design
- cheby1_design: Chebyshev Type I filter design
- cheby2_design: Chebyshev Type II filter design
- ellip_design: Elliptic (Cauer) filter design
- bessel_design: Bessel filter design
- fir_design: FIR filter design using windowed sinc
- apply_filter: Apply filter to signal
- filtfilt: Zero-phase forward-backward filtering
- frequency_response: Compute filter frequency response

References
----------
.. [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
       Processing (3rd ed.). Prentice Hall.
.. [2] Parks, T. W., & Burrus, C. S. (1987). Digital Filter Design.
       Wiley-Interscience.
"""

from typing import Any, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal

# =============================================================================
# Result Types
# =============================================================================


class FilterCoefficients(NamedTuple):
    """
    Filter coefficients from IIR filter design.

    Attributes
    ----------
    b : ndarray
        Numerator (feedforward) coefficients.
    a : ndarray
        Denominator (feedback) coefficients.
    sos : ndarray or None
        Second-order sections representation (more numerically stable).
    """

    b: NDArray[np.floating]
    a: NDArray[np.floating]
    sos: Optional[NDArray[np.floating]]


class FrequencyResponse(NamedTuple):
    """
    Frequency response of a digital filter.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values in Hz.
    magnitude : ndarray
        Magnitude response (linear scale).
    phase : ndarray
        Phase response in radians.
    """

    frequencies: NDArray[np.floating]
    magnitude: NDArray[np.floating]
    phase: NDArray[np.floating]


# =============================================================================
# IIR Filter Design
# =============================================================================


def butter_design(
    order: int,
    cutoff: Union[float, tuple[float, ...]],
    fs: float,
    btype: str = "low",
    output: str = "sos",
) -> FilterCoefficients:
    """
    Design a Butterworth digital filter.

    The Butterworth filter has maximally flat frequency response in the
    passband. It is often used as a "standard" filter when sharp transitions
    are not required.

    Parameters
    ----------
    order : int
        Filter order.
    cutoff : float or tuple
        Cutoff frequency in Hz. For bandpass/bandstop, provide (low, high).
    fs : float
        Sampling frequency in Hz.
    btype : {'low', 'high', 'band', 'bandstop'}, optional
        Filter type. Default is 'low'.
    output : {'sos', 'ba'}, optional
        Output format. 'sos' is recommended for stability. Default is 'sos'.

    Returns
    -------
    coeffs : FilterCoefficients
        Filter coefficients (b, a, sos).

    Examples
    --------
    >>> fs = 1000  # 1 kHz
    >>> coeffs = butter_design(4, 100, fs, btype='low')
    >>> coeffs.sos.shape[0]  # Number of second-order sections
    2

    Notes
    -----
    The Butterworth filter has no ripple in the passband or stopband. The
    -3 dB point occurs at the cutoff frequency.
    """
    # Normalize cutoff frequency to Nyquist
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq

    if output == "sos":
        sos = scipy_signal.butter(order, Wn, btype=btype, output="sos")
        b, a = scipy_signal.sos2tf(sos)
        return FilterCoefficients(b=b, a=a, sos=sos)
    else:
        b, a = scipy_signal.butter(order, Wn, btype=btype, output="ba")
        return FilterCoefficients(b=b, a=a, sos=None)


def cheby1_design(
    order: int,
    ripple: float,
    cutoff: Union[float, tuple[float, ...]],
    fs: float,
    btype: str = "low",
    output: str = "sos",
) -> FilterCoefficients:
    """
    Design a Chebyshev Type I digital filter.

    The Chebyshev Type I filter has equiripple in the passband and monotonic
    in the stopband. It provides a sharper transition than Butterworth for
    the same order.

    Parameters
    ----------
    order : int
        Filter order.
    ripple : float
        Maximum passband ripple in dB.
    cutoff : float or tuple
        Cutoff frequency in Hz. For bandpass/bandstop, provide (low, high).
    fs : float
        Sampling frequency in Hz.
    btype : {'low', 'high', 'band', 'bandstop'}, optional
        Filter type. Default is 'low'.
    output : {'sos', 'ba'}, optional
        Output format. Default is 'sos'.

    Returns
    -------
    coeffs : FilterCoefficients
        Filter coefficients.

    Examples
    --------
    >>> fs = 1000
    >>> coeffs = cheby1_design(4, 0.5, 100, fs)  # 0.5 dB ripple
    >>> len(coeffs.b) > 0
    True
    """
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq

    if output == "sos":
        sos = scipy_signal.cheby1(order, ripple, Wn, btype=btype, output="sos")
        b, a = scipy_signal.sos2tf(sos)
        return FilterCoefficients(b=b, a=a, sos=sos)
    else:
        b, a = scipy_signal.cheby1(order, ripple, Wn, btype=btype, output="ba")
        return FilterCoefficients(b=b, a=a, sos=None)


def cheby2_design(
    order: int,
    attenuation: float,
    cutoff: Union[float, tuple[float, ...]],
    fs: float,
    btype: str = "low",
    output: str = "sos",
) -> FilterCoefficients:
    """
    Design a Chebyshev Type II digital filter.

    The Chebyshev Type II filter has equiripple in the stopband and monotonic
    in the passband. The cutoff frequency is the stopband edge.

    Parameters
    ----------
    order : int
        Filter order.
    attenuation : float
        Minimum stopband attenuation in dB.
    cutoff : float or tuple
        Stopband edge frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    btype : {'low', 'high', 'band', 'bandstop'}, optional
        Filter type. Default is 'low'.
    output : {'sos', 'ba'}, optional
        Output format. Default is 'sos'.

    Returns
    -------
    coeffs : FilterCoefficients
        Filter coefficients.

    Examples
    --------
    >>> fs = 1000
    >>> coeffs = cheby2_design(4, 40, 100, fs)  # 40 dB stopband attenuation
    >>> len(coeffs.b) > 0
    True
    """
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq

    if output == "sos":
        sos = scipy_signal.cheby2(order, attenuation, Wn, btype=btype, output="sos")
        b, a = scipy_signal.sos2tf(sos)
        return FilterCoefficients(b=b, a=a, sos=sos)
    else:
        b, a = scipy_signal.cheby2(order, attenuation, Wn, btype=btype, output="ba")
        return FilterCoefficients(b=b, a=a, sos=None)


def ellip_design(
    order: int,
    passband_ripple: float,
    stopband_attenuation: float,
    cutoff: Union[float, tuple[float, ...]],
    fs: float,
    btype: str = "low",
    output: str = "sos",
) -> FilterCoefficients:
    """
    Design an elliptic (Cauer) digital filter.

    The elliptic filter has equiripple in both passband and stopband. It
    achieves the sharpest transition for a given order but at the cost of
    ripple in both bands.

    Parameters
    ----------
    order : int
        Filter order.
    passband_ripple : float
        Maximum passband ripple in dB.
    stopband_attenuation : float
        Minimum stopband attenuation in dB.
    cutoff : float or tuple
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    btype : {'low', 'high', 'band', 'bandstop'}, optional
        Filter type. Default is 'low'.
    output : {'sos', 'ba'}, optional
        Output format. Default is 'sos'.

    Returns
    -------
    coeffs : FilterCoefficients
        Filter coefficients.

    Examples
    --------
    >>> fs = 1000
    >>> coeffs = ellip_design(4, 0.5, 40, 100, fs)
    >>> len(coeffs.b) > 0
    True
    """
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq

    if output == "sos":
        sos = scipy_signal.ellip(
            order, passband_ripple, stopband_attenuation, Wn, btype=btype, output="sos"
        )
        b, a = scipy_signal.sos2tf(sos)
        return FilterCoefficients(b=b, a=a, sos=sos)
    else:
        b, a = scipy_signal.ellip(
            order, passband_ripple, stopband_attenuation, Wn, btype=btype, output="ba"
        )
        return FilterCoefficients(b=b, a=a, sos=None)


def bessel_design(
    order: int,
    cutoff: Union[float, tuple[float, ...]],
    fs: float,
    btype: str = "low",
    norm: str = "phase",
    output: str = "sos",
) -> FilterCoefficients:
    """
    Design a Bessel/Thomson digital filter.

    The Bessel filter is designed for a maximally flat group delay, making
    it ideal for preserving the waveshape of filtered signals.

    Parameters
    ----------
    order : int
        Filter order.
    cutoff : float or tuple
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    btype : {'low', 'high', 'band', 'bandstop'}, optional
        Filter type. Default is 'low'.
    norm : {'phase', 'delay', 'mag'}, optional
        Normalization type. Default is 'phase'.
    output : {'sos', 'ba'}, optional
        Output format. Default is 'sos'.

    Returns
    -------
    coeffs : FilterCoefficients
        Filter coefficients.

    Examples
    --------
    >>> fs = 1000
    >>> coeffs = bessel_design(4, 100, fs)
    >>> len(coeffs.b) > 0
    True

    Notes
    -----
    The Bessel filter has a slower roll-off than Butterworth, but preserves
    the shape of signals in the passband due to its flat group delay.
    """
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq

    if output == "sos":
        sos = scipy_signal.bessel(order, Wn, btype=btype, norm=norm, output="sos")
        b, a = scipy_signal.sos2tf(sos)
        return FilterCoefficients(b=b, a=a, sos=sos)
    else:
        b, a = scipy_signal.bessel(order, Wn, btype=btype, norm=norm, output="ba")
        return FilterCoefficients(b=b, a=a, sos=None)


# =============================================================================
# FIR Filter Design
# =============================================================================


def fir_design(
    numtaps: int,
    cutoff: Union[float, tuple[float, ...]],
    fs: float,
    window: str = "hamming",
    pass_zero: Union[bool, str] = True,
) -> NDArray[np.floating]:
    """
    Design an FIR filter using the window method.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients). Must be odd for
        Type I filter (pass_zero=True, lowpass).
    cutoff : float or tuple
        Cutoff frequency in Hz. For bandpass, provide (low, high).
    fs : float
        Sampling frequency in Hz.
    window : str, optional
        Window function to use. Default is 'hamming'.
    pass_zero : bool or {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        If True, gain at zero frequency is 1. Default is True.

    Returns
    -------
    h : ndarray
        FIR filter coefficients.

    Examples
    --------
    >>> fs = 1000
    >>> h = fir_design(101, 100, fs)  # 100 Hz lowpass
    >>> len(h)
    101

    Notes
    -----
    FIR filters are always stable and have linear phase when the coefficients
    are symmetric. However, they typically require higher order than IIR
    filters for the same transition sharpness.
    """
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        freq = [c / nyq for c in cutoff]
    else:
        freq = cutoff / nyq

    h = scipy_signal.firwin(
        numtaps,
        freq,
        window=window,
        pass_zero=pass_zero,
    )

    return h


def fir_design_remez(
    numtaps: int,
    bands: ArrayLike,
    desired: ArrayLike,
    fs: float,
    weight: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Design an optimal FIR filter using the Remez exchange algorithm.

    The Parks-McClellan (Remez) algorithm designs an optimal equiripple FIR
    filter that minimizes the maximum error between the desired and actual
    frequency response.

    Parameters
    ----------
    numtaps : int
        Length of the filter.
    bands : array_like
        Band edges in Hz. Must be monotonically increasing with an even number
        of elements.
    desired : array_like
        Desired gain in each band (one value per band).
    fs : float
        Sampling frequency in Hz.
    weight : array_like, optional
        Weight for each band. Default is equal weight.

    Returns
    -------
    h : ndarray
        FIR filter coefficients.

    Examples
    --------
    >>> fs = 1000
    >>> # Lowpass: passband 0-100 Hz, stopband 150-500 Hz
    >>> h = fir_design_remez(101, [0, 100, 150, 500], [1, 0], fs)
    >>> len(h)
    101
    """
    bands_arr = np.asarray(bands)

    h = scipy_signal.remez(
        numtaps,
        bands_arr,
        desired,
        weight=weight,
        fs=fs,
    )

    return h


# =============================================================================
# Filter Application
# =============================================================================


def apply_filter(
    coeffs: Union[FilterCoefficients, tuple[Any, ...], NDArray[Any]],
    x: ArrayLike,
    zi: Optional[ArrayLike] = None,
) -> Union[NDArray[np.floating], tuple[NDArray[np.floating], Any]]:
    """
    Apply a digital filter to a signal.

    Parameters
    ----------
    coeffs : FilterCoefficients, tuple, or ndarray
        Filter coefficients. Can be:
        - FilterCoefficients (uses sos if available)
        - Tuple (b, a) for IIR filter
        - 1D array for FIR filter
    x : array_like
        Input signal.
    zi : array_like, optional
        Initial filter state. If provided, returns (output, final_state).

    Returns
    -------
    y : ndarray or tuple
        Filtered signal. If zi is provided, returns (y, zf).

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
    >>> coeffs = butter_design(4, 100, fs)  # 100 Hz lowpass
    >>> y = apply_filter(coeffs, x)
    >>> len(y) == len(x)
    True
    """
    x = np.asarray(x, dtype=np.float64)

    if isinstance(coeffs, FilterCoefficients):
        if coeffs.sos is not None:
            if zi is not None:
                return scipy_signal.sosfilt(coeffs.sos, x, zi=zi)
            return scipy_signal.sosfilt(coeffs.sos, x)
        else:
            if zi is not None:
                return scipy_signal.lfilter(coeffs.b, coeffs.a, x, zi=zi)
            return scipy_signal.lfilter(coeffs.b, coeffs.a, x)
    elif isinstance(coeffs, tuple) and len(coeffs) == 2:
        b, a = coeffs
        if zi is not None:
            return scipy_signal.lfilter(b, a, x, zi=zi)
        return scipy_signal.lfilter(b, a, x)
    else:
        # Assume FIR coefficients
        h = np.asarray(coeffs)
        if zi is not None:
            return scipy_signal.lfilter(h, [1.0], x, zi=zi)
        return scipy_signal.lfilter(h, [1.0], x)


def filtfilt(
    coeffs: Union[FilterCoefficients, tuple[Any, ...], NDArray[Any]],
    x: ArrayLike,
    padtype: str = "odd",
    padlen: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Apply zero-phase forward-backward filtering.

    The signal is filtered twice, once forward and once backward, which
    eliminates phase distortion but doubles the filter order.

    Parameters
    ----------
    coeffs : FilterCoefficients, tuple, or ndarray
        Filter coefficients.
    x : array_like
        Input signal.
    padtype : {'odd', 'even', 'constant', None}, optional
        Type of padding to use. Default is 'odd'.
    padlen : int, optional
        Length of padding. Default is 3 * max(len(a), len(b)).

    Returns
    -------
    y : ndarray
        Zero-phase filtered signal.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t)
    >>> coeffs = butter_design(4, 100, fs)
    >>> y = filtfilt(coeffs, x)
    >>> len(y) == len(x)
    True

    Notes
    -----
    Zero-phase filtering has no phase distortion but cannot be used for
    real-time applications since it requires the entire signal upfront.
    """
    x = np.asarray(x, dtype=np.float64)

    if isinstance(coeffs, FilterCoefficients):
        if coeffs.sos is not None:
            return scipy_signal.sosfiltfilt(
                coeffs.sos, x, padtype=padtype, padlen=padlen
            )
        else:
            return scipy_signal.filtfilt(
                coeffs.b, coeffs.a, x, padtype=padtype, padlen=padlen
            )
    elif isinstance(coeffs, tuple) and len(coeffs) == 2:
        b, a = coeffs
        return scipy_signal.filtfilt(b, a, x, padtype=padtype, padlen=padlen)
    else:
        h = np.asarray(coeffs)
        return scipy_signal.filtfilt(h, [1.0], x, padtype=padtype, padlen=padlen)


# =============================================================================
# Frequency Response Analysis
# =============================================================================


def frequency_response(
    coeffs: Union[FilterCoefficients, tuple[Any, ...], NDArray[Any]],
    fs: float,
    n_points: int = 512,
    whole: bool = False,
) -> FrequencyResponse:
    """
    Compute the frequency response of a digital filter.

    Parameters
    ----------
    coeffs : FilterCoefficients, tuple, or ndarray
        Filter coefficients.
    fs : float
        Sampling frequency in Hz.
    n_points : int, optional
        Number of frequency points. Default is 512.
    whole : bool, optional
        If True, compute response from 0 to fs (instead of 0 to fs/2).
        Default is False.

    Returns
    -------
    response : FrequencyResponse
        Named tuple with frequencies, magnitude, and phase.

    Examples
    --------
    >>> fs = 1000
    >>> coeffs = butter_design(4, 100, fs)
    >>> response = frequency_response(coeffs, fs)
    >>> len(response.frequencies) == 512
    True
    >>> response.magnitude[0]  # DC gain
    1.0
    """
    if isinstance(coeffs, FilterCoefficients):
        if coeffs.sos is not None:
            w, h = scipy_signal.sosfreqz(coeffs.sos, worN=n_points, whole=whole)
        else:
            w, h = scipy_signal.freqz(coeffs.b, coeffs.a, worN=n_points, whole=whole)
    elif isinstance(coeffs, tuple) and len(coeffs) == 2:
        b, a = coeffs
        w, h = scipy_signal.freqz(b, a, worN=n_points, whole=whole)
    else:
        h_coeffs = np.asarray(coeffs)
        w, h = scipy_signal.freqz(h_coeffs, [1.0], worN=n_points, whole=whole)

    # Convert from normalized frequency to Hz
    frequencies = w * fs / (2 * np.pi)

    magnitude = np.abs(h)
    phase = np.angle(h)

    return FrequencyResponse(frequencies=frequencies, magnitude=magnitude, phase=phase)


def group_delay(
    coeffs: Union[FilterCoefficients, tuple[Any, ...], NDArray[Any]],
    fs: float,
    n_points: int = 512,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the group delay of a digital filter.

    Group delay is the negative derivative of phase with respect to frequency,
    representing the time delay experienced by signal components at each
    frequency.

    Parameters
    ----------
    coeffs : FilterCoefficients, tuple, or ndarray
        Filter coefficients.
    fs : float
        Sampling frequency in Hz.
    n_points : int, optional
        Number of frequency points. Default is 512.

    Returns
    -------
    frequencies : ndarray
        Frequency values in Hz.
    gd : ndarray
        Group delay in samples.

    Examples
    --------
    >>> fs = 1000
    >>> h = fir_design(51, 100, fs)  # Symmetric FIR
    >>> freqs, gd = group_delay(h, fs)
    >>> np.allclose(gd, 25)  # Constant group delay = (N-1)/2
    True
    """
    if isinstance(coeffs, FilterCoefficients):
        b, a = coeffs.b, coeffs.a
    elif isinstance(coeffs, tuple) and len(coeffs) == 2:
        b, a = coeffs
    else:
        b = np.asarray(coeffs)
        a = [1.0]

    w, gd = scipy_signal.group_delay((b, a), w=n_points)
    frequencies = w * fs / (2 * np.pi)

    return frequencies, gd


# =============================================================================
# Utility Functions
# =============================================================================


def filter_order(
    passband_freq: float,
    stopband_freq: float,
    passband_ripple: float,
    stopband_attenuation: float,
    fs: float,
    filter_type: str = "butter",
) -> int:
    """
    Estimate the minimum filter order for given specifications.

    Parameters
    ----------
    passband_freq : float
        Passband edge frequency in Hz.
    stopband_freq : float
        Stopband edge frequency in Hz.
    passband_ripple : float
        Maximum passband ripple in dB.
    stopband_attenuation : float
        Minimum stopband attenuation in dB.
    fs : float
        Sampling frequency in Hz.
    filter_type : {'butter', 'cheby1', 'cheby2', 'ellip'}, optional
        Filter type. Default is 'butter'.

    Returns
    -------
    order : int
        Minimum filter order.

    Examples
    --------
    >>> order = filter_order(100, 150, 0.5, 40, 1000, 'butter')
    >>> order > 0
    True
    """
    nyq = fs / 2
    wp = passband_freq / nyq
    ws = stopband_freq / nyq

    if filter_type == "butter":
        order, _ = scipy_signal.buttord(wp, ws, passband_ripple, stopband_attenuation)
    elif filter_type == "cheby1":
        order, _ = scipy_signal.cheb1ord(wp, ws, passband_ripple, stopband_attenuation)
    elif filter_type == "cheby2":
        order, _ = scipy_signal.cheb2ord(wp, ws, passband_ripple, stopband_attenuation)
    elif filter_type == "ellip":
        order, _ = scipy_signal.ellipord(wp, ws, passband_ripple, stopband_attenuation)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return int(order)


def sos_to_zpk(sos: ArrayLike) -> tuple[NDArray[Any], NDArray[Any], Any]:
    """
    Convert second-order sections to zeros, poles, gain.

    Parameters
    ----------
    sos : array_like
        Second-order sections array with shape (n_sections, 6).

    Returns
    -------
    z : ndarray
        Zeros.
    p : ndarray
        Poles.
    k : float
        Gain.
    """
    return scipy_signal.sos2zpk(np.asarray(sos))


def zpk_to_sos(
    z: ArrayLike, p: ArrayLike, k: float, pairing: str = "nearest"
) -> NDArray[np.floating]:
    """
    Convert zeros, poles, gain to second-order sections.

    Parameters
    ----------
    z : array_like
        Zeros.
    p : array_like
        Poles.
    k : float
        Gain.
    pairing : {'nearest', 'keep_odd', 'minimal'}, optional
        Pole-zero pairing strategy. Default is 'nearest'.

    Returns
    -------
    sos : ndarray
        Second-order sections array.
    """
    return scipy_signal.zpk2sos(np.asarray(z), np.asarray(p), k, pairing=pairing)
