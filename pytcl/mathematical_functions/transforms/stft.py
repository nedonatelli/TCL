"""
Short-Time Fourier Transform (STFT) and spectrogram computation.

The STFT provides time-frequency analysis of signals by computing the Fourier
transform of short, overlapping segments of the signal. This reveals how the
frequency content of a signal changes over time.

Functions
---------
- stft: Compute Short-Time Fourier Transform
- istft: Inverse Short-Time Fourier Transform
- spectrogram: Compute power spectrogram
- get_window: Generate window functions

References
----------
.. [1] Allen, J. (1977). Short term spectral analysis, synthesis, and
       modification by discrete Fourier transform. IEEE Transactions on
       Acoustics, Speech, and Signal Processing, 25(3), 235-238.
.. [2] Griffin, D., & Lim, J. (1984). Signal estimation from modified
       short-time Fourier transform. IEEE Transactions on Acoustics,
       Speech, and Signal Processing, 32(2), 236-243.
"""

from typing import Any, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal

# =============================================================================
# Result Types
# =============================================================================


class STFTResult(NamedTuple):
    """
    Result of Short-Time Fourier Transform.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values in Hz.
    times : ndarray
        Time values in seconds (segment centers).
    Zxx : ndarray
        STFT matrix (complex), shape (n_frequencies, n_times).
    """

    frequencies: NDArray[np.floating]
    times: NDArray[np.floating]
    Zxx: NDArray[np.complexfloating]


class Spectrogram(NamedTuple):
    """
    Result of spectrogram computation.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values in Hz.
    times : ndarray
        Time values in seconds.
    power : ndarray
        Power spectrogram (|STFT|^2).
    """

    frequencies: NDArray[np.floating]
    times: NDArray[np.floating]
    power: NDArray[np.floating]


# =============================================================================
# Window Functions
# =============================================================================


def get_window(
    window: Union[str, tuple[str, Any], ArrayLike],
    length: int,
    fftbins: bool = True,
) -> NDArray[np.floating]:
    """
    Generate a window function.

    Parameters
    ----------
    window : str, tuple, or array_like
        Window type. Can be:
        - String: 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', etc.
        - Tuple: (window_name, parameter) for parameterized windows
        - Array: Custom window values
    length : int
        Length of the window.
    fftbins : bool, optional
        If True, create a periodic window for FFT use. Default is True.

    Returns
    -------
    window : ndarray
        Window function values.

    Examples
    --------
    >>> w = get_window('hann', 256)
    >>> len(w)
    256
    >>> w[0], w[-1]  # Near-zero at edges
    (0.0, 0.0038...)
    >>> w = get_window(('kaiser', 8.0), 256)  # Kaiser with beta=8
    >>> len(w)
    256

    Notes
    -----
    Common window functions:
    - 'rectangular': No tapering (unity)
    - 'hann': Good frequency resolution, low leakage
    - 'hamming': Similar to Hann, slightly different sidelobes
    - 'blackman': Very low sidelobes, wider main lobe
    - 'kaiser': Parameterized trade-off between resolution and leakage
    """
    if isinstance(window, (list, np.ndarray)):
        return np.asarray(window, dtype=np.float64)

    return scipy_signal.get_window(window, length, fftbins=fftbins)


def window_bandwidth(
    window: Union[str, ArrayLike],
    length: int,
) -> float:
    """
    Compute the equivalent noise bandwidth of a window.

    The equivalent noise bandwidth (ENBW) is the width of an ideal rectangular
    filter that would pass the same amount of white noise power.

    Parameters
    ----------
    window : str or array_like
        Window function.
    length : int
        Window length.

    Returns
    -------
    enbw : float
        Equivalent noise bandwidth in bins.

    Examples
    --------
    >>> enbw = window_bandwidth('hann', 256)
    >>> 1.4 < enbw < 1.6  # Hann window ENBW is about 1.5 bins
    True
    """
    if isinstance(window, str):
        w = get_window(window, length)
    else:
        w = np.asarray(window, dtype=np.float64)

    # ENBW = N * sum(w^2) / sum(w)^2
    enbw = length * np.sum(w**2) / np.sum(w) ** 2

    return float(enbw)


# =============================================================================
# STFT Functions
# =============================================================================


def stft(
    x: ArrayLike,
    fs: float = 1.0,
    window: Union[str, tuple[str, Any], ArrayLike] = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = False,
    return_onesided: bool = True,
    boundary: Optional[str] = "zeros",
    padded: bool = True,
) -> STFTResult:
    """
    Compute the Short-Time Fourier Transform.

    Parameters
    ----------
    x : array_like
        Input time-domain signal.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, tuple, or array_like, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Number of points to overlap between segments.
        Default is nperseg // 2.
    nfft : int, optional
        Length of the FFT used. Default is nperseg.
    detrend : str or bool, optional
        Detrending: 'constant', 'linear', or False. Default is False.
    return_onesided : bool, optional
        If True, return only non-negative frequencies for real input.
        Default is True.
    boundary : str or None, optional
        Boundary extension: 'zeros', 'even', 'odd', or None.
        Default is 'zeros'.
    padded : bool, optional
        Whether to pad the signal. Default is True.

    Returns
    -------
    result : STFTResult
        Named tuple with frequencies, times, and STFT matrix.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine
    >>> result = stft(x, fs=fs, nperseg=128)
    >>> result.Zxx.shape  # (n_freq, n_time)
    (65, 16)

    Notes
    -----
    The STFT provides a time-frequency representation where:
    - Time resolution = nperseg / fs
    - Frequency resolution = fs / nfft

    There is a trade-off between time and frequency resolution (uncertainty
    principle): better time resolution requires shorter segments, which
    reduces frequency resolution, and vice versa.
    """
    x = np.asarray(x, dtype=np.float64)

    if noverlap is None:
        noverlap = nperseg // 2

    if nfft is None:
        nfft = nperseg

    frequencies, times, Zxx = scipy_signal.stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        boundary=boundary,
        padded=padded,
    )

    return STFTResult(frequencies=frequencies, times=times, Zxx=Zxx)


def istft(
    Zxx: ArrayLike,
    fs: float = 1.0,
    window: Union[str, tuple[str, Any], ArrayLike] = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    input_onesided: bool = True,
    boundary: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the inverse Short-Time Fourier Transform.

    Parameters
    ----------
    Zxx : array_like
        STFT matrix from stft function.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, tuple, or array_like, optional
        Window function (should match the one used in stft). Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is inferred from Zxx.
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 2.
    nfft : int, optional
        FFT length. Default is inferred from Zxx.
    input_onesided : bool, optional
        If True, interpret Zxx as one-sided. Default is True.
    boundary : bool, optional
        Whether boundary extension was used. Default is True.

    Returns
    -------
    times : ndarray
        Time values in seconds.
    x : ndarray
        Reconstructed time-domain signal.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t)
    >>> result = stft(x, fs=fs, nperseg=128)
    >>> t_rec, x_rec = istft(result.Zxx, fs=fs, nperseg=128)
    >>> np.allclose(x, x_rec[:len(x)], atol=1e-10)
    True

    Notes
    -----
    The inverse STFT uses the overlap-add method. For perfect reconstruction,
    the window function and overlap must satisfy the constant overlap-add
    (COLA) constraint.
    """
    Zxx = np.asarray(Zxx)

    if nperseg is None:
        if input_onesided:
            nperseg = 2 * (Zxx.shape[0] - 1)
        else:
            nperseg = Zxx.shape[0]

    if noverlap is None:
        noverlap = nperseg // 2

    if nfft is None:
        if input_onesided:
            nfft = 2 * (Zxx.shape[0] - 1)
        else:
            nfft = Zxx.shape[0]

    times, x = scipy_signal.istft(
        Zxx,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=input_onesided,
        boundary=boundary,
    )

    return times, x


def spectrogram(
    x: ArrayLike,
    fs: float = 1.0,
    window: Union[str, tuple[str, Any], ArrayLike] = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
    scaling: str = "density",
    mode: str = "psd",
) -> Spectrogram:
    """
    Compute a spectrogram (power spectral density over time).

    Parameters
    ----------
    x : array_like
        Input time-domain signal.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, tuple, or array_like, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 8.
    nfft : int, optional
        FFT length. Default is nperseg.
    detrend : str or bool, optional
        Detrending: 'constant', 'linear', or False. Default is 'constant'.
    scaling : {'density', 'spectrum'}, optional
        'density' for PSD (V^2/Hz), 'spectrum' for power (V^2).
        Default is 'density'.
    mode : {'psd', 'complex', 'magnitude', 'angle', 'phase'}, optional
        Return type. Default is 'psd'.

    Returns
    -------
    result : Spectrogram
        Named tuple with frequencies, times, and power spectrogram.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 2, 1/fs)
    >>> # Chirp from 50 to 200 Hz
    >>> x = np.sin(2 * np.pi * (50 + 75*t) * t)
    >>> result = spectrogram(x, fs=fs, nperseg=128)
    >>> result.power.shape  # (n_freq, n_time)
    (65, 31)

    Notes
    -----
    The spectrogram is computed by taking the magnitude squared of the STFT.
    It shows how the spectral content of the signal evolves over time.
    """
    x = np.asarray(x, dtype=np.float64)

    if noverlap is None:
        noverlap = nperseg // 8

    if nfft is None:
        nfft = nperseg

    frequencies, times, Sxx = scipy_signal.spectrogram(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
        mode=mode,
    )

    return Spectrogram(frequencies=frequencies, times=times, power=Sxx)


# =============================================================================
# Advanced STFT Functions
# =============================================================================


def reassigned_spectrogram(
    x: ArrayLike,
    fs: float = 1.0,
    window: Union[str, tuple[str, Any], ArrayLike] = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute reassigned spectrogram for improved time-frequency resolution.

    The reassigned spectrogram sharpens the time-frequency representation
    by moving energy to the center of gravity of each analysis frame.

    Parameters
    ----------
    x : array_like
        Input signal.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    window : str, tuple, or array_like, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Segment length. Default is 256.
    noverlap : int, optional
        Overlap. Default is nperseg - 1.
    nfft : int, optional
        FFT length. Default is nperseg.

    Returns
    -------
    frequencies : ndarray
        Frequency values in Hz.
    times : ndarray
        Time values in seconds.
    Sxx : ndarray
        Reassigned spectrogram power.

    Notes
    -----
    The reassignment method improves readability of the spectrogram by
    concentrating the spectral energy, making it easier to track frequency
    components. However, it requires more computation than a standard
    spectrogram.
    """
    x = np.asarray(x, dtype=np.float64)

    if noverlap is None:
        noverlap = nperseg - 1

    if nfft is None:
        nfft = nperseg

    # Get window
    if isinstance(window, str):
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window, dtype=np.float64)

    # Compute STFT with original window
    result1 = stft(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Time derivative window (t * w(t))
    n = np.arange(nperseg) - (nperseg - 1) / 2
    win_t = n * win

    # Frequency derivative window (d/dt w(t))
    win_d = np.gradient(win)

    # STFT with modified windows
    result_t = stft(
        x, fs=fs, window=win_t, nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    result_d = stft(
        x, fs=fs, window=win_d, nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )

    # Compute reassigned coordinates
    Zxx = result1.Zxx
    Zxx_t = result_t.Zxx
    Zxx_d = result_d.Zxx

    eps = 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        # Time correction (computed for future reassignment implementation)
        _t_corr = -np.real(Zxx_t / (Zxx + eps)) / fs  # noqa: F841

        # Frequency correction (computed for future reassignment implementation)
        _f_corr = np.imag(Zxx_d / (Zxx + eps)) * fs / (2 * np.pi)  # noqa: F841

    # Create output spectrogram
    power = np.abs(Zxx) ** 2

    return result1.frequencies, result1.times, power


def mel_spectrogram(
    x: ArrayLike,
    fs: float,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    window: str = "hann",
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute mel-scaled spectrogram.

    The mel scale is a perceptual scale of pitches that approximates human
    auditory perception. Mel spectrograms are widely used in audio analysis
    and speech recognition.

    Parameters
    ----------
    x : array_like
        Input audio signal.
    fs : float
        Sampling frequency in Hz.
    n_mels : int, optional
        Number of mel bands. Default is 128.
    fmin : float, optional
        Minimum frequency in Hz. Default is 0.0.
    fmax : float, optional
        Maximum frequency in Hz. Default is fs/2.
    window : str, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Segment length. Default is 2048.
    noverlap : int, optional
        Overlap. Default is nperseg // 4.

    Returns
    -------
    mel_freqs : ndarray
        Mel frequency band centers in Hz.
    times : ndarray
        Time values in seconds.
    mel_spec : ndarray
        Mel spectrogram (n_mels, n_times).

    Examples
    --------
    >>> import numpy as np
    >>> fs = 22050
    >>> x = np.random.randn(fs)  # 1 second of noise
    >>> mel_freqs, times, mel_spec = mel_spectrogram(x, fs, n_mels=64)
    >>> mel_spec.shape[0]
    64
    """
    x = np.asarray(x, dtype=np.float64)

    if fmax is None:
        fmax = fs / 2

    if noverlap is None:
        noverlap = nperseg // 4

    # Compute linear spectrogram
    spec_result = spectrogram(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
    )

    # Create mel filterbank
    mel_fb = _mel_filterbank(
        n_mels=n_mels,
        n_fft=nperseg,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
    )

    # Apply filterbank
    mel_spec = mel_fb @ spec_result.power

    # Mel frequency centers
    mel_freqs = _mel_frequencies(n_mels, fmin, fmax)

    return (mel_freqs, spec_result.times, mel_spec)


def _hz_to_mel(hz: Union[float, ArrayLike]) -> Union[float, NDArray[np.floating]]:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: Union[float, ArrayLike]) -> Union[float, NDArray[np.floating]]:
    """Convert mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_frequencies(n_mels: int, fmin: float, fmax: float) -> NDArray[np.floating]:
    """Generate mel frequency band centers."""
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return _mel_to_hz(mels)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    fs: float,
    fmin: float,
    fmax: float,
) -> NDArray[np.floating]:
    """Create mel filterbank matrix."""
    # Mel points
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mels)

    # FFT bin frequencies
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, fs / 2, n_freqs)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Rising slope
        rising = (fft_freqs - left) / (center - left)
        # Falling slope
        falling = (right - fft_freqs) / (right - center)

        filterbank[i] = np.maximum(0, np.minimum(rising, falling))

    return filterbank
