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
- Allen, J. (1977). Short term spectral analysis, synthesis, and
  modification by discrete Fourier transform. IEEE Transactions on
  Acoustics, Speech, and Signal Processing, 25(3), 235-238.
- Griffin, D., & Lim, J. (1984). Signal estimation from modified
  short-time Fourier transform. IEEE Transactions on Acoustics,
  Speech, and Signal Processing, 32(2), 236-243.
"""

from typing import Any, NamedTuple, Optional, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
        Whatever `spectrogram`'s ``scaling`` and ``mode`` arguments select,
        which at their defaults (``scaling="density"``, ``mode="psd"``) is a
        power spectral *density* in units of V**2/Hz -- not ``|STFT|**2``.
        scipy's density scaling divides by ``fs * sum(window**2)`` and
        doubles the one-sided bins, so the two differ by a factor that
        scales with the sample rate and the window (24,000 at ``fs=1000``
        with a 128-point Hann window). Pass ``scaling="spectrum"`` for a
        power spectrum. With ``mode="complex"`` this field is complex, and
        with ``mode="angle"``/``"phase"`` it is an angle in radians, despite
        the name.
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
    >>> float(w[0]), round(float(w[-1]), 6)  # Near-zero at edges (periodic window)
    (0.0, 0.000151)
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
    (65, 17)

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
    (65, 17)

    Notes
    -----
    Shows how the spectral content of the signal evolves over time.

    The returned ``power`` is not simply the magnitude squared of the STFT:
    it carries whichever normalisation ``scaling`` and ``mode`` select, and
    the default ``scaling="density"`` is a power spectral density, dividing
    by ``fs * sum(window**2)`` and doubling the one-sided bins. Compare
    `reassigned_spectrogram`, which documents and implements the same
    density scaling explicitly. Use ``scaling="spectrum"`` if you want a
    power spectrum whose bins sum to the signal's mean square.
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

    if x.size < nperseg:
        raise ValueError(f"signal has {x.size} samples, fewer than nperseg={nperseg}")

    step = nperseg - noverlap
    frames = sliding_window_view(x, nperseg)[::step]

    # All three transforms share one scaling. Routing the modified windows
    # through `stft` would not work: it normalizes by the window sum, and the
    # derivative window sums to zero, so its transform would be divided by
    # roughly 1e-7. The ratios below are what the reassignment needs, and they
    # are only meaningful if the numerator and denominator are scaled alike.
    ramp = np.arange(nperseg) - (nperseg - 1) / 2  # samples from frame center
    stft_w = np.fft.rfft(frames * win, n=nfft, axis=-1).T
    stft_tw = np.fft.rfft(frames * (ramp * win), n=nfft, axis=-1).T
    stft_dw = np.fft.rfft(frames * (np.gradient(win) * fs), n=nfft, axis=-1).T

    frequencies = np.fft.rfftfreq(nfft, 1 / fs)
    times = (np.arange(frames.shape[0]) * step + (nperseg - 1) / 2) / fs

    # Power spectral density, on the same scaling `spectrogram` uses, so the
    # two are directly comparable: a caller should be able to swap one for the
    # other and see sharper structure rather than different units. One-sided
    # bins carry the energy of their negative-frequency twin, except DC and
    # Nyquist, which have none.
    power = np.abs(stft_w) ** 2 / (fs * np.sum(win**2))
    power[1:] *= 2.0
    if nfft % 2 == 0:
        power[-1] /= 2.0

    # Bins with no energy have no meaningful center of gravity: the ratios
    # below are 0/0 there. They carry nothing to reassign, so they are simply
    # left where they are.
    occupied = power > np.finfo(np.float64).tiny
    ratio_t = np.zeros_like(stft_w)
    ratio_d = np.zeros_like(stft_w)
    np.divide(stft_tw, stft_w, out=ratio_t, where=occupied)
    np.divide(stft_dw, stft_w, out=ratio_d, where=occupied)

    # Reassigned coordinates: the local group delay and instantaneous
    # frequency, which is where the energy in each bin actually sits.
    time_grid, freq_grid = np.meshgrid(times, frequencies)
    reassigned_time = time_grid + np.real(ratio_t) / fs
    reassigned_freq = freq_grid - np.imag(ratio_d) / (2 * np.pi)

    # Scatter each bin's energy into the cell its corrected coordinates land
    # in. Energy that reassigns off the edge of the grid is dropped rather
    # than clamped, because piling it onto the boundary would invent a
    # spectral peak that is not there.
    df = frequencies[1] - frequencies[0] if frequencies.size > 1 else 1.0
    dt = times[1] - times[0] if times.size > 1 else 1.0
    freq_index = np.rint(reassigned_freq / df).astype(np.intp)
    time_index = np.rint((reassigned_time - times[0]) / dt).astype(np.intp)

    inside = (
        occupied
        & (freq_index >= 0)
        & (freq_index < frequencies.size)
        & (time_index >= 0)
        & (time_index < times.size)
    )

    reassigned = np.zeros_like(power)
    np.add.at(
        reassigned,
        (freq_index[inside], time_index[inside]),
        power[inside],
    )

    return frequencies, times, reassigned


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
    return np.asarray(_mel_to_hz(mels))


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
