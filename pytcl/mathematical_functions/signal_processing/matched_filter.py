"""
Matched filtering for signal detection.

Matched filtering is the optimal linear filter for maximizing the signal-to-
noise ratio (SNR) of a known signal in the presence of additive white Gaussian
noise. It is widely used in radar, sonar, and communications.

Functions
---------
- matched_filter: Time-domain matched filtering
- matched_filter_frequency: Frequency-domain matched filtering
- optimal_filter: Optimal filter for colored noise
- pulse_compression: Pulse compression for chirp signals
- generate_lfm_chirp: Generate linear frequency modulated chirp

References
----------
.. [1] Richards, M. A. (2014). Fundamentals of Radar Signal Processing
       (2nd ed.). McGraw-Hill.
.. [2] Turin, G. L. (1960). An introduction to matched filters.
       IRE Transactions on Information Theory, 6(3), 311-329.
"""

from typing import Any, NamedTuple, Optional

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal

# =============================================================================
# Result Types
# =============================================================================


class MatchedFilterResult(NamedTuple):
    """
    Result of matched filter operation.

    Attributes
    ----------
    output : ndarray
        Matched filter output.
    peak_index : int
        Index of the peak (detection point).
    peak_value : float
        Value at the peak.
    snr_gain : float
        Processing gain in dB.
    """

    output: NDArray[np.floating]
    peak_index: int
    peak_value: float
    snr_gain: float


class PulseCompressionResult(NamedTuple):
    """
    Result of pulse compression.

    Attributes
    ----------
    output : ndarray
        Compressed pulse output.
    peak_index : int
        Index of the compressed pulse peak.
    compression_ratio : float
        Ratio of input pulse length to compressed pulse width.
    peak_sidelobe_ratio : float
        Peak-to-sidelobe ratio in dB.
    """

    output: NDArray[np.floating]
    peak_index: int
    compression_ratio: float
    peak_sidelobe_ratio: float


# =============================================================================
# Matched Filter Functions
# =============================================================================


def matched_filter(
    signal: ArrayLike,
    template: ArrayLike,
    normalize: bool = True,
    mode: str = "same",
) -> MatchedFilterResult:
    """
    Apply matched filtering in the time domain.

    The matched filter maximizes the output signal-to-noise ratio (SNR) when
    the input contains a known signal plus white Gaussian noise.

    Parameters
    ----------
    signal : array_like
        Input signal (may contain noise).
    template : array_like
        Template signal to match (the known waveform).
    normalize : bool, optional
        If True, normalize output by template energy. Default is True.
    mode : {'full', 'same', 'valid'}, optional
        Convolution mode. Default is 'same'.

    Returns
    -------
    result : MatchedFilterResult
        Named tuple with filter output, peak location, and SNR gain.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a signal with a pulse
    >>> template = np.array([1, 1, 1, 1, 1])
    >>> signal = np.zeros(100)
    >>> signal[50:55] = template
    >>> result = matched_filter(signal, template)
    >>> 50 <= result.peak_index <= 54
    True

    Notes
    -----
    The matched filter is the time-reversed, conjugated version of the
    template convolved with the signal. For real signals, this is equivalent
    to cross-correlation.

    The theoretical SNR gain of a matched filter is equal to the number of
    samples in the template (for unit-energy template in unit-variance noise).
    """
    signal = np.asarray(signal, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    # Matched filter is correlation with the template
    # Equivalent to convolution with time-reversed conjugate
    output = scipy_signal.correlate(signal, template, mode=mode)

    if normalize:
        template_energy = np.sum(template**2)
        if template_energy > 0:
            output = output / template_energy

    peak_index = int(np.argmax(np.abs(output)))
    peak_value = float(np.abs(output[peak_index]))

    # Theoretical SNR gain = N (number of samples in template)
    snr_gain_linear = len(template)
    snr_gain = 10 * np.log10(snr_gain_linear)

    return MatchedFilterResult(
        output=output,
        peak_index=peak_index,
        peak_value=peak_value,
        snr_gain=snr_gain,
    )


def matched_filter_frequency(
    signal: ArrayLike,
    template: ArrayLike,
    fs: float = 1.0,
    normalize: bool = True,
) -> MatchedFilterResult:
    """
    Apply matched filtering in the frequency domain.

    This implementation uses FFT for efficient computation, especially for
    long signals and templates.

    Parameters
    ----------
    signal : array_like
        Input signal.
    template : array_like
        Template signal to match.
    fs : float, optional
        Sampling frequency (used for output scaling). Default is 1.0.
    normalize : bool, optional
        If True, normalize by template energy. Default is True.

    Returns
    -------
    result : MatchedFilterResult
        Named tuple with filter output, peak location, and SNR gain.

    Examples
    --------
    >>> import numpy as np
    >>> template = np.sin(2 * np.pi * 0.1 * np.arange(50))
    >>> signal = np.zeros(200)
    >>> signal[100:150] = template
    >>> result = matched_filter_frequency(signal, template)
    >>> 100 <= result.peak_index <= 150
    True

    Notes
    -----
    For long signals, frequency-domain matched filtering is more efficient
    as it uses O(N log N) FFT operations instead of O(N^2) convolution.
    """
    signal = np.asarray(signal, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    n_signal = len(signal)
    n_template = len(template)
    n_fft = n_signal + n_template - 1

    # Use next power of 2 for efficiency
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))

    # FFT of signal and conjugate of template FFT
    signal_fft = scipy_fft.fft(signal, n=n_fft)
    template_fft = scipy_fft.fft(template, n=n_fft)

    # Matched filter in frequency domain
    output_fft = signal_fft * np.conj(template_fft)
    output = scipy_fft.ifft(output_fft).real

    # Trim to valid length
    output = output[:n_signal]

    if normalize:
        template_energy = np.sum(template**2)
        if template_energy > 0:
            output = output / template_energy

    peak_index = int(np.argmax(np.abs(output)))
    peak_value = float(np.abs(output[peak_index]))

    snr_gain_linear = len(template)
    snr_gain = 10 * np.log10(snr_gain_linear)

    return MatchedFilterResult(
        output=output,
        peak_index=peak_index,
        peak_value=peak_value,
        snr_gain=snr_gain,
    )


def optimal_filter(
    signal: ArrayLike,
    template: ArrayLike,
    noise_psd: ArrayLike,
    fs: float = 1.0,
) -> NDArray[np.floating]:
    """
    Apply optimal filtering for colored noise.

    The optimal filter (also called the Wiener filter) maximizes SNR when
    the noise is not white (colored noise). It pre-whitens the noise before
    matched filtering.

    Parameters
    ----------
    signal : array_like
        Input signal.
    template : array_like
        Template signal to match.
    noise_psd : array_like
        Noise power spectral density (must be same length as FFT).
    fs : float, optional
        Sampling frequency. Default is 1.0.

    Returns
    -------
    output : ndarray
        Optimal filter output.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.random.randn(256)
    >>> template = np.ones(16)
    >>> noise_psd = np.ones(256)  # White noise
    >>> output = optimal_filter(signal, template, noise_psd)
    >>> len(output) == len(signal)
    True

    Notes
    -----
    The optimal filter in the frequency domain is:
        H(f) = S*(f) / P_n(f)

    where S(f) is the template spectrum and P_n(f) is the noise PSD.
    """
    signal = np.asarray(signal, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)

    n_signal = len(signal)
    n_fft = len(noise_psd)

    # Ensure dimensions match
    if n_fft < n_signal:
        n_fft = n_signal

    # FFT of signal and template
    signal_fft = scipy_fft.fft(signal, n=n_fft)
    template_fft = scipy_fft.fft(template, n=n_fft)

    # Optimal filter: conj(template) / noise_psd
    # Add small regularization to avoid division by zero
    eps = 1e-10 * np.max(noise_psd)
    filter_fft = np.conj(template_fft) / (noise_psd + eps)

    # Apply filter
    output_fft = signal_fft * filter_fft
    output = scipy_fft.ifft(output_fft).real

    return output[:n_signal]


# =============================================================================
# Pulse Compression
# =============================================================================


def pulse_compression(
    signal: ArrayLike,
    reference: ArrayLike,
    window: Optional[str] = None,
) -> PulseCompressionResult:
    """
    Perform pulse compression on a signal.

    Pulse compression is used in radar to achieve the resolution of a short
    pulse while maintaining the energy of a long pulse. It is typically used
    with frequency-modulated (chirp) waveforms.

    Parameters
    ----------
    signal : array_like
        Received signal (possibly containing the transmitted waveform).
    reference : array_like
        Reference waveform (transmitted pulse, e.g., chirp).
    window : str, optional
        Window function to apply to reduce sidelobes. Options: 'hamming',
        'hann', 'blackman', 'kaiser'. Default is None (no windowing).

    Returns
    -------
    result : PulseCompressionResult
        Named tuple with compressed output, compression ratio, and sidelobes.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> chirp = generate_lfm_chirp(0.1, 50, 200, fs)  # 100 ms chirp
    >>> signal = np.zeros(2000)
    >>> signal[500:500+len(chirp)] = chirp
    >>> result = pulse_compression(signal, chirp)
    >>> result.compression_ratio > 1
    True

    Notes
    -----
    The compression ratio is approximately equal to the time-bandwidth product
    of the chirp signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    n_signal = len(signal)
    n_ref = len(reference)
    n_fft = n_signal + n_ref - 1
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))

    # Apply window to reference for sidelobe reduction
    if window is not None:
        win = scipy_signal.get_window(window, n_ref)
        reference = reference * win

    # Frequency domain correlation
    signal_fft = scipy_fft.fft(signal, n=n_fft)
    ref_fft = scipy_fft.fft(reference, n=n_fft)

    output_fft = signal_fft * np.conj(ref_fft)
    output = scipy_fft.ifft(output_fft).real
    output = output[:n_signal]

    # Normalize
    output = output / np.max(np.abs(output))

    peak_index = int(np.argmax(np.abs(output)))
    peak_value = np.abs(output[peak_index])

    # Estimate compressed pulse width (3 dB width)
    threshold = peak_value / np.sqrt(2)  # -3 dB
    above_threshold = np.abs(output) > threshold
    compressed_width = np.sum(above_threshold)

    compression_ratio = n_ref / max(compressed_width, 1)

    # Peak-to-sidelobe ratio
    # Exclude main lobe region around peak
    mainlobe_width = max(int(n_ref / 10), 5)
    sidelobe_mask = np.ones(len(output), dtype=bool)
    start = max(0, peak_index - mainlobe_width)
    end = min(len(output), peak_index + mainlobe_width + 1)
    sidelobe_mask[start:end] = False

    if np.any(sidelobe_mask):
        max_sidelobe = np.max(np.abs(output[sidelobe_mask]))
        if max_sidelobe > 0:
            pslr = 20 * np.log10(peak_value / max_sidelobe)
        else:
            pslr = np.inf
    else:
        pslr = np.inf

    return PulseCompressionResult(
        output=output,
        peak_index=peak_index,
        compression_ratio=compression_ratio,
        peak_sidelobe_ratio=pslr,
    )


# =============================================================================
# Chirp Signal Generation
# =============================================================================


def generate_lfm_chirp(
    duration: float,
    f0: float,
    f1: float,
    fs: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> NDArray[np.floating]:
    """
    Generate a linear frequency modulated (LFM) chirp signal.

    Parameters
    ----------
    duration : float
        Duration of the chirp in seconds.
    f0 : float
        Starting frequency in Hz.
    f1 : float
        Ending frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    amplitude : float, optional
        Signal amplitude. Default is 1.0.
    phase : float, optional
        Initial phase in radians. Default is 0.0.

    Returns
    -------
    chirp : ndarray
        LFM chirp signal.

    Examples
    --------
    >>> chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
    >>> len(chirp)
    44
    >>> chirp[0]  # Should start near amplitude (with phase=0)
    1.0

    Notes
    -----
    The instantaneous frequency varies linearly from f0 to f1 over the
    duration. The time-bandwidth product is (f1 - f0) * duration, which
    determines the pulse compression ratio.
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Linear frequency sweep
    # f(t) = f0 + (f1 - f0) * t / duration
    # Phase integral: phi(t) = 2*pi * (f0*t + (f1-f0)*t^2 / (2*duration))
    chirp_rate = (f1 - f0) / duration
    instantaneous_phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2) + phase

    chirp = amplitude * np.cos(instantaneous_phase)

    return chirp


def generate_nlfm_chirp(
    duration: float,
    f0: float,
    f1: float,
    fs: float,
    beta: float = 1.0,
    amplitude: float = 1.0,
) -> NDArray[np.floating]:
    """
    Generate a non-linear frequency modulated (NLFM) chirp signal.

    NLFM chirps can provide lower sidelobes than LFM chirps without
    requiring windowing, at the cost of slightly reduced SNR.

    Parameters
    ----------
    duration : float
        Duration of the chirp in seconds.
    f0 : float
        Starting frequency in Hz.
    f1 : float
        Ending frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    beta : float, optional
        Non-linearity parameter. Higher values give more non-linearity.
        Default is 1.0.
    amplitude : float, optional
        Signal amplitude. Default is 1.0.

    Returns
    -------
    chirp : ndarray
        NLFM chirp signal.

    Examples
    --------
    >>> chirp = generate_nlfm_chirp(0.001, 1000, 5000, 44100, beta=2.0)
    >>> len(chirp)
    44
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    t_norm = t / duration  # Normalized time [0, 1]

    # Frequency modulation with tanh non-linearity
    bandwidth = f1 - f0
    f_center = (f0 + f1) / 2

    # Non-linear frequency mapping using tanh
    freq_norm = np.tanh(beta * (t_norm - 0.5)) / np.tanh(beta * 0.5)
    instantaneous_freq = f_center + 0.5 * bandwidth * freq_norm

    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / fs

    chirp = amplitude * np.cos(phase)

    return chirp


# =============================================================================
# Ambiguity Function
# =============================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _ambiguity_function_kernel(
    signal: np.ndarray[Any, Any],
    delays: np.ndarray[Any, Any],
    dopplers: np.ndarray[Any, Any],
    fs: float,
    af: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled kernel for ambiguity function computation."""
    n_signal = len(signal)
    n_doppler = len(dopplers)
    n_delay = len(delays)
    t = np.arange(n_signal) / fs

    for i in prange(n_doppler):
        doppler = dopplers[i]
        # Compute Doppler-shifted signal
        shifted = np.empty(n_signal, dtype=np.complex128)
        for k in range(n_signal):
            phase = 2.0 * np.pi * doppler * t[k]
            shifted[k] = signal[k] * (np.cos(phase) + 1j * np.sin(phase))

        for j in range(n_delay):
            delay = delays[j]
            delay_samples = int(delay * fs)

            result = 0.0 + 0.0j
            if delay_samples >= 0:
                if delay_samples < n_signal:
                    for k in range(n_signal - delay_samples):
                        s1 = signal[delay_samples + k]
                        s2_conj = shifted[k].real - 1j * shifted[k].imag
                        result += s1 * s2_conj
            else:
                delay_samples = -delay_samples
                if delay_samples < n_signal:
                    for k in range(n_signal - delay_samples):
                        s1 = signal[k]
                        s2_conj = (
                            shifted[delay_samples + k].real
                            - 1j * shifted[delay_samples + k].imag
                        )
                        result += s1 * s2_conj

            af[i, j] = result


@njit(cache=True, fastmath=True, parallel=True)
def _cross_ambiguity_kernel(
    signal1: np.ndarray[Any, Any],
    signal2: np.ndarray[Any, Any],
    delays: np.ndarray[Any, Any],
    dopplers: np.ndarray[Any, Any],
    fs: float,
    caf: np.ndarray[Any, Any],
) -> None:
    """JIT-compiled kernel for cross-ambiguity function computation."""
    n_signal = len(signal1)
    n_doppler = len(dopplers)
    n_delay = len(delays)
    t = np.arange(n_signal) / fs

    for i in prange(n_doppler):
        doppler = dopplers[i]
        # Compute Doppler-shifted signal2
        shifted = np.empty(n_signal, dtype=np.complex128)
        for k in range(n_signal):
            phase = 2.0 * np.pi * doppler * t[k]
            shifted[k] = signal2[k] * (np.cos(phase) + 1j * np.sin(phase))

        for j in range(n_delay):
            delay = delays[j]
            delay_samples = int(delay * fs)

            result = 0.0 + 0.0j
            if delay_samples >= 0:
                if delay_samples < n_signal:
                    for k in range(n_signal - delay_samples):
                        s1 = signal1[delay_samples + k]
                        s2_conj = shifted[k].real - 1j * shifted[k].imag
                        result += s1 * s2_conj
            else:
                delay_samples = -delay_samples
                if delay_samples < n_signal:
                    for k in range(n_signal - delay_samples):
                        s1 = signal1[k]
                        s2_conj = (
                            shifted[delay_samples + k].real
                            - 1j * shifted[delay_samples + k].imag
                        )
                        result += s1 * s2_conj

            caf[i, j] = result


def ambiguity_function(
    signal: ArrayLike,
    fs: float,
    max_delay: Optional[float] = None,
    max_doppler: Optional[float] = None,
    n_delay: int = 256,
    n_doppler: int = 256,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.complexfloating]]:
    """
    Compute the ambiguity function of a signal.

    The ambiguity function characterizes the resolution and ambiguity
    properties of a radar waveform in delay (range) and Doppler (velocity).

    Parameters
    ----------
    signal : array_like
        Input signal (e.g., radar waveform).
    fs : float
        Sampling frequency in Hz.
    max_delay : float, optional
        Maximum delay in seconds. Default is signal duration.
    max_doppler : float, optional
        Maximum Doppler shift in Hz. Default is fs/2.
    n_delay : int, optional
        Number of delay bins. Default is 256.
    n_doppler : int, optional
        Number of Doppler bins. Default is 256.

    Returns
    -------
    delays : ndarray
        Delay values in seconds.
    dopplers : ndarray
        Doppler frequency values in Hz.
    af : ndarray
        Ambiguity function (2D, complex).

    Examples
    --------
    >>> chirp = generate_lfm_chirp(0.001, 1000, 5000, 44100)
    >>> delays, dopplers, af = ambiguity_function(chirp, 44100)
    >>> af.shape
    (256, 256)
    """
    signal = np.asarray(signal, dtype=np.complex128)
    n_signal = len(signal)
    duration = n_signal / fs

    if max_delay is None:
        max_delay = duration
    if max_doppler is None:
        max_doppler = fs / 2

    delays = np.linspace(-max_delay, max_delay, n_delay)
    dopplers = np.linspace(-max_doppler, max_doppler, n_doppler)

    af = np.zeros((n_doppler, n_delay), dtype=np.complex128)

    # Use JIT-compiled kernel for performance (with parallel execution)
    _ambiguity_function_kernel(signal, delays, dopplers, fs, af)

    # Normalize
    af = np.abs(af) / np.max(np.abs(af))

    return delays, dopplers, af


def cross_ambiguity(
    signal1: ArrayLike,
    signal2: ArrayLike,
    fs: float,
    max_delay: Optional[float] = None,
    max_doppler: Optional[float] = None,
    n_delay: int = 256,
    n_doppler: int = 256,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.complexfloating]]:
    """
    Compute the cross-ambiguity function between two signals.

    Parameters
    ----------
    signal1 : array_like
        First signal.
    signal2 : array_like
        Second signal.
    fs : float
        Sampling frequency in Hz.
    max_delay : float, optional
        Maximum delay in seconds.
    max_doppler : float, optional
        Maximum Doppler shift in Hz.
    n_delay : int, optional
        Number of delay bins.
    n_doppler : int, optional
        Number of Doppler bins.

    Returns
    -------
    delays : ndarray
        Delay values in seconds.
    dopplers : ndarray
        Doppler frequency values in Hz.
    caf : ndarray
        Cross-ambiguity function (2D, complex).
    """
    signal1 = np.asarray(signal1, dtype=np.complex128)
    signal2 = np.asarray(signal2, dtype=np.complex128)

    # Ensure same length
    n_signal = min(len(signal1), len(signal2))
    signal1 = signal1[:n_signal]
    signal2 = signal2[:n_signal]

    duration = n_signal / fs

    if max_delay is None:
        max_delay = duration
    if max_doppler is None:
        max_doppler = fs / 2

    delays = np.linspace(-max_delay, max_delay, n_delay)
    dopplers = np.linspace(-max_doppler, max_doppler, n_doppler)

    caf = np.zeros((n_doppler, n_delay), dtype=np.complex128)

    # Use JIT-compiled kernel for performance (with parallel execution)
    _cross_ambiguity_kernel(signal1, signal2, delays, dopplers, fs, caf)

    # Normalize
    max_val = np.max(np.abs(caf))
    if max_val > 0:
        caf = caf / max_val

    return delays, dopplers, np.abs(caf)
