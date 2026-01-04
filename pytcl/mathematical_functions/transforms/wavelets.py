"""
Wavelet transform utilities.

This module provides continuous and discrete wavelet transform functions,
wavelet generation, and time-frequency analysis tools.

Functions
---------
- cwt: Continuous Wavelet Transform
- dwt: Discrete Wavelet Transform (requires pywavelets)
- idwt: Inverse Discrete Wavelet Transform
- morlet_wavelet: Generate Morlet wavelet
- ricker_wavelet: Generate Ricker (Mexican hat) wavelet
- scales_to_frequencies: Convert wavelet scales to frequencies

References
----------
.. [1] Mallat, S. (2008). A Wavelet Tour of Signal Processing: The Sparse
       Way (3rd ed.). Academic Press.
.. [2] Daubechies, I. (1992). Ten Lectures on Wavelets. SIAM.
"""

from typing import Any, Callable, List, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import is_available

# Use the unified availability check
PYWT_AVAILABLE = is_available("pywt")

# Import pywavelets if available (for use in functions)
if PYWT_AVAILABLE:
    import pywt


# =============================================================================
# Result Types
# =============================================================================


class CWTResult(NamedTuple):
    """
    Result of Continuous Wavelet Transform.

    Attributes
    ----------
    coefficients : ndarray
        CWT coefficient matrix (complex), shape (n_scales, n_samples).
    scales : ndarray
        Scale values used.
    frequencies : ndarray
        Approximate frequencies corresponding to each scale.
    """

    coefficients: NDArray[np.complexfloating]
    scales: NDArray[np.floating]
    frequencies: NDArray[np.floating]


class DWTResult(NamedTuple):
    """
    Result of Discrete Wavelet Transform.

    Attributes
    ----------
    cA : ndarray
        Approximation coefficients at the coarsest level.
    cD : list of ndarray
        Detail coefficients at each level (finest to coarsest).
    levels : int
        Number of decomposition levels.
    wavelet : str
        Wavelet name used.
    """

    cA: NDArray[np.floating]
    cD: List[NDArray[np.floating]]
    levels: int
    wavelet: str


# =============================================================================
# Wavelet Functions
# =============================================================================


def morlet_wavelet(
    M: int,
    w: float = 5.0,
    s: float = 1.0,
    complete: bool = True,
) -> NDArray[np.complexfloating]:
    """
    Generate a Morlet wavelet.

    The Morlet wavelet is a sinusoid windowed by a Gaussian, commonly used
    for time-frequency analysis.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0, the central frequency parameter. Default is 5.0.
    s : float, optional
        Scaling factor. Default is 1.0.
    complete : bool, optional
        If True, use the complete Morlet wavelet with correction term.
        Default is True.

    Returns
    -------
    wavelet : ndarray
        Complex Morlet wavelet.

    Examples
    --------
    >>> wav = morlet_wavelet(128, w=5.0)
    >>> len(wav)
    128
    >>> np.abs(wav[len(wav)//2]) > 0  # Peak in center
    True

    Notes
    -----
    The Morlet wavelet is defined as:
        psi(t) = exp(i*w*t) * exp(-t^2/2)

    With the complete correction:
        psi(t) = (exp(i*w*t) - exp(-w^2/2)) * exp(-t^2/2)
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s

    # Gaussian envelope
    gauss = np.exp(-0.5 * x**2)

    # Complex sinusoid
    sinusoid = np.exp(1j * w * x)

    if complete:
        # Correction term to ensure zero mean
        correction = np.exp(-0.5 * w**2)
        wavelet = (sinusoid - correction) * gauss
    else:
        wavelet = sinusoid * gauss

    # Normalize
    wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))

    return wavelet


def ricker_wavelet(
    points: int,
    a: float = 1.0,
) -> NDArray[np.floating]:
    """
    Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the negative normalized second derivative of a
    Gaussian function. It is real-valued and commonly used in seismology.

    Parameters
    ----------
    points : int
        Number of points in the wavelet.
    a : float, optional
        Width parameter. Default is 1.0.

    Returns
    -------
    wavelet : ndarray
        Ricker wavelet.

    Examples
    --------
    >>> wav = ricker_wavelet(128, a=4.0)
    >>> len(wav)
    128
    >>> wav[len(wav)//2]  # Peak at center
    1.0

    Notes
    -----
    The Ricker wavelet is defined as:
        psi(t) = (1 - 2*(pi*f*t)^2) * exp(-(pi*f*t)^2)

    where f = 1/(sqrt(2)*pi*a) is the central frequency.
    """
    # Native implementation of the Ricker (Mexican hat) wavelet
    # This avoids dependency on scipy.signal.ricker which was removed in some scipy versions
    t = np.arange(points, dtype=np.float64) - (points - 1.0) / 2
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = (t / a) ** 2
    mod = 1 - wsq
    gauss = np.exp(-wsq / 2)
    return A * mod * gauss


def gaussian_wavelet(
    M: int,
    order: int = 1,
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """
    Generate a Gaussian derivative wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    order : int, optional
        Order of the derivative. Default is 1.
    sigma : float, optional
        Standard deviation of the Gaussian. Default is 1.0.

    Returns
    -------
    wavelet : ndarray
        Gaussian derivative wavelet.

    Examples
    --------
    >>> wav = gaussian_wavelet(128, order=1)
    >>> len(wav)
    128
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / sigma

    # Gaussian
    gauss = np.exp(-0.5 * x**2)

    if order == 1:
        wavelet = -x * gauss
    elif order == 2:
        wavelet = (x**2 - 1) * gauss
    elif order == 3:
        wavelet = (3 * x - x**3) * gauss
    elif order == 4:
        wavelet = (3 - 6 * x**2 + x**4) * gauss
    else:
        # General case using Hermite polynomials
        from scipy.special import hermite

        Hn = hermite(order)
        wavelet = ((-1) ** order) * Hn(x) * gauss

    # Normalize
    wavelet = wavelet / np.sqrt(np.sum(wavelet**2))

    return wavelet


# =============================================================================
# Continuous Wavelet Transform
# =============================================================================


def cwt(
    signal: ArrayLike,
    scales: ArrayLike,
    wavelet: Union[str, Callable[[int], NDArray[np.floating]]] = "morlet",
    fs: float = 1.0,
    method: str = "fft",
) -> CWTResult:
    """
    Compute the Continuous Wavelet Transform.

    Parameters
    ----------
    signal : array_like
        Input signal.
    scales : array_like
        Scale values to use.
    wavelet : str or callable, optional
        Wavelet to use. Options: 'morlet', 'ricker', 'gaussian1', 'gaussian2',
        or a callable. Default is 'morlet'.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    method : {'fft', 'conv'}, optional
        Computation method. 'fft' is faster for long signals.
        Default is 'fft'.

    Returns
    -------
    result : CWTResult
        Named tuple with coefficients, scales, and frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t)
    >>> scales = np.arange(1, 128)
    >>> result = cwt(x, scales, wavelet='morlet', fs=fs)
    >>> result.coefficients.shape
    (127, 1000)

    Notes
    -----
    The CWT is computed as:
        W(a, b) = integral s(t) * (1/sqrt(a)) * psi*((t-b)/a) dt

    where a is the scale, b is the translation, and psi is the wavelet.
    """
    signal = np.asarray(signal, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    n = len(signal)

    # Determine wavelet function
    def _morlet_default(M: int) -> NDArray[np.floating]:
        return morlet_wavelet(M, w=5.0)

    def _ricker_default(M: int) -> NDArray[np.floating]:
        return ricker_wavelet(M, a=1.0)

    def _gaussian1_default(M: int) -> NDArray[np.floating]:
        return gaussian_wavelet(M, order=1)

    def _gaussian2_default(M: int) -> NDArray[np.floating]:
        return gaussian_wavelet(M, order=2)

    if callable(wavelet):
        wavelet_func = wavelet
        wavelet_name = "custom"
    elif wavelet == "morlet":
        wavelet_func = _morlet_default
        wavelet_name = "morlet"
    elif wavelet == "ricker":
        wavelet_func = _ricker_default
        wavelet_name = "ricker"
    elif wavelet == "gaussian1":
        wavelet_func = _gaussian1_default
        wavelet_name = "gaussian1"
    elif wavelet == "gaussian2":
        wavelet_func = _gaussian2_default
        wavelet_name = "gaussian2"
    else:
        raise ValueError(f"Unknown wavelet: {wavelet}")

    # Compute CWT
    n_scales = len(scales)
    coefficients = np.zeros((n_scales, n), dtype=np.complex128)

    for i, scale in enumerate(scales):
        # Generate scaled wavelet
        wavelet_length = min(10 * int(scale) + 1, n)
        if wavelet_length < 3:
            wavelet_length = 3

        psi = wavelet_func(wavelet_length)

        # Normalize by sqrt(scale)
        psi = psi / np.sqrt(scale)

        if method == "fft":
            # FFT-based convolution
            n_fft = n + wavelet_length - 1
            n_fft = int(2 ** np.ceil(np.log2(n_fft)))

            from scipy import fft as scipy_fft

            signal_fft = scipy_fft.fft(signal, n=n_fft)
            psi_fft = scipy_fft.fft(np.conj(psi[::-1]), n=n_fft)
            conv = scipy_fft.ifft(signal_fft * psi_fft)
            # Center the result
            start = (wavelet_length - 1) // 2
            coefficients[i, :] = conv[start : start + n]
        else:
            # Direct convolution
            conv = np.convolve(signal, np.conj(psi[::-1]), mode="same")
            coefficients[i, :] = conv

    # Compute approximate frequencies
    frequencies = scales_to_frequencies(scales, wavelet_name, fs)

    return CWTResult(
        coefficients=coefficients,
        scales=scales,
        frequencies=frequencies,
    )


def scales_to_frequencies(
    scales: ArrayLike,
    wavelet: str = "morlet",
    fs: float = 1.0,
) -> NDArray[np.floating]:
    """
    Convert CWT scales to approximate frequencies.

    Parameters
    ----------
    scales : array_like
        Scale values.
    wavelet : str, optional
        Wavelet name. Default is 'morlet'.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    frequencies : ndarray
        Approximate frequencies in Hz.

    Examples
    --------
    >>> scales = np.array([1, 2, 4, 8, 16])
    >>> freqs = scales_to_frequencies(scales, wavelet='morlet', fs=1000)
    >>> len(freqs)
    5
    >>> freqs[0] > freqs[-1]  # Smaller scale = higher frequency
    True
    """
    scales = np.asarray(scales, dtype=np.float64)

    # Center frequency depends on wavelet
    if wavelet == "morlet":
        # For Morlet with w=5.0
        center_freq = 5.0 / (2 * np.pi)
    elif wavelet == "ricker":
        center_freq = 1.0 / (np.sqrt(2) * np.pi)
    elif wavelet == "gaussian1":
        center_freq = 0.5
    elif wavelet == "gaussian2":
        center_freq = 0.5
    else:
        center_freq = 1.0

    frequencies = center_freq * fs / scales

    return frequencies


def frequencies_to_scales(
    frequencies: ArrayLike,
    wavelet: str = "morlet",
    fs: float = 1.0,
) -> NDArray[np.floating]:
    """
    Convert desired frequencies to CWT scales.

    Parameters
    ----------
    frequencies : array_like
        Desired frequencies in Hz.
    wavelet : str, optional
        Wavelet name. Default is 'morlet'.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    scales : ndarray
        Scale values.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)

    if wavelet == "morlet":
        center_freq = 5.0 / (2 * np.pi)
    elif wavelet == "ricker":
        center_freq = 1.0 / (np.sqrt(2) * np.pi)
    else:
        center_freq = 1.0

    scales = center_freq * fs / frequencies

    return scales


# =============================================================================
# Discrete Wavelet Transform
# =============================================================================


def dwt(
    signal: ArrayLike,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "symmetric",
) -> DWTResult:
    """
    Compute the Discrete Wavelet Transform.

    The DWT decomposes a signal into approximation and detail coefficients
    at multiple resolution levels.

    Parameters
    ----------
    signal : array_like
        Input signal.
    wavelet : str, optional
        Wavelet to use (e.g., 'db4', 'haar', 'sym8', 'coif3').
        Default is 'db4'.
    level : int, optional
        Decomposition level. Default is max level for signal length.
    mode : str, optional
        Signal extension mode. Default is 'symmetric'.

    Returns
    -------
    result : DWTResult
        Named tuple with approximation and detail coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(256)
    >>> result = dwt(x, wavelet='db4', level=4)
    >>> len(result.cD)  # 4 levels of detail
    4

    Notes
    -----
    Requires the pywavelets package. Install with: pip install pywavelets

    Common wavelet families:
    - 'haar': Simplest wavelet
    - 'dbN': Daubechies wavelets (N=1..38)
    - 'symN': Symlets (N=2..20)
    - 'coifN': Coiflets (N=1..17)
    - 'biorN.M': Biorthogonal wavelets
    """
    if not PYWT_AVAILABLE:
        raise ImportError(
            "pywavelets is required for DWT. Install with: pip install pywavelets"
        )

    signal = np.asarray(signal, dtype=np.float64)

    if level is None:
        level = pywt.dwt_max_level(len(signal), wavelet)

    # Perform decomposition
    coeffs = pywt.wavedec(signal, wavelet, mode=mode, level=level)

    # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
    cA = coeffs[0]
    cD = coeffs[1:][::-1]  # Reverse to get finest-to-coarsest order

    return DWTResult(
        cA=cA,
        cD=cD,
        levels=level,
        wavelet=wavelet,
    )


def idwt(
    coeffs: DWTResult,
    mode: str = "symmetric",
) -> NDArray[np.floating]:
    """
    Compute the inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs : DWTResult
        DWT coefficients from dwt function.
    mode : str, optional
        Signal extension mode. Default is 'symmetric'.

    Returns
    -------
    signal : ndarray
        Reconstructed signal.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(256)
    >>> result = dwt(x, wavelet='db4', level=4)
    >>> x_rec = idwt(result)
    >>> np.allclose(x, x_rec)
    True
    """
    if not PYWT_AVAILABLE:
        raise ImportError(
            "pywavelets is required for IDWT. Install with: pip install pywavelets"
        )

    # Reconstruct coeffs list in pywt format
    # [cA_n, cD_n, cD_n-1, ..., cD_1]
    pywt_coeffs = [coeffs.cA] + coeffs.cD[::-1]

    signal = pywt.waverec(pywt_coeffs, coeffs.wavelet, mode=mode)

    return signal


def dwt_single_level(
    signal: ArrayLike,
    wavelet: str = "db4",
    mode: str = "symmetric",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute single-level DWT decomposition.

    Parameters
    ----------
    signal : array_like
        Input signal.
    wavelet : str, optional
        Wavelet name. Default is 'db4'.
    mode : str, optional
        Signal extension mode. Default is 'symmetric'.

    Returns
    -------
    cA : ndarray
        Approximation coefficients.
    cD : ndarray
        Detail coefficients.
    """
    if not PYWT_AVAILABLE:
        raise ImportError(
            "pywavelets is required for DWT. Install with: pip install pywavelets"
        )

    signal = np.asarray(signal, dtype=np.float64)
    cA, cD = pywt.dwt(signal, wavelet, mode=mode)

    return cA, cD


def idwt_single_level(
    cA: ArrayLike,
    cD: ArrayLike,
    wavelet: str = "db4",
    mode: str = "symmetric",
) -> NDArray[np.floating]:
    """
    Compute single-level inverse DWT.

    Parameters
    ----------
    cA : array_like
        Approximation coefficients.
    cD : array_like
        Detail coefficients.
    wavelet : str, optional
        Wavelet name. Default is 'db4'.
    mode : str, optional
        Signal extension mode. Default is 'symmetric'.

    Returns
    -------
    signal : ndarray
        Reconstructed signal.
    """
    if not PYWT_AVAILABLE:
        raise ImportError(
            "pywavelets is required for IDWT. Install with: pip install pywavelets"
        )

    cA = np.asarray(cA, dtype=np.float64)
    cD = np.asarray(cD, dtype=np.float64)

    return pywt.idwt(cA, cD, wavelet, mode=mode)


# =============================================================================
# Wavelet Packet Transform
# =============================================================================


def wpt(
    signal: ArrayLike,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "symmetric",
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the Wavelet Packet Transform.

    The WPT provides a more flexible time-frequency decomposition than DWT
    by also decomposing the detail coefficients.

    Parameters
    ----------
    signal : array_like
        Input signal.
    wavelet : str, optional
        Wavelet name. Default is 'db4'.
    level : int, optional
        Decomposition level. Default is 3.
    mode : str, optional
        Signal extension mode. Default is 'symmetric'.

    Returns
    -------
    nodes : dict
        Dictionary mapping node paths to coefficients.
        Path format: 'a' for approximation, 'd' for detail.
        Example: 'aad' means approx->approx->detail.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(256)
    >>> nodes = wpt(x, wavelet='db4', level=2)
    >>> 'aa' in nodes  # Level 2 approximation
    True
    >>> 'dd' in nodes  # Level 2 detail of detail
    True
    """
    if not PYWT_AVAILABLE:
        raise ImportError(
            "pywavelets is required for WPT. Install with: pip install pywavelets"
        )

    signal = np.asarray(signal, dtype=np.float64)

    if level is None:
        level = 3

    wp = pywt.WaveletPacket(signal, wavelet, mode=mode, maxlevel=level)

    # Collect all nodes
    nodes = {}
    for node in wp.get_level(level, "natural"):
        nodes[node.path] = np.array(node.data)

    return nodes


# =============================================================================
# Utility Functions
# =============================================================================


def available_wavelets() -> List[str]:
    """
    List available wavelet families for DWT.

    Returns
    -------
    wavelets : list of str
        Available wavelet names.
    """
    if not PYWT_AVAILABLE:
        return ["morlet", "ricker", "gaussian1", "gaussian2"]

    return pywt.wavelist()


def wavelet_info(wavelet: str) -> dict[str, Any]:
    """
    Get information about a wavelet.

    Parameters
    ----------
    wavelet : str
        Wavelet name.

    Returns
    -------
    info : dict
        Dictionary with wavelet properties.
    """
    if not PYWT_AVAILABLE:
        if wavelet == "morlet":
            return {
                "name": "morlet",
                "family": "complex",
                "orthogonal": False,
                "biorthogonal": False,
            }
        elif wavelet == "ricker":
            return {
                "name": "ricker",
                "family": "real",
                "orthogonal": False,
                "biorthogonal": False,
            }
        else:
            return {"name": wavelet}

    w = pywt.Wavelet(wavelet)

    return {
        "name": w.name,
        "family": w.family_name,
        "orthogonal": w.orthogonal,
        "biorthogonal": w.biorthogonal,
        "symmetry": w.symmetry,
        "filter_length": w.dec_len,
        "vanishing_moments_psi": getattr(w, "vanishing_moments_psi", None),
        "vanishing_moments_phi": getattr(w, "vanishing_moments_phi", None),
    }


def threshold_coefficients(
    coeffs: DWTResult,
    threshold: Union[float, str] = "soft",
    value: Optional[float] = None,
) -> DWTResult:
    """
    Threshold DWT coefficients for denoising.

    Parameters
    ----------
    coeffs : DWTResult
        DWT coefficients.
    threshold : float or {'soft', 'hard'}, optional
        Threshold type or value. Default is 'soft'.
    value : float, optional
        Threshold value. If None, uses universal threshold.

    Returns
    -------
    result : DWTResult
        Thresholded coefficients.
    """
    if not PYWT_AVAILABLE:
        raise ImportError(
            "pywavelets is required. Install with: pip install pywavelets"
        )

    # Estimate noise from finest detail coefficients
    if value is None:
        sigma = np.median(np.abs(coeffs.cD[0])) / 0.6745
        n = sum(len(d) for d in coeffs.cD) + len(coeffs.cA)
        value = sigma * np.sqrt(2 * np.log(n))

    # Apply threshold
    if threshold == "soft":
        mode = "soft"
    elif threshold == "hard":
        mode = "hard"
    else:
        mode = "soft"
        value = threshold

    cA_thresh = pywt.threshold(coeffs.cA, value, mode=mode)
    cD_thresh = [pywt.threshold(d, value, mode=mode) for d in coeffs.cD]

    return DWTResult(
        cA=cA_thresh,
        cD=cD_thresh,
        levels=coeffs.levels,
        wavelet=coeffs.wavelet,
    )
