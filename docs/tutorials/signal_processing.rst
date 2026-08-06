Signal Processing Tutorial
==========================

This tutorial covers digital filtering, spectral analysis, and wavelet
transforms for signal processing applications.

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/tutorials/signal_processing.html"></iframe>
   </div>

Digital Filter Design
---------------------

Design and apply IIR and FIR filters for signal conditioning.

Butterworth Lowpass Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import (
       butter_design, apply_filter, frequency_response
   )

   # Sample rate and cutoff
   fs = 1000  # Hz
   cutoff = 50  # Hz

   # Design 4th order Butterworth lowpass
   coeffs = butter_design(order=4, cutoff=cutoff, fs=fs, btype='low')

   # Generate test signal: 20 Hz + 100 Hz components
   t = np.linspace(0, 1, fs)
   signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)

   # Apply filter
   filtered = apply_filter(coeffs, signal)

   # The 100 Hz component is attenuated

Zero-Phase Filtering
^^^^^^^^^^^^^^^^^^^^

For offline processing, use zero-phase filtering to avoid phase distortion:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import filtfilt

   # Forward-backward filtering (zero phase)
   filtered_zp = filtfilt(coeffs, signal)

Frequency Response Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize filter characteristics:

.. code-block:: python

   # Get frequency response
   resp = frequency_response(coeffs, fs, n_points=512)

   # resp.frequencies: frequency axis (Hz)
   # resp.magnitude: magnitude response
   # resp.phase: phase response (radians)

   # -3 dB point (cutoff)
   cutoff_idx = np.argmin(np.abs(resp.magnitude - 0.707))
   print(f"Cutoff frequency: {resp.frequencies[cutoff_idx]:.1f} Hz")

Spectral Analysis
-----------------

Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions.transforms import power_spectrum

   # Generate noisy signal with two tones
   fs = 1000
   t = np.arange(0, 2, 1/fs)
   signal = (np.sin(2 * np.pi * 50 * t) +
             0.5 * np.sin(2 * np.pi * 120 * t) +
             0.2 * np.random.randn(len(t)))

   # Compute power spectrum
   psd = power_spectrum(signal, fs, window='hann', nperseg=256)

   # psd.frequencies: frequency axis (Hz)
   # psd.psd: power spectral density

   # Find the two strongest spectral peaks
   from scipy.signal import find_peaks

   peaks, _ = find_peaks(psd.psd)
   strongest = peaks[np.argsort(psd.psd[peaks])[-2:]]
   peak_freqs = np.sort(psd.frequencies[strongest])
   print(f"Detected frequencies: {peak_freqs}")

Short-Time Fourier Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For time-frequency analysis of non-stationary signals:

.. code-block:: python

   from pytcl.mathematical_functions.transforms import stft, spectrogram

   # Chirp signal (frequency sweep)
   fs = 1000
   t = np.linspace(0, 2, 2 * fs)
   f0, f1 = 10, 200
   signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 4) * t)

   # Compute STFT
   result = stft(signal, fs, window='hann', nperseg=128)

   # result.frequencies: frequency bins
   # result.times: time centers
   # result.Zxx: complex STFT coefficients

   # Or get power spectrogram directly
   spec = spectrogram(signal, fs, nperseg=128)
   # spec.power: |STFT|^2

Wavelet Transforms
------------------

Continuous Wavelet Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions.transforms import cwt

   # Signal with transient
   fs = 1000
   t = np.linspace(0, 1, fs)
   signal = np.sin(2 * np.pi * 10 * t)
   signal[400:450] += 2 * np.sin(2 * np.pi * 50 * t[400:450])

   # CWT with Morlet wavelet
   scales = np.arange(1, 128)
   result = cwt(signal, scales, wavelet='morlet', fs=fs)

   # result.coefficients: CWT coefficients
   # result.scales: scale values
   # result.frequencies: pseudo-frequencies

Discrete Wavelet Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-resolution analysis:

.. code-block:: python

   from pytcl.mathematical_functions.transforms import dwt, idwt

   # Decompose signal
   result = dwt(signal, wavelet='db4', level=4)

   # result.cA: approximation coefficients (low frequency)
   # result.cD: list of detail coefficients per level

   # Reconstruct
   reconstructed = idwt(result)

Matched Filtering
-----------------

Detect known waveforms in noisy signals:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import matched_filter

   # Known pulse template
   template = np.sin(2 * np.pi * 0.1 * np.arange(50))

   # Signal with pulse at unknown location
   signal = np.random.randn(500) * 0.5
   signal[200:250] += template  # Embed pulse

   # Apply matched filter
   result = matched_filter(signal, template, normalize=True)

   # result.output: filter output
   # result.peak_index: detected location
   # result.snr_gain: processing gain in dB

   print(f"Detected pulse at index {result.peak_index}")
   print(f"SNR gain: {result.snr_gain:.1f} dB")

Complete Example: Radar Signal Processing
-----------------------------------------

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import (
       butter_design, apply_filter, matched_filter, cfar_ca,
       cluster_detections
   )

   # Simulate radar return with targets
   fs = 10000  # Sample rate
   n_samples = 2000
   np.random.seed(42)

   # Transmit pulse: 1 kHz tone burst with decaying envelope (5 ms)
   t_pulse = np.arange(50) / fs
   pulse = np.sin(2 * np.pi * 1000 * t_pulse) * np.exp(-t_pulse / 0.002)

   # Noise floor
   signal = np.random.randn(n_samples) * 0.3

   # Add target echoes at different ranges
   target_ranges = [300, 800, 1200]
   for r in target_ranges:
       signal[r:r+50] += 2.0 * pulse

   # 1. Bandpass filter to reduce out-of-band noise
   coeffs = butter_design(order=4, cutoff=(100, 2000), fs=fs, btype='band')
   filtered = apply_filter(coeffs, signal)

   # 2. Matched filter for pulse compression
   mf_result = matched_filter(filtered, pulse)

   # 3. CFAR detection on the compressed power
   power = np.abs(mf_result.output) ** 2
   cfar = cfar_ca(
       power,
       guard_cells=25,
       ref_cells=100,
       pfa=1e-6
   )

   # 4. Cluster adjacent detection cells into targets
   targets = cluster_detections(cfar.detections, min_separation=50)

   print(f"Detected {len(targets)} targets")
   print(f"Detection peaks at: {targets}")

Next Steps
----------

- See :doc:`radar_detection` for more CFAR algorithms
- Explore :doc:`/api/signal_processing` for complete API reference
- Try :doc:`/api/transforms` for additional transform functions
