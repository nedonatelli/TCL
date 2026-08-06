Signal Processing Fundamentals
==============================

Overview
--------

The Tracker Component Library provides **radar detection, signal filtering, FFT, and wavelet** functionality for processing sensor data.

**Key Modules:**

- ``pytcl.mathematical_functions.signal_processing.detection`` - CFAR detection (Constant False Alarm Rate)
- ``pytcl.mathematical_functions.signal_processing.filters`` - FIR/IIR digital filter design and application
- ``pytcl.mathematical_functions.signal_processing.matched_filter`` - Matched filtering, pulse compression, LFM chirps
- ``pytcl.mathematical_functions.transforms`` - FFT, STFT/spectrogram, wavelets

The wavelet functions (CWT/DWT) are backed by PyWavelets and require the
``signal`` extra (``pip install nrl-tracker[signal]``).

CFAR Detection (Constant False Alarm Rate)
-------------------------------------------

**Why CFAR?**

Raw radar returns contain both signal and noise. CFAR adaptively sets the detection threshold based on the local clutter level to maintain a constant false alarm probability.

.. code-block:: text

   Power
   ^          _
   |         | |        ___ CFAR threshold
   |     ....| |....----    (adapts to local noise)
   | ....    | |    ....
   |_________|_|_________> Range cell

**1D CFAR** (along range/time axis)

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import cfar_ca

   np.random.seed(0)

   # Simulated radar power profile: exponential noise floor plus one target
   radar_return = np.random.exponential(1.0, 1000)
   radar_return[500:505] += 30.0

   result = cfar_ca(
       radar_return,
       guard_cells=4,    # guard band on each side of the test cell
       ref_cells=16,     # training cells on each side
       pfa=1e-6,         # probability of false alarm
   )

   print(f"Detections at indices: {result.detection_indices}")

Output::

   Detections at indices: [500 501 502 503 504]

``cfar_ca`` returns a ``CFARResult`` named tuple with fields
``detections`` (boolean mask), ``threshold`` (per-cell threshold),
``detection_indices``, and ``noise_estimate``.

**CFAR Variants**

Each variant is its own function; all share the
``(signal, guard_cells, ref_cells, pfa)`` interface:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import (
       cfar_go,
       cfar_os,
       cfar_so,
   )

   # OS-CFAR (order statistics): robust when multiple targets fall in the
   # reference window; k selects which order statistic to use
   os_result = cfar_os(radar_return, guard_cells=4, ref_cells=16, pfa=1e-6, k=24)

   # GO-CFAR (greatest-of): conservative at clutter edges (fewer false alarms)
   go_result = cfar_go(radar_return, guard_cells=4, ref_cells=16, pfa=1e-6)

   # SO-CFAR (smallest-of): better detection of closely spaced targets,
   # at the price of more false alarms at clutter edges
   so_result = cfar_so(radar_return, guard_cells=4, ref_cells=16, pfa=1e-6)

**2D CFAR** (range x Doppler)

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import cfar_2d

   np.random.seed(1)

   # Simulated range-Doppler map, dimensions [n_range, n_doppler]
   range_doppler = np.random.exponential(1.0, (256, 128))
   range_doppler[100:104, 50:53] += 25.0  # target

   result_2d = cfar_2d(
       range_doppler,
       guard_cells=(2, 2),
       ref_cells=(8, 8),
       pfa=1e-6,
       method="ca",      # 'ca', 'go', or 'so'
   )

   rows, cols = np.nonzero(result_2d.detections)
   print(f"{result_2d.detections.sum()} detections "
         f"around range cell {rows.mean():.0f}, Doppler cell {cols.mean():.0f}")

Output::

   12 detections around range cell 102, Doppler cell 51

**Detection Probability and CFAR Loss**

Closed-form performance curves for CA-CFAR against a Swerling 1
(exponentially fluctuating) target -- only the Swerling 1 model is
implemented:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import (
       detection_probability,
       snr_loss,
       threshold_factor,
   )

   # detection_probability takes linear SNR, not dB
   for snr_db in (8, 10, 13):
       snr = 10 ** (snr_db / 10)
       pd = detection_probability(snr, pfa=1e-6, n_ref=32)
       print(f"SNR {snr_db:2d} dB -> Pd = {pd:.3f}")

   # CFAR loss: extra SNR needed versus an ideal fixed-threshold detector
   loss_db = snr_loss(n_ref=32, pfa=1e-6, pd=0.5)
   print(f"CA-CFAR loss with 32 reference cells: {loss_db:.2f} dB")

   # The threshold multiplier applied to the noise estimate
   alpha = threshold_factor(pfa=1e-6, n_ref=32)
   print(f"Threshold multiplier: {alpha:.2f}")

Output::

   SNR  8 dB -> Pd = 0.102
   SNR 10 dB -> Pd = 0.216
   SNR 13 dB -> Pd = 0.443
   CA-CFAR loss with 32 reference cells: 0.92 dB
   Threshold multiplier: 17.28

Matched Filtering
-----------------

**Why Matched Filters?**

Maximize signal-to-noise ratio (SNR) for a known signal shape. Optimal for white Gaussian noise.

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import (
       generate_lfm_chirp,
       matched_filter,
   )

   np.random.seed(2)

   # Known transmit waveform: LFM chirp, 20 us, 0-2 MHz sweep at 10 MHz sampling
   fs = 10e6
   duration = 20e-6
   chirp = generate_lfm_chirp(duration, f0=0.0, f1=2e6, fs=fs)

   # Received signal: attenuated echo delayed by 60 samples, in noise
   delay = 60
   received = np.zeros(1000)
   received[delay:delay + len(chirp)] += 0.5 * chirp
   received += 0.1 * np.random.randn(1000)

   result = matched_filter(received, chirp, mode="same")

   # With mode="same" the peak sits at delay + len(template) // 2
   est_delay = result.peak_index - len(chirp) // 2
   est_range = est_delay / fs * 3e8 / 2  # c/2 for the round trip

   print(f"SNR gain: {result.snr_gain:.1f} dB")
   print(f"Estimated delay: {est_delay} samples -> range {est_range:.0f} m")

Output::

   SNR gain: 20.2 dB
   Estimated delay: 60 samples -> range 900 m

**Pulse Compression** (matched filtering for radar pulses)

``pulse_compression`` wraps the matched filter with optional sidelobe
weighting and reports the peak directly in delay samples:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import pulse_compression

   pc = pulse_compression(received, chirp, window="hamming")

   print(f"Peak at delay sample {pc.peak_index}")
   print(f"Compression ratio: {pc.compression_ratio:.0f}")
   print(f"Peak sidelobe ratio: {pc.peak_sidelobe_ratio:.1f} dB")

Output::

   Peak at delay sample 60
   Compression ratio: 67
   Peak sidelobe ratio: 22.8 dB

Why compress? A long pulse carries more energy, but the range resolution of
an uncompressed pulse is set by its duration; after compression it is set by
the bandwidth:

.. code-block:: python

   pulse_duration = 10e-6  # 10 us
   bandwidth = 100e6       # 100 MHz
   c = 3e8

   res_uncompressed = c * pulse_duration / 2
   res_compressed = c / (2 * bandwidth)

   print(f"Uncompressed resolution: {res_uncompressed:.1f} m")
   print(f"Compressed resolution: {res_compressed:.2f} m")
   print(f"Improvement factor: {pulse_duration * bandwidth:.0f}x")

Output::

   Uncompressed resolution: 1500.0 m
   Compressed resolution: 1.50 m
   Improvement factor: 1000x

Digital Filtering
-----------------

**FIR (Finite Impulse Response) Filter Design**

``fir_design`` is a windowed-sinc design; give it the number of taps and
the cutoff frequency:

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import (
       apply_filter,
       fir_design,
   )

   fs = 1000  # sampling frequency (Hz)

   b = fir_design(
       numtaps=101,      # filter length
       cutoff=100.0,     # cutoff frequency (Hz)
       fs=fs,
       window="hamming",
       pass_zero=True,   # low-pass
   )
   print(f"FIR coefficient length: {len(b)}")

   # Generate test signal: 50 Hz + 200 Hz components
   t = np.arange(0, 1, 1 / fs)
   signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

   # Apply the filter (a bare coefficient array is treated as FIR)
   filtered = apply_filter(b, signal)

   print(f"RMS before: {signal.std():.3f}, after 100 Hz low-pass: {filtered.std():.3f}")

Output::

   FIR coefficient length: 101
   RMS before: 0.791, after 100 Hz low-pass: 0.691

For equiripple designs, use ``fir_design_remez``.

**IIR (Infinite Impulse Response) Filter Design**

``butter_design`` returns a ``FilterCoefficients`` named tuple
(fields ``b``, ``a``, ``sos``); second-order sections are used by default
for numerical stability. Chebyshev (``cheby1_design``, ``cheby2_design``),
elliptic (``ellip_design``), and Bessel (``bessel_design``) designs share
the same interface.

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import (
       butter_design,
       filtfilt,
   )

   coeffs = butter_design(
       order=4,
       cutoff=100.0,     # Hz
       fs=fs,
       btype="low",
   )
   print(f"Second-order sections: {coeffs.sos.shape}")

   filtered_iir = apply_filter(coeffs, signal)   # causal filtering
   zero_phase = filtfilt(coeffs, signal)         # zero-phase (offline)

Output::

   Second-order sections: (2, 6)

**FIR vs IIR Comparison**

.. code-block:: text

   Property              FIR              IIR

   Stability            Always stable    Can be unstable (careful design)
   Order                Higher           Lower (for same stopband atten.)
   Phase                Linear phase     Nonlinear phase
   Group Delay          Constant         Varying
   Real-time            More samples     Fewer computations
   Numerical            Better           Prone to roundoff errors
   Implementation       Simple           More complex

   Use FIR for:      Linear phase, robustness, when order is not critical
   Use IIR for:      Low computation cost, tight resource constraints

FFT and Spectral Analysis
--------------------------

**FFT for Radar Processing**

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.transforms import fft

   X = fft(signal)
   magnitude = np.abs(X)

   freqs = np.fft.fftfreq(len(signal), 1 / fs)
   peak = np.argmax(magnitude[: len(signal) // 2])
   print(f"Dominant frequency: {freqs[peak]:.0f} Hz")

Output::

   Dominant frequency: 50 Hz

**Doppler Processing** (FFT along the pulse dimension)

.. code-block:: python

   def doppler_processing(range_time_matrix, prf):
       """
       Form a range-Doppler map from radar returns.

       Args:
           range_time_matrix: (n_range, n_pulses) complex radar data
           prf: Pulse repetition frequency (Hz)

       Returns:
           magnitude: (n_range, n_doppler) magnitude spectrum
           doppler_freqs: Doppler frequency of each bin (Hz)
       """
       range_doppler = fft(range_time_matrix, axis=1)
       magnitude = np.abs(range_doppler)

       n_doppler = range_time_matrix.shape[1]
       doppler_freqs = np.fft.fftfreq(n_doppler, 1 / prf)

       return magnitude, doppler_freqs

**Time-Frequency Analysis** (when target Doppler changes with time)

``spectrogram`` returns a named tuple with ``frequencies``, ``times``, and
``power``:

.. code-block:: python

   from pytcl.mathematical_functions.transforms import spectrogram

   # Signal with changing frequency (like a moving target):
   # frequency sweeps from 10 to 50 Hz
   t = np.linspace(0, 1, 1000)
   freq_t = 10 + 40 * t
   sweep = np.sin(2 * np.pi * np.cumsum(freq_t) / 1000)

   spec = spectrogram(
       sweep,
       fs=1000,
       nperseg=256,   # window length
       noverlap=128,  # overlap
   )

   print(f"{len(spec.frequencies)} frequency bins x {len(spec.times)} time frames")

Output::

   129 frequency bins x 6 time frames

For the complex STFT (and its inverse), use ``stft`` / ``istft`` from the
same module.

Wavelets (Time-Frequency Analysis)
-----------------------------------

**Wavelet Transform** for analyzing non-stationary signals. Requires the
``signal`` extra (PyWavelets); without it these functions raise
``DependencyError``.

.. code-block:: python

   from pytcl.mathematical_functions.transforms import cwt, dwt

   # Signal: sum of two transients
   t = np.linspace(0, 1, 1000)
   transient1 = np.exp(-20 * (t - 0.2) ** 2) * np.sin(2 * np.pi * 100 * t)
   transient2 = np.exp(-20 * (t - 0.8) ** 2) * np.sin(2 * np.pi * 200 * t)
   sig = transient1 + transient2

   # Continuous wavelet transform
   scales = np.arange(1, 64)
   result = cwt(sig, scales, wavelet="morlet", fs=1000)

   print(f"Coefficients: {result.coefficients.shape}")
   print(f"Frequencies: {result.frequencies.min():.1f} - "
         f"{result.frequencies.max():.1f} Hz")

   # Discrete wavelet transform (multi-level decomposition)
   decomp = dwt(sig, wavelet="db4", level=4)
   print(f"DWT levels: {decomp.levels}, "
         f"approximation length: {len(decomp.cA)}")

Output::

   Coefficients: (63, 1000)
   Frequencies: 12.6 - 795.8 Hz
   DWT levels: 4, approximation length: 69

Signal Detection Workflow
-------------------------

**Complete Radar Detection Pipeline**

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import cfar_2d

   class RadarProcessor:
       """End-to-end radar signal processing."""

       def __init__(self, fs, prf, fc=10e9, pfa=1e-5, c=3e8):
           self.fs = fs      # range sampling rate
           self.prf = prf    # pulse repetition frequency
           self.fc = fc      # carrier frequency
           self.pfa = pfa
           self.c = c

       def process_frame(self, raw_iq_data):
           """
           Process one radar frame.

           Args:
               raw_iq_data: (n_range, n_pulses) complex I/Q samples,
                   already pulse-compressed

           Returns:
               List of detected targets
           """
           n_pulses = raw_iq_data.shape[1]

           # Step 1: Doppler processing (FFT across pulses)
           range_doppler = np.fft.fft(raw_iq_data, axis=1)
           magnitude = np.abs(range_doppler) ** 2

           # Step 2: CFAR detection
           result = cfar_2d(
               magnitude,
               guard_cells=(2, 2),
               ref_cells=(8, 8),
               pfa=self.pfa,
           )

           # Step 3: Extract target parameters
           targets = []
           for range_idx, doppler_idx in np.argwhere(result.detections):
               range_m = range_idx * self.c / (2 * self.fs)

               doppler_freqs = np.fft.fftfreq(n_pulses, 1 / self.prf)
               doppler_freq = doppler_freqs[doppler_idx]
               velocity = doppler_freq * self.c / (2 * self.fc)

               snr = magnitude[range_idx, doppler_idx] / result.threshold[range_idx, doppler_idx]

               targets.append({
                   "range": range_m,
                   "velocity": velocity,
                   "snr": 10 * np.log10(snr),
                   "range_cell": int(range_idx),
                   "doppler_cell": int(doppler_idx),
               })

           return targets

   # Synthetic frame: noise plus one target at range cell 40
   np.random.seed(3)
   frame = (np.random.randn(128, 64) + 1j * np.random.randn(128, 64)) / np.sqrt(2)
   frame[40, :] += 2.0 * np.exp(2j * np.pi * 0.2 * np.arange(64))

   proc = RadarProcessor(fs=1e6, prf=1e3)
   targets = proc.process_frame(frame)
   best = max(targets, key=lambda tt: tt["snr"])
   print(f"{len(targets)} detection(s); strongest at range {best['range']:.0f} m, "
         f"velocity {best['velocity']:.1f} m/s")

Output::

   2 detection(s); strongest at range 6000 m, velocity 3.0 m/s

To feed detections into tracking, see the trackers in ``pytcl.trackers``
and :doc:`recipes`.

Multi-Channel Processing
------------------------

**Beamforming** (coherent combination of multiple antenna elements)

.. code-block:: python

   def phased_array_beamform(signal_matrix, angles, spacing_wavelengths=0.5):
       """
       Conventional beamforming (delay-and-sum) for a uniform linear array.

       Args:
           signal_matrix: (n_channels, n_samples) received signals
           angles: array of steering angles (radians)
           spacing_wavelengths: element spacing in wavelengths

       Returns:
           beam_output: (n_angles, n_samples) beamformed signals
       """
       n_channels = signal_matrix.shape[0]
       beam_output = []

       for angle in angles:
           # Phase shifts for each channel to steer the beam
           phase_shifts = (
               2 * np.pi * spacing_wavelengths * np.arange(n_channels) * np.sin(angle)
           )
           weights = np.exp(1j * phase_shifts)

           beam = weights @ signal_matrix
           beam_output.append(beam)

       return np.array(beam_output)

Range Resolution and Ambiguity
-------------------------------

**Range Resolution** = c * tau / 2, where tau is the (compressed) pulse duration

.. code-block:: python

   # Higher resolution requires shorter pulses.
   # But shorter pulses have lower average power and worse SNR;
   # pulse compression (long pulse + matched filter) is the standard fix.

   pulse_durations = [1e-6, 10e-6, 100e-6]  # 1, 10, 100 us
   c = 3e8

   for tau in pulse_durations:
       resolution = c * tau / 2
       print(f"tau={tau*1e6:.0f} us -> resolution = {resolution:.1f} m")

Output::

   tau=1 us -> resolution = 150.0 m
   tau=10 us -> resolution = 1500.0 m
   tau=100 us -> resolution = 15000.0 m

**Doppler Ambiguity** (max unambiguous Doppler)

.. code-block:: python

   # Max unambiguous Doppler velocity: v_max = PRF * lambda / 4

   prf = 10e3               # 10 kHz
   wavelength = 3e8 / 10e9  # 3 cm for 10 GHz carrier

   v_max = prf * wavelength / 4
   print(f"Max unambiguous velocity: +/- {v_max:.1f} m/s")

Output::

   Max unambiguous velocity: +/- 75.0 m/s

Performance Considerations
--------------------------

**Computation Cost**

.. code-block:: text

   Operation               Complexity      Speed

   1D CFAR                O(N)            Fast ~ms
   2D CFAR                O(N*M)          Medium ~10ms
   Matched filter         O(N^2) -> O(N log N) with FFT
   FFT (N samples)        O(N log N)      Fast
   Butterworth IIR        O(N)            Fastest
   FIR filter             O(N * L)        Medium (L = taps)
   Wavelets               O(N log N)      Medium

**Streaming Processing**

For continuous data, buffer samples and run CFAR block-by-block:

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import cfar_ca

   class StreamingCFAR:
       """Accumulate streaming samples and run CFAR on each full block."""

       def __init__(self, block_size=512, guard_cells=4, ref_cells=16, pfa=1e-6):
           self.block_size = block_size
           self.guard_cells = guard_cells
           self.ref_cells = ref_cells
           self.pfa = pfa
           self.buffer = []

       def process_sample(self, sample):
           """Add one sample; returns detection indices when a block completes."""
           self.buffer.append(abs(sample) ** 2)
           if len(self.buffer) < self.block_size:
               return None

           result = cfar_ca(
               np.array(self.buffer),
               guard_cells=self.guard_cells,
               ref_cells=self.ref_cells,
               pfa=self.pfa,
           )
           self.buffer = []
           return result.detection_indices

   np.random.seed(6)
   stream = StreamingCFAR(block_size=512)
   samples = np.random.exponential(1.0, 512)
   samples[300] += 40.0

   for i, s in enumerate(samples):
       hits = stream.process_sample(s)
       if hits is not None:
           print(f"Block complete at sample {i}: detections at {hits}")

Output::

   Block complete at sample 511: detections at [300]

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`api_navigation` - Finding signal processing functions
- :doc:`performance_optimization` - Optimization techniques
- ``examples/signal_processing.py`` - Signal processing examples
