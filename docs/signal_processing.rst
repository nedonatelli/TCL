Signal Processing Fundamentals
==============================

Overview
--------

The Tracker Component Library provides **radar detection, signal filtering, FFT, and wavelet** functionality for processing sensor data.

**Key Modules:**

- ``signal_processing.detection`` - CFAR algorithm (Constant False Alarm Rate)
- ``signal_processing.filtering`` - FIR/IIR digital filters
- ``signal_processing.optimal`` - Matched filtering, likelihood ratio tests
- ``signal_processing.transforms`` - FFT, wavelets, spectral analysis

CFAR Detection (Constant False Alarm Rate)
-------------------------------------------

**Why CFAR?**

Raw radar returns contain both signal and noise. CFAR adaptively sets detection threshold based on local clutter level to maintain constant false alarm probability.

.. code-block:: text

   Raw Signal
   ^
   |     ┌─┐
   |     │ │  ┌─ Detection threshold
   |   ┌─┘ └──┐ (adaptive)
   | ┌─┘      └─┐
   |_┴─────────┴─────→ Range
   
   Green: Signal + Noise
   Red line: CFAR adaptive threshold

**1D CFAR** (along range/time axis)

.. code-block:: python

   from pytcl.signal_processing.detection import cfar_1d
   import numpy as np
   
   # Simulated radar return
   signal = np.zeros(1000)
   signal[500:510] = 10.0  # Target return
   noise = np.random.randn(1000) * 1.0
   radar_return = signal + noise
   
   # CFAR detection
   detections, threshold = cfar_1d(
       radar_return,
       window_size=32,      # Training window size
       guard_size=4,        # Guard band around test cell
       pfa=1e-6,            # Probability of false alarm
       method='cagoesulien'  # Algorithm variant
   )
   
   # Detections is binary array (1 = detection, 0 = no detection)
   detection_indices = np.where(detections)[0]
   print(f"Detections at indices: {detection_indices}")
   
   # Plot
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12, 4))
   plt.plot(radar_return, label='Signal+Noise', alpha=0.7)
   plt.plot(threshold, label='CFAR Threshold', color='red')
   plt.scatter(detection_indices, radar_return[detection_indices], 
               color='green', s=50, label='Detections')
   plt.legend()
   plt.show()

**CFAR Methods**

Different algorithms for different clutter types:

.. code-block:: python

   # CA-CFAR (Cell Averaging): Simple, works well in uniform clutter
   detections, th = cfar_1d(radar_data, method='cell_averaging')
   
   # OS-CFAR (Order Statistics): Robust to multiple targets
   detections, th = cfar_1d(radar_data, method='order_statistics', rank=4)
   
   # SO-CFAR (Smallest Of): Conservative (few false alarms)
   detections, th = cfar_1d(radar_data, method='smallest_of')
   
   # GO-CFAR (Greatest Of): Aggressive (more detections)
   detections, th = cfar_1d(radar_data, method='greatest_of')

**2D CFAR** (range × Doppler)

.. code-block:: python

   from pytcl.signal_processing.detection import cfar_2d
   
   # Simulated range-Doppler map
   # Dimensions: [n_range, n_doppler]
   range_doppler = np.random.randn(256, 128) * 1.0  # Background clutter
   
   # Add target
   range_doppler[100:110, 50:55] += 5.0
   
   # 2D CFAR
   detections_2d, threshold_2d = cfar_2d(
       range_doppler,
       window_size=(32, 32),
       guard_size=(4, 4),
       pfa=1e-6
   )
   
   # Plot
   plt.figure(figsize=(10, 8))
   plt.subplot(2, 1, 1)
   plt.imshow(range_doppler, aspect='auto', origin='lower', cmap='viridis')
   plt.title('Range-Doppler Map')
   plt.colorbar()
   
   plt.subplot(2, 1, 2)
   plt.imshow(detections_2d, aspect='auto', origin='lower', cmap='RdYl')
   plt.title('CFAR Detections')
   plt.colorbar()
   plt.show()

**Probability of Detection vs False Alarm**

.. code-block:: python

   from pytcl.signal_processing.detection import cfar_1d
   
   def compute_roc_curve(signal_snr_db, noise_std, pfa_values):
       """Receiver Operating Characteristic curve"""
       
       # Create test signal
       signal = np.zeros(1000)
       signal[500:510] = 10 ** (signal_snr_db / 20) * noise_std  # Target at SNR
       
       detections_per_pfa = []
       
       for pfa in pfa_values:
           radar_return = signal + np.random.randn(1000) * noise_std
           detections, _ = cfar_1d(radar_return, pfa=pfa)
           pd = np.sum(detections[500:510] == 1) / 10  # Prob of detection
           detections_per_pfa.append(pd)
       
       return detections_per_pfa
   
   # Compute ROC
   snr_db = 6  # 6 dB SNR
   pfa_vals = np.logspace(-8, -3, 10)
   pd_list = compute_roc_curve(snr_db, 1.0, pfa_vals)
   
   plt.semilogx(pfa_vals, pd_list, 'o-', label=f'SNR={snr_db} dB')
   plt.xlabel('Probability of False Alarm')
   plt.ylabel('Probability of Detection')
   plt.grid(True)
   plt.legend()
   plt.show()

Matched Filtering
-----------------

**Why Matched Filters?**

Maximize signal-to-noise ratio (SNR) for known signal shape. Optimal for white Gaussian noise.

.. code-block:: python

   from pytcl.signal_processing.optimal import matched_filter
   import numpy as np
   
   # Known transmit signal (e.g., chirp pulse)
   duration = 1e-6  # 1 microsecond
   fs = 1e6  # 1 MHz sampling
   t = np.arange(0, duration, 1/fs)
   
   # Chirp signal (frequency sweeps 1-10 MHz)
   f_start = 1e6
   f_end = 10e6
   known_signal = np.exp(2j * np.pi * (f_start + (f_end-f_start) * t / duration) * t)
   
   # Received signal = transmitted + target return + noise
   delay = 2e-7  # 2-way propagation delay
   target_return = 0.1 * np.roll(known_signal, int(delay * fs))  # Weak return
   noise = (np.random.randn(len(known_signal)) + 1j * np.random.randn(len(known_signal))) * 0.01
   
   received = target_return + noise
   
   # Apply matched filter
   filtered, peaks = matched_filter(received, known_signal)
   
   print(f"SNR improvement: {20*np.log10(np.abs(peaks).max()):.1f} dB")
   
   # Detect range
   detection_index = np.argmax(np.abs(filtered))
   estimated_distance = detection_index / fs * 3e8 / 2  # c/2 for round trip
   
   print(f"Estimated range: {estimated_distance:.1f} m")

**Pulse Compression** (matched filtering for radar pulses)

.. code-block:: python

   def pulse_compression_example():
       """Demonstrate range resolution improvement via pulse compression"""
       
       # LFM chirp: linear frequency modulation (common in modern radars)
       from pytcl.signal_processing.optimal import matched_filter
       
       # Parameters
       pulse_duration = 10e-6  # 10 μs
       bandwidth = 100e6       # 100 MHz
       fs = 1e9               # 1 GHz sampling
       c = 3e8                # Speed of light
       
       # Uncompressed range resolution
       tau_uncompressed = pulse_duration
       res_uncompressed = c * tau_uncompressed / 2
       
       # Compressed range resolution (determined by bandwidth)
       res_compressed = c / (2 * bandwidth)
       
       print(f"Uncompressed resolution: {res_uncompressed*100:.1f} m")
       print(f"Compressed resolution: {res_compressed*100:.3f} m")
       print(f"Improvement factor: {tau_uncompressed / (1/bandwidth):.0f}x")

Digital Filtering
-----------------

**FIR (Finite Impulse Response) Filter Design**

.. code-block:: python

   from pytcl.signal_processing.filtering import design_fir_filter
   
   # Design a low-pass FIR filter
   fsample = 1000  # Sampling frequency (Hz)
   fpass = 100     # Passband edge (Hz)
   fstop = 150     # Stopband edge (Hz)
   gpass = 1       # Passband ripple (dB)
   gstop = 60      # Stopband attenuation (dB)
   
   b = design_fir_filter(
       order=100,          # Filter order (number of taps)
       frequencies=[fpass, fstop],
       gains=[1, 0],
       fs=fsample
   )
   
   print(f"FIR coefficient length: {len(b)}")
   
   # Apply filter to data
   from pytcl.signal_processing.filtering import apply_filter
   
   # Generate test signal: 50 Hz + 200 Hz components
   t = np.linspace(0, 1, fsample)
   signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*200*t)
   
   # Filter
   filtered = apply_filter(b, signal)
   
   # Plot
   plt.figure(figsize=(12, 4))
   plt.plot(t, signal, label='Original (50+200 Hz)', alpha=0.7)
   plt.plot(t, filtered, label='Filtered (50 Hz LPF)', linewidth=2)
   plt.xlim([0, 0.1])
   plt.legend()
   plt.grid(True)
   plt.show()

**IIR (Infinite Impulse Response) Filter Design**

.. code-block:: python

   from pytcl.signal_processing.filtering import (
       design_butterworth, apply_filter_iir
   )
   
   # Butterworth filter (maximally flat response)
   fsample = 1000
   fpass = 100   # Hz
   order = 4
   
   b, a = design_butterworth(
       order=order,
       cutoff_frequency=fpass,
       fs=fsample,
       filter_type='lowpass'
   )
   
   print(f"Numerator coefficients: {b}")
   print(f"Denominator coefficients: {a}")
   
   # Apply IIR filter
   signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*200*t)
   filtered = apply_filter_iir(b, a, signal)

**FIR vs IIR Comparison**

.. code-block:: text

   Property              FIR              IIR
   
   Stability            Always stable    Can be unstable (careful design)
   Order                Higher           Lower (for same stopband atten.)
   Delay                Linear phase     Nonlinear phase
   Group Delay          Constant         Varying
   Real-time            More samples     Fewer computations
   Numerical            Better           Prone to roundoff errors
   Implementation       Simple           More complex
   
   Use FIR for:      Low latency, linear phase, when order not critical
   Use IIR for:      Low computation cost, tight resource constraints
```

FFT and Spectral Analysis
--------------------------

**FFT for Radar Processing**

.. code-block:: python

   from pytcl.signal_processing.transforms import fft_1d, fft_2d
   import numpy as np
   
   # 1D FFT: Time to frequency domain
   signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t) + noise
   
   X = fft_1d(signal)
   magnitude = np.abs(X)
   
   # Plot magnitude spectrum
   freqs = np.linspace(0, fsample/2, len(X)//2)
   plt.plot(freqs, magnitude[:len(X)//2])
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Magnitude')
   plt.show()

**Doppler Processing** (FFT along time dimension)

.. code-block:: python

   def doppler_processing(range_time_matrix, prf, c=3e8):
       """
       Extract Doppler velocity from radar returns
       
       Args:
           range_time_matrix: (n_range, n_pulses) complex radar data
           prf: Pulse repetition frequency (Hz)
           c: Speed of light
       
       Returns:
           range_doppler: (n_range, n_doppler) magnitude spectrum
       """
       # FFT along time dimension (pulses)
       range_doppler = fft_1d(range_time_matrix, axis=1)
       
       # Get magnitude
       magnitude = np.abs(range_doppler)
       
       # Create velocity axis
       n_doppler = range_doppler.shape[1]
       doppler_freqs = np.fft.fftfreq(n_doppler, 1/prf)
       
       # Convert to velocity
       # v = (f_doppler * c) / (2 * fc)
       # For simplicity, use proportional to Doppler frequency
       
       return magnitude

**Time-Frequency Analysis** (when target Doppler changes with time)

.. code-block:: python

   from pytcl.signal_processing.transforms import spectrogram
   
   # Signal with changing frequency (like moving target)
   t = np.linspace(0, 1, 1000)
   # Frequency increases from 10 to 50 Hz
   freq_t = 10 + 40 * t
   signal = np.sin(2*np.pi*np.cumsum(freq_t)/1000)
   
   # Spectrogram (FFT in windows)
   T, F, Sxx = spectrogram(
       signal,
       fs=1000,
       nperseg=256,  # Window length
       noverlap=128   # Overlap
   )
   
   # Plot
   plt.pcolormesh(T, F, 10*np.log10(Sxx), shading='auto')
   plt.ylabel('Frequency (Hz)')
   plt.xlabel('Time (s)')
   plt.colorbar(label='Power (dB)')
   plt.show()

Wavelets (Time-Frequency Analysis)
-----------------------------------

**Wavelet Transform** for analyzing non-stationary signals

.. code-block:: python

   from pytcl.signal_processing.transforms import continuous_wavelet_transform
   
   # Signal: sum of two transients
   t = np.linspace(0, 1, 1000)
   transient1 = np.exp(-20*(t-0.2)**2) * np.sin(2*np.pi*100*t)
   transient2 = np.exp(-20*(t-0.8)**2) * np.sin(2*np.pi*200*t)
   signal = transient1 + transient2
   
   # Continuous wavelet transform
   scales = np.arange(1, 128)
   coefficients, frequencies = continuous_wavelet_transform(
       signal,
       scales,
       mother_wavelet='morlet',
       fs=1000
   )
   
   # Plot scalogram
   plt.contourf(t, frequencies, np.abs(coefficients), 64)
   plt.colorbar(label='|Coefficient|')
   plt.ylabel('Frequency')
   plt.xlabel('Time')
   plt.title('Scalogram')
   plt.show()

Signal Detection Workflow
-------------------------

**Complete Radar Detection Pipeline**

.. code-block:: python

   class RadarProcessor:
       """End-to-end radar signal processing"""
       
       def __init__(self, fs, prf, c=3e8):
           self.fs = fs
           self.prf = prf
           self.c = c
           self.ca_alarm_rate = 1e-5
       
       def process_frame(self, raw_iq_data):
           """
           Process one radar frame
           
           Args:
               raw_iq_data: (n_range, n_pulses) complex I/Q samples
           
           Returns:
               detections: List of detected targets
           """
           from pytcl.signal_processing.detection import cfar_2d
           
           # Step 1: Pulse compression (matched filtering)
           # (compute reference signal from transmit waveform)
           # compressed = matched_filter_all_ranges(raw_iq_data)
           
           # Step 2: Doppler processing (FFT)
           range_doppler = np.fft.fft(raw_iq_data, axis=1)
           magnitude = np.abs(range_doppler)
           
           # Step 3: CFAR detection
           detections_2d, threshold = cfar_2d(
               magnitude,
               window_size=(8, 8),
               guard_size=(2, 2),
               pfa=self.ca_alarm_rate
           )
           
           # Step 4: Extract target parameters
           detection_cells = np.argwhere(detections_2d)
           
           targets = []
           for range_idx, doppler_idx in detection_cells:
               # Range computation
               range_m = range_idx * self.c / (2 * self.fs)
               
               # Doppler frequency
               doppler_idx_shifted = doppler_idx if doppler_idx < magnitude.shape[1]/2 else doppler_idx - magnitude.shape[1]
               doppler_freq = doppler_idx_shifted * self.prf / magnitude.shape[1]
               
               # Relative velocity
               velocity = doppler_freq * self.c / (2 * 10e9)  # Assumes 10 GHz carrier
               
               # Signal-to-noise ratio
               snr = magnitude[range_idx, doppler_idx] / threshold[range_idx, doppler_idx]
               
               targets.append({
                   'range': range_m,
                   'velocity': velocity,
                   'snr': 10*np.log10(snr),
                   'range_cell': range_idx,
                   'doppler_cell': doppler_idx
               })
           
           return targets
       
       def track_detections(self, detections_sequence):
           """Track detections across frames"""
           # Simple association: nearest neighbor
           tracks = []
           
           for detections in detections_sequence:
               # Associate with existing tracks
               # Update or create new track
               pass
           
           return tracks

Multi-Channel Processing
------------------------

**Beamforming** (coherent combination of multiple antenna elements)

.. code-block:: python

   def phased_array_beamform(signal_matrix, angles):
       """
       Conventional beamforming (delay-and-sum)
       
       Args:
           signal_matrix: (n_channels, n_samples) received signals
           angles: Array of angles to beamform toward
       
       Returns:
           beam_output: Beamformed signals at each angle
       """
       n_channels = signal_matrix.shape[0]
       beam_output = []
       
       for angle in angles:
           # Phase shifts for each channel to steer beam
           # For uniform linear array: phase[ch] = 2π * ch * sin(angle) / wavelength
           
           phase_shifts = 2*np.pi * np.arange(n_channels) * np.sin(angle) / 1.0
           weights = np.exp(1j * phase_shifts)
           
           # Beamform: weighted sum across channels
           beam = np.dot(weights, signal_matrix)
           beam_output.append(beam)
       
       return np.array(beam_output)

Range Resolution and Ambiguity
-------------------------------

**Range Resolution** = c × τ / 2, where τ is pulse duration

.. code-block:: python

   # Higher resolution requires shorter pulses
   # But shorter pulses have:
   # - Lower average power → worse SNR
   # - Pulse compression solution: long pulse + matched filter
   
   pulse_durations = [1e-6, 10e-6, 100e-6]  # 1, 10, 100 μs
   c = 3e8
   
   for tau in pulse_durations:
       resolution = c * tau / 2
       print(f"τ={tau*1e6:.0f}μs → Resolution = {resolution:.1f}m")

**Doppler Ambiguity** (max unambiguous Doppler)

.. code-block:: python

   # Max unambiguous Doppler velocity
   # v_max = PRF * λ / 4
   
   prf = 10e3      # 10 kHz
   wavelength = 3e8 / 10e9  # 3 cm for 10 GHz carrier
   
   v_max = prf * wavelength / 4
   print(f"Max unambiguous velocity: ±{v_max:.1f} m/s")

Performance Considerations
--------------------------

**Computation Cost**

.. code-block:: text

   Operation               Complexity      Speed
   
   1D CFAR                O(N)            Fast ~ms
   2D CFAR                O(N×M)          Medium ~10ms
   Matched filter         O(N²) →O(N)     Fast with FFT
   FFT (N samples)         O(N log N)      Fast
   Butterworth IIR        O(N)            Fastest
   FIR filter             O(N × L)        Medium (L=order)
   Wavelets               O(N log N)      Medium

**Streaming Processing** (on-line)

.. code-block:: python

   class StreamingCFARDetector:
       """CFAR detector for continuous streaming data"""
       
       def __init__(self, window_size=32, guard_size=4):
           self.buffer = np.zeros(window_size + 2*guard_size)
           self.window_size = window_size
           self.guard_size = guard_size
       
       def process_sample(self, sample):
           """Process one sample, return detection"""
           # Shift buffer
           self.buffer = np.roll(self.buffer, -1)
           self.buffer[-1] = np.abs(sample)
           
           # Compute local mean (training window)
           mean_value = (np.sum(self.buffer[:self.guard_size]) + 
                        np.sum(self.buffer[-self.guard_size:])) / (2*self.guard_size)
           
           # Test center sample
           threshold = mean_value * 10  # Adjust threshold factor for desired PFA
           detection = self.buffer[self.guard_size + self.window_size//2] > threshold
           
           return detection

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`api_navigation` - Finding signal processing functions
- :doc:`performance_optimization` - Optimization techniques
- ``examples/signal_processing.py`` - Signal processing examples
