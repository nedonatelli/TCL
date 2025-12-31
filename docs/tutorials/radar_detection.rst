Radar Detection Tutorial
========================

This tutorial covers Constant False Alarm Rate (CFAR) detection algorithms
for radar signal processing.

CFAR Detection Overview
-----------------------

CFAR detectors maintain a constant false alarm probability by adapting
the detection threshold to local noise conditions. This is essential
for radar systems operating in non-uniform clutter environments.

Cell-Averaging CFAR
-------------------

The most common CFAR algorithm averages reference cells to estimate
the noise floor.

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import cfar_ca

   # Simulated radar range profile
   np.random.seed(42)
   n_cells = 500

   # Background noise (exponential for Swerling 0/1 targets)
   signal = np.random.exponential(scale=1.0, size=n_cells)

   # Add targets
   signal[100] = 15.0  # Strong target
   signal[250] = 8.0   # Medium target
   signal[400] = 5.0   # Weak target

   # CA-CFAR detection
   result = cfar_ca(
       signal,
       guard_cells=3,    # Guard cells on each side
       ref_cells=10,     # Reference cells on each side
       pfa=1e-4          # Probability of false alarm
   )

   # result.detections: boolean mask
   # result.threshold: adaptive threshold
   # result.detection_indices: indices of detections

   print(f"Detected targets at: {result.detection_indices}")

CFAR Parameters
^^^^^^^^^^^^^^^

- **guard_cells**: Cells adjacent to cell under test (CUT) that are excluded
  from noise estimation. Prevents target energy from biasing threshold.
- **ref_cells**: Number of cells on each side used to estimate noise level.
- **pfa**: Desired probability of false alarm.

Greatest-Of CFAR (GO-CFAR)
--------------------------

GO-CFAR uses the maximum of leading and lagging averages. Better for
clutter edges but higher detection loss.

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import cfar_go

   result = cfar_go(signal, guard_cells=3, ref_cells=10, pfa=1e-4)

Smallest-Of CFAR (SO-CFAR)
--------------------------

SO-CFAR uses the minimum of leading and lagging averages. Better detection
in clutter but higher false alarms at edges.

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import cfar_so

   result = cfar_so(signal, guard_cells=3, ref_cells=10, pfa=1e-4)

Order-Statistic CFAR (OS-CFAR)
------------------------------

OS-CFAR selects the k-th smallest value from reference cells. Robust
to interfering targets in the reference window.

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import cfar_os

   # Use 3/4 order statistic (robust to one interferer)
   k = int(0.75 * 20)  # 75% of total reference cells
   result = cfar_os(signal, guard_cells=3, ref_cells=10, pfa=1e-4, k=k)

2D CFAR Detection
-----------------

For range-Doppler or image-based detection:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import cfar_2d

   # Simulated range-Doppler map
   n_range = 100
   n_doppler = 64

   # Noise floor
   rdm = np.random.exponential(scale=1.0, size=(n_range, n_doppler))

   # Add targets
   rdm[30, 20] = 20.0   # Target 1
   rdm[70, 45] = 15.0   # Target 2

   # 2D CA-CFAR
   result = cfar_2d(
       rdm,
       guard=(2, 2),     # Guard cells (range, Doppler)
       ref=(5, 5),       # Reference cells
       pfa=1e-5,
       method='ca'
   )

   # Find detection coordinates
   det_range, det_doppler = np.where(result.detections)
   print(f"Detections at (range, Doppler):")
   for r, d in zip(det_range, det_doppler):
       print(f"  ({r}, {d})")

Threshold Factor Calculation
----------------------------

Compute the CFAR threshold multiplier for a given Pfa:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import threshold_factor

   # For CA-CFAR with 20 reference cells
   alpha = threshold_factor(pfa=1e-4, n_ref=20, method='ca')
   print(f"Threshold factor: {alpha:.2f}")

   # Threshold = alpha * noise_estimate

Detection Probability
---------------------

Compute probability of detection for given SNR:

.. code-block:: python

   from pytcl.mathematical_functions.signal_processing import detection_probability

   # Detection probability vs SNR
   snr_db = np.linspace(0, 20, 50)
   snr_linear = 10 ** (snr_db / 10)

   pd = [detection_probability(s, pfa=1e-4, n_ref=20, method='ca')
         for s in snr_linear]

   # Find required SNR for Pd = 0.9
   snr_90 = snr_db[np.argmin(np.abs(np.array(pd) - 0.9))]
   print(f"Required SNR for 90% detection: {snr_90:.1f} dB")

Complete Example: Radar Processing Chain
----------------------------------------

.. code-block:: python

   import numpy as np
   from pytcl.mathematical_functions.signal_processing import (
       matched_filter, cfar_ca
   )
   from pytcl.mathematical_functions.transforms import fft

   # Radar parameters
   fs = 1e6           # Sample rate (1 MHz)
   pri = 1e-3         # Pulse repetition interval
   n_pulses = 32      # Number of pulses for Doppler processing
   n_range = 1000     # Range bins

   np.random.seed(42)

   # Transmit waveform (LFM chirp)
   bw = 5e6           # Bandwidth
   t_pulse = 10e-6    # Pulse duration
   t = np.arange(0, t_pulse, 1/fs)
   chirp = np.exp(1j * np.pi * bw / t_pulse * t**2)

   # Simulate received data (n_range x n_pulses)
   # Noise
   rx_data = (np.random.randn(n_range, n_pulses) +
              1j * np.random.randn(n_range, n_pulses)) * 0.1

   # Add targets with range and Doppler
   targets = [
       {'range': 200, 'doppler': 5},   # Moving target
       {'range': 500, 'doppler': -3},  # Approaching target
       {'range': 750, 'doppler': 0},   # Stationary target
   ]

   for tgt in targets:
       r_idx = tgt['range']
       doppler_phase = np.exp(1j * 2 * np.pi * tgt['doppler'] *
                              np.arange(n_pulses) / n_pulses)
       rx_data[r_idx:r_idx+len(chirp), :] += (
           5.0 * np.outer(chirp, doppler_phase)[:n_range-r_idx, :]
       )

   # Step 1: Pulse compression (matched filtering per pulse)
   compressed = np.zeros((n_range, n_pulses), dtype=complex)
   for p in range(n_pulses):
       mf = matched_filter(rx_data[:, p], chirp)
       compressed[:len(mf.output), p] = mf.output[:n_range]

   # Step 2: Doppler processing (FFT across pulses)
   rdm = np.abs(fft(compressed, axis=1))

   # Step 3: CFAR detection on range-Doppler map
   from pytcl.mathematical_functions.signal_processing import cfar_2d

   detections = cfar_2d(rdm, guard=(2, 2), ref=(5, 3), pfa=1e-6, method='ca')

   det_range, det_doppler = np.where(detections.detections)
   print(f"Detected {len(det_range)} targets:")
   for r, d in zip(det_range, det_doppler):
       print(f"  Range bin {r}, Doppler bin {d}")

Next Steps
----------

- See :doc:`signal_processing` for filter design and spectral analysis
- Explore :doc:`/api/signal_processing` for the complete API reference
