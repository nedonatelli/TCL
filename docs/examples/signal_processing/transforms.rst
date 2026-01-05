Transforms
==========

This example demonstrates FFT, power spectrum, wavelets, and other transforms.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/transforms_fft.html"></iframe>
   </div>

Overview
--------

Transform methods convert signals between domains:

- **Fourier Transform**: Time to frequency domain
- **Short-Time Fourier**: Time-frequency analysis
- **Wavelets**: Multi-resolution analysis
- **Power Spectrum**: Signal power distribution

Fourier Analysis
----------------

**FFT (Fast Fourier Transform)**
   - O(n log n) algorithm
   - Frequency content of signals
   - Foundation for spectral analysis

**Power Spectrum**
   - Signal power vs frequency
   - Periodogram estimation
   - Welch's method for noise reduction

**Spectrogram**
   - Time-frequency representation
   - Short-time Fourier Transform
   - Frequency changes over time

Wavelet Analysis
----------------

**Continuous Wavelet Transform (CWT)**
   - Multi-scale analysis
   - Good time-frequency localization
   - Various mother wavelets

**Discrete Wavelet Transform (DWT)**
   - Efficient decomposition
   - Signal compression
   - Denoising applications

Code Highlights
---------------

The example demonstrates:

- FFT with ``fft()`` and ``ifft()``
- Power spectrum with ``power_spectrum()``
- Spectrogram with ``spectrogram()``
- Wavelet transforms with ``cwt()`` and ``dwt()``

Source Code
-----------

.. literalinclude:: ../../../examples/transforms.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/transforms.py

See Also
--------

- :doc:`signal_processing` - Filter design and detection
