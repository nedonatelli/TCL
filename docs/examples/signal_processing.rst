Signal Processing
=================

This example demonstrates digital filter design, matched filtering, and CFAR detection.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/signal_processing.html"></iframe>
   </div>

Overview
--------

Signal processing fundamentals for radar and tracking:

- **Digital filters**: FIR and IIR filter design
- **Matched filtering**: Optimal detection in noise
- **CFAR detection**: Constant False Alarm Rate processing
- **Spectral analysis**: FFT, power spectrum, spectrograms

Digital Filters
---------------

**FIR Filters**
   - Finite Impulse Response
   - Linear phase available
   - Always stable

**IIR Filters (Butterworth)**
   - Infinite Impulse Response
   - Efficient implementation
   - Maximally flat passband

Matched Filtering
-----------------

Matched filters maximize SNR for known waveforms:

- Correlates signal with template
- Optimal for white Gaussian noise
- Pulse compression for radar

CFAR Detection
--------------

CFAR maintains constant false alarm rate:

- **CA-CFAR**: Cell-averaging
- **GO-CFAR**: Greatest-of
- **OS-CFAR**: Ordered-statistic
- Adaptive threshold estimation

Code Highlights
---------------

The example demonstrates:

- Butterworth filter design with ``butter()``
- FIR filter design with ``firwin()``
- Matched filtering with ``matched_filter()``
- CA-CFAR with ``cfar_ca()``
- Power spectrum estimation

Source Code
-----------

.. literalinclude:: ../../examples/signal_processing.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/signal_processing.py

See Also
--------

- :doc:`transforms` - FFT and spectral analysis
