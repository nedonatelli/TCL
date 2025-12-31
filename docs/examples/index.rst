Examples
========

Standalone example scripts demonstrating pytcl functionality.

These examples are complete, runnable Python scripts that you can use
as starting points for your own applications.

Filtering
---------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`kalman_tracking.py`
     - Linear Kalman filter for 2D position tracking
   * - :download:`ukf_range_bearing.py`
     - Unscented Kalman filter with range-bearing measurements

Multi-Target Tracking
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`multi_target_tracking.py`
     - GNN-based multi-target tracker with OSPA evaluation

Signal Processing
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`cfar_detection.py`
     - CFAR detection algorithms for radar signals
   * - :download:`spectral_analysis.py`
     - Power spectrum and spectrogram computation

Navigation
----------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`coordinate_transforms.py`
     - Geodetic coordinate conversions and geodesic calculations

Running Examples
----------------

All examples can be run directly::

   cd docs/examples
   python kalman_tracking.py

Or from the repository root::

   python docs/examples/kalman_tracking.py

Requirements
------------

Examples require pytcl to be installed::

   pip install -e .

Some examples may require additional dependencies::

   pip install matplotlib  # For visualization
