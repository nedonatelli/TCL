Tracker Component Library
=========================

A Python port of the U.S. Naval Research Laboratory's Tracker Component Library,
providing a comprehensive collection of algorithms for target tracking and
state estimation.

**v1.12.1** — 1,070+ functions | 153 modules | 3,280 tests | 100% mypy --strict | 80% coverage | GPU acceleration

.. note::

   **Performance Optimization**: This release adds Numba JIT compilation, intelligent
   caching with lru_cache, and SparseCostTensor for efficient large-scale assignment
   problems. Type safety maintained with full mypy --strict compliance.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   migration_guide
   gap_analysis
   user_guide/index
   tutorials/index
   notebooks/index
   examples/index
   api/index
   roadmap

Overview
--------

The Tracker Component Library provides:

* **Dynamic Estimation**: Kalman filters (KF, EKF, UKF, CKF, IMM), particle filters,
  smoothers (RTS, fixed-lag, two-filter), information filters (SRIF)
* **Data Association**: GNN, JPDA, MHT, 2D/3D assignment (Hungarian, auction, Murty)
* **Coordinate Systems**: Cartesian, spherical, geodetic conversions; map projections
  (UTM, Mercator, Lambert); rotation representations
* **Dynamic Models**: Constant velocity, acceleration, Singer, coordinated turn,
  and polynomial motion models
* **Navigation**: INS mechanization, INS/GNSS integration, great circle/rhumb line
  navigation, TDOA localization
* **Geophysical Models**: Gravity (WGS84, EGM96/2008), magnetism (WMM, IGRF),
  tidal effects, terrain/DEM utilities
* **Astronomical**: Orbital mechanics, Kepler propagation, Lambert problem,
  reference frame transformations, JPL ephemerides (DE405/430/432s/440),
  relativistic corrections (Schwarzschild, geodetic precession, Shapiro delay)
* **Mathematical Functions**: Special functions (Marcum Q, Lambert W, Debye,
  hypergeometric, Bessel), statistics, numerical integration
* **Signal Processing**: IIR/FIR filters, CFAR detection, FFT, wavelets

Installation
------------

.. code-block:: bash

   pip install nrl-tracker

Or install from source:

.. code-block:: bash

   git clone https://github.com/nedonatelli/TCL.git
   cd TCL
   pip install -e .

Quick Start
-----------

Here's a simple example using the Kalman filter:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import kf_predict, kf_update
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

   # Initial state [x, vx, y, vy]
   x = np.array([0.0, 1.0, 0.0, 0.5])
   P = np.eye(4) * 0.1

   # Motion model
   T = 1.0  # time step
   F = f_constant_velocity(T=T, num_dims=2)
   Q = q_constant_velocity(T=T, sigma_a=0.1, num_dims=2)

   # Predict
   pred = kf_predict(x, P, F, Q)
   print(f"Predicted state: {pred.x}")

   # Measurement model (position only)
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 0.5
   z = np.array([1.1, 0.6])

   # Update
   upd = kf_update(pred.x, pred.P, z, H, R)
   print(f"Updated state: {upd.x}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
