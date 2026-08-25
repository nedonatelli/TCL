Tracker Component Library
=========================

A Python port of the U.S. Naval Research Laboratory's Tracker Component Library,
providing a comprehensive collection of algorithms for target tracking and
state estimation.

|version_badge| — |n_functions| functions | |n_modules| modules | |n_tests| test functions | ty type-checked | 88% coverage | 9 interactive notebooks | GPU acceleration (CuPy + MLX)

.. note::

   **On MATLAB parity**: the core tracking workflow — filter, associate,
   track, evaluate, in Earth-referenced coordinates — is fully ported and
   validated against independent references. The full MATLAB public surface
   is covered at roughly a third by function count; see
   :doc:`matlab_parity_inventory` for the function-level accounting and
   :doc:`matlab_migration_map` for verified name mappings.

.. toctree::
   :maxdepth: 2
   :caption: Start Here

   getting_started
   architecture
   api_navigation
   recipes

.. toctree::
   :maxdepth: 2
   :caption: Filtering & Estimation

   kalman_filter_tuning
   constrained_filtering
   hybrid_filtering
   adaptive_filtering
   information_filters
   advanced_kf_variants
   custom_filter_implementation

.. toctree::
   :maxdepth: 2
   :caption: Tracking & Association

   assignment_association
   particle_filters
   smoothing
   data_structures
   results_io
   sessions

.. toctree::
   :maxdepth: 2
   :caption: Domain-Specific

   coordinate_systems
   astronomical
   atmosphere_models
   navigation_ins
   signal_processing

.. toctree::
   :maxdepth: 2
   :caption: Performance & Advanced

   gpu_acceleration
   performance_optimization
   diagnostics

.. toctree::
   :maxdepth: 2
   :caption: Reference & Learning

   troubleshooting
   migration_v1_to_v2
   migration_guide
   matlab_parity_inventory
   matlab_migration_map
   roadmap
   user_guide/index
   tutorials/index
   notebooks/index
   examples/index
   api/index

Overview
--------

The Tracker Component Library provides:

* **Dynamic Estimation**: Kalman filters (KF, EKF, UKF, CKF, **CEKF**, IMM), particle filters
  (**RBPF**), smoothers (RTS, fixed-lag, two-filter), information filters (SRIF), H-infinity
* **Data Association**: GNN, JPDA, MHT, 2D/3D assignment (Hungarian, auction, Murty)
* **Coordinate Systems**: Cartesian, spherical, geodetic conversions; map projections
  (UTM, Mercator, Lambert); rotation representations
* **Dynamic Models**: Constant velocity, acceleration, Singer, coordinated turn,
  and polynomial motion models
* **Navigation**: INS mechanization, INS/GNSS integration, great circle/rhumb line
  navigation, TDOA localization
* **Geophysical Models**: Gravity (WGS84, EGM96/2008), magnetism (WMM2025 default, WMM2020, IGRF-13, EMM),
  atmosphere (simplified thermosphere), tidal effects, terrain/DEM utilities
* **Astronomical**: Orbital mechanics, Kepler propagation, Lambert problem,
  reference frame transformations, JPL ephemerides (DE405/430/432s/440),
  relativistic corrections (Schwarzschild, geodetic precession, Shapiro delay)
* **Mathematical Functions**: Special functions (Marcum Q, Lambert W, Debye,
  hypergeometric, Bessel), statistics, numerical integration
* **GPU Acceleration**: CuPy (NVIDIA CUDA) and MLX (Apple Silicon) backends for
  batch Kalman filtering, particle filters, and matrix operations (measured on
  MLX: 1.6x at 100 tracks to 40x at 20,000 tracks for batch Kalman)
* **Data Persistence**: HDF5 and SQL backends for tracking data storage and retrieval

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
