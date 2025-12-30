Tracker Component Library
=========================

A Python port of the U.S. Naval Research Laboratory's Tracker Component Library,
providing a comprehensive collection of algorithms for target tracking and
state estimation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide/index
   api/index

Overview
--------

The Tracker Component Library provides:

* **Mathematical Functions**: Special functions, statistics, interpolation,
  numerical integration, geometry, and combinatorics
* **Coordinate Systems**: Conversions between Cartesian, spherical, geodetic
  coordinates; rotation representations; coordinate Jacobians
* **Dynamic Models**: State transition matrices and process noise for various
  motion models (constant velocity, constant acceleration, Singer, coordinated turn)
* **Dynamic Estimation**: Kalman filter family (KF, EKF, UKF, CKF), particle
  filters, and smoothing algorithms

Installation
------------

.. code-block:: bash

   pip install tracker-component-library

Or install from source:

.. code-block:: bash

   git clone https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary.git
   cd TrackerComponentLibrary
   pip install -e .

Quick Start
-----------

Here's a simple example using the Kalman filter:

.. code-block:: python

   import numpy as np
   from tracker_component_library.dynamic_estimation import kf_predict, kf_update
   from tracker_component_library.dynamic_models import f_constant_velocity, q_constant_velocity

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
