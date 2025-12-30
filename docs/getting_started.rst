Getting Started
===============

This guide will help you get started with the Tracker Component Library.

Installation
------------

Requirements
^^^^^^^^^^^^

* Python 3.10 or later
* NumPy >= 1.20
* SciPy >= 1.7

Install from PyPI
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install tracker-component-library

Install from Source
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary.git
   cd TrackerComponentLibrary
   pip install -e ".[dev]"

Basic Concepts
--------------

State Representation
^^^^^^^^^^^^^^^^^^^^

States are represented as 1D NumPy arrays. For kinematic tracking, common
state vectors include:

* **Constant velocity (2D)**: ``[x, vx, y, vy]``
* **Constant acceleration (2D)**: ``[x, vx, ax, y, vy, ay]``
* **Singer model (2D)**: ``[x, vx, ax, y, vy, ay]`` (acceleration is correlated)

Covariance matrices are represented as 2D NumPy arrays of shape ``(n, n)``.

Motion Models
^^^^^^^^^^^^^

The library provides state transition matrices (F) and process noise
covariance matrices (Q) for various motion models:

.. code-block:: python

   from pytcl.dynamic_models import (
       f_constant_velocity,
       f_constant_acceleration,
       f_singer,
       q_constant_velocity,
       q_constant_acceleration,
       q_singer,
   )

   # Constant velocity model
   F_cv = f_constant_velocity(T=1.0, num_dims=2)
   Q_cv = q_constant_velocity(T=1.0, sigma_a=1.0, num_dims=2)

   # Singer maneuvering model
   F_singer = f_singer(T=1.0, tau=10.0, num_dims=2)
   Q_singer = q_singer(T=1.0, tau=10.0, sigma_m=1.0, num_dims=2)

Filters
^^^^^^^

The library provides several filtering algorithms:

**Linear Kalman Filter** - For linear dynamics and measurements:

.. code-block:: python

   from pytcl.dynamic_estimation import kf_predict, kf_update

   pred = kf_predict(x, P, F, Q)
   upd = kf_update(pred.x, pred.P, z, H, R)

**Extended Kalman Filter** - For nonlinear dynamics/measurements:

.. code-block:: python

   from pytcl.dynamic_estimation import ekf_predict, ekf_update

   pred = ekf_predict(x, P, f_func, F_jacobian, Q)
   upd = ekf_update(pred.x, pred.P, z, h_func, H_jacobian, R)

**Unscented Kalman Filter** - For highly nonlinear systems:

.. code-block:: python

   from pytcl.dynamic_estimation import ukf_predict, ukf_update

   pred = ukf_predict(x, P, f_func, Q)
   upd = ukf_update(pred.x, pred.P, z, h_func, R)

**Particle Filter** - For non-Gaussian distributions:

.. code-block:: python

   from pytcl.dynamic_estimation import (
       initialize_particles,
       bootstrap_pf_step,
   )

   state = initialize_particles(x0, P0, N=1000)
   state = bootstrap_pf_step(state.particles, state.weights, z, f, h, Q_sample, R)

Coordinate Systems
^^^^^^^^^^^^^^^^^^

Convert between coordinate systems:

.. code-block:: python

   from pytcl.coordinate_systems import (
       cart2sphere,
       sphere2cart,
       geodetic2ecef,
       ecef2geodetic,
   )

   # Cartesian to spherical
   r, az, el = cart2sphere(np.array([[100, 200, 50]]))

   # Geodetic to ECEF
   x, y, z = geodetic2ecef(lat=40.0, lon=-75.0, alt=100.0)
