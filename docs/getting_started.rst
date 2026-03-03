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

   pip install nrl-tracker

Install from Source
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/nedonatelli/TCL.git
   cd TCL
   pip install -e ".[dev]"

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

Install optional features as needed:

.. code-block:: bash

   # For astronomy features (ephemerides, celestial mechanics)
   pip install nrl-tracker[astronomy]

   # For geodesy features (coordinate transforms)
   pip install nrl-tracker[geodesy]

   # For GPU acceleration (NVIDIA CUDA)
   pip install nrl-tracker[gpu]

   # For GPU acceleration (Apple Silicon M1/M2/M3)
   pip install nrl-tracker[gpu-apple]

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

   # F and H are the Jacobian matrices evaluated at the current state
   F = F_jacobian(x)  # Jacobian of f at x
   H = H_jacobian(x)  # Jacobian of h at x
   pred = ekf_predict(x, P, f_func, F, Q)
   upd = ekf_update(pred.x, pred.P, z, h_func, H, R)

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

   # Q_sample is a callable that samples process noise
   def Q_sample(n_particles, rng=None):
       if rng is None:
           rng = np.random.default_rng()
       return rng.multivariate_normal(np.zeros(n), Q_cov, size=n_particles)

   state = initialize_particles(x0, P0, N=1000)
   state = bootstrap_pf_step(state.particles, state.weights, z, f, h, Q_sample, R)

**Constrained Extended Kalman Filter** - For state constraints (e.g., bounded positions):

.. code-block:: python

   from pytcl.dynamic_estimation.kalman import (
       constrained_ekf_predict,
       constrained_ekf_update,
       ConstraintFunction,
   )

   # Define constraint: 0 <= x[0] <= 100 (position within bounds)
   def constraint_lower(x):
       return -x[0]  # g(x) <= 0 means x[0] >= 0
   
   def constraint_upper(x):
       return x[0] - 100.0  # g(x) <= 0 means x[0] <= 100

   constraint_lower_obj = ConstraintFunction(constraint_lower)
   constraint_upper_obj = ConstraintFunction(constraint_upper)

   pred = constrained_ekf_predict(x, P, f_func, F_jacobian, Q, constraints=[...])
   upd = constrained_ekf_update(pred.x, pred.P, z, h_func, H_jacobian, R, constraints=[...])

**Rao-Blackwellized Particle Filter** - Hybrid linear/nonlinear filtering:

.. code-block:: python

   from pytcl.dynamic_estimation import (
       rbpf_predict,
       rbpf_update,
       RBPFFilter,
   )

   # For systems with nonlinear dynamics in 'y' and linear dynamics in 'x'
   # Each particle maintains its own Kalman filter for the linear part
   filter = RBPFFilter(
       y0=np.array([...]),  # Nonlinear state (particle positions)
       x0_fn=lambda y: np.array([...]),  # Linear state given y
       P0=P_linear,
       N_particles=500,
   )

   filter = rbpf_predict(filter, f_nonlinear, F_linear, Q_linear, Q_nonlinear)
   filter = rbpf_update(filter, z, h_func, R)

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

Atmospheric Models
^^^^^^^^^^^^^^^^^^

Get atmospheric density for satellite drag calculations:

.. code-block:: python

   from pytcl.atmosphere import nrlmsise00

   # NRLMSISE-00 model with solar/geomagnetic activity
   density = nrlmsise00.get_density(
       altitude_km=400.0,
       latitude_deg=45.0,
       longitude_deg=-75.0,
       year=2024,
       day_of_year=100,
       hour_utc=12.0,
       f107=150.0,  # 10.7 cm solar flux (SFU)
       f107a=130.0,  # 81-day average
       kp=3.0,  # Geomagnetic activity index
   )
   print(f"Density: {density:.3e} kg/m³")

   # Get atmospheric composition
   composition = nrlmsise00.get_composition(
       altitude_km=400.0,
       latitude_deg=0.0,
       longitude_deg=0.0,
       year=2024,
       day_of_year=100,
       hour_utc=12.0,
       f107=150.0,
       f107a=130.0,
       kp=3.0,
   )
