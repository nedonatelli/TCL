Getting Started
===============

This guide will help you get started with the Tracker Component Library.

Installation
------------

Requirements
^^^^^^^^^^^^

* Python 3.10 or later
* NumPy >= 1.24
* SciPy >= 1.10
* Numba >= 0.57
* h5py >= 3.8

Install from PyPI
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install nrl-tracker

Install from Source
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/nedonatelli/TCL.git
   cd TCL
   pip install -e .

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

Install optional features as needed:

.. code-block:: bash

   # For astronomy features (ephemerides, celestial mechanics)
   pip install nrl-tracker[astronomy]

   # For geodesy features (coordinate transforms)
   pip install nrl-tracker[geodesy]

   # For terrain data (GEBCO, Earth2014 via NetCDF)
   pip install nrl-tracker[terrain]

   # For visualization (Plotly)
   pip install nrl-tracker[visualization]

   # For signal processing (wavelets)
   pip install nrl-tracker[signal]

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

   import numpy as np

   from pytcl.dynamic_estimation import kf_predict, kf_update

   x = np.array([0.0, 1.0, 0.0, 1.0])  # [x, vx, y, vy]
   P = np.eye(4)
   H = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]])  # measure position only
   R = 0.5 * np.eye(2)
   z = np.array([1.1, 0.9])

   pred = kf_predict(x, P, F_cv, Q_cv)
   upd = kf_update(pred.x, pred.P, z, H, R)

**Extended Kalman Filter** - For nonlinear dynamics/measurements:

.. code-block:: python

   from pytcl.dynamic_estimation import ekf_predict, ekf_update

   def f_func(x):
       return F_cv @ x

   def h_func(x):
       # Range and bearing from the origin
       return np.array([np.hypot(x[0], x[2]), np.arctan2(x[2], x[0])])

   def H_jacobian(x):
       r = np.hypot(x[0], x[2])
       return np.array([
           [x[0] / r, 0.0, x[2] / r, 0.0],
           [-x[2] / r**2, 0.0, x[0] / r**2, 0.0],
       ])

   R_polar = np.diag([0.1, 0.01])
   z_polar = np.array([1.5, 0.8])

   # F and H are the Jacobian matrices evaluated at the current state
   F = F_cv  # Jacobian of f (linear dynamics, so constant)
   pred = ekf_predict(upd.x, upd.P, f_func, F, Q_cv)
   upd = ekf_update(pred.x, pred.P, z_polar, h_func, H_jacobian(pred.x), R_polar)

**Unscented Kalman Filter** - For highly nonlinear systems:

.. code-block:: python

   from pytcl.dynamic_estimation import ukf_predict, ukf_update

   pred = ukf_predict(upd.x, upd.P, f_func, Q_cv)
   upd = ukf_update(pred.x, pred.P, z_polar, h_func, R_polar)

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
       return rng.multivariate_normal(np.zeros(4), Q_cv, size=n_particles)

   state = initialize_particles(x, P, N=1000)
   state = bootstrap_pf_step(
       state.particles, state.weights, z_polar, f_func, h_func, Q_sample, R_polar
   )

**Constrained Extended Kalman Filter** - For state constraints (e.g., bounded positions):

.. code-block:: python

   from pytcl.dynamic_estimation.kalman import (
       constrained_ekf_predict,
       constrained_ekf_update,
       ConstraintFunction,
   )

   # Define constraints: 0 <= x[0] <= 100 (position within bounds)
   def constraint_lower(x):
       return np.array([-x[0]])  # g(x) <= 0 means x[0] >= 0

   def constraint_upper(x):
       return np.array([x[0] - 100.0])  # g(x) <= 0 means x[0] <= 100

   constraints = [
       ConstraintFunction(constraint_lower),
       ConstraintFunction(constraint_upper),
   ]

   # The predict step is unconstrained; constraints apply at the update
   pred = constrained_ekf_predict(upd.x, upd.P, f_func, F_cv, Q_cv)
   upd = constrained_ekf_update(
       pred.x,
       pred.P,
       z_polar,
       h_func,
       H_jacobian(pred.x),
       R_polar,
       constraints=constraints,
   )

**Rao-Blackwellized Particle Filter** - Hybrid linear/nonlinear filtering:

.. code-block:: python

   from pytcl.dynamic_estimation import RBPFFilter

   # Partition the state: nonlinear part 'y' is handled by particles,
   # linear part 'x' is handled by a Kalman filter per particle
   rbpf = RBPFFilter(max_particles=500)
   rbpf.initialize(
       y0=np.array([0.0]),  # Nonlinear state
       x0=np.array([1.0]),  # Linear state
       P0=np.eye(1),
       num_particles=500,
   )

   def g_nl(y):  # Nonlinear transition: y[k+1] = g(y[k])
       return y + 0.1 * np.sin(y)

   def f_lin(x, y):  # Linear transition: x[k+1] = f(x[k], y[k])
       return x

   def h_meas(x, y):  # Measurement combines both parts
       return y + x

   rbpf.predict(g=g_nl, G=np.eye(1), Qy=0.01 * np.eye(1),
                f=f_lin, F=np.eye(1), Qx=0.01 * np.eye(1))
   rbpf.update(z=np.array([1.2]), h=h_meas, H=np.eye(1), R=0.1 * np.eye(1))
   y_est, x_est, P_est = rbpf.estimate()

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

   # Cartesian to spherical (tracking convention: range, azimuth, elevation)
   r, az, el = cart2sphere(np.array([100.0, 200.0, 50.0]), system_type="az-el")

   # Geodetic to ECEF (angles in radians); returns an ECEF [x, y, z] vector
   ecef = geodetic2ecef(lat=np.deg2rad(40.0), lon=np.deg2rad(-75.0), alt=100.0)

Atmospheric Models
^^^^^^^^^^^^^^^^^^

Get atmospheric density for satellite drag calculations:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import simplified_thermosphere

   # Simplified thermosphere model with solar/geomagnetic activity
   output = simplified_thermosphere(
       latitude=np.deg2rad(45.0),
       longitude=np.deg2rad(-75.0),
       altitude=400e3,          # meters
       year=2024,
       day_of_year=100,
       seconds_in_day=12 * 3600.0,
       f107=150.0,   # 10.7 cm solar flux (SFU)
       f107a=130.0,  # 81-day average
       ap=15.0,      # Planetary magnetic index
   )
   print(f"Density: {output.density:.3e} kg/m^3")

   # Composition is available on the same result
   print(f"Atomic oxygen: {output.o_density:.3e} m^-3")
   print(f"Temperature: {output.temperature:.1f} K")
