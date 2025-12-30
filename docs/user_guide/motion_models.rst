Motion Models
=============

This guide covers the discrete-time motion models available for target tracking.

State Transition Matrices
-------------------------

Polynomial Models
^^^^^^^^^^^^^^^^^

**Constant Velocity (CV)**

State: ``[position, velocity]`` per dimension

.. math::

   F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}

.. code-block:: python

   from tracker_component_library.dynamic_models import f_constant_velocity

   # 2D tracking: state = [x, vx, y, vy]
   F = f_constant_velocity(T=1.0, num_dims=2)

**Constant Acceleration (CA)**

State: ``[position, velocity, acceleration]`` per dimension

.. code-block:: python

   from tracker_component_library.dynamic_models import f_constant_acceleration

   # 2D tracking: state = [x, vx, ax, y, vy, ay]
   F = f_constant_acceleration(T=1.0, num_dims=2)

Singer Acceleration Model
^^^^^^^^^^^^^^^^^^^^^^^^^

The Singer model treats acceleration as a first-order Markov process,
suitable for maneuvering targets:

.. math::

   \frac{da}{dt} = -\frac{a}{\tau} + w(t)

where :math:`\tau` is the maneuver time constant.

.. code-block:: python

   from tracker_component_library.dynamic_models import f_singer, f_singer_2d, f_singer_3d

   # tau: maneuver time constant (5-20s for aircraft, 1-5s for ground vehicles)
   F = f_singer(T=1.0, tau=10.0, num_dims=2)

   # Or use convenience functions
   F_2d = f_singer_2d(T=1.0, tau=10.0)
   F_3d = f_singer_3d(T=1.0, tau=10.0)

Coordinated Turn Models
^^^^^^^^^^^^^^^^^^^^^^^

For targets executing turns at constant turn rate:

**2D Coordinated Turn**

State: ``[x, vx, y, vy, omega]`` where omega is turn rate

.. code-block:: python

   from tracker_component_library.dynamic_models import f_coord_turn_2d

   F = f_coord_turn_2d(T=1.0, omega=0.1)  # omega in rad/s

**3D Coordinated Turn**

.. code-block:: python

   from tracker_component_library.dynamic_models import f_coord_turn_3d

   F = f_coord_turn_3d(T=1.0, omega=0.1)

Process Noise Covariance
------------------------

Polynomial Models
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.dynamic_models import (
       q_constant_velocity,
       q_constant_acceleration,
       q_discrete_white_noise,
   )

   # Constant velocity with acceleration noise
   Q_cv = q_constant_velocity(T=1.0, sigma_a=1.0, num_dims=2)

   # Constant acceleration with jerk noise
   Q_ca = q_constant_acceleration(T=1.0, sigma_j=0.5, num_dims=2)

   # General discrete white noise
   Q = q_discrete_white_noise(dim=2, T=1.0, var=1.0, block_size=2)

Singer Model
^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.dynamic_models import q_singer, q_singer_2d, q_singer_3d

   # sigma_m: RMS maneuver level (m/s^2)
   Q = q_singer(T=1.0, tau=10.0, sigma_m=1.0, num_dims=2)

   # Convenience functions
   Q_2d = q_singer_2d(T=1.0, tau=10.0, sigma_m=1.0)
   Q_3d = q_singer_3d(T=1.0, tau=10.0, sigma_m=1.0)

Coordinated Turn
^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.dynamic_models import q_coord_turn_2d, q_coord_turn_3d

   Q_2d = q_coord_turn_2d(T=1.0, sigma_a=1.0, sigma_omega=0.01)
   Q_3d = q_coord_turn_3d(T=1.0, sigma_a=1.0, sigma_omega=0.01)

Continuous-Time Dynamics
------------------------

For simulation or continuous-time modeling:

.. code-block:: python

   from tracker_component_library.dynamic_models import (
       drift_constant_velocity,
       drift_constant_acceleration,
       drift_singer,
       diffusion_constant_velocity,
       continuous_to_discrete,
   )

   # Continuous-time drift function
   x_dot = drift_constant_velocity(x, t, num_dims=2)

   # Convert continuous dynamics to discrete
   F, Q = continuous_to_discrete(A, G, Q_c, T=0.1)

Choosing a Motion Model
-----------------------

.. list-table::
   :header-rows: 1

   * - Scenario
     - Model
     - Notes
   * - Non-maneuvering targets
     - Constant Velocity
     - Simple, computationally efficient
   * - Accelerating targets
     - Constant Acceleration
     - Good for smooth acceleration
   * - Maneuvering targets
     - Singer
     - Models random maneuvers
   * - Turning targets
     - Coordinated Turn
     - Explicitly models turn rate
