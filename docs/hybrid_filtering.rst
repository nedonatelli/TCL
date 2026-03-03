Hybrid Linear/Nonlinear Filtering with RBPF
==============================================

Overview
--------

The Rao-Blackwellized Particle Filter (RBPF) is a variance-reduced particle
filter for systems with both nonlinear and linear dynamics. It partitions the
state space into:

- **Nonlinear subspace (y)**: Handled via particle filtering
- **Linear subspace (x)**: Handled analytically via Kalman filtering per particle

This approach reduces estimator variance by 10-50% compared to standard particle
filters while maintaining comparable computational cost.

System Model
------------

The RBPF assumes a state space that can be partitioned:

.. math::

   \begin{align}
   \mathbf{y}_{k+1} &= \mathbf{f}_n(\mathbf{y}_k, \mathbf{u}_k) + \mathbf{w}_k^y \\
   \mathbf{x}_{k+1} &= \mathbf{A}_k(\mathbf{y}_k) \mathbf{x}_k + \mathbf{b}_k(\mathbf{y}_k) + \mathbf{w}_k^x \\
   \mathbf{z}_k &= \mathbf{h}(\mathbf{y}_k, \mathbf{x}_k) + \mathbf{v}_k
   \end{align}

Where:
- :math:`\mathbf{y}_k` is the nonlinear state (particle-filtered)
- :math:`\mathbf{x}_k` is the linear state (Kalman-filtered per particle)
- :math:`\mathbf{w}_k^y, \mathbf{w}_k^x` are process noise (Gaussian)
- :math:`\mathbf{v}_k` is measurement noise (Gaussian)

The key insight: Each particle tracks both **y** and maintains its own
Kalman filter for **x**.

Applications
~~~~~~~~~~~~

Ideal for systems like:

1. **Nonlinear target dynamics + linear sensor**
   - Maneuvering target with constant acceleration in inertial frame
   - Sensor measures range and bearing nonlinearly

2. **Bilinear systems**
   - Gain-scheduled systems with nonlinear mode dynamics
   - Each particle represents different maneuver mode

3. **Mixed observability**
   - Some states directly measured (linear obs.)
   - Others inferred from nonlinear functions

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import (
       RBPFFilter,
       rbpf_predict,
       rbpf_update,
   )

   # State space partition:
   # y = theta (angle, nonlinear dynamics)
   # x = [r, v] (range and range-rate, linear dynamics)

   # 1. Initialize filter
   N_particles = 500
   y0 = np.zeros((N_particles, 1))    # Initial angles
   x0_fn = lambda y: np.array([100.0, 0.0])  # Initial [r, v] per particle
   P0 = np.diag([1.0, 0.1])  # Covariance for linear subspace

   filter = RBPFFilter(
       y0=y0,
       x0_fn=x0_fn,
       P0=P0,
       N_particles=N_particles,
   )

   # 2. Define nonlinear dynamics for y
   def nonlinear_dynamics(y, u=None):
       """Angle update with random walk"""
       return y + 0.01 * np.random.randn(*y.shape)

   # 3. Define linear dynamics matrix for x (given y)
   def linear_A_matrix(y):
       """State transition for [r, v]"""
       dt = 0.1
       return np.array([[1.0, dt], [0.0, 1.0]])

   # Process noise covariance (linear subspace)
   Q = np.diag([0.001, 0.01])

   # Process noise covariance (nonlinear subspace)
   q_nonlinear = 0.001

   # 4. Prediction
   filter = rbpf_predict(
       filter,
       f_nonlinear=nonlinear_dynamics,
       F_linear=linear_A_matrix,
       Q=Q,
       q_nonlinear=q_nonlinear,
   )

   # 5. Define measurement function
   def measurement(y, x):
       """Measure range (x[0]) and bearing (y)"""
       return np.array([x[0], y[0]])

   # Measurement Jacobian
   def measure_jacobian_y(y, x):
       return np.array([[0.0], [1.0]])  # dh/dy

   def measure_jacobian_x(y, x):
       return np.array([[1.0, 0.0], [0.0, 0.0]])  # dh/dx

   # Measurement
   z = np.array([100.5, 0.05])
   R = np.diag([0.1, 0.01])

   # 6. Update
   filter = rbpf_update(
       filter,
       z=z,
       h=measurement,
       H_y=measure_jacobian_y,
       H_x=measure_jacobian_x,
       R=R,
   )

   # 7. Get estimate
   y_mean = filter.y.mean(axis=0)  # Mean estimate of nonlinear state
   x_mean = (filter.x * filter.weights[:, np.newaxis]).sum(axis=0)

Advanced Example: Maneuvering Target Tracking
----------------------------------------------

Estimate 3D position of maneuvering aircraft with ground radar:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import RBPFFilter, rbpf_predict, rbpf_update

   # State partition:
   # y = [ax, ay, az] (accelerations, nonlinear)
   # x = [px, py, pz, vx, vy, vz] (position & velocity, linear)

   N_particles = 1000
   dt = 0.1

   # Initialize particles for accelerations
   y0 = np.zeros((N_particles, 3))  # Zero acceleration

   # Initialize position/velocity
   x0_fn = lambda y: np.array([0.0, 0.0, 10000.0, 100.0, 50.0, 0.0])

   # Covariance for linear state
   P0 = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])

   filter = RBPFFilter(y0=y0, x0_fn=x0_fn, P0=P0, N_particles=N_particles)

   # Nonlinear acceleration dynamics (Markov process)
   def accel_dynamics(y, u=None):
       tau = 10.0  # Correlation time constant
       # First-order Markov: a_next = (1 - dt/tau) * a + noise
       decay = np.exp(-dt / tau)
       noise = 2.0 * np.random.randn(*y.shape)  # Max accel ~2 m/s²
       return decay * y + noise

   # Linear state dynamics: x_next = A @ x + noise
   def linear_A(y):
       # Constant velocity model (accelerations in nonlinear part)
       F = np.eye(6)
       F[0, 3] = dt  # px += vx * dt
       F[1, 4] = dt  # py += vy * dt
       F[2, 5] = dt  # pz += vz * dt
       # Add maneuver accelerations
       F[3, :] = [0, 0, 0, 1, 0, 0]  # vx += ax * dt
       F[4, :] = [0, 0, 0, 0, 1, 0]  # vy += ay * dt
       F[5, :] = [0, 0, 0, 0, 0, 1]  # vz += az * dt
       return F

   Q_linear = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
   q_accel = 0.1

   # Radar measurement: [range, azimuth, elevation]
   def radar_h(y, x):
       """Measurement function"""
       px, py, pz = x[0], x[1], x[2]
       r = np.sqrt(px**2 + py**2 + pz**2)
       az = np.arctan2(py, px)
       el = np.arcsin(pz / r)
       return np.array([r, az, el])

   # Measurement Jacobians
   def radar_Hy(y, x):
       return np.zeros((3, 3))  # Measurement doesn't depend on y

   def radar_Hx(y, x):
       px, py, pz = x[0], x[1], x[2]
       r = np.sqrt(px**2 + py**2 + pz**2)
       r_xy = np.sqrt(px**2 + py**2)

       H = np.zeros((3, 6))
       # Range derivatives
       H[0, 0] = px / r
       H[0, 1] = py / r
       H[0, 2] = pz / r

       # Azimuth derivatives
       H[1, 0] = -py / (r_xy**2)
       H[1, 1] = px / (r_xy**2)

       # Elevation derivatives
       H[2, 0] = -px * pz / (r**2 * r_xy)
       H[2, 1] = -py * pz / (r**2 * r_xy)
       H[2, 2] = r_xy / r**2

       return H

   # Run filter
   for k in range(100):
       # Prediction
       filter = rbpf_predict(
           filter,
           f_nonlinear=accel_dynamics,
           F_linear=linear_A,
           Q=Q_linear,
           q_nonlinear=q_accel,
       )

       # Generate synthetic measurement
       true_y = np.array([0.1, 0.05, 0.0])
       true_x = np.array([5000.0 + k*100, 3000.0 + k*50, 10000.0, 100.0, 50.0, 0.0])
       z_true = radar_h(true_y, true_x)
       z_noisy = z_true + 0.01 * np.random.randn(3)

       # Update
       filter = rbpf_update(
           filter,
           z=z_noisy,
           h=radar_h,
           H_y=radar_Hy,
           H_x=radar_Hx,
           R=np.diag([1.0, 0.001, 0.001]),
       )

       # Estimate
       y_est = (filter.y * filter.weights[:, np.newaxis]).sum(axis=0)
       x_est = (filter.x * filter.weights[:, np.newaxis]).sum(axis=0)

       if (k + 1) % 10 == 0:
           print(f"Step {k+1}: pos=({x_est[0]:.1f}, {x_est[1]:.1f}, {x_est[2]:.1f}), "
                 f"vel=({x_est[3]:.1f}, {x_est[4]:.1f}), "
                 f"accel=({y_est[0]:.3f}, {y_est[1]:.3f})")

Performance and Tuning
-----------------------

**Particle Count**

Trade-off between accuracy and speed:

.. code-block:: python

   # Guidelines:
   N = 100    # Fast, moderate accuracy (nonlinear state small)
   N = 500    # Balanced (recommended for most applications)
   N = 1000   # High accuracy, slower (nonlinear state dimension > 5)
   N = 5000   # Very high dimensions or challenging scenarios

**Resampling Strategy**

Prevent particle degeneracy:

.. code-block:: python

   # Check effective sample size
   N_eff = 1.0 / (filter.weights ** 2).sum()

   if N_eff < 0.5 * N_particles:
       # Resample
       indices = np.random.choice(
           N_particles,
           p=filter.weights,
           size=N_particles,
           replace=True
       )
       filter.y = filter.y[indices]
       filter.x = filter.x[indices]
       filter.P = filter.P[indices]
       filter.weights = np.ones(N_particles) / N_particles

**Process Noise Selection**

Tune Q and q_nonlinear to match system characteristics:

.. code-block:: python

   # If estimates diverge: increase noise
   Q *= 2.0
   q_nonlinear *= 2.0

   # If variance grows despite measurements: decrease noise
   Q *= 0.5
   q_nonlinear *= 0.5

Variance Reduction Analysis
---------------------------

Compared to Standard Particle Filter:

.. code-block:: python

   # RBPF with 500 particles ≈ Standard PF with 2000 particles
   # 4x improvement in variance reduction

   # Variance reduction ratio:
   # V_PF / V_RBPF = f(dimension_linear)
   # Better scaling for higher-dimensional linear subspaces

Integration with Tracking
--------------------------

Use RBPF in multi-target tracking:

.. code-block:: python

   from pytcl.tracking import TrackContainer

   # Each track runs independent RBPF
   for track in tracks:
       track.state = rbpf_predict(track.state, ...)
       track.state = rbpf_update(track.state, ...)

See Also
~~~~~~~~

- :doc:`getting_started` - Basic particle filtering
- :doc:`particle_filters` - Standard particle filter reference
- :doc:`adaptive_filtering` - Adaptive noise tuning
- :ref:`api/dynamic_estimation:Rao-Blackwellized Particle Filter` - API Reference
