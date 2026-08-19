Hybrid Linear/Nonlinear Filtering with RBPF
==============================================

Overview
--------

The Rao-Blackwellized Particle Filter (RBPF) is a variance-reduced particle
filter for systems with both nonlinear and linear dynamics. It partitions the
state space into:

- **Nonlinear subspace (y)**: Handled via particle filtering
- **Linear subspace (x)**: Handled analytically via Kalman filtering per particle

Because part of the state is marginalized analytically, the RBPF never has
higher estimator variance than a standard particle filter with the same
number of particles; the advantage grows with the dimension of the linear
subspace.

System Model
------------

The RBPF assumes a state space that can be partitioned:

.. math::

   \begin{align}
   \mathbf{y}_{k+1} &= \mathbf{g}(\mathbf{y}_k) + \mathbf{w}_k^y \\
   \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{y}_k) + \mathbf{w}_k^x \\
   \mathbf{z}_k &= \mathbf{h}(\mathbf{x}_k, \mathbf{y}_k) + \mathbf{v}_k
   \end{align}

Where:

- :math:`\mathbf{y}_k` is the nonlinear state (particle-filtered)
- :math:`\mathbf{x}_k` is the linear state (Kalman-filtered per particle)
- :math:`\mathbf{w}_k^y, \mathbf{w}_k^x` are process noise (Gaussian)
- :math:`\mathbf{v}_k` is measurement noise (Gaussian)

The key insight: each particle tracks its own **y** and maintains its own
Kalman filter for **x**.

Applications
~~~~~~~~~~~~

Ideal for systems like:

1. **Nonlinear target dynamics + linear sensor**

   - Maneuvering target with correlated accelerations
   - Sensor measures range and bearing nonlinearly

2. **Bilinear systems**

   - Gain-scheduled systems with nonlinear mode dynamics
   - Each particle represents a different maneuver mode

3. **Mixed observability**

   - Some states directly measured (linear observation)
   - Others inferred from nonlinear functions

Basic Usage
-----------

The filter is driven through ``RBPFFilter`` with four steps: ``initialize``,
``predict``, ``update``, and ``estimate``.

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import RBPFFilter

   np.random.seed(42)

   # State-space partition:
   # y = bearing (nonlinear, particle-filtered)
   # x = [range, range-rate] (linear, Kalman-filtered per particle)

   n_particles = 200
   rbpf = RBPFFilter(max_particles=n_particles)
   rbpf.initialize(
       y0=np.array([0.05]),          # initial bearing
       x0=np.array([100.0, 0.0]),    # initial [r, rdot]
       P0=np.diag([1.0, 0.1]),       # covariance of the linear subspace
       num_particles=n_particles,
   )

   dt = 0.1

   # Nonlinear dynamics for y: bearing random walk
   def g(y):
       return y

   Qy = np.array([[1e-4]])   # process noise for y

   # Linear dynamics for x; may depend on the particle's y via the
   # second argument
   def f(x, y):
       return np.array([x[0] + dt * x[1], x[1]])

   F = np.array([[1.0, dt], [0.0, 1.0]])   # Jacobian of f with respect to x
   Qx = np.diag([1e-3, 1e-2])

   rbpf.predict(g=g, Qy=Qy, f=f, F=F, Qx=Qx)

   # Measurement: range from x, bearing from y
   def h(x, y):
       return np.array([x[0], y[0]])

   H = np.array([[1.0, 0.0], [0.0, 0.0]])  # Jacobian of h with respect to x
   R = np.diag([0.1, 1e-3])
   z = np.array([100.5, 0.05])

   rbpf.update(z=z, h=h, H=H, R=R)

   y_est, x_est, P_est = rbpf.estimate()
   print(f"bearing={y_est[0]:.4f}  range={x_est[0]:.2f}  range-rate={x_est[1]:.3f}")

Output::

   bearing=0.0507  range=100.45  range-rate=0.005

Note the conventions:

- ``f`` and ``h`` take the linear state first: ``f(x, y)`` and ``h(x, y)``
- ``G``, ``F``, and ``H`` are Jacobian *matrices*, not callables; ``H`` is
  the Jacobian of ``h`` with respect to ``x`` only
- ``estimate`` returns the weighted means of both subspaces plus the total
  covariance of the linear subspace (mean of per-particle covariances plus
  spread of per-particle means)

A functional API operating on explicit particle lists is also available:

.. code-block:: python

   from pytcl.dynamic_estimation import rbpf_predict, rbpf_update

   particles = rbpf.get_particles()   # list of RBPFParticle(y, x, P, w)
   particles = rbpf_predict(particles, g, Qy, f, F, Qx)
   particles = rbpf_update(particles, z, h, H, R)

Advanced Example: Maneuvering Target Tracking
----------------------------------------------

Track a maneuvering target whose accelerations follow a first-order Markov
process (nonlinear subspace) while position and velocity stay conditionally
linear:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import RBPFFilter

   np.random.seed(7)

   # State partition:
   # y = [ax, ay] (accelerations, nonlinear)
   # x = [px, py, vx, vy] (position and velocity, linear given y)

   n_particles = 200
   dt = 0.5

   rbpf = RBPFFilter(max_particles=n_particles)
   rbpf.initialize(
       y0=np.zeros(2),
       x0=np.array([5000.0, 3000.0, 100.0, 50.0]),
       P0=np.diag([100.0, 100.0, 25.0, 25.0]),
       num_particles=n_particles,
   )

   # Nonlinear acceleration dynamics (first-order Gauss-Markov)
   tau = 10.0                  # correlation time constant
   decay = np.exp(-dt / tau)

   def g(y):
       return decay * y

   Qy = 0.25 * np.eye(2)       # acceleration process noise

   # Linear dynamics, driven by the particle's acceleration
   def f(x, y):
       px, py, vx, vy = x
       ax, ay = y
       return np.array([
           px + vx * dt + 0.5 * ax * dt**2,
           py + vy * dt + 0.5 * ay * dt**2,
           vx + ax * dt,
           vy + ay * dt,
       ])

   F = np.array([
       [1.0, 0.0, dt, 0.0],
       [0.0, 1.0, 0.0, dt],
       [0.0, 0.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 1.0],
   ])
   Qx = np.diag([1.0, 1.0, 0.1, 0.1])

   # Radar measurement: [range, azimuth] of the position
   def h(x, y):
       r = np.hypot(x[0], x[1])
       az = np.arctan2(x[1], x[0])
       return np.array([r, az])

   R = np.diag([25.0, 1e-4])

   # Simulate the truth and run the filter
   true_x = np.array([5000.0, 3000.0, 100.0, 50.0])
   true_a = np.array([1.0, -0.5])

   for k in range(20):
       rbpf.predict(g=g, Qy=Qy, f=f, F=F, Qx=Qx)

       true_x = f(true_x, true_a)
       z = h(true_x, true_a) + np.array([5.0, 0.01]) * np.random.randn(2)

       # Linearize h about the current estimated position
       _, x_mean, _ = rbpf.estimate()
       px, py = x_mean[:2]
       r = np.hypot(px, py)
       H = np.array([
           [px / r, py / r, 0.0, 0.0],
           [-py / r**2, px / r**2, 0.0, 0.0],
       ])

       rbpf.update(z=z, h=h, H=H, R=R)

       if (k + 1) % 5 == 0:
           y_est, x_est, _ = rbpf.estimate()
           print(
               f"Step {k+1}: pos=({x_est[0]:.0f}, {x_est[1]:.0f}), "
               f"vel=({x_est[2]:.1f}, {x_est[3]:.1f}), "
               f"accel=({y_est[0]:.2f}, {y_est[1]:.2f})"
           )

Output::

   Step 5: pos=(5252, 3134), vel=(102.1, 53.0), accel=(0.08, 0.01)
   Step 10: pos=(5510, 3249), vel=(101.1, 48.7), accel=(-0.52, -0.48)
   Step 15: pos=(5781, 3367), vel=(106.7, 48.8), accel=(0.75, -0.08)
   Step 20: pos=(6046, 3477), vel=(107.4, 45.8), accel=(0.55, -0.26)

Performance and Tuning
-----------------------

**Particle Count**

``num_particles`` sets how many particles are created; ``max_particles``
caps the population (particle merging kicks in above the cap, which is
quadratic in the particle count, so keep ``num_particles <= max_particles``
unless you want merging). Guidelines:

.. code-block:: python

   # Guidelines:
   N = 100    # Fast, moderate accuracy (nonlinear state small)
   N = 500    # Balanced (recommended for most applications)
   N = 1000   # High accuracy, slower (nonlinear state dimension > 5)

**Resampling**

Resampling is built in: after each ``update``, the filter computes the
effective sample size and performs systematic resampling when it drops
below ``resample_threshold * N`` (default threshold 0.5). To monitor
degeneracy yourself:

.. code-block:: python

   weights = np.array([p.w for p in rbpf.particles])
   n_eff = 1.0 / np.sum(weights**2)
   print(f"Effective sample size: {n_eff:.1f} of {len(weights)}")

**Process Noise Selection**

Tune ``Qx`` and ``Qy`` to match system characteristics:

.. code-block:: python

   # If estimates diverge: increase noise
   Qx = Qx * 2.0
   Qy = Qy * 2.0

   # If variance grows despite measurements: decrease noise
   Qx = Qx * 0.5
   Qy = Qy * 0.5

Variance Reduction Analysis
---------------------------

Marginalizing the linear substate means each particle carries an exact
conditional Gaussian instead of a sampled point, so Monte Carlo error is
only incurred in the (smaller) nonlinear subspace. By the law of total
variance this cannot increase estimator variance, and in practice an RBPF
matches the accuracy of a plain particle filter that uses several times as
many particles. The gain is largest when the linear subspace is
high-dimensional relative to the nonlinear one.

Integration with Tracking
--------------------------

For multi-target problems, maintain one ``RBPFFilter`` per track and drive
each filter's ``predict``/``update`` cycle from your data association
logic. The multi-target trackers in ``pytcl.trackers`` manage linear
Kalman filters internally; they do not accept RBPF state, so RBPF-based
tracks must be managed by the application.

See Also
~~~~~~~~

- :doc:`getting_started` - Basic particle filtering
- :doc:`particle_filters` - Standard particle filter reference
- :doc:`adaptive_filtering` - Adaptive noise tuning
- :ref:`Rao-Blackwellized Particle Filter <rbpf>` - API Reference
