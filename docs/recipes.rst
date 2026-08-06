Common Use Cases & Recipes
==========================

This guide provides ready-to-use code for common tracking and estimation scenarios. Each recipe is self-contained and can be adapted to your needs.

See :doc:`architecture` for module organization and :doc:`api_navigation` for function discovery.

A note on return types: the filter functions return named tuples, not plain
pairs. ``kf_predict``/``ekf_predict``/``ukf_predict`` return a
``KalmanPrediction`` with fields ``x`` and ``P``;
``kf_update``/``ekf_update``/``ukf_update`` return a ``KalmanUpdate`` with
fields ``x``, ``P``, ``y``, ``S``, ``K``, and ``likelihood``. Unpack fields
by name.

Basic Single-Target Kalman Filtering
-------------------------------------

**Problem**: Track a single object with position and velocity in 2D, given range/bearing measurements.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity
   from pytcl.coordinate_systems.conversions import sphere2cart

   class SimpleTracker:
       def __init__(self, T=0.1, sigma_a=0.1, meas_std=0.5):
           """
           T: time step (seconds)
           sigma_a: acceleration standard deviation (m/s^2)
           meas_std: measurement standard deviation (meters)
           """
           self.T = T

           # State: [x, vx, y, vy]
           self.x = np.array([100.0, 0.0, 0.0, 0.0])
           self.P = np.eye(4) * 10.0  # Initial uncertainty

           # Motion model (constant velocity)
           self.F = f_constant_velocity(T, num_dims=2)
           self.Q = q_constant_velocity(T, sigma_a, num_dims=2)

           # Measurement model (observe position only)
           self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]])
           self.R = np.eye(2) * meas_std**2

       def process_measurement(self, range_az_el):
           """
           Process a [range, azimuth, elevation] measurement (radians):
           convert to Cartesian and update the filter.
           """
           # Predict
           pred = kf_predict(self.x, self.P, self.F, self.Q)

           # Convert measurement from spherical to Cartesian; keep x, y.
           # 'az-el' puts az=0, el=0 on the +x axis (radar convention).
           r, az, el = range_az_el
           cart_meas = sphere2cart(r, az, el, "az-el")[:2]

           # Update
           upd = kf_update(pred.x, pred.P, cart_meas, self.H, self.R)
           self.x, self.P = upd.x, upd.P

           return self.x[[0, 2]], np.sqrt(np.diag(self.P)[[0, 2]])

   # Usage
   tracker = SimpleTracker()

   # Process measurements over time
   measurements = [
       np.array([100.0, 0.0, 0.0]),      # range=100, az=0, el=0
       np.array([100.5, 0.01, 0.0]),
       np.array([101.2, 0.02, 0.0]),
   ]

   for meas in measurements:
       pos, std = tracker.process_measurement(meas)
       print(f"Position: {np.round(pos, 2)}, Std: {np.round(std, 3)}")

Output::

   Position: [100.   0.], Std: [0.494 0.494]
   Position: [100.29   0.58], Std: [0.381 0.381]
   Position: [100.82   1.46], Std: [0.373 0.373]

Multi-Target Tracking with Assignment
--------------------------------------

**Problem**: Track multiple targets with ambiguous measurements using Global Nearest Neighbor (GNN).

**Recipe**: ``MultiTargetTracker`` takes the motion and measurement models up
front, manages track initiation/confirmation/deletion internally, and is
driven with ``process(measurements, dt)``:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity
   from pytcl.trackers import MultiTargetTracker

   T = 1.0

   # State [x, vx, y, vy]; measurements are [x, y] positions
   F = f_constant_velocity(T, num_dims=2)
   Q = q_constant_velocity(T, sigma_a=0.5, num_dims=2)
   H = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]])
   R = np.eye(2)

   tracker = MultiTargetTracker(
       state_dim=4,
       meas_dim=2,
       F=F,
       H=H,
       Q=Q,
       R=R,
       gate_probability=0.99,
       confirm_hits=3,     # hits needed to confirm a track...
       confirm_window=5,   # ...within this many updates
       max_misses=5,       # delete after this many consecutive misses
   )

   np.random.seed(0)
   for frame in range(10):
       # Two real targets plus one clutter point per frame
       measurements = [
           np.array([100.0 + 5.0 * frame, 50.0]) + np.random.randn(2),
           np.array([200.0 - 3.0 * frame, 80.0 + 2.0 * frame]) + np.random.randn(2),
           np.random.uniform(0.0, 300.0, size=2),   # clutter
       ]
       tracker.process(measurements, dt=T)

   # Confirmed tracks survive; clutter-born tracks never confirm
   for track in tracker.confirmed_tracks:
       pos = track.state[[0, 2]]
       vel = track.state[[1, 3]]
       print(f"Track {track.id}: pos={np.round(pos, 1)}, "
             f"vel={np.round(vel, 1)}, hits={track.hits}")

Output::

   Track 0: pos=[144.2  50. ], vel=[5.  0.3], hits=10
   Track 1: pos=[172.   97.3], vel=[-3.2  1.7], hits=10

``process`` returns the full track list (``Track`` named tuples with
``id``, ``state``, ``covariance``, ``status``, ``hits``, ``misses``, and
``time``); the ``confirmed_tracks`` property filters it to confirmed ones.

Extended Kalman Filter with Nonlinear Dynamics
-----------------------------------------------

**Problem**: Track with nonlinear motion (e.g., coordinated turn model).

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update

   def coordinated_turn(x, T):
       """Coordinated turn transition; state [x, vx, y, vy, omega]."""
       px, vx, py, vy, omega = x
       if abs(omega) > 1e-6:
           s, c = np.sin(omega * T), np.cos(omega * T)
           return np.array([
               px + vx / omega * s - vy / omega * (1 - c),
               vx * c - vy * s,
               py + vx / omega * (1 - c) + vy / omega * s,
               vx * s + vy * c,
               omega,
           ])
       # Straight line when omega is approximately 0
       x_new = x.copy()
       x_new[0] += vx * T
       x_new[2] += vy * T
       return x_new

   def coordinated_turn_jacobian(x, T):
       """Jacobian of the coordinated turn model."""
       px, vx, py, vy, omega = x
       F = np.eye(5)
       if abs(omega) > 1e-6:
           s, c = np.sin(omega * T), np.cos(omega * T)
           F[0, 1] = s / omega
           F[0, 3] = -(1 - c) / omega
           F[1, 1] = c
           F[1, 3] = -s
           F[2, 1] = (1 - c) / omega
           F[2, 3] = s / omega
           F[3, 1] = s
           F[3, 3] = c
           # Sensitivity to the turn rate
           F[0, 4] = vx / omega * (T * c - s / omega) - vy / omega * (T * s - (1 - c) / omega)
           F[1, 4] = -T * (vx * s + vy * c)
           F[2, 4] = vx / omega * (T * s - (1 - c) / omega) + vy / omega * (T * c - s / omega)
           F[3, 4] = T * (vx * c - vy * s)
       else:
           F[0, 1] = T
           F[2, 3] = T
       return F

   class EKFTracker:
       def __init__(self, T=0.1):
           self.T = T

           # State: [x, vx, y, vy, omega]
           self.x = np.array([0.0, 10.0, 0.0, 5.0, 0.05])
           self.P = np.diag([1.0, 1.0, 1.0, 1.0, 0.01])

           self.Q = np.diag([0.01, 0.1, 0.01, 0.1, 1e-4])
           self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0]])  # measure [x, y]
           self.R = np.eye(2) * 0.5**2

       def step(self, z):
           F = coordinated_turn_jacobian(self.x, self.T)
           pred = ekf_predict(
               self.x, self.P,
               lambda x: coordinated_turn(x, self.T),
               F, self.Q,
           )
           upd = ekf_update(
               pred.x, pred.P, z,
               lambda x: x[[0, 2]],
               self.H, self.R,
           )
           self.x, self.P = upd.x, upd.P
           return self.x

   # Usage: simulate a turning target and track it
   np.random.seed(1)
   ekf = EKFTracker(T=0.1)
   truth = np.array([0.0, 10.0, 0.0, 5.0, 0.1])

   for k in range(100):
       truth = coordinated_turn(truth, 0.1)
       z = truth[[0, 2]] + np.random.randn(2) * 0.5
       est = ekf.step(z)

   print(f"True position:      ({truth[0]:.1f}, {truth[2]:.1f})")
   print(f"Estimated position: ({est[0]:.1f}, {est[2]:.1f})")
   print(f"Estimated turn rate: {est[4]:.3f} (true 0.100)")

Output::

   True position:      (61.2, 88.0)
   Estimated position: (61.6, 88.2)
   Estimated turn rate: 0.096 (true 0.100)

Unscented Kalman Filter
-----------------------

**Problem**: Track a nonlinear system without computing Jacobians.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import ukf_predict, ukf_update

   class UKFTracker:
       def __init__(self, T=0.1, alpha=1e-3, beta=2.0, kappa=0.0):
           """
           Unscented Kalman Filter (no Jacobians needed!)

           alpha, beta, kappa: sigma-point parameters
           """
           self.T = T
           self.alpha = alpha
           self.beta = beta
           self.kappa = kappa

           # State: [x, vx, y, vy]
           self.x = np.array([0.0, 1.0, 0.0, 0.0])
           self.P = np.eye(4) * 0.1

           self.Q = np.eye(4) * 0.01
           self.R = np.eye(2) * 0.5**2

       def motion_model(self, x, dt):
           """Nonlinear motion: could be any complex model"""
           x_new = x.copy()
           x_new[0] += x[1] * dt
           x_new[2] += x[3] * dt
           return x_new

       def measurement_model(self, x):
           """Measure position and speed"""
           return np.array([x[0], np.sqrt(x[1]**2 + x[3]**2)])

       def update(self, measurement):
           def f(x):
               return self.motion_model(x, self.T)

           pred = ukf_predict(
               self.x, self.P, f, self.Q,
               alpha=self.alpha, beta=self.beta, kappa=self.kappa,
           )
           upd = ukf_update(
               pred.x, pred.P, measurement, self.measurement_model, self.R,
               alpha=self.alpha, beta=self.beta, kappa=self.kappa,
           )

           self.x = upd.x
           self.P = upd.P

           return self.x[[0, 2]]

   # Usage: measurements are [position, speed]
   np.random.seed(2)
   ukf = UKFTracker()
   for k in range(5):
       z = np.array([0.1 * (k + 1), 1.0]) + np.random.randn(2) * 0.1
       pos = ukf.update(z)
       print(f"Position estimate: {np.round(pos, 3)}")

Output::

   Position estimate: [0.086 0.   ]
   Position estimate: [0.136 0.   ]
   Position estimate: [ 0.205 -0.   ]
   Position estimate: [ 0.329 -0.   ]
   Position estimate: [ 0.413 -0.   ]

INS/GNSS Navigation
--------------------

**Problem**: Fuse inertial measurements (IMU) with occasional GNSS updates.

**Recipe**: use the shipped strapdown mechanization and loose-coupled
error-state filter from ``pytcl.navigation``. All angles are in radians.

.. code-block:: python

   import numpy as np
   from pytcl.navigation.ins import (
       IMUData,
       earth_rate_ned,
       gravity_ned,
       initialize_ins_state,
   )
   from pytcl.navigation.ins_gnss import (
       GNSSMeasurement,
       initialize_ins_gnss,
       loose_coupled_predict,
       loose_coupled_update_position,
       position_std_to_error_state_units,
   )

   # Initial state: stationary, level, facing north
   lat0, lon0, alt0 = np.radians(40.0), np.radians(-74.0), 100.0
   ins0 = initialize_ins_state(lat=lat0, lon=lon0, alt=alt0)
   state = initialize_ins_gnss(ins0, position_std=10.0, velocity_std=1.0)

   # High-rate IMU (100 Hz for 1 s). A stationary, level IMU senses the
   # gravity reaction on the accelerometers and Earth rotation on the
   # gyros; here we corrupt the accelerometer with a small bias.
   dt = 0.01
   accel_true = -gravity_ned(lat0, alt0)     # specific force, NED
   accel_bias = np.array([0.05, 0.0, 0.0])   # 0.05 m/s^2 north bias
   gyro_true = earth_rate_ned(lat0)

   for _ in range(100):
       imu = IMUData(accel=accel_true + accel_bias, gyro=gyro_true, dt=dt)
       state = loose_coupled_predict(state, imu)

   def north_error_m(state):
       return (state.ins_state.position[0] - lat0) * 6.378e6

   print(f"INS-only north drift after 1 s: {north_error_m(state):.3f} m")

   # Low-rate GNSS position fix (lat, lon, alt) pulls the error back.
   # The innovation is [dlat, dlon, dheight] in (rad, rad, m), so a
   # position accuracy quoted in meters must be converted for the
   # horizontal components:
   horiz = position_std_to_error_state_units(5.0, lat0, alt0)  # 5 m accuracy
   gnss = GNSSMeasurement(
       position=np.array([lat0, lon0, alt0]),
       velocity=None,
       position_cov=np.diag([horiz[0]**2, horiz[1]**2, 10.0**2]),
       velocity_cov=None,
       time=1.0,
   )
   result = loose_coupled_update_position(state, gnss)
   state = result.state

   print(f"North error after GNSS update:  {north_error_m(state):.3f} m")

Output::

   INS-only north drift after 1 s: 0.025 m
   North error after GNSS update:  0.005 m

For velocity fixes use ``loose_coupled_update_velocity``, and for combined
position/velocity fixes use ``loose_coupled_update``. Tightly-coupled
(pseudorange) updates are available via ``tight_coupled_update``.

Particle Filter for Nonlinear Non-Gaussian Systems
---------------------------------------------------

**Problem**: Track a system with non-Gaussian noise or highly nonlinear dynamics.

**Recipe**: ``bootstrap_pf_step`` performs predict + weight update +
adaptive resampling in one call. The process noise is specified as a
*sampler*, so any distribution works -- here heavy-tailed Cauchy noise,
which no Kalman filter can represent:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.particle_filters import (
       bootstrap_pf_step,
       initialize_particles,
       particle_mean,
   )

   rng = np.random.default_rng(42)

   # State [x, vx, y, vy]; measurement is the range from the origin
   x0 = np.array([10.0, 1.0, 5.0, 0.0])
   P0 = np.eye(4)
   state = initialize_particles(x0, P0, N=2000, rng=rng)

   dt = 0.1

   def f(x):
       """Constant-velocity motion (per particle)."""
       return np.array([x[0] + x[1] * dt, x[1], x[2] + x[3] * dt, x[3]])

   def h(x):
       """Nonlinear measurement: distance from origin."""
       return np.array([np.hypot(x[0], x[2])])

   def q_sample(n, rng):
       """Heavy-tailed (Cauchy) process noise."""
       return rng.standard_cauchy((n, 4)) * 0.05

   R = np.array([[0.5**2]])

   true_x = x0.copy()
   for k in range(10):
       true_x = f(true_x)
       z = h(true_x) + rng.normal(0.0, 0.5, 1)
       state = bootstrap_pf_step(
           state.particles, state.weights, z, f, h, q_sample, R, rng=rng,
       )

   estimate = particle_mean(state.particles, state.weights)
   print(f"True range: {h(true_x)[0]:.2f}, estimated range: {h(estimate)[0]:.2f}")

Output::

   True range: 12.08, estimated range: 11.65

Resampling happens automatically when the effective sample size drops below
``resample_threshold * N``; ``particle_covariance`` and
``effective_sample_size`` are available from the same module.

Adaptive Kalman Filtering
--------------------------

**Problem**: Track when the measurement noise varies over time.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update

   class AdaptiveKalmanFilter:
       def __init__(self, T=0.1, sigma_a=0.1):
           # State: [x, vx, y, vy]
           self.x = np.array([0.0, 1.0, 0.0, 1.0])
           self.P = np.eye(4) * 0.1

           self.F = np.array([
               [1, T, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, T],
               [0, 0, 0, 1],
           ], dtype=float)
           self.Q = np.eye(4) * sigma_a**2
           self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]])  # measure [x, y]
           self.R = np.eye(2) * 0.5**2

           # Measurement noise adaptation
           self.innovation_history = []
           self.max_history = 30

       def update(self, measurement):
           # Predict
           pred = kf_predict(self.x, self.P, self.F, self.Q)

           # Compute innovation
           innovation = measurement - self.H @ pred.x
           self.innovation_history.append(np.linalg.norm(innovation))

           # Keep history bounded
           if len(self.innovation_history) > self.max_history:
               self.innovation_history = self.innovation_history[1:]

           # Adapt measurement noise based on innovation statistics
           if len(self.innovation_history) >= 10:
               mean_innovation = np.mean(self.innovation_history)
               std_innovation = np.std(self.innovation_history)

               # Inflate R if innovations get erratic (measurement unreliable)
               if std_innovation > mean_innovation * 2:
                   self.R = np.eye(2) * mean_innovation**2
               else:
                   self.R = np.eye(2) * 0.5**2  # Default

           # Update with adapted R
           upd = kf_update(pred.x, pred.P, measurement, self.H, self.R)
           self.x, self.P = upd.x, upd.P

           return self.x[[0, 2]]

   # Usage
   np.random.seed(3)
   adaptive_kf = AdaptiveKalmanFilter()

   for k in range(20):
       z = np.array([0.1 * (k + 1), 0.1 * (k + 1)]) + np.random.randn(2) * 0.5
       pos = adaptive_kf.update(z)

   print(f"Final position estimate: {np.round(pos, 2)}")

Output::

   Final position estimate: [1.91 1.72]

Batch Processing / Smoothing
-----------------------------

**Problem**: Process all data offline to get the best parameter estimates.

**Recipe**: fit a constant-velocity trajectory to all measurements at once
with ``ordinary_least_squares``:

.. code-block:: python

   import numpy as np
   from pytcl.static_estimation import ordinary_least_squares

   # Truth: [x0, vx, y0, vy]
   truth = np.array([100.0, 5.0, 50.0, -2.0])

   np.random.seed(4)
   T = 0.1
   times = np.arange(20) * T
   measurements = np.column_stack([
       truth[0] + truth[1] * times,
       truth[2] + truth[3] * times,
   ]) + np.random.randn(20, 2) * 0.5

   # Linear model: z = A @ [x0, vx, y0, vy]
   A = np.zeros((2 * len(times), 4))
   A[0::2, 0] = 1.0
   A[0::2, 1] = times
   A[1::2, 2] = 1.0
   A[1::2, 3] = times
   b = measurements.ravel()

   result = ordinary_least_squares(A, b)
   print(f"Estimated [x0, vx, y0, vy]: {np.round(result.x, 2)}")

Output::

   Estimated [x0, vx, y0, vy]: [99.83  5.26 49.98 -2.01]

``ordinary_least_squares`` returns an ``LSResult`` with fields ``x``,
``residuals``, ``rank``, and ``singular_values``. For recursive smoothing
of dynamic systems (RTS and two-filter smoothers), see :doc:`smoothing`.

Data Association with Gating
-----------------------------

**Problem**: Only associate measurements within a statistical gate.

**Recipe**: pytcl ships the gating utilities -- ``chi2_gate_threshold``
for the gate size, ``mahalanobis_distance`` for the squared statistical
distance, and ``gnn_association`` to solve the gated assignment:

.. code-block:: python

   import numpy as np
   from pytcl.assignment_algorithms import (
       chi2_gate_threshold,
       gnn_association,
       mahalanobis_distance,
   )

   # Two predicted track positions with innovation covariances
   track_positions = [np.array([10.0, 0.0]), np.array([0.0, 20.0])]
   S_list = [np.eye(2) * 2.0, np.eye(2) * 2.0]

   measurements = [
       np.array([10.5, 0.3]),    # near track 0
       np.array([-0.2, 19.5]),   # near track 1
       np.array([50.0, 50.0]),   # clutter
   ]

   # 99% gate for a 2-dimensional measurement
   gate = chi2_gate_threshold(0.99, num_dimensions=2)
   print(f"Gate threshold: {gate:.2f}")

   # Cost matrix of squared Mahalanobis distances
   cost = np.zeros((len(track_positions), len(measurements)))
   for i, (pred, S) in enumerate(zip(track_positions, S_list)):
       for j, z in enumerate(measurements):
           cost[i, j] = mahalanobis_distance(z - pred, S)

   # Gated global-nearest-neighbor assignment
   assoc = gnn_association(cost, gate_threshold=gate)
   print(f"Track-to-measurement assignment: {assoc.track_to_measurement}")

Output::

   Gate threshold: 9.21
   Track-to-measurement assignment: [0 1]

The clutter point exceeds the gate for both tracks, so it is left
unassigned. For k-best assignments use ``kbest_assign2d`` (Murty's
algorithm), and for higher-dimensional (multi-scan) assignment problems see
``assign3d`` and ``assignment_nd``.

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`api_navigation` - Function discovery
- :doc:`kalman_filter_tuning` - Parameter tuning
- :doc:`performance_optimization` - Performance tips
- ``examples/`` - More complete examples
