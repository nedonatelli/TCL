Common Use Cases & Recipes
==========================

This guide provides ready-to-use code for common tracking and estimation scenarios. Each recipe is self-contained and can be adapted to your needs.

See :doc:`architecture` for module organization and :doc:`api_navigation` for function discovery.

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
           sigma_a: acceleration standard deviation (m/s²)
           meas_std: measurement standard deviation (meters)
           """
           self.T = T
           
           # State: [x, vx, y, vy]
           self.x = np.array([0.0, 0.0, 0.0, 0.0])
           self.P = np.eye(4) * 1.0  # Initial uncertainty
           
           # Motion model (constant velocity)
           self.F = f_constant_velocity(T, num_dims=2)
           self.Q = q_constant_velocity(T, sigma_a, num_dims=2)
           
           # Measurement model (observe position only)
           self.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]])
           self.R = np.eye(2) * meas_std**2
       
       def process_measurement(self, range_bearing):
           """
           Process a range/bearing measurement (from polar: range, azimuth)
           Convert to Cartesian and update filter
           """
           # Predict
           x_pred, P_pred = kf_predict(self.x, self.P, self.F, self.Q)
           
           # Convert measurement from polar to Cartesian
           # Input: [range, azimuth, elevation] → convert to [x, y, z]
           cart_meas = sphere2cart(range_bearing)[:2]  # Take x, y only
           
           # Update
           self.x, self.P = kf_update(x_pred, P_pred, cart_meas, self.H, self.R)
           
           return self.x[:2], np.sqrt(np.diag(self.P)[:2])
   
   # Usage
   tracker = SimpleTracker()
   
   # Process measurements over time
   measurements = [
       np.array([100.0, 0.0, 0.0]),      # range=100, az=0°, el=0°
       np.array([100.5, 0.01, 0.0]),
       np.array([101.2, 0.02, 0.0]),
   ]
   
   for meas in measurements:
       pos, std = tracker.process_measurement(meas)
       print(f"Position: {pos}, Std: {std}")

Multi-Target Tracking with Assignment
--------------------------------------

**Problem**: Track multiple targets with ambiguous measurements using Global Nearest Neighbor (GNN).

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.trackers.multi_tracker_gnn import GNNTracker
   from pytcl.containers import TrackSet, Track
   from pytcl.coordinate_systems.conversions import sphere2cart
   
   class MultiTargetSystem:
       def __init__(self, gate_threshold=5.0):
           self.tracker = GNNTracker(
               gate_threshold=gate_threshold,
               min_observations=2,  # Need 2 observations to confirm track
               max_age=10           # Delete track if not observed for 10 steps
           )
       
       def process_frame(self, measurements_polar):
           """
           Process one frame of measurements
           Input: measurements_polar - list of [range, azimuth, elevation]
           """
           # Convert measurements to Cartesian
           measurements_cart = np.array([
               sphere2cart(m)[:2] for m in measurements_polar
           ])
           
           # Update tracker
           self.tracker.update(measurements_cart)
           
           # Get track information
           confirmed_tracks = self.tracker.tracks['confirmed']
           
           results = []
           for track_id, track in confirmed_tracks.items():
               pos = track.state[:2]
               vel = track.state[2:4]
               results.append({
                   'id': track_id,
                   'position': pos,
                   'velocity': vel,
                   'age': track.age
               })
           
           return results
   
   # Usage
   mtt = MultiTargetSystem()
   
   # Process multiple frames per second
   for frame_idx in range(100):
       # Simulate measurements (3 targets)
       measurements = np.array([
           [100 + frame_idx*0.5, 0.1, 0.0],      # Target 1
           [150 - frame_idx*0.3, 1.5, 0.0],      # Target 2
           [120, 0.5, 0.0],                       # False alarm
       ])
       
       tracks = mtt.process_frame(measurements)
       print(f"Frame {frame_idx}: {len(tracks)} confirmed tracks")

Extended Kalman Filter with Nonlinear Dynamics
-----------------------------------------------

**Problem**: Track with nonlinear motion (e.g., coordinated turn model).

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import extended_kalman_filter
   from pytcl.coordinate_systems.jacobians import jacobian_cart2sphere
   
   def coordinated_turn_motion(x, T, omega):
       """
       Coordinated turn model transition
       x = [x, vx, y, vy, omega]
       """
       x_pred = x.copy()
       if abs(omega) > 1e-6:
           x_pred[0] = x[0] + x[1]/omega * (np.sin(omega*T)) + x[2]/omega * (np.cos(omega*T) - 1)
           x_pred[1] = x[1] * np.cos(omega*T) - x[2] * np.sin(omega*T)
           x_pred[2] = x[2] + x[1]/omega * (1 - np.cos(omega*T)) + x[2]/omega * np.sin(omega*T)
           x_pred[3] = x[1] * np.sin(omega*T) + x[2] * np.cos(omega*T)
       else:
           # Straight line if omega ≈ 0
           x_pred[:4] += x[1:5] * T
       
       return x_pred
   
   def jacobian_motion(x, T, omega):
       """Jacobian of motion model"""
       F = np.eye(5)
       if abs(omega) > 1e-6:
           F[0, 1] = np.sin(omega * T) / omega
           F[0, 3] = (np.cos(omega * T) - 1) / omega
           F[1, 1] = np.cos(omega * T)
           F[1, 3] = -np.sin(omega * T)
           F[2, 1] = (1 - np.cos(omega * T)) / omega
           F[2, 3] = np.sin(omega * T) / omega
           F[3, 1] = np.sin(omega * T)
           F[3, 3] = np.cos(omega * T)
       else:
           F[:4, 1:3] = T * np.eye(2)
       
       return F
   
   class EKFTracker:
       def __init__(self, T=0.1, omega=0.1):
           self.T = T
           self.omega = omega  # Turn rate
           
           # State: [x, vx, y, vy, omega]
           self.x = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
           self.P = np.eye(5) * 0.1
           
           self.Q = np.eye(5) * 0.01
           self.H = np.eye(2, 5)  # Measure position
           self.R = np.eye(2) * 0.5**2
       
       def update(self, measurement):
           # Motion model
           def f_motion(x):
               return coordinated_turn_motion(x, self.T, self.omega)
           
           def F_jacobian(x):
               return jacobian_motion(x, self.T, self.omega)
           
           # Measurement model
           def h_measure(x):
               return x[:2]
           
           def H_jacobian(x):
               return self.H
           
           # Extended Kalman filter step
           result = extended_kalman_filter(
               self.x, self.P, measurement, self.Q, self.R,
               f_motion, F_jacobian, h_measure, H_jacobian
           )
           
           self.x = result.x
           self.P = result.P
           
           return self.x[:2]
   
   # Usage
   ekf = EKFTracker()
   for t in range(100):
       # Simulated measurement
       z = np.array([100 + t*0.5, 50 + t*0.2]) + np.random.randn(2) * 0.5
       pos = ekf.update(z)
       print(f"Position: {pos}")

Unscented Kalman Filter
-----------------------

**Problem**: Track nonlinear system without computing Jacobians.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import unscented_kalman_filter
   
   class UKFTracker:
       def __init__(self, T=0.1, alpha=1e-3, beta=2.0, kappa=1.0):
           """
           Unscented Kalman Filter (no Jacobians needed!)
           
           alpha, beta, kappa: UKF parameters
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
           # Predict
           def f(x):
               return self.motion_model(x, self.T)
           
           # UKF predict
           result = unscented_kalman_filter(
               self.x, self.P, measurement,
               self.Q, self.R, f,
               measurement_model=self.measurement_model,
               alpha=self.alpha, beta=self.beta, kappa=self.kappa
           )
           
           self.x = result.x
           self.P = result.P
           
           return self.x[:2]
   
   # Usage
   ukf = UKFTracker()
   for measurement in measurements:
       pos = ukf.update(measurement)
       print(f"Position: {pos}")

INS/GNSS Navigation
--------------------

**Problem**: Fuse inertial measurements (IMU) with occasional GNSS updates.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.navigation.ins_gnss import ins_update_discrete, gnss_update_discrete
   from pytcl.coordinate_systems.conversions import geodetic2ecef
   
   class INSGNSSFusion:
       def __init__(self):
           """Initialize navigation state"""
           # State: position (lat, lon, alt), velocity (NED), attitude (quaternion)
           self.lat = 40.0  # degrees
           self.lon = -74.0
           self.alt = 0.0
           
           self.v_north = 0.0  # m/s
           self.v_east = 0.0
           self.v_down = 0.0
           
           # Covariance
           self.P_pos = np.eye(3) * 10**2  # 10m std
           self.P_vel = np.eye(3) * 1**2   # 1 m/s std
       
       def process_imu(self, accel, gyro, dt):
           """Process IMU data (high rate, ~100 Hz)"""
           # Simple integration (real system would use quaternion kinematics)
           self.v_north += accel[0] * dt
           self.v_east += accel[1] * dt
           self.v_down += accel[2] * dt
           
           # Update position
           self.lat += self.v_north * dt / 111320  # approx meters to degrees
           self.lon += self.v_east * dt / (111320 * np.cos(np.radians(self.lat)))
           self.alt += -self.v_down * dt
           
           # Increase position uncertainty (process noise)
           self.P_pos += np.eye(3) * 0.01
           self.P_vel += np.eye(3) * 0.1
       
       def process_gnss(self, gnss_pos_ecef, gnss_std=5.0):
           """Process GNSS update (low rate, ~1-10 Hz)"""
           # Measurement: position in ECEF
           H = np.eye(3)
           R = np.eye(3) * gnss_std**2
           
           # Simple update (combine with ECEF state in real system)
           innovation = gnss_pos_ecef - self.get_ecef_position()
           K = self.P_pos @ H.T @ np.linalg.inv(H @ self.P_pos @ H.T + R)
           
           # Update position covariance
           self.P_pos = (np.eye(3) - K @ H) @ self.P_pos
           
           # In real system, would update full state
       
       def get_ecef_position(self):
           """Convert geodetic to ECEF"""
           geodetic = np.array([self.lat, self.lon, self.alt])
           return geodetic2ecef(geodetic)
   
   # Usage
   fusion = INSGNSSFusion()
   
   # High-rate IMU (100 Hz)
   imu_data = [(accel, gyro) for accel, gyro in imu_buffer]
   for accel, gyro in imu_data:
       fusion.process_imu(accel, gyro, dt=0.01)
   
   # Low-rate GNSS (1 Hz)
   if gnss_available:
       fusion.process_gnss(gnss_position)

Particle Filter for Nonlinear Non-Gaussian Systems
---------------------------------------------------

**Problem**: Track system with non-Gaussian noise or highly nonlinear dynamics.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.particle_filters import particle_filter
   
   class ParticleFilterTracker:
       def __init__(self, n_particles=1000):
           """Initialize particle filter"""
           self.n_particles = n_particles
           
           # Particles: [x, vx, y, vy] for each particle
           self.particles = np.random.randn(n_particles, 4) * 0.1
           self.weights = np.ones(n_particles) / n_particles
       
       def motion_model(self, x, noise_scale=0.1):
           """Nonlinear motion with Cauchy-distributed noise (heavy tails)"""
           # Standard motion
           x_pred = x.copy()
           x_pred[0] += x[1] * 0.1
           x_pred[2] += x[3] * 0.1
           
           # Add heavy-tailed noise (Cauchy instead of Gaussian)
           x_pred += np.random.standard_cauchy(4) * noise_scale
           
           return x_pred
       
       def measurement_model(self, x):
           """Nonlinear measurement: distance from origin"""
           return np.sqrt(x[0]**2 + x[2]**2)
       
       def update(self, measurement, meas_std=1.0):
           """Particle filter update step"""
           # Predict all particles
           self.particles = np.array([
               self.motion_model(p) for p in self.particles
           ])
           
           # Compute likelihood for each particle
           predicted_meas = np.array([
               self.measurement_model(p) for p in self.particles
           ])
           
           # Gaussian likelihood
           likelihood = np.exp(-(predicted_meas - measurement)**2 / (2 * meas_std**2))
           
           # Update weights
           self.weights = likelihood * self.weights
           self.weights /= np.sum(self.weights)  # Normalize
           
           # Resample if necessary (effective sample size)
           n_eff = 1 / np.sum(self.weights**2)
           if n_eff < self.n_particles / 2:
               # Low effective sample size → resample
               indices = np.random.choice(self.n_particles, size=self.n_particles, 
                                         p=self.weights)
               self.particles = self.particles[indices]
               self.weights = np.ones(self.n_particles) / self.n_particles
           
           # Estimate state as weighted average
           return np.average(self.particles, axis=0, weights=self.weights)
   
   # Usage
   pf = ParticleFilterTracker(n_particles=2000)
   
   for measurement in measurements:
       state_estimate = pf.update(measurement)
       print(f"Position: {state_estimate[:2]}")

Adaptive Kalman Filtering
--------------------------

**Problem**: Track when measurement noise σ varies over time.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
   
   class AdaptiveKalmanFilter:
       def __init__(self, T=0.1, sigma_a=0.1):
           self.x = np.array([0.0, 1.0, 0.0, 1.0])
           self.P = np.eye(4) * 0.1
           
           self.F = np.array([
               [1, T, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, T],
               [0, 0, 0, 1]
           ])
           self.Q = np.eye(4) * sigma_a**2
           self.H = np.eye(2, 4)
           
           # Measurement noise adaptation
           self.innovation_history = []
           self.max_history = 30
       
       def update(self, measurement):
           # Predict
           x_pred, P_pred = kf_predict(self.x, self.P, self.F, self.Q)
           
           # Compute innovation
           innovation = measurement - self.H @ x_pred
           self.innovation_history.append(np.linalg.norm(innovation))
           
           # Keep history bounded
           if len(self.innovation_history) > self.max_history:
               self.innovation_history = self.innovation_history[1:]
           
           # Adapt measurement noise based on innovation
           if len(self.innovation_history) >= 10:
               mean_innovation = np.mean(self.innovation_history)
               std_innovation = np.std(self.innovation_history)
               
               # Increase R if innovations getting larger (measurement unreliable)
               if std_innovation > mean_innovation * 2:
                   self.R = np.eye(2) * (mean_innovation ** 2)
               else:
                   self.R = np.eye(2) * 0.5**2  # Default
           else:
               self.R = np.eye(2) * 0.5**2
           
           # Update with adapted R
           self.x, self.P = kf_update(x_pred, P_pred, measurement, self.H, self.R)
           
           return self.x[:2]
   
   # Usage
   adaptive_kf = AdaptiveKalmanFilter()
   
   for measurement in measurements:
       pos = adaptive_kf.update(measurement)
       print(f"Position: {pos}")

Batch Processing / Smoothing
-----------------------------

**Problem**: Process all data offline to get the best parameter estimates.

**Recipe**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.batch_estimation import least_squares_batch
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity
   
   class BatchProcessor:
       def __init__(self, measurements, T=0.1):
           """
           Batch process all measurements at once (offline smoothing)
           """
           self.measurements = measurements
           self.T = T
       
       def process(self):
           """Batch least-squares estimation"""
           n = len(self.measurements)
           n_states = n * 4  # [x, vx, y, vy] for each time step
           
           # Build state transition matrix
           F = f_constant_velocity(self.T, num_dims=2)
           
           # Build full system for batch
           Phi = np.eye(n_states)
           for i in range(1, n):
               Phi[i*4:(i+1)*4, (i-1)*4:i*4] = F
           
           # Build measurement matrix
           H_full = np.zeros((n*2, n*4))
           for i in range(n):
               H_full[i*2:(i+1)*2, i*4:(i+1)*4] = np.eye(2, 4)
           
           # Build measurement vector
           z = self.measurements.flatten()
           
           # Solve: x = (H^T R^-1 H)^-1 H^T R^-1 z
           R_inv = np.eye(n*2) * (1 / 0.5**2)
           HTR = H_full.T @ R_inv
           
           x_est = np.linalg.solve(HTR @ H_full, HTR @ z)
           
           return x_est.reshape(n, 4)
   
   # Usage
   all_measurements = np.array([
       [100.0, 50.0], [100.5, 50.2], [101.1, 50.4], ...
   ])
   
   batch = BatchProcessor(all_measurements)
   trajectory = batch.process()
   print(f"Estimated trajectory shape: {trajectory.shape}")

Data Association with Gating
-----------------------------

**Problem**: Only associate measurements within a statistical gate.

**Recipe**:

.. code-block:: python

   import numpy as np
   from scipy.spatial.distance import mahalanobis, cdist
   
   class GatedAssociation:
       def __init__(self, gate_chi2=12.592):
           """
           Gate based on chi-square test
           gate_chi2=12.592 → 99.5% confidence interval for 2-DOF
           """
           self.gate_chi2 = gate_chi2
       
       def associate_measurements(self, tracks, measurements, P_innovation_list):
           """
           Associate using Mahalanobis distance with gating
           """
           n_tracks = len(tracks)
           n_meas = len(measurements)
           
           # Cost matrix
           cost = np.full((n_tracks, n_meas), np.inf)
           
           for i, track in enumerate(tracks):
               for j, meas in enumerate(measurements):
                   # Predicted position
                   pred = track[:2]
                   
                   # Innovation covariance
                   S = P_innovation_list[i]
                   
                   # Mahalanobis distance
                   innovation = meas - pred
                   dist_squared = innovation @ np.linalg.inv(S) @ innovation
                   
                   # Gate check: chi-square test
                   if dist_squared <= self.gate_chi2:
                       cost[i, j] = dist_squared
           
           return cost
   
   # Usage
   gating = GatedAssociation()
   
   cost_matrix = gating.associate_measurements(
       track_predictions, measurements, S_list
   )
   
   # Now solve assignment on gated cost matrix
   from pytcl.assignment.optimization import assignment_nd
   assignments = assignment_nd(cost_matrix)

See Also
--------

- :doc:`architecture` - Module organization
- :doc:`api_navigation` - Function discovery
- :doc:`kalman_filter_tuning` - Parameter tuning
- :doc:`performance_optimization` - Performance tips
- ``examples/`` - More complete examples
