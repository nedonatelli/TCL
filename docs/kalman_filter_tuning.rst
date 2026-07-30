Kalman Filter Tuning Guide
===========================

Overview
--------

This guide explains how to tune Kalman filters for your tracking problem. Proper tuning is critical for filter performance - incorrect noise covariances lead to divergence, missed detections, or excessive smoothing.

Key Sections:

1. **Noise Covariance Estimation** - How to set Q and R
2. **Initialization** - Starting state and covariance selection
3. **Tuning Strategies** - Systematic approaches
4. **Diagnostics** - How to detect filter issues
5. **Common Problems** - Recognition and solutions

Noise Covariance Fundamentals
-----------------------------

**Process Noise (Q):**

Controls how much we expect the target state to deviate from the motion model.

- **Too Small (Q → 0)**: Filter trusts model too much, lags behind maneuvers
- **Too Large (Q → ∞)**: Filter trusts model too little, noisy estimates

**Measurement Noise (R):**

Characterizes sensor measurement accuracy.

- **Too Small (R → 0)**: Filter trusts sensors too much, jerky tracking
- **Too Large (R → ∞)**: Filter ignores measurements, smooth but inaccurate

**Rule of Thumb:**

.. code-block:: python

   # Q: process noise covariance (uncertainty in motion model)
   # For constant velocity model: position doubles each timestep if uncontrolled
   Q = np.diag([
       dt**4/4,  # Position uncertainty (m²)
       dt**2/2,  # Velocity uncertainty (m²/s²)
   ])
   
   # R: measurement noise covariance (sensor accuracy)
   # If sensor has ±0.5m accuracy:
   R = np.array([[0.5**2]])  # Position measurement variance

Estimation Methods
------------------

**Method 1: From Datasheets**

Use manufacturer specifications:

.. code-block:: python

   # GPS accuracy: ±5 meters 95% confidence (~2.5σ)
   # So 1σ ≈ 2.5 meters
   gps_accuracy = 2.5  # meters
   R_gps = np.diag([
       gps_accuracy**2,      # X position
       gps_accuracy**2,      # Y position
   ])

**Method 2: From Historic Data**

Analyze residuals between true state and measurements:

.. code-block:: python

   import numpy as np
   
   # True positions: x_true (from ground truth)
   # Measurements: z_measured
   residuals = z_measured - x_true[:len(z_measured)]
   
   # Estimate R from measurement variance
   R = np.cov(residuals.T)
   print(f"Estimated R:\n{R}")

**Method 3: Adaptive Estimation**

Update noise covariances online using innovations (measurement residuals):

.. code-block:: python

   def adaptive_kalman_filter(x, P, z, F, H, Q, R):
       """EKF with adaptive Q"""
       # Standard EKF prediction
       x_pred = F @ x
       P_pred = F @ P @ F.T + Q
       
       # Innovation
       y = z - H @ x_pred
       S = H @ P_pred @ H.T + R
       
       # Adaptive: increase Q if innovations are large
       innovation_magnitude = np.linalg.norm(y)
       if innovation_magnitude > 3 * np.sqrt(np.trace(S)):
           Q_adapt = Q * 1.5  # Increase process noise
       else:
           Q_adapt = Q
       
       # Continue with standard update...
       return x_new, P_new, Q_adapt

Initialization
--------------

**Initial State (x₀):**

Start with best estimate of target position and velocity:

.. code-block:: python

   # Option 1: Use first measurement
   x0 = np.array([
       z0[0],     # Initial position from first measurement
       0.0,       # Initial velocity (unknown)
   ])
   
   # Option 2: Use two measurements to estimate velocity
   z0 = measurements[0]
   z1 = measurements[1]
   dt = time_steps[1] - time_steps[0]
   
   x0 = np.array([
       z0[0],
       (z1[0] - z0[0]) / dt,  # Velocity estimate
   ])

**Initial Covariance (P₀):**

Reflects uncertainty in initial state:

.. code-block:: python

   import numpy as np
   
   # High uncertainty in velocity (we don't know it yet)
   P0 = np.diag([
       10.0,      # Position uncertainty: ±√10 ≈ ±3.2m
       100.0,     # Velocity uncertainty: ±√100 = ±10 m/s
   ])
   
   # Conservative: Square of initial position uncertainty
   # If measurement has ±5m accuracy:
   pos_uncertainty = 5.0
   P0 = np.diag([pos_uncertainty**2, 1000.0])

Systematic Tuning Strategy
---------------------------

**Step 1: Characterize Measurements**

.. code-block:: python

   import numpy as np
   from scipy import stats
   
   # Analyze measurement noise
   residuals = measurements - true_positions  # If you have ground truth
   
   print(f"Mean: {np.mean(residuals):,.3f}")
   print(f"Std Dev: {np.std(residuals):,.3f}")
   print(f"RMS: {np.sqrt(np.mean(residuals**2)):,.3f}")
   
   # Look for outliers (> 3σ)
   outliers = np.abs(residuals) > 3 * np.std(residuals)
   print(f"Outliers: {np.sum(outliers)} / {len(residuals)}")

**Step 2: Start Conservative**

Begin with high process noise (trusts measurements) and low measurement noise (trusts sensor):

.. code-block:: python

   dt = 0.1  # 10 Hz measurement rate
   
   Q = np.eye(2) * 1.0  # High process noise (high uncertainty in model)
   R = np.eye(1) * 0.1  # Low measurement noise (trust sensor)

**Step 3: Monitor Innovations**

Innovations are the differences between predicted and measured values. They should be zero-mean:

.. code-block:: python

   def compute_innovations(measurements, x_history, P_history, H):
       """Compute normalized innovations for tuning diagnosis"""
       innovations = []
       for z, x, P in zip(measurements, x_history, P_history):
           # Innovation: y = z - H@x
           y = z - H @ x
           
           # Innovation covariance: S = H@P@H' + R
           S = H @ P @ H.T + R
           
           # Normalized innovation (should be ~ N(0,1))
           normalized = y / np.sqrt(np.diag(S))
           innovations.append(normalized)
       
       return np.array(innovations)
   
   # Innovations should have mean ≈ 0, std ≈ 1
   innov = compute_innovations(z_meas, x_hist, P_hist, H)
   print(f"Innovation mean: {np.mean(innov):.4f} (should be ≈ 0)")
   print(f"Innovation std: {np.std(innov):.4f} (should be ≈ 1)")

**Step 4: Adjust Based on Residuals**

- If innovations mean > 0: Filter lags (increase Q or decrease R)
- If innovations std > 1: Filter too confident (decrease Q or increase R)
- If innovations have outliers: robust H∞ filter needed

**Step 5: Validation**

Test on independent data:

.. code-block:: python

   def compute_nees(x_true, x_est, P):
       """Normalized Estimation Error Squared - should be ~1 on average"""
       errors = x_true - x_est
       nees_values = []
       for err, cov in zip(errors, P):
           nees = err.T @ np.linalg.inv(cov) @ err
           nees_values.append(nees)
       return np.array(nees_values)
   
   nees = compute_nees(x_true, x_estimates, P_estimates)
   print(f"NEES mean: {np.mean(nees):.4f}")
   print(f"NEES std: {np.std(nees):.4f}")
   
   # Should be close to 1.0; > 3 indicates inconsistency

Diagnostic Tools
----------------

**Plotting Innovations:**

.. code-block:: python

   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   
   # Plot 1: Innovations over time
   axes[0, 0].plot(innovations)
   axes[0, 0].axhline(0, color='r', linestyle='--')
   axes[0, 0].set_ylabel('Innovation')
   axes[0, 0].set_title('Innovations (should be ~0)')
   
   # Plot 2: Histogram of innovations
   axes[0, 1].hist(innovations.flatten(), bins=30, density=True)
   axes[0, 1].set_title('Innovation Distribution')
   
   # Plot 3: Tracking error
   errors = x_true - x_estimates
   axes[1, 0].plot(errors)
   axes[1, 0].set_ylabel('Error (m)')
   axes[1, 0].set_title('Tracking Error')
   
   # Plot 4: Filter consistency (NEES)
   nees = compute_nees(x_true, x_estimates, P_estimates)
   axes[1, 1].plot(nees)
   axes[1, 1].axhline(1, color='r', linestyle='--')
   axes[1, 1].set_ylabel('NEES')
   axes[1, 1].set_title('Filter Consistency (should be ~1)')
   
   plt.tight_layout()
   plt.show()

Common Problems & Solutions
----------------------------

**Problem 1: Filter Divergence (Error Grows)**

Symptoms: NEES >> 1, innovations increasing

.. code-block:: python

   # Solution: Increase process noise Q
   Q_old = Q
   Q = Q * 10  # Start with 10x increase
   
   # Re-run filter and check NEES

**Problem 2: Lag Behind Maneuvers**

Symptoms: Consistent prediction error after direction change

.. code-block:: python

   # Solution: Increase process noise Q
   # Or use adaptive Q that increases when innovations are large
   
   # Alternative: Better motion model (higher-order)
   # Constant velocity → Constant acceleration

**Problem 3: Jerky Tracking (Follows Noise)**

Symptoms: High-frequency noise in state estimates

.. code-block:: python

   # Solution 1: Increase measurement noise R
   R = R * 10
   
   # Solution 2: Smooth with post-filter
   from scipy.signal import savgol_filter
   x_smooth = savgol_filter(x_estimates, window_length=5, polyorder=2, axis=0)

**Problem 4: Missed Detections**

Symptoms: Occasional measurements not used

.. code-block:: python

   # Solution: Gate measurements
   # Only use if innovation is within expected range
   
   chi2_threshold = 9.21  # 99% confidence for 2-DOF
   
   for z in measurements:
       innovation = z - H @ x_pred
       S = H @ P_pred @ H.T + R
       chi2 = innovation @ np.linalg.inv(S) @ innovation.T
       
       if chi2 < chi2_threshold:
           # Use measurement
           pass
       else:
           # Outlier - skip

**Problem 5: Q or R Values Too Hard to Choose**

Solution: Use Expectation-Maximization (EM) algorithm to learn them automatically:

.. code-block:: python

   # This is complex - use specialized libraries:
   # Option 1: filterpy (Python control filtering library)
   # Option 2: PyMC3 (Bayesian inference)
   
   from filterpy.kalman import KalmanFilter
   # filterpy has built-in EM for Q/R estimation

Filter Selection Guide
----------------------

Choose appropriate filter based on motion model linearity:

================  ========  ============  ==========
Filter Type       Linear    Nonlinear     GPU Ready
================  ========  ============  ==========
Kalman Filter     ✓         ✗             Yes
Extended KF       ✓         ✓             Yes
Unscented KF      ✓         ✓             Yes
Cubature KF       ✓         ✓             Yes
Particle Filter   ✓         ✓             Yes (large N)
================  ========  ============  ==========

Example: Tuning for GPS Tracking
---------------------------------

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update
   
   # 2D position tracking with GPS
   # State: [x, y, vx, vy]
   # Measurement: [x, y] (GPS)
   
   dt = 1.0  # 1 second between measurements
   
   # Motion model: constant velocity
   F = np.array([
       [1, 0, dt, 0],
       [0, 1, 0, dt],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
   ])
   
   # Measurement model: observe position only
   H = np.array([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
   ])
   
   # Process noise: target acceleration uncertainty
   sigma_a = 0.5  # m/s² acceleration noise
   Q = np.diag([
       (sigma_a * dt**4) / 4,
       (sigma_a * dt**4) / 4,
       (sigma_a * dt**2) / 2,
       (sigma_a * dt**2) / 2,
   ])
   
   # Measurement noise: GPS accuracy ±5m
   sigma_gps = 5.0
   R = np.diag([sigma_gps**2, sigma_gps**2])
   
   # Initial state and covariance
   x0 = np.array([0, 0, 0, 0])  # Start at origin, no velocity
   P0 = np.diag([25, 25, 100, 100])  # High uncertainty in velocity
   
   # Run filter
   x = x0
   P = P0
   for z in measurements:
       pred = ekf_predict(x, P, lambda s: F @ s, F, Q)
       upd = ekf_update(pred.x, pred.P, z, lambda s: H @ s, H, R)
       x, P = upd.x, upd.P
       print(f"Position: ({x[0]:.2f}, {x[1]:.2f}), Velocity: ({x[2]:.2f}, {x[3]:.2f})")

References
----------

- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation
- Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
- Bierman, G. J. (1977). Factorization Methods for Discrete Sequential Estimation

See Also
~~~~~~~~

- :doc:`gpu_acceleration` - GPU-accelerated filtering
- Module: ``pytcl.dynamic_estimation.kalman``
- Examples: ``examples/kalman_filter_comparison.py``
