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

For a constant velocity model driven by white acceleration noise with standard
deviation ``sigma_a``, the discrete process noise is

.. math::

   Q = \sigma_a^2
   \begin{bmatrix} T^4/4 & T^3/2 \\ T^3/2 & T^2 \end{bmatrix}

per spatial dimension. Note the :math:`\sigma_a^2` factor: Q is a covariance,
so the noise *variance* scales it, not the standard deviation. Use the shipped
builders from :mod:`pytcl.dynamic_models` instead of hand-rolling this:

.. code-block:: python

    import numpy as np
    from pytcl.dynamic_models import q_constant_velocity

    dt = 1.0
    sigma_a = 0.5  # acceleration noise std [m/s^2]

    # State [x, vx]
    Q = q_constant_velocity(T=dt, sigma_a=sigma_a, num_dims=1)
    print(Q)
    # [[0.0625 0.125 ]
    #  [0.125  0.25  ]]

    # R: measurement noise covariance (sensor accuracy)
    # If sensor has +/-2.5 m (1-sigma) accuracy:
    R = np.array([[2.5**2]])

``q_constant_acceleration``, ``q_singer``, and ``q_coord_turn_2d`` cover
higher-order and maneuvering models.

Estimation Methods
------------------

**Method 1: From Datasheets**

Use manufacturer specifications:

.. code-block:: python

    # GPS accuracy: +/-5 meters 95% confidence (~2 sigma)
    # So 1 sigma ~ 2.5 meters
    gps_accuracy = 2.5  # meters
    R_gps = np.diag([
        gps_accuracy**2,      # X position
        gps_accuracy**2,      # Y position
    ])

**Method 2: From Historic Data**

Analyze residuals between true state and measurements, e.g. from a calibration
run against a surveyed target:

.. code-block:: python

    rng = np.random.default_rng(7)
    true_range = 50.0  # surveyed target position [m]
    z_measured = true_range + rng.normal(0.0, 2.5, size=2000)

    residuals = z_measured - true_range

    # Estimate R from measurement variance
    R = np.atleast_2d(np.var(residuals))
    print(f"Estimated R: {R[0, 0]:.3f} (true value 2.5**2 = 6.25)")
    # Estimated R: 6.060 (true value 2.5**2 = 6.25)

**Method 3: Adaptive Estimation**

Inflate R online when the innovation is implausibly large under the current
model (a chi-squared test on the normalized innovation squared). Built from
the real ``kf_predict`` / ``kf_update`` in :mod:`pytcl.dynamic_estimation.kalman`:

.. code-block:: python

    from pytcl.assignment_algorithms import chi2_gate_threshold
    from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
    from pytcl.dynamic_models import f_constant_velocity

    def adaptive_r_step(x, P, z, F, Q, H, R, inflate=10.0):
        """One KF cycle; inflates R when the innovation is an outlier."""
        pred = kf_predict(x, P, F, Q)
        y = z - H @ pred.x
        S = H @ pred.P @ H.T + R
        nis_value = float(y @ np.linalg.inv(S) @ y)
        if nis_value > chi2_gate_threshold(0.99, len(y)):
            R_used = R * inflate  # distrust this measurement
        else:
            R_used = R
        upd = kf_update(pred.x, pred.P, z, H, R_used)
        return upd.x, upd.P, R_used

    dt = 1.0
    F = f_constant_velocity(dt, num_dims=1)  # state [x, vx]
    Q = q_constant_velocity(dt, sigma_a=0.5, num_dims=1)
    H = np.array([[1.0, 0.0]])
    R = np.array([[6.25]])

    rng = np.random.default_rng(0)
    z_track = 10.0 * np.arange(100) + rng.normal(0.0, 2.5, size=100)
    z_track[[30, 60]] += 40.0  # two multipath spikes

    # Initialize position and velocity from the first two measurements
    x = np.array([z_track[1], (z_track[1] - z_track[0]) / dt])
    P = np.diag([6.25, 12.5])
    inflated_at = []
    for k, z in enumerate(z_track[2:], start=2):
        x, P, R_used = adaptive_r_step(x, P, np.atleast_1d(z), F, Q, H, R)
        if R_used[0, 0] > R[0, 0]:
            inflated_at.append(k)

    print(f"R inflated at steps: {inflated_at}")
    # R inflated at steps: [30, 60]
    print(f"Final state: {np.round(x, 2)}")
    # Final state: [987.     9.16]

Initialization
--------------

**Initial State (x0):**

Start with best estimate of target position and velocity:

.. code-block:: python

    # Option 1: Use first measurement, unknown velocity
    x0 = np.array([
        z_track[0],  # initial position from first measurement
        0.0,         # initial velocity (unknown)
    ])

    # Option 2: Use two measurements to estimate velocity
    x0 = np.array([
        z_track[0],
        (z_track[1] - z_track[0]) / dt,  # velocity estimate
    ])
    print(np.round(x0, 2))
    # [0.31 9.36]

**Initial Covariance (P0):**

Reflects uncertainty in initial state:

.. code-block:: python

    # High uncertainty in velocity (we don't know it yet)
    P0 = np.diag([
        10.0,      # position uncertainty: +/-sqrt(10) ~ +/-3.2 m
        100.0,     # velocity uncertainty: +/-sqrt(100) = +/-10 m/s
    ])

    # Conservative: square of initial position uncertainty
    # If measurement has +/-5 m accuracy:
    pos_uncertainty = 5.0
    P0 = np.diag([pos_uncertainty**2, 1000.0])

Systematic Tuning Strategy
---------------------------

**Step 1: Characterize Measurements**

.. code-block:: python

    # Analyze measurement noise (calibration data from Method 2)
    print(f"Mean: {np.mean(residuals):.3f}")
    print(f"Std Dev: {np.std(residuals):.3f}")
    print(f"RMS: {np.sqrt(np.mean(residuals**2)):.3f}")
    # Mean: -0.100
    # Std Dev: 2.462
    # RMS: 2.464

    # Look for outliers (> 3 sigma)
    outliers = np.abs(residuals) > 3 * np.std(residuals)
    print(f"Outliers: {np.sum(outliers)} / {len(residuals)}")
    # Outliers: 2 / 2000

**Step 2: Start Conservative**

Begin with high process noise (trusts measurements) and low measurement noise (trusts sensor):

.. code-block:: python

    dt = 0.1  # 10 Hz measurement rate

    Q = np.eye(2) * 1.0  # high process noise (high uncertainty in model)
    R = np.eye(1) * 0.1  # low measurement noise (trust sensor)

**Step 3: Monitor Innovations (NIS)**

Innovations are the differences between predicted and measured values. The
Normalized Innovation Squared (NIS) should be chi-squared distributed with
the measurement dimension ``m`` as degrees of freedom, so its mean should be
close to ``m``. ``kf_update`` already returns the innovation ``y`` and its
covariance ``S``; feed them to :func:`~pytcl.performance_evaluation.nis_sequence`:

.. code-block:: python

    from pytcl.performance_evaluation import consistency_test, nis_sequence

    # Simulate a 1D constant velocity target and run the filter
    dt = 1.0
    sigma_a, sigma_z = 0.5, 2.5
    rng = np.random.default_rng(42)
    n_steps = 200

    truth = np.zeros((n_steps, 2))  # [x, vx]
    truth[0] = [0.0, 1.0]
    for k in range(1, n_steps):
        accel = rng.normal(0.0, sigma_a)
        truth[k, 0] = truth[k - 1, 0] + truth[k - 1, 1] * dt + 0.5 * accel * dt**2
        truth[k, 1] = truth[k - 1, 1] + accel * dt
    z_meas = truth[:, 0] + rng.normal(0.0, sigma_z, size=n_steps)

    F = f_constant_velocity(dt, num_dims=1)
    Q = q_constant_velocity(T=dt, sigma_a=sigma_a, num_dims=1)
    H = np.array([[1.0, 0.0]])
    R = np.array([[sigma_z**2]])

    x = np.array([z_meas[0], 0.0])
    P = np.diag([sigma_z**2, 10.0])
    x_hist, P_hist, innovations, innovation_covs = [], [], [], []
    for z in z_meas[1:]:
        pred = kf_predict(x, P, F, Q)
        upd = kf_update(pred.x, pred.P, np.atleast_1d(z), H, R)
        x, P = upd.x, upd.P
        x_hist.append(upd.x)
        P_hist.append(upd.P)
        innovations.append(upd.y)
        innovation_covs.append(upd.S)
    x_hist, P_hist = np.array(x_hist), np.array(P_hist)
    innovations = np.array(innovations)
    innovation_covs = np.array(innovation_covs)

    nis_values = nis_sequence(innovations, innovation_covs)
    print(f"NIS mean: {np.mean(nis_values):.3f} (should be close to m = 1)")
    # NIS mean: 0.959 (should be close to m = 1)

    result = consistency_test(nis_values, df=1)
    print(f"Consistent: {result.is_consistent}, "
          f"95% bounds: [{result.lower_bound:.3f}, {result.upper_bound:.3f}]")
    # Consistent: True, 95% bounds: [0.813, 1.206]

**Step 4: Adjust Based on Innovations**

- If NIS mean > m: filter is overconfident (increase Q or R)
- If NIS mean < m: filter is underconfident (decrease Q or R)
- If NIS has isolated spikes: gate outliers (see Problem 4 below)

**Step 5: Validation (NEES)**

With ground truth (simulation or instrumented test range), check the
Normalized Estimation Error Squared. A consistent filter has average NEES
close to the state dimension ``n``:

.. code-block:: python

    from pytcl.performance_evaluation import average_nees, nees_sequence

    nees_values = nees_sequence(truth[1:], x_hist, P_hist)
    print(f"NEES mean: {average_nees(truth[1:], x_hist, P_hist):.3f} "
          f"(should be close to n = 2)")
    # NEES mean: 1.948 (should be close to n = 2)

    result = consistency_test(nees_values, df=2)
    print(f"Consistent: {result.is_consistent}, "
          f"95% bounds: [{result.lower_bound:.3f}, {result.upper_bound:.3f}]")
    # Consistent: True, 95% bounds: [1.732, 2.287]

.. note::

   The ``consistency_test`` bounds assume independent samples. NEES values
   from a single filter run are correlated, so treat single-run results as
   indicative; for a rigorous test, average NEES across independent Monte
   Carlo runs.

Diagnostic Tools
----------------

**Plotting Innovations:**

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Innovations over time
    axes[0, 0].plot(innovations[:, 0])
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_ylabel('Innovation')
    axes[0, 0].set_title('Innovations (should be ~0 mean)')

    # Plot 2: Histogram of innovations
    axes[0, 1].hist(innovations.flatten(), bins=30, density=True)
    axes[0, 1].set_title('Innovation Distribution')

    # Plot 3: Tracking error
    axes[1, 0].plot(truth[1:, 0] - x_hist[:, 0])
    axes[1, 0].set_ylabel('Error (m)')
    axes[1, 0].set_title('Position Error')

    # Plot 4: Filter consistency (NEES)
    axes[1, 1].plot(nees_values)
    axes[1, 1].axhline(2, color='r', linestyle='--')
    axes[1, 1].set_ylabel('NEES')
    axes[1, 1].set_title('Filter Consistency (should be ~n)')

    plt.tight_layout()
    plt.show()

Common Problems & Solutions
----------------------------

**Problem 1: Filter Divergence (Error Grows)**

Symptoms: NEES >> n, innovations increasing

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
    # Constant velocity -> constant acceleration
    from pytcl.dynamic_models import f_constant_acceleration, q_constant_acceleration

    F_ca = f_constant_acceleration(T=1.0, num_dims=1)   # state [x, vx, ax]
    Q_ca = q_constant_acceleration(T=1.0, sigma_j=0.1, num_dims=1)

**Problem 3: Jerky Tracking (Follows Noise)**

Symptoms: High-frequency noise in state estimates

.. code-block:: python

    # Solution 1: Increase measurement noise R
    R = R * 10

    # Solution 2: Smooth with post-filter
    from scipy.signal import savgol_filter
    x_smooth = savgol_filter(x_hist, window_length=5, polyorder=2, axis=0)

**Problem 4: Outlier Measurements**

Symptoms: Occasional spikes drag the estimate away

.. code-block:: python

    # Solution: Gate measurements before updating.
    # The threshold is a chi-squared quantile for the measurement dimension.
    from pytcl.assignment_algorithms import chi2_gate_threshold, ellipsoidal_gate

    gate = chi2_gate_threshold(0.99, num_dimensions=1)
    print(round(gate, 2))  # 6.63

    pred = kf_predict(x, P, F, Q)
    S = H @ pred.P @ H.T + R
    z_pred = H @ pred.x

    for offset in (1.0, -2.0, 40.0):
        z = z_pred + offset
        if ellipsoidal_gate(z - z_pred, S, gate):
            print(f"offset {offset:+.1f}: accept")
        else:
            print(f"offset {offset:+.1f}: reject (outlier)")
    # offset +1.0: accept
    # offset -2.0: accept
    # offset +40.0: reject (outlier)

**Problem 5: Q or R Values Too Hard to Choose**

Solution: learn them from data with the Expectation-Maximization (EM)
algorithm. This is not built into ``pytcl`` (or ``filterpy``); the ``pykalman``
library ships it as ``KalmanFilter.em()``. Alternatively, grid-search Q and R
and pick the combination whose NIS/NEES statistics are closest to their
expected chi-squared behavior (Steps 3 and 5 above).

Filter Selection Guide
----------------------

Choose appropriate filter based on motion model linearity. "GPU" means a
CUDA implementation exists in :mod:`pytcl.gpu` (kalman, ekf, ukf,
particle_filter); the cubature filter is CPU-only
(``ckf_predict`` / ``ckf_update`` in :mod:`pytcl.dynamic_estimation`).

================  ========  ============  ==========
Filter Type       Linear    Nonlinear     GPU
================  ========  ============  ==========
Kalman Filter     ✓         ✗             Yes
Extended KF       ✓         ✓             Yes
Unscented KF      ✓         ✓             Yes
Cubature KF       ✓         ✓             No
Particle Filter   ✓         ✓             Yes (large N)
================  ========  ============  ==========

Example: Tuning for GPS Tracking
---------------------------------

.. code-block:: python

    from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

    # 2D position tracking with GPS
    # State: [x, vx, y, vy]
    # Measurement: [x, y] (GPS)

    dt = 1.0  # 1 second between measurements

    # Motion model: constant velocity
    F = f_constant_velocity(T=dt, num_dims=2)

    # Process noise: target acceleration uncertainty.
    # Q scales with sigma_a**2 (variance), built per spatial dimension.
    sigma_a = 0.5  # m/s^2 acceleration noise
    Q = q_constant_velocity(T=dt, sigma_a=sigma_a, num_dims=2)

    # Measurement model: observe position only
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    # Measurement noise: GPS accuracy +/-5 m (1 sigma)
    sigma_gps = 5.0
    R = np.diag([sigma_gps**2, sigma_gps**2])

    gps_fixes = np.array([
        [2.1, -3.7],
        [11.8, 4.2],
        [19.5, 10.6],
        [30.2, 16.1],
        [41.0, 24.3],
    ])

    # Initialize from the first fix, with high velocity uncertainty
    x = np.array([gps_fixes[0, 0], 0.0, gps_fixes[0, 1], 0.0])
    P = np.diag([sigma_gps**2, 100.0, sigma_gps**2, 100.0])

    for z in gps_fixes[1:]:
        pred = kf_predict(x, P, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)
        x, P = upd.x, upd.P
        print(f"position: ({x[0]:6.2f}, {x[2]:6.2f})  "
              f"velocity: ({x[1]:5.2f}, {x[3]:5.2f})")
    # position: ( 10.18,   2.88)  velocity: ( 6.47,  5.27)
    # position: ( 18.87,  10.06)  velocity: ( 7.74,  6.36)
    # position: ( 29.05,  16.20)  velocity: ( 8.78,  6.27)
    # position: ( 39.71,  23.56)  velocity: ( 9.42,  6.64)

References
----------

- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation
- Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
- Bierman, G. J. (1977). Factorization Methods for Discrete Sequential Estimation

See Also
~~~~~~~~

- :doc:`gpu_acceleration` - GPU-accelerated filtering
- Module: ``pytcl.dynamic_estimation.kalman``
- Module: ``pytcl.performance_evaluation`` (NEES/NIS consistency tools)
- Examples: ``examples/kalman_filter_comparison.py``
