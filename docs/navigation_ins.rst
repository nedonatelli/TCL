Navigation & Inertial Measurement Systems
=========================================

*Guide to the inertial navigation (INS) and GNSS/INS integration functions in* ``pytcl.navigation``.

This guide covers the complete workflow for building navigation systems with
the shipped INS suite: strapdown mechanization, alignment, coning/sculling
compensation, and loosely/tightly coupled GNSS integration.

**Table of Contents:**

- INS Fundamentals
- INS Mechanization
- Error Modeling & Propagation
- GNSS/INS Integration Architectures
- Practical Implementation
- Diagnostics & Troubleshooting
- Best Practices

INS Fundamentals
----------------

Inertial Navigation Systems compute position, velocity, and attitude by integrating acceleration and rotation rate measurements from accelerometers and gyroscopes.

**Key Sensors:**

- **Accelerometers**: Measure specific force (acceleration + gravity) in body frame
- **Gyroscopes**: Measure rotation rates about body axes
- **Integrated IMU**: Combined 6-DOF sensor package (3 accelerometers, 3 gyroscopes)

**INS Coordinate Frames:**

1. **Body Frame (b-frame)**: Fixed to vehicle, rotated with aircraft/spacecraft
2. **Navigation Frame (n-frame)**: Local tangent plane (NED or ENU)
3. **ECEF Frame**: Earth-centered, Earth-fixed (for global navigation)
4. **Inertial Frame (i-frame)**: Inertial reference (for precision work)

**Mechanization Equation:**

The fundamental INS equations relate specific force and rotation rates to position and velocity:

.. math::

    \dot{\mathbf{v}}^n = \mathbf{C}_b^n \mathbf{f}^b - (\mathbf{2}\boldsymbol{\omega}_{ie}^n + \boldsymbol{\omega}_{en}^n) \times \mathbf{v}^n + \mathbf{g}^n

    \dot{\mathbf{p}}^n = \mathbf{v}^n

    \dot{\mathbf{C}}_b^n = \mathbf{C}_b^n [\boldsymbol{\omega}_{ib}^b]_\times - [\boldsymbol{\omega}_{in}^n]_\times \mathbf{C}_b^n

Where:

- :math:`\mathbf{v}^n`: velocity in nav frame
- :math:`\mathbf{C}_b^n`: direction cosine matrix (body to nav)
- :math:`\mathbf{f}^b`: specific force measurements
- :math:`\boldsymbol{\omega}_{ie}^n`: Earth rotation rate
- :math:`\boldsymbol{\omega}_{en}^n`: transport rate (due to vehicle motion)
- :math:`\mathbf{g}^n`: gravity vector in nav frame

Basic INS Propagation
~~~~~~~~~~~~~~~~~~~~~

The library ships a full NED-frame strapdown mechanization following Groves
(2013), Chapter 5: create an ``INSState`` with ``initialize_ins_state``, wrap
each IMU sample in ``IMUData``, and step with ``mechanize_ins_ned``.

``INSState`` holds ``position`` as geodetic
``[latitude (rad), longitude (rad), altitude (m)]``, ``velocity`` as NED
``[vN, vE, vD]`` (m/s), and ``quaternion`` as the scalar-first body-to-NED
attitude quaternion.

.. code-block:: python

    import numpy as np
    from pytcl.navigation import IMUData, initialize_ins_state, mechanize_ins_ned

    # Initialize at a known position, stationary and level
    state = initialize_ins_state(
        lat=np.radians(40.7128),
        lon=np.radians(-74.0060),
        alt=100.0,
    )

    # Stationary IMU: the accelerometer measures the reaction to gravity
    g = 9.80665
    imu = IMUData(
        accel=np.array([0.0, 0.0, -g]),  # specific force (m/s^2)
        gyro=np.array([0.0, 0.0, 0.0]),  # angular rate (rad/s)
        dt=0.01,                          # sample period (s)
    )

    # Propagate 1 second of data
    for _ in range(100):
        state = mechanize_ins_ned(state, imu)

    print(f"Latitude:  {np.degrees(state.position[0]):.6f} deg")
    print(f"Velocity NED: {state.velocity} m/s")

Passing the previous IMU sample via ``accel_prev`` / ``gyro_prev`` enables
the built-in coning/sculling compensation (see below).

The mechanization building blocks are exported individually if you need
them: ``earth_rate_ned``, ``transport_rate_ned``, ``gravity_ned``,
``normal_gravity``, ``update_attitude_ned``, and ``radii_of_curvature``.

.. code-block:: python

    from pytcl.navigation import earth_rate_ned, transport_rate_ned, gravity_ned

    lat = np.radians(40.7128)
    omega_ie = earth_rate_ned(lat)                            # rad/s
    omega_en = transport_rate_ned(lat, alt=100.0, vN=100.0, vE=50.0)
    g_ned = gravity_ned(lat, 100.0)                           # [0, 0, g]

INS Error Sources & Modeling
-----------------------------

Real inertial sensors exhibit various error characteristics that cause INS drift:

**Accelerometer Errors:**

.. math::

    \mathbf{f}_{measured} = \mathbf{f}_{true} + \mathbf{b}_a + \mathbf{S}_a \mathbf{f}_{true} + \mathbf{n}_a + \text{temp effects}

Where:

- :math:`\mathbf{b}_a`: bias (constant offset, drifts over time)
- :math:`\mathbf{S}_a`: scale factor (gain error)
- :math:`\mathbf{n}_a`: white noise (high frequency)

**Gyroscope Errors:**

- **Bias**: Several types (constant, random walk, rate-dependent)
- **Scale factor**: Gain errors on rotation rates
- **Noise**: Angle random walk and rate random walk
- **Coupling**: Accelerometer-induced errors (g-sensitivity)

**Practical Error Magnitudes (Mid-Grade INS):**

========== ================================ ======================
Sensor     Typical Bias                     Random Walk Rate
========== ================================ ======================
Accel      50-100 mg (0.5-1 m/s^2)          ~0.01 m/s^2/sqrt(hr)
Gyro       50-200 deg/hr (0.01-0.055 deg/s) ~0.3 deg/hr/sqrt(hr)
========== ================================ ======================

**INS Divergence Over Time (No Updates):**

For unaided INS (no GNSS), position error grows approximately as:

.. math::

    \sigma_{\text{position}} \approx 0.5 \, \sigma_{\text{accel\_bias}} \cdot t^2

.. code-block:: python

    def estimate_gnss_outage_duration(accel_bias_std, desired_error=100.0):
        """
        Maximum GNSS outage before position error exceeds the limit.

        Position error ~ 0.5 * bias * t^2, so t = sqrt(2 * error / bias).
        """
        return np.sqrt(2.0 * desired_error / accel_bias_std)

    # 1 mm/s^2 bias, 100 m budget -> about 7.5 minutes
    print(f"Max outage: {estimate_gnss_outage_duration(0.001):.0f} s")

GNSS/INS Integration Architectures
----------------------------------

**1. Loosely Coupled Integration**

GNSS and INS process measurements independently. GNSS provides position/velocity updates to a 15-state error-state Kalman filter.

Advantages:

- Simple implementation
- Works with standard GNSS receivers
- Easy to debug

Disadvantages:

- Slower convergence after GNSS outage
- Cannot use GNSS during high dynamics

The library ships this filter: ``initialize_ins_gnss`` builds an
``INSGNSSState`` (INS state + 15-state error covariance),
``loose_coupled_predict`` runs the mechanization and covariance propagation,
and ``loose_coupled_update`` applies a ``GNSSMeasurement``.

.. important::

   The first three error states are ``[dlat, dlon, dheight]`` in
   ``[rad, rad, m]`` -- the same units as ``INSState.position``. A GNSS
   accuracy quoted in meters must be converted with
   ``position_std_to_error_state_units`` before it can go on the
   measurement covariance diagonal.

.. code-block:: python

    from pytcl.navigation import (
        GNSSMeasurement,
        IMUData,
        initialize_ins_gnss,
        initialize_ins_state,
        loose_coupled_predict,
        loose_coupled_update,
        position_std_to_error_state_units,
    )

    # Initialize INS and the integration filter
    ins_state = initialize_ins_state(
        lat=np.radians(40.7128), lon=np.radians(-74.0060), alt=100.0
    )
    state = initialize_ins_gnss(ins_state, position_std=10.0, velocity_std=1.0)

    # INS prediction at the IMU rate (100 Hz here)
    imu = IMUData(
        accel=np.array([0.0, 0.0, -9.80665]),
        gyro=np.zeros(3),
        dt=0.01,
    )
    for _ in range(100):
        state = loose_coupled_predict(state, imu)

    # GNSS update at 1 Hz: convert the 5 m accuracy to [rad, rad, m] units
    pos_std = position_std_to_error_state_units(
        5.0, lat=state.ins_state.position[0]
    )
    gnss = GNSSMeasurement(
        position=np.array([np.radians(40.7128), np.radians(-74.0060), 100.0]),
        velocity=np.zeros(3),
        position_cov=np.diag(pos_std**2),
        velocity_cov=np.eye(3) * 0.1**2,
        time=1.0,
    )
    result = loose_coupled_update(state, gnss)
    state = result.state

    # result.innovation / result.innovation_cov feed integrity monitoring
    print(f"Position error std (rad, rad, m): "
          f"{np.sqrt(np.diag(state.error_cov)[:3])}")

``loose_coupled_update_position`` and ``loose_coupled_update_velocity``
apply position-only or velocity-only updates with the same interface.

**2. Tightly Coupled Integration**

INS and GNSS share a single Kalman filter. GNSS measurements are raw pseudoranges, not derived position.

Advantages:

- Better performance during signal degradation (works with < 4 satellites)
- Faster convergence
- More robust to GNSS outages

Disadvantages:

- Complex implementation
- Requires raw GNSS data
- Needs GNSS receiver control

Shipped building blocks: ``tight_coupled_update``,
``tight_coupled_measurement_matrix``,
``tight_coupled_pseudorange_innovation``,
``pseudorange_measurement_matrix``, ``satellite_elevation_azimuth``, and
``compute_dop``. The ``INSGNSSState`` carries the receiver ``clock_bias``
and ``clock_drift`` states these need.

**3. Ultra-Tight Coupling**

INS state is used to predict GNSS signal tracking parameters (carrier frequency, code delay). The tracking loops and INS filter are integrated.

Advantages:

- Works in severe signal degradation
- Maintains tracking in high-dynamic environments

Disadvantages:

- Requires custom GNSS receiver
- Complex real-time implementation

This architecture requires receiver internals and is out of scope for the
library.

Practical Implementation Considerations
---------------------------------------

**Coning and Sculling Compensation**

When integrating gyro measurements over finite time steps, simple
integration accumulates rotation errors. The shipped two-sample corrections
follow Savage's algorithm:

.. code-block:: python

    from pytcl.navigation import coning_correction, sculling_correction

    gyro_prev = np.array([0.010, 0.002, 0.001])  # rad/s
    gyro_curr = np.array([0.011, 0.001, 0.002])

    # Coning: cross product of successive angular-rate samples
    delta_coning = coning_correction(gyro_prev, gyro_curr)

    accel_prev = np.array([0.1, 0.0, -9.8])  # m/s^2
    accel_curr = np.array([0.2, 0.1, -9.8])

    # Sculling: the velocity-domain counterpart
    delta_sculling = sculling_correction(accel_prev, accel_curr,
                                         gyro_prev, gyro_curr)

``compensate_imu_data`` applies both at once, and ``mechanize_ins_ned`` /
``loose_coupled_predict`` do it internally when you pass ``accel_prev`` and
``gyro_prev``.

**Alignment & Initialization**

Proper system alignment is critical for INS accuracy. ``coarse_alignment``
levels the platform from averaged stationary accelerometer data, and
``gyrocompass_alignment`` resolves heading from the sensed Earth rotation
(requires navigation-grade gyros):

.. code-block:: python

    from pytcl.navigation import coarse_alignment, gyrocompass_alignment

    # Average stationary accelerometer samples (removes noise)
    accel_samples = np.random.default_rng(42).normal(
        [0.05, -0.02, -9.80665], 0.01, size=(100, 3)
    )
    accel_avg = accel_samples.mean(axis=0)

    lat = np.radians(40.7128)
    roll, pitch = coarse_alignment(accel_avg, lat)

    # Averaged stationary gyro output senses Earth rotation
    OMEGA_E = 7.292115e-5  # rad/s
    gyro_avg = np.array([OMEGA_E * np.cos(lat), 0.0,
                         -OMEGA_E * np.sin(lat)])
    yaw = gyrocompass_alignment(gyro_avg, roll, pitch, lat)

    print(f"Roll:  {np.degrees(roll):.3f} deg")
    print(f"Pitch: {np.degrees(pitch):.3f} deg")
    print(f"Yaw:   {np.degrees(yaw):.3f} deg")

For consumer-grade gyros that cannot sense Earth rotation, initialize
heading from a magnetometer or a known reference instead.

**Handling GNSS Outages**

During GNSS signal loss, only INS measurements are available and the error
covariance grows through ``loose_coupled_predict``. Before applying a GNSS
update after an outage (or a suspect measurement at any time), gate it with
the chi-square innovation test:

.. code-block:: python

    from pytcl.navigation import gnss_outage_detection

    # From the last loose_coupled_update result
    is_outlier = gnss_outage_detection(result.innovation,
                                       result.innovation_cov)
    if not is_outlier:
        state = result.state  # accept the update

State Vector Estimation & Diagnostics
-------------------------------------

**The 15-State Error Vector**

``INSGNSSState.error_cov`` is the covariance of the error state
(``INSErrorState``):

.. code-block:: python

    ERROR_STATE_LABELS = [
        'dlat (rad)', 'dlon (rad)', 'dalt (m)',        # position error (0-2)
        'dvN (m/s)', 'dvE (m/s)', 'dvD (m/s)',         # velocity error (3-5)
        'att_x (rad)', 'att_y (rad)', 'att_z (rad)',   # attitude error (6-8)
        'accel_bias_x', 'accel_bias_y', 'accel_bias_z',  # m/s^2 (9-11)
        'gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z',     # rad/s (12-14)
    ]

    def diagnose_convergence(error_cov):
        """Check whether the filter has converged."""
        std = np.sqrt(np.diag(error_cov))
        diagnostics = {
            'pos_std': std[0:3],       # [rad, rad, m] - NOT meters for lat/lon
            'vel_std': std[3:6],       # m/s
            'att_std_deg': np.degrees(std[6:9]),
            'accel_bias_std': std[9:12],
            'gyro_bias_std': std[12:15],
        }
        is_converged = (
            np.all(diagnostics['pos_std'][:2] < 1e-4)   # ~600 m of latitude
            and diagnostics['pos_std'][2] < 500.0        # altitude (m)
            and np.all(diagnostics['vel_std'] < 50.0)
            and np.all(diagnostics['att_std_deg'] < 10.0)
        )
        return diagnostics, is_converged

    diagnostics, converged = diagnose_convergence(state.error_cov)
    print(f"Position std: {diagnostics['pos_std']}")
    print(f"Converged: {converged}")

``ins_error_state_matrix`` and ``ins_process_noise_matrix`` expose the
underlying error dynamics F and process noise Q if you build a custom
filter around the same error state.

Common Issues & Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem: INS diverges without GNSS updates**

Solution: this is physics, not a bug -- unaided INS drift is unbounded. Keep
``loose_coupled_predict`` running so ``error_cov`` honestly reflects the
drift, and bound the mission time using the outage-duration estimate above.

**Problem: Large attitude errors at startup**

Solution: perform stationary alignment for 30-60 seconds before navigating:

.. code-block:: python

    def stationary_alignment(accel_buffer, gyro_buffer, lat):
        """Align attitude while the vehicle is stationary."""
        roll, pitch = coarse_alignment(accel_buffer.mean(axis=0), lat)
        yaw = gyrocompass_alignment(gyro_buffer.mean(axis=0), roll, pitch, lat)
        return np.array([roll, pitch, yaw])

**Problem: Altitude diverges when the filter loses GNSS**

Solution: the vertical INS channel is inherently unstable. Use a barometer
as a fallback altimeter with appropriate process/measurement noise.

**Problem: Filter barely responds to GNSS updates**

Solution: check the units of ``GNSSMeasurement.position_cov``. Meters on
the latitude/longitude diagonal makes the filter treat GNSS as vastly less
accurate than it is; always convert with
``position_std_to_error_state_units``.

Best Practices
--------------

1. **Initialization & Alignment**

   - Always perform stationary alignment on level ground
   - Verify heading with compass or known reference
   - Allow 1-2 minute convergence period

2. **GNSS/INS Fusion Strategy**

   - Use loosely coupled for robustness with standard receivers
   - Use tightly coupled when raw GNSS data available
   - Implement automatic switching based on GNSS signal quality

3. **Sensor Fusion Architecture**

   - Include barometer for altitude correction
   - Add magnetometer for heading reference
   - Use wheel speed/odometry if available

4. **Error Monitoring**

   - Check innovation sequences (should be white Gaussian noise)
   - Monitor covariance traces to detect filter divergence
   - Gate measurements with ``gnss_outage_detection`` (chi-square test)

5. **Real-Time Performance**

   - Use fixed-lag smoothing to avoid filter lag
   - Implement output buffering for consistent message rates
   - Profile computation time for embedded systems

6. **Sensor Calibration**

   - Pre-flight: accelerometer bias estimation from static data
   - In-flight: the filter's bias states track slow drift
   - Temperature compensation for long mission duration

References & Further Reading
----------------------------

- **Groves** (2013): "Principles of GNSS, Inertial, and Multisensor
  Integrated Navigation Systems" (the mechanization here follows Ch. 5)
- **Titterton & Weston** (2004): "Strapdown Inertial Navigation Technology"
- **Rogers** (2007): "Applied Mathematics in Integrated Navigation Systems"

See Also
~~~~~~~~

- :doc:`coordinate_systems` - Coordinate transformations for navigation frames
- :doc:`astronomical` - Precision reference frames (ECEF/ECI)
- :doc:`recipes` - Ready-to-use Kalman filter implementations
- :doc:`troubleshooting` - Navigation system diagnostics
- ``examples/ins_gnss_navigation.py`` - Full INS/GNSS example
- ``docs/notebooks/07_ins_gnss_integration.ipynb`` - Interactive tutorial
