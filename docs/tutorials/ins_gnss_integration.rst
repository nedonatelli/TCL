INS/GNSS Integration Tutorial
==============================

This tutorial demonstrates how to integrate Inertial Navigation System (INS)
and Global Navigation Satellite System (GNSS) measurements using loosely
and tightly coupled architectures.

INS Basics
----------

Inertial Navigation System Mechanization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

INS mechanization propagates position, velocity, and attitude using
IMU measurements (accelerometers and gyroscopes).

.. code-block:: python

   import numpy as np
   from pytcl.navigation import (
       INSState, IMUData,
       initialize_ins_state, mechanize_ins_ned
   )

   # Initialize INS state
   # Position: latitude (rad), longitude (rad), altitude (m)
   lat = np.radians(37.0)
   lon = np.radians(-122.0)
   alt = 100.0

   # Initial velocity NED (m/s)
   vel_ned = np.array([10.0, 5.0, 0.0])

   # Initial attitude: roll, pitch, yaw (rad)
   attitude = np.array([0.0, 0.0, np.radians(45.0)])

   state = initialize_ins_state(lat, lon, alt, vel_ned, attitude)

   # IMU data (gyros in rad/s, accels in m/s^2)
   imu = IMUData(
       dt=0.01,
       gyro=np.array([0.001, 0.0, 0.005]),  # Small rotation
       accel=np.array([0.0, 0.0, -9.81])    # Gravity only
   )

   # Propagate one step
   new_state = mechanize_ins_ned(state, imu)

Alignment
^^^^^^^^^

Before navigation, the INS must be aligned to determine initial attitude.

**Coarse Alignment (stationary):**

.. code-block:: python

   from pytcl.navigation import coarse_alignment

   # Collect static IMU data
   static_gyro = np.array([0.0, 0.0, 0.0])  # Earth rate negligible
   static_accel = np.array([0.0, 0.0, -9.81])  # Gravity vector

   # Compute initial attitude
   roll, pitch, yaw = coarse_alignment(static_accel, lat)

**Gyrocompass Alignment (finer heading):**

.. code-block:: python

   from pytcl.navigation import gyrocompass_alignment

   # Using Earth rate sensed by gyroscopes
   yaw = gyrocompass_alignment(static_gyro, lat)

Loosely-Coupled Integration
---------------------------

In loosely-coupled integration, the GNSS receiver provides position and
velocity solutions that are used to update the INS error states.

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import (
       initialize_ins_gnss, INSGNSSState,
       ins_error_state_matrix, ins_process_noise_matrix
   )

   # Initial position from GNSS
   gnss_pos = np.array([lat, lon, alt])
   gnss_vel = np.array([10.0, 5.0, 0.0])

   # Initialize integrated state
   ins_gnss = initialize_ins_gnss(
       lat=lat, lon=lon, alt=alt,
       vel_ned=gnss_vel,
       attitude=attitude
   )

   # Error state covariance
   P = np.diag([
       10.0, 10.0, 10.0,        # Position errors (m)
       0.1, 0.1, 0.1,           # Velocity errors (m/s)
       np.radians(1)**2,        # Roll error
       np.radians(1)**2,        # Pitch error
       np.radians(5)**2,        # Heading error
       1e-4, 1e-4, 1e-4,        # Accel biases
       1e-5, 1e-5, 1e-5,        # Gyro biases
   ])

Prediction Step
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import loose_coupled_predict
   from pytcl.dynamic_estimation import kf_predict

   # INS mechanization (predict INS state)
   ins_state = mechanize_ins_ned(ins_gnss.ins_state, imu)

   # Error state dynamics
   dt = imu.dt
   F = ins_error_state_matrix(ins_state, dt)
   Q = ins_process_noise_matrix(
       dt,
       accel_noise=0.01,    # m/s^2/sqrt(Hz)
       gyro_noise=0.001,    # rad/s/sqrt(Hz)
       accel_bias=1e-6,     # m/s^3
       gyro_bias=1e-7       # rad/s^2
   )

   # Kalman prediction
   dx = np.zeros(15)  # Error state
   pred = kf_predict(dx, P, F, Q)
   dx, P = pred.x, pred.P

GNSS Update
^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import (
       loose_coupled_update,
       position_velocity_measurement_matrix
   )
   from pytcl.dynamic_estimation import kf_update

   # GNSS measurement
   gnss_pos_meas = np.array([lat + 1e-6, lon + 1e-6, alt + 2.0])
   gnss_vel_meas = np.array([10.1, 5.05, 0.1])

   # Measurement matrix
   H = position_velocity_measurement_matrix()

   # Innovation (GNSS - INS)
   z_pos = gnss_pos_meas - np.array([
       ins_state.latitude, ins_state.longitude, ins_state.altitude
   ])
   z_vel = gnss_vel_meas - ins_state.velocity
   z = np.concatenate([z_pos, z_vel])

   # Measurement noise
   R = np.diag([2.5, 2.5, 5.0,   # Position (m)
                0.1, 0.1, 0.2])   # Velocity (m/s)

   # Kalman update
   upd = kf_update(dx, P, z, H, R)
   dx, P = upd.x, upd.P

   # Apply correction to INS state
   result = loose_coupled_update(ins_state, dx)
   ins_state = result.corrected_state

Tightly-Coupled Integration
---------------------------

Tightly-coupled integration uses raw GNSS pseudorange and Doppler
measurements directly, providing better performance in degraded
GNSS environments.

Pseudorange Measurement Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import (
       compute_line_of_sight, pseudorange_measurement_matrix,
       tight_coupled_pseudorange_innovation, tight_coupled_update,
       SatelliteInfo, GNSSMeasurement
   )

   # Satellite information (from GNSS receiver)
   satellites = [
       SatelliteInfo(
           prn=1,
           x=15600e3, y=7540e3, z=20140e3,  # ECEF position
           vx=-50.0, vy=100.0, vz=20.0,     # ECEF velocity
           clock_bias=1e-6,
           clock_drift=1e-9
       ),
       # ... more satellites
   ]

   # Pseudorange measurements
   measurements = [
       GNSSMeasurement(prn=1, pseudorange=22345678.9, doppler=-1234.5),
       # ... more measurements
   ]

   # Compute line-of-sight vectors
   user_ecef = geodetic_to_ecef(lat, lon, alt)
   los_vectors = [compute_line_of_sight(user_ecef, sat) for sat in satellites]

   # Measurement matrix
   H = pseudorange_measurement_matrix(los_vectors, len(satellites))

   # Innovations
   innovations = tight_coupled_pseudorange_innovation(
       ins_state, satellites, measurements
   )

   # Measurement noise (pseudorange accuracy)
   R = np.eye(len(satellites)) * 5.0**2  # 5m pseudorange noise

DOP Computation
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import compute_dop, satellite_elevation_azimuth

   # Dilution of precision
   dop = compute_dop(los_vectors)
   print(f"GDOP: {dop.gdop:.2f}")
   print(f"PDOP: {dop.pdop:.2f}")
   print(f"HDOP: {dop.hdop:.2f}")
   print(f"VDOP: {dop.vdop:.2f}")

   # Satellite geometry
   for sat in satellites:
       el, az = satellite_elevation_azimuth(user_ecef, sat)
       print(f"PRN {sat.prn}: El={np.degrees(el):.1f}°, Az={np.degrees(az):.1f}°")

GNSS Outage Detection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import gnss_outage_detection

   # Monitor innovation consistency
   innovation_norm = np.linalg.norm(innovations)
   expected_norm = np.sqrt(np.trace(H @ P @ H.T + R))

   is_outage = gnss_outage_detection(
       innovations, H, P, R, threshold=5.0
   )

   if is_outage:
       print("GNSS outage detected - using INS-only navigation")

Complete Integration Example
----------------------------

.. code-block:: python

   import numpy as np
   from pytcl.navigation import (
       initialize_ins_state, mechanize_ins_ned, IMUData,
       ins_error_state_matrix, ins_process_noise_matrix,
       position_velocity_measurement_matrix, loose_coupled_update
   )
   from pytcl.dynamic_estimation import kf_predict, kf_update

   # Simulation parameters
   dt = 0.01       # IMU rate: 100 Hz
   gnss_rate = 1.0 # GNSS rate: 1 Hz
   duration = 60.0 # seconds

   np.random.seed(42)

   # Initialize
   lat, lon, alt = np.radians(37.0), np.radians(-122.0), 100.0
   vel = np.array([10.0, 5.0, 0.0])
   att = np.array([0.0, 0.0, np.radians(45.0)])

   ins_state = initialize_ins_state(lat, lon, alt, vel, att)
   dx = np.zeros(15)
   P = np.diag([
       10, 10, 10,          # Position
       0.1, 0.1, 0.1,       # Velocity
       0.001, 0.001, 0.01,  # Attitude
       1e-4, 1e-4, 1e-4,    # Accel bias
       1e-5, 1e-5, 1e-5     # Gyro bias
   ])

   # GNSS measurement noise
   R = np.diag([2.5, 2.5, 5.0, 0.1, 0.1, 0.2])
   H = position_velocity_measurement_matrix()

   # Process noise parameters
   accel_noise = 0.01
   gyro_noise = 0.001

   # Simulation loop
   time = 0.0
   gnss_time = 0.0
   trajectory = []

   while time < duration:
       # Simulate IMU (with some noise)
       imu = IMUData(
           dt=dt,
           gyro=np.array([0.001, 0.0, 0.005]) + np.random.randn(3) * gyro_noise,
           accel=np.array([0.1, 0.05, -9.81]) + np.random.randn(3) * accel_noise
       )

       # INS mechanization
       ins_state = mechanize_ins_ned(ins_state, imu)

       # Error state prediction
       F = ins_error_state_matrix(ins_state, dt)
       Q = ins_process_noise_matrix(dt, accel_noise, gyro_noise, 1e-6, 1e-7)
       pred = kf_predict(dx, P, F, Q)
       dx, P = pred.x, pred.P

       # GNSS update (at lower rate)
       if time >= gnss_time:
           # Simulated GNSS measurement
           z = np.concatenate([
               np.random.randn(3) * np.array([2.5, 2.5, 5.0]),
               np.random.randn(3) * np.array([0.1, 0.1, 0.2])
           ])

           upd = kf_update(dx, P, z, H, R)
           dx, P = upd.x, upd.P

           # Apply correction
           result = loose_coupled_update(ins_state, dx)
           ins_state = result.corrected_state
           dx = np.zeros(15)  # Reset error state

           gnss_time += gnss_rate

       trajectory.append([
           ins_state.latitude, ins_state.longitude, ins_state.altitude
       ])
       time += dt

   trajectory = np.array(trajectory)
   print(f"Final position: {np.degrees(trajectory[-1, 0]):.6f}°, "
         f"{np.degrees(trajectory[-1, 1]):.6f}°, {trajectory[-1, 2]:.1f}m")

Next Steps
----------

- See :doc:`/api/navigation` for complete API reference
- Explore :doc:`/user_guide/filtering` for more filter options
- Try :doc:`kalman_filtering` for basic filtering concepts
