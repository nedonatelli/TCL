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
       IMUData,
       initialize_ins_state, mechanize_ins_ned
   )

   # Initialize INS state
   # Position: latitude (rad), longitude (rad), altitude (m)
   lat = np.radians(37.0)
   lon = np.radians(-122.0)
   alt = 100.0

   # Velocity is given as NED components, attitude as Euler angles
   state = initialize_ins_state(
       lat, lon, alt,
       vN=10.0, vE=5.0, vD=0.0,
       roll=0.0, pitch=0.0, yaw=np.radians(45.0),
   )

   # state.position: [lat, lon, alt], state.velocity: [vN, vE, vD],
   # state.quaternion: body-to-NED attitude quaternion

   # IMU data (accels in m/s^2, gyros in rad/s)
   imu = IMUData(
       accel=np.array([0.0, 0.0, -9.81]),   # Gravity only
       gyro=np.array([0.001, 0.0, 0.005]),  # Small rotation
       dt=0.01,
   )

   # Propagate one step
   new_state = mechanize_ins_ned(state, imu)

Alignment
^^^^^^^^^

Before navigation, the INS must be aligned to determine initial attitude.

**Coarse Alignment (stationary):**

.. code-block:: python

   from pytcl.navigation import coarse_alignment

   # Static accelerometer data senses the gravity vector
   static_accel = np.array([0.0, 0.0, -9.81])

   # Leveling: recovers roll and pitch from gravity
   roll, pitch = coarse_alignment(static_accel, lat)

**Gyrocompass Alignment (heading):**

.. code-block:: python

   from pytcl.navigation import gyrocompass_alignment

   # Stationary gyroscopes sense the Earth rotation rate
   omega_ie = 7.292115e-5  # rad/s
   static_gyro = np.array([
       omega_ie * np.cos(lat), 0.0, -omega_ie * np.sin(lat)
   ])

   yaw = gyrocompass_alignment(static_gyro, roll, pitch, lat)

Loosely-Coupled Integration
---------------------------

In loosely-coupled integration, the GNSS receiver provides position and
velocity solutions that are used to update the INS error states.

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import initialize_ins_gnss

   # Wrap the INS state with a 15-state error filter:
   # [position, velocity, attitude, accel bias, gyro bias] errors
   ins_gnss = initialize_ins_gnss(
       state,
       position_std=10.0,             # m
       velocity_std=0.1,              # m/s
       attitude_std=np.radians(1.0),  # rad
       accel_bias_std=1e-2,           # m/s^2
       gyro_bias_std=1e-4,            # rad/s
   )

   # ins_gnss.ins_state: the INS navigation solution
   # ins_gnss.error_state: 15-element error state (zeros after each reset)
   # ins_gnss.error_cov: 15x15 error covariance

Prediction Step
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import loose_coupled_predict

   # One prediction step: mechanizes the INS and propagates the
   # error covariance in a single call
   ins_gnss = loose_coupled_predict(
       ins_gnss, imu,
       accel_noise_std=0.01,   # m/s^2
       gyro_noise_std=0.001,   # rad/s
       accel_bias_std=1e-5,
       gyro_bias_std=1e-7,
   )

Internally this calls ``mechanize_ins_ned`` for the navigation solution and
builds the error-state dynamics with ``ins_error_state_matrix`` and
``ins_process_noise_matrix``, then propagates the covariance with
``kf_predict``.

GNSS Update
^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import (
       GNSSMeasurement, loose_coupled_update,
       position_std_to_error_state_units
   )

   lat_ins, lon_ins, alt_ins = ins_gnss.ins_state.position

   # Position covariance in error-state units [rad, rad, m]:
   # convert a 2.5 m horizontal / 5 m vertical accuracy
   pos_std = position_std_to_error_state_units(2.5, lat_ins, alt_ins)
   position_cov = np.diag([pos_std[0]**2, pos_std[1]**2, 5.0**2])

   # GNSS measurement (position in [lat, lon, alt], velocity in NED)
   gnss = GNSSMeasurement(
       position=np.array([lat_ins + 1e-6, lon_ins + 1e-6, alt_ins + 2.0]),
       velocity=np.array([10.1, 5.05, 0.1]),
       position_cov=position_cov,
       velocity_cov=np.diag([0.1**2, 0.1**2, 0.2**2]),
       time=0.0,
   )

   # Kalman update: corrects the INS state and resets the error state
   result = loose_coupled_update(ins_gnss, gnss)
   ins_gnss = result.state

   # result.innovation: measurement innovation (GNSS - INS)
   # result.innovation_cov: innovation covariance

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
       SatelliteInfo, geodetic_to_ecef
   )

   lat_ins, lon_ins, alt_ins = ins_gnss.ins_state.position
   user_ecef = np.array(geodetic_to_ecef(lat_ins, lon_ins, alt_ins))

   # Satellite observations (positions/velocities in ECEF, from the
   # GNSS receiver; pseudoranges include the receiver clock bias)
   sat_positions = [
       np.array([-8616e3, -13789e3, 21001e3]),
       np.array([3807e3, -22824e3, 13039e3]),
       np.array([-17719e3, -19498e3, 3361e3]),
       np.array([-18391e3, -1866e3, 19071e3]),
   ]
   satellites = [
       SatelliteInfo(
           prn=i + 1,
           position=pos,
           velocity=np.array([-50.0, 100.0, 20.0]),
           pseudorange=float(np.linalg.norm(pos - user_ecef)) + 5.0,
       )
       for i, pos in enumerate(sat_positions)
   ]

   # Line-of-sight unit vector and geometric range to one satellite
   los, rng = compute_line_of_sight(user_ecef, satellites[0].position)

   # Measurement matrix (one row per satellite, plus clock column)
   H = pseudorange_measurement_matrix(user_ecef, satellites)

   # Innovations (measured - predicted pseudoranges)
   innovations, predicted = tight_coupled_pseudorange_innovation(
       ins_gnss, satellites
   )

   # Full tightly-coupled update (5 m pseudorange noise)
   tight_result = tight_coupled_update(ins_gnss, satellites, pseudorange_std=5.0)

   # tight_result.state: corrected INS/GNSS state
   # tight_result.innovations: pseudorange innovations
   # tight_result.dop: dilution of precision values

DOP Computation
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import compute_dop, satellite_elevation_azimuth

   # Dilution of precision from the geometry matrix; passing the user
   # position rotates the split into the local horizontal/vertical frame
   user_lla = ins_gnss.ins_state.position
   gdop, pdop, hdop, vdop = compute_dop(H, user_lla=user_lla)
   print(f"GDOP: {gdop:.2f}")
   print(f"PDOP: {pdop:.2f}")
   print(f"HDOP: {hdop:.2f}")
   print(f"VDOP: {vdop:.2f}")

   # Satellite geometry
   for sat in satellites:
       el, az = satellite_elevation_azimuth(user_lla, sat.position)
       print(f"PRN {sat.prn}: El={np.degrees(el):.1f} deg, "
             f"Az={np.degrees(az):.1f} deg")

GNSS Outage Detection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.navigation import gnss_outage_detection

   # Chi-squared consistency test on the loose-coupled innovations
   is_outage = gnss_outage_detection(
       result.innovation,
       result.innovation_cov,
       threshold=12.592,  # chi-squared 95% for 6 DOF (pos + vel)
   )

   if is_outage:
       print("GNSS outage detected - using INS-only navigation")

Complete Integration Example
----------------------------

.. code-block:: python

   import numpy as np
   from pytcl.navigation import (
       GNSSMeasurement, IMUData,
       initialize_ins_gnss, initialize_ins_state,
       loose_coupled_predict, loose_coupled_update,
       position_std_to_error_state_units
   )

   # Simulation parameters
   dt = 0.01          # IMU rate: 100 Hz
   gnss_period = 1.0  # GNSS rate: 1 Hz
   duration = 60.0    # seconds

   np.random.seed(42)

   # Initialize
   lat, lon, alt = np.radians(37.0), np.radians(-122.0), 100.0
   state = initialize_ins_state(
       lat, lon, alt, vN=10.0, vE=5.0, vD=0.0, yaw=np.radians(45.0)
   )
   ins_gnss = initialize_ins_gnss(
       state, position_std=10.0, velocity_std=0.1,
       attitude_std=np.radians(1.0)
   )

   # Sensor noise parameters
   accel_noise = 0.01
   gyro_noise = 0.001

   # Simulation loop
   time = 0.0
   next_gnss = gnss_period
   trajectory = []

   while time < duration:
       # Simulate IMU: stationary rotation rates, gravity plus noise
       imu = IMUData(
           accel=np.array([0.0, 0.0, -9.81]) + np.random.randn(3) * accel_noise,
           gyro=np.random.randn(3) * gyro_noise,
           dt=dt,
       )

       # INS mechanization + error covariance propagation
       ins_gnss = loose_coupled_predict(
           ins_gnss, imu,
           accel_noise_std=accel_noise, gyro_noise_std=gyro_noise,
       )

       # GNSS update (at lower rate)
       if time >= next_gnss:
           lat_i, lon_i, alt_i = ins_gnss.ins_state.position

           # Simulated GNSS fix near the INS position
           # (2.5 m horizontal / 5 m vertical, 0.1 m/s velocity)
           pos_std = position_std_to_error_state_units(2.5, lat_i, alt_i)
           gnss = GNSSMeasurement(
               position=np.array([
                   lat_i + np.random.randn() * pos_std[0],
                   lon_i + np.random.randn() * pos_std[1],
                   alt_i + np.random.randn() * 5.0,
               ]),
               velocity=ins_gnss.ins_state.velocity + np.random.randn(3) * 0.1,
               position_cov=np.diag([pos_std[0]**2, pos_std[1]**2, 5.0**2]),
               velocity_cov=np.eye(3) * 0.1**2,
               time=time,
           )

           result = loose_coupled_update(ins_gnss, gnss)
           ins_gnss = result.state

           next_gnss += gnss_period

       trajectory.append(ins_gnss.ins_state.position.copy())
       time += dt

   trajectory = np.array(trajectory)
   print(f"Final position: {np.degrees(trajectory[-1, 0]):.6f} deg, "
         f"{np.degrees(trajectory[-1, 1]):.6f} deg, {trajectory[-1, 2]:.1f} m")

Next Steps
----------

- See :doc:`/api/navigation` for complete API reference
- Explore :doc:`/user_guide/filtering` for more filter options
- Try :doc:`kalman_filtering` for basic filtering concepts
