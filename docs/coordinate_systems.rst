Coordinate Systems Deep Dive
=============================

Overview
--------

The Tracker Component Library provides **20+ coordinate system** conversions and transformations essential for multi-sensor tracking. This guide covers all coordinate types, conversions, rotations, and practical usage patterns.

**Key Modules:**

- ``coordinate_systems.conversions`` - Convert between coordinate types
- ``coordinate_systems.rotations`` - Rotation representations and operations
- ``coordinate_systems.jacobians`` - Partial derivatives for filtering
- ``coordinate_systems.projections`` - Map projections (UTM, Mercator, etc.)

All functions are also re-exported at the ``pytcl.coordinate_systems`` package
level, so ``from pytcl.coordinate_systems import cart2sphere`` works too.

Common Coordinate Systems
--------------------------

**Cartesian (XYZ)**

Rectangular coordinates commonly used in physics and tracking.

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import cart2sphere

   x = np.array([100.0, 50.0, 10.0])  # [x, y, z] in meters

   # Convert to spherical; returns a (range, azimuth, elevation) tuple
   r, az, el = cart2sphere(x, system_type='az-el')

**Spherical (Range, Azimuth, Elevation)**

Standard radar/sensor coordinate system. With ``system_type='az-el'`` the
azimuth is measured in the xy-plane from the +x axis and the elevation from
the xy-plane (the tracking convention). The default ``'standard'`` uses the
physics convention (polar angle from +z).

.. code-block:: python

   from pytcl.coordinate_systems.conversions import sphere2cart

   # Arguments are (range, azimuth, elevation) in (meters, radians, radians)
   cart = sphere2cart(1000.0, np.pi/4, np.pi/6, system_type='az-el')
   # Result: array [x, y, z]

**Geodetic (Latitude, Longitude, Altitude)**

WGS84 ellipsoid coordinates used by GPS. All angles are in **radians**.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import geodetic2ecef

   lat = np.radians(40.7128)   # latitude (rad)
   lon = np.radians(-74.0060)  # longitude (rad)
   alt = 10.0                  # altitude above the ellipsoid (m)

   ecef = geodetic2ecef(lat, lon, alt)  # Earth-Centered Earth-Fixed [x, y, z]

**ECEF (Earth-Centered Earth-Fixed)**

Cartesian coordinates fixed to Earth's rotation. Origin at Earth's center.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import ecef2geodetic

   ecef = np.array([
       6378137.0,     # X (meters, roughly Earth's equatorial radius)
       0.0,           # Y
       0.0            # Z
   ])

   lat, lon, alt = ecef2geodetic(ecef)  # returns a (lat, lon, alt) tuple

**ECI (Earth-Centered Inertial)**

Cartesian coordinates fixed to distant stars (non-rotating). Used in orbital mechanics.

.. code-block:: python

   # [X, Y, Z] in meters, inertial frame
   eci = np.array([
       6600000.0,     # X (meters)
       0.0,           # Y
       0.0            # Z
   ])

**Local Coordinates: ENU and NED**

Tangent plane at observer location.

- **ENU (East-North-Up)**: East, North, Up (common in aviation)
- **NED (North-East-Down)**: North, East, Down (common in marine navigation)

.. code-block:: python

   from pytcl.coordinate_systems.conversions import enu2ecef, ned2ecef

   # Observer location (radians, radians, meters)
   lat_ref = np.radians(40.7128)   # NYC
   lon_ref = np.radians(-74.0060)
   ecef_ref = geodetic2ecef(lat_ref, lon_ref, 10.0)

   # ENU: [east, north, up] in meters, relative to the observer
   enu = np.array([100.0, 200.0, 50.0])  # 100m east, 200m north, 50m up
   ecef_target = enu2ecef(enu, lat_ref, lon_ref, ecef_ref)

   # NED: [north, east, down] in meters
   ned = np.array([200.0, 100.0, -50.0])  # 200m N, 100m E, 50m up
   ecef_target = ned2ecef(ned, lat_ref, lon_ref, ecef_ref)

**Cylindrical (rho, phi, z)**

Axially symmetric coordinates.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import cyl2cart

   # Arguments are (rho, phi, z) = (radius, azimuth, height)
   cart = cyl2cart(141.42, np.pi/4, 10.0)  # -> approx [100, 100, 10]

**Polar (rho, phi)** *(2D version of cylindrical)*

.. code-block:: python

   from pytcl.coordinate_systems.conversions import pol2cart

   # Arguments are (radius, azimuth)
   cart = pol2cart(141.42, np.pi/4)  # -> approx [100, 100]

**R-U-V (Range and Direction Cosines)**

Range plus the direction cosines ``u = x/r`` and ``v = y/r``. Useful for
phased-array radars and for Jacobians without the azimuth singularity at
the z-axis.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import cart2ruv, ruv2cart

   # cart2ruv returns a (range, u, v) tuple
   r, u, v = cart2ruv(np.array([100.0, 100.0, 10.0]))

   # Inverse (assumes the +z half-space)
   cart = ruv2cart(r, u, v)  # -> approx [100, 100, 10]

Conversion Matrix Quick Reference
----------------------------------

.. code-block:: text

   Cartesian (X, Y, Z)
        <-> sphere2cart, cart2sphere
   Spherical (Range, Az, El)

   Cartesian (X, Y, Z)
        <-> pol2cart, cart2pol
   Polar (rho, phi)

   Cartesian (X, Y, Z)
        <-> cyl2cart, cart2cyl
   Cylindrical (rho, phi, z)

   ECEF
        <-> geodetic2ecef, ecef2geodetic
   Geodetic (Lat, Lon, Alt)

   ECEF
        <-> enu2ecef, ecef2enu
   ENU (relative to observer)

   ECEF
        <-> ned2ecef, ecef2ned
   NED (relative to observer)

   ECI <-> ECEF via Earth rotation angle
   (pytcl.astronomical.reference_frames: eci_to_ecef(x, gmst))

Practical Examples by Use Case
-------------------------------

Use Case 1: Radar Track Display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Radar gives range, azimuth, elevation. Display on map using lat/lon.

**Solution**:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import (
       sphere2cart, enu2ecef, ecef2geodetic, geodetic2ecef,
   )

   # Radar measurement: r=1 km, az=45 deg (from east), el approx 11 deg
   radar_range, radar_az, radar_el = 1000.0, np.pi/4, 0.2

   # Radar location (radians, radians, meters)
   radar_lat = np.radians(40.7128)   # NYC
   radar_lon = np.radians(-74.0060)
   radar_alt = 100.0

   # Step 1: spherical to Cartesian ENU offset from the radar
   # ('az-el' azimuth is measured from the +x axis, which is east in ENU)
   enu_target = sphere2cart(radar_range, radar_az, radar_el,
                            system_type='az-el')

   # Step 2: ENU offset to absolute ECEF
   radar_ecef = geodetic2ecef(radar_lat, radar_lon, radar_alt)
   ecef_target = enu2ecef(enu_target, radar_lat, radar_lon, radar_ecef)

   # Step 3: ECEF to geodetic (lat/lon/alt)
   lat, lon, alt = ecef2geodetic(ecef_target)

   print(f"Target: Lat={np.degrees(lat):.4f} deg, "
         f"Lon={np.degrees(lon):.4f} deg, Alt={alt:.1f} m")

Output::

   Target: Lat=40.7190 deg, Lon=-73.9978 deg, Alt=298.7 m

Use Case 2: Multi-Sensor Fusion (GPS + IMU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: GPS gives geodetic (lat/lon/alt), IMU gives acceleration in vehicle frame. Filter in common frame.

**Solution**:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import geodetic2ecef
   from pytcl.coordinate_systems.rotations import euler2rotmat

   # GPS solution (radians, radians, meters)
   gps_lat = np.radians(40.7128)
   gps_lon = np.radians(-74.0060)
   gps_alt = 100.0
   gps_ecef = geodetic2ecef(gps_lat, gps_lon, gps_alt)

   # Vehicle attitude: 'ZYX' aerospace sequence takes [yaw, pitch, roll]
   yaw, pitch, roll = 0.2, 0.1, 0.05  # radians
   C_b2n = euler2rotmat([yaw, pitch, roll], 'ZYX')  # body-to-nav DCM

   # Acceleration in body frame [ax, ay, az]
   accel_body = np.array([10.0, 0.5, 0.0])

   # Transform to the local navigation frame
   accel_nav = C_b2n @ accel_body

   print(f"Acceleration in the navigation frame: {accel_nav}")

Use Case 3: Orbital Mechanics (ECI to ECEF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Satellite in inertial frame (ECI), need ground station view (ECEF).

**Solution**: the ECI/ECEF conversion is a rotation about the z-axis by the
Greenwich Mean Sidereal Time (GMST), so first convert the observation time
to a Julian date and compute GMST.

.. code-block:: python

   import numpy as np
   from pytcl.astronomical import cal_to_jd, gmst
   from pytcl.astronomical.reference_frames import eci_to_ecef
   from pytcl.coordinate_systems.conversions import ecef2geodetic

   # Satellite position in ECI (the rotation is unit-agnostic; meters here)
   sat_eci = np.array([6600e3, 0.0, 0.0])

   # Time of observation -> Julian date -> GMST angle (radians)
   jd = cal_to_jd(2026, 2, 26, 12, 0, 0.0)
   theta = gmst(jd)

   # Transform to ECEF (fixed to Earth)
   sat_ecef = eci_to_ecef(sat_eci, theta)

   # Get geodetic position (lat/lon/alt)
   lat, lon, alt = ecef2geodetic(sat_ecef)

   print(f"Satellite lat={np.degrees(lat):.2f} deg, "
         f"lon={np.degrees(lon):.2f} deg")

Rotations: Euler Angles, Quaternions, DCM
------------------------------------------

**Three Ways to Represent Rotation:**

1. **Euler Angles** (3 angles, gimbal lock issues)
2. **Quaternions** (4 parameters, smooth interpolation)
3. **Direction Cosine Matrix (DCM)** (3x3 matrix, 9 parameters)

**Euler Angles (Yaw, Pitch, Roll)**

.. code-block:: python

   from pytcl.coordinate_systems.rotations import euler2rotmat, rotmat2euler

   # The default 'ZYX' aerospace sequence takes angles [yaw, pitch, roll]
   # and builds R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
   yaw, pitch, roll = 0.2, 0.05, 0.1  # radians

   DCM = euler2rotmat([yaw, pitch, roll], 'ZYX')

   # Rotate a vector: v_rotated = DCM @ v
   v = np.array([1.0, 0.0, 0.0])
   v_rotated = DCM @ v

   # Back to Euler angles
   angles_back = rotmat2euler(DCM, 'ZYX')  # [yaw, pitch, roll]

**Quaternions** (better for interpolation, no gimbal lock)

.. code-block:: python

   from pytcl.coordinate_systems.rotations import (
       euler2quat, quat2euler, quat_multiply, quat_rotate
   )

   # Euler to quaternion; [w, x, y, z] scalar-first
   quat = euler2quat([0.2, 0.05, 0.1], 'ZYX')

   # Re-normalize after repeated operations (plain NumPy)
   quat = quat / np.linalg.norm(quat)

   # Compose rotations (quaternion multiplication)
   quat1 = euler2quat([0.1, 0.0, 0.0], 'ZYX')
   quat2 = euler2quat([0.0, 0.05, 0.0], 'ZYX')
   quat_combined = quat_multiply(quat1, quat2)

   # Rotate a vector directly
   v_rotated = quat_rotate(quat, np.array([1.0, 0.0, 0.0]))

   # Back to Euler
   euler_back = quat2euler(quat, 'ZYX')

Related helpers: ``quat_conjugate``, ``quat_inverse``, ``quat2rotmat``,
``rotmat2quat``, and ``slerp(q1, q2, t)`` for interpolation.

**Direction Cosine Matrix (DCM)**

.. code-block:: python

   from pytcl.coordinate_systems.rotations import euler2rotmat

   # DCM: 3x3 orthogonal matrix (R R^T = I, det = +1)
   DCM = euler2rotmat([0.2, 0.05, 0.1], 'ZYX')

   # Verify orthogonality
   assert np.allclose(DCM @ DCM.T, np.eye(3))
   assert np.isclose(np.linalg.det(DCM), 1.0)

   # Rotate vector
   v = np.array([1.0, 0.0, 0.0])
   v_rotated = DCM @ v

   # Rotate back
   v_original = DCM.T @ v_rotated

``is_rotation_matrix(R)`` checks both properties for you.

Jacobians for Nonlinear Filtering
----------------------------------

**Why Jacobians Matter**

In Extended Kalman Filters, you need partial derivatives (Jacobians) for linearization.

**Common Jacobians Available:**

.. code-block:: python

   from pytcl.coordinate_systems.jacobians import (
       spherical_jacobian,      # d(r, az, el) / d(x, y, z)
       spherical_jacobian_inv,  # d(x, y, z) / d(r, az, el)
       geodetic_jacobian,       # d(ECEF) / d(lat, lon, alt)
       enu_jacobian,            # ECEF -> ENU rotation at (lat, lon)
       ned_jacobian,            # ECEF -> NED rotation at (lat, lon)
       polar_jacobian,          # 2D polar
       ruv_jacobian,            # d(r, u, v) / d(x, y, z)
   )

   # Jacobian of the Cartesian -> spherical measurement function
   cart = np.array([1000.0, 500.0, 100.0])
   H = spherical_jacobian(cart, system_type='az-el')  # 3x3 matrix

   # Use in an EKF measurement update
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update
   from pytcl.coordinate_systems.conversions import cart2sphere

   def h_measure(state):
       """Predict the spherical measurement from a Cartesian state."""
       r, az, el = cart2sphere(state[:3], system_type='az-el')
       return np.array([r, az, el])

   def H_jacobian(state):
       """Jacobian of the measurement model (position block only)."""
       H = np.zeros((3, state.shape[0]))
       H[:, :3] = spherical_jacobian(state[:3], system_type='az-el')
       return H

   # ekf_predict / ekf_update take these as the h and H arguments

**Checking a Jacobian Numerically**

The library ships a finite-difference helper, so there is no need to write
your own:

.. code-block:: python

   from pytcl.coordinate_systems.jacobians import (
       numerical_jacobian, spherical_jacobian
   )
   from pytcl.coordinate_systems.conversions import cart2sphere

   cart = np.array([1000.0, 500.0, 100.0])
   J_analytic = spherical_jacobian(cart, system_type='az-el')
   J_numeric = numerical_jacobian(
       lambda p: np.array(cart2sphere(p, system_type='az-el')), cart
   )

   print("Max difference:", np.abs(J_analytic - J_numeric).max())
   # Max difference: ~1e-6 (finite-difference accuracy)

Map Projections
---------------

**When to Use Map Projections**

Project 3D Earth to 2D maps for display or processing.

**UTM (Universal Transverse Mercator)**

Good for local areas with minimal distortion.

.. code-block:: python

   from pytcl.coordinate_systems.projections import (
       geodetic2utm, utm2geodetic
   )

   # Geodetic to UTM (angles in radians)
   lat, lon = np.radians(40.7128), np.radians(-74.0060)  # NYC
   result = geodetic2utm(lat, lon)

   print(f"UTM zone: {result.zone}{result.hemisphere}")
   print(f"Easting: {result.easting:.1f} m, Northing: {result.northing:.1f} m")

   # UTM back to geodetic; returns (lat, lon) in radians
   lat_back, lon_back = utm2geodetic(
       result.easting, result.northing, result.zone, result.hemisphere
   )

Output::

   UTM zone: 18N
   Easting: 583959.4 m, Northing: 4507351.0 m

**Mercator Projection**

Conformal projection (preserves angles). Used by web maps.

.. code-block:: python

   from pytcl.coordinate_systems.projections import mercator, mercator_inverse

   res = mercator(np.radians(40.7128), np.radians(-74.0060))
   print(f"x={res.x:.1f} m, y={res.y:.1f} m")
   # x=-8238310.2 m, y=4942194.8 m

   lat_back, lon_back = mercator_inverse(res.x, res.y)

**Lambert Conformal Conic**

Good for mid-latitude regions (weather maps, regional charts). Takes the
projection origin and two standard parallels:

.. code-block:: python

   from pytcl.coordinate_systems.projections import lambert_conformal_conic

   res = lambert_conformal_conic(
       np.radians(40.7128), np.radians(-74.0060),  # point
       np.radians(39.0), np.radians(-96.0),        # origin lat0, lon0
       np.radians(33.0), np.radians(45.0),         # standard parallels
   )
   print(f"x={res.x:.1f} m, y={res.y:.1f} m")

Also available: ``transverse_mercator``, ``stereographic``,
``polar_stereographic``, ``azimuthal_equidistant`` (each with an inverse).

**Comparison**:

===================  ==========  ==========  ============
Projection           Best Area   Distortion  Complexity
===================  ==========  ==========  ============
UTM                  Local       Low         Simple
Mercator             Global      High eq.    Medium
Lambert Conformal    Regional    Low lat.    Medium
===================  ==========  ==========  ============

Coordinate Transformation Workflow
-----------------------------------

**Five-Step Process for Complex Transformations**

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import (
       sphere2cart, enu2ecef, ecef2geodetic, geodetic2ecef,
   )

   def transform_measurement(radar_meas, radar_lat, radar_lon, radar_alt):
       """
       Transform a radar measurement to global geodetic coordinates.

       Args:
           radar_meas: (range, azimuth, elevation) from the radar
           radar_lat, radar_lon: radar position (radians)
           radar_alt: radar altitude (meters)

       Returns:
           (lat, lon, alt) of the target in radians/meters
       """
       # Step 1: Identify source and destination
       #   Source: Spherical (radar coords); Destination: Geodetic
       # Step 2: Find intermediate frames
       #   Path: Spherical -> Cartesian (ENU) -> ECEF -> Geodetic
       r, az, el = radar_meas

       # Step 3: Spherical to Cartesian (ENU offset from the radar)
       enu = sphere2cart(r, az, el, system_type='az-el')

       # Step 4: ENU to ECEF (absolute position)
       radar_ecef = geodetic2ecef(radar_lat, radar_lon, radar_alt)
       ecef = enu2ecef(enu, radar_lat, radar_lon, radar_ecef)

       # Step 5: ECEF to Geodetic (final destination)
       return ecef2geodetic(ecef)

Performance Considerations
--------------------------

**Vectorization: Convert Multiple Points at Once**

The conversion functions accept arrays, so batches never need a Python loop:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import sphere2cart

   # (n, 3) array of [range, az, el] measurements
   measurements = np.array([
       [100.0, 0.10, 0.05],
       [105.0, 0.12, 0.06],
       [102.0, 0.11, 0.04],
   ])

   # One vectorized call; returns a (3, n) array of [x, y, z] columns
   cart_all = sphere2cart(
       measurements[:, 0], measurements[:, 1], measurements[:, 2],
       system_type='az-el',
   )

   # cart2sphere likewise accepts (3, n) or (n, 3) point arrays

**Caching Jacobians**

.. code-block:: python

   import numpy as np
   from functools import lru_cache
   from pytcl.coordinate_systems.jacobians import spherical_jacobian

   @lru_cache(maxsize=256)
   def cached_jacobian(point_tuple):
       """Cache Jacobians for quantized points (tuples are hashable)."""
       return spherical_jacobian(np.array(point_tuple), system_type='az-el')

   # Quantize to 10 m resolution to raise the cache hit rate
   point = np.array([1003.7, 498.2, 101.4])
   point_quantized = tuple(np.round(point / 10.0) * 10.0)

   J = cached_jacobian(point_quantized)

Common Pitfalls and Solutions
------------------------------

**Pitfall 1: Angle Units (Degrees vs Radians)**

**Problem**: Every pytcl function expects radians; passing degrees produces
silently wrong answers.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import geodetic2ecef

   # Wrong: degrees are interpreted as radians
   ecef = geodetic2ecef(40.0, -74.0, 0.0)

   # Correct
   ecef = geodetic2ecef(np.radians(40.0), np.radians(-74.0), 0.0)

**Pitfall 2: Azimuth Direction Convention**

**Problem**: Different conventions for azimuth reference:

- 0 deg = North (navigation)
- 0 deg = East (math)
- 0 deg = North, +/-180 deg = South (compass)

.. code-block:: python

   from pytcl.coordinate_systems.conversions import cart2sphere

   # TCL 'az-el' convention: azimuth from the +x axis
   r, az, el = cart2sphere(np.array([100.0, 0.0, 0.0]),
                           system_type='az-el')
   # r=100, az=0 (+x direction), el=0

   # Convert compass bearing to math convention
   def compass_to_math_bearing(compass_deg):
       """90 deg compass = 0 deg math, etc."""
       return np.radians(90.0 - compass_deg)

**Pitfall 3: Coordinate Frame Confusion (ECEF vs ECI)**

**Problem**: Mixing inertial and rotating frames.

.. code-block:: python

   from pytcl.astronomical import cal_to_jd, gmst
   from pytcl.astronomical.reference_frames import eci_to_ecef

   # ECI: fixed to distant stars (inertial)
   sat_eci = np.array([6600e3, 0.0, 0.0])

   # ECEF rotates with Earth: you MUST supply the rotation angle
   # for the observation time
   theta = gmst(cal_to_jd(2026, 2, 26, 12, 0, 0.0))
   sat_ecef = eci_to_ecef(sat_eci, theta)

   # Wrong: using ECI coordinates directly as if they were ECEF
   # Correct: rotate between frames using GMST at the observation time

**Pitfall 4: Singularities (Azimuth Undefined at Z-Axis)**

**Problem**: Spherical azimuth is undefined when x=y=0 (looking straight up/down).

.. code-block:: python

   # Problematic: looking straight up, azimuth is arbitrary
   r, az, el = cart2sphere(np.array([0.0, 0.0, 100.0]),
                           system_type='az-el')

   # Use R-U-V instead (range + direction cosines): well defined
   from pytcl.coordinate_systems.conversions import cart2ruv
   r, u, v = cart2ruv(np.array([0.0, 0.0, 100.0]))
   # Result: r=100, u=0, v=0 (boresight)

**Pitfall 5: Altitude Reference Ambiguity**

**Problem**: MSL vs ellipsoid (WGS84) altitude.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import geodetic2ecef

   # TCL uses WGS84 ellipsoid altitude (above ellipsoid)
   # GPS altitude is ellipsoid altitude
   # MSL elevation = WGS84 height - geoid height (see pytcl.gravity.geoid_height)

   lat, lon = np.radians(40.7128), np.radians(-74.0060)
   altitude_wgs84 = 10.0  # above the WGS84 ellipsoid

   ecef = geodetic2ecef(lat, lon, altitude_wgs84)

**Pitfall 6: ENU/NED Reference Point Matters**

**Problem**: ENU/NED are relative to an observer; the reference position is a
required part of the conversion.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import enu2ecef, geodetic2ecef

   enu = np.array([100.0, 200.0, 50.0])  # 100m east, 200m north, 50m up

   # The observer's latitude/longitude orient the tangent plane, and the
   # observer's ECEF position anchors it:
   lat_ref = np.radians(40.7128)   # NYC
   lon_ref = np.radians(-74.0060)
   ecef_ref = geodetic2ecef(lat_ref, lon_ref, 10.0)

   ecef_target = enu2ecef(enu, lat_ref, lon_ref, ecef_ref)

Advanced Topics
---------------

**Batch Conversion with Multiple Frames**

.. code-block:: python

   from pytcl.coordinate_systems.conversions import (
       sphere2cart, enu2ecef, ecef2geodetic, geodetic2ecef,
   )

   def batch_convert_radar_to_geodetic(measurements, radar_positions):
       """
       Convert radar measurements from different radar locations.

       Args:
           measurements: (N, 3) array of [range, az, el]
           radar_positions: (N, 3) array of [lat (rad), lon (rad), alt (m)]

       Returns:
           (N, 3) array of [lat, lon, alt]
       """
       targets = []
       for meas, radar in zip(measurements, radar_positions):
           r, az, el = meas
           radar_lat, radar_lon, radar_alt = radar

           enu = sphere2cart(r, az, el, system_type='az-el')
           radar_ecef = geodetic2ecef(radar_lat, radar_lon, radar_alt)
           ecef = enu2ecef(enu, radar_lat, radar_lon, radar_ecef)
           targets.append(ecef2geodetic(ecef))

       return np.array(targets)

**Time-Varying Reference Frames**

.. code-block:: python

   import numpy as np
   from pytcl.astronomical import gmst
   from pytcl.astronomical.reference_frames import eci_to_ecef
   from pytcl.coordinate_systems.conversions import ecef2geodetic

   def satellite_ground_track(sat_eci, jd_times):
       """
       Project a satellite position to its ground track (geodetic).

       The ECI/ECEF relationship changes with time, so GMST is
       recomputed for every sample.
       """
       track = []
       for jd in jd_times:
           sat_ecef = eci_to_ecef(sat_eci, gmst(jd))
           track.append(ecef2geodetic(sat_ecef))
       return np.array(track)

Conversion Decision Tree
------------------------

**How to choose the right conversion:**

.. code-block:: text

   Start: I have coordinates in ______

   +- Spherical (range, az, el)?
   |  +- Want Cartesian? -> sphere2cart(r, az, el)
   |
   +- Cartesian (x, y, z)?
   |  +- Want Spherical? -> cart2sphere
   |  +- Want Polar (2D)? -> cart2pol
   |  +- Want R-U-V? -> cart2ruv
   |  +- Want Geodetic? -> Need intermediate ECEF
   |      +- Is this local (around observer)? -> enu2ecef first
   |      +- Is this absolute? -> already ECEF, use ecef2geodetic
   |
   +- Geodetic (lat, lon, alt)?
   |  +- Want Cartesian (ECEF)? -> geodetic2ecef(lat, lon, alt)
   |  +- Want local ENU? -> geodetic2enu
   |
   +- ECEF?
   |  +- Want Geodetic? -> ecef2geodetic
   |  +- Want ENU/NED? -> ecef2enu or ecef2ned
   |  +- Want ECI? -> Need GMST -> ecef_to_eci(x, gmst)
   |
   +- ENU/NED (local)?
   |  +- Want ECEF? -> enu2ecef(enu, lat, lon, ecef_ref) or ned2ecef(...)
   |  +- Want local Cartesian? -> Already Cartesian (ENU/NED ARE Cartesian)
   |
   +- ECI (inertial)?
       +- Want ECEF? -> Need GMST -> eci_to_ecef(x, gmst)

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`api_navigation` - Finding coordinate functions
- :doc:`kalman_filter_tuning` - Using Jacobians in filters
- :doc:`troubleshooting` - Coordinate-related errors
- Examples: ``examples/coordinate_visualization.py``
- Examples: ``examples/coordinate_systems.py``
