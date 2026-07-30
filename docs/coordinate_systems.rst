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

Common Coordinate Systems
--------------------------

**Cartesian (XYZ)**

Rectangular coordinates commonly used in physics and tracking.

.. code-block:: python

   x = np.array([100.0, 50.0, 10.0])  # [x, y, z] in meters
   
   from pytcl.coordinate_systems.conversions import cart2sphere
   sphere = cart2sphere(x)  # Convert to spherical
   # Result: [range, azimuth, elevation]

**Spherical (Range, Azimuth, Elevation)**

Standard radar/sensor coordinate system.

.. code-block:: python

   # [range, azimuth, elevation] in [meters, radians, radians]
   sphere = np.array([
       1000.0,           # range (meters)
       np.pi/4,          # azimuth (radians, 0=East, π/2=North)
       np.pi/6           # elevation (radians, 0=horizon, ±π/2=zenith)
   ])
   
   from pytcl.coordinate_systems.conversions import sphere2cart
   cart = sphere2cart(sphere)  # Result: [x, y, z]

**Geodetic (Latitude, Longitude, Altitude)**

WGS84 ellipsoid coordinates used by GPS.

.. code-block:: python

   # [latitude, longitude, altitude] in [degrees or radians, degrees or radians, meters]
   geodetic = np.array([
       40.7128,      # latitude (degrees N)
       -74.0060,     # longitude (degrees E)
       10.0          # altitude above ellipsoid (meters)
   ])
   
   from pytcl.coordinate_systems.conversions import geodetic2ecef
   ecef = geodetic2ecef(geodetic)  # Earth-Centered Earth-Fixed

**ECEF (Earth-Centered Earth-Fixed)**

Cartesian coordinates fixed to Earth's rotation. Origin at Earth's center.

.. code-block:: python

   # [X, Y, Z] in meters, origin at Earth's center
   ecef = np.array([
       6378137.0,     # X (meters, roughly Earth's equatorial radius)
       0.0,           # Y
       0.0            # Z
   ])
   
   from pytcl.coordinate_systems.conversions import ecef2geodetic
   geodetic = ecef2geodetic(ecef)

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

   # ENU: [east, north, up] in meters
   # Relative to observer location
   enu = np.array([100.0, 200.0, 50.0])  # 100m east, 200m north, 50m up
   
   # Convert to ECEF if you know observer position
   from pytcl.coordinate_systems.conversions import enu2ecef
   observer_geodetic = np.array([40.7128, -74.0060, 10.0])  # NYC
   ecef_target = enu2ecef(enu, observer_geodetic)
   
   # NED: [north, east, down] in meters
   ned = np.array([200.0, 100.0, -50.0])  # 200m N, 100m E, 50m up
   
   from pytcl.coordinate_systems.conversions import ned2ecef
   ecef_target = ned2ecef(ned, observer_geodetic)

**Cylindrical (ρ, φ, z)**

Axially symmetric coordinates.

.. code-block:: python

   # [rho, phi, z] = [radius, azimuth, height]
   cyl = np.array([
       141.42,        # ρ = sqrt(x² + y²)
       np.pi/4,       # φ = atan2(y, x)
       10.0           # z
   ])
   
   from pytcl.coordinate_systems.conversions import cyl2cart
   cart = cyl2cart(cyl)  # → [100, 100, 10]

**Polar (ρ, φ)** *(2D version of cylindrical)*

.. code-block:: python

   # [rho, phi] = [radius, azimuth]
   polar = np.array([141.42, np.pi/4])  # 2D plane
   
   from pytcl.coordinate_systems.conversions import pol2cart
   cart = pol2cart(polar)  # → [100, 100]

**RUV (Range, Unit Vector)**

Range and direction as unit vector. Useful for Jacobians without dealing with singularities at z=0.

.. code-block:: python

   # [range, ux, uy, uz] where (ux, uy, uz) is unit direction vector
   ruv = np.array([
       1000.0,                    # range (meters)
       0.7071,  0.7071, 0.0      # unit direction vector
   ])
   
   from pytcl.coordinate_systems.conversions import ruv2cart
   cart = ruv2cart(ruv)

Conversion Matrix Quick Reference
----------------------------------

.. code-block:: text

   Cartesian (X, Y, Z)
        ↕ sphere2cart, cart2sphere
   Spherical (Range, Az, El)
   
   Cartesian (X, Y, Z)
        ↕ pol2cart, cart2pol
   Polar (ρ, φ)
   
   Cartesian (X, Y, Z)
        ↕ cyl2cart, cart2cyl
   Cylindrical (ρ, φ, z)
   
   ECEF
        ↕ geodetic2ecef, ecef2geodetic
   Geodetic (Lat, Lon, Alt)
   
   ECEF
        ↕ enu2ecef, ecef2enu
   ENU (relative to observer)
   
   ECEF
        ↕ ned2ecef, ecef2ned
   NED (relative to observer)
   
   ECI ↔ ECEF via rotation (see reference_frames)

Practical Examples by Use Case
-------------------------------

Use Case 1: Radar Track Display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Radar gives range, azimuth, elevation. Display on map using lat/lon.

**Solution**:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import (
       sphere2cart, ecef2enu, enu2ecef, ecef2geodetic
   )
   
   # Radar measurement
   radar_meas = np.array([1000.0, np.pi/4, 0.2])  # r=1km, az=45°, el≈11°
   
   # Radar location
   radar_lat_lon_alt = np.array([40.7128, -74.0060, 100.0])  # NYC, 100m elevation
   
   # Step 1: Sphere to Cartesian (relative to radar)
   cart_relative = sphere2cart(radar_meas)  # [x, y, z] offset from radar
   
   # Step 2: Cartesian to ENU (still relative)
   # Note: sphere2cart output is [x_east, y_north, z_up] in ENU convention
   enu_target = cart_relative  # Already ENU relative to radar
   
   # Step 3: ENU to ECEF (absolute position)
   ecef_target = enu2ecef(enu_target, radar_lat_lon_alt)
   
   # Step 4: ECEF to Geodetic (lat/lon/alt)
   target_position = ecef2geodetic(ecef_target)
   
   print(f"Target: Lat={np.degrees(target_position[0]):.4f}°, "
         f"Lon={np.degrees(target_position[1]):.4f}°, "
         f"Alt={target_position[2]:.1f}m")

Use Case 2: Multi-Sensor Fusion (GPS + IMU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: GPS gives geodetic (lat/lon/alt), IMU gives acceleration in vehicle frame. Filter in common frame.

**Solution**:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import geodetic2ecef, ecef2enu
   from pytcl.coordinate_systems.rotations import euler2rotmat
   
   # GPS solution
   gps_geodetic = np.array([40.7128, -74.0060, 100.0])
   gps_ecef = geodetic2ecef(gps_geodetic)
   
   # Reference point (IMU navigation origin)
   ref_geodetic = np.array([40.7128, -74.0060, 100.0])
   ref_ecef = geodetic2ecef(ref_geodetic)
   
   # Target relative to reference (ENU frame)
   target_enu = np.array([100.0, 50.0, 10.0])  # 100m E, 50m N, 10m up
   
   # IMU acceleration (in vehicle body frame)
   # Vehicle attitude (heading, pitch, roll)
   euler = np.array([0.2, 0.1, 0.05])  # radians
   DCM = euler2rotmat(euler)  # Direction Cosine Matrix
   
   # Acceleration in body frame [ax, ay, az]
   accel_body = np.array([10.0, 0.5, 0.0])
   
   # Transform to ENU (local tangent plane)
   accel_enu = DCM.T @ accel_body
   
   print(f"Acceleration in ENU: {accel_enu}")

Use Case 3: Orbital Mechanics (ECI ↔ ECEF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Satellite in inertial frame (ECI), need ground station view (ECEF).

**Solution**:

.. code-block:: python

   import numpy as np
   from pytcl.astronomical.reference_frames import ecef_to_eci, eci_to_ecef
   from pytcl.coordinate_systems.conversions import ecef2geodetic
   
   # Satellite position in ECI (inertial frame)
   sat_eci = np.array([6600000.0, 0.0, 0.0])  # X=6600km, Y=0, Z=0
   
   # Time of observation
   import datetime
   t = datetime.datetime(2026, 2, 26, 12, 0, 0)  # UTC
   
   # Transform to ECEF (fixed to Earth)
   sat_ecef = eci_to_ecef(sat_eci, t)
   
   # Get geodetic position (lat/lon/alt)
   sat_geodetic = ecef2geodetic(sat_ecef)
   
   print(f"Satellite lat={np.degrees(sat_geodetic[0]):.2f}°, "
         f"lon={np.degrees(sat_geodetic[1]):.2f}°")

Rotations: Euler Angles, Quaternions, DCM
------------------------------------------

**Three Ways to Represent Rotation:**

1. **Euler Angles** (3 angles, gimbal lock issues)
2. **Quaternions** (4 parameters, smooth interpolation)
3. **Direction Cosine Matrix (DCM)** (3×3 matrix, computationally intensive)

**Euler Angles (Roll, Pitch, Yaw)**

.. code-block:: python

   from pytcl.coordinate_systems.rotations import (
       euler2rotmat, euler2quat, rotmat2euler, quat2euler
   )
   
   # Euler angles [roll, pitch, yaw] in radians
   # Convention: RzRyRx (yaw about Z, pitch about Y, roll about X)
   euler = np.array([
       0.1,       # roll (rotation about X)
       0.05,      # pitch (rotation about Y)
       0.2        # yaw (rotation about Z)
   ])
   
   # Convert to DCM
   DCM = euler2rotmat(euler)
   
   # Rotate a vector: v_rotated = DCM @ v
   v = np.array([1, 0, 0])
   v_rotated = DCM @ v

**Quaternions** (better for interpolation, no gimbal lock)

.. code-block:: python

   from pytcl.coordinate_systems.rotations import (
       euler2quat, quat2euler, quat_multiply, quat_normalize
   )
   
   # Euler to Quaternion
   euler = np.array([0.1, 0.05, 0.2])
   quat = euler2quat(euler)  # [w, x, y, z] scalar-first
   
   # Normalize quaternion (stays unit length)
   quat = quat_normalize(quat)
   
   # Multiple rotations (quaternion multiplication)
   quat1 = euler2quat(np.array([0.1, 0, 0]))
   quat2 = euler2quat(np.array([0, 0.05, 0]))
   quat_combined = quat_multiply(quat1, quat2)
   
   # Back to Euler
   euler_back = quat2euler(quat)

**Direction Cosine Matrix (DCM)**

.. code-block:: python

   from pytcl.coordinate_systems.rotations import euler2rotmat, rotmat2euler
   
   # DCM: 3×3 orthogonal matrix (RT = I, det=1)
   euler = np.array([0.1, 0.05, 0.2])
   DCM = euler2rotmat(euler)
   
   # Verify orthogonality
   assert np.allclose(DCM @ DCM.T, np.eye(3))  # R^T R = I
   assert np.isclose(np.linalg.det(DCM), 1.0)  # det = +1
   
   # Rotate vector
   v = np.array([1, 0, 0])
   v_rotated = DCM @ v
   
   # Rotate back
   v_original = DCM.T @ v_rotated

Jacobians for Nonlinear Filtering
----------------------------------

**Why Jacobians Matter**

In Extended Kalman Filters, you need partial derivatives (Jacobians) for linearization.

**Common Jacobians Available:**

.. code-block:: python

   from pytcl.coordinate_systems.jacobians import (
       spherical_jacobian,
       jacobian_sphere2cart,
       jacobian_cart2pol,
       jacobian_geodetic2ecef,
       jacobian_ecef2enu
   )
   
   # Jacobian of spherical to Cartesian conversion
   # ∂(x,y,z)/∂(r,az,el)
   sphere = np.array([1000.0, 0.5, 0.1])
   H_sphere = jacobian_sphere2cart(sphere)  # 3×3 matrix
   
   # Use in EKF measurement update
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update
   
   def h_measure(state_cart):
       """Measure in spherical, state in Cartesian"""
       return sphere2cart(state_cart)
   
   def H_jacobian(state_cart):
       """Jacobian of measurement model"""
       return jacobian_sphere2cart(state_cart)
   
   # EKF prediction/update uses these

**Computing Jacobians Manually** (if needed):

.. code-block:: python

   import numpy as np
   
   def jacobian_numerical(func, x, delta=1e-8):
       """Numerical Jacobian using finite differences"""
       n = len(x)
       m = len(func(x))
       J = np.zeros((m, n))
       
       for i in range(n):
           x_plus = x.copy()
           x_plus[i] += delta
           x_minus = x.copy()
           x_minus[i] -= delta
           
           J[:, i] = (func(x_plus) - func(x_minus)) / (2 * delta)
       
       return J
   
   # Test: Jacobian of sphere2cart
   from pytcl.coordinate_systems.conversions import sphere2cart
   
   sphere = np.array([1000.0, 0.5, 0.1])
   J_analytic = jacobian_sphere2cart(sphere)
   J_numeric = jacobian_numerical(sphere2cart, sphere)
   
   print("Analytic Jacobian:\n", J_analytic)
   print("Numeric Jacobian:\n", J_numeric)
   print("Difference:", np.linalg.norm(J_analytic - J_numeric))

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
   lat, lon = np.deg2rad(40.7128), np.deg2rad(-74.0060)  # NYC
   result = geodetic2utm(lat, lon)

   print(f"UTM Zone: {result.zone}{result.hemisphere}")
   print(f"Easting: {result.easting:.1f}m, Northing: {result.northing:.1f}m")

   # UTM back to Geodetic
   lat_back, lon_back = utm2geodetic(
       result.easting, result.northing, result.zone, result.hemisphere
   )

**Mercator Projection**

Conformal projection (preserves angles).

.. code-block:: python

   # Similar interface
   # Web maps (Google Maps, OpenStreetMap) use Web Mercator
   # Use for global visualizations that need correct angles

**Lambert Conformal Conic**

Good for mid-latitude regions.

.. code-block:: python

   # For weather maps, regional forecasts
   # Preserves angles and shapes better than Mercator

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
       sphere2cart, enu2ecef, ecef2geodetic
   )
   
   def transform_measurement(radar_meas, radar_position, reference_position):
       """
       Transform radar measurement to global geodetic coordinates
       
       Args:
           radar_meas: [range, azimuth, elevation] from radar
           radar_position: [lat, lon, alt] of radar
           reference_position: [lat, lon, alt] reference frame
       
       Returns:
           target_geodetic: [lat, lon, alt] of target
       """
       
       # Step 1: Identify source and destination
       # Source: Spherical (radar coords)
       # Destination: Geodetic (lat/lon/alt)
       
       # Step 2: Find intermediate frames
       # Path: Spherical → Cartesian (ENU) → ECEF → Geodetic
       
       # Step 3: Spherical to Cartesian (relative to radar)
       cart_relative = sphere2cart(radar_meas)  # [x, y, z] in ENU
       
       # Step 4: ENU to ECEF (absolute position)
       ecef_target = enu2ecef(cart_relative, radar_position)
       
       # Step 5: ECEF to Geodetic (final destination)
       target_geodetic = ecef2geodetic(ecef_target)
       
       return target_geodetic

Performance Considerations
--------------------------

**Vectorization: Convert Multiple Points at Once**

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import sphere2cart
   
   # ❌ Slow: Loop
   measurements = np.array([
       [100, 0.1, 0.05],
       [105, 0.12, 0.06],
       [102, 0.11, 0.04],
   ])
   
   cart_slow = []
   for meas in measurements:
       cart_slow.append(sphere2cart(meas))
   
   # ✅ Fast: Vectorized if function supports it
   cart_fast = np.array([sphere2cart(m) for m in measurements])
   
   # Or reshape for batch processing
   measurements_batch = measurements.reshape(-1, 3)
   # Apply conversion to all rows

**Caching Jacobians**

.. code-block:: python

   from pytcl.coordinate_systems.jacobians import spherical_jacobian
   from functools import lru_cache
   
   @lru_cache(maxsize=256)
   def cached_jacobian(x_tuple):
       """Cache Jacobian with quantization"""
       x = np.array(x_tuple)
       return spherical_jacobian(x)
   
   # Quantize to reduce cache misses
   x_quantized = np.round(x / 10) * 10  # 10m resolution
   x_tuple = tuple(x_quantized)
   
   J = cached_jacobian(x_tuple)

Common Pitfalls and Solutions
------------------------------

**Pitfall 1: Angle Units (Degrees vs Radians)**

**Problem**: Function expects radians, you pass degrees.

.. code-block:: python

   # ❌ Wrong
   geodetic = np.array([40, -74, 0])  # Degrees
   ecef = geodetic2ecef(geodetic)  # Interprets as radians!
   
   # ✅ Correct
   geodetic_rad = np.radians(np.array([40, -74]))
   geodetic = np.array([*geodetic_rad, 0])  # [lat_rad, lon_rad, alt]
   ecef = geodetic2ecef(geodetic)

**Pitfall 2: Azimuth Direction Convention**

**Problem**: Different conventions for azimuth reference:
- 0° = North (navigation)
- 0° = East (math)
- 0° = North, ±180° = South (compass)

.. code-block:: python

   # TCL convention: East=0, North=π/2
   from pytcl.coordinate_systems.conversions import cart2sphere
   
   # Cartesian [100, 0, 0] → Spherical
   sphere = cart2sphere(np.array([100, 0, 0]))  # [100, 0, 0]
   # range=100, azimuth=0 (East), elevation=0
   
   # Convert compass bearing to math convention
   def compass_to_math_bearing(compass_deg):
       """90° compass = 0° math, etc."""
       return np.radians(90 - compass_deg)

**Pitfall 3: Coordinate Frame Confusion (ECEF vs ECI)**

**Problem**: Mixing inertial and rotating frames.

.. code-block:: python

   import datetime
   from pytcl.astronomical.reference_frames import eci_to_ecef, ecef_to_eci
   
   # ECI: Fixed to distant stars (inertial)
   sat_eci = np.array([6600000, 0, 0])
   
   # ECEF: Rotates with Earth
   # MUST specify time of observation
   t = datetime.datetime(2026, 2, 26, 12, 0, 0)
   sat_ecef = eci_to_ecef(sat_eci, t)
   
   # ❌ Wrong: Use ECI coordinates directly without transformation
   # ✅ Correct: Transform between frames with time

**Pitfall 4: Singularities (Azimuth Undefined at Z-Axis)**

**Problem**: Spherical coordinates undefined when x=y=0 (looking straight up/down).

.. code-block:: python

   # ❌ Problematic: Origin
   sphere = cart2sphere(np.array([0, 0, 100]))  # Looking straight up
   # Azimuth is undefined/arbitrary
   
   # ✅ Use RUV instead (range + unit vector)
   from pytcl.coordinate_systems.conversions import cart2ruv
   ruv = cart2ruv(np.array([0, 0, 100]))  # Well-defined
   # Result: [100, 0, 0, 1] (range=100, direction=[0,0,1])

**Pitfall 5: Altitude Reference Ambiguity**

**Problem**: MSL vs ellipsoid (WGS84) altitude.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import geodetic2ecef
   
   # TCL uses WGS84 ellipsoid altitude (above ellipsoid)
   # GPS altitude is ellipsoid altitude
   # Elevation = WGS84 height - geoid height (EGM96/2008)
   
   lat_lon = np.array([40.7128, -74.0060])
   altitude_wgs84 = 10.0  # Above WGS84 ellipsoid
   altitude_msl = 0.0  # Above sea level (approximately)
   
   # Use WGS84 altitude for geodetic2ecef
   geodetic = np.array([lat_lon[0], lat_lon[1], altitude_wgs84])
   ecef = geodetic2ecef(geodetic)

**Pitfall 6: ENU/NED Reference Point Matters**

**Problem**: ENU/NED are relative to observer; forgetting reference point.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import enu2ecef
   
   enu = np.array([100, 200, 50])  # 100m east, 200m north, 50m up
   
   # ❌ Wrong: No reference point
   # Where is this ENU relative to?
   
   # ✅ Correct: Specify observer position
   observer = np.array([40.7128, -74.0060, 10.0])  # NYC
   ecef_target = enu2ecef(enu, observer)

Advanced Topics
---------------

**Batch Conversion with Multiple Frames**

.. code-block:: python

   def batch_convert_radar_to_geodetic(measurements, radar_positions):
       """
       Convert multiple radar measurements from different radar locations
       
       Args:
           measurements: (N, 3) array of [range, az, el]
           radar_positions: (N, 3) array of [lat, lon, alt]
       
       Returns:
           targets_geodetic: (N, 3) array of [lat, lon, alt]
       """
       from pytcl.coordinate_systems.conversions import (
           sphere2cart, enu2ecef, ecef2geodetic
       )
       
       targets = []
       for i in range(len(measurements)):
           # Convert radar meas to ENU (relative)
           enu = sphere2cart(measurements[i])
           
           # Convert to ECEF (absolute)
           ecef = enu2ecef(enu, radar_positions[i])
           
           # Convert to geodetic
           geodetic = ecef2geodetic(ecef)
           targets.append(geodetic)
       
       return np.array(targets)

**Time-Varying Reference Frames**

.. code-block:: python

   import numpy as np
   import datetime
   
   def satellite_to_ground_track(sat_eci, times):
       """
       Project satellite position to ground track (geodetic)
       
       As satellite orbits, ECEF relationship changes with time
       """
       from pytcl.astronomical.reference_frames import eci_to_ecef
       from pytcl.coordinate_systems.conversions import ecef2geodetic
       
       ground_track = []
       for t in times:
           sat_ecef = eci_to_ecef(sat_eci, t)
           ground_point = ecef2geodetic(sat_ecef)
           ground_track.append(ground_point)
       
       return np.array(ground_track)

Conversion Decision Tree
------------------------

**How to choose the right conversion:**

.. code-block:: text

   Start: I have coordinates in ______
   
   ├─ Spherical (range, az, el)?
   │  └─ Want Cartesian? → sphere2cart
   │
   ├─ Cartesian (x, y, z)?
   │  ├─ Want Spherical? → cart2sphere
   │  ├─ Want Polar (2D)? → cart2pol
   │  ├─ Want Geodetic? → Need intermediate ECEF
   │  │   └─ Is this local (around observer)? → ECEF unknown
   │  │   └─ Is this absolute? → Need context
   │  └─ Want ENU/NED?
   │       └─ Use ecef2enu or ecef2ned
   │
   ├─ Geodetic (lat, lon, alt)?
   │  └─ Want Cartesian (ECEF)? → geodetic2ecef
   │
   ├─ ECEF?
   │  ├─ Want Geodetic? → ecef2geodetic
   │  ├─ Want ENU/NED? → ecef2enu or ecef2ned
   │  ├─ Want ECI? → Need time → ecef_to_eci(x, time)
   │  └─ Want Local? → Depends on observer
   │
   ├─ ENU/NED (local)?
   │  ├─ Want ECEF? → enu2ecef(enu, observer) or ned2ecef(ned, observer)
   │  └─ Want local Cartesian? → Already Cartesian (ENU/NED ARE Cartesian)
   │
   └─ ECI (inertial)?
       └─ Want ECEF? → Need time → eci_to_ecef(x, time)

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`api_navigation` - Finding coordinate functions
- :doc:`kalman_filter_tuning` - Using Jacobians in filters
- :doc:`troubleshooting` - Coordinate-related errors
- Examples: ``examples/coordinate_visualization.py``
- Examples: ``examples/coordinate_systems.py``
