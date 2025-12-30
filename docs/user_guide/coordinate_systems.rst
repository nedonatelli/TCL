Coordinate Systems
==================

The library provides comprehensive coordinate system conversions and
rotation representations.

Coordinate Conversions
----------------------

Cartesian and Spherical
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.coordinate_systems import cart2sphere, sphere2cart

   # Cartesian to spherical (range, azimuth, elevation)
   points = np.array([[100, 200, 50]])
   r, az, el = cart2sphere(points)

   # Spherical to Cartesian
   cart = sphere2cart(r, az, el)

Geodetic and ECEF
^^^^^^^^^^^^^^^^^

Convert between geodetic coordinates (latitude, longitude, altitude) and
Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates:

.. code-block:: python

   from tracker_component_library.coordinate_systems import geodetic2ecef, ecef2geodetic

   # Geodetic to ECEF (angles in radians)
   lat = np.deg2rad(40.0)  # 40 degrees North
   lon = np.deg2rad(-75.0)  # 75 degrees West
   alt = 100.0  # meters

   x, y, z = geodetic2ecef(lat, lon, alt)

   # ECEF to geodetic
   lat, lon, alt = ecef2geodetic(x, y, z)

ENU and NED
^^^^^^^^^^^

Local tangent plane coordinates:

.. code-block:: python

   from tracker_component_library.coordinate_systems import ecef2enu, enu2ecef

   # Convert ECEF to local East-North-Up
   origin_lat, origin_lon, origin_alt = np.deg2rad(40.0), np.deg2rad(-75.0), 0.0
   e, n, u = ecef2enu(x, y, z, origin_lat, origin_lon, origin_alt)

Rotation Representations
------------------------

The library supports multiple rotation representations and conversions
between them.

Rotation Matrices
^^^^^^^^^^^^^^^^^

Elementary rotations about principal axes:

.. code-block:: python

   from tracker_component_library.coordinate_systems import rotx, roty, rotz

   # Rotation about x-axis by 30 degrees
   Rx = rotx(np.deg2rad(30))

   # Combined rotation
   R = rotz(yaw) @ roty(pitch) @ rotx(roll)

Quaternions
^^^^^^^^^^^

Unit quaternions for 3D rotations:

.. code-block:: python

   from tracker_component_library.coordinate_systems import (
       quat_from_axis_angle,
       quat_from_rot_mat,
       quat_to_rot_mat,
       quat_multiply,
       quat_rotate,
   )

   # Create quaternion from axis-angle
   axis = np.array([0, 0, 1])  # z-axis
   angle = np.pi / 4  # 45 degrees
   q = quat_from_axis_angle(axis, angle)

   # Convert to rotation matrix
   R = quat_to_rot_mat(q)

   # Rotate a vector
   v = np.array([1, 0, 0])
   v_rotated = quat_rotate(q, v)

Euler Angles
^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.coordinate_systems import (
       euler_to_rot_mat,
       rot_mat_to_euler,
   )

   # Euler angles (roll, pitch, yaw) to rotation matrix
   R = euler_to_rot_mat(roll, pitch, yaw, order="zyx")

   # Rotation matrix to Euler angles
   roll, pitch, yaw = rot_mat_to_euler(R, order="zyx")

Axis-Angle
^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.coordinate_systems import (
       axis_angle_to_rot_mat,
       rot_mat_to_axis_angle,
   )

   axis = np.array([0, 0, 1])
   angle = np.pi / 4
   R = axis_angle_to_rot_mat(axis, angle)

   axis, angle = rot_mat_to_axis_angle(R)

Coordinate Jacobians
--------------------

Jacobians for coordinate transformations are essential for filter design:

.. code-block:: python

   from tracker_component_library.coordinate_systems import (
       cart2sphere_jacobian,
       sphere2cart_jacobian,
       geodetic2ecef_jacobian,
   )

   # Jacobian of Cartesian-to-spherical transformation
   J = cart2sphere_jacobian(x, y, z)

   # Use in EKF measurement update
   H = J @ H_cart  # Transform Cartesian Jacobian to spherical

WGS84 Ellipsoid
---------------

The library uses the WGS84 ellipsoid for geodetic calculations:

.. code-block:: python

   from tracker_component_library.core import WGS84

   print(f"Semi-major axis: {WGS84.a} m")
   print(f"Flattening: {WGS84.f}")
   print(f"Eccentricity: {WGS84.e}")
