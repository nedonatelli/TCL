Coordinate Systems
==================

The library provides comprehensive coordinate system conversions and
rotation representations. All angles are in radians.

Coordinate Conversions
----------------------

Cartesian and Spherical
^^^^^^^^^^^^^^^^^^^^^^^

``cart2sphere`` returns a ``(range, azimuth, elevation)`` tuple. The
``system_type`` keyword selects the convention: ``'standard'`` (physics
convention, polar angle from +z) or ``'az-el'`` (tracking convention,
elevation from the xy-plane):

.. code-block:: python

   import numpy as np

   from pytcl.coordinate_systems import cart2sphere, sphere2cart

   # Cartesian to spherical (range, azimuth, elevation)
   point = np.array([100.0, 200.0, 50.0])
   r, az, el = cart2sphere(point, system_type='az-el')
   print(f"range={r:.3f}, azimuth={az:.6f}, elevation={el:.6f}")

   # Spherical to Cartesian
   cart = sphere2cart(r, az, el, system_type='az-el')
   print(f"cartesian: {cart.round(6)}")

Output:

.. code-block:: text

   range=229.129, azimuth=1.107149, elevation=0.219988
   cartesian: [100. 200.  50.]

Geodetic and ECEF
^^^^^^^^^^^^^^^^^

Convert between geodetic coordinates (latitude, longitude, altitude) and
Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates.
``geodetic2ecef`` returns an ECEF array; ``ecef2geodetic`` takes an ECEF
array and returns a ``(lat, lon, alt)`` tuple:

.. code-block:: python

   from pytcl.coordinate_systems import geodetic2ecef, ecef2geodetic

   # Geodetic to ECEF (angles in radians)
   lat = np.deg2rad(40.0)   # 40 degrees North
   lon = np.deg2rad(-75.0)  # 75 degrees West
   alt = 100.0              # meters

   ecef = geodetic2ecef(lat, lon, alt)
   print(f"ECEF: {ecef.round(1)}")

   # ECEF to geodetic
   lat2, lon2, alt2 = ecef2geodetic(ecef)
   print(f"lat={np.rad2deg(lat2):.6f} deg, lon={np.rad2deg(lon2):.6f} deg, "
         f"alt={alt2:.3f} m")

Output:

.. code-block:: text

   ECEF: [ 1266345.7 -4726066.6  4078049.9]
   lat=40.000000 deg, lon=-75.000000 deg, alt=100.000 m

ENU and NED
^^^^^^^^^^^

Local tangent plane coordinates. ``ecef2enu`` takes the ECEF point, the
reference latitude/longitude, and the reference point's ECEF position:

.. code-block:: python

   from pytcl.coordinate_systems import ecef2enu, enu2ecef

   # Convert ECEF to local East-North-Up about an origin
   origin_lat, origin_lon = np.deg2rad(40.0), np.deg2rad(-75.0)
   origin_ecef = geodetic2ecef(origin_lat, origin_lon, 0.0)

   enu = ecef2enu(ecef, origin_lat, origin_lon, ecef_ref=origin_ecef)
   print(f"ENU: {enu.round(3)}")

Output:

.. code-block:: text

   ENU: [  0.   0. 100.]

``ned2ecef``, ``ecef2ned``, ``enu2ned``, ``ned2enu``, and
``geodetic2enu`` cover the remaining local-frame conversions.

Rotation Representations
------------------------

The library supports multiple rotation representations and conversions
between them.

Rotation Matrices
^^^^^^^^^^^^^^^^^

Elementary rotations about principal axes:

.. code-block:: python

   from pytcl.coordinate_systems import rotx, roty, rotz

   # Rotation about x-axis by 30 degrees
   Rx = rotx(np.deg2rad(30))

   # Combined rotation
   roll, pitch, yaw = 0.1, 0.2, 0.3
   R = rotz(yaw) @ roty(pitch) @ rotx(roll)

Quaternions
^^^^^^^^^^^

Unit quaternions for 3D rotations. Quaternions are built from other
representations (``rotmat2quat``, ``euler2quat``) and converted back with
``quat2rotmat`` / ``quat2euler``:

.. code-block:: python

   from pytcl.coordinate_systems import (
       axisangle2rotmat,
       rotmat2quat,
       quat2rotmat,
       quat_multiply,
       quat_rotate,
   )

   # Create quaternion from axis-angle (via a rotation matrix)
   axis = np.array([0.0, 0.0, 1.0])  # z-axis
   angle = np.pi / 4                 # 45 degrees
   q = rotmat2quat(axisangle2rotmat(axis, angle))

   # Convert to rotation matrix
   R = quat2rotmat(q)

   # Rotate a vector
   v = np.array([1.0, 0.0, 0.0])
   v_rotated = quat_rotate(q, v)
   print(f"Rotated vector: {v_rotated.round(6)}")

Output:

.. code-block:: text

   Rotated vector: [0.707107 0.707107 0.      ]

``quat_conjugate``, ``quat_inverse``, and ``slerp`` (spherical linear
interpolation) are also available.

Euler Angles
^^^^^^^^^^^^

``euler2rotmat`` takes a sequence of angles and a rotation order
(default ``"ZYX"``, i.e. yaw-pitch-roll); ``rotmat2euler`` returns the
angles as an array in the same order:

.. code-block:: python

   from pytcl.coordinate_systems import euler2rotmat, rotmat2euler

   # Euler angles (yaw, pitch, roll) to rotation matrix, ZYX order
   R = euler2rotmat([yaw, pitch, roll], sequence="ZYX")

   # Rotation matrix to Euler angles
   angles = rotmat2euler(R, sequence="ZYX")
   print(f"yaw={angles[0]:.3f}, pitch={angles[1]:.3f}, roll={angles[2]:.3f}")

Output:

.. code-block:: text

   yaw=0.300, pitch=0.200, roll=0.100

Axis-Angle
^^^^^^^^^^

.. code-block:: python

   from pytcl.coordinate_systems import axisangle2rotmat, rotmat2axisangle

   axis = np.array([0.0, 0.0, 1.0])
   angle = np.pi / 4
   R = axisangle2rotmat(axis, angle)

   axis_out, angle_out = rotmat2axisangle(R)
   print(f"axis={axis_out.round(6)}, angle={angle_out:.6f}")

Output:

.. code-block:: text

   axis=[0. 0. 1.], angle=0.785398

Coordinate Jacobians
--------------------

Jacobians for coordinate transformations are essential for filter design:

.. code-block:: python

   from pytcl.coordinate_systems import (
       spherical_jacobian,
       spherical_jacobian_inv,
       geodetic_jacobian,
   )

   # Jacobian of the Cartesian-to-spherical transformation at a point
   J = spherical_jacobian([100.0, 200.0, 50.0], system_type="standard")
   print(f"Jacobian shape: {J.shape}")

   # Jacobian of the geodetic-to-ECEF transformation
   Jg = geodetic_jacobian(lat, lon, alt)

Output:

.. code-block:: text

   Jacobian shape: (3, 3)

``spherical_jacobian`` can be used as the measurement Jacobian ``H`` in
an EKF with spherical (range/angle) measurements. ``polar_jacobian``,
``enu_jacobian``, ``ned_jacobian``, ``ruv_jacobian``, and
``numerical_jacobian`` cover other measurement models.

WGS84 Ellipsoid
---------------

The library uses the WGS84 ellipsoid for geodetic calculations:

.. code-block:: python

   from pytcl.core import WGS84

   print(f"Semi-major axis: {WGS84.a} m")
   print(f"Flattening: {WGS84.f}")
   print(f"Eccentricity: {WGS84.e}")

Output:

.. code-block:: text

   Semi-major axis: 6378137.0 m
   Flattening: 0.0033528106647474805
   Eccentricity: 0.08181919084262149
