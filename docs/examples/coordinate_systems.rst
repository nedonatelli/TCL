Coordinate Systems
==================

This example demonstrates coordinate conversions, rotations, and projections.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/coordinate_rotations.html"></iframe>
   </div>

Overview
--------

Coordinate transformations are fundamental for tracking:

- **Cartesian-Spherical**: Range, azimuth, elevation conversions
- **Geodetic-ECEF**: Earth-fixed coordinates
- **Local frames**: ENU, NED transformations
- **Map projections**: UTM, Mercator, Lambert

Coordinate Systems
------------------

**Cartesian (x, y, z)**
   - Standard 3D coordinates
   - Used for state estimation

**Spherical (range, azimuth, elevation)**
   - Sensor-centric measurements
   - Radar and lidar output

**Geodetic (latitude, longitude, altitude)**
   - Geographic coordinates
   - Navigation reference

**ECEF (Earth-Centered, Earth-Fixed)**
   - Rotates with Earth
   - GPS coordinates

**ENU/NED (Local Tangent Plane)**
   - East-North-Up or North-East-Down
   - Local navigation frame

Rotation Representations
------------------------

- **Rotation matrices**: 3x3 orthogonal matrices
- **Quaternions**: 4D unit vectors, singularity-free
- **Euler angles**: Roll, pitch, yaw
- **Axis-angle**: Rotation axis and angle

Code Highlights
---------------

The example demonstrates:

- ``cart2sphere()`` and ``sphere2cart()``
- ``geodetic_to_ecef()`` and ``ecef_to_geodetic()``
- ``ecef2enu()`` and ``enu2ecef()``
- ``euler2quat()`` and ``quat2euler()``
- ``slerp()`` for quaternion interpolation

Source Code
-----------

.. literalinclude:: ../../examples/coordinate_systems.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/coordinate_systems.py

See Also
--------

- :doc:`coordinate_visualization` - 3D visualizations
- :doc:`navigation_geodesy` - Geodetic calculations
- :doc:`tracking_3d` - Using coordinates in tracking
