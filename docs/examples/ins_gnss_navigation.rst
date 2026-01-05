INS/GNSS Navigation
===================

This example demonstrates Inertial Navigation System (INS) and GNSS integration.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/navigation_trajectory.html"></iframe>
   </div>

Overview
--------

INS/GNSS integration combines:

- **INS**: High-rate, smooth navigation with drift
- **GNSS**: Accurate but noisy absolute position
- **Integration**: Best of both systems

Key Concepts
------------

- **Strapdown mechanization**: Integrating IMU measurements
- **Error state**: Modeling INS drift
- **Loosely coupled**: GNSS position updates
- **Tightly coupled**: GNSS pseudorange updates

INS Mechanization
-----------------

Strapdown INS integrates:

1. **Accelerometers**: Specific force measurements
2. **Gyroscopes**: Angular rate measurements
3. **Attitude update**: Quaternion integration
4. **Velocity update**: Transform and integrate acceleration
5. **Position update**: Integrate velocity

Error Sources
-------------

- **Gyro bias**: Causes heading drift
- **Accelerometer bias**: Causes position drift
- **Scale factor errors**: Proportional errors
- **Coning/sculling**: Integration errors

Code Highlights
---------------

The example demonstrates:

- INS state initialization with ``INSState``
- Strapdown mechanization with ``ins_mechanization()``
- GNSS update with Kalman filter
- Error state estimation and correction
- Trajectory visualization

Source Code
-----------

.. literalinclude:: ../../examples/ins_gnss_navigation.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/ins_gnss_navigation.py

See Also
--------

- :doc:`navigation_geodesy` - Geodetic calculations
- :doc:`coordinate_systems` - Coordinate transformations
- :doc:`kalman_filter_comparison` - Filter for integration
