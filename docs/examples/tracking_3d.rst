3D Target Tracking
==================

This example demonstrates tracking targets in 3D space with range-azimuth-elevation measurements.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../_static/images/examples/tracking_3d.html"></iframe>
   </div>

Overview
--------

3D tracking presents unique challenges:

- **Spherical measurements**: Range, azimuth, and elevation from radar
- **Coordinate transformations**: Converting between measurement and state spaces
- **3D motion models**: Constant velocity, coordinated turn in 3D
- **Visualization**: Displaying tracks and uncertainty in 3D

Key Concepts
------------

- **Spherical-to-Cartesian conversion**: ``sphere2cart()`` and ``cart2sphere()``
- **Measurement Jacobians**: Linearization for EKF updates
- **3D covariance ellipsoids**: Visualizing uncertainty in 3D
- **Helical trajectories**: Constant turn rate with vertical motion

Code Highlights
---------------

The example demonstrates:

- 9-state model: [x, vx, ax, y, vy, ay, z, vz, az]
- Range-azimuth-elevation measurement model
- EKF with spherical measurement Jacobian
- Plotly 3D visualization with trajectory and uncertainty

Source Code
-----------

.. literalinclude:: ../../examples/tracking_3d.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/tracking_3d.py

See Also
--------

- :doc:`multi_target_tracking` - Multiple target tracking
- :doc:`coordinate_systems` - Coordinate transformations
- :doc:`kalman_filter_comparison` - Filter variants
