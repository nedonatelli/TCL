3D Target Tracking
==================

This example demonstrates tracking targets in 3D space with range-azimuth-elevation measurements.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/tracking_3d.html"></iframe>
   </div>

Overview
--------

3D tracking presents unique challenges:

- **Spherical measurements**: Range, azimuth, and elevation from radar
- **Coordinate transformations**: Converting between measurement and state spaces
- **3D motion**: Constant-velocity filtering of maneuvering targets
- **Visualization**: Displaying tracks in 3D

Key Concepts
------------

- **Converted-measurement filtering**: Spherical radar measurements are
  transformed to Cartesian before a linear Kalman filter update
- **RTS smoothing**: Batch smoothing of the full 3D trajectory
- **Multi-sensor fusion**: Combining detections from several 3D sensors
- **Maneuvering targets**: Climbing and descending turns tracked with a
  constant-velocity model

Code Highlights
---------------

The example demonstrates:

- 6-state model: [x, vx, y, vy, z, vz]
- Range-azimuth-elevation measurements converted to Cartesian
- ``kf_predict()``/``kf_update()`` and ``rts_smoother()`` in 3D
- Plotly 3D visualization of trajectories and estimates

Source Code
-----------

.. literalinclude:: ../../../examples/tracking_3d.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/tracking_3d.py

See Also
--------

- :doc:`multi_target_tracking` - Multiple target tracking
- :doc:`../coordinates/coordinate_systems` - Coordinate transformations
- :doc:`../filtering/kalman_filter_comparison` - Filter variants
