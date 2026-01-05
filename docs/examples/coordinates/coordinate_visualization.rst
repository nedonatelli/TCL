Coordinate Visualization
========================

This example provides interactive 3D visualizations of coordinate transforms.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_earth.html"></iframe>
   </div>

Overview
--------

Visualizing coordinate transformations helps understand:

- **Rotation effects**: How rotations change frame orientation
- **Projection distortions**: Map projection artifacts
- **Frame relationships**: ECEF, ENU, NED orientations
- **Interpolation paths**: Quaternion vs Euler interpolation

Key Concepts
------------

- **Frame axes**: Visualizing coordinate system orientation
- **Transformation chains**: Sequential rotations
- **Geodetic surface**: Earth ellipsoid visualization
- **Great circles**: Shortest paths on sphere

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_rotation_axes.html"></iframe>
   </div>

**Rotation Axes**: Visualizing how rotations affect coordinate frame orientation.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_euler_sequence.html"></iframe>
   </div>

**Euler Sequences**: ZYX, ZXZ, and other Euler angle conventions produce different rotation paths.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_slerp.html"></iframe>
   </div>

**SLERP Interpolation**: Spherical linear interpolation provides smooth rotation paths between orientations.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_spherical.html"></iframe>
   </div>

**Spherical Coordinates**: Converting between Cartesian and spherical coordinate systems.

Code Highlights
---------------

The example demonstrates:

- 3D plotting of coordinate frames
- Animated rotation sequences
- Interactive frame selection
- Projection comparison views

Source Code
-----------

.. literalinclude:: ../../../examples/coordinate_visualization.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/coordinate_visualization.py

See Also
--------

- :doc:`coordinate_systems` - Coordinate conversion functions
- :doc:`navigation_geodesy` - Geodetic operations
