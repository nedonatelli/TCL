Coordinate Visualization
========================

This example provides interactive 3D visualizations of coordinate transforms.

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

Code Highlights
---------------

The example demonstrates:

- 3D plotting of coordinate frames
- Animated rotation sequences
- Interactive frame selection
- Projection comparison views

Source Code
-----------

.. literalinclude:: ../../examples/coordinate_visualization.py
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
