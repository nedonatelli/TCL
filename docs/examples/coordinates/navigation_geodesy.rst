Navigation and Geodesy
======================

This example demonstrates geodetic calculations, coordinate conversions, and
great-circle navigation.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/navigation_trajectory.html"></iframe>
   </div>

Overview
--------

Geodesy provides the mathematical foundation for navigation:

- **Geodetic datums**: Earth ellipsoid models (WGS84)
- **Distance calculations**: Vincenty, Haversine methods
- **Local frames**: ECEF and ENU conversions
- **Great circles**: Shortest paths on Earth

Geodetic Calculations
---------------------

**Vincenty's Formulae**
   - High accuracy (< 0.5mm)
   - Works for all distances
   - Handles antipodal points

**Haversine Formula**
   - Simpler calculation
   - Good for short distances
   - Assumes spherical Earth

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_earth.html"></iframe>
   </div>

**Earth Ellipsoid**: The WGS84 reference ellipsoid with coordinate frames at various locations.

Code Highlights
---------------

The example demonstrates:

- ``inverse_geodetic()`` (Vincenty) for accurate distances and azimuths
- ``direct_geodetic()`` for the point reached from a bearing and distance
- ``haversine_distance()`` for quick spherical-Earth distances
- ``geodetic_to_ecef()``/``ecef_to_geodetic()`` and ``ecef_to_enu()``/
  ``enu_to_ecef()`` frame conversions
- Multi-waypoint route planning and sensor coverage analysis

Source Code
-----------

.. literalinclude:: ../../../examples/navigation_geodesy.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/navigation_geodesy.py

See Also
--------

- :doc:`ins_gnss_navigation` - INS/GNSS integration
- :doc:`coordinate_systems` - Coordinate conversions
