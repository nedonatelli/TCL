Navigation and Geodesy
======================

This example demonstrates geodetic calculations, datum conversions, and map projections.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/navigation_trajectory.html"></iframe>
   </div>

Overview
--------

Geodesy provides the mathematical foundation for navigation:

- **Geodetic datums**: Earth ellipsoid models (WGS84)
- **Distance calculations**: Vincenty, Haversine methods
- **Map projections**: UTM, Mercator, Lambert Conformal
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

**Rhumb Lines**
   - Constant bearing paths
   - Longer than great circles
   - Easier navigation

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_earth.html"></iframe>
   </div>

**Earth Ellipsoid**: The WGS84 reference ellipsoid with coordinate frames at various locations.

Map Projections
---------------

**UTM (Universal Transverse Mercator)**
   - Low distortion in zones
   - Standard military/civilian use

**Mercator**
   - Conformal (preserves angles)
   - Used for marine navigation

**Lambert Conformal Conic**
   - Low distortion for mid-latitudes
   - Used for aeronautical charts

Code Highlights
---------------

The example demonstrates:

- ``geodetic_distance_vincenty()`` for accurate distances
- ``geodetic_direct()`` for point from bearing/distance
- ``utm_to_geodetic()`` and ``geodetic_to_utm()``
- Map projection functions

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
