Terrain Modeling
================

This example demonstrates the pytcl.terrain module capabilities, including digital elevation model (DEM) creation, synthetic terrain generation, and terrain analysis.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coord_viz_earth.html"></iframe>
   </div>

Overview
--------

Terrain modeling is essential for:

- **Navigation**: Terrain-aided navigation and TERCOM
- **Simulation**: Realistic environment modeling
- **Line-of-sight**: Radio propagation and visibility
- **Mission planning**: Route optimization

Digital Elevation Models
------------------------

**Flat DEM**
   - Constant elevation surface
   - Useful for testing and baseline comparisons
   - Created with ``create_flat_dem()``

**Synthetic Terrain**
   - Procedurally generated terrain
   - Controllable parameters (amplitude, wavelength)
   - Useful for simulation and testing
   - Created with ``create_synthetic_terrain()``

DEM Properties
--------------

**Grid Structure**
   - Regular lat/lon grid
   - Specified resolution in arcseconds
   - Elevation values at each grid point

**Coordinate System**
   - Geographic coordinates (lat, lon)
   - Elevation in meters above reference

**Analysis Outputs**
   - Min, max, mean elevation
   - Standard deviation
   - Slope and aspect maps

Terrain Analysis
----------------

**Elevation Statistics**
   - Distribution of elevation values
   - Terrain roughness metrics
   - Histogram analysis

**Slope Computation**
   - Gradient magnitude at each point
   - Degrees from horizontal
   - Important for mobility analysis

**Horizon Computation**
   - Visible horizon from observer position
   - Accounts for terrain obstruction
   - Essential for line-of-sight analysis

Applications
------------

**Terrain-Aided Navigation**
   - Match measured terrain to DEM
   - Position fix without GPS
   - Submarine and aircraft navigation

**Viewshed Analysis**
   - Determine visible area from point
   - Radar coverage planning
   - Communication link analysis

**Route Planning**
   - Avoid steep terrain
   - Minimize exposure
   - Optimize fuel consumption

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/navigation_trajectory.html"></iframe>
   </div>

**Terrain-Aided Navigation**: Vehicle trajectories can be matched against terrain models for position updates without GPS.

Code Highlights
---------------

The example demonstrates:

- Flat DEM creation with ``create_flat_dem()``
- Synthetic terrain with ``create_synthetic_terrain()``
- Terrain statistics (min, max, mean, std)
- Slope computation using gradients
- Horizon computation with ``compute_horizon()``

Source Code
-----------

.. literalinclude:: ../../../examples/terrain_demo.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/terrain_demo.py

See Also
--------

- :doc:`ins_gnss_navigation` - Navigation applications
- :doc:`coordinate_systems` - Coordinate transformations
- :doc:`reference_frame_advanced` - Reference frames
