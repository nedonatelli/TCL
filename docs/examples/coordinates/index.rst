Coordinate Systems & Navigation
===============================

Examples demonstrating coordinate systems, navigation, and geodesy.

.. toctree::
   :maxdepth: 1

   coordinate_systems
   coordinate_visualization
   ins_gnss_navigation
   navigation_geodesy

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../../_static/images/examples/coordinate_rotations.html"></iframe>
   </div>

**Coordinate Rotations**: 3D visualization of rotation matrices and transformations.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`coordinate_systems.py <../../../examples/coordinate_systems.py>`
     - Coordinate conversions, rotations, and projections
   * - :download:`coordinate_visualization.py <../../../examples/coordinate_visualization.py>`
     - Interactive 3D visualizations of coordinate transforms

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/navigation_trajectory.html"></iframe>
   </div>

**Navigation Trajectory**: INS trajectory with measurement noise and integration errors.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`ins_gnss_navigation.py <../../../examples/ins_gnss_navigation.py>`
     - INS/GNSS integration for navigation
   * - :download:`navigation_geodesy.py <../../../examples/navigation_geodesy.py>`
     - Geodetic calculations, datum conversions, map projections
