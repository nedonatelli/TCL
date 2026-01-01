Examples
========

Standalone example scripts demonstrating pytcl functionality.

These examples are complete, runnable Python scripts that you can use
as starting points for your own applications.

.. contents:: Example Categories
   :local:
   :depth: 1

Filtering & Estimation
----------------------

.. raw:: html

   <iframe src="/_static/images/examples/kalman_filter_comparison.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Kalman Filter Comparison**: Interactive comparison of Linear KF, EKF, and UKF performance.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`kalman_filter_comparison.py <../../examples/kalman_filter_comparison.py>`
     - Compare Linear Kalman Filter, EKF, and UKF performance
   * - :download:`filter_uncertainty_visualization.py <../../examples/filter_uncertainty_visualization.py>`
     - Visualize filter covariance ellipses and uncertainty propagation
   * - :download:`smoothers_information_filters.py <../../examples/smoothers_information_filters.py>`
     - RTS smoother and information filter demonstrations

Particle Filters
----------------

.. raw:: html

   <iframe src="/_static/images/examples/particle_linear_tracking.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Particle Filter Tracking**: Interactive visualization of particle filter with resampling.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`particle_filters.py <../../examples/particle_filters.py>`
     - Bootstrap particle filter with different resampling methods

Multi-Target Tracking
---------------------

.. raw:: html

   <iframe src="/_static/images/examples/multi_target_tracking_result.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Multi-Target Tracking**: Interactive GNN-based multi-target tracker visualization.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`multi_target_tracking.py <../../examples/multi_target_tracking.py>`
     - GNN-based multi-target tracker with track management
   * - :download:`tracking_3d.py <../../examples/tracking_3d.py>`
     - 3D target tracking with range-azimuth-elevation measurements
   * - :download:`tracking_containers.py <../../examples/tracking_containers.py>`
     - Track and measurement container data structures

Data Association
----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`assignment_algorithms.py <../../examples/assignment_algorithms.py>`
     - Hungarian, auction, GNN, JPDA, and Murty's k-best algorithms

Performance Evaluation
----------------------

.. raw:: html

   <iframe src="/_static/images/examples/performance_evaluation.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Performance Evaluation**: Interactive OSPA metric and tracking quality visualization.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`performance_evaluation.py <../../examples/performance_evaluation.py>`
     - OSPA, GOSPA, track-to-truth assignment metrics

Clustering & Gaussian Mixtures
------------------------------

.. raw:: html

   <iframe src="/_static/images/examples/gaussian_kmeans.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Clustering Comparison**: Interactive K-Means visualization with DBSCAN comparison.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`gaussian_mixtures.py <../../examples/gaussian_mixtures.py>`
     - Gaussian mixture operations, K-means, DBSCAN, hierarchical clustering
   * - :download:`spatial_data_structures.py <../../examples/spatial_data_structures.py>`
     - KD-trees and R-trees for efficient spatial queries

Signal Processing
-----------------

.. raw:: html

   <iframe src="/_static/images/examples/matched_filter.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Signal Processing**: Interactive matched filter and CFAR detection visualization.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`signal_processing.py <../../examples/signal_processing.py>`
     - Filter design, matched filtering, CFAR detection

Coordinate Systems
------------------

.. raw:: html

   <iframe src="/_static/images/examples/coord_viz_rotation_axes.html" width="100%" height="600" frameborder="0" style="margin: 20px 0;"></iframe>

**Coordinate Rotations**: Interactive 3D visualization of rotation matrices.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`coordinate_systems.py <../../examples/coordinate_systems.py>`
     - Coordinate conversions, rotations, and projections
   * - :download:`coordinate_visualization.py <../../examples/coordinate_visualization.py>`
     - Interactive 3D visualizations of coordinate transforms

Navigation & Geodesy
--------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`navigation_geodesy.py <../../examples/navigation_geodesy.py>`
     - Geodetic calculations, datum conversions, map projections
   * - :download:`ins_gnss_navigation.py <../../examples/ins_gnss_navigation.py>`
     - INS/GNSS integration for navigation

Transforms
----------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`transforms.py <../../examples/transforms.py>`
     - FFT, power spectrum, wavelets, and other transforms

Orbital Mechanics
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`orbital_mechanics.py <../../examples/orbital_mechanics.py>`
     - Orbit propagation, Kepler's equation, Lambert problem

Geophysical Models
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`geophysical_models.py <../../examples/geophysical_models.py>`
     - Gravity (EGM96/EGM2008), magnetic field (WMM/IGRF), tidal effects

Static Estimation
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`static_estimation.py <../../examples/static_estimation.py>`
     - Weighted least squares, RANSAC, batch estimation


Running Examples
----------------

All examples can be run directly from the repository root::

   python examples/kalman_filter_comparison.py
   python examples/multi_target_tracking.py

Or from the examples directory::

   cd examples
   python kalman_filter_comparison.py

Requirements
------------

Examples require pytcl to be installed::

   pip install -e .

Some examples require additional dependencies for visualization::

   pip install plotly kaleido  # For interactive and static plots

Generating Documentation Images
-------------------------------

To regenerate the static images shown in this documentation::

   python scripts/generate_example_plots.py

This will create PNG images in ``docs/_static/images/examples/``.
