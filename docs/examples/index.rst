Examples
========

Standalone example scripts demonstrating pytcl functionality.

These examples are complete, runnable Python scripts that you can use
as starting points for your own applications.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Individual Examples

   kalman_filter_comparison
   filter_uncertainty_visualization
   advanced_filters_comparison
   smoothers_information_filters
   particle_filters
   multi_target_tracking
   tracking_3d
   tracking_containers
   assignment_algorithms
   performance_evaluation
   gaussian_mixtures
   spatial_data_structures
   signal_processing
   coordinate_systems
   coordinate_visualization
   ins_gnss_navigation
   navigation_geodesy
   transforms
   orbital_mechanics
   relativity_demo
   ephemeris_demo
   geophysical_models
   magnetism_demo
   atmospheric_modeling
   dynamic_models_demo
   reference_frame_advanced
   special_functions_demo
   static_estimation
   terrain_demo

.. raw:: html

   <style>
   .plotly-container {
       position: relative;
       width: 100%;
       padding-bottom: 56.25%; /* 16:9 aspect ratio */
       margin-bottom: 1rem;
       overflow: hidden;
       border-radius: 8px;
       background: var(--pytcl-bg-secondary, #0d1117);
   }
   .plotly-container.aspect-4-3 {
       padding-bottom: 75%; /* 4:3 aspect ratio for some plots */
   }
   .plotly-container.aspect-wide {
       padding-bottom: 45%; /* Wider aspect ratio */
   }
   .plotly-container.aspect-square {
       padding-bottom: 80%; /* Taller for 3D plots */
   }
   .plotly-iframe {
       position: absolute;
       top: 0;
       left: 0;
       width: 100%;
       height: 100%;
       border: none;
       border-radius: 8px;
   }
   /* Fallback for older approach */
   iframe.plotly-iframe:not(.plotly-container iframe) {
       width: 100%;
       height: 550px;
       border: none;
       border-radius: 8px;
       background: var(--pytcl-bg-secondary, #0d1117);
       margin-bottom: 1rem;
   }
   @media (max-width: 768px) {
       .plotly-container {
           padding-bottom: 75%; /* Taller on mobile */
       }
   }
   </style>

.. contents:: Example Categories
   :local:
   :depth: 1

Filtering & Estimation
----------------------

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/kalman_filter_comparison.html"></iframe>
   </div>

**Kalman Filter Comparison**: Linear KF vs EKF vs UKF performance on 1D tracking.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`kalman_filter_comparison.py <../../examples/kalman_filter_comparison.py>`
     - Compare Linear Kalman Filter, EKF, and UKF performance

**Filter Uncertainty Visualization**: Visualize filter covariance ellipses and uncertainty propagation.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/filter_uncertainty.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`filter_uncertainty_visualization.py <../../examples/filter_uncertainty_visualization.py>`
     - Visualize filter covariance ellipses and uncertainty propagation

**Advanced Filters**: Extended Kalman Filter, Gaussian Sum Filter, and Rao-Blackwellized Particle Filter comparison.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/advanced_filters_comparison.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`advanced_filters_comparison.py <../../examples/advanced_filters_comparison.py>`
     - Constrained EKF, Gaussian Sum Filter, and Rao-Blackwellized Particle Filter

**RTS Smoother vs Kalman Filter**: Fixed-interval smoothing improves estimates using future measurements.

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/smoothers_information_filters_result.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`smoothers_information_filters.py <../../examples/smoothers_information_filters.py>`
     - RTS smoother and information filter demonstrations

Particle Filters
----------------

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/particle_filter_resampling.html"></iframe>
   </div>

**Particle Filter Tracking**: Interactive visualization of particle filter with resampling.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`particle_filters.py <../../examples/particle_filters.py>`
     - Bootstrap particle filter with different resampling methods

**Particle Filtering for Nonlinear Tracking**: Tracking nonlinear trajectories using particle filter.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/particle_filters_demo.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`particle_filters.py <../../examples/particle_filters.py>`
     - Additional particle filter demonstrations and comparison

Multi-Target Tracking
---------------------

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/multi_target_tracking.html"></iframe>
   </div>

**Multi-Target Tracking**: GNN-based multi-target tracker with track management.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`multi_target_tracking.py <../../examples/multi_target_tracking.py>`
     - GNN-based multi-target tracker with track management

**3D Target Tracking**: Helical trajectory tracking with noisy measurements in 3D space.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../_static/images/examples/tracking_3d.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`tracking_3d.py <../../examples/tracking_3d.py>`
     - Advanced 3D target tracking with range-azimuth-elevation measurements

**Track Spatial Distribution**: Track positions and spatial relationships in tracking scenario.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`tracking_containers.py <../../examples/tracking_containers.py>`
     - Track and measurement container data structures

Data Association
----------------

**Assignment Algorithms**: Interactive cost matrix visualization for 2D assignment.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/assignment_algorithms.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`assignment_algorithms.py <../../examples/assignment_algorithms.py>`
     - Hungarian, auction, GNN, JPDA, and Murty's k-best algorithms

Performance Evaluation
----------------------

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/performance_evaluation.html"></iframe>
   </div>

**Performance Evaluation**: Interactive OSPA metric and tracking quality visualization.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`performance_evaluation.py <../../examples/performance_evaluation.py>`
     - OSPA, GOSPA, track-to-truth assignment metrics

Clustering & Gaussian Mixtures
------------------------------

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/clustering_comparison.html"></iframe>
   </div>

**Clustering Comparison**: Interactive K-Means visualization with DBSCAN comparison.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`gaussian_mixtures.py <../../examples/gaussian_mixtures.py>`
     - Gaussian mixture operations, K-means, DBSCAN, hierarchical clustering

**Gaussian Mixture Model**: Clustering with Gaussian mixture models showing cluster centers.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/gaussian_mixtures.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`spatial_data_structures.py <../../examples/spatial_data_structures.py>`
     - KD-trees and R-trees for efficient spatial queries

Signal Processing
-----------------

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/signal_processing.html"></iframe>
   </div>

**Digital Filters**: Butterworth vs FIR filter frequency response comparison.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`signal_processing.py <../../examples/signal_processing.py>`
     - Filter design, matched filtering, CFAR detection

Coordinate Systems
------------------

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/coordinate_rotations.html"></iframe>
   </div>

**Coordinate Rotations**: 3D visualization of rotation matrices and transformations.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`coordinate_systems.py <../../examples/coordinate_systems.py>`
     - Coordinate conversions, rotations, and projections
   * - :download:`coordinate_visualization.py <../../examples/coordinate_visualization.py>`
     - Interactive 3D visualizations of coordinate transforms

**Spherical-Cartesian Transforms**: Converting between coordinate systems in 3D space.

Navigation & Geodesy
--------------------

**Navigation Trajectory**: INS trajectory with measurement noise and integration errors.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/navigation_trajectory.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`ins_gnss_navigation.py <../../examples/ins_gnss_navigation.py>`
     - INS/GNSS integration for navigation
   * - :download:`navigation_geodesy.py <../../examples/navigation_geodesy.py>`
     - Geodetic calculations, datum conversions, map projections

Transforms
----------

**FFT Analysis**: Time and frequency domain visualization of multi-frequency signal.

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/examples/transforms_fft.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`transforms.py <../../examples/transforms.py>`
     - FFT, power spectrum, wavelets, and other transforms

Orbital Mechanics
-----------------

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../_static/images/examples/orbital_propagation.html"></iframe>
   </div>

**Orbit Propagation**: Interactive visualization of orbital mechanics and trajectory analysis.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`orbital_mechanics.py <../../examples/orbital_mechanics.py>`
     - Orbit propagation, Kepler's equation, Lambert problem

**Relativistic Effects**: Gravitational time dilation and relativity effects in space systems.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/relativity_effects.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`relativity_demo.py <../../examples/relativity_demo.py>`
     - Relativistic effects: time dilation, precession, Shapiro delay

Geophysical Models
------------------

**Earth Gravity and Magnetic Field Models**: EGM96/EGM2008 gravity anomaly and WMM/IGRF magnetic field.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/geophysical_models.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`geophysical_models.py <../../examples/geophysical_models.py>`
     - Gravity (EGM96/EGM2008), magnetic field (WMM/IGRF), tidal effects

Static Estimation
-----------------

**Static Position Estimation**: Least squares, weighted least squares, and RANSAC estimation.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/static_estimation.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`static_estimation.py <../../examples/static_estimation.py>`
     - Weighted least squares, RANSAC, batch estimation

Special Functions
-----------------

**Mathematical Special Functions**: Bessel functions, error function, and other special functions used in signal processing.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/special_functions.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`special_functions_demo.py <../../examples/special_functions_demo.py>`
     - Bessel, Marcum Q, Lambert W, hypergeometric functions

Ephemeris & Celestial Mechanics
-------------------------------

**Planetary Ephemeris**: JPL Development Ephemeris (DE) planetary positions and celestial mechanics.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../_static/images/examples/ephemeris_demo.html"></iframe>
   </div>

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`ephemeris_demo.py <../../examples/ephemeris_demo.py>`
     - JPL ephemeris access, planetary positions, celestial coordinates


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
