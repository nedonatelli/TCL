Filtering & Estimation
======================

Examples demonstrating Kalman filters, particle filters, smoothers, and static estimation.

.. toctree::
   :maxdepth: 1

   kalman_filter_comparison
   filter_uncertainty_visualization
   advanced_filters_comparison
   smoothers_information_filters
   particle_filters
   static_estimation

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../../_static/images/examples/kalman_filter_comparison.html"></iframe>
   </div>

**Kalman Filter Comparison**: Linear KF vs EKF vs UKF performance on 1D tracking.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`kalman_filter_comparison.py <../../../examples/kalman_filter_comparison.py>`
     - Compare Linear Kalman Filter, EKF, and UKF performance

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/filter_uncertainty.html"></iframe>
   </div>

**Filter Uncertainty Visualization**: Visualize filter covariance ellipses and uncertainty propagation.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`filter_uncertainty_visualization.py <../../../examples/filter_uncertainty_visualization.py>`
     - Visualize filter covariance ellipses and uncertainty propagation

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/advanced_filters_comparison.html"></iframe>
   </div>

**Advanced Filters**: Extended Kalman Filter, Gaussian Sum Filter, and Rao-Blackwellized Particle Filter comparison.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`advanced_filters_comparison.py <../../../examples/advanced_filters_comparison.py>`
     - Constrained EKF, Gaussian Sum Filter, and Rao-Blackwellized Particle Filter

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../../_static/images/examples/smoothers_information_filters_result.html"></iframe>
   </div>

**RTS Smoother vs Kalman Filter**: Fixed-interval smoothing improves estimates using future measurements.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`smoothers_information_filters.py <../../../examples/smoothers_information_filters.py>`
     - RTS smoother and information filter demonstrations

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/particle_filter_resampling.html"></iframe>
   </div>

**Particle Filter Tracking**: Interactive visualization of particle filter with resampling.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`particle_filters.py <../../../examples/particle_filters.py>`
     - Bootstrap particle filter with different resampling methods

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/particle_filters_demo.html"></iframe>
   </div>

**Particle Filtering for Nonlinear Tracking**: Tracking nonlinear trajectories using particle filter.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/static_estimation.html"></iframe>
   </div>

**Static Position Estimation**: Least squares, weighted least squares, and RANSAC estimation.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :download:`static_estimation.py <../../../examples/static_estimation.py>`
     - Weighted least squares, RANSAC, batch estimation
