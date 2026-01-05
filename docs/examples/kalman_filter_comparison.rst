Kalman Filter Comparison
========================

This example demonstrates different Kalman filter variants for target tracking.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/kalman_filter_comparison.html"></iframe>
   </div>

Overview
--------

This example compares three Kalman filter implementations:

1. **Linear Kalman Filter (KF)** - For linear state-space models with Gaussian noise
2. **Extended Kalman Filter (EKF)** - Linearizes nonlinear models around current estimate
3. **Unscented Kalman Filter (UKF)** - Uses sigma points for better nonlinear approximation

Key Concepts
------------

- **State estimation**: Estimating position and velocity from noisy measurements
- **Filter consistency**: NEES/NIS statistics for filter tuning validation
- **Measurement models**: Linear vs nonlinear (range-bearing) measurements
- **Process noise**: Modeling uncertainty in the motion model

Code Highlights
---------------

The example demonstrates:

- Creating state transition matrices with ``f_constant_velocity()``
- Process noise covariance with ``q_constant_velocity()``
- Sigma point generation with ``sigma_points_merwe()``
- Filter predict/update cycles with ``kf_predict()``, ``kf_update()``
- UKF operations with ``ukf_predict()``, ``ukf_update()``

Source Code
-----------

.. literalinclude:: ../../examples/kalman_filter_comparison.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/kalman_filter_comparison.py

See Also
--------

- :doc:`filter_uncertainty_visualization` - Covariance ellipse visualization
- :doc:`advanced_filters_comparison` - EKF, Gaussian Sum, Rao-Blackwellized PF
- :doc:`smoothers_information_filters` - RTS smoother and information filters
