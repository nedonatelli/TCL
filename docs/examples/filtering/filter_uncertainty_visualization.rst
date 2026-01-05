Filter Uncertainty Visualization
================================

This example visualizes filter covariance ellipses and uncertainty propagation.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/filter_uncertainty.html"></iframe>
   </div>

Overview
--------

Understanding and visualizing filter uncertainty is crucial for:

- **Tuning filter parameters** - Ensuring appropriate uncertainty levels
- **Detecting filter divergence** - Identifying when estimates become unreliable
- **Validating consistency** - Checking that actual errors match predicted uncertainty

Key Concepts
------------

- **Covariance ellipses**: 2D/3D visualization of multivariate Gaussian uncertainty
- **Uncertainty propagation**: How uncertainty grows during prediction steps
- **Measurement updates**: How measurements reduce uncertainty
- **Sigma contours**: 1-sigma, 2-sigma, 3-sigma probability regions

Code Highlights
---------------

The example demonstrates:

- Plotting covariance ellipses from filter covariance matrices
- Animating uncertainty evolution over time
- Comparing predicted vs actual estimation errors
- Visualizing measurement update effects

Source Code
-----------

.. literalinclude:: ../../../examples/filter_uncertainty_visualization.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/filter_uncertainty_visualization.py

See Also
--------

- :doc:`kalman_filter_comparison` - Kalman filter variants
- :doc:`particle_filters` - Particle filter uncertainty representation
