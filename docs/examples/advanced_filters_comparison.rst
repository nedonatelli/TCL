Advanced Filters Comparison
===========================

This example compares advanced filtering techniques for challenging nonlinear problems.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/advanced_filters_comparison.html"></iframe>
   </div>

Overview
--------

When standard Kalman filters are insufficient, advanced techniques provide better performance:

1. **Constrained EKF** - Enforces state constraints during estimation
2. **Gaussian Sum Filter** - Represents multi-modal distributions
3. **Rao-Blackwellized Particle Filter** - Combines analytic and Monte Carlo methods

Key Concepts
------------

- **State constraints**: Physical bounds on state variables
- **Multi-modality**: Distributions with multiple peaks
- **Hybrid filters**: Combining different estimation techniques
- **Marginalization**: Analytically integrating out linear states

Code Highlights
---------------

The example demonstrates:

- Implementing state constraints in EKF updates
- Gaussian mixture representation and merging
- Rao-Blackwellization for linear substructure
- Performance comparison metrics

Source Code
-----------

.. literalinclude:: ../../examples/advanced_filters_comparison.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/advanced_filters_comparison.py

See Also
--------

- :doc:`kalman_filter_comparison` - Basic Kalman filter variants
- :doc:`particle_filters` - Standard particle filters
- :doc:`gaussian_mixtures` - Gaussian mixture operations
