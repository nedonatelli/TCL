Particle Filters
================

This example demonstrates bootstrap particle filters with various resampling methods.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/particle_filter_resampling.html"></iframe>
   </div>

Overview
--------

Particle filters (Sequential Monte Carlo) handle:

- **Nonlinear dynamics** - Arbitrary state transition functions
- **Non-Gaussian noise** - Any noise distribution
- **Multi-modal posteriors** - Multiple hypotheses

Key Concepts
------------

- **Importance sampling**: Weighting particles by likelihood
- **Resampling**: Eliminating low-weight particles
- **Effective sample size**: Measuring particle degeneracy
- **Roughening**: Preventing sample impoverishment

Resampling Methods
------------------

The example compares different resampling strategies:

1. **Multinomial** - Standard random resampling
2. **Systematic** - Evenly spaced samples on CDF
3. **Stratified** - Stratified random sampling
4. **Residual** - Deterministic + random resampling

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/particle_filters_demo.html"></iframe>
   </div>

Code Highlights
---------------

The example demonstrates:

- Bootstrap particle filter initialization
- Weight computation from likelihoods
- Different resampling implementations
- Effective sample size monitoring
- State estimation from weighted particles

Source Code
-----------

.. literalinclude:: ../../../examples/particle_filters.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/particle_filters.py

See Also
--------

- :doc:`advanced_filters_comparison` - Rao-Blackwellized particle filter
- :doc:`kalman_filter_comparison` - Kalman filter alternatives
- :doc:`multi_target_tracking` - Particle filters for MTT
