Dynamic Models
==============

This example demonstrates dynamic models and state transition matrices used in Kalman filtering and target tracking.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/kalman_filter_comparison.html"></iframe>
   </div>

Overview
--------

Dynamic models describe how a target's state evolves over time. They are fundamental to:

- **Kalman filtering**: Prediction step requires state transition
- **Target tracking**: Motion models for different target types
- **Navigation**: INS mechanization and error propagation
- **Simulation**: Generating realistic target trajectories

Key Concepts
------------

**State Transition Matrix (Phi)**
   The discrete-time matrix that propagates state from time k to k+1:
   ``x[k+1] = Phi * x[k] + process_noise``

**Process Noise Covariance (Q)**
   Captures uncertainty in the motion model due to unknown accelerations
   or model mismatch.

**Continuous vs Discrete Time**
   - Continuous: differential equations (dx/dt = F*x)
   - Discrete: difference equations (x[k+1] = Phi*x[k])
   - Conversion: Phi = exp(F*dt)

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/filter_viz_tracking_ellipses.html"></iframe>
   </div>

**State Estimation**: The state transition matrix propagates the state estimate and covariance through time, shown as uncertainty ellipses.

Models Demonstrated
-------------------

**Constant Velocity (CV)**
   - State: [x, vx, y, vy, z, vz]
   - Assumes constant velocity between updates
   - Process noise models unknown accelerations

**Drift Functions**
   - Continuous-time rate of change
   - Position changes at velocity rate
   - Velocity remains constant (for CV model)

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/tracking_3d_kalman.html"></iframe>
   </div>

**3D Tracking**: Dynamic models enable prediction of 3D target trajectories using the state transition matrix.

Code Highlights
---------------

The example demonstrates:

- State transition matrix computation with ``f_constant_velocity()``
- Process noise covariance with ``diffusion_constant_velocity()``
- Drift function evaluation with ``drift_constant_velocity()``
- Continuous to discrete time conversion

Source Code
-----------

.. literalinclude:: ../../../examples/dynamic_models_demo.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/dynamic_models_demo.py

See Also
--------

- :doc:`kalman_filter_comparison` - Kalman filter implementations
- :doc:`multi_target_tracking` - Multi-target tracking
- :doc:`tracking_3d` - 3D tracking example
