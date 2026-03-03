Dynamic Estimation
==================

.. automodule:: pytcl.dynamic_estimation
   :members:
   :undoc-members:
   :show-inheritance:

Kalman Filters
--------------

.. automodule:: pytcl.dynamic_estimation.kalman
   :members:
   :undoc-members:
   :show-inheritance:

Linear Kalman Filter
^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.dynamic_estimation.kalman.linear
   :members:
   :undoc-members:
   :show-inheritance:

Extended Kalman Filter
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.dynamic_estimation.kalman.extended
   :members:
   :undoc-members:
   :show-inheritance:

Constrained Extended Kalman Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

State-constrained filtering using Lagrange multiplier methods. Enforces
equality and inequality constraints on the state estimate.

.. automodule:: pytcl.dynamic_estimation.kalman.constrained
   :members:
   :undoc-members:
   :show-inheritance:

Unscented & Cubature Kalman Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.dynamic_estimation.kalman.unscented
   :members:
   :undoc-members:
   :show-inheritance:

Square-Root Kalman Filters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Numerically stable Kalman filter implementations that propagate the
Cholesky factor of the covariance matrix.

.. automodule:: pytcl.dynamic_estimation.kalman.square_root
   :members:
   :undoc-members:
   :show-inheritance:

Interacting Multiple Model (IMM) Estimator
------------------------------------------

The IMM estimator handles systems with multiple possible dynamic modes.

.. automodule:: pytcl.dynamic_estimation.imm
   :members:
   :undoc-members:
   :show-inheritance:

Particle Filters
----------------

.. automodule:: pytcl.dynamic_estimation.particle_filters
   :members:
   :undoc-members:
   :show-inheritance:

Bootstrap Particle Filter
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.dynamic_estimation.particle_filters.bootstrap
   :members:
   :undoc-members:
   :show-inheritance:

Rao-Blackwellized Particle Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hybrid particle filter for systems with nonlinear and linear subsystems. Each
particle maintains an independent Kalman filter for the linear components,
reducing estimator variance.

.. automodule:: pytcl.dynamic_estimation.rbpf
   :members:
   :undoc-members:
   :show-inheritance:
