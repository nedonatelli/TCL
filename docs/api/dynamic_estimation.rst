Dynamic Estimation
==================

.. automodule:: pytcl.dynamic_estimation
   :no-members:
   :no-undoc-members:

Kalman Filters
--------------

.. automodule:: pytcl.dynamic_estimation.kalman
   :no-members:
   :no-undoc-members:

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

.. _constrained-ekf:

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
   :exclude-members: SRKalmanState, SRKalmanPrediction, SRKalmanUpdate

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
   :no-members:
   :no-undoc-members:

Bootstrap Particle Filter
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.dynamic_estimation.particle_filters.bootstrap
   :members:
   :undoc-members:
   :show-inheritance:

.. _rbpf:

Rao-Blackwellized Particle Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hybrid particle filter for systems with nonlinear and linear subsystems. Each
particle maintains an independent Kalman filter for the linear components,
reducing estimator variance.

.. automodule:: pytcl.dynamic_estimation.rbpf
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Sum Filter
-------------------

.. automodule:: pytcl.dynamic_estimation.gaussian_sum_filter
   :members:
   :undoc-members:
   :show-inheritance:

Information Filter
------------------

.. automodule:: pytcl.dynamic_estimation.information_filter
   :members:
   :undoc-members:
   :show-inheritance:

H Infinity
----------

.. automodule:: pytcl.dynamic_estimation.kalman.h_infinity
   :members:
   :undoc-members:
   :show-inheritance:

Matrix Utils
------------

.. automodule:: pytcl.dynamic_estimation.kalman.matrix_utils
   :members:
   :undoc-members:
   :show-inheritance:

Sr Ukf
------

.. automodule:: pytcl.dynamic_estimation.kalman.sr_ukf
   :members:
   :undoc-members:
   :show-inheritance:

Types
-----

.. automodule:: pytcl.dynamic_estimation.kalman.types
   :members:
   :undoc-members:
   :show-inheritance:

Ud Filter
---------

.. automodule:: pytcl.dynamic_estimation.kalman.ud_filter
   :members:
   :undoc-members:
   :show-inheritance:

Smoothers
---------

.. automodule:: pytcl.dynamic_estimation.smoothers
   :members:
   :undoc-members:
   :show-inheritance:
