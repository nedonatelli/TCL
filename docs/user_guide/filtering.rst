Filtering and State Estimation
==============================

This guide covers the filtering algorithms available in the library.

Kalman Filter Family
--------------------

Linear Kalman Filter
^^^^^^^^^^^^^^^^^^^^

The standard Kalman filter is optimal for linear-Gaussian systems:

.. math::

   x_{k+1} = F x_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)

   z_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)

**Prediction step:**

.. code-block:: python

   from pytcl.dynamic_estimation import kf_predict

   # x: state vector, P: covariance, F: transition matrix, Q: process noise
   prediction = kf_predict(x, P, F, Q)
   x_pred = prediction.x
   P_pred = prediction.P

**Update step:**

.. code-block:: python

   from pytcl.dynamic_estimation import kf_update

   # z: measurement, H: measurement matrix, R: measurement noise
   update = kf_update(x_pred, P_pred, z, H, R)
   x_upd = update.x
   P_upd = update.P

**Combined predict-update:**

.. code-block:: python

   from pytcl.dynamic_estimation import kf_predict_update

   update = kf_predict_update(x, P, z, F, Q, H, R)

Extended Kalman Filter (EKF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For nonlinear systems, the EKF linearizes around the current estimate:

.. code-block:: python

   from pytcl.dynamic_estimation import ekf_predict, ekf_update

   # Define nonlinear dynamics and Jacobian
   def f(x):
       # Nonlinear state transition
       return np.array([x[0] + x[1], x[1] * 0.99])

   def F_jacobian(x):
       return np.array([[1, 1], [0, 0.99]])

   # Predict
   pred = ekf_predict(x, P, f, F_jacobian, Q)

   # Define nonlinear measurement and Jacobian
   def h(x):
       # Range measurement
       return np.array([np.sqrt(x[0]**2 + x[1]**2)])

   def H_jacobian(x):
       r = np.sqrt(x[0]**2 + x[1]**2)
       return np.array([[x[0]/r, x[1]/r]])

   # Update
   upd = ekf_update(pred.x, pred.P, z, h, H_jacobian, R)

**Automatic Jacobian computation:**

.. code-block:: python

   from pytcl.dynamic_estimation import ekf_predict_auto, ekf_update_auto

   # Uses numerical differentiation
   pred = ekf_predict_auto(x, P, f, Q)
   upd = ekf_update_auto(pred.x, pred.P, z, h, R)

Unscented Kalman Filter (UKF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The UKF uses sigma points to capture the mean and covariance through
nonlinear transformations:

.. code-block:: python

   from pytcl.dynamic_estimation import ukf_predict, ukf_update

   # No Jacobians needed!
   pred = ukf_predict(x, P, f, Q)
   upd = ukf_update(pred.x, pred.P, z, h, R)

Cubature Kalman Filter (CKF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CKF uses spherical-radial cubature points, providing a good balance
between accuracy and computational cost:

.. code-block:: python

   from pytcl.dynamic_estimation import ckf_predict, ckf_update

   pred = ckf_predict(x, P, f, Q)
   upd = ckf_update(pred.x, pred.P, z, h, R)

Particle Filters
----------------

For non-Gaussian or highly nonlinear systems, particle filters provide
a flexible Monte Carlo approach:

.. code-block:: python

   from pytcl.dynamic_estimation import (
       initialize_particles,
       bootstrap_pf_step,
       particle_mean,
       particle_covariance,
   )

   # Initialize particles from prior
   state = initialize_particles(x0, P0, N=1000)

   # Define process noise sampler
   def Q_sample(N, rng):
       return rng.multivariate_normal(np.zeros(2), Q, size=N)

   # Run filter step
   state = bootstrap_pf_step(
       state.particles, state.weights,
       z, f, h, Q_sample, R,
       resample_method="systematic"
   )

   # Extract estimates
   x_est = particle_mean(state.particles, state.weights)
   P_est = particle_covariance(state.particles, state.weights)

Smoothing
---------

The library provides RTS (Rauch-Tung-Striebel) smoothing for obtaining
optimal estimates using future measurements:

.. code-block:: python

   from pytcl.dynamic_estimation import kf_smooth

   # After forward filtering, run backward smoothing
   x_smooth, P_smooth = kf_smooth(
       x_filtered, P_filtered,
       x_predicted, P_predicted,
       F
   )

Information Filter
------------------

The information filter is the dual of the Kalman filter, working with
the information matrix (inverse covariance):

.. code-block:: python

   from pytcl.dynamic_estimation import (
       information_filter_predict,
       information_filter_update,
   )

   # Work with information form: y = P^{-1} x, Y = P^{-1}
   y_pred, Y_pred = information_filter_predict(y, Y, F, Q)
   y_upd, Y_upd = information_filter_update(y_pred, Y_pred, z, H, R)
