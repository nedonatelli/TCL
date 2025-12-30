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

Square-Root Kalman Filters
--------------------------

Square-root filters propagate the Cholesky factor of the covariance matrix
instead of the covariance itself. This provides improved numerical stability
and guarantees positive semi-definiteness.

Square-Root Kalman Filter (SRKF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.dynamic_estimation import srkf_predict, srkf_update
   import numpy as np

   # Initialize with Cholesky factors instead of covariances
   x = np.array([0.0, 1.0, 0.0, 0.5])
   P = np.eye(4) * 0.1
   S = np.linalg.cholesky(P)  # S @ S.T = P

   # System matrices
   F = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0],
                 [0, 0, 1, 0.1], [0, 0, 0, 1]])
   Q = np.eye(4) * 0.01
   S_Q = np.linalg.cholesky(Q)

   # Prediction
   pred = srkf_predict(x, S, F, S_Q)
   x_pred, S_pred = pred.x, pred.S

   # Update
   z = np.array([0.1, 0.05])
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 0.1
   S_R = np.linalg.cholesky(R)

   upd = srkf_update(x_pred, S_pred, z, H, S_R)
   x_upd, S_upd = upd.x, upd.S

   # Reconstruct covariance if needed
   P_upd = S_upd @ S_upd.T

U-D Factorization Filter
^^^^^^^^^^^^^^^^^^^^^^^^

Bierman's U-D filter uses a different factorization: ``P = U @ D @ U.T``
where U is unit upper triangular and D is diagonal.

.. code-block:: python

   from pytcl.dynamic_estimation import (
       ud_factorize, ud_reconstruct,
       ud_predict, ud_update
   )

   # Factorize initial covariance
   P = np.diag([0.5, 1.0, 0.5, 1.0])
   U, D = ud_factorize(P)

   # Prediction
   x_pred, U_pred, D_pred = ud_predict(x, U, D, F, Q)

   # Update
   x_upd, U_upd, D_upd, innovation, likelihood = ud_update(
       x_pred, U_pred, D_pred, z, H, R
   )

   # Reconstruct covariance if needed
   P_upd = ud_reconstruct(U_upd, D_upd)

Square-Root UKF
^^^^^^^^^^^^^^^

The square-root UKF combines the benefits of the UKF (no Jacobians needed)
with numerical stability of square-root formulations.

.. code-block:: python

   from pytcl.dynamic_estimation import sr_ukf_predict, sr_ukf_update

   # Nonlinear state transition
   def f(x):
       return np.array([x[0] + x[1] * 0.1, x[1] * 0.99])

   # Nonlinear measurement
   def h(x):
       return np.array([np.sqrt(x[0]**2 + x[1]**2)])

   # Predict and update
   pred = sr_ukf_predict(x, S, f, S_Q)
   upd = sr_ukf_update(pred.x, pred.S, z, h, S_R)

Interacting Multiple Model (IMM) Estimator
------------------------------------------

The IMM estimator handles systems that can switch between multiple
dynamic models. Each model represents a different motion mode
(e.g., constant velocity vs. maneuvering).

Basic IMM Usage
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.dynamic_estimation import imm_predict, imm_update

   # Two modes: constant velocity (CV) and coordinated turn (CT)
   x = np.array([0.0, 10.0, 0.0, 5.0])  # [x, vx, y, vy]
   P = np.eye(4) * 1.0

   # Mode probabilities and transition matrix
   mu = np.array([0.9, 0.1])  # Start in CV mode
   Pi = np.array([[0.95, 0.05],   # CV -> CV, CV -> CT
                  [0.10, 0.90]])  # CT -> CV, CT -> CT

   # Model-specific dynamics
   dt = 0.1
   F_cv = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                    [0, 0, 1, dt], [0, 0, 0, 1]])
   F_ct = ... # Coordinated turn model

   Q_cv = np.eye(4) * 0.01
   Q_ct = np.eye(4) * 0.1  # Higher uncertainty for maneuvering

   # Predict
   pred = imm_predict(
       [x, x], [P, P], mu, Pi,
       [F_cv, F_ct], [Q_cv, Q_ct]
   )

   # Update
   z = np.array([0.5, 0.3])
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 0.1

   upd = imm_update(
       pred.mode_states, pred.mode_covs, pred.mode_probs,
       z, [H, H], [R, R]
   )

   # Combined estimate
   x_est = upd.x
   P_est = upd.P
   mode_probs = upd.mode_probs  # Current mode probabilities

IMMEstimator Class
^^^^^^^^^^^^^^^^^^

For stateful IMM filtering:

.. code-block:: python

   from pytcl.dynamic_estimation import IMMEstimator

   # Initialize
   Pi = np.array([[0.95, 0.05], [0.10, 0.90]])
   imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=Pi)

   imm.initialize(x0, P0)
   imm.set_mode_model(0, F_cv, Q_cv)
   imm.set_mode_model(1, F_ct, Q_ct)
   imm.set_measurement_model(H, R)

   # Filter loop
   for z in measurements:
       result = imm.predict_update(z)
       print(f"State: {result.x}, Mode probs: {imm.mode_probs}")
