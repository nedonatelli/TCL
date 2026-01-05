Nonlinear Filtering Tutorial
============================

This tutorial covers Extended Kalman Filter (EKF) and Unscented Kalman
Filter (UKF) for tracking with nonlinear measurements.

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/tutorials/nonlinear_filtering.html"></iframe>
   </div>

Problem Setup
-------------

We track an object in 2D using range and bearing measurements from a
sensor at the origin. The measurement model is nonlinear:

.. math::

   h(x) = \begin{bmatrix}
       \sqrt{x^2 + y^2} \\
       \arctan(y/x)
   \end{bmatrix}

Extended Kalman Filter
----------------------

The EKF linearizes the nonlinear functions using Jacobians.

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import ekf_predict, ekf_update

   # State: [x, vx, y, vy]
   dt = 0.1

   # Linear dynamics (constant velocity)
   def f(x):
       F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                     [0, 0, 1, dt], [0, 0, 0, 1]])
       return F @ x

   def F_jacobian(x):
       return np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, dt], [0, 0, 0, 1]])

   # Nonlinear measurement (range, bearing)
   def h(x):
       r = np.sqrt(x[0]**2 + x[2]**2)
       theta = np.arctan2(x[2], x[0])
       return np.array([r, theta])

   def H_jacobian(x):
       r = np.sqrt(x[0]**2 + x[2]**2)
       return np.array([
           [x[0]/r, 0, x[2]/r, 0],
           [-x[2]/r**2, 0, x[0]/r**2, 0]
       ])

   # Filter parameters
   Q = np.diag([0.01, 0.1, 0.01, 0.1])
   R = np.diag([0.5, 0.01])  # Range (m), bearing (rad)

   # Initialize
   x = np.array([100.0, -5.0, 50.0, 2.0])
   P = np.diag([10.0, 1.0, 10.0, 1.0])

   # Filter step - compute Jacobians at current/predicted state
   F = F_jacobian(x)  # Evaluate Jacobian at current state
   pred = ekf_predict(x, P, f, F, Q)
   z = np.array([112.0, 0.47])  # Measurement
   H = H_jacobian(pred.x)  # Evaluate Jacobian at predicted state
   upd = ekf_update(pred.x, pred.P, z, h, H, R)

Automatic Jacobian Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If analytical Jacobians are difficult to derive, use numerical differentiation:

.. code-block:: python

   from pytcl.dynamic_estimation import ekf_predict_auto, ekf_update_auto

   # Same f and h functions, no Jacobians needed
   pred = ekf_predict_auto(x, P, f, Q)
   upd = ekf_update_auto(pred.x, pred.P, z, h, R)

Unscented Kalman Filter
-----------------------

The UKF uses sigma points to propagate uncertainty through nonlinear
functions, avoiding Jacobian computation entirely.

.. code-block:: python

   from pytcl.dynamic_estimation import ukf_predict, ukf_update

   # Same f and h functions
   pred = ukf_predict(x, P, f, Q)
   upd = ukf_update(pred.x, pred.P, z, h, R)

The UKF is often more accurate than the EKF for highly nonlinear systems.

Cubature Kalman Filter
----------------------

The CKF uses spherical-radial cubature points, providing a good balance
between accuracy and computational cost:

.. code-block:: python

   from pytcl.dynamic_estimation import ckf_predict, ckf_update

   pred = ckf_predict(x, P, f, Q)
   upd = ckf_update(pred.x, pred.P, z, h, R)

Complete Tracking Example
-------------------------

Here is a complete range-bearing tracking example comparing EKF and UKF:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import (
       ekf_predict, ekf_update,
       ukf_predict, ukf_update
   )

   # Setup
   dt = 0.1
   n_steps = 100
   np.random.seed(42)

   def f(x):
       F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                     [0, 0, 1, dt], [0, 0, 0, 1]])
       return F @ x

   def F_jac(x):
       return np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, dt], [0, 0, 0, 1]])

   def h(x):
       r = np.sqrt(x[0]**2 + x[2]**2)
       theta = np.arctan2(x[2], x[0])
       return np.array([r, theta])

   def H_jac(x):
       r = np.sqrt(x[0]**2 + x[2]**2)
       return np.array([[x[0]/r, 0, x[2]/r, 0],
                        [-x[2]/r**2, 0, x[0]/r**2, 0]])

   Q = np.diag([0.01, 0.1, 0.01, 0.1])
   R = np.diag([1.0, 0.02])

   # Generate truth and measurements
   x_true = np.array([100.0, -5.0, 50.0, 2.0])
   truth, measurements = [], []
   for _ in range(n_steps):
       truth.append(x_true.copy())
       z_true = h(x_true)
       z_noisy = z_true + np.random.multivariate_normal(np.zeros(2), R)
       measurements.append(z_noisy)
       x_true = f(x_true) + np.random.multivariate_normal(np.zeros(4), Q)

   # Run EKF
   x_ekf = np.array([100.0, -5.0, 50.0, 2.0])
   P_ekf = np.diag([10.0, 1.0, 10.0, 1.0])
   ekf_est = []
   for z in measurements:
       F = F_jac(x_ekf)  # Evaluate Jacobian at current state
       pred = ekf_predict(x_ekf, P_ekf, f, F, Q)
       H = H_jac(pred.x)  # Evaluate Jacobian at predicted state
       upd = ekf_update(pred.x, pred.P, z, h, H, R)
       x_ekf, P_ekf = upd.x, upd.P
       ekf_est.append(x_ekf.copy())

   # Run UKF
   x_ukf = np.array([100.0, -5.0, 50.0, 2.0])
   P_ukf = np.diag([10.0, 1.0, 10.0, 1.0])
   ukf_est = []
   for z in measurements:
       pred = ukf_predict(x_ukf, P_ukf, f, Q)
       upd = ukf_update(pred.x, pred.P, z, h, R)
       x_ukf, P_ukf = upd.x, upd.P
       ukf_est.append(x_ukf.copy())

   # Compare
   truth = np.array(truth)
   ekf_est = np.array(ekf_est)
   ukf_est = np.array(ukf_est)

   ekf_rmse = np.sqrt(np.mean((truth[:, 0] - ekf_est[:, 0])**2 +
                              (truth[:, 2] - ekf_est[:, 2])**2))
   ukf_rmse = np.sqrt(np.mean((truth[:, 0] - ukf_est[:, 0])**2 +
                              (truth[:, 2] - ukf_est[:, 2])**2))

   print(f"EKF Position RMSE: {ekf_rmse:.3f}")
   print(f"UKF Position RMSE: {ukf_rmse:.3f}")

Filter Selection Guidelines
---------------------------

- **EKF**: Use when you have analytical Jacobians and moderate nonlinearity
- **UKF**: Use for highly nonlinear systems or when Jacobians are unavailable
- **CKF**: Good compromise between EKF and UKF computational cost

Next Steps
----------

- See :doc:`/user_guide/filtering` for particle filters and smoothing
- Try :doc:`ins_gnss_integration` for navigation applications
