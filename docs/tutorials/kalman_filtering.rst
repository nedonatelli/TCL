Kalman Filtering Tutorial
=========================

This tutorial demonstrates how to implement a Kalman filter for tracking
a moving object using noisy position measurements.

.. raw:: html

   <iframe src="../_static/images/tutorials/kalman_filtering.html" width="100%" height="450" frameborder="0"></iframe>

Problem Setup
-------------

We will track a 2D object moving with constant velocity. The state vector
contains position and velocity in both x and y dimensions:

.. math::

   x = [x, \dot{x}, y, \dot{y}]^T

Step 1: Import Required Modules
-------------------------------

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import kf_predict, kf_update

Step 2: Define the System Model
-------------------------------

The constant velocity model uses the following state transition:

.. code-block:: python

   dt = 0.1  # Time step

   # State transition matrix (constant velocity model)
   F = np.array([
       [1, dt, 0, 0],
       [0, 1,  0, 0],
       [0, 0,  1, dt],
       [0, 0,  0, 1]
   ])

   # Process noise covariance
   q = 0.1  # Process noise intensity
   Q = q * np.array([
       [dt**3/3, dt**2/2, 0,       0],
       [dt**2/2, dt,      0,       0],
       [0,       0,       dt**3/3, dt**2/2],
       [0,       0,       dt**2/2, dt]
   ])

   # Measurement matrix (we observe position only)
   H = np.array([
       [1, 0, 0, 0],
       [0, 0, 1, 0]
   ])

   # Measurement noise covariance
   R = np.eye(2) * 0.5

Step 3: Initialize the Filter
-----------------------------

.. code-block:: python

   # Initial state estimate
   x = np.array([0.0, 1.0, 0.0, 0.5])

   # Initial covariance (high uncertainty)
   P = np.eye(4) * 10.0

Step 4: Generate Simulated Data
-------------------------------

.. code-block:: python

   np.random.seed(42)
   n_steps = 100

   # True trajectory
   true_states = []
   x_true = np.array([0.0, 1.0, 0.0, 0.5])
   for _ in range(n_steps):
       true_states.append(x_true.copy())
       x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), Q)

   # Noisy measurements
   measurements = [H @ s + np.random.multivariate_normal(np.zeros(2), R)
                   for s in true_states]

Step 5: Run the Kalman Filter
-----------------------------

.. code-block:: python

   estimates = []

   for z in measurements:
       # Predict
       pred = kf_predict(x, P, F, Q)
       x, P = pred.x, pred.P

       # Update
       upd = kf_update(x, P, z, H, R)
       x, P = upd.x, upd.P

       estimates.append(x.copy())

Step 6: Analyze Results
-----------------------

.. code-block:: python

   # Convert to arrays
   true_states = np.array(true_states)
   estimates = np.array(estimates)
   measurements = np.array(measurements)

   # Compute position errors
   pos_errors = np.sqrt(
       (true_states[:, 0] - estimates[:, 0])**2 +
       (true_states[:, 2] - estimates[:, 2])**2
   )

   print(f"Mean position error: {np.mean(pos_errors):.3f}")
   print(f"Final position error: {pos_errors[-1]:.3f}")

Complete Example
----------------

Here is the complete code:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import kf_predict, kf_update

   # System parameters
   dt = 0.1
   F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                 [0, 0, 1, dt], [0, 0, 0, 1]])
   q = 0.1
   Q = q * np.array([[dt**3/3, dt**2/2, 0, 0],
                     [dt**2/2, dt, 0, 0],
                     [0, 0, dt**3/3, dt**2/2],
                     [0, 0, dt**2/2, dt]])
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 0.5

   # Initialize
   x = np.array([0.0, 1.0, 0.0, 0.5])
   P = np.eye(4) * 10.0

   # Generate data
   np.random.seed(42)
   x_true = np.array([0.0, 1.0, 0.0, 0.5])
   true_states, measurements = [], []
   for _ in range(100):
       true_states.append(x_true.copy())
       measurements.append(H @ x_true + np.random.multivariate_normal(
           np.zeros(2), R))
       x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), Q)

   # Filter
   estimates = []
   for z in measurements:
       pred = kf_predict(x, P, F, Q)
       upd = kf_update(pred.x, pred.P, z, H, R)
       x, P = upd.x, upd.P
       estimates.append(x.copy())

   # Results
   true_states = np.array(true_states)
   estimates = np.array(estimates)
   rmse = np.sqrt(np.mean((true_states[:, 0] - estimates[:, 0])**2 +
                          (true_states[:, 2] - estimates[:, 2])**2))
   print(f"Position RMSE: {rmse:.3f}")

Next Steps
----------

- Try the :doc:`nonlinear_filtering` tutorial for EKF and UKF examples
- See :doc:`/user_guide/filtering` for more filter variants
- Explore IMM estimators for maneuvering targets
