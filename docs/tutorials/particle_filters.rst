Particle Filtering Tutorial
============================

This tutorial demonstrates Sequential Monte Carlo (particle filter) methods
for nonlinear, non-Gaussian state estimation, and compares a bootstrap
particle filter against an Extended Kalman Filter on the same problem.

Topics covered:

- Bootstrap particle filter (BPF): prediction, weighting, resampling
- The effective sample size (ESS) resampling trigger
- Weight degeneracy and why resampling is needed
- Head-to-head comparison with an EKF on a nonlinear oscillator

Nonlinear System
-----------------

The tutorial's test system has state ``[position, velocity]`` with a
sinusoidal nonlinearity in both the dynamics and the measurement:

.. code-block:: python

   import numpy as np

   def process_model(x, dt):
       return np.array(
           [
               x[0] + x[1] * dt + 0.5 * np.sin(x[0]) * dt**2,
               x[1] + np.sin(x[0]) * dt,
           ]
       )

   def measurement_model(x):
       return np.array([x[0] ** 2])

True Trajectory and Measurements
-----------------------------------

The true trajectory is propagated through ``process_model``; each
measurement is the noisy output of ``measurement_model``:

.. code-block:: python

   np.random.seed(42)
   dt = 0.1
   n_steps = 150

   x_true = np.zeros((n_steps, 2))
   x_true[0] = [0.0, 1.0]
   for k in range(1, n_steps):
       x_true[k] = process_model(x_true[k - 1], dt)

   r_std = 0.5  # measurement noise std
   z_all = np.zeros((n_steps, 1))
   for k in range(n_steps):
       z_all[k] = measurement_model(x_true[k]) + np.random.randn() * r_std

Bootstrap Particle Filter
---------------------------

Each particle is propagated through the (possibly noisy) process model,
reweighted by measurement likelihood, and resampled whenever the effective
sample size drops below half the particle count:

.. code-block:: python

   n_particles = 100
   q_std = 0.05  # process noise std

   particles = np.random.randn(n_particles, 2) * 0.1
   particles[:, 1] = 1.0
   weights = np.ones(n_particles) / n_particles
   x_est = np.zeros((n_steps, 2))

   for k in range(n_steps):
       # Predict
       for i in range(n_particles):
           particles[i] = process_model(particles[i], dt) + np.random.randn(2) * q_std

       # Weight by measurement likelihood
       for i in range(n_particles):
           residual = z_all[k, 0] - measurement_model(particles[i])[0]
           weights[i] *= np.exp(-0.5 * residual**2 / r_std**2)
       weights /= np.sum(weights)

       x_est[k] = np.average(particles, axis=0, weights=weights)

       # Systematic resampling on ESS drop
       if 1.0 / np.sum(weights**2) < n_particles / 2:
           positions = (np.arange(n_particles) + np.random.rand()) / n_particles
           indices = np.searchsorted(np.cumsum(weights), positions)
           particles = particles[indices]
           weights = np.ones(n_particles) / n_particles

   rmse_bpf = np.sqrt(np.mean((x_est - x_true) ** 2))

Comparison with the Extended Kalman Filter
--------------------------------------------

The same trajectory is filtered with a hand-rolled EKF (analytic Jacobians of
``process_model``/``measurement_model``) so the two RMSE curves can be
compared directly:

.. code-block:: python

   x_ekf = np.zeros((n_steps, 2))
   x_ekf[0] = [0.0, 1.0]
   P = np.eye(2)
   Q = np.eye(2) * q_std**2
   R = np.array([[r_std**2]])

   for k in range(1, n_steps):
       x_pred = process_model(x_ekf[k - 1], dt)

       x_prev = x_ekf[k - 1]
       F = np.array([[1.0, dt], [np.cos(x_prev[0]) * dt, 1.0]])
       P = F @ P @ F.T + Q

       z_pred = measurement_model(x_pred)
       residual = z_all[k] - z_pred

       H = np.array([[2 * x_pred[0], 0.0]])
       S = H @ P @ H.T + R
       K = P @ H.T / S[0, 0]

       x_ekf[k] = x_pred + K.flatten() * residual[0]
       P = (np.eye(2) - K @ H) @ P

   rmse_ekf = np.sqrt(np.mean((x_ekf - x_true) ** 2))
   print(f"Bootstrap PF RMSE: {rmse_bpf:.4f}")
   print(f"Extended KF RMSE: {rmse_ekf:.4f}")

Which one wins depends on how strongly the sinusoidal term dominates near
the operating point -- the particle filter tends to hold up better as the
nonlinearity grows, at the cost of more compute per step.

Next Steps
----------

- See :doc:`nonlinear_filtering` for the EKF/UKF/CKF family in more depth
- See :doc:`/api/dynamic_estimation` for the library's particle filter and
  Kalman filter implementations (this tutorial's filters are minimal,
  from-scratch versions for illustration)
- See :doc:`smoothing_algorithms` for improving estimates with a backward pass
