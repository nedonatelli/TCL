Smoothing Algorithms Tutorial
================================

This tutorial demonstrates state smoothing: using the *entire* measurement
sequence, including future measurements, to improve estimates at each past
time step -- something a causal filter cannot do.

Topics covered:

- Forward Kalman filtering as the baseline
- The Rauch-Tung-Striebel (RTS) backward smoothing pass
- Comparing filtered vs. smoothed position/velocity RMSE
- How smoothing shrinks estimation uncertainty relative to filtering alone

System and True Trajectory
------------------------------

The tutorial uses a constant-velocity model with a hidden acceleration phase
in the middle of the trajectory (so the filter has to "catch up" when it
starts, and lags after it ends):

.. code-block:: python

   import numpy as np

   np.random.seed(42)
   dt = 0.1
   n_steps = 100

   F = np.array([[1, dt], [0, 1]])
   H = np.array([[1, 0]])
   Q = np.eye(2) * 0.01
   R = np.array([[0.1]])

   x_true = np.zeros((n_steps, 2))
   x_true[0] = [0.0, 1.0]
   for k in range(1, n_steps):
       x_true[k] = F @ x_true[k - 1]
       if 30 < k < 70:  # hidden acceleration phase
           x_true[k] += np.array([0, 0.1])

   z_all = np.zeros((n_steps, 1))
   for k in range(n_steps):
       z_all[k] = H @ x_true[k] + np.random.randn() * np.sqrt(R[0, 0])

Forward Kalman Filter
------------------------

.. code-block:: python

   x_filt = np.zeros((n_steps, 2))
   P_filt = np.zeros((n_steps, 2, 2))
   x_filt[0], P_filt[0] = [0.0, 1.0], np.eye(2)

   for k in range(1, n_steps):
       x_pred = F @ x_filt[k - 1]
       P_pred = F @ P_filt[k - 1] @ F.T + Q

       innovation = z_all[k] - H @ x_pred
       S = H @ P_pred @ H.T + R
       K = P_pred @ H.T / S[0, 0]

       x_filt[k] = x_pred + K.flatten() * innovation[0]
       P_filt[k] = (np.eye(2) - K @ H) @ P_pred

   rmse_filt = np.sqrt(np.mean((x_filt - x_true) ** 2))

RTS Backward Smoother
-------------------------

The RTS pass walks backward from the last filtered estimate, correcting each
step using the smoothed estimate one step ahead of it:

.. code-block:: python

   x_smooth = np.zeros_like(x_filt)
   P_smooth = np.zeros_like(P_filt)
   x_smooth[-1], P_smooth[-1] = x_filt[-1], P_filt[-1]

   for k in range(n_steps - 2, -1, -1):
       x_pred_next = F @ x_filt[k]
       P_pred_next = F @ P_filt[k] @ F.T + Q

       A = P_filt[k] @ F.T @ np.linalg.inv(P_pred_next)

       x_smooth[k] = x_filt[k] + A @ (x_smooth[k + 1] - x_pred_next)
       P_smooth[k] = P_filt[k] + A @ (P_smooth[k + 1] - P_pred_next) @ A.T

   rmse_smooth = np.sqrt(np.mean((x_smooth - x_true) ** 2))
   print(f"Filter RMSE: {rmse_filt:.4f}")
   print(f"RTS Smoother RMSE: {rmse_smooth:.4f}")

Because the smoother incorporates information from both directions, its
uncertainty (``P_smooth``) is uniformly at or below the filter's
(``P_filt``) at every time step, and its RMSE against the true trajectory is
lower -- most visibly during the hidden-acceleration phase, where the causal
filter has no way to anticipate the change.

Next Steps
----------

- See :doc:`kalman_filtering` for the forward filter on its own
- See :doc:`particle_filters` for smoothing-adjacent techniques on
  nonlinear/non-Gaussian problems
- See :doc:`/api/dynamic_estimation` for the library's filter and smoother
  implementations (fixed-interval, fixed-lag, and two-filter smoothers)
