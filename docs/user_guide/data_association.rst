Data Association
================

This guide covers data association algorithms for multi-target tracking.

Overview
--------

In multi-target tracking, data association is the problem of determining
which measurements correspond to which tracks. The library provides several
approaches:

- **Global Nearest Neighbor (GNN)**: Simple, deterministic assignment
- **Joint Probabilistic Data Association (JPDA)**: Soft association with
  probability weighting

Global Nearest Neighbor (GNN)
-----------------------------

GNN assigns each track to at most one measurement based on minimum cost:

.. code-block:: python

   from pytcl.assignment_algorithms import gnn_association

   # Track states and covariances
   tracks = [np.array([0.0, 1.0]), np.array([10.0, -1.0])]
   covs = [np.eye(2) * 0.5, np.eye(2) * 0.5]

   # Measurements
   measurements = [np.array([0.1]), np.array([10.2]), np.array([50.0])]
   H = np.array([[1.0, 0.0]])
   R = np.array([[0.1]])

   # Associate
   result = gnn_association(tracks, covs, measurements, H, R)

   # result.assignments[i] gives the measurement index for track i
   # (-1 means no assignment)

Joint Probabilistic Data Association (JPDA)
-------------------------------------------

JPDA computes association probabilities for all feasible track-measurement
pairings and updates each track using a weighted combination of innovations.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from pytcl.assignment_algorithms import jpda_update

   # Track states and covariances
   tracks = [np.array([0.0, 1.0]), np.array([10.0, -1.0])]
   covs = [np.eye(2) * 0.5, np.eye(2) * 0.5]

   # Measurements (some may be clutter)
   measurements = np.array([[0.1], [10.2], [50.0]])
   H = np.array([[1.0, 0.0]])
   R = np.array([[0.1]])

   # JPDA update
   result = jpda_update(
       tracks, covs, measurements, H, R,
       detection_prob=0.9,      # Probability of detecting a target
       clutter_density=1e-4,    # Spatial clutter density
       gate_probability=0.99    # Gating probability (chi-squared)
   )

   # Updated tracks
   for i, (x, P) in enumerate(zip(result.states, result.covariances)):
       print(f"Track {i}: state={x}, cov diag={np.diag(P)}")

   # Association probabilities (rows=tracks, cols=[meas0, meas1, ..., no_meas])
   print(f"Association probabilities:\n{result.association_probs}")

Understanding JPDA Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Detection Probability (P_D)**

The probability that a target generates a measurement. Lower values increase
the probability of "no measurement" association:

.. code-block:: python

   # High detection probability: measurements are reliable
   result = jpda_update(tracks, covs, meas, H, R, detection_prob=0.99)

   # Low detection probability: expect missed detections
   result = jpda_update(tracks, covs, meas, H, R, detection_prob=0.5)

**Clutter Density**

The expected number of false alarms per unit volume of measurement space.
Higher values reduce confidence in measurement associations:

.. code-block:: python

   # Low clutter: measurements are mostly from targets
   result = jpda_update(tracks, covs, meas, H, R, clutter_density=1e-6)

   # High clutter: many false alarms expected
   result = jpda_update(tracks, covs, meas, H, R, clutter_density=1e-2)

**Gate Probability**

Controls the validation region size. Measurements outside the gate are
not considered for association:

.. code-block:: python

   # Tight gate: fewer candidate measurements
   result = jpda_update(tracks, covs, meas, H, R, gate_probability=0.95)

   # Loose gate: more candidate measurements
   result = jpda_update(tracks, covs, meas, H, R, gate_probability=0.9999)

JPDA Output Structure
^^^^^^^^^^^^^^^^^^^^^

The ``JPDAUpdate`` result contains:

- ``states``: Updated state estimates for all tracks
- ``covariances``: Updated covariance matrices
- ``innovations``: Innovation vectors for each track
- ``association_probs``: Association probability matrix

.. code-block:: python

   result = jpda_update(tracks, covs, measurements, H, R)

   # Association probabilities shape: (n_tracks, n_meas + 1)
   # Last column is probability of no measurement
   beta = result.association_probs

   for track_idx in range(len(tracks)):
       for meas_idx in range(len(measurements)):
           prob = beta[track_idx, meas_idx]
           print(f"P(track {track_idx} <- meas {meas_idx}) = {prob:.3f}")

       miss_prob = beta[track_idx, -1]
       print(f"P(track {track_idx} missed) = {miss_prob:.3f}")

Computing Likelihoods Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For custom association logic, you can compute the likelihood matrix directly:

.. code-block:: python

   from pytcl.assignment_algorithms import compute_likelihood_matrix

   likelihood_matrix, gated = compute_likelihood_matrix(
       tracks, covs, measurements, H, R,
       detection_prob=0.9,
       gate_threshold=9.21  # Chi-squared threshold for 2D at 99%
   )

   # likelihood_matrix[i, j] = likelihood of track i generating measurement j
   # gated[i, j] = True if measurement j is within track i's gate

Gating
------

Gating reduces computational cost by limiting which measurements are
considered for each track:

.. code-block:: python

   from pytcl.assignment_algorithms import (
       mahalanobis_distance,
       chi2_gate_threshold,
       ellipsoidal_gate
   )

   # Compute gate threshold for 99% probability, 2D measurement
   threshold = chi2_gate_threshold(0.99, df=2)  # Returns ~9.21

   # Check if measurement is within gate
   x = np.array([0.0, 1.0])
   P = np.eye(2) * 0.5
   z = np.array([0.1])
   H = np.array([[1.0, 0.0]])
   R = np.array([[0.1]])

   # Innovation covariance
   S = H @ P @ H.T + R

   # Mahalanobis distance
   y = z - H @ x
   d_sq = mahalanobis_distance(y, S)

   is_gated = d_sq <= threshold

Best Practices
--------------

1. **Tune detection probability** based on your sensor characteristics
2. **Set clutter density** based on environment (urban vs. open area)
3. **Use appropriate gating** to balance computational cost and association quality
4. **Monitor mode probabilities** in IMM-JPDA combinations for insight into target behavior
