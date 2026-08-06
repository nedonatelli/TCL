Multi-Target Tracking Tutorial
==============================

This tutorial covers multi-target tracking algorithms for scenarios with
multiple objects and measurement-to-track association challenges.

.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/tutorials/multi_target_tracking.html"></iframe>
   </div>

Problem Overview
----------------

Multi-target tracking involves:

1. **Data Association**: Matching measurements to existing tracks
2. **Track Management**: Creating, maintaining, and deleting tracks
3. **State Estimation**: Filtering each track's state

Basic Multi-Target Tracker
--------------------------

The ``MultiTargetTracker`` uses Global Nearest Neighbor (GNN) association.

.. code-block:: python

   import numpy as np
   from pytcl.trackers import MultiTargetTracker

   # System model
   dt = 0.1
   F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                 [0, 0, 1, dt], [0, 0, 0, 1]])
   Q = np.eye(4) * 0.1
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 0.5

   # Create tracker (each track runs a linear Kalman filter)
   tracker = MultiTargetTracker(
       state_dim=4,           # [x, vx, y, vy]
       meas_dim=2,            # [x, y]
       F=F, H=H, Q=Q, R=R,
       gate_probability=0.99, # Chi-squared gate probability
       confirm_hits=3,        # Hits to confirm track
       max_misses=5,          # Misses to delete track
   )

Running the Tracker
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simulate measurements from multiple targets
   np.random.seed(42)

   # True target trajectories
   targets = [
       {'x0': np.array([0, 1, 0, 0.5]), 'active': (0, 100)},
       {'x0': np.array([50, -0.5, 20, 1]), 'active': (10, 80)},
       {'x0': np.array([30, 0, 50, -0.8]), 'active': (20, 100)},
   ]

   for t in range(100):
       # Generate measurements
       measurements = []
       for tgt in targets:
           if tgt['active'][0] <= t < tgt['active'][1]:
               x_true = F @ tgt['x0'] if t > tgt['active'][0] else tgt['x0']
               tgt['x0'] = x_true
               z = H @ x_true + np.random.multivariate_normal(np.zeros(2), R)
               measurements.append(z)

       # Add false alarms
       if np.random.rand() < 0.1:
           measurements.append(np.random.rand(2) * 100)

       # Update tracker (measurements is a list of vectors)
       tracker.process(measurements, dt)

       # Print confirmed tracks every 2 seconds
       if t % 20 == 0:
           for track in tracker.confirmed_tracks:
               print(f"t={t}: Track {track.id} at ({track.state[0]:.1f}, "
                     f"{track.state[2]:.1f})")

Data Association Algorithms
---------------------------

Gating
^^^^^^

Filter unlikely measurement-track associations:

.. code-block:: python

   from pytcl.assignment_algorithms import (
       chi2_gate_threshold, ellipsoidal_gate, gate_measurements
   )

   # Predicted measurement and innovation covariance for one track
   x_pred = np.array([5.0, 1.0, 3.0, 0.5])
   P_pred = np.eye(4)
   z_pred = H @ x_pred
   S = H @ P_pred @ H.T + R

   # Chi-squared threshold for a 2D measurement at 99% probability
   threshold = chi2_gate_threshold(0.99, num_dimensions=2)

   # Check if a measurement is in the gate (pass the innovation)
   z = np.array([5.2, 3.1])
   is_valid = ellipsoidal_gate(z - z_pred, S, threshold)

   # Or gate multiple candidates at once
   candidates = np.array([[5.2, 3.1], [10.5, 2.0], [100.0, 50.0]])
   valid_idx, distances = gate_measurements(z_pred, S, candidates, threshold)

Global Nearest Neighbor (GNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.assignment_algorithms import auction, hungarian

   # Cost matrix: tracks x measurements
   # Lower cost = better association
   cost_matrix = np.array([
       [1.2, 5.0, 100.0],   # Track 0 costs
       [4.5, 0.8, 50.0],    # Track 1 costs
       [90.0, 80.0, 2.1],   # Track 2 costs
   ])

   # Hungarian algorithm (optimal)
   track_to_meas, meas_to_track, cost = hungarian(cost_matrix)
   # track_to_meas[i] = measurement index for track i (-1 if unassigned)

   # Auction algorithm (faster for large problems)
   track_to_meas, meas_to_track, cost = auction(
       cost_matrix, epsilon=0.01
   )

Joint Probabilistic Data Association (JPDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JPDA computes association probabilities and updates each track with a
probability-weighted combination of the gated measurements:

.. code-block:: python

   from pytcl.assignment_algorithms import jpda_update

   # Two tracks and three measurements
   track_states = [
       np.array([5.0, 1.0, 3.0, 0.5]),
       np.array([10.0, -0.5, 2.0, 0.2]),
   ]
   track_covariances = [np.eye(4), np.eye(4)]
   measurements = np.array([[5.2, 3.1], [10.5, 2.0], [7.0, 2.5]])

   upd = jpda_update(
       track_states, track_covariances, measurements, H, R,
       detection_prob=0.9,
       clutter_density=1e-4,
   )

   # upd.states: updated state per track
   # upd.covariances: updated covariance per track
   # upd.association_probs[i, j] = P(measurement j from track i)
   #   (last column: missed detection)
   # upd.innovations: combined innovation per track

Multiple Hypothesis Tracking (MHT)
----------------------------------

MHT maintains multiple association hypotheses over time.

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.trackers import MHTTracker, MHTConfig

   config = MHTConfig(
       n_scan=3,                # N-scan pruning depth
       max_hypotheses=100,      # Maximum hypotheses to maintain
       detection_prob=0.9,
       clutter_density=1e-4,
       gate_probability=0.99,
       min_hypothesis_prob=0.01,  # Minimum hypothesis probability
   )

   mht = MHTTracker(
       state_dim=4,
       meas_dim=2,
       F=F, H=H, Q=Q, R=R,
       config=config,
   )

Running MHT
^^^^^^^^^^^

.. code-block:: python

   # Two well-separated targets over a few scans
   all_measurements = [
       [np.array([0.0, 50.0]), np.array([100.0, 50.0])],
       [np.array([0.1, 50.0]), np.array([99.9, 50.1])],
       [np.array([0.2, 50.1]), np.array([99.8, 50.0])],
       [np.array([0.3, 49.9]), np.array([99.7, 50.2])],
   ]

   for t, measurements in enumerate(all_measurements):
       result = mht.process(measurements, dt)

       # Best hypothesis tracks
       for track in result.confirmed_tracks:
           print(f"Track {track.id}: state={track.state}")

       # Hypothesis tree info
       print(f"Active hypotheses: {result.n_hypotheses}")
       print(f"Best hypothesis probability: {result.best_hypothesis_prob:.4f}")

Hypothesis Management
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.trackers import generate_joint_associations

   # Boolean gating matrix: gated[i, j] = True when measurement j
   # falls inside track i's gate
   gated = np.array([
       [True, False, True],
       [False, True, False],
   ])

   # Enumerate every feasible joint association; each entry maps
   # track index -> measurement index
   associations = generate_joint_associations(gated, n_tracks=2, n_meas=3)
   print(f"{len(associations)} feasible joint associations")

Low-probability hypotheses are pruned each scan with
``prune_hypotheses_by_probability``, and ``n_scan_prune`` discards branches
that disagree with the best hypothesis more than ``n_scan`` scans back.
``MHTTracker`` applies both automatically using the ``MHTConfig`` limits.

Track Metrics
-------------

Evaluate tracking performance using standard metrics.

OSPA Metric
^^^^^^^^^^^

.. code-block:: python

   from pytcl.performance_evaluation import ospa

   # True target positions
   truth = [np.array([10.0, 20.0]), np.array([30.0, 40.0]),
            np.array([50.0, 60.0])]

   # Estimated track positions (missing one target)
   estimates = [np.array([10.5, 19.8]), np.array([30.2, 40.5])]

   # OSPA distance (order 2, cutoff 100)
   ospa_result = ospa(truth, estimates, c=100.0, p=2)
   print(f"OSPA: {ospa_result.ospa:.2f}")
   print(f"  Localization: {ospa_result.localization:.2f}")
   print(f"  Cardinality: {ospa_result.cardinality:.2f}")

Track Quality Metrics
^^^^^^^^^^^^^^^^^^^^^

GOSPA is not implemented. Alongside OSPA the library provides the CLEAR MOT
metrics and per-track quality measures, which answer the questions GOSPA is
usually reached for -- how much of each true track was held, and how often
identity was lost.

.. code-block:: python

   from pytcl.performance_evaluation import (
       mot_metrics,
       track_purity,
       track_fragmentation,
       identity_switches,
   )

   # Lists of per-scan position lists (2 scans, 2 targets)
   ground_truth = [
       [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
       [np.array([11.0, 20.5]), np.array([29.5, 40.5])],
   ]
   estimated = [
       [np.array([10.2, 19.9]), np.array([30.1, 40.2])],
       [np.array([11.1, 20.4])],
   ]
   metrics = mot_metrics(ground_truth, estimated, threshold=10.0)
   print(f"MOTA: {metrics.mota:.3f}  MOTP: {metrics.motp:.3f}")

   # label-based measures, given true and estimated track labels per detection
   true_labels = np.array([0, 0, 0, 1, 1, 1])
   est_labels = np.array([0, 0, 1, 1, 1, 1])
   print(f"purity:        {track_purity(true_labels, est_labels):.3f}")
   print(f"fragments:     {track_fragmentation(true_labels, est_labels)}")
   print(f"ID switches:   {identity_switches(true_labels, est_labels)}")

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from pytcl.trackers import MultiTargetTracker
   from pytcl.performance_evaluation import ospa

   # Setup
   np.random.seed(42)
   dt = 0.1

   F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                 [0, 0, 1, dt], [0, 0, 0, 1]])
   Q = np.eye(4) * 0.01
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 1.0

   tracker = MultiTargetTracker(
       state_dim=4, meas_dim=2,
       F=F, H=H, Q=Q, R=R,
       gate_probability=0.99,
       confirm_hits=3, max_misses=5
   )

   # Simulate 3 crossing targets
   n_steps = 100
   targets = [
       np.array([0, 1, 50, 0]),      # Moving right
       np.array([100, -1, 50, 0]),   # Moving left
       np.array([50, 0, 0, 1]),      # Moving up
   ]

   ospa_values = []

   for t in range(n_steps):
       # Propagate true states
       truth_positions = []
       measurements = []

       for i, x in enumerate(targets):
           targets[i] = F @ x
           truth_positions.append([targets[i][0], targets[i][2]])

           # Detection probability 0.9
           if np.random.rand() < 0.9:
               z = H @ targets[i] + np.random.multivariate_normal(
                   np.zeros(2), R
               )
               measurements.append(z)

       # Add clutter
       n_clutter = np.random.poisson(0.5)
       for _ in range(n_clutter):
           measurements.append(np.random.rand(2) * 100)

       # Update tracker
       tracker.process(measurements, dt)
       tracks = tracker.confirmed_tracks

       # Compute OSPA on confirmed track positions
       estimates = [np.array([tr.state[0], tr.state[2]]) for tr in tracks]
       truth = [np.asarray(p) for p in truth_positions]

       ospa_result = ospa(truth, estimates, c=50.0, p=2)
       ospa_values.append(ospa_result.ospa)

   print(f"Mean OSPA: {np.mean(ospa_values):.2f}")
   print(f"Final tracks: {len(tracks)}")

Next Steps
----------

- See :doc:`/api/trackers` for complete tracker API
- Explore :doc:`/api/assignment_algorithms` for association methods
- Check :doc:`/api/performance_evaluation` for more metrics
