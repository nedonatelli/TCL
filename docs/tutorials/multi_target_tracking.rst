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
   from pytcl.trackers import MultiTargetTracker, Track, TrackStatus

   # Create tracker
   tracker = MultiTargetTracker(
       state_dim=4,           # [x, vx, y, vy]
       meas_dim=2,            # [x, y]
       gate_threshold=9.21,   # Chi-squared threshold (99%)
       min_hits=3,            # Hits to confirm track
       max_misses=5,          # Misses to delete track
       filter_type='kf'       # Use Kalman filter
   )

   # System model
   dt = 0.1
   F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                 [0, 0, 1, dt], [0, 0, 0, 1]])
   Q = np.eye(4) * 0.1
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
   R = np.eye(2) * 0.5

   tracker.set_dynamics(F, Q)
   tracker.set_measurement_model(H, R)

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

       # Update tracker
       confirmed_tracks = tracker.update(np.array(measurements))

       # Print confirmed tracks
       for track in confirmed_tracks:
           print(f"t={t}: Track {track.id} at ({track.state[0]:.1f}, "
                 f"{track.state[2]:.1f})")

Data Association Algorithms
---------------------------

Gating
^^^^^^

Filter unlikely measurement-track associations:

.. code-block:: python

   from pytcl.assignment_algorithms import mahalanobis_gate, ellipsoidal_gate

   # Predicted measurement and covariance
   z_pred = H @ x_pred
   S = H @ P_pred @ H.T + R

   # Check if measurement is in gate
   z = np.array([5.2, 3.1])
   is_valid = mahalanobis_gate(z, z_pred, S, threshold=9.21)

   # Or compute gated measurements for multiple candidates
   measurements = np.array([[5.2, 3.1], [10.5, 2.0], [100.0, 50.0]])
   valid_mask = ellipsoidal_gate(measurements, z_pred, S, threshold=9.21)

Global Nearest Neighbor (GNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.assignment_algorithms import (
       auction_algorithm, hungarian_algorithm
   )

   # Cost matrix: tracks x measurements
   # Lower cost = better association
   cost_matrix = np.array([
       [1.2, 5.0, 100.0],   # Track 0 costs
       [4.5, 0.8, 50.0],    # Track 1 costs
       [90.0, 80.0, 2.1],   # Track 2 costs
   ])

   # Hungarian algorithm (optimal)
   track_to_meas, meas_to_track, cost = hungarian_algorithm(cost_matrix)
   # track_to_meas[i] = measurement index for track i (-1 if unassigned)

   # Auction algorithm (faster for large problems)
   track_to_meas, meas_to_track, cost = auction_algorithm(
       cost_matrix, epsilon=0.01
   )

Joint Probabilistic Data Association (JPDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JPDA computes association probabilities and combines innovations:

.. code-block:: python

   from pytcl.assignment_algorithms import (
       jpda_association_probabilities,
       jpda_combined_innovation
   )

   # Likelihood matrix: tracks x measurements
   likelihoods = np.exp(-0.5 * cost_matrix)

   # Add clutter likelihood
   clutter_density = 1e-4

   # Compute association probabilities
   probs = jpda_association_probabilities(
       likelihoods,
       detection_prob=0.9,
       clutter_density=clutter_density
   )
   # probs[i, j] = P(measurement j from track i)

   # Combined innovation for track 0
   innovations = measurements - z_pred[0]  # Innovations to all measurements
   combined = jpda_combined_innovation(innovations, probs[0, :])

Multiple Hypothesis Tracking (MHT)
----------------------------------

MHT maintains multiple association hypotheses over time.

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.trackers import MHTTracker, MHTConfig

   config = MHTConfig(
       max_hypotheses=100,      # Maximum hypotheses to maintain
       n_scan_prune=3,          # N-scan pruning depth
       probability_threshold=0.01,  # Minimum hypothesis probability
       detection_probability=0.9,
       clutter_density=1e-4,
       gate_threshold=16.0
   )

   mht = MHTTracker(
       state_dim=4,
       meas_dim=2,
       config=config
   )

   mht.set_dynamics(F, Q)
   mht.set_measurement_model(H, R)

Running MHT
^^^^^^^^^^^

.. code-block:: python

   for t, measurements in enumerate(all_measurements):
       result = mht.update(measurements)

       # Best hypothesis tracks
       for track in result.confirmed_tracks:
           print(f"Track {track.id}: state={track.state}")

       # Hypothesis tree info
       print(f"Active hypotheses: {result.n_hypotheses}")
       print(f"Best hypothesis probability: {result.best_probability:.4f}")

Hypothesis Management
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.trackers import (
       HypothesisTree, generate_joint_associations,
       n_scan_prune, prune_hypotheses_by_probability
   )

   # Generate all possible associations
   hypotheses = generate_joint_associations(
       n_tracks=3,
       n_measurements=4,
       gating_matrix=valid_associations  # Boolean matrix
   )

   # Prune low-probability hypotheses
   pruned = prune_hypotheses_by_probability(
       hypotheses, probabilities, threshold=0.01
   )

   # N-scan pruning (keep only hypotheses with common history)
   final = n_scan_prune(hypothesis_tree, n_scan=3)

Track Metrics
-------------

Evaluate tracking performance using standard metrics.

OSPA Metric
^^^^^^^^^^^

.. code-block:: python

   from pytcl.performance_evaluation import ospa_distance

   # True target positions
   truth = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

   # Estimated track positions
   estimates = np.array([[10.5, 19.8], [30.2, 40.5]])  # Missing one target

   # OSPA distance (order 2, cutoff 100)
   ospa = ospa_distance(truth, estimates, p=2, c=100.0)
   print(f"OSPA: {ospa.distance:.2f}")
   print(f"  Localization: {ospa.localization:.2f}")
   print(f"  Cardinality: {ospa.cardinality:.2f}")

GOSPA Metric
^^^^^^^^^^^^

.. code-block:: python

   from pytcl.performance_evaluation import gospa_distance

   gospa = gospa_distance(truth, estimates, p=2, c=100.0, alpha=2.0)
   print(f"GOSPA: {gospa.distance:.2f}")

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from pytcl.trackers import MultiTargetTracker
   from pytcl.performance_evaluation import ospa_distance

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
       gate_threshold=9.21,
       min_hits=3, max_misses=5
   )
   tracker.set_dynamics(F, Q)
   tracker.set_measurement_model(H, R)

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
       if measurements:
           tracks = tracker.update(np.array(measurements))
       else:
           tracks = tracker.update(np.empty((0, 2)))

       # Compute OSPA
       if tracks:
           estimates = np.array([[tr.state[0], tr.state[2]] for tr in tracks])
       else:
           estimates = np.empty((0, 2))

       ospa = ospa_distance(
           np.array(truth_positions), estimates, p=2, c=50.0
       )
       ospa_values.append(ospa.distance)

   print(f"Mean OSPA: {np.mean(ospa_values):.2f}")
   print(f"Final tracks: {len(tracks)}")

Next Steps
----------

- See :doc:`/api/trackers` for complete tracker API
- Explore :doc:`/api/assignment_algorithms` for association methods
- Check :doc:`/api/performance_evaluation` for more metrics
