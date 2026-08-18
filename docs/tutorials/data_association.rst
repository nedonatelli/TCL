Data Association Tutorial
==========================

This tutorial demonstrates data association algorithms for matching
measurements to tracked targets in a cluttered, multi-target scenario.

Topics covered:

- Global Nearest Neighbor (GNN) -- greedy assignment
- Optimal assignment via the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
- Cost-matrix formulation with gating
- Simple track initiation/deletion logic driven by association outcome

Scenario Setup
--------------

The tutorial simulates three targets moving on linear trajectories, with
clutter measurements mixed in at each time step. Each step gives every
target a 95% detection probability and adds Poisson(2) clutter measurements
drawn uniformly over the surveillance region:

.. code-block:: python

   import numpy as np

   np.random.seed(42)
   n_steps = 50
   n_targets = 3

   # Target trajectories: x, y, vx, vy
   targets = np.array(
       [
           [10.0, 10.0, 1.0, 0.5],
           [15.0, 5.0, -0.5, 1.5],
           [5.0, 15.0, 0.8, -0.8],
       ]
   )

   meas_list = []
   for k in range(n_steps):
       targets[:, 0] += targets[:, 2] * 0.1
       targets[:, 1] += targets[:, 3] * 0.1

       step_meas = []
       for i in range(n_targets):
           if np.random.rand() < 0.95:
               step_meas.append(targets[i, :2] + np.random.randn(2) * 0.5)
       for _ in range(np.random.poisson(2)):
           step_meas.append(np.random.uniform(0, 20, 2))

       meas_list.append(np.array(step_meas) if step_meas else np.zeros((0, 2)))

Global Nearest Neighbor
------------------------

GNN greedily assigns the lowest-cost (track, measurement) pair under a 5 m
gate, repeating until no pair is left, then ages out tracks that go too long
without a hit and never reached a confirmation threshold:

.. code-block:: python

   # Track fields: x, y, vx, vy, age, confidence
   tracks = [[z[0], z[1], 0, 0, 0, 3] for z in meas_list[0][:n_targets]]

   for k in range(n_steps):
       measurements = meas_list[k]
       n_tracks = len(tracks)
       n_meas = len(measurements)

       if n_tracks > 0 and n_meas > 0:
           cost_matrix = np.zeros((n_tracks, n_meas))
           for i in range(n_tracks):
               track_pos = np.array([tracks[i][0], tracks[i][1]])
               for j in range(n_meas):
                   cost_matrix[i, j] = np.linalg.norm(track_pos - measurements[j])

           assignments = {}
           used_meas = set()
           for _ in range(min(n_tracks, n_meas)):
               best_track, best_meas, min_cost = -1, -1, np.inf
               for i in range(n_tracks):
                   for j in range(n_meas):
                       if j not in used_meas and cost_matrix[i, j] < min_cost:
                           min_cost, best_track, best_meas = cost_matrix[i, j], i, j
               if best_track >= 0 and min_cost < 5.0:  # gate
                   assignments[best_track] = best_meas
                   used_meas.add(best_meas)

           for i, track in enumerate(tracks):
               if i in assignments:
                   z = measurements[assignments[i]]
                   track[0] = 0.7 * track[0] + 0.3 * z[0]
                   track[1] = 0.7 * track[1] + 0.3 * z[1]
                   track[4] += 1
                   track[5] = min(track[5] + 1, 5)
               else:
                   track[4] += 1
                   track[5] = max(track[5] - 1, 0)

           for j in range(n_meas):
               if j not in used_meas:
                   tracks.append([measurements[j][0], measurements[j][1], 0, 0, 0, 0])

       tracks = [t for t in tracks if t[4] < 10 or t[5] >= 3]

   n_gnn_tracks = len(tracks)

GNN is cheap but can commit to a locally optimal pair that blocks a better
overall assignment when tracks are close together.

Optimal Assignment (Hungarian Algorithm)
-----------------------------------------

The Hungarian algorithm instead finds the assignment that minimizes total
cost across all track-measurement pairs simultaneously, over the same
measurement stream:

.. code-block:: python

   from scipy.optimize import linear_sum_assignment

   tracks_h = [[z[0], z[1], 0, 0, 0, 3] for z in meas_list[0][:n_targets]]

   for k in range(n_steps):
       measurements = meas_list[k]
       n_tracks = len(tracks_h)
       n_meas = len(measurements)

       if n_tracks > 0 and n_meas > 0:
           cost_matrix = np.zeros((n_tracks, n_meas))
           for i in range(n_tracks):
               track_pos = np.array([tracks_h[i][0], tracks_h[i][1]])
               for j in range(n_meas):
                   d = np.linalg.norm(track_pos - measurements[j])
                   # Ungated pairs get a large penalty cost instead of being
                   # excluded outright.
                   cost_matrix[i, j] = d if d < 5.0 else 1000.0

           track_idx, meas_idx = linear_sum_assignment(cost_matrix)
           assignments = {
               t: m for t, m in zip(track_idx, meas_idx) if cost_matrix[t, m] < 1000.0
           }

           for i, track in enumerate(tracks_h):
               if i in assignments:
                   z = measurements[assignments[i]]
                   track[0] = 0.7 * track[0] + 0.3 * z[0]
                   track[1] = 0.7 * track[1] + 0.3 * z[1]
                   track[4] += 1
                   track[5] = min(track[5] + 1, 5)
               else:
                   track[4] += 1
                   track[5] = max(track[5] - 1, 0)

           used_meas = set(assignments.values())
           for j in range(n_meas):
               if j not in used_meas:
                   tracks_h.append([measurements[j][0], measurements[j][1], 0, 0, 0, 0])

       tracks_h = [t for t in tracks_h if t[4] < 10 or t[5] >= 3]

   n_hungarian_tracks = len(tracks_h)

Track Maintenance
------------------

Both approaches share the same simple track lifecycle, already applied
inside each loop above: an associated measurement pulls the track's
position; a missed association ages the track down; and a track that goes
too long without a hit and never reached a confirmation threshold is
dropped -- the same ``t[4] < 10 or t[5] >= 3`` rule filters both
``tracks`` (GNN) and ``tracks_h`` (Hungarian):

.. code-block:: python

   print(f"GNN confirmed tracks: {n_gnn_tracks}")
   print(f"Hungarian confirmed tracks: {n_hungarian_tracks}")

Next Steps
----------

- See :doc:`multi_target_tracking` for a full tracker built around
  association output (track confirmation, deletion, OSPA scoring)
- See :doc:`/api/assignment_algorithms` for the library's GNN, JPDA, and
  Hungarian-based association routines (this tutorial reimplements a
  minimal version of both for illustration)
- See :doc:`/api/trackers` for the tracker classes that wrap association
  with state estimation
