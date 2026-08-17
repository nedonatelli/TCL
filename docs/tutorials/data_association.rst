Data Association Tutorial
==========================

This tutorial demonstrates data association algorithms for matching
measurements to tracked targets in a cluttered, multi-target scenario.

Topics covered:

- Global Nearest Neighbor (GNN) — greedy assignment
- Optimal assignment via the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
- Cost-matrix formulation with gating
- Simple track initiation/deletion logic driven by association outcome

Scenario Setup
--------------

The tutorial simulates three targets moving on linear trajectories, with
clutter measurements mixed in at each time step:

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

   # Each step: 95% detection probability per target, plus Poisson(2) clutter
   # drawn uniformly over the surveillance region.

Global Nearest Neighbor
------------------------

GNN greedily assigns the lowest-cost (track, measurement) pair, repeating
until no pair is left under the gate:

.. code-block:: python

   cost_matrix = np.zeros((n_tracks, n_meas))
   for i in range(n_tracks):
       track_pos = np.array([tracks[i][0], tracks[i][1]])
       for j in range(n_meas):
           cost_matrix[i, j] = np.linalg.norm(track_pos - measurements[j])

   assignments = {}
   used_meas = set()
   for _ in range(min(n_tracks, n_meas)):
       # pick the globally smallest remaining cost under a 5 m gate
       ...

GNN is cheap but can commit to a locally optimal pair that blocks a better
overall assignment when tracks are close together.

Optimal Assignment (Hungarian Algorithm)
-----------------------------------------

The Hungarian algorithm instead finds the assignment that minimizes total
cost across all track-measurement pairs simultaneously:

.. code-block:: python

   from scipy.optimize import linear_sum_assignment

   # Ungated pairs get a large penalty cost instead of being excluded outright
   cost_matrix = np.where(cost_matrix < 5.0, cost_matrix, 1000.0)
   track_indices, meas_indices = linear_sum_assignment(cost_matrix)

   assignments = {
       t: m
       for t, m in zip(track_indices, meas_indices)
       if cost_matrix[t, m] < 1000.0
   }

Track Maintenance
------------------

Both approaches share the same simple track lifecycle: an associated
measurement pulls the track's position and re-derives a one-step velocity
estimate; a missed association ages the track down. Tracks that go too long
without a hit and never reached a confirmation threshold are dropped:

.. code-block:: python

   tracks = [t for t in tracks if t[4] < 10 or t[5] >= 3]  # age < 10 or confirmed

Next Steps
----------

- See :doc:`multi_target_tracking` for a full tracker built around
  association output (track confirmation, deletion, OSPA scoring)
- See :doc:`/api/assignment_algorithms` for the library's GNN, JPDA, and
  Hungarian-based association routines (this tutorial reimplements a
  minimal version of both for illustration)
- See :doc:`/api/trackers` for the tracker classes that wrap association
  with state estimation
