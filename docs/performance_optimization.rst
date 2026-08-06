Performance Optimization Guide
==============================

Overview
--------

This guide covers CPU-side optimization techniques for the Tracker Component Library. For GPU acceleration, see :doc:`gpu_acceleration`.

Key Techniques:

1. **Profiling** - Identify bottlenecks
2. **Vectorization** - Batch operations
3. **Algorithm Selection** - Choose O(n) over O(n^2)
4. **Caching** - Reuse computed values
5. **Numba JIT** - Compile hotspots to machine code
6. **Sparse Data Structures** - Reduce memory overhead

Before Optimizing
------------------

**Profile First!**

Never optimize without profiling. Most time is spent in a few functions:

.. code-block:: python

   import cProfile
   import pstats

   import numpy as np

   from pytcl.dynamic_estimation.kalman import ekf_update

   rng = np.random.default_rng(42)
   measurements = rng.normal(size=(1000, 2))
   x = np.zeros(4)
   P = np.eye(4)
   H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
   R = np.eye(2) * 0.1

   def profile_tracking_algorithm():
       global x, P
       for z in measurements:
           upd = ekf_update(x, P, z, lambda s: H @ s, H, R)
           x, P = upd.x, upd.P

   # Profile
   profiler = cProfile.Profile()
   profiler.enable()
   profile_tracking_algorithm()
   profiler.disable()

   # View results
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions

Alternative: Using **line_profiler** for detailed line-by-line analysis:

.. code-block:: bash

   pip install line_profiler
   kernprof -l -v your_script.py  # Profiles lines with @profile decorator

Vectorization & Batching
-------------------------

**Problem: Redundant Work Inside the Measurement Loop**

.. code-block:: python

   # SLOW: rebuild the measurement function and Jacobian every iteration
   for z in measurements:
       H_step = np.eye(2, 4)
       upd = ekf_update(x, P, z, lambda s: H_step @ s, H_step, R)
       x, P = upd.x, upd.P

   # FASTER: hoist anything constant out of the loop. There is no batched
   # CPU EKF entry point (see pytcl.gpu for batched linear filters); the cost
   # is dominated by the Jacobian evaluation and the covariance update, so
   # precompute H when the measurement model is linear.
   x = np.zeros(4)
   P = np.eye(4)
   H = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0]])
   h = lambda s: H @ s
   for z in measurements:
       upd = ekf_update(x, P, z, h, H, R)
       x, P = upd.x, upd.P

**Problem: Repeated Coordinate Conversions**

``sphere2cart(r, az, el)`` accepts scalars or arrays. Pass arrays instead of
looping; the result for n points has shape ``(3, n)``.

.. code-block:: python

   from pytcl.coordinate_systems.conversions import sphere2cart

   r = rng.uniform(1e3, 1e4, size=500)
   az = rng.uniform(-np.pi, np.pi, size=500)
   el = rng.uniform(-np.pi / 2, np.pi / 2, size=500)

   # SLOW: convert one point at a time
   cartesian_coords = []
   for i in range(len(r)):
       cartesian_coords.append(sphere2cart(r[i], az[i], el[i]))

   # FAST: vectorized conversion, shape (3, 500)
   cartesian = sphere2cart(r, az, el)

**Problem: Data Association with Many Targets**

.. code-block:: python

   targets = rng.normal(size=(50, 2))
   meas_xy = rng.normal(size=(60, 2))

   def compute_distance(a, b):
       return np.linalg.norm(a - b)

   # SLOW: compute the cost matrix element by element
   n_targets, n_measurements = len(targets), len(meas_xy)
   cost = np.zeros((n_targets, n_measurements))
   for i, target in enumerate(targets):
       for j, measurement in enumerate(meas_xy):
           cost[i, j] = compute_distance(target, measurement)

   # FAST: vectorized distance computation
   from scipy.spatial.distance import cdist
   cost = cdist(targets, meas_xy, metric='euclidean')

Algorithm Selection
-------------------

**Assignment Problems:**

===================  ========  =======  ==========================
Algorithm            Time      Optimal  Best For
===================  ========  =======  ==========================
Greedy               O(n^2)    No       Quick estimates
Hungarian (Munkres)  O(n^3)    Yes      Small problems: n < 1000
Auction              O(n^3)    ~Yes     Large, well-scaled costs
===================  ========  =======  ==========================

``hungarian`` and ``auction`` both return a 3-tuple
``(row_ind, col_ind, total_cost)``; ``greedy_assignment_nd`` returns an
``AssignmentNDResult`` named tuple with ``assignments`` and ``cost`` fields.

.. code-block:: python

   from pytcl.assignment_algorithms import auction, greedy_assignment_nd, hungarian

   cost_matrix = cdist(targets, meas_xy[:50], metric='euclidean')

   # For < 1000 targets: Hungarian gives the optimal assignment
   row_ind, col_ind, total_cost = hungarian(cost_matrix)

   # Auction algorithm: near-optimal, scales well
   row_ind, col_ind, total_cost = auction(cost_matrix)

   # Greedy: fastest, but suboptimal
   result = greedy_assignment_nd(cost_matrix)
   assignments, total_cost = result.assignments, result.cost

Caching with ``lru_cache``
---------------------------

The library already caches several expensive computations internally:

.. code-block:: python

   # Cached with lru_cache inside the library (you don't need to do anything):
   from pytcl.gravity import legendre_scaling_factors  # cached per n_max
   from pytcl.coordinate_systems.jacobians import enu_jacobian  # cached, quantized lat/lon

   # Clenshaw recursion coefficients in pytcl.gravity.clenshaw are also
   # cached internally.

   # For custom functions in your code:
   from functools import lru_cache

   precomputed_values = rng.normal(size=128)

   @lru_cache(maxsize=128)
   def expensive_lookup_table(index):
       # Computed once, reused on subsequent calls
       return precomputed_values[index]

   # Query many times - only computed once per distinct index
   for i in range(10000):
       result = expensive_lookup_table(i % 128)

**Caching for Jacobian Computations:**

The ENU and NED Jacobians in ``pytcl.coordinate_systems.jacobians`` already
apply this pattern: inputs are quantized to about 1 m resolution and results
are memoized with ``lru_cache``, giving a 25-40% speedup when repeatedly
called with similar latitudes and longitudes. Use the shipped functions
instead of rolling your own cache:

.. code-block:: python

   # Repeated calls with nearby lat/lon hit the internal cache
   lats = rng.uniform(0.6999, 0.7001, size=1000)
   lons = rng.uniform(-1.2001, -1.1999, size=1000)
   for lat, lon in zip(lats, lons):
       J = enu_jacobian(lat, lon)

Numba JIT Compilation
---------------------

The library uses Numba in selected hotspots (gating, clustering, particle
filters, signal processing). You can use it for custom code:

.. code-block:: python

   from numba import njit

   # Compile to machine code on first call
   @njit(cache=True)
   def compute_range_rate(positions, velocities, receiver_pos):
       """Compute range-rate (dot product) in a compiled loop."""
       n = len(positions)
       range_rates = np.zeros(n)

       for i in range(n):
           # Compiled to machine code - no interpreter overhead
           relative_pos = positions[i] - receiver_pos
           range_rates[i] = np.dot(relative_pos, velocities[i]) / np.linalg.norm(relative_pos)

       return range_rates

   pos = rng.normal(size=(1000, 3)) * 1e4
   vel = rng.normal(size=(1000, 3)) * 10
   receiver_pos = np.zeros(3)

   # First call: compilation (slow)
   range_rates = compute_range_rate(pos, vel, receiver_pos)

   # Subsequent calls: machine code (fast)
   pos_new = rng.normal(size=(1000, 3)) * 1e4
   vel_new = rng.normal(size=(1000, 3)) * 10
   range_rates = compute_range_rate(pos_new, vel_new, receiver_pos)

**Numba Tips:**

- ``cache=True`` allows reuse across runs
- Avoid Python objects - use numpy arrays
- Keep functions simple (no complex control flow)
- ``@njit`` is shorthand for ``@jit(nopython=True)`` - prefer it for numerical kernels
- Test with small input first (compilation can fail on edge cases)

Example: Batched Prediction for Many Targets

Do not hand-roll a batched Kalman predict - the library ships one in
``pytcl.gpu`` that runs on MLX (Apple Silicon) or CuPy (NVIDIA) and falls
back to NumPy when neither is installed. Measured end-to-end on MLX for
batch linear predict+update, the speedup over a per-track CPU loop is about
1.6x at 100 tracks, 13x at 1,000 tracks, and 40x at 20,000 tracks.

.. code-block:: python

   from pytcl.gpu import batch_kf_predict, to_cpu, to_gpu

   states = np.zeros((100, 4))                # (n_targets, state_dim)
   covariances = np.stack([np.eye(4)] * 100)  # (n_targets, state_dim, state_dim)
   F = np.eye(4)
   F[0, 2] = F[1, 3] = 1.0
   Q = np.eye(4) * 0.1

   x_b, P_b = to_gpu(states), to_gpu(covariances)
   pred = batch_kf_predict(x_b, P_b, F, Q)    # predict all targets at once
   x_pred, P_pred = to_cpu(pred.x), to_cpu(pred.P)

Sparse Data Structures
----------------------

For large assignment problems with few valid assignments (sparse cost
matrix), convert the dense matrix to a ``SparseCostTensor`` and use the
sparse greedy solver:

.. code-block:: python

   from pytcl.assignment_algorithms import (
       SparseCostTensor,
       greedy_assignment_nd_sparse,
   )

   # Traditional: full matrix (memory wasted on infinite costs)
   n = 2000
   cost_dense = np.full((n, n), np.inf)
   valid_rows = rng.integers(0, n, size=100)
   valid_cols = rng.integers(0, n, size=100)
   cost_dense[valid_rows, valid_cols] = rng.uniform(0, 10, size=100)
   # Memory: 2000 * 2000 * 8 bytes = 32 MB, ~100 finite entries

   # Sparse: only store valid entries
   sparse_cost = SparseCostTensor.from_dense(cost_dense)
   print(f"valid entries: {sparse_cost.n_valid}")
   print(f"memory savings: {sparse_cost.memory_savings:.1%}")

   assignments = greedy_assignment_nd_sparse(sparse_cost)

Output:

.. code-block:: text

   valid entries: 100
   memory savings: 100.0%

**Benefits:**

- Memory scales with the number of finite entries, not the matrix size
- ``SparseCostTensor.sparsity`` and ``memory_savings`` report the reduction
- Works for N-dimensional cost tensors, not just 2-D matrices

Real-World Example: Multi-Sensor Tracking
------------------------------------------

Optimize a realistic tracking scenario. Association uses the shipped
``gated_gnn_association`` (chi-squared gating plus global nearest neighbor)
and the state update uses ``kf_update``:

.. code-block:: python

   import time

   from pytcl.assignment_algorithms import gated_gnn_association
   from pytcl.dynamic_estimation.kalman import kf_update

   class OptimizedTracker:
       def __init__(self, n_targets):
           self.states = np.zeros((n_targets, 4))
           self.covariances = np.stack([np.eye(4)] * n_targets)
           self.F = np.eye(4)  # Constant-velocity model
           self.F[0, 2] = self.F[1, 3] = 1.0
           self.H = np.eye(2, 4)  # Observe position only
           self.R = np.eye(2) * 0.1

       def predict(self, Q):
           """Predict all targets (vectorized)"""
           self.states = self.states @ self.F.T
           self.covariances = np.einsum(
               "ij,njk,lk->nil", self.F, self.covariances, self.F
           ) + Q

       def update(self, measurements):
           """Gate, associate, and update with measurements"""
           assoc = gated_gnn_association(
               self.states,
               self.covariances,
               measurements,
               self.H,
               gate_probability=0.99,
           )
           # track_to_measurement[i] is the measurement index for track i,
           # or -1 if the track got no measurement
           for i, j in enumerate(assoc.track_to_measurement):
               if j >= 0:
                   upd = kf_update(
                       self.states[i], self.covariances[i],
                       measurements[j], self.H, self.R,
                   )
                   self.states[i] = upd.x
                   self.covariances[i] = upd.P

   # Usage with timing
   tracker = OptimizedTracker(n_targets=100)
   measurement_sequence = [
       tracker.states[:, :2] + rng.normal(scale=0.3, size=(100, 2))
       for _ in range(50)
   ]

   start = time.perf_counter()

   for meas in measurement_sequence:
       tracker.predict(Q=np.eye(4) * 0.1)
       tracker.update(meas)

   elapsed = time.perf_counter() - start
   print(f"Tracking {len(measurement_sequence)} frames: {elapsed:.2f}s")

Performance Checklist
---------------------

Before shipping - verify these optimizations:

- **Profile hotspots** - Know where time is spent
- **Vectorize loops** - Use numpy operations, not Python loops
- **Choose right algorithm** - O(n) vs O(n^2) matters
- **Cache values** - Don't recompute constants
- **Consider Numba** - For tight numerical loops
- **Use sparse structures** - For large matrices with many invalid entries
- **GPU acceleration** - If data is large enough
- **Thread pool** - For independent operations

Common Mistakes
---------------

**Mistake 1: Premature Optimization**

.. code-block:: python

   # Don't start with complex Numba code, sparse matrices, or GPU.
   #
   # Do this first:
   #   1. Write simple, readable code
   #   2. Profile it
   #   3. Optimize bottlenecks only

**Mistake 2: Micro-optimizations on Non-Critical Code**

.. code-block:: python

   # Profile shows 99% time in parsing input
   # But you optimize the filter algorithm
   #
   # Profile first to find the real bottleneck

**Mistake 3: Memory Allocation in Loop**

.. code-block:: python

   def compute_distances(out, targets, measurements):
       out[:] = cdist(targets, measurements)

   # SLOW: allocates a new array each iteration
   for _ in range(10):
       cost = np.zeros((n_targets, n_measurements))  # Allocate
       compute_distances(cost, targets, meas_xy)     # Fill

   # FAST: allocate once, reuse
   cost = np.empty((n_targets, n_measurements))
   for _ in range(10):
       compute_distances(cost, targets, meas_xy)

**Mistake 4: Ignoring NumPy Broadcasting**

.. code-block:: python

   target_positions = rng.normal(size=(100, 3))
   sensor_positions = rng.normal(size=(100, 3))

   # SLOW: Python loop
   ranges = []
   for pos_target, pos_sensor in zip(target_positions, sensor_positions):
       r = np.linalg.norm(pos_target - pos_sensor)
       ranges.append(r)

   # FAST: NumPy broadcasting
   ranges = np.linalg.norm(target_positions - sensor_positions, axis=1)

Resources
---------

- `NumPy Broadcasting Guide <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
- `Numba Performance Tips <https://numba.readthedocs.io/en/stable/user/performance-tips.html>`_
- `Python Speed Profile Guide <https://docs.python.org/3/library/profile.html>`_
- `SciPy Optimization <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_

See Also
~~~~~~~~

- :doc:`gpu_acceleration` - GPU-accelerated operations
- :doc:`kalman_filter_tuning` - Filter parameter tuning
- Module: ``pytcl.dynamic_estimation.kalman``
- Examples: ``examples/performance_evaluation.py``
