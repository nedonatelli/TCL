Performance Optimization Guide
==============================

Overview
--------

This guide covers CPU-side optimization techniques for the Tracker Component Library. For GPU acceleration, see :doc:`gpu_acceleration`.

Key Techniques:

1. **Profiling** - Identify bottlenecks
2. **Vectorization** - Batch operations
3. **Algorithm Selection** - Choose O(n) over O(n²)
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
   from io import StringIO
   
   def profile_tracking_algorithm():
       # Your tracking code here
       for z in measurements:
           upd = ekf_update(x, P, z, h, H, R)
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

**Problem: Loop Over Measurements**

.. code-block:: python

   # ❌ SLOW: Python loop - slow interpreter overhead
   for z in measurements:  # measurements shape: (10000, 2)
       upd = ekf_update(x, P, z, h, H, R)
       x, P = upd.x, upd.P

   # ✅ FASTER: keep the per-step work minimal -- hoist anything constant out
   # of the loop. There is no batched EKF entry point; the cost is dominated by
   # the Jacobian evaluation and the covariance update, so precompute H when the
   # measurement model is linear.
   H = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0]])
   for z in measurements:
       upd = ekf_update(x, P, z, lambda s: H @ s, H, R)
       x, P = upd.x, upd.P

**Problem: Repeated Coordinate Conversions**

.. code-block:: python

   # ❌ SLOW: Convert one at a time
   cartesian_coords = []
   for spherical in spherical_array:
       cartesian = sphere2cart(spherical)
       cartesian_coords.append(cartesian)
   
   # ✅ FAST: Vectorized conversion
   from pytcl.coordinate_systems.conversions import sphere2cart
   cartesian_coords = sphere2cart(spherical_array)

**Problem: Data Association with Many Targets**

.. code-block:: python

   # ❌ SLOW: Compute full cost matrix
   cost = np.zeros((n_targets, n_measurements))
   for i, target in enumerate(targets):
       for j, measurement in enumerate(measurements):
           cost[i, j] = compute_distance(target, measurement)
   
   # ✅ FAST: Vectorized distance computation
   from scipy.spatial.distance import cdist
   cost = cdist(targets, measurements, metric='euclidean')

Algorithm Selection
-------------------

**Assignment Problems:**

===================  ======  =======  ===============
Algorithm            Time    Optimal  Best For
===================  ======  =======  ===============
Greedy               O(n²)   No       Quick estimates
Hungarian (Munkres)  O(n³)   Yes      Small: n < 1000
Auction              O(n³)   Yes      Modern libraries
Auction + heuristic  O(n²)   ~Yes     Large: n > 1000
===================  ======  =======  ===============

.. code-block:: python

   from pytcl.assignment_algorithms import relaxation_assignment_nd
   
   # For < 100 targets: Hungarian is fine
   # assignments = relaxation_assignment_nd(cost_matrix, method='hungarian')
   
   # For > 1000 targets: Use greedy + auctions
   # assignments = relaxation_assignment_nd(cost_matrix, method='auction')
   
   # For sparse matrices (mostly infinite costs):
   from pytcl.assignment_algorithms import greedy_assignment_nd
   assignments = greedy_assignment_nd(cost_matrix)

Caching with ``lru_cache``
---------------------------

The library automatically caches expensive functions:

.. code-block:: python

   # These are already cached (you don't need to do anything):
   from pytcl.sphericalHarmonics import legendre_scaling_factors  # Cached
   from pytcl.special_functions import clenshaw_recursion  # Cached
   
   # For custom functions in your code:
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def expensive_lookup_table(index):
       # Computed once, reused on subsequent calls
       return precomputed_values[index]
   
   # Query many times - only computed once
   for i in range(10000):
       result = expensive_lookup_table(i % 128)

**Caching for Jacobian Computations:**

.. code-block:: python

   @lru_cache(maxsize=256)
   def cached_jacobian(angle_rounded):
       """Cache Jacobian with angle quantization"""
       # Helper function - quantize to 0.01 degree resolution
       return compute_jacobian(angle_rounded)
   
   for angle in angles:
       # Quantize to 0.01 degree for caching
       angle_rounded = round(angle / 0.01) * 0.01
       J = cached_jacobian(angle_rounded)

Numba JIT Compilation
---------------------

The library uses Numba for matrix operations. You can use it for custom code:

.. code-block:: python

   from numba import njit
   import numpy as np
   
   # Compile to machine code on first call
   @njit(cache=True)
   def compute_range_rate(positions, velocities):
       """Compute range-rate (dot product) - 5-10x speedup"""
       n = len(positions)
       range_rates = np.zeros(n)
       
       for i in range(n):
           # Convert to machine code - no interpreter overhead
           relative_pos = positions[i] - receiver_pos
           range_rates[i] = np.dot(relative_pos, velocities[i]) / np.linalg.norm(relative_pos)
       
       return range_rates
   
   # First call: compilation (slow)
   range_rates = compute_range_rate(pos, vel)
   
   # Subsequent calls: machine code (fast)
   range_rates = compute_range_rate(pos_new, vel_new)

**Numba Tips:**

- Cache=True allows reuse across runs
- Avoid Python objects - use numpy arrays
- Keep functions simple (no complex control flow)
- Use ``@njit`` for pure functions, ``@jit`` for methods
- Test with small input first (compilation can fail on edge cases)

Example: Multi-target Tracking Loop

.. code-block:: python

   @njit(cache=True)
   def kalman_predict_batch(states, covariances, F, Q):
       """Predict for all targets at once"""
       n_targets = len(states)
       x_pred = np.zeros_like(states)
       P_pred = np.zeros_like(covariances)
       
       for i in range(n_targets):
           x_pred[i] = F @ states[i]
           P_pred[i] = F @ covariances[i] @ F.T + Q
       
       return x_pred, P_pred

Sparse Data Structures
----------------------

For large assignment problems with few valid assignments (sparse cost matrix):

.. code-block:: python

   from pytcl.assignment_algorithms import greedy_assignment_nd
   import numpy as np
   
   # Traditional: Full matrix (memory waste on infinite costs)
   cost_dense = np.full((10000, 10000), np.inf)
   cost_dense[valid_pairs] = actual_costs  # Only 100 valid pairs
   
   # Memory: 10000 × 10000 × 8 bytes = 800 MB
   
   # Sparse: Only store valid entries
   assignments = greedy_assignment_nd(cost_dense)  # Efficient indexing
   # Memory: ~100 entries × 8 bytes = 1 KB

**Benefits:**

- Memory savings: 50-90% reduction
- Speed: O(n_valid log n_valid) vs O(n² log n²)
- Automatic selection in ``relaxation_assignment_nd()``

Real-World Example: Multi-Sensor Tracking
------------------------------------------

Optimize a realistic tracking scenario:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update
   from pytcl.assignment_algorithms import relaxation_assignment_nd
   from scipy.spatial.distance import cdist
   import time
   
   class OptimizedTracker:
       def __init__(self, n_targets):
           self.states = np.zeros((n_targets, 4))
           self.covariances = np.eye(4)[np.newaxis, :, :].repeat(n_targets, axis=0)
           self.F = np.eye(4)  # Simple constant-velocity model
           self.H = np.eye(2, 4)  # Observe position only
           
           # Cache computations
           self.prev_cost_matrix = None
       
       def predict(self, Q):
           """Predict all targets"""
           # Vectorized prediction
           self.states = self.states @ self.F.T
           self.covariances = np.array([
               self.F @ P @ self.F.T + Q for P in self.covariances
           ])
       
       def compute_association_cost(self, measurements):
           """Fast data association using vectorized operations"""
           # Vectorized distance: O(n × m) not O(n × m) loop iterations
           positions = self.states[:, :2]
           distances = cdist(positions, measurements, metric='mahalanobis')
           
           # Gate: only nearby measurements matter
           cost_matrix = np.where(distances < 5, distances, np.inf)
           return cost_matrix
       
       def update(self, measurements):
           """Update with measurements"""
           cost = self.compute_association_cost(measurements)
           assignments = relaxation_assignment_nd(cost)
           
           for target_idx, meas_idx in assignments:
               if meas_idx >= 0:  # Valid assignment
                   z = measurements[meas_idx]
                   # Update state using measurement
                   innovation = z - self.H @ self.states[target_idx]
                   self.states[target_idx] += 0.5 * innovation
   
   # Usage with timing
   tracker = OptimizedTracker(n_targets=100)
   
   start = time.perf_counter()
   
   for measurements in measurement_sequence:
       tracker.predict(Q=np.eye(4) * 0.1)
       tracker.update(measurements)
   
   elapsed = time.perf_counter() - start
   print(f"Tracking {len(measurement_sequence)} frames: {elapsed:.2f}s")

Performance Checklist
---------------------

Before shipping - verify these optimizations:

✓ **Profile hotspots** - Know where time is spent
✓ **Vectorize loops** - Use numpy operations, not Python loops
✓ **Choose right algorithm** - O(n) vs O(n²) matters
✓ **Cache values** - Don't recompute constants
✓ **Consider Numba** - For tight numerical loops
✓ **Use sparse structures** - For large matrices with many zeros
✓ **GPU acceleration** - If data is large enough
✓ **Thread pool** - For independent operations

Common Mistakes
---------------

**Mistake 1: Premature Optimization**

.. code-block:: python

   # ❌ Don't start with this
   # Complex Numba code, sparse matrices, GPU
   
   # ✅ Do this first
   # Simple, readable code
   # Profile it
   # Optimize bottlenecks only

**Mistake 2: Micro-optimizations on Non-Critical Code**

.. code-block:: python

   # Profile shows 99% time in parsing input
   # But you optimize the filter algorithm
   
   # ✅ Profile first to find real bottleneck

**Mistake 3: Memory Allocation in Loop**

.. code-block:: python

   # ❌ SLOW: Allocates new arrays each iteration
   for z in measurements:
       cost = np.zeros((n_targets, n_measurements))  # Allocate
       cost[:] = compute_distances(...)  # Fill
   
   # ✅ FAST: Allocate once, reuse
   cost = np.zeros((n_targets, n_measurements))
   for z in measurements:
       np.fill_diagonal(cost, 0)  # Reset instead of reallocate
       compute_distances(cost, ...)

**Mistake 4: Ignoring NumPy Broadcasting**

.. code-block:: python

   # ❌ SLOW: Python loop
   ranges = []
   for pos_target, pos_sensor in zip(target_positions, sensor_positions):
       r = np.linalg.norm(pos_target - pos_sensor)
       ranges.append(r)
   
   # ✅ FAST: NumPy broadcasting
   ranges = np.linalg.norm(target_positions - sensor_positions, axis=1)

Resources
---------

- `NumPy Performance Guide <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
- `Numba Performance Tips <http://numba.pydata.org/numba-doc/latest/user/performance-tips.html>`_
- `Python Speed Profile Guide <https://docs.python.org/3/library/profile.html>`_
- `SciPy Optimization <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_

See Also
~~~~~~~~

- :doc:`gpu_acceleration` - GPU-accelerated operations
- :doc:`kalman_filter_tuning` - Filter parameter tuning
- Module: ``pytcl.dynamic_estimation.kalman``
- Examples: ``examples/performance_evaluation.py``
