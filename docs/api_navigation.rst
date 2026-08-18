API Navigation Guide
====================

Overview
--------

The Tracker Component Library provides **1044 public names** across
**145 modules**. This guide shows how to discover and use them
effectively.

.. Figures are contract-tested against the package by
   tests/contract/test_docs_architecture.py::test_package_counts_match_reality
   -- update both this line and docs/architecture.rst together, and rerun
   that test, if either number changes.

Key Resources:

- **API Documentation**: Auto-generated from docstrings (:doc:`api/index`)
- **Architecture Guide**: Module organization and structure (:doc:`architecture`)
- **Examples**: Working code in ``examples/`` folder
- **Tutorials**: Interactive notebooks in ``docs/notebooks/``

Quick Discovery Methods
-----------------------

Method 1: Python ``help()`` and ``dir()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Explore module contents**:

.. code-block:: python

   import pytcl

   # List all modules
   print(dir(pytcl))

   # List functions in a submodule
   from pytcl import coordinate_systems
   print([x for x in dir(coordinate_systems) if not x.startswith('_')])

   # Get help on a function
   from pytcl.coordinate_systems import sphere2cart
   help(sphere2cart)  # Shows docstring, signature, examples

**Output shows**:

- Function signature with type hints
- Docstring describing what it does
- Parameters and return values
- Usage examples (often included)

Method 2: Interactive Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use IDE autocomplete**:

.. code-block:: text

   from pytcl.dynamic_estimation import kalman
   kalman.<Tab>  # shows all available functions
   # kf_predict, kf_update, ekf_predict, ekf_update, ukf_predict, ...

**Use Jupyter notebook autocomplete**:

.. code-block:: text

   import pytcl.coordinate_systems.rotations as rot
   rot.euler<Tab>
   # Suggestions: euler2rotmat, euler2quat

Method 3: Search Functions by Category
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Find functions related to a task**:

.. code-block:: python

   # All Kalman filter predict/update pairs
   from pytcl.dynamic_estimation import kalman
   print([x for x in dir(kalman) if 'kf_' in x.lower()][:10])
   # ['ckf_predict', 'ckf_spherical_cubature_points', 'ckf_update',
   #  'constrained_ekf_predict', 'constrained_ekf_update', 'ekf_predict',
   #  'ekf_predict_auto', 'ekf_update', 'ekf_update_auto',
   #  'iterated_ekf_update']

   # All coordinate conversions
   from pytcl.coordinate_systems import conversions
   print(sorted(x for x in dir(conversions) if '2' in x)[:12])
   # ['cart2cyl', 'cart2pol', 'cart2ruv', 'cart2sphere', 'cyl2cart',
   #  'ecef2enu', 'ecef2geodetic', 'ecef2ned', 'ecef2sez', 'enu2ecef',
   #  'enu2ned', 'geodetic2ecef']

Method 4: Browse the Sphinx API Docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Browse the rendered documentation**:

- Open :doc:`api/index`
- Click a module name (e.g., ``dynamic_estimation``)
- Browse all functions with full documentation

Common Discovery Workflows
---------------------------

Workflow 1: "I need to do Kalman filtering"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Visit :doc:`api/index` and search for "kalman"

**Step 2**: See available filters (all functional predict/update pairs):

- ``kf_predict``, ``kf_update`` - Linear Kalman filter
- ``ekf_predict``, ``ekf_update`` - Extended KF (using Jacobians)
- ``ukf_predict``, ``ukf_update`` - Unscented KF (sigma points)
- ``ckf_predict``, ``ckf_update`` - Cubature KF (deterministic points)
- ``imm_predict``, ``imm_update`` - Interacting Multiple Model (in ``pytcl.dynamic_estimation``)

**Step 3**: Pick the right one:

.. code-block:: python

   # Linear system -> use standard KF
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update

   # Nonlinear, have a Jacobian -> use EKF
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update

   # Nonlinear, don't want Jacobian code -> use UKF
   from pytcl.dynamic_estimation.kalman import ukf_predict, ukf_update

**Step 4**: See tuning guide at :doc:`kalman_filter_tuning`

Workflow 2: "I need coordinate conversion"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Find what conversions are available:

.. code-block:: python

   from pytcl.coordinate_systems import conversions

   # List all conversion functions
   funcs = [x for x in dir(conversions)
            if not x.startswith('_') and x.islower()]
   print(sorted(funcs)[:12])
   # ['cart2cyl', 'cart2pol', 'cart2ruv', 'cart2sphere', 'cyl2cart',
   #  'ecef2enu', 'ecef2geodetic', 'ecef2ned', 'ecef2sez', 'enu2ecef',
   #  'enu2ned', 'geocentric_radius']

**Step 2**: Match your need:

.. code-block:: python

   # Cartesian to spherical coordinates
   from pytcl.coordinate_systems.conversions import cart2sphere

   # ECEF (Earth-Centered Earth-Fixed) to geodetic
   from pytcl.coordinate_systems.conversions import ecef2geodetic

   # East-North-Up to ECEF
   from pytcl.coordinate_systems.conversions import enu2ecef

**Step 3**: Use it:

.. code-block:: python

   import numpy as np
   cart_coords = np.array([1.0, 0.0, 0.0])
   r, az, el = cart2sphere(cart_coords, system_type='az-el')
   print(r, az, el)  # 1.0 0.0 0.0 (range, azimuth, elevation)

Workflow 3: "I need data association"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Look at assignment algorithms:

.. code-block:: python

   from pytcl import assignment_algorithms

   # Available methods
   print([x for x in dir(assignment_algorithms) if 'assign' in x][:8])
   # ['assign2d', 'assign3d', 'assign3d_auction', 'assign3d_lagrangian',
   #  'assignment_from_flow_solution', 'assignment_nd',
   #  'assignment_to_flow_network', 'auction_assignment_nd']

**Step 2**: Understand when to use each:

.. code-block:: python

   import numpy as np

   cost_matrix = np.array([[4.0, 1.0, 3.0],
                           [2.0, 0.0, 5.0],
                           [3.0, 2.0, 2.0]])

   # 2D problems: Jonker-Volgenant via assign2d
   from pytcl.assignment_algorithms import assign2d
   result = assign2d(cost_matrix)
   # result.row_indices, result.col_indices, result.cost

   # Multi-frame (S-D) problems: Lagrangian relaxation on a cost tensor
   from pytcl.assignment_algorithms import relaxation_assignment_nd

   # Top-K solutions (need multiple hypotheses)
   from pytcl.assignment_algorithms import murty
   top_k = murty(cost_matrix, k=3)
   # top_k.assignments, top_k.costs, top_k.n_found

**Step 3**: See performance guide at :doc:`performance_optimization`

Workflow 4: "I need navigation functions"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Browse navigation modules:

.. code-block:: python

   from pytcl.navigation import (
       geodesy,           # Direct/inverse geodesic problems on the ellipsoid
       great_circle,      # Great-circle distance, waypoints, TDOA location
       rhumb,             # Rhumb-line navigation
       ins,               # Strapdown INS mechanization
       ins_gnss,          # INS/GNSS integration
   )

**Step 2**: Pick the right function (all are re-exported from
``pytcl.navigation``):

.. code-block:: python

   # Distance between lat/lon points (spherical Earth)
   from pytcl.navigation import great_circle_distance

   # Ellipsoidal direct/inverse problems
   from pytcl.navigation import direct_geodetic, inverse_geodetic

   # INS mechanization
   from pytcl.navigation import mechanize_ins_ned

**Step 3**: Use it (angles in radians):

.. code-block:: python

   import numpy as np
   from pytcl.navigation import great_circle_distance

   # New York to London
   nyc_lat, nyc_lon = np.radians(40.7128), np.radians(-74.0060)
   london_lat, london_lon = np.radians(51.5074), np.radians(-0.1278)

   distance = great_circle_distance(nyc_lat, nyc_lon, london_lat, london_lon)
   print(f"{distance / 1000:.1f} km")  # 5570.2 km

Function Naming Conventions
----------------------------

**Conversion Functions**: ``source2destination``

.. code-block:: text

   cart2sphere      # Cartesian -> Spherical
   sphere2cart      # Spherical -> Cartesian
   ecef2geodetic    # ECEF -> Geodetic
   ecef2enu         # ECEF -> East-North-Up
   euler2rotmat     # Euler angles -> Direction Cosine Matrix

**Prediction/Update**: ``prefix_verb``

.. code-block:: text

   kf_predict, kf_update           # Kalman filter
   ekf_predict, ekf_update         # Extended Kalman filter
   ukf_predict, ukf_update         # Unscented Kalman filter

**Property/Getter Functions**: ``noun_property`` or ``get_noun``

.. code-block:: text

   sun_position                 # Astronomy
   compute_dop                  # Navigation (dilution of precision)
   get_magnetic_cache_info      # Geophysics

**Check/Validate Functions**: ``is_noun``

.. code-block:: text

   is_rotation_matrix           # Rotations
   is_deep_space                # SGP4 (deep-space vs near-Earth TLE)

API Reference by Use Case
---------------------------

Multi-Target Tracking
~~~~~~~~~~~~~~~~~~~~~

**Essential Functions**:

.. code-block:: python

   # Tracking system
   from pytcl.trackers import MultiTargetTracker

   # Data structures
   from pytcl.trackers import Track

   # Coordinate conversions
   from pytcl.coordinate_systems.conversions import sphere2cart

   # Filter (inside tracker)
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update

   # Dynamic model
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

   # Data association (inside tracker, but useful for custom)
   from pytcl.assignment_algorithms import gnn_association, assign2d

   # Performance evaluation
   from pytcl.performance_evaluation import nees, nis, ospa

**See**: :doc:`architecture` section "Pattern 2: Multi-Target Tracking"

Navigation and Geomatics
~~~~~~~~~~~~~~~~~~~~~~~~~

**Essential Functions**:

.. code-block:: python

   # Geodetic calculations
   from pytcl.navigation import (
       geodetic_to_ecef, ecef_to_geodetic, great_circle_distance
   )

   # INS mechanization
   from pytcl.navigation import mechanize_ins_ned

   # Coordinate frames
   from pytcl.coordinate_systems.rotations import (
       euler2rotmat, rotmat2euler, quat2rotmat
   )

   # Projections (map coordinates)
   from pytcl.coordinate_systems.projections import (
       geodetic2utm, utm2geodetic
   )

   # Geophysical models
   from pytcl.gravity import models as gravity_models
   from pytcl.magnetism import wmm, igrf
   from pytcl.atmosphere import models as atmosphere_models

**See**: :doc:`architecture` section "Pattern 3: INS/GNSS Navigation"

Satellite Operations
~~~~~~~~~~~~~~~~~~~~~

**Essential Functions**:

.. code-block:: python

   # Orbit propagation
   from pytcl.astronomical.orbital_mechanics import kepler_propagate

   # Reference frame transforms
   from pytcl.astronomical.reference_frames import (
       ecef_to_eci, eci_to_ecef, precession_matrix_iau76, nutation_matrix
   )

   # Ephemeris (planets, sun, moon)
   from pytcl.astronomical.ephemerides import (
       sun_position, moon_position
   )

   # SGP4 (TLE propagation)
   from pytcl.astronomical.sgp4 import sgp4_propagate

   # Relativistic corrections
   from pytcl.astronomical.relativity import proper_time_rate, shapiro_delay

**See**: :doc:`architecture` section "Pattern 4: Satellite Propagation"

Advanced Signal Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Essential Functions**:

.. code-block:: python

   # CFAR detection
   from pytcl.mathematical_functions.signal_processing.detection import (
       cfar_ca, cfar_2d
   )

   # Filtering
   from pytcl.mathematical_functions.signal_processing.filters import (
       butter_design, apply_filter
   )

   # Optimal detection
   from pytcl.mathematical_functions.signal_processing.matched_filter import (
       matched_filter, pulse_compression
   )

   # Special functions (for detection threshold calculation)
   from pytcl.mathematical_functions.special_functions import marcum_q

Searching for Functions: Advanced Tips
---------------------------------------

**Search by keyword using grep**:

.. code-block:: bash

   # Find all "distance" functions in the codebase
   grep -r "def .*distance" pytcl/

   # Find functions with "kalman" in the name
   grep -ri "def .*kalman" pytcl/

**Search in Jupyter**:

.. code-block:: python

   # Find all functions containing a keyword, e.g. "slerp"
   import pytcl
   import inspect

   def find_functions(keyword):
       for name in dir(pytcl):
           module = getattr(pytcl, name)
           if inspect.ismodule(module):
               for func_name in dir(module):
                   if keyword.lower() in func_name.lower():
                       func = getattr(module, func_name)
                       if callable(func):
                           print(f"{name}.{func_name}")

   find_functions("slerp")
   # coordinate_systems.slerp

**View function source code**:

.. code-block:: python

   from pytcl.coordinate_systems import sphere2cart
   import inspect

   # View source
   print(inspect.getsource(sphere2cart))

   # Find where function is defined
   print(inspect.getfile(sphere2cart))

Type Hints and Signatures
-------------------------

**All functions have complete type hints**:

.. code-block:: python

   from pytcl.coordinate_systems.conversions import cart2sphere
   import inspect

   sig = inspect.signature(cart2sphere)
   print(sig)
   # (abridged output)
   # (cart_points: ArrayLike,
   #  system_type: Literal['standard', 'az-el', 'range-az-el'] = 'standard')
   #  -> Tuple[ndarray, ndarray, ndarray]

   # Understand the parameters
   for param_name, param in sig.parameters.items():
       print(f"{param_name}: {param.annotation}")

**Benefits**:

- IDE autocomplete shows expected types
- Type checking (ty, mypy, pyright) catches errors
- Self-documenting code

Common Errors and Solutions
----------------------------

**ImportError: No module named 'pytcl.xxx'**

Solution: Check the correct import path

.. code-block:: python

   # Wrong: kalman is not a top-level module
   # from pytcl.kalman import kf_predict

   # Correct
   from pytcl.dynamic_estimation.kalman import kf_predict

**TypeError: missing required positional arguments**

Solution: Check the function signature -- many conversions take separate
scalar/array arguments rather than one packed vector

.. code-block:: python

   from pytcl.coordinate_systems.conversions import sphere2cart
   help(sphere2cart)  # Shows the (r, az, el) parameters

   # Wrong: one packed array
   # sphere2cart(np.array([1000.0, 0.5, 0.1]))
   # TypeError: sphere2cart() missing 2 required positional
   # arguments: 'az' and 'el'

   # Correct: separate arguments
   cart = sphere2cart(1000.0, 0.5, 0.1, system_type='az-el')

**"Function not found" but you know it exists**

Solution: Check alternative names

.. code-block:: python

   # Maybe it's in a different module
   import pytcl

   # Search all modules
   for module_name in dir(pytcl):
       module = getattr(pytcl, module_name)
       if hasattr(module, 'your_function_name'):
           print(f"Found in pytcl.{module_name}")

Getting Help
------------

**In Python REPL**:

.. code-block:: text

   from pytcl.dynamic_estimation.kalman import kf_predict

   # View docstring
   help(kf_predict)

   # View signature
   import inspect
   print(inspect.signature(kf_predict))

**In IPython/Jupyter**:

.. code-block:: text

   from pytcl.dynamic_estimation.kalman import kf_predict

   # View docstring in sidebar
   kf_predict?

   # View source code
   kf_predict??

**Online Resources**:

- :doc:`api/index` - Full API reference
- :doc:`architecture` - Module organization
- :doc:`kalman_filter_tuning` - Filter usage guide
- :doc:`gpu_acceleration` - GPU usage
- ``examples/`` - Working code samples
- ``docs/notebooks/`` - Interactive tutorials

API Patterns and Conventions
-----------------------------

**NumPy Arrays**:

Most functions accept/return N-dimensional NumPy arrays:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import cart2sphere

   # Single point, shape (3,)
   r, az, el = cart2sphere(np.array([1.0, 2.0, 3.0]))

   # Multiple points at once: shape (3, n) or (n, 3)
   r, az, el = cart2sphere(np.array([[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]]))

**Return Values**:

Most functions return named tuples for clarity:

.. code-block:: python

   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update

   x, P = np.zeros(4), np.eye(4)          # state and covariance
   F, Q = np.eye(4), np.eye(4) * 0.01     # transition and process noise

   pred = kf_predict(x, P, F, Q)
   # pred.x = predicted state, pred.P = predicted covariance

   z, H, R = np.array([1.0, 2.0]), np.eye(2, 4), np.eye(2)
   upd = kf_update(pred.x, pred.P, z, H, R)
   # upd.x, upd.P, plus upd.y (innovation), upd.S (innovation
   # covariance), upd.K (gain), upd.likelihood

**Optional Parameters**:

Many functions have optional parameters with sensible defaults:

.. code-block:: python

   from pytcl.assignment_algorithms import assign2d

   # Simple usage with defaults
   result = assign2d(cost_matrix)

   # Advanced: allow non-assignment at a fixed cost, or maximize profit
   result = assign2d(cost_matrix, cost_of_non_assignment=10.0)
   result = assign2d(cost_matrix, maximize=True)

Best Practices
--------------

1. **Start with the highest-level API**

   Use: ``from pytcl.trackers import MultiTargetTracker``

   Avoid: Implementing by combining 10 lower-level functions

2. **Check examples for your use case**

   ``examples/`` folder has code for:

   - Multi-target tracking
   - INS/GNSS navigation
   - Satellite operations
   - Signal processing

3. **Use the type hints**

   Enable a type checker (ty, mypy, or pyright) in your IDE, and trust IDE autocomplete.

4. **Understand the math**

   Reference :doc:`kalman_filter_tuning` for filter parameters and
   :doc:`performance_optimization` for profiling. Don't guess at
   parameters.

5. **Profile before optimizing**

   Use cProfile to find bottlenecks, and see
   :doc:`performance_optimization` for GPU acceleration. Don't
   prematurely optimize.

See Also
~~~~~~~~

- :doc:`architecture` - Module organization and patterns
- :doc:`getting_started` - Quick start guide
- :doc:`kalman_filter_tuning` - Filter parameter guide
- :doc:`gpu_acceleration` - GPU computation
- :doc:`performance_optimization` - Profiling and optimization
- Examples: ``examples/`` for working code
- Tutorials: ``docs/notebooks/`` for interactive learning
