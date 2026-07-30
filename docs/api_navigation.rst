API Navigation Guide
====================

Overview
--------

The Tracker Component Library provides **1,400+ functions** across **180+ modules**. This guide shows how to discover and use them effectively.

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

.. code-block:: python

   from pytcl.dynamic_estimation import kalman
   kalman.  # Press Tab - shows all available functions
   # kf_predict, kf_update, extended_kalman_filter, etc.

**Use Jupyter notebook autocomplete**:

.. code-block:: python

   import pytcl.coordinate_systems.rotations as rot
   rot.euler  # Press Tab
   # Suggestions: euler2rotmat, euler2quat, euler_rate2body_rate, etc.

Method 3: Search Functions by Category
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Find functions related to a task**:

.. code-block:: python

   # All Kalman filter variants
   from pytcl.dynamic_estimation import kalman
   print([x for x in dir(kalman) if 'kalman' in x.lower()])
   # extended_kalman_filter, unscented_kalman_filter, cubature_kalman_filter
   
   # All coordinate conversions
   from pytcl.coordinate_systems import conversions
   print([x for x in dir(conversions) if '2' in x])
   # cart2sphere, sphere2cart, cart2pol, pol2cart, etc.

Method 4: Search Documentation Online
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Browse Sphinx API docs**:

Visit the documentation at https://readthedocs.org/ and search:

- Click "API" in the navigation
- Click a module name (e.g., ``kalman``)
- Browse all functions with full documentation

Common Discovery Workflows
---------------------------

Workflow 1: "I need to do Kalman filtering"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Visit :doc:`api/index` and search for "kalman"

**Step 2**: See available filters:
  - ``kf_predict``, ``kf_update`` - Linear Kalman filter
  - ``extended_kalman_filter`` - Nonlinear (using Jacobians)
  - ``unscented_kalman_filter`` - Nonlinear (using sigma points)
  - ``cubature_kalman_filter`` - Deterministic points
  - ``imm_filter`` - Multiple model

**Step 3**: Pick the right one:

.. code-block:: python

   # Linear system → use standard KF
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
   
   # Nonlinear, need Jacobian → use EKF
   from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update
   
   # Nonlinear, don't want Jacobian code → use UKF
   from pytcl.dynamic_estimation.kalman import ukf_predict, ukf_update

**Step 4**: See tuning guide at :doc:`kalman_filter_tuning`

Workflow 2: "I need coordinate conversion"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Find what conversions are available:

.. code-block:: python

   from pytcl.coordinate_systems import conversions
   
   # List all conversion functions
   funcs = [x for x in dir(conversions) if not x.startswith('_')]
   print(funcs)
   # Output: ['cart2cyl', 'cart2pol', 'cart2ruv', 'cart2sphere', 
   #          'cyl2cart', 'pol2cart', 'ruv2cart', 'sphere2cart',
   #          'ecef2enu', 'enu2ecef', 'ecef2geodetic', ...]

**Step 2**: Match your need:

.. code-block:: python

   # Cartesian to spherical coordinates
   from pytcl.coordinate_systems.conversions import cart2sphere
   
   # ECEF (Earth-Centered Earth-Fixed) to geodetic
   from pytcl.coordinate_systems.conversions import ecef2geodetic
   
   # East-North-Up to Cartesian
   from pytcl.coordinate_systems.conversions import enu2ecef

**Step 3**: Use it:

.. code-block:: python

   import numpy as np
   cart_coords = np.array([1.0, 0.0, 0.0])
   spherical = cart2sphere(cart_coords)
   print(spherical)  # [1., 0., 0.] (range, azimuth, elevation)

Workflow 3: "I need data association"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Look at assignment algorithms:

.. code-block:: python

   from pytcl.assignment_algorithms import assign2d, assign3d, murty
   
   # Available methods
   print(dir(optimization))
   # relaxation_assignment_nd, greedy_assignment_nd, murty, etc.

**Step 2**: Understand when to use each:

.. code-block:: python

   # Small problems (< 500 assignments): any method works
   from pytcl.assignment_algorithms import relaxation_assignment_nd
   assignments = relaxation_assignment_nd(cost_matrix)
   
   # Large problems (> 1000): consider auction or sparse
   from pytcl.assignment_algorithms import relaxation_assignment_nd
   assignments = relaxation_assignment_nd(cost_matrix, method='auction')
   
   # Top-K solutions (need multiple options)
   from pytcl.assignment_algorithms import murty
   top_k_solutions = murty(cost_matrix, k=5)

**Step 3**: See performance guide at :doc:`performance_optimization`

Workflow 4: "I need navigation functions"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Browse navigation modules:

.. code-block:: python

   from pytcl.navigation import (
       geodesy,           # Great circle, rhumb line
       ins_gnss,          # INS/GNSS fusion
       ephemerides,       # Celestial bodies
       tdoa                # Time difference localization
   )

**Step 2**: Pick the right module:

.. code-block:: python

   # Distance/bearing between lat/lon points
   from pytcl.navigation import great_circle_distance
   
   # INS propagation with gravity
   from pytcl.navigation import mechanize_ins_ned
   
   # Sun, Moon, planet positions
   from pytcl.astronomical.ephemerides import sun_position

**Step 3**: Use it with coordinate conversions:

.. code-block:: python

   from pytcl.navigation import great_circle_distance
   
   # New York to London
   nyc = np.array([40.7128, -74.0060])  # lat, lon
   london = np.array([51.5074, -0.1278])
   distance = great_circle_distance(nyc, london)
   print(f"{distance / 1000:.1f} km")  # ~5570 km

Function Naming Conventions
----------------------------

**Conversion Functions**: ``source2destination``

.. code-block:: python

   cart2sphere      # Cartesian → Spherical
   sphere2cart      # Spherical → Cartesian
   ecef2geodetic    # ECEF → Geodetic
   ecef2enu         # ECEF → East-North-Up
   euler2rotmat        # Euler angles → Direction Cosine Matrix

**Prediction/Update**: ``verb_noun``

.. code-block:: python

   kf_predict, kf_update           # Kalman filter
   ekf_predict, ekf_update         # Extended Kalman filter
   particle_filter_predict         # Particle filter

**Property/Getter Functions**: ``get_noun`` or ``compute_noun``

.. code-block:: python

   sun_position()              # Astronomy
   compute_distance()              # Math
   get_magnetic_cache_info()       # Geophysics

**Check/Validate Functions**: ``is_noun`` or ``check_noun``

.. code-block:: python

   is_positive_definite()          # Utility
   is_proper_rotation()            # Rotations

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
   from pytcl.assignment_algorithms import relaxation_assignment_nd
   
   # Performance evaluation
   from pytcl.performance_evaluation.metrics import (
       nees, nis, cramer_rao_bound
   )

**See**: :doc:`architecture` section "Pattern 2: Multi-Target Tracking"

Navigation and Geomatics
~~~~~~~~~~~~~~~~~~~~~~~~~

**Essential Functions**:

.. code-block:: python

   # Geodetic calculations
   from pytcl.navigation.geodesy import (
       geodetic2ecef, ecef2geodetic, great_circle_distance
   )
   
   # INS mechanization
   from pytcl.navigation import mechanize_ins_ned
   
   # Coordinate frames
   from pytcl.coordinate_systems.rotations import (
       euler2rotmat, rotmat2euler, quat2dcm
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
       ecef_to_eci, eci_to_ecef, apply_precession_nutation
   )
   
   # Ephemeris (planets, sun, moon)
   from pytcl.astronomical.ephemerides import (
       sun_position, get_moon_position
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
   from pytcl.mathematical_functions.signal_processing.detection import cfar_ca, cfar_2d
   
   # Filtering
   from pytcl.mathematical_functions.signal_processing.filters import (
       design_butterworth, apply_filter
   )
   
   # Optimal detection
   from pytcl.mathematical_functions.signal_processing.matched_filter import (
       matched_filter, likelihood_ratio_test
   )
   
   # Special functions (for detection threshold calculation)
   from pytcl.mathematical_functions.special_functions import marcum_q

Searching for Functions: Advanced Tips
---------------------------------------

**Search by keyword using grep**:

.. code-block:: bash

   # Find all "distance" functions in the codebase
   grep -r "def.*distance" pytcl/
   
   # Find functions with "kalman" in name
   grep -r "def kalman" pytcl/

**Search in Jupyter**:

.. code-block:: python

   # Find all functions containing "filter"
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
   
   find_functions("filter")

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
   # (cart_coords: NDArray[np.floating[Any]], ...) -> NDArray[np.floating[Any]]
   
   # Understand the parameters
   for param_name, param in sig.parameters.items():
       print(f"{param_name}: {param.annotation}")

**Benefits**:
  - IDE autocomplete shows expected types
  - Type checking (mypy, pyright) catches errors
  - Self-documenting code

Common Errors and Solutions
----------------------------

**ImportError: No module named 'pytcl.xxx'**

Solution: Check the correct import path

.. code-block:: python

   # ❌ Wrong
   from pytcl.dynamic_estimation.kalman import kf_predict  # Module doesn't exist here
   
   # ✅ Correct
   from pytcl.dynamic_estimation.kalman import kf_predict

**TypeError: Unexpected argument type**

Solution: Check function signature

.. code-block:: python

   from pytcl.coordinate_systems.conversions import cart2sphere
   help(cart2sphere)  # Shows expected types
   
   # ❌ Wrong: function expects ndarray
   cart2sphere([1.0, 0.0, 0.0])  # List instead of ndarray
   
   # ✅ Correct
   import numpy as np
   cart2sphere(np.array([1.0, 0.0, 0.0]))

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
   
   # Single value
   result = func(np.array([1.0, 2.0, 3.0]))  # shape (3,)
   
   # Multiple values
   results = func(np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]))  # shape (2, 3)

**Return Values**:

Most functions return named tuples or dataclasses for clarity:

.. code-block:: python

   from pytcl.dynamic_estimation.kalman import KalmanUpdate
   
   result = kf_predict(x, P, F, Q)
   # result.x = predicted state
   # result.P = predicted covariance
   # result.S = innovation covariance (if applicable)

**Optional Parameters**:

Many functions have optional parameters with sensible defaults:

.. code-block:: python

   from pytcl.assignment_algorithms import relaxation_assignment_nd
   
   # Simple usage with default method
   assignments = relaxation_assignment_nd(cost_matrix)
   
   # Advanced: specify method
   assignments = relaxation_assignment_nd(cost_matrix, method='hungarian')

Best Practices
--------------

1. **Start with the highest-level API**
   
   ✅ Use: ``from pytcl.trackers import MultiTargetTracker``
   
   ❌ Avoid: Implementing by combining 10 lower-level functions

2. **Check examples for your use case**
   
   ``examples/`` folder has code for:
   - Multi-target tracking
   - INS/GNSS navigation
   - Satellite operations
   - Signal processing

3. **Use the type hints**
   
   ✅ Enable mypy checking in your IDE
   
   ✅ Trust IDE autocomplete
   
   ❌ Don't ignore type errors

4. **Understand the math**
   
   ✅ Reference :doc:`kalman_filter_tuning` for filter parameters
   
   ✅ Use :doc:`performance_optimization` for profiling
   
   ❌ Don't guess at parameters

5. **Profile before optimizing**
   
   ✅ Use cProfile to find bottlenecks
   
   ✅ See :doc:`performance_optimization` for GPU acceleration
   
   ❌ Don't prematurely optimize

See Also
~~~~~~~~

- :doc:`architecture` - Module organization and patterns
- :doc:`getting_started` - Quick start guide
- :doc:`kalman_filter_tuning` - Filter parameter guide
- :doc:`gpu_acceleration` - GPU computation
- :doc:`performance_optimization` - Profiling and optimization
- Examples: ``examples/`` for working code
- Tutorials: ``docs/notebooks/`` for interactive learning
