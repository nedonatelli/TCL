Library Architecture
====================

Overview
--------

``pytcl`` is **145 modules** across **22 implemented packages**, exporting about
1046 public names. There is no framework object to inherit from: the library is a
set of composable functions and small result types, so a tracker is something you
assemble rather than something you subclass.

Three placeholder packages — ``misc``, ``physical_values`` and ``scheduling``
— used to exist with no modules, mirroring the MATLAB directory layout. They
were removed before 2.0.0: an importable-but-empty package implied support
that does not exist, which is worse than the honest ``ImportError``.
``pytcl.transponders`` was one of them until v2.2.0, when AIS decoding (see
:mod:`pytcl.transponders.ais`) gave it real content; if the remaining three
are ported (see :doc:`matlab_parity_inventory`), they return the same way.

Subsystem Map
-------------

Packages grouped by the role they play. Arrows point from a consumer toward what
it depends on; only the structurally significant edges are drawn.

.. mermaid::

   graph TD
       subgraph Estimation
           DE["dynamic_estimation<br/>17 modules"]
           DM["dynamic_models<br/>7 modules"]
           SE["static_estimation<br/>3 modules"]
       end

       subgraph Association_and_Tracking
           AA["assignment_algorithms<br/>9 modules"]
           TR["trackers<br/>5 modules"]
           CL["clustering<br/>4 modules"]
           CO["containers<br/>8 modules"]
       end

       subgraph Geometry_and_Navigation
           CS["coordinate_systems<br/>5 modules"]
           NA["navigation<br/>5 modules"]
           AS["astronomical<br/>9 modules"]
       end

       subgraph Environment_Models
           GR["gravity<br/>5 modules"]
           MA["magnetism<br/>3 modules"]
           AT["atmosphere<br/>5 modules"]
           TE["terrain<br/>3 modules"]
       end

       subgraph Foundation
           MF["mathematical_functions<br/>25 modules"]
           CR["core<br/>7 modules"]
       end

       subgraph Support
           IO["io<br/>12 modules"]
           GP["gpu<br/>7 modules"]
           PL["plotting<br/>4 modules"]
           PE["performance_evaluation<br/>2 modules"]
       end

       TR --> DE
       TR --> AA
       TR --> CO
       DE --> DM
       DE --> MF
       AA --> MF
       CL --> CO
       NA --> CS
       NA --> GR
       AS --> CS
       SE --> MF
       CS --> MF
       PE --> MF
       IO --> CO
       GP --> MF
       PL --> CO
       DM --> CR
       MF --> CR

Tracking Pipeline
-----------------

The order in which these subsystems are normally composed. Every stage is a
function call over plain arrays and small result objects, so any one of them can
be replaced without touching the others.

.. mermaid::

   flowchart LR
       M["Raw measurements<br/>range, bearing, elevation"] --> C
       C["coordinate_systems<br/>convert to a common frame"] --> G
       G["assignment_algorithms<br/>gating"] --> A
       A["assignment_algorithms<br/>data association"] --> F
       F["dynamic_estimation<br/>predict and update"] --> TM
       TM["trackers<br/>initiation, confirmation, deletion"] --> P
       P["io<br/>persist tracks"] --> E
       E["performance_evaluation<br/>NEES, NIS, OSPA"]
       DM["dynamic_models<br/>F and Q"] -.-> F
       E -.->|tune| DM

Per-detection Measurement Covariance
------------------------------------

``SingleTargetTracker.update`` and ``MultiTargetTracker.process`` accept a
covariance per detection, not just the fixed ``R`` given to the constructor:

.. code-block:: python

   # each detection carries the covariance that actually applies to it
   tracks = tracker.process(detections, dt, measurement_covariances=covariances)

This matters whenever the measurement error is not the same for every
detection, and a converted polar detection is the common case. Its Cartesian
covariance is ``J R_polar J.T``, which is anisotropic and grows with range:
down-range spread stays at ``sigma_range`` while cross-range spread is
``r * sigma_bearing``.

No single ``R`` describes that. Size it to the down-range term and the gate is
too tight at long range -- true detections fall outside it and the tracker
starts duplicate tracks. Size it to the cross-range term and cardinality is
right, but the covariance is inflated and the filter under-reports its own
accuracy. Supplying each detection's covariance avoids the choice; both the
gate and the Kalman gain then use the covariance that applies.

Estimator Families
------------------

What ``dynamic_estimation`` actually provides. These are exposed as
``*_predict`` / ``*_update`` function pairs rather than filter classes, which is
why the examples below thread state through explicitly.

.. mermaid::

   graph LR
       KF["Linear Kalman<br/>kf_predict / kf_update"]
       KF --> EKF["Extended<br/>ekf_predict / ekf_update"]
       KF --> UKF["Unscented<br/>ukf_predict / ukf_update"]
       KF --> SR["Square-root and UD<br/>numerically robust"]
       KF --> CE["Constrained<br/>ConstrainedEKF"]
       KF --> INF["Information and SRIF"]
       KF --> IMM["IMM<br/>maneuver switching"]
       PF["Particle filters"] --> RBPF["Rao-Blackwellised"]
       GSF["Gaussian sum"]
       KF --> SM["Smoothers<br/>RTS, fixed-lag, two-filter"]
       PF --> SM

Package Reference
-----------------

Counts are measured from the packages themselves rather than asserted;
``tests/test_docs_architecture.py`` fails if this table drifts.

.. list-table::
   :header-rows: 1
   :widths: 26 10 10 54

   * - Package
     - Modules
     - Public
     - Purpose
   * - ``mathematical_functions``
     - 25
     - 70
     - Special functions, statistics, transforms, signal processing, geometry
   * - ``dynamic_estimation``
     - 17
     - 88
     - Kalman variants, particle filters, smoothers, information filters
   * - ``assignment_algorithms``
     - 9
     - 52
     - 2-D, 3-D and N-D assignment, gating, JPDA, k-best, network flow
   * - ``astronomical``
     - 9
     - 149
     - Orbital mechanics, SGP4/SDP4, TLEs, ephemerides, reference frames
   * - ``containers``
     - 8
     - 35
     - Spatial indices (k-d tree, R-tree, VP-tree, cover tree) and track sets
   * - ``core``
     - 7
     - 72
     - Constants, exceptions, validation, array helpers, data paths
   * - ``dynamic_models``
     - 7
     - 37
     - Motion models and process-noise matrices, continuous and discrete
   * - ``gpu``
     - 7
     - 38
     - CuPy and MLX backends for array-heavy routines
   * - ``io``
     - 12
     - 37
     - Track and measurement persistence, including HDF5
   * - ``coordinate_systems``
     - 5
     - 69
     - Frame conversions, rotations, Jacobians, map projections
   * - ``gravity``
     - 5
     - 58
     - Spherical-harmonic gravity, EGM, solid Earth tides
   * - ``navigation``
     - 5
     - 101
     - Geodesy, great circle and rhumb line, INS and INS/GNSS
   * - ``clustering``
     - 4
     - 28
     - k-means, DBSCAN, hierarchical, Gaussian mixtures
   * - ``plotting``
     - 4
     - 30
     - Track, uncertainty and coverage visualization
   * - ``trackers``
     - 5
     - 19
     - End-to-end single- and multi-target trackers
   * - ``atmosphere``
     - 5
     - 57
     - Simplified thermosphere, standard atmospheres, ionosphere,
       humidity, refractivity
   * - ``magnetism``
     - 3
     - 34
     - WMM, WMMHR, IGRF, EMM
   * - ``static_estimation``
     - 3
     - 38
     - Least squares, robust estimation, model selection
   * - ``terrain``
     - 3
     - 30
     - DEM handling, line of sight, horizon and viewshed
   * - ``performance_evaluation``
     - 2
     - 21
     - NEES, NIS, OSPA, Cramer-Rao bounds
   * - ``diagnostics``
     - 1
     - 10
     - Opt-in loguru logging, ASCII-safe rich progress bars and track tables
   * - ``transponders``
     - 1
     - 5
     - AIS NMEA decoding, checksum validation, and position-report
       extraction (pyais)

Composition Examples
--------------------

Each of these runs as written; the same test that checks the table also checks
that every import on this page resolves.

Filtering a constant-velocity target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity
   from pytcl.dynamic_estimation import kf_predict, kf_update

   T = 0.1
   F = f_constant_velocity(T, num_dims=2)
   Q = q_constant_velocity(T, sigma_a=0.1, num_dims=2)

   # f_constant_velocity is block diagonal -- one (position, velocity) pair
   # per spatial dimension -- so the state is [x, vx, y, vy], not
   # [x, y, vx, vy]. Getting this backwards silently measures velocity.
   x = np.array([0.0, 1.0, 0.0, 0.5])   # x, vx, y, vy
   P = np.eye(4) * 0.1

   pred = kf_predict(x, P, F, Q)

   H = np.array([[1.0, 0.0, 0.0, 0.0],    # observe x
                 [0.0, 0.0, 1.0, 0.0]])   # observe y
   R = np.eye(2) * 0.05
   upd = kf_update(pred.x, pred.P, np.array([0.1, 0.05]), H, R)

Associating measurements to tracks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pytcl.assignment_algorithms import assign2d

   # cost[i, j] is the cost of assigning track i to measurement j
   cost = np.array([[4.0, 1.0, 3.0],
                    [2.0, 0.0, 5.0],
                    [3.0, 2.0, 2.0]])
   result = assign2d(cost)
   # result.col_indices[i] is the measurement assigned to track i

Geodetic conversion and range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pytcl.navigation import geodetic_to_ecef, inverse_geodetic

   dc_lat, dc_lon = np.radians(38.9), np.radians(-77.0)
   ny_lat, ny_lon = np.radians(40.7), np.radians(-74.0)

   x, y, z = geodetic_to_ecef(dc_lat, dc_lon, 0.0)
   distance, azimuth, back_azimuth = inverse_geodetic(dc_lat, dc_lon, ny_lat, ny_lon)

Optional Dependencies
---------------------

A core install pulls only ``numpy``, ``scipy``, ``numba``, ``h5py``,
``loguru``, ``msgspec`` and ``rich``.
Everything else is opt-in, so an import failing inside one of these packages
usually means a missing extra rather than a bug.

.. mermaid::

   graph LR
       CORE["core install<br/>numpy, scipy, numba, h5py,<br/>loguru, msgspec, rich"]
       CORE --> AST["astronomy<br/>astropy, jplephem"]
       CORE --> GEO["geodesy<br/>pyproj, geographiclib"]
       CORE --> TER["terrain<br/>netCDF4"]
       CORE --> VIS["visualization<br/>plotly"]
       CORE --> SIG["signal<br/>pywavelets"]
       CORE --> GPU["gpu, gpu-apple<br/>cupy, mlx"]
       AST --> ALL["all<br/>everything except gpu"]
       GEO --> ALL
       TER --> ALL
       VIS --> ALL
       SIG --> ALL

Angles and Units
----------------

Every angle at an API boundary is in **radians** and every distance in
**meters**, unless a function documents otherwise. Passing degrees where radians
are expected is the most common way to get results that look plausible and are
wrong: as a latitude in radians, ``-5`` is -286 degrees, which is past the poles.

See Also
--------

- :doc:`getting_started` - installation and first steps
- :doc:`performance_optimization` - profiling and GPU acceleration
- :doc:`examples/index` - runnable example scripts
