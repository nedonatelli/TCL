Library Architecture
====================

Overview
--------

The Tracker Component Library contains **153 modules** organized into 8 major subsystems:

.. image:: /_static/architecture.png
   :alt: Architecture diagram showing subsystem organization
   :align: center

1. **Dynamic Estimation** (15 modules) - Filtering and state estimation
2. **Data Association** (8 modules) - Assignment and correlation algorithms
3. **Coordinate Systems** (20 modules) - Conversions and transformations
4. **Dynamic Models** (12 modules) - Motion models and process noise
5. **Navigation** (18 modules) - INS, guidance, geodesy
6. **Geophysical** (15 modules) - Gravity, magnetism, weather, terrain
7. **Tracking** (12 modules) - Multi-target tracking and containers
8. **Mathematical** (25 modules) - Special functions, distributions, integration

High-Level Subsystem View
--------------------------

.. code-block:: text

   pytcl/
   ├── core/                          # Utility: types, decorators, constants
   ├── containers/                    # Data structures: TrackSet, EstimationSet
   │
   ├── dynamic_estimation/            # Filtering engines
   │   ├── kalman/                    # KF, EKF, UKF, CKF, IMM
   │   ├── particle_filters/          # PF, SISR, SIR
   │   ├── batch_estimation/          # Least squares, smoothing
   │   └── measurement_update/        # Update equations
   │
   ├── dynamic_models/                # Motion models
   │   ├── continuous_time/           # White noise acc, Gauss-Markov
   │   ├── discrete_time/             # CT to DT conversions
   │   └── process_noise/             # Q matrix generation
   │
   ├── assignment_algorithms/         # Data association
   │   ├── two_dimensional/           # 2D assignment
   │   ├── three_dimensional/         # 3D/MHT assignment
   │   └── optimization/              # Hungarian, auction, Murty
   │
   ├── trackers/                      # Tracker implementations
   │   └── multi_tracker_gnn/         # GNN tracker
   │
   ├── coordinate_systems/            # Coordinate transformations
   │   ├── conversions/               # Cart↔spherical, geodetic, etc
   │   ├── rotations/                 # Euler angles, quaternions, DCM
   │   ├── jacobians/                 # Partial derivatives
   │   └── projections/               # Map projections (UTM, Mercator)
   │
   ├── navigation/                    # Navigation
   │   ├── geodesy/                   # Great circle, rhumb line
   │   ├── ins_gnss/                  # INS mechanization
   │   ├── ephemerides/               # JPL ephemerides  
   │   └── tdoa/                      # Time difference of arrival
   │
   ├── geophysical/                   # Environmental models
   │   ├── gravity/                   # WGS84, EGM96/2008
   │   ├── magnetism/                 # WMM, IGRF
   │   ├── atmosphere/                # NRLMSISE00
   │   └── terrain/                   # Terrain elevation
   │
   ├── astronomical/                  # Orbital mechanics
   │   ├── orbital_mechanics/         # Keplerian propagation
   │   ├── lambert/                   # Lambert's problem
   │   ├── reference_frames/          # ECEF↔ECI, precession, nutation
   │   ├── relativity/                # Relativistic corrections
   │   ├── sgp4/                      # SGP4 propagator
   │   └── ephemeris/                 # Ephemeris functions
   │
   ├── mathematical_functions/        # Math utilities
   │   ├── special_functions/         # Marcum Q, Lambert W, Bessel
   │   ├── distributions/             # Probability distributions
   │   ├── statistics/                # Statistical functions
   │   ├── numerical/                 # Integration, interpolation
   │   └── optimization/              # Minimization, root finding
   │
   ├── signal_processing/             # Signal processing
   │   ├── filtering/                 # IIR, FIR filters
   │   ├── detection/                 # CFAR, matched filtering
   │   └── transforms/                # FFT, wavelets
   │
   ├── clustering/                    # Pattern recognition
   │   └── algorithms/                # K-means, DBSCAN, hierarchical
   │
   ├── performance_evaluation/        # Metrics and analysis
   │   ├── metrics/                   # NEES, NIS, Cramer-Rao
   │   ├── consistency/               # Filter consistency checks
   │   └── benchmarking/              # Performance profiling
   │
   ├── gpu/                           # GPU acceleration
   │   ├── cupy_backend/              # NVIDIA acceleration
   │   └── mlx_backend/               # Apple Silicon acceleration
   │
   └── plotting/                      # Visualization

Subsystem Details
-----------------

Dynamic Estimation Subsystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Implement state estimation algorithms (filtering).

**When to use**: Any system where you need to estimate hidden state from noisy measurements.

**Modules**:

- ``kalman`` - Linear/nonlinear Kalman variants (KF, EKF, UKF, CKF)
- ``particle_filters`` - Nonlinear, non-Gaussian filtering
- ``batch_estimation`` - Offline smoothing and batch processing
- ``measurement_update`` - Generic update equations

**Quick Example**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import KalmanFilter
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

   # Setup filter
   T = 0.1  # time step
   x = np.array([0, 1, 0, 0])  # state: pos, vel (2D)
   P = np.eye(4) * 0.1
   
   F = f_constant_velocity(T, num_dims=2)
   Q = q_constant_velocity(T, sigma_a=0.1, num_dims=2)
   H = np.eye(2, 4)  # observe position
   R = np.eye(2) * 0.5
   
   kf = KalmanFilter(F, H, Q, R)
   
   # Predict and update
   measurements = [np.array([1.1, 0.5]), np.array([2.3, 1.2]), ...]
   for z in measurements:
       x, P = kf.predict(x, P)
       x, P = kf.update(x, P, z)

**Dependency Graph**:
  - Requires: ``dynamic_models`` (F, Q matrices)
  - Requires: ``coordinate_systems`` (if fusing different coordinate types)
  - Optional: ``performance_evaluation`` (diagnostic metrics)

Data Association Subsystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Match observations (measurements) to tracks (targets).

**When to use**: Multi-target tracking with multiple sensors or measurement ambiguity.

**Modules**:

- ``two_dimensional`` - 2D assignment (gating, MHT)
- ``three_dimensional`` - 3D assignment, Murty algorithm
- ``optimization`` - Hungarian, auction algorithms

**Quick Example**:

.. code-block:: python

   from pytcl.assignment.optimization import assignment_nd
   from scipy.spatial.distance import cdist
   
   # Compute cost matrix (lower = better match)
   positions = np.array([[0, 0], [1, 1], [2, 2]])  # 3 targets
   measurements = np.array([[0.1, 0.1], [2.1, 2.1]])  # 2 measurements
   
   cost = cdist(positions[:, :2], measurements, metric='euclidean')
   cost[cost > 0.5] = np.inf  # gate: only close matches
   
   # Solve assignment
   assignments = assignment_nd(cost)  # [(target_i, meas_j), ...]

**Dependency Graph**:
  - Requires: Computed cost matrix (usually from coordinate_systems)
  - Optional: ``clustering`` (pre-grouping tracks)

Coordinate Systems Subsystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Transform between different coordinate representations and reference frames.

**When to use**: Working with data from different sensors or coordinate systems.

**Modules**:

- ``conversions`` - Spherical ↔ Cartesian, geodetic, RUV, ENU
- ``rotations`` - Euler angles, quaternions, rotation matrices (DCM)
- ``jacobians`` - Partial derivatives for Jacobian matrices
- ``projections`` - Map projections (UTM, Mercator, Lambert Conformal)

**Quick Example**:

.. code-block:: python

   from pytcl.coordinate_systems.conversions import sphere2cart, cart2sphere
   from pytcl.coordinate_systems.rotations import euler2dcm
   
   # Spherical to Cartesian
   spherical = np.array([1.0, np.pi/4, np.pi/6])  # range, az, el
   cartesian = sphere2cart(spherical)
   
   # Rotation
   euler = np.array([0, 0, np.pi/4])  # heading, pitch, roll
   DCM = euler2dcm(euler)

**Dependency Graph**:
  - Depends on: ``mathematical_functions`` (for special functions)
  - Used by: Virtually all other subsystems
  - Provides: Jacobians for ``dynamic_estimation``

Navigation Subsystem
~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Navigation algorithms including INS mechanization, geodetic calculations, and celestial navigation.

**When to use**: Building navigation systems (INS/GNSS, satellite operations, ground vehicles).

**Modules**:

- ``ins_gnss`` - INS/GNSS integration, gravity modeling
- ``geodesy`` - Great circle distance, rhumb line, coordinate transformations
- ``ephemerides`` - JPL ephemeris data (Sun, Moon, planets)
- ``tdoa`` - Time difference of arrival localization

**Quick Example**:

.. code-block:: python

   from pytcl.navigation.geodesy import geodetic2ecef, great_circle_distance
   
   # Convert geodetic to ECEF
   lat, lon, alt = 40.0, -74.0, 0.0  # degrees, degrees, meters
   ecef = geodetic2ecef(np.array([lat, lon, alt]))
   
   # Great circle distance between two points
   pos1 = np.array([40.0, -74.0])  # NYC
   pos2 = np.array([51.5, -0.1])   # London
   distance = great_circle_distance(pos1, pos2)

**Dependency Graph**:
  - Depends on: ``coordinate_systems``, ``mathematical_functions``
  - Used by: ``dynamic_estimation`` (for state propagation)
  - Provides: Environmental models for tracking

Geophysical Subsystem
~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Model environmental factors (gravity, magnetic field, atmosphere, terrain).

**When to use**: Accurate modeling of forces on vehicles, environmental corrections.

**Modules**:

- ``gravity`` - WGS84 ellipsoid, EGM96/2008 geoid, gravity anomalies
- ``magnetism`` - World Magnetic Model (WMM), IGRF
- ``atmosphere`` - NRLMSISE-00 model for density, temperature
- ``terrain`` - Terrain elevation from digital elevation models

**Quick Example**:

.. code-block:: python

   from pytcl.gravity.gravity_models import gravity_force
   from pytcl.magnetism.magnetic_models import magnetic_field
   from pytcl.atmosphere.nrlmsise00 import atmospheric_density
   
   # Compute gravitational acceleration
   ecef_pos = np.array([6378000, 0, 0])  # Earth surface
   g_accel = gravity_force(ecef_pos)  # m/s^2
   
   # Magnetic field
   mag_field = magnetic_field(ecef_pos)  # nT
   
   # Atmospheric density
   lat, lon, alt = 40.0, -74.0, 35000  # 35,000 feet
   rho = atmospheric_density(lat, lon, alt)

**Dependency Graph**:
  - Depends on: ``coordinate_systems``, ``mathematical_functions``
  - Used by: ``navigation``, ``astronomical`` (for satellite propagation)

Astronomical Subsystem
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Orbital mechanics, ephemerides, and celestial reference frames.

**When to use**: Satellite operations, space vehicle navigation, astronomical calculations.

**Modules**:

- ``orbital_mechanics`` - Keplerian propagation, anomaly conversions
- ``lambert`` - Lambert's problem (bi-elliptic solutions)
- ``reference_frames`` - ECEF↔ECI, precession, nutation, polar motion
- ``relativity`` - Relativistic corrections (Schwarzschild, Shapiro delay)
- ``sgp4`` - SGP4 propagator with perturbations
- ``ephemeris`` - JPL ephemeris interface

**Quick Example**:

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import propagate_kepler
   from pytcl.astronomical.reference_frames import ecef2eci, eci2ecef
   
   # Keplerian propagation
   semimajor_axis = 6800e3  # meters
   eccentricity = 0.001
   inclination = 98.0  # degrees (sun-synchronous)
   
   state = np.array([semimajor_axis, eccentricity, inclination, 
                     0, 0, 0])  # Keplerian elements
   
   # Propagate 1 hour
   state_prop = propagate_kepler(state, t=3600)

**Dependency Graph**:
  - Depends on: ``coordinate_systems``, ``navigation``, ``geophysical``
  - Used by: Satellite tracking systems
  - Provides: Ephemeris data to navigation systems

Mathematical Functions Subsystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Special functions, probability distributions, numerical methods.

**When to use**: Implementing advanced algorithms (Bessel functions, special distributions).

**Modules**:

- ``special_functions`` - Marcum Q, Lambert W, Bessel, hypergeometric
- ``distributions`` - Rayleigh, Rician, non-central chi-square
- ``statistics`` - Statistical tests, confidence intervals
- ``numerical`` - Integration, interpolation, root finding
- ``optimization`` - Minimization, root solving

**Quick Example**:

.. code-block:: python

   from pytcl.mathematical_functions.special_functions import (
       marcum_q, lambert_w
   )
   from pytcl.mathematical_functions.distributions import (
       rician_pdf, rayleigh_pdf
   )
   
   # Marcum Q function (detection probability)
   a, b = 2.0, 3.0
   q_val = marcum_q(a, b)  # Detection probability
   
   # Rician distribution (fading signals)
   signal = 1.0
   noise = 0.5
   x = 2.0
   pdf_val = rician_pdf(x, signal, noise)

**Dependency Graph**:
  - Depends on: NumPy, SciPy (foundation)
  - Used by: Nearly all other subsystems

Tracking Subsystem
~~~~~~~~~~~~~~~~~~~

**Purpose**: Multi-target tracking implementations and related containers.

**When to use**: Building complete tracking systems from filters and association.

**Modules**:

- ``multi_tracker_gnn`` - Global Nearest Neighbor tracker
- ``containers`` - TrackSet, EstimationSet data structures
- ``clustering`` - Track clustering algorithms

**Quick Example**:

.. code-block:: python

   from pytcl.trackers.multi_tracker_gnn import GNNTracker
   from pytcl.containers import TrackSet, Track
   
   # Create tracker
   tracker = GNNTracker(
       gate_threshold=5.0,
       new_track_threshold=3,
       delete_threshold=5
   )
   
   # Process measurements
   for measurements in measurement_stream:
       tracks = tracker.update(measurements)

**Dependency Graph**:
  - Depends on: ``dynamic_estimation``, ``assignment_algorithms``, ``coordinate_systems``
  - Used by: Complete tracking applications

Common Architecture Patterns
----------------------------

Pattern 1: Single-Target Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Tracking one object with one or more sensors.

**Workflow**:

.. code-block:: python

   # 1. Get measurements
   measurements = sensor.get_observations()
   
   # 2. Convert to common frame
   from pytcl.coordinate_systems import sphere2cart
   cartesian_meas = sphere2cart(measurements)
   
   # 3. Filter update
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
   x_pred, P_pred = kf_predict(x, P, F, Q)
   x_upd, P_upd = kf_update(x_pred, P_pred, cartesian_meas, H, R)

**Modules Used**:
  - ``dynamic_estimation``: Filter
  - ``dynamic_models``: Motion model (F, Q)
  - ``coordinate_systems``: Measurement conversion

Pattern 2: Multi-Target Tracking with Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Tracking multiple targets with data association.

**Workflow**:

.. code-block:: python

   # 1. Predict all tracks
   for track in tracks:
       track.x_pred, track.P_pred = kf_predict(track.x, track.P, F, Q)
   
   # 2. Compute assignment cost
   from scipy.spatial.distance import cdist
   positions = np.array([t.x_pred[:2] for t in tracks])
   measurements = get_measurements()
   cost = cdist(positions, measurements)
   cost[cost > gate_dist] = np.inf
   
   # 3. Solve assignment
   from pytcl.assignment.optimization import assignment_nd
   assignments = assignment_nd(cost)
   
   # 4. Update assigned tracks
   for track_idx, meas_idx in assignments:
       if meas_idx >= 0:
           x_upd, P_upd = kf_update(
               tracks[track_idx].x_pred,
               tracks[track_idx].P_pred,
               measurements[meas_idx],
               H, R
           )

**Modules Used**:
  - ``dynamic_estimation``: Filter
  - ``assignment_algorithms``: Data association
  - ``coordinate_systems``: Gating

Pattern 3: INS/GNSS Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Fusing inertial and GPS data for robust navigation.

**Workflow**:

.. code-block:: python

   from pytcl.navigation.ins_gnss import ins_update, gnss_update
   
   # 1. IMU propagation (continuous)
   accel, gyro = imu.read()
   state, covariance = ins_update(state, covariance, accel, gyro, dt)
   
   # 2. GPS update (intermittent)
   if gps.has_fix():
       gnss_pos = gps.get_position()
       state, covariance = gnss_update(state, covariance, gnss_pos, gnss_std)

**Modules Used**:
  - ``navigation``: INS mechanization
  - ``dynamic_estimation``: Filter equations
  - ``coordinate_systems``: Frame conversions

Pattern 4: Satellite Propagation with Corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Accurate satellite orbit prediction.

**Workflow**:

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import propagate_kepler
   from pytcl.astronomical.reference_frames import ecef2eci
   from pytcl.geophysical.gravity import perturbations
   
   # 1. Initial state in ECI
   state_eci = get_initial_state()
   
   # 2. Propagate with perturbations
   for t in time_steps:
       # Keplerian propagation
       state_eci = propagate_kepler(state_eci, dt)
       
       # Apply perturbations
       perturbations_eci = perturbations(state_eci)
       state_eci += perturbations_eci * dt
   
   # 3. Convert to ECEF for ground tracking
   state_ecef = ecef2eci(state_eci)

**Modules Used**:
  - ``astronomical``: Orbit propagation
  - ``geophysical``: Environmental effects
  - ``coordinate_systems``: Frame conversions

Dependency Resolution Guide
----------------------------

**Question**: How do I find what modules I need?

**Answer**: Follow this decision tree:

1. **Are you doing state estimation?**
   
   → YES: Use ``dynamic_estimation``
   
   - Need motion model? → Use ``dynamic_models``
   - Need to convert measurements? → Use ``coordinate_systems``

2. **Are you tracking multiple targets?**
   
   → YES: Use ``assignment_algorithms``
   
   - Need advanced assignment? → Use ``three_dimensional``

3. **Are you working with different coordinate systems?**
   
   → YES: Use ``coordinate_systems``
   
   - Need rotations? → Use ``rotations``
   - Need projections? → Use ``projections``

4. **Are you doing navigation?**
   
   → YES: Use ``navigation``
   
   - Need geodetic calculations? → Use ``geodesy``
   - Need INS/GNSS fusion? → Use ``ins_gnss``
   - Need satellite operations? → Use ``astronomical``

5. **Do you need environmental modeling?**
   
   → YES: Use ``geophysical``
   
   - Gravity effects? → Use ``gravity``
   - Magnetic field? → Use ``magnetism``
   - Atmospheric? → Use ``atmosphere``

Module Import Patterns
----------------------

**Single Function Import** (for specific use):

.. code-block:: python

   from pytcl.coordinate_systems.conversions import sphere2cart
   result = sphere2cart(data)

**Submodule Import** (for multiple related functions):

.. code-block:: python

   from pytcl.dynamic_estimation import kalman
   x_pred, P_pred = kalman.kf_predict(x, P, F, Q)

**Top-Level Import** (for convenience):

.. code-block:: python

   import pytcl
   result = pytcl.sphere2cart(data)  # If re-exported

**Check available functions in module**:

.. code-block:: python

   import pytcl.coordinate_systems.rotations as rotations
   print([name for name in dir(rotations) if not name.startswith('_')])

Best Practices by Use Case
---------------------------

**Rapid Prototyping**
  → Use top-level functions, simple examples

**Production Tracking System**
  → Use ``trackers.multi_tracker_gnn`` directly
  → Add diagnostic checks from ``performance_evaluation``

**Research/Development**
  → Use individual modules for maximum flexibility
  → Combine filters with custom models

**High-Performance Real-Time**
  → Use GPU acceleration (``gpu`` module)
  → Pre-compute Jacobians
  → Use Numba-accelerated functions
  → See :doc:`performance_optimization`

**Safety-Critical Systems**
  → Use consistency checks from ``performance_evaluation``
  → Monitor NEES/NIS values
  → Use multiple independent filters for voting
  → Validate with ``kalman_filter_tuning`` diagnostics

See Also
~~~~~~~~

- :doc:`kalman_filter_tuning` - Detailed filter parameter tuning
- :doc:`gpu_acceleration` - Using GPU computation
- :doc:`performance_optimization` - Performance profiling and optimization
- :doc:`getting_started` - Getting started guide
- Examples: ``examples/`` folder with complete working code
