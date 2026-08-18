MATLAB TCL parity inventory
===========================

A function-level comparison of pytcl against the MATLAB library at commit
``593ce51`` of the U.S. Naval Research Laboratory's `Tracker Component
Library <https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary>`_,
produced by walking every directory of the MATLAB repository rather than by
asserting a percentage.

Why this document exists
------------------------

Earlier documentation claimed "full feature parity" and "100% MATLAB parity".
Those claims were scoped to a tier-1/tier-2 component list that was never
published alongside them, and one of their checkmarks — NRLMSISE-00 — proved
to be a placeholder shipped under the model's name (gh-79). This inventory
replaces assertion with enumeration.

Method
------

The MATLAB public surface was counted under MATLAB's actual visibility rules:
a regular ``.m`` file exposes exactly its first function (later declarations
are file-local subfunctions), and ``classdef`` files expose their methods
except those in ``Access = private`` blocks. Third-party code, sample code and
compiled artifacts are excluded. That yields **2,549 public names** across
1,843 files — against **1,899 public names** in pytcl.

Automated name matching is a lower bound only: MATLAB ships one file per
variant where pytcl uses a keyword argument, and the port renamed
systematically (``KalmanUpdate`` → ``kf_update``). The verdicts below come
from reading each area's function list against pytcl's modules.

Summary
-------

**pytcl ports the core tracking workflow comprehensively and validates it
against independent references, but the MATLAB library's full surface is
substantially broader.** By function count, coverage is roughly a third of
the MATLAB public surface. By workflow — filter, associate, track, evaluate,
in Earth-referenced coordinates — coverage is near-complete, and in several
places pytcl exceeds the original (OSPA/MOT metrics, ionosphere models,
R-trees and cover trees, min-cost flow, SQL/HDF5 track storage, GPU
backends).

.. list-table::
   :header-rows: 1
   :widths: 30 10 12 48

   * - MATLAB area
     - Public
     - Coverage
     - Notes
   * - Dynamic_Estimation
     - 113
     - Core strong
     - KF/EKF/UKF/CKF, square-root and UD forms, SRIF, information filter,
       H-infinity, IMM, particle filters, RBPF, RTS/two-filter/fixed-lag/
       fixed-interval smoothers all ported. **Missing:** EnKF, ESRIF,
       quasi-Monte-Carlo Kalman variants, BLUE polar/spherical measurement
       updates, progressive Gaussian update, pure-propagation filter,
       reduced-state filters, batch least-squares estimators, PCRLB/Riccati
       analysis tools.
   * - Dynamic_Models
     - 62
     - Partial
     - CV/CA, coordinated turn (2D/3D/polar), Singer, Gauss-Markov F and Q
       matrices ported. **Missing:** weave/spiral maneuver models,
       flat-to-curved-Earth dynamics, geodesic Brownian motion, trajectory
       generators with constrained endpoints, process-noise suggestion
       tooling.
   * - Assignment_Algorithms
     - 45
     - Core strong
     - 2D assignment (Hungarian, auction, Murty k-best), 3D assignment,
       JPDA with gating, min-cost flow all ported and oracle-validated.
       **Missing:** bottleneck assignment, knapsack, transportation problem,
       stable-matching family, missed-detection LR matrix builders,
       assignment-probability calculators beyond JPDA.
   * - Coordinate_Systems
     - 331
     - Partial
     - Cartesian/spherical/polar/geodetic/ENU/NED/SEZ conversions, rotations,
       quaternions, r-u-v (range plus direction cosines, ``cart2ruv``/
       ``ruv2cart``), UTM and standard projections ported and validated to
       sub-mm against pyproj/EPSG. **Missing:** the angle-only direction-
       cosine UV measurement system (``spher2Uv``/``uv2Spher``-style
       conversions, u-v-w unit vectors, and their Jacobians -- the natural
       measurement space of a planar phased array; distinct from the ported
       r-u-v triple), most of the 65-function time
       suite (pytcl covers UTC/TAI/TT/GPS and Julian dates; TDB/TCB/TCG,
       Besselian epochs, sidereal local time variants absent), the 30
       measurement Jacobians and 14 Hessians as standalone functions, exotic
       projections (bipolar, gnomonic, azimuthal-equidistant families),
       ellipsoidal-harmonic coordinates, RF transform-parameter estimation.
   * - Mathematical_Functions
     - 1,506
     - Selective
     - The largest area and the largest gap, though much of it is generic
       numerics rather than tracking. Ported well: signal processing (CFAR
       family, matched filtering, STFT/CWT/DWT), core statistics (12
       distribution classes vs MATLAB's ~40), interpolation, basic matrix
       operations, special functions. The cubature-point library (~148
       files -- a signature strength of the MATLAB TCL) is now well
       covered: the degree-5/7 Gaussian rules including the full
       ``seventh_order_cubature_points`` algorithm surface, spherical-radial
       points (with the ``beta`` generalization), Genz-Keister nested
       rules, the 14th-order and 2nd-order (Julier) rules, Student-t
       cubature points, Smolyak sparse grids over the Genz-Keister
       sequences, and tensor Gauss-Hermite. **Remaining:** Gaussian LCD
       samples and the uniform region-cubature rules (Cube/Simplex/Sphere/
       Spherical-Surface and beyond; see :doc:`roadmap`). Also thin or
       absent: combinatorics (113 vs
       18), polynomials (55; no pytcl counterpart), geometry beyond basics
       (81), continuous optimization (37), specific integrals/derivatives,
       graph algorithms, accurate-arithmetic helpers.
   * - Astronomical_Code
     - 28
     - Partial
     - SGP4 (validated against the official library), Kepler propagation,
       orbital elements, Lambert solvers, JPL ephemerides via jplephem.
       **Missing:** Hipparcos catalog access, aberration and light-deflection
       corrections, EOP acquisition, angles-only initial orbit determination,
       two-point velocity determination, equinoctial Kepler solver.
   * - Atmosphere_and_Refraction
     - 34
     - Weak
     - U.S. Standard Atmosphere 1976/ISA validated; barometric thermosphere
       with documented limits (gh-79). **Missing:** the entire refraction
       suite (astronomical refraction add/remove, standard refraction ray
       tracing, refractivity models), all humidity conversions, Jacchia
       model, NRLMSISE-00 proper. pytcl adds ionosphere models (Klobuchar,
       TEC) that the MATLAB area lacks.
   * - Gravity
     - 14
     - Good
     - EGM coefficient loading, geoid height, normal gravity, tide offsets
       (solid/pole/ocean) ported. **Missing:** lunar gravity coefficients,
       polar-motion/drift coefficient adjustments, ellipsoidal parameter
       conversions.
   * - Magnetism
     - 12
     - Split
     - Coefficient loading and field evaluation for WMM/IGRF/EMM ported and
       validated to sub-nT. **Missing:** the coordinate-system half — apex
       and quasi-dipole coordinates, centered-dipole transforms,
       magnetic-heading conversions, field-line tracing.
   * - Navigation
     - 21
     - Good
     - Direct/indirect geodesic, great-circle and rhumb problems ported,
       validated against GeographicLib; rhumb intersection is ported as
       ``rhumb_intersect`` and great-circle TDOA as ``great_circle_tdoa_loc``.
       **Missing:** geodesic and great-circle intersections, surface angles.
   * - Static_Estimation
     - 11
     - Divergent
     - The name overlaps; the content mostly does not. MATLAB's area is
       target localization: TDOA, Doppler-only init, range-rate-only,
       direction-only static estimators — **almost all absent** from pytcl.
       pytcl's ``static_estimation`` instead ports the general estimators
       (OLS/TLS/RANSAC/robust/RLS/MLE) that MATLAB keeps elsewhere.
   * - Clustering_and_Mixture_Reduction
     - 14
     - Good
     - Runnalls and West mixture reduction, k-means(++), EM clustering
       ported; pytcl adds DBSCAN and hierarchical clustering. **Missing:**
       ISE-based reduction and its gradients, brute-force reduction,
       windowed grid centroiding.
   * - Container_Classes
     - 265
     - Different emphasis
     - k-d tree and metric tree ported (as KDTree/BallTree/VPTree),
       ClusterSet ported; pytcl adds R-trees, cover trees and the whole
       track/measurement container layer. **Missing:** AVL tree, binary
       heaps, disjoint sets, linked lists, interval class, B-spline class —
       largely idiomatic-Python non-goals, but absent nonetheless.
   * - Performance_Evaluation
     - 12
     - Split
     - NEES (with confidence bounds), RMSE ported; pytcl adds the standard
       OSPA metric and CLEAR-MOT metrics. **Missing:** MATLAB's MOSPA/MMOSPA
       family (``calcMOSPAError``, ``MMOSPA2Tar2D`` — a related but distinct
       OSPA-family capability), AEE/GAE, jitter, MERF, non-credibility
       index.
   * - Terrain
     - 3
     - Good
     - Earth2014 loading ported (plus GEBCO, which MATLAB lacks); solid-tide
       shift ported. EGM2008 terrain coefficients not ported.
   * - Misc
     - 52
     - Mostly N/A
     - Bit manipulation, MATLAB plotting, file utilities — superseded by
       numpy/matplotlib/stdlib. Substantive absences: GSHHG coastline data
       access and ``pointIsOnLand``.
   * - Transponders
     - 3
     - Partial
     - MATLAB's ``Transponders`` directory holds exactly three functions:
       ``decodeAISString`` (parse an NMEA sentence to a struct, wrapping
       libais, a C library), ``decodeAISPosReports2Mat`` (extract position
       reports into a matrix), and ``NMEAChecksum`` (compute/validate an
       NMEA sentence's trailing ``*hh`` checksum, usable whether decoding
       or constructing one). pytcl (v2.2.0) ports the first two as
       :mod:`pytcl.transponders.ais`'s ``decode_ais`` and
       ``ais_position_reports``, which reassemble multipart
       ``!AIVDM``/``!AIVDO`` sentences and extract position reports (message
       types 1/2/3/18/19), normalizing ITU-R M.1371 "not available"
       sentinels to NaN -- via `pyais <https://pypi.org/project/pyais/>`_,
       a pure-Python decoder, rather than a libais binding, so the two are
       functionally equivalent but not the same implementation. Validated
       against 6,808 real position reports from 299 ships off Norway (see
       :doc:`results_io`). **Missing:** a standalone
       ``NMEAChecksum``-equivalent (pytcl validates checksums internally
       inside pyais's decode path, not as its own callable function), and
       NMEA sentence *construction* in either direction (encoding a
       message back to ``!AIVDM`` text) -- neither is ported.
   * - Scheduling
     - 4
     - Absent
     - Interval scheduling algorithms not ported.
   * - Physical_Values
     - 14
     - Partial
     - Physical constants ported. Radar bands, water permittivity and
       reflection-coefficient models absent.

What pytcl has that the MATLAB library does not
-----------------------------------------------

The comparison runs both ways. pytcl adds: the standard OSPA metric and CLEAR-MOT evaluation,
ionospheric delay models, R-trees and cover trees, DBSCAN and hierarchical
clustering, min-cost-flow assignment, SQL and HDF5 track storage with
migration tooling, dual-backend GPU acceleration, and a validation suite of
6,200+ tests with a 43-file validation suite checking against independent references — the
MATLAB library distributes no test suite at all.

Honest bottom line
------------------

The defensible claim is not "full feature parity". It is: **the core tracking
workflow is ported, oracle-validated, and in places extended; the MATLAB
library's long tail — its cubature collection, refraction suite, time-scale
zoo, localization estimators and specialized coordinate systems — is
substantially unported, at roughly a third of the full public surface by
function count.** Areas above marked Absent, Weak or Divergent are the honest
priority list for anyone who needs them.
