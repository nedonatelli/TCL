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
R-trees and cover trees, STFT/wavelet transforms and matched filtering,
SQL/HDF5 track storage, GPU backends). Min-cost flow is *not* among them:
MATLAB ships ``Mathematical_Functions/Graph_Algorithms/minCostFlow.m``.

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
       H-infinity, IMM, particle filters, RBPF, and RTS/two-filter/fixed-lag
       smoothers all ported (``fixed_interval_smoother`` is a documented
       alias for the RTS smoother, not a fourth algorithm). **Missing:**
       EnKF, ESRIF, quasi-Monte-Carlo Kalman variants, BLUE polar/spherical
       measurement updates, progressive Gaussian update, pure-propagation
       filter, reduced-state filters, batch least-squares estimators,
       PCRLB/Riccati analysis tools. Also absent, and larger than that list
       suggests: the entire ``Measurement_Update/Update_Parts`` tree (30
       files, 27% of the area -- separate gain, measurement-prediction and
       update-with-prediction entry points, which pytcl does not decompose
       its updates into), continuous-discrete propagation
       (``State_Propagation/Continuous_Time``, 6 files), divided-difference
       filters, one/two-point initialization, and batch/FIR smoothers. By
       counterpart count roughly 20 of the 113 MATLAB functions have a
       pytcl equivalent; "Core strong" describes the KF family's depth, not
       breadth across the area.
   * - Dynamic_Models
     - 62
     - Partial
     - CV/CA (arbitrary polynomial order, via ``f_poly_kal``/``q_poly_kal``),
       coordinated turn (2D/3D/polar) and Singer F and Q matrices ported.
       **Missing:** Gauss-Markov at orders other than 2 -- MATLAB's
       ``FGaussMarkov``/``QGaussMarkov`` are arbitrary-order (order 0 is
       Ornstein-Uhlenbeck, 1 is integrated OU, 2 is Singer) and pytcl ports
       only the order-2 Singer case, with no ``order`` argument; also
       weave/spiral maneuver models, flat-to-curved-Earth dynamics, geodesic
       Brownian motion, trajectory generators with constrained endpoints,
       process-noise suggestion tooling.
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
       ``ruv2cart``) and the standard projections ported; UTM is validated to
       sub-mm against pyproj/EPSG, but ``stereographic`` and
       ``azimuthal_equidistant`` are spherical approximations that diverge
       from PROJ by kilometres away from the projection centre (their own
       docstrings tabulate this; gh-25). The angle-only direction-cosine UV
       measurement system's core conversions are now ported
       (``coordinate_systems.conversions.uv``, v2.8.0: u-v <->
       spherical, full bistatic r-u-v, camera-to-uv); its Jacobians,
       Hessians and cubature/Taylor covariance conversions are not.
       **Missing:** most of the 65-function time
       suite (pytcl covers UTC/TAI/TT/GPS and Julian dates; TDB/TCB/TCG,
       Besselian epochs, sidereal local time variants absent), 26 of the 30
       measurement Jacobians (spherical, polar, r-u-v, ENU/NED and geodetic
       Jacobians *are* ported, in ``coordinate_systems/jacobians/``) and all
       14 Hessians, the ellipsoidal azimuthal-equidistant family (pytcl's
       ``azimuthal_equidistant`` is a spherical approximation), other exotic
       projections (bipolar, gnomonic), ellipsoidal-harmonic coordinates, RF
       transform-parameter estimation.
   * - Mathematical_Functions
     - 1,440
     - Selective
     - The largest area and the largest gap, though much of it is generic
       numerics rather than tracking. Ported well: the CFAR family
       (pytcl adds GO/SO/2-D beyond MATLAB's CA and OS), core statistics
       (12 distribution classes vs MATLAB's 54), interpolation, special
       functions. Note that pytcl's STFT/CWT/DWT and matched filtering are
       **not** ports -- the MATLAB library contains no STFT, wavelet or
       matched-filter code at all, so they belong under "what pytcl adds"
       below, not under parity. Signal processing overall is thin against
       the original: roughly 6 of 51 MATLAB names, with the entire
       17-file ``Array_Processing`` subtree (tapering, beam patterns,
       subarray weights) unported. Basic matrix operations likewise: about
       10 of 99, with the ``Joint_Matrix_Diagonalization`` and ``Tensors``
       subtrees absent. The cubature-point library (~148
       files -- a signature strength of the MATLAB TCL) is now well
       covered: the degree-5/7 Gaussian rules including the full
       ``seventh_order_cubature_points`` algorithm surface, spherical-radial
       points (with the ``beta`` generalization), Genz-Keister nested
       rules, the 14th-order and 2nd-order (Julier) rules, Student-t
       cubature points, Smolyak sparse grids over the Genz-Keister
       sequences, and tensor Gauss-Hermite. Gaussian LCD samples
       (``gaussian_lcd_samples``) and the general-dimension uniform
       region-cubature rules (Cube/Simplex/Sphere/Spherical-Surface) have
       since shipped. **Remaining:** the 48 dimension-specialized
       subdirectory files within that same region-cubature subset (fixed
       2D/3D formulas in Cube/, Square/, Tetrahedra/, Triangles/), three
       1-D quadrature building blocks, and the region types outside that
       subset (``Prism``, ``Pyramid``, ``Cross_Polytope``, ``Exp_Weight``,
       ``Weighted_Ellipse``, ``Hexagon``, ``Spherical_Shell`` -- 33 files --
       plus loose torus and n-dimensional-shell rules; see :doc:`roadmap`).
       There is no cone or wedge rule anywhere in the MATLAB library. Also thin or
       absent: combinatorics (113 vs
       18), polynomials (55; no pytcl counterpart), geometry beyond basics
       (81), continuous optimization (37), most specific
       integrals/derivatives (the Carlson symmetric and incomplete elliptic
       integrals *are* ported, in ``special_functions/elliptic.py``), graph
       algorithms other than min-cost flow, accurate-arithmetic helpers.
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
     - Partial
     - U.S. Standard Atmosphere 1976/ISA validated; barometric thermosphere
       with documented limits (gh-79); all 10 humidity conversions, both
       dew-point functions, the refractivity helpers and the
       astronomical-refraction group (``SinclairAtmos``,
       ``removeAstroRefrac``/``addAstroRefrac`` with all three algorithms,
       ``simpAstroRefParam`` transcribed from the in-tree SOFA source)
       and the complete ``Standard_Exponential_Model`` suite (bistatic
       r-u-v ray tracing, bias approximation, cubature conversions,
       refractivity reduction) ported and validated against MATLAB
       fixtures; speed of sound (ideal-gas and Cramer algorithms).
       **Missing:** the gas-table speed-of-sound algorithm (needs
       NRLMSISE-00 output), the Jacchia 1971 model (pure MATLAB and
       portable, but its validation oracle -- Sun position via
       ``readJPLEphem``, ``GCRS2ITRS``, ``Cal2UTC`` -- is the SOFA MEX
       chain plus JPL ephemeris data, so it is deferred as an
       astronomy-integration task), NRLMSISE-00 proper (the MATLAB ``.m`` files are MEX
       stubs; the implementation is Brodowski's public-domain C port in
       ``3rd_Party_Libraries/nrlmsise-00-bc9a2fe/``, portable but a
       standalone project). pytcl adds ionosphere models (Klobuchar, TEC)
       that the MATLAB area lacks.
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
       Great-circle intersection is ported too, as ``great_circle_intersect``
       (plus a path-endpoint form, ``great_circle_path_intersect``).
       **Missing:** the *geodesic* (ellipsoidal) intersection, surface
       angles, the at-a-fixed-height ``*ProbGen`` generalizations of the
       direct/indirect geodesic and rhumb problems, geodesic midpoint, and
       standalone pseudorange localization (``pseudoRangeLoc``) -- pytcl has
       only the filter-component pieces of the last.
   * - Static_Estimation
     - 11
     - Split
     - Seven of the 11 localization estimators are ported in
       ``static_estimation.localization`` (v2.8.0):
       ``TDOAOnlyStaticLocEst``, ``rangeOnlyStaticLocEstNP``,
       ``RROnlyStaticVelEst``, ``getAdHocCartCov``, and — via the
       ``polyRootsMultiDim`` port — ``TDOA2Cart``,
       ``rangeRate2StaticPos`` and ``rangeRateRatio2StaticPos2D``, all
       validated against MATLAB fixtures. **Missing:**
       ``directionOnlyStaticLocEst``, ``computePolyMeasFIM``, and the
       two ``Uses_External_Solver`` files (need the SCS solver).
       pytcl's ``static_estimation`` also ports the general estimators
       (OLS/TLS/robust/MLE) that MATLAB keeps elsewhere; RANSAC and
       recursive least squares are pytcl originals with no MATLAB
       counterpart anywhere in the library.
   * - Clustering_and_Mixture_Reduction
     - 14
     - Good
     - Runnalls and West mixture reduction, k-means(++)
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
     - Partial
     - The Earth2014 *dataset* is read (plus GEBCO, which MATLAB lacks), but
       not MATLAB's function: ``getEarth2014TerrainCoeffs`` returns degree-2160
       spherical-harmonic coefficients from ``.bshc`` files, while pytcl parses
       the 1-arcmin geodetic grid products and has no ``.bshc`` parser.
       Solid-tide shift ported. So of the three public functions, one
       (``solidTideShift``) has a genuine equivalent and both
       coefficient-returning functions -- Earth2014 and EGM2008 terrain --
       are unported.
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
       :doc:`results_io`). ``NMEAChecksum`` is ported too, as
       ``nmea_checksum``; ``decode_ais`` and ``ais_position_reports``
       validate the trailing ``*hh`` by default (``validate_checksum=True``,
       NMEA 4.10 TAG blocks handled). **Missing:**
       NMEA sentence *construction* in either direction (encoding a
       message back to ``!AIVDM`` text) -- neither is ported.
   * - Scheduling
     - 4
     - Absent
     - Interval scheduling algorithms not ported.
   * - Physical_Values
     - 14
     - Partial
     - Physical constants partially ported: 43 module-level constants
       against ``Constants.m``'s 92 ``properties (Constant)``, with the
       WGS72, EGM96/EGM2008 and lunar (GL0900C/JPL) constant families and
       the atomic/electromagnetic set (electron and proton mass, fine
       structure, Rydberg, Faraday) all absent. Radar bands, water
       permittivity, reflection-coefficient models, ``elementAMU`` and
       ``gasProp`` absent.

What pytcl has that the MATLAB library does not
-----------------------------------------------

The comparison runs both ways. pytcl adds: the standard OSPA metric and CLEAR-MOT evaluation,
ionospheric delay models, R-trees and cover trees, DBSCAN and hierarchical
clustering, min-cost-flow assignment, SQL and HDF5 track storage with
migration tooling, dual-backend GPU acceleration, and a test suite of 8,000+
cases that includes 49 validation files checking against independent
references — the MATLAB library distributes no test suite at all.

Honest bottom line
------------------

The defensible claim is not "full feature parity". It is: **the core tracking
workflow is ported, oracle-validated, and in places extended; the MATLAB
library's long tail — its cubature collection, refraction suite, time-scale
zoo, localization estimators and specialized coordinate systems — is
substantially unported, at roughly a third of the full public surface by
function count.** Areas above marked Absent, Weak or Divergent are the honest
priority list for anyone who needs them.
