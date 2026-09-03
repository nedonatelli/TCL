MATLAB-to-pytcl migration map
=============================

The explicit function mappings and calling-convention differences between the
MATLAB `Tracker Component Library
<https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary>`_ and
pytcl. Companion to :doc:`matlab_parity_inventory`, which says *what* is
ported; this says *how to call it*.

Every pytcl target in the tables below was resolved by import when the tables
were generated — none is asserted from memory. Signature examples are quoted
from the sources of both libraries.

Calling conventions
-------------------

**Naming.** MATLAB ``camelCase`` becomes ``snake_case``. Families rename
systematically: ``Kalman*`` → ``kf_*``, ``disc*Pred`` → ``*_predict``,
``*Update`` → ``*_update``, ``calc*``/``get*`` prefixes are dropped, and the
MATLAB distribution-class suffix ``D`` disappears (``GaussianD`` →
``Gaussian``).

**Argument order is not preserved — check every call.** The canonical trap:

.. code-block:: text

   MATLAB:  [xUpdate,PUpdate,innov,Pzz,W] = KalmanUpdate(xPred,PPred,z,R,H)
   pytcl:   kf_update(x, P, z, H, R) -> KalmanUpdate(x, P, y, S, K, likelihood)

MATLAB puts ``R`` before ``H`` and makes ``H`` optional; pytcl puts ``H``
before ``R`` and requires both. For a square measurement model a transposed
call runs without error and produces garbage.

**Return values.** MATLAB multiple outputs become NamedTuples, with renames:
``innov`` → ``y``, ``Pzz`` → ``S``, ``W`` (gain) → ``K``. Access by field
name; unpacking by position reproduces the MATLAB order only where documented.

**Array layout.** MATLAB states are ``xDim×1`` column vectors and point sets
are one column per point. pytcl filters take 1-D arrays for single states and
``(N, dim)`` row-per-item arrays for batches (including the GPU batch API).
The coordinate-conversion functions are the exception: they accept the
MATLAB-style ``(3, n)`` column layout directly, plus ``(3,)`` and ``(n, 3)``
with automatic transposition.

**Indexing and sentinels.** MATLAB is 1-based and marks unassigned rows with
``0`` in its assignment vectors (``col4row``). pytcl is 0-based and returns
explicit pairs plus explicit absence:

.. code-block:: text

   MATLAB:  col4row = [1; 2; 0]          % row 3 unassigned
   pytcl:   Assignment2DResult(row_indices=[0, 1], col_indices=[0, 1],
                               cost=..., unassigned_rows=[2], unassigned_cols=[])

**Units.** Both libraries use radians and SI units at API boundaries. No
degree/radian conversion is needed when porting call sites.

**Time arguments.** MATLAB TCL passes two-part Julian dates ``(Jul1, Jul2)``
for precision; pytcl time functions take a single float Julian date. Expect
~4e-5 s quantization at contemporary epochs, which matters only for
sub-millisecond timing work.

**Optional arguments.** MATLAB skips optionals positionally with ``[]``;
pytcl uses keywords. MATLAB variant selectors that were separate arguments
(``systemType`` integers) become string keywords (``system_type='az-el'``),
and MATLAB features absent from a pytcl signature — e.g. the bistatic
``zTx``/``zRx`` arguments of ``Cart2Sphere`` — are unported, not renamed.

Function mappings
-----------------

Curated mappings cover the systematic renames; same-name matches are listed
for completeness. Absence from these tables means no counterpart exists — see
:doc:`matlab_parity_inventory` for the per-area accounting of what is
unported.

Assignment Algorithms
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``assign2D``
     - ``pytcl.assignment_algorithms.two_dimensional.assign2d``
   * - ``assign2DHungarian``
     - ``pytcl.assignment_algorithms.two_dimensional.hungarian``
   * - ``assign3D``
     - ``pytcl.assignment_algorithms.three_dimensional.assign3d``
   * - ``calcSetJPDAUpdate``
     - ``pytcl.assignment_algorithms.jpda.jpda_update``
   * - ``kBest2DAssign``
     - ``pytcl.assignment_algorithms.two_dimensional.murty``

Astronomical Code
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``orbEls2State``
     - ``pytcl.astronomical.orbital_mechanics.orbital_elements_to_state``
   * - ``propagateOrbitKepler``
     - ``pytcl.astronomical.orbital_mechanics.kepler_propagate``
   * - ``propagateOrbitSGP4``
     - ``pytcl.astronomical.sgp4.sgp4_propagate``
   * - ``readJPLEphem``
     - ``pytcl.astronomical.ephemerides.DEEphemeris``
   * - ``solveKeplersEq``
     - ``pytcl.astronomical.orbital_mechanics.mean_to_eccentric_anomaly``
   * - ``state2OrbEls``
     - ``pytcl.astronomical.orbital_mechanics.state_to_orbital_elements``

Clustering and Mixture Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``RunnalsGaussMixRed``
     - ``pytcl.clustering.gaussian_mixture.reduce_mixture_runnalls``
   * - ``WestGaussReduction``
     - ``pytcl.clustering.gaussian_mixture.reduce_mixture_west``
   * - ``kMeanspp``
     - ``pytcl.clustering.kmeans.kmeans``

Container Classes
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``ClusterSet``
     - ``pytcl.containers.cluster_set.ClusterSet``
   * - ``kdTree``
     - ``pytcl.containers.kd_tree.KDTree``
   * - ``metricTree``
     - ``pytcl.containers.vptree.VPTree``

Coordinate Systems
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``Cal2UTC``
     - ``pytcl.astronomical.time_systems.cal_to_jd``
   * - ``Cart2Ellipse``
     - ``pytcl.coordinate_systems.conversions.ecef2geodetic``
   * - ``Cart2Pol``
     - ``pytcl.coordinate_systems.conversions.cart2pol``
   * - ``Cart2Ruv``
     - ``pytcl.coordinate_systems.conversions.cart2ruv``
   * - ``Cart2Sphere``
     - ``pytcl.coordinate_systems.conversions.cart2sphere``
   * - ``ECEF2ENU``
     - ``pytcl.coordinate_systems.conversions.ecef2enu``
   * - ``ECEF2NED``
     - ``pytcl.coordinate_systems.conversions.ecef2ned``
   * - ``ENU2ECEF``
     - ``pytcl.coordinate_systems.conversions.enu2ecef``
   * - ``NED2ECEF``
     - ``pytcl.coordinate_systems.conversions.ned2ecef``
   * - ``TAI2TT``
     - ``pytcl.astronomical.time_systems.tai_to_tt``
   * - ``TT2GAST``
     - ``pytcl.astronomical.time_systems.gast``
   * - ``TT2GMST``
     - ``pytcl.astronomical.time_systems.gmst``
   * - ``UKFUpdate``
     - ``pytcl.dynamic_estimation.kalman.unscented.ukf_update``
   * - ``UTC2TAI``
     - ``pytcl.astronomical.time_systems.utc_to_tai``
   * - ``discUKFPred``
     - ``pytcl.dynamic_estimation.kalman.unscented.ukf_predict``
   * - ``ellips2Cart``
     - ``pytcl.coordinate_systems.conversions.geodetic2ecef``
   * - ``findUTMZone``
     - ``pytcl.coordinate_systems.projections.utm_central_meridian``
   * - ``pol2Cart``
     - ``pytcl.coordinate_systems.conversions.spherical.pol2cart``
   * - ``quat2RotMat``
     - ``pytcl.coordinate_systems.rotations.rotations.quat2rotmat``
   * - ``rotMat2Quat``
     - ``pytcl.coordinate_systems.rotations.rotations.rotmat2quat``
   * - ``ruv2Cart``
     - ``pytcl.coordinate_systems.conversions.spherical.ruv2cart``

Dynamic Estimation
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``EKFUpdate``
     - ``pytcl.dynamic_estimation.kalman.extended.ekf_update``
   * - ``HInfinityUpdate``
     - ``pytcl.dynamic_estimation.kalman.h_infinity.hinf_update``
   * - ``KalmanBatchSmoother``
     - ``pytcl.dynamic_estimation.smoothers.rts_smoother``
   * - ``KalmanIntervalSmoother``
     - ``pytcl.dynamic_estimation.smoothers.fixed_interval_smoother``
   * - ``KalmanUpdate``
     - ``pytcl.dynamic_estimation.kalman.linear.kf_update``
   * - ``cubKalUpdate``
     - ``pytcl.dynamic_estimation.kalman.unscented.ckf_update``
   * - ``discCubKalPred``
     - ``pytcl.dynamic_estimation.kalman.unscented.ckf_predict``
   * - ``discEKFPred``
     - ``pytcl.dynamic_estimation.kalman.extended.ekf_predict``
   * - ``discKalPred``
     - ``pytcl.dynamic_estimation.kalman.linear.kf_predict``
   * - ``infoFilterDiscPred``
     - ``pytcl.dynamic_estimation.kalman.linear.information_filter_predict``
   * - ``infoFilterUpdate``
     - ``pytcl.dynamic_estimation.kalman.linear.information_filter_update``
   * - ``multipleModelPred``
     - ``pytcl.dynamic_estimation.imm.imm_predict``
   * - ``multipleModelUpdate``
     - ``pytcl.dynamic_estimation.imm.imm_update``
   * - ``sqrtDiscKalPred``
     - ``pytcl.dynamic_estimation.kalman.square_root.srkf_predict``
   * - ``sqrtKalmanUpdate``
     - ``pytcl.dynamic_estimation.kalman.square_root.srkf_update``

Dynamic Models
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``FCoordTurn2D``
     - ``pytcl.dynamic_models.discrete_time.coordinated_turn.f_coord_turn_2d``
   * - ``FGaussMarkov``
     - ``pytcl.dynamic_models.discrete_time.singer.f_singer`` -- **order 2
       only.** MATLAB's is arbitrary-order; orders 0 (Ornstein-Uhlenbeck)
       and 1 (integrated OU) have no pytcl counterpart.
   * - ``FPolyKal``
     - ``pytcl.dynamic_models.discrete_time.polynomial.f_poly_kal``
       (arbitrary ``order``, matching MATLAB; ``f_constant_velocity`` is
       the ``order=1`` case)
   * - ``QCoordTurn``
     - ``pytcl.dynamic_models.process_noise.q_coord_turn_2d``
   * - ``QGaussMarkov``
     - ``pytcl.dynamic_models.process_noise.q_singer`` -- **order 2 only**,
       same limitation as ``FGaussMarkov`` above.
   * - ``QPolyKal``
     - ``pytcl.dynamic_models.process_noise.q_poly_kal`` (arbitrary
       ``order``, matching MATLAB; ``q_constant_velocity`` is the
       ``order=1`` case)

Gravity
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``getEGMGeoidHeight``
     - ``pytcl.gravity.egm.geoid_height``
   * - ``gravSolidTideOffset``
     - ``pytcl.gravity.tides.solid_earth_tide_displacement``

Magnetism
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``getIGRFCoeffs``
     - ``pytcl.magnetism.igrf.create_igrf14_coefficients``
   * - ``getWMMCoeffs``
     - ``pytcl.magnetism.wmm.wmm``

Mathematical Functions
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``BellNumber``
     - ``pytcl.mathematical_functions.combinatorics.combinatorics.bell_number``
   * - ``BetaD``
     - ``pytcl.mathematical_functions.statistics.distributions.Beta``
   * - ``CatalanNumber``
     - ``pytcl.mathematical_functions.combinatorics.combinatorics.catalan_number``
   * - ``ChiSquareD``
     - ``pytcl.mathematical_functions.statistics.distributions.ChiSquared``
   * - ``Debye``
     - ``pytcl.mathematical_functions.special_functions.debye.debye``
   * - ``ExponentialD``
     - ``pytcl.mathematical_functions.statistics.distributions.Exponential``
   * - ``GammaD``
     - ``pytcl.mathematical_functions.statistics.distributions.Gamma``
   * - ``GaussianD``
     - ``pytcl.mathematical_functions.statistics.distributions.Gaussian``
   * - ``GaussianMixtureD``
     - ``pytcl.clustering.gaussian_mixture.GaussianMixture``
   * - ``MarcumQ``
     - ``pytcl.mathematical_functions.special_functions.marcum_q.marcum_q``
   * - ``PoissonD``
     - ``pytcl.mathematical_functions.statistics.distributions.Poisson``
   * - ``StudentTD``
     - ``pytcl.mathematical_functions.statistics.distributions.StudentT``
   * - ``UniformD``
     - ``pytcl.mathematical_functions.statistics.distributions.Uniform``
   * - ``VonMisesD``
     - ``pytcl.mathematical_functions.statistics.distributions.VonMises``
   * - ``WishartD``
     - ``pytcl.mathematical_functions.statistics.distributions.Wishart``
   * - ``cholSemiDef``
     - ``pytcl.mathematical_functions.basic_matrix.decompositions.chol_semi_def``
   * - ``commutationMatrix``
     - ``pytcl.mathematical_functions.basic_matrix.special_matrices.commutation_matrix``
   * - ``duplicationMatrix``
     - ``pytcl.mathematical_functions.basic_matrix.special_matrices.duplication_matrix``
   * - ``eliminationMatrix``
     - ``pytcl.mathematical_functions.basic_matrix.special_matrices.elimination_matrix``
   * - ``erfI``
     - ``pytcl.mathematical_functions.special_functions.error_functions.erfi``
   * - ``fallingFactorial``
     - ``pytcl.mathematical_functions.special_functions.hypergeometric.falling_factorial``
   * - ``getNextPermutation``
     - ``pytcl.mathematical_functions.combinatorics.next_permutation``
   * - ``nullspace``
     - ``pytcl.mathematical_functions.basic_matrix.decompositions.null_space``
   * - ``perm``
     - ``pytcl.mathematical_functions.special_functions.gamma_functions.perm``
   * - ``polyRootsMultiDim``
     - ``pytcl.mathematical_functions.polynomials.poly_roots_multi_dim``
   * - ``polygamma``
     - ``pytcl.mathematical_functions.special_functions.gamma_functions.polygamma``
   * - ``spherHarmonicEval``
     - ``pytcl.gravity.spherical_harmonics.spherical_harmonic_sum``
   * - ``subfactorial``
     - ``pytcl.mathematical_functions.combinatorics.combinatorics.subfactorial``
   * - ``totalLeastSquares``
     - ``pytcl.static_estimation.least_squares.total_least_squares``
   * - ``tria``
     - ``pytcl.mathematical_functions.basic_matrix.decompositions.tria``
   * - ``triangleArea``
     - ``pytcl.mathematical_functions.geometry.geometry.triangle_area``
   * - ``vec``
     - ``pytcl.core.array_utils.vec``

Navigation
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``directGeodeticProb``
     - ``pytcl.navigation.geodesy.direct_geodetic``
   * - ``directRhumbProblem``
     - ``pytcl.navigation.rhumb.direct_rhumb``
   * - ``directRhumbSpherProblem``
     - ``pytcl.navigation.rhumb.direct_rhumb_spherical``
   * - ``greatCircleAzimuth``
     - ``pytcl.navigation.great_circle.great_circle_inverse``
   * - ``greatCircleDistance``
     - ``pytcl.navigation.great_circle.great_circle_distance``
   * - ``greatCircleIntersect``
     - ``pytcl.navigation.great_circle.great_circle_intersect``
   * - ``greatCircleTDOALoc``
     - ``pytcl.navigation.great_circle.great_circle_tdoa_loc``
   * - ``indirectGeodeticProb``
     - ``pytcl.navigation.geodesy.inverse_geodetic``
   * - ``indirectRhumbProblem``
     - ``pytcl.navigation.rhumb.indirect_rhumb``
   * - ``indirectRhumbSpherProblem``
     - ``pytcl.navigation.rhumb.indirect_rhumb_spherical``
   * - ``rhumbIntersect``
     - ``pytcl.navigation.rhumb.rhumb_intersect``

Performance Evaluation
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``calcNEES``
     - ``pytcl.performance_evaluation.estimation_metrics.nees``
   * - ``calcRMSE``
     - ``pytcl.performance_evaluation.estimation_metrics.rmse``

Static Estimation
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``TDOAOnlyStaticLocEst``
     - ``pytcl.static_estimation.localization.tdoa_only_static_loc_est``
   * - ``rangeOnlyStaticLocEstNP``
     - ``pytcl.static_estimation.localization.range_only_static_loc_est_np``
   * - ``RROnlyStaticVelEst``
     - ``pytcl.static_estimation.localization.rr_only_static_vel_est``
   * - ``getAdHocCartCov``
     - ``pytcl.static_estimation.localization.ad_hoc_cart_cov``
   * - ``TDOA2Cart``
     - ``pytcl.static_estimation.localization.tdoa_to_cart``
   * - ``rangeRate2StaticPos``
     - ``pytcl.static_estimation.localization.range_rate_to_static_pos``
   * - ``rangeRateRatio2StaticPos2D``
     - ``pytcl.static_estimation.localization.range_rate_ratio_to_static_pos_2d``
   * - ``rotAxis2Vec``
     - ``pytcl.coordinate_systems.rotations.rot_axis_to_vec``
