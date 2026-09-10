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
   * - ``BesselEpoch2TDB``
     - ``pytcl.astronomical.time_scales.besselian_epoch2tdb``
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
   * - ``HessianChainRule``
     - ``pytcl.coordinate_systems.hessians.measurement_hessians.hessian_chain_rule``
   * - ``HessianOfAffineTransFun``
     - ``pytcl.coordinate_systems.hessians.measurement_hessians.hessian_of_affine_trans_fun``
   * - ``CartCD2ITRS``
     - ``pytcl.magnetism.coordinates.cart_cd2itrs``
   * - ``ITRS2CartCD``
     - ``pytcl.magnetism.coordinates.itrs2cart_cd``
   * - ``ITRS2MagneticApex``
     - ``pytcl.magnetism.coordinates.itrs2magnetic_apex``
   * - ``ITRS2QD``
     - ``pytcl.magnetism.coordinates.itrs2qd``
   * - ``JulDate2JulEpoch``
     - ``pytcl.astronomical.time_scales.jul_date2jul_epoch``
   * - ``JulEpoch2JulDate``
     - ``pytcl.astronomical.time_scales.jul_epoch2jul_date``
   * - ``MMOSPA2Tar2D``
     - ``pytcl.performance_evaluation.mospa.mmospa2tar_2d``
   * - ``MMOSPAApprox``
     - ``pytcl.performance_evaluation.mospa.mmospa_approx``
   * - ``NED2ECEF``
     - ``pytcl.coordinate_systems.conversions.ned2ecef``
   * - ``NRLMSISE00Alt4Pres``
     - ``pytcl.atmosphere.nrlmsise00.nrlmsise00_pressure_altitude``
   * - ``NRLMSISE00GasTemp``
     - ``pytcl.atmosphere.nrlmsise00.nrlmsise00_gas_temp``
   * - ``TAI2TT``
     - ``pytcl.astronomical.time_systems.tai_to_tt``
   * - ``TDOAGradient``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.tdoa_gradient``
   * - ``TCB2TDB``
     - ``pytcl.astronomical.time_scales.tcb2tdb``
   * - ``TCG2TT``
     - ``pytcl.astronomical.time_scales.tcg2tt``
   * - ``TDB2BesselEpoch``
     - ``pytcl.astronomical.time_scales.tdb2besselian_epoch``
   * - ``TDB2TCB``
     - ``pytcl.astronomical.time_scales.tdb2tcb``
   * - ``TDB2TT``
     - ``pytcl.astronomical.time_scales.tdb2tt``
   * - ``TT2GAST``
     - ``pytcl.astronomical.time_scales.tt2gast``
   * - ``TT2GMST``
     - ``pytcl.astronomical.time_scales.tt2gmst``
   * - ``TT2LAST``
     - ``pytcl.astronomical.time_scales.tt2last``
   * - ``TT2LMST``
     - ``pytcl.astronomical.time_scales.tt2lmst``
   * - ``TT2TCG``
     - ``pytcl.astronomical.time_scales.tt2tcg``
   * - ``TT2TDB``
     - ``pytcl.astronomical.time_scales.tt2tdb``
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
     - ``pytcl.coordinate_systems.conversions.spherical.ruv2cart`` (aligned
       monostatic) / ``pytcl.coordinate_systems.conversions.uv.ruv2cart_bistatic``
       (full bistatic)
   * - ``Cart2Ruv``
     - ``pytcl.coordinate_systems.conversions.uv.cart2ruv_bistatic``
   * - ``ruv2Ruv``
     - ``pytcl.coordinate_systems.conversions.uv.ruv2ruv``
   * - ``uv2SpherAng``
     - ``pytcl.coordinate_systems.conversions.uv.uv2spher_ang``
   * - ``spherAng2Uv``
     - ``pytcl.coordinate_systems.conversions.uv.spher_ang2uv``
   * - ``stateRuv2Cart``
     - ``pytcl.coordinate_systems.conversions.uv.state_ruv2cart``
   * - ``cameraCoords2UVCoords``
     - ``pytcl.coordinate_systems.conversions.uv.camera_coords2uv``

Dynamic Estimation
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - MATLAB
     - pytcl
   * - ``EKFUpdate``
     - ``pytcl.dynamic_estimation.kalman.extended.ekf_update``
   * - ``EnKFDiscPred``
     - ``pytcl.dynamic_estimation.kalman.ensemble.enkf_predict``
   * - ``EnKFUpdate``
     - ``pytcl.dynamic_estimation.kalman.ensemble.enkf_update``
   * - ``ESRIFDiscPred``
     - ``pytcl.dynamic_estimation.information_filter.esrif_predict``
   * - ``ESRIFUpdate``
     - ``pytcl.dynamic_estimation.information_filter.esrif_update``
   * - ``discQMCKalPred``
     - ``pytcl.dynamic_estimation.kalman.qmc.qmc_kf_predict``
   * - ``QMCKalUpdate``
     - ``pytcl.dynamic_estimation.kalman.qmc.qmc_kf_update``
   * - ``QMCKalMeasPred``
     - ``pytcl.dynamic_estimation.kalman.qmc.qmc_kf_meas_pred``
   * - ``QMCKalUpdateWithPred``
     - ``pytcl.dynamic_estimation.kalman.qmc.qmc_kf_update_with_pred``
   * - ``calcQMCKalmanGain``
     - ``pytcl.dynamic_estimation.kalman.qmc.calc_qmc_kalman_gain``
   * - ``BLUEPolarMeasUpdateApprox``
     - ``pytcl.dynamic_estimation.kalman.blue.blue_polar_meas_update``
   * - ``BLUESpherMeasUpdateApprox``
     - ``pytcl.dynamic_estimation.kalman.blue.blue_spher_meas_update``
   * - ``batchLSLinMeasLinDyn``
     - ``pytcl.dynamic_estimation.batch_estimation.batch_ls_lin_meas_lin_dyn``
   * - ``batchLSNonlinMeasLinDyn``
     - ``pytcl.dynamic_estimation.batch_estimation.batch_ls_nonlin_meas_lin_dyn``
   * - ``batchLSNonlinMeasLinDynLM``
     - ``pytcl.dynamic_estimation.batch_estimation.batch_ls_nonlin_meas_lin_dyn_lm``
   * - ``batchLSNonlinMeasNonlinDyn``
     - ``pytcl.dynamic_estimation.batch_estimation.batch_ls_nonlin_meas_nonlin_dyn``
   * - ``batchLSNonlinMeasNonlinDynLM``
     - ``pytcl.dynamic_estimation.batch_estimation.batch_ls_nonlin_meas_nonlin_dyn_lm``
   * - ``twoPointDiffInit``
     - ``pytcl.dynamic_estimation.batch_estimation.two_point_diff_init``
   * - ``HInfinityUpdate``
     - ``pytcl.dynamic_estimation.kalman.h_infinity.hinf_update``
   * - ``KalmanBatchSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.kalman_batch_smoother``
   * - ``FPInfoBatchSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.fp_info_batch_smoother``
   * - ``EKalmanBatchSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.ekalman_batch_smoother``
   * - ``sqrtCubKalBatchSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.sqrt_cub_kal_batch_smoother``
   * - ``sqrtInfoBatchSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.sqrt_info_batch_smoother``
   * - ``KalmanIntervalSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.kalman_interval_smoother``
   * - ``FPInfoIntervalSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.fp_info_interval_smoother``
   * - ``KalmanFIRSmoother``
     - ``pytcl.dynamic_estimation.batch_smoothers.kalman_fir_smoother``
   * - ``KalmanFIRSmootherCoeffs``
     - ``pytcl.dynamic_estimation.batch_smoothers.kalman_fir_smoother_coeffs``
   * - ``uv2SpherAngCubature``
     - ``pytcl.coordinate_systems.conversions.covariance_conversions.uv2spher_ang_cubature``
   * - ``ruv2RuvCubature``
     - ``pytcl.coordinate_systems.conversions.covariance_conversions.ruv2ruv_cubature``
   * - ``cameraCoords2UVCoordsCubature``
     - ``pytcl.coordinate_systems.conversions.covariance_conversions.camera_coords2uv_cubature``
   * - ``monostatRuv2CartTaylor``
     - ``pytcl.coordinate_systems.conversions.covariance_conversions.monostat_ruv2cart_taylor``
   * - ``speedOfSoundInAir``
     - ``pytcl.atmosphere.models.speed_of_sound_gas_table`` (algorithm 0) /
       ``speed_of_sound_ideal_gas`` (1) / ``speed_of_sound_cramer`` (2)
   * - ``calcCartRRJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_cart_rr_jacob``
   * - ``calcPolarConvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_polar_conv_jacob``
   * - ``calcPolarJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_polar_jacob``
   * - ``calcPolarRRConvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_polar_rr_conv_jacob``
   * - ``calcPolarRRJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_polar_rr_jacob``
   * - ``calcRuvConvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_ruv_conv_jacob``
   * - ``calcRuvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_ruv_jacob``
   * - ``calcRuvRRConvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_ruv_rr_conv_jacob``
   * - ``calcRuvRRJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_ruv_rr_jacob``
   * - ``calcSpherConvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_spher_conv_jacob``
   * - ``calcSpherInvJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_spher_inv_jacob``
   * - ``calcSpherJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_spher_jacob``
   * - ``calcSpherRRJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_spher_rr_jacob``
   * - ``normVecJacob``
     - ``pytcl.coordinate_systems.jacobians.measurement_jacobians.norm_vec_jacob``
   * - ``polAngGradient``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.pol_ang_gradient``
   * - ``rangeGradient``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.range_gradient``
   * - ``rangeRateGradient``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.range_rate_gradient``
   * - ``spherAngGradient``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.spher_ang_gradient``
   * - ``uGradient2D``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.u_gradient_2d``
   * - ``uGradient3D``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.u_gradient_3d``
   * - ``uvGradient``
     - ``pytcl.coordinate_systems.jacobians.component_gradients.uv_gradient``
   * - ``calcSpherConvHessian``
     - ``pytcl.coordinate_systems.hessians.measurement_hessians.calc_spher_conv_hessian``
   * - ``calcSpherHessian``
     - ``pytcl.coordinate_systems.hessians.measurement_hessians.calc_spher_hessian``
   * - ``calcSpherInvHessian``
     - ``pytcl.coordinate_systems.hessians.measurement_hessians.calc_spher_inv_hessian``
   * - ``polarU2DCrossGrad``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.polar_u_2d_cross_grad``
   * - ``polarU2DCrossHessian``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.polar_u_2d_cross_hessian``
   * - ``rangeHessian``
     - ``pytcl.coordinate_systems.hessians.component_hessians.range_hessian``
   * - ``spherAngHessian``
     - ``pytcl.coordinate_systems.hessians.component_hessians.spher_ang_hessian``
   * - ``spherAngUvCrossGrad``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.spher_ang_uv_cross_grad``
   * - ``spherAngUvCrossHessian``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.spher_ang_uv_cross_hessian``
   * - ``uHessian2D``
     - ``pytcl.coordinate_systems.hessians.component_hessians.u_hessian_2d``
   * - ``uHessian3D``
     - ``pytcl.coordinate_systems.hessians.component_hessians.u_hessian_3d``
   * - ``uPolar2DCrossGrad``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.u_polar_2d_cross_grad``
   * - ``uPolar2DCrossHessian``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.u_polar_2d_cross_hessian``
   * - ``uvHessian``
     - ``pytcl.coordinate_systems.hessians.component_hessians.uv_hessian``
   * - ``uvSpherAngCrossGrad``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.uv_spher_ang_cross_grad``
   * - ``uvSpherAngCrossHessian``
     - ``pytcl.coordinate_systems.hessians.cross_derivatives.uv_spher_ang_cross_hessian``
   * - ``calcMOSPAError``
     - ``pytcl.performance_evaluation.mospa.calc_mospa_error``
   * - ``partitionIntervals``
     - ``pytcl.scheduling.partition_intervals``
   * - ``scheduleIntervals``
     - ``pytcl.scheduling.schedule_intervals``
   * - ``scheduleMinLatenessDense``
     - ``pytcl.scheduling.schedule_min_lateness_dense``
   * - ``scheduleWeightedIntervals``
     - ``pytcl.scheduling.schedule_weighted_intervals``
   * - ``geogHeading2Mag``
     - ``pytcl.magnetism.coordinates.geog_heading2mag``
   * - ``magHeading2Geog``
     - ``pytcl.magnetism.coordinates.mag_heading2geog``
   * - ``spherCD2SpherITRS``
     - ``pytcl.magnetism.coordinates.spher_cd2spher_itrs``
   * - ``spherITRS2SpherCD``
     - ``pytcl.magnetism.coordinates.spher_itrs2spher_cd``
   * - ``trace2EarthMagApex``
     - ``pytcl.magnetism.coordinates.trace2earth_mag_apex``
   * - ``RiccatiPredNoClutter``
     - ``pytcl.dynamic_estimation.performance_prediction.riccati_pred_no_clutter``
   * - ``RiccatiPostNoClutter``
     - ``pytcl.dynamic_estimation.performance_prediction.riccati_post_no_clutter``
   * - ``FIMPredNoClutter``
     - ``pytcl.dynamic_estimation.performance_prediction.fim_pred_no_clutter``
   * - ``FIMPostNoClutter``
     - ``pytcl.dynamic_estimation.performance_prediction.fim_post_no_clutter``
   * - ``PCRLBPredAdd``
     - ``pytcl.dynamic_estimation.performance_prediction.pcrlb_pred_add``
   * - ``PCRLBUpdateAddNoClutter``
     - ``pytcl.dynamic_estimation.performance_prediction.pcrlb_update_add_no_clutter``
   * - ``correctAssocProbApprox``
     - ``pytcl.dynamic_estimation.performance_prediction.correct_assoc_prob_approx``
   * - ``trackPurityLinApprox``
     - ``pytcl.dynamic_estimation.performance_prediction.track_purity_lin_approx``
   * - ``linTargetIsUntrackable``
     - ``pytcl.dynamic_estimation.performance_prediction.lin_target_is_untrackable``
   * - ``DiscPriorPModel``
     - ``pytcl.dynamic_estimation.performance_prediction.disc_prior_p_model``
   * - ``sqrtDiscCubKalPred``
     - ``pytcl.dynamic_estimation.kalman.sqrt_cubature.sqrt_ckf_predict``
   * - ``sqrtCubKalUpdate``
     - ``pytcl.dynamic_estimation.kalman.sqrt_cubature.sqrt_ckf_update``
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
   * - ``directionOnlyStaticLocEst``
     - ``pytcl.static_estimation.localization.direction_only_static_loc_est``
   * - ``computePolyMeasFIM``
     - ``pytcl.static_estimation.localization.poly_meas_fim``
