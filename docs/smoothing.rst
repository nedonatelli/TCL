Smoothing Algorithms & Offline Estimation
=========================================

*Guide to batch and fixed-lag smoothing for improved state estimation using forward and backward passes.*

Smoothing differs from filtering: while filters use only past measurements, smoothers use all available measurements (past and future) for better accuracy. Ideal for offline analysis, trajectory refinement, and post-processing. All smoothers on this page live in ``pytcl.dynamic_estimation``.

**Table of Contents:**

- Smoothing Fundamentals
- Rauch-Tung-Striebel (RTS) Smoother
- Fixed-Lag Smoothing
- Two-Filter Approach
- Fixed-Interval Smoothing
- Performance Comparison
- Applications
- Common Issues & Solutions
- Best Practices
- Troubleshooting

Smoothing Fundamentals
----------------------

**Filtering vs Smoothing:**

- **Filter**: :math:`p(x_k | z_{1:k})` -- estimate at time :math:`k` using measurements up to :math:`k`
- **Smoother**: :math:`p(x_k | z_{1:N})` -- estimate at time :math:`k` using all :math:`N` measurements

**Why Smoothing?**

1. Better accuracy: ~30-50% lower RMS error vs filtering
2. No real-time constraint: can use all data
3. Ideal for trajectory analysis and post-processing
4. Symmetric uncertainty (less bias than filtering)

**Trade-off:**

- Filtering: Real-time, lower latency
- Smoothing: Offline only (or delayed, for fixed-lag), requires the measurements after :math:`k`

**Mathematical Concept:**

Smoothing likelihood:

.. math::

   p(x_k | z_{1:N}) \propto p(z_{k+1:N} | x_k) \, p(x_k | z_{1:k})

where :math:`p(x_k | z_{1:k})` is the forward filter estimate and :math:`p(z_{k+1:N} | x_k)` carries the information in the future measurements.

Rauch-Tung-Striebel (RTS) Smoother
----------------------------------

The RTS smoother runs a standard Kalman filter forward, then sweeps backward correcting each estimate with information from the future. The backward recursion is

.. math::

   \mathbf{G}_k = \mathbf{P}_{k|k} \mathbf{F}^T \mathbf{P}_{k+1|k}^{-1}

.. math::

   \hat{\mathbf{x}}_{k|N} = \hat{\mathbf{x}}_{k|k}
   + \mathbf{G}_k (\hat{\mathbf{x}}_{k+1|N} - \hat{\mathbf{x}}_{k+1|k})

.. math::

   \mathbf{P}_{k|N} = \mathbf{P}_{k|k}
   + \mathbf{G}_k (\mathbf{P}_{k+1|N} - \mathbf{P}_{k+1|k}) \mathbf{G}_k^T

``rts_smoother`` performs both passes and returns an ``RTSResult`` named tuple with the smoothed trajectory (``x_smooth``, ``P_smooth``) and, for comparison, the forward-filter results (``x_filt``, ``P_filt``):

.. code-block:: python

    import numpy as np

    from pytcl.dynamic_estimation import rts_smoother

    # 1D constant-velocity model, position-only measurements
    rng = np.random.default_rng(7)
    dt = 0.1
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])

    n_steps = 100
    true_pos = np.linspace(0.0, 10.0, n_steps)
    measurements = [np.array([p + rng.normal(0.0, 1.0)]) for p in true_pos]

    x0 = np.array([measurements[0][0], 0.0])
    P0 = np.eye(2)

    result = rts_smoother(x0, P0, measurements, F, Q, H, R)

    x_filt = np.array(result.x_filt)
    x_smooth = np.array(result.x_smooth)

    rms_filt = np.sqrt(np.mean((x_filt[:, 0] - true_pos) ** 2))
    rms_smooth = np.sqrt(np.mean((x_smooth[:, 0] - true_pos) ** 2))
    print(f"Filter RMS position error:   {rms_filt:.4f}")
    print(f"Smoother RMS position error: {rms_smooth:.4f}")
    print(f"Improvement: {(1 - rms_smooth / rms_filt) * 100:.1f}%")
    # Filter RMS position error:   0.4347
    # Smoother RMS position error: 0.2715
    # Improvement: 37.5%

    mean_var_filt = np.mean([np.trace(P) for P in result.P_filt])
    mean_var_smooth = np.mean([np.trace(P) for P in result.P_smooth])
    print(f"Mean trace(P), filter:   {mean_var_filt:.4f}")
    print(f"Mean trace(P), smoother: {mean_var_smooth:.4f}")
    # Mean trace(P), filter:   0.4381
    # Mean trace(P), smoother: 0.1385

Building the Passes Yourself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you need custom logic between steps (time-varying models, gating, logging), assemble the same algorithm from the single-step primitives: ``kf_predict`` / ``kf_update`` for the forward pass and ``kf_smooth`` for the backward recursion. ``rts_smoother_single_step`` is an equivalent backward step that returns a ``SmoothedState`` named tuple instead of a plain tuple.

.. code-block:: python

    from pytcl.dynamic_estimation import kf_predict, kf_smooth, kf_update

    # Forward pass, storing filtered and predicted quantities
    x, P = x0, P0
    x_filt_list, P_filt_list = [], []
    x_pred_list, P_pred_list = [], []
    for z in measurements:
        pred = kf_predict(x, P, F, Q)
        x_pred_list.append(pred.x)
        P_pred_list.append(pred.P)
        upd = kf_update(pred.x, pred.P, z, H, R)
        x, P = upd.x, upd.P
        x_filt_list.append(x)
        P_filt_list.append(P)

    # Backward pass
    x_s = [None] * n_steps
    P_s = [None] * n_steps
    x_s[-1], P_s[-1] = x_filt_list[-1], P_filt_list[-1]
    for k in range(n_steps - 2, -1, -1):
        x_s[k], P_s[k] = kf_smooth(
            x_filt_list[k], P_filt_list[k],
            x_pred_list[k + 1], P_pred_list[k + 1],
            x_s[k + 1], P_s[k + 1],
            F,
        )

    diff = max(np.max(np.abs(a - b)) for a, b in zip(x_s, result.x_smooth))
    print(f"max difference vs rts_smoother: {diff:.2e}")
    # max difference vs rts_smoother: 0.00e+00

Fixed-Lag Smoothing
-------------------

**Real-Time Smoothing with Bounded Delay**

``fixed_lag_smoother`` provides near-real-time smoothing: at time :math:`k` it outputs the smoothed estimate for time :math:`k - L` using measurements up to :math:`k`, keeping only the last :math:`L` filter results in memory. The returned ``FixedLagResult.x_smooth`` list follows that convention -- entry :math:`k` (for :math:`k \ge L`) is the estimate for time :math:`k - L`; the first :math:`L` entries are plain filtered estimates.

.. code-block:: python

    from pytcl.dynamic_estimation import fixed_lag_smoother

    lag = 5
    result_lag = fixed_lag_smoother(x0, P0, measurements, F, Q, H, R, lag=lag)

    # Align: entry k estimates time k - lag
    est = np.array([result_lag.x_smooth[k][0] for k in range(lag, n_steps)])
    truth = true_pos[: n_steps - lag]

    rms_lag = np.sqrt(np.mean((est - truth) ** 2))
    rms_filt_aligned = np.sqrt(
        np.mean((x_filt[: n_steps - lag, 0] - truth) ** 2)
    )
    print(f"Filter RMS (same interval): {rms_filt_aligned:.4f}")
    print(f"Fixed-lag RMS (lag=5):      {rms_lag:.4f}")
    # Filter RMS (same interval): 0.4416
    # Fixed-lag RMS (lag=5):      0.3408

A lag of 5-10 steps typically recovers a large fraction of the full smoothing gain at a fixed, bounded delay.

Two-Filter Approach
-------------------

**Forward and Backward Filtering**

Alternative to RTS (the Fraser-Potter form): run forward and backward filters independently, then combine their estimates in information form. Useful for parallel implementation. ``two_filter_smoother`` needs an initial condition for the backward filter -- typically a rough guess at the final state with a diffuse (large) covariance:

.. code-block:: python

    from pytcl.dynamic_estimation import two_filter_smoother

    x0_bwd = np.array([true_pos[-1], 1.0])  # rough guess at final state
    P0_bwd = np.eye(2) * 100.0              # diffuse prior

    result_tf = two_filter_smoother(
        x0, P0, x0_bwd, P0_bwd, measurements, F, Q, H, R
    )

    x_tf = np.array(result_tf.x_smooth)
    rms_tf = np.sqrt(np.mean((x_tf[:, 0] - true_pos) ** 2))
    print(f"Two-filter RMS position error: {rms_tf:.4f}")
    # Two-filter RMS position error: 0.2714

Fixed-Interval Smoothing
------------------------

Fixed-interval smoothing is the general problem of smoothing over a complete, fixed data interval; the RTS smoother is its standard solution. ``fixed_interval_smoother`` is provided as an alias with the same signature and ``RTSResult`` return:

.. code-block:: python

    from pytcl.dynamic_estimation import fixed_interval_smoother

    result_fi = fixed_interval_smoother(x0, P0, measurements, F, Q, H, R)
    print(np.allclose(np.array(result_fi.x_smooth), x_smooth))
    # True

Performance Comparison
----------------------

Summarizing the position RMS errors from the runs above (fixed-lag is evaluated on its slightly shorter aligned interval):

.. code-block:: python

    print(f"filter           {rms_filt:.4f}")
    print(f"fixed-lag (L=5)  {rms_lag:.4f}")
    print(f"two-filter       {rms_tf:.4f}")
    print(f"RTS              {rms_smooth:.4f}")
    # filter           0.4347
    # fixed-lag (L=5)  0.3408
    # two-filter       0.2714
    # RTS              0.2715

The ordering is typical: fixed-lag recovers part of the smoothing gain at bounded delay, while the full-interval smoothers (RTS and two-filter, which are algebraically equivalent) do best.

Applications
------------

**1. Post-Mission Analysis**

Analyze a complete recorded flight or mission with batch smoothing:

.. code-block:: python

    def analyze_mission(measurements, x0, P0, F, Q, H, R):
        """Process a complete recorded mission with batch smoothing."""
        result = rts_smoother(x0, P0, measurements, F, Q, H, R)
        std_devs = np.array([np.sqrt(np.diag(P)) for P in result.P_smooth])
        reduction = 1.0 - (
            np.mean([np.trace(P) for P in result.P_smooth])
            / np.mean([np.trace(P) for P in result.P_filt])
        )
        return {
            "trajectory": np.array(result.x_smooth),
            "uncertainties": std_devs,
            "uncertainty_reduction": reduction,
        }

    report = analyze_mission(measurements, x0, P0, F, Q, H, R)
    print(f"Uncertainty reduction: {report['uncertainty_reduction'] * 100:.1f}%")
    # Uncertainty reduction: 68.4%

**2. Multi-Sensor Fusion with Latency**

A delayed measurement that arrives :math:`d` steps late can be incorporated by re-smoothing the affected interval: pass the measurement list with the late value filled in at its true time index and rerun ``rts_smoother`` (or ``fixed_lag_smoother`` with ``lag >= d`` for a streaming system). The smoother handles the time correlation automatically.

**3. GPS/INS Trajectory Refinement**

INS provides a high-rate dead-reckoned trajectory; GPS provides occasional absolute fixes. Model the INS error dynamics as the state, treat GPS fixes as measurements (with ``None`` at epochs without a fix -- ``rts_smoother`` accepts missing measurements), and smooth to distribute each GPS correction backward across the preceding INS-only stretch.

Common Issues & Solutions
-------------------------

**Problem: Backward Pass Diverges**

The RTS gain inverts the predicted covariance :math:`\mathbf{P}_{k+1|k}`. If that matrix is near-singular (near-perfect measurements, degenerate dynamics), the backward pass amplifies noise. Increase process noise :math:`\mathbf{Q}`, or check the conditioning of :math:`\mathbf{P}_{k+1|k}` before smoothing.

**Problem: Backward Covariance Grows**

Indicates the filter was too optimistic (:math:`\mathbf{P}` too small). Solutions:

1. Increase process noise Q
2. Reduce measurement noise R (if overestimated)
3. Use the Joseph form covariance update in a custom forward pass:

.. code-block:: python

    def joseph_covariance_update(P_pred, K, H, R):
        """Joseph stabilized covariance update (numerically robust)."""
        n = P_pred.shape[0]
        I_KH = np.eye(n) - K @ H
        return I_KH @ P_pred @ I_KH.T + K @ R @ K.T

    pred = kf_predict(x0, P0, F, Q)
    upd = kf_update(pred.x, pred.P, measurements[0], H, R)

    P_joseph = joseph_covariance_update(pred.P, upd.K, H, R)
    print(np.allclose(P_joseph, upd.P))
    # True

**Problem: Smoother Solution Violates Constraints**

Smoothed estimates are unconstrained least-squares solutions; if physical constraints (road networks, terrain, speed limits) must hold, project each smoothed state onto the constraint set as a post-processing step, or use a constrained filtering formulation for the forward pass.

Best Practices
--------------

1. **Batch Smoothing** (Offline Analysis)

   - Use ``rts_smoother`` for complete datasets
   - Provides optimal estimates given all data
   - ~30-50% accuracy improvement over filtering

2. **Fixed-Lag** (Near Real-Time)

   - When you need low latency but can wait for the lag
   - Typical lag: 5-10 timesteps
   - Good balance: much of the smoothing gain at a bounded delay

3. **Monitor Consistency**

   - Check that trace(P_smooth) < trace(P_filt) on average (if not, filter issue)
   - Verify the backward pass using simulated data with known truth
   - Compare forward-only vs full smoother

4. **Memory Efficient**

   - ``fixed_lag_smoother`` stores only the last ``lag`` filter results
   - For very long missions, smooth in overlapping batches and discard batch edges

5. **Numerical Stability**

   - Use the Joseph covariance form in custom forward passes
   - Check matrix condition numbers before the backward pass

6. **Parameter Tuning**

   - Validate process/measurement noise with smoothed residuals
   - Residuals should be white noise (check autocorrelation)
   - Use the smoother to detect parameter mismatches

Troubleshooting
---------------

**Diagnostic Checklist:**

.. code-block:: python

    def diagnose_smoother(measurements, result, H):
        """Quality checks on an RTSResult."""
        issues = []

        # Check 1: smoother should beat the filter
        tr_filter = np.mean([np.trace(P) for P in result.P_filt])
        tr_smooth = np.mean([np.trace(P) for P in result.P_smooth])
        if tr_filter / tr_smooth < 1.1:
            issues.append("Minimal improvement - check data quality")

        # Check 2: backward covariance growth
        if np.trace(result.P_smooth[0]) > 10 * np.trace(result.P_smooth[-1]):
            issues.append("Backward covariance grows - check Q/R tuning")

        # Check 3: residual whiteness (lag-1 autocorrelation)
        x_s = np.array(result.x_smooth)
        residuals = np.array(
            [z - H @ x for z, x in zip(measurements, x_s)]
        )
        r0 = residuals[:, 0] - residuals[:, 0].mean()
        acf = np.correlate(r0, r0, mode="full")[len(r0) - 1 :]
        if acf[1] > 0.5 * acf[0]:
            issues.append("Residuals auto-correlated - check model")

        return issues if issues else ["Smoother diagnostics OK"]

    print(diagnose_smoother(measurements, result, H))
    # ['Smoother diagnostics OK']

See Also
--------

- :doc:`recipes` -- Ready-to-use smoothing implementations
- :doc:`kalman_filter_tuning` -- Parameter selection for smoothers
- :doc:`troubleshooting` -- Smoothing issues and debugging
- API: ``pytcl.dynamic_estimation.smoothers`` module (``rts_smoother``, ``fixed_lag_smoother``, ``two_filter_smoother``, ``fixed_interval_smoother``, ``kf_smooth``, ``rts_smoother_single_step``)
