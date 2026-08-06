Information Filters and SRIF
=============================

Information filters represent uncertainty by the **inverse of covariance** rather than covariance itself. This approach offers superior numerical stability, natural handling of weak measurements, and efficient batch processing. This guide covers the information filter and the Square Root Information Filter (SRIF) as implemented in ``pytcl.dynamic_estimation``.

.. contents:: Contents
   :local:
   :depth: 2

When to Use Information Filters
-------------------------------

**Advantages over Kalman Filters:**

1. **Numerical Stability**: Works well with poorly-conditioned systems

   - No matrix inversion of covariance during the update (direct accumulation of information)
   - Square Root Information Filter is more robust than standard Kalman

2. **Weak Measurement Handling**: Graceful degradation when measurements have high uncertainty

   - Kalman gain can become ill-conditioned with high :math:`\mathbf{R}`
   - Information filter naturally handles weak or missing measurements

3. **Batch Processing**: Natural for off-line data processing and smoothing

4. **Decentralized Systems**: Multiple information sources combine trivially (additive information)

5. **Initial Uncertainty**: Easy to handle unknown initial state (zero information)

**When NOT to use:**

- You need real-time performance and computational resources are limited (Kalman is simpler)
- Your system is well-conditioned and measurements are always present (standard Kalman is fine)
- You need fast forward-only filtering of large data streams (Kalman is more standard)

Information Filter Fundamentals
-------------------------------

**Standard Kalman Filter** represents uncertainty as covariance :math:`\mathbf{P}`:

.. math::

   \mathbf{P}_k = E[(\mathbf{x}_k - \hat{\mathbf{x}}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T]

**Information Filter** represents uncertainty by the information matrix:

.. math::

   \mathbf{Y}_k = \mathbf{P}_k^{-1}

and the information vector:

.. math::

   \mathbf{y}_k = \mathbf{P}_k^{-1} \hat{\mathbf{x}}_k = \mathbf{Y}_k \hat{\mathbf{x}}_k

Recover the state estimate via:

.. math::

   \hat{\mathbf{x}}_k = \mathbf{Y}_k^{-1} \mathbf{y}_k

**Information Gain:**

When a measurement :math:`z_k` arrives with covariance :math:`\mathbf{R}_k`, the information gain is:

.. math::

   \Delta \mathbf{Y}_k = \mathbf{H}_k^T \mathbf{R}_k^{-1} \mathbf{H}_k

.. math::

   \Delta \mathbf{y}_k = \mathbf{H}_k^T \mathbf{R}_k^{-1} z_k

The information is updated additively:

.. math::

   \mathbf{Y}_{k|k} = \mathbf{Y}_{k|k-1} + \Delta \mathbf{Y}_k, \qquad
   \mathbf{y}_{k|k} = \mathbf{y}_{k|k-1} + \Delta \mathbf{y}_k

**Key Insight:** Information matrices add directly (unlike covariances which combine via matrix algebra). This makes decentralized fusion trivial.

Working in Information Form
---------------------------

``pytcl`` provides ``state_to_information`` and ``information_to_state`` to convert between the two representations:

.. code-block:: python

    import numpy as np

    from pytcl.dynamic_estimation import (
        information_to_state,
        state_to_information,
    )

    x = np.array([1.0, 0.5])
    P = np.eye(2) * 0.1

    y, Y = state_to_information(x, P)
    print(y)
    # [10.  5.]
    print(Y)
    # [[10.  0.]
    #  [ 0. 10.]]

    x_back, P_back = information_to_state(y, Y)
    print(x_back)
    # [1.  0.5]

Prediction and Update Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prediction step is more complex than in the Kalman filter because it requires inverting the information matrix. For the discrete linear system

.. math::

   \mathbf{x}_{k+1} = \mathbf{F}_k \mathbf{x}_k + \mathbf{w}_k, \quad
   \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)

the predicted information is

.. math::

   \mathbf{Y}_{k+1|k} = [\mathbf{F}_k \mathbf{Y}_{k|k}^{-1} \mathbf{F}_k^T + \mathbf{Q}_k]^{-1}

while the measurement update is purely additive. ``information_filter_predict`` and ``information_filter_update`` implement both steps:

.. code-block:: python

    from pytcl.dynamic_estimation import (
        information_filter_predict,
        information_filter_update,
    )

    # Constant-velocity model, position-only measurement
    dt = 0.1
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])

    y_pred, Y_pred = information_filter_predict(y, Y, F, Q)

    z = np.array([1.07])
    y_upd, Y_upd = information_filter_update(y_pred, Y_pred, z, H, R)

    x_upd, P_upd = information_to_state(y_upd, Y_upd)
    print(x_upd)
    # [1.05363339 0.50032733]
    print(np.sqrt(np.diag(P_upd)))
    # [0.30138795 0.33141565]

Filtering a Measurement Sequence
--------------------------------

``information_filter`` runs the full predict/update recursion over a list of measurements and returns an ``InformationFilterResult`` with both information-form (``y_filt``, ``Y_filt``) and state-form (``x_filt``, ``P_filt``) estimates. Use ``None`` in the measurement list for missing measurements -- a skipped update is simply "no information added".

.. code-block:: python

    from pytcl.dynamic_estimation import information_filter

    # 1D position tracking with constant velocity (truth: pos = 0.1 * k)
    rng = np.random.default_rng(42)
    x0 = np.array([0.0, 1.0])  # pos, vel
    P0 = np.eye(2) * 0.1

    y0, Y0 = state_to_information(x0, P0)

    measurements = [
        np.array([0.1 * (k + 1) + rng.normal(0.0, 0.5)]) for k in range(20)
    ]

    result = information_filter(y0, Y0, measurements, F, Q, H, R)

    for k in (0, 5, 10, 15, 19):
        x_k = result.x_filt[k]
        sig = np.sqrt(np.diag(result.P_filt[k]))
        print(
            f"k={k:2d}  x = [{x_k[0]:6.3f}, {x_k[1]:6.3f}]"
            f"  sigma = [{sig[0]:.3f}, {sig[1]:.3f}]"
        )
    # k= 0  x = [ 0.128,  1.002]  sigma = [0.301, 0.331]
    # k= 5  x = [ 0.411,  0.910]  sigma = [0.291, 0.385]
    # k=10  x = [ 1.018,  0.996]  sigma = [0.306, 0.403]
    # k=15  x = [ 1.654,  1.060]  sigma = [0.313, 0.403]
    # k=19  x = [ 2.056,  1.049]  sigma = [0.315, 0.400]

Unknown Initial State
---------------------

Information filters excel when the initial state is completely unknown: set :math:`\mathbf{Y}_0 = 0` (zero information) and let measurements build up information. The information matrix is rank-deficient until enough independent measurements have constrained every state component:

.. code-block:: python

    y0_unknown = np.zeros(2)
    Y0_unknown = np.zeros((2, 2))  # No prior information at all

    result_u = information_filter(
        y0_unknown, Y0_unknown, measurements, F, Q, H, R
    )

    print(result_u.x_filt[0], np.linalg.matrix_rank(result_u.Y_filt[0]))
    # [0.25235854 0.        ] 1
    print(result_u.x_filt[2], np.linalg.matrix_rank(result_u.Y_filt[2]))
    # [0.41570303 2.11459482] 2

After one position-only measurement the information matrix has rank 1 -- position is constrained but velocity is not. By the third measurement the rank is full and a (still noisy) velocity estimate exists. A covariance-form Kalman filter cannot represent "no prior" exactly; it must approximate it with a large :math:`\mathbf{P}_0`.

Square Root Information Filter (SRIF)
-------------------------------------

Standard information filters invert matrices at each predict step, which can be numerically delicate. The SRIF instead carries the **square root** of the information matrix, :math:`\mathbf{R}` with :math:`\mathbf{R}^T \mathbf{R} = \mathbf{Y}`, and the transformed information vector :math:`\mathbf{r} = \mathbf{R} \hat{\mathbf{x}}`.

The measurement update stacks the prior square root with the whitened measurement and triangularizes by QR decomposition:

.. math::

   \begin{bmatrix} \mathbf{R}_{k|k-1} & \mathbf{r}_{k|k-1} \\
   \mathbf{R}_{z}^{-T} \mathbf{H}_k & \mathbf{R}_{z}^{-T} z_k \end{bmatrix}
   \xrightarrow{\text{QR}}
   \begin{bmatrix} \mathbf{R}_{k|k} & \mathbf{r}_{k|k} \\ 0 & e_k \end{bmatrix}

where :math:`\mathbf{R}_z` is a Cholesky factor of the measurement noise covariance. No information matrix is ever formed explicitly during the update, and the effective condition number is the square root of the full-matrix version.

``pytcl`` ships ``srif_predict``, ``srif_update``, and the batch driver ``srif_filter`` (returning an ``SRIFResult`` with ``r_filt``, ``R_filt``, ``x_filt``, ``P_filt``). Note that the current ``srif_predict`` implementation propagates through covariance space rather than triangularizing the stacked square-root arrays directly -- see its docstring for details.

.. code-block:: python

    from pytcl.dynamic_estimation import srif_filter

    # SRIF initial condition: R0.T @ R0 = inv(P0), r0 = R0 @ x0
    R0 = np.linalg.cholesky(np.linalg.inv(P0)).T
    r0 = R0 @ x0

    result_srif = srif_filter(r0, R0, measurements, F, Q, H, R)

    diff = max(
        np.max(np.abs(a - b))
        for a, b in zip(result_srif.x_filt, result.x_filt)
    )
    print(f"max |x_srif - x_if| = {diff:.2e}")
    # max |x_srif - x_if| = 1.33e-15

    print(result_srif.x_filt[-1])
    # [2.05631183 1.04852902]

On this well-conditioned toy problem both filters agree to machine precision; the SRIF's advantage appears on poorly-conditioned problems (large dynamic range in state uncertainties, nearly-singular measurement geometry).

**Related square-root variants.** ``pytcl.dynamic_estimation`` also provides square-root and factorized forms of the *covariance* filter: ``srkf_predict`` / ``srkf_update`` / ``srkf_predict_update`` (square-root Kalman filter propagating a Cholesky factor of :math:`\mathbf{P}`) and ``ud_factorize`` / ``ud_predict`` / ``ud_update`` / ``ud_reconstruct`` (Bierman-Thornton U-D factorization). These serve the same numerical-robustness goal from the covariance side.

Multi-Sensor Fusion
-------------------

Because information is additive, independent sensors can be processed one at a time with ``information_filter_update`` -- each scalar sensor contributes :math:`\mathbf{h}^T z / r` to the information vector and :math:`\mathbf{h}^T \mathbf{h} / r` to the information matrix:

.. code-block:: python

    # 2D localization with 3 independent scalar sensors
    x0_fusion = np.array([0.0, 0.0])  # [x, y]
    P0_fusion = np.eye(2) * 10.0

    y_cur, Y_cur = state_to_information(x0_fusion, P0_fusion)

    sensors = [
        # (measurement, measurement row, variance)
        (1.1, np.array([[1.0, 0.0]]), 0.5),  # x position
        (0.9, np.array([[0.0, 1.0]]), 0.5),  # y position
        (1.4, np.array([[1.0, 1.0]]) / np.sqrt(2.0), 0.8),  # diagonal range
    ]

    for z_s, H_s, r_s in sensors:
        y_cur, Y_cur = information_filter_update(
            y_cur, Y_cur, np.array([z_s]), H_s, np.array([[r_s]])
        )

    x_est, P_est = information_to_state(y_cur, Y_cur)
    print(x_est)
    # [1.06163716 0.87116097]
    print(np.sqrt(np.diag(P_est)))
    # [0.62237366 0.62237366]

For decentralized architectures where each node ships its information contribution to a fusion center, ``fuse_information`` adds a list of ``InformationState`` tuples in one call:

.. code-block:: python

    from pytcl.dynamic_estimation import InformationState, fuse_information

    y_prior, Y_prior = state_to_information(x0_fusion, P0_fusion)

    contributions = [InformationState(y=y_prior, Y=Y_prior)]
    for z_s, H_s, r_s in sensors:
        contributions.append(
            InformationState(
                y=H_s.T @ np.array([z_s]) / r_s,
                Y=H_s.T @ H_s / r_s,
            )
        )

    fused = fuse_information(contributions)
    x_fused, P_fused = information_to_state(fused.y, fused.Y)
    print(x_fused)
    # [1.06163716 0.87116097]

The one-shot fused result is identical to the sequential updates -- addition is order-independent.

Comparison: Information Filter vs Kalman Filter
-----------------------------------------------

**Numerical Behavior:**

+----------------------------------+-----------------------+---------------------+
| Aspect                           | Kalman Filter         | Information Filter  |
+==================================+=======================+=====================+
| State representation             | x-hat, P (covariance) | y, Y (information)  |
+----------------------------------+-----------------------+---------------------+
| Measurement update               | Multiplicative (K)    | Additive (Y, y)     |
+----------------------------------+-----------------------+---------------------+
| Prediction step                  | Direct (P_pred)       | Requires inversion  |
+----------------------------------+-----------------------+---------------------+
| Weak measurement (R -> inf)      | Kalman gain -> 0      | Information -> 0    |
+----------------------------------+-----------------------+---------------------+
| Unknown initial state            | Approximate (large P) | Exact (Y = 0)       |
+----------------------------------+-----------------------+---------------------+
| Numerical stability              | Good                  | Better (SRIF)       |
+----------------------------------+-----------------------+---------------------+
| Batch processing                 | Awkward               | Natural             |
+----------------------------------+-----------------------+---------------------+
| Decentralized fusion             | Complex               | Trivial (add info)  |
+----------------------------------+-----------------------+---------------------+

**When Information Filter Excels:**

1. **Unknown initial state**: Set :math:`\mathbf{Y}_0 = 0` (zero information)
2. **Intermittent measurements**: Missing measurements = no information update (pass ``None``)
3. **Batch smoothing**: Natural representation for all-at-once processing
4. **Multi-source fusion**: Information from all sensors adds directly

Diagnostics
-----------

The information matrix itself is the primary health indicator. Monitor its rank (are all states observable yet?) and condition number (is state recovery numerically safe?):

.. code-block:: python

    Y_final = result.Y_filt[-1]
    rank = np.linalg.matrix_rank(Y_final)
    cond = np.linalg.cond(Y_final)
    print(f"rank {rank} of {Y_final.shape[0]}, condition number {cond:.2f}")
    # rank 2 of 2, condition number 3.49

A rank-deficient :math:`\mathbf{Y}` means some state directions are unconstrained and ``information_to_state`` cannot recover a unique estimate. A large condition number warns that the recovered covariance is unreliable -- switch to the SRIF.

Tuning Guidelines
-----------------

**Choose Information Filter When:**

1. System is poorly conditioned (high aspect ratios in state/measurement spaces)
2. Multiple sensors with different measurement rates/uncertainties
3. Batch processing or offline smoothing
4. Unknown or weak initial state
5. Need numerical robustness

**Parameter Selection:**

- **Weak initial information**: instead of exactly zero, :math:`\mathbf{Y}_0 = \epsilon \mathbf{I}` with :math:`\epsilon = 10^{-6}` to :math:`10^{-8}` keeps early state recovery well-posed
- **SRIF vs Standard IF**: use ``srif_filter`` whenever numerics are critical
- **Regularization**: add :math:`\delta \mathbf{I}` to prevent singularity:

  .. math::

     \mathbf{Y}_k \leftarrow \mathbf{Y}_k + \delta \mathbf{I}, \quad \delta = 10^{-10}

Common Pitfalls
---------------

1. **Singular Information Matrix**: Measurements don't constrain all states

   - *Fix*: Monitor rank, add weak priors, ensure observable states

2. **Numerical Inversion**: Computing :math:`\mathbf{Y}^{-1}` becomes ill-conditioned

   - *Fix*: Use the SRIF or regularize (add a small :math:`\delta \mathbf{I}`)

3. **Unbounded Information Accumulation**: With negligible process noise, information grows without bound

   - *Fix*: Model realistic process noise; monitor condition numbers

4. **Costly Prediction Step**: The predict step requires inversion, unlike the Kalman form

   - *Fix*: Hybrid approach (Kalman forward, information form for fusion/batch steps)

See Also
--------

- :doc:`kalman_filter_tuning` -- Standard filtering approach
- :doc:`smoothing` -- RTS smoother for trajectory refinement
- :doc:`adaptive_filtering` -- Tuning parameters online
- :doc:`troubleshooting` -- Numerical issues diagnosis

**References:**

- Bierman (1977) -- *Factorization Methods for Discrete Sequential Estimation* -- Foundational SRIF work
- Bar-Shalom, Li, Kirubarajan (2001) -- *Estimation with Applications to Tracking and Navigation*
- Kailath, Sayed, Hassibi (2000) -- *Linear Estimation* -- Comprehensive treatment with information filters
- Dyer & McReynolds (1969) -- *Extension of Square Root Filtering to Include Process Noise* -- Original SRIF
- Mutambara (1998) -- *Decentralized Estimation and Control for Multisensor Systems*
