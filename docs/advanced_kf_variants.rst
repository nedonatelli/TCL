Advanced Kalman Filter Variants
===============================

Beyond the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF), advanced variants use sophisticated numerical integration schemes, sigma-point strategies, and ensemble methods to achieve superior accuracy for highly nonlinear systems. This guide covers the Cubature Kalman Filter, sigma-point filters, numerical-Jacobian (central difference) filtering, the Ensemble Kalman Filter, and their practical applications. pytcl ships all of these except the EnKF in ``pytcl.dynamic_estimation``.

.. contents:: Contents
   :local:
   :depth: 3

When to Use Advanced KF Variants
=================================

**Problem Scenarios:**

1. **Highly Nonlinear Systems**: EKF linearization is too coarse

   - Launch vehicle ascent (extreme acceleration changes)
   - Radar tracking in near-field (range-dependent nonlinearity)
   - Atmospheric re-entry (drag coefficient varies drastically)

2. **Non-Gaussian Error Distributions**: Measurements are heavy-tailed

   - *Solution*: Ensemble or particle filter approaches

3. **Ill-Conditioned Jacobians**: Linearization is numerically unstable

   - *Solution*: Cubature (uses numerical integration instead)

4. **Large Computational Budget**: Can afford extra complexity

   - Cubature/Sigma-point: Slightly more expensive than EKF
   - Ensemble: Computationally intensive but parallelizable

5. **High-Dimensional Systems**: Need scalable uncertainty propagation

   - *Solution*: Ensemble Kalman Filter (scales to 1000s of states)

Cubature Kalman Filter (CKF)
=============================

The Cubature Kalman Filter uses **cubature integration rules** to compute transformed mean and covariance through nonlinear functions with high accuracy.

**Key Idea:**

Numerical integration via cubature points (symmetric sampling):

.. math::

   \int_{\mathbb{R}^n} g(\mathbf{x}) \,
   \mathcal{N}(\mathbf{x}; \mathbf{0}, \mathbf{I}) \, d\mathbf{x}
   \approx \frac{1}{2n} \sum_{i=1}^{2n} g(\sqrt{n} \, \boldsymbol{\xi}_i)

where :math:`\boldsymbol{\xi}_i` are the :math:`2n` cubature points.

**Advantages:**

- No Jacobian computation required (derivative-free)
- Third-order numerical accuracy for Gaussian inputs
- Better accuracy than UKF for many nonlinear problems
- Symmetric sampling provides numerical stability

Theory: Spherical Cubature Rule
--------------------------------

For an :math:`n`-dimensional system, use :math:`2n` cubature points on a sphere:

.. math::

   \boldsymbol{\xi}_i = \sqrt{n} \, \mathbf{e}_i, \qquad
   \boldsymbol{\xi}_{i+n} = -\sqrt{n} \, \mathbf{e}_i, \qquad
   i = 1, \ldots, n

where :math:`\mathbf{e}_i` are standard basis vectors.

**Transformed Mean:**

.. math::

   \hat{\mathbf{y}} = \frac{1}{2n} \sum_{i=1}^{2n}
   g(\mathbf{m} + \mathbf{S} \boldsymbol{\xi}_i)

where :math:`\mathbf{P} = \mathbf{S} \mathbf{S}^T` (Cholesky decomposition).

**Transformed Covariance:**

.. math::

   \mathbf{Q}_{yy} = \frac{1}{2n} \sum_{i=1}^{2n}
   \left(g(\mathbf{m} + \mathbf{S} \boldsymbol{\xi}_i) - \hat{\mathbf{y}}\right)
   \left(g(\mathbf{m} + \mathbf{S} \boldsymbol{\xi}_i) - \hat{\mathbf{y}}\right)^T

Using the pytcl Implementation
------------------------------

pytcl ships the CKF as ``ckf_predict`` / ``ckf_update``, plus
``ckf_spherical_cubature_points`` if you want the raw points:

.. code-block:: python

    import numpy as np
    from pytcl.dynamic_estimation import ckf_spherical_cubature_points

    # 2n points, each with weight 1/(2n)
    points, weights = ckf_spherical_cubature_points(2)
    print(points)
    print(weights)
    # [[ 1.41421356  0.        ]
    #  [ 0.          1.41421356]
    #  [-1.41421356  0.        ]
    #  [ 0.         -1.41421356]]
    # [0.25 0.25 0.25 0.25]

``ckf_predict(x, P, f, Q)`` propagates the state through a nonlinear
dynamics function ``f`` and returns a ``KalmanPrediction`` named tuple
``(x, P)``. ``ckf_update(x, P, z, h, R)`` applies a nonlinear measurement
function ``h`` and returns a ``KalmanUpdate`` named tuple
``(x, P, y, S, K, likelihood)``.

**Example: Nonlinear Pendulum Tracking**

.. code-block:: python

    from pytcl.dynamic_estimation import ckf_predict, ckf_update

    def f_pendulum(x, dt=0.1, g=9.81, L=1.0):
        """Nonlinear pendulum dynamics (Euler step; use RK4 in practice)."""
        theta, theta_dot = x
        return np.array([
            theta + theta_dot * dt,
            theta_dot - (g / L) * np.sin(theta) * dt,
        ])

    def h_pendulum(x):
        """Measure sin(angle): nonlinear position sensor on the arc."""
        return np.array([np.sin(x[0])])

    rng = np.random.default_rng(0)
    Q = np.diag([1e-4, 1e-3])
    R = np.array([[0.01]])

    # Simulate the true pendulum and noisy measurements once; the UKF
    # and SR-UKF sections below reuse the same data for comparison
    true_x = np.array([0.5, 0.0])
    truths_pend, zs_pend = [], []
    for _ in range(100):
        true_x = f_pendulum(true_x)
        truths_pend.append(true_x.copy())
        zs_pend.append(h_pendulum(true_x) + rng.normal(0.0, 0.1, size=1))

    # Filter from a deliberately offset initial estimate
    x = np.array([0.3, 0.0])
    P = np.diag([0.1, 0.1])
    for k, z in enumerate(zs_pend):
        pred = ckf_predict(x, P, f_pendulum, Q)
        upd = ckf_update(pred.x, pred.P, z, h_pendulum, R)
        x, P = upd.x, upd.P
        if k % 33 == 0:
            err = x[0] - truths_pend[k][0]
            print(f"step {k:2d}: theta_err={err:+.4f}  "
                  f"sigma_theta={np.sqrt(P[0, 0]):.4f}")
    # step  0: theta_err=+0.0044  sigma_theta=0.1035
    # step 33: theta_err=-0.0304  sigma_theta=0.0719
    # step 66: theta_err=+0.0825  sigma_theta=0.0675
    # step 99: theta_err=-0.0976  sigma_theta=0.0685

Sigma-Point Kalman Filters
===========================

**Unscented Kalman Filter (UKF)** and variants use sigma points (deterministic samples) to represent the probability distribution.

Unscented Transform
-------------------

Given mean :math:`\mathbf{m}` and covariance :math:`\mathbf{P}`, generate :math:`2n+1` sigma points:

.. math::

   \boldsymbol{\sigma}_0 = \mathbf{m}

.. math::

   \boldsymbol{\sigma}_i = \mathbf{m} + \sqrt{n + \kappa} \, \mathbf{S}_i, \qquad
   i = 1, \ldots, n

.. math::

   \boldsymbol{\sigma}_{i+n} = \mathbf{m} - \sqrt{n + \kappa} \, \mathbf{S}_i, \qquad
   i = 1, \ldots, n

where :math:`\mathbf{S}` is the Cholesky decomposition of :math:`\mathbf{P}`, and :math:`\kappa` is a tuning parameter.

**Weights:**

.. math::

   W_0^m = \frac{\kappa}{n + \kappa}, \qquad
   W_0^c = \frac{\kappa}{n + \kappa} + (1 - \alpha^2 + \beta)

.. math::

   W_i^m = W_i^c = \frac{1}{2(n + \kappa)}, \qquad i = 1, \ldots, 2n

pytcl exposes both sigma-point sets and the transform itself:

- ``sigma_points_merwe(x, P, alpha, beta, kappa)``: Van der Merwe's scaled
  points (the modern default)
- ``sigma_points_julier(x, P, kappa)``: Julier's original parameterization
- ``unscented_transform(sigmas, Wm, Wc, noise_cov)``: mean and covariance
  of transformed points

.. code-block:: python

    from pytcl.dynamic_estimation import (
        sigma_points_merwe,
        sigma_points_julier,
        unscented_transform,
    )

    x = np.array([0.3, 0.0])
    P = np.diag([0.1, 0.1])

    sp = sigma_points_merwe(x, P, alpha=1e-3, beta=2.0, kappa=0.0)
    print(sp.points.shape)  # 2n+1 points for n=2
    # (5, 2)

    # Propagate sigma points through the dynamics, then recover the
    # transformed mean and covariance (with process noise added)
    propagated = np.array([f_pendulum(s) for s in sp.points])
    y, Pyy = unscented_transform(propagated, sp.Wm, sp.Wc, noise_cov=Q)
    print(np.round(y, 4))
    # [ 0.3    -0.2754]

    sp_j = sigma_points_julier(x, P, kappa=1.0)
    print(sp_j.points.shape)
    # (5, 2)

**Full UKF cycle:**

``ukf_predict`` / ``ukf_update`` wrap sigma-point generation and the
unscented transform into single predict/update calls with the same
signatures and return types as the CKF:

.. code-block:: python

    from pytcl.dynamic_estimation import ukf_predict, ukf_update

    x = np.array([0.3, 0.0])
    P = np.diag([0.1, 0.1])
    for z in zs_pend:
        pred = ukf_predict(x, P, f_pendulum, Q, alpha=1e-3, beta=2.0, kappa=0.0)
        upd = ukf_update(pred.x, pred.P, z, h_pendulum, R,
                         alpha=1e-3, beta=2.0, kappa=0.0)
        x, P = upd.x, upd.P

    print(f"final theta_err={x[0] - truths_pend[-1][0]:+.4f}")
    # final theta_err=-0.0976

On this mildly nonlinear problem the UKF and CKF agree to about four
decimal places; they diverge on problems with stronger curvature.

Square-Root UKF
---------------

For long-running filters or ill-conditioned covariances, the square-root
form propagates the Cholesky factor :math:`\mathbf{S}` (where
:math:`\mathbf{P} = \mathbf{S}\mathbf{S}^T`) directly, guaranteeing a
positive semi-definite covariance:

.. code-block:: python

    from scipy.linalg import cholesky
    from pytcl.dynamic_estimation import sr_ukf_predict, sr_ukf_update

    x = np.array([0.3, 0.0])
    S = cholesky(np.diag([0.1, 0.1]), lower=True)
    S_Q = cholesky(Q, lower=True)
    S_R = cholesky(R, lower=True)

    for z in zs_pend:
        pred = sr_ukf_predict(x, S, f_pendulum, S_Q)
        upd = sr_ukf_update(pred.x, pred.S, z, h_pendulum, S_R)
        x, S = upd.x, upd.S

    P_sr = S @ S.T  # reconstruct covariance when needed
    print(f"final theta_err={x[0] - truths_pend[-1][0]:+.4f}  "
          f"sigma_theta={np.sqrt(P_sr[0, 0]):.4f}")
    # final theta_err=-0.0976  sigma_theta=0.0684

Central Difference (Numerical-Jacobian) Filtering
=================================================

When the dynamics or measurement functions are only available as code, the
Jacobian can be approximated by **central differences** instead of derived
analytically.

**Key Idea:**

.. math::

   \mathbf{F}[i, j] \approx
   \frac{f_i(\mathbf{x} + \delta \mathbf{e}_j) - f_i(\mathbf{x} - \delta \mathbf{e}_j)}{2\delta}

where :math:`\delta` is the difference step size.

**Advantages:**

- No Jacobian code needed (numerical differentiation)
- Better approximation than forward differences (:math:`O(\delta^2)` vs :math:`O(\delta)`)
- Works for complex or implicit dynamics
- Slightly more expensive than EKF (2n extra function calls per Jacobian)

pytcl implements this as ``numerical_jacobian`` (central differences with
step ``dx``) and wraps the full EKF cycle around it as ``ekf_predict_auto``
and ``ekf_update_auto``. Note that ``ekf_predict_auto`` evaluates the
Jacobian at the *prior* state before propagating, which is the correct
linearization point; a common bug in hand-rolled versions is differentiating
at the already-predicted state.

.. code-block:: python

    from pytcl.dynamic_estimation import numerical_jacobian

    def h_radar(x):
        """Radar measurement [range, range_rate] of state [px, py, vx, vy]."""
        pos, vel = x[:2], x[2:]
        r = np.hypot(pos[0], pos[1])
        return np.array([r, pos @ vel / r])

    x_test = np.array([1000.0, 500.0, 10.0, -5.0])
    H = numerical_jacobian(h_radar, x_test)
    print(np.round(H, 6))
    # [[ 0.894427  0.447214  0.        0.      ]
    #  [ 0.003578 -0.007155  0.894427  0.447214]]

The first row is the unit position vector (the analytical range gradient),
confirming the central-difference approximation.

**Example: Radar Tracking with Automatic Jacobians**

.. code-block:: python

    from pytcl.dynamic_estimation import ekf_predict_auto, ekf_update_auto

    def h_radar_full(x):
        """Radar measurement [range, bearing, range_rate]."""
        pos, vel = x[:2], x[2:]
        r = np.hypot(pos[0], pos[1])
        return np.array([r, np.arctan2(pos[1], pos[0]), pos @ vel / r])

    dt = 0.1
    F = np.array([[1.0, 0.0, dt, 0.0],
                  [0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    def f_cv(x):
        return F @ x

    rng = np.random.default_rng(1)
    Q_cv = np.diag([0.01, 0.01, 0.1, 0.1])
    R_radar = np.diag([25.0, 1e-4, 1.0])  # 5 m, 10 mrad, 1 m/s

    truth = np.array([1000.0, 500.0, -20.0, 5.0])
    x = truth + np.array([50.0, -50.0, 5.0, -2.0])
    P = np.diag([2500.0, 2500.0, 100.0, 100.0])

    print(f"initial position error: {np.hypot(*(x[:2] - truth[:2])):.1f}")
    for k in range(100):
        truth = f_cv(truth)
        z = h_radar_full(truth) + rng.normal(0.0, [5.0, 0.01, 1.0])
        pred = ekf_predict_auto(x, P, f_cv, Q_cv)
        upd = ekf_update_auto(pred.x, pred.P, z, h_radar_full, R_radar)
        x, P = upd.x, upd.P

    print(f"final position error:   {np.hypot(*(x[:2] - truth[:2])):.1f}")
    # initial position error: 70.7
    # final position error:   0.9

Because the Jacobians are computed automatically, adding a measurement
channel (bearing here) only requires changing the measurement function --
there is no derivative code to keep in sync.

Ensemble Kalman Filter (EnKF)
=============================

The Ensemble Kalman Filter represents uncertainty via an ensemble (collection) of state realizations rather than explicit covariance matrices.

.. note::

   pytcl does **not** ship an Ensemble Kalman Filter. The class below is a
   self-contained reference implementation included for completeness; for
   the variants above, use the pytcl functions directly.

**Key Advantages:**

1. **Scalability**: Works efficiently in very high dimensions (1000s-millions of states)
2. **Non-Gaussian Errors**: Naturally handles non-Gaussian distributions
3. **Nonlinearity Handling**: Implicit handling via ensemble propagation
4. **Parallelization**: Each ensemble member can run independently

**Algorithm:**

Given ensemble :math:`\{\mathbf{x}^{(i)}\}_{i=1}^{N}` with :math:`N` members:

1. **Predict**: Propagate each member independently
2. **Update**: Add random perturbations to measurements, update ensemble members

.. code-block:: python

    from scipy.linalg import cholesky

    class EnsembleKalmanFilter:
        """
        Ensemble Kalman Filter (EnKF): reference implementation.

        Represents uncertainty via an ensemble of state realizations.
        Naturally handles high-dimensional systems and nonlinearity.
        """

        def __init__(self, x0, P0, num_members=100, rng=None):
            """
            Parameters
            ----------
            x0 : (n,) array
                Mean state
            P0 : (n, n) array
                Initial covariance
            num_members : int
                Number of ensemble members (typically 50-1000)
            rng : numpy.random.Generator, optional
            """
            self.n = len(x0)
            self.num_members = num_members
            self.rng = rng if rng is not None else np.random.default_rng()

            L = cholesky(P0, lower=True)
            self.ensemble = (
                x0[:, np.newaxis]
                + L @ self.rng.standard_normal((self.n, num_members))
            )

        def get_state(self):
            """Return mean and covariance from ensemble."""
            x_mean = np.mean(self.ensemble, axis=1)

            anomalies = self.ensemble - x_mean[:, np.newaxis]
            P = (anomalies @ anomalies.T) / (self.num_members - 1)
            return x_mean, P

        def predict(self, f_func, Q):
            """Propagate ensemble members and add process noise."""
            for i in range(self.num_members):
                self.ensemble[:, i] = f_func(self.ensemble[:, i])

            L_Q = cholesky(Q, lower=True)
            self.ensemble += L_Q @ self.rng.standard_normal(
                (self.n, self.num_members)
            )

        def update(self, z, h_func, R):
            """Update ensemble via perturbed measurements."""
            m = len(z)

            z_ensemble = np.array([
                h_func(self.ensemble[:, i]) for i in range(self.num_members)
            ]).T
            z_mean = np.mean(z_ensemble, axis=1)
            Z_anom = z_ensemble - z_mean[:, np.newaxis]

            Pzz = (Z_anom @ Z_anom.T) / (self.num_members - 1) + R

            x_mean = np.mean(self.ensemble, axis=1)
            X_anom = self.ensemble - x_mean[:, np.newaxis]
            Pxz = (X_anom @ Z_anom.T) / (self.num_members - 1)

            K = Pxz @ np.linalg.inv(Pzz)

            # Perturbed measurements: one noisy copy per member
            L_R = cholesky(R, lower=True)
            z_pert = z[:, np.newaxis] + L_R @ self.rng.standard_normal(
                (m, self.num_members)
            )

            self.ensemble += K @ (z_pert - z_ensemble)

**Example: Atmospheric Data Assimilation (Simplified)**

.. code-block:: python

    def f_temp_diffusion(x, dt=0.01, diffusion=0.1):
        """Temperature diffusion: dT/dt = alpha * d2T/dx2 (simplified)."""
        x_new = x.copy()
        x_new[1:-1] += diffusion * dt * (x[:-2] - 2 * x[1:-1] + x[2:])
        return x_new

    def h_temp_obs(x):
        """Observe temperature at every 5th grid point."""
        return x[::5]

    n_grid = 50
    idx = np.arange(n_grid)

    # Spatially correlated initial covariance (length scale 3 cells):
    # observing every 5th point then also corrects its neighbors
    P0 = 4.0 * np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / 3.0) ** 2)
    P0 += 1e-6 * np.eye(n_grid)

    x_true = 20.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, n_grid))
    x0 = x_true + cholesky(P0, lower=True) @ \
        np.random.default_rng(3).standard_normal(n_grid)

    enkf = EnsembleKalmanFilter(x0, P0, num_members=100,
                                rng=np.random.default_rng(4))

    Q_grid = np.eye(n_grid) * 1e-4
    R_obs = np.eye(10) * 0.5
    obs_rng = np.random.default_rng(5)

    print(f"prior RMSE: {np.sqrt(np.mean((x0 - x_true) ** 2)):.3f}")
    for k in range(100):
        x_true = f_temp_diffusion(x_true)
        enkf.predict(f_temp_diffusion, Q_grid)

        z = h_temp_obs(x_true) + np.sqrt(0.5) * obs_rng.standard_normal(10)
        enkf.update(z, h_temp_obs, R_obs)

        if k % 33 == 0:
            x_est, P_est = enkf.get_state()
            rmse = np.sqrt(np.mean((x_est - x_true) ** 2))
            print(f"step {k:2d}: RMSE={rmse:.3f}  "
                  f"mean var={np.mean(np.diag(P_est)):.3f}")
    # prior RMSE: 1.777
    # step  0: RMSE=0.908  mean var=0.793
    # step 33: RMSE=0.997  mean var=0.312
    # step 66: RMSE=0.966  mean var=0.284
    # step 99: RMSE=0.968  mean var=0.266

The first assimilation cycle halves the error. Note the classic EnKF
caveat visible in the output: the ensemble variance keeps shrinking while
the actual error plateaus, i.e. the ensemble slowly becomes overconfident.
Production EnKF systems counter this with covariance inflation (see Common
Pitfalls below).

Comparison: Advanced KF Variants
=================================

**Accuracy and Computational Cost:**

+--------------------------------+----------+-----------+-----------+-----------+
| Filter Type                    | CKF      | UKF       | EKF-auto  | EnKF      |
+================================+==========+===========+===========+===========+
| **Accuracy**                   |          |           |           |           |
|  Nonlinearity (mild)           | EKF+     | EKF+      | EKF       | EKF+      |
|  Nonlinearity (strong)         | Good     | Good      | Fair      | Good      |
|  Non-Gaussian errors           | Fair     | Fair      | Fair      | **Good**  |
+--------------------------------+----------+-----------+-----------+-----------+
| **Speed** (relative to EKF)    |          |           |           |           |
|  Single step                   | 1.5x     | 1.3x      | 2.0x      | Nx        |
|  Function evaluations          | 2n       | 2n+1      | 2n        | N members |
+--------------------------------+----------+-----------+-----------+-----------+
| **Jacobian Required**          | No       | No        | No        | No        |
|                                |          |           | (numeric) |           |
+--------------------------------+----------+-----------+-----------+-----------+
| **Memory (relative)**          | 1x       | 1x        | 1x        | Nx        |
+--------------------------------+----------+-----------+-----------+-----------+
| **Parallelizable**             | No       | No        | No        | **Yes**   |
+--------------------------------+----------+-----------+-----------+-----------+
| **High Dimensions** (n>1000)   | No       | No        | No        | Yes       |
+--------------------------------+----------+-----------+-----------+-----------+

**When to Use Each:**

1. **Cubature Kalman Filter** (``ckf_predict`` / ``ckf_update``)

   - Moderate-dimensional systems (n < 100)
   - Smooth nonlinearities
   - Need derivative-free approach
   - Avoid for high dimensions or hard real-time constraints

2. **Unscented Kalman Filter** (``ukf_predict`` / ``ukf_update``)

   - Balance accuracy and speed
   - Most nonlinearities
   - Standard choice for modern tracking
   - Well-understood theory and tuning
   - Use ``sr_ukf_predict`` / ``sr_ukf_update`` for numerical robustness

3. **Numerical-Jacobian EKF** (``ekf_predict_auto`` / ``ekf_update_auto``)

   - Complex dynamics only available as code
   - Numerical precision issues make analytical Jacobians unreliable
   - Slightly more expensive than EKF with analytical Jacobians
   - Not significantly more accurate than EKF for most problems

4. **Ensemble Kalman Filter** (not shipped; see reference implementation above)

   - Very high dimensions (1000s-millions)
   - Non-Gaussian errors
   - Parallelizable across ensemble members
   - Data assimilation (geophysics, oceanography)
   - More complex, requires careful tuning
   - Smaller ensemble means sampling errors

Mixing Variants
===============

Because pytcl's filters are plain functions operating on ``(x, P)`` pairs,
variants compose freely: predictions and updates from different filters can
be interleaved in a single cycle, e.g. a cheap CKF time update with a UKF
measurement update, or different filters for different sensors.

.. code-block:: python

    # One hybrid cycle on the pendulum problem
    x = np.array([0.3, 0.0])
    P = np.diag([0.1, 0.1])

    pred = ckf_predict(x, P, f_pendulum, Q)          # CKF time update
    upd = ukf_update(pred.x, pred.P, zs_pend[0],     # UKF measurement update
                     h_pendulum, R)
    print(type(upd).__name__, np.round(upd.x, 4))
    # KalmanUpdate [ 0.4988 -0.4342]

Practical Diagnostics
======================

Innovation-based consistency checks work identically for every variant
because each update returns the innovation ``y`` and its covariance ``S``.
pytcl provides ``nis`` (Normalized Innovation Squared) and
``consistency_test`` in ``pytcl.performance_evaluation``:

.. code-block:: python

    from pytcl.performance_evaluation import consistency_test, nis

    x = np.array([0.3, 0.0])
    P = np.diag([0.1, 0.1])
    nis_values = []
    for z in zs_pend:
        pred = ukf_predict(x, P, f_pendulum, Q)
        upd = ukf_update(pred.x, pred.P, z, h_pendulum, R)
        nis_values.append(nis(upd.y, upd.S))
        x, P = upd.x, upd.P

    result = consistency_test(np.array(nis_values), df=1)
    print(f"mean NIS: {result.mean_value:.3f} (expect ~1 for df=1)")
    print(f"95% bounds: [{result.lower_bound:.3f}, {result.upper_bound:.3f}]")
    print(f"consistent: {result.is_consistent}")
    # mean NIS: 0.828 (expect ~1 for df=1)
    # 95% bounds: [0.742, 1.296]
    # consistent: True

A mean NIS near the measurement dimension indicates the filter's innovation
covariance matches reality; values far above suggest an overconfident filter
(``Q`` or ``R`` too small), far below an underconfident one. Note that
consecutive NIS values from a single run are correlated, so treat the
chi-squared bounds as indicative rather than exact (they are strictly valid
for independent samples, e.g. across Monte Carlo runs).

Tuning Guidelines
=================

**CKF Tuning:**

- Usually minimal tuning needed (derivative-free, symmetric, no parameters)
- Primary parameter: process noise :math:`Q` (same as standard Kalman)

**UKF Tuning:**

- :math:`\alpha` (spread): Typically :math:`10^{-3}` (start conservative)
- :math:`\beta` (prior knowledge): 2.0 for Gaussian
- :math:`\kappa` (secondary): Often 0, or :math:`3-n` for some applications

**Numerical-Jacobian EKF Tuning:**

- ``dx`` (step size): default :math:`10^{-7}`; problem-dependent
- Smaller ``dx``: More accurate linearization but numerically sensitive
- Larger ``dx``: More robust but less accurate

**EnKF Tuning:**

- Ensemble size :math:`N`: 50-1000 typical

  - Larger :math:`N`: Better approximation, more expensive
  - Smaller :math:`N`: Faster, but sampling errors

- Localization: For spatial systems, limit update region

**Rule of Thumb:**

1. Start with standard Kalman, then UKF, then an advanced variant
2. Use **CKF** if derivatives cause numerical issues
3. Use **EnKF** if dimension > 500
4. Switch to **ekf_predict_auto/ekf_update_auto** when you only need
   EKF-level accuracy but have no analytical Jacobian

Common Pitfalls
===============

1. **Tuning Proliferation**: Advanced filters have more parameters

   - *Fix*: Use defaults initially, tune conservatively

2. **High Ensemble Size Overhead**: EnKF with 1000 members is expensive

   - *Fix*: Use localization, data assimilation techniques

3. **Numerical Issues in Derivatives**: Finite differences can amplify roundoff errors

   - *Fix*: Use an appropriate ``dx``, consider analytical Jacobians

4. **Overconfidence in Ensemble Mean**: EnKF ensemble can collapse

   - *Fix*: Monitor ensemble spread, use covariance inflation

5. **Mode Switches**: IMM + advanced filter combinations complex

   - *Fix*: Test thoroughly, start simple

See Also
========

- :doc:`kalman_filter_tuning` -- Basics and standard Kalman
- :doc:`adaptive_filtering` -- Parameter tuning online
- :doc:`information_filters` -- Numerically stable alternatives
- :doc:`particle_filters` -- For multi-modal distributions
- :doc:`troubleshooting` -- Debugging filter issues

**References:**

- Arasaratnam & Haykin (2009) -- *Cubature Kalman Filters* -- Foundational CKF paper
- Sarkka (2013) -- *Bayesian Filtering and Smoothing* -- Comprehensive sigma-point theory
- Evensen (2003) -- *Ensemble Kalman Filter* -- Ensemble methods origins
- Bar-Shalom, Li, Kirubarajan (2001) -- *Estimation with Applications* -- Comprehensive reference
