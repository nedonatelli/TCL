"""
Constrained Extended Kalman Filter (CEKF).

Extends the Extended Kalman Filter to enforce constraints on the state
estimate. Uses Lagrange multiplier method to project onto constraint manifold
while maintaining positive definite covariance.

References
----------
- Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear
  Approaches. Wiley-Interscience.
- Simon, D. & Simon, D. L. (2010). Constrained Kalman filtering via
  density function truncation. Journal of Guidance, Control, and Dynamics.
"""

import warnings
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.extended import ekf_predict, ekf_update
from pytcl.dynamic_estimation.kalman.linear import KalmanPrediction, KalmanUpdate

# Relative regularization for the G P Gt solves in
# ConstrainedEKF._project_onto_constraints. Scaled to trace(G P Gt) rather
# than an absolute constant so it stays negligible next to G P Gt at any
# covariance magnitude while still guarding the inverse near machine
# precision; see that method's docstring for the failure it replaces.
_REG_EPS_REL = np.finfo(np.float64).eps

# Relative floor for the projected covariance's eigenvalues, in the same
# method. An absolute 1e-10 either overstates a genuinely near-zero
# constrained-direction eigenvalue (harmless but misleading below
# P-scale 1e-6) or, once P's own eigenvalues drop below 1e-10, clamps
# the *unconstrained* direction up to 1e-10 too -- a 100x inflation at
# P-scale 1e-12. Scaled to the projected covariance's own largest
# eigenvalue instead: measured worst-case negative-eigenvalue roundoff
# noise from eigh, across 20000 randomized projections (n=2..7,
# P-scale 1e-15..1e3), was 1.43e-12 relative to that eigenvalue;
# sqrt(machine epsilon) sits four orders of magnitude above that.
_EIG_FLOOR_REL = np.sqrt(np.finfo(np.float64).eps)


class ConstraintFunction:
    """Base class for state constraints."""

    def __init__(
        self,
        g: Callable[[NDArray[Any]], NDArray[Any]],
        G: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        constraint_type: str = "inequality",
    ):
        """
        Define a constraint: g(x) ≤ 0 (inequality) or g(x) = 0 (equality).

        Parameters
        ----------
        g : callable
            Constraint function: g(x) -> scalar or array
            Inequality: g(x) ≤ 0
            Equality: g(x) = 0
        G : callable, optional
            Jacobian of g with respect to x: ∂g/∂x
            If None, computed numerically.
        constraint_type : {'inequality', 'equality'}
            Constraint type.
        """
        self.g = g
        self.G = G
        self.constraint_type = constraint_type

    def evaluate(self, x: NDArray[Any]) -> NDArray[Any]:
        """Evaluate constraint at state x."""
        return np.atleast_1d(np.asarray(self.g(x), dtype=np.float64))

    def jacobian(self, x: NDArray[Any]) -> NDArray[Any]:
        """Compute constraint Jacobian at x."""
        if self.G is not None:
            return np.atleast_2d(np.asarray(self.G(x), dtype=np.float64))
        else:
            # Numerical differentiation
            eps = 1e-6
            n = len(x)
            g_x = self.evaluate(x)
            m = len(g_x)
            J = np.zeros((m, n))
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += eps
                g_plus = self.evaluate(x_plus)
                J[:, i] = (g_plus - g_x) / eps
            return J

    def is_satisfied(self, x: NDArray[Any], tol: float = 1e-6) -> bool:
        """Check if constraint is satisfied."""
        g_val = self.evaluate(x)
        if self.constraint_type == "inequality":
            return bool(np.all(g_val <= tol))
        else:  # equality
            return np.allclose(g_val, 0, atol=tol)


def _violation(g_val: NDArray[Any], constraint_type: str) -> NDArray[Any]:
    """Per-row violation, on the same scale `is_satisfied` thresholds."""
    if constraint_type == "inequality":
        return np.maximum(g_val, 0.0)
    return np.abs(g_val)


class ConstrainedEKF:
    """
    Extended Kalman Filter with state constraints.

    Enforces linear and/or nonlinear constraints on state estimate using
    Lagrange multiplier method with covariance projection.

    Attributes
    ----------
    constraints : list of ConstraintFunction
        List of active constraints.
    """

    def __init__(self) -> None:
        """Initialize Constrained EKF."""
        self.constraints: list[ConstraintFunction] = []

    def add_constraint(self, constraint: ConstraintFunction) -> None:
        """
        Add a constraint to the filter.

        Parameters
        ----------
        constraint : ConstraintFunction
            Constraint to enforce.
        """
        self.constraints.append(constraint)

    def predict(
        self,
        x: ArrayLike,
        P: ArrayLike,
        f: Callable[[NDArray[Any]], NDArray[Any]],
        F: ArrayLike,
        Q: ArrayLike,
    ) -> KalmanPrediction:
        """
        Constrained EKF prediction step.

        Performs standard EKF prediction. Constraints are neither enforced
        nor checked here -- the body is a plain ``ekf_predict`` call and
        ``self.constraints`` is not consulted, so a predicted state may
        violate them silently. Enforcement happens in the update step.

        Parameters
        ----------
        x : array_like
            Current state estimate, shape (n,).
        P : array_like
            Current state covariance, shape (n, n).
        f : callable
            Nonlinear state transition function.
        F : array_like
            Jacobian of f at current state.
        Q : array_like
            Process noise covariance, shape (n, n).

        Returns
        -------
        result : KalmanPrediction
            Predicted state and covariance.
        """
        return ekf_predict(x, P, f, F, Q)

    def update(
        self,
        x: ArrayLike,
        P: ArrayLike,
        z: ArrayLike,
        h: Callable[[NDArray[Any]], NDArray[Any]],
        H: ArrayLike,
        R: ArrayLike,
    ) -> KalmanUpdate:
        """
        Constrained EKF update step.

        Performs standard EKF update, then projects result onto constraint
        manifold.

        Parameters
        ----------
        x : array_like
            Predicted state estimate, shape (n,).
        P : array_like
            Predicted state covariance, shape (n, n).
        z : array_like
            Measurement, shape (m,).
        h : callable
            Nonlinear measurement function.
        H : array_like
            Jacobian of h at current state.
        R : array_like
            Measurement noise covariance, shape (m, m).

        Returns
        -------
        result : KalmanUpdate
            Updated state and covariance (constrained).
        """
        # Standard EKF update
        result = ekf_update(x, P, z, h, H, R)
        x_upd = result.x
        P_upd = result.P

        # Apply constraint projection
        if self.constraints:
            x_upd, P_upd = self._project_onto_constraints(x_upd, P_upd)

        return KalmanUpdate(
            x=x_upd,
            P=P_upd,
            y=result.y,
            S=result.S,
            K=result.K,
            likelihood=result.likelihood,
        )

    def _project_onto_constraints(
        self,
        x: NDArray[Any],
        P: NDArray[Any],
        max_iter: int = 10,
        tol: float = 1e-6,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Project state and covariance onto constraint manifold.

        Uses iterative Lagrange multiplier method with covariance
        projection to enforce constraints while maintaining positive
        definiteness.

        Parameters
        ----------
        x : ndarray
            State estimate, shape (n,).
        P : ndarray
            Covariance matrix, shape (n, n).
        max_iter : int
            Maximum iterations for constraint projection.
        tol : float
            Convergence tolerance.

        Returns
        -------
        x_proj : ndarray
            Constrained state estimate.
        P_proj : ndarray
            Projected covariance.
        """
        x_proj = x.copy()
        P_proj = P.copy()

        # Check which constraints are violated
        violated: list[ConstraintFunction] = [
            c for c in self.constraints if not c.is_satisfied(x_proj)
        ]

        if not violated:
            return x_proj, P_proj

        # The state iteration below uses the *incoming* covariance as a fixed
        # metric. Shrinking P_proj inside the loop -- as this used to do --
        # collapses it along G after the first step, so every later Newton
        # iteration is multiplied by an almost-zero gain and the state stalls
        # short of the constraint surface. The covariance is projected once,
        # after the state has converged.
        P_metric = P_proj.copy()
        active: list[ConstraintFunction] = []
        state_pinv_warned = False

        # Iterative projection
        for iteration in range(max_iter):
            converged = True

            for constraint in violated:
                g_val = constraint.evaluate(x_proj)
                G = constraint.jacobian(x_proj)

                # Only process violated rows of this constraint: a multi-row
                # ConstraintFunction (e.g. a two-sided box, g = [x-hi, -x+lo])
                # mixes satisfied and violated rows, and an unmasked G makes
                # G P G^T singular whenever two rows are anti-parallel at the
                # same state (opposite bounds on one variable both entering
                # the solve). `mask` used to be computed and never applied,
                # so every row entered regardless of violation.
                if constraint.constraint_type == "inequality":
                    mask = g_val > tol
                else:
                    mask = np.abs(g_val) > tol

                if not np.any(mask):
                    continue

                converged = False

                if constraint not in active:
                    active.append(constraint)

                G = G[mask]
                g_val = g_val[mask]

                # Covariance-weighted projection onto the linearised
                # constraint surface (Simon 2010, "Kalman filtering with state
                # constraints"). Linearising g about x gives
                #     g(x) + G (x_new - x) = 0,
                # and minimizing (x_new - x)ᵀ P⁻¹ (x_new - x) subject to that
                # yields
                #     x_new = x - P Gᵀ (G P Gᵀ)⁻¹ g(x),
                # i.e. λ = -(G P Gᵀ)⁻¹ g(x).
                #
                # This previously used -(G P Gᵀ)⁻¹ (G x + g), carrying a
                # spurious G x term. Since G x has nothing to do with how far
                # the constraint is violated, it dominated whenever the state
                # was far from the origin and threw the estimate across the
                # feasible region instead of onto its boundary.
                GP = G @ P_metric
                GPGt = GP @ G.T

                # Regularize relative to the problem's own scale rather than
                # by a fixed absolute amount: an absolute 1e-6 swamps GPGt
                # once P shrinks below that, and the Newton step converges
                # geometrically with ratio mu / (GPGt + mu) instead of in one
                # step, stalling short of the constraint surface within
                # max_iter. Scaling mu to trace(GPGt) keeps that ratio at
                # _REG_EPS_REL regardless of P's magnitude.
                m_dim = GPGt.shape[0]
                mu = np.eye(m_dim) * (_REG_EPS_REL * np.trace(GPGt) / m_dim)

                try:
                    GPGt_inv = np.linalg.inv(GPGt + mu)
                    lam = -GPGt_inv @ g_val
                except np.linalg.LinAlgError:
                    # GPGt is exactly singular (e.g. the constraint has zero
                    # sensitivity to the current uncertainty), so no relative
                    # regularization can restore invertibility. This can
                    # recur every iteration for the same call; warn once per
                    # call rather than once per iteration.
                    if not state_pinv_warned:
                        warnings.warn(
                            "constrained EKF state projection: G P G^T is "
                            "singular and could not be regularized; "
                            "falling back to np.linalg.pinv",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        state_pinv_warned = True
                    lam = -np.linalg.pinv(GPGt) @ g_val

                x_proj = x_proj + P_metric @ G.T @ lam

            if converged:
                break
        else:
            unsatisfied = [c for c in violated if not c.is_satisfied(x_proj, tol)]
            if unsatisfied:
                worst = max(
                    float(np.max(_violation(c.evaluate(x_proj), c.constraint_type)))
                    for c in unsatisfied
                )
                warnings.warn(
                    f"constrained EKF state projection did not converge "
                    f"after {max_iter} iterations; largest constraint "
                    f"violation {worst:.3e} exceeds tol {tol:.3e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Covariance projection, once per constraint that was active:
        #     P <- P - P Gᵀ (G P Gᵀ)⁻¹ G P
        # evaluated at the converged state, restricted to the rows that
        # ended up sitting on the boundary there (|g| <= tol). This is a
        # different predicate from the state loop's `g > tol` /
        # `abs(g) > tol` violated-row mask: by convergence, every row that
        # was driven to the boundary has g approx 0, so the violated-row
        # mask would now exclude exactly the rows that belong in the
        # active set. A row never violated during the state loop can also
        # end up here if correcting another row of the same constraint
        # pushed it onto its own boundary, which is why this is
        # recomputed fresh from `x_proj` rather than reusing the state
        # loop's per-iteration mask.
        cov_pinv_warned = False
        for constraint in active:
            G = constraint.jacobian(x_proj)
            g_val = constraint.evaluate(x_proj)
            active_mask = np.abs(g_val) <= tol
            if not np.any(active_mask):
                continue
            G = G[active_mask]
            GP = G @ P_proj
            GPGt = GP @ G.T
            m_dim = GPGt.shape[0]
            mu = np.eye(m_dim) * (_REG_EPS_REL * np.trace(GPGt) / m_dim)
            try:
                GPGt_inv = np.linalg.inv(GPGt + mu)
            except np.linalg.LinAlgError:
                # Can recur once per active constraint; warn once per call
                # rather than once per constraint.
                if not cov_pinv_warned:
                    warnings.warn(
                        "constrained EKF covariance projection: G P G^T is "
                        "singular and could not be regularized; falling "
                        "back to np.linalg.pinv",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    cov_pinv_warned = True
                GPGt_inv = np.linalg.pinv(GPGt)
            P_proj = P_proj - GP.T @ GPGt_inv @ GP

            # Ensure symmetry
            P_proj = (P_proj + P_proj.T) / 2

            # Enforce positive definiteness, flooring relative to this
            # matrix's own scale rather than an absolute constant -- see
            # _EIG_FLOOR_REL.
            eigvals, eigvecs = np.linalg.eigh(P_proj)
            eig_floor = _EIG_FLOOR_REL * np.max(np.abs(eigvals))
            # An eigenvalue more negative than -eig_floor is too large to
            # be eigh roundoff on this projection (_EIG_FLOOR_REL is
            # already sized to that roundoff -- see its definition above),
            # so the projected covariance is genuinely non-PSD rather than
            # merely rank-deficient in the constrained direction.
            most_neg_eig = eigvals.min()
            if most_neg_eig < -eig_floor:
                raise np.linalg.LinAlgError(
                    f"constrained EKF covariance projection is not "
                    f"positive semi-definite: eigenvalue {most_neg_eig:.6e} "
                    f"is too negative to be eigh roundoff"
                )
            if np.any(eigvals < eig_floor):
                eigvals[eigvals < eig_floor] = eig_floor
                P_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return x_proj, P_proj


def constrained_ekf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[Any]], NDArray[Any]],
    F: ArrayLike,
    Q: ArrayLike,
) -> KalmanPrediction:
    """
    Convenience function for constrained EKF prediction.

    Parameters
    ----------
    x : array_like
        Current state estimate.
    P : array_like
        Current covariance.
    f : callable
        Nonlinear dynamics function.
    F : array_like
        Jacobian of f.
    Q : array_like
        Process noise covariance.

    Returns
    -------
    result : KalmanPrediction
        Predicted state and covariance.
    """
    cekf = ConstrainedEKF()
    return cekf.predict(x, P, f, F, Q)


def constrained_ekf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[Any]], NDArray[Any]],
    H: ArrayLike,
    R: ArrayLike,
    constraints: Optional[list[ConstraintFunction]] = None,
) -> KalmanUpdate:
    """
    Convenience function for constrained EKF update.

    Parameters
    ----------
    x : array_like
        Predicted state.
    P : array_like
        Predicted covariance.
    z : array_like
        Measurement.
    h : callable
        Nonlinear measurement function.
    H : array_like
        Jacobian of h.
    R : array_like
        Measurement noise covariance.
    constraints : list, optional
        List of ConstraintFunction objects.

    Returns
    -------
    result : KalmanUpdate
        Updated state and covariance.
    """
    cekf = ConstrainedEKF()
    if constraints:
        for c in constraints:
            cekf.add_constraint(c)
    return cekf.update(x, P, z, h, H, R)


__all__ = [
    "ConstraintFunction",
    "ConstrainedEKF",
    "constrained_ekf_predict",
    "constrained_ekf_update",
]
