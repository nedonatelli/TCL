"""
Constrained Extended Kalman Filter (CEKF).

Extends the Extended Kalman Filter to enforce constraints on the state
estimate. Uses Lagrange multiplier method to project onto constraint manifold
while maintaining positive definite covariance.

References
----------
.. [1] Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear
       Approaches. Wiley-Interscience.
.. [2] Simon, D. & Simon, D. L. (2010). Constrained Kalman filtering via
       density function truncation. Journal of Guidance, Control, and Dynamics.
"""

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.extended import ekf_predict, ekf_update
from pytcl.dynamic_estimation.kalman.linear import KalmanPrediction, KalmanUpdate


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
            return np.all(g_val <= tol)
        else:  # equality
            return np.allclose(g_val, 0, atol=tol)


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

        Performs standard EKF prediction (constraints not enforced here,
        only checked). Constraint enforcement happens in update step.

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

        # Iterative projection
        for iteration in range(max_iter):
            converged = True

            for constraint in violated:
                g_val = constraint.evaluate(x_proj)
                G = constraint.jacobian(x_proj)

                # Only process violated constraints
                if constraint.constraint_type == "inequality":
                    mask = g_val > tol
                else:
                    mask = np.abs(g_val) > tol

                if not np.any(mask):
                    continue

                converged = False

                # Lagrange multiplier for this constraint
                # λ = -(G P Gᵀ + μ)⁻¹ G (x - x_ref)
                # where x_ref is desired state (we use x)

                GP = G @ P_proj
                GPGt = GP @ G.T

                # Add small regularization for numerical stability
                mu = np.eye(GPGt.shape[0]) * 1e-6

                try:
                    GPGt_inv = np.linalg.inv(GPGt + mu)
                    lam = -GPGt_inv @ (G @ x_proj + g_val)

                    # State correction
                    x_corr = P_proj @ G.T @ lam
                    x_proj = x_proj + x_corr

                    # Covariance projection
                    # P_proj = P - P G^T (G P G^T)^{-1} G P
                    P_proj = P_proj - GP.T @ GPGt_inv @ GP

                    # Ensure symmetry
                    P_proj = (P_proj + P_proj.T) / 2

                    # Enforce positive definiteness
                    eigvals, eigvecs = np.linalg.eigh(P_proj)
                    if np.any(eigvals < 0):
                        eigvals[eigvals < 1e-10] = 1e-10
                        P_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T

                except np.linalg.LinAlgError:
                    # If inversion fails, use pseudoinverse
                    GPGt_pinv = np.linalg.pinv(GPGt)
                    lam = -GPGt_pinv @ (G @ x_proj + g_val)
                    x_proj = x_proj + P_proj @ G.T @ lam

            if converged:
                break

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
