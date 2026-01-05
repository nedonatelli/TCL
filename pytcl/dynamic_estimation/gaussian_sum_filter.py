"""Gaussian Sum Filter (GSF) for nonlinear state estimation.

The Gaussian Sum Filter represents the posterior distribution as a weighted
mixture of Gaussians. This allows approximation of multi-modal distributions
and nonlinear systems better than a single EKF.

Each component has:
- Weight (w_i): Probability of that Gaussian
- Mean (μ_i): Component state estimate
- Covariance (P_i): Component uncertainty

The filter performs:
1. Predict: Propagate each component independently
2. Update: Update each component with measurement, adapt weights
3. Manage: Prune low-weight components, merge similar ones

References
----------
Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with
Applications to Tracking and Navigation. Wiley-Interscience.
"""

from typing import Callable, List, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.extended import ekf_predict, ekf_update


class GaussianComponent(NamedTuple):
    """Single Gaussian component in mixture."""

    x: NDArray[np.floating]  # State estimate
    P: NDArray[np.floating]  # Covariance
    w: float  # Weight (probability)


class GaussianSumFilter:
    """Gaussian Sum Filter for nonlinear state estimation.

    A mixture model approach that represents the posterior distribution
    as a weighted sum of Gaussians. Useful for multi-modal distributions
    and nonlinear systems.

    Attributes
    ----------
    components : list[GaussianComponent]
        Current mixture components (state, covariance, weight).
    max_components : int
        Maximum components to maintain (via pruning/merging).
    merge_threshold : float
        KL divergence threshold for merging components.
    prune_threshold : float
        Weight threshold below which to prune components.
    """

    def __init__(
        self,
        max_components: int = 5,
        merge_threshold: float = 0.01,
        prune_threshold: float = 1e-3,
    ):
        """Initialize Gaussian Sum Filter.

        Parameters
        ----------
        max_components : int
            Maximum number of Gaussian components to maintain.
        merge_threshold : float
            KL divergence threshold for merging. Components with KL
            divergence below this are merged.
        prune_threshold : float
            Weight threshold for pruning. Components with weight below
            this are removed.
        """
        self.components: List[GaussianComponent] = []
        self.max_components = max_components
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold

    def initialize(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        num_components: int = 1,
    ) -> None:
        """Initialize filter with initial state.

        Parameters
        ----------
        x0 : array_like
            Initial state estimate, shape (n,).
        P0 : array_like
            Initial covariance, shape (n, n).
        num_components : int
            Number of components to initialize with. If > 1, will
            create multiple components with slightly perturbed means.
        """
        x0 = np.asarray(x0, dtype=np.float64)
        P0 = np.asarray(P0, dtype=np.float64)

        self.components = []
        weight = 1.0 / num_components

        for i in range(num_components):
            if i == 0:
                x = x0.copy()
            else:
                # Slight perturbation for diversity
                x = x0 + np.random.randn(x0.shape[0]) * np.sqrt(np.diag(P0)) * 0.1

            self.components.append(GaussianComponent(x=x, P=P0.copy(), w=weight))

    def predict(
        self,
        f: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        F: ArrayLike,
        Q: ArrayLike,
    ) -> None:
        """Predict step: propagate each component.

        Parameters
        ----------
        f : callable
            Nonlinear state transition function f(x).
        F : array_like
            Jacobian of f, shape (n, n).
        Q : array_like
            Process noise covariance, shape (n, n).
        """
        F = np.asarray(F, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        new_components = []
        for comp in self.components:
            # EKF predict for each component
            pred = ekf_predict(comp.x, comp.P, f, F, Q)
            new_components.append(GaussianComponent(x=pred.x, P=pred.P, w=comp.w))

        self.components = new_components

    def update(
        self,
        z: ArrayLike,
        h: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        H: ArrayLike,
        R: ArrayLike,
    ) -> None:
        """Update step: update each component, adapt weights.

        Parameters
        ----------
        z : array_like
            Measurement, shape (m,).
        h : callable
            Nonlinear measurement function h(x).
        H : array_like
            Jacobian of h, shape (m, n).
        R : array_like
            Measurement noise covariance, shape (m, m).
        """
        z = np.asarray(z, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        # Update each component and compute likelihoods
        likelihoods = []
        updated_components = []

        for comp in self.components:
            # EKF update for component
            upd = ekf_update(comp.x, comp.P, z, h, H, R)

            # Likelihood from this measurement
            likelihood = upd.likelihood

            updated_components.append(GaussianComponent(x=upd.x, P=upd.P, w=comp.w))
            likelihoods.append(likelihood)

        # Adapt weights based on measurement likelihood
        likelihoods = np.array(likelihoods)
        weights = np.array([c.w for c in updated_components])

        # Normalize weights by likelihood (Bayesian update)
        weights = weights * likelihoods
        weight_sum = np.sum(weights)

        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback: equal weights if all likelihoods zero
            weights = np.ones(len(updated_components)) / len(updated_components)

        # Update components with new weights
        self.components = [
            GaussianComponent(x=c.x, P=c.P, w=w)
            for c, w in zip(updated_components, weights)
        ]

        # Manage components (prune, merge)
        self._prune_components()
        self._merge_components()

    def _prune_components(self) -> None:
        """Remove components with weight below threshold."""
        self.components = [c for c in self.components if c.w >= self.prune_threshold]

        if len(self.components) == 0:
            # Failsafe: keep best component
            self.components = [max(self.components, key=lambda c: c.w)]

        # Renormalize weights
        total_weight = sum(c.w for c in self.components)
        if total_weight > 0:
            self.components = [
                GaussianComponent(x=c.x, P=c.P, w=c.w / total_weight)
                for c in self.components
            ]

    def _merge_components(self) -> None:
        """Merge similar components to keep count manageable."""
        while len(self.components) > self.max_components:
            # Find pair with smallest KL divergence
            best_i, best_j, best_kl = 0, 1, float("inf")

            for i in range(len(self.components)):
                for j in range(i + 1, len(self.components)):
                    kl = self._kl_divergence(self.components[i], self.components[j])
                    if kl < best_kl:
                        best_kl = kl
                        best_i = best_j = i
                        best_j = j

            if best_kl < self.merge_threshold:
                # Merge components i and j
                ci = self.components[best_i]
                cj = self.components[best_j]

                # Merged weight
                w_new = ci.w + cj.w

                # Merged mean (weighted average)
                x_new = (ci.w * ci.x + cj.w * cj.x) / w_new

                # Merged covariance (weighted average of covariances
                # plus covariance of means)
                P_new = (ci.w * ci.P + cj.w * cj.P) / w_new
                dx_i = ci.x - x_new
                dx_j = cj.x - x_new
                P_new += (
                    ci.w * np.outer(dx_i, dx_i) + cj.w * np.outer(dx_j, dx_j)
                ) / w_new

                # Create merged component
                merged = GaussianComponent(x=x_new, P=P_new, w=w_new)

                # Replace with merged, remove old
                self.components = [
                    c
                    for i, c in enumerate(self.components)
                    if i != best_i and i != best_j
                ]
                self.components.append(merged)
            else:
                # Can't merge more, stop
                break

    @staticmethod
    def _kl_divergence(c1: GaussianComponent, c2: GaussianComponent) -> float:
        """Compute KL divergence between two Gaussians.

        KL(N1 || N2) = 0.5 * [tr(P2^{-1}P1) + (μ2-μ1)^T P2^{-1}
        (μ2-μ1) - n + ln|P2|/|P1|]

        Parameters
        ----------
        c1, c2 : GaussianComponent
            Gaussian components.

        Returns
        -------
        kl : float
            KL divergence from c1 to c2.
        """
        dx = c2.x - c1.x
        n = len(c1.x)

        try:
            P2_inv = np.linalg.inv(c2.P)
            logdet_ratio = np.linalg.slogdet(c2.P)[1] - np.linalg.slogdet(c1.P)[1]

            trace_term = np.trace(P2_inv @ c1.P)
            quad_term = dx @ P2_inv @ dx
            kl = 0.5 * (trace_term + quad_term - n + logdet_ratio)

            return float(np.clip(kl, 0, np.inf))
        except np.linalg.LinAlgError:
            # Singular matrix, return large KL
            return 1e6

    def estimate(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get overall state estimate (weighted mean and covariance).

        Returns
        -------
        x : ndarray
            Weighted mean of components.
        P : ndarray
            Weighted covariance of components.
        """
        if not self.components:
            raise ValueError("No components initialized")

        # Weighted mean
        x_est = np.zeros_like(self.components[0].x)
        for comp in self.components:
            x_est += comp.w * comp.x

        # Weighted covariance
        P_est = np.zeros_like(self.components[0].P)
        for comp in self.components:
            dx = comp.x - x_est
            P_est += comp.w * (comp.P + np.outer(dx, dx))

        return x_est, P_est

    def get_components(self) -> List[GaussianComponent]:
        """Get current mixture components.

        Returns
        -------
        components : list[GaussianComponent]
            List of components with (x, P, w).
        """
        return self.components.copy()

    def get_num_components(self) -> int:
        """Get number of current components."""
        return len(self.components)


def gaussian_sum_filter_predict(
    components: List[GaussianComponent],
    f: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    F: ArrayLike,
    Q: ArrayLike,
) -> List[GaussianComponent]:
    """Convenience function for GSF prediction.

    Parameters
    ----------
    components : list[GaussianComponent]
        Current mixture components.
    f : callable
        Nonlinear state transition function.
    F : array_like
        Jacobian of f.
    Q : array_like
        Process noise covariance.

    Returns
    -------
    components_new : list[GaussianComponent]
        Predicted components.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.dynamic_estimation.gaussian_sum_filter import GaussianComponent
    >>> # Two-component mixture for position-velocity state
    >>> comp1 = GaussianComponent(
    ...     x=np.array([0.0, 1.0]),  # moving right
    ...     P=np.eye(2) * 0.5,
    ...     w=0.5
    ... )
    >>> comp2 = GaussianComponent(
    ...     x=np.array([0.0, -1.0]),  # moving left
    ...     P=np.eye(2) * 0.5,
    ...     w=0.5
    ... )
    >>> components = [comp1, comp2]
    >>> # Constant velocity dynamics
    >>> dt = 0.1
    >>> f = lambda x: np.array([x[0] + x[1] * dt, x[1]])
    >>> F = np.array([[1, dt], [0, 1]])
    >>> Q = np.eye(2) * 0.01
    >>> predicted = gaussian_sum_filter_predict(components, f, F, Q)
    >>> len(predicted)
    2
    >>> predicted[0].w  # weights unchanged in prediction
    0.5
    """
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    new_components = []
    for comp in components:
        pred = ekf_predict(comp.x, comp.P, f, F, Q)
        new_components.append(GaussianComponent(x=pred.x, P=pred.P, w=comp.w))

    return new_components


def gaussian_sum_filter_update(
    components: List[GaussianComponent],
    z: ArrayLike,
    h: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    H: ArrayLike,
    R: ArrayLike,
) -> List[GaussianComponent]:
    """Convenience function for GSF update.

    Parameters
    ----------
    components : list[GaussianComponent]
        Predicted mixture components.
    z : array_like
        Measurement.
    h : callable
        Nonlinear measurement function.
    H : array_like
        Jacobian of h.
    R : array_like
        Measurement noise covariance.

    Returns
    -------
    components_new : list[GaussianComponent]
        Updated components with adapted weights.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.dynamic_estimation.gaussian_sum_filter import GaussianComponent
    >>> # Two-component mixture
    >>> comp1 = GaussianComponent(
    ...     x=np.array([1.0, 0.5]),
    ...     P=np.eye(2) * 0.5,
    ...     w=0.5
    ... )
    >>> comp2 = GaussianComponent(
    ...     x=np.array([3.0, 0.5]),
    ...     P=np.eye(2) * 0.5,
    ...     w=0.5
    ... )
    >>> components = [comp1, comp2]
    >>> # Position measurement near component 1
    >>> z = np.array([1.1])
    >>> h = lambda x: np.array([x[0]])
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[0.1]])
    >>> updated = gaussian_sum_filter_update(components, z, h, H, R)
    >>> len(updated)
    2
    >>> # Component 1 should have higher weight (closer to measurement)
    >>> updated[0].w > updated[1].w
    True
    """
    z = np.asarray(z, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    likelihoods = []
    updated_components = []

    for comp in components:
        upd = ekf_update(comp.x, comp.P, z, h, H, R)
        likelihood = upd.likelihood

        updated_components.append(GaussianComponent(x=upd.x, P=upd.P, w=comp.w))
        likelihoods.append(likelihood)

    # Adapt weights
    likelihoods = np.array(likelihoods)
    weights = np.array([c.w for c in updated_components])

    weights = weights * likelihoods
    weight_sum = np.sum(weights)

    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(updated_components)) / len(updated_components)

    return [
        GaussianComponent(x=c.x, P=c.P, w=w)
        for c, w in zip(updated_components, weights)
    ]
