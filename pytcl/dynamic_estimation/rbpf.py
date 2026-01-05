"""Rao-Blackwellized Particle Filter (RBPF).

The RBPF partitions the state into a nonlinear part (handled by particles) and
a linear part (handled by Kalman filters for each particle). This provides
better estimation quality than plain particle filters for systems with both
nonlinear and linear dynamics.

The algorithm:
1. Maintain N particles, each with:
   - Position in nonlinear state space (y)
   - Kalman filter state (x, P) for linear subspace
   - Weight w based on measurement likelihood
2. For each time step:
   - Predict: Propagate nonlinear particles, update KF for each
   - Update: Compute measurement likelihood, adapt weights
   - Resample: When effective sample size is low, draw new particles
   - Merge: Combine nearby particles to reduce variance

References:
- Doucet et al., "On Sequential Monte Carlo Sampling with Adaptive Weights"
  (Doucet & Tadic, 2003)
- Andrieu et al., "Particle Methods for Change Detection, System Identification"
  (IEEE SPM, 2004)
"""

from typing import Any, Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from pytcl.dynamic_estimation.kalman.extended import ekf_predict, ekf_update


class RBPFParticle(NamedTuple):
    """Rao-Blackwellized particle with nonlinear and linear components.

    Parameters
    ----------
    y : NDArray
        Nonlinear state component (propagated by particle transition)
    x : NDArray
        Linear state component (estimated by Kalman filter for this particle)
    P : NDArray
        Covariance of linear state component
    w : float
        Particle weight (typically normalized to sum to 1)
    """

    y: NDArray[Any]
    x: NDArray[Any]
    P: NDArray[Any]
    w: float


class RBPFFilter:
    """Rao-Blackwellized Particle Filter.

    Combines particle filtering for nonlinear states with Kalman filtering
    for conditionally-linear states. For a system partitioned as:
        - y: nonlinear state (particles)
        - x: linear state given y (Kalman filter)

    Attributes
    ----------
    particles : list[RBPFParticle]
        Current particles with nonlinear/linear states and weights
    max_particles : int
        Maximum number of particles (default 100)
    resample_threshold : float
        Resample when N_eff < resample_threshold * N (default 0.5)
    merge_threshold : float
        Merge nearby particles when KL divergence < threshold (default 0.5)
    """

    def __init__(
        self,
        max_particles: int = 100,
        resample_threshold: float = 0.5,
        merge_threshold: float = 0.5,
    ):
        """Initialize RBPF.

        Parameters
        ----------
        max_particles : int
            Maximum number of particles to maintain
        resample_threshold : float
            Resample threshold as fraction of max particles
        merge_threshold : float
            KL divergence threshold for merging particles
        """
        self.particles: list[RBPFParticle] = []
        self.max_particles = max_particles
        self.resample_threshold = resample_threshold
        self.merge_threshold = merge_threshold

    def initialize(
        self,
        y0: NDArray[Any],
        x0: NDArray[Any],
        P0: NDArray[Any],
        num_particles: int = 100,
    ) -> None:
        """Initialize particles.

        Parameters
        ----------
        y0 : NDArray
            Initial nonlinear state (broadcasted to all particles)
        x0 : NDArray
            Initial linear state (broadcasted to all particles)
        P0 : NDArray
            Initial linear state covariance (same for all particles)
        num_particles : int
            Number of particles to initialize
        """
        self.particles = []
        weight = 1.0 / num_particles

        # Add small noise to particle y values to break ties
        ny = y0.shape[0]

        for i in range(num_particles):
            # Nonlinear component: small perturbation around y0
            y = y0 + np.random.randn(ny) * 1e-6
            # Linear component: same for all particles (improved by update)
            x = x0.copy()
            P = P0.copy()

            particle = RBPFParticle(y=y, x=x, P=P, w=weight)
            self.particles.append(particle)

    def predict(
        self,
        g: Callable[[NDArray[Any]], NDArray[Any]],
        G: NDArray[Any],
        Qy: NDArray[Any],
        f: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
        F: NDArray[Any],
        Qx: NDArray[Any],
    ) -> None:
        """Predict step: propagate particles and linear states.

        Parameters
        ----------
        g : callable
            Nonlinear state transition: y[k+1] = g(y[k])
        G : NDArray
            Jacobian of g with respect to y (for covariance propagation)
        Qy : NDArray
            Process noise covariance for nonlinear state
        f : callable
            Linear transition: x[k+1] = f(x[k], y[k])
        F : NDArray
            Jacobian matrix dF/dx (linearized around y)
        Qx : NDArray
            Process noise covariance for linear state
        """
        new_particles = []

        for particle in self.particles:
            # Predict nonlinear component
            y_pred = g(particle.y)
            # Add process noise
            y_pred = y_pred + np.random.multivariate_normal(
                np.zeros(y_pred.shape[0]), Qy
            )

            # Create wrapper for linear dynamics with current y_pred
            def f_wrapper(x: NDArray[Any]) -> NDArray[Any]:
                return f(x, y_pred)

            # Predict linear component using EKF
            pred = ekf_predict(particle.x, particle.P, f_wrapper, F, Qx)

            new_particle = RBPFParticle(
                y=y_pred,
                x=pred.x,
                P=pred.P,
                w=particle.w,
            )
            new_particles.append(new_particle)

        self.particles = new_particles

    def update(
        self,
        z: NDArray[Any],
        h: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
        H: NDArray[Any],
        R: NDArray[Any],
    ) -> None:
        """Update step: adapt particle weights based on measurement.

        Parameters
        ----------
        z : NDArray
            Measurement vector
        h : callable
            Measurement function: z = h(x, y)
        H : NDArray
            Jacobian matrix dH/dx (measurement sensitivity)
        R : NDArray
            Measurement noise covariance
        """
        weights = np.zeros(len(self.particles))
        new_particles = []

        for i, particle in enumerate(self.particles):
            # Create wrapper for measurement function with current y
            def h_wrapper(x: NDArray[Any]) -> NDArray[Any]:
                return h(x, particle.y)

            # Update linear component (Kalman update)
            upd = ekf_update(particle.x, particle.P, z, h_wrapper, H, R)

            # Weight: measurement likelihood from Kalman update
            likelihood = upd.likelihood

            # Unnormalized weight
            weights[i] = particle.w * likelihood

            new_particle = RBPFParticle(
                y=particle.y,
                x=upd.x,
                P=upd.P,
                w=particle.w,  # Will renormalize below
            )
            new_particles.append(new_particle)

        # Normalize weights
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights = weights / w_sum
        else:
            # Uniform weights if all likelihoods are zero
            weights = np.ones(len(self.particles)) / len(self.particles)

        # Update particles with new weights
        self.particles = [
            RBPFParticle(y=p.y, x=p.x, P=p.P, w=w)
            for p, w in zip(new_particles, weights)
        ]

        # Resample if needed
        self._resample_if_needed()

        # Merge if too many particles
        self._merge_particles()

    def estimate(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Estimate state as weighted mean and covariance.

        Returns
        -------
        y_est : NDArray
            Weighted mean of nonlinear components
        x_est : NDArray
            Weighted mean of linear components
        P_est : NDArray
            Weighted covariance (includes mixture and linear uncertainties)
        """
        if not self.particles:
            raise ValueError("No particles to estimate")

        weights = np.array([p.w for p in self.particles])

        # Nonlinear state: weighted mean
        y_particles = np.array([p.y for p in self.particles])
        y_est = np.average(y_particles, axis=0, weights=weights)

        # Linear state: weighted mean and covariance
        x_particles = np.array([p.x for p in self.particles])
        x_est = np.average(x_particles, axis=0, weights=weights)

        # Covariance: E[(x - x_est)(x - x_est)^T] = E[Cov[x|y]] + Cov[E[x|y]]
        # = weighted_mean(P) + weighted_cov(x)

        # Weighted mean of covariances
        P_mean = np.zeros((self.particles[0].P.shape[0], self.particles[0].P.shape[1]))
        for p in self.particles:
            P_mean += p.w * p.P

        # Weighted covariance of means
        P_cov = np.zeros((self.particles[0].P.shape[0], self.particles[0].P.shape[1]))
        for p in self.particles:
            dx = p.x - x_est
            P_cov += p.w * np.outer(dx, dx)

        P_est = P_mean + P_cov

        return y_est, x_est, P_est

    def get_particles(self) -> list[RBPFParticle]:
        """Get current particles.

        Returns
        -------
        list[RBPFParticle]
            Current particle list
        """
        return self.particles.copy()

    def _resample_if_needed(self) -> None:
        """Resample particles if effective sample size is too low.

        Uses systematic resampling to reduce variance.
        """
        weights = np.array([p.w for p in self.particles])

        # Effective sample size
        N_eff = 1.0 / np.sum(weights**2)

        threshold = self.resample_threshold * len(self.particles)

        if N_eff < threshold:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        """Perform systematic resampling."""
        weights = np.array([p.w for p in self.particles])
        n = len(self.particles)

        # Cumulative sum
        cs = np.cumsum(weights)

        # Resample indices
        indices = []
        u = np.random.uniform(0, 1.0 / n)

        j = 0
        for i in range(n):
            while u > cs[j]:
                j += 1
            indices.append(j)
            u += 1.0 / n

        # Create new particles with uniform weights
        new_particles = []
        weight = 1.0 / n

        for idx in indices:
            p = self.particles[idx]
            new_particles.append(
                RBPFParticle(y=p.y.copy(), x=p.x.copy(), P=p.P.copy(), w=weight)
            )

        self.particles = new_particles

    def _merge_particles(self) -> None:
        """Merge nearby particles to reduce variance."""
        if len(self.particles) <= 1:
            return

        # Find closest pair by KL divergence
        max_iter = len(self.particles) - self.max_particles

        for _ in range(max_iter):
            if len(self.particles) <= self.max_particles:
                break

            best_div = np.inf
            best_i, best_j = 0, 1

            # Find closest pair
            for i in range(len(self.particles)):
                for j in range(i + 1, len(self.particles)):
                    div = self._kl_divergence(
                        self.particles[i].P,
                        self.particles[j].P,
                        self.particles[i].x,
                        self.particles[j].x,
                    )
                    if div < best_div:
                        best_div = div
                        best_i, best_j = i, j

            if best_div < self.merge_threshold:
                # Merge particles i and j
                p_i = self.particles[best_i]
                p_j = self.particles[best_j]

                # Weighted merge
                w_total = p_i.w + p_j.w
                w_i = p_i.w / w_total
                w_j = p_j.w / w_total

                # Merged nonlinear state
                y_merged = w_i * p_i.y + w_j * p_j.y

                # Merged linear state and covariance
                x_merged = w_i * p_i.x + w_j * p_j.x

                # Merged covariance
                P_merged = (
                    w_i * p_i.P
                    + w_j * p_j.P
                    + w_i * np.outer(p_i.x - x_merged, p_i.x - x_merged)
                    + w_j * np.outer(p_j.x - x_merged, p_j.x - x_merged)
                )

                merged_particle = RBPFParticle(
                    y=y_merged, x=x_merged, P=P_merged, w=w_total
                )

                # Replace particles
                if best_i < best_j:
                    self.particles[best_i] = merged_particle
                    self.particles.pop(best_j)
                else:
                    self.particles[best_j] = merged_particle
                    self.particles.pop(best_i)
            else:
                break

        # Renormalize weights
        w_sum = sum(p.w for p in self.particles)
        if w_sum > 0:
            self.particles = [
                RBPFParticle(y=p.y, x=p.x, P=p.P, w=p.w / w_sum) for p in self.particles
            ]

    @staticmethod
    def _kl_divergence(
        P1: NDArray[Any], P2: NDArray[Any], x1: NDArray[Any], x2: NDArray[Any]
    ) -> float:
        """Compute KL divergence between two Gaussians.

        KL(N(x1, P1) || N(x2, P2)) = 0.5 * [
            trace(P2^-1 @ P1) + (x2-x1)^T @ P2^-1 @ (x2-x1) - n + ln(|P2|/|P1|)
        ]

        Parameters
        ----------
        P1 : NDArray
            Covariance of first Gaussian
        P2 : NDArray
            Covariance of second Gaussian
        x1 : NDArray
            Mean of first Gaussian
        x2 : NDArray
            Mean of second Gaussian

        Returns
        -------
        float
            KL divergence (always >= 0)
        """
        try:
            P2_inv = np.linalg.inv(P2)
            n = P1.shape[0]

            # Trace term
            trace_term = np.trace(P2_inv @ P1)

            # Mean difference term
            dx = x2 - x1
            mean_term = dx @ P2_inv @ dx

            # Determinant term
            det_term = np.linalg.slogdet(P2)[1] - np.linalg.slogdet(P1)[1]

            kl = 0.5 * (trace_term + mean_term - n + det_term)
            return float(np.maximum(kl, 0.0))  # Ensure non-negative
        except (np.linalg.LinAlgError, ValueError):
            return np.inf


# Convenience functions for functional interface


def rbpf_predict(
    particles: list[RBPFParticle],
    g: Callable[[NDArray[Any]], NDArray[Any]],
    G: NDArray[Any],
    Qy: NDArray[Any],
    f: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
    F: NDArray[Any],
    Qx: NDArray[Any],
) -> list[RBPFParticle]:
    """Predict step for RBPF particles.

    Parameters
    ----------
    particles : list[RBPFParticle]
        Current particles
    g : callable
        Nonlinear state transition
    G : NDArray
        Jacobian of nonlinear transition
    Qy : NDArray
        Process noise covariance for nonlinear state
    f : callable
        Linear state transition
    F : NDArray
        Jacobian of linear transition
    Qx : NDArray
        Process noise covariance for linear state

    Returns
    -------
    list[RBPFParticle]
        Predicted particles

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.dynamic_estimation.rbpf import RBPFParticle
    >>> np.random.seed(42)
    >>> # 3 particles with nonlinear bearing and linear position
    >>> particles = [
    ...     RBPFParticle(y=np.array([0.1]), x=np.array([0.0, 1.0]),
    ...                  P=np.eye(2) * 0.5, w=1/3),
    ...     RBPFParticle(y=np.array([0.0]), x=np.array([0.0, 1.0]),
    ...                  P=np.eye(2) * 0.5, w=1/3),
    ...     RBPFParticle(y=np.array([-0.1]), x=np.array([0.0, 1.0]),
    ...                  P=np.eye(2) * 0.5, w=1/3),
    ... ]
    >>> # Nonlinear dynamics for bearing
    >>> g = lambda y: y  # bearing stays constant
    >>> G = np.eye(1)
    >>> Qy = np.eye(1) * 0.01
    >>> # Linear dynamics for position
    >>> f = lambda x, y: np.array([x[0] + x[1] * 0.1, x[1]])
    >>> F = np.array([[1, 0.1], [0, 1]])
    >>> Qx = np.eye(2) * 0.01
    >>> predicted = rbpf_predict(particles, g, G, Qy, f, F, Qx)
    >>> len(predicted)
    3
    """
    new_particles = []

    for particle in particles:
        # Predict nonlinear component
        y_pred = g(particle.y)
        y_pred = y_pred + np.random.multivariate_normal(np.zeros(y_pred.shape[0]), Qy)

        # Create wrapper for linear dynamics with current y_pred
        def f_wrapper(x: NDArray[Any]) -> NDArray[Any]:
            return f(x, y_pred)

        # Predict linear component
        pred = ekf_predict(particle.x, particle.P, f_wrapper, F, Qx)

        new_particle = RBPFParticle(
            y=y_pred,
            x=pred.x,
            P=pred.P,
            w=particle.w,
        )
        new_particles.append(new_particle)

    return new_particles


def rbpf_update(
    particles: list[RBPFParticle],
    z: NDArray[Any],
    h: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
    H: NDArray[Any],
    R: NDArray[Any],
) -> list[RBPFParticle]:
    """Update step for RBPF particles.

    Parameters
    ----------
    particles : list[RBPFParticle]
        Predicted particles
    z : NDArray
        Measurement
    h : callable
        Measurement function
    H : NDArray
        Jacobian of measurement function
    R : NDArray
        Measurement noise covariance

    Returns
    -------
    list[RBPFParticle]
        Updated particles with adapted weights

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.dynamic_estimation.rbpf import RBPFParticle
    >>> # 3 particles at different bearings
    >>> particles = [
    ...     RBPFParticle(y=np.array([0.5]), x=np.array([1.0, 0.0]),
    ...                  P=np.eye(2) * 0.5, w=1/3),
    ...     RBPFParticle(y=np.array([0.0]), x=np.array([1.0, 0.0]),
    ...                  P=np.eye(2) * 0.5, w=1/3),
    ...     RBPFParticle(y=np.array([-0.5]), x=np.array([1.0, 0.0]),
    ...                  P=np.eye(2) * 0.5, w=1/3),
    ... ]
    >>> # Position measurement
    >>> z = np.array([1.1])
    >>> h = lambda x, y: np.array([x[0]])  # measure position
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[0.1]])
    >>> updated = rbpf_update(particles, z, h, H, R)
    >>> len(updated)
    3
    >>> # Weights should sum to 1
    >>> abs(sum(p.w for p in updated) - 1.0) < 1e-10
    True
    """
    weights = np.zeros(len(particles))
    new_particles = []

    for i, particle in enumerate(particles):
        # Create wrapper for measurement function with current y
        def h_wrapper(x: NDArray[Any]) -> NDArray[Any]:
            return h(x, particle.y)

        # Update linear component
        upd = ekf_update(particle.x, particle.P, z, h_wrapper, H, R)

        # Weight by measurement likelihood
        weights[i] = particle.w * upd.likelihood

        new_particle = RBPFParticle(
            y=particle.y,
            x=upd.x,
            P=upd.P,
            w=particle.w,
        )
        new_particles.append(new_particle)

    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(len(particles)) / len(particles)

    # Update with new weights
    return [
        RBPFParticle(y=p.y, x=p.x, P=p.P, w=w) for p, w in zip(new_particles, weights)
    ]
