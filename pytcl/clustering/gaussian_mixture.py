"""
Gaussian Mixture operations for tracking applications.

This module provides Gaussian mixture representation, moment matching,
and mixture reduction algorithms (Runnalls', West's) commonly used in
multi-target tracking for hypothesis pruning.

References
----------
.. [1] A. R. Runnalls, "Kullback-Leibler Approach to Gaussian Mixture Reduction,"
       IEEE Trans. Aerospace and Electronic Systems, vol. 43, no. 3, 2007.
.. [2] M. West, "Approximating posterior distributions by mixture,"
       Journal of the Royal Statistical Society, Series B, vol. 55, no. 2, 1993.
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class GaussianComponent(NamedTuple):
    """A single Gaussian component in a mixture.

    Attributes
    ----------
    weight : float
        Component weight (0, 1], must sum to 1 across mixture.
    mean : ndarray
        Mean vector, shape (n,).
    covariance : ndarray
        Covariance matrix, shape (n, n).
    """

    weight: float
    mean: NDArray[np.floating]
    covariance: NDArray[np.floating]


class MergeResult(NamedTuple):
    """Result of merging two Gaussian components.

    Attributes
    ----------
    component : GaussianComponent
        The merged Gaussian component.
    cost : float
        KL-divergence-based merge cost.
    """

    component: GaussianComponent
    cost: float


class ReductionResult(NamedTuple):
    """Result of mixture reduction.

    Attributes
    ----------
    components : list of GaussianComponent
        Reduced set of components.
    n_original : int
        Original number of components.
    n_reduced : int
        Reduced number of components.
    total_cost : float
        Total cost incurred by merging.
    """

    components: List[GaussianComponent]
    n_original: int
    n_reduced: int
    total_cost: float


def moment_match(
    weights: ArrayLike,
    means: List[ArrayLike],
    covariances: List[ArrayLike],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute moment-matched mean and covariance from mixture components.

    Given a Gaussian mixture, computes a single Gaussian that matches
    the first two moments (mean and covariance) of the mixture.

    Parameters
    ----------
    weights : array_like
        Component weights, shape (k,). Must sum to 1.
    means : list of array_like
        Component means, each shape (n,).
    covariances : list of array_like
        Component covariances, each shape (n, n).

    Returns
    -------
    mean : ndarray
        Moment-matched mean, shape (n,).
    covariance : ndarray
        Moment-matched covariance, shape (n, n).

    Examples
    --------
    >>> import numpy as np
    >>> weights = [0.5, 0.5]
    >>> means = [np.array([0., 0.]), np.array([2., 0.])]
    >>> covs = [np.eye(2) * 0.1, np.eye(2) * 0.1]
    >>> m, P = moment_match(weights, means, covs)
    >>> m  # Should be [1., 0.]
    array([1., 0.])
    """
    weights = np.asarray(weights, dtype=np.float64)
    means = [np.asarray(m, dtype=np.float64).flatten() for m in means]
    covariances = [np.asarray(P, dtype=np.float64) for P in covariances]

    k = len(weights)
    n = len(means[0])

    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum

    # Combined mean: x = sum_i w_i * x_i
    mean = np.zeros(n)
    for i in range(k):
        mean += weights[i] * means[i]

    # Combined covariance: P = sum_i w_i * (P_i + (x_i - x)(x_i - x)^T)
    covariance = np.zeros((n, n))
    for i in range(k):
        diff = means[i] - mean
        covariance += weights[i] * (covariances[i] + np.outer(diff, diff))

    # Ensure symmetry
    covariance = (covariance + covariance.T) / 2

    return mean, covariance


def runnalls_merge_cost(
    c1: GaussianComponent,
    c2: GaussianComponent,
) -> float:
    """
    Compute Runnalls' KL-divergence-based merge cost for two components.

    The cost approximates the increase in KL-divergence when merging
    components c1 and c2 into a single moment-matched Gaussian.

    Parameters
    ----------
    c1 : GaussianComponent
        First component.
    c2 : GaussianComponent
        Second component.

    Returns
    -------
    cost : float
        Merge cost (non-negative). Lower cost means components are
        more similar and merging causes less information loss.

    Notes
    -----
    The cost function is:
        cost = 0.5 * [w_m * log|P_m| - w_1 * log|P_1| - w_2 * log|P_2|]

    where w_m = w_1 + w_2, P_m is the moment-matched covariance.
    """
    w1, w2 = c1.weight, c2.weight
    w_merged = w1 + w2

    if w_merged < 1e-15:
        return 0.0

    # Merged mean
    mean_merged = (w1 * c1.mean + w2 * c2.mean) / w_merged

    # Merged covariance
    diff1 = c1.mean - mean_merged
    diff2 = c2.mean - mean_merged
    P_merged = (
        w1 * (c1.covariance + np.outer(diff1, diff1))
        + w2 * (c2.covariance + np.outer(diff2, diff2))
    ) / w_merged

    # Ensure numerical stability
    P_merged = (P_merged + P_merged.T) / 2

    # Compute log determinants with numerical safeguards
    try:
        log_det_merged = np.linalg.slogdet(P_merged)[1]
        log_det_1 = np.linalg.slogdet(c1.covariance)[1]
        log_det_2 = np.linalg.slogdet(c2.covariance)[1]
    except np.linalg.LinAlgError:
        return np.inf

    # Cost: 0.5 * (w_m * log|P_m| - w_1 * log|P_1| - w_2 * log|P_2|)
    cost = 0.5 * (w_merged * log_det_merged - w1 * log_det_1 - w2 * log_det_2)

    return max(0.0, cost)  # Ensure non-negative


def merge_gaussians(
    c1: GaussianComponent,
    c2: GaussianComponent,
) -> MergeResult:
    """
    Merge two Gaussian components via moment matching.

    Parameters
    ----------
    c1 : GaussianComponent
        First component.
    c2 : GaussianComponent
        Second component.

    Returns
    -------
    result : MergeResult
        Merged component and merge cost.

    Examples
    --------
    >>> c1 = GaussianComponent(0.3, np.array([0., 0.]), np.eye(2) * 0.1)
    >>> c2 = GaussianComponent(0.2, np.array([1., 0.]), np.eye(2) * 0.1)
    >>> result = merge_gaussians(c1, c2)
    >>> result.component.weight
    0.5
    """
    w_merged = c1.weight + c2.weight
    cost = runnalls_merge_cost(c1, c2)

    if w_merged < 1e-15:
        # Return first component if both have zero weight
        return MergeResult(c1, 0.0)

    # Merged mean
    mean_merged = (c1.weight * c1.mean + c2.weight * c2.mean) / w_merged

    # Merged covariance
    diff1 = c1.mean - mean_merged
    diff2 = c2.mean - mean_merged
    P_merged = (
        c1.weight * (c1.covariance + np.outer(diff1, diff1))
        + c2.weight * (c2.covariance + np.outer(diff2, diff2))
    ) / w_merged

    # Ensure symmetry
    P_merged = (P_merged + P_merged.T) / 2

    merged = GaussianComponent(w_merged, mean_merged, P_merged)
    return MergeResult(merged, cost)


def prune_mixture(
    components: List[GaussianComponent],
    weight_threshold: float = 1e-5,
) -> List[GaussianComponent]:
    """
    Remove components with weights below threshold and renormalize.

    Parameters
    ----------
    components : list of GaussianComponent
        Input components.
    weight_threshold : float
        Components with weight below this are removed.

    Returns
    -------
    pruned : list of GaussianComponent
        Pruned and renormalized components.

    Examples
    --------
    >>> comps = [
    ...     GaussianComponent(0.9, np.array([0.]), np.array([[1.]])),
    ...     GaussianComponent(1e-6, np.array([10.]), np.array([[1.]])),
    ... ]
    >>> pruned = prune_mixture(comps, weight_threshold=1e-5)
    >>> len(pruned)
    1
    >>> pruned[0].weight  # Renormalized
    1.0
    """
    # Filter by threshold
    surviving = [c for c in components if c.weight >= weight_threshold]

    if not surviving:
        # Keep highest weight component if all would be pruned
        surviving = [max(components, key=lambda c: c.weight)]

    # Renormalize weights
    total_weight = sum(c.weight for c in surviving)
    if total_weight > 0:
        surviving = [
            GaussianComponent(c.weight / total_weight, c.mean, c.covariance)
            for c in surviving
        ]

    return surviving


def reduce_mixture_runnalls(
    components: List[GaussianComponent],
    max_components: int,
    weight_threshold: float = 1e-5,
) -> ReductionResult:
    """
    Reduce mixture using Runnalls' greedy algorithm.

    Iteratively merges the pair of components with the smallest
    KL-divergence merge cost until the target number is reached.

    Parameters
    ----------
    components : list of GaussianComponent
        Input mixture components.
    max_components : int
        Maximum number of components in output.
    weight_threshold : float
        Components below this weight are pruned first.

    Returns
    -------
    result : ReductionResult
        Reduced mixture with cost information.

    References
    ----------
    .. [1] A. R. Runnalls, "Kullback-Leibler Approach to Gaussian Mixture
           Reduction," IEEE Trans. Aerospace and Electronic Systems, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> # 4 components, reduce to 2
    >>> comps = [
    ...     GaussianComponent(0.25, np.array([0., 0.]), np.eye(2) * 0.1),
    ...     GaussianComponent(0.25, np.array([0.1, 0.]), np.eye(2) * 0.1),
    ...     GaussianComponent(0.25, np.array([5., 5.]), np.eye(2) * 0.1),
    ...     GaussianComponent(0.25, np.array([5.1, 5.]), np.eye(2) * 0.1),
    ... ]
    >>> result = reduce_mixture_runnalls(comps, max_components=2)
    >>> len(result.components)
    2
    """
    n_original = len(components)

    if n_original == 0:
        return ReductionResult([], 0, 0, 0.0)

    # First, prune low-weight components
    working = prune_mixture(components, weight_threshold)

    # If already at or below target, return
    if len(working) <= max_components:
        return ReductionResult(working, n_original, len(working), 0.0)

    total_cost = 0.0

    # Greedy merging
    while len(working) > max_components:
        n = len(working)
        min_cost = np.inf
        best_i, best_j = 0, 1

        # Find pair with minimum merge cost
        for i in range(n):
            for j in range(i + 1, n):
                cost = runnalls_merge_cost(working[i], working[j])
                if cost < min_cost:
                    min_cost = cost
                    best_i, best_j = i, j

        # Merge the best pair
        merged = merge_gaussians(working[best_i], working[best_j])
        total_cost += merged.cost

        # Build new list: remove i and j, add merged
        new_working = []
        for k in range(n):
            if k != best_i and k != best_j:
                new_working.append(working[k])
        new_working.append(merged.component)
        working = new_working

    # Renormalize final weights
    total_weight = sum(c.weight for c in working)
    if total_weight > 0:
        working = [
            GaussianComponent(c.weight / total_weight, c.mean, c.covariance)
            for c in working
        ]

    return ReductionResult(working, n_original, len(working), total_cost)


def west_merge_cost(
    c1: GaussianComponent,
    c2: GaussianComponent,
) -> float:
    """
    Compute West's merge cost for two components.

    West's algorithm uses a simpler cost based on weighted Mahalanobis
    distance between component means.

    Parameters
    ----------
    c1 : GaussianComponent
        First component.
    c2 : GaussianComponent
        Second component.

    Returns
    -------
    cost : float
        Merge cost based on weighted mean separation.
    """
    w1, w2 = c1.weight, c2.weight
    w_merged = w1 + w2

    if w_merged < 1e-15:
        return 0.0

    # Mean difference
    diff = c1.mean - c2.mean

    # Use average covariance for distance computation
    P_avg = (w1 * c1.covariance + w2 * c2.covariance) / w_merged

    try:
        # Mahalanobis distance squared
        P_inv = np.linalg.inv(P_avg)
        mahal_sq = diff @ P_inv @ diff
    except np.linalg.LinAlgError:
        return np.inf

    # West's cost: (w1 * w2 / w_merged) * d^2
    cost = (w1 * w2 / w_merged) * mahal_sq

    return max(0.0, cost)


def reduce_mixture_west(
    components: List[GaussianComponent],
    max_components: int,
    weight_threshold: float = 1e-5,
) -> ReductionResult:
    """
    Reduce mixture using West's algorithm.

    Similar to Runnalls' but uses a simpler cost function based on
    weighted Mahalanobis distance between means.

    Parameters
    ----------
    components : list of GaussianComponent
        Input mixture components.
    max_components : int
        Maximum number of components in output.
    weight_threshold : float
        Components below this weight are pruned first.

    Returns
    -------
    result : ReductionResult
        Reduced mixture with cost information.

    References
    ----------
    .. [1] M. West, "Approximating posterior distributions by mixture,"
           Journal of the Royal Statistical Society, Series B, 1993.

    Examples
    --------
    >>> import numpy as np
    >>> comps = [
    ...     GaussianComponent(0.25, np.array([0., 0.]), np.eye(2) * 0.1),
    ...     GaussianComponent(0.25, np.array([0.1, 0.]), np.eye(2) * 0.1),
    ...     GaussianComponent(0.25, np.array([5., 5.]), np.eye(2) * 0.1),
    ...     GaussianComponent(0.25, np.array([5.1, 5.]), np.eye(2) * 0.1),
    ... ]
    >>> result = reduce_mixture_west(comps, max_components=2)
    >>> len(result.components)
    2
    """
    n_original = len(components)

    if n_original == 0:
        return ReductionResult([], 0, 0, 0.0)

    # First, prune low-weight components
    working = prune_mixture(components, weight_threshold)

    # If already at or below target, return
    if len(working) <= max_components:
        return ReductionResult(working, n_original, len(working), 0.0)

    total_cost = 0.0

    # Greedy merging using West's cost
    while len(working) > max_components:
        n = len(working)
        min_cost = np.inf
        best_i, best_j = 0, 1

        # Find pair with minimum merge cost
        for i in range(n):
            for j in range(i + 1, n):
                cost = west_merge_cost(working[i], working[j])
                if cost < min_cost:
                    min_cost = cost
                    best_i, best_j = i, j

        # Merge the best pair
        merged = merge_gaussians(working[best_i], working[best_j])
        total_cost += min_cost

        # Build new list: remove i and j, add merged
        new_working = []
        for k in range(n):
            if k != best_i and k != best_j:
                new_working.append(working[k])
        new_working.append(merged.component)
        working = new_working

    # Renormalize final weights
    total_weight = sum(c.weight for c in working)
    if total_weight > 0:
        working = [
            GaussianComponent(c.weight / total_weight, c.mean, c.covariance)
            for c in working
        ]

    return ReductionResult(working, n_original, len(working), total_cost)


class GaussianMixture:
    """
    Gaussian Mixture representation with reduction operations.

    Provides an object-oriented interface for Gaussian mixture operations
    including density evaluation, sampling, and reduction.

    Parameters
    ----------
    components : list of GaussianComponent, optional
        Initial components. If None, creates empty mixture.

    Attributes
    ----------
    components : list of GaussianComponent
        Current mixture components.

    Examples
    --------
    >>> import numpy as np
    >>> gm = GaussianMixture()
    >>> gm.add_component(0.5, np.array([0., 0.]), np.eye(2) * 0.1)
    >>> gm.add_component(0.5, np.array([2., 2.]), np.eye(2) * 0.1)
    >>> len(gm)
    2
    >>> gm.mean  # Mixture mean
    array([1., 1.])
    """

    def __init__(self, components: Optional[List[GaussianComponent]] = None):
        if components is None:
            self.components: List[GaussianComponent] = []
        else:
            self.components = list(components)

    def __len__(self) -> int:
        return len(self.components)

    def add_component(
        self,
        weight: float,
        mean: ArrayLike,
        covariance: ArrayLike,
    ) -> None:
        """
        Add a component to the mixture.

        Parameters
        ----------
        weight : float
            Component weight.
        mean : array_like
            Mean vector.
        covariance : array_like
            Covariance matrix.
        """
        mean = np.asarray(mean, dtype=np.float64).flatten()
        covariance = np.asarray(covariance, dtype=np.float64)
        self.components.append(GaussianComponent(weight, mean, covariance))

    def normalize_weights(self) -> None:
        """Normalize component weights to sum to 1."""
        total = sum(c.weight for c in self.components)
        if total > 0:
            self.components = [
                GaussianComponent(c.weight / total, c.mean, c.covariance)
                for c in self.components
            ]

    @property
    def weights(self) -> NDArray[np.floating]:
        """Component weights as array."""
        return np.array([c.weight for c in self.components])

    @property
    def means(self) -> List[NDArray[np.floating]]:
        """List of component means."""
        return [c.mean for c in self.components]

    @property
    def covariances(self) -> List[NDArray[np.floating]]:
        """List of component covariances."""
        return [c.covariance for c in self.components]

    @property
    def mean(self) -> NDArray[np.floating]:
        """Mixture mean (moment-matched)."""
        if not self.components:
            raise ValueError("Mixture is empty")
        m, _ = moment_match(self.weights, self.means, self.covariances)
        return m

    @property
    def covariance(self) -> NDArray[np.floating]:
        """Mixture covariance (moment-matched)."""
        if not self.components:
            raise ValueError("Mixture is empty")
        _, P = moment_match(self.weights, self.means, self.covariances)
        return P

    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        if not self.components:
            return 0
        return len(self.components[0].mean)

    def pdf(self, x: ArrayLike) -> float:
        """
        Evaluate mixture probability density at x.

        Parameters
        ----------
        x : array_like
            Point at which to evaluate density.

        Returns
        -------
        density : float
            Mixture probability density.
        """
        x = np.asarray(x, dtype=np.float64).flatten()

        if not self.components:
            return 0.0

        density = 0.0
        for c in self.components:
            density += c.weight * self._gaussian_pdf(x, c.mean, c.covariance)

        return density

    def _gaussian_pdf(
        self,
        x: NDArray[np.floating],
        mean: NDArray[np.floating],
        cov: NDArray[np.floating],
    ) -> float:
        """Evaluate single Gaussian PDF."""
        n = len(x)
        diff = x - mean

        try:
            cov_inv = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0

            mahal_sq = diff @ cov_inv @ diff
            log_pdf = -0.5 * (n * np.log(2 * np.pi) + log_det + mahal_sq)
            return np.exp(log_pdf)
        except np.linalg.LinAlgError:
            return 0.0

    def sample(
        self,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.floating]:
        """
        Draw samples from the mixture.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        samples : ndarray
            Samples, shape (n_samples, n) or (n,) if n_samples=1.
        """
        if not self.components:
            raise ValueError("Mixture is empty")

        if rng is None:
            rng = np.random.default_rng()

        # Choose components based on weights
        weights = self.weights
        weights = weights / weights.sum()  # Normalize
        component_indices = rng.choice(len(self.components), size=n_samples, p=weights)

        # Sample from chosen components
        samples = []
        for idx in component_indices:
            c = self.components[idx]
            sample = rng.multivariate_normal(c.mean, c.covariance)
            samples.append(sample)

        samples = np.array(samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def prune(self, weight_threshold: float = 1e-5) -> "GaussianMixture":
        """
        Remove low-weight components.

        Parameters
        ----------
        weight_threshold : float
            Minimum weight to retain.

        Returns
        -------
        pruned : GaussianMixture
            New mixture with low-weight components removed.
        """
        pruned = prune_mixture(self.components, weight_threshold)
        return GaussianMixture(pruned)

    def reduce_runnalls(self, max_components: int) -> "GaussianMixture":
        """
        Reduce mixture using Runnalls' algorithm.

        Parameters
        ----------
        max_components : int
            Maximum number of components.

        Returns
        -------
        reduced : GaussianMixture
            Reduced mixture.
        """
        result = reduce_mixture_runnalls(self.components, max_components)
        return GaussianMixture(result.components)

    def reduce_west(self, max_components: int) -> "GaussianMixture":
        """
        Reduce mixture using West's algorithm.

        Parameters
        ----------
        max_components : int
            Maximum number of components.

        Returns
        -------
        reduced : GaussianMixture
            Reduced mixture.
        """
        result = reduce_mixture_west(self.components, max_components)
        return GaussianMixture(result.components)

    def copy(self) -> "GaussianMixture":
        """Create a deep copy of the mixture."""
        return GaussianMixture(
            [
                GaussianComponent(c.weight, c.mean.copy(), c.covariance.copy())
                for c in self.components
            ]
        )


__all__ = [
    "GaussianComponent",
    "MergeResult",
    "ReductionResult",
    "moment_match",
    "runnalls_merge_cost",
    "merge_gaussians",
    "prune_mixture",
    "reduce_mixture_runnalls",
    "west_merge_cost",
    "reduce_mixture_west",
    "GaussianMixture",
]
