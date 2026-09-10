"""
Bernoulli filter for joint target existence and state estimation.

The Bernoulli filter is the exact Bayes filter for a target that may or
may not exist: the posterior is a Bernoulli random finite set with
existence probability ``r`` and, conditioned on existence, a spatial
density. This implementation uses a single-Gaussian spatial density with
moment matching of the prediction/update mixtures, the common Kalman
(linear-Gaussian) realization of the filter.

It is the principled replacement for ad-hoc M-of-N track initiation and
termination logic: ``r`` rises as detections consistent with the dynamics
accumulate and decays through missed detections.

References
----------
- B. Ristic, B.-T. Vo, B.-N. Vo, and A. Farina, "A Tutorial on
  Bernoulli Filters: Theory, Implementation and Applications,"
  IEEE Transactions on Signal Processing, 61(13), 2013.
"""

from typing import List, NamedTuple, Optional

import msgspec
import numpy as np
from numpy.typing import ArrayLike, NDArray


class BernoulliConfig(msgspec.Struct, frozen=True):
    """Configuration for the Bernoulli filter.

    Attributes
    ----------
    birth_prob : float
        Probability ``p_b`` that a target is born between scans, given
        it does not currently exist. Default 0.01.
    survival_prob : float
        Probability ``p_s`` that an existing target survives to the
        next scan. Default 0.99.
    detection_prob : float
        Probability of detection ``Pd``. Default 0.9.
    clutter_intensity : float
        Poisson clutter intensity ``kappa`` (expected clutter
        measurements per unit measurement volume), assumed uniform.
        Must be positive: the update equations divide detection
        densities by it. A truly clutter-free sensor can pass a small
        value. Default 1e-4.
    """

    birth_prob: float = 0.01
    survival_prob: float = 0.99
    detection_prob: float = 0.9
    clutter_intensity: float = 1e-4


class BernoulliState(NamedTuple):
    """Bernoulli filter posterior.

    Attributes
    ----------
    r : float
        Probability that the target exists.
    x : ndarray
        State mean, conditioned on existence.
    P : ndarray
        State covariance, conditioned on existence.
    """

    r: float
    x: NDArray[np.floating]
    P: NDArray[np.floating]


def _gaussian_density(
    z: NDArray[np.floating],
    mean: NDArray[np.floating],
    cov: NDArray[np.floating],
) -> float:
    """Multivariate normal density N(z; mean, cov)."""
    m = len(z)
    diff = z - mean
    chol = np.linalg.cholesky(cov)
    alpha = np.linalg.solve(chol, diff)
    mahal_sq = float(alpha @ alpha)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return float(np.exp(-0.5 * (mahal_sq + m * np.log(2.0 * np.pi) + log_det)))


def _moment_match(
    weights: List[float],
    means: List[NDArray[np.floating]],
    covs: List[NDArray[np.floating]],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Collapse a normalized Gaussian mixture to a single Gaussian."""
    x = np.zeros_like(means[0])
    for w, m in zip(weights, means):
        x = x + w * m
    P = np.zeros_like(covs[0])
    for w, m, c in zip(weights, means, covs):
        d = m - x
        P = P + w * (c + np.outer(d, d))
    return x, P


def bernoulli_predict(
    state: BernoulliState,
    F: ArrayLike,
    Q: ArrayLike,
    birth_mean: ArrayLike,
    birth_cov: ArrayLike,
    config: Optional[BernoulliConfig] = None,
) -> BernoulliState:
    """
    Bernoulli filter prediction step.

    The predicted existence probability combines birth and survival,

    .. math::

        r_{k|k-1} = p_b (1 - r_{k-1}) + p_s r_{k-1},

    and the predicted spatial density is the corresponding mixture of
    the birth density and the survived (Kalman-predicted) density,
    moment-matched to a single Gaussian (Ristic et al. 2013, Sec. V).

    Parameters
    ----------
    state : BernoulliState
        Current posterior.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    birth_mean : array_like
        Mean of the birth density.
    birth_cov : array_like
        Covariance of the birth density.
    config : BernoulliConfig, optional
        Filter parameters. Uses defaults if not provided.

    Returns
    -------
    predicted : BernoulliState
        Predicted Bernoulli state.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers.bernoulli import (
    ...     BernoulliConfig, BernoulliState, bernoulli_predict
    ... )
    >>> config = BernoulliConfig(birth_prob=0.1, survival_prob=0.9)
    >>> state = BernoulliState(r=0.5, x=np.zeros(2), P=np.eye(2))
    >>> F = np.eye(2)
    >>> Q = 0.01 * np.eye(2)
    >>> pred = bernoulli_predict(state, F, Q, np.zeros(2), 10 * np.eye(2),
    ...                          config)
    >>> round(pred.r, 3)  # 0.1 * 0.5 + 0.9 * 0.5
    0.5
    """
    cfg = config or BernoulliConfig()
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    x_b = np.asarray(birth_mean, dtype=np.float64)
    P_b = np.asarray(birth_cov, dtype=np.float64)

    r_pred = cfg.birth_prob * (1.0 - state.r) + cfg.survival_prob * state.r

    x_s = F @ state.x
    P_s = F @ state.P @ F.T + Q

    if r_pred <= 0.0:
        # Nothing can exist; carry the birth density as the placeholder
        # spatial density (it is what a future birth would use).
        return BernoulliState(r=0.0, x=x_b, P=P_b)

    w_birth = cfg.birth_prob * (1.0 - state.r) / r_pred
    w_survive = cfg.survival_prob * state.r / r_pred
    x_pred, P_pred = _moment_match([w_birth, w_survive], [x_b, x_s], [P_b, P_s])

    return BernoulliState(r=float(r_pred), x=x_pred, P=P_pred)


def bernoulli_update(
    state: BernoulliState,
    measurements: List[ArrayLike],
    H: ArrayLike,
    R: ArrayLike,
    config: Optional[BernoulliConfig] = None,
) -> BernoulliState:
    """
    Bernoulli filter update step.

    With Poisson clutter of uniform intensity ``kappa`` and detection
    probability ``Pd``, the existence update is (Ristic et al. 2013,
    eq. (87))

    .. math::

        \\delta_k = P_d \\Bigl(1 -
            \\sum_{z} \\frac{q(z)}{\\kappa}\\Bigr), \\qquad
        r_k = r_{k|k-1} \\frac{1 - \\delta_k}{1 - r_{k|k-1} \\delta_k},

    where :math:`q(z) = \\mathcal{N}(z; H x_{k|k-1}, S)` is the
    predicted measurement density. The spatial density becomes a
    mixture of the missed-detection term (weight :math:`1 - P_d`) and
    one Kalman-updated term per measurement (weight
    :math:`P_d\\, q(z)/\\kappa`), normalized and moment-matched to a
    single Gaussian.

    Parameters
    ----------
    state : BernoulliState
        Predicted Bernoulli state.
    measurements : list of array_like
        Measurements received this scan (possibly empty).
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.
    config : BernoulliConfig, optional
        Filter parameters. Uses defaults if not provided.

    Returns
    -------
    updated : BernoulliState
        Updated Bernoulli state.

    Raises
    ------
    ValueError
        If ``config.clutter_intensity`` is not positive: the update
        divides detection densities by it.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers.bernoulli import (
    ...     BernoulliConfig, BernoulliState, bernoulli_update
    ... )
    >>> config = BernoulliConfig(detection_prob=0.9, clutter_intensity=1e-3)
    >>> state = BernoulliState(r=0.5, x=np.zeros(2), P=np.eye(2))
    >>> H = np.eye(2)
    >>> R = 0.1 * np.eye(2)
    >>> upd = bernoulli_update(state, [np.array([0.1, -0.1])], H, R, config)
    >>> upd.r > state.r  # an on-prediction detection raises existence
    True
    >>> miss = bernoulli_update(state, [], H, R, config)
    >>> miss.r < state.r  # no detection lowers existence
    True
    """
    cfg = config or BernoulliConfig()
    if cfg.clutter_intensity <= 0.0:
        raise ValueError(
            "clutter_intensity must be positive: the Bernoulli update "
            "divides detection densities by it. For a clutter-free "
            "sensor pass a small positive value."
        )

    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Z = [np.asarray(z, dtype=np.float64).flatten() for z in measurements]

    z_pred = H @ state.x
    S = H @ state.P @ H.T + R

    q = [_gaussian_density(z, z_pred, S) for z in Z]

    ratio_sum = sum(qi / cfg.clutter_intensity for qi in q)
    delta = cfg.detection_prob * (1.0 - ratio_sum)

    denom = 1.0 - state.r * delta
    r_upd = state.r * (1.0 - delta) / denom
    r_upd = float(min(max(r_upd, 0.0), 1.0))

    # Spatial density: missed-detection term plus one Kalman-updated
    # term per measurement.
    weights = [1.0 - cfg.detection_prob]
    means = [state.x]
    covs = [state.P]

    if Z:
        K = state.P @ H.T @ np.linalg.inv(S)
        P_upd = state.P - K @ S @ K.T
        for z, qi in zip(Z, q):
            weights.append(cfg.detection_prob * qi / cfg.clutter_intensity)
            means.append(state.x + K @ (z - z_pred))
            covs.append(P_upd)

    total = sum(weights)
    weights = [w / total for w in weights]
    x_upd, P_upd_mm = _moment_match(weights, means, covs)

    return BernoulliState(r=r_upd, x=x_upd, P=P_upd_mm)


__all__ = [
    "BernoulliConfig",
    "BernoulliState",
    "bernoulli_predict",
    "bernoulli_update",
]
