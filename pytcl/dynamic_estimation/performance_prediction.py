"""
Tracker performance prediction.

Ports of the MATLAB TCL ``Dynamic_Estimation/Performance_Prediction``
directory: asymptotic Kalman filter covariances under a detection
probability PD <= 1 (the Boers-Driessen modified Riccati equation),
the corresponding asymptotic Fisher information matrices, the
recursive posterior Cramer-Rao lower bound (PCRLB) steps with
additive noise, track purity and correct-association approximations
for closely spaced targets, an untrackability test, and the prior
target-distribution model used to evaluate the PCRLB without Monte
Carlo runs.

The discrete algebraic Riccati equations that the MATLAB originals
solve with ``RiccatiSolveD`` (a QZ-based solver) are solved here with
:func:`scipy.linalg.solve_discrete_are`, which solves the identical
equation; the stabilizing solution is unique, so results agree to
numerical precision.

References
----------
.. [1] Y. Boers and H. Driessen, "Modified Riccati equation and its
   application to target tracking," IEE Proceedings Radar, Sonar and
   Navigation, vol. 153, no. 1, pp. 7-12, Feb. 2006.
.. [2] P. Tichavsky, C. H. Muravchik, and A. Nehorai, "Posterior
   Cramer-Rao bounds for discrete-time nonlinear filtering," IEEE
   Transactions on Signal Processing, vol. 46, no. 5, pp. 1386-1396,
   May 1998.
.. [3] M. Hernandez, B. Ristic, A. Farina, and L. Timmoneri, "A
   comparison of two Cramer-Rao bounds for nonlinear filtering with
   Pd<1," IEEE Transactions on Signal Processing, vol. 52, no. 9,
   pp. 2361-2370, Sep. 2004.
.. [4] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with
   Applications to Tracking and Navigation. New York: John Wiley and
   Sons, Inc, 2001, ch. 5.2.5 and 7.5.
.. [5] D. F. Crouse, "Basic tracking using nonlinear 3D monostatic
   and bistatic measurements," IEEE Aerospace and Electronic Systems
   Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
"""

import warnings
from typing import Callable, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag, solve_discrete_are
from scipy.special import gammaln

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    fifth_order_cubature_points,
    transform_cubature_points,
)


class RiccatiResult(NamedTuple):
    """Asymptotic covariance from a modified Riccati equation.

    Attributes
    ----------
    P : ndarray
        (x_dim, x_dim) average asymptotic error covariance.
    converged : bool
        True if the PD < 1 fixed-point iteration converged (always
        True when PD = 1, where the algebraic solution is used
        directly).
    """

    P: NDArray[np.floating]
    converged: bool


class TrackPurityResult(NamedTuple):
    """Result of :func:`track_purity_lin_approx`.

    Attributes
    ----------
    pc : float
        Average track purity over the batch (probability that an
        assigned measurement belongs to the track).
    p_inv : ndarray
        (x_dim, x_dim) inverse covariance of a typical track at the
        end of the batch.
    """

    pc: float
    p_inv: NDArray[np.floating]


class PriorModel(NamedTuple):
    """Result of :func:`disc_prior_p_model`.

    Attributes
    ----------
    x : ndarray
        (x_dim,) mean of the prior target distribution at step k.
    P : ndarray
        (x_dim, x_dim) covariance of that distribution.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


def _rcond(a: NDArray) -> float:
    """Reciprocal condition number from the singular values (the
    MATLAB originals use the 1-norm estimate; both serve only as a
    singularity guard against a 1e-15 threshold)."""
    sv = np.linalg.svd(a, compute_uv=False)
    if sv[0] == 0.0:
        return 0.0
    return float(sv[-1] / sv[0])


def _elementwise_converged(
    p: NDArray, p_prev: NDArray, rel_tol: float, abs_tol: float
) -> bool:
    diff = np.abs(p - p_prev)
    return bool(np.all((diff <= rel_tol * np.abs(p)) | (diff <= abs_tol)))


def riccati_pred_no_clutter(
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    pd: float = 1.0,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-13,
    max_iter: int = 5000,
) -> RiccatiResult:
    """
    Asymptotic predicted Kalman covariance with detection probability.

    Solves the prior Riccati equation modified for a detection
    probability PD <= 1 [1]_: the average asymptotic covariance of a
    linear Kalman filter just before a (possible) measurement update,

    ``P = F P F' - PD * F P H' (H P H' + R)^-1 H P F' + Q``.

    Parameters
    ----------
    h : array_like
        (z_dim, x_dim) measurement matrix.
    f : array_like
        (x_dim, x_dim) state transition matrix.
    r : array_like
        (z_dim, z_dim) measurement covariance.
    q : array_like
        (x_dim, x_dim) nonsingular process noise covariance.
    pd : float, optional
        Detection probability per scan; default 1.
    rel_tol, abs_tol : float, optional
        Elementwise convergence tolerances for the PD < 1 iteration
        (an element converges when either is met). Defaults 1e-10 and
        1e-13.
    max_iter : int, optional
        Iteration cap; default 5000.

    Returns
    -------
    result : RiccatiResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[10.0]])
    >>> Q = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    >>> res = riccati_pred_no_clutter(H, F, R, Q, 0.5)
    >>> res.converged
    True
    >>> full = riccati_pred_no_clutter(H, F, R, Q, 1.0)
    >>> bool(res.P[0, 0] > full.P[0, 0])
    True

    Notes
    -----
    Port of ``RiccatiPredNoClutter.m``. The original's tolerance
    arguments are tangled (the signature orders them ``AbsTol,
    RelTol`` while the documentation lists the reverse); the port uses
    unambiguous keywords with the original's effective defaults. The
    PD = 1 seed is computed with SciPy's DARE solver instead of
    ``RiccatiSolveD``.
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    f = np.asarray(f, dtype=np.float64)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    q = np.asarray(q, dtype=np.float64)

    # The PD=1 solution: the standard prediction-form DARE.
    p_prev = solve_discrete_are(f.T, h.T, q, r)
    if pd == 1.0:
        return RiccatiResult(p_prev, True)

    for _ in range(max_iter):
        hph_r = h @ p_prev @ h.T + r
        gain_term = np.linalg.solve(hph_r, h) @ p_prev @ f.T
        p = f @ p_prev @ f.T - pd * (f @ p_prev @ h.T) @ gain_term + q
        if _elementwise_converged(p, p_prev, rel_tol, abs_tol):
            return RiccatiResult(p, True)
        p_prev = p
    return RiccatiResult(p_prev, False)


def riccati_post_no_clutter(
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    pd: float = 1.0,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-13,
    max_iter: int = 5000,
) -> RiccatiResult:
    """
    Asymptotic posterior Kalman covariance with detection probability.

    Solves the posterior form of the modified Riccati equation [1]_:
    the average asymptotic covariance of a linear Kalman filter just
    after a (possible with probability PD) measurement update.

    Parameters
    ----------
    h, f, r, q, pd, rel_tol, abs_tol, max_iter
        As in :func:`riccati_pred_no_clutter`.

    Returns
    -------
    result : RiccatiResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[10.0]])
    >>> Q = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    >>> post = riccati_post_no_clutter(H, F, R, Q, 0.5)
    >>> pred = riccati_pred_no_clutter(H, F, R, Q, 0.5)
    >>> bool(post.P[0, 0] < pred.P[0, 0])
    True

    Notes
    -----
    Port of ``RiccatiPostNoClutter.m``. The original errors out when
    called with exactly six arguments (its defaulting logic tests a
    parameter that does not exist yet); the port's keyword arguments
    avoid the tangle. The PD = 1 seed uses SciPy's DARE solver with
    the original's cross-term formulation.
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    f = np.asarray(f, dtype=np.float64)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    q = np.asarray(q, dtype=np.float64)

    p_prev = solve_discrete_are(f.T, f.T @ h.T, q, h @ q @ h.T + r, s=q @ h.T)
    if pd == 1.0:
        return RiccatiResult(p_prev, True)

    for _ in range(max_iter):
        fpf = f @ p_prev @ f.T
        cross = fpf @ h.T + q @ h.T
        s_mat = h @ fpf @ h.T + h @ q @ h.T + r
        p = fpf + q - pd * cross @ np.linalg.solve(s_mat, cross.T)
        p = (p + p.T) / 2.0
        if _elementwise_converged(p, p_prev, rel_tol, abs_tol):
            return RiccatiResult(p, True)
        p_prev = p
    return RiccatiResult(p_prev, False)


def fim_post_no_clutter(
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    pd: float = 1.0,
) -> NDArray[np.floating]:
    """
    Asymptotic posterior Fisher information with detection probability.

    The information-reduction-factor form of the asymptotic FIM after
    a measurement update; its inverse is the asymptotic PCRLB. For
    PD = 1 the inverse equals :func:`riccati_post_no_clutter`'s
    output; for PD < 1 it is smaller (the bound is not tight).

    Parameters
    ----------
    h : array_like
        (z_dim, x_dim) measurement matrix.
    f : array_like
        (x_dim, x_dim) state transition matrix.
    r : array_like
        (z_dim, z_dim) positive definite measurement covariance.
    q : array_like
        (x_dim, x_dim) process noise covariance. A singular q selects
        an iterative recursion; the FIM must then be positive
        definite.
    pd : float, optional
        Detection probability per scan; default 1.

    Returns
    -------
    J : ndarray
        (x_dim, x_dim) asymptotic posterior Fisher information matrix.

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[10.0]])
    >>> Q = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    >>> J = fim_post_no_clutter(H, F, R, Q, 1.0)
    >>> P = riccati_post_no_clutter(H, F, R, Q, 1.0).P
    >>> np.allclose(np.linalg.inv(J), P, rtol=1e-8)
    True

    Notes
    -----
    Port of ``FIMPostNoClutter.m`` (Tichavsky's recursion rewritten as
    a DARE, solved with SciPy).
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    f = np.asarray(f, dtype=np.float64)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    q = np.asarray(q, dtype=np.float64)
    x_dim = h.shape[1]

    if _rcond(q) < 1e-15:
        max_iter = 5000
        rel_tol = 1e-12
        abs_tol = 1e-15
        j_prev = np.eye(x_dim)
        j_z = pd * h.T @ np.linalg.solve(r, h)
        for _ in range(max_iter):
            j = np.linalg.inv(q + f @ np.linalg.solve(j_prev, f.T)) + j_z
            j = (j + j.T) / 2.0
            if _elementwise_converged(j, j_prev, rel_tol, abs_tol):
                return j
            j_prev = j
        warnings.warn(
            "fim_post_no_clutter: max iterations reached without convergence",
            stacklevel=2,
        )
        return j_prev

    q_inv = np.linalg.inv(q)
    d11 = f.T @ q_inv @ f
    d12 = -np.linalg.solve(q.T, f).T  # -F'/Q
    d22 = q_inv + pd * (h.T @ np.linalg.solve(r, h))
    a = np.linalg.solve(d11, d12)
    j = solve_discrete_are(a, np.eye(x_dim), d22 - d12.T @ a, d11)
    return (j + j.T) / 2.0


def fim_pred_no_clutter(
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    pd: float = 1.0,
) -> NDArray[np.floating]:
    """
    Asymptotic predicted Fisher information with detection probability.

    Propagates :func:`fim_post_no_clutter`'s asymptotic posterior FIM
    through one prediction step without a measurement update.

    Parameters
    ----------
    h, f, r, q, pd
        As in :func:`fim_post_no_clutter`.

    Returns
    -------
    J : ndarray
        (x_dim, x_dim) asymptotic prior Fisher information matrix.

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[10.0]])
    >>> Q = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    >>> J = fim_pred_no_clutter(H, F, R, Q, 1.0)
    >>> P = riccati_pred_no_clutter(H, F, R, Q, 1.0).P
    >>> np.allclose(np.linalg.inv(J), P, rtol=1e-8)
    True

    Notes
    -----
    Port of ``FIMPredNoClutter.m``.
    """
    f = np.asarray(f, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    j_post = fim_post_no_clutter(h, f, r, q, pd)

    if _rcond(q) < 1e-15:
        j = np.linalg.inv(q + f @ np.linalg.solve(j_post, f.T))
    else:
        q_inv = np.linalg.inv(q)
        d11 = f.T @ q_inv @ f
        d12 = -np.linalg.solve(q.T, f).T
        j = q_inv - d12.T @ np.linalg.solve(j_post + d11, d12)
    return (j + j.T) / 2.0


def pcrlb_pred_add(
    j_prior: ArrayLike,
    x_prior: Optional[ArrayLike],
    p_prior: Optional[ArrayLike],
    q: ArrayLike,
    f: Union[ArrayLike, Callable],
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    PCRLB Fisher information prediction with additive process noise.

    Propagates a Fisher information matrix through one discrete time
    step. The dynamics enter either as a fixed transition matrix, or
    as a Jacobian function averaged over a Gaussian prior on the true
    target state via cubature integration [5]_.

    Parameters
    ----------
    j_prior : array_like
        (x_dim, x_dim) Fisher information matrix at the previous step.
    x_prior : array_like or None
        (x_dim,) mean of the true-state distribution at the previous
        step; only used when ``f`` is callable.
    p_prior : array_like or None
        (x_dim, x_dim) covariance of that distribution. A zero (or
        None) covariance with callable ``f`` evaluates the Jacobian at
        ``x_prior`` only.
    q : array_like
        (x_dim, x_dim) additive process noise covariance.
    f : array_like or callable
        Fixed (x_dim, x_dim) transition matrix, or a Jacobian function
        ``f(x) -> (x_dim, x_dim)``.
    xi, w : array_like, optional
        Cubature points (num_points, x_dim) and weights for the
        Jacobian average. Default: fifth-order points.

    Returns
    -------
    j_pred : ndarray
        (x_dim, x_dim) predicted Fisher information matrix.

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> Q = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    >>> J = np.diag([2.0, 1.0])
    >>> j_pred = pcrlb_pred_add(J, None, None, Q, F)
    >>> j_pred.shape
    (2, 2)

    Notes
    -----
    Port of ``PCRLBPredAdd.m``, with one deliberate fix of an upstream
    defect: the original's cubature-point arguments can never be used
    (it tests ``nargin<8`` in a seven-argument function, so
    user-supplied points are always discarded and recomputed); the
    port honors ``xi``/``w`` when given.
    """
    j_prior = np.asarray(j_prior, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    x_dim = j_prior.shape[0]

    const_trans = True
    f_mat: NDArray[np.floating]
    f_jac: Optional[Callable] = f if callable(f) else None
    if f_jac is not None:
        p_prior_arr = (
            np.zeros((x_dim, x_dim))
            if p_prior is None
            else np.asarray(p_prior, dtype=np.float64)
        )
        if np.array_equal(p_prior_arr, np.zeros((x_dim, x_dim))):
            f_mat = np.atleast_2d(f_jac(np.asarray(x_prior, dtype=np.float64).ravel()))
        else:
            const_trans = False
    else:
        f_mat = np.asarray(f, dtype=np.float64)

    q_inv = np.linalg.pinv(q)
    if const_trans:
        d12 = -f_mat.T @ q_inv
        d11 = f_mat.T @ q_inv @ f_mat
    else:
        if xi is None:
            xi_arr, w_arr = fifth_order_cubature_points(x_dim)
        else:
            xi_arr = np.asarray(xi, dtype=np.float64)
            w_arr = np.asarray(w, dtype=np.float64).ravel()
        x_prior_arr = np.asarray(x_prior, dtype=np.float64).ravel()
        pts, _ = transform_cubature_points(
            xi_arr, w_arr, x_prior_arr, np.linalg.cholesky(p_prior_arr)
        )
        d12 = np.zeros((x_dim, x_dim))
        d11 = np.zeros((x_dim, x_dim))
        assert f_jac is not None
        for k in range(len(w_arr)):
            f_k = np.atleast_2d(f_jac(pts[k, :]))
            d12 = d12 - w_arr[k] * f_k.T @ q_inv
            d11 = d11 + w_arr[k] * f_k.T @ q_inv @ f_k

    j_pred = q_inv - d12.T @ np.linalg.pinv(j_prior + d11) @ d12
    return (j_pred + j_pred.T) / 2.0


def pcrlb_update_add_no_clutter(
    j_pred: ArrayLike,
    x_cur: Optional[ArrayLike],
    p_cur: Optional[ArrayLike],
    r: ArrayLike,
    pd: float,
    h: Union[ArrayLike, Callable],
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    PCRLB Fisher information measurement update, no clutter.

    Adds one measurement's information (scaled by the detection
    probability, the information reduction factor without clutter
    [3]_) to a predicted Fisher information matrix. The measurement
    model enters as a fixed matrix or as a Jacobian function averaged
    over a Gaussian distribution of the true state.

    Parameters
    ----------
    j_pred : array_like
        (x_dim, x_dim) predicted Fisher information matrix (zeros for
        the first measurement).
    x_cur : array_like or None
        (x_dim,) mean of the true-state distribution at the current
        step; only used when ``h`` is callable.
    p_cur : array_like or None
        (x_dim, x_dim) covariance of that distribution, or None/zeros
        to evaluate the Jacobian at ``x_cur`` only.
    r : array_like
        (z_dim, z_dim) additive measurement noise covariance.
    pd : float
        Detection probability at the current step.
    h : array_like or callable
        Fixed (z_dim, x_dim) measurement matrix, or a Jacobian
        function ``h(x) -> (z_dim, x_dim)``.
    xi, w : array_like, optional
        Cubature points and weights. Default: fifth-order points.

    Returns
    -------
    j_post : ndarray
        (x_dim, x_dim) updated Fisher information matrix.

    Examples
    --------
    >>> import numpy as np
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[10.0]])
    >>> j = pcrlb_update_add_no_clutter(np.zeros((2, 2)), None, None,
    ...                                 R, 0.5, H)
    >>> float(j[0, 0])
    0.05

    Notes
    -----
    Port of ``PCRLBUpdateAddNoClutter.m``. Multiple independent
    measurements at one time can be applied by calling this
    sequentially.
    """
    j_pred = np.asarray(j_pred, dtype=np.float64)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    x_dim = j_pred.shape[0]

    const_meas = True
    h_mat: NDArray[np.floating]
    h_jac: Optional[Callable] = h if callable(h) else None
    if h_jac is not None:
        p_cur_arr = (
            np.zeros((x_dim, x_dim))
            if p_cur is None
            else np.asarray(p_cur, dtype=np.float64)
        )
        if np.array_equal(p_cur_arr, np.zeros((x_dim, x_dim))):
            h_mat = np.atleast_2d(h_jac(np.asarray(x_cur, dtype=np.float64).ravel()))
        else:
            const_meas = False
    else:
        h_mat = np.atleast_2d(np.asarray(h, dtype=np.float64))

    r_inv = np.linalg.pinv(r)
    if const_meas:
        j_post = j_pred + pd * h_mat.T @ r_inv @ h_mat
    else:
        if xi is None:
            xi_arr, w_arr = fifth_order_cubature_points(x_dim)
        else:
            xi_arr = np.asarray(xi, dtype=np.float64)
            w_arr = np.asarray(w, dtype=np.float64).ravel()
        x_cur_arr = np.asarray(x_cur, dtype=np.float64).ravel()
        pts, _ = transform_cubature_points(
            xi_arr, w_arr, x_cur_arr, np.linalg.cholesky(p_cur_arr)
        )
        assert h_jac is not None
        j_meas = np.zeros((x_dim, x_dim))
        for k in range(len(w_arr)):
            h_k = np.atleast_2d(h_jac(pts[k, :]))
            j_meas = j_meas + w_arr[k] * (h_k.T @ r_inv @ h_k)
        j_post = j_pred + pd * j_meas

    return (j_post + j_post.T) / 2.0


def _assoc_volume_constant(m: int) -> float:
    """The constant C_m from the correct-association approximation."""
    gamma_ratio = np.exp(gammaln((m + 1) / 2.0) - gammaln(m / 2.0 + 1.0))
    return float(2.0 ** (m - 1) * np.pi ** ((m - 1) / 2.0) * gamma_ratio)


def correct_assoc_prob_approx(
    z_dim: int,
    beta: float,
    det_s_pred: float,
) -> float:
    """
    Approximate probability of correct measurement-to-track assignment.

    For a cluster of closely spaced targets with density ``beta`` per
    unit measurement volume and average innovation covariance
    determinant ``det_s_pred``, approximates the probability that a
    track is assigned its own measurement (PD = 1, no false alarms).

    Parameters
    ----------
    z_dim : int
        Dimensionality of the measurements.
    beta : float
        Targets per unit volume in the measurement coordinate system.
    det_s_pred : float
        Average determinant of the targets' innovation covariance.

    Returns
    -------
    pc : float
        Approximate correct-association probability.

    Examples
    --------
    >>> pc = correct_assoc_prob_approx(2, 1e-4, 50.0**2)
    >>> bool(0.0 < pc < 1.0)
    True
    >>> correct_assoc_prob_approx(2, 0.0, 50.0**2)
    1.0

    Notes
    -----
    Port of ``correctAssocProbApprox.m``.
    """
    return float(np.exp(-_assoc_volume_constant(z_dim) * beta * np.sqrt(det_s_pred)))


def track_purity_lin_approx(
    beta: float,
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    num_steps: int,
    p_inv_init: ArrayLike,
) -> TrackPurityResult:
    """
    Average track purity for closely spaced targets, linear models.

    Approximates the probability that a measurement assigned to a
    typical track actually belongs to it, averaged over ``num_steps``
    scans from turn-on, for targets of density ``beta`` per unit
    measurement volume (PD = 1, no false alarms). Also returns the
    track's final inverse covariance.

    Parameters
    ----------
    beta : float
        Targets per unit volume in the measurement coordinates.
    h : array_like
        (z_dim, x_dim) measurement matrix.
    f : array_like
        (x_dim, x_dim) invertible state transition matrix.
    r : array_like
        (z_dim, z_dim) measurement covariance.
    q : array_like
        (x_dim, x_dim) process noise covariance.
    num_steps : int
        Number of scans in the batch.
    p_inv_init : array_like
        (x_dim, x_dim) inverse covariance of a track's prior at
        turn-on (before the first measurement update).

    Returns
    -------
    result : TrackPurityResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> res = track_purity_lin_approx(1e-4, H, F, np.array([[100.0]]),
    ...                               0.1 * np.eye(2), 10, np.eye(2))
    >>> bool(0.0 < res.pc <= 1.0)
    True

    Notes
    -----
    Port of ``trackPurityLinApprox.m``, with one deliberate fix of an
    upstream defect: the original's initial update reads
    ``PInvInit+H'*R\\H``, which MATLAB's precedence parses as
    ``(H'*R)\\H`` -- a dimension error whenever z_dim differs from
    x_dim (the usual case), and not the information-form update its
    own loop body applies. The port computes ``H' R^-1 H`` as the loop
    does.
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    f = np.asarray(f, dtype=np.float64)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    q = np.asarray(q, dtype=np.float64)
    m = h.shape[0]
    c_m = _assoc_volume_constant(m)

    p_inv = np.asarray(p_inv_init, dtype=np.float64) + h.T @ np.linalg.solve(r, h)
    pc = 0.0
    for _ in range(num_steps):
        # Information-filter prediction of the inverse covariance.
        p_inv_finv = np.linalg.solve(f.T, p_inv.T).T
        d_inv = f.T + p_inv_finv @ q
        p_inv = np.linalg.solve(d_inv, p_inv_finv)
        # Innovation covariance assuming correct association, then the
        # association-corrupted measurement covariance of Eq. 7.
        q_k = h @ np.linalg.solve(p_inv, h.T)
        s_k = r + q_k
        omega = (2.0 / (m + 2)) * ((m + 1) * q_k - r)
        xi_val = c_m * beta * np.sqrt(np.linalg.det(s_k))
        phi = xi_val * np.exp(-xi_val)
        r_cur = r + omega * phi
        s_k = r_cur + q_k
        pc += float(np.exp(-c_m * beta * np.sqrt(np.linalg.det(s_k))))
        p_inv = p_inv + h.T @ np.linalg.solve(r_cur, h)
    return TrackPurityResult(pc / num_steps, p_inv)


def lin_target_is_untrackable(
    target_params: Union[ArrayLike, Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]],
    pd: float,
    lambda_fa: float,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-13,
    max_iter: int = 5000,
) -> bool:
    """
    Test whether a linear-model target is untrackable.

    A target is untrackable when a track-oriented MHT's best possible
    target-association score never exceeds the missed-detection score,
    given asymptotic track accuracy: ``PD / (lambda_fa *
    sqrt((2 pi)^m det(S)))  <=  1 - PD`` with S the asymptotic
    innovation covariance.

    Parameters
    ----------
    target_params : array_like or tuple
        Either the (z_dim, z_dim) asymptotic innovation covariance
        matrix S, or a tuple ``(h, f, r, q)`` of linear-model
        parameters from which the asymptotic predicted covariance is
        computed via :func:`riccati_pred_no_clutter`.
    pd : float
        Detection probability per scan (should be < 1).
    lambda_fa : float
        False alarm density in the measurement coordinates (false
        alarms per unit volume).
    rel_tol, abs_tol : float, optional
        Tolerances for the Riccati iteration (tuple form only).
    max_iter : int, optional
        Iteration cap for the Riccati iteration (tuple form only).

    Returns
    -------
    untrackable : bool
        True if the target is untrackable.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.diag([100.0, 100.0])
    >>> lin_target_is_untrackable(S, 0.5, 1e-8)
    False
    >>> lin_target_is_untrackable(S, 0.5, 10.0)
    True

    Notes
    -----
    Port of ``linTargetIsUntrackable.m`` (the MATLAB struct input maps
    to the ``(h, f, r, q)`` tuple).
    """
    if isinstance(target_params, tuple):
        h, f, r, q = target_params
        h = np.atleast_2d(np.asarray(h, dtype=np.float64))
        m = h.shape[0]
        p_pred = riccati_pred_no_clutter(
            h, f, r, q, pd, rel_tol=rel_tol, abs_tol=abs_tol, max_iter=max_iter
        ).P
        det_s_pred = float(
            np.linalg.det(h @ p_pred @ h.T + np.atleast_2d(np.asarray(r)))
        )
    else:
        s = np.atleast_2d(np.asarray(target_params, dtype=np.float64))
        m = s.shape[0]
        det_s_pred = float(np.linalg.det(s))

    max_is_target_score = pd / (lambda_fa * np.sqrt((2.0 * np.pi) ** m * det_s_pred))
    missed_detect_score = 1.0 - pd
    return bool(max_is_target_score <= missed_detect_score)


def disc_prior_p_model(
    k: int,
    x_init: ArrayLike,
    f: Optional[ArrayLike] = None,
    q: Optional[ArrayLike] = None,
    T: Optional[float] = None,
    q0: Optional[float] = None,
) -> PriorModel:
    """
    Prior target distribution under a discrete-time linear model.

    Gives the mean and covariance of the target-state distribution at
    discrete time ``k`` when the state at time 0 is known exactly
    (``x_init``) and evolves under a linear model with additive
    process noise -- the prior needed to evaluate the PCRLB for random
    tracks without Monte Carlo runs [5]_.

    Two parameterizations (pass exactly one):

    - ``f``/``q``: an arbitrary transition matrix and process noise
      covariance; the mean is ``F^k x_init`` and the covariance is
      ``sum_{n=0}^{k-1} F^n Q F^n'``.
    - ``T``/``q0``: sample period and power spectral density of a
      3D Cartesian polynomial motion model chosen by the state
      length: 3 (position-only white noise), 6 (discretized
      continuous white noise acceleration), or 9 (jerk), with
      position-major state ordering ``[x, y, z, vx, ...]``.

    Parameters
    ----------
    k : int
        Discrete time step at which the prior is desired.
    x_init : array_like
        (x_dim,) state at time 0.
    f : array_like, optional
        (x_dim, x_dim) transition matrix (generic form).
    q : array_like, optional
        (x_dim, x_dim) process noise covariance (generic form).
    T : float, optional
        Sample period (polynomial-model form).
    q0 : float, optional
        Process noise power spectral density (polynomial-model form).

    Returns
    -------
    result : PriorModel

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> Q = 0.1 * np.eye(2)
    >>> res = disc_prior_p_model(3, [0.0, 1.0], f=F, q=Q)
    >>> res.x
    array([3., 1.])
    >>> res6 = disc_prior_p_model(4, np.arange(6.0), T=0.5, q0=2.0)
    >>> res6.P.shape
    (6, 6)

    Notes
    -----
    Port of ``DiscPriorPModel.m`` (its ``mode`` flag maps onto which
    keyword pair is given). The polynomial-model form supports state
    lengths 3, 6 and 9; the original silently assumes 6 for any
    unlisted length and then fails, so the port raises instead.
    """
    x_init = np.asarray(x_init, dtype=np.float64).ravel()

    if f is not None:
        if q is None:
            raise ValueError("the generic form needs both f and q")
        f = np.asarray(f, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        x_hat = np.linalg.matrix_power(f, k) @ x_init
        x_dim = len(x_init)
        p = np.zeros((x_dim, x_dim))
        for n in range(k):
            f_pow = np.linalg.matrix_power(f, n)
            p += f_pow @ q @ f_pow.T
        return PriorModel(x_hat, (p + p.T) / 2.0)

    if T is None or q0 is None:
        raise ValueError("pass either f and q, or T and q0")

    x_dim = len(x_init)
    if x_dim == 3:
        qt11 = k * T * q0
        p = np.diag([qt11, qt11, qt11])
        return PriorModel(x_init.copy(), p)
    if x_dim == 6:
        f_tilde = np.array([[1.0, k * T], [0.0, 1.0]])
        q_tilde = q0 * np.array(
            [
                [k**3 * T**3 / 3.0, k**2 * T**2 / 2.0],
                [k**2 * T**2 / 2.0, k * T],
            ]
        )
        n_per = 2
    elif x_dim == 9:
        f_tilde = np.array(
            [
                [1.0, k * T, k**2 * T**2 / 2.0],
                [0.0, 1.0, k * T],
                [0.0, 0.0, 1.0],
            ]
        )
        q_tilde = q0 * np.array(
            [
                [k**5 * T**5 / 20.0, k**4 * T**4 / 8.0, k**3 * T**3 / 6.0],
                [k**4 * T**4 / 8.0, k**3 * T**3 / 3.0, k**2 * T**2 / 2.0],
                [k**3 * T**3 / 6.0, k**2 * T**2 / 2.0, k * T],
            ]
        )
        n_per = 3
    else:
        raise ValueError("the polynomial-model form supports state lengths 3, 6 or 9")

    f_sum = block_diag(f_tilde, f_tilde, f_tilde)
    q_sum = block_diag(q_tilde, q_tilde, q_tilde)
    # Reorder from per-axis [pos, deriv, ...] blocks to position-major
    # [x, y, z, vx, vy, vz, ...] ordering.
    sel = np.concatenate([np.arange(d, 3 * n_per, n_per) for d in range(n_per)])
    f_sum = f_sum[np.ix_(sel, sel)]
    q_sum = q_sum[np.ix_(sel, sel)]
    p = (q_sum + q_sum.T) / 2.0
    return PriorModel(f_sum @ x_init, p)


__all__ = [
    "PriorModel",
    "RiccatiResult",
    "TrackPurityResult",
    "correct_assoc_prob_approx",
    "disc_prior_p_model",
    "fim_post_no_clutter",
    "fim_pred_no_clutter",
    "lin_target_is_untrackable",
    "pcrlb_pred_add",
    "pcrlb_update_add_no_clutter",
    "riccati_pred_no_clutter",
    "riccati_post_no_clutter",
    "track_purity_lin_approx",
]
