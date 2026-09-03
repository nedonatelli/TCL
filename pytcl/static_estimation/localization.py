"""
Closed-form static localization estimators.

Ports of the polynomial-free estimators from the MATLAB TCL
``Static_Estimation`` directory: TDOA least-squares emitter localization,
bistatic range-only localization, range-rate-only velocity estimation,
and an ad-hoc Cartesian covariance from radar sensor parameters.

References
----------
.. [1] M. D. Gillette and H. F. Silverman, "A linear closed-form algorithm
   for source localization from time-differences of arrival," IEEE Signal
   Processing Letters, vol. 15, pp. 1-4, 2008.
.. [2] M. Malanowski and K. Kulpa, "Two methods for target localization in
   multistatic passive radar," IEEE Transactions on Aerospace and
   Electronic Systems, vol. 48, no. 1, pp. 572-580, Jan. 2012.
.. [3] D. F. Crouse, "Basic tracking using nonlinear 3D monostatic and
   bistatic measurements," IEEE Aerospace and Electronic Systems
   Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
"""

from typing import NamedTuple, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.coordinate_systems.rotations.rotations import rot_axis_to_vec
from pytcl.core.constants import SPEED_OF_LIGHT
from pytcl.mathematical_functions.polynomials import poly_roots_multi_dim


def _create_a_and_w(
    ref_rx_loc: NDArray[np.floating],
    non_ref_rx_locs: NDArray[np.floating],
    time_delays: NDArray[np.floating],
    c: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """One reference receiver's block of the TDOA linear system."""
    dm0 = time_delays * c
    w = 0.5 * (dm0**2 - np.sum(non_ref_rx_locs**2, axis=0) + np.sum(ref_rx_loc**2))
    num_rx = non_ref_rx_locs.shape[1]
    A = np.zeros((num_rx, 4))
    A[:, :3] = ref_rx_loc[np.newaxis, :] - non_ref_rx_locs.T
    A[:, 3] = dm0
    return A, w


def tdoa_only_static_loc_est(
    time_delays: Union[ArrayLike, Sequence[ArrayLike]],
    ref_rx_locs: ArrayLike,
    non_ref_rx_locs: Union[ArrayLike, Sequence[ArrayLike]],
    c: float = SPEED_OF_LIGHT,
) -> NDArray[np.floating]:
    """
    Closed-form least-squares emitter location from TDOA measurements.

    A minimum of one reference receiver and four TDOA measurements is
    needed for observability in 3D. For minimal (exactly-determined)
    systems use ``tdoa_to_cart`` instead (not yet ported).

    Parameters
    ----------
    time_delays : array_like or sequence of array_like
        With a single reference receiver, an (n,) vector of time
        differences between each receiver and the reference. With
        multiple references, a sequence whose i-th element holds the
        delay vector for the receivers paired with the i-th reference.
        The form (array or sequence of arrays) must match
        ``non_ref_rx_locs``.
    ref_rx_locs : array_like
        (3,) location of the single reference receiver, or (3, num_refs)
        locations of all reference receivers.
    non_ref_rx_locs : array_like or sequence of array_like
        With a single reference, a (3, n) matrix of receiver locations.
        With multiple references, a sequence whose i-th element is the
        (3, n_i) matrix of receivers paired with the i-th reference.
    c : float, optional
        Propagation speed. Default: speed of light.

    Returns
    -------
    source_loc : ndarray
        (3,) emitter location. Exact in an error-free setting; otherwise
        a least-squares solution with respect to a non-standard cost
        function.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([27.0, 0.0, -42.0])
    >>> ref = np.array([9.0, 39.0, 100.0])
    >>> rx = np.array([[65.0, 64.0, -128.0, 0.0],
    ...                [10.0, 71.0, 6.0, -20.0],
    ...                [-60.0, 43.0, 12.0, 4.0]])
    >>> c = 341.0
    >>> tdoa = (np.linalg.norm(t[:, None] - rx, axis=0)
    ...         - np.linalg.norm(t - ref)) / c
    >>> np.round(tdoa_only_static_loc_est(tdoa, ref, rx, c), 9)
    array([ 27.,   0., -42.])

    Notes
    -----
    Port of ``TDOAOnlyStaticLocEst.m``, implementing the linear
    closed-form algorithm of [1]_.
    """
    ref_arr = np.asarray(ref_rx_locs, dtype=np.float64)

    if isinstance(non_ref_rx_locs, np.ndarray) or (
        not isinstance(non_ref_rx_locs, (list, tuple))
    ):
        non_ref = np.asarray(non_ref_rx_locs, dtype=np.float64)
        delays = np.asarray(time_delays, dtype=np.float64)
        A, w = _create_a_and_w(ref_arr, non_ref, delays, c)
    else:
        num_refs = ref_arr.shape[1]
        blocks = []
        ws = []
        for i in range(num_refs):
            non_ref = np.asarray(non_ref_rx_locs[i], dtype=np.float64)
            delays = np.asarray(time_delays[i], dtype=np.float64)
            a_cur, w_cur = _create_a_and_w(ref_arr[:, i], non_ref, delays, c)
            blocks.append(a_cur)
            ws.append(w_cur)
        total_rx = sum(b.shape[0] for b in blocks)
        A = np.zeros((total_rx, 3 + num_refs))
        w = np.concatenate(ws)
        row = 0
        for i, a_cur in enumerate(blocks):
            n = a_cur.shape[0]
            A[row : row + n, :3] = a_cur[:, :3]
            A[row : row + n, 3 + i] = a_cur[:, 3]
            row += n

    if A.shape[0] < 4:
        raise ValueError(
            "Not enough received signals to solve the problem. A minimum "
            "of four TDOA measurements is required."
        )

    xs = np.linalg.pinv(A) @ w
    return xs[:3]


class RangeOnlyLocEst(NamedTuple):
    """Result of :func:`range_only_static_loc_est_np`.

    Attributes
    ----------
    x_est : ndarray
        (3,) Cartesian location estimate, or (3, 2) holding both
        solutions when only the minimal three measurements are given.
    p_taylor : ndarray or None
        (3, 3, num_sol) Taylor-series covariance(s), present when a
        measurement covariance was supplied.
    p_crlb : ndarray or None
        (3, 3, num_sol) Cramer-Rao lower bound covariance(s), present
        when a measurement covariance was supplied.
    """

    x_est: NDArray[np.floating]
    p_taylor: Optional[NDArray[np.floating]]
    p_crlb: Optional[NDArray[np.floating]]


def range_only_static_loc_est_np(
    r_bi: ArrayLike,
    z_loc1: ArrayLike,
    z_loc2: ArrayLike,
    method: int = 1,
    r_cov: Optional[ArrayLike] = None,
) -> RangeOnlyLocEst:
    """
    Target location in 3D from bistatic range-only measurements.

    One receiver and multiple transmitters (or vice versa); the sensors
    cannot all be coplanar. With noisy measurements, results degrade as
    the geometry approaches coplanarity.

    Parameters
    ----------
    r_bi : array_like
        (num_meas,) bistatic range measurements, num_meas >= 3.
    z_loc1 : array_like
        (3, num_meas) transmitter locations (with one receiver), or
        receiver locations (with one transmitter).
    z_loc2 : array_like
        (3,) location of the single receiver (or transmitter). It may
        not be collocated with any sensor in ``z_loc1``.
    method : int, optional
        0 for the spherical-interpolation method of [2]_ (requires
        num_meas > 3), 1 (default) for the spherical-intersection
        technique of [2]_.
    r_cov : array_like, optional
        (num_meas, num_meas) measurement covariance. When given, the
        Taylor-series and CRLB covariances are computed (method 1 only,
        as in the original).

    Returns
    -------
    result : RangeOnlyLocEst
        Location estimate and, when ``r_cov`` was supplied, the two
        covariance estimates.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([4e3, -2e3, 3e3])
    >>> rx = np.array([100.0, 200.0, -50.0])
    >>> tx = np.array([[0.0, 8e3, -6e3, 2e3, -3e3],
    ...                [0.0, 1e3, 5e3, -7e3, 2e3],
    ...                [0.0, -2e3, 1e3, 4e3, 9e3]])
    >>> r = np.linalg.norm(t[:, None] - tx, axis=0) + np.linalg.norm(t - rx)
    >>> np.round(range_only_static_loc_est_np(r, tx, rx).x_est, 6)
    array([ 4000., -2000.,  3000.])

    Notes
    -----
    Port of ``rangeOnlyStaticLocEstNP.m``. Two behaviors of the original
    are preserved deliberately: covariance outputs are only defined for
    method 1 (the original references variables that method 0 never
    creates), and the covariance of a uniquely-selected solution is
    linearized about solution 1's position even when solution 2 was the
    one selected (the original uses ``xEst1`` in ``Delta`` regardless of
    which solution won the residual comparison).
    """
    r_bi = np.asarray(r_bi, dtype=np.float64).ravel()
    z_loc1 = np.asarray(z_loc1, dtype=np.float64)
    z_loc2 = np.asarray(z_loc2, dtype=np.float64).ravel()

    meas_dim = len(r_bi)
    if meas_dim < 3:
        raise ValueError("A minimum of three measurements are required.")

    # Move the receiver to the origin.
    z_tx = z_loc1 - z_loc2[:, np.newaxis]

    S = z_tx.T
    s_star = np.linalg.pinv(S)

    # Equation 10 of [2].
    z = 0.5 * (np.sum(S * S, axis=1) - r_bi**2)

    x_est2: Optional[NDArray[np.floating]] = None
    if method == 0:
        if meas_dim == 3:
            raise ValueError("Method 0 does not work with num_meas == 3.")
        T = np.eye(meas_dim) - S @ s_star
        # Equation 16.
        r_t = -(r_bi @ T @ z) / (r_bi @ T @ r_bi)
        x_est = s_star @ (z + r_bi * r_t)
        num_sol = 1
    elif method == 1:
        a = s_star @ z  # Equation 17.
        b = s_star @ r_bi  # Equation 18.

        # Equation 21; the real part adds robustness to noise.
        root_term = np.real(
            np.sqrt(complex(4.0 * (a @ b) ** 2 - 4.0 * ((b @ b) - 1.0) * (a @ a)))
        )
        denom = 2.0 * (b @ b - 1.0)
        r_mono1 = (-2.0 * a @ b - root_term) / denom
        r_mono2 = (-2.0 * a @ b + root_term) / denom

        # Equation 19, with the residual norm choosing between the two
        # candidate solutions in the overdetermined case.
        x_est1 = a + b * r_mono1
        diff = x_est1[:, np.newaxis] - z_tx
        d1 = np.linalg.norm(
            r_bi - np.linalg.norm(x_est1) - np.sqrt(np.sum(diff * diff, axis=0))
        )

        x_est2 = a + b * r_mono2
        diff = x_est2[:, np.newaxis] - z_tx
        d2 = np.linalg.norm(
            r_bi - np.linalg.norm(x_est2) - np.sqrt(np.sum(diff * diff, axis=0))
        )

        if meas_dim == 3:
            x_est = np.column_stack((x_est1, x_est2))
            rt1 = r_mono1
            rt2 = r_mono2
            num_sol = 2
        else:
            if d1 < d2:
                x_est = x_est1
                rt1 = r_mono1
            else:
                x_est = x_est2
                rt1 = r_mono2
            num_sol = 1
    else:
        raise ValueError("Unknown method specified.")

    p_taylor: Optional[NDArray[np.floating]] = None
    p_crlb: Optional[NDArray[np.floating]] = None
    if r_cov is not None:
        r_cov_arr = np.asarray(r_cov, dtype=np.float64)
        p_taylor = np.zeros((3, 3, num_sol))
        p_crlb = np.zeros((3, 3, num_sol))

        delta = S - np.outer(r_bi, x_est1) / np.linalg.norm(x_est1)
        gamma = np.diag(r_bi)
        dxdr = np.linalg.lstsq(delta, np.eye(meas_dim) * rt1 - gamma, rcond=None)[0]
        p_taylor[:, :, 0] = dxdr @ r_cov_arr @ dxdr.T
        p_crlb[:, :, 0] = np.linalg.pinv(dxdr @ np.linalg.pinv(r_cov_arr) @ dxdr.T)

        if num_sol > 1:
            delta = S - np.outer(r_bi, x_est2) / np.linalg.norm(x_est2)
            dxdr = np.linalg.lstsq(delta, np.eye(meas_dim) * rt2 - gamma, rcond=None)[0]
            p_taylor[:, :, 1] = dxdr @ r_cov_arr @ dxdr.T
            p_crlb[:, :, 1] = np.linalg.pinv(dxdr @ np.linalg.pinv(r_cov_arr) @ dxdr.T)

    # Adjust for the receiver not being at the origin.
    if x_est.ndim == 1:
        x_est = x_est + z_loc2
    else:
        x_est = x_est + z_loc2[:, np.newaxis]

    return RangeOnlyLocEst(x_est, p_taylor, p_crlb)


def rr_only_static_vel_est(
    rr: ArrayLike,
    x_tx: Optional[ArrayLike],
    x_rx: ArrayLike,
    z_tar: ArrayLike,
    use_half_range: bool = False,
) -> NDArray[np.floating]:
    """
    Least-squares target velocity from bistatic range-rate measurements.

    Works in 2D and 3D; produces a least-squares estimate when more than
    the minimum number of measurements (2 in 2D, 3 in 3D) is given. Uses
    a non-relativistic model and ignores atmospheric effects.

    Parameters
    ----------
    rr : array_like
        (num_meas,) range rates.
    x_tx : array_like or None
        (2*d, num_meas) stacked transmitter position/velocity states, or
        a single (2*d,) state shared by all measurements. Pass None when
        the target itself is the transmitter (an emitter).
    x_rx : array_like
        (2*d, num_meas) stacked receiver states, or a single (2*d,)
        state shared by all measurements.
    z_tar : array_like
        (d,) Cartesian target position.
    use_half_range : bool, optional
        True if the range rates are one-way (monostatic convention);
        they are doubled internally. Default False.

    Returns
    -------
    v_est : ndarray
        (d,) least-squares Cartesian velocity estimate.

    Examples
    --------
    An emitter (the target is the transmitter) observed by three moving
    receivers; error-free one-way range rates recover its velocity:

    >>> import numpy as np
    >>> z_tar = np.array([1.5, -0.4, 2.2])
    >>> v_tar = np.array([0.3, 1.1, -0.7])
    >>> x_rx = np.array([[0.5, -1.2, 2.0],
    ...                  [1.0, 0.3, -1.5],
    ...                  [-0.6, 1.8, 0.4],
    ...                  [0.1, -0.5, 0.7],
    ...                  [-0.2, 0.4, 0.1],
    ...                  [0.3, 0.2, -0.4]])
    >>> h = z_tar[:, None] - x_rx[:3]
    >>> h = h / np.linalg.norm(h, axis=0)
    >>> rr = np.sum(h * (v_tar[:, None] - x_rx[3:]), axis=0)
    >>> np.round(rr_only_static_vel_est(rr, None, x_rx, z_tar), 9)
    array([ 0.3,  1.1, -0.7])

    Notes
    -----
    Port of ``RROnlyStaticVelEst.m``, implementing Equation 41 in
    Section IV E of [3]_, with the target-is-transmitter case handled
    specially to remove the singularity.
    """
    rr = np.asarray(rr, dtype=np.float64).ravel()
    if use_half_range:
        rr = 2.0 * rr

    num_meas = len(rr)
    x_rx = np.asarray(x_rx, dtype=np.float64)
    if x_rx.ndim == 1:
        x_rx = x_rx[:, np.newaxis]
    if x_rx.shape[1] == 1:
        x_rx = np.tile(x_rx, (1, num_meas))

    z_tar = np.asarray(z_tar, dtype=np.float64).ravel()
    pos_dim = len(z_tar)

    z_rx = x_rx[:pos_dim, :]
    v_rx = x_rx[pos_dim : 2 * pos_dim, :]

    h = z_tar[:, np.newaxis] - z_rx
    h = h / np.linalg.norm(h, axis=0)

    if x_tx is not None:
        x_tx_arr = np.asarray(x_tx, dtype=np.float64)
        if x_tx_arr.ndim == 1:
            x_tx_arr = x_tx_arr[:, np.newaxis]
        if x_tx_arr.shape[1] == 1:
            x_tx_arr = np.tile(x_tx_arr, (1, num_meas))
        z_tx = x_tx_arr[:pos_dim, :]
        v_tx = x_tx_arr[pos_dim : 2 * pos_dim, :]
        hi = z_tar[:, np.newaxis] - z_tx
        hi = hi / np.linalg.norm(hi, axis=0)

        r_dot_b = rr + np.sum(h * v_rx, axis=0) + np.sum(hi * v_tx, axis=0)
        Hv = h.T + hi.T
    else:
        # The target is the transmitter.
        r_dot_b = rr + np.sum(h * v_rx, axis=0)
        Hv = h.T

    return np.linalg.lstsq(Hv, r_dot_b, rcond=None)[0]


def ad_hoc_cart_cov(
    bandwidth: float,
    beamwidth: ArrayLike,
    snr: float,
    x: Optional[ArrayLike] = None,
    dim: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Ad-hoc Cartesian covariance from radar sensor parameters.

    Builds a 2D or 3D covariance whose principal axes are the range and
    cross-range resolutions at the estimated target location, rotated
    from the x-axis into the target direction.

    Parameters
    ----------
    bandwidth : float
        Radar bandwidth in Hz.
    beamwidth : array_like
        Scalar beamwidth (azimuth and elevation equal), or a length-2
        vector [azimuth, elevation], in radians.
    snr : float
        Signal-to-noise ratio. As in the original, the value enters the
        range-resolution formula directly (the MATLAB documentation
        calls it dB but the code applies no conversion).
    x : array_like, optional
        (2,) or (3,) estimated Cartesian target location.
        Default [1, 0, 0].
    dim : int, optional
        2 for polar (range, azimuth) or 3 for spherical measurements.
        Default: the dimensionality of ``x``.

    Returns
    -------
    V : ndarray
        (dim, dim) covariance matrix.

    Notes
    -----
    Port of ``getAdHocCartCov.m``.

    Examples
    --------
    >>> import numpy as np
    >>> V = ad_hoc_cart_cov(5e6, [np.deg2rad(2), np.deg2rad(10)], 10.0,
    ...                     [1e3, 1e3, 1e3])
    >>> V.shape
    (3, 3)
    >>> bool(np.allclose(V, V.T)) and bool(np.all(np.linalg.eigvalsh(V) > 0))
    True
    """
    if x is None:
        x_arr = np.array([1.0, 0.0, 0.0])
    else:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
    if dim is None:
        dim = len(x_arr)

    beamwidth_arr = np.atleast_1d(np.asarray(beamwidth, dtype=np.float64))
    az_beamwidth = beamwidth_arr[0]
    el_beamwidth = beamwidth_arr[1] if len(beamwidth_arr) == 2 else az_beamwidth

    r = np.linalg.norm(x_arr[:dim])
    range_res = SPEED_OF_LIGHT / (2.0 * bandwidth * np.sqrt(2.0 * snr))
    az_angle_res = 2.0 * r * np.sin(az_beamwidth / 2.0)
    el_angle_res = 2.0 * r * np.sin(el_beamwidth / 2.0)

    R2 = rot_axis_to_vec(x_arr[:dim], "x")
    if dim == 3:
        V = np.diag(
            [
                (range_res / 2.0) ** 2,
                (az_angle_res / 2.0) ** 2,
                (el_angle_res / 2.0) ** 2,
            ]
        )
        theta = np.pi - np.arctan2(R2[2, 1], R2[2, 2])
        c, s = np.cos(theta), np.sin(theta)
        R1 = np.array([[c, -s], [s, c]])
        V[1:3, 1:3] = R1 @ V[1:3, 1:3] @ R1.T
    elif dim == 2:
        V = np.diag([(range_res / 2.0) ** 2, (az_angle_res / 2.0) ** 2])
    else:
        raise ValueError("dim must be 2 or 3")

    return R2 @ V @ R2.T


class PolyStaticEst(NamedTuple):
    """Result of a polynomial-solver-based static estimator.

    Attributes
    ----------
    z_cart : ndarray
        (dim, num_sol) real Cartesian solutions that survived the
        complex and sign filters. Geometric ambiguity generally leaves
        more than one column.
    exit_code : int
        Exit code of :func:`~pytcl.mathematical_functions.polynomials.\
poly_roots_multi_dim` (0 on success).
    """

    z_cart: NDArray[np.floating]
    exit_code: int


def _real_solutions(roots, abs_tol, rel_tol):
    """MATLAB's complex-solution filter: a column is kept when ANY of
    its coordinates is numerically real (a preserved upstream quirk —
    the subsequent sign filter removes most of what slips through)."""
    keep = (
        np.sum(
            (np.abs(roots.imag) < abs_tol)
            | (np.abs(roots.imag) < rel_tol * np.abs(roots.real)),
            axis=0,
        )
        != 0
    )
    return roots[:, keep].real


def tdoa_to_cart(
    tdoa: ArrayLike,
    l_rx1: ArrayLike,
    l_rx2: ArrayLike,
    c: float = SPEED_OF_LIGHT,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-7,
    max_deg_increases: Optional[int] = None,
    use_motzkin_null: bool = False,
) -> PolyStaticEst:
    """
    Target location from a minimal set of TDOA measurements.

    Exactly 2 measurements in 2D or 3 in 3D — the minimal number for
    observability, unlike the overdetermined least-squares
    :func:`tdoa_only_static_loc_est`. The hyperbolic equations are
    turned into simultaneous multivariate polynomials and solved with
    :func:`~pytcl.mathematical_functions.polynomials.poly_roots_multi_dim`.

    Parameters
    ----------
    tdoa : array_like
        (dim,) time differences of arrival; ``tdoa[i]`` is the arrival
        time at ``l_rx2[:, i]`` minus the arrival time at
        ``l_rx1[:, i]``.
    l_rx1 : array_like
        (dim, dim) reference sensor positions, one column per
        measurement, or a single (dim,) position shared by all
        measurements.
    l_rx2 : array_like
        (dim, dim) non-reference sensor positions.
    c : float, optional
        Propagation speed. Default: speed of light.
    abs_tol, rel_tol : float, optional
        Tolerances used both to decide whether a root is numerically
        real and to discard sign-flipped ghost solutions introduced by
        the squaring. Defaults 1e-9 and 1e-7.
    max_deg_increases : int, optional
        Passed through to the polynomial solver.
    use_motzkin_null : bool, optional
        Passed through to the polynomial solver. Default False.

    Returns
    -------
    result : PolyStaticEst
        Real solutions and the solver exit code. As in the original, if
        the sign filter would discard every candidate, the first one is
        kept anyway.

    Examples
    --------
    >>> import numpy as np
    >>> S1 = np.array([9.0, 39.0, 100.0])
    >>> S2 = np.array([65.0, 10.0, -60.0])
    >>> S3 = np.array([64.0, 71.0, 43.0])
    >>> S4 = np.array([-128.0, 6.0, 12.0])
    >>> t = np.array([27.0, 0.0, -42.0])
    >>> c = 341.0
    >>> d = lambda a, b: np.linalg.norm(t - a) - np.linalg.norm(t - b)
    >>> tdoa = np.array([d(S2, S1), d(S3, S1), d(S3, S4)]) / c
    >>> l_rx1 = np.column_stack([S1, S1, S4])
    >>> l_rx2 = np.column_stack([S2, S3, S3])
    >>> res = tdoa_to_cart(tdoa, l_rx1, l_rx2, c)
    >>> np.round(res.z_cart[:, 0], 6)
    array([ 27.,  -0., -42.])

    Notes
    -----
    Port of ``TDOA2Cart.m``, following the polynomial formulation of
    M. P. Williams, "Solving polynomial equations using linear
    algebra," Johns Hopkins Technical Digest, vol. 28, no. 4,
    pp. 354-363, 2010 (with the sign-of-u typo of the paper fixed, as
    in the original).
    """
    tdoa = np.asarray(tdoa, dtype=np.float64).ravel()
    l_rx1 = np.asarray(l_rx1, dtype=np.float64)
    l_rx2 = np.asarray(l_rx2, dtype=np.float64)

    num_dim = l_rx1.shape[0]
    if l_rx1.ndim == 1 or l_rx1.shape[1] == 1:
        l_rx1 = np.tile(l_rx1.reshape(num_dim, 1), (1, num_dim))

    # The equations use the opposite naming, so swap.
    l_rx1, l_rx2 = l_rx2, l_rx1

    u = (l_rx1 + l_rx2) / 2.0
    v = (l_rx1 - l_rx2) / 2.0

    x_polys = []
    if num_dim == 2:
        for k in range(2):
            u1, u2 = u[:, k]
            v1, v2 = v[:, k]
            delta = c * tdoa[k] / 2.0
            xp = np.zeros((3, 3))
            xp[1, 0] = -2.0 * (u1 * v1**2 + u2 * v1 * v2 - u1 * delta**2)
            xp[2, 0] = v1**2 - delta**2
            xp[0, 1] = -2.0 * (u1 * v1 * v2 + u2 * v2**2 - u2 * delta**2)
            xp[1, 1] = 2.0 * v1 * v2
            xp[0, 2] = v2**2 - delta**2
            xp[0, 0] = (
                -(u1**2) * delta**2
                - u2**2 * delta**2
                - v1**2 * delta**2
                - v2**2 * delta**2
                + u1**2 * v1**2
                + 2.0 * u1 * u2 * v1 * v2
                + u2**2 * v2**2
                + delta**4
            )
            x_polys.append(xp)
    elif num_dim == 3:
        for k in range(3):
            u1, u2, u3 = u[:, k]
            v1, v2, v3 = v[:, k]
            delta = c * tdoa[k] / 2.0
            uv = u1 * v1 + u2 * v2 + u3 * v3
            xp = np.zeros((3, 3, 3))
            xp[1, 0, 0] = -2.0 * v1 * uv + 2.0 * u1 * delta**2
            xp[2, 0, 0] = (v1 - delta) * (v1 + delta)
            xp[0, 1, 0] = -2.0 * v2 * uv + 2.0 * u2 * delta**2
            xp[0, 2, 0] = (v2 - delta) * (v2 + delta)
            xp[1, 1, 0] = 2.0 * v1 * v2
            xp[0, 0, 1] = -2.0 * v3 * uv + 2.0 * u3 * delta**2
            xp[1, 0, 1] = 2.0 * v1 * v3
            xp[0, 1, 1] = 2.0 * v2 * v3
            xp[0, 0, 2] = (v3 - delta) * (v3 + delta)
            xp[0, 0, 0] = (
                uv**2
                - (u1**2 + u2**2 + u3**2 + v1**2 + v2**2 + v3**2) * delta**2
                + delta**4
            )
            x_polys.append(xp)
    else:
        raise ValueError("The dimensionality of the locations is invalid.")

    the_roots, exit_code = poly_roots_multi_dim(
        x_polys, max_deg_increases, use_motzkin_null
    )
    z_cart = _real_solutions(the_roots, abs_tol, rel_tol)

    # The squaring makes sign-flipped ghosts; discard solutions whose
    # recomputed TDOA disagrees, judged by the same tolerances so TDOA
    # values near zero do not spuriously fail a sign comparison.
    num_sol = z_cart.shape[1]
    keep = np.ones(num_sol, dtype=bool)
    for s in range(num_sol):
        for k in range(num_dim):
            tdoa_comp = (
                np.linalg.norm(z_cart[:, s] - l_rx1[:, k])
                - np.linalg.norm(z_cart[:, s] - l_rx2[:, k])
            ) / c
            abs_diff = abs(tdoa_comp - tdoa[k])
            if not (abs_diff < abs_tol or abs_diff < rel_tol * abs(tdoa[k])):
                keep[s] = False
                break

    # As in the original: never return an empty set if candidates
    # existed.
    if num_sol > 0 and not keep.any():
        keep[0] = True

    return PolyStaticEst(z_cart[:, keep], exit_code)


def range_rate_to_static_pos(
    rr: ArrayLike,
    s_rx: ArrayLike,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-7,
    max_deg_increases: Optional[int] = None,
    use_motzkin_null: bool = False,
) -> PolyStaticEst:
    """
    Stationary-emitter location from minimal range-rate measurements.

    Given the minimum number of range rates for observability (2 in
    2D, 3 in 3D) from moving receivers, locate a stationary emitter —
    e.g. drones taking Doppler measurements of a stationary phone with
    a known broadcast frequency. None of the receivers may be
    stationary.

    Parameters
    ----------
    rr : array_like
        (dim,) range-rate measurements.
    s_rx : array_like
        (2*dim, dim) stacked receiver position and velocity per
        measurement; ``s_rx[:, i] = [x, y(, z), xdot, ydot(, zdot)]``.
    abs_tol, rel_tol : float, optional
        Tolerances for the numerically-real and sign-consistency
        filters. Defaults 1e-9 and 1e-7.
    max_deg_increases, use_motzkin_null : optional
        Passed through to the polynomial solver.

    Returns
    -------
    result : PolyStaticEst
        Real solutions (the true emitter plus geometric ambiguities)
        and the solver exit code.

    Examples
    --------
    >>> import numpy as np
    >>> u_true = np.array([1e3, 5e3])
    >>> s = np.array([[500.0, 1100.0], [2500.0, 2500.0]])
    >>> s_dot = np.array([[300.0, 300.0], [0.0, 0.0]])
    >>> rr = np.array(
    ...     [
    ...         -s_dot[:, k] @ (u_true - s[:, k]) / np.linalg.norm(u_true - s[:, k])
    ...         for k in range(2)
    ...     ]
    ... )
    >>> res = range_rate_to_static_pos(rr, np.vstack([s, s_dot]))
    >>> bool(
    ...     np.min(np.linalg.norm(res.z_cart - u_true[:, None], axis=0)) < 1e-3
    ... )
    True

    Notes
    -----
    Port of ``rangeRate2StaticPos.m``, implementing concepts of
    D. F. Crouse, "General multivariate polynomial target localization
    and initial estimation," Journal of Advances in Information
    Fusion, vol. 13, no. 1, pp. 68-91, Jun. 2018. The 3D coefficient
    hypermatrices are scaled by 1e-3 exactly as in the original.
    """
    rr = np.asarray(rr, dtype=np.float64).ravel()
    s_rx = np.asarray(s_rx, dtype=np.float64)

    num_dim = s_rx.shape[0] // 2
    l_list = s_rx[:num_dim, :]
    l_dot_list = s_rx[num_dim:, :]

    poly_mats = []
    if num_dim == 2:
        for k in range(2):
            loc = l_list[:, k]
            l_dot = l_dot_list[:, k]
            r_dot = rr[k]
            l_tilde = 2.0 * (l_dot * (loc @ l_dot) - r_dot**2 * loc)
            c_tilde = r_dot**2 * (loc @ loc) - (loc @ l_dot) ** 2
            cm = np.zeros((3, 3))
            cm[2, 0] = r_dot**2 - l_dot[0] ** 2
            cm[0, 2] = r_dot**2 - l_dot[1] ** 2
            cm[1, 1] = -2.0 * l_dot[0] * l_dot[1]
            cm[1, 0] = l_tilde[0]
            cm[0, 1] = l_tilde[1]
            cm[0, 0] = c_tilde
            poly_mats.append(cm)
    elif num_dim == 3:
        for k in range(3):
            loc = l_list[:, k]
            l_dot = l_dot_list[:, k]
            r_dot = rr[k]
            l_tilde = 2.0 * (l_dot * (loc @ l_dot) - r_dot**2 * loc)
            c_tilde = r_dot**2 * (loc @ loc) - (loc @ l_dot) ** 2
            cm = np.zeros((3, 3, 3))
            cm[2, 0, 0] = r_dot**2 - l_dot[0] ** 2
            cm[0, 2, 0] = r_dot**2 - l_dot[1] ** 2
            cm[0, 0, 2] = r_dot**2 - l_dot[2] ** 2
            cm[1, 1, 0] = -2.0 * l_dot[0] * l_dot[1]
            cm[1, 0, 1] = -2.0 * l_dot[0] * l_dot[2]
            cm[0, 1, 1] = -2.0 * l_dot[1] * l_dot[2]
            cm[1, 0, 0] = l_tilde[0]
            cm[0, 1, 0] = l_tilde[1]
            cm[0, 0, 1] = l_tilde[2]
            cm[0, 0, 0] = c_tilde
            poly_mats.append(cm / 1e3)
    else:
        raise ValueError("Invalid dimensionality")

    the_roots, exit_code = poly_roots_multi_dim(
        poly_mats, max_deg_increases, use_motzkin_null
    )
    z_cart = _real_solutions(the_roots, abs_tol, rel_tol)

    num_sol = z_cart.shape[1]
    keep = np.ones(num_sol, dtype=bool)
    for s in range(num_sol):
        for k in range(num_dim):
            diff = z_cart[:, s] - l_list[:, k]
            rr_comp = -l_dot_list[:, k] @ diff / np.linalg.norm(diff)
            abs_diff = abs(rr_comp - rr[k])
            if not (abs_diff < abs_tol or abs_diff < rel_tol * abs(rr[k])):
                keep[s] = False
                break

    return PolyStaticEst(z_cart[:, keep], exit_code)


def range_rate_ratio_to_static_pos_2d(
    f_rat: ArrayLike,
    s_r_ref: ArrayLike,
    s_rx: ArrayLike,
    c: float = SPEED_OF_LIGHT,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-7,
    max_deg_increases: Optional[int] = None,
    use_motzkin_null: bool = False,
) -> PolyStaticEst:
    """
    2D emitter location from Doppler frequency ratios alone.

    A stationary emitter of UNKNOWN transmission frequency can be
    localized from moving sensors using only the ratios of the
    frequencies each sensor measures: the unknown frequency cancels in
    the ratio. Three sensors are needed — one reference and two
    others.

    Parameters
    ----------
    f_rat : array_like
        (2,) frequency ratios; the numerator is the reference sensor's
        measured frequency, the denominator the i-th other sensor's.
    s_r_ref : array_like
        (4,) reference sensor state ``[x, y, xdot, ydot]``.
    s_rx : array_like
        (4, 2) states of the two other sensors.
    c : float, optional
        Propagation speed of the signal. Default: speed of light.
    abs_tol, rel_tol : float, optional
        Tolerances for the numerically-real and sign-consistency
        filters. Defaults 1e-9 and 1e-7.
    max_deg_increases, use_motzkin_null : optional
        Passed through to the polynomial solver.

    Returns
    -------
    result : PolyStaticEst
        Real 2D solutions (the emitter plus geometric ambiguities) and
        the solver exit code.

    Notes
    -----
    Port of ``rangeRateRatio2StaticPos2D.m`` (same reference as
    :func:`range_rate_to_static_pos`). The problem is lifted to five
    variables ``[tx, ty, r1, r2, r3]`` — the target position plus the
    range to each sensor — before solving.
    """
    f_rat = np.asarray(f_rat, dtype=np.float64).ravel()
    s_r_ref = np.asarray(s_r_ref, dtype=np.float64).ravel()
    s_rx = np.asarray(s_rx, dtype=np.float64)

    l_rx1 = s_r_ref[:2]
    l_rx1_dot = s_r_ref[2:4]
    l_rx = s_rx[:2, :]
    l_rx_dot = s_rx[2:4, :]

    # Variables are ordered [tx, ty, r1, r2, r3].
    poly_mats = []

    # The r1 range-definition equation.
    l1x, l1y = l_rx1
    cm = np.zeros((3, 3, 3, 3, 3))
    cm[0, 0, 2, 0, 0] = 1.0
    cm[2, 0, 0, 0, 0] = -1.0
    cm[0, 2, 0, 0, 0] = -1.0
    cm[1, 0, 0, 0, 0] = 2.0 * l1x
    cm[0, 1, 0, 0, 0] = 2.0 * l1y
    cm[0, 0, 0, 0, 0] = -(l1x**2) - l1y**2
    poly_mats.append(cm)

    # The r2 and r3 range-definition equations.
    for cur_r in (2, 3):
        ljx, ljy = l_rx[:, cur_r - 2]
        cm = np.zeros((3, 3, 3, 3, 3))
        if cur_r == 2:
            cm[0, 0, 0, 2, 0] = 1.0
        else:
            cm[0, 0, 0, 0, 2] = 1.0
        cm[2, 0, 0, 0, 0] = -1.0
        cm[0, 2, 0, 0, 0] = -1.0
        cm[1, 0, 0, 0, 0] = 2.0 * ljx
        cm[0, 1, 0, 0, 0] = 2.0 * ljy
        cm[0, 0, 0, 0, 0] = -(ljx**2) - ljy**2
        poly_mats.append(cm)

    # The measurement equations.
    l1x_dot, l1y_dot = l_rx1_dot
    for cur_r in (1, 2):
        lj = l_rx[:, cur_r - 1]
        lj_dot = l_rx_dot[:, cur_r - 1]
        ljx_dot, ljy_dot = lj_dot
        f1j = f_rat[cur_r - 1]

        cm = np.zeros((2, 2, 2, 2, 2))
        cm[0, 0, 1, 0, 0] = -f1j * (lj @ lj_dot)

        rj_coeff = l_rx1 @ l_rx1_dot
        if cur_r == 1:
            cm[0, 0, 0, 1, 0] = rj_coeff
        else:
            cm[0, 0, 0, 0, 1] = rj_coeff

        r1rj_coeff = c * (f1j - 1.0)
        if cur_r == 1:
            cm[0, 0, 1, 1, 0] = r1rj_coeff
        else:
            cm[0, 0, 1, 0, 1] = r1rj_coeff

        cm[1, 0, 1, 0, 0] = f1j * ljx_dot
        cm[0, 1, 1, 0, 0] = f1j * ljy_dot

        if cur_r == 1:
            cm[1, 0, 0, 1, 0] = -l1x_dot
            cm[0, 1, 0, 1, 0] = -l1y_dot
        else:
            cm[1, 0, 0, 0, 1] = -l1x_dot
            cm[0, 1, 0, 0, 1] = -l1y_dot

        poly_mats.append(cm)

    the_roots, exit_code = poly_roots_multi_dim(
        poly_mats, max_deg_increases, use_motzkin_null
    )
    z_cart = _real_solutions(the_roots, abs_tol, rel_tol)
    z_cart = z_cart[:2, :]

    num_sol = z_cart.shape[1]
    keep = np.ones(num_sol, dtype=bool)
    for s in range(num_sol):
        diff = z_cart[:, s] - l_rx1
        rr_ref = -l_rx1_dot @ diff / np.linalg.norm(diff)
        for k in range(2):
            diff = z_cart[:, s] - l_rx[:, k]
            rr_cur = -l_rx_dot[:, k] @ diff / np.linalg.norm(diff)
            f_rat_cur = (1.0 - rr_ref / c) / (1.0 - rr_cur / c)
            abs_diff = abs(f_rat_cur - f_rat[k])
            if not (abs_diff < abs_tol or abs_diff < rel_tol * abs(f_rat[k])):
                keep[s] = False
                break

    return PolyStaticEst(z_cart[:, keep], exit_code)


__all__ = [
    "PolyStaticEst",
    "RangeOnlyLocEst",
    "ad_hoc_cart_cov",
    "range_only_static_loc_est_np",
    "range_rate_ratio_to_static_pos_2d",
    "range_rate_to_static_pos",
    "rr_only_static_vel_est",
    "tdoa_only_static_loc_est",
    "tdoa_to_cart",
]
