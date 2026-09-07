"""
Measurement conversions with covariances.

Ports of the MATLAB TCL ``Conversions_with_Covariances`` measurement
converters for direction-cosine coordinate systems: cubature-based
conversion of u-v measurements to spherical angles, of bistatic
r-u-v(-w) measurements between bistatic channel geometries, and of
camera image-plane measurements to u-v direction cosines, plus the
Taylor-series (CM1/CM2/CM3) debiased conversion of monostatic r-u-v
measurements to Cartesian coordinates.

Measurements are column vectors, batches are (d, N), and covariance
batches are (d, d, N) stacked along the last axis, as in the MATLAB
originals.

The cubature conversions default to
:func:`~pytcl.mathematical_functions.numerical_integration.cubature_points.fifth_order_cubature_points`.
MATLAB's ``fifthOrderCubPoints`` implements different (equally valid)
degree-5 rules, so default-point results agree with MATLAB only to
the rules' own truncation error on these non-polynomial conversions;
pass explicit ``xi``/``w`` to pin the rule exactly.

References
----------
.. [1] D. F. Crouse, "Basic tracking using nonlinear 3D monostatic
   and bistatic measurements," IEEE Aerospace and Electronic Systems
   Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
.. [2] X. Tian and Y. Bar-Shalom, "Coordinate conversion and tracking
   for very long range radars," IEEE Transactions on Aerospace and
   Electronic Systems, vol. 45, no. 3, pp. 1073-1088, Jul. 2009.
.. [3] J. R. Cookson, Z. T. Chance, and L. F. Urbano, "Consistent
   estimation for very long range radars," MIT Lincoln Laboratory,
   Lexington, MA, Tech. Rep. 1184, Dec. 2017.
"""

from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.coordinate_systems.conversions.uv import (
    camera_coords2uv,
    ruv2ruv,
    uv2spher_ang,
)
from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    fifth_order_cubature_points,
    transform_cubature_points,
)


class ConvertedMeasurements(NamedTuple):
    """A batch of converted measurements with covariances.

    Attributes
    ----------
    z : ndarray
        (d, N) converted measurement means.
    R : ndarray
        (d, d, N) covariance matrices of the converted measurements.
    """

    z: NDArray[np.floating]
    R: NDArray[np.floating]


def _as_columns(z: ArrayLike, rows: int) -> NDArray[np.floating]:
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim == 1:
        z_arr = z_arr[:, np.newaxis]
    if z_arr.shape[0] < rows:
        raise ValueError(f"expected at least {rows} rows, got {z_arr.shape[0]}")
    return z_arr


def _rep_sqrt(s_r: ArrayLike, d: int, n: int) -> NDArray[np.floating]:
    s = np.asarray(s_r, dtype=np.float64)
    if s.ndim == 2:
        s = np.repeat(s[:, :, np.newaxis], n, axis=2)
    if s.shape != (d, d, n):
        raise ValueError(f"covariance stack must be ({d}, {d}, {n}), got {s.shape}")
    return s


def _points(
    xi: Optional[ArrayLike], w: Optional[ArrayLike], dim: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if xi is None:
        return fifth_order_cubature_points(dim)
    return (
        np.asarray(xi, dtype=np.float64),
        np.asarray(w, dtype=np.float64).ravel(),
    )


def _wrap_pi(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap to [-pi, pi), MATLAB's wrapRange(x, -pi, pi)."""
    return np.mod(x + np.pi, 2.0 * np.pi) - np.pi


def _spher_ang_to_unit(
    az_el: NDArray[np.floating], system_type: int
) -> NDArray[np.floating]:
    """Unit vectors from (2, N) azimuth-elevation pairs, matching the
    MATLAB ``spher2Cart`` axis conventions."""
    az = az_el[0, :]
    el = az_el[1, :]
    if system_type == 0:
        return np.vstack([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    if system_type == 1:
        return np.vstack([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    if system_type == 2:
        return np.vstack([np.cos(az) * np.sin(el), np.sin(az) * np.sin(el), np.cos(el)])
    if system_type == 3:
        return np.vstack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    raise ValueError("Invalid system type specified.")


def _mean_direction_az_el(
    az_el: NDArray[np.floating],
    system_type: int,
    w: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Port of ``meanDirectionAzEl.m``: the weighted mean direction of
    azimuth-elevation points, computed through unit vectors."""
    unit = _spher_ang_to_unit(az_el, system_type)
    mean_vec = unit @ w
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return uv2spher_ang(mean_vec[:, np.newaxis], system_type).ravel()


def uv2spher_ang_cubature(
    z_uv: ArrayLike,
    s_r: ArrayLike,
    system_type: int = 0,
    m_s: Optional[ArrayLike] = None,
    m_uv: Optional[ArrayLike] = None,
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
) -> ConvertedMeasurements:
    """
    Convert Gaussian u-v measurements to spherical angles by cubature.

    Approximates the mean and covariance of noise-corrupted u-v
    direction-cosine measurements converted into [azimuth; elevation]
    spherical angles, propagating cubature points through
    :func:`~pytcl.coordinate_systems.conversions.uv.uv2spher_ang`.
    The mean is a proper mean direction and azimuth differences are
    wrapped; elevation is not wrapped, so measurements should not be
    at the elevation extremes.

    Parameters
    ----------
    z_uv : array_like
        (2, N) u-v direction cosine measurements.
    s_r : array_like
        (2, 2, N) lower-triangular square roots of the measurement
        covariances (a single (2, 2) matrix applies to all).
    system_type : int, optional
        Spherical axis convention (0-3), as in ``uv2spher_ang``.
    m_s, m_uv : array_like, optional
        (3, 3) rotations from the global frame to the spherical and
        u-v frames. Default identity.
    xi, w : array_like, optional
        Cubature points (num_points, 2) and weights. Default:
        fifth-order points.

    Returns
    -------
    result : ConvertedMeasurements
        ``z`` is (2, N) [azimuth; elevation] in radians; ``R`` is
        (2, 2, N).

    Examples
    --------
    >>> import numpy as np
    >>> res = uv2spher_ang_cubature([0.3, 0.4], 1e-3 * np.eye(2))
    >>> res.z.shape, res.R.shape
    ((2, 1), (2, 2, 1))

    Notes
    -----
    Port of ``uv2SpherAngCubature.m``.
    """
    z_uv = _as_columns(z_uv, 2)
    n = z_uv.shape[1]
    s_r_arr = _rep_sqrt(s_r, 2, n)
    xi_arr, w_arr = _points(xi, w, 2)

    az_el = np.zeros((2, n))
    r_az_el = np.zeros((2, 2, n))
    for k in range(n):
        pts, _ = transform_cubature_points(xi_arr, w_arr, z_uv[:, k], s_r_arr[:, :, k])
        xi_conv = uv2spher_ang(pts.T, system_type, m_s, m_uv)
        mean_val = _mean_direction_az_el(xi_conv, system_type, w_arr)
        diff = xi_conv - mean_val[:, np.newaxis]
        diff[0, :] = _wrap_pi(diff[0, :])
        p = (diff * w_arr[np.newaxis, :]) @ diff.T
        az_el[:, k] = mean_val
        r_az_el[:, :, k] = p
    return ConvertedMeasurements(az_el, r_az_el)


def ruv2ruv_cubature(
    z: ArrayLike,
    s_r: ArrayLike,
    use_half_range: Union[bool, Tuple[bool, bool]] = False,
    z_tx1: Optional[ArrayLike] = None,
    z_rx1: Optional[ArrayLike] = None,
    m1: Optional[ArrayLike] = None,
    z_tx2: Optional[ArrayLike] = None,
    z_rx2: Optional[ArrayLike] = None,
    m2: Optional[ArrayLike] = None,
    include_w: int = 0,
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
) -> ConvertedMeasurements:
    """
    Convert bistatic r-u-v(-w) measurements between channels by cubature.

    Approximates the moments of bistatic r-u-v (or r-u-v-w)
    measurements taken by one transmitter/receiver pair converted into
    the coordinate system of another, e.g. to gate bistatic receptions
    against a transmit beam.

    Parameters
    ----------
    z : array_like
        (3, N) [r; u; v] or (4, N) [r; u; v; w] measurements.
    s_r : array_like
        Lower-triangular square roots of the measurement covariances,
        (d, d, N) or a single (d, d).
    use_half_range : bool or (bool, bool), optional
        Whether the source and destination ranges are one-way; a
        scalar applies to both. Default False.
    z_tx1, z_rx1, m1 : array_like, optional
        Transmitter position, receiver position and receiver rotation
        of the source channel. Defaults: origin, origin, identity.
    z_tx2, z_rx2, m2 : array_like, optional
        The same for the destination channel.
    include_w : int, optional
        0 (default): no w component in the output. 1: include w and
        average as a mean direction (linear covariance about it;
        singular R). 2: include w with plain weighted averaging (the
        direction loses unit magnitude).
    xi, w : array_like, optional
        Cubature points (num_points, d) and weights. Default:
        fifth-order points of the measurement's dimension (the
        original always generates 3-D points, so its r-u-v-w input
        requires explicit points; the port's default follows the
        input).

    Returns
    -------
    result : ConvertedMeasurements
        ``z`` is (3, N), or (4, N) when ``include_w`` is nonzero (R is
        then singular).

    Examples
    --------
    >>> import numpy as np
    >>> s_r = np.diag([50.0, 1e-3, 1e-3])
    >>> rx2 = np.array([1e4, 2e4, 0.0])
    >>> res = ruv2ruv_cubature([9e4, 0.1, 0.2], s_r, False,
    ...                        None, rx2, None, None, None)
    >>> res.z.shape, res.R.shape
    ((3, 1), (3, 3, 1))

    Notes
    -----
    Port of ``ruv2RuvCubature.m``.
    """
    z_arr = _as_columns(z, 3)
    d_in = z_arr.shape[0]
    n = z_arr.shape[1]
    num_dim = 4 if include_w else 3
    s_r_arr = _rep_sqrt(s_r, d_in, n)
    xi_arr, w_arr = _points(xi, w, d_in)

    z_conv = np.zeros((num_dim, n))
    r_conv = np.zeros((num_dim, num_dim, n))
    if include_w in (0, 2):
        for k in range(n):
            pts, _ = transform_cubature_points(
                xi_arr, w_arr, z_arr[:, k], s_r_arr[:, :, k]
            )
            conv = ruv2ruv(
                pts.T,
                use_half_range,
                z_tx1,
                z_rx1,
                m1,
                z_tx2,
                z_rx2,
                m2,
                bool(include_w),
            )
            mean_val = conv @ w_arr
            diff = conv - mean_val[:, np.newaxis]
            z_conv[:, k] = mean_val
            r_conv[:, :, k] = (diff * w_arr[np.newaxis, :]) @ diff.T
    elif include_w == 1:
        for k in range(n):
            pts, _ = transform_cubature_points(
                xi_arr, w_arr, z_arr[:, k], s_r_arr[:, :, k]
            )
            conv = ruv2ruv(
                pts.T, use_half_range, z_tx1, z_rx1, m1, z_tx2, z_rx2, m2, True
            )
            # Deal with any converted points of zero direction magnitude.
            bad = ~np.all(np.isfinite(conv[1:, :]), axis=0)
            conv[1, bad] = 1.0
            conv[2, bad] = 0.0
            conv[3, bad] = 0.0

            z_mean = conv @ w_arr
            # The mean direction is the linear average projected onto
            # the unit sphere.
            z_mean[1:] = z_mean[1:] / np.linalg.norm(z_mean[1:])
            if not np.all(np.isfinite(z_mean[1:])):
                # Faithful transcription of the original's guard; not
                # reachable through ruv2ruv itself, whose in-front
                # directions cannot sum to a zero-norm mean.
                z_mean[1:] = [1.0, 0.0, 0.0]  # pragma: no cover

            diff = conv - z_mean[:, np.newaxis]
            z_conv[:, k] = z_mean
            r_conv[:, :, k] = (diff * w_arr[np.newaxis, :]) @ diff.T
    else:
        raise ValueError("Unknown value of include_w provided.")
    return ConvertedMeasurements(z_conv, r_conv)


def camera_coords2uv_cubature(
    z_cam: ArrayLike,
    s_r: ArrayLike,
    a: ArrayLike,
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
) -> ConvertedMeasurements:
    """
    Convert Gaussian camera measurements to u-v cosines by cubature.

    Converts measurements in the image plane of a perspective camera
    (distance units, not pixels) into u-v direction cosines in the
    camera's own frame, without rotation to global coordinates.

    Parameters
    ----------
    z_cam : array_like
        (2, N) [x; y] image-plane measurements.
    s_r : array_like
        (2, 2, N) lower-triangular square roots of the measurement
        covariances (a single (2, 2) matrix applies to all).
    a : array_like
        (3, 3) camera matrix, as in
        :func:`~pytcl.coordinate_systems.conversions.uv.camera_coords2uv`;
        the third row must be [0, 0, 1].
    xi, w : array_like, optional
        Cubature points (num_points, 2) and weights. Default:
        fifth-order points.

    Returns
    -------
    result : ConvertedMeasurements
        ``z`` is (2, N) u-v pairs; ``R`` is (2, 2, N).

    Examples
    --------
    >>> import numpy as np
    >>> a = np.diag([35e-3, 35e-3, 1.0])
    >>> res = camera_coords2uv_cubature([1e-2, -2e-2], 1e-4 * np.eye(2), a)
    >>> res.z.shape, res.R.shape
    ((2, 1), (2, 2, 1))

    Notes
    -----
    Port of ``cameraCoords2UVCoordsCubature.m``.
    """
    z_cam = _as_columns(z_cam, 2)
    n = z_cam.shape[1]
    s_r_arr = _rep_sqrt(s_r, 2, n)
    xi_arr, w_arr = _points(xi, w, 2)

    z_uv = np.zeros((2, n))
    r_uv = np.zeros((2, 2, n))
    for k in range(n):
        pts, _ = transform_cubature_points(xi_arr, w_arr, z_cam[:, k], s_r_arr[:, :, k])
        conv = camera_coords2uv(pts.T, a, None, include_w=False)
        mean_val = conv @ w_arr
        diff = conv - mean_val[:, np.newaxis]
        z_uv[:, k] = mean_val
        r_uv[:, :, k] = (diff * w_arr[np.newaxis, :]) @ diff.T
    return ConvertedMeasurements(z_uv, r_uv)


def _cm1_conversion(
    r: float, u: float, v: float, sr2: float, su2: float, sv2: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The first-order CM1 conversion of Tian and Bar-Shalom."""
    w = np.sqrt(abs(1.0 - u * u - v * v))
    dfzdr = w
    dfzdu = -r * u / w
    dfzdv = -r * v / w
    r2 = r * r
    u2 = u * u
    v2 = v * v
    z = max(r * w, 0.0)
    cart = np.array([r * u, r * v, z])
    r11 = u2 * sr2 + r2 * su2
    r22 = v2 * sr2 + r2 * sv2
    r33 = dfzdr**2 * sr2 + dfzdu**2 * su2 + dfzdv**2 * sv2
    r12 = u * v * sr2
    r13 = u * dfzdr * sr2 + r * dfzdu * su2
    r23 = v * dfzdr * sr2 + r * dfzdv * sv2
    cov = np.array([[r11, r12, r13], [r12, r22, r23], [r13, r23, r33]])
    return cart, cov


def _cm2_corrected_conversion(
    r: float, u: float, v: float, sr2: float, su2: float, sv2: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The corrected second-order CM2 conversion of Cookson et al."""
    su4 = su2 * su2
    sv4 = sv2 * sv2
    r2 = r * r
    u2 = u * u
    u4 = u2 * u2
    v2 = v * v
    v4 = v2 * v2
    w = np.sqrt(abs(1.0 - u2 - v2))
    w2 = w * w
    w4 = w2 * w2
    w6 = w4 * w2
    w3 = w2 * w
    cz = (
        -(1.0 / 2.0) * (r / w + r * u2 / w3) * su2
        - (1.0 / 2.0) * (r / w + r * v2 / w3) * sv2
    )
    z = max(r * w + cz, 0.0)
    cart = np.array([r * u, r * v, z])
    sxx = u2 * sr2 + r2 * su2 + sr2 * su2
    syy = v2 * sr2 + r2 * sv2 + sr2 * sv2
    szz = (
        w2 * sr2
        + (r2 / w2) * (u2 * su2 + v2 * sv2)
        + (u2 / w2) * sr2 * su2
        + (v2 / w2) * sr2 * sv2
        + (r2 / 2.0) * (u2 * v2 / w6 - (u2 + v2) / w4 - 1.0 / w2) * su2 * sv2
        + (r2 / 2.0) * (u4 / w6 + 2.0 * u2 / w4 + 1.0 / w2) * su4
        + (r2 / 2.0) * (v4 / w6 + 2.0 * v2 / w4 + 1.0 / w2) * sv4
    )
    sxy = u * v * sr2
    sxz = u * w * sr2 - (r2 * u / w) * su2 - (u / w) * sr2 * su2
    syz = v * w * sr2 - (r2 * v / w) * sv2 - (v / w) * sr2 * sv2
    cov = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    return cart, cov


def _cm3_conversion(
    r: float, u: float, v: float, sr2: float, su2: float, sv2: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The third-order CM3 conversion of Cookson et al."""
    su4 = su2 * su2
    su6 = su4 * su2
    sv4 = sv2 * sv2
    sv6 = sv4 * sv2
    r2 = r * r
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u6 = u4 * u2
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v6 = v4 * v2
    w = np.sqrt(abs(1.0 - u2 - v2))
    w2 = w * w
    w3 = w2 * w
    w4 = w3 * w
    w5 = w4 * w
    w6 = w5 * w
    w8 = w6 * w2
    w10 = w8 * w2
    cz = (
        -(1.0 / 2.0) * (r / w + r * u2 / w3) * su2
        - (1.0 / 2.0) * (r / w + r * v2 / w3) * sv2
    )
    z = max(r * w + cz, 0.0)
    cart = np.array([r * u, r * v, z])
    sxx = u2 * sr2 + r2 * su2 + sr2 * su2
    syy = v2 * sr2 + r2 * sv2 + sr2 * sv2
    szz = (
        w2 * sr2
        + (r2 / w2) * (u2 * su2 + v2 * sv2)
        - sr2 * su2
        - sr2 * sv2
        + (7.0 * r2 * u2 * v2 / w6 + r2 * (u2 + v2) / w4) * su2 * sv2
        + (1.0 / 2.0) * (7.0 * r2 * u4 / w6 + 8.0 * r2 * u2 / w4 + r2 / w2) * su4
        + (1.0 / 2.0) * (7.0 * r2 * v4 / w6 + 8.0 * r2 * v2 / w4 + r2 / w2) * sv4
        + (3.0 / 4.0) * (u4 / w6 + 2.0 * u2 / w4 + 1.0 / w2) * sr2 * su4
        + (3.0 / 4.0) * (v4 / w6 + 2.0 * v2 / w4 + 1.0 / w2) * sr2 * sv4
        + (1.0 / 4.0)
        * (
            45.0 * r2 * u4 * v2 / w10
            + (36.0 * r2 * u2 * v2 + 6.0 * r2 * u4) / w8
            + (6.0 * r2 * u2 + 3.0 * r2 * v2) / w6
        )
        * su4
        * sv2
        + (1.0 / 4.0)
        * (
            45.0 * r2 * u2 * v4 / w10
            + (36.0 * r2 * u2 * v2 + 6.0 * r2 * v4) / w8
            + (6.0 * r2 * v2 + 3.0 * r2 * u2) / w6
        )
        * su2
        * sv4
        + (15.0 / 4.0) * (r2 * u6 / w10 + 2.0 * r2 * u4 / w8 + r2 * u2 / w6) * su6
        + (15.0 / 4.0) * (r2 * v6 / w10 + 2.0 * r2 * v4 / w8 + r2 * v2 / w6) * sv6
        + (1.0 / 2.0)
        * (3.0 * u2 * v2 / w6 + (u2 + v2) / w4 + 1.0 / w2)
        * sr2
        * su2
        * sv2
    )
    sxy = u * v * sr2
    sxz = (
        u * w * sr2
        - (r2 * u / w) * su2
        - (1.0 / 2.0) * (u3 / w3 + 3.0 * u / w) * sr2 * su2
        - (1.0 / 2.0) * (u * v2 / w3 + u / w) * sr2 * sv2
        - (1.0 / 2.0) * (3.0 * r2 * u * v2 / w5 + r2 * u / w3) * su2 * sv2
        - (3.0 / 2.0) * (r2 * u3 / w5 + r2 * u / w3) * su4
    )
    syz = (
        v * w * sr2
        - (r2 * v / w) * sv2
        - (1.0 / 2.0) * (u2 * v / w3 + v / w) * sr2 * su2
        - (1.0 / 2.0) * (v3 / w3 + 3.0 * v / w) * sr2 * sv2
        - (1.0 / 2.0) * (3.0 * r2 * u2 * v / w5 + r2 * v / w3) * su2 * sv2
        - (3.0 / 2.0) * (r2 * v3 / w5 + r2 * v / w3) * sv4
    )
    cov = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    return cart, cov


def _cm2_original_conversion(
    r: float, u: float, v: float, sr2: float, su2: float, sv2: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The uncorrected CM2 conversion of Tian and Bar-Shalom (the
    original's undocumented algorithm 3)."""
    temp1 = np.sqrt(abs(1.0 - u * u - v * v))
    dfzdr = temp1
    dfzdu = -r * u / temp1
    dfzdv = -r * v / temp1
    temp2 = (1.0 - u * u - v * v) ** 1.5
    d2fzdu = r * (v * v - 1.0) / temp2
    d2fzdv = r * (u * u - 1.0) / temp2
    d2fzdrdu = -u / temp1
    d2fzdrdv = -v / temp1
    d2fzdudv = -r * u * v / temp2
    cz = 0.5 * d2fzdu * su2 + 0.5 * d2fzdv * sv2
    z = max(r * temp1 - cz, 0.0)
    cart = np.array([r * u, r * v, z])
    r11 = r * r * su2 + u * u * sr2 + sr2 * su2
    r22 = r * r * sv2 + v * v * sr2 + sr2 * sv2
    r33 = (
        cz * cz
        + dfzdr**2 * sr2
        + dfzdu**2 * su2
        + dfzdv**2 * sv2
        + cz * (d2fzdu * su2 + d2fzdv * sv2)
        + (3.0 / 4.0) * d2fzdu * su2**2
        + (3.0 / 4.0) * d2fzdv * sv2**2
        + d2fzdrdu**2 * sr2 * su2
        + d2fzdrdv**2 * sr2 * sv2
        + d2fzdudv**2 * su2 * sv2
    )
    r12 = u * v * sr2
    r13 = dfzdu * r * su2 + dfzdr * u * sr2
    r23 = dfzdv * r * sv2 + dfzdr * v * sr2
    cov = np.array([[r11, r12, r13], [r12, r22, r23], [r13, r23, r33]])
    return cart, cov


_TAYLOR_CONVERSIONS = {
    0: _cm1_conversion,
    1: _cm2_corrected_conversion,
    2: _cm3_conversion,
    3: _cm2_original_conversion,
}


def monostat_ruv2cart_taylor(
    z_meas: ArrayLike,
    r: ArrayLike,
    use_half_range: bool = False,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
    algorithm: int = 1,
) -> ConvertedMeasurements:
    """
    Debiased Taylor-series conversion of monostatic r-u-v to Cartesian.

    Approximates the Cartesian mean and covariance of Gaussian
    monostatic [r; u; v] measurements using the classic
    converted-measurement Taylor expansions. Covariance cross terms of
    the input are neglected, as in the source expressions.

    Parameters
    ----------
    z_meas : array_like
        (3, N) [r; u; v] measurements; r is the one-way range when
        ``use_half_range``, otherwise the two-way range.
    r : array_like
        (3, 3, N) measurement covariances (a single (3, 3) applies to
        all).
    use_half_range : bool, optional
        Whether the range is already one-way. Default False (the range
        and its covariance are halved internally).
    z_rx : array_like, optional
        (3, N) receiver positions (a single (3,) applies to all).
        Default origin.
    m : array_like, optional
        (3, 3, N) rotations from the global frame to the receiver's
        (a single (3, 3) applies to all). Default identity.
    algorithm : int, optional
        0: the first-order CM1 conversion of [2]_. 1 (default): the
        second-order CM2 conversion as corrected in [3]_. 2: the
        third-order CM3 conversion of [3]_. 3: the uncorrected CM2
        conversion of [2]_ (present but undocumented upstream).

    Returns
    -------
    result : ConvertedMeasurements
        ``z`` is (3, N) Cartesian positions; ``R`` is (3, 3, N).

    Examples
    --------
    >>> import numpy as np
    >>> R = np.diag([1.0, 1e-6, 25e-6])
    >>> res = monostat_ruv2cart_taylor([1e5, 0.3, 0.0], R, True)
    >>> np.round(res.z.ravel(), 1).tolist()
    [30000.0, 0.0, 95392.6]

    Notes
    -----
    Port of ``monostatRuv2CartTaylor.m``, with one deliberate fix of
    an upstream defect: given several measurements with
    ``useHalfRange=false``, the original halves only the FIRST
    measurement's range (``zMeas(1)=zMeas(1)/2``) while scaling every
    covariance, corrupting measurements 2..N; the port halves every
    range. Its single-measurement behavior is unchanged. The z
    component is clamped nonnegative (the debiasing breaks down at
    extreme angles), as upstream.
    """
    z_arr = _as_columns(z_meas, 3).copy()
    n = z_arr.shape[1]
    r_arr = np.asarray(r, dtype=np.float64)
    if r_arr.ndim == 2:
        r_arr = np.repeat(r_arr[:, :, np.newaxis], n, axis=2)
    r_arr = r_arr.copy()
    m_arr = np.asarray(np.eye(3) if m is None else m, dtype=np.float64)
    if m_arr.ndim == 2:
        m_arr = np.repeat(m_arr[:, :, np.newaxis], n, axis=2)
    if z_rx is None:
        z_rx_arr = np.zeros((3, n))
    else:
        z_rx_arr = _as_columns(z_rx, 3)
        if z_rx_arr.shape[1] == 1:
            z_rx_arr = np.repeat(z_rx_arr, n, axis=1)

    if not use_half_range:
        z_arr[0, :] = z_arr[0, :] / 2.0
        d = np.diag([0.5, 1.0, 1.0])
        for k in range(n):
            r_arr[:, :, k] = d @ r_arr[:, :, k] @ d.T

    try:
        conversion = _TAYLOR_CONVERSIONS[algorithm]
    except KeyError:
        raise ValueError("Unknown algorithm specified") from None

    z_cart = np.zeros((3, n))
    r_cart = np.zeros((3, 3, n))
    for k in range(n):
        m_inv = m_arr[:, :, k].T
        cart, cov = conversion(
            float(z_arr[0, k]),
            float(z_arr[1, k]),
            float(z_arr[2, k]),
            float(r_arr[0, 0, k]),
            float(r_arr[1, 1, k]),
            float(r_arr[2, 2, k]),
        )
        z_cart[:, k] = m_inv @ cart + z_rx_arr[:, k]
        r_cart[:, :, k] = m_inv @ cov @ m_inv.T
    return ConvertedMeasurements(z_cart, r_cart)


__all__ = [
    "ConvertedMeasurements",
    "camera_coords2uv_cubature",
    "monostat_ruv2cart_taylor",
    "ruv2ruv_cubature",
    "uv2spher_ang_cubature",
]
