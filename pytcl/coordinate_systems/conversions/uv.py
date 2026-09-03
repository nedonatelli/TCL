"""
Direction-cosine u-v measurement coordinates.

The u-v(-w) system is the natural measurement space of a planar phased
array: u and v are the first two components of a unit direction vector
in the sensor's coordinate system. This module ports the MATLAB TCL
angle-only u-v conversions and the full bistatic r-u-v conversions with
sensor offsets and pointing rotations (the simplified aligned-monostatic
``ruv2cart``/``cart2ruv`` live in
:mod:`pytcl.coordinate_systems.conversions.spherical`).

All conventions follow D. F. Crouse, "Basic tracking using nonlinear
3D monostatic and bistatic measurements," IEEE Aerospace and
Electronic Systems Magazine, vol. 29, no. 8, Part II, pp. 4-53,
Aug. 2014.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_columns(z: ArrayLike, rows: int) -> NDArray[np.floating]:
    """(rows,) or (rows, N) input as a (rows, N) float array."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.shape[0] < rows:
        raise ValueError(f"expected at least {rows} rows, got {arr.shape[0]}")
    return arr


def _broadcast_sensor(
    val: Optional[ArrayLike], n: int, rows: int
) -> NDArray[np.floating]:
    """A sensor location argument as (rows, n), defaulting to zeros."""
    if val is None:
        return np.zeros((rows, n))
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.shape[1] == 1:
        arr = np.tile(arr, (1, n))
    return arr[:rows, :]


def _broadcast_rot(m: Optional[ArrayLike], n: int) -> NDArray[np.floating]:
    """A rotation argument as (3, 3, N), defaulting to identity."""
    if m is None:
        return np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, n))
    arr = np.asarray(m, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.shape[2] == 1:
        arr = np.tile(arr, (1, 1, n))
    return arr


def uv2spher_ang(
    uv: ArrayLike,
    system_type: int = 0,
    m_s: Optional[ArrayLike] = None,
    m_uv: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert u-v direction cosines to spherical azimuth and elevation.

    Parameters
    ----------
    uv : array_like
        (2, N) u-v pairs, or (3, N) u-v-w unit vectors. With only u-v
        given, w is taken positive (in front of the sensor).
    system_type : int, optional
        Spherical axis convention:

        - 0 (default): azimuth counterclockwise from x in the x-y
          plane, elevation up from the x-y plane.
        - 1: azimuth counterclockwise from z in the z-x plane,
          elevation up from the z-x plane (z-axis boresight).
        - 2: like 0, but the second angle is measured down from the
          z-axis (pi/2 - elevation).
        - 3: like 0, but azimuth is measured clockwise from the y-axis
          (East-of-North bearings in an ENU frame).
    m_s : array_like, optional
        (3, 3) rotation from the global frame to the frame the
        spherical angles are expressed in. Default identity.
    m_uv : array_like, optional
        (3, 3) rotation from the global frame to the frame the u-v
        coordinates are expressed in. Default identity.

    Returns
    -------
    az_el : ndarray
        (2, N) azimuth and elevation in radians.

    Examples
    --------
    >>> import numpy as np
    >>> az_el = uv2spher_ang(np.array([0.0, 0.0]))  # the +z boresight
    >>> np.round(az_el.ravel(), 6).tolist()
    [0.0, 1.570796]

    Notes
    -----
    Port of ``uv2SpherAng.m`` (Crouse 2014, see the module docstring).
    """
    uv_arr = _as_columns(uv, 2).copy()
    if m_uv is None:
        m_uv = np.eye(3)
    if m_s is None:
        m_s = np.eye(3)

    if uv_arr.shape[0] < 3:
        # real() guards rounding pushing the sqrt argument negative.
        w = np.sqrt(np.maximum(0.0, 1.0 - uv_arr[0, :] ** 2 - uv_arr[1, :] ** 2))
        uv_arr = np.vstack([uv_arr, w])

    uv_arr = np.asarray(m_s) @ np.asarray(m_uv).T @ uv_arr[:3, :]

    u, v, w = uv_arr[0, :], uv_arr[1, :], uv_arr[2, :]

    az_el = np.zeros((2, uv_arr.shape[1]))
    if system_type == 0:
        az_el[0, :] = np.arctan2(v, u)
        az_el[1, :] = np.arcsin(w)
    elif system_type == 1:
        az_el[0, :] = np.arctan2(u, w)
        az_el[1, :] = np.arcsin(v)
    elif system_type == 2:
        az_el[0, :] = np.arctan2(v, u)
        az_el[1, :] = np.arccos(w)
    elif system_type == 3:
        az_el[0, :] = np.arctan2(u, v)
        az_el[1, :] = np.arcsin(w)
    else:
        raise ValueError("Invalid system type specified.")
    return az_el


def spher_ang2uv(
    az_el: ArrayLike,
    system_type: int = 0,
    include_w: bool = False,
    m_s: Optional[ArrayLike] = None,
    m_uv: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert spherical azimuth and elevation to u-v direction cosines.

    Parameters
    ----------
    az_el : array_like
        (2, N) azimuth and elevation in radians.
    system_type : int, optional
        Spherical axis convention; see :func:`uv2spher_ang`.
    include_w : bool, optional
        Also return the third unit-vector component w. Default False.
    m_s, m_uv : array_like, optional
        Rotations as in :func:`uv2spher_ang`.

    Returns
    -------
    uv : ndarray
        (2, N) u-v pairs, or (3, N) u-v-w unit vectors when
        ``include_w`` is True.

    Examples
    --------
    >>> import numpy as np
    >>> uv = spher_ang2uv(np.array([0.4, 0.7]), include_w=True)
    >>> az_el = uv2spher_ang(uv)
    >>> np.allclose(az_el.ravel(), [0.4, 0.7])
    True

    Notes
    -----
    Port of ``spherAng2Uv.m`` (Crouse 2014, see the module docstring).
    """
    az_el_arr = _as_columns(az_el, 2)
    if m_uv is None:
        m_uv = np.eye(3)
    if m_s is None:
        m_s = np.eye(3)

    azimuth = az_el_arr[0, :].copy()
    elevation = az_el_arr[1, :].copy()

    if system_type == 2:
        elevation = np.pi / 2.0 - elevation
        system_type = 0
    elif system_type == 3:
        azimuth = np.pi / 2.0 - azimuth
        system_type = 0

    if system_type == 0:
        vec = np.vstack(
            [
                np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )
    elif system_type == 1:
        vec = np.vstack(
            [
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
                np.cos(azimuth) * np.cos(elevation),
            ]
        )
    else:
        raise ValueError("Invalid system type specified.")

    uv = np.asarray(m_uv) @ np.asarray(m_s).T @ vec
    return uv if include_w else uv[:2, :]


def ruv2cart_bistatic(
    z: ArrayLike,
    use_half_range: bool = False,
    z_tx: Optional[ArrayLike] = None,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Convert bistatic r-u-v(-w) measurements to global Cartesian points.

    The full conversion with transmitter/receiver offsets and receiver
    pointing rotations; the aligned-monostatic simplification is
    :func:`pytcl.coordinate_systems.conversions.spherical.ruv2cart`.

    Parameters
    ----------
    z : array_like
        (3, N) r-u-v or (4, N) r-u-v-w measurements. The range is the
        bistatic range (transmitter to target to receiver).
    use_half_range : bool, optional
        True if the ranges are one-way (monostatic convention).
        Default False.
    z_tx : array_like, optional
        (3, N) transmitter positions, or a single (3,) position shared
        by all measurements. Default: the origin.
    z_rx : array_like, optional
        Receiver positions, like ``z_tx``. Default: the origin.
    m : array_like, optional
        (3, 3, N) rotations from the global frame to each receiver's
        local frame (the local z-axis is the pointing direction), or a
        single (3, 3) shared rotation. Default identity.

    Returns
    -------
    z_c : ndarray
        (3, N) global Cartesian positions.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([10.0, 0.0, 0.0])  # r-u-v along the local z-axis
    >>> np.round(ruv2cart_bistatic(z).ravel(), 12)
    array([0., 0., 5.])

    Notes
    -----
    Port of ``ruv2Cart.m`` (Crouse 2014, see the module docstring).
    """
    z_arr = _as_columns(z, 3)
    n = z_arr.shape[1]
    m_arr = _broadcast_rot(m, n)
    z_rx_arr = _broadcast_sensor(z_rx, n, 3)
    z_tx_arr = _broadcast_sensor(z_tx, n, 3)

    r_b = z_arr[0, :].copy()
    if use_half_range:
        r_b = 2.0 * r_b

    has_w = z_arr.shape[0] > 3

    z_c = np.zeros((3, n))
    for k in range(n):
        if has_w:
            u_vec = z_arr[1:4, k]
        else:
            u, v = z_arr[1, k], z_arr[2, k]
            uv_mag2 = u**2 + v**2
            if uv_mag2 > 1.0:
                uv_mag = np.sqrt(uv_mag2)
                u, v = u / uv_mag, v / uv_mag
            u_vec = np.array([u, v, np.sqrt(max(0.0, 1.0 - u**2 - v**2))])

        # The transmitter in the receiver's local coordinate system.
        z_tx_l = m_arr[:, :, k] @ (z_tx_arr[:, k] - z_rx_arr[:, k])

        denom = 2.0 * (r_b[k] - u_vec @ z_tx_l)
        r1 = (r_b[k] ** 2 - z_tx_l @ z_tx_l) / denom if r_b[k] != 0.0 else 0.0

        z_l = r1 * u_vec
        z_c[:, k] = np.linalg.solve(m_arr[:, :, k], z_l) + z_rx_arr[:, k]
    return z_c


def cart2ruv_bistatic(
    z_c: ArrayLike,
    use_half_range: bool = False,
    z_tx: Optional[ArrayLike] = None,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
    include_w: bool = False,
) -> NDArray[np.floating]:
    """
    Convert global Cartesian points to bistatic r-u-v(-w) measurements.

    Parameters
    ----------
    z_c : array_like
        (3, N) global Cartesian positions.
    use_half_range : bool, optional
        Halve the returned bistatic range (monostatic convention).
        Default False.
    z_tx, z_rx, m : array_like, optional
        Transmitter/receiver positions and receiver rotations as in
        :func:`ruv2cart_bistatic`.
    include_w : bool, optional
        Also return the third direction cosine w. Default False.

    Returns
    -------
    z : ndarray
        (3, N) r-u-v or (4, N) r-u-v-w measurements.

    Examples
    --------
    >>> import numpy as np
    >>> z_c = np.array([0.0, 0.0, 5.0])
    >>> np.round(cart2ruv_bistatic(z_c).ravel(), 12)
    array([10.,  0.,  0.])

    Notes
    -----
    Port of ``Cart2Ruv.m`` (Crouse 2014, see the module docstring).
    """
    z_c_arr = _as_columns(z_c, 3)
    n = z_c_arr.shape[1]
    m_arr = _broadcast_rot(m, n)
    z_rx_arr = _broadcast_sensor(z_rx, n, 3)
    z_tx_arr = _broadcast_sensor(z_tx, n, 3)

    z = np.zeros((4 if include_w else 3, n))
    for k in range(n):
        z_c_l = m_arr[:, :, k] @ (z_c_arr[:, k] - z_rx_arr[:, k])
        r1 = np.linalg.norm(z_c_arr[:, k] - z_rx_arr[:, k])
        r2 = np.linalg.norm(z_c_arr[:, k] - z_tx_arr[:, k])
        z[0, k] = r1 + r2
        z[1, k] = z_c_l[0] / r1
        z[2, k] = z_c_l[1] / r1
        if include_w:
            z[3, k] = z_c_l[2] / r1

    if use_half_range:
        z[0, :] = z[0, :] / 2.0
    return z


def ruv2ruv(
    z: ArrayLike,
    use_half_range: "bool | tuple" = False,
    z_tx1: Optional[ArrayLike] = None,
    z_rx1: Optional[ArrayLike] = None,
    m1: Optional[ArrayLike] = None,
    z_tx2: Optional[ArrayLike] = None,
    z_rx2: Optional[ArrayLike] = None,
    m2: Optional[ArrayLike] = None,
    include_w: Optional[bool] = None,
) -> NDArray[np.floating]:
    """
    Convert bistatic r-u-v(-w) measurements between two bistatic pairs.

    Converts measurements taken by one transmitter/receiver pair into
    the coordinate system of another (possibly rotated, displaced)
    pair, via Cartesian coordinates.

    Parameters
    ----------
    z : array_like
        (3, N) r-u-v or (4, N) r-u-v-w measurements.
    use_half_range : bool or tuple of (bool, bool), optional
        One-way-range convention for the input and output systems; a
        scalar applies to both. Default False.
    z_tx1, z_rx1, m1 : array_like, optional
        Transmitter/receiver positions and receiver rotation of the
        system the measurements come from.
    z_tx2, z_rx2, m2 : array_like, optional
        The same for the system converted into.
    include_w : bool, optional
        Include w in the output. Default: True when the input is
        (4, N), else False.

    Returns
    -------
    z_new : ndarray
        The measurements in the second system.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([100.0, 0.3, -0.2])
    >>> rx2 = np.array([10.0, -5.0, 2.0])
    >>> out = ruv2ruv(z, False, None, None, None, None, rx2)
    >>> back = ruv2ruv(out, False, None, rx2)
    >>> np.allclose(back.ravel(), z)
    True

    Notes
    -----
    Port of ``ruv2Ruv.m``: composes :func:`ruv2cart_bistatic` and
    :func:`cart2ruv_bistatic`.
    """
    z_arr = _as_columns(z, 3)
    num_dim = z_arr.shape[0]

    if isinstance(use_half_range, bool):
        half1 = half2 = use_half_range
    else:
        half1, half2 = use_half_range

    if include_w is None:
        # 4 rows means r-u-v-w, so keep the w coordinate.
        include_w = num_dim == 4

    z_cart = ruv2cart_bistatic(z_arr, half1, z_tx1, z_rx1, m1)
    return cart2ruv_bistatic(z_cart, half2, z_tx2, z_rx2, m2, include_w)


def state_ruv2cart(x: ArrayLike) -> NDArray[np.floating]:
    """
    Convert an r-u-v state with derivatives to a Cartesian state.

    Handles 6-element states [r, u, v, rdot, udot, vdot] and 9-element
    states with second derivatives appended, converting position,
    velocity and (when present) acceleration into 3D Cartesian
    components. The measurement is monostatic with w > 0.

    Parameters
    ----------
    x : array_like
        (6, N) or (9, N) r-u-v states.

    Returns
    -------
    cart_states : ndarray
        (6, N) or (9, N) Cartesian states [x, y, z, xdot, ydot, zdot
        (, xddot, yddot, zddot)].

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([100.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    >>> np.round(state_ruv2cart(x).ravel(), 12)
    array([  0.,   0., 100.,   0.,   0.,   5.])

    Notes
    -----
    Port of ``stateRuv2Cart.m``. The local basis vectors u1 (radial),
    u2 and u3 and the coefficients follow the original expressions
    verbatim.
    """
    x_arr = _as_columns(x, 6)
    num_dim, n = x_arr.shape
    cart = np.zeros((num_dim, n))

    r = x_arr[0, :]
    u = x_arr[1, :]
    v = x_arr[2, :]
    r_dot = x_arr[3, :]
    u_dot = x_arr[4, :]
    v_dot = x_arr[5, :]

    w2 = np.maximum(0.0, 1.0 - u**2 - v**2)
    w = np.sqrt(w2)
    diff_v2 = 1.0 - v**2
    diff_v = np.sqrt(diff_v2)
    denom = np.sqrt(w2 * diff_v2)

    u1 = np.vstack([u, v, w])
    u2 = np.vstack([w / diff_v, np.zeros(n), -u / diff_v])
    u3 = np.vstack([-u * v / diff_v, diff_v, -v * (w / diff_v)])

    cart[0:3, :] = r * u1

    c1 = (u_dot * diff_v2 + u * v * v_dot) / denom
    c2 = v_dot / diff_v

    cart[3:6, :] = r_dot * u1 + (r * c1) * u2 + (r * c2) * u3

    if num_dim > 6:
        r_ddot = x_arr[6, :]
        u_ddot = x_arr[7, :]
        v_ddot = x_arr[8, :]

        c3 = -((w + u**2 / w) * (u_dot - u_dot * v**2 + u * v * v_dot)) / diff_v**3
        c4 = v * (-(u**2) / w - w) * (-u_dot * diff_v2 - u * v * v_dot) / diff_v2**2
        c5 = -v_dot / diff_v
        c6 = (w / diff_v) * (-v * u_dot * diff_v2 - u * v_dot) / diff_v**3 + u * (
            diff_v / w
        ) * (-u * v * u_dot * diff_v2 - u**2 * v_dot + v_dot * diff_v2**2) / diff_v**5

        c1_dot = (
            (u_dot * diff_v2 + u * v * v_dot)
            * (v * v_dot * (2.0 - u**2 - 2.0 * v**2) + u * u_dot * diff_v2)
        ) / denom**3 + (
            u_ddot * diff_v2 - v * u_dot * v_dot + u * (v * v_ddot + v_dot**2)
        ) / denom
        c2_dot = (v_ddot * diff_v2 + v * v_dot**2) / diff_v**3

        a1 = r_ddot + r * (c1 * c3 + c2 * c5)
        a2 = 2.0 * r_dot * c1 + r * (c1_dot + c2 * c6)
        a3 = 2.0 * r_dot * c2 + r * (c2_dot + c1 * c4)

        cart[6:9, :] = a1 * u1 + a2 * u2 + a3 * u3
    return cart


def camera_coords2uv(
    z_cam: ArrayLike,
    a: ArrayLike,
    m: Optional[ArrayLike] = None,
    include_w: bool = True,
) -> NDArray[np.floating]:
    """
    Convert camera pixel coordinates to u-v(-w) direction cosines.

    Parameters
    ----------
    z_cam : array_like
        (2, N) camera coordinates [x, y] in the image plane.
    a : array_like
        (3, 3) camera intrinsics-style matrix; its third row must be
        [0, 0, a33].
    m : array_like, optional
        (3, 3) rotation from the global frame to the camera's frame;
        the returned directions are rotated back into the global
        frame. Default identity (no rotation applied).
    include_w : bool, optional
        Include the third component. Default True.

    Returns
    -------
    dir_vecs : ndarray
        (3, N) u-v-w unit vectors, or (2, N) u-v pairs when
        ``include_w`` is False.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.diag([500.0, 500.0, 1.0])
    >>> d = camera_coords2uv(np.array([0.0, 0.0]), a)
    >>> np.round(d.ravel(), 12)
    array([0., 0., 1.])

    Notes
    -----
    Port of ``cameraCoords2UVCoords.m``.
    """
    a = np.asarray(a, dtype=np.float64)
    if not (a[2, 0] == 0.0 and a[2, 1] == 0.0):
        raise ValueError("The third row of a has the wrong format.")

    z_cam_arr = _as_columns(z_cam, 2)

    a33 = a[2, 2]
    a11, a12, a13 = a[0, :] / a33
    a21, a22, a23 = a[1, :] / a33

    x_c = z_cam_arr[0, :]
    y_c = z_cam_arr[1, :]

    denom = np.sqrt(
        a13**2 * (a21**2 + a22**2)
        - 2.0 * a11 * a13 * a21 * a23
        - 2.0 * a12 * a22 * (a11 * a21 + a13 * a23)
        + a12**2 * (a21**2 + a23**2)
        + a11**2 * (a22**2 + a23**2)
        + (-2.0 * a13 * (a21**2 + a22**2) + 2.0 * (a11 * a21 + a12 * a22) * a23) * x_c
        + (a21**2 + a22**2) * x_c**2
        + (2.0 * a13 * (a11 * a21 + a12 * a22) - 2.0 * (a11**2 + a12**2) * a23) * y_c
        - 2.0 * (a11 * a21 + a12 * a22) * x_c * y_c
        + (a11**2 + a12**2) * y_c**2
    )
    sign_val = np.sign(a11 * a22 - a12 * a21)

    u = sign_val * (-a13 * a22 + a22 * x_c + a12 * (a23 - y_c)) / denom
    v = sign_val * (a13 * a21 - a21 * x_c - a11 * (a23 - y_c)) / denom

    if m is not None:
        w = np.sqrt(np.maximum(0.0, 1.0 - u**2 - v**2))
        uvw = np.asarray(m, dtype=np.float64).T @ np.vstack([u, v, w])
        return uvw if include_w else uvw[:2, :]
    if include_w:
        w = np.sqrt(np.maximum(0.0, 1.0 - u**2 - v**2))
        return np.vstack([u, v, w])
    return np.vstack([u, v])


__all__ = [
    "camera_coords2uv",
    "cart2ruv_bistatic",
    "ruv2cart_bistatic",
    "ruv2ruv",
    "spher_ang2uv",
    "state_ruv2cart",
    "uv2spher_ang",
]
