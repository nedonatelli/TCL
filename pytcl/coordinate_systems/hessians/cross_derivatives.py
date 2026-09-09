"""
Cross gradients and Hessians between angular measurement systems.

Ports of the MATLAB TCL's ``Jacobians/Cross_Gradients`` and
``Hessians/Cross_Hessians``: derivatives of one angular
parameterization with respect to another -- u-v(-w) direction cosines
versus spherical angles (with independently rotated sensor frames
``ms``/``muv``), and the 2-D direction cosine versus polar angle
pair.

The 2-D functions' MATLAB sources call ``rotMat2D2Angle``, which is
not defined anywhere in the MATLAB library -- passing rotation
matrices there errors upstream. The obvious intended helper (the
angle of a 2-D rotation matrix) is implemented here, so the rotation
arguments work.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "uv_spher_ang_cross_grad",
    "spher_ang_uv_cross_grad",
    "u_polar_2d_cross_grad",
    "polar_u_2d_cross_grad",
    "uv_spher_ang_cross_hessian",
    "spher_ang_uv_cross_hessian",
    "u_polar_2d_cross_hessian",
    "polar_u_2d_cross_hessian",
]


def _rot_mat_2d_to_angle(m: ArrayLike | None) -> float:
    # The helper the MATLAB sources reference but never define.
    if m is None:
        return 0.0
    mm = np.asarray(m, dtype=np.float64)
    return float(np.arctan2(mm[1, 0], mm[0, 0]))


def _mix(ms: ArrayLike | None, muv: ArrayLike | None, uv_first: bool):
    s = np.eye(3) if ms is None else np.asarray(ms, dtype=np.float64)
    u = np.eye(3) if muv is None else np.asarray(muv, dtype=np.float64)
    return u @ s.T if uv_first else s @ u.T


def uv_spher_ang_cross_grad(
    az_el: ArrayLike,
    system_type: int = 0,
    include_w: bool = False,
    ms: ArrayLike | None = None,
    muv: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of u-v(-w) direction cosines w.r.t. spherical angles.

    The spherical angles are defined in the frame rotated by ``ms``,
    the direction cosines in the frame rotated by ``muv``.

    Port of ``uvSpherAngCrossGrad``.

    Parameters
    ----------
    az_el : array_like
        (2,) azimuth and elevation in radians.
    system_type : int, optional
        Spherical angle convention 0-3. Default 0.
    include_w : bool, optional
        Also return the third direction cosine's row. Default False.
    ms, muv : array_like, optional
        (3, 3) global-to-local rotations of the spherical and u-v
        sensor frames. Default: identity.

    Returns
    -------
    J : ndarray
        (2, 2) d[u; v]/d[az, el] ((3, 2) with ``include_w``).

    Examples
    --------
    >>> J = uv_spher_ang_cross_grad([0.5, 0.2])
    >>> J.shape
    (2, 2)
    """
    az, el = np.asarray(az_el, dtype=np.float64).reshape(-1)[:2]
    m = _mix(ms, muv, uv_first=True)
    sa, ca = np.sin(az), np.cos(az)
    se, ce = np.sin(el), np.cos(el)
    rows = 3 if include_w else 2
    j = np.zeros((rows, 2))
    for i in range(rows):
        m1, m2, m3 = m[i, 0], m[i, 1], m[i, 2]
        if system_type == 0:
            j[i, 0] = ce * (m2 * ca - m1 * sa)
            j[i, 1] = m3 * ce - (m1 * ca + m2 * sa) * se
        elif system_type == 1:
            j[i, 0] = ce * (m1 * ca - m3 * sa)
            j[i, 1] = m2 * ce - (m3 * ca + m1 * sa) * se
        elif system_type == 2:
            j[i, 0] = (m2 * ca - m1 * sa) * se
            j[i, 1] = m1 * ca * ce + m2 * ce * sa - m3 * se
        elif system_type == 3:
            j[i, 0] = ce * (m1 * ca - m2 * sa)
            j[i, 1] = m3 * ce - (m2 * ca + m1 * sa) * se
        else:
            raise ValueError("Invalid system type specified.")
    return j


def spher_ang_uv_cross_grad(
    uv: ArrayLike,
    system_type: int = 0,
    ms: ArrayLike | None = None,
    muv: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of spherical angles w.r.t. u-v direction cosines.

    Port of ``spherAngUvCrossGrad``.

    Parameters
    ----------
    uv : array_like
        (2,) [u, v] (w inferred on the +w hemisphere) or (3,)
        [u, v, w].
    system_type : int, optional
        Spherical angle convention 0-3. Default 0.
    ms, muv : array_like, optional
        (3, 3) frame rotations as in :func:`uv_spher_ang_cross_grad`.

    Returns
    -------
    J : ndarray
        (2, 2) d[az; el]/d[u, v].

    Examples
    --------
    >>> J = spher_ang_uv_cross_grad([0.1, 0.2])
    >>> J.shape
    (2, 2)
    """
    zv = np.asarray(uv, dtype=np.float64).reshape(-1)
    if zv.size > 2:
        u, v, w = zv[:3]
    else:
        u, v = zv[:2]
        w = np.sqrt(1.0 - u**2 - v**2)
    m = _mix(ms, muv, uv_first=False)
    u1, v1, w1 = m @ (u, v, w)
    du1du = m[0, 0] - m[0, 2] * u / w
    du1dv = m[0, 1] - m[0, 2] * v / w
    dv1du = m[1, 0] - m[1, 2] * u / w
    dv1dv = m[1, 1] - m[1, 2] * v / w
    dw1du = m[2, 0] - m[2, 2] * u / w
    dw1dv = m[2, 1] - m[2, 2] * v / w

    j = np.zeros((2, 2))
    if system_type in (0, 2, 3):
        denom1 = u1**2 + v1**2
        denom2 = np.sqrt(1.0 - w1**2)
        j[0, 0] = (-v1 * du1du + u1 * dv1du) / denom1
        j[0, 1] = (-v1 * du1dv + u1 * dv1dv) / denom1
        j[1, 0] = dw1du / denom2
        j[1, 1] = dw1dv / denom2
        if system_type == 2:
            j[1, :] = -j[1, :]
        elif system_type == 3:
            j[0, :] = -j[0, :]
    elif system_type == 1:
        denom1 = u1**2 + w1**2
        denom2 = np.sqrt(1.0 - v1**2)
        j[0, 0] = (w1 * du1du - u1 * dw1du) / denom1
        j[0, 1] = (w1 * du1dv - u1 * dw1dv) / denom1
        j[1, 0] = dv1du / denom2
        j[1, 1] = dv1dv / denom2
    else:
        raise ValueError("Invalid system type specified.")
    return j


def u_polar_2d_cross_grad(
    azimuth: ArrayLike,
    system_type: int = 0,
    mp: ArrayLike | None = None,
    mu: ArrayLike | None = None,
    include_v: bool = False,
) -> NDArray[np.float64]:
    """
    Gradient of the 2-D direction cosine(s) w.r.t. the polar angle.

    Port of ``uPolar2DCrossGrad`` (whose rotation arguments are
    unusable upstream; see the module docstring).

    Parameters
    ----------
    azimuth : array_like
        Scalar or (N,) polar angles in radians.
    system_type : int, optional
        Polar convention (0 or 1). Default 0.
    mp, mu : array_like, optional
        (2, 2) rotations of the polar and direction-cosine frames.
        Default: identity.
    include_v : bool, optional
        Also return the second direction cosine's row. Default False.

    Returns
    -------
    J : ndarray
        (1, N) ((2, N) with ``include_v``).

    Examples
    --------
    >>> u_polar_2d_cross_grad(0.0)
    array([[-0.]])
    """
    theta_u = _rot_mat_2d_to_angle(mu)
    theta_p = _rot_mat_2d_to_angle(mp)
    az = np.atleast_1d(np.asarray(azimuth, dtype=np.float64)).reshape(-1)
    rows = 2 if include_v else 1
    j = np.zeros((rows, az.size))
    if system_type == 0:
        a = az - theta_p + theta_u
        j[0] = -np.sin(a)
        if include_v:
            j[1] = np.cos(a)
    elif system_type == 1:
        a = az + theta_p - theta_u
        j[0] = np.cos(a)
        if include_v:
            j[1] = -np.sin(a)
    else:
        raise ValueError("Invalid system type specified.")
    return j


def polar_u_2d_cross_grad(
    u_list: ArrayLike, system_type: int = 0
) -> NDArray[np.float64]:
    """
    Gradient of the polar angle w.r.t. the 2-D direction cosine(s).

    Port of ``polarU2DCrossGrad``.

    Parameters
    ----------
    u_list : array_like
        (1, N) or (N,) u values (v inferred on the +v half-plane), or
        (2, N) [u; v] columns.
    system_type : int, optional
        Polar convention (0 or 1). Default 0.

    Returns
    -------
    J : ndarray
        (1, N) d(az)/du, or (2, N) [d(az)/du; d(az)/dv] when v was
        given.

    Examples
    --------
    >>> polar_u_2d_cross_grad(0.0)
    array([[-1.]])
    """
    z = np.asarray(u_list, dtype=np.float64)
    if z.ndim < 2:
        z = z.reshape(1, -1)
    has_v = z.shape[0] > 1
    n = z.shape[1]
    j = np.zeros((1 + has_v, n))
    for k in range(n):
        u = z[0, k]
        v = z[1, k] if has_v else np.sqrt(1.0 - u**2)
        if system_type == 0:
            j[0, k] = -1.0 / v
            if has_v:
                j[1, k] = 1.0 / u
        elif system_type == 1:
            j[0, k] = 1.0 / v
            if has_v:
                j[1, k] = -1.0 / u
        else:
            raise ValueError("Invalid system type specified.")
    return j


def uv_spher_ang_cross_hessian(
    az_el: ArrayLike,
    system_type: int = 0,
    include_w: bool = False,
    ms: ArrayLike | None = None,
    muv: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Hessians of u-v(-w) direction cosines w.r.t. spherical angles.

    Port of ``uvSpherAngCrossHessian``.

    Parameters
    ----------
    az_el, system_type, include_w, ms, muv
        As in :func:`uv_spher_ang_cross_grad`.

    Returns
    -------
    H : ndarray
        (2, 2, 2) stack over (az, el) for u and v ((2, 2, 3) with
        ``include_w``).

    Examples
    --------
    >>> H = uv_spher_ang_cross_hessian([0.5, 0.2])
    >>> H.shape
    (2, 2, 2)
    """
    az, el = np.asarray(az_el, dtype=np.float64).reshape(-1)[:2]
    m = _mix(ms, muv, uv_first=True)
    sa, ca = np.sin(az), np.cos(az)
    se, ce = np.sin(el), np.cos(el)
    rows = 3 if include_w else 2
    out = np.zeros((2, 2, rows))
    for i in range(rows):
        m1, m2, m3 = m[i, 0], m[i, 1], m[i, 2]
        if system_type == 0:
            daz2 = -ce * (m1 * ca + m2 * sa)
            dazdel = (-m2 * ca + m1 * sa) * se
            del2 = -ce * (m1 * ca + m2 * sa) - m3 * se
        elif system_type == 1:
            daz2 = -ce * (m3 * ca + m1 * sa)
            dazdel = (-m1 * ca + m3 * sa) * se
            del2 = -ce * (m3 * ca + m1 * sa) - m2 * se
        elif system_type == 2:
            daz2 = -(m1 * ca + m2 * sa) * se
            dazdel = ce * (m2 * ca - m1 * sa)
            del2 = -m3 * ce - (m1 * ca + m2 * sa) * se
        elif system_type == 3:
            daz2 = -ce * (m2 * ca + m1 * sa)
            dazdel = (-m1 * ca + m2 * sa) * se
            del2 = -ce * (m2 * ca + m1 * sa) - m3 * se
        else:
            raise ValueError("Invalid system type specified.")
        out[0, 0, i] = daz2
        out[0, 1, i] = out[1, 0, i] = dazdel
        out[1, 1, i] = del2
    return out


def spher_ang_uv_cross_hessian(
    uv: ArrayLike,
    system_type: int = 0,
    ms: ArrayLike | None = None,
    muv: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Hessians of spherical angles w.r.t. u-v direction cosines.

    Port of ``spherAngUvCrossHessian``.

    Parameters
    ----------
    uv, system_type, ms, muv
        As in :func:`spher_ang_uv_cross_grad`.

    Returns
    -------
    H : ndarray
        (2, 2, 2) stack over (u, v) for azimuth and elevation.

    Examples
    --------
    >>> H = spher_ang_uv_cross_hessian([0.1, 0.2])
    >>> H.shape
    (2, 2, 2)
    """
    zv = np.asarray(uv, dtype=np.float64).reshape(-1)
    if zv.size > 2:
        u, v, w = zv[:3]
    else:
        u, v = zv[:2]
        w = np.sqrt(1.0 - u**2 - v**2)
    m = _mix(ms, muv, uv_first=False)
    u1, v1, w1 = m @ (u, v, w)
    du1du = m[0, 0] - m[0, 2] * u / w
    du1dv = m[0, 1] - m[0, 2] * v / w
    dv1du = m[1, 0] - m[1, 2] * u / w
    dv1dv = m[1, 1] - m[1, 2] * v / w
    dw1du = m[2, 0] - m[2, 2] * u / w
    dw1dv = m[2, 1] - m[2, 2] * v / w
    w3 = w**3
    d2u1dudu = m[0, 2] * (v**2 - 1.0) / w3
    d2u1dudv = -m[0, 2] * u * v / w3
    d2u1dvdv = m[0, 2] * (u**2 - 1.0) / w3
    d2v1dudu = m[1, 2] * (v**2 - 1.0) / w3
    d2v1dudv = -m[1, 2] * u * v / w3
    d2v1dvdv = m[1, 2] * (u**2 - 1.0) / w3
    d2w1dudu = m[2, 2] * (v**2 - 1.0) / w3
    d2w1dudv = -m[2, 2] * u * v / w3
    d2w1dvdv = m[2, 2] * (u**2 - 1.0) / w3

    if system_type in (0, 2, 3):
        denom1 = (u1**2 + v1**2) ** 2
        denom2 = np.sqrt(1.0 - w1**2) ** 3
        dazdu2 = (
            -2 * u1 * (2 * du1du * dv1du * u1 - du1du**2 * v1 + dv1du**2 * v1)
            + (2 * du1du * dv1du + d2v1dudu * u1 - d2u1dudu * v1) * (u1**2 + v1**2)
        ) / denom1
        dazdudv = (
            d2v1dudv * u1**3
            + v1**2 * (du1dv * dv1du + du1du * dv1dv - d2u1dudv * v1)
            - u1**2 * (du1dv * dv1du + du1du * dv1dv + d2u1dudv * v1)
            + u1 * v1 * (2 * du1du * du1dv - 2 * dv1du * dv1dv + d2v1dudv * v1)
        ) / denom1
        dazdv2 = (
            -2 * u1 * (2 * du1dv * dv1dv * u1 - du1dv**2 * v1 + dv1dv**2 * v1)
            + (2 * du1dv * dv1dv + d2v1dvdv * u1 - d2u1dvdv * v1) * (u1**2 + v1**2)
        ) / denom1
        deldu2 = (d2w1dudu + dw1du**2 * w1 - d2w1dudu * w1**2) / denom2
        deldudv = (d2w1dudv + dw1du * dw1dv * w1 - d2w1dudv * w1**2) / denom2
        deldv2 = (d2w1dvdv + dw1dv**2 * w1 - d2w1dvdv * w1**2) / denom2
        if system_type == 2:
            deldu2, deldudv, deldv2 = (
                (-(dw1du**2) * w1 + d2w1dudu * (-1.0 + w1**2)) / denom2,
                (-dw1du * dw1dv * w1 + d2w1dudv * (-1.0 + w1**2)) / denom2,
                (-(dw1dv**2) * w1 + d2w1dvdv * (-1.0 + w1**2)) / denom2,
            )
        elif system_type == 3:
            dazdu2, dazdudv, dazdv2 = -dazdu2, -dazdudv, -dazdv2
    elif system_type == 1:
        denom1 = (u1**2 + w1**2) ** 2
        denom2 = np.sqrt(1.0 - v1**2) ** 3
        dazdu2 = (
            2 * u1 * (2 * du1du * dw1du * u1 - du1du**2 * w1 + dw1du**2 * w1)
            + (-2 * du1du * dw1du - d2w1dudu * u1 + d2u1dudu * w1) * (u1**2 + w1**2)
        ) / denom1
        dazdudv = (
            u1**2 * (du1dv * dw1du + du1du * dw1dv - d2w1dudv * u1)
            + u1 * (-2 * du1du * du1dv + 2 * dw1du * dw1dv + d2u1dudv * u1) * w1
            - (du1dv * dw1du + du1du * dw1dv + d2w1dudv * u1) * w1**2
            + d2u1dudv * w1**3
        ) / denom1
        dazdv2 = (
            2 * u1 * (2 * du1dv * dw1dv * u1 - du1dv**2 * w1 + dw1dv**2 * w1)
            + (-2 * du1dv * dw1dv - d2w1dvdv * u1 + d2u1dvdv * w1) * (u1**2 + w1**2)
        ) / denom1
        deldu2 = (d2v1dudu + dv1du**2 * v1 - d2v1dudu * v1**2) / denom2
        deldudv = (d2v1dudv + dv1du * dv1dv * v1 - d2v1dudv * v1**2) / denom2
        deldv2 = (d2v1dvdv + dv1dv**2 * v1 - d2v1dvdv * v1**2) / denom2
    else:
        raise ValueError("Invalid system type specified.")

    out = np.zeros((2, 2, 2))
    out[0, 0, 0] = dazdu2
    out[0, 1, 0] = out[1, 0, 0] = dazdudv
    out[1, 1, 0] = dazdv2
    out[0, 0, 1] = deldu2
    out[0, 1, 1] = out[1, 0, 1] = deldudv
    out[1, 1, 1] = deldv2
    return out


def u_polar_2d_cross_hessian(
    azimuth: ArrayLike,
    system_type: int = 0,
    include_v: bool = False,
) -> NDArray[np.float64]:
    """
    Second derivative of the 2-D direction cosine(s) w.r.t. the angle.

    Port of ``uPolar2DCrossHessian``.

    Parameters
    ----------
    azimuth : array_like
        Scalar or (N,) polar angles in radians.
    system_type : int, optional
        Polar convention (0 or 1). Default 0.
    include_v : bool, optional
        Also return the second direction cosine's row. Default False.

    Returns
    -------
    H : ndarray
        (1, N) ((2, N) with ``include_v``).

    Examples
    --------
    >>> u_polar_2d_cross_hessian(0.0)
    array([[-1.]])
    """
    az = np.atleast_1d(np.asarray(azimuth, dtype=np.float64)).reshape(-1)
    rows = 2 if include_v else 1
    h = np.zeros((rows, az.size))
    if system_type == 0:
        h[0] = -np.cos(az)
        if include_v:
            h[1] = -np.sin(az)
    elif system_type == 1:
        h[0] = -np.sin(az)
        if include_v:
            h[1] = -np.cos(az)
    else:
        raise ValueError("Invalid system type specified.")
    return h


def polar_u_2d_cross_hessian(
    u_list: ArrayLike, system_type: int = 0
) -> NDArray[np.float64]:
    """
    Second derivatives of the polar angle w.r.t. the direction cosines.

    Port of ``polarU2DCrossHessian``.

    Parameters
    ----------
    u_list : array_like
        (1, N) or (N,) u values (v inferred on the +v half-plane), or
        (2, N) [u; v] columns.
    system_type : int, optional
        Polar convention (0 or 1). Default 0.

    Returns
    -------
    H : ndarray
        (1, 1, N), or (2, 2, N) when v was given, matching the MATLAB
        layout.

    Examples
    --------
    >>> polar_u_2d_cross_hessian(0.0)[..., 0]
    array([[-0.]])
    """
    z = np.asarray(u_list, dtype=np.float64)
    if z.ndim < 2:
        z = z.reshape(1, -1)
    has_v = z.shape[0] > 1
    n = z.shape[1]
    dim = 1 + has_v
    h = np.zeros((dim, dim, n))
    for k in range(n):
        u = z[0, k]
        v = z[1, k] if has_v else np.sqrt(1.0 - u**2)
        sign = 1.0 if system_type == 0 else -1.0
        if system_type not in (0, 1):
            raise ValueError("Invalid system type specified.")
        h[0, 0, k] = -sign * u / v**3
        if has_v:
            h[0, 1, k] = sign / v**2
            h[1, 0, k] = -sign / u**2
            h[1, 1, k] = sign * v / u**3
    return h
