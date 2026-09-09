"""
Hessians of individual measurement components.

Ports of the MATLAB TCL's ``Coordinate_Systems/Hessians/
Component_Hessians``: second derivatives of bistatic range, spherical
angles and direction cosines with respect to Cartesian position.
Conventions (``system_type``, ``m``, ``l_tx``/``l_rx``,
``use_half_range``) follow
:mod:`pytcl.coordinate_systems.jacobians.component_gradients`. All
functions take one point; a Hessian stack has shape
``(n, n, num_out)``.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "range_hessian",
    "spher_ang_hessian",
    "uv_hessian",
    "u_hessian_2d",
    "u_hessian_3d",
]


def _p(x: ArrayLike, size: int) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).reshape(-1)[:size]


def range_hessian(
    x: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Hessian of bistatic range with respect to Cartesian position.

    A transmitter collocated with the target contributes nothing, as
    in the gradient. Port of ``rangeHessian``.

    Parameters
    ----------
    x : array_like
        (d,) target position, d in {1, 2, 3}.
    use_half_range : bool, optional
        Divide by two for one-way ranges. Default False.
    l_tx, l_rx : array_like, optional
        Transmitter/receiver positions. Default: the origin.

    Returns
    -------
    H : ndarray
        (d, d) Hessian.

    Examples
    --------
    >>> H = range_hessian([3.0, 4.0])
    >>> H.shape
    (2, 2)
    """
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    d = xv.size
    tx = np.zeros(d) if l_tx is None else _p(l_tx, d)
    rx = np.zeros(d) if l_rx is None else _p(l_rx, d)

    def _term(delta: NDArray[np.float64]) -> NDArray[np.float64]:
        nrm = np.linalg.norm(delta)
        return -np.outer(delta, delta) / nrm**3 + np.eye(d) / nrm

    delta_rx = xv - rx
    delta_tx = xv - tx
    h = _term(delta_rx)
    if np.linalg.norm(delta_tx) != 0:
        h = h + _term(delta_tx)
    if use_half_range:
        h = h / 2.0
    return h


def spher_ang_hessian(
    x: ArrayLike,
    system_type: int = 0,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Hessians of spherical azimuth and elevation w.r.t. position.

    Port of ``spherAngHessian``.

    Parameters
    ----------
    x : array_like
        (3,) global Cartesian position.
    system_type : int, optional
        Angle convention 0-3. Default 0.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local rotation. Default: identity.

    Returns
    -------
    H : ndarray
        (3, 3, 2) stack: H[:, :, 0] the azimuth Hessian, H[:, :, 1]
        the elevation Hessian.

    Examples
    --------
    >>> H = spher_ang_hessian([1e3, 2e3, 500.0])
    >>> H.shape
    (3, 3, 2)
    """
    rot = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(3) if l_rx is None else _p(l_rx, 3)
    xv, yv, zv = rot @ (_p(x, 3) - rx)

    haz = np.zeros((3, 3))
    hel = np.zeros((3, 3))
    r4 = (xv**2 + yv**2 + zv**2) ** 2
    if system_type in (0, 2, 3):
        r2xy = xv**2 + yv**2
        rxy = np.sqrt(r2xy)
        haz[0, 0] = 2 * xv * yv / r2xy**2
        haz[1, 1] = -2 * xv * yv / r2xy**2
        haz[0, 1] = haz[1, 0] = -(xv - yv) * (xv + yv) / r2xy**2
        hel[0, 0] = (zv * (2 * xv**4 + xv**2 * yv**2 - yv**2 * (yv**2 + zv**2))) / (
            rxy**3 * r4
        )
        hel[1, 1] = -(zv * (xv**4 - 2 * yv**4 + xv**2 * (-(yv**2) + zv**2))) / (
            rxy**3 * r4
        )
        hel[2, 2] = -2 * rxy * zv / r4
        hel[0, 1] = hel[1, 0] = xv * yv * zv * (3 * r2xy + zv**2) / (rxy**3 * r4)
        hel[0, 2] = hel[2, 0] = -(xv * (xv**2 + yv**2 - zv**2)) / (rxy * r4)
        hel[1, 2] = hel[2, 1] = -(yv * (xv**2 + yv**2 - zv**2)) / (rxy * r4)
        if system_type == 2:
            hel = -hel
        elif system_type == 3:
            haz = -haz
    elif system_type == 1:
        r2xz = xv**2 + zv**2
        rxz = np.sqrt(r2xz)
        haz[0, 0] = -2 * xv * zv / r2xz**2
        haz[2, 2] = 2 * xv * zv / r2xz**2
        haz[0, 2] = haz[2, 0] = (xv - zv) * (xv + zv) / r2xz**2
        hel[0, 0] = (yv * (2 * xv**4 + xv**2 * zv**2 - zv**2 * (yv**2 + zv**2))) / (
            rxz**3 * r4
        )
        hel[1, 1] = -2 * yv * rxz / r4
        hel[2, 2] = -(yv * (xv**4 - 2 * zv**4 + xv**2 * (yv - zv) * (yv + zv))) / (
            rxz**3 * r4
        )
        hel[0, 1] = hel[1, 0] = -(xv * (xv**2 - yv**2 + zv**2)) / (rxz * r4)
        hel[0, 2] = hel[2, 0] = (
            xv * yv * zv * (3 * xv**2 + yv**2 + 3 * zv**2) / (rxz**3 * r4)
        )
        hel[1, 2] = hel[2, 1] = -(zv * (xv**2 - yv**2 + zv**2)) / (rxz * r4)
    else:
        raise ValueError("Invalid system type specified.")

    out = np.zeros((3, 3, 2))
    out[:, :, 0] = rot.T @ haz @ rot
    out[:, :, 1] = rot.T @ hel @ rot
    return out


def uv_hessian(
    x: ArrayLike,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
    include_w: bool = False,
) -> NDArray[np.float64]:
    """
    Hessians of the u-v(-w) direction cosines w.r.t. position.

    Port of ``uvHessian``.

    Parameters
    ----------
    x : array_like
        (3,) global Cartesian position.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local rotation. Default: identity.
    include_w : bool, optional
        Also return the third direction cosine's Hessian. Default
        False.

    Returns
    -------
    H : ndarray
        (3, 3, 2) stack ((3, 3, 3) with ``include_w``).

    Examples
    --------
    >>> H = uv_hessian([1e3, 2e3, 5e3])
    >>> H.shape
    (3, 3, 2)
    """
    rot = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(3) if l_rx is None else _p(l_rx, 3)
    local = rot @ (_p(x, 3) - rx)
    xv, yv, zv = local
    x2, y2, z2 = xv * xv, yv * yv, zv * zv
    r5 = np.linalg.norm(local) ** 5

    hu = (
        np.array(
            [
                [
                    -3 * xv * (y2 + z2),
                    -yv * (-2 * x2 + y2 + z2),
                    -zv * (-2 * x2 + y2 + z2),
                ],
                [-yv * (-2 * x2 + y2 + z2), -xv * (x2 - 2 * y2 + z2), 3 * xv * yv * zv],
                [-zv * (-2 * x2 + y2 + z2), 3 * xv * yv * zv, -xv * (x2 + y2 - 2 * z2)],
            ]
        )
        / r5
    )
    hv = (
        np.array(
            [
                [-yv * (-2 * x2 + y2 + z2), -xv * (x2 - 2 * y2 + z2), 3 * xv * yv * zv],
                [
                    -xv * (x2 - 2 * y2 + z2),
                    -3 * yv * (x2 + z2),
                    -zv * (x2 - 2 * y2 + z2),
                ],
                [3 * xv * yv * zv, -zv * (x2 - 2 * y2 + z2), -yv * (x2 + y2 - 2 * z2)],
            ]
        )
        / r5
    )

    rows = 3 if include_w else 2
    out = np.zeros((3, 3, rows))
    out[:, :, 0] = rot.T @ hu @ rot
    out[:, :, 1] = rot.T @ hv @ rot
    if include_w:
        hw = (
            np.array(
                [
                    [
                        -zv * (-2 * x2 + y2 + z2),
                        3 * xv * yv * zv,
                        -xv * (x2 + y2 - 2 * z2),
                    ],
                    [
                        3 * xv * yv * zv,
                        -zv * (x2 - 2 * y2 + z2),
                        -yv * (x2 + y2 - 2 * z2),
                    ],
                    [
                        -xv * (x2 + y2 - 2 * z2),
                        -yv * (x2 + y2 - 2 * z2),
                        -3 * (x2 + y2) * zv,
                    ],
                ]
            )
            / r5
        )
        out[:, :, 2] = rot.T @ hw @ rot
    return out


def u_hessian_2d(
    x: ArrayLike,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
    include_v: bool = False,
) -> NDArray[np.float64]:
    """
    Hessian of the 2-D direction cosine u (optionally v) w.r.t. position.

    Port of ``uHessian2D``.

    Parameters
    ----------
    x : array_like
        (2,) position.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (2, 2) global-to-local rotation. Default: identity.
    include_v : bool, optional
        Also return the second direction cosine's Hessian. Default
        False.

    Returns
    -------
    H : ndarray
        (2, 2, 1) stack ((2, 2, 2) with ``include_v``).

    Examples
    --------
    >>> H = u_hessian_2d([3.0, 4.0])
    >>> H.shape
    (2, 2, 1)
    """
    rot = np.eye(2) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(2) if l_rx is None else _p(l_rx, 2)
    local = rot @ (_p(x, 2) - rx)
    xv, yv = local
    r5 = np.linalg.norm(local) ** 5

    hu = (
        np.array(
            [
                [-3 * xv * yv**2, -yv * (-2 * xv**2 + yv**2)],
                [-yv * (-2 * xv**2 + yv**2), -xv * (xv**2 - 2 * yv**2)],
            ]
        )
        / r5
    )
    rows = 2 if include_v else 1
    out = np.zeros((2, 2, rows))
    out[:, :, 0] = rot.T @ hu @ rot
    if include_v:
        hv = (
            np.array(
                [
                    [-yv * (-2 * xv**2 + yv**2), -xv * (xv**2 - 2 * yv**2)],
                    [-xv * (xv**2 - 2 * yv**2), -3 * xv**2 * yv],
                ]
            )
            / r5
        )
        out[:, :, 1] = rot.T @ hv @ rot
    return out


def u_hessian_3d(
    x: ArrayLike,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Hessian of the first 3-D direction cosine u w.r.t. position.

    Port of ``uHessian3D``.

    Parameters
    ----------
    x : array_like
        (3,) position.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local rotation. Default: identity.

    Returns
    -------
    H : ndarray
        (3, 3, 1) stack.

    Examples
    --------
    >>> H = u_hessian_3d([1e3, 2e3, 5e3])
    >>> H.shape
    (3, 3, 1)
    """
    return uv_hessian(x, l_rx, m)[:, :, :1]
