"""
Gradients of individual measurement components.

Ports of the MATLAB TCL's ``Coordinate_Systems/Jacobians/
Component_Gradients``: partial derivatives of bistatic range, range
rate, spherical/polar angles, direction cosines and TDOA with respect
to Cartesian position (or state, for range rate). These are the
building blocks the full measurement Jacobians compose.

Conventions shared across the family (matching the MATLAB sources):

- Positions/states are column vectors; a ``(d,)`` input yields one 2-D
  Jacobian, a ``(d, N)`` input a ``(rows, d, N)`` stack.
- ``l_tx``/``l_rx`` are transmitter/receiver positions (states, for
  range rate: position stacked over velocity); omitted means the
  origin (at rest).
- ``m`` is the rotation taking global coordinates into the receiver's
  local frame: ``x_local = m @ (x_global - l_rx)``.
- ``use_half_range`` divides bistatic quantities by two (one-way
  monostatic convention).
- ``system_type`` follows the reference integer conventions: 0 =
  azimuth from +x in the x-y plane, elevation up from that plane; 1 =
  azimuth from +z in the z-x plane, elevation toward +y; 2 = like 0
  but elevation measured down from +z (zenith angle); 3 = like 0 but
  azimuth measured clockwise from +y.

References
----------
- D. F. Crouse, "Basic tracking using nonlinear 3D monostatic and
  bistatic measurements," IEEE Aerospace and Electronic Systems
  Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.constants import SPEED_OF_LIGHT

__all__ = [
    "range_gradient",
    "range_rate_gradient",
    "spher_ang_gradient",
    "pol_ang_gradient",
    "uv_gradient",
    "u_gradient_2d",
    "u_gradient_3d",
    "tdoa_gradient",
]


def _columns(x: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    """Return (d, N) columns and whether the input was a single point."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis], True
    return arr, False


def _pack(stack: NDArray[np.float64], single: bool) -> NDArray[np.float64]:
    """Collapse a (rows, d, N) stack to (rows, d) for a single point."""
    return stack[:, :, 0] if single else stack


def range_gradient(
    x: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of bistatic range with respect to Cartesian position.

    A transmitter exactly collocated with the target is assumed to
    move with it, dropping the transmitter term.

    Port of ``rangeGradient``.

    Parameters
    ----------
    x : array_like
        (d,) or (d, N) target positions, d in {1, 2, 3}.
    use_half_range : bool, optional
        Divide the bistatic range by two. Default False.
    l_tx, l_rx : array_like, optional
        Transmitter/receiver positions. Default: the origin.

    Returns
    -------
    J : ndarray
        (1, d) gradient, or a (1, d, N) stack.

    Examples
    --------
    >>> range_gradient([3.0, 4.0])
    array([[1.2, 1.6]])
    """
    pts, single = _columns(x)
    d, n = pts.shape
    tx = np.zeros(d) if l_tx is None else np.asarray(l_tx, dtype=np.float64)[:d]
    rx = np.zeros(d) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:d]

    out = np.zeros((1, d, n))
    for k in range(n):
        delta_rx = pts[:, k] - rx
        delta_tx = pts[:, k] - tx
        g = delta_rx / np.linalg.norm(delta_rx)
        if np.any(delta_tx != 0):
            g = g + delta_tx / np.linalg.norm(delta_tx)
        out[0, :, k] = g
    if use_half_range:
        out /= 2.0
    return _pack(out, single)


def range_rate_gradient(
    x: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of bistatic range rate with respect to the target state.

    The state stacks position over velocity (2d elements, d in
    {1, 2, 3}); ``l_tx``/``l_rx`` are transmitter/receiver *states* of
    the same layout, so moving platforms are handled. A transmitter
    collocated with the target moves with it.

    Port of ``rangeRateGradient``.

    Parameters
    ----------
    x : array_like
        (2d,) or (2d, N) target states.
    use_half_range : bool, optional
        Divide by two for one-way rates. Default False.
    l_tx, l_rx : array_like, optional
        Transmitter/receiver states. Default: the origin, at rest.

    Returns
    -------
    J : ndarray
        (1, 2d) gradient, or a (1, 2d, N) stack.

    Examples
    --------
    >>> J = range_rate_gradient([0.0, 5e3, 100.0, 0.0])
    >>> J.shape
    (1, 4)
    """
    pts, single = _columns(x)
    x_dim, n = pts.shape
    d = x_dim // 2
    tx = np.zeros(x_dim) if l_tx is None else np.asarray(l_tx, dtype=np.float64)
    rx = np.zeros(x_dim) if l_rx is None else np.asarray(l_rx, dtype=np.float64)
    if d not in (1, 2, 3):
        raise ValueError("Invalid state dimensionality.")

    def _a(v: NDArray[np.float64]) -> NDArray[np.float64]:
        # norm(v)^2 I - v v^T, written out as in the reference.
        return np.dot(v, v) * np.eye(d) - np.outer(v, v)

    out = np.zeros((1, x_dim, n))
    for k in range(n):
        xc = pts[:, k]
        j = np.zeros(x_dim)
        dtr = xc[:d] - rx[:d]
        dtl = xc[:d] - tx[:d]
        if d == 1:
            drb = dtr / np.linalg.norm(dtr)
            if dtl[0] != 0:
                drb = drb + dtl / np.linalg.norm(dtl)
            j[1] = drb[0]
        else:
            dvr = xc[d:] - rx[d:]
            dvl = xc[d:] - tx[d:]
            drb = dtr / np.linalg.norm(dtr)
            j[:d] = _a(dtr) @ dvr / np.linalg.norm(dtr) ** 3
            if np.any(dtl != 0):
                drb = drb + dtl / np.linalg.norm(dtl)
                j[:d] = j[:d] + _a(dtl) @ dvl / np.linalg.norm(dtl) ** 3
            j[d:] = drb
        out[0, :, k] = j
    if use_half_range:
        out /= 2.0
    return _pack(out, single)


def spher_ang_gradient(
    x: ArrayLike,
    system_type: int = 0,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of spherical azimuth and elevation w.r.t. position.

    Port of ``spherAngGradient``.

    Parameters
    ----------
    x : array_like
        (3,) or (3, N) global Cartesian positions.
    system_type : int, optional
        Angle convention 0-3 (see the module docstring). Default 0.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local rotation. Default: identity.

    Returns
    -------
    J : ndarray
        (2, 3) [d(az); d(el)] rows, or a (2, 3, N) stack.

    Examples
    --------
    >>> J = spher_ang_gradient([1e3, 2e3, 500.0])
    >>> J.shape
    (2, 3)
    """
    pts, single = _columns(x)
    n = pts.shape[1]
    rot = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(3) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:3]

    local = rot @ (pts[:3] - rx[:, np.newaxis])
    out = np.zeros((2, 3, n))
    for k in range(n):
        xv, yv, zv = local[:, k]
        r = np.linalg.norm(local[:, k])
        j = np.zeros((2, 3))
        if system_type == 0:
            rxy = np.sqrt(xv**2 + yv**2)
            j[0, 0] = -yv / (xv**2 + yv**2)
            j[1, 0] = -xv * zv / (r**2 * rxy)
            j[0, 1] = xv / (xv**2 + yv**2)
            j[1, 1] = -yv * zv / (r**2 * rxy)
            j[1, 2] = rxy / r**2
        elif system_type == 1:
            rzx = np.sqrt(zv**2 + xv**2)
            j[0, 0] = zv / (zv**2 + xv**2)
            j[1, 0] = -xv * yv / (r**2 * rzx)
            j[1, 1] = rzx / r**2
            j[0, 2] = -xv / (zv**2 + xv**2)
            j[1, 2] = -zv * yv / (r**2 * rzx)
        elif system_type == 2:
            rxy = np.sqrt(xv**2 + yv**2)
            j[0, 0] = -yv / (xv**2 + yv**2)
            j[1, 0] = xv * zv / (r**2 * rxy)
            j[0, 1] = xv / (xv**2 + yv**2)
            j[1, 1] = yv * zv / (r**2 * rxy)
            j[1, 2] = -rxy / r**2
        elif system_type == 3:
            rxy = np.sqrt(xv**2 + yv**2)
            j[0, 0] = yv / (xv**2 + yv**2)
            j[1, 0] = -xv * zv / (r**2 * rxy)
            j[0, 1] = -xv / (xv**2 + yv**2)
            j[1, 1] = -yv * zv / (r**2 * rxy)
            j[1, 2] = rxy / r**2
        else:
            raise ValueError("Invalid system type specified.")
        out[:, :, k] = j @ rot
    return _pack(out, single)


def pol_ang_gradient(
    x: ArrayLike,
    system_type: int = 0,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of a 2-D polar angle with respect to position.

    ``system_type`` 0 measures counterclockwise from +x, 1 clockwise
    from +y.

    Port of ``polAngGradient``.

    Parameters
    ----------
    x : array_like
        (2,) or (2, N) positions.
    system_type : int, optional
        Angle convention (0 or 1). Default 0.
    l_rx : array_like, optional
        Receiver position. Default: the origin.

    Returns
    -------
    J : ndarray
        (1, 2) gradient, or a (1, 2, N) stack.

    Examples
    --------
    >>> pol_ang_gradient([2.0, 0.0])
    array([[-0. ,  0.5]])
    """
    pts, single = _columns(x)
    n = pts.shape[1]
    rx = np.zeros(2) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:2]

    out = np.zeros((1, 2, n))
    for k in range(n):
        xl, yl = pts[:2, k] - rx
        r2 = xl**2 + yl**2
        if system_type == 0:
            out[0, :, k] = (-yl / r2, xl / r2)
        elif system_type == 1:
            out[0, :, k] = (yl / r2, -xl / r2)
        else:
            raise ValueError("Invalid system type specified.")
    return _pack(out, single)


def uv_gradient(
    x: ArrayLike,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
    include_w: bool = False,
) -> NDArray[np.float64]:
    """
    Gradient of u-v(-w) direction cosines with respect to position.

    Port of ``uvGradient``.

    Parameters
    ----------
    x : array_like
        (3,) or (3, N) global Cartesian positions.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local rotation. Default: identity.
    include_w : bool, optional
        Also return the gradient of the third direction cosine.
        Default False.

    Returns
    -------
    J : ndarray
        (2, 3) [du; dv] rows ((3, 3) with ``include_w``), or the
        corresponding stack.

    Examples
    --------
    >>> uv_gradient([0.0, 0.0, 10.0])
    array([[0.1, 0. , 0. ],
           [0. , 0.1, 0. ]])
    """
    pts, single = _columns(x)
    n = pts.shape[1]
    rot = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(3) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:3]

    rows = 3 if include_w else 2
    out = np.zeros((rows, 3, n))
    for k in range(n):
        local = rot @ (pts[:3, k] - rx)
        r = np.linalg.norm(local)
        xv, yv, zv = local
        du = np.array([(yv**2 + zv**2) / r**3, -xv * yv / r**3, -xv * zv / r**3])
        dv = np.array([-xv * yv / r**3, (xv**2 + zv**2) / r**3, -yv * zv / r**3])
        out[0, :, k] = du @ rot
        out[1, :, k] = dv @ rot
        if include_w:
            dw = np.array([-xv * zv / r**3, -yv * zv / r**3, (xv**2 + yv**2) / r**3])
            out[2, :, k] = dw @ rot
    return _pack(out, single)


def u_gradient_2d(
    x: ArrayLike,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
    include_v: bool = False,
) -> NDArray[np.float64]:
    """
    Gradient of the 2-D direction cosine u (optionally v) w.r.t. position.

    Port of ``uGradient2D``.

    Parameters
    ----------
    x : array_like
        (2,) or (2, N) positions.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (2, 2) global-to-local rotation. Default: identity.
    include_v : bool, optional
        Also return the second direction cosine's gradient. Default
        False.

    Returns
    -------
    J : ndarray
        (1, 2) ((2, 2) with ``include_v``), or the corresponding stack.

    Examples
    --------
    >>> u_gradient_2d([0.0, 4.0])
    array([[0.25, 0.  ]])
    """
    pts, single = _columns(x)
    n = pts.shape[1]
    rot = np.eye(2) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(2) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:2]

    rows = 2 if include_v else 1
    out = np.zeros((rows, 2, n))
    for k in range(n):
        local = rot @ (pts[:2, k] - rx)
        r3 = np.linalg.norm(local) ** 3
        xv, yv = local
        out[0, :, k] = np.array([yv**2 / r3, -xv * yv / r3]) @ rot
        if include_v:
            out[1, :, k] = np.array([-xv * yv / r3, xv**2 / r3]) @ rot
    return _pack(out, single)


def u_gradient_3d(
    x: ArrayLike,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Gradient of the first 3-D direction cosine u w.r.t. position.

    Port of ``uGradient3D``.

    Parameters
    ----------
    x : array_like
        (3,) or (3, N) positions.
    l_rx : array_like, optional
        Receiver position. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local rotation. Default: identity.

    Returns
    -------
    J : ndarray
        (1, 3) gradient, or a (1, 3, N) stack.

    Examples
    --------
    >>> u_gradient_3d([0.0, 5.0, 0.0])
    array([[0.2, 0. , 0. ]])
    """
    pts, single = _columns(x)
    n = pts.shape[1]
    rot = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    rx = np.zeros(3) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:3]
    out = np.zeros((1, 3, n))
    for k in range(n):
        local = rot @ (pts[:3, k] - rx)
        r = np.linalg.norm(local)
        xv, yv, zv = local
        du = np.array([(yv**2 + zv**2) / r**3, -xv * yv / r**3, -xv * zv / r**3])
        out[0, :, k] = du @ rot
    return _pack(out, single)


def tdoa_gradient(
    x: ArrayLike,
    l_ref: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    c: float = SPEED_OF_LIGHT,
) -> NDArray[np.float64]:
    """
    Gradient of a TDOA measurement with respect to position.

    The time difference of arrival between the receiver at ``l_rx``
    and the reference receiver at ``l_ref``, in seconds.

    Port of ``TDOAGradient``.

    Parameters
    ----------
    x : array_like
        (d,) or (d, N) emitter positions.
    l_ref : array_like, optional
        Reference receiver position. Default: the origin.
    l_rx : array_like, optional
        Second receiver position. Default: the origin.
    c : float, optional
        Propagation speed. Default: the vacuum speed of light.

    Returns
    -------
    J : ndarray
        (1, d) gradient, or a (1, d, N) stack.

    Examples
    --------
    >>> J = tdoa_gradient([1e3, 2e3], l_ref=[0.0, 0.0], l_rx=[5e3, 0.0])
    >>> J.shape
    (1, 2)
    """
    pts, single = _columns(x)
    d, n = pts.shape
    ref = np.zeros(d) if l_ref is None else np.asarray(l_ref, dtype=np.float64)[:d]
    rx = np.zeros(d) if l_rx is None else np.asarray(l_rx, dtype=np.float64)[:d]

    out = np.zeros((1, d, n))
    for k in range(n):
        d1 = pts[:, k] - ref
        d2 = pts[:, k] - rx
        out[0, :, k] = (d2 / np.linalg.norm(d2) - d1 / np.linalg.norm(d1)) / c
    return _pack(out, single)
