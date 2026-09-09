"""
Full measurement Jacobians for tracking measurement models.

Ports of the MATLAB TCL's top-level ``Coordinate_Systems/Jacobians``
functions and the ``Converted_Jacobians`` family: Jacobians of
bistatic spherical, polar and r-u-v(-w) measurements -- with and
without range rate -- with respect to Cartesian position or state,
composed from :mod:`.component_gradients`. The ``*_conv_*`` variants
take the measurement itself, convert it back to Cartesian and
evaluate the Jacobian there (the form needed when converting measured
covariances).

Conventions (``system_type``, ``m``, ``l_tx``/``l_rx``,
``use_half_range``) are those of :mod:`.component_gradients`. Each
function's ``use_half_range`` default matches its MATLAB source: the
spherical and polar families default to True in the monostatic call
form, the r-u-v family to False.

References
----------
- D. F. Crouse, "Basic tracking using nonlinear 3D monostatic and
  bistatic measurements," IEEE Aerospace and Electronic Systems
  Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.coordinate_systems.conversions.uv import ruv2cart_bistatic
from pytcl.coordinate_systems.jacobians.component_gradients import (
    pol_ang_gradient,
    range_gradient,
    range_rate_gradient,
    spher_ang_gradient,
    uv_gradient,
)

__all__ = [
    "calc_spher_jacob",
    "calc_spher_inv_jacob",
    "calc_spher_rr_jacob",
    "calc_polar_jacob",
    "calc_polar_rr_jacob",
    "calc_ruv_jacob",
    "calc_ruv_rr_jacob",
    "calc_cart_rr_jacob",
    "calc_spher_conv_jacob",
    "calc_polar_conv_jacob",
    "calc_polar_rr_conv_jacob",
    "calc_ruv_conv_jacob",
    "calc_ruv_rr_conv_jacob",
    "norm_vec_jacob",
]


def _vec(v: ArrayLike | None, size: int) -> NDArray[np.float64] | None:
    return None if v is None else np.asarray(v, dtype=np.float64).reshape(-1)[:size]


def _spher_half_range_default(
    use_half_range: bool | None, l_tx: ArrayLike | None
) -> bool:
    # The spherical-family MATLAB default: half range in the pure
    # monostatic call form (no transmitter given), full bistatic range
    # once a transmitter appears.
    if use_half_range is None:
        return l_tx is None
    return use_half_range


def calc_spher_jacob(
    x: ArrayLike,
    system_type: int = 0,
    use_half_range: bool | None = None,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Jacobian of a bistatic [range; azimuth; elevation] measurement.

    Port of ``calcSpherJacob``.

    Parameters
    ----------
    x : array_like
        (3,) target position in global Cartesian coordinates.
    system_type : int, optional
        Spherical angle convention 0-3. Default 0.
    use_half_range : bool, optional
        One-way range convention. Default: True when no transmitter is
        given (monostatic), False otherwise -- the MATLAB default.
    l_tx, l_rx : array_like, optional
        Transmitter/receiver positions. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local receiver rotation. Default: identity.

    Returns
    -------
    J : ndarray
        (3, 3) Jacobian; rows d(r), d(az), d(el).

    Examples
    --------
    >>> J = calc_spher_jacob([1e3, 2e3, 500.0])
    >>> J.shape
    (3, 3)
    """
    xv = _vec(x, 3)
    uhr = _spher_half_range_default(use_half_range, l_tx)
    return np.vstack(
        [
            range_gradient(xv, uhr, _vec(l_tx, 3), _vec(l_rx, 3)),
            spher_ang_gradient(xv, system_type, _vec(l_rx, 3), m),
        ]
    )


def calc_spher_inv_jacob(z: ArrayLike, system_type: int = 0) -> NDArray[np.float64]:
    """
    Jacobian of Cartesian position w.r.t. a monostatic spherical triple.

    Port of ``calcSpherInvJacob``.

    Parameters
    ----------
    z : array_like
        (3,) measurement [r, az, el].
    system_type : int, optional
        Spherical angle convention 0-3. Default 0.

    Returns
    -------
    J : ndarray
        (3, 3) Jacobian; columns d/dr, d/d(az), d/d(el).

    Examples
    --------
    >>> J = calc_spher_inv_jacob([1.0, 0.0, 0.0])
    >>> float(J[0, 0])
    1.0
    """
    r, az, el = np.asarray(z, dtype=np.float64).reshape(-1)[:3]
    sa, ca = np.sin(az), np.cos(az)
    se, ce = np.sin(el), np.cos(el)
    j = np.zeros((3, 3))
    if system_type == 0:
        j[:, 0] = (ca * ce, ce * sa, se)
        j[:, 1] = (-r * ce * sa, r * ca * ce, 0.0)
        j[:, 2] = (-r * ca * se, -r * sa * se, r * ce)
    elif system_type == 1:
        j[:, 0] = (ce * sa, se, ca * ce)
        j[:, 1] = (r * ca * ce, 0.0, -r * ce * sa)
        j[:, 2] = (-r * sa * se, r * ce, -r * ca * se)
    elif system_type == 2:
        j[:, 0] = (ca * se, sa * se, ce)
        j[:, 1] = (-r * sa * se, r * ca * se, 0.0)
        j[:, 2] = (r * ca * ce, r * ce * sa, -r * se)
    elif system_type == 3:
        j[:, 0] = (sa * ce, ca * ce, se)
        j[:, 1] = (r * ce * ca, -r * sa * ce, 0.0)
        j[:, 2] = (-r * sa * se, -r * ca * se, r * ce)
    else:
        raise ValueError("Invalid system type specified.")
    return j


def calc_spher_rr_jacob(
    x: ArrayLike,
    system_type: int = 0,
    use_half_range: bool | None = None,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Jacobian of [range; azimuth; elevation; range rate] w.r.t. state.

    The state stacks position over velocity; ``l_tx``/``l_rx`` are
    six-element states so moving platforms are handled.

    Port of ``calcSpherRRJacob``.

    Parameters
    ----------
    x : array_like
        (6,) target state.
    system_type, use_half_range, l_tx, l_rx, m
        As in :func:`calc_spher_jacob`, with states for the platforms.

    Returns
    -------
    J : ndarray
        (4, 6) Jacobian.

    Examples
    --------
    >>> J = calc_spher_rr_jacob([1e3, 2e3, 500.0, 10.0, -20.0, 5.0])
    >>> J.shape
    (4, 6)
    """
    xv = _vec(x, 6)
    uhr = _spher_half_range_default(use_half_range, l_tx)
    tx = _vec(l_tx, 6)
    rx = _vec(l_rx, 6)
    j = np.zeros((4, 6))
    j[0, :3] = range_gradient(
        xv[:3], uhr, None if tx is None else tx[:3], None if rx is None else rx[:3]
    )
    j[1:3, :3] = spher_ang_gradient(
        xv[:3], system_type, None if rx is None else rx[:3], m
    )
    j[3, :] = range_rate_gradient(xv, uhr, tx, rx)
    return j


def calc_polar_jacob(
    x: ArrayLike,
    system_type: int = 0,
    use_half_range: bool = True,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Jacobian of a 2-D bistatic [range; angle] measurement.

    Port of ``calcPolarJacob``.

    Parameters
    ----------
    x : array_like
        (2,) target position.
    system_type : int, optional
        Polar angle convention (0 or 1). Default 0.
    use_half_range : bool, optional
        One-way range convention. Default True (the MATLAB default).
    l_tx, l_rx : array_like, optional
        Transmitter/receiver positions. Default: the origin.

    Returns
    -------
    J : ndarray
        (2, 2) Jacobian.

    Examples
    --------
    >>> J = calc_polar_jacob([3.0, 4.0])
    >>> J.shape
    (2, 2)
    """
    xv = _vec(x, 2)
    return np.vstack(
        [
            range_gradient(xv, use_half_range, _vec(l_tx, 2), _vec(l_rx, 2)),
            pol_ang_gradient(xv, system_type, _vec(l_rx, 2)),
        ]
    )


def calc_polar_rr_jacob(
    x: ArrayLike,
    system_type: int = 0,
    use_half_range: bool = True,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Jacobian of a 2-D [range; angle; range rate] w.r.t. the state.

    Port of ``calcPolarRRJacob``.

    Parameters
    ----------
    x : array_like
        (4,) target state [x, y, vx, vy].
    system_type, use_half_range, l_tx, l_rx
        As in :func:`calc_polar_jacob`, with four-element states for
        the platforms.

    Returns
    -------
    J : ndarray
        (3, 4) Jacobian.

    Examples
    --------
    >>> J = calc_polar_rr_jacob([3e3, 4e3, 10.0, -5.0])
    >>> J.shape
    (3, 4)
    """
    xv = _vec(x, 4)
    tx = _vec(l_tx, 4)
    rx = _vec(l_rx, 4)
    j = np.zeros((3, 4))
    j[0, :2] = range_gradient(
        xv[:2],
        use_half_range,
        None if tx is None else tx[:2],
        None if rx is None else rx[:2],
    )
    j[1, :2] = pol_ang_gradient(xv[:2], system_type, None if rx is None else rx[:2])
    j[2, :] = range_rate_gradient(xv, use_half_range, tx, rx)
    return j


def calc_ruv_jacob(
    x: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
    include_w: bool = False,
) -> NDArray[np.float64]:
    """
    Jacobian of a bistatic r-u-v(-w) measurement w.r.t. position.

    Port of ``calcRuvJacob``.

    Parameters
    ----------
    x : array_like
        (3,) target position.
    use_half_range : bool, optional
        One-way range convention. Default False (the MATLAB default).
    l_tx, l_rx : array_like, optional
        Transmitter/receiver positions. Default: the origin.
    m : array_like, optional
        (3, 3) global-to-local receiver rotation. Default: identity.
    include_w : bool, optional
        Include the redundant third direction cosine row. Default
        False.

    Returns
    -------
    J : ndarray
        (3, 3) Jacobian ((4, 3) with ``include_w``).

    Examples
    --------
    >>> J = calc_ruv_jacob([1e3, 2e3, 5e3])
    >>> J.shape
    (3, 3)
    """
    xv = _vec(x, 3)
    return np.vstack(
        [
            range_gradient(xv, use_half_range, _vec(l_tx, 3), _vec(l_rx, 3)),
            uv_gradient(xv, _vec(l_rx, 3), m, include_w),
        ]
    )


def calc_ruv_rr_jacob(
    x: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
    include_w: bool = False,
) -> NDArray[np.float64]:
    """
    Jacobian of [r; u; v(; w); range rate] w.r.t. the target state.

    Port of ``calcRuvRRJacob``.

    Parameters
    ----------
    x : array_like
        (6,) target state.
    use_half_range, l_tx, l_rx, m, include_w
        As in :func:`calc_ruv_jacob`, with six-element states for the
        platforms.

    Returns
    -------
    J : ndarray
        (4, 6) Jacobian ((5, 6) with ``include_w``).

    Examples
    --------
    >>> J = calc_ruv_rr_jacob([1e3, 2e3, 5e3, 10.0, -20.0, 5.0])
    >>> J.shape
    (4, 6)
    """
    xv = _vec(x, 6)
    tx = _vec(l_tx, 6)
    rx = _vec(l_rx, 6)
    rows = 5 if include_w else 4
    j = np.zeros((rows, 6))
    j[0, :3] = range_gradient(
        xv[:3],
        use_half_range,
        None if tx is None else tx[:3],
        None if rx is None else rx[:3],
    )
    j[1 : rows - 1, :3] = uv_gradient(
        xv[:3], None if rx is None else rx[:3], m, include_w
    )
    j[rows - 1, :] = range_rate_gradient(xv, use_half_range, tx, rx)
    return j


def calc_cart_rr_jacob(
    x: ArrayLike,
    components: int = 0,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Jacobian of a Cartesian-position(-plus-range-rate) measurement.

    With ``components=0`` the measurement is [position; range rate];
    any other value drops the range-rate row (position only), matching
    the MATLAB semantics.

    Port of ``calcCartRRJacob``.

    Parameters
    ----------
    x : array_like
        (2d,) target state, d in {1, 2, 3}.
    components : int, optional
        0 to include the range-rate row. Default 0.
    use_half_range, l_tx, l_rx
        As in
        :func:`~pytcl.coordinate_systems.jacobians.component_gradients.range_rate_gradient`.

    Returns
    -------
    J : ndarray
        (d+1, 2d) Jacobian ((d, 2d) without range rate).

    Examples
    --------
    >>> J = calc_cart_rr_jacob([1e3, 2e3, 5e3, 10.0, -20.0, 5.0])
    >>> J.shape
    (4, 6)
    """
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    x_dim = xv.size
    d = x_dim // 2
    if components == 0:
        j = np.zeros((d + 1, x_dim))
        j[:d, :d] = np.eye(d)
        j[d, :] = range_rate_gradient(xv, use_half_range, l_tx, l_rx)
    else:
        j = np.zeros((d, x_dim))
        j[:d, :d] = np.eye(d)
    return j


def norm_vec_jacob(x: ArrayLike) -> NDArray[np.float64]:
    """
    Jacobian of the unit vector x/norm(x) with respect to x.

    Port of ``normVecJacob``.

    Parameters
    ----------
    x : array_like
        (d,) vector.

    Returns
    -------
    J : ndarray
        (d, d) Jacobian.

    Examples
    --------
    >>> norm_vec_jacob([2.0, 0.0])
    array([[0. , 0. ],
           [0. , 0.5]])
    """
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(xv)
    return -np.outer(xv, xv) / n**3 + np.eye(xv.size) / n


# ---------------------------------------------------------------------
# Measurement-to-Cartesian inversions (faithful ports of the bistatic
# spher2Cart/pol2Cart used by the converted Jacobians).
# ---------------------------------------------------------------------


def _spher2cart_bistatic(
    z: NDArray[np.float64],
    system_type: int,
    use_half_range: bool,
    l_tx: NDArray[np.float64] | None,
    l_rx: NDArray[np.float64] | None,
    m: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    r, az, el = z[:3]
    if system_type == 2:
        el = np.pi / 2 - el
        system_type = 0
    elif system_type == 3:
        az = np.pi / 2 - az
        system_type = 0
    if system_type == 0:
        u = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    elif system_type == 1:
        u = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    else:
        raise ValueError("Invalid system type specified.")
    rot = np.eye(3) if m is None else m
    if l_tx is None:
        # Monostatic receiver-at-origin form.
        scale = r if use_half_range else r / 2.0
        return rot.T @ (scale * u)
    rx = np.zeros(3) if l_rx is None else l_rx
    if use_half_range:
        r = 2.0 * r
    tx_local = rot @ (l_tx - rx)
    r1 = (r**2 - np.dot(tx_local, tx_local)) / (2.0 * (r - np.dot(u, tx_local)))
    return rot.T @ (r1 * u) + rx


def _pol2cart_bistatic(
    z: NDArray[np.float64],
    system_type: int,
    use_half_range: bool,
    l_tx: NDArray[np.float64] | None,
    l_rx: NDArray[np.float64] | None,
    m: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    rb, az = z[:2]
    if system_type == 0:
        u = np.array([np.cos(az), np.sin(az)])
    elif system_type == 1:
        u = np.array([np.sin(az), np.cos(az)])
    else:
        raise ValueError("Invalid system type specified.")
    rot = np.eye(2) if m is None else m
    tx = np.zeros(2) if l_tx is None else l_tx
    rx = np.zeros(2) if l_rx is None else l_rx
    if use_half_range:
        rb = 2.0 * rb
    if np.all(tx == rx):
        return np.linalg.solve(rot, (rb / 2.0) * u) + rx
    tx_local = rot @ (tx - rx)
    r1 = (rb**2 - np.dot(tx_local, tx_local)) / (2.0 * (rb - np.dot(u, tx_local)))
    return np.linalg.solve(rot, r1 * u) + rx


def calc_spher_conv_jacob(
    z: ArrayLike,
    system_type: int = 0,
    use_half_range: bool | None = None,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Spherical measurement Jacobian evaluated at the measurement itself.

    The measurement is inverted to Cartesian and the Jacobian of
    :func:`calc_spher_jacob` is evaluated there -- the form needed to
    transform a measured covariance.

    Port of ``calcSpherConvJacob``.

    Parameters
    ----------
    z : array_like
        (3,) measurement [r, az, el].
    system_type, use_half_range, l_tx, l_rx, m
        As in :func:`calc_spher_jacob`.

    Returns
    -------
    J : ndarray
        (3, 3) Jacobian.

    Examples
    --------
    >>> J = calc_spher_conv_jacob([1e3, 0.5, 0.2])
    >>> J.shape
    (3, 3)
    """
    zv = _vec(z, 3)
    uhr = _spher_half_range_default(use_half_range, l_tx)
    rot = None if m is None else np.asarray(m, dtype=np.float64)
    x = _spher2cart_bistatic(zv, system_type, uhr, _vec(l_tx, 3), _vec(l_rx, 3), rot)
    return calc_spher_jacob(x, system_type, uhr, l_tx, l_rx, m)


def calc_polar_conv_jacob(
    z: ArrayLike,
    system_type: int = 0,
    use_half_range: bool = True,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Polar measurement Jacobian evaluated at the measurement itself.

    Port of ``calcPolarConvJacob``.

    Parameters
    ----------
    z : array_like
        (2,) measurement [r, angle].
    system_type, use_half_range, l_tx, l_rx
        As in :func:`calc_polar_jacob`.
    m : array_like, optional
        (2, 2) global-to-local receiver rotation used in the
        measurement inversion. Default: identity.

    Returns
    -------
    J : ndarray
        (2, 2) Jacobian.

    Examples
    --------
    >>> J = calc_polar_conv_jacob([2e3, 0.7])
    >>> J.shape
    (2, 2)
    """
    zv = _vec(z, 2)
    rot = None if m is None else np.asarray(m, dtype=np.float64)
    x = _pol2cart_bistatic(
        zv, system_type, use_half_range, _vec(l_tx, 2), _vec(l_rx, 2), rot
    )
    return calc_polar_jacob(x, system_type, use_half_range, l_tx, l_rx)


def calc_polar_rr_conv_jacob(
    z: ArrayLike,
    system_type: int = 0,
    use_half_range: bool = True,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Converted-polar Jacobian with a pass-through range-rate row.

    Port of ``calcPolarRRConvJacob``.

    Parameters
    ----------
    z : array_like
        (3,) measurement [r, angle, range rate].
    system_type, use_half_range, l_tx, l_rx, m
        As in :func:`calc_polar_conv_jacob`.

    Returns
    -------
    J : ndarray
        (3, 3) block-diagonal Jacobian.

    Examples
    --------
    >>> J = calc_polar_rr_conv_jacob([2e3, 0.7, 15.0])
    >>> float(J[2, 2])
    1.0
    """
    zv = _vec(z, 3)
    j = np.zeros((3, 3))
    j[:2, :2] = calc_polar_conv_jacob(
        zv[:2], system_type, use_half_range, l_tx, l_rx, m
    )
    j[2, 2] = 1.0
    return j


def calc_ruv_conv_jacob(
    z: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    r-u-v measurement Jacobian evaluated at the measurement itself.

    Port of ``calcRuvConvJacob``.

    Parameters
    ----------
    z : array_like
        (3,) measurement [r, u, v].
    use_half_range, l_tx, l_rx, m
        As in :func:`calc_ruv_jacob`.

    Returns
    -------
    J : ndarray
        (3, 3) Jacobian.

    Examples
    --------
    >>> J = calc_ruv_conv_jacob([2e3, 0.1, 0.2])
    >>> J.shape
    (3, 3)
    """
    zv = _vec(z, 3)
    x = ruv2cart_bistatic(zv, use_half_range, l_tx, l_rx, m).reshape(-1)
    return calc_ruv_jacob(x, use_half_range, l_tx, l_rx, m)


def calc_ruv_rr_conv_jacob(
    z: ArrayLike,
    use_half_range: bool = False,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Converted r-u-v Jacobian with a pass-through range-rate row.

    Port of ``calcRuvRRConvJacob``.

    Parameters
    ----------
    z : array_like
        (4,) measurement [r, u, v, range rate].
    use_half_range, l_tx, l_rx, m
        As in :func:`calc_ruv_conv_jacob`.

    Returns
    -------
    J : ndarray
        (4, 4) block-diagonal Jacobian.

    Examples
    --------
    >>> J = calc_ruv_rr_conv_jacob([2e3, 0.1, 0.2, 25.0])
    >>> float(J[3, 3])
    1.0
    """
    zv = _vec(z, 4)
    j = np.zeros((4, 4))
    j[:3, :3] = calc_ruv_conv_jacob(zv[:3], use_half_range, l_tx, l_rx, m)
    j[3, 3] = 1.0
    return j
