"""
Composed measurement Hessians and Hessian-transformation helpers.

Ports of the MATLAB TCL's top-level ``Coordinate_Systems/Hessians``
functions: the spherical measurement Hessian (direct, inverse and
converted forms) and the affine/chain-rule helpers used to push
Hessians through transformed functions.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.coordinate_systems.hessians.component_hessians import (
    range_hessian,
    spher_ang_hessian,
)
from pytcl.coordinate_systems.jacobians.measurement_jacobians import (
    _spher2cart_bistatic,
    _spher_half_range_default,
    _vec,
)

__all__ = [
    "calc_spher_hessian",
    "calc_spher_inv_hessian",
    "calc_spher_conv_hessian",
    "hessian_of_affine_trans_fun",
    "hessian_chain_rule",
]


def calc_spher_hessian(
    x: ArrayLike,
    system_type: int = 0,
    use_half_range: bool | None = None,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Hessians of a bistatic [range; azimuth; elevation] measurement.

    Port of ``calcSpherHessian``.

    Parameters
    ----------
    x : array_like
        (3,) target position.
    system_type, use_half_range, l_tx, l_rx, m
        As in
        :func:`~pytcl.coordinate_systems.jacobians.measurement_jacobians.calc_spher_jacob`
        (the same MATLAB half-range default applies).

    Returns
    -------
    H : ndarray
        (3, 3, 3) stack: range, azimuth and elevation Hessians.

    Examples
    --------
    >>> H = calc_spher_hessian([1e3, 2e3, 500.0])
    >>> H.shape
    (3, 3, 3)
    """
    xv = _vec(x, 3)
    uhr = _spher_half_range_default(use_half_range, l_tx)
    out = np.zeros((3, 3, 3))
    out[:, :, 0] = range_hessian(xv, uhr, _vec(l_tx, 3), _vec(l_rx, 3))
    out[:, :, 1:] = spher_ang_hessian(xv, system_type, _vec(l_rx, 3), m)
    return out


def calc_spher_inv_hessian(z: ArrayLike, system_type: int = 0) -> NDArray[np.float64]:
    """
    Hessians of Cartesian position w.r.t. a spherical triple.

    Port of ``calcSpherInvHessian``.

    Parameters
    ----------
    z : array_like
        (3,) measurement [r, az, el].
    system_type : int, optional
        Spherical angle convention 0-3. Default 0.

    Returns
    -------
    H : ndarray
        (3, 3, 3) stack: the Hessians of x, y and z with respect to
        (r, az, el).

    Examples
    --------
    >>> H = calc_spher_inv_hessian([1e3, 0.5, 0.2])
    >>> H.shape
    (3, 3, 3)
    """
    r, az, el = np.asarray(z, dtype=np.float64).reshape(-1)[:3]
    sa, ca = np.sin(az), np.cos(az)
    se, ce = np.sin(el), np.cos(el)

    def _sym(daa: float, dee: float, dra: float, dre: float, dae: float):
        h = np.zeros((3, 3))
        h[1, 1] = daa
        h[2, 2] = dee
        h[0, 1] = h[1, 0] = dra
        h[0, 2] = h[2, 0] = dre
        h[1, 2] = h[2, 1] = dae
        return h

    if system_type == 0:
        hx = _sym(-r * ca * ce, -r * ca * ce, -ce * sa, -ca * se, r * sa * se)
        hy = _sym(-r * ce * sa, -r * ce * sa, ca * ce, -sa * se, -r * ca * se)
        hz = _sym(0.0, -r * se, 0.0, ce, 0.0)
    elif system_type == 1:
        hx = _sym(-r * ce * sa, -r * ce * sa, ca * ce, -sa * se, -r * ca * se)
        hy = _sym(0.0, -r * se, 0.0, ce, 0.0)
        hz = _sym(-r * ca * ce, -r * ca * ce, -ce * sa, -ca * se, r * sa * se)
    elif system_type == 2:
        hx = _sym(-r * ca * se, -r * ca * se, -sa * se, ca * ce, -r * ce * sa)
        hy = _sym(-r * sa * se, -r * sa * se, ca * se, ce * sa, r * ca * ce)
        hz = _sym(0.0, -r * ce, 0.0, -se, 0.0)
    elif system_type == 3:
        hx = _sym(-r * sa * ce, -r * sa * ce, ce * ca, -sa * se, -r * ca * se)
        hy = _sym(-r * ce * ca, -r * ce * ca, -sa * ce, -ca * se, r * sa * se)
        hz = _sym(0.0, -r * se, 0.0, ce, 0.0)
    else:
        raise ValueError("Invalid system type specified.")

    out = np.zeros((3, 3, 3))
    out[:, :, 0] = hx
    out[:, :, 1] = hy
    out[:, :, 2] = hz
    return out


def calc_spher_conv_hessian(
    z: ArrayLike,
    system_type: int = 0,
    use_half_range: bool | None = None,
    l_tx: ArrayLike | None = None,
    l_rx: ArrayLike | None = None,
    m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Spherical measurement Hessian evaluated at the measurement itself.

    Port of ``calcSpherConvHessian``.

    Parameters
    ----------
    z : array_like
        (3,) measurement [r, az, el].
    system_type, use_half_range, l_tx, l_rx, m
        As in :func:`calc_spher_hessian`.

    Returns
    -------
    H : ndarray
        (3, 3, 3) stack.

    Examples
    --------
    >>> H = calc_spher_conv_hessian([1e3, 0.5, 0.2])
    >>> H.shape
    (3, 3, 3)
    """
    zv = _vec(z, 3)
    uhr = _spher_half_range_default(use_half_range, l_tx)
    rot = None if m is None else np.asarray(m, dtype=np.float64)
    x = _spher2cart_bistatic(zv, system_type, uhr, _vec(l_tx, 3), _vec(l_rx, 3), rot)
    return calc_spher_hessian(x, system_type, uhr, l_tx, l_rx, m)


def hessian_of_affine_trans_fun(h: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """
    Hessian stack of ``M @ f(x)`` given the Hessian stack of ``f(x)``.

    Port of ``HessianOfAffineTransFun``.

    Parameters
    ----------
    h : array_like
        (n, n, num_out) Hessian stack of f.
    m : array_like
        (num_out_new, num_out) linear map applied to f's outputs.

    Returns
    -------
    H : ndarray
        (n, n, num_out_new) Hessian stack of M @ f.

    Examples
    --------
    >>> import numpy as np
    >>> H = np.zeros((2, 2, 2)); H[:, :, 0] = np.eye(2)
    >>> hessian_of_affine_trans_fun(H, [[0.0, 1.0], [1.0, 0.0]]).shape
    (2, 2, 2)
    """
    stack = np.asarray(h, dtype=np.float64)
    mm = np.asarray(m, dtype=np.float64)
    # For each fixed (row, col) of the Hessian, mix the output layers.
    return np.einsum("ok,ijk->ijo", mm, stack)


def hessian_chain_rule(
    h_f: ArrayLike,
    h_g: ArrayLike,
    j_f: ArrayLike,
    j_g: ArrayLike,
) -> NDArray[np.float64]:
    """
    Hessian stack of the composition f(g(x)).

    Given f's Hessians/Jacobian in its own argument and g's
    Hessians/Jacobian in x, returns the Hessians of f(g(x)) in x.

    Port of ``HessianChainRule``.

    Parameters
    ----------
    h_f : array_like
        (num_in, num_in, num_out) Hessian stack of f at g(x).
    h_g : array_like
        (num_x, num_x, num_in) Hessian stack of g at x.
    j_f : array_like
        (num_out, num_in) Jacobian of f at g(x).
    j_g : array_like
        (num_in, num_x) Jacobian of g at x.

    Returns
    -------
    H : ndarray
        (num_x, num_x, num_out) Hessian stack of the composition.

    Examples
    --------
    >>> import numpy as np
    >>> hf = np.zeros((2, 2, 2)); hg = np.zeros((2, 2, 2))
    >>> hessian_chain_rule(hf, hg, np.eye(2), 2 * np.eye(2)).shape
    (2, 2, 2)
    """
    hf = np.asarray(h_f, dtype=np.float64)
    hg = np.asarray(h_g, dtype=np.float64)
    jf = np.asarray(j_f, dtype=np.float64)
    jg = np.asarray(j_g, dtype=np.float64)
    num_out = hf.shape[2]
    num_in = hf.shape[0]
    n_x = jg.shape[1]
    out = np.zeros((n_x, n_x, num_out))
    for k in range(num_out):
        acc = jg.T @ hf[:, :, k] @ jg
        for k2 in range(num_in):
            acc = acc + jf[k, k2] * hg[:, :, k2]
        out[:, :, k] = acc
    return out
