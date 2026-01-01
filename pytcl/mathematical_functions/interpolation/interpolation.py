"""
Interpolation methods.

This module provides interpolation functions for 1D, 2D, and 3D data,
commonly used in tracking for measurement interpolation and terrain models.
"""

from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate


def interp1d(
    x: ArrayLike,
    y: ArrayLike,
    kind: Literal[
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ] = "linear",
    fill_value: Union[float, Tuple[float, float], str] = np.nan,
    bounds_error: bool = False,
) -> Callable[[ArrayLike], NDArray[np.floating]]:
    """
    Create a 1D interpolation function.

    Parameters
    ----------
    x : array_like
        Sample points (must be monotonically increasing).
    y : array_like
        Sample values.
    kind : str, optional
        Interpolation method:
        - 'linear': Linear interpolation (default)
        - 'nearest': Nearest neighbor
        - 'zero', 'slinear', 'quadratic', 'cubic': Spline of order 0, 1, 2, 3
        - 'previous', 'next': Previous/next value
    fill_value : float, tuple, or 'extrapolate', optional
        Value for points outside data range. Default is NaN.
        Use 'extrapolate' to extrapolate beyond bounds.
    bounds_error : bool, optional
        If True, raise error for out-of-bounds. Default is False.

    Returns
    -------
    f : callable
        Interpolation function that takes x values and returns y values.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([0, 1, 4, 9])
    >>> f = interp1d(x, y, kind='quadratic')
    >>> f(1.5)
    array(2.25)

    See Also
    --------
    scipy.interpolate.interp1d : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    interp_func = interpolate.interp1d(
        x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error
    )

    def wrapper(x_new: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(interp_func(x_new), dtype=np.float64)

    return wrapper


def linear_interp(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: Optional[float] = None,
    right: Optional[float] = None,
) -> NDArray[np.floating]:
    """
    One-dimensional linear interpolation.

    Parameters
    ----------
    x : array_like
        X-coordinates at which to evaluate.
    xp : array_like
        X-coordinates of data points (must be increasing).
    fp : array_like
        Y-coordinates of data points.
    left : float, optional
        Value for x < xp[0]. Default is fp[0].
    right : float, optional
        Value for x > xp[-1]. Default is fp[-1].

    Returns
    -------
    y : ndarray
        Interpolated values.

    Examples
    --------
    >>> linear_interp(2.5, [1, 2, 3], [1, 4, 9])
    6.5

    See Also
    --------
    numpy.interp : Underlying implementation.
    """
    return np.asarray(np.interp(x, xp, fp, left=left, right=right), dtype=np.float64)


def cubic_spline(
    x: ArrayLike,
    y: ArrayLike,
    bc_type: Literal["not-a-knot", "clamped", "natural", "periodic"] = "not-a-knot",
) -> interpolate.CubicSpline:
    """
    Create a cubic spline interpolation.

    Parameters
    ----------
    x : array_like
        Sample points (must be strictly increasing).
    y : array_like
        Sample values.
    bc_type : str, optional
        Boundary condition type:
        - 'not-a-knot': Default, uses continuity conditions.
        - 'clamped': First derivatives at endpoints are zero.
        - 'natural': Second derivatives at endpoints are zero.
        - 'periodic': Periodic boundary conditions.

    Returns
    -------
    cs : CubicSpline
        Cubic spline object. Call cs(x_new) to interpolate.

    Examples
    --------
    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.sin(x)
    >>> cs = cubic_spline(x, y)
    >>> cs(np.pi/2)
    array(0.99999...)

    See Also
    --------
    scipy.interpolate.CubicSpline : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return interpolate.CubicSpline(x, y, bc_type=bc_type)


def pchip(
    x: ArrayLike,
    y: ArrayLike,
) -> interpolate.PchipInterpolator:
    """
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

    PCHIP preserves monotonicity and avoids overshooting, making it
    suitable for data that should not have spurious oscillations.

    Parameters
    ----------
    x : array_like
        Sample points (must be strictly increasing).
    y : array_like
        Sample values.

    Returns
    -------
    p : PchipInterpolator
        PCHIP interpolator object.

    Notes
    -----
    Unlike cubic splines, PCHIP will not overshoot if the data is
    monotonic, making it more suitable for physical quantities that
    must stay positive or bounded.

    See Also
    --------
    scipy.interpolate.PchipInterpolator : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return interpolate.PchipInterpolator(x, y)


def akima(
    x: ArrayLike,
    y: ArrayLike,
) -> interpolate.Akima1DInterpolator:
    """
    Akima interpolation.

    Akima interpolation is a smooth interpolation method that avoids
    excessive oscillation compared to cubic splines.

    Parameters
    ----------
    x : array_like
        Sample points (must be strictly increasing).
    y : array_like
        Sample values.

    Returns
    -------
    a : Akima1DInterpolator
        Akima interpolator object.

    See Also
    --------
    scipy.interpolate.Akima1DInterpolator : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return interpolate.Akima1DInterpolator(x, y)


def interp2d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    kind: Literal["linear", "cubic", "quintic"] = "linear",
) -> interpolate.RegularGridInterpolator:
    """
    Create a 2D interpolation function on a regular grid.

    Parameters
    ----------
    x : array_like
        Grid coordinates along first axis.
    y : array_like
        Grid coordinates along second axis.
    z : array_like
        Values on the grid of shape (len(x), len(y)).
    kind : str, optional
        Interpolation method: 'linear', 'cubic', or 'quintic'.
        Default is 'linear'.

    Returns
    -------
    f : RegularGridInterpolator
        Interpolation function. Call f((xi, yi)) to interpolate.

    Examples
    --------
    >>> x = np.linspace(0, 4, 5)
    >>> y = np.linspace(0, 4, 5)
    >>> z = np.outer(x, y)  # z = x * y
    >>> f = interp2d(x, y, z)
    >>> f([[2.5, 2.5]])
    array([6.25])

    See Also
    --------
    scipy.interpolate.RegularGridInterpolator : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return interpolate.RegularGridInterpolator(
        (x, y), z, method=kind, bounds_error=False, fill_value=np.nan
    )


def interp3d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    values: ArrayLike,
    kind: Literal["linear", "nearest"] = "linear",
) -> interpolate.RegularGridInterpolator:
    """
    Create a 3D interpolation function on a regular grid.

    Parameters
    ----------
    x : array_like
        Grid coordinates along first axis.
    y : array_like
        Grid coordinates along second axis.
    z : array_like
        Grid coordinates along third axis.
    values : array_like
        Values on the grid of shape (len(x), len(y), len(z)).
    kind : str, optional
        Interpolation method: 'linear' or 'nearest'. Default is 'linear'.

    Returns
    -------
    f : RegularGridInterpolator
        Interpolation function.

    See Also
    --------
    scipy.interpolate.RegularGridInterpolator : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    return interpolate.RegularGridInterpolator(
        (x, y, z), values, method=kind, bounds_error=False, fill_value=np.nan
    )


def rbf_interpolate(
    points: ArrayLike,
    values: ArrayLike,
    kernel: Literal[
        "linear",
        "thin_plate_spline",
        "cubic",
        "quintic",
        "multiquadric",
        "inverse_multiquadric",
        "gaussian",
    ] = "thin_plate_spline",
    smoothing: float = 0.0,
) -> interpolate.RBFInterpolator:
    """
    Radial Basis Function (RBF) interpolation.

    RBF interpolation works with scattered (non-grid) data in any dimension.

    Parameters
    ----------
    points : array_like
        Data point coordinates of shape (n_samples, n_dims).
    values : array_like
        Values at data points of shape (n_samples,) or (n_samples, n_values).
    kernel : str, optional
        RBF kernel function. Default is 'thin_plate_spline'.
    smoothing : float, optional
        Smoothing parameter. 0 means exact interpolation. Default is 0.

    Returns
    -------
    rbf : RBFInterpolator
        RBF interpolation object.

    Examples
    --------
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> values = np.array([0, 1, 1, 2])
    >>> rbf = rbf_interpolate(points, values)
    >>> rbf([[0.5, 0.5]])
    array([1.])

    See Also
    --------
    scipy.interpolate.RBFInterpolator : Underlying implementation.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    return interpolate.RBFInterpolator(
        points, values, kernel=kernel, smoothing=smoothing
    )


def barycentric(
    x: ArrayLike,
    y: ArrayLike,
) -> interpolate.BarycentricInterpolator:
    """
    Barycentric polynomial interpolation.

    This is a numerically stable method for polynomial interpolation.

    Parameters
    ----------
    x : array_like
        Sample points.
    y : array_like
        Sample values.

    Returns
    -------
    p : BarycentricInterpolator
        Barycentric interpolator object.

    See Also
    --------
    scipy.interpolate.BarycentricInterpolator : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return interpolate.BarycentricInterpolator(x, y)


def krogh(
    x: ArrayLike,
    y: ArrayLike,
) -> interpolate.KroghInterpolator:
    """
    Krogh interpolation.

    Polynomial interpolation using divided differences.

    Parameters
    ----------
    x : array_like
        Sample points.
    y : array_like
        Sample values (can include derivatives at points).

    Returns
    -------
    k : KroghInterpolator
        Krogh interpolator object.

    See Also
    --------
    scipy.interpolate.KroghInterpolator : Underlying implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return interpolate.KroghInterpolator(x, y)


def spherical_interp(
    lat: ArrayLike,
    lon: ArrayLike,
    values: ArrayLike,
) -> interpolate.RBFInterpolator:
    """
    Interpolation on a spherical surface.

    Converts lat/lon to 3D Cartesian coordinates and uses RBF interpolation.

    Parameters
    ----------
    lat : array_like
        Latitude in radians of shape (n_samples,).
    lon : array_like
        Longitude in radians of shape (n_samples,).
    values : array_like
        Values at sample points.

    Returns
    -------
    interp : RBFInterpolator
        Interpolation function. Call with 3D Cartesian coordinates.

    Notes
    -----
    To interpolate at new lat/lon points:
    1. Convert lat/lon to Cartesian: x=cos(lat)*cos(lon), y=cos(lat)*sin(lon),
       z=sin(lat)
    2. Call interp([[x, y, z]])
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Convert to Cartesian coordinates on unit sphere
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    points = np.column_stack([x, y, z])

    return interpolate.RBFInterpolator(points, values, kernel="thin_plate_spline")


__all__ = [
    "interp1d",
    "linear_interp",
    "cubic_spline",
    "pchip",
    "akima",
    "interp2d",
    "interp3d",
    "rbf_interpolate",
    "barycentric",
    "krogh",
    "spherical_interp",
]
