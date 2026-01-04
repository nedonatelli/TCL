"""
Covariance ellipse plotting utilities.

This module provides functions for visualizing uncertainty as ellipses
in 2D and 3D spaces, commonly used in tracking and estimation applications.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import is_available, requires

# Lazy flag for backward compatibility
HAS_PLOTLY = is_available("plotly")


def covariance_ellipse_points(
    mean: ArrayLike,
    cov: ArrayLike,
    n_std: float = 2.0,
    n_points: int = 100,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate points for a 2D covariance ellipse.

    Parameters
    ----------
    mean : array_like
        Center of the ellipse [x, y].
    cov : array_like
        2x2 covariance matrix.
    n_std : float, optional
        Number of standard deviations for ellipse size. Default is 2.0.
    n_points : int, optional
        Number of points to generate. Default is 100.

    Returns
    -------
    x : ndarray
        X coordinates of ellipse points.
    y : ndarray
        Y coordinates of ellipse points.

    Examples
    --------
    >>> mean = [0, 0]
    >>> cov = [[1, 0.5], [0.5, 2]]
    >>> x, y = covariance_ellipse_points(mean, cov, n_std=2.0)
    >>> len(x) == 100
    True
    """
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    if cov.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2")

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Handle negative eigenvalues (numerical issues)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Compute rotation angle
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, n_points)
    a = n_std * np.sqrt(eigenvalues[0])
    b = n_std * np.sqrt(eigenvalues[1])

    # Ellipse in standard position
    x_std = a * np.cos(t)
    y_std = b * np.sin(t)

    # Rotate and translate
    x = mean[0] + x_std * np.cos(angle) - y_std * np.sin(angle)
    y = mean[1] + x_std * np.sin(angle) + y_std * np.cos(angle)

    return x, y


def covariance_ellipsoid_points(
    mean: ArrayLike,
    cov: ArrayLike,
    n_std: float = 2.0,
    n_points: int = 20,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate points for a 3D covariance ellipsoid surface.

    Parameters
    ----------
    mean : array_like
        Center of the ellipsoid [x, y, z].
    cov : array_like
        3x3 covariance matrix.
    n_std : float, optional
        Number of standard deviations for ellipsoid size. Default is 2.0.
    n_points : int, optional
        Number of points along each angular dimension. Default is 20.

    Returns
    -------
    x : ndarray
        X coordinates of surface points (n_points x n_points).
    y : ndarray
        Y coordinates of surface points (n_points x n_points).
    z : ndarray
        Z coordinates of surface points (n_points x n_points).

    Examples
    --------
    >>> mean = [0, 0, 0]
    >>> cov = np.diag([1, 2, 3])
    >>> x, y, z = covariance_ellipsoid_points(mean, cov)
    >>> x.shape == (20, 20)
    True
    """
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    if cov.shape != (3, 3):
        raise ValueError("Covariance matrix must be 3x3")

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Handle negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)

    # Generate unit sphere points
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    u, v = np.meshgrid(u, v)

    # Unit sphere
    x_sphere = np.sin(v) * np.cos(u)
    y_sphere = np.sin(v) * np.sin(u)
    z_sphere = np.cos(v)

    # Stack for transformation
    sphere_points = np.stack([x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()])

    # Scale by eigenvalues and rotate
    scaling = n_std * np.sqrt(eigenvalues)
    transformed = eigenvectors @ np.diag(scaling) @ sphere_points

    # Reshape and translate
    x = transformed[0].reshape(n_points, n_points) + mean[0]
    y = transformed[1].reshape(n_points, n_points) + mean[1]
    z = transformed[2].reshape(n_points, n_points) + mean[2]

    return x, y, z


def ellipse_parameters(cov: ArrayLike) -> Tuple[float, float, float]:
    """
    Extract ellipse parameters from a 2x2 covariance matrix.

    Parameters
    ----------
    cov : array_like
        2x2 covariance matrix.

    Returns
    -------
    semi_major : float
        Semi-major axis length (1-sigma).
    semi_minor : float
        Semi-minor axis length (1-sigma).
    angle : float
        Rotation angle in radians (counter-clockwise from x-axis).

    Examples
    --------
    >>> cov = [[4, 0], [0, 1]]
    >>> a, b, theta = ellipse_parameters(cov)
    >>> np.isclose(a, 2.0)
    True
    >>> np.isclose(b, 1.0)
    True
    """
    cov = np.asarray(cov, dtype=np.float64)

    if cov.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2")

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    semi_major = np.sqrt(max(eigenvalues[0], 0))
    semi_minor = np.sqrt(max(eigenvalues[1], 0))
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    return semi_major, semi_minor, angle


def confidence_region_radius(n_dims: int, confidence: float = 0.95) -> float:
    """
    Compute the chi-squared radius for a confidence region.

    For a multivariate Gaussian, the squared Mahalanobis distance follows
    a chi-squared distribution with n_dims degrees of freedom.

    Parameters
    ----------
    n_dims : int
        Number of dimensions.
    confidence : float, optional
        Confidence level (0 to 1). Default is 0.95.

    Returns
    -------
    radius : float
        The chi-squared radius (number of standard deviations).

    Examples
    --------
    >>> r = confidence_region_radius(2, 0.95)
    >>> np.isclose(r, np.sqrt(5.991), rtol=0.01)
    True
    """
    from scipy import stats

    chi2_val = stats.chi2.ppf(confidence, df=n_dims)
    return np.sqrt(chi2_val)


@requires("plotly", extra="visualization")
def plot_covariance_ellipse(
    mean: ArrayLike,
    cov: ArrayLike,
    n_std: float = 2.0,
    fill: bool = True,
    color: str = "blue",
    opacity: float = 0.3,
    name: Optional[str] = None,
    showlegend: bool = True,
) -> Any:
    """
    Create a Plotly trace for a covariance ellipse.

    Parameters
    ----------
    mean : array_like
        Center of the ellipse [x, y].
    cov : array_like
        2x2 covariance matrix.
    n_std : float, optional
        Number of standard deviations. Default is 2.0.
    fill : bool, optional
        Whether to fill the ellipse. Default is True.
    color : str, optional
        Color for the ellipse. Default is "blue".
    opacity : float, optional
        Opacity for the fill (0 to 1). Default is 0.3.
    name : str, optional
        Name for the trace. Default is None.
    showlegend : bool, optional
        Whether to show in legend. Default is True.

    Returns
    -------
    trace : go.Scatter
        Plotly scatter trace for the ellipse.

    Raises
    ------
    DependencyError
        If plotly is not installed.
    """
    import plotly.graph_objects as go

    x, y = covariance_ellipse_points(mean, cov, n_std=n_std)

    if fill:
        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=1),
            fill="toself",
            fillcolor=f"rgba({_color_to_rgb(color)}, {opacity})",
            name=name or f"{n_std}σ covariance",
            showlegend=showlegend,
        )
    else:
        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2),
            name=name or f"{n_std}σ covariance",
            showlegend=showlegend,
        )


@requires("plotly", extra="visualization")
def plot_covariance_ellipses(
    means: List[ArrayLike],
    covariances: List[ArrayLike],
    n_std: float = 2.0,
    colors: Optional[List[str]] = None,
    opacity: float = 0.3,
    show_centers: bool = True,
) -> Any:
    """
    Create a figure with multiple covariance ellipses.

    Parameters
    ----------
    means : list of array_like
        Centers of the ellipses.
    covariances : list of array_like
        2x2 covariance matrices.
    n_std : float, optional
        Number of standard deviations. Default is 2.0.
    colors : list of str, optional
        Colors for each ellipse. Default cycles through a palette.
    opacity : float, optional
        Opacity for the fills. Default is 0.3.
    show_centers : bool, optional
        Whether to show center points. Default is True.

    Returns
    -------
    fig : go.Figure
        Plotly figure with the ellipses.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    default_colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
    ]
    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(means))]

    for i, (mean, cov) in enumerate(zip(means, covariances)):
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        # Add ellipse
        trace = plot_covariance_ellipse(
            mean,
            cov,
            n_std=n_std,
            color=colors[i],
            opacity=opacity,
            name=f"Ellipse {i + 1}",
            showlegend=True,
        )
        fig.add_trace(trace)

        # Add center point
        if show_centers:
            fig.add_trace(
                go.Scatter(
                    x=[mean[0]],
                    y=[mean[1]],
                    mode="markers",
                    marker=dict(color=colors[i], size=8),
                    showlegend=False,
                )
            )

    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        title=f"{n_std}σ Covariance Ellipses",
    )

    return fig


@requires("plotly", extra="visualization")
def plot_covariance_ellipsoid(
    mean: ArrayLike,
    cov: ArrayLike,
    n_std: float = 2.0,
    color: str = "blue",
    opacity: float = 0.5,
    name: Optional[str] = None,
) -> Any:
    """
    Create a Plotly surface trace for a 3D covariance ellipsoid.

    Parameters
    ----------
    mean : array_like
        Center of the ellipsoid [x, y, z].
    cov : array_like
        3x3 covariance matrix.
    n_std : float, optional
        Number of standard deviations. Default is 2.0.
    color : str, optional
        Color for the surface. Default is "blue".
    opacity : float, optional
        Opacity (0 to 1). Default is 0.5.
    name : str, optional
        Name for the trace.

    Returns
    -------
    trace : go.Surface
        Plotly surface trace for the ellipsoid.
    """
    import plotly.graph_objects as go

    x, y, z = covariance_ellipsoid_points(mean, cov, n_std=n_std)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name=name or f"{n_std}σ ellipsoid",
    )


def _color_to_rgb(color: str) -> str:
    """Convert a color name to RGB string for rgba()."""
    color_map = {
        "blue": "0, 100, 255",
        "red": "255, 50, 50",
        "green": "50, 200, 50",
        "orange": "255, 165, 0",
        "purple": "128, 0, 128",
        "cyan": "0, 255, 255",
        "magenta": "255, 0, 255",
        "yellow": "255, 255, 0",
        "black": "0, 0, 0",
        "gray": "128, 128, 128",
        "grey": "128, 128, 128",
    }
    return color_map.get(color.lower(), "0, 100, 255")


__all__ = [
    "covariance_ellipse_points",
    "covariance_ellipsoid_points",
    "ellipse_parameters",
    "confidence_region_radius",
    "plot_covariance_ellipse",
    "plot_covariance_ellipses",
    "plot_covariance_ellipsoid",
]
