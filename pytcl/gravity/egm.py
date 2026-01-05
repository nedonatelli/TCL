"""
Earth Gravitational Models (EGM96 and EGM2008).

This module provides support for high-degree Earth gravitational models
including EGM96 (degree 360) and EGM2008 (degree 2190). These models
represent the Earth's gravitational field using spherical harmonic
coefficients.

The models support:
- Geoid height computation
- Gravity disturbance/anomaly calculation
- Deflection of the vertical

References
----------
.. [1] Lemoine, F.G., et al. "The Development of the Joint NASA GSFC and
       NIMA Geopotential Model EGM96." NASA Technical Paper, 1998.
.. [2] Pavlis, N.K., et al. "The development and evaluation of the Earth
       Gravitational Model 2008 (EGM2008)." JGR 117.B4 (2012).
.. [3] National Geospatial-Intelligence Agency. "EGM2008 Model Coefficients."
       https://earth-info.nga.mil/
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .clenshaw import clenshaw_gravity, clenshaw_potential
from .models import WGS84, normal_gravity_somigliana

# Module logger
_logger = logging.getLogger("pytcl.gravity.egm")


class EGMCoefficients(NamedTuple):
    """Earth Gravitational Model coefficients.

    Parameters
    ----------
    C : ndarray
        Cosine coefficients (fully normalized), shape (n_max+1, n_max+1).
    S : ndarray
        Sine coefficients (fully normalized), shape (n_max+1, n_max+1).
    GM : float
        Gravitational parameter in m^3/s^2.
    R : float
        Reference radius in meters.
    n_max : int
        Maximum degree of the model.
    model_name : str
        Model name ("EGM96" or "EGM2008").
    """

    C: NDArray[np.floating]
    S: NDArray[np.floating]
    GM: float
    R: float
    n_max: int
    model_name: str


class GeoidResult(NamedTuple):
    """Geoid height computation result.

    Parameters
    ----------
    height : float
        Geoid height above reference ellipsoid in meters.
    lat : float
        Latitude in radians.
    lon : float
        Longitude in radians.
    model : str
        Model name used.
    """

    height: float
    lat: float
    lon: float
    model: str


class GravityDisturbance(NamedTuple):
    """Gravity disturbance result.

    Parameters
    ----------
    delta_g_r : float
        Radial component of gravity disturbance in m/s^2.
    delta_g_lat : float
        Northward component in m/s^2.
    delta_g_lon : float
        Eastward component in m/s^2.
    magnitude : float
        Total magnitude of disturbance in m/s^2.
    """

    delta_g_r: float
    delta_g_lat: float
    delta_g_lon: float
    magnitude: float


# Model parameters
EGM_PARAMETERS: Dict[str, Dict[str, float]] = {
    "EGM96": {
        "GM": 3.986004415e14,  # m^3/s^2
        "R": 6378136.3,  # m (reference radius)
        "n_max_full": 360,
    },
    "EGM2008": {
        "GM": 3.986004415e14,  # m^3/s^2
        "R": 6378136.3,  # m (reference radius)
        "n_max_full": 2190,
    },
}


def get_data_dir() -> Path:
    """Get the pytcl data directory.

    The data directory is located at ~/.pytcl/data/ by default.
    Can be overridden by setting the PYTCL_DATA_DIR environment variable.

    Returns
    -------
    Path
        Path to the data directory.

    Examples
    --------
    >>> data_dir = get_data_dir()
    >>> str(data_dir).endswith('.pytcl/data') or 'PYTCL_DATA_DIR' in dir()
    True
    """
    env_dir = os.environ.get("PYTCL_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        data_dir = Path.home() / ".pytcl" / "data"

    return data_dir


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def parse_egm_file(
    filepath: Path,
    n_max: Optional[int] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Parse an EGM coefficient file in NGA format.

    The file format is:
        n m C_nm S_nm [sigma_C sigma_S]

    where n is degree, m is order, C_nm and S_nm are the normalized
    coefficients, and sigma values are optional uncertainties.

    Parameters
    ----------
    filepath : Path
        Path to the coefficient file.
    n_max : int, optional
        Maximum degree to load. If None, loads all coefficients.

    Returns
    -------
    C : ndarray
        Cosine coefficients, shape (n_max+1, n_max+1).
    S : ndarray
        Sine coefficients, shape (n_max+1, n_max+1).
    """
    # First pass: determine maximum degree in file
    max_n_in_file = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                n = int(parts[0])
                max_n_in_file = max(max_n_in_file, n)

    # Determine actual n_max to use
    if n_max is None:
        actual_n_max = max_n_in_file
    else:
        actual_n_max = min(n_max, max_n_in_file)

    # Initialize coefficient arrays
    C = np.zeros((actual_n_max + 1, actual_n_max + 1))
    S = np.zeros((actual_n_max + 1, actual_n_max + 1))

    # C[0,0] = 1 by convention (central term)
    C[0, 0] = 1.0

    # Second pass: read coefficients
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            n = int(parts[0])
            m = int(parts[1])

            if n > actual_n_max:
                continue
            if m > n:
                continue

            try:
                C[n, m] = float(parts[2])
                S[n, m] = float(parts[3])
            except (ValueError, IndexError):
                continue

    return C, S


def create_test_coefficients(n_max: int = 36) -> EGMCoefficients:
    """Create test coefficients for low-degree testing.

    Creates a simplified set of coefficients based on published
    low-degree values from EGM2008 for testing purposes.

    Parameters
    ----------
    n_max : int
        Maximum degree (default 36).

    Returns
    -------
    EGMCoefficients
        Test coefficient set.

    Examples
    --------
    >>> coef = create_test_coefficients(n_max=10)
    >>> coef.n_max
    10
    >>> coef.C[0, 0]  # Central term
    1.0
    >>> abs(coef.C[2, 0]) > 0  # J2 term present
    True
    """
    C = np.zeros((n_max + 1, n_max + 1))
    S = np.zeros((n_max + 1, n_max + 1))

    # Central term
    C[0, 0] = 1.0

    # Low-degree coefficients from EGM2008 (normalized)
    # These are the dominant terms that determine the overall geoid shape

    # n=2 (oblateness and ellipticity)
    C[2, 0] = -0.484165371736e-03
    C[2, 1] = -0.186987635955e-09
    S[2, 1] = 0.119528012031e-08
    C[2, 2] = 0.243914352398e-05
    S[2, 2] = -0.140016683654e-05

    # n=3
    C[3, 0] = 0.957254173792e-06
    C[3, 1] = 0.202998882184e-05
    S[3, 1] = 0.248513158716e-06
    C[3, 2] = 0.904627768605e-06
    S[3, 2] = -0.619025944205e-06
    C[3, 3] = 0.721072657057e-06
    S[3, 3] = 0.141435626958e-05

    # n=4
    C[4, 0] = 0.539873863789e-06
    C[4, 1] = -0.536321616971e-06
    S[4, 1] = -0.473440265853e-06
    C[4, 2] = 0.350694105785e-06
    S[4, 2] = 0.662671572540e-06
    C[4, 3] = 0.990771803829e-06
    S[4, 3] = -0.200928369177e-06
    C[4, 4] = -0.188560802735e-06
    S[4, 4] = 0.308853169333e-06

    # n=5-6 (a few more terms for better accuracy)
    if n_max >= 5:
        C[5, 0] = 0.686702913736e-07
        C[6, 0] = -0.149953927978e-06

    return EGMCoefficients(
        C=C,
        S=S,
        GM=EGM_PARAMETERS["EGM2008"]["GM"],
        R=EGM_PARAMETERS["EGM2008"]["R"],
        n_max=n_max,
        model_name="EGM2008_TEST",
    )


@lru_cache(maxsize=4)
def _load_coefficients_cached(
    model: str,
    n_max: Optional[int],
) -> EGMCoefficients:
    """Cached coefficient loading (internal function).

    Parameters
    ----------
    model : str
        Model name.
    n_max : int or None
        Maximum degree.

    Returns
    -------
    EGMCoefficients
    """
    if model not in EGM_PARAMETERS:
        raise ValueError(f"Unknown model: {model}. Use 'EGM96' or 'EGM2008'.")

    params = EGM_PARAMETERS[model]

    # Determine coefficient file path
    data_dir = get_data_dir()
    filepath = data_dir / f"{model}.cof"

    _logger.debug("Loading %s coefficients from %s", model, filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {filepath}\n"
            f"Please download the {model} coefficients from:\n"
            f"  https://earth-info.nga.mil/\n"
            f"and save to: {filepath}\n"
            f"Or use create_test_coefficients() for testing."
        )

    # Parse the file
    actual_n_max = n_max if n_max is not None else int(params["n_max_full"])
    C, S = parse_egm_file(filepath, actual_n_max)

    _logger.info(
        "Loaded %s coefficients: n_max=%d, array_size=%.1f MB",
        model,
        C.shape[0] - 1,
        C.nbytes / 1024 / 1024 * 2,  # Both C and S arrays
    )

    return EGMCoefficients(
        C=C,
        S=S,
        GM=params["GM"],
        R=params["R"],
        n_max=C.shape[0] - 1,
        model_name=model,
    )


def load_egm_coefficients(
    model: str = "EGM2008",
    n_max: Optional[int] = None,
) -> EGMCoefficients:
    """Load EGM coefficients from file.

    Parameters
    ----------
    model : str
        Model name, either "EGM96" or "EGM2008".
    n_max : int, optional
        Maximum degree to load. If None, loads the full model.

    Returns
    -------
    EGMCoefficients
        Loaded coefficient structure.

    Raises
    ------
    FileNotFoundError
        If the coefficient file is not found.
    ValueError
        If an unknown model is specified.

    Examples
    --------
    >>> coef = load_egm_coefficients("EGM2008", n_max=360)
    >>> coef.n_max
    360
    """
    return _load_coefficients_cached(model, n_max)


def geoid_height(
    lat: float,
    lon: float,
    model: str = "EGM2008",
    n_max: Optional[int] = None,
    coefficients: Optional[EGMCoefficients] = None,
) -> float:
    """Compute geoid height above WGS84 ellipsoid.

    The geoid is the equipotential surface of the Earth's gravity field
    that best fits mean sea level. This function computes the height
    of the geoid above the WGS84 reference ellipsoid.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    model : str
        Model name ("EGM96" or "EGM2008").
    n_max : int, optional
        Maximum degree to use. Default uses full model.
    coefficients : EGMCoefficients, optional
        Pre-loaded coefficients. If None, loads from file.

    Returns
    -------
    float
        Geoid height in meters.

    Examples
    --------
    >>> # Geoid height at equator, prime meridian
    >>> N = geoid_height(0, 0)  # Should be approximately 17 m
    """
    if coefficients is None:
        coefficients = load_egm_coefficients(model, n_max)
    elif n_max is not None and n_max < coefficients.n_max:
        # Use subset of provided coefficients
        C = coefficients.C[: n_max + 1, : n_max + 1].copy()
        S = coefficients.S[: n_max + 1, : n_max + 1].copy()
        coefficients = EGMCoefficients(
            C=C,
            S=S,
            GM=coefficients.GM,
            R=coefficients.R,
            n_max=n_max,
            model_name=coefficients.model_name,
        )

    # Use reference radius as radial distance (on geoid)
    r = coefficients.R

    # Compute disturbing potential using Clenshaw summation
    # Exclude n=0,1 terms (reference field)
    C_dist = coefficients.C.copy()
    S_dist = coefficients.S.copy()
    C_dist[0, 0] = 0.0  # Remove central term
    if coefficients.n_max >= 1:
        C_dist[1, :] = 0.0
        S_dist[1, :] = 0.0

    T = clenshaw_potential(
        lat,
        lon,
        r,
        C_dist,
        S_dist,
        coefficients.R,
        coefficients.GM,
        coefficients.n_max,
    )

    # Normal gravity at the point (on ellipsoid surface)
    gamma = normal_gravity_somigliana(lat, WGS84)

    # Bruns' formula: N = T / gamma
    N = T / gamma

    return N


def geoid_heights(
    lats: NDArray[np.floating],
    lons: NDArray[np.floating],
    model: str = "EGM2008",
    n_max: Optional[int] = None,
    coefficients: Optional[EGMCoefficients] = None,
) -> NDArray[np.floating]:
    """Compute geoid heights for multiple points.

    Parameters
    ----------
    lats : ndarray
        Geodetic latitudes in radians.
    lons : ndarray
        Longitudes in radians.
    model : str
        Model name.
    n_max : int, optional
        Maximum degree.
    coefficients : EGMCoefficients, optional
        Pre-loaded coefficients. If None, loads from file.

    Returns
    -------
    ndarray
        Geoid heights in meters.

    Examples
    --------
    >>> import numpy as np
    >>> coef = create_test_coefficients(n_max=10)
    >>> lats = np.array([0.0, np.pi/4])  # Equator and 45°N
    >>> lons = np.array([0.0, np.pi/2])  # Prime meridian and 90°E
    >>> heights = geoid_heights(lats, lons, coefficients=coef)
    >>> len(heights)
    2
    """
    # Load coefficients once
    if coefficients is None:
        coefficients = load_egm_coefficients(model, n_max)

    # Compute for each point
    heights = np.zeros(len(lats))
    for i in range(len(lats)):
        heights[i] = geoid_height(
            lats[i], lons[i], model, n_max, coefficients=coefficients
        )

    return heights


def gravity_disturbance(
    lat: float,
    lon: float,
    h: float = 0.0,
    model: str = "EGM2008",
    n_max: Optional[int] = None,
    coefficients: Optional[EGMCoefficients] = None,
) -> GravityDisturbance:
    """Compute gravity disturbance from EGM model.

    The gravity disturbance is the difference between actual gravity
    and normal (reference ellipsoid) gravity at the same point.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float
        Height above ellipsoid in meters.
    model : str
        Model name.
    n_max : int, optional
        Maximum degree.
    coefficients : EGMCoefficients, optional
        Pre-loaded coefficients.

    Returns
    -------
    GravityDisturbance
        Gravity disturbance components.

    Examples
    --------
    >>> coef = create_test_coefficients(n_max=10)
    >>> dist = gravity_disturbance(0, 0, h=0, coefficients=coef)
    >>> isinstance(dist.magnitude, float)
    True
    """
    if coefficients is None:
        coefficients = load_egm_coefficients(model, n_max)

    # Radial distance (approximate)
    r = coefficients.R + h

    # Compute gravity disturbance using Clenshaw summation
    # Exclude n=0,1 terms
    C_dist = coefficients.C.copy()
    S_dist = coefficients.S.copy()
    C_dist[0, 0] = 0.0
    if coefficients.n_max >= 1:
        C_dist[1, :] = 0.0
        S_dist[1, :] = 0.0

    g_r, g_lat, g_lon = clenshaw_gravity(
        lat,
        lon,
        r,
        C_dist,
        S_dist,
        coefficients.R,
        coefficients.GM,
        coefficients.n_max,
    )

    magnitude = np.sqrt(g_r**2 + g_lat**2 + g_lon**2)

    return GravityDisturbance(
        delta_g_r=g_r,
        delta_g_lat=g_lat,
        delta_g_lon=g_lon,
        magnitude=magnitude,
    )


def gravity_anomaly(
    lat: float,
    lon: float,
    h: float = 0.0,
    model: str = "EGM2008",
    n_max: Optional[int] = None,
    coefficients: Optional[EGMCoefficients] = None,
) -> float:
    """Compute free-air gravity anomaly.

    The gravity anomaly is the difference between observed gravity
    and normal gravity, with the normal gravity evaluated at the
    geoid rather than at the observation point.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float
        Height above ellipsoid in meters.
    model : str
        Model name.
    n_max : int, optional
        Maximum degree.
    coefficients : EGMCoefficients, optional
        Pre-loaded coefficients. If None, loads from file.

    Returns
    -------
    float
        Gravity anomaly in m/s^2 (typically reported in mGal = 1e-5 m/s^2).

    Examples
    --------
    >>> coef = create_test_coefficients(n_max=10)
    >>> anomaly = gravity_anomaly(0, 0, h=0, coefficients=coef)
    >>> isinstance(anomaly, float)
    True
    """
    disturbance = gravity_disturbance(lat, lon, h, model, n_max, coefficients)

    # For free-air anomaly, we mainly care about the radial component
    # at the observation height
    return disturbance.delta_g_r


def deflection_of_vertical(
    lat: float,
    lon: float,
    model: str = "EGM2008",
    n_max: Optional[int] = None,
    coefficients: Optional[EGMCoefficients] = None,
) -> Tuple[float, float]:
    """Compute deflection of the vertical.

    The deflection of the vertical is the angle between the direction
    of gravity (plumb line) and the normal to the reference ellipsoid.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    model : str
        Model name.
    n_max : int, optional
        Maximum degree.
    coefficients : EGMCoefficients, optional
        Pre-loaded coefficients. If None, loads from file.

    Returns
    -------
    xi : float
        North-south deflection component in arcseconds.
    eta : float
        East-west deflection component in arcseconds.

    Notes
    -----
    Positive xi means the plumb line points more north than the normal.
    Positive eta means the plumb line points more east than the normal.

    Examples
    --------
    >>> coef = create_test_coefficients(n_max=10)
    >>> xi, eta = deflection_of_vertical(0, 0, coefficients=coef)
    >>> isinstance(xi, float) and isinstance(eta, float)
    True
    """
    if coefficients is None:
        coefficients = load_egm_coefficients(model, n_max)

    # Compute geoid height gradient
    # Use finite differences
    delta = 1e-6  # Small angle in radians (~0.2 arcsec)

    N_center = geoid_height(lat, lon, coefficients=coefficients)
    N_north = geoid_height(lat + delta, lon, coefficients=coefficients)
    N_east = geoid_height(lat, lon + delta, coefficients=coefficients)

    # Geoid height gradient
    # dN/dlat (meters per radian)
    dN_dlat = (N_north - N_center) / delta

    # dN/dlon (meters per radian)
    dN_dlon = (N_east - N_center) / delta

    # Earth radius at the point
    R = coefficients.R

    # Deflection components (small angle approximation)
    # xi = -dN/ds_north / R = -dN/dlat / R (in radians)
    # eta = -dN/ds_east / (R*cos(lat)) = -dN/dlon / (R*cos(lat))
    xi_rad = -dN_dlat / R
    eta_rad = -dN_dlon / (R * np.cos(lat))

    # Convert to arcseconds
    RAD_TO_ARCSEC = 3600.0 * 180.0 / np.pi
    xi = xi_rad * RAD_TO_ARCSEC
    eta = eta_rad * RAD_TO_ARCSEC

    return xi, eta


__all__ = [
    "EGMCoefficients",
    "GeoidResult",
    "GravityDisturbance",
    "EGM_PARAMETERS",
    "get_data_dir",
    "parse_egm_file",
    "create_test_coefficients",
    "load_egm_coefficients",
    "geoid_height",
    "geoid_heights",
    "gravity_disturbance",
    "gravity_anomaly",
    "deflection_of_vertical",
]
