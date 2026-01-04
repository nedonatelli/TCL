"""
Enhanced Magnetic Model (EMM) and World Magnetic Model High Resolution (WMMHR).

This module provides support for high-degree magnetic field models:
- EMM2017: Degree 790, resolves anomalies down to 51 km wavelength
- WMMHR2025: Degree 133, resolves anomalies down to ~300 km wavelength

These models provide higher spatial resolution than the standard WMM (degree 12)
by including crustal magnetic field contributions.

References
----------
.. [1] Maus, S., et al. "EMAG2: A 2-arc min resolution Earth Magnetic
       Anomaly Grid compiled from satellite, airborne, and marine
       magnetic measurements." Geochemistry, Geophysics, Geosystems 10.8 (2009).
.. [2] NOAA National Centers for Environmental Information,
       "Enhanced Magnetic Model (EMM)."
       https://www.ncei.noaa.gov/products/enhanced-magnetic-model
.. [3] NOAA National Centers for Environmental Information,
       "World Magnetic Model High Resolution (WMMHR)."
       https://www.ncei.noaa.gov/products/world-magnetic-model-high-resolution
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .wmm import MagneticResult

# Model parameters
EMM_PARAMETERS: dict[str, dict[str, Any]] = {
    "EMM2017": {
        "n_max": 790,
        "epoch": 2017.0,
        "valid_start": 2000.0,
        "valid_end": 2022.0,
        "reference_radius": 6371.2,  # km
        "file_size_mb": 24.5,
    },
    "WMMHR2025": {
        "n_max": 133,
        "n_max_sv": 15,  # Secular variation only to degree 15
        "epoch": 2025.0,
        "valid_start": 2025.0,
        "valid_end": 2030.0,
        "reference_radius": 6371.2,  # km
        "file_size_kb": 534,
    },
}


class HighResCoefficients(NamedTuple):
    """High-resolution magnetic model coefficients.

    Extends MagneticCoefficients with additional metadata for
    high-degree models like EMM and WMMHR.

    Attributes
    ----------
    g : ndarray
        Main field cosine coefficients (nT), shape (n_max+1, n_max+1).
    h : ndarray
        Main field sine coefficients (nT), shape (n_max+1, n_max+1).
    g_dot : ndarray
        Secular variation of g (nT/year), shape (n_max_sv+1, n_max_sv+1).
    h_dot : ndarray
        Secular variation of h (nT/year), shape (n_max_sv+1, n_max_sv+1).
    epoch : float
        Reference epoch (decimal year).
    n_max : int
        Maximum degree for main field.
    n_max_sv : int
        Maximum degree for secular variation (may be < n_max).
    model_name : str
        Model identifier ("EMM2017" or "WMMHR2025").
    """

    g: NDArray[np.floating]
    h: NDArray[np.floating]
    g_dot: NDArray[np.floating]
    h_dot: NDArray[np.floating]
    epoch: float
    n_max: int
    n_max_sv: int
    model_name: str


def get_data_dir() -> Path:
    """Get the pytcl data directory for magnetic coefficients.

    The data directory is located at ~/.pytcl/data/ by default.
    Can be overridden by setting the PYTCL_DATA_DIR environment variable.

    Returns
    -------
    Path
        Path to the data directory.
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


def parse_emm_file(
    filepath: Path,
    n_max: Optional[int] = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    float,
    int,
]:
    """Parse an EMM/WMMHR coefficient file.

    The file format is similar to WMM but with more coefficients:
        n m g_nm h_nm g_dot_nm h_dot_nm

    Parameters
    ----------
    filepath : Path
        Path to the coefficient file.
    n_max : int, optional
        Maximum degree to load. If None, loads all coefficients.

    Returns
    -------
    g : ndarray
        Cosine coefficients.
    h : ndarray
        Sine coefficients.
    g_dot : ndarray
        Secular variation of g.
    h_dot : ndarray
        Secular variation of h.
    epoch : float
        Model epoch.
    actual_n_max : int
        Maximum degree loaded.
    """
    epoch = 2017.0  # Default
    max_n_in_file = 0
    max_n_sv = 0

    # First pass: determine dimensions and epoch
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Parse header line (first line with epoch info)
            if line_num == 0:
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        epoch = float(parts[0])
                    except ValueError:
                        pass
                continue

            # Skip comment lines
            if line.startswith("#") or line.startswith("9999"):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    n = int(parts[0])
                    max_n_in_file = max(max_n_in_file, n)

                    # Check if secular variation is present
                    if len(parts) >= 6:
                        g_dot_val = float(parts[4])
                        h_dot_val = float(parts[5])
                        if g_dot_val != 0 or h_dot_val != 0:
                            max_n_sv = max(max_n_sv, n)
                except (ValueError, IndexError):
                    continue

    # Determine actual limits
    if n_max is None:
        actual_n_max = max_n_in_file
    else:
        actual_n_max = min(n_max, max_n_in_file)

    actual_n_max_sv = min(max_n_sv, actual_n_max)

    # Initialize coefficient arrays
    g = np.zeros((actual_n_max + 1, actual_n_max + 1))
    h = np.zeros((actual_n_max + 1, actual_n_max + 1))
    g_dot = np.zeros((actual_n_max_sv + 1, actual_n_max_sv + 1))
    h_dot = np.zeros((actual_n_max_sv + 1, actual_n_max_sv + 1))

    # Second pass: read coefficients
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line_num == 0:
                continue
            if line.startswith("#") or line.startswith("9999"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                n = int(parts[0])
                m = int(parts[1])

                if n > actual_n_max or m > n:
                    continue

                g[n, m] = float(parts[2])
                h[n, m] = float(parts[3])

                # Secular variation (if present)
                if len(parts) >= 6 and n <= actual_n_max_sv:
                    g_dot[n, m] = float(parts[4])
                    h_dot[n, m] = float(parts[5])

            except (ValueError, IndexError):
                continue

    return g, h, g_dot, h_dot, epoch, actual_n_max


def create_test_coefficients(n_max: int = 36) -> HighResCoefficients:
    """Create test coefficients for low-degree testing.

    Creates a simplified set of coefficients for testing purposes,
    using the same low-degree terms as WMM2020 extended with
    synthetic crustal field terms.

    Parameters
    ----------
    n_max : int
        Maximum degree (default 36).

    Returns
    -------
    HighResCoefficients
        Test coefficient set.
    """
    n_max_sv = min(n_max, 12)  # SV only to degree 12

    g = np.zeros((n_max + 1, n_max + 1))
    h = np.zeros((n_max + 1, n_max + 1))
    g_dot = np.zeros((n_max_sv + 1, n_max_sv + 1))
    h_dot = np.zeros((n_max_sv + 1, n_max_sv + 1))

    # Core field coefficients from WMM2020 (degrees 1-12)
    # n=1 (dipole)
    g[1, 0] = -29404.5
    g[1, 1] = -1450.7
    h[1, 1] = 4652.9

    # n=2
    g[2, 0] = -2500.0
    g[2, 1] = 2982.0
    g[2, 2] = 1676.8
    h[2, 1] = -2991.6
    h[2, 2] = -734.8

    # n=3
    g[3, 0] = 1363.9
    g[3, 1] = -2381.0
    g[3, 2] = 1236.2
    g[3, 3] = 525.7
    h[3, 1] = -82.2
    h[3, 2] = 241.8
    h[3, 3] = -542.9

    # n=4
    g[4, 0] = 903.1
    g[4, 1] = 809.4
    g[4, 2] = 86.2
    g[4, 3] = -309.4
    g[4, 4] = 47.9
    h[4, 1] = 282.0
    h[4, 2] = -158.4
    h[4, 3] = 199.8
    h[4, 4] = -350.1

    # n=5
    g[5, 0] = -234.4
    g[5, 1] = 363.1
    g[5, 2] = 47.7
    g[5, 3] = 187.8
    g[5, 4] = -140.7
    g[5, 5] = -151.2
    h[5, 1] = 46.7
    h[5, 2] = 196.9
    h[5, 3] = -119.4
    h[5, 4] = 16.0
    h[5, 5] = 100.1

    # n=6
    g[6, 0] = 65.6
    g[6, 1] = 65.5
    g[6, 2] = 51.9
    g[6, 3] = -61.1
    g[6, 4] = 16.9
    g[6, 5] = 0.7
    g[6, 6] = -91.1
    h[6, 1] = -77.0
    h[6, 2] = 25.0
    h[6, 3] = 26.4
    h[6, 4] = -16.3
    h[6, 5] = 28.9
    h[6, 6] = 4.1

    # Secular variation (degrees 1-6 only for test)
    g_dot[1, 0] = 6.7
    g_dot[1, 1] = 7.7
    h_dot[1, 1] = -25.1

    g_dot[2, 0] = -11.5
    g_dot[2, 1] = -7.1
    g_dot[2, 2] = -2.2
    h_dot[2, 1] = -30.2
    h_dot[2, 2] = -23.9

    # Add synthetic crustal field for higher degrees (small values)
    # These simulate the crustal field contribution
    np.random.seed(42)  # Reproducible
    for n in range(13, n_max + 1):
        # Crustal field decays with increasing degree
        amplitude = 10.0 / n  # nT
        for m in range(n + 1):
            g[n, m] = amplitude * (2 * np.random.random() - 1)
            h[n, m] = amplitude * (2 * np.random.random() - 1)

    return HighResCoefficients(
        g=g,
        h=h,
        g_dot=g_dot,
        h_dot=h_dot,
        epoch=2020.0,
        n_max=n_max,
        n_max_sv=n_max_sv,
        model_name="EMM_TEST",
    )


@lru_cache(maxsize=4)
def _load_coefficients_cached(
    model: str,
    n_max: Optional[int],
) -> HighResCoefficients:
    """Cached coefficient loading (internal function).

    Parameters
    ----------
    model : str
        Model name ("EMM2017" or "WMMHR2025").
    n_max : int or None
        Maximum degree.

    Returns
    -------
    HighResCoefficients
    """
    if model not in EMM_PARAMETERS:
        raise ValueError(f"Unknown model: {model}. Use 'EMM2017' or 'WMMHR2025'.")

    params = EMM_PARAMETERS[model]

    # Determine coefficient file path
    data_dir = get_data_dir()

    # Try different file extensions
    for ext in [".cof", ".COF", ".txt", ".dat"]:
        filepath = data_dir / f"{model}{ext}"
        if filepath.exists():
            break
    else:
        ncei_base = "https://www.ncei.noaa.gov/products"
        emm_url = f"{ncei_base}/enhanced-magnetic-model"
        wmmhr_url = f"{ncei_base}/world-magnetic-model-high-resolution"
        raise FileNotFoundError(
            f"Coefficient file not found for {model}\n"
            f"Please download the {model} coefficients from:\n"
            f"  {emm_url} (EMM)\n"
            f"  {wmmhr_url} (WMMHR)\n"
            f"and save to: {data_dir}/{model}.cof\n"
            f"Or use create_test_coefficients() for testing."
        )

    # Parse the file
    actual_n_max = n_max if n_max is not None else params["n_max"]
    g, h, g_dot, h_dot, epoch, loaded_n_max = parse_emm_file(filepath, actual_n_max)

    n_max_sv = params.get("n_max_sv", loaded_n_max)
    n_max_sv = min(n_max_sv, g_dot.shape[0] - 1)

    return HighResCoefficients(
        g=g,
        h=h,
        g_dot=g_dot,
        h_dot=h_dot,
        epoch=epoch,
        n_max=loaded_n_max,
        n_max_sv=n_max_sv,
        model_name=model,
    )


def load_emm_coefficients(
    model: str = "EMM2017",
    n_max: Optional[int] = None,
) -> HighResCoefficients:
    """Load high-resolution magnetic model coefficients.

    Parameters
    ----------
    model : str
        Model name, either "EMM2017" or "WMMHR2025".
    n_max : int, optional
        Maximum degree to load. If None, loads the full model.

    Returns
    -------
    HighResCoefficients
        Loaded coefficient structure.

    Raises
    ------
    FileNotFoundError
        If the coefficient file is not found.
    ValueError
        If an unknown model is specified.

    Examples
    --------
    >>> coef = load_emm_coefficients("WMMHR2025", n_max=50)
    >>> coef.n_max
    50
    """
    return _load_coefficients_cached(model, n_max)


def _high_res_field_spherical(
    lat: float,
    lon: float,
    r: float,
    year: float,
    coeffs: HighResCoefficients,
    n_max_eval: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute magnetic field using high-resolution coefficients.

    Similar to magnetic_field_spherical but handles different n_max
    for main field and secular variation.

    Parameters
    ----------
    lat : float
        Geocentric latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from Earth's center in km.
    year : float
        Decimal year.
    coeffs : HighResCoefficients
        Model coefficients.
    n_max_eval : int, optional
        Maximum degree to evaluate. Default uses full model.

    Returns
    -------
    B_r : float
        Radial component in nT.
    B_theta : float
        Colatitude component in nT.
    B_phi : float
        Longitude component in nT.
    """
    from pytcl.gravity.spherical_harmonics import associated_legendre

    if n_max_eval is None:
        n_max_eval = coeffs.n_max
    else:
        n_max_eval = min(n_max_eval, coeffs.n_max)

    a = 6371.2  # Reference radius in km

    # Time adjustment
    dt = year - coeffs.epoch

    # Create time-adjusted coefficients
    # Main field always uses full n_max
    g = coeffs.g.copy()
    h = coeffs.h.copy()

    # Secular variation only up to n_max_sv
    n_max_sv = min(coeffs.n_max_sv, n_max_eval)
    for n in range(1, n_max_sv + 1):
        for m in range(n + 1):
            if n < coeffs.g_dot.shape[0] and m < coeffs.g_dot.shape[1]:
                g[n, m] += dt * coeffs.g_dot[n, m]
                h[n, m] += dt * coeffs.h_dot[n, m]

    # Colatitude
    theta = np.pi / 2 - lat
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Compute associated Legendre functions
    P = associated_legendre(n_max_eval, n_max_eval, cos_theta, normalized=True)

    # Compute dP/dtheta
    dP = np.zeros((n_max_eval + 1, n_max_eval + 1))
    if abs(sin_theta) > 1e-10:
        for n in range(1, n_max_eval + 1):
            for m in range(n + 1):
                if m == n:
                    dP[n, m] = n * cos_theta / sin_theta * P[n, m]
                elif n > m:
                    factor = np.sqrt((n - m) * (n + m + 1))
                    if m + 1 <= n:
                        dP[n, m] = (
                            n * cos_theta / sin_theta * P[n, m]
                            - factor * P[n, m + 1] / sin_theta
                            if m + 1 <= n_max_eval
                            else n * cos_theta / sin_theta * P[n, m]
                        )

    # Initialize field components
    B_r = 0.0
    B_theta = 0.0
    B_phi = 0.0

    # Sum over spherical harmonic degrees and orders
    r_ratio = a / r

    for n in range(1, n_max_eval + 1):
        r_power = r_ratio ** (n + 2)

        for m in range(n + 1):
            cos_m_lon = np.cos(m * lon)
            sin_m_lon = np.sin(m * lon)

            gnm = g[n, m]
            hnm = h[n, m]

            B_r += (n + 1) * r_power * P[n, m] * (gnm * cos_m_lon + hnm * sin_m_lon)
            B_theta += -r_power * dP[n, m] * (gnm * cos_m_lon + hnm * sin_m_lon)

            if abs(sin_theta) > 1e-10:
                B_phi += (
                    r_power
                    * m
                    * P[n, m]
                    / sin_theta
                    * (gnm * sin_m_lon - hnm * cos_m_lon)
                )

    return B_r, B_theta, B_phi


def emm(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2020.0,
    model: str = "EMM2017",
    n_max: Optional[int] = None,
    coefficients: Optional[HighResCoefficients] = None,
) -> MagneticResult:
    """Compute magnetic field using Enhanced Magnetic Model.

    The EMM provides higher spatial resolution than WMM by including
    crustal magnetic field contributions up to degree 790.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in km. Default 0.
    year : float, optional
        Decimal year. Default 2020.0.
    model : str, optional
        Model name ("EMM2017" or "WMMHR2025"). Default "EMM2017".
    n_max : int, optional
        Maximum degree to evaluate. Default uses full model.
    coefficients : HighResCoefficients, optional
        Pre-loaded coefficients. If None, loads from file.

    Returns
    -------
    result : MagneticResult
        Magnetic field components and derived quantities.

    Examples
    --------
    >>> import numpy as np
    >>> # Use test coefficients for demonstration
    >>> coef = create_test_coefficients(n_max=36)
    >>> result = emm(np.radians(40), np.radians(-105), 1.0, 2020.0, coefficients=coef)
    >>> print(f"Declination: {np.degrees(result.D):.2f}°")
    >>> print(f"Total intensity: {result.F:.0f} nT")
    """
    if coefficients is None:
        coefficients = load_emm_coefficients(model, n_max)

    # Geocentric approximation
    a = 6371.2  # km
    lat_gc = lat
    r = a + h

    # Compute field in spherical coordinates
    B_r, B_theta, B_phi = _high_res_field_spherical(
        lat_gc, lon, r, year, coefficients, n_max
    )

    # Convert to geodetic coordinates
    X = -B_theta  # North
    Y = B_phi  # East
    Z = -B_r  # Down

    # Derived quantities
    H = np.sqrt(X * X + Y * Y)
    F = np.sqrt(H * H + Z * Z)
    incl = np.arctan2(Z, H)
    D = np.arctan2(Y, X)

    return MagneticResult(X=X, Y=Y, Z=Z, H=H, F=F, I=incl, D=D)


def wmmhr(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2025.0,
    n_max: Optional[int] = None,
    coefficients: Optional[HighResCoefficients] = None,
) -> MagneticResult:
    """Compute magnetic field using World Magnetic Model High Resolution.

    WMMHR2025 extends WMM to degree 133, providing higher spatial
    resolution for applications requiring more accurate local field values.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in km. Default 0.
    year : float, optional
        Decimal year. Default 2025.0.
    n_max : int, optional
        Maximum degree to evaluate. Default uses full model (133).
    coefficients : HighResCoefficients, optional
        Pre-loaded coefficients. If None, loads from file.

    Returns
    -------
    result : MagneticResult
        Magnetic field components and derived quantities.

    Notes
    -----
    WMMHR2025 is valid for 2025.0 - 2030.0. For dates outside this
    range, results may be less accurate.

    Secular variation is only computed for degrees 1-15. Higher degree
    terms represent static crustal field.
    """
    return emm(
        lat,
        lon,
        h,
        year,
        model="WMMHR2025",
        n_max=n_max,
        coefficients=coefficients,
    )


def emm_declination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2020.0,
    model: str = "EMM2017",
    n_max: Optional[int] = None,
    coefficients: Optional[HighResCoefficients] = None,
) -> float:
    """Compute magnetic declination using EMM/WMMHR.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in km.
    year : float, optional
        Decimal year.
    model : str, optional
        Model name.
    n_max : int, optional
        Maximum degree to evaluate.
    coefficients : HighResCoefficients, optional
        Pre-loaded coefficients.

    Returns
    -------
    float
        Magnetic declination in radians.
    """
    result = emm(lat, lon, h, year, model, n_max, coefficients)
    return result.D


def emm_inclination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2020.0,
    model: str = "EMM2017",
    n_max: Optional[int] = None,
    coefficients: Optional[HighResCoefficients] = None,
) -> float:
    """Compute magnetic inclination using EMM/WMMHR.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in km.
    year : float, optional
        Decimal year.
    model : str, optional
        Model name.
    n_max : int, optional
        Maximum degree to evaluate.
    coefficients : HighResCoefficients, optional
        Pre-loaded coefficients.

    Returns
    -------
    float
        Magnetic inclination (dip) in radians.
    """
    result = emm(lat, lon, h, year, model, n_max, coefficients)
    return result.I


def emm_intensity(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2020.0,
    model: str = "EMM2017",
    n_max: Optional[int] = None,
    coefficients: Optional[HighResCoefficients] = None,
) -> float:
    """Compute total magnetic field intensity using EMM/WMMHR.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in km.
    year : float, optional
        Decimal year.
    model : str, optional
        Model name.
    n_max : int, optional
        Maximum degree to evaluate.
    coefficients : HighResCoefficients, optional
        Pre-loaded coefficients.

    Returns
    -------
    float
        Total magnetic field intensity in nT.
    """
    result = emm(lat, lon, h, year, model, n_max, coefficients)
    return result.F


__all__ = [
    "HighResCoefficients",
    "EMM_PARAMETERS",
    "get_data_dir",
    "parse_emm_file",
    "create_test_coefficients",
    "load_emm_coefficients",
    "emm",
    "wmmhr",
    "emm_declination",
    "emm_inclination",
    "emm_intensity",
]
