"""
World Magnetic Model (WMM) implementation.

The WMM is the standard model used by the U.S. Department of Defense,
the U.K. Ministry of Defence, NATO, and the International Hydrographic
Organization for navigation, attitude, and heading referencing.

References
----------
.. [1] Chulliat et al., "The US/UK World Magnetic Model for 2020-2025,"
       NOAA Technical Report, 2020.
.. [2] https://www.ngdc.noaa.gov/geomag/WMM/
"""

from functools import lru_cache
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.gravity.spherical_harmonics import associated_legendre

# =============================================================================
# Cache Configuration
# =============================================================================

# Default cache size (number of unique location/time combinations to cache)
_DEFAULT_CACHE_SIZE = 1024

# Precision for rounding inputs (radians for lat/lon, km for radius, years for time)
# These control how aggressively similar inputs are grouped
_CACHE_PRECISION = {
    "lat": 6,  # ~0.1 meter precision at Earth surface
    "lon": 6,
    "r": 3,  # 1 meter precision
    "year": 2,  # ~4 day precision
}


class MagneticResult(NamedTuple):
    """Result of magnetic field computation.

    Attributes
    ----------
    X : float
        Northward component (nT).
    Y : float
        Eastward component (nT).
    Z : float
        Downward component (nT).
    H : float
        Horizontal intensity (nT).
    F : float
        Total intensity (nT).
    I : float
        Inclination (dip angle) in radians.
    D : float
        Declination in radians.
    """

    X: float
    Y: float
    Z: float
    H: float
    F: float
    I: float
    D: float


class MagneticCoefficients(NamedTuple):
    """Spherical harmonic coefficients for magnetic model.

    Attributes
    ----------
    g : ndarray
        Main field cosine coefficients (nT).
    h : ndarray
        Main field sine coefficients (nT).
    g_dot : ndarray
        Secular variation of g (nT/year).
    h_dot : ndarray
        Secular variation of h (nT/year).
    epoch : float
        Reference epoch (decimal year).
    n_max : int
        Maximum degree.
    """

    g: NDArray[np.floating]
    h: NDArray[np.floating]
    g_dot: NDArray[np.floating]
    h_dot: NDArray[np.floating]
    epoch: float
    n_max: int


def create_wmm2020_coefficients() -> MagneticCoefficients:
    """
    Create WMM2020 model coefficients.

    Returns
    -------
    coeffs : MagneticCoefficients
        WMM2020 spherical harmonic coefficients.

    Notes
    -----
    These are the official WMM2020 coefficients valid from 2020.0 to 2025.0.
    For use beyond 2025, updated coefficients should be obtained from NOAA.
    """
    n_max = 12
    g = np.zeros((n_max + 1, n_max + 1))
    h = np.zeros((n_max + 1, n_max + 1))
    g_dot = np.zeros((n_max + 1, n_max + 1))
    h_dot = np.zeros((n_max + 1, n_max + 1))

    # WMM2020 main field coefficients (selected low-degree terms)
    # Full model has coefficients up to n=12
    # Units: nT (nanotesla)

    # n=1
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
    g[6, 2] = -19.1
    g[6, 3] = 73.0
    g[6, 4] = -62.7
    g[6, 5] = 0.6
    g[6, 6] = -24.2
    h[6, 1] = -76.7
    h[6, 2] = 25.4
    h[6, 3] = -9.2
    h[6, 4] = 55.9
    h[6, 5] = -17.0
    h[6, 6] = 8.4

    # n=7
    g[7, 0] = 80.6
    g[7, 1] = -76.7
    g[7, 2] = -8.2
    g[7, 3] = -26.6
    g[7, 4] = 3.0
    g[7, 5] = -14.9
    g[7, 6] = 10.4
    g[7, 7] = -18.3
    h[7, 1] = 0.2
    h[7, 2] = -21.5
    h[7, 3] = 15.5
    h[7, 4] = 13.8
    h[7, 5] = -13.5
    h[7, 6] = -0.1
    h[7, 7] = 8.8

    # n=8
    g[8, 0] = 24.4
    g[8, 1] = 6.0
    g[8, 2] = -2.0
    g[8, 3] = -5.8
    g[8, 4] = 0.1
    g[8, 5] = 10.9
    g[8, 6] = -1.3
    g[8, 7] = -6.5
    g[8, 8] = -2.0
    h[8, 1] = -20.4
    h[8, 2] = 13.4
    h[8, 3] = 12.1
    h[8, 4] = -6.4
    h[8, 5] = -8.4
    h[8, 6] = 8.6
    h[8, 7] = 2.2
    h[8, 8] = -7.0

    # n=9
    g[9, 0] = 5.0
    g[9, 1] = 8.4
    g[9, 2] = 3.0
    g[9, 3] = -1.5
    g[9, 4] = 0.1
    g[9, 5] = -3.8
    g[9, 6] = 4.3
    g[9, 7] = -1.4
    g[9, 8] = -2.4
    g[9, 9] = -6.0
    h[9, 1] = 0.9
    h[9, 2] = -1.4
    h[9, 3] = 3.8
    h[9, 4] = -5.3
    h[9, 5] = -0.3
    h[9, 6] = 0.4
    h[9, 7] = 1.7
    h[9, 8] = -0.9
    h[9, 9] = 4.6

    # n=10
    g[10, 0] = -1.8
    g[10, 1] = -0.7
    g[10, 2] = 2.1
    g[10, 3] = 2.1
    g[10, 4] = -2.4
    g[10, 5] = -1.8
    g[10, 6] = -0.5
    g[10, 7] = 0.6
    g[10, 8] = 0.9
    g[10, 9] = -0.8
    g[10, 10] = -0.2
    h[10, 1] = 0.8
    h[10, 2] = -0.4
    h[10, 3] = -0.2
    h[10, 4] = 0.7
    h[10, 5] = 0.3
    h[10, 6] = 2.2
    h[10, 7] = -2.5
    h[10, 8] = 0.5
    h[10, 9] = 0.6
    h[10, 10] = -0.4

    # n=11
    g[11, 0] = 3.0
    g[11, 1] = -1.5
    g[11, 2] = -0.2
    g[11, 3] = -0.3
    g[11, 4] = 0.5
    g[11, 5] = 1.3
    g[11, 6] = -1.2
    g[11, 7] = 0.7
    g[11, 8] = 0.4
    g[11, 9] = 0.0
    g[11, 10] = 0.6
    g[11, 11] = -0.5
    h[11, 1] = -0.2
    h[11, 2] = 0.4
    h[11, 3] = 0.5
    h[11, 4] = 0.4
    h[11, 5] = -0.6
    h[11, 6] = 0.3
    h[11, 7] = 0.0
    h[11, 8] = -0.4
    h[11, 9] = 0.1
    h[11, 10] = -0.3
    h[11, 11] = -0.3

    # n=12
    g[12, 0] = -0.2
    g[12, 1] = -0.2
    g[12, 2] = -0.1
    g[12, 3] = 0.1
    g[12, 4] = 0.5
    g[12, 5] = 1.1
    g[12, 6] = -0.3
    g[12, 7] = -0.4
    g[12, 8] = -0.3
    g[12, 9] = 0.2
    g[12, 10] = -0.5
    g[12, 11] = 0.4
    g[12, 12] = -0.2
    h[12, 1] = 0.1
    h[12, 2] = 0.5
    h[12, 3] = 0.0
    h[12, 4] = -0.2
    h[12, 5] = 0.3
    h[12, 6] = -0.4
    h[12, 7] = 0.3
    h[12, 8] = 0.3
    h[12, 9] = -0.1
    h[12, 10] = -0.1
    h[12, 11] = -0.1
    h[12, 12] = -0.2

    # Secular variation (nT/year) - all terms
    # n=1
    g_dot[1, 0] = 6.7
    g_dot[1, 1] = 7.7
    h_dot[1, 1] = -25.1

    # n=2
    g_dot[2, 0] = -11.5
    g_dot[2, 1] = -7.1
    g_dot[2, 2] = -2.2
    h_dot[2, 1] = -30.2
    h_dot[2, 2] = -23.9

    # n=3
    g_dot[3, 0] = 2.8
    g_dot[3, 1] = -6.2
    g_dot[3, 2] = 3.4
    g_dot[3, 3] = -12.2
    h_dot[3, 1] = 5.7
    h_dot[3, 2] = -1.0
    h_dot[3, 3] = 1.1

    # n=4
    g_dot[4, 0] = -1.1
    g_dot[4, 1] = -1.6
    g_dot[4, 2] = -6.0
    g_dot[4, 3] = 5.4
    g_dot[4, 4] = -5.5
    h_dot[4, 1] = 0.2
    h_dot[4, 2] = 6.4
    h_dot[4, 3] = 3.1
    h_dot[4, 4] = -12.0

    # n=5
    g_dot[5, 0] = -0.3
    g_dot[5, 1] = 0.1
    g_dot[5, 2] = -0.6
    g_dot[5, 3] = 0.2
    g_dot[5, 4] = 0.3
    g_dot[5, 5] = 1.0
    h_dot[5, 1] = -0.4
    h_dot[5, 2] = 2.1
    h_dot[5, 3] = 3.4
    h_dot[5, 4] = -0.9
    h_dot[5, 5] = -1.2

    # n=6
    g_dot[6, 0] = -0.6
    g_dot[6, 1] = -0.4
    g_dot[6, 2] = 0.5
    g_dot[6, 3] = 1.4
    g_dot[6, 4] = -1.4
    g_dot[6, 5] = 0.0
    g_dot[6, 6] = 0.8
    h_dot[6, 1] = -0.2
    h_dot[6, 2] = -0.9
    h_dot[6, 3] = 0.3
    h_dot[6, 4] = 0.1
    h_dot[6, 5] = -0.1
    h_dot[6, 6] = 0.4

    # n=7
    g_dot[7, 0] = -0.1
    g_dot[7, 1] = -0.3
    g_dot[7, 2] = 0.3
    g_dot[7, 3] = 0.2
    g_dot[7, 4] = -0.5
    g_dot[7, 5] = 0.2
    g_dot[7, 6] = -0.2
    g_dot[7, 7] = 0.6
    h_dot[7, 1] = -0.5
    h_dot[7, 2] = 0.4
    h_dot[7, 3] = 0.1
    h_dot[7, 4] = 0.4
    h_dot[7, 5] = -0.2
    h_dot[7, 6] = 0.4
    h_dot[7, 7] = 0.3

    # n=8
    g_dot[8, 0] = 0.0
    g_dot[8, 1] = 0.0
    g_dot[8, 2] = 0.1
    g_dot[8, 3] = -0.2
    g_dot[8, 4] = 0.4
    g_dot[8, 5] = 0.3
    g_dot[8, 6] = 0.0
    g_dot[8, 7] = 0.1
    g_dot[8, 8] = -0.1
    h_dot[8, 1] = 0.1
    h_dot[8, 2] = -0.1
    h_dot[8, 3] = 0.3
    h_dot[8, 4] = 0.0
    h_dot[8, 5] = 0.2
    h_dot[8, 6] = -0.1
    h_dot[8, 7] = -0.1
    h_dot[8, 8] = 0.0

    return MagneticCoefficients(
        g=g,
        h=h,
        g_dot=g_dot,
        h_dot=h_dot,
        epoch=2020.0,
        n_max=n_max,
    )


# Default WMM2020 coefficients
WMM2020 = create_wmm2020_coefficients()


# =============================================================================
# Cached Computation Core
# =============================================================================


def _quantize_inputs(
    lat: float, lon: float, r: float, year: float
) -> Tuple[float, float, float, float]:
    """Round inputs to cache precision for consistent cache hits."""
    return (
        round(lat, _CACHE_PRECISION["lat"]),
        round(lon, _CACHE_PRECISION["lon"]),
        round(r, _CACHE_PRECISION["r"]),
        round(year, _CACHE_PRECISION["year"]),
    )


@lru_cache(maxsize=_DEFAULT_CACHE_SIZE)
def _magnetic_field_spherical_cached(
    lat: float,
    lon: float,
    r: float,
    year: float,
    n_max: int,
    coeff_id: int,
) -> Tuple[float, float, float]:
    """
    Cached core computation of magnetic field in spherical coordinates.

    This is the internal cached version. The coefficient arrays are identified
    by their id() since NamedTuples with numpy arrays aren't hashable.

    Parameters
    ----------
    lat : float
        Geocentric latitude in radians (quantized).
    lon : float
        Longitude in radians (quantized).
    r : float
        Radial distance in km (quantized).
    year : float
        Decimal year (quantized).
    n_max : int
        Maximum spherical harmonic degree.
    coeff_id : int
        Unique identifier for the coefficient set.

    Returns
    -------
    B_r, B_theta, B_phi : tuple of float
        Magnetic field components in spherical coordinates (nT).
    """
    # Retrieve coefficients from registry
    coeffs = _coefficient_registry.get(coeff_id)
    if coeffs is None:
        raise ValueError(f"Coefficient set {coeff_id} not found in registry")

    return _compute_magnetic_field_spherical_impl(lat, lon, r, year, coeffs)


# Registry to hold coefficient sets by id
_coefficient_registry: dict[str, Any] = {}


def _register_coefficients(coeffs: "MagneticCoefficients") -> int:
    """Register a coefficient set and return its unique ID."""
    coeff_id = id(coeffs)
    if coeff_id not in _coefficient_registry:
        _coefficient_registry[coeff_id] = coeffs
    return coeff_id


def _compute_magnetic_field_spherical_impl(
    lat: float,
    lon: float,
    r: float,
    year: float,
    coeffs: "MagneticCoefficients",
) -> Tuple[float, float, float]:
    """
    Core implementation of magnetic field computation.

    This contains the actual spherical harmonic expansion logic,
    separated for clarity and to support caching.
    """
    n_max = coeffs.n_max
    a = 6371.2  # Reference radius in km (WMM convention)

    # Time adjustment
    dt = year - coeffs.epoch

    # Adjusted coefficients
    g = coeffs.g + dt * coeffs.g_dot
    h = coeffs.h + dt * coeffs.h_dot

    # Colatitude
    theta = np.pi / 2 - lat
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Compute associated Legendre functions (Schmidt semi-normalized)
    P = associated_legendre(n_max, n_max, cos_theta, normalized=True)

    # Compute dP/dtheta
    dP = np.zeros((n_max + 1, n_max + 1))
    if abs(sin_theta) > 1e-10:
        for n in range(1, n_max + 1):
            for m in range(n + 1):
                if m == n:
                    dP[n, m] = n * cos_theta / sin_theta * P[n, m]
                elif n > m:
                    factor = np.sqrt((n - m) * (n + m + 1))
                    if m + 1 <= n:
                        dP[n, m] = (
                            n * cos_theta / sin_theta * P[n, m]
                            - factor * P[n, m + 1] / sin_theta
                            if m + 1 <= n_max
                            else n * cos_theta / sin_theta * P[n, m]
                        )

    # Initialize field components
    B_r = 0.0
    B_theta = 0.0
    B_phi = 0.0

    # Sum over spherical harmonic degrees and orders
    r_ratio = a / r

    for n in range(1, n_max + 1):
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


# =============================================================================
# Cache Management
# =============================================================================


def get_magnetic_cache_info() -> dict[str, Any]:
    """
    Get information about the magnetic field computation cache.

    Returns
    -------
    info : dict
        Dictionary containing cache statistics:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - maxsize: Maximum cache size
        - currsize: Current number of cached entries
        - hit_rate: Ratio of hits to total calls (0-1)

    Examples
    --------
    >>> from pytcl.magnetism import get_magnetic_cache_info
    >>> info = get_magnetic_cache_info()
    >>> print(f"Cache hit rate: {info['hit_rate']:.1%}")
    """
    cache_info = _magnetic_field_spherical_cached.cache_info()
    total = cache_info.hits + cache_info.misses
    hit_rate = cache_info.hits / total if total > 0 else 0.0

    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "hit_rate": hit_rate,
    }


def clear_magnetic_cache() -> None:
    """
    Clear the magnetic field computation cache.

    This can be useful when memory is constrained or when switching
    between different coefficient sets.

    Examples
    --------
    >>> from pytcl.magnetism import clear_magnetic_cache
    >>> clear_magnetic_cache()  # Free cached computations
    """
    _magnetic_field_spherical_cached.cache_clear()
    _coefficient_registry.clear()


def configure_magnetic_cache(
    maxsize: Optional[int] = None,
    precision: Optional[dict[str, Any]] = None,
) -> None:
    """
    Configure the magnetic field computation cache.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of entries in the cache. If None, keeps current.
        Set to 0 to disable caching.
    precision : dict, optional
        Dictionary with keys 'lat', 'lon', 'r', 'year' specifying
        decimal places for rounding. Higher values = more precision
        but fewer cache hits.

    Notes
    -----
    Changing cache configuration clears the existing cache.

    Examples
    --------
    >>> from pytcl.magnetism import configure_magnetic_cache
    >>> # Increase cache size for batch processing
    >>> configure_magnetic_cache(maxsize=4096)
    >>> # Reduce precision for more cache hits
    >>> configure_magnetic_cache(precision={'lat': 4, 'lon': 4, 'r': 2, 'year': 1})
    """
    global _magnetic_field_spherical_cached

    if precision is not None:
        for key in ["lat", "lon", "r", "year"]:
            if key in precision:
                _CACHE_PRECISION[key] = precision[key]

    if maxsize is not None:
        # Recreate the cached function with new maxsize
        clear_magnetic_cache()

        @lru_cache(maxsize=maxsize)
        def new_cached(
            lat: float,
            lon: float,
            r: float,
            year: float,
            n_max: int,
            coeff_id: int,
        ) -> Tuple[float, float, float]:
            coeffs = _coefficient_registry.get(coeff_id)
            if coeffs is None:
                raise ValueError(f"Coefficient set {coeff_id} not found")
            return _compute_magnetic_field_spherical_impl(lat, lon, r, year, coeffs)

        _magnetic_field_spherical_cached = new_cached


def magnetic_field_spherical(
    lat: float,
    lon: float,
    r: float,
    year: float,
    coeffs: MagneticCoefficients = WMM2020,
    use_cache: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute magnetic field in spherical coordinates.

    Parameters
    ----------
    lat : float
        Geocentric latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radial distance from Earth's center in km.
    year : float
        Decimal year (e.g., 2023.5 for mid-2023).
    coeffs : MagneticCoefficients, optional
        Model coefficients. Default WMM2020.
    use_cache : bool, optional
        Whether to use LRU caching for repeated queries. Default True.
        Set to False for single-use queries or when memory is constrained.

    Returns
    -------
    B_r : float
        Radial component (positive outward) in nT.
    B_theta : float
        Colatitude component (positive southward) in nT.
    B_phi : float
        Longitude component (positive eastward) in nT.

    Notes
    -----
    Results are cached by default using LRU caching. Inputs are quantized
    to a configurable precision before caching to improve hit rates for
    nearby queries. Use `get_magnetic_cache_info()` to check cache
    statistics and `clear_magnetic_cache()` to free memory.
    """
    if use_cache:
        # Quantize inputs for cache key
        q_lat, q_lon, q_r, q_year = _quantize_inputs(lat, lon, r, year)

        # Register coefficients and get ID
        coeff_id = _register_coefficients(coeffs)

        return _magnetic_field_spherical_cached(
            q_lat, q_lon, q_r, q_year, coeffs.n_max, coeff_id
        )
    else:
        # Direct computation without caching
        return _compute_magnetic_field_spherical_impl(lat, lon, r, year, coeffs)


def wmm(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
    coeffs: MagneticCoefficients = WMM2020,
) -> MagneticResult:
    """
    Compute magnetic field using World Magnetic Model.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above WGS84 ellipsoid in km. Default 0.
    year : float, optional
        Decimal year. Default 2023.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients. Default WMM2020.

    Returns
    -------
    result : MagneticResult
        Magnetic field components and derived quantities.

    Examples
    --------
    >>> import numpy as np
    >>> result = wmm(np.radians(40), np.radians(-105), 1.0, 2023.0)
    >>> print(f"Declination: {np.degrees(result.D):.2f}°")
    >>> print(f"Inclination: {np.degrees(result.I):.2f}°")
    >>> print(f"Total intensity: {result.F:.0f} nT")
    """
    # Convert geodetic to geocentric
    # Simplified: assume spherical Earth for radius calculation
    a = 6371.2  # km

    # Geocentric latitude (approximate, ignoring ellipticity for simplicity)
    lat_gc = lat
    r = a + h

    # Compute field in spherical coordinates
    B_r, B_theta, B_phi = magnetic_field_spherical(lat_gc, lon, r, year, coeffs)

    # Convert to geodetic coordinates (X, Y, Z)
    # X = North, Y = East, Z = Down
    # For spherical approximation:
    # X = -B_theta (theta increases southward)
    # Y = B_phi
    # Z = -B_r (r increases outward, Z positive down)

    X = -B_theta
    Y = B_phi
    Z = -B_r

    # Derived quantities
    H = np.sqrt(X * X + Y * Y)  # Horizontal intensity
    F = np.sqrt(H * H + Z * Z)  # Total intensity

    # Inclination (dip angle)
    incl = np.arctan2(Z, H)

    # Declination
    D = np.arctan2(Y, X)

    return MagneticResult(
        X=X,
        Y=Y,
        Z=Z,
        H=H,
        F=F,
        I=incl,
        D=D,
    )


def magnetic_declination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
    coeffs: MagneticCoefficients = WMM2020,
) -> float:
    """
    Compute magnetic declination (variation).

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above ellipsoid in km. Default 0.
    year : float, optional
        Decimal year. Default 2023.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients.

    Returns
    -------
    D : float
        Magnetic declination in radians.
        Positive = east of true north.
        Negative = west of true north.

    Examples
    --------
    >>> import numpy as np
    >>> # Declination in Denver, CO
    >>> D = magnetic_declination(np.radians(39.7), np.radians(-105.0))
    >>> print(f"Declination: {np.degrees(D):.1f}°")
    """
    result = wmm(lat, lon, h, year, coeffs)
    return result.D


def magnetic_inclination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
    coeffs: MagneticCoefficients = WMM2020,
) -> float:
    """
    Compute magnetic inclination (dip angle).

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above ellipsoid in km. Default 0.
    year : float, optional
        Decimal year. Default 2023.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients.

    Returns
    -------
    I : float
        Magnetic inclination in radians.
        Positive = field points into Earth (Northern hemisphere).
        Negative = field points out of Earth (Southern hemisphere).
    """
    result = wmm(lat, lon, h, year, coeffs)
    return result.I


def magnetic_field_intensity(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
    coeffs: MagneticCoefficients = WMM2020,
) -> float:
    """
    Compute total magnetic field intensity.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height above ellipsoid in km. Default 0.
    year : float, optional
        Decimal year. Default 2023.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients.

    Returns
    -------
    F : float
        Total magnetic field intensity in nT.
    """
    result = wmm(lat, lon, h, year, coeffs)
    return result.F


__all__ = [
    "MagneticResult",
    "MagneticCoefficients",
    "WMM2020",
    "create_wmm2020_coefficients",
    "magnetic_field_spherical",
    "wmm",
    "magnetic_declination",
    "magnetic_inclination",
    "magnetic_field_intensity",
    # Cache management
    "get_magnetic_cache_info",
    "clear_magnetic_cache",
    "configure_magnetic_cache",
]
