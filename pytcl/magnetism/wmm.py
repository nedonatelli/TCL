"""
World Magnetic Model (WMM) implementation.

The WMM is the standard model used by the U.S. Department of Defense,
the U.K. Ministry of Defense, NATO, and the International Hydrographic
Organization for navigation, attitude, and heading referencing.

References
----------
- Chulliat et al., "The US/UK World Magnetic Model for 2020-2025,"
  NOAA Technical Report, 2020.
- https://www.ngdc.noaa.gov/geomag/WMM/
"""

from functools import lru_cache
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.gravity.spherical_harmonics import (
    associated_legendre,
    associated_legendre_derivative,
)

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
    I: float  # noqa: E741
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


# Official WMM2020 coefficients (NOAA/NCEI WMM.COF, epoch 2020.0):
# columns are n, m, g (nT), h (nT), g_dot (nT/yr), h_dot (nT/yr)
_WMM2020_COF = """\
  1  0  -29404.5       0.0        6.7        0.0
  1  1   -1450.7    4652.9        7.7      -25.1
  2  0   -2500.0       0.0      -11.5        0.0
  2  1    2982.0   -2991.6       -7.1      -30.2
  2  2    1676.8    -734.8       -2.2      -23.9
  3  0    1363.9       0.0        2.8        0.0
  3  1   -2381.0     -82.2       -6.2        5.7
  3  2    1236.2     241.8        3.4       -1.0
  3  3     525.7    -542.9      -12.2        1.1
  4  0     903.1       0.0       -1.1        0.0
  4  1     809.4     282.0       -1.6        0.2
  4  2      86.2    -158.4       -6.0        6.9
  4  3    -309.4     199.8        5.4        3.7
  4  4      47.9    -350.1       -5.5       -5.6
  5  0    -234.4       0.0       -0.3        0.0
  5  1     363.1      47.7        0.6        0.1
  5  2     187.8     208.4       -0.7        2.5
  5  3    -140.7    -121.3        0.1       -0.9
  5  4    -151.2      32.2        1.2        3.0
  5  5      13.7      99.1        1.0        0.5
  6  0      65.9       0.0       -0.6        0.0
  6  1      65.6     -19.1       -0.4        0.1
  6  2      73.0      25.0        0.5       -1.8
  6  3    -121.5      52.7        1.4       -1.4
  6  4     -36.2     -64.4       -1.4        0.9
  6  5      13.5       9.0       -0.0        0.1
  6  6     -64.7      68.1        0.8        1.0
  7  0      80.6       0.0       -0.1        0.0
  7  1     -76.8     -51.4       -0.3        0.5
  7  2      -8.3     -16.8       -0.1        0.6
  7  3      56.5       2.3        0.7       -0.7
  7  4      15.8      23.5        0.2       -0.2
  7  5       6.4      -2.2       -0.5       -1.2
  7  6      -7.2     -27.2       -0.8        0.2
  7  7       9.8      -1.9        1.0        0.3
  8  0      23.6       0.0       -0.1        0.0
  8  1       9.8       8.4        0.1       -0.3
  8  2     -17.5     -15.3       -0.1        0.7
  8  3      -0.4      12.8        0.5       -0.2
  8  4     -21.1     -11.8       -0.1        0.5
  8  5      15.3      14.9        0.4       -0.3
  8  6      13.7       3.6        0.5       -0.5
  8  7     -16.5      -6.9        0.0        0.4
  8  8      -0.3       2.8        0.4        0.1
  9  0       5.0       0.0       -0.1        0.0
  9  1       8.2     -23.3       -0.2       -0.3
  9  2       2.9      11.1       -0.0        0.2
  9  3      -1.4       9.8        0.4       -0.4
  9  4      -1.1      -5.1       -0.3        0.4
  9  5     -13.3      -6.2       -0.0        0.1
  9  6       1.1       7.8        0.3       -0.0
  9  7       8.9       0.4       -0.0       -0.2
  9  8      -9.3      -1.5       -0.0        0.5
  9  9     -11.9       9.7       -0.4        0.2
 10  0      -1.9       0.0        0.0        0.0
 10  1      -6.2       3.4       -0.0       -0.0
 10  2      -0.1      -0.2       -0.0        0.1
 10  3       1.7       3.5        0.2       -0.3
 10  4      -0.9       4.8       -0.1        0.1
 10  5       0.6      -8.6       -0.2       -0.2
 10  6      -0.9      -0.1       -0.0        0.1
 10  7       1.9      -4.2       -0.1       -0.0
 10  8       1.4      -3.4       -0.2       -0.1
 10  9      -2.4      -0.1       -0.1        0.2
 10 10      -3.9      -8.8       -0.0       -0.0
 11  0       3.0       0.0       -0.0        0.0
 11  1      -1.4      -0.0       -0.1       -0.0
 11  2      -2.5       2.6       -0.0        0.1
 11  3       2.4      -0.5        0.0        0.0
 11  4      -0.9      -0.4       -0.0        0.2
 11  5       0.3       0.6       -0.1       -0.0
 11  6      -0.7      -0.2        0.0        0.0
 11  7      -0.1      -1.7       -0.0        0.1
 11  8       1.4      -1.6       -0.1       -0.0
 11  9      -0.6      -3.0       -0.1       -0.1
 11 10       0.2      -2.0       -0.1        0.0
 11 11       3.1      -2.6       -0.1       -0.0
 12  0      -2.0       0.0        0.0        0.0
 12  1      -0.1      -1.2       -0.0       -0.0
 12  2       0.5       0.5       -0.0        0.0
 12  3       1.3       1.3        0.0       -0.1
 12  4      -1.2      -1.8       -0.0        0.1
 12  5       0.7       0.1       -0.0       -0.0
 12  6       0.3       0.7        0.0        0.0
 12  7       0.5      -0.1       -0.0       -0.0
 12  8      -0.2       0.6        0.0        0.1
 12  9      -0.5       0.2       -0.0       -0.0
 12 10       0.1      -0.9       -0.0       -0.0
 12 11      -1.1      -0.0       -0.0        0.0
 12 12      -0.3       0.5       -0.1       -0.1
"""


def create_wmm2020_coefficients() -> MagneticCoefficients:
    """
    Create WMM2020 model coefficients.

    Returns
    -------
    coeffs : MagneticCoefficients
        WMM2020 spherical harmonic coefficients.

    Examples
    --------
    >>> coeffs = create_wmm2020_coefficients()
    >>> coeffs.epoch
    2020.0
    >>> coeffs.n_max
    12

    Notes
    -----
    These are the official WMM2020 coefficients valid from 2020.0 to 2025.0,
    embedded verbatim from the NOAA/NCEI WMM.COF distribution file.
    For use beyond 2025, updated coefficients should be obtained from NOAA.
    """
    n_max = 12
    g = np.zeros((n_max + 1, n_max + 1))
    h = np.zeros((n_max + 1, n_max + 1))
    g_dot = np.zeros((n_max + 1, n_max + 1))
    h_dot = np.zeros((n_max + 1, n_max + 1))

    for line in _WMM2020_COF.strip().split("\n"):
        n_s, m_s, g_s, h_s, gd_s, hd_s = line.split()
        n, m = int(n_s), int(m_s)
        g[n, m] = float(g_s)
        h[n, m] = float(h_s)
        g_dot[n, m] = float(gd_s)
        h_dot[n, m] = float(hd_s)

    return MagneticCoefficients(
        g=g, h=h, g_dot=g_dot, h_dot=h_dot, epoch=2020.0, n_max=n_max
    )


# Official WMM2025 coefficients (NOAA/NCEI WMM.COF, epoch 2025.0):
# columns are n, m, g (nT), h (nT), g_dot (nT/yr), h_dot (nT/yr)
_WMM2025_COF = """\
  1  0  -29351.8       0.0       12.0        0.0
  1  1   -1410.8    4545.4        9.7      -21.5
  2  0   -2556.6       0.0      -11.6        0.0
  2  1    2951.1   -3133.6       -5.2      -27.7
  2  2    1649.3    -815.1       -8.0      -12.1
  3  0    1361.0       0.0       -1.3        0.0
  3  1   -2404.1     -56.6       -4.2        4.0
  3  2    1243.8     237.5        0.4       -0.3
  3  3     453.6    -549.5      -15.6       -4.1
  4  0     895.0       0.0       -1.6        0.0
  4  1     799.5     278.6       -2.4       -1.1
  4  2      55.7    -133.9       -6.0        4.1
  4  3    -281.1     212.0        5.6        1.6
  4  4      12.1    -375.6       -7.0       -4.4
  5  0    -233.2       0.0        0.6        0.0
  5  1     368.9      45.4        1.4       -0.5
  5  2     187.2     220.2        0.0        2.2
  5  3    -138.7    -122.9        0.6        0.4
  5  4    -142.0      43.0        2.2        1.7
  5  5      20.9     106.1        0.9        1.9
  6  0      64.4       0.0       -0.2        0.0
  6  1      63.8     -18.4       -0.4        0.3
  6  2      76.9      16.8        0.9       -1.6
  6  3    -115.7      48.8        1.2       -0.4
  6  4     -40.9     -59.8       -0.9        0.9
  6  5      14.9      10.9        0.3        0.7
  6  6     -60.7      72.7        0.9        0.9
  7  0      79.5       0.0       -0.0        0.0
  7  1     -77.0     -48.9       -0.1        0.6
  7  2      -8.8     -14.4       -0.1        0.5
  7  3      59.3      -1.0        0.5       -0.8
  7  4      15.8      23.4       -0.1        0.0
  7  5       2.5      -7.4       -0.8       -1.0
  7  6     -11.1     -25.1       -0.8        0.6
  7  7      14.2      -2.3        0.8       -0.2
  8  0      23.2       0.0       -0.1        0.0
  8  1      10.8       7.1        0.2       -0.2
  8  2     -17.5     -12.6        0.0        0.5
  8  3       2.0      11.4        0.5       -0.4
  8  4     -21.7      -9.7       -0.1        0.4
  8  5      16.9      12.7        0.3       -0.5
  8  6      15.0       0.7        0.2       -0.6
  8  7     -16.8      -5.2       -0.0        0.3
  8  8       0.9       3.9        0.2        0.2
  9  0       4.6       0.0       -0.0        0.0
  9  1       7.8     -24.8       -0.1       -0.3
  9  2       3.0      12.2        0.1        0.3
  9  3      -0.2       8.3        0.3       -0.3
  9  4      -2.5      -3.3       -0.3        0.3
  9  5     -13.1      -5.2        0.0        0.2
  9  6       2.4       7.2        0.3       -0.1
  9  7       8.6      -0.6       -0.1       -0.2
  9  8      -8.7       0.8        0.1        0.4
  9  9     -12.9      10.0       -0.1        0.1
 10  0      -1.3       0.0        0.1        0.0
 10  1      -6.4       3.3        0.0        0.0
 10  2       0.2       0.0        0.1       -0.0
 10  3       2.0       2.4        0.1       -0.2
 10  4      -1.0       5.3       -0.0        0.1
 10  5      -0.6      -9.1       -0.3       -0.1
 10  6      -0.9       0.4        0.0        0.1
 10  7       1.5      -4.2       -0.1        0.0
 10  8       0.9      -3.8       -0.1       -0.1
 10  9      -2.7       0.9       -0.0        0.2
 10 10      -3.9      -9.1       -0.0       -0.0
 11  0       2.9       0.0        0.0        0.0
 11  1      -1.5       0.0       -0.0       -0.0
 11  2      -2.5       2.9        0.0        0.1
 11  3       2.4      -0.6        0.0       -0.0
 11  4      -0.6       0.2        0.0        0.1
 11  5      -0.1       0.5       -0.1       -0.0
 11  6      -0.6      -0.3        0.0       -0.0
 11  7      -0.1      -1.2       -0.0        0.1
 11  8       1.1      -1.7       -0.1       -0.0
 11  9      -1.0      -2.9       -0.1        0.0
 11 10      -0.2      -1.8       -0.1        0.0
 11 11       2.6      -2.3       -0.1        0.0
 12  0      -2.0       0.0        0.0        0.0
 12  1      -0.2      -1.3        0.0       -0.0
 12  2       0.3       0.7       -0.0        0.0
 12  3       1.2       1.0       -0.0       -0.1
 12  4      -1.3      -1.4       -0.0        0.1
 12  5       0.6      -0.0       -0.0       -0.0
 12  6       0.6       0.6        0.1       -0.0
 12  7       0.5      -0.1       -0.0       -0.0
 12  8      -0.1       0.8        0.0        0.0
 12  9      -0.4       0.1        0.0       -0.0
 12 10      -0.2      -1.0       -0.1       -0.0
 12 11      -1.3       0.1       -0.0        0.0
 12 12      -0.7       0.2       -0.1       -0.1
"""


def create_wmm2025_coefficients() -> MagneticCoefficients:
    """
    Create WMM2025 model coefficients.

    Returns
    -------
    coeffs : MagneticCoefficients
        WMM2025 spherical harmonic coefficients.

    Examples
    --------
    >>> coeffs = create_wmm2025_coefficients()
    >>> coeffs.epoch
    2025.0
    >>> coeffs.n_max
    12

    Notes
    -----
    These are the official WMM2025 coefficients valid from 2025.0 to 2030.0,
    embedded verbatim from the NOAA/NCEI WMM.COF distribution file.
    """
    n_max = 12
    g = np.zeros((n_max + 1, n_max + 1))
    h = np.zeros((n_max + 1, n_max + 1))
    g_dot = np.zeros((n_max + 1, n_max + 1))
    h_dot = np.zeros((n_max + 1, n_max + 1))

    for line in _WMM2025_COF.strip().split("\n"):
        n_s, m_s, g_s, h_s, gd_s, hd_s = line.split()
        n, m = int(n_s), int(m_s)
        g[n, m] = float(g_s)
        h[n, m] = float(h_s)
        g_dot[n, m] = float(gd_s)
        h_dot[n, m] = float(hd_s)

    return MagneticCoefficients(
        g=g, h=h, g_dot=g_dot, h_dot=h_dot, epoch=2025.0, n_max=n_max
    )


WMM2020 = create_wmm2020_coefficients()
WMM2025 = create_wmm2025_coefficients()


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
_coefficient_registry: dict[int, Any] = {}


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

    # WMM Gauss coefficients require Schmidt semi-normalized Legendre
    # functions: P_schmidt = P_full / sqrt(2n+1). Derive both P and
    # dP/dtheta from the fully normalized implementations, using
    # dP/dtheta = -sin(theta) * dP/dx with x = cos(theta).
    P_full = associated_legendre(n_max, n_max, cos_theta, normalized=True)
    dP_full = associated_legendre_derivative(
        n_max, n_max, cos_theta, P_full, normalized=True
    )
    # associated_legendre is geodesy fully normalized:
    # sqrt((2 - delta_0m)(2n+1)(n-m)!/(n+m)!). Schmidt semi-normalization
    # keeps the sqrt(2 - delta_0m) factor, so Schmidt = full / sqrt(2n+1).
    scale = np.ones((n_max + 1, n_max + 1))
    scale /= np.sqrt(2 * np.arange(n_max + 1) + 1)[:, np.newaxis]
    P = P_full * scale
    dP = -sin_theta * dP_full * scale

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
    >>> 0.0 <= info['hit_rate'] <= 1.0
    True
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
    >>> # Restore the default precision
    >>> configure_magnetic_cache(precision={'lat': 6, 'lon': 6, 'r': 3, 'year': 2})
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
    coeffs: MagneticCoefficients = WMM2025,
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
        Model coefficients. Default WMM2025.
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

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import magnetic_field_spherical
    >>> # Compute at Earth's surface (40°N, 105°W, sea level)
    >>> lat = np.radians(40)
    >>> lon = np.radians(-105)
    >>> r = 6371.2  # Earth mean radius in km
    >>> B_r, B_theta, B_phi = magnetic_field_spherical(lat, lon, r, 2023.0)
    >>> # All components should be on order of tens to tens of thousands of nT
    >>> bool(20000 < (B_r**2 + B_theta**2 + B_phi**2)**0.5 < 70000)
    True
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
    year: float = 2025.0,
    coeffs: MagneticCoefficients = WMM2025,
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
        Decimal year. Default 2025.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients. Default WMM2025.

    Returns
    -------
    result : MagneticResult
        Magnetic field components and derived quantities.

    Examples
    --------
    >>> import numpy as np
    >>> result = wmm(np.radians(40), np.radians(-105), 1.0, 2023.0)
    >>> print(f"Declination: {np.degrees(result.D):.2f}°")
    Declination: 7.83°
    >>> print(f"Inclination: {np.degrees(result.I):.2f}°")
    Inclination: 66.23°
    >>> print(f"Total intensity: {result.F:.0f} nT")
    Total intensity: 51573 nT
    """
    # Convert geodetic (WGS84) to geocentric spherical coordinates
    a_wgs = 6378.137  # WGS84 semi-major axis, km
    e2 = 6.694379990141e-3  # WGS84 first eccentricity squared

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    rc = a_wgs / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    p = (rc + h) * cos_lat
    z = (rc * (1.0 - e2) + h) * sin_lat
    r = np.sqrt(p * p + z * z)
    lat_gc = np.arcsin(z / r)

    # Compute field in geocentric spherical coordinates
    B_r, B_theta, B_phi = magnetic_field_spherical(lat_gc, lon, r, year, coeffs)

    # Geocentric NED components
    X_gc = -B_theta  # theta increases southward
    Y = B_phi
    Z_gc = -B_r  # r increases outward, Z positive down

    # Rotate from geocentric to geodetic frame (WMM report, eq. 17)
    psi = lat_gc - lat
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    X = X_gc * cos_psi - Z_gc * sin_psi
    Z = X_gc * sin_psi + Z_gc * cos_psi

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
    year: float = 2025.0,
    coeffs: MagneticCoefficients = WMM2025,
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
        Decimal year. Default 2025.0.
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
    >>> # Declination in Denver, CO (easterly, i.e. positive)
    >>> D = magnetic_declination(np.radians(39.7), np.radians(-105.0))
    >>> print(f"Declination: {np.degrees(D):.1f}°")
    Declination: 7.6°
    """
    result = wmm(lat, lon, h, year, coeffs)
    return result.D


def magnetic_inclination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2025.0,
    coeffs: MagneticCoefficients = WMM2025,
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
        Decimal year. Default 2025.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients.

    Returns
    -------
    I : float
        Magnetic inclination in radians.
        Positive = field points into Earth (Northern hemisphere).
        Negative = field points out of Earth (Southern hemisphere).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import magnetic_inclination
    >>> # Inclination at 40°N, 105°W (Denver)
    >>> lat = np.radians(40)
    >>> lon = np.radians(-105)
    >>> I = magnetic_inclination(lat, lon, 1.6, 2023.0)
    >>> # Northern hemisphere: inclination should be positive
    >>> bool(I > 0)
    True
    >>> # Typical values in US are 50-70 degrees
    >>> bool(0.8 < I < 1.3)  # ~46-74 degrees
    True
    """
    result = wmm(lat, lon, h, year, coeffs)
    return result.I


def magnetic_field_intensity(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2025.0,
    coeffs: MagneticCoefficients = WMM2025,
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
        Decimal year. Default 2025.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients.

    Returns
    -------
    F : float
        Total magnetic field intensity in nT.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import magnetic_field_intensity
    >>> # Field intensity at magnetic equator vs pole
    >>> F_eq = magnetic_field_intensity(0, 0, 0, 2023.0)  # Equator
    >>> F_pole = magnetic_field_intensity(np.radians(80), 0, 0, 2023.0)  # Near pole
    >>> # Field is stronger at poles
    >>> bool(F_pole > F_eq)
    True
    >>> # Typical Earth field is 25,000 to 65,000 nT
    >>> bool(25000 < F_eq < 35000)  # Equatorial field is weaker
    True
    >>> bool(55000 < F_pole < 65000)  # Polar field is stronger
    True
    """
    result = wmm(lat, lon, h, year, coeffs)
    return result.F


__all__ = [
    "MagneticResult",
    "MagneticCoefficients",
    "WMM2020",
    "WMM2025",
    "create_wmm2020_coefficients",
    "create_wmm2025_coefficients",
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
