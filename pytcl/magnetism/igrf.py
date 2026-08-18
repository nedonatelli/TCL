"""
International Geomagnetic Reference Field (IGRF) implementation.

The IGRF is a standard mathematical description of the Earth's main
magnetic field, used widely in studies of the Earth's interior,
its ionosphere and magnetosphere, and in various applications.

References
----------
- Alken et al., "International Geomagnetic Reference Field: the
  thirteenth generation," Earth, Planets and Space, 2021.
- https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
"""

from typing import NamedTuple, Optional

import numpy as np

from pytcl.magnetism.wmm import (
    MagneticCoefficients,
    MagneticResult,
    wmm,
)


class IGRFModel(NamedTuple):
    """IGRF model for a specific epoch.

    Attributes
    ----------
    epoch : float
        Reference epoch year.
    coeffs : MagneticCoefficients
        Spherical harmonic coefficients.
    valid_from : float
        Start of validity period.
    valid_to : float
        End of validity period.
    """

    epoch: float
    coeffs: MagneticCoefficients
    valid_from: float
    valid_to: float


# Official IGRF-13 coefficients for epoch 2020.0 with 2020-2025 secular
# variation, extracted from the IAGA igrf13coeffs.txt distribution file:
# columns are n, m, g (nT), h (nT), g_dot (nT/yr), h_dot (nT/yr)
_IGRF13_2020_COF = """\
1  0  -29404.8       0.0      5.7      0.0
  1  1   -1450.9    4652.5      7.4    -25.9
  2  0   -2499.6       0.0    -11.0      0.0
  2  1    2982.0   -2991.6     -7.0    -30.2
  2  2    1677.0    -734.6     -2.1    -22.4
  3  0    1363.2       0.0      2.2      0.0
  3  1   -2381.2     -82.1     -5.9      6.0
  3  2    1236.2     241.9      3.1     -1.1
  3  3     525.7    -543.4    -12.0      0.5
  4  0     903.0       0.0     -1.2      0.0
  4  1     809.5     281.9     -1.6     -0.1
  4  2      86.3    -158.4     -5.9      6.5
  4  3    -309.4     199.7      5.2      3.6
  4  4      48.0    -349.7     -5.1     -5.0
  5  0    -234.3       0.0     -0.3      0.0
  5  1     363.2      47.7      0.5      0.0
  5  2     187.8     208.3     -0.6      2.5
  5  3    -140.7    -121.2      0.2     -0.6
  5  4    -151.2      32.3      1.3      3.0
  5  5      13.5      98.9      0.9      0.3
  6  0      66.0       0.0     -0.5      0.0
  6  1      65.5     -19.1     -0.3      0.0
  6  2      72.9      25.1      0.4     -1.6
  6  3    -121.5      52.8      1.3     -1.3
  6  4     -36.2     -64.5     -1.4      0.8
  6  5      13.5       8.9      0.0      0.0
  6  6     -64.7      68.1      0.9      1.0
  7  0      80.6       0.0     -0.1      0.0
  7  1     -76.7     -51.5     -0.2      0.6
  7  2      -8.2     -16.9      0.0      0.6
  7  3      56.5       2.2      0.7     -0.8
  7  4      15.8      23.5      0.1     -0.2
  7  5       6.4      -2.2     -0.5     -1.1
  7  6      -7.2     -27.2     -0.8      0.1
  7  7       9.8      -1.8      0.8      0.3
  8  0      23.7       0.0      0.0      0.0
  8  1       9.7       8.4      0.1     -0.2
  8  2     -17.6     -15.3     -0.1      0.6
  8  3      -0.5      12.8      0.4     -0.2
  8  4     -21.1     -11.7     -0.1      0.5
  8  5      15.3      14.9      0.4     -0.3
  8  6      13.7       3.6      0.3     -0.4
  8  7     -16.5      -6.9     -0.1      0.5
  8  8      -0.3       2.8      0.4      0.0
  9  0       5.0       0.0      0.0      0.0
  9  1       8.4     -23.4      0.0      0.0
  9  2       2.9      11.0      0.0      0.0
  9  3      -1.5       9.8      0.0      0.0
  9  4      -1.1      -5.1      0.0      0.0
  9  5     -13.2      -6.3      0.0      0.0
  9  6       1.1       7.8      0.0      0.0
  9  7       8.8       0.4      0.0      0.0
  9  8      -9.3      -1.4      0.0      0.0
  9  9     -11.9       9.6      0.0      0.0
 10  0      -1.9       0.0      0.0      0.0
 10  1      -6.2       3.4      0.0      0.0
 10  2      -0.1      -0.2      0.0      0.0
 10  3       1.7       3.6      0.0      0.0
 10  4      -0.9       4.8      0.0      0.0
 10  5       0.7      -8.6      0.0      0.0
 10  6      -0.9      -0.1      0.0      0.0
 10  7       1.9      -4.3      0.0      0.0
 10  8       1.4      -3.4      0.0      0.0
 10  9      -2.4      -0.1      0.0      0.0
 10 10      -3.8      -8.8      0.0      0.0
 11  0       3.0       0.0      0.0      0.0
 11  1      -1.4       0.0      0.0      0.0
 11  2      -2.5       2.5      0.0      0.0
 11  3       2.3      -0.6      0.0      0.0
 11  4      -0.9      -0.4      0.0      0.0
 11  5       0.3       0.6      0.0      0.0
 11  6      -0.7      -0.2      0.0      0.0
 11  7      -0.1      -1.7      0.0      0.0
 11  8       1.4      -1.6      0.0      0.0
 11  9      -0.6      -3.0      0.0      0.0
 11 10       0.2      -2.0      0.0      0.0
 11 11       3.1      -2.6      0.0      0.0
 12  0      -2.0       0.0      0.0      0.0
 12  1      -0.1      -1.2      0.0      0.0
 12  2       0.5       0.5      0.0      0.0
 12  3       1.3       1.4      0.0      0.0
 12  4      -1.2      -1.8      0.0      0.0
 12  5       0.7       0.1      0.0      0.0
 12  6       0.3       0.8      0.0      0.0
 12  7       0.5      -0.2      0.0      0.0
 12  8      -0.3       0.6      0.0      0.0
 12  9      -0.5       0.2      0.0      0.0
 12 10       0.1      -0.9      0.0      0.0
 12 11      -1.1       0.0      0.0      0.0
 12 12      -0.3       0.5      0.0      0.0
 13  0       0.1       0.0      0.0      0.0
 13  1      -0.9      -0.9      0.0      0.0
 13  2       0.5       0.6      0.0      0.0
 13  3       0.7       1.4      0.0      0.0
 13  4      -0.3      -0.4      0.0      0.0
 13  5       0.8      -1.3      0.0      0.0
 13  6       0.0      -0.1      0.0      0.0
 13  7       0.8       0.3      0.0      0.0
 13  8       0.0      -0.1      0.0      0.0
 13  9       0.4       0.5      0.0      0.0
 13 10       0.1       0.5      0.0      0.0
 13 11       0.5      -0.4      0.0      0.0
 13 12      -0.5      -0.4      0.0      0.0
 13 13      -0.4      -0.6      0.0      0.0
"""


def create_igrf13_coefficients() -> MagneticCoefficients:
    """
    Create IGRF-13 model coefficients for epoch 2020.

    Returns
    -------
    coeffs : MagneticCoefficients
        IGRF-13 spherical harmonic coefficients (epoch 2020.0, with
        2020-2025 secular variation), embedded verbatim from the official
        IAGA igrf13coeffs.txt distribution.

    Examples
    --------
    >>> coeffs = create_igrf13_coefficients()
    >>> coeffs.epoch
    2020.0
    >>> coeffs.n_max
    13

    Notes
    -----
    IGRF-13 is valid from 1900.0 to 2025.0. This function returns the
    coefficients for the 2020.0 epoch; for earlier epochs the historical
    tables should be interpolated.

    IGRF-13's validity window ends 2025.0 and no IGRF-14 release is
    embedded in this library yet. Calls with ``year`` beyond 2025.0
    extrapolate the 2020-2025 secular variation (``g_dot``/``h_dot``)
    linearly past the model's official range -- treat results for
    ``year > 2025.0`` as an unofficial extrapolation, not IGRF-13 output.
    """
    n_max = 13
    g = np.zeros((n_max + 1, n_max + 1))
    h = np.zeros((n_max + 1, n_max + 1))
    g_dot = np.zeros((n_max + 1, n_max + 1))
    h_dot = np.zeros((n_max + 1, n_max + 1))

    for line in _IGRF13_2020_COF.strip().split("\n"):
        n_s, m_s, g_s, h_s, gd_s, hd_s = line.split()
        n, m = int(n_s), int(m_s)
        g[n, m] = float(g_s)
        h[n, m] = float(h_s)
        g_dot[n, m] = float(gd_s)
        h_dot[n, m] = float(hd_s)

    return MagneticCoefficients(
        g=g, h=h, g_dot=g_dot, h_dot=h_dot, epoch=2020.0, n_max=n_max
    )


IGRF13 = create_igrf13_coefficients()


def igrf(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
    coeffs: Optional[MagneticCoefficients] = None,
) -> MagneticResult:
    """
    Compute magnetic field using IGRF model.

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
        Model coefficients. Default IGRF13.

    Returns
    -------
    result : MagneticResult
        Magnetic field components and derived quantities.

    Examples
    --------
    >>> import numpy as np
    >>> result = igrf(np.radians(45), np.radians(-75), 0, 2023.0)
    >>> print(f"Total field: {result.F:.0f} nT")
    Total field: 53370 nT
    """
    if coeffs is None:
        coeffs = IGRF13

    # Same geodetic evaluation as the WMM (WGS84 geodetic-to-geocentric
    # conversion, Schmidt-normalized synthesis, frame rotation)
    return wmm(lat, lon, h, year, coeffs)


def igrf_declination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
) -> float:
    """
    Compute magnetic declination using IGRF.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height in km. Default 0.
    year : float, optional
        Decimal year. Default 2023.0.

    Returns
    -------
    D : float
        Declination in radians.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import igrf_declination
    >>> # Declination at Denver (40°N, 105°W)
    >>> lat = np.radians(40)
    >>> lon = np.radians(-105)
    >>> D = igrf_declination(lat, lon, 1.6, 2023.0)
    >>> # Denver lies west of the agonic line: easterly declination (~8° E)
    >>> print(f"Declination: {np.degrees(D):.1f}°")
    Declination: 7.8°
    >>> bool(0 < D < 0.35)  # East is positive
    True
    """
    return igrf(lat, lon, h, year).D


def igrf_inclination(
    lat: float,
    lon: float,
    h: float = 0.0,
    year: float = 2023.0,
) -> float:
    """
    Compute magnetic inclination using IGRF.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h : float, optional
        Height in km. Default 0.
    year : float, optional
        Decimal year. Default 2023.0.

    Returns
    -------
    I : float
        Inclination in radians.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import igrf_inclination
    >>> # Inclination comparison: equator vs pole
    >>> I_eq = igrf_inclination(0, 0, 0, 2023.0)  # Equator
    >>> I_pole = igrf_inclination(np.radians(85), 0, 0, 2023.0)  # Near pole
    >>> # The dip equator is offset from the geographic equator; at (0°N, 0°E)
    >>> # in the Gulf of Guinea the inclination is moderately negative (~-30°)
    >>> bool(-0.6 < I_eq < 0)
    True
    >>> bool(abs(I_pole) > 1.4)  # ~80+ degrees near the pole
    True
    """
    return igrf(lat, lon, h, year).I


def dipole_moment(coeffs: MagneticCoefficients = IGRF13) -> float:
    """
    Compute the centered dipole moment.

    Parameters
    ----------
    coeffs : MagneticCoefficients, optional
        Model coefficients. Default IGRF13.

    Returns
    -------
    M : float
        Dipole moment in nT * km^3.

    Notes
    -----
    The dipole moment is computed from the n=1 Gauss coefficients:
    M = a^3 * sqrt(g10^2 + g11^2 + h11^2)

    Examples
    --------
    >>> from pytcl.magnetism import dipole_moment, IGRF13
    >>> # Compute Earth's dipole moment from IGRF-13
    >>> M = dipole_moment(IGRF13)
    >>> # Earth's dipole moment is approximately 7.9 × 10^22 A·m²
    >>> # In nT·km³ units, this is about 7.9 × 10^15
    >>> 7e15 < M < 8.5e15
    True
    """
    a = 6371.2  # Reference radius in km
    g10 = coeffs.g[1, 0]
    g11 = coeffs.g[1, 1]
    h11 = coeffs.h[1, 1]

    M = a**3 * np.sqrt(g10**2 + g11**2 + h11**2)
    return M


def dipole_axis(
    coeffs: MagneticCoefficients = IGRF13,
) -> tuple[float, float]:
    """
    Compute the geocentric dipole axis direction.

    Parameters
    ----------
    coeffs : MagneticCoefficients, optional
        Model coefficients. Default IGRF13.

    Returns
    -------
    lat : float
        Latitude of the north geomagnetic pole in radians.
    lon : float
        Longitude of the north geomagnetic pole in radians.

    Notes
    -----
    The geomagnetic pole is where the centered dipole axis
    intersects the Earth's surface.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import dipole_axis, IGRF13
    >>> # Compute geomagnetic pole location
    >>> lat, lon = dipole_axis(IGRF13)
    >>> # Geomagnetic north pole is around 80.6°N, 72.7°W
    >>> print(f"{np.degrees(lat):.2f}°N, {np.degrees(lon):.2f}°E")
    80.59°N, -72.68°E
    >>> bool(70 < np.degrees(lat) < 85)
    True
    >>> bool(-100 < np.degrees(lon) < -60)
    True
    """
    g10 = coeffs.g[1, 0]
    g11 = coeffs.g[1, 1]
    h11 = coeffs.h[1, 1]

    # The dipole moment vector is proportional to (g11, h11, g10); the
    # NORTH geomagnetic pole lies along its antipode (g10 < 0 for Earth)
    b0 = np.sqrt(g10**2 + g11**2 + h11**2)
    theta = np.arccos(-g10 / b0)
    phi = np.arctan2(-h11, -g11)

    # Convert colatitude to latitude
    lat = np.pi / 2 - theta

    return lat, phi


def magnetic_north_pole(
    year: float = 2023.0,
    coeffs: MagneticCoefficients = IGRF13,
) -> tuple[float, float]:
    """
    Compute the location of the magnetic north pole.

    The magnetic north pole is where the field is vertical
    (inclination = 90°). This differs from the geomagnetic pole
    due to non-dipole field contributions.

    Parameters
    ----------
    year : float, optional
        Decimal year. Default 2023.0.
    coeffs : MagneticCoefficients, optional
        Model coefficients. Default IGRF13.

    Returns
    -------
    lat : float
        Latitude of magnetic north pole in radians.
    lon : float
        Longitude of magnetic north pole in radians.

    Notes
    -----
    This uses an iterative search starting from the dipole pole.
    The magnetic pole moves over time.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.magnetism import magnetic_north_pole, dipole_axis
    >>> # Magnetic north pole location (2023)
    >>> lat, lon = magnetic_north_pole(2023.0)
    >>> # The pole has drifted past 86°N toward the Siberian side of
    >>> # the Arctic (east longitude), around 86°N, 146°E in 2023
    >>> bool(75 < np.degrees(lat) < 90)
    True
    >>> bool(130 < np.degrees(lon) < 180)
    True
    >>> # Compare with geomagnetic pole
    >>> geo_lat, geo_lon = dipole_axis()
    >>> # Magnetic pole differs from geomagnetic pole
    >>> bool(abs(lat - geo_lat) > 0.01)  # Different locations
    True
    """
    from scipy.optimize import minimize

    lat_cap = np.radians(89.9)

    def horizontal_intensity(x: np.ndarray) -> float:
        # Keep the search off the exact geographic pole, where the
        # synthesis's singularity guard makes H spuriously zero
        if abs(x[0]) > lat_cap:
            return 1e9
        return float(igrf(x[0], x[1], 0.0, year, coeffs).H)

    # Coarse grid over the Arctic to find the global basin (the H surface
    # has local minima that trap a single descent), then refine
    best = (np.inf, np.pi / 2, 0.0)
    for lat_deg in np.arange(78.0, 89.5, 1.0):
        for lon_deg in np.arange(-180.0, 180.0, 10.0):
            x = np.array([np.radians(lat_deg), np.radians(lon_deg)])
            val = horizontal_intensity(x)
            if val < best[0]:
                best = (val, x[0], x[1])

    result = minimize(
        horizontal_intensity,
        np.array([best[1], best[2]]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-3, "maxiter": 500},
    )
    lat, lon = float(result.x[0]), float(result.x[1])

    return lat, lon


__all__ = [
    "IGRFModel",
    "IGRF13",
    "create_igrf13_coefficients",
    "igrf",
    "igrf_declination",
    "igrf_inclination",
    "dipole_moment",
    "dipole_axis",
    "magnetic_north_pole",
]
