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

from typing import NamedTuple, Tuple
import numpy as np
from numpy.typing import NDArray

from pytcl.gravity.spherical_harmonics import associated_legendre


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

    # Secular variation (nT/year) - selected terms
    g_dot[1, 0] = 6.7
    g_dot[1, 1] = 7.7
    h_dot[1, 1] = -25.1

    g_dot[2, 0] = -11.5
    g_dot[2, 1] = -7.1
    g_dot[2, 2] = -2.2
    h_dot[2, 1] = -30.2
    h_dot[2, 2] = -23.9

    g_dot[3, 0] = 2.8
    g_dot[3, 1] = -6.2
    g_dot[3, 2] = 3.4
    g_dot[3, 3] = -12.2
    h_dot[3, 1] = 5.7
    h_dot[3, 2] = -1.0
    h_dot[3, 3] = 1.1

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


def magnetic_field_spherical(
    lat: float,
    lon: float,
    r: float,
    year: float,
    coeffs: MagneticCoefficients = WMM2020,
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

    Returns
    -------
    B_r : float
        Radial component (positive outward) in nT.
    B_theta : float
        Colatitude component (positive southward) in nT.
    B_phi : float
        Longitude component (positive eastward) in nT.
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
    # WMM uses Schmidt semi-normalization
    P = associated_legendre(n_max, n_max, cos_theta, normalized=True)

    # Compute dP/dtheta = -sin(theta) * dP/d(cos(theta))
    # For efficiency, use recurrence relation
    dP = np.zeros((n_max + 1, n_max + 1))
    if abs(sin_theta) > 1e-10:
        for n in range(1, n_max + 1):
            for m in range(n + 1):
                if m == n:
                    dP[n, m] = n * cos_theta / sin_theta * P[n, m]
                elif n > m:
                    # Recurrence relation for derivative
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

            # Gauss coefficients
            gnm = g[n, m]
            hnm = h[n, m]

            # Field contributions
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
    I = np.arctan2(Z, H)

    # Declination
    D = np.arctan2(Y, X)

    return MagneticResult(
        X=X,
        Y=Y,
        Z=Z,
        H=H,
        F=F,
        I=I,
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
]
