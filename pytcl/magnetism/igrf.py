"""
International Geomagnetic Reference Field (IGRF) implementation.

The IGRF is a standard mathematical description of the Earth's main
magnetic field, used widely in studies of the Earth's interior,
its ionosphere and magnetosphere, and in various applications.

References
----------
.. [1] Alken et al., "International Geomagnetic Reference Field: the
       thirteenth generation," Earth, Planets and Space, 2021.
.. [2] https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
"""

from typing import NamedTuple, Optional

import numpy as np

from pytcl.magnetism.wmm import (
    MagneticCoefficients,
    MagneticResult,
    magnetic_field_spherical,
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


def create_igrf13_coefficients() -> MagneticCoefficients:
    """
    Create IGRF-13 model coefficients for epoch 2020.

    Returns
    -------
    coeffs : MagneticCoefficients
        IGRF-13 spherical harmonic coefficients.

    Notes
    -----
    IGRF-13 is valid from 1900.0 to 2025.0. This function returns
    the coefficients for the 2020.0 epoch. For other epochs, the
    coefficients should be interpolated.
    """
    n_max = 13  # IGRF goes to degree 13

    g = np.zeros((n_max + 1, n_max + 1))
    h = np.zeros((n_max + 1, n_max + 1))
    g_dot = np.zeros((n_max + 1, n_max + 1))
    h_dot = np.zeros((n_max + 1, n_max + 1))

    # IGRF-13 coefficients for 2020.0 epoch
    # Units: nT

    # n=1
    g[1, 0] = -29404.8
    g[1, 1] = -1450.9
    h[1, 1] = 4652.5

    # n=2
    g[2, 0] = -2499.6
    g[2, 1] = 2982.0
    g[2, 2] = 1677.0
    h[2, 1] = -2991.6
    h[2, 2] = -734.6

    # n=3
    g[3, 0] = 1363.2
    g[3, 1] = -2381.2
    g[3, 2] = 1236.2
    g[3, 3] = 525.7
    h[3, 1] = -82.1
    h[3, 2] = 241.9
    h[3, 3] = -543.4

    # n=4
    g[4, 0] = 903.0
    g[4, 1] = 809.5
    g[4, 2] = 86.3
    g[4, 3] = -309.4
    g[4, 4] = 48.0
    h[4, 1] = 281.9
    h[4, 2] = -158.4
    h[4, 3] = 199.7
    h[4, 4] = -349.7

    # n=5
    g[5, 0] = -234.3
    g[5, 1] = 363.2
    g[5, 2] = 47.7
    g[5, 3] = 187.8
    g[5, 4] = -140.7
    g[5, 5] = -151.2
    h[5, 1] = 46.9
    h[5, 2] = 196.9
    h[5, 3] = -119.3
    h[5, 4] = 16.0
    h[5, 5] = 100.2

    # n=6
    g[6, 0] = 66.0
    g[6, 1] = 65.5
    g[6, 2] = -19.1
    g[6, 3] = 72.9
    g[6, 4] = -62.6
    g[6, 5] = 0.6
    g[6, 6] = -24.2
    h[6, 1] = -76.7
    h[6, 2] = 25.4
    h[6, 3] = -9.2
    h[6, 4] = 55.8
    h[6, 5] = -17.0
    h[6, 6] = 8.4

    # n=7
    g[7, 0] = 80.4
    g[7, 1] = -76.6
    g[7, 2] = -8.2
    g[7, 3] = -26.6
    g[7, 4] = 3.0
    g[7, 5] = -14.9
    g[7, 6] = 10.4
    g[7, 7] = -18.3
    h[7, 1] = 0.2
    h[7, 2] = -21.5
    h[7, 3] = 15.4
    h[7, 4] = 13.8
    h[7, 5] = -13.5
    h[7, 6] = -0.1
    h[7, 7] = 8.9

    # n=8
    g[8, 0] = 24.2
    g[8, 1] = 5.8
    g[8, 2] = -2.0
    g[8, 3] = -5.8
    g[8, 4] = 0.1
    g[8, 5] = 11.0
    g[8, 6] = -1.4
    g[8, 7] = -6.5
    g[8, 8] = -2.0
    h[8, 1] = -20.3
    h[8, 2] = 13.4
    h[8, 3] = 12.0
    h[8, 4] = -6.4
    h[8, 5] = -8.5
    h[8, 6] = 8.5
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
    h[9, 3] = 3.7
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
    g[12, 5] = 1.0
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

    # n=13
    g[13, 0] = -0.1
    g[13, 1] = -0.2
    g[13, 2] = 0.1
    g[13, 3] = 0.1
    g[13, 4] = 0.0
    g[13, 5] = 0.0
    g[13, 6] = 0.1
    g[13, 7] = 0.0
    g[13, 8] = 0.0
    g[13, 9] = 0.0
    g[13, 10] = 0.0
    g[13, 11] = 0.0
    g[13, 12] = 0.0
    g[13, 13] = 0.0
    h[13, 1] = 0.1
    h[13, 2] = 0.0
    h[13, 3] = 0.0
    h[13, 4] = 0.1
    h[13, 5] = 0.0
    h[13, 6] = 0.0
    h[13, 7] = 0.0
    h[13, 8] = 0.0
    h[13, 9] = 0.0
    h[13, 10] = 0.0
    h[13, 11] = 0.0
    h[13, 12] = 0.0
    h[13, 13] = 0.0

    # Secular variation (SV) for 2020-2025 (nT/year)
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
    h_dot[3, 1] = 6.0
    h_dot[3, 2] = -1.1
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


# Default IGRF-13 coefficients
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
    """
    if coeffs is None:
        coeffs = IGRF13

    # Reference radius in km
    a = 6371.2

    # Approximate geocentric coordinates
    lat_gc = lat  # Simplified
    r = a + h

    # Compute field in spherical coordinates
    B_r, B_theta, B_phi = magnetic_field_spherical(lat_gc, lon, r, year, coeffs)

    # Convert to geodetic (X=North, Y=East, Z=Down)
    X = -B_theta
    Y = B_phi
    Z = -B_r

    # Derived quantities
    H = np.sqrt(X * X + Y * Y)
    F = np.sqrt(H * H + Z * Z)
    incl = np.arctan2(Z, H)
    D = np.arctan2(Y, X)

    return MagneticResult(X=X, Y=Y, Z=Z, H=H, F=F, I=incl, D=D)


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
    """
    g10 = coeffs.g[1, 0]
    g11 = coeffs.g[1, 1]
    h11 = coeffs.h[1, 1]

    # Colatitude of pole
    theta = np.arctan2(np.sqrt(g11**2 + h11**2), g10)

    # Longitude of pole
    phi = np.arctan2(h11, g11)

    # Convert to latitude
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
    """
    # Start from dipole pole
    lat, lon = dipole_axis(coeffs)

    # Simple iterative refinement
    for _ in range(10):
        result = igrf(lat, lon, 0, year, coeffs)

        # At the pole, H should be zero
        if result.H < 1.0:  # Close enough (1 nT)
            break

        # Move toward where H is smaller
        # This is a simplified search
        lat += 0.01 * np.sign(np.pi / 2 - abs(lat))

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
