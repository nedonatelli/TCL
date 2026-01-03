"""
Ionospheric models for radio propagation and navigation applications.

This module provides ionospheric models used for computing signal delays,
electron density profiles, and Total Electron Content (TEC) estimates.
These are essential for GPS/GNSS corrections and radio wave propagation.

Models
------
- Klobuchar: GPS broadcast ionospheric model (single-frequency correction)
- NeQuick: Galileo ionospheric model placeholder
- IRI: International Reference Ionosphere simplified model

References
----------
.. [1] Klobuchar, J.A. (1987). "Ionospheric Time-Delay Algorithm for
       Single-Frequency GPS Users". IEEE Transactions on Aerospace and
       Electronic Systems, AES-23(3), 325-331.
.. [2] Nava, B., Coisson, P., & Radicella, S.M. (2008). "A new version
       of the NeQuick ionosphere electron density model". Journal of
       Atmospheric and Solar-Terrestrial Physics, 70(15), 1856-1862.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
F_L1 = 1575.42e6  # GPS L1 frequency (Hz)
F_L2 = 1227.60e6  # GPS L2 frequency (Hz)


class IonosphereState(NamedTuple):
    """
    Ionospheric state at a given location and time.

    Attributes
    ----------
    tec : float or ndarray
        Total Electron Content in TECU (10^16 electrons/m²).
    delay_l1 : float or ndarray
        Ionospheric delay at L1 frequency in meters.
    delay_l2 : float or ndarray
        Ionospheric delay at L2 frequency in meters.
    f_peak : float or ndarray
        Critical frequency of F2 layer in MHz.
    h_peak : float or ndarray
        Height of F2 layer peak in km.
    """

    tec: float | NDArray[np.float64]
    delay_l1: float | NDArray[np.float64]
    delay_l2: float | NDArray[np.float64]
    f_peak: float | NDArray[np.float64]
    h_peak: float | NDArray[np.float64]


class KlobucharCoefficients(NamedTuple):
    """
    Klobuchar ionospheric model coefficients.

    These coefficients are broadcast by GPS satellites in the navigation message.

    Attributes
    ----------
    alpha : ndarray
        Amplitude coefficients (4 values) in seconds.
    beta : ndarray
        Period coefficients (4 values) in seconds.
    """

    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]


# Default Klobuchar coefficients (typical mid-latitude values)
DEFAULT_KLOBUCHAR = KlobucharCoefficients(
    alpha=np.array([3.82e-8, 1.49e-8, -5.96e-8, -5.96e-8]),
    beta=np.array([1.43e5, 0.0, -3.28e5, 1.13e5]),
)


def klobuchar_delay(
    latitude: ArrayLike,
    longitude: ArrayLike,
    elevation: ArrayLike,
    azimuth: ArrayLike,
    gps_time: ArrayLike,
    coefficients: KlobucharCoefficients | None = None,
) -> NDArray[np.float64]:
    """
    Compute ionospheric delay using the Klobuchar model.

    The Klobuchar model is the standard GPS broadcast ionospheric
    correction model. It provides single-frequency ionospheric
    delay estimates accurate to about 50% RMS.

    Parameters
    ----------
    latitude : array_like
        User geodetic latitude in radians.
    longitude : array_like
        User geodetic longitude in radians.
    elevation : array_like
        Satellite elevation angle in radians.
    azimuth : array_like
        Satellite azimuth angle in radians.
    gps_time : array_like
        GPS time of week in seconds.
    coefficients : KlobucharCoefficients, optional
        Ionospheric coefficients from GPS navigation message.
        If None, uses default mid-latitude values.

    Returns
    -------
    delay : ndarray
        Ionospheric delay in meters (at L1 frequency).

    Examples
    --------
    >>> # User at 40°N, 105°W, satellite at 45° elevation
    >>> delay = klobuchar_delay(
    ...     np.radians(40), np.radians(-105),
    ...     np.radians(45), np.radians(180),
    ...     gps_time=43200  # Noon
    ... )
    >>> delay > 0
    True

    Notes
    -----
    The Klobuchar model assumes a thin-shell ionosphere at 350 km altitude
    and uses a cosine model for diurnal variation. It typically removes
    about 50% of the ionospheric delay.

    References
    ----------
    .. [1] IS-GPS-200, Interface Specification.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    azimuth = np.asarray(azimuth, dtype=np.float64)
    gps_time = np.asarray(gps_time, dtype=np.float64)

    if coefficients is None:
        coefficients = DEFAULT_KLOBUCHAR

    alpha = coefficients.alpha
    beta = coefficients.beta

    # Semi-circles (GPS convention)
    phi_u = latitude / np.pi  # User latitude in semi-circles
    lam_u = longitude / np.pi  # User longitude in semi-circles

    # Earth's central angle (semi-circles)
    psi = 0.0137 / (elevation / np.pi + 0.11) - 0.022

    # Ionospheric pierce point latitude (semi-circles)
    phi_i = phi_u + psi * np.cos(azimuth)
    phi_i = np.clip(phi_i, -0.416, 0.416)

    # Ionospheric pierce point longitude (semi-circles)
    lam_i = lam_u + psi * np.sin(azimuth) / np.cos(phi_i * np.pi)

    # Geomagnetic latitude (semi-circles)
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)

    # Local time at ionospheric pierce point (seconds)
    t = 43200 * lam_i + gps_time
    t = np.mod(t, 86400)

    # Obliquity factor
    F = 1.0 + 16.0 * (0.53 - elevation / np.pi) ** 3

    # Ionospheric delay computation
    # Amplitude
    AMP = alpha[0] + alpha[1] * phi_m + alpha[2] * phi_m**2 + alpha[3] * phi_m**3
    AMP = np.maximum(AMP, 0)

    # Period
    PER = beta[0] + beta[1] * phi_m + beta[2] * phi_m**2 + beta[3] * phi_m**3
    PER = np.maximum(PER, 72000)

    # Phase
    x = 2 * np.pi * (t - 50400) / PER

    # Ionospheric time delay (seconds)
    delay_sec = np.where(
        np.abs(x) < 1.57,
        F * (5e-9 + AMP * (1 - x**2 / 2 + x**4 / 24)),
        F * 5e-9,
    )

    # Convert to meters
    delay_m = delay_sec * SPEED_OF_LIGHT

    return delay_m


def dual_frequency_tec(
    pseudorange_l1: ArrayLike,
    pseudorange_l2: ArrayLike,
) -> NDArray[np.float64]:
    """
    Compute Total Electron Content from dual-frequency pseudoranges.

    This method uses the dispersive nature of the ionosphere to
    estimate TEC from the difference in L1 and L2 pseudoranges.

    Parameters
    ----------
    pseudorange_l1 : array_like
        L1 pseudorange in meters.
    pseudorange_l2 : array_like
        L2 pseudorange in meters.

    Returns
    -------
    tec : ndarray
        Total Electron Content in TECU (10^16 electrons/m²).

    Notes
    -----
    The ionospheric delay is proportional to TEC and inversely
    proportional to frequency squared:
        delay = 40.3 * TEC / f²

    The difference in delays at L1 and L2 gives:
        P2 - P1 = 40.3 * TEC * (1/f1² - 1/f2²)

    This is the standard dual-frequency ionospheric correction method.
    """
    pseudorange_l1 = np.asarray(pseudorange_l1, dtype=np.float64)
    pseudorange_l2 = np.asarray(pseudorange_l2, dtype=np.float64)

    # Ionospheric coefficient
    K = 40.3  # m³/s²

    # Frequency squared terms
    f1_sq = F_L1**2
    f2_sq = F_L2**2

    # TEC from pseudorange difference
    # P2 - P1 = K * TEC * (1/f2² - 1/f1²) / 10^16
    # Note: negative because f1 > f2
    delta_inv_f_sq = 1 / f2_sq - 1 / f1_sq
    tec = (pseudorange_l2 - pseudorange_l1) / (K * delta_inv_f_sq) / 1e16

    return tec


def ionospheric_delay_from_tec(
    tec: ArrayLike,
    frequency: float = F_L1,
) -> NDArray[np.float64]:
    """
    Compute ionospheric delay from Total Electron Content.

    Parameters
    ----------
    tec : array_like
        Total Electron Content in TECU (10^16 electrons/m²).
    frequency : float, optional
        Signal frequency in Hz. Default is GPS L1.

    Returns
    -------
    delay : ndarray
        Ionospheric delay in meters.

    Notes
    -----
    The ionospheric delay for a signal is:
        delay = 40.3 * TEC * 10^16 / f²
    """
    tec = np.asarray(tec, dtype=np.float64)

    K = 40.3  # m³/s²
    delay = K * tec * 1e16 / frequency**2

    return delay


def simple_iri(
    latitude: ArrayLike,
    longitude: ArrayLike,
    altitude: ArrayLike,
    hour: ArrayLike,
    month: int = 6,
    solar_flux: float = 150.0,
) -> IonosphereState:
    """
    Simplified International Reference Ionosphere (IRI) model.

    This provides approximate electron density and TEC values based on
    simplified IRI physics. For accurate predictions, use the full IRI
    model or external services.

    Parameters
    ----------
    latitude : array_like
        Geodetic latitude in radians.
    longitude : array_like
        Geodetic longitude in radians.
    altitude : array_like
        Altitude in meters.
    hour : array_like
        Local hour (0-24).
    month : int, optional
        Month of year (1-12). Default is 6 (June).
    solar_flux : float, optional
        F10.7 solar flux in SFU. Default is 150 (moderate activity).

    Returns
    -------
    state : IonosphereState
        Ionospheric state with TEC, delays, and F2 layer parameters.

    Notes
    -----
    This is a simplified empirical model suitable for educational purposes
    and rough estimates. For operational use, the full IRI-2020 model
    should be employed.

    Examples
    --------
    >>> state = simple_iri(np.radians(40), np.radians(-105), 300e3, 12)
    >>> state.tec > 0
    True
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    altitude = np.asarray(altitude, dtype=np.float64)
    hour = np.asarray(hour, dtype=np.float64)

    # Convert latitude to degrees for calculations
    lat_deg = np.degrees(latitude)
    # lon_deg not used in simplified model but kept for future expansion

    # Simplified F2 layer critical frequency (foF2) model
    # Based on typical diurnal and latitudinal variations
    lat_factor = np.cos(latitude) ** 0.8
    hour_angle = 2 * np.pi * (hour - 14) / 24  # Peak around 14:00 local
    diurnal = 0.5 * (1 + np.cos(hour_angle))

    # Solar activity factor
    solar_factor = 0.5 + 0.5 * (solar_flux - 70) / 180
    solar_factor = np.clip(solar_factor, 0.3, 1.2)

    # Seasonal factor (simplified)
    season_angle = 2 * np.pi * (month - 1) / 12
    season_factor = 1.0 + 0.2 * np.cos(season_angle - np.pi * np.sign(lat_deg))

    # F2 layer critical frequency (MHz)
    f_peak = 5.0 + 8.0 * lat_factor * diurnal * solar_factor * season_factor
    f_peak = np.maximum(f_peak, 2.0)

    # F2 layer peak height (km)
    h_peak = 250 + 100 * (1 - lat_factor) + 50 * (1 - diurnal)

    # Simplified TEC calculation (TECU)
    # TEC roughly scales with foF2 squared
    base_tec = 0.5 * f_peak**2
    tec = base_tec * solar_factor * season_factor

    # Ionospheric delays
    delay_l1 = ionospheric_delay_from_tec(tec, F_L1)
    delay_l2 = ionospheric_delay_from_tec(tec, F_L2)

    # Handle scalar vs array output
    if np.ndim(latitude) == 0:
        return IonosphereState(
            tec=float(tec),
            delay_l1=float(delay_l1),
            delay_l2=float(delay_l2),
            f_peak=float(f_peak),
            h_peak=float(h_peak),
        )

    return IonosphereState(
        tec=tec,
        delay_l1=delay_l1,
        delay_l2=delay_l2,
        f_peak=f_peak,
        h_peak=h_peak,
    )


def magnetic_latitude(
    latitude: ArrayLike,
    longitude: ArrayLike,
) -> NDArray[np.float64]:
    """
    Compute approximate geomagnetic latitude.

    Uses a simple dipole approximation with the magnetic pole at
    approximately 80.5°N, 72.8°W.

    Parameters
    ----------
    latitude : array_like
        Geodetic latitude in radians.
    longitude : array_like
        Geodetic longitude in radians.

    Returns
    -------
    mag_lat : ndarray
        Geomagnetic latitude in radians.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)

    # Approximate magnetic pole location (2020 epoch)
    pole_lat = np.radians(80.5)
    pole_lon = np.radians(-72.8)

    # Spherical law of cosines for angular distance
    cos_mag_lat = np.sin(latitude) * np.sin(pole_lat) + np.cos(latitude) * np.cos(
        pole_lat
    ) * np.cos(longitude - pole_lon)

    # Geomagnetic colatitude
    mag_colat = np.arccos(np.clip(cos_mag_lat, -1, 1))

    # Geomagnetic latitude
    mag_lat = np.pi / 2 - mag_colat

    return mag_lat


def scintillation_index(
    magnetic_latitude: ArrayLike,
    hour: ArrayLike,
    kp_index: float = 3.0,
) -> NDArray[np.float64]:
    """
    Estimate ionospheric scintillation index S4.

    Provides a rough estimate of amplitude scintillation based on
    geomagnetic latitude, local time, and geomagnetic activity.

    Parameters
    ----------
    magnetic_latitude : array_like
        Geomagnetic latitude in radians.
    hour : array_like
        Local hour (0-24).
    kp_index : float, optional
        Kp geomagnetic activity index (0-9). Default is 3 (moderate).

    Returns
    -------
    s4 : ndarray
        S4 amplitude scintillation index (0-1).

    Notes
    -----
    S4 > 0.3 indicates moderate scintillation.
    S4 > 0.6 indicates strong scintillation that may affect receivers.
    """
    magnetic_latitude = np.asarray(magnetic_latitude, dtype=np.float64)
    hour = np.asarray(hour, dtype=np.float64)

    # Scintillation is most intense:
    # - At equatorial latitudes (within ±20° of magnetic equator)
    # - At high latitudes (auroral zone, |lat| > 60°)
    # - Post-sunset to midnight (local time 19-24)
    # - During high geomagnetic activity

    mag_lat_deg = np.abs(np.degrees(magnetic_latitude))

    # Equatorial contribution
    equatorial = np.exp(-((mag_lat_deg - 15) ** 2) / 200)

    # Auroral contribution
    auroral = np.exp(-((mag_lat_deg - 70) ** 2) / 100)

    # Combined latitude factor
    lat_factor = np.maximum(equatorial, 0.3 * auroral)

    # Local time factor (peak at ~20:00 local)
    hour_angle = 2 * np.pi * (hour - 20) / 24
    time_factor = 0.5 * (1 + np.cos(hour_angle))

    # Geomagnetic activity factor
    kp_factor = 0.3 + 0.7 * (kp_index / 9)

    # S4 estimate
    s4 = 0.8 * lat_factor * time_factor * kp_factor

    return np.clip(s4, 0, 1)


__all__ = [
    "IonosphereState",
    "KlobucharCoefficients",
    "DEFAULT_KLOBUCHAR",
    "klobuchar_delay",
    "dual_frequency_tec",
    "ionospheric_delay_from_tec",
    "simple_iri",
    "magnetic_latitude",
    "scintillation_index",
    # Constants
    "SPEED_OF_LIGHT",
    "F_L1",
    "F_L2",
]
