"""
NRLMSISE-00 Atmospheric Model

High-fidelity thermosphere/atmosphere model from the U.S. Naval Research
Laboratory. Provides density, temperature, and composition profiles for
altitudes from -5 km to 1000 km.

References
----------
.. [1] Picone, J. M., A. E. Hedin, D. P. Drob, and A. C. Aikin (2002),
       "NRLMSISE-00 empirical model of the atmosphere: Statistical
       comparisons and scientific issues," J. Geophys. Res., 107(A12), 1468,
       doi:10.1029/2002JA009430
.. [2] NASA GSFC NRLMSISE-00 Model:
       https://ccmc.gsfc.nasa.gov/models/nrlmsise00
"""

from typing import NamedTuple
import numpy as np
from numpy.typing import ArrayLike, NDArray


class NRLMSISE00Output(NamedTuple):
    """
    Output from NRLMSISE-00 atmospheric model.
    
    Attributes
    ----------
    density : float or ndarray
        Total atmospheric density in kg/m³.
    temperature : float or ndarray
        Temperature at altitude (K).
    exosphere_temperature : float or ndarray
        Exospheric temperature (K).
    he_density : float or ndarray
        Helium density in m⁻³.
    o_density : float or ndarray
        Atomic oxygen density in m⁻³.
    n2_density : float or ndarray
        N₂ density in m⁻³.
    o2_density : float or ndarray
        O₂ density in m⁻³.
    ar_density : float or ndarray
        Argon density in m⁻³.
    h_density : float or ndarray
        Hydrogen density in m⁻³.
    n_density : float or ndarray
        Atomic nitrogen density in m⁻³.
    """
    
    density: float | NDArray[np.float64]
    temperature: float | NDArray[np.float64]
    exosphere_temperature: float | NDArray[np.float64]
    he_density: float | NDArray[np.float64]
    o_density: float | NDArray[np.float64]
    n2_density: float | NDArray[np.float64]
    o2_density: float | NDArray[np.float64]
    ar_density: float | NDArray[np.float64]
    h_density: float | NDArray[np.float64]
    n_density: float | NDArray[np.float64]


class F107Index(NamedTuple):
    """
    Solar activity indices for NRLMSISE-00.
    
    Attributes
    ----------
    f107 : float
        10.7 cm solar radio flux (daily, SFU).
    f107a : float
        10.7 cm solar radio flux (81-day average, SFU).
    ap : float or ndarray
        Planetary magnetic index (Ap index).
    ap_array : ndarray, optional
        Ap values for each 3-hour interval of the day (8 values).
        If not provided, derived from ap value.
    """
    
    f107: float
    f107a: float
    ap: float | NDArray[np.float64]
    ap_array: NDArray[np.float64] | None = None


# NRLMSISE-00 Coefficients (simplified structure)
# Note: Full model requires extensive coefficient tables from NOAA
# These are placeholder structures that would be populated from data files

class NRLMSISE00:
    """
    NRLMSISE-00 High-Fidelity Atmosphere Model.
    
    This is a comprehensive thermosphere model covering altitudes from
    approximately -5 km to 1000 km, with detailed chemical composition
    and temperature profiles.
    
    Parameters
    ----------
    use_meter_altitude : bool, optional
        If True, expect altitude input in meters. If False, expect km.
        Default is True (meters).
    
    Notes
    -----
    The model requires:
    - Solar flux index (F10.7)
    - Magnetic activity index (Ap)
    - Geographic location (latitude, longitude)
    - Time of day
    
    The implementation loads coefficient tables from data files that
    should be obtained from NOAA's Space Weather Prediction Center.
    """
    
    def __init__(self, use_meter_altitude: bool = True):
        """Initialize NRLMSISE-00 model."""
        self.use_meter_altitude = use_meter_altitude
        self._coefficients_loaded = False
        # TODO: Load coefficient tables from data files
    
    def __call__(
        self,
        latitude: ArrayLike,
        longitude: ArrayLike,
        altitude: ArrayLike,
        year: int,
        day_of_year: int,
        seconds_in_day: float,
        f107: float = 150.0,
        f107a: float = 150.0,
        ap: float | ArrayLike = 4.0,
    ) -> NRLMSISE00Output:
        """
        Compute atmospheric density and composition.
        
        Parameters
        ----------
        latitude : array_like
            Geodetic latitude in radians.
        longitude : array_like
            Longitude in radians.
        altitude : array_like
            Altitude in meters (or km if use_meter_altitude=False).
        year : int
            Year (e.g., 2024).
        day_of_year : int
            Day of year (1-366).
        seconds_in_day : float
            Seconds since midnight (0-86400).
        f107 : float, optional
            10.7 cm solar flux (daily value, SFU). Default 150.
        f107a : float, optional
            10.7 cm solar flux (81-day average, SFU). Default 150.
        ap : float or array_like, optional
            Planetary magnetic index. Can be single value or 8-element
            array of 3-hour Ap values. Default 4.0.
        
        Returns
        -------
        output : NRLMSISE00Output
            Atmospheric properties (density, temperature, composition).
        
        Notes
        -----
        This is a placeholder for the actual NRLMSISE-00 implementation.
        The full model requires:
        1. Loading coefficient tables from NOAA data
        2. Implementing complex interpolation algorithms
        3. Handling special cases (high solar activity, magnetic storms)
        """
        raise NotImplementedError(
            "NRLMSISE-00 implementation requires coefficient tables from NOAA. "
            "Please download NRLMSISE-00 data files from: "
            "https://ccmc.gsfc.nasa.gov/models/nrlmsise00"
        )


def nrlmsise00(
    latitude: ArrayLike,
    longitude: ArrayLike,
    altitude: ArrayLike,
    year: int,
    day_of_year: int,
    seconds_in_day: float,
    f107: float = 150.0,
    f107a: float = 150.0,
    ap: float | ArrayLike = 4.0,
) -> NRLMSISE00Output:
    """
    Compute NRLMSISE-00 atmospheric properties.
    
    This is a module-level convenience function wrapping the NRLMSISE00 class.
    
    Parameters
    ----------
    latitude : array_like
        Geodetic latitude in radians.
    longitude : array_like
        Longitude in radians.
    altitude : array_like
        Altitude in meters.
    year : int
        Year (e.g., 2024).
    day_of_year : int
        Day of year (1-366).
    seconds_in_day : float
        Seconds since midnight (0-86400).
    f107 : float, optional
        10.7 cm solar flux (daily value, SFU). Default 150.
    f107a : float, optional
        10.7 cm solar flux (81-day average, SFU). Default 150.
    ap : float or array_like, optional
        Planetary magnetic index. Default 4.0.
    
    Returns
    -------
    output : NRLMSISE00Output
        Atmospheric properties.
    
    Notes
    -----
    See NRLMSISE00 class for more details.
    
    Examples
    --------
    >>> # ISS altitude (~400 km), magnetic latitude = 40°, quiet geomagnetic activity
    >>> output = nrlmsise00(
    ...     latitude=np.radians(40),
    ...     longitude=np.radians(-75),
    ...     altitude=400_000,  # 400 km
    ...     year=2024,
    ...     day_of_year=1,
    ...     seconds_in_day=43200,
    ...     f107=150,  # Average solar activity
    ...     f107a=150,
    ...     ap=5  # Quiet conditions
    ... )
    >>> print(f"Density at ISS: {output.density:.2e} kg/m³")
    """
    model = NRLMSISE00()
    return model(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        year=year,
        day_of_year=day_of_year,
        seconds_in_day=seconds_in_day,
        f107=f107,
        f107a=f107a,
        ap=ap,
    )


__all__ = [
    "NRLMSISE00",
    "NRLMSISE00Output",
    "F107Index",
    "nrlmsise00",
]
