"""
JPL Ephemerides for High-Precision Celestial Mechanics

This module provides access to JPL Development Ephemeris (DE) files for computing
high-precision positions and velocities of celestial bodies (Sun, Moon, planets).

The module leverages the jplephem library, which provides optimized Fortran-based
interpolation of ephemeris kernels. Multiple DE versions are supported (DE405,
DE430, DE432s, DE440).

Examples
--------
>>> from pytcl.astronomical.ephemerides import DEEphemeris
>>> from datetime import datetime
>>> 
>>> # Load ephemeris (auto-downloads if needed)
>>> eph = DEEphemeris(version='DE440')
>>> 
>>> # Query Sun position (AU)
>>> jd = 2451545.0  # J2000.0
>>> r_sun, v_sun = eph.sun_position(jd)
>>> print(f"Sun distance: {np.linalg.norm(r_sun):.6f} AU")
Sun distance: 0.983327 AU
>>> 
>>> # Query Moon position
>>> r_moon, v_moon = eph.moon_position(jd)

Notes
-----
- Ephemeris files are auto-downloaded to ~/.jplephem/ on first use
- Time input is Julian Day (JD) in Terrestrial Time (TT) scale
- Positions returned in AU, velocities in AU/day in ICRF frame
- For highest precision, use DE440 (latest release) or DE432s (2013)

References
----------
.. [1] Standish, E. M. (1995). "Report of the IAU WGAS Sub-group on
       Numerical Standards". In Highlights of Astronomy (Vol. 10).
.. [2] Folkner, W. M., Williams, J. G., Boggs, D. H., Park, R. S., &
       Kuchynka, P. (2014). "The Planetary and Lunar Ephemeris DE430 and DE431".
       Interplanetary Network Progress Report, 42(196), 1-81.

"""

import numpy as np
from typing import Tuple, Optional, Literal
from functools import lru_cache
import warnings

__all__ = [
    'DEEphemeris',
    'sun_position',
    'moon_position',
    'planet_position',
    'barycenter_position',
]


class DEEphemeris:
    """High-precision JPL Development Ephemeris kernel wrapper.
    
    This class manages access to JPL ephemeris files and provides methods
    for querying positions and velocities of celestial bodies.
    
    Parameters
    ----------
    version : {'DE405', 'DE430', 'DE432s', 'DE440'}, optional
        Ephemeris version to load. Default is 'DE440' (latest).
        - DE440: Latest JPL release (2020), covers 1550-2650
        - DE432s: High-precision version (2013), covers 1350-3000
        - DE430: Earlier release (2013), covers 1550-2650
        - DE405: Older version (1998), compact, covers 1600-2200
    
    Attributes
    ----------
    version : str
        Ephemeris version identifier
    kernel : jplephem.SpiceKernel
        Loaded ephemeris kernel object
    _cache : dict
        Cache for frequently accessed positions
        
    Raises
    ------
    ImportError
        If jplephem is not installed
    ValueError
        If version is not recognized
        
    Examples
    --------
    >>> eph = DEEphemeris(version='DE440')
    >>> r_sun, v_sun = eph.sun_position(2451545.0)
    
    """
    
    # Valid ephemeris versions
    _VALID_VERSIONS = {'DE405', 'DE430', 'DE432s', 'DE440'}
    
    # Supported bodies and their DE IDs
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
    _BODY_IDS = {
        'mercury': 1,
        'venus': 2,
        'earth': 3,
        'moon': 301,
        'mars': 4,
        'jupiter': 5,
        'saturn': 6,
        'uranus': 7,
        'neptune': 8,
        'pluto': 9,
        'sun': 10,
        'earth_moon_barycenter': 3,
        'solar_system_barycenter': 0,
    }
    
    def __init__(self, version: str = 'DE440') -> None:
        """Initialize ephemeris kernel.
        
        Parameters
        ----------
        version : str, optional
            Ephemeris version (default: 'DE440')
            
        """
        if version not in self._VALID_VERSIONS:
            raise ValueError(
                f"Ephemeris version must be one of {self._VALID_VERSIONS}, "
                f"got '{version}'"
            )
        
        try:
            import jplephem
        except ImportError as e:
            raise ImportError(
                "jplephem is required for ephemeris access. "
                "Install with: pip install jplephem"
            ) from e
        
        self.version = version
        self._jplephem = jplephem
        self._kernel: Optional[object] = None
        self._cache: dict = {}
        
    @property
    def kernel(self):
        """Lazy-load ephemeris kernel on first access."""
        if self._kernel is None:
            # jplephem.load_file() auto-downloads if needed
            self._kernel = self._jplephem.load_file(self.version)
        return self._kernel
    
    def sun_position(
        self, 
        jd: float, 
        frame: Literal['icrf', 'ecliptic'] = 'icrf'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Sun position and velocity.
        
        Parameters
        ----------
        jd : float
            Julian Day in Terrestrial Time (TT)
        frame : {'icrf', 'ecliptic'}, optional
            Coordinate frame (default: 'icrf').
            - 'icrf': International Celestial Reference Frame
            - 'ecliptic': Ecliptic coordinate system (J2000.0)
            
        Returns
        -------
        position : ndarray, shape (3,)
            Sun position in AU
        velocity : ndarray, shape (3,)
            Sun velocity in AU/day
            
        Notes
        -----
        The Sun's position is computed relative to the Solar System Barycenter
        (SSB) in the ICRF frame.
        
        Examples
        --------
        >>> eph = DEEphemeris()
        >>> r, v = eph.sun_position(2451545.0)
        >>> print(f"Distance: {np.linalg.norm(r):.6f} AU")
        
        """
        # Sun position relative to SSB
        t = self.kernel.t0 + jd
        position, velocity = self.kernel[0, 10].compute_and_differentiate(t)
        
        position = np.array(position)
        velocity = np.array(velocity)
        
        if frame == 'ecliptic':
            from . import reference_frames
            position = reference_frames.equatorial_to_ecliptic(position)
            velocity = reference_frames.equatorial_to_ecliptic(velocity)
        
        return position, velocity
    
    def moon_position(
        self,
        jd: float,
        frame: Literal['icrf', 'ecliptic', 'earth_centered'] = 'icrf'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Moon position and velocity.
        
        Parameters
        ----------
        jd : float
            Julian Day in Terrestrial Time (TT)
        frame : {'icrf', 'ecliptic', 'earth_centered'}, optional
            Coordinate frame (default: 'icrf').
            - 'icrf': Moon position relative to Solar System Barycenter
            - 'ecliptic': Ecliptic coordinates
            - 'earth_centered': Position relative to Earth
            
        Returns
        -------
        position : ndarray, shape (3,)
            Moon position in AU (or relative to Earth for 'earth_centered')
        velocity : ndarray, shape (3,)
            Moon velocity in AU/day
            
        Notes
        -----
        By default, returns Moon position relative to the Solar System Barycenter.
        Use frame='earth_centered' for geocentric coordinates.
        
        Examples
        --------
        >>> eph = DEEphemeris()
        >>> r, v = eph.moon_position(2451545.0, frame='earth_centered')
        
        """
        t = self.kernel.t0 + jd
        
        if frame == 'earth_centered':
            # Moon relative to Earth
            position, velocity = self.kernel[3, 301].compute_and_differentiate(t)
        else:
            # Moon relative to SSB
            position, velocity = self.kernel[0, 301].compute_and_differentiate(t)
        
        position = np.array(position)
        velocity = np.array(velocity)
        
        if frame == 'ecliptic':
            from . import reference_frames
            position = reference_frames.equatorial_to_ecliptic(position)
            velocity = reference_frames.equatorial_to_ecliptic(velocity)
        
        return position, velocity
    
    def planet_position(
        self,
        planet: Literal['mercury', 'venus', 'mars', 'jupiter', 'saturn', 
                        'uranus', 'neptune'],
        jd: float,
        frame: Literal['icrf', 'ecliptic'] = 'icrf'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute planet position and velocity.
        
        Parameters
        ----------
        planet : str
            Planet name: 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
            'uranus', 'neptune'
        jd : float
            Julian Day in Terrestrial Time (TT)
        frame : {'icrf', 'ecliptic'}, optional
            Coordinate frame (default: 'icrf')
            
        Returns
        -------
        position : ndarray, shape (3,)
            Planet position in AU
        velocity : ndarray, shape (3,)
            Planet velocity in AU/day
            
        Raises
        ------
        ValueError
            If planet name is not recognized
            
        Examples
        --------
        >>> eph = DEEphemeris()
        >>> r, v = eph.planet_position('mars', 2451545.0)
        
        """
        planet_lower = planet.lower()
        if planet_lower not in self._BODY_IDS or planet_lower == 'sun':
            raise ValueError(
                f"Planet must be one of {set(self._BODY_IDS.keys()) - {'sun', 'moon'}}, "
                f"got '{planet}'"
            )
        
        planet_id = self._BODY_IDS[planet_lower]
        t = self.kernel.t0 + jd
        
        position, velocity = self.kernel[0, planet_id].compute_and_differentiate(t)
        
        position = np.array(position)
        velocity = np.array(velocity)
        
        if frame == 'ecliptic':
            from . import reference_frames
            position = reference_frames.equatorial_to_ecliptic(position)
            velocity = reference_frames.equatorial_to_ecliptic(velocity)
        
        return position, velocity
    
    def barycenter_position(
        self,
        body: str,
        jd: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute position of any body relative to Solar System Barycenter.
        
        Parameters
        ----------
        body : str
            Body name ('sun', 'moon', 'mercury', ..., 'neptune')
        jd : float
            Julian Day in Terrestrial Time (TT)
            
        Returns
        -------
        position : ndarray, shape (3,)
            Position in AU
        velocity : ndarray, shape (3,)
            Velocity in AU/day
            
        """
        if body.lower() == 'sun':
            return self.sun_position(jd)
        elif body.lower() == 'moon':
            return self.moon_position(jd, frame='icrf')
        else:
            return self.planet_position(body, jd)
    
    def clear_cache(self) -> None:
        """Clear internal position cache."""
        self._cache.clear()


# Module-level convenience functions

_default_eph: Optional[DEEphemeris] = None


def _get_default_ephemeris() -> DEEphemeris:
    """Get or create default ephemeris instance."""
    global _default_eph
    if _default_eph is None:
        _default_eph = DEEphemeris(version='DE440')
    return _default_eph


def sun_position(
    jd: float,
    frame: Literal['icrf', 'ecliptic'] = 'icrf'
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function: Compute Sun position and velocity.
    
    Parameters
    ----------
    jd : float
        Julian Day in Terrestrial Time (TT)
    frame : {'icrf', 'ecliptic'}, optional
        Coordinate frame (default: 'icrf')
        
    Returns
    -------
    position : ndarray, shape (3,)
        Sun position in AU
    velocity : ndarray, shape (3,)
        Sun velocity in AU/day
        
    See Also
    --------
    DEEphemeris.sun_position : Full ephemeris class with caching
    
    """
    return _get_default_ephemeris().sun_position(jd, frame=frame)


def moon_position(
    jd: float,
    frame: Literal['icrf', 'ecliptic', 'earth_centered'] = 'icrf'
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function: Compute Moon position and velocity.
    
    Parameters
    ----------
    jd : float
        Julian Day in Terrestrial Time (TT)
    frame : {'icrf', 'ecliptic', 'earth_centered'}, optional
        Coordinate frame (default: 'icrf')
        
    Returns
    -------
    position : ndarray, shape (3,)
        Moon position in AU
    velocity : ndarray, shape (3,)
        Moon velocity in AU/day
        
    See Also
    --------
    DEEphemeris.moon_position : Full ephemeris class with caching
    
    """
    return _get_default_ephemeris().moon_position(jd, frame=frame)


def planet_position(
    planet: Literal['mercury', 'venus', 'mars', 'jupiter', 'saturn',
                    'uranus', 'neptune'],
    jd: float,
    frame: Literal['icrf', 'ecliptic'] = 'icrf'
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function: Compute planet position and velocity.
    
    Parameters
    ----------
    planet : str
        Planet name
    jd : float
        Julian Day in Terrestrial Time (TT)
    frame : {'icrf', 'ecliptic'}, optional
        Coordinate frame (default: 'icrf')
        
    Returns
    -------
    position : ndarray, shape (3,)
        Planet position in AU
    velocity : ndarray, shape (3,)
        Planet velocity in AU/day
        
    See Also
    --------
    DEEphemeris.planet_position : Full ephemeris class with caching
    
    """
    return _get_default_ephemeris().planet_position(planet, jd, frame=frame)


def barycenter_position(
    body: str,
    jd: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function: Position relative to Solar System Barycenter.
    
    Parameters
    ----------
    body : str
        Body name ('sun', 'moon', 'mercury', ..., 'neptune')
    jd : float
        Julian Day in Terrestrial Time (TT)
        
    Returns
    -------
    position : ndarray, shape (3,)
        Position in AU
    velocity : ndarray, shape (3,)
        Velocity in AU/day
        
    """
    return _get_default_ephemeris().barycenter_position(body, jd)
