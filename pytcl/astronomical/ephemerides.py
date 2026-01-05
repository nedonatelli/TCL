"""
JPL Ephemerides for High-Precision Celestial Mechanics

This module provides access to JPL Development Ephemeris (DE) files for computing
high-precision positions and velocities of celestial bodies (Sun, Moon, planets).

The module leverages the jplephem library, which provides optimized Fortran-based
interpolation of ephemeris kernels. Multiple DE versions are supported (DE405,
DE430, DE432s, DE440).

Constants
---------
AU_PER_KM : float
    Astronomical Unit in kilometers (1 AU = 149597870.7 km)
KM_PER_DAY_TO_AU_PER_DAY : float
    Conversion factor for velocity from km/day to AU/day

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

from typing import Any, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.core.exceptions import DependencyError

# Constants for unit conversion
AU_PER_KM = 1.0 / 149597870.7  # 1 AU in km
KM_PER_DAY_TO_AU_PER_DAY = AU_PER_KM  # velocity conversion factor
EPSILON_J2000 = 0.4090910179  # Mean obliquity of the ecliptic at J2000.0 (radians)

__all__ = [
    "DEEphemeris",
    "sun_position",
    "moon_position",
    "planet_position",
    "barycenter_position",
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
    DependencyError
        If jplephem is not installed
    ValueError
        If version is not recognized

    Examples
    --------
    >>> eph = DEEphemeris(version='DE440')
    >>> r_sun, v_sun = eph.sun_position(2451545.0)

    """

    # Valid ephemeris versions
    _VALID_VERSIONS = {"DE405", "DE430", "DE432s", "DE440"}

    # Supported bodies and their DE IDs
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
    _BODY_IDS = {
        "mercury": 1,
        "venus": 2,
        "earth": 3,
        "moon": 301,
        "mars": 4,
        "jupiter": 5,
        "saturn": 6,
        "uranus": 7,
        "neptune": 8,
        "pluto": 9,
        "sun": 10,
        "earth_moon_barycenter": 3,
        "solar_system_barycenter": 0,
    }

    def __init__(self, version: str = "DE440") -> None:
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
            raise DependencyError(
                "jplephem is required for ephemeris access.",
                package="jplephem",
                feature="JPL ephemeris access",
                install_command="pip install pytcl[astronomy]",
            ) from e

        self.version = version
        self._jplephem = jplephem
        self._kernel: Optional[object] = None
        self._cache: dict[str, Any] = {}

    @property
    def kernel(self) -> Optional[object]:
        """Lazy-load ephemeris kernel on first access.

        Note: This requires jplephem to be installed and the kernel file
        to be available locally or downloadable from the JPL servers.
        """
        if self._kernel is None:
            try:
                # Try to load using jplephem SPK module
                import os
                import urllib.request

                from jplephem.daf import DAF
                from jplephem.spk import SPK

                # Try to construct kernel filename
                kernel_name = f"de{self.version[2:]}.bsp"
                kernel_url = f"https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/{kernel_name}"

                # Try to download if not exists
                kernel_path = os.path.expanduser(f"~/.jplephem/{kernel_name}")
                os_dir = os.path.dirname(kernel_path)
                if not os.path.exists(os_dir):
                    os.makedirs(os_dir, exist_ok=True)

                if not os.path.exists(kernel_path):
                    try:
                        urllib.request.urlretrieve(kernel_url, kernel_path)
                    except Exception as e:
                        raise RuntimeError(
                            f"Could not download ephemeris kernel from {kernel_url}. "
                            f"Please download manually and place at {kernel_path}"
                        ) from e

                # Load the kernel using DAF and SPK
                daf = DAF(open(kernel_path, "rb"))
                self._kernel = SPK(daf)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load ephemeris kernel for version {self.version}. "
                    f"Ensure jplephem is installed and kernel files are accessible. "
                    f"Error: {str(e)}"
                ) from e

        return self._kernel

    def sun_position(
        self, jd: float, frame: Literal["icrf", "ecliptic"] = "icrf"
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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
        # Sun position relative to SSB (in km)
        segment = self.kernel[0, 10]
        position, velocity = segment.compute_and_differentiate(jd)

        # Convert from km to AU
        position = np.array(position) * AU_PER_KM
        velocity = np.array(velocity) * KM_PER_DAY_TO_AU_PER_DAY

        if frame == "ecliptic":
            from . import reference_frames

            position = reference_frames.equatorial_to_ecliptic(position, EPSILON_J2000)
            velocity = reference_frames.equatorial_to_ecliptic(velocity, EPSILON_J2000)

        return position, velocity

    def moon_position(
        self, jd: float, frame: Literal["icrf", "ecliptic", "earth_centered"] = "icrf"
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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
        if frame == "earth_centered":
            # Moon relative to Earth
            segment = self.kernel[3, 301]
            position, velocity = segment.compute_and_differentiate(jd)
        else:
            # Moon relative to SSB: need to compute Earth->Moon, then add Earth->SSB
            # Get Earth barycenter position
            earth_segment = self.kernel[0, 3]
            earth_pos, earth_vel = earth_segment.compute_and_differentiate(jd)

            # Get Moon position relative to Earth
            moon_segment = self.kernel[3, 301]
            moon_rel_earth_pos, moon_rel_earth_vel = (
                moon_segment.compute_and_differentiate(jd)
            )

            # Moon position relative to SSB
            position = earth_pos + moon_rel_earth_pos
            velocity = earth_vel + moon_rel_earth_vel

        # Convert from km to AU
        position = np.array(position) * AU_PER_KM
        velocity = np.array(velocity) * KM_PER_DAY_TO_AU_PER_DAY

        if frame == "ecliptic":
            from . import reference_frames

            position = reference_frames.equatorial_to_ecliptic(position, EPSILON_J2000)
            velocity = reference_frames.equatorial_to_ecliptic(velocity, EPSILON_J2000)

        return position, velocity

    def planet_position(
        self,
        planet: Literal[
            "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"
        ],
        jd: float,
        frame: Literal["icrf", "ecliptic"] = "icrf",
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
        if planet_lower not in self._BODY_IDS or planet_lower == "sun":
            raise ValueError(
                f"Planet must be one of {set(self._BODY_IDS.keys()) - {'sun', 'moon'}}, "
                f"got '{planet}'"
            )

        planet_id = self._BODY_IDS[planet_lower]
        segment = self.kernel[0, planet_id]
        position, velocity = segment.compute_and_differentiate(jd)

        # Convert from km to AU
        position = np.array(position) * AU_PER_KM
        velocity = np.array(velocity) * KM_PER_DAY_TO_AU_PER_DAY

        if frame == "ecliptic":
            from . import reference_frames

            position = reference_frames.equatorial_to_ecliptic(position, EPSILON_J2000)
            velocity = reference_frames.equatorial_to_ecliptic(velocity, EPSILON_J2000)

        return position, velocity

    def barycenter_position(
        self, body: str, jd: float
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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
        if body.lower() == "sun":
            return self.sun_position(jd)
        elif body.lower() == "moon":
            return self.moon_position(jd, frame="icrf")
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
        _default_eph = DEEphemeris(version="DE440")
    return _default_eph


def sun_position(
    jd: float, frame: Literal["icrf", "ecliptic"] = "icrf"
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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

    Examples
    --------
    >>> r, v = sun_position(2451545.0)  # J2000.0  # doctest: +SKIP
    >>> np.linalg.norm(r)  # Distance from SSB  # doctest: +SKIP
    0.00...

    See Also
    --------
    DEEphemeris.sun_position : Full ephemeris class with caching

    """
    return _get_default_ephemeris().sun_position(jd, frame=frame)


def moon_position(
    jd: float, frame: Literal["icrf", "ecliptic", "earth_centered"] = "icrf"
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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

    Examples
    --------
    >>> r, v = moon_position(2451545.0, 'earth_centered')  # doctest: +SKIP
    >>> np.linalg.norm(r) * 149597870.7  # Distance in km  # doctest: +SKIP
    402...

    See Also
    --------
    DEEphemeris.moon_position : Full ephemeris class with caching

    """
    return _get_default_ephemeris().moon_position(jd, frame=frame)


def planet_position(
    planet: Literal[
        "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"
    ],
    jd: float,
    frame: Literal["icrf", "ecliptic"] = "icrf",
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    body: str, jd: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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

    Examples
    --------
    >>> r, v = barycenter_position('mars', 2451545.0)  # doctest: +SKIP
    >>> np.linalg.norm(r)  # Mars distance from SSB  # doctest: +SKIP
    1.4...

    """
    return _get_default_ephemeris().barycenter_position(body, jd)
