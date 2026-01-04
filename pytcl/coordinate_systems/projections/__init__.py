"""
Map Projections for Tracking and Navigation.

This module provides common map projections for converting between
geodetic coordinates and planar map coordinates:

- Mercator: Cylindrical conformal projection
- Transverse Mercator: Basis for UTM
- UTM: Universal Transverse Mercator with zone handling
- Stereographic: Azimuthal conformal projection
- Lambert Conformal Conic: Conic projection for mid-latitudes
- Azimuthal Equidistant: Preserves distances from center

All projections use WGS84 ellipsoid parameters by default.

Examples
--------
>>> import numpy as np
>>> from pytcl.coordinate_systems.projections import geodetic2utm, utm2geodetic
>>> # Convert to UTM
>>> result = geodetic2utm(np.radians(45.0), np.radians(-75.5))
>>> print(f"Zone {result.zone}{result.hemisphere}: "
...       f"E={result.easting:.1f}, N={result.northing:.1f}")
>>> # Convert back
>>> lat, lon = utm2geodetic(result.easting, result.northing,
...                         result.zone, result.hemisphere)
"""

from pytcl.coordinate_systems.projections.projections import (
    WGS84_A,  # Constants; Result types
)
from pytcl.coordinate_systems.projections.projections import (
    WGS84_B,
    WGS84_E,
    WGS84_E2,
    WGS84_EP2,
    WGS84_F,
    ProjectionResult,
    UTMResult,
    azimuthal_equidistant,
    azimuthal_equidistant_inverse,
    geodetic2utm,
    geodetic2utm_batch,
    lambert_conformal_conic,
    lambert_conformal_conic_inverse,
    mercator,
    mercator_inverse,
    polar_stereographic,
    stereographic,
    stereographic_inverse,
    transverse_mercator,
    transverse_mercator_inverse,
    utm2geodetic,
    utm_central_meridian,
    utm_zone,
)

__all__ = [
    # Constants
    "WGS84_A",
    "WGS84_B",
    "WGS84_F",
    "WGS84_E",
    "WGS84_E2",
    "WGS84_EP2",
    # Result types
    "ProjectionResult",
    "UTMResult",
    # Mercator
    "mercator",
    "mercator_inverse",
    # Transverse Mercator
    "transverse_mercator",
    "transverse_mercator_inverse",
    # UTM
    "utm_zone",
    "utm_central_meridian",
    "geodetic2utm",
    "utm2geodetic",
    "geodetic2utm_batch",
    # Stereographic
    "stereographic",
    "stereographic_inverse",
    "polar_stereographic",
    # Lambert Conformal Conic
    "lambert_conformal_conic",
    "lambert_conformal_conic_inverse",
    # Azimuthal Equidistant
    "azimuthal_equidistant",
    "azimuthal_equidistant_inverse",
]
