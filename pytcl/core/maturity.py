"""
Module maturity classification system for the Tracker Component Library.

This module provides a standardized way to indicate the production-readiness
and stability of different modules within pyTCL. The maturity levels help
users understand which APIs are stable and which may change.

Maturity Levels
---------------
STABLE (3)
    Production-ready. Thoroughly tested, well-documented, and API is frozen.
    Breaking changes only in major version bumps.

MATURE (2)
    Ready for production use. Good test coverage and documentation.
    Minor API adjustments possible in minor versions.

EXPERIMENTAL (1)
    Functional but may change. Limited testing or documentation.
    API may change in any release.

DEPRECATED (0)
    Scheduled for removal. Use the recommended replacement.

Examples
--------
Check the maturity level of a module:

>>> from pytcl.core.maturity import get_maturity, MaturityLevel
>>> level = get_maturity("pytcl.dynamic_estimation.kalman.linear")
>>> level == MaturityLevel.STABLE
True

List all stable modules:

>>> from pytcl.core.maturity import get_modules_by_maturity, MaturityLevel
>>> stable_modules = get_modules_by_maturity(MaturityLevel.STABLE)

See Also
--------
pytcl.core.optional_deps : Optional dependency management.
"""

from enum import IntEnum
from typing import Dict, List


class MaturityLevel(IntEnum):
    """Maturity level classification for modules.

    Attributes
    ----------
    DEPRECATED : int
        Level 0. Scheduled for removal.
    EXPERIMENTAL : int
        Level 1. Functional but unstable API.
    MATURE : int
        Level 2. Production-ready with possible minor changes.
    STABLE : int
        Level 3. Production-ready with frozen API.
    """

    DEPRECATED = 0
    EXPERIMENTAL = 1
    MATURE = 2
    STABLE = 3


# Module maturity classifications
# Keys are module paths relative to pytcl (e.g., "dynamic_estimation.kalman.linear")
MODULE_MATURITY: Dict[str, MaturityLevel] = {
    # =========================================================================
    # STABLE (3) - Production-ready, frozen API
    # =========================================================================
    # Core
    "core.constants": MaturityLevel.STABLE,
    "core.exceptions": MaturityLevel.STABLE,
    "core.validation": MaturityLevel.STABLE,
    "core.array_utils": MaturityLevel.STABLE,
    "core.optional_deps": MaturityLevel.STABLE,
    # Kalman Filters
    "dynamic_estimation.kalman.linear": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.extended": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.unscented": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.types": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.matrix_utils": MaturityLevel.STABLE,
    # Motion Models
    "dynamic_models.constant_velocity": MaturityLevel.STABLE,
    "dynamic_models.constant_acceleration": MaturityLevel.STABLE,
    "dynamic_models.coordinated_turn": MaturityLevel.STABLE,
    "dynamic_models.singer": MaturityLevel.STABLE,
    "dynamic_models.process_noise.constant_velocity": MaturityLevel.STABLE,
    "dynamic_models.process_noise.constant_acceleration": MaturityLevel.STABLE,
    # Coordinate Systems
    "coordinate_systems.conversions.geodetic": MaturityLevel.STABLE,
    "coordinate_systems.conversions.spherical": MaturityLevel.STABLE,
    "coordinate_systems.rotations.rotations": MaturityLevel.STABLE,
    "coordinate_systems.rotations.quaternions": MaturityLevel.STABLE,
    # Assignment Algorithms
    "assignment_algorithms.hungarian": MaturityLevel.STABLE,
    "assignment_algorithms.auction": MaturityLevel.STABLE,
    "assignment_algorithms.gating": MaturityLevel.STABLE,
    # Containers
    "containers.kd_tree": MaturityLevel.STABLE,
    "containers.base": MaturityLevel.STABLE,
    # Mathematical Functions
    "mathematical_functions.special": MaturityLevel.STABLE,
    # =========================================================================
    # MATURE (2) - Production-ready, minor changes possible
    # =========================================================================
    # Kalman Filters
    "dynamic_estimation.kalman.square_root": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.ud_filter": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.sr_ukf": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.cubature": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.constrained": MaturityLevel.MATURE,
    "dynamic_estimation.information_filter": MaturityLevel.MATURE,
    "dynamic_estimation.imm": MaturityLevel.MATURE,
    "dynamic_estimation.h_infinity": MaturityLevel.MATURE,
    # Particle Filters
    "dynamic_estimation.particle_filters.bootstrap": MaturityLevel.MATURE,
    "dynamic_estimation.particle_filters.resampling": MaturityLevel.MATURE,
    # Smoothers
    "dynamic_estimation.smoothers.rts": MaturityLevel.MATURE,
    "dynamic_estimation.smoothers.fixed_lag": MaturityLevel.MATURE,
    # Motion Models
    "dynamic_models.process_noise.coordinated_turn": MaturityLevel.MATURE,
    "dynamic_models.process_noise.singer": MaturityLevel.MATURE,
    # Assignment Algorithms
    "assignment_algorithms.jpda": MaturityLevel.MATURE,
    "assignment_algorithms.mht": MaturityLevel.MATURE,
    "assignment_algorithms.murty": MaturityLevel.MATURE,
    "assignment_algorithms.assignment_3d": MaturityLevel.MATURE,
    # Containers
    "containers.ball_tree": MaturityLevel.MATURE,
    "containers.rtree": MaturityLevel.MATURE,
    "containers.vptree": MaturityLevel.MATURE,
    "containers.covertree": MaturityLevel.MATURE,
    "containers.track_list": MaturityLevel.MATURE,
    "containers.measurement_set": MaturityLevel.MATURE,
    "containers.cluster_set": MaturityLevel.MATURE,
    # Navigation
    "navigation.ins.strapdown": MaturityLevel.MATURE,
    "navigation.ins.error_model": MaturityLevel.MATURE,
    "navigation.gnss.positioning": MaturityLevel.MATURE,
    "navigation.geodesy": MaturityLevel.MATURE,
    "navigation.great_circle": MaturityLevel.MATURE,
    # Coordinate Systems
    "coordinate_systems.jacobians.jacobians": MaturityLevel.MATURE,
    "coordinate_systems.projections.utm": MaturityLevel.MATURE,
    "coordinate_systems.projections.mercator": MaturityLevel.MATURE,
    # Mathematical Functions
    "mathematical_functions.signal_processing.filters": MaturityLevel.MATURE,
    "mathematical_functions.signal_processing.detection": MaturityLevel.MATURE,
    "mathematical_functions.transforms.fft": MaturityLevel.MATURE,
    "mathematical_functions.transforms.wavelets": MaturityLevel.MATURE,
    # Static Estimation
    "static_estimation.least_squares": MaturityLevel.MATURE,
    "static_estimation.robust": MaturityLevel.MATURE,
    "static_estimation.ransac": MaturityLevel.MATURE,
    # Astronomical
    "astronomical.orbital_mechanics": MaturityLevel.MATURE,
    "astronomical.ephemerides": MaturityLevel.MATURE,
    "astronomical.reference_frames": MaturityLevel.MATURE,
    # =========================================================================
    # EXPERIMENTAL (1) - Functional but API may change
    # =========================================================================
    # Advanced Filters
    "dynamic_estimation.kalman.gaussian_sum": MaturityLevel.EXPERIMENTAL,
    "dynamic_estimation.kalman.rao_blackwellized": MaturityLevel.EXPERIMENTAL,
    # Geophysical Models
    "geophysical.gravity.egm": MaturityLevel.EXPERIMENTAL,
    "geophysical.magnetism.wmm": MaturityLevel.EXPERIMENTAL,
    "geophysical.tides": MaturityLevel.EXPERIMENTAL,
    # Terrain
    "terrain.dem": MaturityLevel.EXPERIMENTAL,
    "terrain.loaders": MaturityLevel.EXPERIMENTAL,
    "terrain.analysis": MaturityLevel.EXPERIMENTAL,
    # Relativity
    "astronomical.relativity": MaturityLevel.EXPERIMENTAL,
    "astronomical.satellite.sgp4": MaturityLevel.EXPERIMENTAL,
}


def get_maturity(module_path: str) -> MaturityLevel:
    """
    Get the maturity level of a module.

    Parameters
    ----------
    module_path : str
        Module path relative to pytcl (e.g., "dynamic_estimation.kalman.linear")
        or full path (e.g., "pytcl.dynamic_estimation.kalman.linear").

    Returns
    -------
    MaturityLevel
        The module's maturity level. Returns EXPERIMENTAL if not classified.

    Examples
    --------
    >>> get_maturity("dynamic_estimation.kalman.linear")
    <MaturityLevel.STABLE: 3>
    >>> get_maturity("pytcl.core.constants")
    <MaturityLevel.STABLE: 3>
    """
    # Strip pytcl. prefix if present
    if module_path.startswith("pytcl."):
        module_path = module_path[6:]

    return MODULE_MATURITY.get(module_path, MaturityLevel.EXPERIMENTAL)


def get_modules_by_maturity(level: MaturityLevel) -> List[str]:
    """
    Get all modules at a specific maturity level.

    Parameters
    ----------
    level : MaturityLevel
        The maturity level to filter by.

    Returns
    -------
    list of str
        Module paths at the specified maturity level.

    Examples
    --------
    >>> stable = get_modules_by_maturity(MaturityLevel.STABLE)
    >>> "core.constants" in stable
    True
    """
    return [path for path, mat in MODULE_MATURITY.items() if mat == level]


def get_maturity_summary() -> Dict[MaturityLevel, int]:
    """
    Get a summary count of modules at each maturity level.

    Returns
    -------
    dict
        Mapping from MaturityLevel to count of modules.

    Examples
    --------
    >>> summary = get_maturity_summary()
    >>> summary[MaturityLevel.STABLE] > 0
    True
    """
    summary = {level: 0 for level in MaturityLevel}
    for level in MODULE_MATURITY.values():
        summary[level] += 1
    return summary


def is_stable(module_path: str) -> bool:
    """
    Check if a module is stable (production-ready with frozen API).

    Parameters
    ----------
    module_path : str
        Module path to check.

    Returns
    -------
    bool
        True if the module is stable.

    Examples
    --------
    >>> is_stable("dynamic_estimation.kalman.linear")
    True
    >>> is_stable("terrain.dem")
    False
    """
    return get_maturity(module_path) == MaturityLevel.STABLE


def is_production_ready(module_path: str) -> bool:
    """
    Check if a module is production-ready (STABLE or MATURE).

    Parameters
    ----------
    module_path : str
        Module path to check.

    Returns
    -------
    bool
        True if the module is STABLE or MATURE.

    Examples
    --------
    >>> is_production_ready("dynamic_estimation.kalman.linear")
    True
    >>> is_production_ready("dynamic_estimation.imm")
    True
    >>> is_production_ready("terrain.dem")
    False
    """
    level = get_maturity(module_path)
    return level >= MaturityLevel.MATURE


def format_maturity_badge(level: MaturityLevel) -> str:
    """
    Get a formatted badge string for a maturity level.

    Parameters
    ----------
    level : MaturityLevel
        The maturity level.

    Returns
    -------
    str
        A badge string suitable for documentation.

    Examples
    --------
    >>> format_maturity_badge(MaturityLevel.STABLE)
    '|stable|'
    """
    badges = {
        MaturityLevel.STABLE: "|stable|",
        MaturityLevel.MATURE: "|mature|",
        MaturityLevel.EXPERIMENTAL: "|experimental|",
        MaturityLevel.DEPRECATED: "|deprecated|",
    }
    return badges.get(level, "|unknown|")


__all__ = [
    "MaturityLevel",
    "MODULE_MATURITY",
    "get_maturity",
    "get_modules_by_maturity",
    "get_maturity_summary",
    "is_stable",
    "is_production_ready",
    "format_maturity_badge",
]
