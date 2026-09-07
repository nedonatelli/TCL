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

Notes
-----
**How modules are classified.** The levels are a promise about breakage, so
they are assigned from evidence rather than impression:

- **STABLE** is never assigned automatically. Freezing an API is a release
  commitment, not something a coverage number implies; promoting a module
  here is a deliberate act.
- **MATURE** requires at least 90% line coverage, no behaviour change in the
  current release, and a code path CI can actually execute.
- **EXPERIMENTAL** covers everything else: below that coverage bar, changed
  behaviour this release, or -- for :mod:`pytcl.gpu` -- a CuPy branch that no
  CI runner can reach, so the reported coverage reflects only the MLX half.

Two invariants are enforced by ``tests/unit/test_maturity_comprehensive.py``:
every registered path must import, and every module must be registered. The
first exists because 32 of 78 entries once named modules from a pre-2.0
layout, and since lookup is by exact path they silently reported
EXPERIMENTAL instead of their recorded level. The second exists because whole
subsystems shipped -- ``io``, ``diagnostics``, ``transponders``, ``gpu`` --
without ever being classified.

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
    # Reclassified from STABLE: ensure_positive_definite began rejecting
    # singular matrices, which STABLE would have deferred to a major bump.
    # The frozen-API claim was aspirational rather than enforced.
    "core.validation": MaturityLevel.MATURE,
    "core.array_utils": MaturityLevel.STABLE,
    "core.optional_deps": MaturityLevel.STABLE,
    # Kalman Filters
    "dynamic_estimation.kalman.linear": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.extended": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.unscented": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.types": MaturityLevel.STABLE,
    "dynamic_estimation.kalman.matrix_utils": MaturityLevel.STABLE,
    # Motion Models
    "dynamic_models.discrete_time.polynomial": MaturityLevel.STABLE,
    "dynamic_models.discrete_time.coordinated_turn": MaturityLevel.STABLE,
    "dynamic_models.discrete_time.singer": MaturityLevel.STABLE,
    "dynamic_models.process_noise.polynomial": MaturityLevel.STABLE,
    # Coordinate Systems
    "coordinate_systems.conversions.geodetic": MaturityLevel.STABLE,
    "coordinate_systems.conversions.spherical": MaturityLevel.STABLE,
    "coordinate_systems.rotations.rotations": MaturityLevel.STABLE,
    # Assignment Algorithms
    "assignment_algorithms.two_dimensional.assignment": MaturityLevel.STABLE,
    "assignment_algorithms.gating": MaturityLevel.STABLE,
    # Containers
    "containers.kd_tree": MaturityLevel.STABLE,
    "containers.base": MaturityLevel.STABLE,
    # Mathematical Functions
    # =========================================================================
    # MATURE (2) - Production-ready, minor changes possible
    # =========================================================================
    # Kalman Filters
    "dynamic_estimation.kalman.square_root": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.ud_filter": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.sr_ukf": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.constrained": MaturityLevel.MATURE,
    "dynamic_estimation.information_filter": MaturityLevel.MATURE,
    "dynamic_estimation.imm": MaturityLevel.MATURE,
    "dynamic_estimation.kalman.h_infinity": MaturityLevel.MATURE,
    # Particle Filters
    "dynamic_estimation.particle_filters.bootstrap": MaturityLevel.MATURE,
    # Smoothers
    "dynamic_estimation.smoothers": MaturityLevel.MATURE,
    # Motion Models
    "dynamic_models.process_noise.coordinated_turn": MaturityLevel.MATURE,
    "dynamic_models.process_noise.singer": MaturityLevel.MATURE,
    # Assignment Algorithms
    "assignment_algorithms.jpda": MaturityLevel.MATURE,
    "trackers.mht": MaturityLevel.MATURE,
    "assignment_algorithms.two_dimensional.kbest": MaturityLevel.MATURE,
    "assignment_algorithms.three_dimensional.assignment": MaturityLevel.MATURE,
    # Containers
    "containers.rtree": MaturityLevel.MATURE,
    "containers.vptree": MaturityLevel.MATURE,
    "containers.covertree": MaturityLevel.MATURE,
    "containers.track_list": MaturityLevel.MATURE,
    "containers.measurement_set": MaturityLevel.MATURE,
    "containers.cluster_set": MaturityLevel.MATURE,
    # Navigation
    "navigation.ins": MaturityLevel.MATURE,
    "navigation.ins_gnss": MaturityLevel.MATURE,
    "navigation.geodesy": MaturityLevel.MATURE,
    "navigation.great_circle": MaturityLevel.MATURE,
    # Coordinate Systems
    "coordinate_systems.jacobians.jacobians": MaturityLevel.MATURE,
    "coordinate_systems.projections.projections": MaturityLevel.MATURE,
    # Mathematical Functions
    "mathematical_functions.signal_processing.filters": MaturityLevel.MATURE,
    "mathematical_functions.signal_processing.detection": MaturityLevel.MATURE,
    "mathematical_functions.transforms.fourier": MaturityLevel.MATURE,
    "mathematical_functions.transforms.wavelets": MaturityLevel.MATURE,
    # Static Estimation
    "static_estimation.least_squares": MaturityLevel.MATURE,
    "static_estimation.robust": MaturityLevel.MATURE,
    # Astronomical
    "astronomical.orbital_mechanics": MaturityLevel.MATURE,
    "astronomical.ephemerides": MaturityLevel.MATURE,
    "astronomical.reference_frames": MaturityLevel.MATURE,
    # =========================================================================
    # EXPERIMENTAL (1) - Functional but API may change
    # =========================================================================
    # Advanced Filters
    "dynamic_estimation.gaussian_sum_filter": MaturityLevel.EXPERIMENTAL,
    "dynamic_estimation.rbpf": MaturityLevel.EXPERIMENTAL,
    # Geophysical Models
    "gravity.egm": MaturityLevel.EXPERIMENTAL,
    "magnetism.wmm": MaturityLevel.EXPERIMENTAL,
    "gravity.tides": MaturityLevel.EXPERIMENTAL,
    # Terrain
    "terrain.dem": MaturityLevel.EXPERIMENTAL,
    "terrain.loaders": MaturityLevel.EXPERIMENTAL,
    # Relativity
    "astronomical.relativity": MaturityLevel.EXPERIMENTAL,
    "astronomical.sgp4": MaturityLevel.EXPERIMENTAL,
    # =========================================================================
    # Added Mature entries (see the audit note in the module docstring)
    # =========================================================================
    # assignment_algorithms
    "assignment_algorithms.data_association": MaturityLevel.MATURE,
    "assignment_algorithms.dijkstra_min_cost": MaturityLevel.MATURE,
    "assignment_algorithms.nd_assignment": MaturityLevel.MATURE,
    "assignment_algorithms.network_flow": MaturityLevel.MATURE,
    # astronomical
    "astronomical.lambert": MaturityLevel.MATURE,
    "astronomical.special_orbits": MaturityLevel.MATURE,
    "astronomical.time_systems": MaturityLevel.MATURE,
    # atmosphere
    "atmosphere.humidity": MaturityLevel.EXPERIMENTAL,
    "atmosphere.ionosphere": MaturityLevel.MATURE,
    "atmosphere.models": MaturityLevel.MATURE,
    "atmosphere.refraction": MaturityLevel.EXPERIMENTAL,
    "atmosphere.thermosphere": MaturityLevel.MATURE,
    # clustering
    "clustering.kmeans": MaturityLevel.MATURE,
    # core
    "core.paths": MaturityLevel.MATURE,
    # diagnostics
    "diagnostics.render": MaturityLevel.MATURE,
    # dynamic_estimation
    "dynamic_estimation.configs": MaturityLevel.MATURE,
    # gravity
    "gravity.clenshaw": MaturityLevel.MATURE,
    "gravity.models": MaturityLevel.MATURE,
    "gravity.spherical_harmonics": MaturityLevel.MATURE,
    # io
    "io.asdf_io": MaturityLevel.MATURE,
    "io.compat": MaturityLevel.MATURE,
    "io.hdf5_storage": MaturityLevel.MATURE,
    "io.hdf5_track_storage": MaturityLevel.MATURE,
    "io.migration": MaturityLevel.MATURE,
    "io.readers": MaturityLevel.MATURE,
    "io.serialize": MaturityLevel.MATURE,
    "io.session": MaturityLevel.MATURE,
    "io.sql_storage": MaturityLevel.MATURE,
    # magnetism
    "magnetism.emm": MaturityLevel.MATURE,
    "magnetism.igrf": MaturityLevel.MATURE,
    # mathematical_functions
    "mathematical_functions.basic_matrix.special_matrices": MaturityLevel.MATURE,
    "mathematical_functions.combinatorics.combinatorics": MaturityLevel.MATURE,
    "mathematical_functions.geometry.geometry": MaturityLevel.MATURE,
    "mathematical_functions.interpolation.interpolation": MaturityLevel.MATURE,
    "mathematical_functions.numerical_integration.cubature_points": MaturityLevel.MATURE,
    "mathematical_functions.numerical_integration.lcd_samples": MaturityLevel.MATURE,
    "mathematical_functions.numerical_integration.quadrature": MaturityLevel.MATURE,
    "mathematical_functions.numerical_integration.region_cubature": MaturityLevel.MATURE,
    "mathematical_functions.special_functions.bessel": MaturityLevel.MATURE,
    "mathematical_functions.special_functions.elliptic": MaturityLevel.MATURE,
    "mathematical_functions.special_functions.error_functions": MaturityLevel.MATURE,
    "mathematical_functions.special_functions.gamma_functions": MaturityLevel.MATURE,
    "mathematical_functions.special_functions.lambert_w": MaturityLevel.MATURE,
    "mathematical_functions.special_functions.marcum_q": MaturityLevel.MATURE,
    "mathematical_functions.statistics.estimators": MaturityLevel.MATURE,
    "mathematical_functions.transforms.stft": MaturityLevel.MATURE,
    # performance_evaluation
    "performance_evaluation.estimation_metrics": MaturityLevel.MATURE,
    "performance_evaluation.track_metrics": MaturityLevel.MATURE,
    # plotting
    "plotting.coordinates": MaturityLevel.MATURE,
    "plotting.ellipses": MaturityLevel.MATURE,
    # trackers
    "trackers.configs": MaturityLevel.MATURE,
    "trackers.single_target": MaturityLevel.MATURE,
    # =========================================================================
    # Added Experimental entries (see the audit note in the module docstring)
    # =========================================================================
    # astronomical
    "astronomical.tle": MaturityLevel.EXPERIMENTAL,  # 88% coverage
    # clustering
    "clustering.dbscan": MaturityLevel.EXPERIMENTAL,  # 80% coverage
    "clustering.gaussian_mixture": MaturityLevel.EXPERIMENTAL,  # 86% coverage
    "clustering.hierarchical": MaturityLevel.EXPERIMENTAL,  # 83% coverage
    # core
    "core.maturity": MaturityLevel.EXPERIMENTAL,  # behaviour changed this release
    # dynamic_models
    "dynamic_models.continuous_time.dynamics": MaturityLevel.EXPERIMENTAL,  # 88% coverage
    # gpu
    "gpu._backend": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    "gpu.ekf": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    "gpu.kalman": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    "gpu.matrix_utils": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    "gpu.particle_filter": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    "gpu.ukf": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    "gpu.utils": MaturityLevel.EXPERIMENTAL,  # CuPy branch unverifiable in CI
    # io
    "io.dataframes": MaturityLevel.EXPERIMENTAL,  # 90% coverage
    "io.storage": MaturityLevel.EXPERIMENTAL,  # 70% coverage
    "io.track_database": MaturityLevel.EXPERIMENTAL,  # behaviour changed this release
    # mathematical_functions
    "mathematical_functions.basic_matrix.decompositions": MaturityLevel.EXPERIMENTAL,  # 86% coverage
    "mathematical_functions.signal_processing.matched_filter": MaturityLevel.EXPERIMENTAL,  # 66% coverage
    "mathematical_functions.special_functions.debye": MaturityLevel.EXPERIMENTAL,  # 55% coverage
    "mathematical_functions.special_functions.hypergeometric": MaturityLevel.EXPERIMENTAL,  # 65% coverage
    "mathematical_functions.statistics.distributions": MaturityLevel.EXPERIMENTAL,  # 83% coverage
    # navigation
    "navigation.rhumb": MaturityLevel.EXPERIMENTAL,  # 85% coverage
    # plotting
    "plotting.metrics": MaturityLevel.EXPERIMENTAL,  # behaviour changed this release
    "plotting.tracks": MaturityLevel.EXPERIMENTAL,  # 85% coverage
    # coordinate_systems
    "coordinate_systems.conversions.uv": MaturityLevel.EXPERIMENTAL,  # new in v2.8.0
    # dynamic_estimation
    "dynamic_estimation.kalman.ensemble": MaturityLevel.EXPERIMENTAL,  # new this release
    "dynamic_estimation.kalman.qmc": MaturityLevel.EXPERIMENTAL,  # new this release
    "dynamic_estimation.kalman.blue": MaturityLevel.EXPERIMENTAL,  # new this release
    "dynamic_estimation.batch_estimation": MaturityLevel.EXPERIMENTAL,  # new this release
    # mathematical_functions
    "mathematical_functions.polynomials.multivariate": MaturityLevel.EXPERIMENTAL,  # new this release
    # static_estimation
    "static_estimation.localization": MaturityLevel.EXPERIMENTAL,  # new this release
    "static_estimation.maximum_likelihood": MaturityLevel.EXPERIMENTAL,  # 90% coverage
    # terrain
    "terrain.visibility": MaturityLevel.EXPERIMENTAL,  # 88% coverage
    # trackers
    "trackers.hypothesis": MaturityLevel.EXPERIMENTAL,  # 81% coverage
    "trackers.multi_target": MaturityLevel.EXPERIMENTAL,  # behaviour changed this release
    # transponders
    "transponders.ais": MaturityLevel.EXPERIMENTAL,  # behaviour changed this release
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
