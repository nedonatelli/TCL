"""
Optional dependencies management for the Tracker Component Library.

This module provides a unified system for handling optional dependencies,
including lazy imports, availability checks, and helpful error messages.

The system supports:
- Lazy imports that only load modules when accessed
- Availability flags for conditional code paths
- Decorators for functions requiring optional dependencies
- Helpful error messages with installation instructions

Examples
--------
Check if a dependency is available:

>>> from pytcl.core.optional_deps import is_available
>>> if is_available("plotly"):
...     import plotly.graph_objects as go

Use a decorator to require a dependency:

>>> from pytcl.core.optional_deps import requires
>>> @requires("plotly", extra="visualization")
... def create_3d_plot(data):
...     import plotly.graph_objects as go
...     return go.Figure(data)

Import with a helpful error on failure:

>>> from pytcl.core.optional_deps import import_optional
>>> go = import_optional("plotly.graph_objects", package="plotly", extra="visualization")
"""

import importlib
import logging
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar

from pytcl.core.exceptions import DependencyError

# Module logger
_logger = logging.getLogger("pytcl.core.optional_deps")

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Package Configuration
# =============================================================================

# Mapping of package names to their pip install extras
# Format: package_name -> (extra_name, pip_package_name)
PACKAGE_EXTRAS: dict[str, tuple[str, str]] = {
    # Visualization
    "plotly": ("visualization", "plotly"),
    # Astronomy
    "astropy": ("astronomy", "astropy"),
    "jplephem": ("astronomy", "jplephem"),
    # Geodesy
    "pyproj": ("geodesy", "pyproj"),
    "geographiclib": ("geodesy", "geographiclib"),
    # Optimization
    "cvxpy": ("optimization", "cvxpy"),
    # Signal processing
    "pywt": ("signal", "pywavelets"),
    "pywavelets": ("signal", "pywavelets"),
    # Terrain data
    "netCDF4": ("terrain", "netCDF4"),
    # GPU acceleration
    "cupy": ("gpu", "cupy-cuda12x"),
    # Apple Silicon GPU acceleration
    "mlx": ("gpu-apple", "mlx"),
}

# Friendly names for features provided by each package
PACKAGE_FEATURES: dict[str, str] = {
    "plotly": "interactive visualization",
    "astropy": "astronomical calculations",
    "jplephem": "JPL ephemeris access",
    "pyproj": "coordinate transformations",
    "geographiclib": "geodetic calculations",
    "cvxpy": "convex optimization",
    "pywt": "wavelet transforms",
    "pywavelets": "wavelet transforms",
    "netCDF4": "NetCDF file reading",
    "cupy": "GPU acceleration",
    "mlx": "Apple Silicon GPU acceleration",
}


# =============================================================================
# Availability Cache
# =============================================================================

# Cache of package availability checks
_availability_cache: dict[str, bool] = {}


def _clear_cache() -> None:
    """Clear the availability cache. Mainly for testing."""
    _availability_cache.clear()


def is_available(package: str) -> bool:
    """
    Check if an optional package is available.

    Parameters
    ----------
    package : str
        Name of the package to check (e.g., "plotly", "pywt").

    Returns
    -------
    bool
        True if the package is installed and can be imported.

    Examples
    --------
    >>> from pytcl.core.optional_deps import is_available
    >>> if is_available("plotly"):
    ...     from plotly import graph_objects as go
    ...     # use plotly
    ... else:
    ...     print("Plotly not available")

    Notes
    -----
    Results are cached for performance. Use ``_clear_cache()`` if you
    need to re-check availability (e.g., after installing a package).
    """
    if package in _availability_cache:
        return _availability_cache[package]

    try:
        importlib.import_module(package)
        available = True
        _logger.debug("Optional package '%s' is available", package)
    except ImportError:
        available = False
        _logger.debug("Optional package '%s' is not available", package)

    _availability_cache[package] = available
    return available


# =============================================================================
# Import Helpers
# =============================================================================


def _get_install_command(package: str, extra: Optional[str] = None) -> str:
    """Generate the pip install command for a package."""
    if extra:
        return f"pip install pytcl[{extra}]"

    # Check if we know the extra for this package
    if package in PACKAGE_EXTRAS:
        extra_name, _ = PACKAGE_EXTRAS[package]
        return f"pip install pytcl[{extra_name}]"

    # Default to direct package install
    pip_package = package
    if package in PACKAGE_EXTRAS:
        _, pip_package = PACKAGE_EXTRAS[package]
    return f"pip install {pip_package}"


def _get_feature_name(package: str) -> str:
    """Get a friendly feature name for a package."""
    return PACKAGE_FEATURES.get(package, f"{package} functionality")


def import_optional(
    module_name: str,
    *,
    package: Optional[str] = None,
    extra: Optional[str] = None,
    feature: Optional[str] = None,
) -> ModuleType:
    """
    Import an optional module with a helpful error message on failure.

    Parameters
    ----------
    module_name : str
        Full module path to import (e.g., "plotly.graph_objects").
    package : str, optional
        Package name for error message. If not provided, extracted from
        module_name.
    extra : str, optional
        Name of the pytcl extra that provides this dependency
        (e.g., "visualization", "astronomy").
    feature : str, optional
        Description of the feature requiring this dependency.

    Returns
    -------
    module : ModuleType
        The imported module.

    Raises
    ------
    DependencyError
        If the module cannot be imported.

    Examples
    --------
    >>> go = import_optional(
    ...     "plotly.graph_objects",
    ...     package="plotly",
    ...     extra="visualization",
    ...     feature="3D plotting"
    ... )
    """
    if package is None:
        package = module_name.split(".")[0]

    try:
        module = importlib.import_module(module_name)
        _logger.debug("Successfully imported optional module '%s'", module_name)
        return module
    except ImportError as e:
        if feature is None:
            feature = _get_feature_name(package)

        install_cmd = _get_install_command(package, extra)

        msg = f"{package} is required for {feature}. " f"Install with: {install_cmd}"
        _logger.warning("Failed to import optional module '%s': %s", module_name, e)
        raise DependencyError(
            msg,
            package=package,
            feature=feature,
            install_command=install_cmd,
        ) from e


# =============================================================================
# Decorator for Optional Dependencies
# =============================================================================


def requires(
    *packages: str,
    extra: Optional[str] = None,
    feature: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark a function as requiring optional dependencies.

    When the decorated function is called, it checks if the required
    packages are available. If not, it raises a DependencyError with
    a helpful message.

    Parameters
    ----------
    *packages : str
        One or more package names required by the function.
    extra : str, optional
        Name of the pytcl extra that provides these dependencies.
    feature : str, optional
        Description of the feature provided by the function.

    Returns
    -------
    decorator : callable
        Decorator that wraps the function with dependency checking.

    Examples
    --------
    >>> from pytcl.core.optional_deps import requires
    >>>
    >>> @requires("plotly", extra="visualization")
    ... def create_plot(data):
    ...     import plotly.graph_objects as go
    ...     return go.Figure(data)
    >>>
    >>> # This will raise DependencyError if plotly is not installed
    >>> create_plot([1, 2, 3])

    Multiple packages:

    >>> @requires("astropy", "jplephem", extra="astronomy")
    ... def compute_ephemeris(body, time):
    ...     from astropy.time import Time
    ...     import jplephem
    ...     # ...

    Notes
    -----
    The decorator checks availability at call time, not at definition
    time. This allows the module to be imported even if the optional
    dependencies are not installed.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            missing = [pkg for pkg in packages if not is_available(pkg)]

            if missing:
                # Get feature name from first package if not provided
                feat = feature or _get_feature_name(missing[0])

                if len(missing) == 1:
                    pkg_str = missing[0]
                    install_cmd = _get_install_command(missing[0], extra)
                else:
                    pkg_str = ", ".join(missing)
                    install_cmd = _get_install_command(missing[0], extra)

                msg = (
                    f"{pkg_str} {'is' if len(missing) == 1 else 'are'} required "
                    f"for {feat}. Install with: {install_cmd}"
                )
                raise DependencyError(
                    msg,
                    package=missing[0],
                    feature=feat,
                    install_command=install_cmd,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Availability Flags (for backward compatibility)
# =============================================================================


# These flags are computed lazily when first accessed
class _AvailabilityFlags:
    """Lazy availability flags for common optional packages."""

    @property
    def HAS_PLOTLY(self) -> bool:
        """True if plotly is available."""
        return is_available("plotly")

    @property
    def HAS_PYWT(self) -> bool:
        """True if pywavelets is available."""
        return is_available("pywt")

    @property
    def PYWT_AVAILABLE(self) -> bool:
        """True if pywavelets is available (alias)."""
        return is_available("pywt")

    @property
    def HAS_JPLEPHEM(self) -> bool:
        """True if jplephem is available."""
        return is_available("jplephem")

    @property
    def HAS_ASTROPY(self) -> bool:
        """True if astropy is available."""
        return is_available("astropy")

    @property
    def HAS_PYPROJ(self) -> bool:
        """True if pyproj is available."""
        return is_available("pyproj")

    @property
    def HAS_CVXPY(self) -> bool:
        """True if cvxpy is available."""
        return is_available("cvxpy")

    @property
    def HAS_NETCDF4(self) -> bool:
        """True if netCDF4 is available."""
        return is_available("netCDF4")

    @property
    def HAS_CUPY(self) -> bool:
        """True if cupy is available."""
        return is_available("cupy")

    @property
    def HAS_MLX(self) -> bool:
        """True if mlx is available (Apple Silicon)."""
        return is_available("mlx")


# Create singleton instance
_flags = _AvailabilityFlags()

# Export individual flags for convenient access
HAS_PLOTLY = property(lambda self: _flags.HAS_PLOTLY)
HAS_PYWT = property(lambda self: _flags.HAS_PYWT)
PYWT_AVAILABLE = property(lambda self: _flags.PYWT_AVAILABLE)
HAS_JPLEPHEM = property(lambda self: _flags.HAS_JPLEPHEM)
HAS_ASTROPY = property(lambda self: _flags.HAS_ASTROPY)
HAS_PYPROJ = property(lambda self: _flags.HAS_PYPROJ)
HAS_CVXPY = property(lambda self: _flags.HAS_CVXPY)
HAS_NETCDF4 = property(lambda self: _flags.HAS_NETCDF4)
HAS_CUPY = property(lambda self: _flags.HAS_CUPY)
HAS_MLX = property(lambda self: _flags.HAS_MLX)


# =============================================================================
# Lazy Module Loader
# =============================================================================


class LazyModule:
    """
    A lazy module loader that imports the module on first access.

    This allows optional dependencies to be "imported" at module level
    without triggering an import error until they're actually used.

    Parameters
    ----------
    module_name : str
        Full module path to import.
    package : str, optional
        Package name for error messages.
    extra : str, optional
        pytcl extra that provides this dependency.
    feature : str, optional
        Feature description for error messages.

    Examples
    --------
    >>> from pytcl.core.optional_deps import LazyModule
    >>> go = LazyModule("plotly.graph_objects", package="plotly")
    >>> # No import yet...
    >>> fig = go.Figure()  # Import happens here
    """

    def __init__(
        self,
        module_name: str,
        *,
        package: Optional[str] = None,
        extra: Optional[str] = None,
        feature: Optional[str] = None,
    ):
        self._module_name = module_name
        self._package = package or module_name.split(".")[0]
        self._extra = extra
        self._feature = feature
        self._module: Optional[ModuleType] = None

    def _load(self) -> ModuleType:
        """Load the module if not already loaded."""
        if self._module is None:
            self._module = import_optional(
                self._module_name,
                package=self._package,
                extra=self._extra,
                feature=self._feature,
            )
        return self._module

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the loaded module."""
        module = self._load()
        return getattr(module, name)

    def __dir__(self) -> list[str]:
        """Return module attributes for tab completion."""
        try:
            module = self._load()
            return dir(module)
        except DependencyError:
            return []


# =============================================================================
# Convenience Functions
# =============================================================================


def check_dependencies(*packages: str, extra: Optional[str] = None) -> None:
    """
    Check that all required packages are available.

    Parameters
    ----------
    *packages : str
        Package names to check.
    extra : str, optional
        pytcl extra for installation hint.

    Raises
    ------
    DependencyError
        If any package is not available.

    Examples
    --------
    >>> from pytcl.core.optional_deps import check_dependencies
    >>> check_dependencies("plotly", extra="visualization")
    >>> # Raises DependencyError if plotly is not installed
    """
    missing = [pkg for pkg in packages if not is_available(pkg)]

    if missing:
        feature = _get_feature_name(missing[0])
        install_cmd = _get_install_command(missing[0], extra)

        if len(missing) == 1:
            msg = f"{missing[0]} is required. Install with: {install_cmd}"
        else:
            msg = f"{', '.join(missing)} are required. Install with: {install_cmd}"

        raise DependencyError(
            msg,
            package=missing[0],
            feature=feature,
            install_command=install_cmd,
        )


__all__ = [
    # Core functions
    "is_available",
    "import_optional",
    "requires",
    "check_dependencies",
    # Lazy loading
    "LazyModule",
    # Configuration
    "PACKAGE_EXTRAS",
    "PACKAGE_FEATURES",
    # Availability flags (backward compatibility)
    "HAS_PLOTLY",
    "HAS_PYWT",
    "PYWT_AVAILABLE",
    "HAS_JPLEPHEM",
    "HAS_ASTROPY",
    "HAS_PYPROJ",
    "HAS_CVXPY",
    "HAS_NETCDF4",
    "HAS_CUPY",
    "HAS_MLX",
    # Internal (for testing)
    "_clear_cache",
    "_flags",
]
