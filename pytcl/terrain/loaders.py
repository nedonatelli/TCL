"""
GEBCO and Earth2014 terrain data loaders.

This module provides functions for loading elevation data from:
- GEBCO (General Bathymetric Chart of the Oceans): Global bathymetry and topography
- Earth2014: High-resolution global terrain model with multiple representations

Both datasets require external files that are too large to bundle with the package.
Users should download files from the official sources and place them in ~/.pytcl/data/.

References
----------
.. [1] GEBCO Compilation Group (2024) GEBCO 2024 Grid.
       https://www.gebco.net/data-products/gridded-bathymetry-data/
.. [2] Hirt, C. and Rexer, M. (2015) Earth2014: 1 arc-min shape, topography,
       bedrock and ice-sheet models - available as gridded data and degree-10,800
       spherical harmonics. Int. J. Appl. Earth Observ. Geoinform. 39, 103-112.
       https://ddfe.curtin.edu.au/models/Earth2014/
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from pytcl.core.exceptions import DependencyError

from .dem import DEMGrid

# Model parameters
_GEBCO_BASE_URL = "https://www.gebco.net/data-products/gridded-bathymetry-data"

GEBCO_PARAMETERS: dict[str, dict[str, Any]] = {
    "GEBCO2024": {
        "resolution_arcsec": 15.0,
        "n_lat": 43200,
        "n_lon": 86400,
        "format": "NetCDF",
        "file_size_mb": 7500,
        "url": f"{_GEBCO_BASE_URL}/gebco2024-grid",
    },
    "GEBCO2023": {
        "resolution_arcsec": 15.0,
        "n_lat": 43200,
        "n_lon": 86400,
        "format": "NetCDF",
        "file_size_mb": 7500,
        "url": f"{_GEBCO_BASE_URL}/gebco2023-grid",
    },
    "GEBCO2022": {
        "resolution_arcsec": 15.0,
        "n_lat": 43200,
        "n_lon": 86400,
        "format": "NetCDF",
        "file_size_mb": 7500,
        "url": f"{_GEBCO_BASE_URL}/",
    },
}

EARTH2014_PARAMETERS: dict[str, dict[str, Any]] = {
    "SUR": {
        "description": "Physical surface (topography, ice surface, 0 over oceans)",
        "file_pattern": "Earth2014.SUR2014.1min.geod.bin",
    },
    "BED": {
        "description": "Bedrock (topography minus ice/water)",
        "file_pattern": "Earth2014.BED2014.1min.geod.bin",
    },
    "TBI": {
        "description": "Topography, bedrock, ice (land/ocean/ice surface)",
        "file_pattern": "Earth2014.TBI2014.1min.geod.bin",
    },
    "RET": {
        "description": "Rock-equivalent topography (2670 kg/m3 density)",
        "file_pattern": "Earth2014.RET2014.1min.geod.bin",
    },
    "ICE": {
        "description": "Ice sheet thickness",
        "file_pattern": "Earth2014.ICE2014.1min.geod.bin",
    },
}

# Earth2014 grid parameters (1 arc-minute resolution)
EARTH2014_N_LAT = 10800
EARTH2014_N_LON = 21600
EARTH2014_RESOLUTION_ARCSEC = 60.0  # 1 arc-minute
EARTH2014_FILE_SIZE_MB = 455


class GEBCOMetadata(NamedTuple):
    """GEBCO dataset metadata.

    Attributes
    ----------
    version : str
        GEBCO version (e.g., "GEBCO2024").
    resolution_arcsec : float
        Grid resolution in arc-seconds.
    lat_min : float
        Minimum latitude in radians.
    lat_max : float
        Maximum latitude in radians.
    lon_min : float
        Minimum longitude in radians.
    lon_max : float
        Maximum longitude in radians.
    source : str
        Data source identifier.
    """

    version: str
    resolution_arcsec: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    source: str


class Earth2014Metadata(NamedTuple):
    """Earth2014 dataset metadata.

    Attributes
    ----------
    layer : str
        Layer type (SUR, BED, TBI, RET, ICE).
    description : str
        Layer description.
    resolution_arcsec : float
        Grid resolution in arc-seconds.
    lat_min : float
        Minimum latitude in radians.
    lat_max : float
        Maximum latitude in radians.
    lon_min : float
        Minimum longitude in radians.
    lon_max : float
        Maximum longitude in radians.
    """

    layer: str
    description: str
    resolution_arcsec: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def get_data_dir() -> Path:
    """Get the pytcl data directory for terrain files.

    The data directory is located at ~/.pytcl/data/ by default.
    Can be overridden by setting the PYTCL_DATA_DIR environment variable.

    Returns
    -------
    Path
        Path to the data directory.
    """
    env_dir = os.environ.get("PYTCL_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        data_dir = Path.home() / ".pytcl" / "data"

    return data_dir


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _find_gebco_file(version: str = "GEBCO2024") -> Path:
    """Find GEBCO NetCDF file in data directory.

    Parameters
    ----------
    version : str
        GEBCO version (e.g., "GEBCO2024", "GEBCO2023").

    Returns
    -------
    Path
        Path to the GEBCO file.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    data_dir = get_data_dir()

    # Try common file patterns
    patterns = [
        f"{version}.nc",
        f"{version}_*.nc",
        f"{version.lower()}.nc",
        f"GEBCO_{version[-4:]}_Grid.nc",
        f"gebco_{version[-4:]}.nc",
    ]

    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            return matches[0]

    url = GEBCO_PARAMETERS.get(version, {}).get("url", "https://www.gebco.net/")
    raise FileNotFoundError(
        f"GEBCO file not found for {version}\n"
        f"Please download from: {url}\n"
        f"Save as: {data_dir}/{version}.nc\n"
        f"Or use create_test_gebco_dem() for testing."
    )


def _find_earth2014_file(layer: str = "SUR") -> Path:
    """Find Earth2014 binary file in data directory.

    Parameters
    ----------
    layer : str
        Earth2014 layer (SUR, BED, TBI, RET, ICE).

    Returns
    -------
    Path
        Path to the Earth2014 file.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if layer not in EARTH2014_PARAMETERS:
        raise ValueError(
            f"Unknown Earth2014 layer: {layer}. "
            f"Valid layers: {list(EARTH2014_PARAMETERS.keys())}"
        )

    data_dir = get_data_dir()
    file_pattern = EARTH2014_PARAMETERS[layer]["file_pattern"]
    filepath = data_dir / file_pattern

    if filepath.exists():
        return filepath

    # Try alternative patterns
    alt_patterns = [
        f"earth2014.{layer.lower()}2014.1min.geod.bin",
        f"{layer}2014.1min.geod.bin",
        f"Earth2014_{layer}.bin",
    ]

    for pattern in alt_patterns:
        alt_path = data_dir / pattern
        if alt_path.exists():
            return alt_path

    raise FileNotFoundError(
        f"Earth2014 {layer} file not found.\n"
        f"Please download from: https://ddfe.curtin.edu.au/models/Earth2014/\n"
        f"Save as: {data_dir}/{file_pattern}\n"
        f"Or use create_test_earth2014_dem() for testing."
    )


def parse_gebco_netcdf(
    filepath: Path,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
) -> tuple[NDArray[np.floating], float, float, float, float]:
    """Parse GEBCO NetCDF file and extract region.

    Parameters
    ----------
    filepath : Path
        Path to GEBCO NetCDF file.
    lat_min : float, optional
        Minimum latitude in radians (default: -90°).
    lat_max : float, optional
        Maximum latitude in radians (default: +90°).
    lon_min : float, optional
        Minimum longitude in radians (default: -180°).
    lon_max : float, optional
        Maximum longitude in radians (default: +180°).

    Returns
    -------
    data : ndarray
        Elevation data array.
    lat_min_actual : float
        Actual minimum latitude of extracted region.
    lat_max_actual : float
        Actual maximum latitude of extracted region.
    lon_min_actual : float
        Actual minimum longitude of extracted region.
    lon_max_actual : float
        Actual maximum longitude of extracted region.
    """
    try:
        import netCDF4 as nc
    except ImportError as e:
        raise DependencyError(
            "netCDF4 is required for loading GEBCO files.",
            package="netCDF4",
            feature="NetCDF file reading",
            install_command="pip install pytcl[terrain]",
        ) from e

    # Set defaults for global extent
    if lat_min is None:
        lat_min = np.radians(-90.0)
    if lat_max is None:
        lat_max = np.radians(90.0)
    if lon_min is None:
        lon_min = np.radians(-180.0)
    if lon_max is None:
        lon_max = np.radians(180.0)

    with nc.Dataset(filepath, "r") as dataset:
        # Get coordinate arrays
        lats = dataset.variables["lat"][:]
        lons = dataset.variables["lon"][:]

        # Convert bounds to degrees for indexing
        lat_min_deg = np.degrees(lat_min)
        lat_max_deg = np.degrees(lat_max)
        lon_min_deg = np.degrees(lon_min)
        lon_max_deg = np.degrees(lon_max)

        # Find index ranges
        lat_indices = np.where((lats >= lat_min_deg) & (lats <= lat_max_deg))[0]
        lon_indices = np.where((lons >= lon_min_deg) & (lons <= lon_max_deg))[0]

        if len(lat_indices) == 0 or len(lon_indices) == 0:
            raise ValueError("Requested region is outside GEBCO coverage")

        i_start, i_end = lat_indices[0], lat_indices[-1] + 1
        j_start, j_end = lon_indices[0], lon_indices[-1] + 1

        # Extract elevation data
        # GEBCO stores elevation in 'elevation' variable
        elevation = dataset.variables["elevation"][i_start:i_end, j_start:j_end]

        # Get actual bounds
        lat_min_actual = np.radians(float(lats[i_start]))
        lat_max_actual = np.radians(float(lats[i_end - 1]))
        lon_min_actual = np.radians(float(lons[j_start]))
        lon_max_actual = np.radians(float(lons[j_end - 1]))

    return (
        np.asarray(elevation, dtype=np.float64),
        lat_min_actual,
        lat_max_actual,
        lon_min_actual,
        lon_max_actual,
    )


def parse_earth2014_binary(
    filepath: Path,
    layer: str,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
) -> tuple[NDArray[np.floating], float, float, float, float]:
    """Parse Earth2014 binary file and extract region.

    Earth2014 files are stored as int16 big-endian binary data,
    organized as rows from south to north, columns from west to east.

    Parameters
    ----------
    filepath : Path
        Path to Earth2014 binary file.
    layer : str
        Layer type (SUR, BED, TBI, RET, ICE).
    lat_min : float, optional
        Minimum latitude in radians (default: -90°).
    lat_max : float, optional
        Maximum latitude in radians (default: +90°).
    lon_min : float, optional
        Minimum longitude in radians (default: -180°).
    lon_max : float, optional
        Maximum longitude in radians (default: +180°).

    Returns
    -------
    data : ndarray
        Elevation data array in meters.
    lat_min_actual : float
        Actual minimum latitude of extracted region.
    lat_max_actual : float
        Actual maximum latitude of extracted region.
    lon_min_actual : float
        Actual minimum longitude of extracted region.
    lon_max_actual : float
        Actual maximum longitude of extracted region.
    """
    # Set defaults for global extent
    if lat_min is None:
        lat_min = np.radians(-90.0)
    if lat_max is None:
        lat_max = np.radians(90.0)
    if lon_min is None:
        lon_min = np.radians(-180.0)
    if lon_max is None:
        lon_max = np.radians(180.0)

    # Earth2014 grid parameters
    # Grid is cell-centered, excluding poles and integer meridians
    # First point: (-89.9917°, -179.9917°)
    # Last point: (89.9917°, 179.9917°)
    d_lat = np.radians(1.0 / 60.0)  # 1 arc-minute
    d_lon = np.radians(1.0 / 60.0)

    # Cell centers start half a cell from -90/-180
    lat_start = np.radians(-90.0 + 1.0 / 120.0)  # -89.9917°
    lon_start = np.radians(-180.0 + 1.0 / 120.0)  # -179.9917°

    # Convert requested bounds to indices
    lat_min_deg = np.degrees(lat_min)
    lat_max_deg = np.degrees(lat_max)
    lon_min_deg = np.degrees(lon_min)
    lon_max_deg = np.degrees(lon_max)

    # Compute row/column indices
    i_start = max(0, int(np.floor((np.radians(lat_min_deg) - lat_start) / d_lat)))
    i_end = min(
        EARTH2014_N_LAT, int(np.ceil((np.radians(lat_max_deg) - lat_start) / d_lat)) + 1
    )
    j_start = max(0, int(np.floor((np.radians(lon_min_deg) - lon_start) / d_lon)))
    j_end = min(
        EARTH2014_N_LON, int(np.ceil((np.radians(lon_max_deg) - lon_start) / d_lon)) + 1
    )

    # Read binary data
    # File is stored as int16 big-endian, rows from south to north
    n_rows = i_end - i_start
    n_cols = j_end - j_start

    data = np.zeros((n_rows, n_cols), dtype=np.float64)

    with open(filepath, "rb") as f:
        for i, row_idx in enumerate(range(i_start, i_end)):
            # Seek to start of row
            row_offset = row_idx * EARTH2014_N_LON * 2  # 2 bytes per int16
            col_offset = j_start * 2
            f.seek(row_offset + col_offset)

            # Read row segment
            row_data = np.frombuffer(
                f.read(n_cols * 2), dtype=">i2"
            )  # big-endian int16
            data[i, :] = row_data.astype(np.float64)

    # Compute actual bounds
    lat_min_actual = lat_start + i_start * d_lat
    lat_max_actual = lat_start + (i_end - 1) * d_lat
    lon_min_actual = lon_start + j_start * d_lon
    lon_max_actual = lon_start + (j_end - 1) * d_lon

    return data, lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual


@lru_cache(maxsize=8)
def _load_gebco_cached(
    version: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> DEMGrid:
    """Cached GEBCO loading (internal function)."""
    filepath = _find_gebco_file(version)
    data, lat_min_a, lat_max_a, lon_min_a, lon_max_a = parse_gebco_netcdf(
        filepath, lat_min, lat_max, lon_min, lon_max
    )

    return DEMGrid(
        data,
        lat_min_a,
        lat_max_a,
        lon_min_a,
        lon_max_a,
        nodata_value=-9999.0,
        name=version,
    )


@lru_cache(maxsize=8)
def _load_earth2014_cached(
    layer: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> DEMGrid:
    """Cached Earth2014 loading (internal function)."""
    filepath = _find_earth2014_file(layer)
    data, lat_min_a, lat_max_a, lon_min_a, lon_max_a = parse_earth2014_binary(
        filepath, layer, lat_min, lat_max, lon_min, lon_max
    )

    return DEMGrid(
        data,
        lat_min_a,
        lat_max_a,
        lon_min_a,
        lon_max_a,
        nodata_value=-9999.0,
        name=f"Earth2014-{layer}",
    )


def load_gebco(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    version: str = "GEBCO2024",
) -> DEMGrid:
    """Load GEBCO bathymetry/topography data for a region.

    GEBCO (General Bathymetric Chart of the Oceans) provides global
    bathymetry and land topography at 15 arc-second resolution.

    Parameters
    ----------
    lat_min : float
        Minimum latitude in radians.
    lat_max : float
        Maximum latitude in radians.
    lon_min : float
        Minimum longitude in radians.
    lon_max : float
        Maximum longitude in radians.
    version : str, optional
        GEBCO version ("GEBCO2024", "GEBCO2023", "GEBCO2022").
        Default is "GEBCO2024".

    Returns
    -------
    DEMGrid
        DEM grid containing bathymetry/topography data.

    Raises
    ------
    FileNotFoundError
        If the GEBCO file is not found.
    DependencyError
        If netCDF4 is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> dem = load_gebco(
    ...     lat_min=np.radians(35.0),
    ...     lat_max=np.radians(40.0),
    ...     lon_min=np.radians(-125.0),
    ...     lon_max=np.radians(-120.0),
    ...     version="GEBCO2024"
    ... )
    >>> point = dem.get_elevation(np.radians(37.5), np.radians(-122.5))
    >>> print(f"Elevation: {point.elevation:.1f} m")

    Notes
    -----
    GEBCO data files are not included in the package due to their size (~7.5 GB).
    Download from: https://www.gebco.net/data-products/gridded-bathymetry-data/
    Save to: ~/.pytcl/data/GEBCO2024.nc
    """
    if version not in GEBCO_PARAMETERS:
        raise ValueError(
            f"Unknown GEBCO version: {version}. "
            f"Valid versions: {list(GEBCO_PARAMETERS.keys())}"
        )

    return _load_gebco_cached(version, lat_min, lat_max, lon_min, lon_max)


def load_earth2014(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    layer: str = "SUR",
) -> DEMGrid:
    """Load Earth2014 terrain data for a region.

    Earth2014 provides global topography at 1 arc-minute resolution with
    multiple layer representations.

    Parameters
    ----------
    lat_min : float
        Minimum latitude in radians.
    lat_max : float
        Maximum latitude in radians.
    lon_min : float
        Minimum longitude in radians.
    lon_max : float
        Maximum longitude in radians.
    layer : str, optional
        Layer type. Options:
        - "SUR": Physical surface (topography, ice surface, 0 over oceans)
        - "BED": Bedrock (topography minus ice/water)
        - "TBI": Topography, bedrock, ice sheet surfaces
        - "RET": Rock-equivalent topography
        - "ICE": Ice sheet thickness
        Default is "SUR".

    Returns
    -------
    DEMGrid
        DEM grid containing terrain data.

    Raises
    ------
    FileNotFoundError
        If the Earth2014 file is not found.
    ValueError
        If an invalid layer is specified.

    Examples
    --------
    >>> import numpy as np
    >>> dem = load_earth2014(
    ...     lat_min=np.radians(35.0),
    ...     lat_max=np.radians(40.0),
    ...     lon_min=np.radians(-125.0),
    ...     lon_max=np.radians(-120.0),
    ...     layer="SUR"
    ... )
    >>> point = dem.get_elevation(np.radians(37.5), np.radians(-122.5))
    >>> print(f"Elevation: {point.elevation:.1f} m")

    Notes
    -----
    Earth2014 data files are not included in the package (~455 MB per layer).
    Download from: https://ddfe.curtin.edu.au/models/Earth2014/
    Save to: ~/.pytcl/data/Earth2014.SUR2014.1min.geod.bin (for SUR layer)

    References
    ----------
    Hirt, C. and Rexer, M. (2015) Earth2014: 1 arc-min shape, topography,
    bedrock and ice-sheet models. Int. J. Appl. Earth Observ. Geoinform. 39.
    """
    if layer not in EARTH2014_PARAMETERS:
        raise ValueError(
            f"Unknown Earth2014 layer: {layer}. "
            f"Valid layers: {list(EARTH2014_PARAMETERS.keys())}"
        )

    return _load_earth2014_cached(layer, lat_min, lat_max, lon_min, lon_max)


def create_test_gebco_dem(
    lat_min: float = np.radians(35.0),
    lat_max: float = np.radians(40.0),
    lon_min: float = np.radians(-125.0),
    lon_max: float = np.radians(-120.0),
    resolution_arcsec: float = 60.0,
    include_bathymetry: bool = True,
    seed: Optional[int] = 42,
) -> DEMGrid:
    """Create synthetic GEBCO-like DEM for testing.

    Generates a synthetic terrain with both land and ocean areas,
    useful for testing without requiring actual GEBCO data files.

    Parameters
    ----------
    lat_min : float, optional
        Minimum latitude in radians. Default is 35°N.
    lat_max : float, optional
        Maximum latitude in radians. Default is 40°N.
    lon_min : float, optional
        Minimum longitude in radians. Default is 125°W.
    lon_max : float, optional
        Maximum longitude in radians. Default is 120°W.
    resolution_arcsec : float, optional
        Grid resolution in arc-seconds. Default is 60.0 (1 arc-min).
    include_bathymetry : bool, optional
        If True, includes ocean bathymetry. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    DEMGrid
        Synthetic DEM mimicking GEBCO characteristics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute grid dimensions
    d_lat = np.radians(resolution_arcsec / 3600)
    d_lon = np.radians(resolution_arcsec / 3600)

    n_lat = max(2, int(np.ceil((lat_max - lat_min) / d_lat)) + 1)
    n_lon = max(2, int(np.ceil((lon_max - lon_min) / d_lon)) + 1)

    # Create coordinate arrays
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Generate terrain
    # Land areas: positive elevation up to 4000m
    # Ocean areas: negative depth down to -6000m

    # Create a coastline pattern
    coastline = 0.5 + 0.3 * np.sin(3 * lat_grid) + 0.2 * np.cos(5 * lon_grid)
    is_ocean = (lon_grid - lon_min) / (lon_max - lon_min) > coastline

    # Land topography
    land_elev = 500 + 1500 * np.sin(10 * lat_grid) * np.cos(8 * lon_grid)
    land_elev += 500 * np.sin(20 * lon_grid)
    land_elev = np.maximum(land_elev, 0)

    # Ocean bathymetry
    if include_bathymetry:
        ocean_depth = -2000 - 2000 * np.sin(5 * lon_grid) * np.cos(7 * lat_grid)
        ocean_depth -= 1000 * np.sin(15 * lat_grid)
        ocean_depth = np.minimum(ocean_depth, -100)
    else:
        ocean_depth = np.zeros_like(land_elev)

    # Combine
    elevation = np.where(is_ocean, ocean_depth, land_elev)

    # Add random noise
    noise = np.random.randn(n_lat, n_lon) * 50
    elevation += noise

    return DEMGrid(
        elevation,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        nodata_value=-9999.0,
        name="GEBCO_TEST",
    )


def create_test_earth2014_dem(
    lat_min: float = np.radians(35.0),
    lat_max: float = np.radians(40.0),
    lon_min: float = np.radians(-125.0),
    lon_max: float = np.radians(-120.0),
    layer: str = "SUR",
    resolution_arcsec: float = 60.0,
    seed: Optional[int] = 42,
) -> DEMGrid:
    """Create synthetic Earth2014-like DEM for testing.

    Generates a synthetic terrain mimicking Earth2014 characteristics,
    useful for testing without requiring actual data files.

    Parameters
    ----------
    lat_min : float, optional
        Minimum latitude in radians. Default is 35°N.
    lat_max : float, optional
        Maximum latitude in radians. Default is 40°N.
    lon_min : float, optional
        Minimum longitude in radians. Default is 125°W.
    lon_max : float, optional
        Maximum longitude in radians. Default is 120°W.
    layer : str, optional
        Layer type to simulate. Default is "SUR".
    resolution_arcsec : float, optional
        Grid resolution in arc-seconds. Default is 60.0 (1 arc-min).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    DEMGrid
        Synthetic DEM mimicking Earth2014 characteristics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute grid dimensions
    d_lat = np.radians(resolution_arcsec / 3600)
    d_lon = np.radians(resolution_arcsec / 3600)

    n_lat = max(2, int(np.ceil((lat_max - lat_min) / d_lat)) + 1)
    n_lon = max(2, int(np.ceil((lon_max - lon_min) / d_lon)) + 1)

    # Create coordinate arrays
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Generate base topography
    base_elev = 500 + 1500 * np.sin(10 * lat_grid) * np.cos(8 * lon_grid)
    base_elev += 500 * np.sin(20 * lon_grid) + 300 * np.cos(15 * lat_grid)

    if layer == "SUR":
        # Physical surface: positive values only (0 over oceans)
        elevation = np.maximum(base_elev, 0)
    elif layer == "BED":
        # Bedrock: can be negative (below sea level)
        elevation = base_elev - 200  # Bedrock slightly below surface
    elif layer == "TBI":
        # Topography, bedrock, ice: mixed representation
        elevation = base_elev
    elif layer == "RET":
        # Rock-equivalent: adjusted for density
        elevation = base_elev * 0.9  # Simplified approximation
    elif layer == "ICE":
        # Ice thickness: only in certain regions
        # Simulate some ice coverage at high latitudes
        ice_coverage = lat_grid > np.radians(38.0)
        elevation = np.where(ice_coverage, 500 + 200 * np.random.rand(n_lat, n_lon), 0)
    else:
        elevation = base_elev

    # Add random noise
    noise = np.random.randn(n_lat, n_lon) * 30
    elevation += noise

    return DEMGrid(
        elevation,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        nodata_value=-9999.0,
        name=f"Earth2014_TEST_{layer}",
    )


def get_gebco_metadata(version: str = "GEBCO2024") -> GEBCOMetadata:
    """Get metadata for a GEBCO version.

    Parameters
    ----------
    version : str
        GEBCO version.

    Returns
    -------
    GEBCOMetadata
        Metadata about the dataset.
    """
    if version not in GEBCO_PARAMETERS:
        raise ValueError(f"Unknown GEBCO version: {version}")

    params = GEBCO_PARAMETERS[version]
    return GEBCOMetadata(
        version=version,
        resolution_arcsec=params["resolution_arcsec"],
        lat_min=np.radians(-90.0),
        lat_max=np.radians(90.0),
        lon_min=np.radians(-180.0),
        lon_max=np.radians(180.0),
        source=params["url"],
    )


def get_earth2014_metadata(layer: str = "SUR") -> Earth2014Metadata:
    """Get metadata for an Earth2014 layer.

    Parameters
    ----------
    layer : str
        Layer type.

    Returns
    -------
    Earth2014Metadata
        Metadata about the dataset.
    """
    if layer not in EARTH2014_PARAMETERS:
        raise ValueError(f"Unknown Earth2014 layer: {layer}")

    params = EARTH2014_PARAMETERS[layer]
    return Earth2014Metadata(
        layer=layer,
        description=params["description"],
        resolution_arcsec=EARTH2014_RESOLUTION_ARCSEC,
        lat_min=np.radians(-90.0),
        lat_max=np.radians(90.0),
        lon_min=np.radians(-180.0),
        lon_max=np.radians(180.0),
    )


__all__ = [
    # Parameters
    "GEBCO_PARAMETERS",
    "EARTH2014_PARAMETERS",
    # Metadata types
    "GEBCOMetadata",
    "Earth2014Metadata",
    # Data directory
    "get_data_dir",
    # Loading functions
    "load_gebco",
    "load_earth2014",
    # Test data generation
    "create_test_gebco_dem",
    "create_test_earth2014_dem",
    # Metadata functions
    "get_gebco_metadata",
    "get_earth2014_metadata",
    # Parsing functions (for advanced use)
    "parse_gebco_netcdf",
    "parse_earth2014_binary",
]
