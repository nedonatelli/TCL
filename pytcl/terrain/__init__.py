"""
Terrain Models and Visibility Analysis.

This module provides tools for working with digital elevation models (DEMs)
and computing terrain-related effects:

- Digital Elevation Model interface for elevation queries
- GEBCO bathymetry/topography data loading
- Earth2014 global terrain model loading
- Terrain gradient and slope calculations
- Line-of-sight and viewshed analysis
- Terrain masking for radar/sensor coverage

Submodules
----------
dem
    DEM data structures and elevation query interface.
loaders
    GEBCO and Earth2014 data loaders.
visibility
    Line-of-sight, viewshed, and terrain masking functions.

Examples
--------
>>> from pytcl.terrain import DEMGrid, create_synthetic_terrain
>>> import numpy as np
>>> # Create synthetic terrain for testing
>>> dem = create_synthetic_terrain(
...     lat_min=np.radians(35.0),
...     lat_max=np.radians(36.0),
...     lon_min=np.radians(-120.0),
...     lon_max=np.radians(-119.0),
...     amplitude=500.0
... )
>>> # Query elevation
>>> point = dem.get_elevation(np.radians(35.5), np.radians(-119.5))
>>> print(f"Elevation: {point.elevation:.1f} m")

>>> # Check line of sight
>>> from pytcl.terrain import line_of_sight
>>> los = line_of_sight(
...     dem,
...     obs_lat=np.radians(35.2), obs_lon=np.radians(-119.8), obs_height=10.0,
...     tgt_lat=np.radians(35.8), tgt_lon=np.radians(-119.2), tgt_height=10.0
... )
>>> print(f"Visible: {los.visible}, Clearance: {los.clearance:.1f} m")

>>> # Load GEBCO/Earth2014 data (requires external files)
>>> from pytcl.terrain import load_gebco, load_earth2014, create_test_gebco_dem
>>> # Use test data for demonstration
>>> dem = create_test_gebco_dem()
"""

# DEM interface
from pytcl.terrain.dem import (
    DEMGrid,
    DEMMetadata,
    DEMPoint,
    TerrainGradient,
    create_flat_dem,
    create_synthetic_terrain,
    get_elevation_profile,
    interpolate_dem,
    merge_dems,
)

# Data loaders
from pytcl.terrain.loaders import (
    EARTH2014_PARAMETERS,
    GEBCO_PARAMETERS,
    Earth2014Metadata,
    GEBCOMetadata,
    create_test_earth2014_dem,
    create_test_gebco_dem,
    get_data_dir,
    get_earth2014_metadata,
    get_gebco_metadata,
    load_earth2014,
    load_gebco,
)

# Visibility functions
from pytcl.terrain.visibility import (
    HorizonPoint,
    LOSResult,
    ViewshedResult,
    compute_horizon,
    line_of_sight,
    radar_coverage_map,
    terrain_masking_angle,
    viewshed,
)

__all__ = [
    # DEM data structures
    "DEMPoint",
    "TerrainGradient",
    "DEMMetadata",
    "DEMGrid",
    # DEM functions
    "get_elevation_profile",
    "interpolate_dem",
    "merge_dems",
    "create_flat_dem",
    "create_synthetic_terrain",
    # Visibility data structures
    "LOSResult",
    "ViewshedResult",
    "HorizonPoint",
    # Visibility functions
    "line_of_sight",
    "viewshed",
    "compute_horizon",
    "terrain_masking_angle",
    "radar_coverage_map",
    # Data loaders
    "GEBCO_PARAMETERS",
    "EARTH2014_PARAMETERS",
    "GEBCOMetadata",
    "Earth2014Metadata",
    "get_data_dir",
    "load_gebco",
    "load_earth2014",
    "create_test_gebco_dem",
    "create_test_earth2014_dem",
    "get_gebco_metadata",
    "get_earth2014_metadata",
]
