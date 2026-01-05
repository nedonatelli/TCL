"""
Digital Elevation Model (DEM) Interface.

This module provides a generic interface for working with digital elevation
models, including:
- Elevation queries at geographic coordinates
- Terrain gradient and slope calculations
- Profile extraction along paths
- Support for various DEM data formats

The interface is designed to work with common DEM formats including:
- SRTM (Shuttle Radar Topography Mission)
- ASTER GDEM
- GEBCO (General Bathymetric Chart of the Oceans)
- Earth2014
- ETOPO1/ETOPO2

References
----------
.. [1] Farr, T. G., et al. "The Shuttle Radar Topography Mission."
       Reviews of Geophysics 45.2 (2007).
.. [2] GEBCO Compilation Group (2023) GEBCO 2023 Grid
.. [3] Hirt, C. and Rexer, M. "Earth2014: 1 arc-min shape, topography,
       bedrock and ice-sheet models." Geophysical Journal International
       198.3 (2014): 1544-1555.
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class DEMPoint(NamedTuple):
    """Single point elevation query result.

    Parameters
    ----------
    latitude : float
        Latitude in radians.
    longitude : float
        Longitude in radians.
    elevation : float
        Elevation in meters above reference (MSL or ellipsoid).
    valid : bool
        Whether the elevation value is valid (not a fill/no-data value).
    """

    latitude: float
    longitude: float
    elevation: float
    valid: bool


class TerrainGradient(NamedTuple):
    """Terrain gradient at a point.

    Parameters
    ----------
    slope : float
        Terrain slope in radians (angle from horizontal).
    aspect : float
        Aspect (downslope direction) in radians, measured clockwise from north.
    dz_dx : float
        Elevation gradient in east direction (m/m).
    dz_dy : float
        Elevation gradient in north direction (m/m).
    """

    slope: float
    aspect: float
    dz_dx: float
    dz_dy: float


class DEMMetadata(NamedTuple):
    """Metadata for a DEM dataset.

    Parameters
    ----------
    name : str
        Name of the DEM dataset.
    resolution : float
        Grid resolution in arc-seconds.
    lat_min : float
        Minimum latitude in radians.
    lat_max : float
        Maximum latitude in radians.
    lon_min : float
        Minimum longitude in radians.
    lon_max : float
        Maximum longitude in radians.
    vertical_datum : str
        Vertical reference (e.g., 'MSL', 'WGS84').
    horizontal_datum : str
        Horizontal reference (e.g., 'WGS84').
    """

    name: str
    resolution: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    vertical_datum: str
    horizontal_datum: str


class DEMGrid:
    """In-memory DEM grid for elevation queries.

    This class provides a simple in-memory representation of a DEM grid
    for efficient elevation queries. Data can be loaded from arrays or
    from external files.

    Parameters
    ----------
    data : ndarray
        2D array of elevation values (rows=latitude, cols=longitude).
        Latitude increases with row index (south to north).
    lat_min : float
        Minimum latitude in radians.
    lat_max : float
        Maximum latitude in radians.
    lon_min : float
        Minimum longitude in radians.
    lon_max : float
        Maximum longitude in radians.
    nodata_value : float, optional
        Value representing no data. Default is -9999.
    name : str, optional
        Name identifier for the DEM. Default is "Custom DEM".

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple 10x10 DEM grid
    >>> data = np.random.rand(10, 10) * 1000  # 0-1000m elevation
    >>> dem = DEMGrid(
    ...     data,
    ...     lat_min=np.radians(35.0),
    ...     lat_max=np.radians(36.0),
    ...     lon_min=np.radians(-120.0),
    ...     lon_max=np.radians(-119.0)
    ... )
    >>> elev = dem.get_elevation(np.radians(35.5), np.radians(-119.5))
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        nodata_value: float = -9999.0,
        name: str = "Custom DEM",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.nodata_value = nodata_value
        self.name = name

        # Compute grid parameters
        self.n_lat, self.n_lon = self.data.shape
        self.d_lat = (lat_max - lat_min) / (self.n_lat - 1) if self.n_lat > 1 else 0
        self.d_lon = (lon_max - lon_min) / (self.n_lon - 1) if self.n_lon > 1 else 0

    def get_metadata(self) -> DEMMetadata:
        """Get DEM metadata.

        Returns
        -------
        DEMMetadata
            Metadata about this DEM.
        """
        # Convert d_lat to arc-seconds for resolution
        resolution_arcsec = np.degrees(self.d_lat) * 3600 if self.d_lat > 0 else 0

        return DEMMetadata(
            name=self.name,
            resolution=resolution_arcsec,
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
            vertical_datum="MSL",
            horizontal_datum="WGS84",
        )

    def _in_bounds(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within DEM bounds."""
        return (
            self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max
        )

    def _get_indices(self, lat: float, lon: float) -> Tuple[int, int, float, float]:
        """Get grid indices and fractional parts for interpolation.

        Returns
        -------
        i : int
            Row index (latitude).
        j : int
            Column index (longitude).
        frac_i : float
            Fractional part for latitude interpolation.
        frac_j : float
            Fractional part for longitude interpolation.
        """
        # Compute continuous indices
        if self.d_lat > 0:
            idx_lat = (lat - self.lat_min) / self.d_lat
        else:
            idx_lat = 0.0

        if self.d_lon > 0:
            idx_lon = (lon - self.lon_min) / self.d_lon
        else:
            idx_lon = 0.0

        # Get integer indices
        i = int(np.floor(idx_lat))
        j = int(np.floor(idx_lon))

        # Clamp to valid range
        i = max(0, min(i, self.n_lat - 2))
        j = max(0, min(j, self.n_lon - 2))

        # Fractional parts
        frac_i = idx_lat - i
        frac_j = idx_lon - j

        return i, j, frac_i, frac_j

    def get_elevation(
        self,
        lat: float,
        lon: float,
        interpolation: str = "bilinear",
    ) -> DEMPoint:
        """Get elevation at a geographic coordinate.

        Parameters
        ----------
        lat : float
            Latitude in radians.
        lon : float
            Longitude in radians.
        interpolation : str, optional
            Interpolation method: 'nearest' or 'bilinear'. Default is 'bilinear'.

        Returns
        -------
        DEMPoint
            Elevation query result with validity flag.
        """
        if not self._in_bounds(lat, lon):
            return DEMPoint(lat, lon, self.nodata_value, False)

        i, j, frac_i, frac_j = self._get_indices(lat, lon)

        if interpolation == "nearest":
            # Round to nearest grid point
            ni = i + (1 if frac_i >= 0.5 else 0)
            nj = j + (1 if frac_j >= 0.5 else 0)
            ni = min(ni, self.n_lat - 1)
            nj = min(nj, self.n_lon - 1)
            elev = self.data[ni, nj]
        else:  # bilinear
            # Get four surrounding points
            z00 = self.data[i, j]
            z01 = self.data[i, j + 1] if j + 1 < self.n_lon else z00
            z10 = self.data[i + 1, j] if i + 1 < self.n_lat else z00
            z11 = (
                self.data[i + 1, j + 1]
                if (i + 1 < self.n_lat and j + 1 < self.n_lon)
                else z00
            )

            # Check for nodata
            values = [z00, z01, z10, z11]
            if any(v == self.nodata_value for v in values):
                return DEMPoint(lat, lon, self.nodata_value, False)

            # Bilinear interpolation
            elev = (
                z00 * (1 - frac_i) * (1 - frac_j)
                + z01 * (1 - frac_i) * frac_j
                + z10 * frac_i * (1 - frac_j)
                + z11 * frac_i * frac_j
            )

        valid = elev != self.nodata_value
        return DEMPoint(lat, lon, float(elev), valid)

    def get_elevations(
        self,
        lats: NDArray[np.floating],
        lons: NDArray[np.floating],
        interpolation: str = "bilinear",
    ) -> NDArray[np.floating]:
        """Get elevations at multiple geographic coordinates.

        Parameters
        ----------
        lats : ndarray
            Latitudes in radians.
        lons : ndarray
            Longitudes in radians.
        interpolation : str, optional
            Interpolation method. Default is 'bilinear'.

        Returns
        -------
        ndarray
            Array of elevation values. Invalid points contain nodata_value.
        """
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        elevations = np.full(lats.shape, self.nodata_value)

        for idx in np.ndindex(lats.shape):
            result = self.get_elevation(lats[idx], lons[idx], interpolation)
            elevations[idx] = result.elevation

        return elevations

    def get_gradient(
        self,
        lat: float,
        lon: float,
        earth_radius: float = 6371000.0,
    ) -> TerrainGradient:
        """Compute terrain gradient at a point.

        Uses central differences to estimate the gradient.

        Parameters
        ----------
        lat : float
            Latitude in radians.
        lon : float
            Longitude in radians.
        earth_radius : float, optional
            Earth radius in meters for converting angular to linear distances.
            Default is 6371000.0.

        Returns
        -------
        TerrainGradient
            Terrain gradient including slope and aspect.
        """
        # Get elevations at surrounding points
        z_center = self.get_elevation(lat, lon)
        if not z_center.valid:
            return TerrainGradient(0.0, 0.0, 0.0, 0.0)

        # Use grid spacing for finite differences
        dlat = self.d_lat if self.d_lat > 0 else np.radians(1 / 3600)  # 1 arcsec
        dlon = self.d_lon if self.d_lon > 0 else np.radians(1 / 3600)

        # Get neighboring elevations
        z_n = self.get_elevation(lat + dlat, lon)
        z_s = self.get_elevation(lat - dlat, lon)
        z_e = self.get_elevation(lat, lon + dlon)
        z_w = self.get_elevation(lat, lon - dlon)

        # Convert angular distances to meters
        dx = earth_radius * np.cos(lat) * dlon
        dy = earth_radius * dlat

        # Compute gradients using central differences
        if z_n.valid and z_s.valid:
            dz_dy = (z_n.elevation - z_s.elevation) / (2 * dy)
        elif z_n.valid:
            dz_dy = (z_n.elevation - z_center.elevation) / dy
        elif z_s.valid:
            dz_dy = (z_center.elevation - z_s.elevation) / dy
        else:
            dz_dy = 0.0

        if z_e.valid and z_w.valid:
            dz_dx = (z_e.elevation - z_w.elevation) / (2 * dx)
        elif z_e.valid:
            dz_dx = (z_e.elevation - z_center.elevation) / dx
        elif z_w.valid:
            dz_dx = (z_center.elevation - z_w.elevation) / dx
        else:
            dz_dx = 0.0

        # Compute slope and aspect
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

        # Aspect: direction of steepest descent, measured clockwise from north
        if dz_dx == 0 and dz_dy == 0:
            aspect = 0.0
        else:
            # atan2(-dz_dx, -dz_dy) gives direction of steepest descent
            aspect = np.arctan2(-dz_dx, -dz_dy)
            if aspect < 0:
                aspect += 2 * np.pi

        return TerrainGradient(slope, aspect, dz_dx, dz_dy)


def get_elevation_profile(
    dem: DEMGrid,
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    n_points: int = 100,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract elevation profile along a path.

    Parameters
    ----------
    dem : DEMGrid
        DEM grid to query.
    lat_start : float
        Starting latitude in radians.
    lon_start : float
        Starting longitude in radians.
    lat_end : float
        Ending latitude in radians.
    lon_end : float
        Ending longitude in radians.
    n_points : int, optional
        Number of sample points along the path. Default is 100.

    Returns
    -------
    distances : ndarray
        Array of distances from start in meters.
    elevations : ndarray
        Array of elevation values in meters.

    Examples
    --------
    >>> import numpy as np
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119),
    ...     elevation=500)
    >>> dists, elevs = get_elevation_profile(
    ...     dem, np.radians(35.2), np.radians(-119.8),
    ...     np.radians(35.8), np.radians(-119.2), n_points=10)
    >>> len(dists) == 10
    True
    """
    # Generate points along path
    lats = np.linspace(lat_start, lat_end, n_points)
    lons = np.linspace(lon_start, lon_end, n_points)

    # Get elevations
    elevations = dem.get_elevations(lats, lons)

    # Compute distances using great circle approximation
    earth_radius = 6371000.0
    dlat = lats - lat_start
    dlon = lons - lon_start
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_start) * np.cos(lats) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = earth_radius * c

    return distances, elevations


def interpolate_dem(
    dem: DEMGrid,
    new_lat_min: float,
    new_lat_max: float,
    new_lon_min: float,
    new_lon_max: float,
    new_n_lat: int,
    new_n_lon: int,
    method: str = "bilinear",
) -> DEMGrid:
    """Interpolate DEM to a new grid.

    Parameters
    ----------
    dem : DEMGrid
        Source DEM grid.
    new_lat_min : float
        New minimum latitude in radians.
    new_lat_max : float
        New maximum latitude in radians.
    new_lon_min : float
        New minimum longitude in radians.
    new_lon_max : float
        New maximum longitude in radians.
    new_n_lat : int
        Number of latitude points in new grid.
    new_n_lon : int
        Number of longitude points in new grid.
    method : str, optional
        Interpolation method. Default is 'bilinear'.

    Returns
    -------
    DEMGrid
        New interpolated DEM grid.

    Examples
    --------
    >>> import numpy as np
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119),
    ...     elevation=100)
    >>> new_dem = interpolate_dem(
    ...     dem,
    ...     np.radians(35.2), np.radians(35.8),
    ...     np.radians(-119.8), np.radians(-119.2),
    ...     new_n_lat=5, new_n_lon=5)
    >>> new_dem.data.shape
    (5, 5)
    """
    # Create new coordinate arrays
    new_lats = np.linspace(new_lat_min, new_lat_max, new_n_lat)
    new_lons = np.linspace(new_lon_min, new_lon_max, new_n_lon)

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(new_lons, new_lats)

    # Interpolate elevations
    new_data = np.zeros((new_n_lat, new_n_lon))
    for i in range(new_n_lat):
        for j in range(new_n_lon):
            result = dem.get_elevation(lat_grid[i, j], lon_grid[i, j], method)
            new_data[i, j] = result.elevation

    return DEMGrid(
        new_data,
        new_lat_min,
        new_lat_max,
        new_lon_min,
        new_lon_max,
        dem.nodata_value,
        f"{dem.name} (resampled)",
    )


def merge_dems(
    dems: List[DEMGrid],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_arcsec: float = 30.0,
) -> DEMGrid:
    """Merge multiple DEMs into a single grid.

    Parameters
    ----------
    dems : list of DEMGrid
        List of DEM grids to merge.
    lat_min : float
        Output minimum latitude in radians.
    lat_max : float
        Output maximum latitude in radians.
    lon_min : float
        Output minimum longitude in radians.
    lon_max : float
        Output maximum longitude in radians.
    resolution_arcsec : float, optional
        Output resolution in arc-seconds. Default is 30.0.

    Returns
    -------
    DEMGrid
        Merged DEM grid.

    Examples
    --------
    >>> import numpy as np
    >>> dem1 = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119), elevation=100)
    >>> dem2 = create_flat_dem(
    ...     np.radians(36), np.radians(37),
    ...     np.radians(-120), np.radians(-119), elevation=200)
    >>> merged = merge_dems(
    ...     [dem1, dem2],
    ...     np.radians(35), np.radians(37),
    ...     np.radians(-120), np.radians(-119))
    >>> merged.name
    'Merged DEM'
    """
    # Compute output grid dimensions
    d_lat = np.radians(resolution_arcsec / 3600)
    d_lon = np.radians(resolution_arcsec / 3600)

    n_lat = int(np.ceil((lat_max - lat_min) / d_lat)) + 1
    n_lon = int(np.ceil((lon_max - lon_min) / d_lon)) + 1

    # Create output arrays
    new_lats = np.linspace(lat_min, lat_max, n_lat)
    new_lons = np.linspace(lon_min, lon_max, n_lon)
    new_data = np.full((n_lat, n_lon), -9999.0)

    # Fill output grid from input DEMs
    for i, lat in enumerate(new_lats):
        for j, lon in enumerate(new_lons):
            for dem in dems:
                result = dem.get_elevation(lat, lon)
                if result.valid:
                    new_data[i, j] = result.elevation
                    break

    return DEMGrid(
        new_data,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        -9999.0,
        "Merged DEM",
    )


def create_flat_dem(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    elevation: float = 0.0,
    resolution_arcsec: float = 30.0,
) -> DEMGrid:
    """Create a flat DEM with constant elevation.

    Useful for testing or as a placeholder when terrain data is unavailable.

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
    elevation : float, optional
        Constant elevation in meters. Default is 0.0.
    resolution_arcsec : float, optional
        Grid resolution in arc-seconds. Default is 30.0.

    Returns
    -------
    DEMGrid
        Flat DEM grid.

    Examples
    --------
    >>> import numpy as np
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119),
    ...     elevation=500)
    >>> dem.name
    'Flat DEM'
    >>> result = dem.get_elevation(np.radians(35.5), np.radians(-119.5))
    >>> abs(result.elevation - 500) < 1
    True
    """
    d_lat = np.radians(resolution_arcsec / 3600)
    d_lon = np.radians(resolution_arcsec / 3600)

    n_lat = max(2, int(np.ceil((lat_max - lat_min) / d_lat)) + 1)
    n_lon = max(2, int(np.ceil((lon_max - lon_min) / d_lon)) + 1)

    data = np.full((n_lat, n_lon), elevation)

    return DEMGrid(
        data,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        -9999.0,
        "Flat DEM",
    )


def create_synthetic_terrain(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    base_elevation: float = 0.0,
    amplitude: float = 1000.0,
    wavelength_km: float = 50.0,
    resolution_arcsec: float = 30.0,
    seed: Optional[int] = None,
) -> DEMGrid:
    """Create synthetic terrain for testing.

    Generates terrain using a combination of sinusoidal functions and
    random noise to simulate realistic topography.

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
    base_elevation : float, optional
        Base elevation in meters. Default is 0.0.
    amplitude : float, optional
        Amplitude of elevation variations in meters. Default is 1000.0.
    wavelength_km : float, optional
        Characteristic wavelength of terrain features in km. Default is 50.0.
    resolution_arcsec : float, optional
        Grid resolution in arc-seconds. Default is 30.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DEMGrid
        Synthetic terrain DEM.

    Examples
    --------
    >>> import numpy as np
    >>> dem = create_synthetic_terrain(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119),
    ...     base_elevation=500, amplitude=200, seed=42)
    >>> dem.name
    'Synthetic Terrain'
    >>> dem.data.min() < dem.data.max()  # Has elevation variation
    True
    """
    if seed is not None:
        np.random.seed(seed)

    d_lat = np.radians(resolution_arcsec / 3600)
    d_lon = np.radians(resolution_arcsec / 3600)

    n_lat = max(2, int(np.ceil((lat_max - lat_min) / d_lat)) + 1)
    n_lon = max(2, int(np.ceil((lon_max - lon_min) / d_lon)) + 1)

    # Create coordinate arrays
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Convert to approximate distances from center
    earth_radius = 6371.0  # km
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2

    x_km = earth_radius * np.cos(lat_center) * (lon_grid - lon_center)
    y_km = earth_radius * (lat_grid - lat_center)

    # Generate terrain with multiple frequency components
    k = 2 * np.pi / wavelength_km

    elevation = base_elevation
    elevation += amplitude * 0.5 * np.sin(k * x_km) * np.cos(k * y_km)
    elevation += amplitude * 0.3 * np.sin(2 * k * x_km + 0.5)
    elevation += amplitude * 0.2 * np.cos(1.5 * k * y_km + 0.3)

    # Add some random variation
    noise = np.random.randn(n_lat, n_lon) * amplitude * 0.1
    # Smooth the noise
    from scipy.ndimage import gaussian_filter

    noise = gaussian_filter(noise, sigma=2)
    elevation += noise

    return DEMGrid(
        elevation,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        -9999.0,
        "Synthetic Terrain",
    )
