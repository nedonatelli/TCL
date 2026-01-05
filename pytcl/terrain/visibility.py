"""
Terrain Visibility and Masking Functions.

This module provides functions for computing terrain visibility and masking,
including:
- Line-of-sight (LOS) analysis between points
- Viewshed computation (visible area from a point)
- Terrain masking for radar/sensor coverage
- Horizon computation

These functions are essential for:
- Radar coverage analysis
- Communication link budget calculations
- Sensor placement optimization
- Target detection range estimation

References
----------
.. [1] Wang, J., Robinson, G.J., and White, K. "A fast solution to local
       viewshed computation using grid-based digital elevation models."
       Photogrammetric Engineering & Remote Sensing 62.10 (1996): 1157-1164.
.. [2] De Floriani, L. and Magillo, P. "Algorithms for visibility computation
       on terrains: a survey." Environment and Planning B 30.5 (2003): 709-728.
"""

from typing import List, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .dem import DEMGrid


class LOSResult(NamedTuple):
    """Line-of-sight analysis result.

    Parameters
    ----------
    visible : bool
        Whether there is unobstructed line of sight between observer and target.
    grazing_angle : float
        Grazing angle to terrain in radians (angle above/below obstacle).
        Positive means clearing terrain, negative means blocked.
    obstacle_distance : float
        Distance to the blocking obstacle in meters (if blocked).
        0 if line of sight is clear.
    obstacle_elevation : float
        Elevation of blocking obstacle in meters (if blocked).
    clearance : float
        Minimum clearance above terrain along path in meters.
        Negative if blocked.
    """

    visible: bool
    grazing_angle: float
    obstacle_distance: float
    obstacle_elevation: float
    clearance: float


class ViewshedResult(NamedTuple):
    """Viewshed computation result.

    Parameters
    ----------
    visible : ndarray
        2D boolean array indicating visibility from observer.
    observer_lat : float
        Observer latitude in radians.
    observer_lon : float
        Observer longitude in radians.
    observer_height : float
        Observer height above terrain in meters.
    lat_min : float
        Minimum latitude of viewshed grid in radians.
    lat_max : float
        Maximum latitude of viewshed grid in radians.
    lon_min : float
        Minimum longitude of viewshed grid in radians.
    lon_max : float
        Maximum longitude of viewshed grid in radians.
    """

    visible: NDArray[np.bool_]
    observer_lat: float
    observer_lon: float
    observer_height: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class HorizonPoint(NamedTuple):
    """Point on the terrain horizon.

    Parameters
    ----------
    azimuth : float
        Azimuth from observer in radians (clockwise from north).
    elevation_angle : float
        Elevation angle to horizon in radians (above horizontal).
    distance : float
        Distance to horizon point in meters.
    terrain_elevation : float
        Terrain elevation at horizon point in meters.
    """

    azimuth: float
    elevation_angle: float
    distance: float
    terrain_elevation: float


def line_of_sight(
    dem: DEMGrid,
    obs_lat: float,
    obs_lon: float,
    obs_height: float,
    tgt_lat: float,
    tgt_lon: float,
    tgt_height: float,
    n_samples: int = 100,
    earth_radius: float = 6371000.0,
    refraction_coeff: float = 0.0,
) -> LOSResult:
    """Compute line of sight between observer and target.

    Parameters
    ----------
    dem : DEMGrid
        Digital elevation model.
    obs_lat : float
        Observer latitude in radians.
    obs_lon : float
        Observer longitude in radians.
    obs_height : float
        Observer height above terrain in meters.
    tgt_lat : float
        Target latitude in radians.
    tgt_lon : float
        Target longitude in radians.
    tgt_height : float
        Target height above terrain in meters.
    n_samples : int, optional
        Number of sample points along path. Default is 100.
    earth_radius : float, optional
        Earth radius in meters. Default is 6371000.0.
    refraction_coeff : float, optional
        Atmospheric refraction coefficient (typically 0.13 for radio).
        Set to 0 for optical line of sight. Default is 0.0.

    Returns
    -------
    LOSResult
        Line-of-sight analysis result.

    Notes
    -----
    The refraction coefficient models atmospheric bending of radio waves.
    A typical value for radio frequencies is 0.13 (4/3 Earth model).
    For optical line of sight, use 0.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.terrain.dem import create_flat_dem
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119), elevation=100)
    >>> result = line_of_sight(
    ...     dem,
    ...     np.radians(35.3), np.radians(-119.7), 10,
    ...     np.radians(35.7), np.radians(-119.3), 10)
    >>> result.visible  # Clear LOS over flat terrain
    True
    """
    # Effective Earth radius for refraction
    if refraction_coeff > 0:
        effective_radius = earth_radius / (1 - refraction_coeff)
    else:
        effective_radius = earth_radius

    # Get observer terrain elevation
    obs_terrain = dem.get_elevation(obs_lat, obs_lon)
    obs_elev = obs_terrain.elevation if obs_terrain.valid else 0.0
    obs_total = obs_elev + obs_height

    # Get target terrain elevation
    tgt_terrain = dem.get_elevation(tgt_lat, tgt_lon)
    tgt_elev = tgt_terrain.elevation if tgt_terrain.valid else 0.0
    tgt_total = tgt_elev + tgt_height

    # Sample points along path
    sample_lats = np.linspace(obs_lat, tgt_lat, n_samples)
    sample_lons = np.linspace(obs_lon, tgt_lon, n_samples)

    # Compute distances from observer
    dlat = sample_lats - obs_lat
    dlon = sample_lons - obs_lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(obs_lat) * np.cos(sample_lats) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = earth_radius * c

    total_distance = distances[-1]
    if total_distance < 1e-6:
        return LOSResult(True, np.pi / 2, 0.0, 0.0, float("inf"))

    # Compute LOS height at each sample point (linear interpolation)
    t = distances / total_distance
    los_heights = obs_total * (1 - t) + tgt_total * t

    # Apply Earth curvature correction
    # For a spherical Earth, the drop due to curvature is approximately:
    # h_drop = d^2 / (2 * R_effective)
    curvature_drop = distances * (total_distance - distances) / (2 * effective_radius)

    # Get terrain elevations along path
    terrain_elevs = np.zeros(n_samples)
    for i in range(n_samples):
        result = dem.get_elevation(sample_lats[i], sample_lons[i])
        terrain_elevs[i] = result.elevation if result.valid else 0.0

    # Compute clearance (LOS height above terrain, accounting for curvature)
    # The terrain appears lower by curvature_drop due to Earth's curvature
    effective_terrain = terrain_elevs - curvature_drop
    clearances = los_heights - effective_terrain

    # Find minimum clearance (skip endpoints)
    if n_samples > 2:
        min_clearance_idx = np.argmin(clearances[1:-1]) + 1
        min_clearance = clearances[min_clearance_idx]
    else:
        min_clearance_idx = 0
        min_clearance = float("inf")

    # Check visibility
    visible = min_clearance >= 0

    if visible:
        # Compute grazing angle (angle above obstacle)
        if min_clearance < float("inf"):
            grazing_dist = distances[min_clearance_idx]
            if grazing_dist > 0:
                grazing_angle = np.arctan(min_clearance / grazing_dist)
            else:
                grazing_angle = np.pi / 2
        else:
            grazing_angle = np.pi / 2
        obstacle_distance = 0.0
        obstacle_elevation = 0.0
    else:
        # Find the blocking point
        blocked_idx = min_clearance_idx
        obstacle_distance = distances[blocked_idx]
        obstacle_elevation = terrain_elevs[blocked_idx]

        # Grazing angle (negative when blocked)
        if obstacle_distance > 0:
            grazing_angle = np.arctan(min_clearance / obstacle_distance)
        else:
            grazing_angle = -np.pi / 2

    return LOSResult(
        visible, grazing_angle, obstacle_distance, obstacle_elevation, min_clearance
    )


def viewshed(
    dem: DEMGrid,
    obs_lat: float,
    obs_lon: float,
    obs_height: float,
    max_range: float = 50000.0,
    target_height: float = 0.0,
    n_radials: int = 360,
    samples_per_radial: int = 100,
    earth_radius: float = 6371000.0,
    refraction_coeff: float = 0.0,
) -> ViewshedResult:
    """Compute viewshed (visible area) from an observer location.

    Parameters
    ----------
    dem : DEMGrid
        Digital elevation model.
    obs_lat : float
        Observer latitude in radians.
    obs_lon : float
        Observer longitude in radians.
    obs_height : float
        Observer height above terrain in meters.
    max_range : float, optional
        Maximum analysis range in meters. Default is 50000.0.
    target_height : float, optional
        Target height above terrain in meters. Default is 0.0.
    n_radials : int, optional
        Number of radial directions to analyze. Default is 360.
    samples_per_radial : int, optional
        Number of samples per radial. Default is 100.
    earth_radius : float, optional
        Earth radius in meters. Default is 6371000.0.
    refraction_coeff : float, optional
        Atmospheric refraction coefficient. Default is 0.0.

    Returns
    -------
    ViewshedResult
        Viewshed computation result with visibility grid.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.terrain.dem import create_flat_dem
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119), elevation=100)
    >>> result = viewshed(
    ...     dem, np.radians(35.5), np.radians(-119.5), 20,
    ...     max_range=10000, n_radials=36, samples_per_radial=10)
    >>> result.visible.any()  # Some cells visible
    True
    """
    # Convert max range to angular distance
    max_angular_range = max_range / earth_radius

    # Create output grid matching DEM
    visible = np.zeros((dem.n_lat, dem.n_lon), dtype=bool)

    # Effective Earth radius for refraction
    if refraction_coeff > 0:
        effective_radius = earth_radius / (1 - refraction_coeff)
    else:
        effective_radius = earth_radius

    # Get observer terrain elevation
    obs_terrain = dem.get_elevation(obs_lat, obs_lon)
    obs_elev = obs_terrain.elevation if obs_terrain.valid else 0.0
    obs_total = obs_elev + obs_height

    # Process each radial direction
    azimuths = np.linspace(0, 2 * np.pi, n_radials, endpoint=False)

    for azimuth in azimuths:
        # Maximum slope angle seen so far along this radial
        max_slope = -np.inf

        # Sample points along radial
        for i_sample in range(1, samples_per_radial + 1):
            # Distance along radial
            dist_frac = i_sample / samples_per_radial
            angular_dist = max_angular_range * dist_frac
            linear_dist = angular_dist * earth_radius

            # Compute target coordinates using spherical trig
            tgt_lat = np.arcsin(
                np.sin(obs_lat) * np.cos(angular_dist)
                + np.cos(obs_lat) * np.sin(angular_dist) * np.cos(azimuth)
            )
            dlon = np.arctan2(
                np.sin(azimuth) * np.sin(angular_dist) * np.cos(obs_lat),
                np.cos(angular_dist) - np.sin(obs_lat) * np.sin(tgt_lat),
            )
            tgt_lon = obs_lon + dlon

            # Normalize longitude to [-pi, pi]
            while tgt_lon > np.pi:
                tgt_lon -= 2 * np.pi
            while tgt_lon < -np.pi:
                tgt_lon += 2 * np.pi

            # Check if within DEM bounds
            if not dem._in_bounds(tgt_lat, tgt_lon):
                continue

            # Get target terrain elevation
            tgt_terrain = dem.get_elevation(tgt_lat, tgt_lon)
            if not tgt_terrain.valid:
                continue

            tgt_total = tgt_terrain.elevation + target_height

            # Apply Earth curvature correction
            curvature_drop = linear_dist**2 / (2 * effective_radius)
            effective_tgt = tgt_total - curvature_drop

            # Compute slope angle from observer
            elevation_diff = effective_tgt - obs_total
            slope_angle = np.arctan2(elevation_diff, linear_dist)

            # Visible if slope angle exceeds maximum seen so far
            is_visible = slope_angle >= max_slope

            if is_visible:
                # Find nearest grid cell and mark visible
                i, j, _, _ = dem._get_indices(tgt_lat, tgt_lon)
                if 0 <= i < dem.n_lat and 0 <= j < dem.n_lon:
                    visible[i, j] = True

            # Update maximum slope
            max_slope = max(max_slope, slope_angle)

    # Observer location is always visible
    i_obs, j_obs, _, _ = dem._get_indices(obs_lat, obs_lon)
    if 0 <= i_obs < dem.n_lat and 0 <= j_obs < dem.n_lon:
        visible[i_obs, j_obs] = True

    return ViewshedResult(
        visible,
        obs_lat,
        obs_lon,
        obs_height,
        dem.lat_min,
        dem.lat_max,
        dem.lon_min,
        dem.lon_max,
    )


def compute_horizon(
    dem: DEMGrid,
    obs_lat: float,
    obs_lon: float,
    obs_height: float,
    n_azimuths: int = 360,
    max_range: float = 50000.0,
    samples_per_radial: int = 100,
    earth_radius: float = 6371000.0,
) -> List[HorizonPoint]:
    """Compute terrain horizon profile from an observer location.

    Parameters
    ----------
    dem : DEMGrid
        Digital elevation model.
    obs_lat : float
        Observer latitude in radians.
    obs_lon : float
        Observer longitude in radians.
    obs_height : float
        Observer height above terrain in meters.
    n_azimuths : int, optional
        Number of azimuth directions. Default is 360.
    max_range : float, optional
        Maximum analysis range in meters. Default is 50000.0.
    samples_per_radial : int, optional
        Number of samples per radial. Default is 100.
    earth_radius : float, optional
        Earth radius in meters. Default is 6371000.0.

    Returns
    -------
    list of HorizonPoint
        Horizon points for each azimuth direction.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.terrain.dem import create_flat_dem
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119), elevation=100)
    >>> horizon = compute_horizon(
    ...     dem, np.radians(35.5), np.radians(-119.5), 10,
    ...     n_azimuths=8, max_range=10000, samples_per_radial=10)
    >>> len(horizon)
    8
    """
    max_angular_range = max_range / earth_radius

    # Get observer terrain elevation
    obs_terrain = dem.get_elevation(obs_lat, obs_lon)
    obs_elev = obs_terrain.elevation if obs_terrain.valid else 0.0
    obs_total = obs_elev + obs_height

    horizon_points = []
    azimuths = np.linspace(0, 2 * np.pi, n_azimuths, endpoint=False)

    for azimuth in azimuths:
        max_elevation_angle = -np.pi / 2
        horizon_dist = 0.0
        horizon_elev = 0.0

        # Sample along radial to find horizon
        for i_sample in range(1, samples_per_radial + 1):
            dist_frac = i_sample / samples_per_radial
            angular_dist = max_angular_range * dist_frac
            linear_dist = angular_dist * earth_radius

            # Compute target coordinates
            tgt_lat = np.arcsin(
                np.sin(obs_lat) * np.cos(angular_dist)
                + np.cos(obs_lat) * np.sin(angular_dist) * np.cos(azimuth)
            )
            dlon = np.arctan2(
                np.sin(azimuth) * np.sin(angular_dist) * np.cos(obs_lat),
                np.cos(angular_dist) - np.sin(obs_lat) * np.sin(tgt_lat),
            )
            tgt_lon = obs_lon + dlon

            # Normalize longitude
            while tgt_lon > np.pi:
                tgt_lon -= 2 * np.pi
            while tgt_lon < -np.pi:
                tgt_lon += 2 * np.pi

            if not dem._in_bounds(tgt_lat, tgt_lon):
                continue

            # Get terrain elevation
            tgt_terrain = dem.get_elevation(tgt_lat, tgt_lon)
            if not tgt_terrain.valid:
                continue

            # Apply Earth curvature
            curvature_drop = linear_dist**2 / (2 * earth_radius)
            effective_elev = tgt_terrain.elevation - curvature_drop

            # Compute elevation angle
            elevation_diff = effective_elev - obs_total
            elev_angle = np.arctan2(elevation_diff, linear_dist)

            # Update horizon if this is highest angle seen
            if elev_angle > max_elevation_angle:
                max_elevation_angle = elev_angle
                horizon_dist = linear_dist
                horizon_elev = tgt_terrain.elevation

        horizon_points.append(
            HorizonPoint(azimuth, max_elevation_angle, horizon_dist, horizon_elev)
        )

    return horizon_points


def terrain_masking_angle(
    dem: DEMGrid,
    obs_lat: float,
    obs_lon: float,
    obs_height: float,
    azimuth: float,
    max_range: float = 50000.0,
    n_samples: int = 100,
    earth_radius: float = 6371000.0,
) -> float:
    """Compute terrain masking angle in a specific direction.

    The masking angle is the minimum elevation angle at which a target
    would be visible (not blocked by terrain).

    Parameters
    ----------
    dem : DEMGrid
        Digital elevation model.
    obs_lat : float
        Observer latitude in radians.
    obs_lon : float
        Observer longitude in radians.
    obs_height : float
        Observer height above terrain in meters.
    azimuth : float
        Azimuth direction in radians (clockwise from north).
    max_range : float, optional
        Maximum analysis range in meters. Default is 50000.0.
    n_samples : int, optional
        Number of sample points. Default is 100.
    earth_radius : float, optional
        Earth radius in meters. Default is 6371000.0.

    Returns
    -------
    float
        Masking angle in radians above horizontal.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.terrain.dem import create_flat_dem
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119), elevation=100)
    >>> angle = terrain_masking_angle(
    ...     dem, np.radians(35.5), np.radians(-119.5), 10, azimuth=0)
    >>> -np.pi/2 <= angle <= np.pi/2  # Valid angle range
    True
    """
    max_angular_range = max_range / earth_radius

    obs_terrain = dem.get_elevation(obs_lat, obs_lon)
    obs_elev = obs_terrain.elevation if obs_terrain.valid else 0.0
    obs_total = obs_elev + obs_height

    max_elevation_angle = -np.pi / 2

    for i_sample in range(1, n_samples + 1):
        dist_frac = i_sample / n_samples
        angular_dist = max_angular_range * dist_frac
        linear_dist = angular_dist * earth_radius

        tgt_lat = np.arcsin(
            np.sin(obs_lat) * np.cos(angular_dist)
            + np.cos(obs_lat) * np.sin(angular_dist) * np.cos(azimuth)
        )
        dlon = np.arctan2(
            np.sin(azimuth) * np.sin(angular_dist) * np.cos(obs_lat),
            np.cos(angular_dist) - np.sin(obs_lat) * np.sin(tgt_lat),
        )
        tgt_lon = obs_lon + dlon

        while tgt_lon > np.pi:
            tgt_lon -= 2 * np.pi
        while tgt_lon < -np.pi:
            tgt_lon += 2 * np.pi

        if not dem._in_bounds(tgt_lat, tgt_lon):
            continue

        tgt_terrain = dem.get_elevation(tgt_lat, tgt_lon)
        if not tgt_terrain.valid:
            continue

        curvature_drop = linear_dist**2 / (2 * earth_radius)
        effective_elev = tgt_terrain.elevation - curvature_drop

        elevation_diff = effective_elev - obs_total
        elev_angle = np.arctan2(elevation_diff, linear_dist)

        max_elevation_angle = max(max_elevation_angle, elev_angle)

    return max_elevation_angle


def radar_coverage_map(
    dem: DEMGrid,
    radar_lat: float,
    radar_lon: float,
    radar_height: float,
    min_elevation: float = 0.0,
    max_range: float = 100000.0,
    target_height: float = 1000.0,
    n_radials: int = 360,
    samples_per_radial: int = 200,
    earth_radius: float = 6371000.0,
    refraction_coeff: float = 0.13,
) -> ViewshedResult:
    """Compute radar coverage map accounting for terrain masking.

    Similar to viewshed but with radar-specific parameters including
    minimum elevation angle and atmospheric refraction.

    Parameters
    ----------
    dem : DEMGrid
        Digital elevation model.
    radar_lat : float
        Radar latitude in radians.
    radar_lon : float
        Radar longitude in radians.
    radar_height : float
        Radar antenna height above terrain in meters.
    min_elevation : float, optional
        Minimum radar elevation angle in radians. Default is 0.0.
    max_range : float, optional
        Maximum radar range in meters. Default is 100000.0.
    target_height : float, optional
        Target altitude above terrain in meters. Default is 1000.0.
    n_radials : int, optional
        Number of radial directions. Default is 360.
    samples_per_radial : int, optional
        Samples per radial. Default is 200.
    earth_radius : float, optional
        Earth radius in meters. Default is 6371000.0.
    refraction_coeff : float, optional
        Atmospheric refraction coefficient (0.13 for 4/3 Earth). Default is 0.13.

    Returns
    -------
    ViewshedResult
        Radar coverage map.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.terrain.dem import create_flat_dem
    >>> dem = create_flat_dem(
    ...     np.radians(35), np.radians(36),
    ...     np.radians(-120), np.radians(-119), elevation=100)
    >>> coverage = radar_coverage_map(
    ...     dem, np.radians(35.5), np.radians(-119.5), 30,
    ...     max_range=20000, n_radials=36, samples_per_radial=20)
    >>> coverage.visible.any()  # Some coverage exists
    True
    """
    # Compute basic viewshed with refraction
    result = viewshed(
        dem,
        radar_lat,
        radar_lon,
        radar_height,
        max_range=max_range,
        target_height=target_height,
        n_radials=n_radials,
        samples_per_radial=samples_per_radial,
        earth_radius=earth_radius,
        refraction_coeff=refraction_coeff,
    )

    # Apply minimum elevation constraint
    if min_elevation > 0:
        visible = result.visible.copy()
        effective_radius = earth_radius / (1 - refraction_coeff)

        radar_terrain = dem.get_elevation(radar_lat, radar_lon)
        radar_elev = radar_terrain.elevation if radar_terrain.valid else 0.0
        radar_total = radar_elev + radar_height

        # Check each visible cell against minimum elevation
        for i in range(dem.n_lat):
            for j in range(dem.n_lon):
                if not visible[i, j]:
                    continue

                # Get cell coordinates
                cell_lat = dem.lat_min + i * dem.d_lat
                cell_lon = dem.lon_min + j * dem.d_lon

                # Compute distance
                dlat = cell_lat - radar_lat
                dlon = cell_lon - radar_lon
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(radar_lat) * np.cos(cell_lat) * np.sin(dlon / 2) ** 2
                )
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist = earth_radius * c

                if dist < 1e-6:
                    continue

                # Get cell elevation
                cell_terrain = dem.get_elevation(cell_lat, cell_lon)
                cell_elev = cell_terrain.elevation if cell_terrain.valid else 0.0
                cell_total = cell_elev + target_height

                # Apply curvature
                curvature_drop = dist**2 / (2 * effective_radius)
                effective_cell = cell_total - curvature_drop

                # Compute elevation angle
                elev_angle = np.arctan2(effective_cell - radar_total, dist)

                # Mask if below minimum elevation
                if elev_angle < min_elevation:
                    visible[i, j] = False

        return ViewshedResult(
            visible,
            result.observer_lat,
            result.observer_lon,
            result.observer_height,
            result.lat_min,
            result.lat_max,
            result.lon_min,
            result.lon_max,
        )

    return result
