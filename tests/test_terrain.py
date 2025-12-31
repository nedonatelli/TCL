"""
Tests for terrain module (DEM interface and visibility).
"""

import numpy as np
import pytest

from pytcl.terrain import (  # DEM data structures; DEM functions; Visibility data structures; Visibility functions
    DEMGrid,
    DEMMetadata,
    DEMPoint,
    HorizonPoint,
    LOSResult,
    TerrainGradient,
    ViewshedResult,
    compute_horizon,
    create_flat_dem,
    create_synthetic_terrain,
    get_elevation_profile,
    interpolate_dem,
    line_of_sight,
    merge_dems,
    radar_coverage_map,
    terrain_masking_angle,
    viewshed,
)

# ============================================================================
# DEM Data Structure Tests
# ============================================================================


class TestDEMPoint:
    """Tests for DEMPoint named tuple."""

    def test_creation(self):
        """Test DEMPoint creation."""
        point = DEMPoint(
            latitude=np.radians(35.0),
            longitude=np.radians(-120.0),
            elevation=100.0,
            valid=True,
        )
        assert point.latitude == pytest.approx(np.radians(35.0))
        assert point.longitude == pytest.approx(np.radians(-120.0))
        assert point.elevation == 100.0
        assert point.valid is True

    def test_invalid_point(self):
        """Test DEMPoint with invalid flag."""
        point = DEMPoint(0.0, 0.0, -9999.0, False)
        assert point.valid is False
        assert point.elevation == -9999.0


class TestTerrainGradient:
    """Tests for TerrainGradient named tuple."""

    def test_creation(self):
        """Test TerrainGradient creation."""
        grad = TerrainGradient(
            slope=np.radians(10.0),
            aspect=np.radians(45.0),
            dz_dx=0.1,
            dz_dy=0.15,
        )
        assert grad.slope == pytest.approx(np.radians(10.0))
        assert grad.aspect == pytest.approx(np.radians(45.0))
        assert grad.dz_dx == 0.1
        assert grad.dz_dy == 0.15


class TestDEMMetadata:
    """Tests for DEMMetadata named tuple."""

    def test_creation(self):
        """Test DEMMetadata creation."""
        meta = DEMMetadata(
            name="Test DEM",
            resolution=30.0,
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            vertical_datum="MSL",
            horizontal_datum="WGS84",
        )
        assert meta.name == "Test DEM"
        assert meta.resolution == 30.0


# ============================================================================
# DEMGrid Tests
# ============================================================================


class TestDEMGrid:
    """Tests for DEMGrid class."""

    @pytest.fixture
    def simple_dem(self):
        """Create a simple 10x10 DEM for testing."""
        # Create elevation data that increases from SW to NE
        data = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                data[i, j] = i * 100 + j * 50  # 0-1350m range

        return DEMGrid(
            data,
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            name="Test DEM",
        )

    def test_creation(self, simple_dem):
        """Test DEMGrid creation."""
        assert simple_dem.n_lat == 10
        assert simple_dem.n_lon == 10
        assert simple_dem.name == "Test DEM"

    def test_get_metadata(self, simple_dem):
        """Test metadata retrieval."""
        meta = simple_dem.get_metadata()
        assert meta.name == "Test DEM"
        assert meta.lat_min == pytest.approx(np.radians(35.0))
        assert meta.lat_max == pytest.approx(np.radians(36.0))
        assert meta.vertical_datum == "MSL"

    def test_get_elevation_center(self, simple_dem):
        """Test elevation query at center point."""
        # Query center of DEM
        lat = np.radians(35.5)
        lon = np.radians(-119.5)
        point = simple_dem.get_elevation(lat, lon)

        assert point.valid
        # Center should be approximately middle elevation
        assert 400 < point.elevation < 900

    def test_get_elevation_corner(self, simple_dem):
        """Test elevation query at corner."""
        # SW corner (lowest point)
        lat = np.radians(35.0)
        lon = np.radians(-120.0)
        point = simple_dem.get_elevation(lat, lon)

        assert point.valid
        assert point.elevation == pytest.approx(0.0, abs=1)

    def test_get_elevation_out_of_bounds(self, simple_dem):
        """Test elevation query outside bounds."""
        lat = np.radians(40.0)  # Outside bounds
        lon = np.radians(-119.5)
        point = simple_dem.get_elevation(lat, lon)

        assert not point.valid
        assert point.elevation == simple_dem.nodata_value

    def test_get_elevation_nearest(self, simple_dem):
        """Test nearest neighbor interpolation."""
        lat = np.radians(35.5)
        lon = np.radians(-119.5)
        point = simple_dem.get_elevation(lat, lon, interpolation="nearest")

        assert point.valid

    def test_get_elevations_batch(self, simple_dem):
        """Test batch elevation queries."""
        lats = np.radians([35.2, 35.5, 35.8])
        lons = np.radians([-119.8, -119.5, -119.2])
        elevs = simple_dem.get_elevations(lats, lons)

        assert len(elevs) == 3
        # Elevations should increase (data increases NE)
        assert elevs[0] < elevs[1] < elevs[2]

    def test_get_gradient(self, simple_dem):
        """Test gradient computation."""
        lat = np.radians(35.5)
        lon = np.radians(-119.5)
        grad = simple_dem.get_gradient(lat, lon)

        assert isinstance(grad, TerrainGradient)
        # Slope should be positive (terrain is not flat)
        assert grad.slope >= 0
        # Gradients should be non-zero
        assert grad.dz_dx != 0 or grad.dz_dy != 0


class TestDEMGridFlat:
    """Tests for flat DEM."""

    def test_create_flat_dem(self):
        """Test flat DEM creation."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=500.0,
            resolution_arcsec=300.0,  # 5 arcmin
        )

        # Query anywhere
        point = dem.get_elevation(np.radians(35.5), np.radians(-119.5))
        assert point.elevation == pytest.approx(500.0)
        assert point.valid

    def test_flat_dem_gradient(self):
        """Test that flat DEM has zero slope."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=100.0,
        )

        grad = dem.get_gradient(np.radians(35.5), np.radians(-119.5))
        assert grad.slope == pytest.approx(0.0, abs=1e-6)


class TestSyntheticTerrain:
    """Tests for synthetic terrain generation."""

    def test_create_synthetic_terrain(self):
        """Test synthetic terrain creation."""
        dem = create_synthetic_terrain(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            base_elevation=500.0,
            amplitude=200.0,
            seed=42,
        )

        assert dem.n_lat > 0
        assert dem.n_lon > 0

        # Check elevation range
        point = dem.get_elevation(np.radians(35.5), np.radians(-119.5))
        assert point.valid

    def test_synthetic_terrain_reproducibility(self):
        """Test synthetic terrain is reproducible with seed."""
        dem1 = create_synthetic_terrain(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            seed=123,
        )
        dem2 = create_synthetic_terrain(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            seed=123,
        )

        np.testing.assert_array_almost_equal(dem1.data, dem2.data)


# ============================================================================
# DEM Utility Function Tests
# ============================================================================


class TestElevationProfile:
    """Tests for elevation profile extraction."""

    def test_get_elevation_profile(self):
        """Test profile extraction."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=100.0,
        )

        distances, elevations = get_elevation_profile(
            dem,
            lat_start=np.radians(35.2),
            lon_start=np.radians(-119.8),
            lat_end=np.radians(35.8),
            lon_end=np.radians(-119.2),
            n_points=50,
        )

        assert len(distances) == 50
        assert len(elevations) == 50
        # Start distance should be 0
        assert distances[0] == pytest.approx(0.0)
        # All elevations should be 100 for flat DEM
        np.testing.assert_array_almost_equal(elevations, 100.0)


class TestInterpolateDEM:
    """Tests for DEM interpolation."""

    def test_interpolate_dem(self):
        """Test DEM resampling."""
        # Create source DEM
        source = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=250.0,
            resolution_arcsec=120.0,
        )

        # Interpolate to finer grid
        result = interpolate_dem(
            source,
            new_lat_min=np.radians(35.25),
            new_lat_max=np.radians(35.75),
            new_lon_min=np.radians(-119.75),
            new_lon_max=np.radians(-119.25),
            new_n_lat=20,
            new_n_lon=20,
        )

        assert result.n_lat == 20
        assert result.n_lon == 20
        # Elevations should still be ~250
        point = result.get_elevation(np.radians(35.5), np.radians(-119.5))
        assert point.elevation == pytest.approx(250.0, abs=1)


class TestMergeDEMs:
    """Tests for DEM merging."""

    def test_merge_dems(self):
        """Test merging multiple DEMs."""
        dem1 = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(35.5),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.5),
            elevation=100.0,
        )
        dem2 = create_flat_dem(
            lat_min=np.radians(35.5),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.5),
            elevation=200.0,
        )

        merged = merge_dems(
            [dem1, dem2],
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.5),
            resolution_arcsec=120.0,
        )

        # Check that both regions have correct elevations
        point1 = merged.get_elevation(np.radians(35.25), np.radians(-119.75))
        point2 = merged.get_elevation(np.radians(35.75), np.radians(-119.75))

        assert point1.valid
        assert point1.elevation == pytest.approx(100.0, abs=10)
        assert point2.valid
        assert point2.elevation == pytest.approx(200.0, abs=10)


# ============================================================================
# Visibility Data Structure Tests
# ============================================================================


class TestLOSResult:
    """Tests for LOSResult named tuple."""

    def test_creation(self):
        """Test LOSResult creation."""
        result = LOSResult(
            visible=True,
            grazing_angle=np.radians(5.0),
            obstacle_distance=0.0,
            obstacle_elevation=0.0,
            clearance=100.0,
        )
        assert result.visible is True
        assert result.clearance == 100.0


class TestViewshedResult:
    """Tests for ViewshedResult named tuple."""

    def test_creation(self):
        """Test ViewshedResult creation."""
        visible = np.zeros((10, 10), dtype=bool)
        visible[5, 5] = True

        result = ViewshedResult(
            visible=visible,
            observer_lat=np.radians(35.5),
            observer_lon=np.radians(-119.5),
            observer_height=10.0,
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
        )
        assert result.observer_height == 10.0
        assert result.visible[5, 5]


class TestHorizonPoint:
    """Tests for HorizonPoint named tuple."""

    def test_creation(self):
        """Test HorizonPoint creation."""
        hp = HorizonPoint(
            azimuth=np.radians(45.0),
            elevation_angle=np.radians(2.0),
            distance=10000.0,
            terrain_elevation=500.0,
        )
        assert hp.distance == 10000.0
        assert hp.terrain_elevation == 500.0


# ============================================================================
# Line of Sight Tests
# ============================================================================


class TestLineOfSight:
    """Tests for line_of_sight function."""

    @pytest.fixture
    def flat_dem(self):
        """Create flat DEM for LOS testing."""
        return create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
        )

    def test_los_flat_terrain_visible(self, flat_dem):
        """Test LOS over flat terrain is visible."""
        result = line_of_sight(
            flat_dem,
            obs_lat=np.radians(35.2),
            obs_lon=np.radians(-119.8),
            obs_height=10.0,
            tgt_lat=np.radians(35.8),
            tgt_lon=np.radians(-119.2),
            tgt_height=10.0,
        )

        assert result.visible
        assert result.clearance > 0

    def test_los_same_point(self, flat_dem):
        """Test LOS to same point."""
        result = line_of_sight(
            flat_dem,
            obs_lat=np.radians(35.5),
            obs_lon=np.radians(-119.5),
            obs_height=10.0,
            tgt_lat=np.radians(35.5),
            tgt_lon=np.radians(-119.5),
            tgt_height=10.0,
        )

        assert result.visible

    def test_los_with_obstacle(self):
        """Test LOS blocked by obstacle."""
        # Create DEM with ridge in middle
        data = np.zeros((20, 20))
        data[9:11, :] = 500.0  # Ridge across middle

        dem = DEMGrid(
            data,
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
        )

        result = line_of_sight(
            dem,
            obs_lat=np.radians(35.2),  # South of ridge
            obs_lon=np.radians(-119.5),
            obs_height=10.0,
            tgt_lat=np.radians(35.8),  # North of ridge
            tgt_lon=np.radians(-119.5),
            tgt_height=10.0,
            n_samples=200,
        )

        assert not result.visible
        assert result.clearance < 0

    def test_los_with_refraction(self, flat_dem):
        """Test LOS with atmospheric refraction."""
        result = line_of_sight(
            flat_dem,
            obs_lat=np.radians(35.2),
            obs_lon=np.radians(-119.8),
            obs_height=10.0,
            tgt_lat=np.radians(35.8),
            tgt_lon=np.radians(-119.2),
            tgt_height=10.0,
            refraction_coeff=0.13,  # 4/3 Earth model
        )

        assert result.visible


# ============================================================================
# Viewshed Tests
# ============================================================================


class TestViewshed:
    """Tests for viewshed function."""

    def test_viewshed_flat_terrain(self):
        """Test viewshed on flat terrain."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
            resolution_arcsec=120.0,  # Coarse for speed
        )

        result = viewshed(
            dem,
            obs_lat=np.radians(35.5),
            obs_lon=np.radians(-119.5),
            obs_height=100.0,
            max_range=50000.0,
            n_radials=36,  # Every 10 degrees for speed
            samples_per_radial=20,
        )

        assert isinstance(result, ViewshedResult)
        # Observer location should be visible
        assert result.visible.any()

    def test_viewshed_observer_visible(self):
        """Test that observer location is marked visible."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
        )

        result = viewshed(
            dem,
            obs_lat=np.radians(35.5),
            obs_lon=np.radians(-119.5),
            obs_height=10.0,
            n_radials=8,
            samples_per_radial=10,
        )

        # Find observer grid cell
        i, j, _, _ = dem._get_indices(np.radians(35.5), np.radians(-119.5))
        assert result.visible[i, j]


# ============================================================================
# Horizon Tests
# ============================================================================


class TestComputeHorizon:
    """Tests for compute_horizon function."""

    def test_horizon_flat_terrain(self):
        """Test horizon on flat terrain."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
        )

        horizon = compute_horizon(
            dem,
            obs_lat=np.radians(35.5),
            obs_lon=np.radians(-119.5),
            obs_height=100.0,
            n_azimuths=8,
            samples_per_radial=20,
        )

        assert len(horizon) == 8
        # All horizon points should be HorizonPoint
        for hp in horizon:
            assert isinstance(hp, HorizonPoint)
            # On flat terrain with elevated observer, horizon should be below horizontal
            # (looking down due to Earth curvature)
            assert hp.elevation_angle < np.radians(5)

    def test_horizon_coverage(self):
        """Test horizon covers all azimuths."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
        )

        horizon = compute_horizon(
            dem,
            obs_lat=np.radians(35.5),
            obs_lon=np.radians(-119.5),
            obs_height=10.0,
            n_azimuths=36,
        )

        assert len(horizon) == 36
        # Check azimuths are evenly spaced
        azimuths = [hp.azimuth for hp in horizon]
        expected_step = 2 * np.pi / 36
        for i in range(1, len(azimuths)):
            assert azimuths[i] - azimuths[i - 1] == pytest.approx(expected_step)


# ============================================================================
# Terrain Masking Tests
# ============================================================================


class TestTerrainMaskingAngle:
    """Tests for terrain_masking_angle function."""

    def test_masking_angle_flat(self):
        """Test masking angle on flat terrain."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
        )

        angle = terrain_masking_angle(
            dem,
            obs_lat=np.radians(35.5),
            obs_lon=np.radians(-119.5),
            obs_height=100.0,
            azimuth=0.0,  # North
        )

        # On flat terrain, masking angle should be negative (looking down)
        assert angle < np.radians(5)


# ============================================================================
# Radar Coverage Tests
# ============================================================================


class TestRadarCoverageMap:
    """Tests for radar_coverage_map function."""

    def test_radar_coverage_basic(self):
        """Test basic radar coverage map."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
            resolution_arcsec=120.0,
        )

        result = radar_coverage_map(
            dem,
            radar_lat=np.radians(35.5),
            radar_lon=np.radians(-119.5),
            radar_height=50.0,
            min_elevation=0.0,
            max_range=50000.0,
            target_height=1000.0,
            n_radials=16,
            samples_per_radial=20,
        )

        assert isinstance(result, ViewshedResult)
        assert result.visible.any()

    def test_radar_coverage_min_elevation(self):
        """Test radar coverage with minimum elevation constraint."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=0.0,
            resolution_arcsec=120.0,
        )

        # With very high minimum elevation, coverage should be limited
        result = radar_coverage_map(
            dem,
            radar_lat=np.radians(35.5),
            radar_lon=np.radians(-119.5),
            radar_height=10.0,
            min_elevation=np.radians(30.0),  # 30 degree minimum
            max_range=50000.0,
            target_height=100.0,  # Low flying target
            n_radials=16,
            samples_per_radial=20,
        )

        # Coverage should be very limited with high min elevation
        visible_fraction = np.sum(result.visible) / result.visible.size
        assert visible_fraction < 0.5  # Less than half should be visible


# ============================================================================
# Integration Tests
# ============================================================================


class TestTerrainIntegration:
    """Integration tests for terrain module."""

    def test_synthetic_terrain_workflow(self):
        """Test complete workflow with synthetic terrain."""
        # Create synthetic terrain
        dem = create_synthetic_terrain(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            base_elevation=500.0,
            amplitude=300.0,
            seed=42,
            resolution_arcsec=60.0,
        )

        # Get metadata
        meta = dem.get_metadata()
        assert meta.name == "Synthetic Terrain"

        # Query elevation
        point = dem.get_elevation(np.radians(35.5), np.radians(-119.5))
        assert point.valid

        # Compute gradient
        grad = dem.get_gradient(np.radians(35.5), np.radians(-119.5))
        assert grad.slope >= 0

        # Extract profile
        distances, elevations = get_elevation_profile(
            dem,
            lat_start=np.radians(35.2),
            lon_start=np.radians(-119.8),
            lat_end=np.radians(35.8),
            lon_end=np.radians(-119.2),
            n_points=20,
        )
        assert len(distances) == 20

        # Check line of sight
        los = line_of_sight(
            dem,
            obs_lat=np.radians(35.2),
            obs_lon=np.radians(-119.8),
            obs_height=100.0,
            tgt_lat=np.radians(35.8),
            tgt_lon=np.radians(-119.2),
            tgt_height=100.0,
        )
        assert isinstance(los, LOSResult)

    def test_multiple_dem_queries(self):
        """Test multiple sequential DEM queries."""
        dem = create_flat_dem(
            lat_min=np.radians(35.0),
            lat_max=np.radians(36.0),
            lon_min=np.radians(-120.0),
            lon_max=np.radians(-119.0),
            elevation=100.0,
        )

        # Query many points
        for i in range(100):
            lat = np.radians(35.0 + np.random.rand())
            lon = np.radians(-120.0 + np.random.rand())
            point = dem.get_elevation(lat, lon)
            assert point.valid
            assert point.elevation == pytest.approx(100.0)
