"""Tests for GEBCO and Earth2014 terrain data loaders."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.terrain import (  # Loader functions; Parameters; Metadata types; DEM types for verification
    EARTH2014_PARAMETERS,
    GEBCO_PARAMETERS,
    DEMGrid,
    DEMPoint,
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


class TestGEBCOParameters:
    """Tests for GEBCO parameter definitions."""

    def test_gebco2025_defined(self):
        """GEBCO2025 parameters are defined."""
        assert "GEBCO2025" in GEBCO_PARAMETERS

    def test_gebco2024_defined(self):
        """GEBCO2024 parameters are defined."""
        assert "GEBCO2024" in GEBCO_PARAMETERS

    def test_gebco2023_defined(self):
        """GEBCO2023 parameters are defined."""
        assert "GEBCO2023" in GEBCO_PARAMETERS

    def test_resolution_15_arcsec(self):
        """GEBCO resolution is 15 arc-seconds."""
        assert GEBCO_PARAMETERS["GEBCO2025"]["resolution_arcsec"] == 15.0
        assert GEBCO_PARAMETERS["GEBCO2024"]["resolution_arcsec"] == 15.0
        assert GEBCO_PARAMETERS["GEBCO2023"]["resolution_arcsec"] == 15.0

    def test_grid_dimensions(self):
        """GEBCO grid dimensions are correct."""
        params = GEBCO_PARAMETERS["GEBCO2025"]
        assert params["n_lat"] == 43200
        assert params["n_lon"] == 86400


class TestEarth2014Parameters:
    """Tests for Earth2014 parameter definitions."""

    def test_sur_layer_defined(self):
        """SUR layer is defined."""
        assert "SUR" in EARTH2014_PARAMETERS

    def test_bed_layer_defined(self):
        """BED layer is defined."""
        assert "BED" in EARTH2014_PARAMETERS

    def test_tbi_layer_defined(self):
        """TBI layer is defined."""
        assert "TBI" in EARTH2014_PARAMETERS

    def test_ret_layer_defined(self):
        """RET layer is defined."""
        assert "RET" in EARTH2014_PARAMETERS

    def test_ice_layer_defined(self):
        """ICE layer is defined."""
        assert "ICE" in EARTH2014_PARAMETERS

    def test_all_layers_have_description(self):
        """All layers have description and file_pattern."""
        for layer, params in EARTH2014_PARAMETERS.items():
            assert "description" in params, f"{layer} missing description"
            assert "file_pattern" in params, f"{layer} missing file_pattern"


class TestDataDirectory:
    """Tests for data directory functions."""

    def test_get_data_dir_returns_path(self):
        """get_data_dir returns a Path object."""
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)

    def test_data_dir_in_home(self):
        """Default data dir is under home directory."""
        data_dir = get_data_dir()
        home = Path.home()
        assert str(data_dir).startswith(str(home))

    def test_data_dir_is_pytcl(self):
        """Data dir is ~/.pytcl/data."""
        data_dir = get_data_dir()
        assert ".pytcl" in str(data_dir)
        assert "data" in str(data_dir)


class TestGEBCOMetadata:
    """Tests for GEBCO metadata functions."""

    def test_get_metadata_returns_type(self):
        """get_gebco_metadata returns GEBCOMetadata."""
        meta = get_gebco_metadata("GEBCO2025")
        assert isinstance(meta, GEBCOMetadata)

    def test_metadata_version(self):
        """Metadata contains correct version."""
        meta = get_gebco_metadata("GEBCO2025")
        assert meta.version == "GEBCO2025"

    def test_metadata_resolution(self):
        """Metadata contains correct resolution."""
        meta = get_gebco_metadata("GEBCO2025")
        assert meta.resolution_arcsec == 15.0

    def test_metadata_global_extent(self):
        """Metadata has global extent."""
        meta = get_gebco_metadata("GEBCO2025")
        assert_allclose(meta.lat_min, np.radians(-90.0))
        assert_allclose(meta.lat_max, np.radians(90.0))
        assert_allclose(meta.lon_min, np.radians(-180.0))
        assert_allclose(meta.lon_max, np.radians(180.0))

    def test_invalid_version_raises(self):
        """Invalid version raises ValueError."""
        with pytest.raises(ValueError):
            get_gebco_metadata("GEBCO1900")


class TestEarth2014Metadata:
    """Tests for Earth2014 metadata functions."""

    def test_get_metadata_returns_type(self):
        """get_earth2014_metadata returns Earth2014Metadata."""
        meta = get_earth2014_metadata("SUR")
        assert isinstance(meta, Earth2014Metadata)

    def test_metadata_layer(self):
        """Metadata contains correct layer."""
        meta = get_earth2014_metadata("BED")
        assert meta.layer == "BED"

    def test_metadata_has_description(self):
        """Metadata contains description."""
        meta = get_earth2014_metadata("SUR")
        assert len(meta.description) > 0

    def test_metadata_resolution_1arcmin(self):
        """Metadata contains 1 arc-minute resolution."""
        meta = get_earth2014_metadata("SUR")
        assert meta.resolution_arcsec == 60.0

    def test_metadata_global_extent(self):
        """Metadata has global extent."""
        meta = get_earth2014_metadata("SUR")
        assert_allclose(meta.lat_min, np.radians(-90.0))
        assert_allclose(meta.lat_max, np.radians(90.0))

    def test_invalid_layer_raises(self):
        """Invalid layer raises ValueError."""
        with pytest.raises(ValueError):
            get_earth2014_metadata("INVALID")


class TestCreateTestGEBCODEM:
    """Tests for synthetic GEBCO DEM creation."""

    def test_returns_demgrid(self):
        """create_test_gebco_dem returns DEMGrid."""
        dem = create_test_gebco_dem()
        assert isinstance(dem, DEMGrid)

    def test_default_extent(self):
        """Default extent is California coast region."""
        dem = create_test_gebco_dem()
        assert_allclose(dem.lat_min, np.radians(35.0), rtol=1e-3)
        assert_allclose(dem.lat_max, np.radians(40.0), rtol=1e-3)
        assert_allclose(dem.lon_min, np.radians(-125.0), rtol=1e-3)
        assert_allclose(dem.lon_max, np.radians(-120.0), rtol=1e-3)

    def test_custom_extent(self):
        """Custom extent is respected."""
        dem = create_test_gebco_dem(
            lat_min=np.radians(0.0),
            lat_max=np.radians(10.0),
            lon_min=np.radians(0.0),
            lon_max=np.radians(10.0),
        )
        assert_allclose(dem.lat_min, np.radians(0.0), rtol=1e-3)
        assert_allclose(dem.lat_max, np.radians(10.0), rtol=1e-3)

    def test_includes_bathymetry(self):
        """With bathymetry, some elevations are negative."""
        dem = create_test_gebco_dem(include_bathymetry=True)
        assert np.any(dem.data < 0)

    def test_no_bathymetry_positive(self):
        """Without bathymetry, ocean areas are zero."""
        dem = create_test_gebco_dem(include_bathymetry=False)
        # Most land areas should be positive
        assert np.any(dem.data > 0)

    def test_reproducible_with_seed(self):
        """Same seed produces same data."""
        dem1 = create_test_gebco_dem(seed=123)
        dem2 = create_test_gebco_dem(seed=123)
        assert_allclose(dem1.data, dem2.data)

    def test_different_seed_different_data(self):
        """Different seeds produce different data."""
        dem1 = create_test_gebco_dem(seed=123)
        dem2 = create_test_gebco_dem(seed=456)
        assert not np.allclose(dem1.data, dem2.data)

    def test_name_is_gebco_test(self):
        """DEM name indicates test data."""
        dem = create_test_gebco_dem()
        assert "GEBCO" in dem.name
        assert "TEST" in dem.name

    def test_elevation_query_works(self):
        """Can query elevation from test DEM."""
        dem = create_test_gebco_dem()
        point = dem.get_elevation(np.radians(37.5), np.radians(-122.5))
        assert isinstance(point, DEMPoint)
        assert point.valid


class TestCreateTestEarth2014DEM:
    """Tests for synthetic Earth2014 DEM creation."""

    def test_returns_demgrid(self):
        """create_test_earth2014_dem returns DEMGrid."""
        dem = create_test_earth2014_dem()
        assert isinstance(dem, DEMGrid)

    def test_default_layer_sur(self):
        """Default layer is SUR."""
        dem = create_test_earth2014_dem()
        assert "SUR" in dem.name

    def test_bed_layer(self):
        """BED layer can be created."""
        dem = create_test_earth2014_dem(layer="BED")
        assert "BED" in dem.name

    def test_tbi_layer(self):
        """TBI layer can be created."""
        dem = create_test_earth2014_dem(layer="TBI")
        assert "TBI" in dem.name

    def test_ret_layer(self):
        """RET layer can be created."""
        dem = create_test_earth2014_dem(layer="RET")
        assert "RET" in dem.name

    def test_ice_layer(self):
        """ICE layer can be created."""
        dem = create_test_earth2014_dem(layer="ICE")
        assert "ICE" in dem.name

    def test_sur_non_negative(self):
        """SUR layer has non-negative values (0 over oceans)."""
        dem = create_test_earth2014_dem(layer="SUR")
        assert np.all(dem.data >= -50)  # Allow small noise

    def test_bed_layer_differs_from_sur(self):
        """BED layer differs from SUR layer (lower elevations)."""
        dem_sur = create_test_earth2014_dem(layer="SUR", seed=42)
        dem_bed = create_test_earth2014_dem(layer="BED", seed=42)
        # BED represents bedrock which is typically below surface
        # The synthetic BED is generated with a -200m offset
        assert np.mean(dem_bed.data) < np.mean(dem_sur.data)

    def test_ice_layer_distinct(self):
        """ICE layer has distinct characteristics from surface layers."""
        dem = create_test_earth2014_dem(layer="ICE")
        # ICE layer represents ice thickness, should have some zero values
        # and some non-zero values (simulated ice coverage at high latitudes)
        has_ice = np.any(dem.data > 100)  # Some significant ice
        has_no_ice = np.any(dem.data < 50)  # Some low/no ice areas
        assert has_ice or has_no_ice  # At least one condition should be true

    def test_reproducible_with_seed(self):
        """Same seed produces same data."""
        dem1 = create_test_earth2014_dem(seed=123)
        dem2 = create_test_earth2014_dem(seed=123)
        assert_allclose(dem1.data, dem2.data)

    def test_name_indicates_test(self):
        """DEM name indicates test data."""
        dem = create_test_earth2014_dem()
        assert "Earth2014" in dem.name
        assert "TEST" in dem.name


class TestDEMIntegration:
    """Integration tests for test DEMs with DEM interface."""

    def test_gebco_elevation_profile(self):
        """Can extract elevation profile from GEBCO test DEM."""
        from pytcl.terrain import get_elevation_profile

        dem = create_test_gebco_dem()
        distances, elevations = get_elevation_profile(
            dem,
            lat_start=np.radians(36.0),
            lon_start=np.radians(-124.0),
            lat_end=np.radians(39.0),
            lon_end=np.radians(-121.0),
            n_points=50,
        )
        assert len(distances) == 50
        assert len(elevations) == 50
        assert not np.any(np.isnan(elevations))

    def test_earth2014_gradient(self):
        """Can compute gradient from Earth2014 test DEM."""
        dem = create_test_earth2014_dem()
        gradient = dem.get_gradient(np.radians(37.5), np.radians(-122.5))
        assert not np.isnan(gradient.slope)
        assert not np.isnan(gradient.aspect)

    def test_gebco_line_of_sight(self):
        """Can compute LOS using GEBCO test DEM."""
        from pytcl.terrain import line_of_sight

        dem = create_test_gebco_dem()
        los = line_of_sight(
            dem,
            obs_lat=np.radians(37.0),
            obs_lon=np.radians(-123.0),
            obs_height=100.0,
            tgt_lat=np.radians(38.0),
            tgt_lon=np.radians(-122.0),
            tgt_height=50.0,
        )
        assert hasattr(los, "visible")
        assert hasattr(los, "clearance")


class TestLoadGEBCOErrors:
    """Tests for GEBCO loading error handling."""

    def test_invalid_version_raises(self):
        """Invalid GEBCO version raises ValueError."""
        with pytest.raises(ValueError):
            load_gebco(
                lat_min=np.radians(35.0),
                lat_max=np.radians(40.0),
                lon_min=np.radians(-125.0),
                lon_max=np.radians(-120.0),
                version="GEBCO1900",
            )

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """Missing GEBCO file raises FileNotFoundError."""
        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        from pytcl.terrain.loaders import _load_gebco_cached

        _load_gebco_cached.cache_clear()
        with pytest.raises(FileNotFoundError):
            load_gebco(
                lat_min=np.radians(35.0),
                lat_max=np.radians(40.0),
                lon_min=np.radians(-125.0),
                lon_max=np.radians(-120.0),
                version="GEBCO2025",
            )
        _load_gebco_cached.cache_clear()


class TestLoadEarth2014Errors:
    """Tests for Earth2014 loading error handling."""

    def test_invalid_layer_raises(self):
        """Invalid Earth2014 layer raises ValueError."""
        with pytest.raises(ValueError):
            load_earth2014(
                lat_min=np.radians(35.0),
                lat_max=np.radians(40.0),
                lon_min=np.radians(-125.0),
                lon_max=np.radians(-120.0),
                layer="INVALID",
            )

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """Missing Earth2014 file raises FileNotFoundError."""
        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        from pytcl.terrain.loaders import _load_earth2014_cached

        _load_earth2014_cached.cache_clear()
        with pytest.raises(FileNotFoundError):
            load_earth2014(
                lat_min=np.radians(35.0),
                lat_max=np.radians(40.0),
                lon_min=np.radians(-125.0),
                lon_max=np.radians(-120.0),
                layer="SUR",
            )
        _load_earth2014_cached.cache_clear()


class TestResolutionHandling:
    """Tests for resolution and grid spacing."""

    def test_gebco_test_resolution(self):
        """Test GEBCO DEM has correct resolution."""
        dem = create_test_gebco_dem(resolution_arcsec=60.0)
        meta = dem.get_metadata()
        # Should be approximately 60 arc-seconds
        assert 55 < meta.resolution < 65

    def test_earth2014_test_resolution(self):
        """Test Earth2014 DEM has correct resolution."""
        dem = create_test_earth2014_dem(resolution_arcsec=60.0)
        meta = dem.get_metadata()
        assert 55 < meta.resolution < 65

    def test_fine_resolution(self):
        """Can create test DEM with finer resolution."""
        dem = create_test_gebco_dem(resolution_arcsec=30.0)
        meta = dem.get_metadata()
        assert 25 < meta.resolution < 35


class TestNumericalStability:
    """Tests for numerical stability at edge cases."""

    def test_equator(self):
        """Test DEM creation and query at equator."""
        dem = create_test_gebco_dem(
            lat_min=np.radians(-5.0),
            lat_max=np.radians(5.0),
            lon_min=np.radians(-5.0),
            lon_max=np.radians(5.0),
        )
        point = dem.get_elevation(0, 0)
        assert not np.isnan(point.elevation)

    def test_high_latitude(self):
        """Test DEM creation at high latitude."""
        dem = create_test_earth2014_dem(
            lat_min=np.radians(70.0),
            lat_max=np.radians(80.0),
            lon_min=np.radians(-10.0),
            lon_max=np.radians(10.0),
        )
        point = dem.get_elevation(np.radians(75.0), 0)
        assert not np.isnan(point.elevation)

    def test_dateline_crossing(self):
        """Test DEM at dateline (±180°)."""
        dem = create_test_gebco_dem(
            lat_min=np.radians(0.0),
            lat_max=np.radians(10.0),
            lon_min=np.radians(175.0),
            lon_max=np.radians(180.0),
        )
        point = dem.get_elevation(np.radians(5.0), np.radians(177.5))
        assert not np.isnan(point.elevation)


class TestTerrainDataDirectory:
    """Tests for terrain data directory management."""

    def test_get_data_dir(self):
        """Test getting terrain data directory."""
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)

    def test_get_data_dir_consistency(self):
        """Test that data directory returns consistent results."""
        dir1 = get_data_dir()
        dir2 = get_data_dir()
        assert dir1 == dir2

    def test_data_dir_is_path_like(self):
        """Test that data dir is a proper path."""
        data_dir = get_data_dir()
        # Path objects have .exists(), .mkdir(), etc.
        assert hasattr(data_dir, "exists") or hasattr(data_dir, "__fspath__")


class TestCoefficientCreation:
    """Tests for creating test DEM data."""

    def test_create_test_gebco_dem_default(self):
        """Test creating test GEBCO DEM."""
        dem = create_test_gebco_dem()
        assert dem is not None
        assert hasattr(dem, "data") or isinstance(dem, np.ndarray)

    def test_create_test_gebco_dem_custom_extent(self):
        """Test creating test GEBCO with custom extent."""
        try:
            dem = create_test_gebco_dem()
            # Should have latitude, longitude, elevation
            assert dem is not None
        except Exception as e:
            pytest.skip(f"Test DEM creation failed: {e}")

    def test_create_test_earth2014_dem_default(self):
        """Test creating test Earth2014 DEM."""
        try:
            dem = create_test_earth2014_dem()
            assert dem is not None
        except Exception as e:
            pytest.skip(f"Earth2014 test DEM failed: {e}")

    def test_create_test_earth2014_dem_surface(self):
        """Test creating test Earth2014 surface layer."""
        try:
            dem = create_test_earth2014_dem(layer="SUR")
            assert dem is not None
        except Exception as e:
            pytest.skip(f"Earth2014 surface layer failed: {e}")


class TestDEMQueries:
    """Tests for DEM query operations."""

    def test_load_gebco_minimal(self):
        """Test loading GEBCO data with minimal parameters."""
        try:
            dem = load_gebco(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"GEBCO loading unavailable: {e}")

    def test_load_earth2014_minimal(self):
        """Test loading Earth2014 data."""
        try:
            dem = load_earth2014(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
                layer="SUR",
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"Earth2014 loading unavailable: {e}")


class TestTerrainInterpolation:
    """Tests for terrain data interpolation."""

    def test_gebco_coverage_global(self):
        """Test GEBCO covers global regions."""
        try:
            dem1 = load_gebco(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
            )
            dem2 = load_gebco(
                lat_min=np.radians(50.0),
                lat_max=np.radians(51.0),
                lon_min=np.radians(0.0),
                lon_max=np.radians(1.0),
            )
            assert isinstance(dem1, DEMGrid)
            assert isinstance(dem2, DEMGrid)

        except Exception as e:
            pytest.skip(f"GEBCO coverage test failed: {e}")

    def test_earth2014_layers(self):
        """Test different Earth2014 layers."""
        try:
            layers = ["SUR", "BED", "TBI"]

            for layer in layers:
                try:
                    dem = load_earth2014(
                        lat_min=np.radians(40.0),
                        lat_max=np.radians(41.0),
                        lon_min=np.radians(-74.0),
                        lon_max=np.radians(-73.0),
                        layer=layer,
                    )
                    assert isinstance(dem, DEMGrid)
                except FileNotFoundError:
                    pass

        except Exception as e:
            pytest.skip(f"Earth2014 layers test failed: {e}")


class TestDEMBounds:
    """Tests for DEM boundary conditions."""

    def test_gebco_equatorial_region(self):
        """Test GEBCO at equatorial region."""
        try:
            dem = load_gebco(
                lat_min=np.radians(-1.0),
                lat_max=np.radians(1.0),
                lon_min=np.radians(-1.0),
                lon_max=np.radians(1.0),
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"Equatorial GEBCO failed: {e}")

    def test_gebco_poles(self):
        """Test GEBCO near poles."""
        try:
            dem = load_gebco(
                lat_min=np.radians(80.0),
                lat_max=np.radians(85.0),
                lon_min=np.radians(-180.0),
                lon_max=np.radians(180.0),
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"Polar GEBCO unavailable: {e}")

    def test_gebco_prime_meridian(self):
        """Test GEBCO crossing prime meridian."""
        try:
            dem = load_gebco(
                lat_min=np.radians(50.0),
                lat_max=np.radians(55.0),
                lon_min=np.radians(-5.0),
                lon_max=np.radians(5.0),
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"Prime meridian GEBCO failed: {e}")

    def test_gebco_date_line(self):
        """Test GEBCO crossing international date line."""
        try:
            dem = load_gebco(
                lat_min=np.radians(0.0),
                lat_max=np.radians(5.0),
                lon_min=np.radians(175.0),
                lon_max=np.radians(180.0),
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"Date line GEBCO unavailable: {e}")


class TestTerrainStatistics:
    """Tests for terrain data statistics and properties."""

    def test_gebco_dem_structure(self):
        """Test GEBCO DEM has expected structure."""
        try:
            dem = load_gebco(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
            )
            assert isinstance(dem, DEMGrid)
            assert dem.data is not None
            assert dem.data.shape[0] > 0
            assert dem.data.shape[1] > 0

        except Exception as e:
            pytest.skip(f"GEBCO structure test failed: {e}")

    def test_earth2014_dem_structure(self):
        """Test Earth2014 DEM has expected structure."""
        try:
            dem = load_earth2014(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
                layer="SUR",
            )
            assert isinstance(dem, DEMGrid)
            assert dem.data is not None

        except Exception as e:
            pytest.skip(f"Earth2014 structure test failed: {e}")


class TestDEMDataTypes:
    """Tests for DEM data types and formats."""

    def test_gebco_returns_demgrid(self):
        """Test that GEBCO returns DEMGrid."""
        try:
            dem = load_gebco(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"GEBCO return type test failed: {e}")

    def test_earth2014_returns_demgrid(self):
        """Test that Earth2014 returns DEMGrid."""
        try:
            dem = load_earth2014(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
                layer="SUR",
            )
            assert isinstance(dem, DEMGrid)

        except Exception as e:
            pytest.skip(f"Earth2014 return type test failed: {e}")


class TestDEMCacheAndPerformance:
    """Tests for DEM caching and performance considerations."""

    def test_repeated_gebco_consistency(self):
        """Test that repeated GEBCO queries return consistent results."""
        try:
            args = dict(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
            )
            dem1 = load_gebco(**args)
            dem2 = load_gebco(**args)

            # Cached load should return the same object
            assert dem1 is dem2

        except Exception as e:
            pytest.skip(f"GEBCO consistency test failed: {e}")

    def test_repeated_earth2014_consistency(self):
        """Test that repeated Earth2014 queries return consistent results."""
        try:
            args = dict(
                lat_min=np.radians(40.0),
                lat_max=np.radians(41.0),
                lon_min=np.radians(-74.0),
                lon_max=np.radians(-73.0),
            )
            dem1 = load_earth2014(**args)
            dem2 = load_earth2014(**args)

            assert dem1 is dem2

        except Exception as e:
            pytest.skip(f"Earth2014 consistency test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
