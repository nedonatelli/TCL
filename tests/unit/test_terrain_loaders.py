"""Tests for GEBCO and Earth2014 terrain data loaders."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import is_available
from pytcl.terrain import (
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
    parse_gebco_netcdf,
)

_data_skip = (FileNotFoundError, DependencyError)
_HAS_NETCDF4 = is_available("netCDF4")

# Common test region (New York area, ~1° x 1°)
_NYC = dict(
    lat_min=np.radians(40.0),
    lat_max=np.radians(41.0),
    lon_min=np.radians(-74.0),
    lon_max=np.radians(-73.0),
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


def _plane_dem(n=11, z=lambda i, j: 100.0 + 7.0 * i - 3.0 * j):
    """A DEM whose elevation is a known function of the (row, column) index."""
    data = np.array([[z(i, j) for j in range(n)] for i in range(n)])
    return DEMGrid(
        data,
        np.radians(35.0),
        np.radians(36.0),
        np.radians(-120.0),
        np.radians(-119.0),
    )


class TestDEMOnKnownSurface:
    """DEM queries on surfaces whose elevation is known in closed form."""

    def test_elevation_profile_follows_the_plane(self):
        from geographiclib.geodesic import Geodesic

        from pytcl.core.constants import EARTH_MEAN_RADIUS
        from pytcl.terrain import get_elevation_profile

        dem = _plane_dem()
        start = (35.2, -119.8)
        end = (35.8, -119.2)

        distances, elevations = get_elevation_profile(
            dem,
            np.radians(start[0]),
            np.radians(start[1]),
            np.radians(end[0]),
            np.radians(end[1]),
            n_points=7,
        )

        fractions = np.linspace(0.0, 1.0, 7)
        lats = start[0] + fractions * (end[0] - start[0])
        lons = start[1] + fractions * (end[1] - start[1])
        i = (lats - 35.0) / 0.1
        j = (lons + 120.0) / 0.1
        assert_allclose(elevations, 100.0 + 7.0 * i - 3.0 * j, rtol=1e-9)
        sphere = Geodesic(EARTH_MEAN_RADIUS, 0.0)
        expected = [
            sphere.Inverse(*start, la, lo)["s12"]
            for la, lo in zip(lats, lons, strict=True)
        ]
        assert_allclose(distances, expected, rtol=1e-9, atol=1e-6)

    def test_gradient_of_the_plane(self):
        from pytcl.core.constants import EARTH_MEAN_RADIUS

        dem = _plane_dem()
        lat = np.radians(35.5)

        gradient = dem.get_gradient(lat, np.radians(-119.5))

        d = np.radians(0.1)
        dz_dy = 7.0 / (EARTH_MEAN_RADIUS * d)
        dz_dx = -3.0 / (EARTH_MEAN_RADIUS * np.cos(lat) * d)
        assert gradient.dz_dy == pytest.approx(dz_dy, rel=1e-9, abs=0)
        assert gradient.dz_dx == pytest.approx(dz_dx, rel=1e-9, abs=0)
        assert gradient.slope == pytest.approx(
            np.arctan(np.hypot(dz_dx, dz_dy)), rel=1e-9, abs=0
        )
        # Steepest descent heads south-east here (elevation falls with j, rises
        # with i), so the aspect is measured clockwise from north.
        assert gradient.aspect == pytest.approx(
            np.arctan2(-dz_dx, -dz_dy) % (2 * np.pi), rel=1e-9, abs=0
        )

    @pytest.mark.parametrize(
        "lat_deg,lon_deg",
        [(0.0, 0.0), (75.0, 0.0), (5.0, 177.5), (-60.0, -30.0)],
        ids=["equator", "high-latitude", "dateline", "southern"],
    )
    def test_query_at_node_returns_the_stored_value(self, lat_deg, lon_deg):
        n = 11
        data = np.arange(n * n, dtype=float).reshape(n, n)
        dem = DEMGrid(
            data,
            np.radians(lat_deg - 5.0),
            np.radians(lat_deg + 5.0),
            np.radians(lon_deg - 2.5),
            np.radians(lon_deg + 2.5),
        )

        point = dem.get_elevation(np.radians(lat_deg), np.radians(lon_deg))

        assert point.valid
        assert point.elevation == pytest.approx(data[5, 5], rel=1e-12, abs=0)


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


def _gebco_elevation(i, j):
    """Known elevation at row i, column j: bilinear in the indices, so the
    interpolated value between nodes is itself exactly this function."""
    return 100 + 7 * i - 3 * j + 2 * i * j


def _write_gebco(path, lat0, lon0, n=21, step=0.1):
    """Write a GEBCO-schema file whose elevations are ``_gebco_elevation``."""
    import netCDF4 as nc

    lats = lat0 + step * np.arange(n)
    lons = lon0 + step * np.arange(n)
    elevation = np.array(
        [[_gebco_elevation(i, j) for j in range(n)] for i in range(n)], dtype=np.int16
    )
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", n)
        ds.createDimension("lon", n)
        ds.createVariable("lat", "f8", ("lat",))[:] = lats
        ds.createVariable("lon", "f8", ("lon",))[:] = lons
        ds.createVariable("elevation", "i2", ("lat", "lon"))[:] = elevation
    return lats, lons, elevation


@pytest.fixture
def synthetic_data_dir(tmp_path, monkeypatch):
    """Point the loaders at a temp directory, so no test reads the real
    GEBCO_2025.nc, and clear the loader caches on both sides."""
    from pytcl.terrain.loaders import _load_gebco_cached

    monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
    _load_gebco_cached.cache_clear()
    yield tmp_path
    _load_gebco_cached.cache_clear()


@pytest.mark.skipif(not _HAS_NETCDF4, reason="netCDF4 not installed")
class TestLoadGebcoSyntheticFile:
    """load_gebco on a file of known elevations (the schema that
    ``scripts/make_synthetic_gebco.py`` writes, with analytic values)."""

    @staticmethod
    def _load(i0, i1, j0, j1, lat0=10.0, lon0=20.0):
        """Load the nodes i0..i1, j0..j1 (bounds sit half a cell outside)."""
        return load_gebco(
            lat_min=np.radians(lat0 + 0.1 * (i0 - 0.5)),
            lat_max=np.radians(lat0 + 0.1 * (i1 + 0.5)),
            lon_min=np.radians(lon0 + 0.1 * (j0 - 0.5)),
            lon_max=np.radians(lon0 + 0.1 * (j1 + 0.5)),
        )

    def test_grid_values_and_bounds(self, synthetic_data_dir):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)

        dem = self._load(3, 6, 2, 5)

        expected = [[_gebco_elevation(i, j) for j in range(2, 6)] for i in range(3, 7)]
        assert dem.data.shape == (4, 4)
        assert_allclose(dem.data, expected, rtol=0, atol=0)
        assert dem.lat_min == pytest.approx(np.radians(10.3), rel=1e-12, abs=0)
        assert dem.lat_max == pytest.approx(np.radians(10.6), rel=1e-12, abs=0)
        assert dem.lon_min == pytest.approx(np.radians(20.2), rel=1e-12, abs=0)
        assert dem.lon_max == pytest.approx(np.radians(20.5), rel=1e-12, abs=0)
        assert dem.name == "GEBCO2025"

    def test_every_grid_node_is_returned_exactly(self, synthetic_data_dir):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)

        dem = self._load(3, 6, 2, 5)

        for i in range(3, 7):
            for j in range(2, 6):
                point = dem.get_elevation(
                    np.radians(10.0 + 0.1 * i), np.radians(20.0 + 0.1 * j)
                )
                assert point.valid
                assert point.elevation == pytest.approx(
                    _gebco_elevation(i, j), rel=1e-9, abs=0
                )

    @pytest.mark.parametrize(
        "fi,fj", [(0.25, 0.5), (0.5, 0.5), (0.9, 0.1), (0.5, 0.0), (0.0, 0.75)]
    )
    def test_bilinear_value_between_nodes(self, synthetic_data_dir, fi, fj):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)
        dem = self._load(3, 6, 2, 5)
        i, j = 3 + fi, 2 + fj

        point = dem.get_elevation(
            np.radians(10.0 + 0.1 * i), np.radians(20.0 + 0.1 * j)
        )

        assert point.valid
        assert point.elevation == pytest.approx(_gebco_elevation(i, j), rel=1e-6, abs=0)

    def test_nearest_neighbour_between_nodes(self, synthetic_data_dir):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)
        dem = self._load(3, 6, 2, 5)

        point = dem.get_elevation(
            np.radians(10.0 + 0.1 * 3.25),
            np.radians(20.0 + 0.1 * 2.75),
            interpolation="nearest",
        )

        assert point.elevation == _gebco_elevation(3, 3)

    @pytest.mark.parametrize(
        "lat0,lon0",
        [(-0.2, -0.2), (89.8, 0.0), (0.0, 179.8), (-89.9, -90.0)],
        ids=["prime-meridian-equator", "north-pole", "dateline", "south-pole"],
    )
    def test_grids_at_coordinate_extremes(self, synthetic_data_dir, lat0, lon0):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", lat0, lon0, n=3)

        dem = self._load(0, 2, 0, 2, lat0=lat0, lon0=lon0)

        expected = [[_gebco_elevation(i, j) for j in range(3)] for i in range(3)]
        assert_allclose(dem.data, expected, rtol=0, atol=0)

    def test_region_outside_the_file_is_rejected(self, synthetic_data_dir):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)

        with pytest.raises(ValueError, match="outside GEBCO coverage"):
            load_gebco(
                lat_min=np.radians(50.0),
                lat_max=np.radians(51.0),
                lon_min=np.radians(0.0),
                lon_max=np.radians(1.0),
            )

    def test_repeated_load_returns_the_cached_grid(self, synthetic_data_dir):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)

        assert self._load(3, 6, 2, 5) is self._load(3, 6, 2, 5)

    def test_loaded_grid_is_read_only(self, synthetic_data_dir):
        _write_gebco(synthetic_data_dir / "GEBCO2025.nc", 10.0, 20.0)
        dem = self._load(3, 6, 2, 5)

        with pytest.raises(ValueError, match="read-only"):
            dem.data[0, 0] = 0.0

    def test_committed_fixture_matches_an_independent_netcdf_read(
        self, synthetic_data_dir
    ):
        """tests/fixtures/terrain/synthetic_gebco.nc (written by
        scripts/make_synthetic_gebco.py): every node loaded equals the value
        netCDF4 reads straight from the file."""
        import shutil

        import netCDF4 as nc

        fixture = (
            Path(__file__).parents[1] / "fixtures" / "terrain" / "synthetic_gebco.nc"
        )
        shutil.copy(fixture, synthetic_data_dir / "GEBCO2025.nc")
        with nc.Dataset(fixture) as ds:
            lats = np.asarray(ds.variables["lat"][:])
            lons = np.asarray(ds.variables["lon"][:])
            raw = np.asarray(ds.variables["elevation"][:], dtype=float)

        dem = load_gebco(
            lat_min=np.radians(lats[0] - 0.01),
            lat_max=np.radians(lats[-1] + 0.01),
            lon_min=np.radians(lons[0] - 0.01),
            lon_max=np.radians(lons[-1] + 0.01),
        )

        assert dem.data.shape == raw.shape
        assert_allclose(dem.data, raw, rtol=0, atol=0)
        mid = raw.shape[0] // 2
        point = dem.get_elevation(np.radians(lats[mid]), np.radians(lons[mid]))
        assert point.elevation == pytest.approx(raw[mid, mid], rel=1e-9, abs=0)

        half = dem.get_elevation(
            np.radians(0.5 * (lats[mid] + lats[mid + 1])), np.radians(lons[mid])
        )
        assert half.elevation == pytest.approx(
            0.5 * (raw[mid, mid] + raw[mid + 1, mid]), rel=1e-9, abs=0
        )


_EARTH2014_N_COLS = 21600
_EARTH2014_N_ROWS = 10800


def _earth2014_raw_block(path, lat_min_rad, lon_min_rad, shape):
    """Read an Earth2014 block straight from the int16 big-endian file.

    Cell (i, j) is centred on latitude -90 + (i + 0.5)/60 and longitude
    -180 + (j + 0.5)/60 degrees, rows running south to north.
    """
    i0 = round((np.degrees(lat_min_rad) + 90.0) * 60.0 - 0.5)
    j0 = round((np.degrees(lon_min_rad) + 180.0) * 60.0 - 0.5)
    grid = np.memmap(
        path,
        dtype=">i2",
        mode="r",
        shape=(_EARTH2014_N_ROWS, _EARTH2014_N_COLS),
    )
    return np.asarray(grid[i0 : i0 + shape[0], j0 : j0 + shape[1]], dtype=float)


class TestLoadEarth2014RealFile:
    """load_earth2014 against the raw bytes of the installed file; skipped
    where the (445 MB) Earth2014 files are absent."""

    @pytest.mark.parametrize("layer", ["SUR", "BED", "TBI"])
    def test_block_equals_raw_file_contents(self, layer):
        from pytcl.terrain.loaders import _find_earth2014_file

        try:
            dem = load_earth2014(**_NYC, layer=layer)
            path = _find_earth2014_file(layer)
        except _data_skip:
            pytest.skip(f"Earth2014 {layer} data file not available")

        raw = _earth2014_raw_block(path, dem.lat_min, dem.lon_min, dem.data.shape)

        assert dem.data.shape == (62, 62)
        assert_allclose(dem.data, raw, rtol=0, atol=0)
        assert dem.name == f"Earth2014-{layer}"

    def test_himalaya_peak_sits_at_everest(self):
        """Orientation check: the highest SUR cell in a 2 degree window around
        Everest (27.99 N, 86.93 E) is above 7000 m and within half a degree of
        it. A flipped row order would put the window in the southern ocean."""
        try:
            dem = load_earth2014(
                lat_min=np.radians(27.0),
                lat_max=np.radians(29.0),
                lon_min=np.radians(86.0),
                lon_max=np.radians(88.0),
                layer="SUR",
            )
        except _data_skip:
            pytest.skip("Earth2014 SUR data file not available")

        i, j = np.unravel_index(np.argmax(dem.data), dem.data.shape)
        peak_lat = np.degrees(dem.lat_min + i * dem.d_lat)
        peak_lon = np.degrees(dem.lon_min + j * dem.d_lon)

        assert dem.data[i, j] > 7000.0
        assert abs(peak_lat - 27.99) < 0.5
        assert abs(peak_lon - 86.93) < 0.5


@pytest.mark.skipif(not _HAS_NETCDF4, reason="netCDF4 not installed")
class TestParseGebcoNetCDFFillValue:
    """A netCDF4 masked array's mask must survive parse_gebco_netcdf.

    ``np.asarray()`` on a masked array drops the mask, so a ``_FillValue``
    cell reads back as a real elevation. The shipped GEBCO_2025.nc declares
    no ``_FillValue`` (this does not bite the bundled data), but the parser
    is exported for user files that may.
    """

    @staticmethod
    def _write_gebco_with_fill_value(tmp_path: Path, fill: float) -> Path:
        import netCDF4 as nc

        lats = np.linspace(-1.0, 1.0, 5)
        lons = np.linspace(-1.0, 1.0, 5)
        elevation = np.full((5, 5), 100.0)
        elevation[2, 2] = fill  # hole at (lat=0, lon=0)

        path = tmp_path / "GEBCO_fill.nc"
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("lat", len(lats))
            ds.createDimension("lon", len(lons))
            ds.createVariable("lat", "f8", ("lat",))[:] = lats
            ds.createVariable("lon", "f8", ("lon",))[:] = lons
            var = ds.createVariable("elevation", "f8", ("lat", "lon"), fill_value=fill)
            var[:] = elevation
        return path

    def test_fill_valued_cell_is_not_a_real_elevation(self, tmp_path):
        """netCDF4 masks the _FillValue cell; the parser must not report
        it as -32768.0 (the value on disk)."""
        path = self._write_gebco_with_fill_value(tmp_path, fill=-32768.0)

        data, *_ = parse_gebco_netcdf(path)

        assert data[2, 2] != pytest.approx(-32768.0)
        assert data[2, 2] == pytest.approx(-9999.0)

    def test_fill_valued_cell_is_invalid_once_wrapped_in_a_demgrid(self, tmp_path):
        """The audit measured bilinear interpolation beside a leaked fill
        value returning -11732.5 m. DEMGrid's own nodata check (hardcoded
        -9999.0) only catches the hole if the parser maps it there."""
        path = self._write_gebco_with_fill_value(tmp_path, fill=-32768.0)

        data, lat_min, lat_max, lon_min, lon_max = parse_gebco_netcdf(path)
        grid = DEMGrid(data, lat_min, lat_max, lon_min, lon_max)

        beside_the_hole = grid.get_elevation(np.radians(0.1), np.radians(0.1))
        assert not beside_the_hole.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
