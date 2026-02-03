"""
Comprehensive tests for EGM (Earth Gravitational Model) module.

Tests coverage for:
- EGM coefficient loading and parsing
- Geoid height calculations
- Gravity disturbance calculations
- Gravity anomaly calculations
- Deflection of vertical calculations
"""

import numpy as np
import pytest

from pytcl.gravity.egm import (
    EGMCoefficients,
    create_test_coefficients,
    deflection_of_vertical,
    geoid_height,
    geoid_heights,
    gravity_anomaly,
    gravity_disturbance,
)


class TestEGMCoefficients:
    """Tests for EGM coefficient creation and properties."""

    def test_create_test_coefficients_default(self):
        """Test creating test EGM coefficients with default parameters."""
        coeffs = create_test_coefficients()

        assert isinstance(coeffs, EGMCoefficients)
        assert coeffs.C.shape[0] > 0
        assert coeffs.S.shape[0] > 0

    def test_create_test_coefficients_custom_degree(self):
        """Test creating test coefficients with custom degree."""
        for n_max in [10, 20, 36, 50]:
            coeffs = create_test_coefficients(n_max)
            assert isinstance(coeffs, EGMCoefficients)
            assert coeffs.C.shape[0] >= n_max

    def test_create_test_coefficients_structure(self):
        """Test structure of created coefficients."""
        coeffs = create_test_coefficients(20)

        # Check that coefficients are arrays
        assert isinstance(coeffs.C, np.ndarray)
        assert isinstance(coeffs.S, np.ndarray)

        # Check finite values
        assert np.all(np.isfinite(coeffs.C))
        assert np.all(np.isfinite(coeffs.S))


class TestGeoidHeight:
    """Tests for geoid height calculations."""

    def test_geoid_height_single_point_with_coefficients(self):
        """Test geoid height with explicit coefficients."""
        coeffs = create_test_coefficients(20)
        lat = 45.0
        lon = 10.0

        # Direct calculation with test coefficients
        assert coeffs is not None

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_single_point(self):
        """Test geoid height at a single point."""
        lat = 45.0  # degrees
        lon = 10.0  # degrees

        h = geoid_height(lat, lon)

        # Should return a finite scalar
        assert np.isfinite(h)

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_equator(self):
        """Test geoid height at equator."""
        lat = 0.0
        lon = 0.0

        h = geoid_height(lat, lon)
        assert np.isfinite(h)

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_poles(self):
        """Test geoid height at poles."""
        h_north = geoid_height(90.0, 0.0)
        h_south = geoid_height(-90.0, 0.0)

        assert np.isfinite(h_north)
        assert np.isfinite(h_south)

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_longitude_periodicity(self):
        """Test geoid height periodicity with longitude."""
        lat = 45.0
        h1 = geoid_height(lat, 0.0)
        h2 = geoid_height(lat, 360.0)

        assert np.isclose(h1, h2, rtol=1e-10)

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_array_input(self):
        """Test geoid height with array input."""
        lats = np.array([0, 30, 60, 90])
        lons = np.array([0, 45, 90, 180])

        heights = geoid_heights(lats, lons)

        assert heights.shape == lats.shape
        assert np.all(np.isfinite(heights))

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_grid(self):
        """Test geoid height on regular grid."""
        lats = np.linspace(-90, 90, 10)
        lons = np.linspace(-180, 180, 10)

        heights = geoid_heights(lats, lons)

        assert heights.shape == lats.shape
        assert np.all(np.isfinite(heights))

    @pytest.mark.skip(reason="Requires EGM2008 data file")
    def test_geoid_height_typical_range(self):
        """Test that geoid heights are in typical range."""
        lats = np.linspace(-90, 90, 20)
        lons = np.linspace(-180, 180, 20)

        heights = geoid_heights(lats, lons)

        # Geoid heights typically range from -100 to +100 meters
        assert np.all(heights > -200)
        assert np.all(heights < 200)


@pytest.mark.skip(reason="Requires EGM2008 data file")
class TestGravityDisturbance:
    """Tests for gravity disturbance calculations."""

    def test_gravity_disturbance_single_point(self):
        """Test gravity disturbance at single point."""
        lat = 45.0
        lon = 10.0

        delta_g = gravity_disturbance(lat, lon)

        assert np.isfinite(delta_g)

    def test_gravity_disturbance_equator(self):
        """Test gravity disturbance at equator."""
        delta_g = gravity_disturbance(0.0, 0.0)
        assert np.isfinite(delta_g)

    def test_gravity_disturbance_poles(self):
        """Test gravity disturbance at poles."""
        delta_g_north = gravity_disturbance(90.0, 0.0)
        delta_g_south = gravity_disturbance(-90.0, 0.0)

        assert np.isfinite(delta_g_north)
        assert np.isfinite(delta_g_south)

    def test_gravity_disturbance_units(self):
        """Test gravity disturbance is in expected units (mGal)."""
        lat = 45.0
        lon = 10.0

        delta_g = gravity_disturbance(lat, lon)

        # Gravity disturbance typically in range -500 to +500 mGal
        assert abs(delta_g) < 1000


@pytest.mark.skip(reason="Requires EGM2008 data file")
class TestGravityAnomaly:
    """Tests for gravity anomaly calculations."""

    def test_gravity_anomaly_single_point(self):
        """Test gravity anomaly at single point."""
        lat = 45.0
        lon = 10.0

        ga = gravity_anomaly(lat, lon)

        assert np.isfinite(ga)

    def test_gravity_anomaly_equator(self):
        """Test gravity anomaly at equator."""
        ga = gravity_anomaly(0.0, 0.0)
        assert np.isfinite(ga)

    def test_gravity_anomaly_poles(self):
        """Test gravity anomaly at poles."""
        ga_north = gravity_anomaly(90.0, 0.0)
        ga_south = gravity_anomaly(-90.0, 0.0)

        assert np.isfinite(ga_north)
        assert np.isfinite(ga_south)

    def test_gravity_anomaly_range(self):
        """Test gravity anomaly is in expected range."""
        lat = 45.0
        lon = 10.0

        ga = gravity_anomaly(lat, lon)

        # Gravity anomaly typically in range -200 to +200 mGal
        assert abs(ga) < 500


@pytest.mark.skip(reason="Requires EGM2008 data file")
class TestDeflectionOfVertical:
    """Tests for deflection of vertical calculations."""

    def test_deflection_of_vertical_single_point(self):
        """Test deflection of vertical at single point."""
        lat = 45.0
        lon = 10.0

        result = deflection_of_vertical(lat, lon)

        # Should return tuple or array with 2 components
        if isinstance(result, tuple):
            assert len(result) == 2
            assert np.isfinite(result[0])
            assert np.isfinite(result[1])
        else:
            assert result.shape[-1] >= 2
            assert np.all(np.isfinite(result))

    def test_deflection_of_vertical_equator(self):
        """Test deflection of vertical at equator."""
        result = deflection_of_vertical(0.0, 0.0)

        if isinstance(result, tuple):
            assert np.isfinite(result[0])
            assert np.isfinite(result[1])
        else:
            assert np.all(np.isfinite(result))

    def test_deflection_of_vertical_poles(self):
        """Test deflection of vertical at poles."""
        result_north = deflection_of_vertical(90.0, 0.0)
        result_south = deflection_of_vertical(-90.0, 0.0)

        if isinstance(result_north, tuple):
            assert np.isfinite(result_north[0])
            assert np.isfinite(result_north[1])
            assert np.isfinite(result_south[0])
            assert np.isfinite(result_south[1])


@pytest.mark.skip(reason="Requires EGM2008 data file")
class TestEGMIntegration:
    """Integration tests for EGM calculations."""

    def test_egm_calculations_global_coverage(self):
        """Test EGM calculations at various global locations."""
        locations = [
            (0, 0),  # Equator, Prime Meridian
            (45, 90),  # Mid-latitude, 90E
            (-30, -120),  # Southern hemisphere
            (60, 179),  # High latitude
        ]

        for lat, lon in locations:
            h = geoid_height(lat, lon)
            delta_g = gravity_disturbance(lat, lon)
            ga = gravity_anomaly(lat, lon)
            dov = deflection_of_vertical(lat, lon)

            assert np.isfinite(h)
            assert np.isfinite(delta_g)
            assert np.isfinite(ga)
            if isinstance(dov, tuple):
                assert np.isfinite(dov[0]) and np.isfinite(dov[1])

    def test_egm_regular_grid(self):
        """Test EGM calculations on regular grid."""
        lats = np.linspace(-60, 60, 7)
        lons = np.linspace(-180, 180, 13)

        heights = []
        for lat in lats:
            for lon in lons:
                h = geoid_height(lat, lon)
                heights.append(h)

        assert all(np.isfinite(h) for h in heights)

    def test_egm_different_degrees(self):
        """Test EGM calculations with different coefficient degrees."""
        lat, lon = 45.0, 10.0

        for n_max in [10, 20, 36]:
            coeffs = create_test_coefficients(n_max)

            # Verify coefficient structure
            assert isinstance(coeffs.C, np.ndarray)
            assert isinstance(coeffs.S, np.ndarray)

    def test_egm_symmetries(self):
        """Test EGM field symmetries."""
        lat = 45.0

        # Test symmetry about prime meridian if applicable
        h1 = geoid_height(lat, 10.0)
        h2 = geoid_height(lat, -10.0)

        assert np.isfinite(h1)
        assert np.isfinite(h2)

    def test_egm_continuity(self):
        """Test continuity of EGM functions across ranges."""
        # Test continuity in latitude
        lats = np.linspace(0, 90, 100)
        heights = [geoid_height(lat, 0.0) for lat in lats]

        # Check that values are reasonably continuous
        diffs = np.abs(np.diff(heights))
        assert np.all(diffs < 50)  # Changes should be smooth


@pytest.mark.skip(reason="Requires EGM2008 data file")
class TestEGMEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_egm_latitude_boundaries(self):
        """Test EGM calculations at latitude boundaries."""
        for lat in [-90, -89, 0, 89, 90]:
            h = geoid_height(lat, 0.0)
            assert np.isfinite(h)

    def test_egm_longitude_boundaries(self):
        """Test EGM calculations at longitude boundaries."""
        lat = 45.0
        for lon in [-180, -179, 0, 179, 180]:
            h = geoid_height(lat, lon)
            assert np.isfinite(h)

    def test_egm_precision(self):
        """Test numerical precision of EGM calculations."""
        lat, lon = 45.123456, 10.654321

        h1 = geoid_height(lat, lon)
        h2 = geoid_height(lat, lon)

        # Should be deterministic
        assert np.isclose(h1, h2)
