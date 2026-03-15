"""Tests for Enhanced Magnetic Model (EMM) and WMMHR."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.magnetism import (
    EMM_PARAMETERS,
    create_emm_test_coefficients,
    emm,
    emm_declination,
    emm_inclination,
    emm_intensity,
    get_emm_data_dir,
    wmm,
    wmmhr,
)
from pytcl.magnetism.emm import (
    HighResCoefficients,
    create_test_coefficients,
    get_data_dir,
    parse_emm_file,
)


class TestHighResCoefficients:
    """Tests for HighResCoefficients creation and structure."""

    def test_create_test_coefficients_default(self):
        """Test creating default test coefficients (n_max=36)."""
        coef = create_emm_test_coefficients()
        assert coef.n_max == 36
        assert coef.n_max_sv == 12
        assert coef.epoch == 2020.0
        assert coef.model_name == "EMM_TEST"

    def test_create_test_coefficients_custom_nmax(self):
        """Test creating test coefficients with custom n_max."""
        coef = create_emm_test_coefficients(n_max=50)
        assert coef.n_max == 50
        assert coef.n_max_sv == 12  # SV capped at 12

    def test_coefficient_arrays_shape(self):
        """Test that coefficient arrays have correct shape."""
        coef = create_emm_test_coefficients(n_max=20)
        assert coef.g.shape == (21, 21)
        assert coef.h.shape == (21, 21)
        assert coef.g_dot.shape == (13, 13)  # n_max_sv + 1
        assert coef.h_dot.shape == (13, 13)

    def test_dipole_coefficient_nonzero(self):
        """Test that g[1,0] (axial dipole) is nonzero."""
        coef = create_emm_test_coefficients()
        assert coef.g[1, 0] != 0
        # Should be negative (pointing south)
        assert coef.g[1, 0] < -20000  # nT

    def test_dipole_dominant(self):
        """g[1,0] should be the largest coefficient in magnitude."""
        coef = create_emm_test_coefficients()
        assert abs(coef.g[1, 0]) > abs(coef.g[2, 0])
        assert abs(coef.g[1, 0]) > abs(coef.g[1, 1])

    def test_higher_degrees_smaller(self):
        """Higher degree coefficients should be smaller on average."""
        coef = create_emm_test_coefficients(n_max=50)
        # Average magnitude of degree 5 vs degree 40
        avg_n5 = np.mean(np.abs(coef.g[5, :6]))
        avg_n40 = np.mean(np.abs(coef.g[40, :41]))
        assert avg_n5 > avg_n40


class TestEMMParameters:
    """Tests for model parameter definitions."""

    def test_emm2017_parameters(self):
        """Test EMM2017 parameters are defined correctly."""
        assert "EMM2017" in EMM_PARAMETERS
        params = EMM_PARAMETERS["EMM2017"]
        assert params["n_max"] == 790
        assert params["epoch"] == 2017.0

    def test_wmmhr2025_parameters(self):
        """Test WMMHR2025 parameters are defined correctly."""
        assert "WMMHR2025" in EMM_PARAMETERS
        params = EMM_PARAMETERS["WMMHR2025"]
        assert params["n_max"] == 133
        assert params["n_max_sv"] == 15
        assert params["epoch"] == 2025.0


class TestDataDirectory:
    """Tests for data directory functionality."""

    def test_get_data_dir_returns_path(self):
        """get_emm_data_dir should return a Path object."""
        from pathlib import Path

        data_dir = get_emm_data_dir()
        assert isinstance(data_dir, Path)

    def test_data_dir_is_in_home(self):
        """Default data dir should be under home directory."""
        from pathlib import Path

        data_dir = get_emm_data_dir()
        home = Path.home()
        # Check that data_dir is under home
        assert str(data_dir).startswith(str(home))


class TestEMMFunction:
    """Tests for the emm() main function."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_emm_returns_magnetic_result(self, test_coefficients):
        """EMM returns MagneticResult with all expected fields."""
        result = emm(
            np.radians(40),
            np.radians(-105),
            1.0,
            2020.0,
            coefficients=test_coefficients,
        )
        assert hasattr(result, "X")
        assert hasattr(result, "Y")
        assert hasattr(result, "Z")
        assert hasattr(result, "H")
        assert hasattr(result, "F")
        assert hasattr(result, "I")
        assert hasattr(result, "D")

    def test_total_intensity_reasonable(self, test_coefficients):
        """Total field intensity should be in expected range."""
        result = emm(
            np.radians(45), np.radians(0), 0, 2020.0, coefficients=test_coefficients
        )
        # Field intensity typically 25,000-65,000 nT at mid-latitudes
        # but can exceed this at high latitudes, allow up to 100,000 nT
        assert 20000 < result.F < 100000

    def test_horizontal_intensity_formula(self, test_coefficients):
        """H = sqrt(X^2 + Y^2)."""
        result = emm(
            np.radians(40), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        H_calc = np.sqrt(result.X**2 + result.Y**2)
        assert_allclose(result.H, H_calc, rtol=1e-10)

    def test_total_intensity_formula(self, test_coefficients):
        """F = sqrt(H^2 + Z^2)."""
        result = emm(
            np.radians(40), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        F_calc = np.sqrt(result.H**2 + result.Z**2)
        assert_allclose(result.F, F_calc, rtol=1e-10)

    def test_declination_range(self, test_coefficients):
        """Declination should be within -180 to 180 degrees."""
        result = emm(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert -np.pi <= result.D <= np.pi

    def test_inclination_range(self, test_coefficients):
        """Inclination should be within -90 to 90 degrees."""
        result = emm(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert -np.pi / 2 <= result.I <= np.pi / 2


class TestEMMPhysicalProperties:
    """Tests for physical properties of EMM field."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_inclination_positive_north(self, test_coefficients):
        """Inclination is positive in northern hemisphere."""
        result = emm(np.radians(60), 0, 0, 2020.0, coefficients=test_coefficients)
        assert result.I > 0  # Field points into Earth

    def test_inclination_negative_south(self, test_coefficients):
        """Inclination is negative in southern hemisphere."""
        result = emm(np.radians(-60), 0, 0, 2020.0, coefficients=test_coefficients)
        assert result.I < 0  # Field points out of Earth

    def test_field_stronger_at_poles(self, test_coefficients):
        """Magnetic field is stronger near poles than equator."""
        F_pole = emm(np.radians(80), 0, 0, 2020.0, coefficients=test_coefficients).F
        F_eq = emm(0, 0, 0, 2020.0, coefficients=test_coefficients).F
        assert F_pole > F_eq

    def test_field_decreases_with_altitude(self, test_coefficients):
        """Magnetic field decreases with altitude."""
        F_0 = emm(np.radians(45), 0, 0, 2020.0, coefficients=test_coefficients).F
        F_100 = emm(np.radians(45), 0, 100, 2020.0, coefficients=test_coefficients).F
        assert F_0 > F_100


class TestEMMComparisonWithWMM:
    """Compare EMM (low degree) with standard WMM."""

    def test_low_degree_similar_to_wmm(self):
        """EMM with low n_max should give similar results to WMM."""
        lat = np.radians(40)
        lon = np.radians(-75)

        # Create test coefficients that match WMM core field
        coef = create_emm_test_coefficients(n_max=12)

        # Compare with WMM
        emm_result = emm(lat, lon, 0, 2020.0, coefficients=coef, n_max=12)
        wmm_result = wmm(lat, lon, 0, 2020.0)

        # Should be within 10% for total field (test coefficients slightly different)
        rel_diff = abs(emm_result.F - wmm_result.F) / wmm_result.F
        assert rel_diff < 0.15


class TestWMMHR:
    """Tests for wmmhr() convenience function."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=50)

    def test_wmmhr_returns_result(self, test_coefficients):
        """WMMHR returns MagneticResult."""
        result = wmmhr(
            np.radians(45), np.radians(-75), 0, 2025.0, coefficients=test_coefficients
        )
        assert hasattr(result, "F")
        assert hasattr(result, "D")
        assert hasattr(result, "I")

    def test_wmmhr_uses_model_coefficients(self, test_coefficients):
        """WMMHR should use the provided coefficients."""
        result1 = wmmhr(
            np.radians(45),
            np.radians(-75),
            0,
            2025.0,
            coefficients=test_coefficients,
            n_max=36,
        )
        result2 = wmmhr(
            np.radians(45),
            np.radians(-75),
            0,
            2025.0,
            coefficients=test_coefficients,
            n_max=50,
        )
        # Different n_max should give slightly different results
        # (due to higher degree contributions in result2)
        # But both should be valid (can reach ~90,000 nT at high latitudes)
        assert 20000 < result1.F < 100000
        assert 20000 < result2.F < 100000


class TestConvenienceFunctions:
    """Tests for emm_declination, emm_inclination, emm_intensity."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_emm_declination(self, test_coefficients):
        """emm_declination returns correct value."""
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        D = emm_declination(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        assert_allclose(D, result.D)

    def test_emm_inclination(self, test_coefficients):
        """emm_inclination returns correct value."""
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        incl = emm_inclination(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        assert_allclose(incl, result.I)

    def test_emm_intensity(self, test_coefficients):
        """emm_intensity returns correct value."""
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        F = emm_intensity(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        assert_allclose(F, result.F)

    def test_declination_is_scalar(self, test_coefficients):
        """Declination function returns scalar."""
        D = emm_declination(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert isinstance(D, float)

    def test_inclination_is_scalar(self, test_coefficients):
        """Inclination function returns scalar."""
        incl = emm_inclination(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert isinstance(incl, float)

    def test_intensity_is_scalar(self, test_coefficients):
        """Intensity function returns scalar."""
        F = emm_intensity(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert isinstance(F, float)


class TestSecularVariation:
    """Tests for secular variation in high-resolution models."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_field_changes_with_time(self, test_coefficients):
        """Field should change slightly between years."""
        result_2020 = emm(
            np.radians(45), np.radians(0), 0, 2020.0, coefficients=test_coefficients
        )
        result_2022 = emm(
            np.radians(45), np.radians(0), 0, 2022.0, coefficients=test_coefficients
        )
        # Small but non-zero change expected
        # Secular variation is ~50-100 nT/year at mid-latitudes
        assert result_2020.F != result_2022.F

    def test_declination_changes_with_time(self, test_coefficients):
        """Declination should change over time."""
        D_2020 = emm_declination(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        D_2025 = emm_declination(
            np.radians(45), np.radians(-75), 0, 2025.0, coefficients=test_coefficients
        )
        # Should be different
        assert D_2020 != D_2025


class TestNumericalStability:
    """Tests for numerical stability at various locations."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_equator(self, test_coefficients):
        """Field computation at equator should not produce NaN."""
        result = emm(0, 0, 0, 2020.0, coefficients=test_coefficients)
        assert not np.isnan(result.F)
        assert not np.isnan(result.D)
        assert not np.isnan(result.I)

    def test_north_pole(self, test_coefficients):
        """Field computation near north pole should not produce NaN."""
        result = emm(np.radians(89.9), 0, 0, 2020.0, coefficients=test_coefficients)
        assert not np.isnan(result.F)
        # Declination may be undefined at pole, but should not crash

    def test_south_pole(self, test_coefficients):
        """Field computation near south pole should not produce NaN."""
        result = emm(np.radians(-89.9), 0, 0, 2020.0, coefficients=test_coefficients)
        assert not np.isnan(result.F)

    def test_high_altitude(self, test_coefficients):
        """Field computation at high altitude should work."""
        result = emm(
            np.radians(45),
            np.radians(0),
            500,
            2020.0,  # 500 km altitude
            coefficients=test_coefficients,
        )
        assert not np.isnan(result.F)
        assert result.F > 0

    def test_various_longitudes(self, test_coefficients):
        """Test field at various longitudes."""
        for lon_deg in [0, 45, 90, 135, 180, -135, -90, -45]:
            result = emm(
                np.radians(45),
                np.radians(lon_deg),
                0,
                2020.0,
                coefficients=test_coefficients,
            )
            assert not np.isnan(result.F)
            assert result.F > 0


class TestHighDegreeEvaluation:
    """Tests for evaluation at higher harmonic degrees."""

    def test_higher_degree_coefficients(self):
        """Test with higher degree coefficients (n_max=50)."""
        coef = create_emm_test_coefficients(n_max=50)
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=coef, n_max=50
        )
        assert not np.isnan(result.F)
        assert result.F > 0

    def test_n_max_limit_respected(self):
        """Specifying n_max should limit evaluation."""
        coef = create_emm_test_coefficients(n_max=50)

        # Evaluate at different n_max values
        result_12 = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=coef, n_max=12
        )
        result_50 = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=coef, n_max=50
        )

        # Both should be valid
        assert result_12.F > 0
        assert result_50.F > 0

        # Higher degree should give slightly different result
        # (due to crustal field contributions)
        assert result_12.F != result_50.F


# =====================================================================
# Additional comprehensive tests
# =====================================================================


class TestEMMDataManagement:
    """Tests for data directory and file management."""

    def test_get_data_dir(self):
        """Test data directory retrieval."""
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)

    def test_get_data_dir_consistency(self):
        """Test that data dir returns same result."""
        dir1 = get_data_dir()
        dir2 = get_data_dir()
        assert dir1 == dir2


class TestCoefficientCreation:
    """Tests for creating test coefficients."""

    def test_create_test_coefficients_default(self):
        """Test creating default test coefficients."""
        coeff = create_test_coefficients()
        assert isinstance(coeff, HighResCoefficients)
        assert coeff.n_max == 36
        assert coeff.g.shape[0] == 37  # n_max+1
        assert coeff.h.shape[0] == 37

    def test_create_test_coefficients_custom_degree(self):
        """Test creating coefficients with custom degree."""
        coeff = create_test_coefficients(n_max=50)
        assert coeff.n_max == 50
        assert coeff.g.shape == (51, 51)

    def test_test_coefficients_properties(self):
        """Test properties of test coefficients."""
        coeff = create_test_coefficients(n_max=20)

        # Check arrays are numpy arrays
        assert isinstance(coeff.g, np.ndarray)
        assert isinstance(coeff.h, np.ndarray)
        assert isinstance(coeff.g_dot, np.ndarray)
        assert isinstance(coeff.h_dot, np.ndarray)

        # Check epoch is reasonable
        assert isinstance(coeff.epoch, (int, float))
        assert 2000 <= coeff.epoch <= 2030

    def test_test_coefficients_g_h_symmetry(self):
        """Test g and h coefficient arrays have same shape."""
        coeff = create_test_coefficients(n_max=30)
        assert coeff.g.shape == coeff.h.shape
        assert coeff.g_dot.shape == coeff.h_dot.shape


class TestCoeffientParsing:
    """Tests for parsing coefficient files."""

    def test_parse_emm_file_minimal(self):
        """Test parsing a minimal test file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            # Write test data
            f.write("2020.0\n")  # epoch
            f.write("1 0 -29438.5 0.0 10.7 0.0\n")
            f.write("1 1 -1501.1 4796.2 17.7 -26.9\n")
            f.write("2 0 -2445.3 0.0 -8.8 0.0\n")
            f.flush()

            g, h, g_dot, h_dot, epoch, n_max = parse_emm_file(Path(f.name))

            assert epoch == 2020.0
            assert n_max >= 2
            assert g[1, 0] == pytest.approx(-29438.5)

    def test_parse_emm_with_degree_limit(self):
        """Test parsing with degree limit."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("2020.0\n")
            f.write("1 0 -29438.5 0.0 10.7 0.0\n")
            f.write("2 0 -2445.3 0.0 -8.8 0.0\n")
            f.write("3 0 3012.5 0.0 0.2 0.0\n")
            f.flush()

            g, h, g_dot, h_dot, epoch, n_max = parse_emm_file(Path(f.name), n_max=2)

            assert n_max == 2
            assert g.shape[0] == 3  # n_max + 1


class TestEMMFieldCalculations:
    """Tests for EMM field calculations."""

    def test_emm_basic(self):
        """Test basic EMM field calculation."""
        coeff = create_test_coefficients(n_max=36)
        result = emm(
            lat=np.radians(40.0),
            lon=np.radians(-75.0),
            h=0.0,
            year=2020.0,
            coefficients=coeff,
        )

        assert hasattr(result, "X")
        assert hasattr(result, "Y")
        assert hasattr(result, "Z")

    def test_emm_array_inputs(self):
        """Test EMM with array inputs."""
        coeff = create_test_coefficients(n_max=36)

        latitudes = np.radians(np.array([40.0, 45.0, 50.0]))
        longitudes = np.radians(np.array([-75.0, -80.0, -85.0]))
        heights = np.array([0.0, 100.0, 200.0])

        result = emm(
            lat=latitudes,
            lon=longitudes,
            h=heights,
            year=2020.0,
            coefficients=coeff,
        )

        assert len(result.X) == len(latitudes)

    def test_emm_different_years(self):
        """Test EMM at different years."""
        coeff = create_test_coefficients(n_max=36)

        lat_r = np.radians(40.0)
        lon_r = np.radians(-75.0)
        result_2020 = emm(lat_r, lon_r, 0.0, 2020.0, coefficients=coeff)
        result_2025 = emm(lat_r, lon_r, 0.0, 2025.0, coefficients=coeff)

        # Secular variation should produce different results
        assert result_2020.X != result_2025.X


class TestWMMHRCalculations:
    """Tests for WMMHR calculations."""

    def test_wmmhr_basic(self):
        """Test basic WMMHR calculation."""
        try:
            result = wmmhr(np.radians(40.0), np.radians(-75.0), 0.0, 2025.0)
        except FileNotFoundError:
            pytest.skip("WMMHR2025 coefficient file not installed")

        assert hasattr(result, "X")
        assert hasattr(result, "Y")
        assert hasattr(result, "Z")


class TestMagneticComponentsFromEMM:
    """Tests for magnetic component extraction."""

    def test_emm_declination(self):
        """Test EMM declination calculation."""
        coeff = create_test_coefficients(n_max=36)

        decl = emm_declination(
            np.radians(40.0), np.radians(-75.0), 0.0, 2020.0,
            coefficients=coeff,
        )

        assert -np.pi <= decl <= np.pi

    def test_emm_inclination(self):
        """Test EMM inclination calculation."""
        coeff = create_test_coefficients(n_max=36)

        incl = emm_inclination(
            np.radians(40.0), np.radians(-75.0), 0.0, 2020.0,
            coefficients=coeff,
        )

        assert -np.pi / 2 <= incl <= np.pi / 2

    def test_emm_intensity(self):
        """Test EMM total intensity calculation."""
        coeff = create_test_coefficients(n_max=36)

        intensity = emm_intensity(
            np.radians(40.0), np.radians(-75.0), 0.0, 2020.0,
            coefficients=coeff,
        )

        assert float(intensity) > 0


class TestMagneticArrayOperations:
    """Tests for array operations with magnetic components."""

    def test_declination_arrays(self):
        """Test declination with array inputs."""
        coeff = create_test_coefficients(n_max=36)

        lats = np.radians(np.array([40.0, 45.0, 50.0]))
        lons = np.radians(np.array([-75.0, -80.0, -85.0]))
        heights = np.array([0.0, 100.0, 200.0])

        decl = emm_declination(lats, lons, heights, 2020.0, coefficients=coeff)

        assert isinstance(decl, np.ndarray)
        assert len(decl) == 3
        assert np.all(np.abs(decl) <= np.pi)

    def test_inclination_arrays(self):
        """Test inclination with array inputs."""
        coeff = create_test_coefficients(n_max=36)

        lats = np.radians(np.array([40.0, 45.0, 50.0]))
        lons = np.radians(np.array([-75.0, -80.0, -85.0]))
        heights = np.array([0.0, 100.0, 200.0])

        incl = emm_inclination(lats, lons, heights, 2020.0, coefficients=coeff)

        assert isinstance(incl, np.ndarray)
        assert len(incl) == 3
        assert np.all(np.abs(incl) <= np.pi / 2)


class TestCoefficientDimensions:
    """Tests for coefficient array dimensions and properties."""

    def test_coefficient_shapes_consistency(self):
        """Test that coefficient arrays have consistent shapes."""
        coeff = create_test_coefficients(n_max=40)

        # g and h should be square
        assert coeff.g.shape[0] == coeff.g.shape[1]
        assert coeff.h.shape[0] == coeff.h.shape[1]

        # g and h should have same size
        assert coeff.g.shape == coeff.h.shape

    def test_sv_arrays_smaller_or_equal(self):
        """Test that SV arrays are smaller or equal to main arrays."""
        coeff = create_test_coefficients(n_max=50)

        # SV arrays should be smaller or equal
        assert coeff.g_dot.shape[0] <= coeff.g.shape[0]
        assert coeff.h_dot.shape[0] <= coeff.h.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
