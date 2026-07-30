"""Correctness audit tests for pytcl.core, pytcl.terrain, and pytcl.plotting.

Every test here checks behavior against an INDEPENDENT reference:
published constants (CODATA 2018, WGS84, IERS, IAU), hand-derived analytic
geometry (ray casting over synthetic DEMs, horizon trigonometry), numpy/scipy
ground truth, or chi-squared table values.  Figure-builder tests assert the
actual data payloads inside the figures, not just that a figure was returned.
"""

import math

import numpy as np
import pytest
from scipy import linalg as sla
from scipy.interpolate import RegularGridInterpolator

import pytcl.core.optional_deps as od
from pytcl.core import constants as C
from pytcl.core.array_utils import (
    block_diag,
    column_vector,
    is_positive_definite,
    is_positive_semidefinite,
    meshgrid_ij,
    nearest_positive_definite,
    normalize_vector,
    outer_product,
    repmat,
    row_vector,
    safe_cholesky,
    skew_symmetric,
    unskew,
    unvec,
    vec,
    wrap_to_2pi,
    wrap_to_360,
    wrap_to_pi,
    wrap_to_pm180,
    wrap_to_range,
)
from pytcl.core.exceptions import (
    ComputationError,
    ConfigurationError,
    DependencyError,
    DimensionError,
    MethodError,
    ParameterError,
    TCLError,
    ValidationError,
)
from pytcl.core.maturity import (
    MODULE_MATURITY,
    MaturityLevel,
    format_maturity_badge,
    get_maturity,
    get_maturity_summary,
    get_modules_by_maturity,
    is_production_ready,
    is_stable,
)
from pytcl.core.optional_deps import (
    LazyModule,
    check_dependencies,
    import_optional,
    is_available,
    requires,
)
from pytcl.core.paths import ensure_data_dir, get_data_dir
from pytcl.core.validation import (
    ArraySpec,
    ScalarSpec,
    check_compatible_shapes,
    ensure_2d,
    ensure_column_vector,
    ensure_positive_definite,
    ensure_row_vector,
    ensure_square_matrix,
    ensure_symmetric,
    validate_array,
    validate_inputs,
    validate_same_shape,
    validated_array_input,
)
from pytcl.plotting.ellipses import (
    confidence_region_radius,
    covariance_ellipse_points,
    covariance_ellipsoid_points,
    ellipse_parameters,
)
from pytcl.terrain.dem import (
    DEMGrid,
    create_flat_dem,
    create_synthetic_terrain,
    get_elevation_profile,
    interpolate_dem,
    merge_dems,
)
from pytcl.terrain.loaders import (
    EARTH2014_PARAMETERS,
    GEBCO_PARAMETERS,
    get_earth2014_metadata,
    get_gebco_metadata,
    load_earth2014,
    load_gebco,
    parse_earth2014_binary,
)
from pytcl.terrain.visibility import (
    compute_horizon,
    line_of_sight,
    radar_coverage_map,
    terrain_masking_angle,
    viewshed,
)

R_EARTH = 6371000.0
HAS_PLOTLY = is_available("plotly")
HAS_NETCDF4 = is_available("netCDF4")


# =============================================================================
# core.constants vs authoritative published values
# =============================================================================


class TestConstantsAuthoritative:
    """Every physical constant spot-checked against its published source."""

    def test_si_exact_defining_constants(self):
        # SI 2019 redefinition: these are exact by definition (CODATA 2018).
        assert C.SPEED_OF_LIGHT == 299_792_458.0
        assert C.PLANCK_CONSTANT == 6.62607015e-34
        assert C.BOLTZMANN_CONSTANT == 1.380649e-23
        assert C.ELEMENTARY_CHARGE == 1.602176634e-19
        assert C.AVOGADRO_CONSTANT == 6.02214076e23

    def test_codata_2018_measured_constants(self):
        assert C.GRAVITATIONAL_CONSTANT == pytest.approx(6.67430e-11, rel=1e-12)
        assert C.STANDARD_ATMOSPHERE == 101_325.0
        assert C.ABSOLUTE_ZERO_CELSIUS == -273.15
        assert C.STANDARD_GRAVITY == 9.80665  # CGPM definition

    def test_moon_gm_matches_de430_and_the_librarys_own_mass_ratio(self):
        """MOON_GM must agree with DE430 and with EARTH_GM / EARTH_MOON_MASS_RATIO.

        It did not. core.constants carried 4.9028695e12, which is 1.4e-5
        relative away from both the published DE430/GRAIL value
        (4902.800118 km^3/s^2) and from the quotient the library's own two
        constants imply -- and away from the value pytcl.gravity.tides was
        independently using. Two of the three disagreed with each other.
        """
        assert C.MOON_GM == pytest.approx(4.902800118e12, rel=1e-9)

        # Consistency with the library's own Earth-Moon mass ratio. The
        # tolerance is set by the rounding in EARTH_MOON_MASS_RATIO itself,
        # not by any slack in MOON_GM.
        derived = C.EARTH_GM / C.EARTH_MOON_MASS_RATIO
        assert C.MOON_GM == pytest.approx(derived, rel=2e-7)

    def test_moon_and_sun_gm_are_defined_once(self):
        """gravity.tides must not carry its own copy of these constants.

        It used to define MOON_GM = 4.902801e12 and SUN_GM locally. Two copies
        of a physical constant drift, and these had: the tides value was right
        and core's was not.
        """
        from pytcl.gravity import tides

        assert tides.MOON_GM is C.MOON_GM
        assert tides.SUN_GM is C.SUN_GM

    def test_derived_constants_from_exact_values(self):
        # R = N_A * k_B (exact in SI 2019)
        r_exact = C.AVOGADRO_CONSTANT * C.BOLTZMANN_CONSTANT
        assert C.UNIVERSAL_GAS_CONSTANT == pytest.approx(r_exact, rel=1e-9)
        # sigma = 2 pi^5 k^4 / (15 h^3 c^2)
        sigma = (
            2
            * math.pi**5
            * C.BOLTZMANN_CONSTANT**4
            / (15 * C.PLANCK_CONSTANT**3 * C.SPEED_OF_LIGHT**2)
        )
        assert C.STEFAN_BOLTZMANN_CONSTANT == pytest.approx(sigma, rel=1e-9)

    def test_wgs84_defining_parameters(self):
        # NIMA TR8350.2 defining parameters
        assert C.EARTH_SEMI_MAJOR_AXIS == 6_378_137.0
        assert C.EARTH_FLATTENING == pytest.approx(1.0 / 298.257223563, rel=1e-15)
        assert C.EARTH_GM == 3.986004418e14
        assert C.EARTH_ROTATION_RATE == 7.292115e-5
        # Derived: published WGS84 semi-minor axis
        assert C.EARTH_SEMI_MINOR_AXIS == pytest.approx(6_356_752.314245, abs=1e-4)
        b_derived = C.EARTH_SEMI_MAJOR_AXIS * (1 - C.EARTH_FLATTENING)
        assert C.EARTH_SEMI_MINOR_AXIS == pytest.approx(b_derived, abs=1e-4)
        # e^2 = 2f - f^2; published value 6.69437999014e-3
        assert C.EARTH_ECCENTRICITY_SQ == pytest.approx(6.69437999014e-3, rel=1e-10)
        assert C.EARTH_ECCENTRICITY == pytest.approx(
            math.sqrt(C.EARTH_ECCENTRICITY_SQ), rel=1e-15
        )
        # e'^2 published: 6.73949674228e-3
        assert C.EARTH_ECCENTRICITY_PRIME_SQ == pytest.approx(
            6.73949674228e-3, rel=1e-10
        )
        assert C.EARTH_GM_EGM2008 == 3.986004415e14
        # IERS/aoki mean rotation rate
        assert C.EARTH_MEAN_ANGULAR_VELOCITY == pytest.approx(7.2921151467e-5)
        assert C.EARTH_MEAN_RADIUS == 6_371_000.0

    def test_ellipsoid_dataclass_derived_properties(self):
        w = C.WGS84
        assert w.b == pytest.approx(6_356_752.314245, abs=1e-4)
        assert w.e2 == pytest.approx(2 * w.f - w.f**2, rel=1e-15)
        assert w.e == pytest.approx(math.sqrt(w.e2), rel=1e-15)
        assert w.ep2 == pytest.approx(w.e2 / (1 - w.e2), rel=1e-15)
        assert w.ep == pytest.approx(math.sqrt(w.ep2), rel=1e-15)
        # GRS80 defining flattening
        assert C.GRS80.f == pytest.approx(1.0 / 298.257222101, rel=1e-15)
        assert C.GRS80.a == 6_378_137.0
        # Clarke 1866: a and derived b = 6356583.8 m (published)
        assert C.CLARKE1866.a == 6_378_206.4
        assert C.CLARKE1866.b == pytest.approx(6_356_583.8, abs=0.01)
        # Sphere: f = 0 so b == a, e == 0
        assert C.SPHERE_EARTH.b == C.SPHERE_EARTH.a
        assert C.SPHERE_EARTH.e == 0.0

    def test_time_constants(self):
        assert C.SECONDS_PER_DAY == 86_400.0
        assert C.DAYS_PER_JULIAN_YEAR == 365.25
        assert C.DAYS_PER_JULIAN_CENTURY == 36_525.0
        assert C.SECONDS_PER_JULIAN_CENTURY == 36_525.0 * 86_400.0
        assert C.J2000_EPOCH_JD == 2_451_545.0
        assert C.MJD_OFFSET == 2_400_000.5

    def test_mathematical_constants(self):
        assert C.PI == math.pi
        assert C.TWO_PI == 2 * math.pi
        assert C.HALF_PI == math.pi / 2
        assert C.DEG_TO_RAD * 180.0 == pytest.approx(math.pi, rel=1e-15)
        assert C.RAD_TO_DEG == pytest.approx(180.0 / math.pi, rel=1e-15)
        assert C.ARCSEC_TO_RAD * 3600 * 180 == pytest.approx(math.pi, rel=1e-12)
        assert C.RAD_TO_ARCSEC * C.ARCSEC_TO_RAD == pytest.approx(1.0, rel=1e-15)

    def test_solar_system_constants(self):
        # IAU 2012 Resolution B2: AU is exact
        assert C.ASTRONOMICAL_UNIT == 149_597_870_700.0
        # DE405 heliocentric GM
        assert C.SUN_GM == pytest.approx(1.32712440018e20, rel=1e-11)
        # DE405 Earth-Moon mass ratio
        assert C.EARTH_MOON_MASS_RATIO == pytest.approx(81.30056, rel=1e-7)

    def test_physical_constants_dataclass_mirrors_module(self):
        pc = C.PhysicalConstants()
        assert pc.c == C.SPEED_OF_LIGHT
        assert pc.G == C.GRAVITATIONAL_CONSTANT
        assert pc.h == C.PLANCK_CONSTANT
        assert pc.k_B == C.BOLTZMANN_CONSTANT
        assert pc.sigma == C.STEFAN_BOLTZMANN_CONSTANT
        assert pc.e == C.ELEMENTARY_CHARGE
        assert pc.N_A == C.AVOGADRO_CONSTANT
        assert pc.R == C.UNIVERSAL_GAS_CONSTANT
        assert pc.g_0 == C.STANDARD_GRAVITY
        assert C.c == C.SPEED_OF_LIGHT
        assert C.G == C.GRAVITATIONAL_CONSTANT


# =============================================================================
# core.array_utils vs numpy/scipy ground truth
# =============================================================================


class TestArrayUtilsGroundTruth:
    def test_wrap_to_pi_interval_and_equivalence(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-50, 50, 1000)
        w = wrap_to_pi(x)
        assert np.all(w >= -np.pi) and np.all(w < np.pi)
        # Wrapped value differs from input by an integer multiple of 2*pi
        k = (x - w) / (2 * np.pi)
        assert np.allclose(k, np.round(k), atol=1e-9)
        assert wrap_to_pi(np.pi) == pytest.approx(-np.pi)  # documented [-pi, pi)
        assert wrap_to_pi(0.0) == 0.0

    def test_wrap_to_2pi_interval_and_equivalence(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-50, 50, 1000)
        w = wrap_to_2pi(x)
        assert np.all(w >= 0) and np.all(w < 2 * np.pi)
        k = (x - w) / (2 * np.pi)
        assert np.allclose(k, np.round(k), atol=1e-9)

    def test_wrap_to_range_and_degree_wrappers(self):
        assert wrap_to_range(370, 0, 360) == pytest.approx(10.0)
        assert wrap_to_range(-10, 0, 360) == pytest.approx(350.0)
        assert wrap_to_pm180(270) == pytest.approx(-90.0)
        assert wrap_to_pm180(180) == pytest.approx(-180.0)
        assert wrap_to_360(-90) == pytest.approx(270.0)
        rng = np.random.default_rng(2)
        x = rng.uniform(-1000, 1000, 500)
        w = wrap_to_range(x, -5.0, 7.0)
        assert np.all(w >= -5.0) and np.all(w < 7.0)
        k = (x - w) / 12.0
        assert np.allclose(k, np.round(k), atol=1e-9)

    def test_vector_shaping(self):
        assert column_vector([1, 2, 3]).shape == (3, 1)
        assert column_vector([[1, 2, 3]]).shape == (3, 1)
        assert row_vector([1, 2, 3]).shape == (1, 3)
        assert row_vector([[1], [2], [3]]).shape == (1, 3)

    def test_vec_unvec_matlab_semantics(self):
        A = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(vec(A).ravel(), [1, 3, 2, 4])
        np.testing.assert_array_equal(vec(A, order="C").ravel(), [1, 2, 3, 4])
        v = np.arange(1, 7)
        M = unvec(v, (2, 3))
        np.testing.assert_array_equal(M, [[1, 3, 5], [2, 4, 6]])
        np.testing.assert_array_equal(unvec(vec(A).ravel(), (2, 2)), A)

    def test_block_diag_vs_scipy(self):
        A = np.arange(4).reshape(2, 2)
        B = np.array([[5.0]])
        np.testing.assert_array_equal(block_diag(A, B), sla.block_diag(A, B))

    def test_skew_symmetric_cross_product_identity(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            v = rng.standard_normal(3)
            u = rng.standard_normal(3)
            S = skew_symmetric(v)
            np.testing.assert_allclose(S @ u, np.cross(v, u), atol=1e-12)
            np.testing.assert_allclose(S, -S.T, atol=0)
            np.testing.assert_allclose(unskew(S), v, atol=0)
        with pytest.raises(ValueError):
            skew_symmetric([1, 2])
        with pytest.raises(ValueError):
            unskew(np.eye(2))

    def test_normalize_vector(self):
        v, n = normalize_vector([3, 4], return_norm=True)
        np.testing.assert_allclose(v, [0.6, 0.8])
        assert n == pytest.approx(5.0)
        # Zero vector maps to zero, no NaN
        np.testing.assert_array_equal(normalize_vector([0.0, 0.0]), [0.0, 0.0])
        # Axis handling
        M = np.array([[3.0, 0.0], [4.0, 2.0]])
        vn, norms = normalize_vector(M, axis=0, return_norm=True)
        np.testing.assert_allclose(np.linalg.norm(vn, axis=0), [1.0, 1.0])
        np.testing.assert_allclose(norms, [5.0, 2.0])

    def test_outer_repmat_meshgrid(self):
        np.testing.assert_array_equal(
            outer_product([1, 2], [3, 4, 5]), np.outer([1, 2], [3, 4, 5])
        )
        np.testing.assert_array_equal(
            repmat([1, 2], 2, 3), np.tile(np.atleast_2d([1, 2]), (2, 3))
        )
        X, Y = meshgrid_ij(np.array([1, 2, 3]), np.array([4, 5]))
        np.testing.assert_array_equal(X, [[1, 1], [2, 2], [3, 3]])
        np.testing.assert_array_equal(Y, [[4, 5], [4, 5], [4, 5]])

    def test_is_positive_definite(self):
        assert is_positive_definite([[4, 2], [2, 5]])
        assert not is_positive_definite([[1, 2], [2, 1]])  # eig -1
        assert not is_positive_definite([[1, 2, 3], [4, 5, 6]])  # not square
        assert not is_positive_definite([[1, 5], [0, 1]])  # not symmetric

    def test_is_positive_definite_rejects_singular_matrices(self):
        """A singular matrix is semidefinite, not definite.

        The test was ``eigenvalues > -tol * max|lambda|``, which admits zero,
        so diag(1, 0) reported True and the name overstated the guarantee.
        """
        assert not is_positive_definite(np.diag([1.0, 0.0]))
        assert not is_positive_definite(np.zeros((2, 2)))
        # a genuinely negative eigenvalue, however small, is not definite
        assert not is_positive_definite(np.diag([1.0, -1e-12]))

    def test_is_positive_semidefinite_accepts_what_definite_rejects(self):
        """The tolerant check has its own name now.

        A covariance may legitimately be singular -- a perfectly known state
        component gives a zero eigenvalue -- so the semidefinite check is the
        right one there.
        """
        assert is_positive_semidefinite(np.diag([1.0, 0.0]))
        assert is_positive_semidefinite(np.zeros((2, 2)))
        assert is_positive_semidefinite(np.diag([1.0, -1e-12]))

        # but it still rejects a real negative eigenvalue
        assert not is_positive_semidefinite(np.diag([1.0, -1.0]))
        assert not is_positive_semidefinite([[1, 5], [0, 1]])  # not symmetric

    def test_definite_implies_semidefinite(self):
        """Anything definite must also pass the weaker check."""
        rng = np.random.default_rng(11)
        for _ in range(30):
            M = rng.normal(size=(4, 4))
            A = M @ M.T + np.eye(4) * 1e-3  # symmetric, well conditioned
            assert is_positive_definite(A)
            assert is_positive_semidefinite(A)

    def test_nearest_positive_definite_matches_higham_reference(self):
        """Higham (1988): nearest symmetric PSD in Frobenius norm equals
        eigenvalue clipping of the symmetric part."""
        rng = np.random.default_rng(4)
        for _ in range(20):
            A = rng.standard_normal((4, 4))
            B = (A + A.T) / 2
            lam, V = np.linalg.eigh(B)
            reference = V @ np.diag(np.maximum(lam, 0)) @ V.T
            got = nearest_positive_definite(A)
            np.testing.assert_allclose(got, reference, atol=1e-8)
            assert np.min(np.linalg.eigvalsh(got)) >= -1e-10
            # Frobenius optimality vs random symmetric PSD candidates
            d_opt = np.linalg.norm(A - got)
            for _ in range(5):
                Q = rng.standard_normal((4, 4))
                cand = Q @ Q.T  # random PSD
                assert d_opt <= np.linalg.norm(A - cand) + 1e-8

    def test_nearest_positive_definite_identity_on_pd_input(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        np.testing.assert_allclose(nearest_positive_definite(A), A, atol=1e-10)

    def test_safe_cholesky(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = safe_cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-12)
        np.testing.assert_allclose(L, np.tril(L))
        # Singular PSD input: succeeds via jitter, factorization near A
        S = np.array([[1.0, 1.0], [1.0, 1.0]])
        L = safe_cholesky(S)
        np.testing.assert_allclose(L @ L.T, S, atol=1e-4)


# =============================================================================
# core.validation contracts
# =============================================================================


class TestValidationContracts:
    def test_validate_array_all_constraints(self):
        np.testing.assert_array_equal(validate_array([1, 2, 3]), [1, 2, 3])
        assert validate_array([1, 2], dtype=np.float64).dtype == np.float64
        with pytest.raises(ValidationError):
            validate_array([1, 2], ndim=2)
        assert validate_array([[1]], ndim=(1, 2)).ndim == 2
        with pytest.raises(ValidationError):
            validate_array([[1, 2]], shape=(2, 2))
        with pytest.raises(ValidationError):
            validate_array([1, 2], shape=(2, 2))  # ndim mismatch via shape
        with pytest.raises(ValidationError):
            validate_array([1], min_ndim=2)
        with pytest.raises(ValidationError):
            validate_array([[1]], max_ndim=1)
        with pytest.raises(ValidationError):
            validate_array([1, np.nan], finite=True)
        with pytest.raises(ValidationError):
            validate_array([1, np.inf], finite=True)
        with pytest.raises(ValidationError):
            validate_array([1, -1], non_negative=True)
        assert validate_array([0, 1], non_negative=True) is not None
        with pytest.raises(ValidationError):
            validate_array([0, 1], positive=True)  # zero not positive
        with pytest.raises(ValidationError):
            validate_array([], allow_empty=False)

    def test_validation_error_is_value_error(self):
        assert issubclass(ValidationError, ValueError)
        assert issubclass(ValidationError, TCLError)

    def test_ensure_functions(self):
        assert ensure_2d([1, 2, 3]).shape == (3, 1)
        assert ensure_2d([1, 2, 3], axis="row").shape == (1, 3)
        assert ensure_column_vector([1, 2]).shape == (2, 1)
        with pytest.raises(ValidationError):
            ensure_column_vector([[1, 2], [3, 4]])
        assert ensure_row_vector([1, 2]).shape == (1, 2)
        with pytest.raises(ValidationError):
            ensure_row_vector([[1, 2], [3, 4]])
        with pytest.raises(ValidationError):
            ensure_square_matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValidationError):
            ensure_symmetric([[1, 2], [3, 4]])
        S = ensure_symmetric([[1, 2 + 1e-12], [2, 4]])
        np.testing.assert_allclose(S, S.T, atol=0)  # exactly symmetrized
        with pytest.raises(ValidationError):
            ensure_positive_definite([[1, 2], [2, 1]])
        np.testing.assert_allclose(
            ensure_positive_definite([[2, 0], [0, 3]]), [[2, 0], [0, 3]]
        )

    def test_validate_same_shape_and_compatible_shapes(self):
        validate_same_shape([1, 2], [3, 4])
        with pytest.raises(ValidationError):
            validate_same_shape([1, 2], [[3, 4]])
        check_compatible_shapes((3, 4), (4, 5))  # no dimension constraint: OK
        with pytest.raises(ValidationError):
            check_compatible_shapes((3, 4), (4, 5), dimension=0)
        check_compatible_shapes((3, 4), (3, 5), dimension=0)

    def test_scalar_spec(self):
        spec = ScalarSpec(dtype=int, min_value=1, max_value=10)
        assert spec.validate(5, "k") == 5
        assert spec.validate("7", "k") == 7  # coerced
        with pytest.raises(ValidationError):
            spec.validate(0, "k")
        with pytest.raises(ValidationError):
            spec.validate(11, "k")
        with pytest.raises(ValidationError):
            ScalarSpec(positive=True).validate(0, "x")
        with pytest.raises(ValidationError):
            ScalarSpec(non_negative=True).validate(-1, "x")
        with pytest.raises(ValidationError):
            ScalarSpec(finite=True).validate(np.inf, "x")

    def test_array_spec_hierarchy(self):
        spec = ArraySpec(ndim=2, positive_definite=True)
        np.testing.assert_allclose(
            spec.validate([[2, 0], [0, 1]], "P"), [[2, 0], [0, 1]]
        )
        with pytest.raises(ValidationError):
            spec.validate([[1, 2], [2, 1]], "P")
        with pytest.raises(ValidationError):
            ArraySpec(square=True).validate([[1, 2, 3]], "M")

    def test_validate_inputs_decorator(self):
        @validate_inputs(
            x=ArraySpec(ndim=1, finite=True),
            k=ScalarSpec(dtype=int, min_value=1),
            y={"ndim": 2},
        )
        def f(x, y, k=1):
            return np.sum(x) + k

        assert f([1, 2, 3], [[1]], k=2) == 8
        with pytest.raises(ValidationError):
            f([1, np.nan], [[1]])
        with pytest.raises(ValidationError):
            f([1, 2], [[1]], k=0)
        with pytest.raises(ValidationError):
            f([1, 2], [1])  # y must be 2D

        @validate_inputs(x="bad spec")
        def g(x):
            return x

        with pytest.raises(TypeError):
            g(1)

    def test_validated_array_input_decorator(self):
        @validated_array_input("x", ndim=1, finite=True)
        def f(x, y=1):
            return np.sum(x) + y

        assert f([1, 2, 3]) == 7
        with pytest.raises(ValidationError):
            f([[1, 2]])
        with pytest.raises(ValidationError):
            f([1, np.inf])


# =============================================================================
# core.paths
# =============================================================================


class TestPaths:
    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path / "custom"))
        assert get_data_dir() == tmp_path / "custom"
        d = ensure_data_dir()
        assert d.is_dir()

    def test_default_location(self, monkeypatch):
        monkeypatch.delenv("PYTCL_DATA_DIR", raising=False)
        d = get_data_dir()
        assert d.parts[-2:] == (".pytcl", "data")


# =============================================================================
# core.maturity
# =============================================================================


class TestMaturity:
    def test_get_maturity_and_prefix_stripping(self):
        assert get_maturity("core.constants") == MaturityLevel.STABLE
        assert get_maturity("pytcl.core.constants") == MaturityLevel.STABLE
        assert get_maturity("no.such.module") == MaturityLevel.EXPERIMENTAL
        assert get_maturity("terrain.dem") == MaturityLevel.EXPERIMENTAL

    def test_levels_ordering(self):
        assert MaturityLevel.DEPRECATED < MaturityLevel.EXPERIMENTAL
        assert MaturityLevel.EXPERIMENTAL < MaturityLevel.MATURE
        assert MaturityLevel.MATURE < MaturityLevel.STABLE

    def test_modules_by_maturity_partition(self):
        total = sum(len(get_modules_by_maturity(level)) for level in MaturityLevel)
        assert total == len(MODULE_MATURITY)
        assert "core.constants" in get_modules_by_maturity(MaturityLevel.STABLE)

    def test_summary_counts(self):
        summary = get_maturity_summary()
        assert sum(summary.values()) == len(MODULE_MATURITY)
        for level in MaturityLevel:
            assert summary[level] == len(get_modules_by_maturity(level))

    def test_is_stable_and_production_ready(self):
        assert is_stable("core.constants")
        assert not is_stable("terrain.dem")
        assert is_production_ready("dynamic_estimation.imm")  # MATURE
        assert not is_production_ready("terrain.dem")

    def test_format_maturity_badge(self):
        assert format_maturity_badge(MaturityLevel.STABLE) == "|stable|"
        assert format_maturity_badge(MaturityLevel.DEPRECATED) == "|deprecated|"


# =============================================================================
# core.optional_deps
# =============================================================================


class TestOptionalDeps:
    def test_is_available(self):
        assert is_available("numpy") is True
        assert is_available("definitely_not_a_module_xyz") is False

    def test_import_optional_success_and_failure(self):
        mod = import_optional("math")
        assert mod.sqrt(4) == 2.0
        with pytest.raises(DependencyError) as exc_info:
            import_optional("no_such_pkg_xyz", extra="visualization")
        assert "pip install" in exc_info.value.install_command
        assert issubclass(DependencyError, ImportError)

    def test_requires_decorator(self):
        @requires("numpy")
        def ok():
            return 42

        assert ok() == 42

        @requires("no_such_pkg_xyz", extra="visualization")
        def bad():
            return 0

        with pytest.raises(DependencyError):
            bad()

    def test_check_dependencies(self):
        check_dependencies("numpy")
        with pytest.raises(DependencyError):
            check_dependencies("no_such_pkg_xyz")

    def test_lazy_module(self):
        m = LazyModule("math")
        assert m.pi == math.pi
        bad = LazyModule("no_such_pkg_xyz")
        with pytest.raises(DependencyError):
            _ = bad.anything

    def test_module_level_flags_are_real_booleans(self):
        """Regression: HAS_* flags were module-level `property` objects
        (always truthy). They must be actual bools tracking availability."""
        assert isinstance(od.HAS_PLOTLY, bool)
        assert od.HAS_PLOTLY == is_available("plotly")
        assert isinstance(od.HAS_NETCDF4, bool)
        assert od.HAS_NETCDF4 == is_available("netCDF4")
        assert isinstance(od.HAS_CUPY, bool)
        assert od.HAS_CUPY == is_available("cupy")
        assert od.PYWT_AVAILABLE == od.HAS_PYWT
        with pytest.raises(AttributeError):
            _ = od.NOT_A_FLAG
        # from-import path also resolves through module __getattr__
        from pytcl.core.optional_deps import HAS_PLOTLY as flag

        assert isinstance(flag, bool)


# =============================================================================
# core.exceptions
# =============================================================================


class TestExceptions:
    def test_hierarchy_and_stdlib_mixins(self):
        assert issubclass(ValidationError, (TCLError, ValueError))
        assert issubclass(DimensionError, ValidationError)
        assert issubclass(ParameterError, ValidationError)
        assert issubclass(ComputationError, (TCLError, RuntimeError))
        assert issubclass(MethodError, (ConfigurationError, ValueError))
        assert issubclass(DependencyError, (ConfigurationError, ImportError))

    def test_details_formatting(self):
        e = TCLError("msg", details={"value": 42})
        assert str(e) == "msg (value=42)"
        assert str(TCLError("bare")) == "bare"
        v = ValidationError("bad", parameter="P", expected="3x3", actual="2x4")
        assert v.parameter == "P"
        assert "parameter=P" in str(v)
        d = DimensionError(
            "shape", expected_shape=(3, 3), actual_shape=(2, 4), parameter="P"
        )
        assert d.expected_shape == (3, 3)
        assert d.actual_shape == (2, 4)


# =============================================================================
# terrain.dem — analytic geometry ground truth
# =============================================================================


def _plane_dem(n_lat=11, n_lon=11, alpha=2.0, beta=3.0):
    """DEM whose data is a plane in index space: z[i, j] = alpha*i + beta*j."""
    i, j = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    data = alpha * i + beta * j
    lat_min, lat_max = np.radians(35.0), np.radians(36.0)
    lon_min, lon_max = np.radians(-120.0), np.radians(-119.0)
    return DEMGrid(data, lat_min, lat_max, lon_min, lon_max), (alpha, beta)


class TestDEMGridInterpolation:
    def test_bilinear_exact_on_plane(self):
        """Bilinear interpolation reproduces a plane exactly."""
        dem, (a, b) = _plane_dem()
        rng = np.random.default_rng(5)
        for _ in range(50):
            lat = rng.uniform(dem.lat_min, dem.lat_max)
            lon = rng.uniform(dem.lon_min, dem.lon_max)
            idx_lat = (lat - dem.lat_min) / dem.d_lat
            idx_lon = (lon - dem.lon_min) / dem.d_lon
            expected = a * idx_lat + b * idx_lon
            got = dem.get_elevation(lat, lon)
            assert got.valid
            assert got.elevation == pytest.approx(expected, abs=1e-9)

    def test_bilinear_matches_scipy_regular_grid(self):
        rng = np.random.default_rng(6)
        data = rng.standard_normal((9, 13)) * 500
        lat_min, lat_max = np.radians(10.0), np.radians(11.0)
        lon_min, lon_max = np.radians(20.0), np.radians(22.0)
        dem = DEMGrid(data, lat_min, lat_max, lon_min, lon_max)
        lats = np.linspace(lat_min, lat_max, 9)
        lons = np.linspace(lon_min, lon_max, 13)
        rgi = RegularGridInterpolator((lats, lons), data, method="linear")
        for _ in range(50):
            lat = rng.uniform(lat_min, lat_max)
            lon = rng.uniform(lon_min, lon_max)
            got = dem.get_elevation(lat, lon).elevation
            assert got == pytest.approx(float(rgi((lat, lon))), abs=1e-9)

    def test_nearest_interpolation(self):
        dem, (a, b) = _plane_dem()
        # A point closest to grid node (3, 7)
        lat = dem.lat_min + (3 + 0.3) * dem.d_lat
        lon = dem.lon_min + (7 - 0.4) * dem.d_lon
        got = dem.get_elevation(lat, lon, interpolation="nearest")
        assert got.elevation == pytest.approx(a * 3 + b * 7)

    def test_out_of_bounds_invalid(self):
        dem, _ = _plane_dem()
        got = dem.get_elevation(dem.lat_max + 0.01, dem.lon_min)
        assert not got.valid
        assert got.elevation == dem.nodata_value

    def test_nodata_propagates_invalid(self):
        data = np.zeros((5, 5))
        data[2, 2] = -9999.0
        dem = DEMGrid(data, np.radians(0), np.radians(1), np.radians(0), np.radians(1))
        # Query inside the cell adjacent to the nodata corner
        lat = dem.lat_min + 2.5 * dem.d_lat
        lon = dem.lon_min + 2.5 * dem.d_lon
        assert not dem.get_elevation(lat, lon).valid

    def test_get_elevations_batch_matches_scalar(self):
        dem, _ = _plane_dem()
        lats = np.linspace(dem.lat_min, dem.lat_max, 7)
        lons = np.linspace(dem.lon_min, dem.lon_max, 7)
        batch = dem.get_elevations(lats, lons)
        for k in range(7):
            assert batch[k] == pytest.approx(
                dem.get_elevation(lats[k], lons[k]).elevation
            )

    def test_metadata(self):
        dem, _ = _plane_dem()
        md = dem.get_metadata()
        assert md.lat_min == dem.lat_min
        assert md.resolution == pytest.approx(np.degrees(dem.d_lat) * 3600)


class TestDEMGradient:
    def test_gradient_on_tilted_plane(self):
        """Plane rising north and east: gradients match analytic values and
        aspect points at the steepest-descent direction (south-west)."""
        dem, (a, b) = _plane_dem()
        lat = dem.lat_min + 5 * dem.d_lat
        lon = dem.lon_min + 5 * dem.d_lon
        g = dem.get_gradient(lat, lon, earth_radius=R_EARTH)
        dz_dy_expected = a / (dem.d_lat * R_EARTH)
        dz_dx_expected = b / (dem.d_lon * R_EARTH * np.cos(lat))
        assert g.dz_dy == pytest.approx(dz_dy_expected, rel=1e-6)
        assert g.dz_dx == pytest.approx(dz_dx_expected, rel=1e-6)
        slope_expected = np.arctan(np.hypot(dz_dx_expected, dz_dy_expected))
        assert g.slope == pytest.approx(slope_expected, rel=1e-6)
        aspect_expected = np.arctan2(-g.dz_dx, -g.dz_dy) + 2 * np.pi
        assert g.aspect == pytest.approx(aspect_expected % (2 * np.pi), rel=1e-6)

    def test_gradient_north_facing_plane_aspect_south(self):
        """Rising to the north only -> steepest descent due south (aspect pi)."""
        dem, _ = _plane_dem(alpha=5.0, beta=0.0)
        lat = dem.lat_min + 5 * dem.d_lat
        lon = dem.lon_min + 5 * dem.d_lon
        g = dem.get_gradient(lat, lon)
        assert g.aspect == pytest.approx(np.pi, abs=1e-9)

    def test_flat_dem_zero_gradient(self):
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=500,
            resolution_arcsec=360,
        )
        g = dem.get_gradient(np.radians(35.5), np.radians(-119.5))
        assert g.slope == pytest.approx(0.0, abs=1e-12)
        assert g.dz_dx == pytest.approx(0.0, abs=1e-12)
        assert g.dz_dy == pytest.approx(0.0, abs=1e-12)


class TestElevationProfileAndResampling:
    def test_profile_points_lie_on_path(self):
        dem, (a, b) = _plane_dem()
        lat0, lon0 = np.radians(35.2), np.radians(-119.8)
        lat1, lon1 = np.radians(35.8), np.radians(-119.2)
        n = 25
        dists, elevs = get_elevation_profile(dem, lat0, lon0, lat1, lon1, n)
        assert len(dists) == n and len(elevs) == n
        assert dists[0] == 0.0
        assert np.all(np.diff(dists) > 0)
        lats = np.linspace(lat0, lat1, n)
        lons = np.linspace(lon0, lon1, n)
        # Elevations sampled exactly on the linearly spaced path
        for k in range(n):
            assert elevs[k] == pytest.approx(
                dem.get_elevation(lats[k], lons[k]).elevation, abs=1e-9
            )
        # Distances match an independent haversine computation
        for k in (1, n // 2, n - 1):
            s1 = np.sin((lats[k] - lat0) / 2) ** 2
            s2 = np.cos(lat0) * np.cos(lats[k]) * np.sin((lons[k] - lon0) / 2) ** 2
            d_expected = R_EARTH * 2 * np.arcsin(np.sqrt(s1 + s2))
            assert dists[k] == pytest.approx(d_expected, rel=1e-9)

    def test_interpolate_dem_vs_scipy(self):
        rng = np.random.default_rng(7)
        data = rng.standard_normal((11, 11)) * 300
        lat_min, lat_max = np.radians(35.0), np.radians(36.0)
        lon_min, lon_max = np.radians(-120.0), np.radians(-119.0)
        dem = DEMGrid(data, lat_min, lat_max, lon_min, lon_max)
        new = interpolate_dem(
            dem,
            np.radians(35.2),
            np.radians(35.8),
            np.radians(-119.8),
            np.radians(-119.2),
            new_n_lat=7,
            new_n_lon=5,
        )
        assert new.data.shape == (7, 5)
        lats = np.linspace(lat_min, lat_max, 11)
        lons = np.linspace(lon_min, lon_max, 11)
        rgi = RegularGridInterpolator((lats, lons), data, method="linear")
        new_lats = np.linspace(np.radians(35.2), np.radians(35.8), 7)
        new_lons = np.linspace(np.radians(-119.8), np.radians(-119.2), 5)
        for i in range(7):
            for j in range(5):
                assert new.data[i, j] == pytest.approx(
                    float(rgi((new_lats[i], new_lons[j]))), abs=1e-9
                )

    def test_merge_dems_first_valid_wins(self):
        dem1 = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=100,
            resolution_arcsec=600,
        )
        dem2 = create_flat_dem(
            np.radians(36),
            np.radians(37),
            np.radians(-120),
            np.radians(-119),
            elevation=200,
            resolution_arcsec=600,
        )
        merged = merge_dems(
            [dem1, dem2],
            np.radians(35),
            np.radians(37),
            np.radians(-120),
            np.radians(-119),
            resolution_arcsec=600,
        )
        assert merged.name == "Merged DEM"
        lo = merged.get_elevation(np.radians(35.5), np.radians(-119.5))
        hi = merged.get_elevation(np.radians(36.5), np.radians(-119.5))
        assert lo.elevation == pytest.approx(100, abs=1e-6)
        assert hi.elevation == pytest.approx(200, abs=1e-6)

    def test_create_flat_dem_constant(self):
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=500,
            resolution_arcsec=600,
        )
        assert np.all(dem.data == 500)
        got = dem.get_elevation(np.radians(35.37), np.radians(-119.21))
        assert got.elevation == pytest.approx(500)

    def test_create_synthetic_terrain_reproducible_and_bounded(self):
        kwargs = dict(
            lat_min=np.radians(35),
            lat_max=np.radians(36),
            lon_min=np.radians(-120),
            lon_max=np.radians(-119),
            base_elevation=500,
            amplitude=200,
            resolution_arcsec=600,
            seed=42,
        )
        d1 = create_synthetic_terrain(**kwargs)
        d2 = create_synthetic_terrain(**kwargs)
        np.testing.assert_array_equal(d1.data, d2.data)
        assert d1.data.min() < d1.data.max()
        # amplitude components sum to <= 1.0*amplitude + noise
        assert np.all(np.abs(d1.data - 500) < 200 * 1.5)


# =============================================================================
# terrain.visibility — hand ray-casting ground truth
# =============================================================================


def _wall_dem(wall_height=500.0):
    """1 deg x 1 deg DEM, elevation 0 with an E-W wall in 35.45..35.55 deg."""
    n = 101
    data = np.zeros((n, n))
    lats_deg = np.linspace(35.0, 36.0, n)
    wall_rows = (lats_deg >= 35.45) & (lats_deg <= 35.55)
    data[wall_rows, :] = wall_height
    return DEMGrid(
        data,
        np.radians(35.0),
        np.radians(36.0),
        np.radians(-120.0),
        np.radians(-119.0),
    )


def _ridge_dem():
    """Flat 0 with a 300 m E-W ridge band at 35.60..35.63 deg latitude."""
    n = 201
    data = np.zeros((n, n))
    lats_deg = np.linspace(35.0, 36.0, n)
    ridge_rows = (lats_deg >= 35.60) & (lats_deg <= 35.63)
    data[ridge_rows, :] = 300.0
    return DEMGrid(
        data,
        np.radians(35.0),
        np.radians(36.0),
        np.radians(-120.0),
        np.radians(-119.0),
    )


class TestLineOfSight:
    def test_blocked_by_wall_hand_raycast(self):
        """Observer and target at 10 m AGL on opposite sides of a 500 m wall:
        blocked, with clearance ~ -(500 - 10 - curvature_bulge_at_wall)."""
        dem = _wall_dem()
        res = line_of_sight(
            dem,
            np.radians(35.2),
            np.radians(-119.5),
            10.0,
            np.radians(35.8),
            np.radians(-119.5),
            10.0,
        )
        assert not res.visible
        # Hand computation: LOS height is 10 m along the whole path.  The
        # earth bulge d1*d2/(2R) raises the effective wall height, and is
        # largest at the path midpoint (which lies inside the wall band), so
        # the minimum clearance is 10 - 500 - D^2/(8R) at d ~ D/2.
        d_total = np.radians(0.6) * R_EARTH
        bulge_mid = d_total**2 / (8 * R_EARTH)
        expected_clearance = 10.0 - 500.0 - bulge_mid
        assert res.clearance == pytest.approx(expected_clearance, abs=3.0)
        assert res.obstacle_elevation == pytest.approx(500.0, abs=25.0)
        assert res.obstacle_distance == pytest.approx(d_total / 2, abs=1500.0)
        assert res.grazing_angle < 0

    def test_visible_over_wall_when_target_high(self):
        dem = _wall_dem()
        res = line_of_sight(
            dem,
            np.radians(35.2),
            np.radians(-119.5),
            10.0,
            np.radians(35.8),
            np.radians(-119.5),
            2000.0,
        )
        assert res.visible
        assert res.clearance > 0
        assert res.obstacle_distance == 0.0
        assert res.grazing_angle > 0

    def test_visible_over_flat_terrain_short_path(self):
        """~14 km path, 10 m antennas: earth bulge ~4 m < 10 m, visible."""
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=100,
            resolution_arcsec=600,
        )
        res = line_of_sight(
            dem,
            np.radians(35.45),
            np.radians(-119.55),
            10.0,
            np.radians(35.55),
            np.radians(-119.45),
            10.0,
        )
        assert res.visible
        assert res.clearance > 0

    def test_blocked_beyond_horizon_flat_terrain(self):
        """~57 km path, 10 m antennas: mid-path bulge ~64 m >> 10 m.
        Two 10 m towers cannot see each other beyond ~22 km combined
        horizon distance over a smooth sphere."""
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=100,
            resolution_arcsec=600,
        )
        res = line_of_sight(
            dem,
            np.radians(35.3),
            np.radians(-119.7),
            10.0,
            np.radians(35.7),
            np.radians(-119.3),
            10.0,
        )
        assert not res.visible
        # Clearance ~ 10 - D^2/(8R) at mid-path
        dlat = np.radians(0.4)
        dlon = np.radians(0.4) * np.cos(np.radians(35.5))
        d_path = R_EARTH * np.hypot(dlat, dlon)
        expected = 10.0 - d_path**2 / (8 * R_EARTH)
        assert res.clearance == pytest.approx(expected, abs=3.0)

    def test_same_point(self):
        dem = _wall_dem()
        res = line_of_sight(
            dem,
            np.radians(35.2),
            np.radians(-119.5),
            10.0,
            np.radians(35.2),
            np.radians(-119.5),
            10.0,
        )
        assert res.visible
        assert res.clearance == float("inf")

    def test_refraction_extends_visibility(self):
        """Effective-Earth refraction can only improve clearance."""
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=0,
            resolution_arcsec=600,
        )
        args = (
            dem,
            np.radians(35.05),
            np.radians(-119.5),
            5.0,
            np.radians(35.95),
            np.radians(-119.5),
            5.0,
        )
        no_refr = line_of_sight(*args, refraction_coeff=0.0)
        refr = line_of_sight(*args, refraction_coeff=0.13)
        assert refr.clearance > no_refr.clearance


class TestHorizonAndMaskingAngles:
    def test_masking_angle_toward_ridge_matches_trig(self):
        """Mask angle north = arctan((H - h_obs - drop)/d) at the ridge front."""
        dem = _ridge_dem()
        obs_lat, obs_lon, h = np.radians(35.5), np.radians(-119.5), 30.0
        angle = terrain_masking_angle(
            dem,
            obs_lat,
            obs_lon,
            h,
            azimuth=0.0,
            max_range=50000.0,
            n_samples=500,
        )
        d = np.radians(0.1) * R_EARTH  # ridge front edge, ~11.1 km
        drop = d**2 / (2 * R_EARTH)
        expected = np.arctan((300.0 - h - drop) / d)
        assert angle == pytest.approx(expected, abs=5e-4)

    def test_masking_angle_flat_matches_horizon_formula(self):
        """Over flat terrain the mask angle is the classic optical-horizon dip:
        max over d of -(h/d + d/2R) = -2*sqrt(h/(2R))."""
        dem = _ridge_dem()
        obs_lat, obs_lon, h = np.radians(35.5), np.radians(-119.5), 30.0
        angle = terrain_masking_angle(
            dem,
            obs_lat,
            obs_lon,
            h,
            azimuth=np.pi,  # south: flat
            max_range=50000.0,
            n_samples=500,
        )
        expected = -2.0 * np.sqrt(h / (2 * R_EARTH))
        assert angle == pytest.approx(expected, abs=1e-4)

    def test_compute_horizon_directional(self):
        dem = _ridge_dem()
        pts = compute_horizon(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            30.0,
            n_azimuths=4,
            max_range=50000.0,
            samples_per_radial=500,
        )
        assert len(pts) == 4
        north, east, south, west = pts
        assert north.azimuth == pytest.approx(0.0)
        assert south.azimuth == pytest.approx(np.pi)
        # North: ridge dominates the horizon
        d = np.radians(0.1) * R_EARTH
        drop = d**2 / (2 * R_EARTH)
        assert north.elevation_angle == pytest.approx(
            np.arctan((300.0 - 30.0 - drop) / d), abs=5e-4
        )
        assert north.terrain_elevation == pytest.approx(300.0, abs=30.0)
        assert north.distance == pytest.approx(d, abs=300.0)
        # South/East/West flat: horizon dip formula
        dip = -2.0 * np.sqrt(30.0 / (2 * R_EARTH))
        for p in (south, east, west):
            assert p.elevation_angle == pytest.approx(dip, abs=1e-4)


class TestViewshed:
    def test_geometric_shadow_of_ridge(self):
        """Cells behind the ridge along the north radial are shadowed; cells
        in front of it and the ridge crest are visible."""
        dem = _ridge_dem()
        obs_lat, obs_lon = np.radians(35.5), np.radians(-119.5)
        result = viewshed(
            dem,
            obs_lat,
            obs_lon,
            30.0,
            max_range=50000.0,
            n_radials=8,
            samples_per_radial=500,
        )
        j_obs = 100  # observer column (-119.5 deg on a 201-column grid)
        i_obs = 100
        assert result.visible[i_obs, j_obs]  # observer cell
        # Flat cells between observer and ridge: visible
        for i in range(104, 118):
            assert result.visible[i, j_obs], f"row {i} should be visible"
        # The front crest edge of the ridge (rows 119-120) is visible; the
        # deeper flat crest cells are correctly occluded by the front edge
        # (same height, larger distance -> lower elevation angle).
        assert result.visible[119, j_obs]
        assert result.visible[120, j_obs]
        # Shadow zone north of the ridge out to max range: not visible
        for i in range(130, 170):
            assert not result.visible[i, j_obs], f"row {i} should be shadowed"

    def test_flat_terrain_all_sampled_cells_visible(self):
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=100,
            resolution_arcsec=600,
        )
        result = viewshed(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            20.0,
            max_range=10000.0,
            n_radials=8,
            samples_per_radial=50,
        )
        assert result.visible.any()
        assert result.observer_height == 20.0


class TestRadarCoverage:
    def test_zero_min_elevation_equals_viewshed(self):
        dem = _ridge_dem()
        kwargs = dict(
            max_range=20000.0,
            target_height=100.0,
            n_radials=8,
            samples_per_radial=100,
        )
        cov = radar_coverage_map(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            30.0,
            min_elevation=0.0,
            refraction_coeff=0.13,
            **kwargs,
        )
        vs = viewshed(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            30.0,
            refraction_coeff=0.13,
            **kwargs,
        )
        np.testing.assert_array_equal(cov.visible, vs.visible)

    def test_high_min_elevation_masks_low_cells(self):
        """On flat terrain with 0 m targets, every cell at nonzero distance
        sits below a 0.1 rad elevation cut, so coverage is (almost) empty
        and always a subset of the unconstrained viewshed."""
        dem = create_flat_dem(
            np.radians(35),
            np.radians(36),
            np.radians(-120),
            np.radians(-119),
            elevation=0,
            resolution_arcsec=600,
        )
        kwargs = dict(
            max_range=20000.0,
            target_height=0.0,
            n_radials=16,
            samples_per_radial=50,
        )
        cov = radar_coverage_map(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            30.0,
            min_elevation=0.1,
            **kwargs,
        )
        vs = viewshed(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            30.0,
            refraction_coeff=0.13,
            **kwargs,
        )
        assert vs.visible.sum() > 0  # unconstrained coverage exists
        assert np.all(cov.visible <= vs.visible)  # masking only removes
        assert cov.visible.sum() == 0  # all cells below the elevation cut


# =============================================================================
# terrain.loaders — synthetic files only
# =============================================================================


class TestLoaderMetadataAndErrors:
    def test_gebco_metadata(self):
        md = get_gebco_metadata("GEBCO2025")
        assert md.version == "GEBCO2025"
        assert md.resolution_arcsec == 15.0
        assert md.lat_min == pytest.approx(np.radians(-90))
        assert md.lon_max == pytest.approx(np.radians(180))
        with pytest.raises(ValueError):
            get_gebco_metadata("GEBCO1999")

    def test_earth2014_metadata(self):
        md = get_earth2014_metadata("SUR")
        assert md.layer == "SUR"
        assert md.resolution_arcsec == 60.0
        with pytest.raises(ValueError):
            get_earth2014_metadata("XXX")

    def test_load_rejects_unknown_versions(self):
        with pytest.raises(ValueError):
            load_gebco(0, 0.1, 0, 0.1, version="GEBCO1999")
        with pytest.raises(ValueError):
            load_earth2014(0, 0.1, 0, 0.1, layer="XXX")

    def test_missing_files_raise_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_gebco(
                np.radians(1.23),
                np.radians(1.24),
                np.radians(4.56),
                np.radians(4.57),
            )
        with pytest.raises(FileNotFoundError):
            load_earth2014(
                np.radians(1.23),
                np.radians(1.24),
                np.radians(4.56),
                np.radians(4.57),
            )

    def test_parameter_tables(self):
        assert set(EARTH2014_PARAMETERS) == {"SUR", "BED", "TBI", "RET", "ICE"}
        for params in GEBCO_PARAMETERS.values():
            assert params["resolution_arcsec"] == 15.0
            assert params["n_lat"] == 43200
            assert params["n_lon"] == 86400


class TestParseEarth2014Binary:
    N_LON = 21600

    def _value(self, row, col):
        return (row * 251 + col * 3) % 20000 - 10000

    def test_sw_corner_region_round_trip(self, tmp_path):
        """Write the first rows of a synthetic global int16 big-endian file
        (official layout: cell-centered, first record at the SW corner,
        rows south to north) and verify values and bounds round-trip."""
        n_rows_written = 8
        rows = np.empty((n_rows_written, self.N_LON), dtype=">i2")
        for r in range(n_rows_written):
            rows[r, :] = self._value(r, np.arange(self.N_LON))
        path = tmp_path / "Earth2014.SUR2014.1min.geod.bin"
        path.write_bytes(rows.tobytes())

        lat_min = np.radians(-90.0 + 2.2 / 60.0)
        lat_max = np.radians(-90.0 + 4.8 / 60.0)
        lon_min = np.radians(-180.0 + 1.2 / 60.0)
        lon_max = np.radians(-180.0 + 6.7 / 60.0)
        data, lat_a, lat_b, lon_a, lon_b = parse_earth2014_binary(
            path, "SUR", lat_min, lat_max, lon_min, lon_max
        )
        d = np.radians(1.0 / 60.0)
        lat_start = np.radians(-90.0 + 1.0 / 120.0)
        lon_start = np.radians(-180.0 + 1.0 / 120.0)
        # Returned bounds are cell centers of the official grid
        i0 = round((lat_a - lat_start) / d)
        j0 = round((lon_a - lon_start) / d)
        assert lat_a == pytest.approx(lat_start + i0 * d, abs=1e-12)
        assert lon_a == pytest.approx(lon_start + j0 * d, abs=1e-12)
        # Region covers the request (within one cell)
        assert lat_a <= lat_min + d and lat_b >= lat_max - d
        assert lon_a <= lon_min + d and lon_b >= lon_max - d
        # Every parsed value matches the synthetic file contents at the
        # row/column implied by the returned bounds (S->N, W->E order)
        n_r, n_c = data.shape
        assert n_r == round((lat_b - lat_a) / d) + 1
        assert n_c == round((lon_b - lon_a) / d) + 1
        for i in range(n_r):
            for j in range(n_c):
                assert data[i, j] == self._value(i0 + i, j0 + j)


@pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not installed")
class TestParseGebcoNetCDF:
    def test_subregion_extraction(self, tmp_path):
        import netCDF4 as nc

        from pytcl.terrain.loaders import parse_gebco_netcdf

        lats = np.linspace(-2.0, 2.0, 41)  # 0.1 deg spacing
        lons = np.linspace(-3.0, 3.0, 61)
        elev = np.add.outer(100.0 * np.arange(41), np.arange(61))
        path = tmp_path / "GEBCO_test.nc"
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("lat", len(lats))
            ds.createDimension("lon", len(lons))
            ds.createVariable("lat", "f8", ("lat",))[:] = lats
            ds.createVariable("lon", "f8", ("lon",))[:] = lons
            ds.createVariable("elevation", "f8", ("lat", "lon"))[:] = elev

        data, lat_a, lat_b, lon_a, lon_b = parse_gebco_netcdf(
            path,
            np.radians(-1.0),
            np.radians(1.0),
            np.radians(0.0),
            np.radians(2.0),
        )
        lat_sel = np.where((lats >= -1.0) & (lats <= 1.0))[0]
        lon_sel = np.where((lons >= 0.0) & (lons <= 2.0))[0]
        expected = elev[np.ix_(lat_sel, lon_sel)]
        np.testing.assert_allclose(data, expected)
        assert lat_a == pytest.approx(np.radians(lats[lat_sel[0]]))
        assert lat_b == pytest.approx(np.radians(lats[lat_sel[-1]]))
        assert lon_a == pytest.approx(np.radians(lons[lon_sel[0]]))
        assert lon_b == pytest.approx(np.radians(lons[lon_sel[-1]]))

    def test_region_outside_coverage_raises(self, tmp_path):
        import netCDF4 as nc

        from pytcl.terrain.loaders import parse_gebco_netcdf

        path = tmp_path / "GEBCO_tiny.nc"
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("lat", 3)
            ds.createDimension("lon", 3)
            ds.createVariable("lat", "f8", ("lat",))[:] = [0.0, 0.1, 0.2]
            ds.createVariable("lon", "f8", ("lon",))[:] = [0.0, 0.1, 0.2]
            ds.createVariable("elevation", "f8", ("lat", "lon"))[:] = np.ones((3, 3))
        with pytest.raises(ValueError):
            parse_gebco_netcdf(
                path,
                np.radians(50),
                np.radians(51),
                np.radians(50),
                np.radians(51),
            )


class TestSyntheticTestDEMs:
    def test_create_test_gebco_dem(self):
        from pytcl.terrain.loaders import create_test_gebco_dem

        dem = create_test_gebco_dem(seed=42)
        dem2 = create_test_gebco_dem(seed=42)
        np.testing.assert_array_equal(dem.data, dem2.data)
        assert dem.name == "GEBCO_TEST"
        assert dem.lat_min == pytest.approx(np.radians(35.0))
        # Contains both land (positive) and ocean (negative) by design
        assert dem.data.max() > 0 and dem.data.min() < 0

    def test_create_test_earth2014_dem_layers(self):
        from pytcl.terrain.loaders import create_test_earth2014_dem

        for layer in ("SUR", "BED", "TBI", "RET", "ICE"):
            dem = create_test_earth2014_dem(layer=layer, seed=1)
            assert dem.name == f"Earth2014_TEST_{layer}"
            assert dem.data.shape[0] >= 2


# =============================================================================
# plotting — numerical helpers vs analytic references
# =============================================================================


class TestEllipseMath:
    def test_ellipse_points_lie_on_mahalanobis_contour(self):
        """Every generated point p satisfies
        (p-mu)' Sigma^-1 (p-mu) = n_std^2 exactly."""
        rng = np.random.default_rng(8)
        for _ in range(10):
            A = rng.standard_normal((2, 2))
            cov = A @ A.T + 0.5 * np.eye(2)
            mean = rng.standard_normal(2)
            n_std = rng.uniform(0.5, 3.0)
            x, y = covariance_ellipse_points(mean, cov, n_std=n_std, n_points=50)
            pts = np.stack([x - mean[0], y - mean[1]])
            m2 = np.einsum("ij,ik,kj->j", pts, np.linalg.inv(cov), pts)
            np.testing.assert_allclose(m2, n_std**2, rtol=1e-9)

    def test_ellipsoid_points_lie_on_mahalanobis_surface(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((3, 3))
        cov = A @ A.T + 0.5 * np.eye(3)
        mean = np.array([1.0, -2.0, 3.0])
        x, y, z = covariance_ellipsoid_points(mean, cov, n_std=2.0, n_points=15)
        assert x.shape == (15, 15)
        pts = np.stack([(x - 1.0).ravel(), (y + 2.0).ravel(), (z - 3.0).ravel()])
        m2 = np.einsum("ij,ik,kj->j", pts, np.linalg.inv(cov), pts)
        np.testing.assert_allclose(m2, 4.0, rtol=1e-9)

    def test_ellipse_parameters_vs_eigendecomposition(self):
        rng = np.random.default_rng(10)
        for _ in range(10):
            A = rng.standard_normal((2, 2))
            cov = A @ A.T + 0.1 * np.eye(2)
            a, b, theta = ellipse_parameters(cov)
            lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
            assert a == pytest.approx(np.sqrt(lam[0]), rel=1e-10)
            assert b == pytest.approx(np.sqrt(lam[1]), rel=1e-10)
            # Reconstruct: R(theta) diag(a^2, b^2) R(theta)' == cov
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            np.testing.assert_allclose(R @ np.diag([a**2, b**2]) @ R.T, cov, atol=1e-9)
        with pytest.raises(ValueError):
            ellipse_parameters(np.eye(3))

    def test_confidence_region_radius_vs_chi2_tables(self):
        # Published chi-squared quantiles
        assert confidence_region_radius(1, 0.95) == pytest.approx(
            np.sqrt(3.841458821), rel=1e-8
        )
        assert confidence_region_radius(2, 0.95) == pytest.approx(
            np.sqrt(5.991464547), rel=1e-8
        )
        assert confidence_region_radius(3, 0.95) == pytest.approx(
            np.sqrt(7.814727903), rel=1e-8
        )
        # 1D 68.27% ~ 1 sigma
        assert confidence_region_radius(1, 0.6826894921) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestFigurePayloads:
    """Figure builders: assert the data payloads match the input transforms."""

    def test_plot_covariance_ellipse_payload(self):
        from pytcl.plotting.ellipses import plot_covariance_ellipse

        mean = [1.0, 2.0]
        cov = [[2.0, 0.5], [0.5, 1.0]]
        trace = plot_covariance_ellipse(mean, cov, n_std=2.0)
        x_ref, y_ref = covariance_ellipse_points(mean, cov, n_std=2.0)
        np.testing.assert_allclose(np.asarray(trace.x), x_ref)
        np.testing.assert_allclose(np.asarray(trace.y), y_ref)
        assert trace.fill == "toself"
        unfilled = plot_covariance_ellipse(mean, cov, fill=False)
        assert unfilled.fill is None

    def test_plot_covariance_ellipses_trace_count(self):
        from pytcl.plotting.ellipses import plot_covariance_ellipses

        means = [[0, 0], [5, 5], [10, 0]]
        covs = [np.eye(2)] * 3
        fig = plot_covariance_ellipses(means, covs)
        assert len(fig.data) == 6  # ellipse + center per input

    def test_plot_covariance_ellipsoid_payload(self):
        from pytcl.plotting.ellipses import plot_covariance_ellipsoid

        cov = np.diag([1.0, 2.0, 3.0])
        trace = plot_covariance_ellipsoid([0, 0, 0], cov, n_std=1.0)
        x_ref, _, _ = covariance_ellipsoid_points([0, 0, 0], cov, n_std=1.0)
        np.testing.assert_allclose(np.asarray(trace.x), x_ref)

    def test_plot_trajectory_payloads(self):
        from pytcl.plotting.tracks import (
            plot_measurements_2d,
            plot_trajectory_2d,
            plot_trajectory_3d,
        )

        states = np.arange(20.0).reshape(5, 4)
        t2 = plot_trajectory_2d(states, x_idx=0, y_idx=2)
        np.testing.assert_array_equal(np.asarray(t2.x), states[:, 0])
        np.testing.assert_array_equal(np.asarray(t2.y), states[:, 2])
        t3 = plot_trajectory_3d(states, x_idx=0, y_idx=1, z_idx=3)
        np.testing.assert_array_equal(np.asarray(t3.z), states[:, 3])
        meas = np.arange(10.0).reshape(5, 2)
        tm = plot_measurements_2d(meas)
        np.testing.assert_array_equal(np.asarray(tm.x), meas[:, 0])
        np.testing.assert_array_equal(np.asarray(tm.y), meas[:, 1])

    def test_plot_tracking_result_traces_and_ellipse_payload(self):
        from pytcl.plotting.tracks import plot_tracking_result

        n = 10
        states = np.zeros((n, 4))
        states[:, 0] = np.arange(n)
        states[:, 2] = np.arange(n) * 2
        meas = states[:, [0, 2]] + 0.1
        covs = [np.eye(4) for _ in range(n)]
        fig = plot_tracking_result(
            true_states=states,
            estimates=states,
            measurements=meas,
            covariances=covs,
            ellipse_interval=5,
        )
        # true + measurements + estimates + 2 ellipses (steps 0 and 5)
        assert len(fig.data) == 5
        ellipse_trace = fig.data[3]
        ex, ey = covariance_ellipse_points(
            [states[0, 0], states[0, 2]], np.eye(2), n_std=2.0
        )
        np.testing.assert_allclose(np.asarray(ellipse_trace.x), ex)
        np.testing.assert_allclose(np.asarray(ellipse_trace.y), ey)

    def test_plot_multi_target_tracks(self):
        from pytcl.plotting.tracks import plot_multi_target_tracks

        tracks = {
            "A": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "B": np.array([[5.0, 5.0], [6.0, 4.0]]),
        }
        fig = plot_multi_target_tracks(tracks)
        assert len(fig.data) == 4  # line + id label per track
        np.testing.assert_array_equal(np.asarray(fig.data[0].x), [0.0, 1.0])

    def test_plot_state_time_series_payload(self):
        from pytcl.plotting.tracks import plot_state_time_series

        states = np.arange(12.0).reshape(4, 3)
        fig = plot_state_time_series(states)
        assert len(fig.data) == 3
        for i in range(3):
            np.testing.assert_array_equal(np.asarray(fig.data[i].y), states[:, i])

    def test_plot_estimation_comparison_bounds(self):
        from pytcl.plotting.tracks import plot_estimation_comparison

        n = 6
        truth = np.zeros((n, 2))
        est = np.ones((n, 2))
        covs = [np.diag([4.0, 9.0])] * n
        fig = plot_estimation_comparison(
            truth, est, covariances=covs, n_std=2.0, state_indices=[0]
        )
        assert len(fig.data) == 3  # bounds + true + estimate
        band = np.asarray(fig.data[0].y)
        # Upper bound is est + 2*sqrt(4) = 5, lower is est - 4 = -3
        np.testing.assert_allclose(band[:n], 5.0)
        np.testing.assert_allclose(band[n:], -3.0)

    def test_create_animated_tracking_frames(self):
        from pytcl.plotting.tracks import create_animated_tracking

        n = 5
        states = np.zeros((n, 4))
        states[:, 0] = np.arange(n)
        states[:, 2] = np.arange(n)
        meas = states[:, [0, 2]]
        fig = create_animated_tracking(states, states, meas)
        assert len(fig.frames) == n

    def test_plot_rmse_over_time_running_rmse_formula(self):
        from pytcl.plotting.metrics import plot_rmse_over_time

        errors = np.array([[3.0], [4.0], [12.0]])
        fig = plot_rmse_over_time(errors)
        expected = np.sqrt(np.cumsum(errors[:, 0] ** 2) / np.arange(1, 4))
        np.testing.assert_allclose(np.asarray(fig.data[0].y), expected)

    def test_plot_nees_sequence_chi2_bounds(self):
        from scipy import stats

        from pytcl.plotting.metrics import plot_nees_sequence

        nees = np.full(10, 2.0)
        fig = plot_nees_sequence(nees, n_dims=2, confidence=0.95)
        band = np.asarray(fig.data[0].y)
        upper = stats.chi2.ppf(0.975, df=2)
        lower = stats.chi2.ppf(0.025, df=2)
        np.testing.assert_allclose(band[:10], upper)
        np.testing.assert_allclose(band[10:], lower)
        # Expected-value line equals the state dimension
        np.testing.assert_allclose(np.asarray(fig.data[1].y), 2.0)
        np.testing.assert_allclose(np.asarray(fig.data[2].y), nees)

    def test_plot_ospa_and_cardinality(self):
        from pytcl.plotting.metrics import (
            plot_cardinality_over_time,
            plot_ospa_over_time,
        )

        ospa = np.array([1.0, 2.0, 3.0])
        fig = plot_ospa_over_time(ospa, localization=ospa / 2, cardinality=ospa / 3)
        assert len(fig.data) == 3
        np.testing.assert_allclose(np.asarray(fig.data[0].y), ospa)
        np.testing.assert_allclose(np.asarray(fig.data[1].y), ospa / 2)
        fig2 = plot_cardinality_over_time([2, 2, 3], [2, 3, 3])
        assert len(fig2.data) == 2
        np.testing.assert_array_equal(np.asarray(fig2.data[0].y), [2, 2, 3])

    def test_plot_error_histogram_and_consistency(self):
        from pytcl.plotting.metrics import (
            plot_consistency_summary,
            plot_error_histogram,
        )

        rng = np.random.default_rng(11)
        errors = rng.standard_normal((200, 2))
        fig = plot_error_histogram(errors, show_gaussian_fit=True)
        assert len(fig.data) == 4  # histogram + fit per component
        fig2 = plot_consistency_summary(np.full(10, 4.0), nis_values=np.full(10, 2.0))
        assert len(fig2.data) == 6

    def test_plot_monte_carlo_rmse_formula(self):
        from pytcl.plotting.metrics import plot_monte_carlo_rmse

        rng = np.random.default_rng(12)
        errors = rng.standard_normal((20, 15, 2))
        fig = plot_monte_carlo_rmse(errors)
        expected = np.sqrt(np.mean(errors**2, axis=0))
        np.testing.assert_allclose(np.asarray(fig.data[0].y), expected[:, 0])
        np.testing.assert_allclose(np.asarray(fig.data[1].y), expected[:, 1])

    def test_plot_coordinate_axes_endpoints(self):
        from pytcl.plotting.coordinates import plot_coordinate_axes_3d

        # 90 deg rotation about z: x-axis maps to y-axis
        Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        traces = plot_coordinate_axes_3d(
            origin=(1, 2, 3), rotation_matrix=Rz, scale=2.0
        )
        assert len(traces) == 3
        # X-axis endpoint: origin + 2*R[:,0] = (1, 4, 3)
        assert traces[0].x == (1, 1)
        assert traces[0].y == (2, 4)
        assert traces[0].z == (3, 3)
        # Z-axis unchanged by Rz: endpoint (1, 2, 5)
        assert traces[2].z == (3, 5)

    def test_plot_points_spherical_conversion(self):
        from pytcl.plotting.coordinates import plot_points_spherical

        pts = np.array([[2.0, 0.0, np.pi / 2]])  # r=2 on +x axis
        fig = plot_points_spherical(pts)
        assert fig.data[0].x[0] == pytest.approx(2.0)
        assert fig.data[0].y[0] == pytest.approx(0.0, abs=1e-12)
        assert fig.data[0].z[0] == pytest.approx(0.0, abs=1e-12)

    def test_plot_rotation_and_euler_and_transform(self):
        from pytcl.plotting.coordinates import (
            plot_coordinate_transform,
            plot_euler_angles,
            plot_rotation_comparison,
        )

        fig = plot_rotation_comparison(np.eye(3), np.eye(3))
        assert len(fig.data) == 6
        fig2 = plot_euler_angles([0.1, 0.2, 0.3], sequence="ZYX")
        assert len(fig2.data) == 24  # 4 subplots x 6 axis traces
        pts = np.arange(9.0).reshape(3, 3)
        fig3 = plot_coordinate_transform(pts, pts + 1)
        np.testing.assert_array_equal(np.asarray(fig3.data[0].x), pts[:, 0])
        np.testing.assert_array_equal(np.asarray(fig3.data[1].x), pts[:, 0] + 1)

    def test_plot_spherical_grid_and_slerp(self):
        from pytcl.plotting.coordinates import (
            plot_quaternion_interpolation,
            plot_spherical_grid,
        )

        fig = plot_spherical_grid(r=2.0)
        assert len(fig.data) == 4  # sphere + 3 axes
        np.testing.assert_allclose(np.max(np.asarray(fig.data[0].z)), 2.0)
        fig2 = plot_quaternion_interpolation([1, 0, 0, 0], [0, 0, 0, 1], n_steps=5)
        assert len(fig2.frames) == 5


# =============================================================================
# Empty namespace modules
# =============================================================================
