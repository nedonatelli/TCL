"""``magnetic_field_spherical`` against the closed-form dipole field.

The exported spherical-coordinate entry point had no test reaching it (gh-49).
Its sibling `magnetic_field` is exercised; this one is not, and they are separate
code paths with separate coordinate conventions.

The oracle here is analysis rather than another implementation. A geomagnetic
model truncated to degree 1 *is* a dipole, and a dipole has an exact closed form.
Feeding the evaluator a synthetic coefficient set with a single nonzero term
turns the whole spherical-harmonic machinery into something with a known answer,
which pins down every part of it that a full-model comparison would leave
ambiguous: the Schmidt normalization, the ``(a/r)^(n+2)`` radial dependence, the
sign of each component, and the handedness of the colatitude and longitude
derivatives.

That matters because this package shipped WMM magnetism roughly 180 degrees
wrong once already. A test comparing two of this library's own routines would
have agreed with itself throughout.

Companion coverage: ``test_magnetic_coefficients.py`` pins the shipped
coefficient tables to the official NOAA and IAGA files. Correct tables plus a
correct evaluator is what makes the modeled field correct; neither test alone
establishes it.

Reference: the geomagnetic potential in Schmidt semi-normalized form,

    V = a * sum_n (a/r)^(n+1) * sum_m (g_nm cos(m phi) + h_nm sin(m phi)) P_nm

with B = -grad V. See Chapman & Bartels, *Geomagnetism* (1940), or the WMM
Technical Report section 2.
"""

import numpy as np
import pytest

from pytcl.magnetism.wmm import MagneticCoefficients, magnetic_field_spherical

# The WMM/IGRF geomagnetic reference radius, not an Earth radius: it is a
# defined constant of the model, and the field scales as (a/r)^(n+2) from it.
REFERENCE_RADIUS_KM = 6371.2
N_MAX = 12

# Component magnitudes here are of order 1e4 nT and the Legendre recursion
# accumulates roughly 1e-13 relative error over twelve degrees, so an absolute
# tolerance alone is the wrong instrument. The floor covers components that are
# analytically zero, where a relative tolerance says nothing. Both are far below
# the ~150 nT accuracy the WMM itself claims.
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_FLOOR_NT = 1e-6


def _coefficients(**terms: float) -> MagneticCoefficients:
    """A synthetic model with only the named terms set.

    ``_coefficients(g10=-30000.0)`` is a pure axial dipole. Keys are the usual
    ``g``/``h`` notation with single-digit degree and order.
    """
    g = np.zeros((N_MAX + 1, N_MAX + 1))
    h = np.zeros((N_MAX + 1, N_MAX + 1))
    for key, value in terms.items():
        table = g if key[0] == "g" else h
        table[int(key[1]), int(key[2])] = value
    return MagneticCoefficients(
        g=g,
        h=h,
        g_dot=np.zeros_like(g),
        h_dot=np.zeros_like(h),
        epoch=2025.0,
        n_max=N_MAX,
    )


class TestAxialDipole:
    """Only ``g[1,0]`` set. The classic centered, aligned dipole.

    B_r     =  2 (a/r)^3 g10 cos(theta)
    B_theta =    (a/r)^3 g10 sin(theta)
    B_phi   =  0
    """

    G10 = -30000.0
    LATITUDES_DEG = [-89.0, -60.0, -30.0, 0.0, 30.0, 45.0, 60.0, 89.0]

    @pytest.mark.parametrize("lat_deg", LATITUDES_DEG)
    @pytest.mark.parametrize("lon_deg", [0.0, 90.0, 187.5, -45.0])
    def test_components_match_the_closed_form(self, lat_deg, lon_deg):
        """Exact, at every latitude and longitude."""
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        colatitude = np.pi / 2 - lat

        b_r, b_theta, b_phi = magnetic_field_spherical(
            lat,
            lon,
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(g10=self.G10),
            use_cache=False,
        )

        assert b_r == pytest.approx(
            2 * self.G10 * np.cos(colatitude),
            rel=RELATIVE_TOLERANCE,
            abs=ABSOLUTE_FLOOR_NT,
        )
        assert b_theta == pytest.approx(
            self.G10 * np.sin(colatitude), rel=RELATIVE_TOLERANCE, abs=ABSOLUTE_FLOOR_NT
        )
        assert b_phi == pytest.approx(0.0, abs=ABSOLUTE_FLOOR_NT), (
            "an axially symmetric field has no east-west component"
        )

    @pytest.mark.parametrize("radius_multiple", [1.0, 1.5, 2.0, 3.0, 10.0])
    def test_the_field_falls_off_as_the_inverse_cube(self, radius_multiple):
        """A dipole is ``(a/r)^3``, from the ``(n+1)=2`` potential exponent.

        Getting this exponent wrong is a whole-model error that no single-radius
        test can see, and satellite-altitude users are the ones who would find
        it.
        """
        b_r, _, _ = magnetic_field_spherical(
            np.radians(89.999999),
            0.0,
            REFERENCE_RADIUS_KM * radius_multiple,
            2025.0,
            coeffs=_coefficients(g10=self.G10),
            use_cache=False,
        )
        assert b_r == pytest.approx(
            2 * self.G10 / radius_multiple**3, rel=RELATIVE_TOLERANCE
        )

    def test_the_field_is_twice_as_strong_at_the_pole_as_at_the_equator(self):
        """A textbook dipole identity, and a check on the factor of 2 in B_r."""
        pole = magnetic_field_spherical(
            np.radians(89.999999),
            0.0,
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(g10=self.G10),
            use_cache=False,
        )
        equator = magnetic_field_spherical(
            0.0,
            0.0,
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(g10=self.G10),
            use_cache=False,
        )
        assert np.linalg.norm(pole) == pytest.approx(
            2 * np.linalg.norm(equator), rel=1e-6
        )

    def test_the_field_points_inward_in_the_northern_hemisphere(self):
        """The sign that makes a compass work.

        ``g10`` is negative for Earth, so ``B_r = 2 g10 cos(theta)`` is negative
        north of the equator: the field enters the surface there. This is the
        assertion that fails when the model comes out 180 degrees wrong.
        """
        north, _, _ = magnetic_field_spherical(
            np.radians(60.0),
            0.0,
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(g10=self.G10),
            use_cache=False,
        )
        south, _, _ = magnetic_field_spherical(
            np.radians(-60.0),
            0.0,
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(g10=self.G10),
            use_cache=False,
        )
        assert north < 0.0 < south


class TestEquatorialDipole:
    """Only ``g[1,1]`` and ``h[1,1]`` set, giving longitude dependence.

    With ``S(phi) = g11 cos(phi) + h11 sin(phi)``:

        B_r     =  2 (a/r)^3 S(phi) sin(theta)
        B_theta =   -(a/r)^3 S(phi) cos(theta)
        B_phi   =    (a/r)^3 (g11 sin(phi) - h11 cos(phi))

    The axial case above leaves ``B_phi`` identically zero and the longitude
    argument unused, so on its own it cannot tell whether longitude is wired up
    at all, let alone with the right sign.
    """

    G11, H11 = -1500.0, 4600.0
    POINTS_DEG = [
        (0.0, 0.0),
        (30.0, 45.0),
        (-60.0, 120.0),
        (45.0, -170.0),
        (10.0, 359.0),
    ]

    @pytest.mark.parametrize("lat_deg,lon_deg", POINTS_DEG)
    def test_components_match_the_closed_form(self, lat_deg, lon_deg):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        colatitude = np.pi / 2 - lat
        combined = self.G11 * np.cos(lon) + self.H11 * np.sin(lon)

        b_r, b_theta, b_phi = magnetic_field_spherical(
            lat,
            lon,
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(g11=self.G11, h11=self.H11),
            use_cache=False,
        )

        assert b_r == pytest.approx(
            2 * combined * np.sin(colatitude),
            rel=RELATIVE_TOLERANCE,
            abs=ABSOLUTE_FLOOR_NT,
        )
        assert b_theta == pytest.approx(
            -combined * np.cos(colatitude),
            rel=RELATIVE_TOLERANCE,
            abs=ABSOLUTE_FLOOR_NT,
        )
        assert b_phi == pytest.approx(
            self.G11 * np.sin(lon) - self.H11 * np.cos(lon),
            rel=RELATIVE_TOLERANCE,
            abs=ABSOLUTE_FLOOR_NT,
        )

    def test_the_field_actually_varies_with_longitude(self):
        """Guard the guard: the closed form above is only meaningful if the
        evaluator reads ``lon`` at all. A stub ignoring it would satisfy every
        axial-dipole assertion in this file."""
        sampled = [
            magnetic_field_spherical(
                np.radians(20.0),
                np.radians(lon_deg),
                REFERENCE_RADIUS_KM,
                2025.0,
                coeffs=_coefficients(g11=self.G11, h11=self.H11),
                use_cache=False,
            )[0]
            for lon_deg in (0.0, 90.0, 180.0, 270.0)
        ]
        assert len(set(np.round(sampled, 6))) == len(sampled), (
            f"B_r is the same at four longitudes ({sampled}), so longitude is "
            f"being ignored"
        )


class TestSuperpositionAndCaching:
    """Properties that hold for any coefficient set."""

    def test_the_field_of_a_sum_is_the_sum_of_the_fields(self):
        """The potential is linear in the coefficients, so the field is too.

        A normalization applied once per call rather than once per term would
        break this while leaving each individual case looking right.
        """
        point = (np.radians(37.0), np.radians(-122.0), REFERENCE_RADIUS_KM, 2025.0)
        axial = np.array(
            magnetic_field_spherical(
                *point, coeffs=_coefficients(g10=-30000.0), use_cache=False
            )
        )
        equatorial = np.array(
            magnetic_field_spherical(
                *point, coeffs=_coefficients(g11=-1500.0, h11=4600.0), use_cache=False
            )
        )
        combined = np.array(
            magnetic_field_spherical(
                *point,
                coeffs=_coefficients(g10=-30000.0, g11=-1500.0, h11=4600.0),
                use_cache=False,
            )
        )
        np.testing.assert_allclose(
            combined,
            axial + equatorial,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_FLOOR_NT,
        )

    # The cached path rounds its inputs before the lookup: latitude and
    # longitude to 1e-6 rad (about 6.4 m at the surface), radius to 1e-3 km,
    # decimal year to 0.01. So `use_cache` is not free -- it answers for the
    # nearest grid point, not for the point asked about. The two tests below
    # separate "the cache is broken" from "the cache quantizes as documented",
    # which a single equality assertion cannot do.
    EARTH_LIKE = dict(g10=-29350.0, g11=-1410.0, h11=4545.0)

    def test_caching_is_exact_for_an_input_already_on_the_quantization_grid(self):
        """No rounding to do, so the two paths must agree bit for bit.

        This is the assertion that fails if the cache returns a stale entry or
        is keyed on the wrong thing.
        """
        point = (round(0.9, 6), round(-0.002, 6), round(6371.2, 3), round(2025.0, 2))
        coefficients = _coefficients(**self.EARTH_LIKE)

        cached = magnetic_field_spherical(*point, coeffs=coefficients, use_cache=True)
        uncached = magnetic_field_spherical(
            *point, coeffs=coefficients, use_cache=False
        )
        assert cached == uncached, (
            f"on-grid input gave {cached} cached and {uncached} uncached; with "
            f"no rounding to do these must be identical"
        )

    def test_caching_off_the_grid_stays_within_what_quantization_permits(self):
        """Off-grid, the cached answer is for a point up to ~6.4 m away.

        The horizontal gradient of the field is roughly 0.005 nT/m, so that
        displacement is worth a few hundredths of a nT. Asserting a bound of
        1 nT keeps the check meaningful while staying two orders of magnitude
        under the ~150 nT accuracy the WMM itself claims -- a cache that
        returned a genuinely wrong entry would miss by thousands.
        """
        point = (np.radians(51.5), np.radians(-0.1278), REFERENCE_RADIUS_KM, 2025.0)
        coefficients = _coefficients(**self.EARTH_LIKE)

        cached = np.array(
            magnetic_field_spherical(*point, coeffs=coefficients, use_cache=True)
        )
        uncached = np.array(
            magnetic_field_spherical(*point, coeffs=coefficients, use_cache=False)
        )
        largest = float(np.max(np.abs(cached - uncached)))
        assert largest < 1.0, (
            f"cached and uncached differ by {largest:.4f} nT, more than input "
            f"quantization can account for"
        )

    def test_an_empty_model_produces_no_field(self):
        """All-zero coefficients must give exactly zero, not a residual."""
        result = magnetic_field_spherical(
            np.radians(45.0),
            np.radians(90.0),
            REFERENCE_RADIUS_KM,
            2025.0,
            coeffs=_coefficients(),
            use_cache=False,
        )
        assert result == (0.0, 0.0, 0.0)


def test_the_default_model_gives_a_plausible_earth_field():
    """The shipped default, sanity-checked against the real Earth.

    Surface field strength runs from roughly 22,000 nT near the South Atlantic
    Anomaly to roughly 67,000 nT near the poles. This will not catch a subtle
    error -- the closed-form tests above are for that -- but it does catch a
    default model that is empty, misscaled, or in the wrong units.
    """
    for lat_deg, lon_deg in [(0.0, 0.0), (45.0, -75.0), (-30.0, 25.0), (80.0, 100.0)]:
        components = magnetic_field_spherical(
            np.radians(lat_deg),
            np.radians(lon_deg),
            REFERENCE_RADIUS_KM,
            2025.0,
            use_cache=False,
        )
        strength = float(np.linalg.norm(components))
        assert 20000.0 < strength < 70000.0, (
            f"default model gives |B| = {strength:.1f} nT at "
            f"({lat_deg}, {lon_deg}), outside the range the real field occupies"
        )
