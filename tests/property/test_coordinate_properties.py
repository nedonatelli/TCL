"""Property-based round-trip tests for coordinate_systems conversions.

Covers two transform pairs:

- ``cart2sphere`` / ``sphere2cart`` (pytcl/coordinate_systems/conversions/spherical.py),
  across all three ``system_type`` values -- their azimuth/elevation conventions
  genuinely differ (see the module docstring there), so a property that holds
  for one is not free for the others.
- ``geodetic2ecef`` / ``ecef2geodetic`` (.../conversions/geodetic.py), across
  both ``ecef2geodetic`` methods ("iterative" Bowring, "direct" Vermeille).

Generators deliberately hit the hard cases instead of avoiding them: Cartesian
points spanning 1e-6 to 1e7 in magnitude (log-uniform, so every decade gets
exercised -- a linear ``st.floats(1e-6, 1e7)`` would spend almost all its mass
near the top of the range), exact axis alignment (the pole/singularity
direction), and the origin; geodetic latitudes including exactly +/-90 deg,
longitudes including +/-180 deg (the antimeridian), and both positive and
negative altitudes.

A recurring theme below: spherical/geodetic coordinates have a genuine
mathematical singularity at the poles (azimuth/longitude is not a function of
position there -- every azimuth maps to the same point). Whether that
singularity is *observable* in float64 depends on whether the pole angle
itself is exactly representable:

- ``np.sin(0.0) == 0.0`` exactly (0.0 is exact, and IEEE 754 guarantees
  sin(0) = 0 exactly). So the "standard" system's north pole (``el = 0.0``)
  destroys azimuth's *magnitude* exactly and irrecoverably: x = y = 0.0 bit
  for bit. It does not destroy azimuth entirely, though -- IEEE 754's signed
  zero survives the multiplication (``0.0 * cos(az)`` carries the sign of
  ``cos(az)``), and ``atan2`` of a pair of signed zeros is defined to return
  exactly 0.0 or +/-pi depending on that sign. So one bit of az (which half
  of the circle ``cos(az)`` fell in) is recoverable even at this exact
  singularity; see ``TestSphereCartRoundTrip`` below for the verified
  breakdown and a counterexample that falsified the naive "always 0.0"
  assumption this docstring originally made.
- ``np.pi`` and ``np.pi / 2`` are *not* exactly representable, so
  ``np.sin(np.pi)`` and ``np.cos(np.pi / 2)`` are ~1e-16, not 0.0. The
  "standard" south pole (``el = pi``) and both "az-el" poles
  (``el = +/-pi/2``) therefore leave a tiny but nonzero, azimuth-proportional
  residual in x/y -- so azimuth is (accidentally) numerically recoverable
  there, even though it is mathematically meaningless. Same story for
  geodetic longitude at ``lat = +/-pi/2``. A third, distinct regime -- ``el``
  nonzero but deep in float64's subnormal range -- collapses the same way as
  the exact zero above but *without* the clean signed-zero characterization
  (x and y both quantize to the same one or two subnormal steps regardless
  of az); see ``test_near_pole_subnormal_azimuth_regression``.

Each test below says explicitly which case it is in and why.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from pytcl.coordinate_systems.conversions.geodetic import ecef2geodetic, geodetic2ecef
from pytcl.coordinate_systems.conversions.spherical import cart2sphere, sphere2cart

SYSTEM_TYPES = ["standard", "az-el", "range-az-el"]
GEODETIC_METHODS = ["iterative", "direct"]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _log_uniform_component(min_exp: int = -6, max_exp: int = 7) -> st.SearchStrategy:
    """A signed float with magnitude log-uniform over [10**min_exp, 10**max_exp].

    Sampling the decimal exponent uniformly (rather than the value itself)
    guarantees Hypothesis explores every magnitude decade from 1e-6 to 1e7,
    per the task brief -- a plain ``st.floats(1e-6, 1e7)`` would draw almost
    exclusively values near 1e7.
    """
    sign = st.sampled_from([-1.0, 1.0])
    exponent = st.integers(min_value=min_exp, max_value=max_exp)
    mantissa = st.floats(
        min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False
    )
    return st.builds(lambda s, e, m: s * m * (10.0**e), sign, exponent, mantissa)


@st.composite
def cartesian_points(draw) -> np.ndarray:
    """3-vectors covering the hard cases named in the brief.

    ``kind`` deliberately biases toward the exact axes (1/4 of draws land
    exactly on one coordinate axis -- the pole/singularity direction for
    every ``system_type``) and the exact origin, rather than leaving them to
    the vanishing probability of an independent-per-component draw landing
    on zero.
    """
    kind = draw(st.sampled_from(["generic", "axis_x", "axis_y", "axis_z", "origin"]))
    if kind == "origin":
        return np.zeros(3, dtype=np.float64)
    comp = _log_uniform_component()
    if kind == "axis_x":
        return np.array([draw(comp), 0.0, 0.0])
    if kind == "axis_y":
        return np.array([0.0, draw(comp), 0.0])
    if kind == "axis_z":
        return np.array([0.0, 0.0, draw(comp)])
    return np.array([draw(comp), draw(comp), draw(comp)])


@st.composite
def spherical_triples(draw, system_type: str):
    """(r, az, el) triples, including the exact pole elevation for ``system_type``.

    ``el`` is drawn from an explicit pole-edge sample half the time so the
    exact singularity is exercised, not just approached; the rest of the time
    it's a generic angle anywhere in the valid range (including near-pole
    values that stress the arccos/arctan2 conditioning without hitting it
    exactly).
    """
    r = draw(st.one_of(st.just(0.0), _log_uniform_component().map(abs)))
    if system_type == "standard":
        el_edges = [0.0, np.pi]
        el = draw(
            st.one_of(
                st.sampled_from(el_edges),
                st.floats(
                    min_value=0.0,
                    max_value=np.pi,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        az = draw(
            st.floats(
                min_value=0.0,
                max_value=2 * np.pi,
                allow_nan=False,
                allow_infinity=False,
            )
        )
    else:
        el_edges = [-np.pi / 2, np.pi / 2]
        el = draw(
            st.one_of(
                st.sampled_from(el_edges),
                st.floats(
                    min_value=-np.pi / 2,
                    max_value=np.pi / 2,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        az = draw(
            st.floats(
                min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False
            )
        )
    return r, az, el


def _lat_strategy() -> st.SearchStrategy:
    return st.one_of(
        st.sampled_from([np.pi / 2, -np.pi / 2, 0.0]),
        st.floats(
            min_value=-np.pi / 2,
            max_value=np.pi / 2,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _lon_strategy() -> st.SearchStrategy:
    return st.one_of(
        st.sampled_from([np.pi, -np.pi, 0.0]),
        st.floats(
            min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False
        ),
    )


def _alt_strategy() -> st.SearchStrategy:
    # -5 km (below the Dead Sea) to 4e7 m (past geostationary altitude),
    # matching the range already used in tests/validation/test_coordinate_audit.py
    # so the two suites stress the same envelope.
    return st.one_of(
        st.sampled_from([-5000.0, 0.0, 4e7]),
        st.floats(
            min_value=-5000.0, max_value=4e7, allow_nan=False, allow_infinity=False
        ),
    )


# ---------------------------------------------------------------------------
# cart2sphere / sphere2cart
# ---------------------------------------------------------------------------


class TestCartSphereRoundTrip:
    """cart -> sphere -> cart always recovers the point.

    This direction has no singularity to worry about: whatever azimuth
    ``cart2sphere`` reports at a pole (r*sin(el) == 0, so az is a free
    parameter that doesn't affect x/y), ``sphere2cart`` multiplies it right
    back by that same zero factor. The *point* is always well defined even
    though az may not be "the" azimuth of anything.

    Tolerance: r = sqrt(x^2+y^2+z^2) rounds the sum of squares to the nearest
    representable double before taking the square root. When one component is
    smaller than roughly sqrt(eps) ~= 1.5e-8 times another (e.g. x=1e-6 next
    to z=1e7, a 13-decade ratio -- squarely inside the brief's requested
    range), x^2 vanishes below the ULP of z^2 and is rounded away *before*
    any trig runs: this is lost at the addition inside ``r``, not by
    ``sphere2cart``, and no algorithm computing r this way can avoid it in
    float64. So the recoverable precision on any one component is bounded by
    about ``sqrt(eps) * r``, not ``eps * r``; atol is set accordingly, with
    rtol carrying the ordinary (non-degenerate) case where roundoff is a few
    ULP of each component itself.
    """

    @given(cartesian_points(), st.sampled_from(SYSTEM_TYPES))
    def test_point_roundtrips(self, p, system_type):
        r, az, el = cart2sphere(p, system_type)
        back = sphere2cart(r, az, el, system_type)
        note(f"p={p} system={system_type} -> r={r} az={az} el={el} back={back}")
        atol = max(r, 1.0) * 1e-7  # ~6.7x the sqrt(eps)*r bound derived above
        np.testing.assert_allclose(back, p, rtol=1e-9, atol=atol)

    @given(cartesian_points(), st.sampled_from(SYSTEM_TYPES))
    def test_range_matches_norm(self, p, system_type):
        r, _az, _el = cart2sphere(p, system_type)
        assert r == pytest.approx(np.linalg.norm(p), rel=1e-12, abs=1e-12)


class TestSphereCartRoundTrip:
    """sphere -> cart -> sphere: r and el always recover; az does not at the
    exact standard-system north pole (a true, exact singularity -- see the
    module docstring). Elsewhere, including at the other, "accidentally"
    representable poles, az recovers too.
    """

    @given(
        st.sampled_from(SYSTEM_TYPES).flatmap(
            lambda s: spherical_triples(s).map(lambda t: (s, t))
        )
    )
    def test_range_roundtrips(self, system_and_triple):
        system_type, (r, az, el) = system_and_triple
        cart = sphere2cart(r, az, el, system_type)
        r2, _az2, _el2 = cart2sphere(cart, system_type)
        note(f"system={system_type} r={r} az={az} el={el} -> r2={r2}")
        np.testing.assert_allclose(r2, r, rtol=1e-9, atol=1e-9)

    @given(
        st.sampled_from(SYSTEM_TYPES).flatmap(
            lambda s: spherical_triples(s).map(lambda t: (s, t))
        )
    )
    def test_elevation_roundtrips(self, system_and_triple):
        system_type, (r, az, el) = system_and_triple
        assume(
            r > 0
        )  # at r == 0 every (az, el) maps to the origin; el is not recoverable
        cart = sphere2cart(r, az, el, system_type)
        _r2, _az2, el2 = cart2sphere(cart, system_type)
        note(f"system={system_type} r={r} az={az} el={el} -> el2={el2}")
        # Away from a pole, el recovers to a few ULP. Right at/near a pole,
        # arccos (standard) or the effective arcsin (az-el, via atan2 with a
        # near-zero xy_range) is ill-conditioned: d(arccos(u))/du diverges
        # like 1/sqrt(1-u^2), so an eps-sized error in z/r is amplified by
        # roughly 1/sin(el) (standard) or 1/cos(el) (az-el) near the pole.
        # atol = 1e-6 upper-bounds that amplification for any el in range
        # (worst case near the pole itself, where the amplified term is
        # capped by the fact sin/cos themselves are bounded by 1).
        np.testing.assert_allclose(el2, el, atol=1e-6)

    @given(spherical_triples("standard"))
    def test_standard_azimuth_roundtrips_except_at_exact_north_pole(self, triple):
        r, az, el = triple
        assume(
            r > 0
        )  # at r == 0 every (az, el) maps to the origin; az is not recoverable
        cart = sphere2cart(r, az, el, "standard")
        _r2, az2, _el2 = cart2sphere(cart, "standard")
        if el == 0.0:
            # True, exact singularity: np.sin(0.0) == 0.0 bit for bit, so
            # x == y == 0.0 exactly -- the *magnitude* of az is destroyed.
            # But it is not fully destroyed: IEEE 754 multiplication's sign
            # rule (sign(a*b) = sign(a) xor sign(b)) means
            # `0.0 * cos(az)` and `0.0 * sin(az)` are *signed* zeros whose
            # signs still encode sign(cos(az)) and sign(sin(az)). atan2 of a
            # pair of signed zeros is defined by IEEE 754 to return exactly
            # 0.0 when the real part's sign is non-negative, and exactly
            # +/-pi when it is negative -- so one bit of az (which half of
            # the circle cos(az) was on) survives.
            #
            # Counterexample that falsified the naive "always reports 0.0"
            # claim: (r=1, az=2.0, el=0.0) -> az2 == pi, not 0.0 (cos(2.0) <
            # 0). Verified directly against IEEE 754 semantics, not a pytcl
            # bug -- see task-2-report.md.
            if np.cos(az) >= 0:
                assert az2 == 0.0
            else:
                assert abs(az2) == np.pi
        elif r * abs(np.sin(el)) < 1e-300:
            # Second, distinct near-pole regime, found by Hypothesis at
            # (r=1, az=1.0, el=5e-324 -- the smallest positive subnormal
            # double): el != 0.0 bit-for-bit, so this isn't the exact
            # signed-zero case above, but x = r*sin(el)*cos(az) and
            # y = r*sin(el)*sin(az) both land deep in float64's subnormal
            # range, which has only ~4.94e-324 (2**-1074) of spacing between
            # representable values. x and y round to the *same* one or two
            # subnormal steps regardless of az, destroying the x/y ratio
            # (and hence az) even though el is nonzero: the counterexample
            # recovers az2 == pi/4 for a true az of 1.0. This is gradual,
            # not a hard cutoff -- empirically (see task-2-report.md) the
            # recovered-az error stays under 1e-6 down to about
            # r*sin(el) ~ 1e-317 and only then grows past it, so the 1e-300
            # guard here sits ~17 orders of magnitude inside the safe zone.
            # Not a pytcl bug: any float64 implementation computing
            # r*sin(el)*cos(az) this way hits the same wall. No further
            # assertion applies here -- r and el already round-trip fine in
            # this regime (checked by test_range_roundtrips /
            # test_elevation_roundtrips, whose error at these el is bounded
            # by el itself, far under their atol).
            note(f"deep-subnormal regime: r={r} az={az} el={el} -> az2={az2}")
        else:
            # Every other el (including south pole el == pi, where sin(pi)
            # ~= 1.22e-16 leaves an az-proportional residual -- see module
            # docstring) numerically recovers az. Near-pole el amplifies
            # roundoff in the same way elevation does (see
            # test_elevation_roundtrips); atol=1e-6 covers that.
            note(f"r={r} az={az} el={el} -> az2={az2}")
            assert (
                abs(np.remainder(az2 - az, 2 * np.pi)) < 1e-6
                or abs(np.remainder(az2 - az, 2 * np.pi) - 2 * np.pi) < 1e-6
            )

    def test_north_pole_azimuth_signed_zero_regression(self):
        """Pinned counterexample from the property above (r=1, az=2.0,
        el=0.0): recovered azimuth at the exact standard-system north pole
        is pi, not 0.0, because cos(2.0) < 0 and 0.0 * cos(2.0) is a
        *negative* signed zero -- atan2(+0.0, -0.0) == pi by IEEE 754. A
        naive fix that always expects 0.0 at the pole would be wrong.
        """
        cart = sphere2cart(1.0, 2.0, 0.0, "standard")
        _r2, az2, _el2 = cart2sphere(cart, "standard")
        assert az2 == np.pi

    def test_near_pole_subnormal_azimuth_regression(self):
        """Pinned counterexample from the property above (r=1, az=1.0,
        el=5e-324, the smallest positive subnormal double): recovered
        azimuth is pi/4, unrelated to the true az=1.0, because x and y both
        underflow to the same 4.94e-324 subnormal step. el is nonzero here
        -- this is a different failure mode than the exact-zero regression
        above, not a duplicate of it.
        """
        cart = sphere2cart(1.0, 1.0, 5e-324, "standard")
        _r2, az2, _el2 = cart2sphere(cart, "standard")
        assert az2 == pytest.approx(np.pi / 4)

    @given(st.sampled_from(["az-el", "range-az-el"]), spherical_triples("az-el"))
    def test_azel_azimuth_roundtrips_everywhere(self, system_type, triple):
        r, az, el = triple
        assume(
            r > 0
        )  # at r == 0 every (az, el) maps to the origin; az is not recoverable
        cart = sphere2cart(r, az, el, system_type)
        _r2, az2, _el2 = cart2sphere(cart, system_type)
        note(f"system={system_type} r={r} az={az} el={el} -> az2={az2}")
        # No r*abs(cos(el)) < 1e-300 deep-subnormal guard here, unlike the
        # standard system's r*sin(el) guard above -- and deliberately not
        # kept "for symmetry," because it would be unreachable dead code
        # that implies a risk this system can't have: the standard system's
        # collapse is possible because el == 0.0 is *exactly* representable,
        # so el (and sin(el)) can be driven all the way down to the smallest
        # positive subnormal (5e-324). az-el's pole is at +/-pi/2, which is
        # NOT exactly representable -- the closest a float64 el can ever get
        # to the true real pi/2 is bounded below by float64's own spacing
        # there (~1.11e-16 relative), so cos(el) can never go below
        # np.cos(np.pi/2) ~= 6.12e-17 for any el this generator (or any
        # float64 value) can produce. There is no reachable "el nonzero but
        # deep in subnormal range" regime to guard against here.
        #
        # Away from that (nonexistent) regime, neither az-el pole
        # (el == +/-pi/2) is an exact floating-point zero: that same
        # ~6.12e-17 floor leaves a tiny but az-proportional residual in
        # x/y, so az is numerically recoverable everywhere in this system,
        # including exactly at the sampled pole edge value -- see module
        # docstring.
        diff = abs(np.remainder(az2 - az, 2 * np.pi))
        assert min(diff, 2 * np.pi - diff) < 1e-6


# ---------------------------------------------------------------------------
# geodetic2ecef / ecef2geodetic
# ---------------------------------------------------------------------------


class TestGeodeticRoundTrip:
    """geodetic -> ECEF -> geodetic, across both ecef2geodetic methods.

    Longitude is mathematically undefined at the poles (every meridian meets
    there), but -- exactly parallel to the az-el spherical pole above -- this
    implementation numerically recovers it anyway: at lat = +/-pi/2,
    cos(lat) is not exactly 0.0 (pi/2 is not exactly representable), so the
    ECEF x/y components carry a tiny but lon-proportional residual instead of
    collapsing to a true (0, 0). That is asserted explicitly below, not
    assumed away.

    Tolerance: lat/lon atol=1e-9 rad and alt atol=1e-6 m for "direct"
    (Vermeille's closed-form solution) matches the precision already
    established for these functions in
    tests/validation/test_coordinate_audit.py::test_ecef2geodetic_edge_latitudes.
    "iterative" (Bowring, 5 fixed-point iterations) gets a looser alt
    atol=1e-2 m, matching the tol_m used for that method in the same file's
    pyproj cross-check -- 5 iterations of Bowring's method converges to
    sub-cm but isn't a closed form, so it doesn't hit the same floor.

    lon at the exact pole uses the *same* 1e-9 atol as everywhere else, not
    a looser one -- measured, not assumed. A first draft of this test gave
    the pole case a 1e-4 atol on the theory that x/y there (~1e-17 x
    (N+alt)) were "near the edge of what N/alt additions preserve," but
    directly sweeping both ecef2geodetic methods over the full lon/alt
    generator range at lat = +/-pi/2 (and at the float64 values nearest
    pi/2 short of it, i.e. what this generator can actually produce) found
    a worst-case lon error of 2.22e-16 -- 2 ULP, indistinguishable from the
    away-from-pole case, not the 11-order-of-magnitude-looser regression a
    1e-4 atol would have silently let through. The theorized "edge of what
    N/alt can preserve" mechanism doesn't actually bind: cos(lat) at the
    pole is a fixed ~6.12e-17 (not itself degrading further with N/alt), so
    it just scales x/y down uniformly rather than eating precision.
    """

    @given(
        lat=_lat_strategy(),
        lon=_lon_strategy(),
        alt=_alt_strategy(),
        method=st.sampled_from(GEODETIC_METHODS),
    )
    def test_lat_lon_alt_roundtrip(self, lat, lon, alt, method):
        ecef = geodetic2ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef2geodetic(ecef, method=method)
        note(
            f"lat={lat} lon={lon} alt={alt} method={method} "
            f"-> lat2={lat2} lon2={lon2} alt2={alt2}"
        )
        alt_atol = 1e-2 if method == "iterative" else 1e-6
        assert abs(lat2 - lat) < 1e-9
        assert abs(alt2 - alt) < alt_atol

        # lon round-trips to the same 1e-9 atol everywhere, pole included --
        # see the class docstring for the measurement that replaced an
        # earlier, unjustifiably loose pole-only tolerance here.
        diff = abs(np.remainder(lon2 - lon, 2 * np.pi))
        assert min(diff, 2 * np.pi - diff) < 1e-9

    @given(
        lat=_lat_strategy(),
        lon=_lon_strategy(),
        alt=_alt_strategy(),
    )
    def test_ecef_point_roundtrips_iterative(self, lat, lon, alt):
        """The ECEF *point* (not the angles) round-trips through the default
        method regardless of pole/antimeridian conditioning -- this is the
        property most consumers actually rely on (e.g. sensor fusion working
        in ECEF)."""
        ecef = geodetic2ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef2geodetic(ecef)
        ecef2 = geodetic2ecef(lat2, lon2, alt2)
        note(f"lat={lat} lon={lon} alt={alt} ecef={ecef} ecef2={ecef2}")
        np.testing.assert_allclose(ecef2, ecef, rtol=1e-9, atol=1e-6)
