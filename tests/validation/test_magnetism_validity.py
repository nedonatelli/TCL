"""Validity-window warnings and cache-content-keying for WMM/EMM (v2.11.1 2.4/2.5).

Task 2.4: ``wmm()`` and ``emm()`` extrapolated the linear secular-variation
terms arbitrarily far outside a coefficient set's declared validity window
with no warning at all. ``EMM_PARAMETERS[model]["valid_start"/"valid_end"]``
were declared and never read; WMM carries no explicit window, but its
five-year validity (epoch to epoch + 5.0) is documented in every
``create_wmmYYYY_coefficients`` docstring. Measured before the fix:
``wmm(40N, 105W, 2000.0)`` (using the default WMM2025 coefficients, whose
window is 2025.0-2030.0) returned D=9.651 deg, F=54426.3 nT against IGRF-14's
own 2000.0-epoch values of D=10.431 deg, F=54147.7 nT -- 0.78 deg and 279 nT
off, with zero warnings. This suite pins that same wrong-but-now-warned value
(the fix adds a warning, it does not change what gets returned) and checks
both directions of the window and both models.

Task 2.5: ``magnetic_field_spherical``'s LRU cache keyed its entries on
``id(coeffs)``. Mutating a registered coefficient array in place left the
id unchanged, so the cache kept serving the field computed from the
pre-mutation contents. The backing ``_coefficient_registry`` also held
strong references forever, so every throwaway coefficient set that ever
passed through the cache stayed alive for the life of the process. This
suite reproduces the stale-field defect (a mutation must change the
result) and the retention defect (an unreferenced coefficient set must be
collectible), and checks that legitimate cache reuse still works.
"""

import gc
import warnings

import numpy as np
import pytest

from pytcl.magnetism.coordinates import geog_heading2mag, mag_heading2geog
from pytcl.magnetism.emm import EMM_PARAMETERS, create_test_coefficients, emm
from pytcl.magnetism.igrf import IGRF14
from pytcl.magnetism.wmm import (
    WMM2020,
    WMM2025,
    MagneticCoefficients,
    clear_magnetic_cache,
    create_wmm2025_coefficients,
    get_magnetic_cache_info,
    magnetic_field_spherical,
    wmm,
)

DENVER_LAT = np.radians(40.0)
DENVER_LON = np.radians(-105.0)


# =============================================================================
# Task 2.4: WMM validity window
# =============================================================================


class TestWMMValidityWindow:
    def test_warns_before_its_validity_window(self):
        """WMM2025 is valid from 2025.0; 2000.0 is 25 years early."""
        with pytest.warns(UserWarning, match="valid.*2025"):
            wmm(DENVER_LAT, DENVER_LON, 0.0, 2000.0)

    def test_warns_beyond_its_validity_window(self):
        """WMM2025 is valid through 2030.0; 2100.0 is 70 years late."""
        with pytest.warns(UserWarning, match="valid.*2030"):
            wmm(DENVER_LAT, DENVER_LON, 0.0, 2100.0)

    def test_inside_the_window_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wmm(DENVER_LAT, DENVER_LON, 0.0, 2026.7)

    def test_at_the_epoch_boundary_is_silent(self):
        """year == epoch is the start of the window, not before it."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wmm(DENVER_LAT, DENVER_LON, 0.0, WMM2025.epoch)

    def test_extrapolation_past_2000_matches_the_measured_defect(self):
        """The fix adds a warning; it must not change the extrapolated value.

        Pins the exact figures from the v2.11.1 audit: D=9.651 deg,
        F=54426.3 nT, 0.78 deg / 279 nT off IGRF-14's true 2000.0 field.
        A regression here means the fix altered wmm()'s output, which the
        patch-release constraints forbid.
        """
        with pytest.warns(UserWarning):
            result = wmm(DENVER_LAT, DENVER_LON, 0.0, 2000.0)
        assert np.degrees(result.D) == pytest.approx(9.651, abs=1e-2)
        assert result.F == pytest.approx(54426.3, abs=1.0)

    def test_older_release_is_valid_over_its_own_window(self):
        """WMM2020 (epoch 2020.0) treats 2000.0 as before its own window."""
        with pytest.warns(UserWarning, match="valid.*2020"):
            wmm(DENVER_LAT, DENVER_LON, 0.0, 2000.0, WMM2020)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wmm(DENVER_LAT, DENVER_LON, 0.0, 2023.0, WMM2020)


# =============================================================================
# Task 2.4: EMM / WMMHR validity window
# =============================================================================


class TestEMMValidityWindow:
    """EMM_PARAMETERS is nested per model: EMM2017 valid 2000-2022 (n_max
    790, epoch 2017.0); WMMHR2025 valid 2025-2030 (n_max 133, epoch 2025.0).
    Uses synthetic in-memory coefficients throughout (``create_test_coefficients``)
    so these run without the real (multi-MB, not vendored) .COF files -- the
    window check reads ``EMM_PARAMETERS[model]``, independent of which
    coefficients object was actually supplied.
    """

    def test_reads_its_declared_validity_parameters(self):
        """valid_start/valid_end were declared and never consulted."""
        coef = create_test_coefficients(n_max=36)
        valid_end = EMM_PARAMETERS["EMM2017"]["valid_end"]
        with pytest.warns(UserWarning, match="valid.*2022"):
            emm(DENVER_LAT, DENVER_LON, 0.0, valid_end + 5.0, coefficients=coef)

    def test_warns_before_its_validity_window(self):
        coef = create_test_coefficients(n_max=36)
        valid_start = EMM_PARAMETERS["EMM2017"]["valid_start"]
        with pytest.warns(UserWarning, match="valid.*2000"):
            emm(DENVER_LAT, DENVER_LON, 0.0, valid_start - 5.0, coefficients=coef)

    def test_inside_its_window_is_silent(self):
        coef = create_test_coefficients(n_max=36)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emm(DENVER_LAT, DENVER_LON, 0.0, 2020.0, coefficients=coef)

    def test_at_its_window_boundaries_is_silent(self):
        """year == valid_start and year == valid_end are inside the
        (inclusive) window, not outside it -- the WMM analogue of this
        (`TestWMMValidityWindow.test_at_the_epoch_boundary_is_silent`) only
        checks one boundary since WMM's window is derived (epoch is always
        the start); EMM's two bounds are independently declared, so both
        need their own check against an off-by-one in either comparison.
        """
        coef = create_test_coefficients(n_max=36)
        valid_start = EMM_PARAMETERS["EMM2017"]["valid_start"]
        valid_end = EMM_PARAMETERS["EMM2017"]["valid_end"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emm(DENVER_LAT, DENVER_LON, 0.0, valid_start, coefficients=coef)
            emm(DENVER_LAT, DENVER_LON, 0.0, valid_end, coefficients=coef)

    def test_extrapolation_past_2200_matches_the_measured_defect(self):
        """EMM2017 at 2200.0: a 180-year linear extrapolation, zero warnings
        before the fix. Value is unpinned (it depends on the synthetic test
        coefficients, not the real EMM2017 table) but must now warn.
        """
        coef = create_test_coefficients(n_max=36)
        with pytest.warns(UserWarning, match="valid.*2022"):
            result = emm(DENVER_LAT, DENVER_LON, 0.0, 2200.0, coefficients=coef)
        assert np.isfinite(result.F)

    def test_wmmhr2025_uses_its_own_window_not_emm2017s(self):
        """The two models' windows do not collide: WMMHR2025 is 2025-2030,
        EMM2017 is 2000-2022. A year inside one and outside the other must
        warn only under the model it is actually outside of.
        """
        coef = create_test_coefficients(n_max=36)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emm(
                DENVER_LAT,
                DENVER_LON,
                0.0,
                2026.0,
                model="WMMHR2025",
                coefficients=coef,
            )
        with pytest.warns(UserWarning, match="valid.*2025"):
            emm(
                DENVER_LAT,
                DENVER_LON,
                0.0,
                2020.0,
                model="WMMHR2025",
                coefficients=coef,
            )


# =============================================================================
# Task 2.5: cache keyed on coefficient content, not identity
# =============================================================================


class TestMagneticFieldCache:
    def setup_method(self):
        clear_magnetic_cache()

    def test_cache_does_not_serve_stale_fields_after_in_place_mutation(self):
        coeffs = create_wmm2025_coefficients()
        args = (DENVER_LAT, DENVER_LON, 6371.2, 2026.7)
        before = magnetic_field_spherical(*args, coeffs=coeffs)[0]
        coeffs.g[2, 0] *= 2.0
        after = magnetic_field_spherical(*args, coeffs=coeffs)[0]
        assert after != pytest.approx(before), "cache served a stale field"

    def test_mutated_field_matches_an_uncached_computation(self):
        """Not just "different" -- the post-mutation cached value must be
        close to what a fresh, uncached computation on the mutated
        coefficients gives (not bit-exact: the cache quantizes lat/lon/r/year
        to a configurable precision before keying, by design -- see
        `_quantize_inputs` -- so a small, expected rounding difference
        remains between the two paths).
        """
        coeffs = create_wmm2025_coefficients()
        args = (DENVER_LAT, DENVER_LON, 6371.2, 2026.7)
        magnetic_field_spherical(*args, coeffs=coeffs)  # populate the cache
        coeffs.g[2, 0] *= 2.0
        cached = magnetic_field_spherical(*args, coeffs=coeffs, use_cache=True)
        uncached = magnetic_field_spherical(*args, coeffs=coeffs, use_cache=False)
        assert cached == pytest.approx(uncached, rel=1e-4)

    def test_unmutated_coefficients_still_hit_the_cache(self):
        """The fix must not degrade into "never cache" -- repeated calls on
        the same, unmutated coefficients should still be served from cache.
        """
        coeffs = create_wmm2025_coefficients()
        args = (DENVER_LAT, DENVER_LON, 6371.2, 2026.7)
        magnetic_field_spherical(*args, coeffs=coeffs)
        before_hits = get_magnetic_cache_info()["hits"]
        magnetic_field_spherical(*args, coeffs=coeffs)
        after_hits = get_magnetic_cache_info()["hits"]
        assert after_hits == before_hits + 1

    def test_registry_does_not_retain_unreferenced_models(self):
        """500 throwaway models leaving 500 live objects was the measured
        defect; using 50 here since that is all a property test needs to
        show the growth is not O(calls).
        """
        baseline = sum(
            1 for o in gc.get_objects() if isinstance(o, MagneticCoefficients)
        )
        args = (DENVER_LAT, DENVER_LON, 6371.2, 2026.7)
        for _ in range(50):
            coeffs = create_wmm2025_coefficients()
            magnetic_field_spherical(*args, coeffs=coeffs)
            del coeffs
        gc.collect()
        after = sum(1 for o in gc.get_objects() if isinstance(o, MagneticCoefficients))
        assert after - baseline < 5


# =============================================================================
# Task 2.4 follow-up: coordinates.py must not measure IGRF coefficients
# against WMM's validity window
# =============================================================================


class TestCoordinatesAcceptIGRFCoefficientsWithoutFalseWarning:
    """`geog_heading2mag`/`mag_heading2geog` (and their siblings in
    coordinates.py) accept any `MagneticCoefficients`, IGRF14 included --
    most of the module's other functions default to it. Their shared
    `_declination_at` helper originally called the public, WMM-window-
    checked `wmm()`, so passing `IGRF14` through it at a year outside
    WMM2025's window (but squarely inside IGRF's) produced a false-positive
    "before WMM2025's valid window ... prefer an older WMM release or
    IGRF" warning -- wrong on every count: these are already IGRF
    coefficients. Reproduces the exact call the review flagged.

    `_declination_at` now always calls the unwarned `_wmm_core`, the same
    fix applied to `igrf.py` and for the same reason: it is generic
    infrastructure that accepts either family of coefficients, and there
    is no reliable way to tell which family a given `MagneticCoefficients`
    came from (see the report's discussion of why a generic provenance tag
    was rejected). One consequence: WMM coefficients routed through this
    same path also stop warning here -- callers who want the WMM window
    warning get it by calling `wmm()` directly, which is untouched by this
    fix and still warns (`TestWMMValidityWindow`).
    """

    def test_geog_heading2mag_with_igrf_coefficients_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            geog_heading2mag([DENVER_LAT, DENVER_LON, 0.0], 0.5, IGRF14, year=1990.0)

    def test_mag_heading2geog_with_igrf_coefficients_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mag_heading2geog([DENVER_LAT, DENVER_LON, 0.0], 0.5, IGRF14, year=1990.0)
