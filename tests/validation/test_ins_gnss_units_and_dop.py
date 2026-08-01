"""INS/GNSS error-state units, and the frame DOP is reported in (gh-19).

Two defects that a converging filter hides.

**Units.** The first three error states are ``[dlat, dlon, dheight]`` in
``[rad, rad, m]``, but ``initialize_ins_gnss`` put a meters-valued
``position_std`` straight onto all three diagonal entries, and the default
measurement covariance was in m^2 against innovations in radians.

The consequence is subtler than "the gains are wrong by a factor of two". With
the shipped defaults the two errors cancelled: covariance and measurement noise
were *both* wrongly in meters, so the ratio came out right and the filter
absorbed a sensible half of each innovation. The damage appears the moment a
caller supplies a correctly scaled ``position_cov`` -- then the filter's own
covariance is larger by the square of an Earth radius, roughly 1e13, and it
absorbs essentially 100% of every measurement. The INS contributes nothing and
the filter is a pass-through for GNSS, while still looking like it is fusing.

**DOP frame.** HDOP and VDOP are read off the first two and third diagonal
entries of ``inv(H^T H)``, so they are horizontal and vertical only if the
position columns of ``H`` span the local horizontal plane. ``tight_coupled_update``
passed an ECEF geometry matrix, whose x and y axes point at the equator no
matter where the user is. At mid-latitudes the reported HDOP and VDOP were
close to each other's true values. GDOP and PDOP are traces, hence
rotation-invariant, and were right all along.

The oracle for the units is the geodetic relation itself -- a northward meter
is ``1/(R_N + h)`` radians of latitude -- and for DOP it is the same
constellation built directly in ENU, where the horizontal/vertical split is
true by construction.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman import kf_update
from pytcl.navigation.ins import initialize_ins_state, radii_of_curvature
from pytcl.navigation.ins_gnss import (
    GNSSMeasurement,
    SatelliteInfo,
    compute_dop,
    initialize_ins_gnss,
    loose_coupled_update_position,
    position_measurement_matrix,
    position_std_to_error_state_units,
    tight_coupled_update,
)

LAT = np.radians(45.0)
LON = np.radians(-75.0)
ALT = 100.0


class TestPositionStdConversion:
    """A meters-valued uncertainty, expressed in the states the filter uses."""

    @pytest.mark.parametrize(
        "lat_deg", [-80.0, -45.0, -10.0, 0.0, 10.0, 45.0, 60.0, 80.0]
    )
    def test_latitude_component_is_the_meridional_relation(self, lat_deg):
        """``dlat = dnorth / (R_N + h)``, the definition of meridian curvature."""
        lat = np.radians(lat_deg)
        meridian_radius, _ = radii_of_curvature(lat)

        std = position_std_to_error_state_units(10.0, lat, ALT)
        assert std[0] == pytest.approx(10.0 / (meridian_radius + ALT), rel=1e-12)

    @pytest.mark.parametrize("lat_deg", [-80.0, -45.0, 0.0, 45.0, 80.0])
    def test_longitude_component_is_the_prime_vertical_relation(self, lat_deg):
        """``dlon = deast / ((R_E + h) cos(lat))``.

        The ``cos(lat)`` is what makes this different from the latitude case,
        and dropping it is the error that would look plausible at the equator
        and be 40% wrong at 45 degrees.
        """
        lat = np.radians(lat_deg)
        _, transverse_radius = radii_of_curvature(lat)

        std = position_std_to_error_state_units(10.0, lat, ALT)
        expected = 10.0 / ((transverse_radius + ALT) * np.cos(lat))
        assert std[1] == pytest.approx(expected, rel=1e-12)

    def test_height_is_left_in_meters(self):
        """The third state is already meters; converting it would be the bug."""
        assert position_std_to_error_state_units(10.0, LAT, ALT)[2] == 10.0

    def test_longitude_uncertainty_grows_toward_the_poles(self):
        """A meter of easting spans more longitude the further north you are."""
        at = [
            position_std_to_error_state_units(10.0, np.radians(d), 0.0)[1]
            for d in (0.0, 30.0, 60.0, 85.0)
        ]
        assert all(later > earlier for earlier, later in zip(at, at[1:])), at

    def test_latitude_uncertainty_barely_changes_with_latitude(self):
        """The meridian radius spans only ``(1 - e^2)^(-3/2)``, about 1.01.

        This is the check that the two components are not accidentally the same
        expression: longitude swings by an order of magnitude over the same
        span while latitude moves by one percent.
        """
        at = [
            position_std_to_error_state_units(10.0, np.radians(d), 0.0)[0]
            for d in (0.0, 30.0, 60.0, 85.0)
        ]
        assert 1.0 < max(at) / min(at) < 1.02, at

        longitude = [
            position_std_to_error_state_units(10.0, np.radians(d), 0.0)[1]
            for d in (0.0, 30.0, 60.0, 85.0)
        ]
        assert max(longitude) / min(longitude) > 10.0, longitude

    def test_the_conversion_stays_finite_at_the_pole(self):
        """``cos(lat)`` reaches zero there, so the relation genuinely diverges.

        Returning ``inf`` would poison the covariance and every subsequent
        update; the cosine is floored instead.
        """
        std = position_std_to_error_state_units(10.0, np.radians(90.0), 0.0)
        assert np.all(np.isfinite(std))
        assert std[1] > std[0]

    def test_zero_uncertainty_converts_to_zero(self):
        assert np.all(position_std_to_error_state_units(0.0, LAT, ALT) == 0.0)


class TestFilterWeighting:
    """What the units are for: the gain the filter actually applies."""

    @staticmethod
    def _absorbed_fraction(position_cov):
        """Fraction of a 10 m north offset the filter takes from the fix."""
        ins = initialize_ins_state(lat=LAT, lon=LON, alt=ALT)
        state = initialize_ins_gnss(ins, position_std=10.0)

        meridian_radius, _ = radii_of_curvature(LAT)
        offset = 10.0 / (meridian_radius + ALT)
        gnss = GNSSMeasurement(
            position=np.array([LAT + offset, LON, ALT]),
            velocity=None,
            position_cov=position_cov,
            velocity_cov=None,
            time=0.0,
        )
        result = loose_coupled_update_position(state, gnss)
        return (result.state.ins_state.position[0] - LAT) / offset

    def test_a_correctly_scaled_measurement_gives_the_textbook_gain(self):
        """P = 10 m, R = 2 m, so the filter should take 100/(100+4) of it.

        This is the case the unit mismatch destroyed. With meters-squared on
        the radian diagonal the filter's own covariance was larger by about
        1e13 and it took the whole innovation every time, regardless of how
        good or bad the fix claimed to be.
        """
        two_meters = position_std_to_error_state_units(2.0, LAT, ALT)
        cov = np.diag([two_meters[0] ** 2, two_meters[1] ** 2, 2.0**2])

        assert self._absorbed_fraction(cov) == pytest.approx(100 / 104, abs=1e-3)

    def test_a_worse_fix_moves_the_filter_less(self):
        """Monotonicity: the gain has to respond to the measurement noise.

        Under the old units it did not -- every fix, from millimeter to
        kilometer, produced the same near-unit gain.
        """
        fractions = []
        for meters in (1.0, 5.0, 20.0, 100.0):
            std = position_std_to_error_state_units(meters, LAT, ALT)
            cov = np.diag([std[0] ** 2, std[1] ** 2, meters**2])
            fractions.append(self._absorbed_fraction(cov))

        assert all(b < a for a, b in zip(fractions, fractions[1:])), fractions
        assert fractions[0] > 0.98, "a 1 m fix against a 10 m filter is nearly trusted"
        assert fractions[-1] < 0.02, "a 100 m fix should barely move a 10 m filter"

    def test_the_initial_covariance_is_in_error_state_units(self):
        ins = initialize_ins_state(lat=LAT, lon=LON, alt=ALT)
        state = initialize_ins_gnss(ins, position_std=10.0)

        expected = position_std_to_error_state_units(10.0, LAT, ALT)
        np.testing.assert_allclose(
            np.sqrt(np.diag(state.error_cov)[:3]), expected, rtol=1e-12
        )

    def test_the_gain_matches_a_hand_computed_kalman_update(self):
        """Independent of the INS/GNSS plumbing entirely.

        Builds the same update by hand from P, H and R, so a change in how the
        state is assembled cannot make this agree for the wrong reason.
        """
        ins = initialize_ins_state(lat=LAT, lon=LON, alt=ALT)
        state = initialize_ins_gnss(ins, position_std=10.0)

        two_meters = position_std_to_error_state_units(2.0, LAT, ALT)
        R = np.diag([two_meters[0] ** 2, two_meters[1] ** 2, 2.0**2])
        meridian_radius, _ = radii_of_curvature(LAT)
        offset = 10.0 / (meridian_radius + ALT)

        by_hand = kf_update(
            np.zeros(15),
            state.error_cov,
            np.array([offset, 0.0, 0.0]),
            position_measurement_matrix(),
            R,
        )
        assert by_hand.x[0] / offset == pytest.approx(
            self._absorbed_fraction(R), rel=1e-9
        )


class TestDopFrame:
    """HDOP and VDOP mean nothing without a local frame."""

    # Six satellites at assorted elevations and azimuths, as seen from the
    # user. Enough for a well-conditioned solution; a symmetric four-satellite
    # set at one elevation is singular.
    SATELLITES_EL_AZ_DEG = [
        (70, 30),
        (45, 120),
        (30, 210),
        (55, 300),
        (20, 60),
        (80, 180),
    ]

    @classmethod
    def _geometry(cls):
        """The same constellation in ENU and in ECEF.

        ENU is the reference: there, horizontal and vertical are true by
        construction, so its HDOP and VDOP are the values the ECEF path has to
        reproduce.
        """
        enu = np.array(
            [
                [
                    np.cos(np.radians(el)) * np.sin(np.radians(az)),
                    np.cos(np.radians(el)) * np.cos(np.radians(az)),
                    np.sin(np.radians(el)),
                ]
                for el, az in cls.SATELLITES_EL_AZ_DEG
            ]
        )
        sin_lat, cos_lat = np.sin(LAT), np.cos(LAT)
        sin_lon, cos_lon = np.sin(LON), np.cos(LON)
        ecef_to_enu = np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ]
        )
        ones = np.ones((len(cls.SATELLITES_EL_AZ_DEG), 1))
        return (
            np.hstack([-enu, ones]),
            np.hstack([-(enu @ ecef_to_enu), ones]),
        )

    def test_an_ecef_matrix_with_the_user_position_matches_the_enu_reference(self):
        """The fix: supplying ``user_lla`` rotates before splitting."""
        h_enu, h_ecef = self._geometry()

        reference = compute_dop(h_enu)
        rotated = compute_dop(h_ecef, user_lla=np.array([LAT, LON, ALT]))

        np.testing.assert_allclose(rotated, reference, rtol=1e-10)

    def test_an_ecef_matrix_without_the_user_position_is_wrong(self):
        """Guard the guard.

        If the ECEF and ENU splits happened to agree, the test above would pass
        for a rotation that does nothing. At 45 degrees they do not agree --
        the reported values are close to each other's truth.
        """
        h_enu, h_ecef = self._geometry()

        _, _, hdop_true, vdop_true = compute_dop(h_enu)
        _, _, hdop_ecef, vdop_ecef = compute_dop(h_ecef)

        assert abs(hdop_ecef - hdop_true) > 0.3 * hdop_true
        assert abs(vdop_ecef - vdop_true) > 0.2 * vdop_true

    def test_gdop_and_pdop_are_rotation_invariant(self):
        """They are traces, so the frame never mattered for them.

        Worth pinning: it is what makes this a reporting bug in two of the four
        numbers rather than in all of them.
        """
        h_enu, h_ecef = self._geometry()

        gdop_enu, pdop_enu, _, _ = compute_dop(h_enu)
        gdop_ecef, pdop_ecef, _, _ = compute_dop(h_ecef)

        assert gdop_ecef == pytest.approx(gdop_enu, rel=1e-10)
        assert pdop_ecef == pytest.approx(pdop_enu, rel=1e-10)

    def test_pdop_is_the_quadrature_sum_of_hdop_and_vdop(self):
        """A relation that only holds in a frame where the split is real.

        In ECEF without rotation it holds too, because the same three diagonal
        entries are being summed -- which is exactly why it cannot be the only
        check. Paired with the reference comparison above, it pins the split.
        """
        h_enu, h_ecef = self._geometry()
        for dop in (
            compute_dop(h_enu),
            compute_dop(h_ecef, user_lla=np.array([LAT, LON, ALT])),
        ):
            _, pdop, hdop, vdop = dop
            assert pdop**2 == pytest.approx(hdop**2 + vdop**2, rel=1e-10)

    def test_a_singular_geometry_reports_infinite_dop(self):
        """Too few satellites to solve; the rotation must not mask that."""
        h = np.hstack([-np.eye(3)[:2], np.ones((2, 1))])
        assert all(np.isinf(compute_dop(h, user_lla=np.array([LAT, LON, ALT]))))

    def test_tight_coupled_update_reports_dop_in_the_local_frame(self):
        """The call site, not just the function.

        Every other test here exercises ``compute_dop`` directly, so all of
        them keep passing if ``tight_coupled_update`` stops passing the user
        position -- which is the actual defect gh-19 describes. This builds a
        real constellation around a mid-latitude user and checks the DOP it
        reports against the ENU reference.
        """
        from pytcl.navigation.geodesy import geodetic_to_ecef

        user_ecef = np.array(geodetic_to_ecef(LAT, LON, ALT))
        sin_lat, cos_lat = np.sin(LAT), np.cos(LAT)
        sin_lon, cos_lon = np.sin(LON), np.cos(LON)
        enu_to_ecef = np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ]
        )

        satellites = []
        for prn, (el, az) in enumerate(self.SATELLITES_EL_AZ_DEG, start=1):
            direction_enu = np.array(
                [
                    np.cos(np.radians(el)) * np.sin(np.radians(az)),
                    np.cos(np.radians(el)) * np.cos(np.radians(az)),
                    np.sin(np.radians(el)),
                ]
            )
            sat_ecef = user_ecef + 2.0e7 * (enu_to_ecef @ direction_enu)
            satellites.append(
                SatelliteInfo(
                    prn=prn,
                    position=sat_ecef,
                    velocity=np.zeros(3),
                    pseudorange=float(np.linalg.norm(sat_ecef - user_ecef)),
                )
            )

        ins = initialize_ins_state(lat=LAT, lon=LON, alt=ALT)
        state = initialize_ins_gnss(ins)
        result = tight_coupled_update(state, satellites)

        h_enu, _ = self._geometry()
        _, _, hdop_reference, vdop_reference = compute_dop(h_enu)
        _, _, hdop_reported, vdop_reported = result.dop

        assert hdop_reported == pytest.approx(hdop_reference, rel=1e-6), (
            "tight_coupled_update reported an HDOP that is not horizontal in "
            "the user's local frame, so the ECEF geometry matrix is not being "
            "rotated (gh-19)"
        )
        assert vdop_reported == pytest.approx(vdop_reference, rel=1e-6)
