"""Great-circle oracle tests for pytcl.navigation.great_circle.

Task 3.3 (v2.11.1 correctness patch): ``great_circle_intersect`` returned
whichever antipodal solution ``n1 x n2`` happened to produce instead of the
documented "intersections closest to the given points" -- for az1 = 135 deg,
az2 = 225 deg it returned a point whose azimuth from point 1 is 315 deg, i.e.
backwards along az1. ``cross_track_distance``'s ``along_track`` used
``arccos``, which is unsigned, so a point 5 deg behind the start returned the
identical distance as a point 5 deg ahead.

Both are fixed by transcribing the positive-distance branch rule from
MATLAB's ``greatCircleIntersect.m`` (Baselga & Martinez-Llario 2018): the
correct intersection is the one reached by a positive distance along the
observed azimuth, not whichever branch the vector algebra happens to return.

The oracle here is ``geographiclib.geodesic.Geodesic`` instantiated with
flattening 0 -- i.e. Karney's algorithm specialized to an exact sphere of
the module's own ``EARTH_RADIUS``. That keeps the reference computation on
the same spherical model as the code under test (no ellipsoidal-vs-spherical
mismatch from the default WGS84 instance) while still being a completely
independent implementation from this module's normal-vector cross-product
approach.
"""

import numpy as np
import pytest

geographiclib = pytest.importorskip("geographiclib.geodesic")

from pytcl.navigation.great_circle import (  # noqa: E402
    EARTH_RADIUS,
    cross_track_distance,
    great_circle_intersect,
)

GEO = geographiclib.Geodesic(EARTH_RADIUS, 0.0)


def _oracle_azimuth(lat1, lon1, lat2, lon2):
    """Azimuth from point 1 to point 2, in [0, 2*pi), via geographiclib."""
    result = GEO.Inverse(
        np.degrees(lat1), np.degrees(lon1), np.degrees(lat2), np.degrees(lon2)
    )
    return np.radians(result["azi1"]) % (2 * np.pi)


def _oracle_point_along(lat1, lon1, azimuth, distance):
    """Point reached from (lat1, lon1) heading azimuth for signed distance,
    via geographiclib. Negative distance extends the line backwards."""
    result = GEO.Direct(
        np.degrees(lat1), np.degrees(lon1), np.degrees(azimuth), distance
    )
    return np.radians(result["lat2"]), np.radians(result["lon2"])


class TestGreatCircleIntersectBranch:
    """great_circle_intersect must return the intersection reached by a
    positive distance along each point's observed azimuth, not whichever
    antipodal solution the underlying vector algebra happens to produce."""

    @staticmethod
    def _angular_diff(a, b):
        return abs((a - b + np.pi) % (2 * np.pi) - np.pi)

    @pytest.mark.parametrize(
        "az1_deg,az2_deg",
        [
            (135.0, 225.0),  # the measured defect case
            (45.0, 315.0),  # already correct before the fix; must stay so
        ],
    )
    def test_intersection_reached_by_forward_azimuth_from_both_points(
        self, az1_deg, az2_deg
    ):
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, np.radians(10.0)
        az1, az2 = np.radians(az1_deg), np.radians(az2_deg)

        result = great_circle_intersect(lat1, lon1, az1, lat2, lon2, az2)
        assert result.valid

        oracle_az_from_1 = _oracle_azimuth(lat1, lon1, result.lat1, result.lon1)
        oracle_az_from_2 = _oracle_azimuth(lat2, lon2, result.lat1, result.lon1)

        assert self._angular_diff(oracle_az_from_1, az1) < 1e-6
        assert self._angular_diff(oracle_az_from_2, az2) < 1e-6

    def test_matches_matlab_worked_crossfix_example(self):
        # greatCircleIntersect.m's own EXAMPLE: two sensors (near Tokyo and
        # Sydney) observe a target (near San Francisco) and localize it by
        # crossfix. Bearings come from the oracle (geographiclib), not from
        # this module's own azimuth function, so the whole computation is
        # independently checked end to end.
        lat_a, lon_a = np.radians(34.685169), np.radians(139.443632)
        lat_c, lon_c = np.radians(-33.8617), np.radians(151.2117)
        lat_x, lon_x = np.radians(37.7917), np.radians(-122.4633)

        az_ax = _oracle_azimuth(lat_a, lon_a, lat_x, lon_x)
        az_cx = _oracle_azimuth(lat_c, lon_c, lat_x, lon_x)

        result = great_circle_intersect(lat_a, lon_a, az_ax, lat_c, lon_c, az_cx)
        assert result.valid
        assert result.lat1 == pytest.approx(lat_x, abs=1e-6)
        assert result.lon1 == pytest.approx(lon_x, abs=1e-6)

    @pytest.mark.parametrize(
        "az1_deg,az2_deg",
        [
            (135.0, 225.0),  # branch swap required (the measured defect)
            (45.0, 315.0),  # no swap required -- the antipode-formula blind
            # spot: a reintroduced longitude bug in the *second* point would
            # pass every other test here undetected, since nothing else
            # inspects (lat2, lon2) when the primary branch is already right.
        ],
    )
    def test_two_returned_points_are_genuinely_antipodal(self, az1_deg, az2_deg):
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, np.radians(10.0)
        az1, az2 = np.radians(az1_deg), np.radians(az2_deg)

        result = great_circle_intersect(lat1, lon1, az1, lat2, lon2, az2)
        assert result.valid

        # Oracle-measured separation between the two returned points, not
        # this module's own great_circle_distance -- independent of the
        # code under test, including whatever branch it selected.
        separation = GEO.Inverse(
            np.degrees(result.lat1),
            np.degrees(result.lon1),
            np.degrees(result.lat2),
            np.degrees(result.lon2),
        )["s12"]
        assert separation == pytest.approx(np.pi * EARTH_RADIUS, rel=1e-9)


class TestCrossTrackAlongTrackSigned:
    """along_track must be signed: positive ahead of the path start,
    negative behind it, matching the sign and magnitude of an independently
    constructed on-path point at a known signed distance."""

    @pytest.mark.parametrize("signed_distance", [2_000_000.0, -1_500_000.0])
    def test_along_track_matches_oracle_point_on_path(self, signed_distance):
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, np.radians(10.0)

        path_azimuth = _oracle_azimuth(lat1, lon1, lat2, lon2)
        lat_p, lon_p = _oracle_point_along(lat1, lon1, path_azimuth, signed_distance)

        result = cross_track_distance(lat_p, lon_p, lat1, lon1, lat2, lon2)

        assert result.cross_track == pytest.approx(0.0, abs=1e-3)
        assert result.along_track == pytest.approx(signed_distance, rel=1e-6)

    def test_along_track_is_signed_and_antisymmetric(self):
        # Task 3.3's measured defect: a point 5 deg behind the start
        # returned the same along_track as a point 5 deg ahead.
        start = (0.0, 0.0)
        end = (0.0, np.radians(10.0))

        ahead = cross_track_distance(
            0.0, np.radians(5.0), start[0], start[1], end[0], end[1]
        )
        behind = cross_track_distance(
            0.0, np.radians(-5.0), start[0], start[1], end[0], end[1]
        )

        assert ahead.along_track > 0
        assert behind.along_track < 0
        assert behind.along_track == pytest.approx(-ahead.along_track, rel=1e-9)
