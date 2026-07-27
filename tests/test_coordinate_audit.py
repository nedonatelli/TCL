"""Correctness audit tests for pytcl.coordinate_systems.

Every public function is validated against an independent reference
(scipy.spatial.transform for rotations, pyproj for geodetic conversions and
map projections) or against mathematical invariants (round-trips, orthogonality,
numerical Jacobians).

Tolerances: reference-exact formulas are held to near machine precision
(1e-9 rad / 1e-6 m); iterative inversions to their documented convergence
(1e-8); the azimuthal equidistant projection to its documented spherical
approximation (0.5% of distance from center).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from pytcl.coordinate_systems import (
    axisangle2rotmat,
    azimuthal_equidistant,
    azimuthal_equidistant_inverse,
    cart2cyl,
    cart2pol,
    cart2ruv,
    cart2sphere,
    cross_covariance_transform,
    cyl2cart,
    dcm_rate,
    ecef2enu,
    ecef2geodetic,
    ecef2ned,
    enu2ecef,
    enu2ned,
    enu_jacobian,
    euler2quat,
    euler2rotmat,
    geocentric_radius,
    geodetic2ecef,
    geodetic2enu,
    geodetic2utm,
    geodetic_jacobian,
    is_rotation_matrix,
    lambert_conformal_conic,
    lambert_conformal_conic_inverse,
    mercator,
    mercator_inverse,
    meridional_radius,
    ned2ecef,
    ned2enu,
    ned_jacobian,
    numerical_jacobian,
    pol2cart,
    polar_jacobian,
    polar_jacobian_inv,
    polar_stereographic,
    prime_vertical_radius,
    quat2euler,
    quat2rotmat,
    quat_conjugate,
    quat_inverse,
    quat_multiply,
    quat_rotate,
    rodrigues2rotmat,
    rotmat2axisangle,
    rotmat2euler,
    rotmat2quat,
    rotmat2rodrigues,
    rotx,
    roty,
    rotz,
    ruv2cart,
    ruv_jacobian,
    slerp,
    sphere2cart,
    spherical_jacobian,
    spherical_jacobian_inv,
    stereographic,
    stereographic_inverse,
    transverse_mercator,
    transverse_mercator_inverse,
    utm2geodetic,
    utm_central_meridian,
    utm_zone,
)
from pytcl.coordinate_systems.conversions.geodetic import (
    ecef2sez,
    geodetic2sez,
    sez2ecef,
    sez2geodetic,
)

RNG_SEED = 20260726


def scalar_first(rot: Rotation) -> np.ndarray:
    """scipy quaternion [x, y, z, w] -> pytcl convention [w, x, y, z]."""
    q = rot.as_quat()
    return np.r_[q[3], q[:3]]


def assert_quat_close(q1, q2, atol=1e-12):
    """Compare quaternions up to the q ~ -q sign ambiguity."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    err = min(np.abs(q1 - q2).max(), np.abs(q1 + q2).max())
    assert err < atol, f"quaternion mismatch: {err}"


class TestRotationsVsScipy:
    """Rotation representations validated against scipy.spatial.transform."""

    def test_rotx_roty_rotz(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            a = rng.uniform(-np.pi, np.pi)
            np.testing.assert_allclose(
                rotx(a), Rotation.from_euler("x", a).as_matrix(), atol=1e-15
            )
            np.testing.assert_allclose(
                roty(a), Rotation.from_euler("y", a).as_matrix(), atol=1e-15
            )
            np.testing.assert_allclose(
                rotz(a), Rotation.from_euler("z", a).as_matrix(), atol=1e-15
            )

    @pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ", "YXZ", "XZX"])
    def test_euler2rotmat_matches_scipy_intrinsic(self, seq):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            ang = rng.uniform(-np.pi, np.pi, 3)
            np.testing.assert_allclose(
                euler2rotmat(ang, seq),
                Rotation.from_euler(seq, ang).as_matrix(),
                atol=1e-14,
            )

    @pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ"])
    def test_rotmat2euler_matches_scipy(self, seq):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            mat = Rotation.random(rng=rng).as_matrix()
            np.testing.assert_allclose(
                rotmat2euler(mat, seq),
                Rotation.from_matrix(mat).as_euler(seq),
                atol=1e-12,
            )

    @pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ"])
    def test_rotmat2euler_roundtrip(self, seq):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            mat = Rotation.random(rng=rng).as_matrix()
            np.testing.assert_allclose(
                euler2rotmat(rotmat2euler(mat, seq), seq), mat, atol=1e-12
            )

    @pytest.mark.parametrize("pitch_sign", [1.0, -1.0])
    def test_zyx_gimbal_lock(self, pitch_sign):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(10):
            yaw, roll = rng.uniform(-np.pi, np.pi, 2)
            mat = euler2rotmat([yaw, pitch_sign * np.pi / 2, roll], "ZYX")
            rec = rotmat2euler(mat, "ZYX")
            # Angles are not unique at the singularity, the rotation must be.
            np.testing.assert_allclose(euler2rotmat(rec, "ZYX"), mat, atol=1e-8)

    @pytest.mark.parametrize("y_sign", [1.0, -1.0])
    def test_xyz_gimbal_lock(self, y_sign):
        rng = np.random.default_rng(RNG_SEED)
        for eps in [0.0, 1e-8]:
            x, z = rng.uniform(-np.pi, np.pi, 2)
            mat = euler2rotmat([x, y_sign * (np.pi / 2 - eps), z], "XYZ")
            rec = rotmat2euler(mat, "XYZ")
            np.testing.assert_allclose(euler2rotmat(rec, "XYZ"), mat, atol=1e-6)

    def test_quat2rotmat_matches_scipy(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            rot = Rotation.random(rng=rng)
            np.testing.assert_allclose(
                quat2rotmat(scalar_first(rot)), rot.as_matrix(), atol=1e-14
            )

    def test_rotmat2quat_matches_scipy(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            rot = Rotation.random(rng=rng)
            assert_quat_close(rotmat2quat(rot.as_matrix()), scalar_first(rot))

    def test_euler2quat_quat2euler_vs_scipy(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            ang = rng.uniform(-1.4, 1.4, 3)
            q = euler2quat(ang, "ZYX")
            assert_quat_close(q, scalar_first(Rotation.from_euler("ZYX", ang)))
            np.testing.assert_allclose(quat2euler(q, "ZYX"), ang, atol=1e-12)

    def test_quat_multiply_matches_scipy_composition(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            r1, r2 = Rotation.random(rng=rng), Rotation.random(rng=rng)
            assert_quat_close(
                quat_multiply(scalar_first(r1), scalar_first(r2)),
                scalar_first(r1 * r2),
            )

    def test_quat_rotate_matches_scipy_apply(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            rot = Rotation.random(rng=rng)
            v = rng.normal(size=3)
            np.testing.assert_allclose(
                quat_rotate(scalar_first(rot), v), rot.apply(v), atol=1e-12
            )

    def test_quat_conjugate_inverse_properties(self):
        rng = np.random.default_rng(RNG_SEED)
        q = scalar_first(Rotation.random(rng=rng)) * 2.5  # non-unit
        np.testing.assert_allclose(
            quat_multiply(q, quat_inverse(q)), [1, 0, 0, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            quat_conjugate(q), [q[0], -q[1], -q[2], -q[3]], atol=1e-15
        )

    def test_axisangle_rodrigues_vs_scipy_rotvec(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            rot = Rotation.random(rng=rng)
            rvec = rot.as_rotvec()
            angle = np.linalg.norm(rvec)
            np.testing.assert_allclose(
                axisangle2rotmat(rvec / angle, angle), rot.as_matrix(), atol=1e-13
            )
            np.testing.assert_allclose(
                rodrigues2rotmat(rvec), rot.as_matrix(), atol=1e-13
            )
            # inverse maps (axis sign ambiguity resolved by comparing products)
            axis, ang = rotmat2axisangle(rot.as_matrix())
            np.testing.assert_allclose(np.asarray(axis) * ang, rvec, atol=1e-9)
            np.testing.assert_allclose(
                rotmat2rodrigues(rot.as_matrix()), rvec, atol=1e-9
            )

    def test_rotmat2axisangle_180deg(self):
        axis_in = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
        mat = axisangle2rotmat(axis_in, np.pi)
        axis, ang = rotmat2axisangle(mat)
        # arccos((trace-1)/2) near trace = -1 is sqrt(eps)-accurate
        assert abs(ang - np.pi) < 1e-7
        np.testing.assert_allclose(np.abs(axis), np.abs(axis_in), atol=1e-7)

    def test_rotmat2axisangle_near_180deg(self):
        # Angles just below pi must still recompose to the same rotation
        axis_in = np.array([-2.0, 1.0, 2.0]) / 3.0
        for delta in [1e-9, 1e-8, 1e-7, 1e-5, 1e-3]:
            mat = axisangle2rotmat(axis_in, np.pi - delta)
            axis, ang = rotmat2axisangle(mat)
            assert abs(np.linalg.norm(axis) - 1) < 1e-9
            np.testing.assert_allclose(axisangle2rotmat(axis, ang), mat, atol=1e-6)

    def test_slerp_matches_scipy(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            r1, r2 = Rotation.random(rng=rng), Rotation.random(rng=rng)
            ref = Slerp([0, 1], Rotation.concatenate([r1, r2]))
            for t in [0.0, 0.3, 0.5, 0.9, 1.0]:
                assert_quat_close(
                    slerp(scalar_first(r1), scalar_first(r2), t),
                    scalar_first(ref(t)),
                    atol=1e-9,
                )

    def test_dcm_rate_finite_difference(self):
        rng = np.random.default_rng(RNG_SEED)
        mat = Rotation.random(rng=rng).as_matrix()
        omega = rng.normal(size=3)
        dt = 1e-7
        fwd = mat @ Rotation.from_rotvec(omega * dt).as_matrix()
        np.testing.assert_allclose(dcm_rate(mat, omega), (fwd - mat) / dt, atol=1e-6)

    def test_is_rotation_matrix(self):
        rng = np.random.default_rng(RNG_SEED)
        assert is_rotation_matrix(Rotation.random(rng=rng).as_matrix())
        assert not is_rotation_matrix(np.eye(3) * 2)
        assert not is_rotation_matrix(np.diag([1.0, 1.0, -1.0]))  # det = -1

    def test_rotation_matrix_properties(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            mat = euler2rotmat(rng.uniform(-np.pi, np.pi, 3), "ZYX")
            np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-14)
            assert abs(np.linalg.det(mat) - 1) < 1e-12


class TestSphericalConversions:
    """Round-trip and reference validation of cart/sphere/cyl/pol/ruv."""

    def edge_points(self):
        rng = np.random.default_rng(RNG_SEED)
        pts = rng.normal(size=(50, 3)) * rng.choice([1e-3, 1.0, 1e6], size=(50, 1))
        edges = np.array(
            [
                [0, 0, 1],
                [0, 0, -1],
                [-1, 0, 0],
                [0, -1, 0],
                [1e-12, 1e-12, 1],
                [-5, -5, -5],
            ]
        )
        return np.vstack([pts, edges])

    @pytest.mark.parametrize("system", ["standard", "az-el"])
    def test_cart_sphere_roundtrip(self, system):
        for p in self.edge_points():
            r, az, el = cart2sphere(p, system)
            back = sphere2cart(r, az, el, system)
            np.testing.assert_allclose(back, p, atol=1e-9 * max(1.0, np.linalg.norm(p)))

    def test_cart2sphere_azel_reference(self):
        # 45 deg az, elevation of (1,1,1) is atan(1/sqrt(2))
        r, az, el = cart2sphere([1.0, 1.0, 1.0], "az-el")
        assert abs(r - np.sqrt(3)) < 1e-12
        assert abs(az - np.pi / 4) < 1e-12
        assert abs(el - np.arctan2(1, np.sqrt(2))) < 1e-12

    def test_cart2sphere_standard_ranges(self):
        rng = np.random.default_rng(RNG_SEED)
        for p in rng.normal(size=(50, 3)):
            _, az, el = cart2sphere(p, "standard")
            assert 0 <= az < 2 * np.pi
            assert 0 <= el <= np.pi

    def test_cart2sphere_input_shapes_agree(self):
        rng = np.random.default_rng(RNG_SEED)
        pts = rng.normal(size=(3, 20))
        r1, a1, e1 = cart2sphere(pts, "az-el")
        r2, a2, e2 = cart2sphere(pts.T, "az-el")
        np.testing.assert_allclose([r1, a1, e1], [r2, a2, e2])

    def test_pol_cyl_roundtrips(self):
        for p in self.edge_points():
            rho, phi, z = cart2cyl(p)
            np.testing.assert_allclose(
                cyl2cart(rho, phi, z), p, atol=1e-9 * max(1.0, np.linalg.norm(p))
            )
            r, theta = cart2pol(p[:2])
            np.testing.assert_allclose(
                pol2cart(r, theta),
                p[:2],
                atol=1e-9 * max(1.0, np.linalg.norm(p[:2])),
            )

    def test_ruv_roundtrip_upper_hemisphere(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            p = rng.normal(size=3) * 100
            p[2] = abs(p[2]) + 0.1  # ruv2cart assumes z >= 0
            r, u, v = cart2ruv(p)
            np.testing.assert_allclose(ruv2cart(r, u, v), p, atol=1e-6)

    def test_cart2ruv_reference(self):
        az, el = np.radians(45), np.radians(30)
        p = sphere2cart(100.0, az, el, "az-el")
        r, u, v = cart2ruv(p)
        assert abs(r - 100.0) < 1e-9
        assert abs(u - np.cos(az) * np.cos(el)) < 1e-12
        assert abs(v - np.sin(az) * np.cos(el)) < 1e-12


class TestGeodeticVsPyproj:
    """Geodetic conversions validated against pyproj."""

    pyproj = pytest.importorskip("pyproj")

    @pytest.fixture(scope="class")
    def ecef_transformer(self):
        return self.pyproj.Transformer.from_crs(
            "EPSG:4979", "EPSG:4978", always_xy=True
        )

    def test_geodetic2ecef_vs_pyproj(self, ecef_transformer):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            lat = rng.uniform(-89.9, 89.9)
            lon = rng.uniform(-180, 180)
            alt = rng.uniform(-5e3, 4e7)
            ref = ecef_transformer.transform(lon, lat, alt)
            got = geodetic2ecef(np.radians(lat), np.radians(lon), alt)
            np.testing.assert_allclose(got, ref, atol=1e-6)

    @pytest.mark.parametrize("method", ["iterative", "direct"])
    def test_ecef2geodetic_vs_pyproj(self, ecef_transformer, method):
        rng = np.random.default_rng(RNG_SEED)
        # iterative: Bowring converges to sub-mm; direct: Vermeille is exact
        tol_m = 1e-2 if method == "iterative" else 1e-6
        for _ in range(50):
            lat = rng.uniform(-89.9, 89.9)
            lon = rng.uniform(-180, 180)
            alt = rng.uniform(-5e3, 4e7)
            ecef = ecef_transformer.transform(lon, lat, alt)
            lat2, lon2, alt2 = ecef2geodetic(np.array(ecef), method=method)
            assert abs(np.degrees(lat2) - lat) * 1.11e5 < tol_m
            assert abs(alt2 - alt) < tol_m

    @pytest.mark.parametrize("method", ["iterative", "direct"])
    @pytest.mark.parametrize(
        "lat_deg,alt", [(90.0, 0.0), (-90.0, 1e4), (0.0, 0.0), (89.999, 100.0)]
    )
    def test_ecef2geodetic_edge_latitudes(self, ecef_transformer, method, lat_deg, alt):
        ecef = ecef_transformer.transform(10.0, lat_deg, alt)
        lat2, _, alt2 = ecef2geodetic(np.array(ecef), method=method)
        assert abs(np.degrees(lat2) - lat_deg) < 1e-9
        assert abs(alt2 - alt) < 1e-6

    def test_enu_against_geodesic(self):
        # A point 1 km due north (geodesic) must be ~[0, 1000, -d^2/2R]
        geod = self.pyproj.Geod(ellps="WGS84")
        lat0, lon0 = np.radians(37.0), np.radians(-122.0)
        for az, expect_idx in [(0.0, 1), (90.0, 0)]:
            lon1, lat1, _ = geod.fwd(np.degrees(lon0), np.degrees(lat0), az, 1000.0)
            enu = geodetic2enu(np.radians(lat1), np.radians(lon1), 0.0, lat0, lon0, 0.0)
            assert abs(enu[expect_idx] - 1000.0) < 0.1
            assert abs(enu[1 - expect_idx]) < 0.1
            # Earth curvature drop: d^2 / (2 R) ~ 0.078 m
            assert abs(enu[2] + 1000.0**2 / (2 * 6.371e6)) < 0.01

    def test_enu_ned_sez_roundtrips_and_consistency(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(30):
            lat = rng.uniform(-1.4, 1.4)
            lon = rng.uniform(-np.pi, np.pi)
            ref = geodetic2ecef(lat, lon, rng.uniform(0, 1e4))
            p = ref + rng.normal(scale=1e5, size=3)
            enu = ecef2enu(p, lat, lon, ref)
            ned = ecef2ned(p, lat, lon, ref)
            sez = ecef2sez(p, lat, lon, ref)
            np.testing.assert_allclose(enu2ecef(enu, lat, lon, ref), p, atol=1e-6)
            np.testing.assert_allclose(ned2ecef(ned, lat, lon, ref), p, atol=1e-6)
            np.testing.assert_allclose(sez2ecef(sez, lat, lon, ref), p, atol=1e-6)
            np.testing.assert_allclose(enu2ned(enu), ned, atol=1e-9)
            np.testing.assert_allclose(ned2enu(ned), enu, atol=1e-9)
            # SEZ = [-N, E, -D] (Vallado convention)
            np.testing.assert_allclose(sez, [-ned[0], ned[1], -ned[2]], atol=1e-6)

    def test_sez_geodetic_roundtrip(self):
        lat0, lon0, alt0 = np.radians(18.3), np.radians(-66.75), 500.0
        sez = np.array([-30000.0, 20000.0, 35000.0])
        lat, lon, alt = sez2geodetic(sez, lat0, lon0, alt0)
        back = geodetic2sez(lat, lon, alt, lat0, lon0, alt0)
        np.testing.assert_allclose(back, sez, atol=1e-5)

    def test_radii(self):
        geod = self.pyproj.Geod(ellps="WGS84")
        for lat_deg in [0.0, 30.0, 60.0, 85.0]:
            lat = np.radians(lat_deg)
            # geocentric radius = |ECEF| of a surface point
            ecef = geodetic2ecef(lat, 0.0, 0.0)
            assert abs(geocentric_radius(lat) - np.linalg.norm(ecef)) < 1e-6
            # N cos(lat) = distance of surface point from spin axis
            assert abs(prime_vertical_radius(lat) * np.cos(lat) - ecef[0]) < 1e-6
            # M = d(meridian arc)/d(lat), via geodesic arc over 2e-4 deg
            d = 1e-4
            _, _, arc = geod.inv(0.0, lat_deg - d, 0.0, lat_deg + d)
            m_num = arc / np.radians(2 * d)
            assert abs(meridional_radius(lat) - m_num) < 1.0


class TestProjectionsVsPyproj:
    """Map projections validated against pyproj CRS transforms."""

    pyproj = pytest.importorskip("pyproj")

    def transformer(self, proj4):
        crs = self.pyproj.CRS.from_proj4(proj4)
        return self.pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def test_mercator_vs_pyproj(self):
        rng = np.random.default_rng(RNG_SEED)
        tr = self.transformer("+proj=merc +lon_0=0 +k=1 +ellps=WGS84")
        for _ in range(50):
            lat, lon = rng.uniform(-85, 85), rng.uniform(-179, 179)
            xr, yr = tr.transform(lon, lat)
            res = mercator(np.radians(lat), np.radians(lon))
            assert np.hypot(res.x - xr, res.y - yr) < 1e-6
            lat2, lon2 = mercator_inverse(xr, yr)
            assert abs(np.degrees(lat2) - lat) < 1e-9
            assert abs(np.degrees(lon2) - lon) < 1e-9

    def test_mercator_scale_vs_pyproj_factors(self):
        proj = self.pyproj.Proj(
            self.pyproj.CRS.from_proj4("+proj=merc +lon_0=0 +k=1 +ellps=WGS84")
        )
        factors = proj.get_factors(-75.0, 45.0)
        res = mercator(np.radians(45.0), np.radians(-75.0))
        assert abs(res.scale - factors.meridional_scale) < 1e-9

    def test_transverse_mercator_and_utm_vs_pyproj(self):
        rng = np.random.default_rng(RNG_SEED)
        tr = self.transformer("+proj=utm +zone=18 +ellps=WGS84")
        for _ in range(20):
            lat, lon = rng.uniform(0.1, 80), rng.uniform(-77.9, -72.1)
            xr, yr = tr.transform(lon, lat)
            res = geodetic2utm(np.radians(lat), np.radians(lon), zone=18)
            # Redfearn series: ~1 mm within a UTM zone
            assert np.hypot(res.easting - xr, res.northing - yr) < 1e-2
            lat2, lon2 = utm2geodetic(xr, yr, 18, "N")
            assert abs(np.degrees(lat2) - lat) * 1.11e5 < 1e-2
            assert abs(np.degrees(lon2) - lon) * 1.11e5 < 1e-2
        # raw transverse_mercator drives geodetic2utm; check the offset wiring
        res_tm = transverse_mercator(
            np.radians(45.0), np.radians(-75.5), lon0=np.radians(-75.0), k0=0.9996
        )
        res_utm = geodetic2utm(np.radians(45.0), np.radians(-75.5))
        assert abs(res_tm.x + 500000.0 - res_utm.easting) < 1e-9
        assert abs(res_tm.y - res_utm.northing) < 1e-9
        lat3, lon3 = transverse_mercator_inverse(
            res_tm.x, res_tm.y, lon0=np.radians(-75.0), k0=0.9996
        )
        assert abs(lat3 - np.radians(45.0)) < 1e-10
        assert abs(lon3 - np.radians(-75.5)) < 1e-10

    def test_utm_southern_hemisphere(self):
        tr = self.transformer("+proj=utm +zone=56 +south +ellps=WGS84")
        xr, yr = tr.transform(151.2, -33.9)  # Sydney
        res = geodetic2utm(np.radians(-33.9), np.radians(151.2))
        assert res.zone == 56 and res.hemisphere == "S"
        assert np.hypot(res.easting - xr, res.northing - yr) < 1e-2
        lat2, lon2 = utm2geodetic(res.easting, res.northing, 56, "S")
        assert abs(np.degrees(lat2) + 33.9) < 1e-7
        assert abs(np.degrees(lon2) - 151.2) < 1e-7

    def test_utm_zone_rules(self):
        rad = np.radians
        assert utm_zone(rad(-75.5)) == 18
        assert utm_zone(rad(-179.9)) == 1
        assert utm_zone(rad(179.9)) == 60
        assert utm_zone(rad(180.0)) == 1  # wraps to -180
        # Norway and Svalbard exceptions
        assert utm_zone(rad(3.5), rad(60)) == 32
        assert utm_zone(rad(5), rad(75)) == 31
        assert utm_zone(rad(10), rad(75)) == 33
        assert utm_zone(rad(25), rad(75)) == 35
        assert utm_zone(rad(34), rad(80)) == 37
        assert abs(np.degrees(utm_central_meridian(31)) - 3.0) < 1e-9

    @pytest.mark.parametrize("north", [True, False])
    def test_polar_stereographic_vs_pyproj(self, north):
        rng = np.random.default_rng(RNG_SEED)
        sign = 1 if north else -1
        proj4 = (
            f"+proj=stere +lat_0={sign * 90} +lat_ts={sign * 90} "
            "+lon_0=0 +k_0=0.994 +ellps=WGS84"
        )
        tr = self.transformer(proj4)
        proj = self.pyproj.Proj(self.pyproj.CRS.from_proj4(proj4))
        for _ in range(30):
            lat = sign * rng.uniform(55, 90)
            lon = rng.uniform(-180, 180)
            xr, yr = tr.transform(lon, lat)
            res = polar_stereographic(np.radians(lat), np.radians(lon), north=north)
            assert np.hypot(res.x - xr, res.y - yr) < 1e-6
            factors = proj.get_factors(lon, lat)
            assert abs(res.scale - factors.meridional_scale) < 1e-9
            conv_err = res.convergence - np.radians(factors.meridian_convergence)
            assert abs(np.arctan2(np.sin(conv_err), np.cos(conv_err))) < 1e-9

    def test_stereographic_self_roundtrip(self):
        rng = np.random.default_rng(RNG_SEED)
        lat0, lon0 = np.radians(52.0), np.radians(5.0)
        for _ in range(50):
            lat, lon = np.radians(rng.uniform(20, 84)), np.radians(rng.uniform(-40, 40))
            res = stereographic(lat, lon, lat0, lon0)
            lat2, lon2 = stereographic_inverse(res.x, res.y, lat0, lon0)
            assert abs(lat2 - lat) < 1e-10
            assert abs(lon2 - lon) < 1e-10

    def test_stereographic_center_and_conformal_scale(self):
        lat0, lon0 = np.radians(52.0), np.radians(5.0)
        res = stereographic(lat0, lon0, lat0, lon0, k0=0.9999)
        assert abs(res.x) < 1e-9 and abs(res.y) < 1e-9
        assert abs(res.scale - 0.9999) < 1e-9

    @pytest.mark.parametrize(
        "lat0,lon0,lat1,lat2,lat_range,lon_range",
        [
            (39.0, -96.0, 33.0, 45.0, (20, 60), (-130, -60)),
            (-30.0, 135.0, -20.0, -40.0, (-55, -10), (110, 155)),
        ],
    )
    def test_lambert_conformal_conic_vs_pyproj(
        self, lat0, lon0, lat1, lat2, lat_range, lon_range
    ):
        rng = np.random.default_rng(RNG_SEED)
        tr = self.transformer(
            f"+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} "
            f"+lon_0={lon0} +ellps=WGS84"
        )
        args = (np.radians(lat0), np.radians(lon0), np.radians(lat1), np.radians(lat2))
        for _ in range(30):
            lat = rng.uniform(*lat_range)
            lon = rng.uniform(*lon_range)
            xr, yr = tr.transform(lon, lat)
            res = lambert_conformal_conic(np.radians(lat), np.radians(lon), *args)
            assert np.hypot(res.x - xr, res.y - yr) < 1e-6
            lat_b, lon_b = lambert_conformal_conic_inverse(xr, yr, *args)
            assert abs(np.degrees(lat_b) - lat) * 1.11e5 < 1e-4
            assert abs(np.degrees(lon_b) - lon) * 1.11e5 < 1e-4

    def test_azimuthal_equidistant_vs_pyproj(self):
        rng = np.random.default_rng(RNG_SEED)
        lat0, lon0 = 38.9, -77.0
        tr = self.transformer(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84")
        for _ in range(50):
            lat, lon = rng.uniform(25, 55), rng.uniform(-95, -60)
            xr, yr = tr.transform(lon, lat)
            res = azimuthal_equidistant(
                np.radians(lat), np.radians(lon), np.radians(lat0), np.radians(lon0)
            )
            # Documented spherical (authalic) approximation of the ellipsoidal
            # geodesic projection: allow 0.5% of distance + 50 m floor.
            dist = np.hypot(xr, yr)
            assert np.hypot(res.x - xr, res.y - yr) < 0.005 * dist + 50.0
            lat2, lon2 = azimuthal_equidistant_inverse(
                res.x, res.y, np.radians(lat0), np.radians(lon0)
            )
            assert abs(lat2 - np.radians(lat)) < 1e-9
            assert abs(lon2 - np.radians(lon)) < 1e-9

    def test_azimuthal_equidistant_preserves_center_distance(self):
        # Radial distance from center must equal the sphere arc distance
        lat0, lon0 = np.radians(38.9), np.radians(-77.0)
        res = azimuthal_equidistant(np.radians(48.9), lon0, lat0, lon0)
        # 10 degrees of latitude ~ 10 * 111.1 km on the authalic sphere
        rho = np.hypot(res.x, res.y)
        assert abs(rho - np.radians(10.0) * 6371007.181) < 1e-3


class TestJacobians:
    """Jacobians validated against numerical differentiation."""

    @pytest.mark.parametrize("system", ["standard", "az-el"])
    def test_spherical_jacobian_vs_numerical(self, system):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(30):
            p = rng.normal(size=3) * 100

            def f(x):
                r, az, el = cart2sphere(x, system)
                return np.array([r, az, el])

            jac = spherical_jacobian(p, system)
            jac_num = numerical_jacobian(f, p, dx=1e-5)
            np.testing.assert_allclose(jac, jac_num, atol=1e-6)

    @pytest.mark.parametrize("system", ["standard", "az-el"])
    def test_spherical_jacobian_inv_and_product(self, system):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(30):
            r = rng.uniform(10, 1e4)
            az = rng.uniform(-np.pi, np.pi)
            el = rng.uniform(-1.4, 1.4) if system == "az-el" else rng.uniform(0.1, 3.0)

            def f(s):
                return sphere2cart(s[0], s[1], s[2], system)

            jac_inv = spherical_jacobian_inv(r, az, el, system)
            jac_num = numerical_jacobian(f, [r, az, el], dx=1e-6)
            np.testing.assert_allclose(jac_inv, jac_num, atol=1e-4, rtol=1e-6)
            # forward and inverse Jacobians must be matrix inverses
            p = sphere2cart(r, az, el, system)
            np.testing.assert_allclose(
                spherical_jacobian(p, system) @ jac_inv, np.eye(3), atol=1e-9
            )

    def test_polar_jacobians(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(30):
            p = rng.normal(size=2) * 100

            def f(x):
                r, theta = cart2pol(x)
                return np.array([r, theta])

            jac = polar_jacobian(p)
            np.testing.assert_allclose(
                jac, numerical_jacobian(f, p, dx=1e-6), atol=1e-6
            )
            r, theta = cart2pol(p)
            np.testing.assert_allclose(
                jac @ polar_jacobian_inv(r, theta), np.eye(2), atol=1e-9
            )

    def test_ruv_jacobian_vs_numerical(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(30):
            p = rng.normal(size=3) * 100
            p[2] = abs(p[2]) + 1

            def f(x):
                r, u, v = cart2ruv(x)
                return np.array([r, u, v])

            np.testing.assert_allclose(
                ruv_jacobian(p), numerical_jacobian(f, p, dx=1e-5), atol=1e-6
            )

    def test_enu_ned_jacobians(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            # quantize to the documented 1e-5 rad cache resolution
            lat = round(rng.uniform(-1.4, 1.4), 5)
            lon = round(rng.uniform(-np.pi, np.pi), 5)
            j_enu = enu_jacobian(lat, lon)
            j_ned = ned_jacobian(lat, lon)
            # proper rotations
            for jac in (j_enu, j_ned):
                np.testing.assert_allclose(jac @ jac.T, np.eye(3), atol=1e-12)
                assert abs(np.linalg.det(jac) - 1) < 1e-12
            # consistent with ecef2enu / ecef2ned
            ref = geodetic2ecef(lat, lon, 0.0)
            d = rng.normal(size=3) * 1000
            np.testing.assert_allclose(
                j_enu @ d, ecef2enu(ref + d, lat, lon, ref), atol=1e-6
            )
            np.testing.assert_allclose(
                j_ned @ d, ecef2ned(ref + d, lat, lon, ref), atol=1e-6
            )

    def test_geodetic_jacobian_vs_numerical(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(30):
            lla = [
                rng.uniform(-1.4, 1.4),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(0, 1e4),
            ]

            def f(x):
                return geodetic2ecef(x[0], x[1], x[2])

            jac = geodetic_jacobian(*lla)
            jac_num = numerical_jacobian(f, lla, dx=1e-7)
            np.testing.assert_allclose(jac, jac_num, atol=1e-6 * np.abs(jac_num).max())

    def test_cross_covariance_transform(self):
        rng = np.random.default_rng(RNG_SEED)
        jac = rng.normal(size=(3, 3))
        cov = np.diag([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            cross_covariance_transform(jac, cov), jac @ cov @ jac.T
        )

    def test_numerical_jacobian_analytic_case(self):
        def f(x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        jac = numerical_jacobian(f, [3.0, 2.0])
        np.testing.assert_allclose(jac, [[6.0, 0.0], [2.0, 3.0]], atol=1e-5)
