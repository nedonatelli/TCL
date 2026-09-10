"""
Magnetic coordinate systems.

Ports of the MATLAB TCL's ``Magnetism`` coordinate half: centered
dipole (CD) transforms, magnetic apex and quasi-dipole (QD)
coordinates via field-line tracing, and magnetic/geographic heading
conversions.

The field-line tracer integrates along the normalized magnetic flux
direction until the ellipsoidal height stops increasing, then locates
the apex by root-finding on the level condition -- algorithmically
the MATLAB implementation, with SciPy's adaptive RK45 and event
root-finding in place of the hand-rolled adaptive step and
``fminbnd`` (a documented swap; the fixtures pin agreement).

Positions are ITRS/ECEF Cartesian in meters unless stated otherwise.

References
----------
- A. D. Richmond, "Ionospheric electrodynamics using magnetic apex
  coordinates," Journal of Geomagnetism and Geoelectricity, vol. 47,
  no. 2, pp. 191-212, 1995.
- J. T. Emmert, A. D. Richmond, and D. P. Drob, "A computationally
  compact representation of magnetic-apex and quasi-dipole
  coordinates with smooth base vectors," Journal of Geophysical
  Research, vol. 115, no. A8, 2010.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from pytcl.coordinate_systems.conversions.geodetic import ecef2geodetic
from pytcl.core.constants import EARTH_SEMI_MAJOR_AXIS
from pytcl.magnetism.igrf import IGRF14
from pytcl.magnetism.wmm import (
    WMM2025,
    MagneticCoefficients,
    _compute_magnetic_field_spherical_impl,
)

__all__ = [
    "cd_rotation_matrix",
    "itrs2cart_cd",
    "cart_cd2itrs",
    "spher_itrs2spher_cd",
    "spher_cd2spher_itrs",
    "geog_heading2mag",
    "mag_heading2geog",
    "trace2earth_mag_apex",
    "itrs2magnetic_apex",
    "itrs2qd",
]

_REF_RADIUS_KM = 6371.2  # The magnetic models' reference radius.


def _dipole_terms(
    coeffs: MagneticCoefficients, year: float | None
) -> tuple[float, float, float]:
    dt = 0.0 if year is None else year - coeffs.epoch
    g = coeffs.g + dt * coeffs.g_dot
    h = coeffs.h + dt * coeffs.h_dot
    return float(g[1, 0]), float(g[1, 1]), float(h[1, 1])


def cd_rotation_matrix(
    g10: float | None = None,
    g11: float | None = None,
    h11: float | None = None,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
) -> NDArray[np.float64]:
    """
    Rotation taking ITRS vectors into centered-dipole coordinates.

    The CD +z axis points along the dipole axis toward the northern
    magnetic pole, derived from the degree-1 Gauss coefficients
    (either given directly or taken from ``coeffs`` at ``year``).

    Shared core of the MATLAB ``ITRS2CartCD``/``CartCD2ITRS`` pair.

    Parameters
    ----------
    g10, g11, h11 : float, optional
        Degree-1 Gauss coefficients. Either give all three or none.
    coeffs : MagneticCoefficients, optional
        Model supplying the coefficients when not given directly.
        Default IGRF14.
    year : float, optional
        Decimal year for secular variation. Default: the model epoch.

    Returns
    -------
    R : ndarray
        (3, 3) rotation matrix; ``R @ v_itrs`` is in CD coordinates.

    Examples
    --------
    >>> R = cd_rotation_matrix()
    >>> R.shape
    (3, 3)
    """
    if g10 is None:
        g10, g11, h11 = _dipole_terms(coeffs, year)
    pole = np.array([-g11, -h11, -g10], dtype=np.float64)
    pole = pole / np.linalg.norm(pole)
    phi_n = np.arctan2(pole[1], pole[0])
    theta_n = np.arccos(pole[2])  # colatitude of the CD pole
    cy, sy = np.cos(-theta_n), np.sin(-theta_n)
    cz, sz = np.cos(-phi_n), np.sin(-phi_n)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rz


def itrs2cart_cd(
    z_cart: ArrayLike,
    g10: float | None = None,
    g11: float | None = None,
    h11: float | None = None,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
) -> NDArray[np.float64]:
    """
    Rotate Cartesian ITRS positions into centered-dipole coordinates.

    Port of ``ITRS2CartCD``.

    Parameters
    ----------
    z_cart : array_like
        (3,) or (3, N) ITRS positions.
    g10, g11, h11, coeffs, year
        As in :func:`cd_rotation_matrix`.

    Returns
    -------
    z_cd : ndarray
        The positions in CD coordinates, same shape.

    Examples
    --------
    >>> z = itrs2cart_cd([6378137.0, 0.0, 0.0])
    >>> z.shape
    (3,)
    """
    r = cd_rotation_matrix(g10, g11, h11, coeffs, year)
    return r @ np.asarray(z_cart, dtype=np.float64)


def cart_cd2itrs(
    z_cd: ArrayLike,
    g10: float | None = None,
    g11: float | None = None,
    h11: float | None = None,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
) -> NDArray[np.float64]:
    """
    Rotate centered-dipole Cartesian positions back into the ITRS.

    Port of ``CartCD2ITRS``.

    Parameters
    ----------
    z_cd : array_like
        (3,) or (3, N) CD positions.
    g10, g11, h11, coeffs, year
        As in :func:`cd_rotation_matrix`.

    Returns
    -------
    z_itrs : ndarray
        The positions in the ITRS, same shape.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([6378137.0, 0.0, 0.0])
    >>> back = cart_cd2itrs(itrs2cart_cd(z))
    >>> bool(np.allclose(back, z))
    True
    """
    r = cd_rotation_matrix(g10, g11, h11, coeffs, year)
    return r.T @ np.asarray(z_cd, dtype=np.float64)


def _sphere2cart(z: NDArray[np.float64]) -> NDArray[np.float64]:
    r, az, el = z
    return r * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])


def _cart2sphere(v: NDArray[np.float64]) -> NDArray[np.float64]:
    r = np.linalg.norm(v)
    return np.array([r, np.arctan2(v[1], v[0]), np.arcsin(v[2] / r)])


def spher_itrs2spher_cd(
    z_spher: ArrayLike,
    g10: float | None = None,
    g11: float | None = None,
    h11: float | None = None,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
) -> NDArray[np.float64]:
    """
    Convert spherical ITRS coordinates to spherical CD coordinates.

    Port of ``spherITRS2SpherCD``.

    Parameters
    ----------
    z_spher : array_like
        (3,) [r, azimuth, elevation] or (2,) [azimuth, elevation]
        (angles in radians; unit radius assumed for the 2-element
        form).
    g10, g11, h11, coeffs, year
        As in :func:`cd_rotation_matrix`.

    Returns
    -------
    z_cd : ndarray
        The spherical CD coordinates, same number of elements.

    Examples
    --------
    >>> z = spher_itrs2spher_cd([6.4e6, 0.5, 0.2])
    >>> z.shape
    (3,)
    """
    z = np.asarray(z_spher, dtype=np.float64).reshape(-1)
    two = z.size < 3
    full = np.concatenate([[1.0], z]) if two else z
    out = _cart2sphere(itrs2cart_cd(_sphere2cart(full), g10, g11, h11, coeffs, year))
    return out[1:] if two else out


def spher_cd2spher_itrs(
    z_cd: ArrayLike,
    g10: float | None = None,
    g11: float | None = None,
    h11: float | None = None,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
) -> NDArray[np.float64]:
    """
    Convert spherical CD coordinates back to spherical ITRS.

    Port of ``spherCD2SpherITRS``.

    Parameters
    ----------
    z_cd : array_like
        (3,) [r, azimuth, elevation] or (2,) [azimuth, elevation].
    g10, g11, h11, coeffs, year
        As in :func:`cd_rotation_matrix`.

    Returns
    -------
    z_spher : ndarray
        The spherical ITRS coordinates, same number of elements.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([6.4e6, 0.5, 0.2])
    >>> back = spher_cd2spher_itrs(spher_itrs2spher_cd(z))
    >>> bool(np.allclose(back, z))
    True
    """
    z = np.asarray(z_cd, dtype=np.float64).reshape(-1)
    two = z.size < 3
    full = np.concatenate([[1.0], z]) if two else z
    out = _cart2sphere(cart_cd2itrs(_sphere2cart(full), g10, g11, h11, coeffs, year))
    return out[1:] if two else out


def _declination_at(
    lat: float,
    lon: float,
    h: float,
    coeffs: MagneticCoefficients,
    year: float | None,
) -> float:
    """Angle of magnetic north east of geographic north (radians)."""
    from pytcl.magnetism.wmm import wmm

    y = coeffs.epoch if year is None else year
    res = wmm(lat, lon, h / 1000.0, y, coeffs)  # wmm takes km
    return float(res.D)


def geog_heading2mag(
    point_lla: ArrayLike,
    heading: float,
    coeffs: MagneticCoefficients = WMM2025,
    year: float | None = None,
) -> float:
    """
    Convert a geographic heading to a magnetic heading.

    The magnetic heading is the geographic heading minus the local
    magnetic declination (the direction of the horizontal magnetic
    flux, east of geographic north).

    Port of ``geogHeading2Mag``, evaluated through pytcl's validated
    field models.

    Parameters
    ----------
    point_lla : array_like
        (3,) geodetic [latitude, longitude, height] in radians and
        meters (WGS-84).
    heading : float
        Geographic heading in radians east of north.
    coeffs : MagneticCoefficients, optional
        Field model. Default WMM2025 (the MATLAB default is the WMM).
    year : float, optional
        Decimal year. Default: the model epoch.

    Returns
    -------
    heading_mag : float
        Heading in radians east of magnetic north.

    Examples
    --------
    >>> h = geog_heading2mag([0.7, -1.2, 0.0], 0.5)
    >>> isinstance(h, float)
    True
    """
    lat, lon, h = np.asarray(point_lla, dtype=np.float64).reshape(-1)[:3]
    return float(heading - _declination_at(lat, lon, h, coeffs, year))


def mag_heading2geog(
    point_lla: ArrayLike,
    heading_mag: float,
    coeffs: MagneticCoefficients = WMM2025,
    year: float | None = None,
) -> float:
    """
    Convert a magnetic heading to a geographic heading.

    Port of ``magHeading2Geog``.

    Parameters
    ----------
    point_lla : array_like
        (3,) geodetic [latitude, longitude, height] in radians and
        meters (WGS-84).
    heading_mag : float
        Heading in radians east of magnetic north.
    coeffs, year
        As in :func:`geog_heading2mag`.

    Returns
    -------
    heading_geo : float
        Heading in radians east of geographic north.

    Examples
    --------
    >>> p = [0.7, -1.2, 0.0]
    >>> round(mag_heading2geog(p, geog_heading2mag(p, 0.5)), 12)
    0.5
    """
    lat, lon, h = np.asarray(point_lla, dtype=np.float64).reshape(-1)[:3]
    return float(heading_mag + _declination_at(lat, lon, h, coeffs, year))


def _b_cartesian(
    x_m: NDArray[np.float64],
    coeffs: MagneticCoefficients,
    year: float,
) -> NDArray[np.float64]:
    """Magnetic flux density in ITRS Cartesian components (nT)."""
    r_m = np.linalg.norm(x_m)
    lat_gc = np.arcsin(x_m[2] / r_m)
    lon = np.arctan2(x_m[1], x_m[0])
    b_r, b_theta, b_phi = _compute_magnetic_field_spherical_impl(
        lat_gc, lon, r_m / 1000.0, year, coeffs
    )
    st, ct = np.cos(lat_gc), np.sin(lat_gc)  # sin/cos of colatitude
    sp, cp = np.sin(lon), np.cos(lon)
    e_r = np.array([st * cp, st * sp, ct])
    e_theta = np.array([ct * cp, ct * sp, -st])
    e_phi = np.array([-sp, cp, 0.0])
    return b_r * e_r + b_theta * e_theta + b_phi * e_phi


def _up_vector(x_m: NDArray[np.float64]) -> NDArray[np.float64]:
    lat, lon, _ = (float(v) for v in ecef2geodetic(x_m))
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _height_of(x_m: NDArray[np.float64]) -> float:
    return float(ecef2geodetic(x_m)[2])


_INF_HEIGHT = 1e37  # The reference's declare-it-infinite threshold (m).


def trace2earth_mag_apex(
    z_cart: ArrayLike,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> tuple[NDArray[np.float64], float]:
    """
    Trace the magnetic field line through a point to its apex.

    Integration follows the normalized flux direction toward
    increasing ellipsoidal height until the height peaks; the apex is
    located by root-finding on the level condition. A field line that
    climbs beyond 1e37 m (near the magnetic poles) is declared to
    reach infinity, as in the reference.

    Port of ``trace2EarthMagApex`` (SciPy RK45 with event
    root-finding in place of the hand-rolled adaptive step).

    Parameters
    ----------
    z_cart : array_like
        (3,) ITRS position in meters.
    coeffs : MagneticCoefficients, optional
        Field model. Default IGRF14.
    year : float, optional
        Decimal year. Default: the model epoch.
    rel_tol, abs_tol : float, optional
        Integration tolerances. Defaults 1e-6 and 1e-9, the MATLAB
        defaults.

    Returns
    -------
    apex_point : ndarray
        (3,) ITRS position of the apex in meters (infinite entries
        for a polar-escaping line).
    sign_val : float
        +1 when the field at the start points toward increasing
        height (magnetically southern side), -1 otherwise.

    Examples
    --------
    >>> apex, sign = trace2earth_mag_apex([6.4e6, 1e5, 2e6])
    >>> apex.shape
    (3,)
    """
    x0 = np.asarray(z_cart, dtype=np.float64).reshape(3)
    y = coeffs.epoch if year is None else year

    def _dxds(_s: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        b = _b_cartesian(x, coeffs, y)
        return b / np.linalg.norm(b)

    sign_val = 1.0 if float(_up_vector(x0) @ _dxds(0.0, x0)) >= 0 else -1.0

    def _deriv(s: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return sign_val * _dxds(s, x)

    def _level(s: float, x: NDArray[np.float64]) -> float:
        return float(_up_vector(x) @ _deriv(s, x))

    def _escaped(_s: float, x: NDArray[np.float64]) -> float:
        return _height_of(x) - _INF_HEIGHT

    _level.terminal = True
    _level.direction = -1
    _escaped.terminal = True

    sol = solve_ivp(
        _deriv,
        (0.0, 1e40),
        x0,
        method="RK45",
        rtol=rel_tol,
        atol=abs_tol,
        events=(_level, _escaped),
        dense_output=True,
    )
    if sol.t_events[1].size:  # escaped toward infinity near a pole
        return np.full(3, np.inf), sign_val
    if not sol.t_events[0].size:
        raise RuntimeError("Field-line tracing did not reach an apex.")
    return np.asarray(sol.y_events[0][0], dtype=np.float64), sign_val


def _magnetic_potential(
    x_m: NDArray[np.float64],
    coeffs: MagneticCoefficients,
    year: float,
) -> float:
    """Scalar potential V in T*m (B = -grad V)."""
    r_m = np.linalg.norm(x_m)
    lat_gc = np.arcsin(x_m[2] / r_m)
    lon = np.arctan2(x_m[1], x_m[0])
    from pytcl.magnetism._schmidt import _schmidt_legendre

    n_max = coeffs.n_max
    dt = year - coeffs.epoch
    g = coeffs.g + dt * coeffs.g_dot
    h = coeffs.h + dt * coeffs.h_dot
    cos_theta = np.cos(np.pi / 2 - lat_gc)
    p = _schmidt_legendre(n_max, cos_theta)
    r_km = r_m / 1000.0
    v = 0.0
    for n in range(1, n_max + 1):
        r_pow = (_REF_RADIUS_KM / r_km) ** (n + 1)
        for m in range(n + 1):
            v += (
                r_pow
                * p[n, m]
                * (g[n, m] * np.cos(m * lon) + h[n, m] * np.sin(m * lon))
            )
    # nT * km -> T * m.
    return float(_REF_RADIUS_KM * v * 1e-6)


def itrs2magnetic_apex(
    z_cart: ArrayLike,
    h_r: float = 0.0,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert an ITRS position to (modified) magnetic apex coordinates.

    The apex latitude follows Richmond's definition from the apex
    height of the field line through the point; the longitude is the
    centered-dipole longitude of the apex; the third coordinate is
    the signed magnetic scalar potential at the point.

    Port of ``ITRS2MagneticApex``.

    Parameters
    ----------
    z_cart : array_like
        (3,) ITRS position in meters.
    h_r : float, optional
        Reference height in meters for modified apex coordinates.
        Default 0 (standard apex coordinates).
    coeffs, year, rel_tol, abs_tol
        As in :func:`trace2earth_mag_apex`.

    Returns
    -------
    z_apex : ndarray
        (3,) [apex latitude (rad), apex longitude (rad), potential
        (T m)]. The latitude is NaN when the field line's apex sits
        below ``h_r`` (the modified-apex definition leaves it
        undefined there); the longitude is NaN for a polar-escaping
        line.
    apex_point : ndarray
        (3,) the traced apex position in the ITRS (meters).

    Examples
    --------
    >>> z, apex = itrs2magnetic_apex([6.4e6, 1e5, 2e6])
    >>> z.shape
    (3,)
    """
    x0 = np.asarray(z_cart, dtype=np.float64).reshape(3)
    y = coeffs.epoch if year is None else year
    a = EARTH_SEMI_MAJOR_AXIS
    apex_point, sign_val = trace2earth_mag_apex(x0, coeffs, y, rel_tol, abs_tol)
    if not np.all(np.isfinite(apex_point)):
        h_a = np.inf
        phi = np.nan
    else:
        h_a = _height_of(apex_point)
        g10, g11, h11 = _dipole_terms(coeffs, y)
        phi = float(_cart2sphere(itrs2cart_cd(apex_point, g10, g11, h11))[1])
    with np.errstate(invalid="ignore"):
        lambda_a = -sign_val * np.arccos(np.sqrt((a + h_r) / (a + h_a)))
    v_coord = -sign_val * _magnetic_potential(x0, coeffs, y)
    return np.array([lambda_a, phi, v_coord]), apex_point


def itrs2qd(
    z_cart: ArrayLike,
    coeffs: MagneticCoefficients = IGRF14,
    year: float | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert an ITRS position to quasi-dipole coordinates.

    The QD latitude uses the point's own ellipsoidal height against
    the apex height; the longitude is the centered-dipole longitude
    of the apex.

    Port of ``ITRS2QD``.

    Parameters
    ----------
    z_cart : array_like
        (3,) ITRS position in meters.
    coeffs, year, rel_tol, abs_tol
        As in :func:`trace2earth_mag_apex`.

    Returns
    -------
    z_qd : ndarray
        (3,) [QD latitude (rad), QD longitude (rad), ellipsoidal
        height (m)]. The longitude is NaN for a polar-escaping line.
    apex_point : ndarray
        (3,) the traced apex position in the ITRS (meters).

    Examples
    --------
    >>> z, apex = itrs2qd([6.4e6, 1e5, 2e6])
    >>> z.shape
    (3,)
    """
    x0 = np.asarray(z_cart, dtype=np.float64).reshape(3)
    y = coeffs.epoch if year is None else year
    a = EARTH_SEMI_MAJOR_AXIS
    apex_point, sign_val = trace2earth_mag_apex(x0, coeffs, y, rel_tol, abs_tol)
    h = _height_of(x0)
    if not np.all(np.isfinite(apex_point)):
        h_a = np.inf
        phi = np.nan
    else:
        h_a = _height_of(apex_point)
        g10, g11, h11 = _dipole_terms(coeffs, y)
        phi = float(_cart2sphere(itrs2cart_cd(apex_point, g10, g11, h11))[1])
    lambda_qd = -sign_val * np.arccos(np.sqrt((a + h) / (a + h_a)))
    return np.array([lambda_qd, phi, h]), apex_point
