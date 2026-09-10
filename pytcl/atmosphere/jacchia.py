"""
Jacchia 1971 reference atmosphere.

Port of ``jacchiaAtmosParam``: exospheric temperature from solar flux,
geomagnetic activity, and the diurnal bulge; the empirical temperature
profile; and density from Montenbruck & Gill's bi-polynomial fit with
the semiannual, geomagnetic, and seasonal-latitude corrections.

Known deviations from the MATLAB original, all deliberate and loud:

- The MATLAB function is unrunnable as shipped: it references three
  Earth-orientation variables (``deltaTTUT1``, ``xpyp``, ``dXdY``) that
  are never assigned. The oracle fixtures were captured with a patched
  copy that takes the solar declination and local hour angle as inputs
  (see ``scripts/matlab_capture/capture_jacchia.m``); this port computes
  them with pytcl's own time-scale chain and a low-precision solar
  position (accurate to ~0.01 degrees, far below the sensitivity of the
  diurnal term).
- MATLAB computes pressure as ``rho * R * T / M`` with the molar mass in
  g/mol, which yields kPa while documenting Pa. This port returns true
  pascals (a factor of 1000 larger than MATLAB's number).
- The seasonal-latitude correction's ``sin(lat)^3 / |sin(lat)|`` is
  0/0 at the equator; MATLAB silently returns NaN there. This port uses
  the limit value 0 (the factor is ``sign(sin lat) * sin^2 lat``).
- The mean-molecular-mass polynomial behind the pressure output is a
  fit for the 90-100 km mixing region; at higher altitudes it diverges
  and MATLAB silently returns a negative pressure. This port emits a
  ``RuntimeWarning`` and returns NaN pressure where the polynomial's
  molar mass is not positive.

References
----------
- L. G. Jacchia, "Revised Static Models of the Thermosphere and
  Exosphere with Empirical Temperature Profiles," SAO Special Report
  332, Cambridge, MA, 1971.
- O. Montenbruck and E. Gill, Satellite Orbits: Models, Methods and
  Applications, Springer, 2000 (Tables 3.9 and 3.10).
"""

import warnings
from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike

from pytcl.core.constants import UNIVERSAL_GAS_CONSTANT

# Mean-molecular-mass profile coefficients (powers of Z - 90 km, g/mol).
_CN = np.array(
    [
        28.82678,
        -7.40066e-2,
        -1.19407e-2,
        4.51103e-4,
        -8.21895e-6,
        1.07561e-5,
        -6.97444e-7,
    ]
)

_T0 = 183.0  # Boundary temperature at 90 km, K.
_Z0 = 90.0  # Boundary altitude, km.
_ZX = 125.0  # Inflection altitude, km.

# The eight 6x5 bi-polynomial coefficient tables of Montenbruck & Gill
# (Tables 3.9/3.10), keyed by the (T < 850, altitude band) tree of the
# MATLAB original. Transcribed from jacchiaAtmosParam.m, which itself
# warns the values "were entered manually and may include some typing
# errors" -- the oracle fixtures pin this transcription to MATLAB's.
_CIJ_LOW_T = {
    "z90_180": np.array(
        [
            [-0.3520856e2, 0.3912622e1, -0.8649259e2, 0.1504119e3, -0.7109428e2],
            [0.1129210e4, 0.1198158e4, 0.8633794e3, -0.3577091e4, 0.1970558e4],
            [-0.1527475e5, -0.3558481e5, 0.1899243e5, 0.2508241e5, -0.1968253e5],
            [0.9302042e5, 0.3646554e6, -0.3290364e6, -0.1209631e5, 0.8438137e5],
            [-0.2734394e6, -0.1576097e7, 0.1685831e7, -0.4282943e6, -0.1345593e6],
            [0.3149696e6, 0.2487723e7, -0.2899124e7, 0.1111904e7, 0.3294095e4],
        ]
    ),
    "z180_500": np.array(
        [
            [0.2311910e2, 0.1355298e3, -0.8424310e3, 0.1287331e4, -0.6181209e3],
            [-0.1057776e4, 0.6087973e3, 0.8690566e4, -0.1715922e5, 0.9052671e4],
            [0.1177230e5, -0.3164132e5, -0.1076323e4, 0.6302629e5, -0.4312459e5],
            [-0.5827663e5, 0.2188167e6, -0.2422912e6, 0.2461286e5, 0.6044096e5],
            [0.1254589e6, -0.5434710e6, 0.8123016e6, -0.4490438e6, 0.5007458e5],
            [-0.9452922e5, 0.4408026e6, -0.7379410e6, 0.5095273e6, -0.1154192e6],
        ]
    ),
    "z500_1000": np.array(
        [
            [-0.1815722e4, 0.9792972e4, -0.1831374e5, 0.1385255e5, -0.3451234e4],
            [0.9851221e4, -0.5397525e5, 0.9993169e5, -0.7259456e5, 0.1622553e5],
            [-0.1822932e5, 0.1002430e6, -0.1784481e6, 0.1145178e6, -0.1641934e5],
            [0.1298113e5, -0.7113430e5, 0.1106375e6, -0.3825777e5, -0.1666915e5],
            [-0.1533510e4, 0.7815537e4, 0.7037562e4, -0.4674636e5, 0.3516946e5],
            [-0.1263680e4, 0.7265792e4, -0.2092909e5, 0.2936094e5, -0.1491676e5],
        ]
    ),
    "z1000_2500": np.array(
        [
            [0.3548698e3, -0.2508685e4, 0.6252742e4, -0.6755376e4, 0.2675763e4],
            [-0.5370852e3, 0.4182586e4, -0.1151114e5, 0.1338915e5, -0.5610580e4],
            [-0.2349586e2, -0.8941841e3, 0.4417927e4, -0.6732817e4, 0.3312608e4],
            [0.3407073e3, -0.1531588e4, 0.2179045e4, -0.8841341e3, -0.1369769e3],
            [-0.1698470e3, 0.8985697e3, -0.1704797e4, 0.1363098e4, -0.3812417e3],
            [0.2494943e2, -0.1389618e3, 0.2820058e3, -0.2472862e3, 0.7896439e2],
        ]
    ),
}
_CIJ_HIGH_T = {
    "z90_180": np.array(
        [
            [-0.5335412e2, 0.2900557e2, -0.2046439e2, 0.7977149e1, -0.1335853e1],
            [0.1977533e4, -0.7091478e3, 0.4398538e3, -0.1568720e3, 0.2615466e2],
            [-0.2993620e5, 0.5187286e4, -0.1989795e4, 0.3643166e3, -0.5700669e2],
            [0.2112068e6, -0.4483029e4, -0.1349971e5, 0.9510012e4, -0.1653725e4],
            [-0.7209722e6, -0.7684101e5, 0.1256236e6, -0.6805699e5, 0.1181257e5],
            [0.9625966e6, 0.2123127e6, -0.2622793e6, 0.1337130e6, -0.2329995e5],
        ]
    ),
    "z180_500": np.array(
        [
            [0.4041761e2, -0.1305719e3, 0.1466809e3, -0.7120296e2, 0.1269605e2],
            [-0.8127720e3, 0.2273565e4, -0.2577261e4, 0.1259045e4, -0.2254978e3],
            [0.5130043e4, -0.1501308e5, 0.1717142e5, -0.8441698e4, 0.1518796e4],
            [-0.1600170e5, 0.4770469e5, -0.5473492e5, 0.2699668e5, -0.4870306e4],
            [0.2384718e5, -0.7199064e5, 0.8284653e5, -0.4098358e5, 0.7411926e4],
            [-0.1363104e5, 0.4153499e5, -0.4793581e5, 0.2377854e5, -0.4310233e4],
        ]
    ),
    "z500_1000": np.array(
        [
            [-0.4021335e2, -0.1326983e3, 0.3778864e3, -0.2808660e3, 0.6513531e2],
            [0.4255789e3, 0.3528126e3, -0.2077888e4, 0.1726543e4, -0.4191477e3],
            [-0.1821662e4, 0.7905357e3, 0.3934271e4, -0.3969334e4, 0.1027991e4],
            [0.3070231e4, -0.2941540e4, -0.3276639e4, 0.4420217e4, -0.1230778e4],
            [-0.2196848e4, 0.2585118e4, 0.1382776e4, -0.2533006e4, 0.7451387e3],
            [0.5494959e3, -0.6604225e3, -0.3328077e3, 0.6335703e3, -0.1879812e3],
        ]
    ),
    "z1000_2500": np.array(
        [
            [0.1281061e2, -0.3389179e3, 0.6861935e3, -0.4667627e3, 0.1029662e3],
            [0.2024251e3, 0.1668302e3, -0.1147876e4, 0.9918940e3, -0.2430215e3],
            [-0.5750743e3, 0.8259823e3, 0.2329832e3, -0.6503359e3, 0.1997989e3],
            [0.5106207e3, -0.1032012e4, 0.4851874e3, 0.8214097e2, -0.6527048e2],
            [-0.1898953e3, 0.4347501e3, -0.2986011e3, 0.5423180e2, 0.5039459e1],
            [0.2569577e2, -0.6282710e2, 0.4971077e2, -0.1404385e2, 0.8450500e0],
        ]
    ),
}


class JacchiaState(NamedTuple):
    """Jacchia 1971 atmospheric state.

    Attributes
    ----------
    density : float
        Total mass density, kg/m^3.
    pressure : float
        Pressure in Pa (ideal-gas law; the MATLAB original returns kPa
        while documenting Pa -- see the module docstring).
    temperature : float
        Temperature at the requested altitude, K.
    exospheric_temperature : float
        Exospheric temperature, K.
    """

    density: float
    pressure: float
    temperature: float
    exospheric_temperature: float


def _get_coeff(z_km: float, te: float) -> np.ndarray:
    """Select the Montenbruck & Gill bi-polynomial table."""
    if z_km < 90 or z_km > 2500:
        raise ValueError("altitude must be within 90 km to 2500 km")
    if te < 500 or te > 1900:
        raise ValueError("exospheric temperature must be within 500 K to 1900 K")
    table = _CIJ_LOW_T if te < 850 else _CIJ_HIGH_T
    if z_km < 180:
        return table["z90_180"]
    if z_km < 500:
        return table["z180_500"]
    if z_km < 1000:
        return table["z500_1000"]
    return table["z1000_2500"]


def _jacchia_from_sun_geometry(
    jul1: float,
    jul2: float,
    lat: float,
    z_km: float,
    f107: float,
    f107b: float,
    kp: float,
    sun_dec: float,
    local_hour_angle: float,
) -> Tuple[float, float, float, float]:
    """The Jacchia 1971 model proper, with the solar geometry given.

    This is the part the MATLAB oracle fixtures pin bit-tight: the
    capture script feeds ``sun_dec``/``local_hour_angle`` explicitly
    because the original's internal astro chain is unrunnable as
    shipped (see the module docstring).
    """
    # Exospheric temperature: solar flux, diurnal bulge, geomagnetic.
    tc = 379.0 + 3.24 * f107b + 1.3 * (f107 - f107b)

    beta = np.radians(-37.0)
    p = np.radians(6.0)
    gamma = np.radians(43.0)
    m = 2.2
    n = 3.0
    r = 0.3

    eta = 0.5 * abs(lat - sun_dec)
    theta = 0.5 * abs(lat + sun_dec)
    tau = local_hour_angle + beta + p * np.sin(local_hour_angle + gamma)
    tau = np.mod(tau + np.pi, 2 * np.pi) - np.pi
    tl = tc * (
        1
        + r
        * (
            np.sin(theta) ** m
            + (np.cos(eta) ** m - np.sin(theta) ** m) * np.cos(tau / 2) ** n
        )
    )

    t_high = 28 * kp + 0.03 * np.exp(kp)  # above 350 km
    t_low = 14 * kp + 0.02 * np.exp(kp)  # below 350 km
    f = 0.5 * (np.tanh(0.04 * (z_km - 350)) + 1)
    tg = f * t_high + (1 - f) * t_low
    te = tg + tl

    # Temperature profile.
    tx = 371.668 + 0.0518806 * te - 294.3505 * np.exp(-0.00216222 * te)
    A = 2 * (te - tx) / np.pi
    B = 4.5e-6
    t1 = 1.9 * (tx - _T0) / (_ZX - _Z0)
    t3 = -1.7 * (tx - _T0) / (_ZX - _Z0) ** 3
    t4 = -0.8 * (tx - _T0) / (_ZX - _Z0) ** 4
    dz = z_km - _ZX
    if z_km > _ZX:
        temp = tx + A * np.arctan(t1 * dz * (1 + B * dz**2.5) / A)
    else:
        temp = tx + t1 * dz + t3 * dz**3 + t4 * dz**4

    # Standard density from the bi-polynomial fit.
    cij = _get_coeff(z_km, te)
    zi = (z_km / 1000.0) ** np.arange(6)
    tj = (te / 1000.0) ** np.arange(5)
    log_rho = float(zi @ cij @ tj)

    # Semiannual correction (tropical years since 1958-01-01).
    phi = (jul1 + jul2 - 2400000.5 - 36204) / 365.2422
    tau_sa = phi + 0.09544 * (
        (0.5 + 0.5 * np.sin(2 * np.pi * phi + 6.035)) ** 1.65 - 0.5
    )
    f_z = (5.876e-7 * z_km**2.331 + 0.06328) * np.exp(z_km * -2.868e-3)
    g_t = 0.02835 + 0.3817 * (1 + 0.4671 * np.sin(2 * np.pi * tau_sa + 4.137)) * np.sin(
        4 * np.pi * tau_sa + 4.259
    )
    log_rho += f_z * g_t

    # Geomagnetic correction (below 350 km only).
    if z_km < 350:
        log_rho += (0.012 * kp + 1.2e-5 * np.exp(kp)) * (1 - f)

    # Seasonal-latitude correction. sin^3(lat)/|sin(lat)| ==
    # sign(sin lat) * sin^2(lat); the MATLAB form is NaN at the equator,
    # where the limit (and this port's value) is 0.
    sin_lat = np.sin(lat)
    log_rho += (
        0.014
        * (z_km - 90)
        * np.exp(-0.0013 * (z_km - 90) ** 2)
        * np.sin(np.pi * phi - 1.72)
        * np.sign(sin_lat)
        * sin_lat**2
    )

    rho = 10.0**log_rho

    # Pressure via the mean molecular mass polynomial. M is in g/mol;
    # dividing by (M * 1e-3) kg/mol yields true pascals -- the MATLAB
    # original divides by M in g/mol and returns kPa while documenting
    # Pa (see the module docstring). The polynomial is a fit for the
    # mixing region near the 90 km boundary: at high altitude it
    # diverges (negative "molar mass"), where the MATLAB original
    # silently returns a negative pressure. This port warns and
    # returns NaN there instead.
    mmass = float(np.polyval(_CN[::-1], z_km - 90))
    if mmass > 0:
        pressure = rho * UNIVERSAL_GAS_CONSTANT * temp / (mmass * 1e-3)
    else:
        warnings.warn(
            "jacchia_atmos_param: the mean-molecular-mass polynomial is "
            f"unphysical at {z_km:.0f} km (M = {mmass:.1f} g/mol); the "
            "pressure output is meaningless there (the MATLAB original "
            "silently returns a negative value) and is set to NaN. "
            "Density and temperature are unaffected.",
            RuntimeWarning,
            stacklevel=3,
        )
        pressure = float("nan")

    return float(rho), float(pressure), float(temp), float(te)


def _sun_ra_dec(jd_tt: float) -> Tuple[float, float]:
    """Low-precision apparent solar RA/declination of date (radians).

    Standard almanac polynomial expressions, accurate to about 0.01
    degrees -- far below the sensitivity of Jacchia's diurnal-bulge
    term to the solar direction.
    """
    n = jd_tt - 2451545.0
    L = np.radians(np.mod(280.460 + 0.9856474 * n, 360.0))
    g = np.radians(np.mod(357.528 + 0.9856003 * n, 360.0))
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    return float(ra), float(dec)


def jacchia_atmos_param(
    jul1: float,
    jul2: float,
    point: ArrayLike,
    f107: float,
    f107b: float,
    kp: float,
) -> JacchiaState:
    """
    Jacchia 1971 atmospheric density, pressure, and temperature.

    Port of ``jacchiaAtmosParam``. Valid for altitudes of 90 km to
    2500 km and exospheric temperatures of 500 K to 1900 K (both
    enforced, as in the original).

    Parameters
    ----------
    jul1, jul2 : float
        Two-part Julian date in UTC (days); the full date is the sum.
    point : array_like
        Geodetic ``[latitude, longitude, altitude]`` with latitude and
        longitude in radians and altitude in meters.
    f107 : float
        10.7 cm solar radio flux averaged over the previous day, in
        units of 1e-22 W/(m^2 Hz).
    f107b : float
        Same flux averaged over three 27-day solar rotations.
    kp : float
        Three-hourly planetary geomagnetic index for a time 6.7 hours
        before the epoch.

    Returns
    -------
    state : JacchiaState
        Density (kg/m^3), pressure (Pa), temperature (K), and
        exospheric temperature (K).

    Raises
    ------
    ValueError
        If the altitude or the resulting exospheric temperature is
        outside the model's tabulated domain.

    Examples
    --------
    >>> import numpy as np
    >>> state = jacchia_atmos_param(
    ...     2451545.0, 0.25, [np.radians(40.0), np.radians(-75.0), 400e3],
    ...     150.0, 150.0, 3.0,
    ... )
    >>> 1e-13 < state.density < 1e-11  # ~ISS-altitude density
    True
    >>> 600.0 < state.exospheric_temperature < 1400.0
    True
    """
    point = np.asarray(point, dtype=np.float64).ravel()
    lat, lon, alt = float(point[0]), float(point[1]), float(point[2])
    z_km = alt / 1000.0

    # Solar geometry from pytcl's own chain: UTC -> TT, apparent solar
    # RA/dec of date, and the local apparent sidereal time for the hour
    # angle (lha = LAST - RA).
    from pytcl.astronomical.reference_frames import gast_iau82
    from pytcl.astronomical.time_systems import get_leap_seconds, jd_to_cal

    year, month, day, *_ = jd_to_cal(jul1 + jul2)
    tai_utc = get_leap_seconds(int(year), int(month), int(day))
    jd_tt = jul1 + jul2 + (tai_utc + 32.184) / 86400.0
    ra, dec = _sun_ra_dec(jd_tt)
    last = gast_iau82(jul1 + jul2, jd_tt) + lon
    lha = np.mod(last - ra + np.pi, 2 * np.pi) - np.pi

    rho, pressure, temp, te = _jacchia_from_sun_geometry(
        jul1, jul2, lat, z_km, f107, f107b, kp, dec, lha
    )
    return JacchiaState(rho, pressure, temp, te)


__all__ = [
    "JacchiaState",
    "jacchia_atmos_param",
]
