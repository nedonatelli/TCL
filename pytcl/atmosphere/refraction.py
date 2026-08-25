"""
Atmospheric refractivity for radar and optical propagation.

Ports from the MATLAB Tracker Component Library's
``Atmosphere_and_Refraction`` directory: refractivity helpers,
astronomical refraction (add/remove with the Sinclair atmosphere model),
and the standard-exponential-model radar refraction suite (bistatic
r-u-v ray tracing, bias approximation, cubature-based Gaussian
conversions).

Conventions
-----------
Refractivity is ``N = (n - 1) * 1e6`` where ``n`` is the index of
refraction. The standard exponential atmosphere model takes the
refractivity at height ``h`` above sea level to be
``N = Ns * exp(-ce * h)`` where ``Ns`` is the sea-level refractivity and
``ce`` is the decay constant returned by :func:`atmos_exp_decay_const`.
"""

import warnings
from typing import NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import IntegrationWarning, quad, solve_bvp, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import erf

from pytcl.atmosphere.humidity import H2O_MOLAR_MASS
from pytcl.core.constants import (
    EARTH_ECCENTRICITY_SQ,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
    STANDARD_ATMOSPHERE,
    STANDARD_RELATIVE_HUMIDITY,
    STANDARD_TEMPERATURE,
    UNIVERSAL_GAS_CONSTANT,
)
from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    cubature_point_moments,
    fifth_order_cubature_points,
    transform_cubature_points,
)
from pytcl.navigation.geodesy import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    osculating_sphere,
)

__all__ = [
    "AstroRefParams",
    "AstroRefractionResult",
    "CubatureConversionResult",
    "ExpDecayConstResult",
    "RuvStdRefracResult",
    "SinclairAtmosResult",
    "StdRefracBiasResult",
    "add_astro_refraction",
    "approx_refractivity",
    "atmos_exp_decay_const",
    "cart2ruv_std_refrac",
    "cart2ruv_std_refrac_cubature",
    "reduce_std_refrac_to_sphere",
    "remove_astro_refraction",
    "ruv2cart_std_refrac",
    "ruv2cart_std_refrac_cubature",
    "simple_astro_ref_params",
    "sinclair_atmosphere",
    "std_refrac_bias_approx",
]


class ExpDecayConstResult(NamedTuple):
    """
    Exponential-atmosphere decay constant and 1-km refractivity change.

    Attributes
    ----------
    ce : float or ndarray
        Decay constant of the refractivity in inverse meters.
    delta_n : float or ndarray
        Change in refractivity going 1 km up from sea level (negative).
    """

    ce: Union[float, NDArray[np.floating]]
    delta_n: Union[float, NDArray[np.floating]]


def approx_refractivity(
    temperature: ArrayLike,
    pressure: ArrayLike,
    water_vapor_pressure: ArrayLike,
) -> Union[float, NDArray[np.floating]]:
    """
    Approximate the refractivity of air from temperature and pressure.

    Implements Equation 6 in Annex I of ITU-R P.453-11. The
    approximation does not depend on frequency.

    Port of ``approxRefractivity.m``.

    References
    ----------
    - International Telecommunication Union, "Recommendation ITU-R
      P.453-11: The radio refractive index: Its formula and refractivity
      data," Tech. Rep., Jul. 2015.

    Parameters
    ----------
    temperature : array_like
        Temperature(s) in Kelvin.
    pressure : array_like
        Total atmospheric pressure(s) in Pascals (dry pressure plus the
        partial pressure of water vapor).
    water_vapor_pressure : array_like
        Partial pressure(s) of water vapor in Pascals.

    Returns
    -------
    refractivity : float or ndarray
        The refractivity ``N = (n - 1) * 1e6`` of the atmosphere.

    Examples
    --------
    >>> round(float(approx_refractivity(288.15, 101325.0, 853.3)), 4)
    311.2452
    """
    T = np.asarray(temperature, dtype=np.float64)
    # Pascals -> hectopascals.
    P = np.asarray(pressure, dtype=np.float64) / 100.0
    Pw = np.asarray(water_vapor_pressure, dtype=np.float64) / 100.0
    N = 77.6 * (P / T) - 5.6 * (Pw / T) + 3.75e5 * (Pw / T**2)
    return float(N) if N.ndim == 0 else N


def atmos_exp_decay_const(
    ns: ArrayLike,
) -> ExpDecayConstResult:
    """
    Decay constant of the exponential refractivity model.

    Given the refractivity of air at sea level, obtain the approximate
    decay constant for refractivity as a function of height per Appendix A
    of the CRPL Exponential Reference Atmosphere, along with the
    change in refractivity 1 km above sea level. The refractivity at
    height ``h`` above sea level is then ``N = ns * exp(-ce * h)``.

    Port of ``atmosExpDecayConst4Refrac.m``.

    References
    ----------
    - B. R. Bean and G. D. Thayer, CRPL Exponential Reference Atmosphere.
      Washington, D.C.: U.S. Department of Commerce, National Bureau of
      Standards, Oct. 1959.

    Parameters
    ----------
    ns : array_like
        Refractivity of air at sea level.

    Returns
    -------
    result : ExpDecayConstResult
        Named tuple of ``ce`` (decay constant, inverse meters) and
        ``delta_n`` (refractivity change 1 km up from sea level), each
        with the same shape as ``ns``.

    Examples
    --------
    >>> ce, delta_n = atmos_exp_decay_const(313.0)
    >>> print(f"{ce:.6e}")
    1.438586e-04
    >>> round(delta_n, 4)
    -41.9388
    """
    ns_arr = np.asarray(ns, dtype=np.float64)
    delta_n = -7.32 * np.exp(0.005577 * ns_arr)
    ce = np.log(ns_arr / (ns_arr + delta_n)) / 1e3
    if ce.ndim == 0:
        return ExpDecayConstResult(float(ce), float(delta_n))
    return ExpDecayConstResult(ce, delta_n)


class AstroRefParams(NamedTuple):
    """
    Constants of the ``A*tan(z) + B*tan^3(z)`` astronomical refraction model.

    Attributes
    ----------
    a : float
        The tan(z) coefficient in radians.
    b : float
        The tan^3(z) coefficient in radians.
    """

    a: float
    b: float


class SinclairAtmosResult(NamedTuple):
    """
    Atmospheric parameters of the Sinclair model at the queried heights.

    Attributes
    ----------
    n : float or ndarray
        Index of refraction.
    dndr : float or ndarray
        Derivative of the index of refraction with respect to height,
        in inverse meters.
    temperature : float or ndarray
        Temperature in Kelvin.
    pressure : float or ndarray
        Pressure in Pascals.
    """

    n: Union[float, NDArray[np.floating]]
    dndr: Union[float, NDArray[np.floating]]
    temperature: Union[float, NDArray[np.floating]]
    pressure: Union[float, NDArray[np.floating]]


class AstroRefractionResult(NamedTuple):
    """
    A zenith distance with the refraction correction that produced it.

    Attributes
    ----------
    zenith_distance : float or ndarray
        The converted zenith distance(s) in radians: refraction-free for
        :func:`remove_astro_refraction`, refraction-corrupted for
        :func:`add_astro_refraction`. An empty array signals inputs
        outside the algorithm's validity region (matching the MATLAB
        functions, which return empty matrices).
    delta_z : float or ndarray
        The refraction correction in radians that was applied, with
        ``z_true = z_observed + delta_z``.
    """

    zenith_distance: Union[float, NDArray[np.floating]]
    delta_z: Union[float, NDArray[np.floating]]


def simple_astro_ref_params(
    rel_humid: float = STANDARD_RELATIVE_HUMIDITY,
    pressure: float = STANDARD_ATMOSPHERE,
    temperature: float = STANDARD_TEMPERATURE,
    wavelength: float = 0.574e-6,
) -> AstroRefParams:
    """
    Constants for the simple ``A*tan(z) + B*tan^3(z)`` refraction model.

    Port of ``simpAstroRefParam.m`` (a MEX wrapper in MATLAB). This
    function uses computations derived from the IAU SOFA ``refco``
    routine; it is not itself software provided by or endorsed by SOFA.
    It differs from the original in taking SI inputs (Pascals, Kelvin,
    meters) and converting internally to the hPa/Celsius/micrometer
    units the fit was built for.

    Parameters
    ----------
    rel_humid : float, optional
        Relative humidity at the observer as a fraction in [0, 1].
        Default 0.
    pressure : float, optional
        Atmospheric pressure at the observer in Pascals. Default
        101325 Pa.
    temperature : float, optional
        Air temperature at the observer in Kelvin. Default 288.15 K.
    wavelength : float, optional
        Observation wavelength in meters; values above 100 micrometers
        select the radio-frequency fit instead of the optical/IR one.
        Default 0.574e-6 m (yellow light).

    Returns
    -------
    params : AstroRefParams
        The coefficients ``a`` and ``b`` in radians.

    Examples
    --------
    >>> a, b = simple_astro_ref_params(0.5, 101325.0, 288.15)
    >>> print(f"{a:.9e} {b:.9e}")
    2.767559220e-04 -3.167276124e-07
    """
    # SOFA refco restricts inputs to safe ranges before evaluating.
    tc = min(max(temperature - 273.15, -150.0), 200.0)
    phpa = min(max(pressure / 100.0, 0.0), 10000.0)
    rh = min(max(rel_humid, 0.0), 1.0)
    wl_um = min(max(wavelength * 1e6, 0.1), 1e6)

    optical = wl_um <= 100.0

    if phpa > 0.0:
        ps = 10.0 ** ((0.7859 + 0.03477 * tc) / (1.0 + 0.00412 * tc)) * (
            1.0 + phpa * (4.5e-6 + 6e-10 * tc * tc)
        )
        pw = rh * ps / (1.0 - (1.0 - rh) * ps / phpa)
    else:
        pw = 0.0

    tk = tc + 273.15
    if optical:
        wlsq = wl_um * wl_um
        gamma = (
            (77.53484e-6 + (4.39108e-7 + 3.666e-9 / wlsq) / wlsq) * phpa
            - 11.2684e-6 * pw
        ) / tk
    else:
        gamma = (77.6890e-6 * phpa - (6.3938e-6 - 0.375463 / tk) * pw) / tk

    beta = 4.4474e-6 * tk
    if not optical:
        beta -= 0.0074 * pw * beta

    return AstroRefParams(gamma * (1.0 - beta), -gamma * (beta - gamma / 2.0))


def sinclair_atmosphere(
    height: ArrayLike,
    obs_lat_lon_alt: ArrayLike,
    rel_humid: float = STANDARD_RELATIVE_HUMIDITY,
    pressure: float = STANDARD_ATMOSPHERE,
    temperature: float = STANDARD_TEMPERATURE,
    wavelength: float = 0.574e-6,
    tropopause_height: float = 11000.0,
) -> SinclairAtmosResult:
    """
    Atmospheric parameters for the Sinclair refraction model.

    A two-layer troposphere/stratosphere model of the index of refraction
    and its height derivative, used by algorithm 0 of
    :func:`remove_astro_refraction`. Follows Chapter 7.2 of Hohenkerk's
    treatment in the Explanatory Supplement and the 1982 HM Nautical
    Almanac Office technical note.

    Port of ``SinclairAtmos.m``.

    Parameters
    ----------
    height : array_like
        Height(s) above the reference ellipsoid in meters at which to
        evaluate the model.
    obs_lat_lon_alt : array_like
        The observer's WGS-84 ``[latitude, longitude, height]`` with
        angles in radians and height in meters. Only the latitude and
        height are read.
    rel_humid : float, optional
        Relative humidity at the observer in [0, 1]. Default 0.
    pressure : float, optional
        Pressure at the observer in Pascals. Default 101325 Pa.
    temperature : float, optional
        Temperature at the observer in Kelvin. Default 288.15 K.
    wavelength : float, optional
        Observation wavelength in meters. Default 0.574e-6 m.
    tropopause_height : float, optional
        Assumed top of the troposphere in meters. Default 11000 m.

    Returns
    -------
    result : SinclairAtmosResult
        Named tuple of ``n``, ``dndr``, ``temperature`` and ``pressure``
        at each queried height.

    References
    ----------
    - C. Y. Hohenkerk and A. T. Sinclair, "The computation of angular
      atmospheric refraction at large zenith angles," HM Nautical Almanac
      Office, Tech. Rep. NAO TN No. 63, Apr. 1985.

    Examples
    --------
    >>> obs = [0.61, 0.0, 100.0]
    >>> res = sinclair_atmosphere(1000.0, obs, 0.5, 101325.0, 288.15)
    >>> print(f"{res.n:.9f} {res.temperature:.2f}")
    1.000254179 282.30
    """
    h = np.asarray(height, dtype=np.float64)
    obs = np.asarray(obs_lat_lon_alt, dtype=np.float64)
    h0 = obs[2]

    # Geocentric (spherical) latitude of the observer.
    phi = np.arctan((1 - EARTH_ECCENTRICITY_SQ) * np.tan(obs[0]))

    # Pascals -> millibars; meters -> micrometers.
    p0_mb = pressure * 0.01
    lam = wavelength * 1e6

    # Constants from Equation 7.82 of the reference.
    r_gas = 1000.0 * UNIVERSAL_GAS_CONSTANT  # J/(kmol K)
    md = 28.966  # Molecular mass of dry air in amu.
    mw = H2O_MOLAR_MASS
    # Exponent of the temperature dependence of water vapor pressure.
    delta = 18.36
    alpha = 0.0065  # Tropospheric temperature lapse rate in K/m.

    # Equation 7.83: partial pressure of water vapor at the observer (mb).
    pw0 = rel_humid * (temperature / 247.1) ** delta
    g_bar = 9.784 * (1 - 0.0026 * np.cos(2 * phi) - 0.00000028 * h0)
    a_const = (287.604 + 1.6288 / lam**2 + 0.0136 / lam**4) * (273.15 / 1013.25) * 1e-6
    c2 = g_bar * md / r_gas
    gamma = c2 / alpha
    c5 = pw0 * (1 - mw / md) * gamma / (delta - gamma)
    c6 = a_const * (p0_mb + c5) / temperature
    c7 = (a_const * c5 + 11.2684e-6 * pw0) / temperature
    c8 = alpha * (gamma - 1) * c6 / temperature
    c9 = alpha * (delta - 1) * c7 / temperature

    n = np.zeros_like(h)
    dndr = np.zeros_like(h)
    t_out = np.zeros_like(h)
    p_out = np.zeros_like(h)

    ht = tropopause_height
    strat = h > ht

    # Refraction and temperature at the tropopause (Equation 7.85).
    tt = temperature - alpha * (ht - h0)
    t_rat_t = tt / temperature
    nt = 1 + (c6 * t_rat_t ** (gamma - 2) - c7 * t_rat_t ** (delta - 2)) * t_rat_t
    pwt = pw0 * t_rat_t**delta
    pt = (p0_mb + c5) * t_rat_t**gamma - pwt * (1 - mw / md) * gamma / (delta - gamma)

    # Stratosphere: Equation 7.86.
    t_out[strat] = tt
    n[strat] = 1 + (nt - 1) * np.exp(-c2 * (h[strat] - ht) / tt)
    dndr[strat] = -(c2 / tt) * (nt - 1) * np.exp(-c2 * (h[strat] - ht) / tt)
    p_out[strat] = pt * np.exp(-c2 * (h[strat] - ht) / tt) / 0.01

    # Troposphere: Equation 7.85.
    trop = ~strat
    t_out[trop] = temperature - alpha * (h[trop] - h0)
    t_rat = t_out[trop] / temperature
    n[trop] = 1 + (c6 * t_rat ** (gamma - 2) - c7 * t_rat ** (delta - 2)) * t_rat
    dndr[trop] = -c8 * t_rat ** (gamma - 2) + c9 * t_rat ** (delta - 2)
    pw = pw0 * t_rat**delta
    p_out[trop] = (
        (p0_mb + c5) * t_rat**gamma - pw * (1 - mw / md) * gamma / (delta - gamma)
    ) / 0.01

    if n.ndim == 0:
        return SinclairAtmosResult(float(n), float(dndr), float(t_out), float(p_out))
    return SinclairAtmosResult(n, dndr, t_out, p_out)


def remove_astro_refraction(
    algorithm: int,
    obs_lat_lon_alt: ArrayLike,
    z_observed: ArrayLike,
    rel_humid: float = STANDARD_RELATIVE_HUMIDITY,
    pressure: float = STANDARD_ATMOSPHERE,
    temperature: float = STANDARD_TEMPERATURE,
    wavelength: float = 0.574e-6,
) -> AstroRefractionResult:
    """
    Remove atmospheric refraction from an observed zenith distance.

    Given refraction-corrupted zenith distances of an object outside the
    atmosphere seen by a near-surface observer, compute the true
    (refraction-free) zenith distances using low-precision atmospheric
    models.

    Port of ``removeAstroRefrac.m``.

    Parameters
    ----------
    algorithm : int
        ``0`` numerical ray integration through the Sinclair atmosphere
        (the most precise; valid for observed zenith distances up to 100
        degrees and observers below the tropopause); ``1`` the
        Saastamoinen formula (sea-level observer, zenith distances below
        70 degrees); ``2`` the IAU ``A*tan(z) + B*tan^3(z)`` model with
        Newton-Raphson correction (sea-level observer, any positive
        zenith distance, degrading near the horizon).
    obs_lat_lon_alt : array_like
        The observer's WGS-84 ``[latitude, longitude, height]``, radians
        and meters. Only algorithm 0 reads it (latitude and height; the
        longitude does not matter). Ignored by algorithms 1 and 2.
    z_observed : array_like
        Refraction-corrupted positive zenith distance(s) in radians,
        measured down from the local vertical.
    rel_humid : float, optional
        Relative humidity at the observer in [0, 1]. Default 0.
    pressure : float, optional
        Pressure at the observer in Pascals. Default 101325 Pa.
    temperature : float, optional
        Temperature at the observer in Kelvin. Default 288.15 K.
    wavelength : float, optional
        Observation wavelength in meters. Default 0.574e-6 m.

    Returns
    -------
    result : AstroRefractionResult
        ``zenith_distance`` holds the true zenith distances,
        ``z_true = z_observed + delta_z``. Both fields are empty arrays
        when any input is outside the algorithm's validity region
        (mirroring the MATLAB function's empty-matrix return).

    Raises
    ------
    ValueError
        If any observed zenith distance is negative, the algorithm is
        unknown, or (algorithm 0) the observer is above the tropopause.

    Examples
    --------
    >>> obs = [0.61, 0.0, 100.0]
    >>> z_true, dz = remove_astro_refraction(0, obs, 1.2, 0.5, 101325.0, 288.15)
    >>> print(f"{z_true:.9f} {dz:.4e}")
    1.200706341 7.0634e-04
    >>> z_true, dz = remove_astro_refraction(2, obs, 1.2, 0.5, 101325.0, 288.15)
    >>> print(f"{z_true:.9f}", dz > 0)
    1.200705016 True
    """
    z0 = np.asarray(z_observed, dtype=np.float64)
    scalar_in = z0.ndim == 0
    z0 = np.atleast_1d(z0)
    empty = np.empty(0, dtype=np.float64)

    if np.any(z0 < 0):
        raise ValueError("The observed zenith distance must be positive.")

    if algorithm == 0:
        obs = np.asarray(obs_lat_lon_alt, dtype=np.float64)
        lat, lon, h0 = obs[0], obs[1], obs[2]
        # Radius of the Earth at this latitude (independent of longitude).
        x, y, z_e = geodetic_to_ecef(lat, lon, 0.0)
        re = float(np.sqrt(x**2 + y**2 + z_e**2))
        r0 = re + h0
        ht = 11000.0  # Assumed top of the troposphere.
        rt = ht + re
        hs = 80000.0  # Height at which refraction is negligible.
        rs = hs + re

        # Unbounded zenith distances could keep the integrator from
        # terminating.
        if np.any(z0 > 100 * np.pi / 180):
            return AstroRefractionResult(empty, empty)
        if h0 > ht:
            raise ValueError(
                "The algorithm is not meant for observers above the "
                "height of the troposphere (11000m)"
            )

        def _atmos(h):
            return sinclair_atmosphere(
                h,
                obs,
                rel_humid,
                pressure,
                temperature,
                wavelength,
                ht,
            )

        n0 = _atmos(h0).n
        nt = _atmos(ht).n
        ns = _atmos(hs).n

        def _zenith_dist_to_r(z_cur, z0_cur):
            # Equation 7.84: six Newton iterations for the radius at
            # which the ray reaches zenith distance z along its path.
            r = np.full_like(np.asarray(z_cur, dtype=np.float64), r0)
            for _ in range(6):
                res = _atmos(r - re)
                r = r - (res.n * r - n0 * r0 * np.sin(z0_cur) / np.sin(z_cur)) / (
                    res.n + r * res.dndr
                )
            return r

        def _integrand(z_cur, z0_cur):
            # Equation 7.87.
            r_val = _zenith_dist_to_r(z_cur, z0_cur)
            res = _atmos(r_val - re)
            return r_val * res.dndr / (res.n + r_val * res.dndr)

        z_true = np.zeros_like(z0)
        delta_z = np.zeros_like(z0)
        for i in range(z0.size):
            z0_cur = z0.flat[i]
            if z0_cur < 1e-20:
                # The model has exactly zero refraction at the zenith;
                # the integrator struggles with such small angles.
                z_true.flat[i] = z0_cur
                delta_z.flat[i] = 0.0
                continue
            # Zenith distances of the tropopause and of the top of the
            # stratosphere along this ray (Equation 7.88).
            zt = np.arcsin(n0 * r0 * np.sin(z0_cur) / (nt * rt))
            zs = np.arcsin(n0 * r0 * np.sin(z0_cur) / (ns * rs))

            xit = -quad(
                _integrand,
                zt,
                z0_cur,
                args=(z0_cur,),
                epsabs=1e-10,
                epsrel=1e-6,
            )[0]
            xis = -quad(
                _integrand,
                zs,
                zt,
                args=(z0_cur,),
                epsabs=1e-10,
                epsrel=1e-6,
            )[0]
            delta_z.flat[i] = xit + xis
            z_true.flat[i] = z0_cur + delta_z.flat[i]
    elif algorithm == 1:
        # Saastamoinen's formula. Pressure in millibars.
        p_mb = pressure * 0.01
        delta = 18.36
        pw0 = rel_humid * (temperature / 247.1) ** delta
        q = (p_mb - 0.156 * pw0) / temperature

        if np.any(z0 > 70 * np.pi / 180):
            return AstroRefractionResult(empty, empty)

        arcsec = (1 / 60) * (1 / 60) * (np.pi / 180)
        tan_z = np.tan(z0)
        delta_z = arcsec * (
            16.271 * q * tan_z * (1 + 0.0000394 * q * tan_z**2)
            - 0.0000749 * p_mb * (tan_z + tan_z**3)
        )
        z_true = z0 + delta_z
    elif algorithm == 2:
        # The IAU A*tan(z)+B*tan^3(z) model with the Newton-Raphson
        # correction and input bounding used in SOFA's iauAtioq.
        cel_min = 1e-6
        sel_min = 0.05
        a, b = simple_astro_ref_params(rel_humid, pressure, temperature, wavelength)
        r = np.maximum(np.sin(z0), cel_min)
        z = np.maximum(np.cos(z0), sel_min)
        tan_z = r / z
        w = b * tan_z * tan_z
        delta_z = (a + w) * tan_z / (1 + (a + 3 * w) / (z * z))
        z_true = z0 + delta_z
    else:
        raise ValueError(
            f"An invalid value for the algorithm was provided: {algorithm}"
        )

    if scalar_in:
        return AstroRefractionResult(float(z_true[0]), float(delta_z[0]))
    return AstroRefractionResult(z_true, delta_z)


def add_astro_refraction(
    algorithm: int,
    obs_lat_lon_alt: ArrayLike,
    z_true: ArrayLike,
    rel_humid: float = STANDARD_RELATIVE_HUMIDITY,
    pressure: float = STANDARD_ATMOSPHERE,
    temperature: float = STANDARD_TEMPERATURE,
    wavelength: float = 0.574e-6,
) -> AstroRefractionResult:
    """
    Add atmospheric refraction to a true zenith distance.

    The inverse of :func:`remove_astro_refraction`: given true
    (refraction-free) zenith distances of an object outside the
    atmosphere, compute the refraction-corrupted apparent zenith
    distances. The inverse problem is solved by a fixed 20 iterations of
    the forward model, which is generally sufficient for convergence to
    working precision.

    Port of ``addAstroRefrac.m``.

    Parameters
    ----------
    algorithm : int
        Same choices and validity regions as
        :func:`remove_astro_refraction`.
    obs_lat_lon_alt : array_like
        The observer's WGS-84 ``[latitude, longitude, height]``, radians
        and meters. Read only by algorithm 0.
    z_true : array_like
        True positive zenith distance(s) in radians.
    rel_humid : float, optional
        Relative humidity at the observer in [0, 1]. Default 0.
    pressure : float, optional
        Pressure at the observer in Pascals. Default 101325 Pa.
    temperature : float, optional
        Temperature at the observer in Kelvin. Default 288.15 K.
    wavelength : float, optional
        Observation wavelength in meters. Default 0.574e-6 m.

    Returns
    -------
    result : AstroRefractionResult
        ``zenith_distance`` holds the refraction-corrupted zenith
        distances, ``z0 = z_true - delta_z``. Both fields are empty
        arrays when the point falls outside the algorithm's validity
        region during iteration.

    Examples
    --------
    >>> obs = [0.61, 0.0, 100.0]
    >>> z_t = 1.200706341
    >>> z0, dz = add_astro_refraction(0, obs, z_t, 0.5, 101325.0, 288.15)
    >>> print(f"{z0:.6f}")
    1.200000
    """
    _, delta_z = remove_astro_refraction(
        algorithm,
        obs_lat_lon_alt,
        z_true,
        rel_humid,
        pressure,
        temperature,
        wavelength,
    )

    z_true_arr = np.asarray(z_true, dtype=np.float64)
    if np.size(delta_z) != 0:
        for _ in range(20):
            _, delta_z = remove_astro_refraction(
                algorithm,
                obs_lat_lon_alt,
                z_true_arr - delta_z,
                rel_humid,
                pressure,
                temperature,
                wavelength,
            )
            if np.size(delta_z) == 0:
                # The observation ended up too far underground.
                break

    if np.size(delta_z) == 0:
        empty = np.empty(0, dtype=np.float64)
        return AstroRefractionResult(empty, empty)
    z0 = z_true_arr - delta_z
    if z_true_arr.ndim == 0:
        return AstroRefractionResult(float(z0), float(delta_z))
    return AstroRefractionResult(z0, delta_z)


# ---------------------------------------------------------------------------
# Standard exponential atmospheric model (radar refraction)
# ---------------------------------------------------------------------------


class RuvStdRefracResult(NamedTuple):
    """
    Refraction-corrupted bistatic r-u-v measurements and ray directions.

    Attributes
    ----------
    z : ndarray
        The measurements, shape (3, N) or (4, N) with ``include_w``:
        bistatic range then the direction cosines of the apparent target
        direction in the receiver's local frame.
    u_tx : ndarray
        Unit vectors (3, N) in ECEF pointing from the transmitter toward
        the refraction-corrupted apparent target position.
    u_tar_rx : ndarray
        Unit vectors (3, N) of the apparent direction of the receiver as
        seen by the target.
    u_tar_tx : ndarray
        Unit vectors (3, N) of the apparent direction of the transmitter
        as seen by the target.
    """

    z: NDArray[np.float64]
    u_tx: NDArray[np.float64]
    u_tar_rx: NDArray[np.float64]
    u_tar_tx: NDArray[np.float64]


class StdRefracBiasResult(NamedTuple):
    """
    Approximate refraction biases of a monostatic radar measurement.

    Attributes
    ----------
    delta_r_one_way : float
        Bias in the one-way range in meters (add to the true range to get
        the measured range).
    delta_theta : float
        Bias in the elevation angle in radians.
    """

    delta_r_one_way: float
    delta_theta: float


class CubatureConversionResult(NamedTuple):
    """
    First two moments of a measurement converted by cubature integration.

    Attributes
    ----------
    mean : ndarray
        Converted mean(s), shape (3, N).
    covariance : ndarray
        Converted covariance matrices, shape (3, 3, N).
    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]


def _std_decay_const(ns: float) -> float:
    """The CRPL decay constant used by every exponential-model default."""
    delta_n = -7.32 * np.exp(0.005577 * ns)
    return float(np.log(ns / (ns + delta_n)) / 1000.0)


def _refractivity_2d(x, y, ns, r_e, ce):
    """Refractivity 1e6*(n-1) at 2D point(s) (x, y) from Earth's center."""
    return 1e-6 * ns * np.exp(-ce * (np.sqrt(x**2 + y**2) - r_e))


def _ray_rhs(x, y, ns, r_e, ce):
    """RHS of the 2D ray ODE; vectorized over columns of y for solve_bvp."""
    n_val = _refractivity_2d(x, y[0], ns, r_e, ce)
    return np.vstack(
        [
            y[1],
            ce
            * (1 + y[1] ** 2)
            * (x * y[1] - y[0])
            * n_val
            / ((n_val + 1) * np.sqrt(x**2 + y[0] ** 2)),
        ]
    )


def _ray_rhs_jac(x, y, ns, r_e, ce):
    """Jacobian of :func:`_ray_rhs`, shape (2, 2, m)."""
    n_val = _refractivity_2d(x, y[0], ns, r_e, ce)
    m = np.size(x)
    jac = np.zeros((2, 2, m))
    jac[0, 1, :] = 1.0
    jac[1, 0, :] = (
        ce
        * (1 + y[1] ** 2)
        * (-n_val)
        * (
            ce * y[0] * (x * y[1] - y[0]) * np.sqrt(x**2 + y[0] ** 2)
            + x * (x + y[0] * y[1]) * (n_val + 1)
        )
        / ((x**2 + y[0] ** 2) ** (3 / 2) * (n_val + 1) ** 2)
    )
    jac[1, 1, :] = (
        ce
        * (x - 2 * y[0] * y[1] + 3 * x * y[1] ** 2)
        * n_val
        / ((n_val + 1) * np.sqrt(x**2 + y[0] ** 2))
    )
    return jac


def _trace_ray(x0, y0, x1, y1, ns, r_e, ce):
    """Solve the two-point ray BVP between (x0, y0) and (x1, y1)."""
    num_steps = max(20, int(np.ceil(20 * np.hypot(x1 - x0, y1 - y0) / 400e3)))
    x_mesh = np.linspace(x0, x1, num_steps)
    slope = (y1 - y0) / (x1 - x0)
    intercept = y1 - slope * x1
    y_init = np.vstack([x_mesh * slope + intercept, np.full(num_steps, slope)])
    sol = solve_bvp(
        lambda x, y: _ray_rhs(x, y, ns, r_e, ce),
        lambda ya, yb: np.array([ya[0] - y0, yb[0] - y1]),
        x_mesh,
        y_init,
        fun_jac=lambda x, y: _ray_rhs_jac(x, y, ns, r_e, ce),
        bc_jac=lambda ya, yb: (
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        ),
        tol=1e-8,
        max_nodes=100000,
    )
    return sol


def _path_length(sol, x0, x1, ns, r_e, ce):
    """Optical path length along a traced ray (the apparent range)."""

    def _fun(x):
        y = sol.sol(x)
        return (1 + _refractivity_2d(x, y[0], ns, r_e, ce)) * np.sqrt(1 + y[1] ** 2)

    return quad(_fun, x0, x1, epsabs=1e-13, epsrel=1e-13, limit=200)[0]


def _vertical_range(y0, y_max, ns, r_e, ce):
    """Closed-form apparent range for a purely radial (vertical) path."""
    return (
        ((np.exp(ce * (r_e - y0)) - np.exp(ce * (r_e - y_max))) * ns) / (1e6 * ce)
        + y_max
        - y0
    )


def _local_ray_frame(x_obs, vec_to_tar):
    """
    Rotation from ECEF into the 2D ray-tracing frame.

    The frame's y axis is the observer's (spherical) vertical, the x axis
    the horizontal projection of the observer-to-target vector; the ray
    stays in the x-y plane. Returns the rotation matrix and the norm of
    the horizontal projection (whose smallness flags near-vertical rays).
    """
    u_vert = x_obs / np.linalg.norm(x_obs)
    vec = vec_to_tar - np.dot(vec_to_tar, u_vert) * u_vert
    horiz_norm = np.linalg.norm(vec)
    if horiz_norm < 1e-3:
        return None, horiz_norm
    u_horiz = vec / horiz_norm
    rot = np.vstack([u_horiz, u_vert, np.cross(u_horiz, u_vert)])
    return rot, horiz_norm


def _atmos_refrac_meas(x_obs, x_obj, ns, ce, r_e):
    """
    Apparent one-way range and arrival/departure directions between two
    points in the exponential atmosphere (sphere-centered ECEF frame).
    """
    vec_to_tar = x_obj - x_obs
    rot, horiz_norm = _local_ray_frame(x_obs, vec_to_tar)

    y0 = np.linalg.norm(x_obs)
    if rot is None:
        # Near-vertical path: refraction bending is negligible, so the
        # apparent range is a closed-form radial integral.
        x1 = 0.0
        y1_rel = np.dot(vec_to_tar, x_obs / y0)
        y_max = np.hypot(x1, y1_rel + y0)
        rng = _vertical_range(y0, y_max, ns, r_e, ce)
        u_arrive = vec_to_tar / np.linalg.norm(vec_to_tar)
        return rng, u_arrive, -u_arrive

    vec_local = rot @ vec_to_tar
    x1 = vec_local[0]
    y1 = vec_local[1] + y0

    sol = _trace_ray(0.0, y0, x1, y1, ns, r_e, ce)
    rng = _path_length(sol, 0.0, x1, ns, r_e, ce)

    theta0 = np.arctan(sol.y[1, 0])
    u_arrive = rot.T @ np.array([np.cos(theta0), np.sin(theta0), 0.0])
    theta1 = np.arctan(sol.y[1, -1])
    u_depart = -(rot.T @ np.array([np.cos(theta1), np.sin(theta1), 0.0]))
    return rng, u_arrive, u_depart


def _apparent_cart_from_ruv(z, use_half_range, z_tx, z_rx, m):
    """
    The refraction-free bistatic r-u-v to Cartesian conversion used to
    seed the ray shooting (a single-measurement port of MATLAB's
    ``ruv2Cart``).
    """
    r_b = 2 * z[0] if use_half_range else z[0]
    if z.shape[0] > 3:
        u_vec = z[1:4].copy()
    else:
        u, v = z[1], z[2]
        uv_mag2 = u**2 + v**2
        if uv_mag2 > 1:
            uv_mag = np.sqrt(uv_mag2)
            u, v = u / uv_mag, v / uv_mag
        u_vec = np.array([u, v, np.sqrt(max(1 - u**2 - v**2, 0.0))])
    z_tx_local = m @ (z_tx - z_rx)
    if r_b == 0:
        r1 = 0.0
    else:
        r1 = (r_b**2 - np.dot(z_tx_local, z_tx_local)) / (
            2 * (r_b - np.dot(u_vec, z_tx_local))
        )
    return m.T @ (r1 * u_vec) + z_rx


def cart2ruv_std_refrac(
    z_c: ArrayLike,
    use_half_range: bool = False,
    z_tx: Optional[ArrayLike] = None,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
    ns: float = 313.0,
    include_w: bool = False,
    ce: Optional[float] = None,
    r_e: Optional[float] = None,
    sphere_center: Optional[ArrayLike] = None,
) -> RuvStdRefracResult:
    """
    Convert Cartesian points to refraction-corrupted bistatic r-u-v.

    Traces rays through the standard exponential atmosphere
    (``N = ns * exp(-ce * h)``) over a locally osculating spherical Earth
    to determine the apparent bistatic range and direction cosines of
    each target as seen by the receiver. Not suitable for
    satellite-to-satellite paths grazing the atmosphere, and the ray
    tracer can fail for paths going too far underground or for targets
    collocated with the receiver or the transmitter.

    Port of ``Cart2RuvStdRefrac.m``.

    Parameters
    ----------
    z_c : array_like
        Cartesian target positions in global ECEF coordinates, shape
        (3, N) or (3,).
    use_half_range : bool, optional
        Whether the bistatic range is halved (one-way range in the
        monostatic case). Default False.
    z_tx : array_like, optional
        Transmitter ECEF position, shape (3,). Default: the origin.
    z_rx : array_like, optional
        Receiver ECEF position, shape (3,). Default: the origin.
    m : array_like, optional
        3x3 rotation from global axes to the receiver's local axes (the
        receiver boresight is its local z axis). Default: identity.
    ns : float, optional
        Refractivity reduced to the reference sphere. Default 313.
    include_w : bool, optional
        Include the third direction cosine, making ``z`` 4xN. Default
        False.
    ce : float, optional
        Decay constant of the exponential model in inverse meters.
        Default: derived from ``ns`` via the CRPL standard constants
        (:func:`atmos_exp_decay_const`).
    r_e : float, optional
        Radius of the spherical-Earth approximation. Default: the
        osculating sphere at the receiver
        (:func:`pytcl.navigation.geodesy.osculating_sphere`).
    sphere_center : array_like, optional
        ECEF offset of the sphere's center. Defaults to the osculating
        sphere's offset when ``r_e`` is defaulted, and to zeros when
        ``r_e`` is given.

    Returns
    -------
    result : RuvStdRefracResult
        The measurements and the apparent ray directions at both ends.

    Examples
    --------
    >>> z_rx = np.array([6378137.0, 0.0, 0.0])
    >>> z_tar = np.array([6428137.0, 100e3, 0.0])
    >>> res = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx)
    >>> print(f"{res.z[0, 0]:.3f}")
    111808.239
    """
    z_c = np.atleast_2d(np.asarray(z_c, dtype=np.float64))
    if z_c.shape[0] == 1:
        z_c = z_c.T
    num_meas = z_c.shape[1]
    z_tx = np.zeros(3) if z_tx is None else np.asarray(z_tx, dtype=np.float64).ravel()
    z_rx = np.zeros(3) if z_rx is None else np.asarray(z_rx, dtype=np.float64).ravel()
    m = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    if ce is None:
        ce = _std_decay_const(ns)
    if r_e is None:
        lat, lon, _ = ecef_to_geodetic(*z_rx)
        r_e, sphere_center = osculating_sphere(float(lat), float(lon))
    elif sphere_center is None:
        sphere_center = np.zeros(3)
    sphere_center = np.asarray(sphere_center, dtype=np.float64).ravel()

    z_c = z_c - sphere_center[:, None]
    z_tx = z_tx - sphere_center
    z_rx = z_rx - sphere_center

    z = np.zeros((4 if include_w else 3, num_meas))
    u_tx = np.zeros((3, num_meas))
    u_tar_tx = np.zeros((3, num_meas))
    u_tar_rx = np.zeros((3, num_meas))

    monostatic = np.array_equal(z_rx, z_tx)
    for cur in range(num_meas):
        if monostatic:
            rng, u_arrive, u_tar_rx[:, cur] = _atmos_refrac_meas(
                z_tx, z_c[:, cur], ns, ce, r_e
            )
            r = 2 * rng
            u_tx[:, cur] = u_arrive
            u_tar_tx[:, cur] = u_tar_rx[:, cur]
        else:
            r2, u_arrive, u_tar_rx[:, cur] = _atmos_refrac_meas(
                z_rx, z_c[:, cur], ns, ce, r_e
            )
            r1, u_tx[:, cur], u_tar_tx[:, cur] = _atmos_refrac_meas(
                z_tx, z_c[:, cur], ns, ce, r_e
            )
            r = r1 + r2
        if use_half_range:
            r = r / 2
        u = m @ u_arrive
        z[0, cur] = r
        z[1:, cur] = u[: z.shape[0] - 1]
    return RuvStdRefracResult(z, u_tx, u_tar_rx, u_tar_tx)


def ruv2cart_std_refrac(
    z_ruv: ArrayLike,
    use_half_range: bool = False,
    z_tx: Optional[ArrayLike] = None,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
    ns: float = 313.0,
    ce: Optional[float] = None,
    r_e: Optional[float] = None,
    sphere_center: Optional[ArrayLike] = None,
    x_max: float = 1000e3,
) -> NDArray[np.float64]:
    """
    Convert refraction-corrupted bistatic r-u-v points to Cartesian.

    The inverse of :func:`cart2ruv_std_refrac`: shoots a ray from the
    receiver in the apparent direction through the exponential
    atmosphere (an initial value problem) and searches along the traced
    path for the point whose accumulated bistatic range matches the
    measurement. Fails if the target is collocated with the transmitter
    or the receiver.

    Port of ``ruv2CartStdRefrac.m``. Deviation from MATLAB: the MATLAB
    function's near-vertical branch aborts the whole measurement loop
    (an upstream bug); this port processes remaining measurements.

    Parameters
    ----------
    z_ruv : array_like
        Measurements, shape (3, N) or (4, N) ([r; u; v] or [r; u; v; w]),
        or a single measurement of shape (3,) or (4,).
    use_half_range : bool, optional
        Whether the ranges in ``z_ruv`` are halved. Default False.
    z_tx, z_rx, m, ns, ce, r_e, sphere_center
        As in :func:`cart2ruv_std_refrac`.
    x_max : float, optional
        Maximum horizontal displacement searched along the ray, in
        meters. Default 1000 km.

    Returns
    -------
    z_cart : ndarray
        Cartesian target positions in global ECEF coordinates, (3, N).

    Examples
    --------
    >>> z_rx = np.array([6378137.0, 0.0, 0.0])
    >>> z_tar = np.array([6428137.0, 100e3, 0.0])
    >>> ruv = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx).z
    >>> back = ruv2cart_std_refrac(ruv, True, z_rx, z_rx)
    >>> np.allclose(back[:, 0], z_tar, atol=0.5)
    True
    """
    z_ruv = np.atleast_2d(np.asarray(z_ruv, dtype=np.float64))
    if z_ruv.shape[0] == 1:
        z_ruv = z_ruv.T
    num_meas = z_ruv.shape[1]
    z_tx = np.zeros(3) if z_tx is None else np.asarray(z_tx, dtype=np.float64).ravel()
    z_rx = np.zeros(3) if z_rx is None else np.asarray(z_rx, dtype=np.float64).ravel()
    m = np.eye(3) if m is None else np.asarray(m, dtype=np.float64)
    if ce is None:
        ce = _std_decay_const(ns)
    if r_e is None:
        lat, lon, _ = ecef_to_geodetic(*z_rx)
        r_e, sphere_center = osculating_sphere(float(lat), float(lon))
    elif sphere_center is None:
        sphere_center = np.zeros(3)
    sphere_center = np.asarray(sphere_center, dtype=np.float64).ravel()

    z_rx = z_rx - sphere_center
    z_tx = z_tx - sphere_center

    z_ruv = z_ruv.copy()
    if use_half_range:
        z_ruv[0, :] = 2 * z_ruv[0, :]

    monostatic = np.array_equal(z_rx, z_tx)
    u_vert = z_rx / np.linalg.norm(z_rx)
    y0 = np.linalg.norm(z_rx)
    z_cart = np.zeros((3, num_meas))

    def _tx_range(tar_loc):
        # Apparent one-way range from the transmitter via the forward
        # model (monostatic-from-Tx with half range).
        res = cart2ruv_std_refrac(
            tar_loc.reshape(3, 1),
            True,
            z_tx,
            z_tx,
            None,
            ns,
            False,
            ce,
            r_e,
            np.zeros(3),
        )
        return res.z[0, 0]

    for cur in range(num_meas):
        apparent = _apparent_cart_from_ruv(z_ruv[:, cur], False, z_tx, z_rx, m)
        vec_to_tar = apparent - z_rx
        rot, horiz_norm = _local_ray_frame(z_rx, vec_to_tar)
        biased_range = z_ruv[0, cur]

        if rot is None:
            # Near-vertical: search the radial distance directly against
            # the closed-form range.
            if z_ruv.shape[0] == 3:
                w = np.sqrt(max(1 - z_ruv[1, cur] ** 2 - z_ruv[2, cur] ** 2, 0.0))
                u_meas = np.array([z_ruv[1, cur], z_ruv[2, cur], w])
            else:
                u_meas = z_ruv[1:4, cur]

            def _cost_vertical(y_max):
                r_rx = _vertical_range(y0, y_max, ns, r_e, ce)
                if monostatic:
                    r_tx = r_rx
                else:
                    # Faithful to MATLAB's rangeCostVertical, which uses
                    # u*range as an absolute position; that is only
                    # geometrically right for a receiver at the origin.
                    # Positions near the Earth's center overflow the
                    # exponential model; MATLAB propagates Inf silently
                    # and the bounded search carries on.
                    with (
                        np.errstate(over="ignore", invalid="ignore"),
                        warnings.catch_warnings(),
                    ):
                        warnings.simplefilter("ignore", IntegrationWarning)
                        r_tx = _tx_range(u_meas * r_rx)
                return (biased_range - r_tx - r_rx) ** 2

            res = minimize_scalar(
                _cost_vertical,
                bounds=(y0, y0 + x_max),
                method="bounded",
                options={"xatol": 1e-8},
            )
            y_true = res.x
            # A purely vertical offset needs no horizontal frame.
            z_cart[:, cur] = u_vert * (y_true - y0) + z_rx + sphere_center
            continue

        vec_local = rot @ vec_to_tar
        x1 = vec_local[0]
        y1 = vec_local[1] + y0
        y0_dot = (y1 - y0) / x1

        ivp = solve_ivp(
            lambda x, y: _ray_rhs(x, y.reshape(2, 1), ns, r_e, ce).ravel(),
            (0.0, x_max),
            [y0, y0_dot],
            method="RK45",
            rtol=1e-12,
            atol=1e-12,
            dense_output=True,
        )

        def _path_range(x_end):
            def _fun(x):
                y = ivp.sol(x)
                return (1 + _refractivity_2d(x, y[0], ns, r_e, ce)) * np.sqrt(
                    1 + y[1] ** 2
                )

            return quad(_fun, 0.0, x_end, epsabs=1e-10, epsrel=1e-10, limit=200)[0]

        def _cost(x_end):
            r_rx = _path_range(x_end)
            if monostatic:
                r_tx = r_rx
            else:
                y_end = ivp.sol(x_end)
                tar_local = np.array([x_end, y_end[0] - y0, 0.0])
                r_tx = _tx_range(rot.T @ tar_local + z_rx)
            return (biased_range - r_tx - r_rx) ** 2

        res = minimize_scalar(
            _cost,
            bounds=(0.0, x_max),
            method="bounded",
            options={"xatol": 1e-8},
        )
        x_true = res.x
        y_true = ivp.sol(x_true)[0]
        local = np.array([x_true, y_true - y0, 0.0])
        z_cart[:, cur] = rot.T @ local + z_rx + sphere_center
    return z_cart


def std_refrac_bias_approx(
    path_length: float,
    elevation: float,
    radar_height: float,
    ns: float = 313.0,
    ce: Optional[float] = None,
    r_e: Optional[float] = None,
    algorithm: int = 1,
) -> StdRefracBiasResult:
    """
    Approximate range and elevation biases due to standard refraction.

    For a monostatic radar at a given height observing a target at a
    given path length and elevation, approximate the offsets that
    refraction through the standard exponential atmosphere adds to the
    one-way range and to the elevation angle.

    Port of ``stdRefracBiasApprox.m``.

    Parameters
    ----------
    path_length : float
        Length of the refraction-free path to the target in meters.
    elevation : float
        Elevation angle of the target above the radar's local horizontal
        in radians.
    radar_height : float
        Height of the radar above the reference sphere in meters.
    ns : float, optional
        Refractivity reduced to the reference sphere. Default 313.
    ce : float, optional
        Decay constant in inverse meters. Default: CRPL standard.
    r_e : float, optional
        Radius of the reference sphere. Default: the WGS-84 mean radius
        ``(2a + b) / 3``.
    algorithm : int, optional
        ``1`` (default) numerical ray tracing (the same BVP solve as
        :func:`cart2ruv_std_refrac`); ``0`` the closed-form
        Kerce-Blair-Brown approximation, valid for elevations up to 49
        degrees.

    Returns
    -------
    result : StdRefracBiasResult
        The one-way range bias in meters and the elevation bias in
        radians.

    References
    ----------
    - J. C. Kerce, W. D. Blair, and G. C. Brown, "Modeling refraction
      errors for simulation studies of multisensor target tracking,"
      Proc. 36th Southeastern Symposium on System Theory, Mar. 2004.

    Examples
    --------
    >>> res = std_refrac_bias_approx(100e3, 0.1, 100.0)
    >>> print(f"{res.delta_r_one_way:.4f} {res.delta_theta:.6e}")
    15.9561 1.420764e-03
    >>> res0 = std_refrac_bias_approx(100e3, 0.1, 100.0, algorithm=0)
    >>> print(f"{res0.delta_r_one_way:.4f}")
    15.8175
    """
    if ce is None:
        ce = _std_decay_const(ns)
    if r_e is None:
        r_e = (2 * EARTH_SEMI_MAJOR_AXIS + EARTH_SEMI_MINOR_AXIS) / 3
    r_radar = r_e + radar_height

    if algorithm == 0:
        alpha = ns / 1e6
        beta = ce
        if elevation > np.deg2rad(49):
            raise ValueError("This algorithm does not work for angles >49 degrees.")
        # The F function of Kerce, Blair and Brown, with the asymptotic
        # erf branch corrected from the paper.
        l_upper = (
            np.sqrt(beta / (2 * r_radar))
            * np.cos(elevation)
            * (path_length + r_radar * (1 / np.cos(elevation)) * np.tan(elevation))
        )
        l_lower = np.sqrt(beta * r_radar / 2) * np.tan(elevation)
        if elevation <= np.deg2rad(10):
            erf_diff = erf(l_upper) - erf(l_lower)
        else:
            erf_diff = 1 / (l_lower * np.sqrt(np.pi)) * np.exp(-(l_lower**2)) - 1 / (
                l_upper * np.sqrt(np.pi)
            ) * np.exp(-(l_upper**2))
        f_val = (
            np.sqrt(np.pi * r_radar / (2 * beta))
            * np.cos(elevation)
            * np.exp(-beta * (r_radar - r_e))
            * np.exp(beta * r_radar * np.tan(elevation) ** 2 / 2)
            * erf_diff
        )
        return StdRefracBiasResult(
            float(alpha * f_val),
            float(alpha * beta * np.cos(elevation) * f_val),
        )
    if algorithm != 1:
        raise ValueError(f"algorithm must be 0 or 1, got {algorithm}")

    x1 = path_length * np.cos(elevation)
    y0 = r_radar
    y1 = r_radar + np.sin(elevation) * path_length

    if abs(np.pi / 2 - elevation) < 1e-3:
        # Nearly vertical: closed-form radial integral, zero bending.
        y_max = r_radar + path_length
        delta_r = ((np.exp(ce * (r_e - y0)) - np.exp(ce * (r_e - y_max))) * ns) / (
            1e6 * ce
        )
        return StdRefracBiasResult(float(delta_r), 0.0)

    sol = _trace_ray(0.0, y0, x1, y1, ns, r_e, ce)
    rng = _path_length(sol, 0.0, x1, ns, r_e, ce)
    return StdRefracBiasResult(
        float(rng - path_length),
        float(np.arctan(sol.y[1, 0]) - elevation),
    )


def reduce_std_refrac_to_sphere(
    n_meas: float,
    height: float,
    exp_const: float = 0.005577,
    mult_const: float = 7.32,
    xatol: float = 1e-4,
) -> NDArray[np.float64]:
    """
    Reduce a measured refractivity to the reference sphere.

    Given the atmospheric refractivity measured at a height above sea
    level, determine the equivalent sea-level refractivity under the
    standard exponential model. The model is scanned over sea-level
    refractivities in [200, 450]; each sign change brackets a candidate
    solution refined by bounded scalar minimization, so **one or two
    solutions** can be returned.

    Port of ``reduceStdRefrac2Spher.m``.

    Parameters
    ----------
    n_meas : float
        The measured refractivity ``(n - 1) * 1e6``.
    height : float
        Height of the measurement above mean sea level in meters.
    exp_const, mult_const : float, optional
        Parameters of the decay model ``deltaN = -mult_const *
        exp(exp_const * N)`` per kilometer. Defaults are the CRPL
        standard values 0.005577 and 7.32.
    xatol : float, optional
        Absolute tolerance of the bounded search (MATLAB ``fminbnd``
        default 1e-4).

    Returns
    -------
    ns_values : ndarray
        The candidate sea-level refractivities, shape (num_solutions,).
        May be empty if the measurement is inconsistent with the model.

    Examples
    --------
    >>> vals = reduce_std_refrac_to_sphere(300.0, 1000.0)
    >>> print(f"{vals[0]:.4f}")
    352.1814
    """
    num_points = 100
    ns_grid = np.linspace(200.0, 450.0, num_points)

    def _model_minus_meas(ns):
        return (
            ns * (ns / (ns - mult_const * np.exp(exp_const * ns))) ** (-height / 1000)
            - n_meas
        )

    vals = _model_minus_meas(ns_grid)
    crossings = np.flatnonzero(np.diff(vals > 0))
    solutions = np.zeros(crossings.size)
    for i, idx in enumerate(crossings):
        res = minimize_scalar(
            lambda ns: _model_minus_meas(ns) ** 2,
            bounds=(ns_grid[idx], ns_grid[idx + 1]),
            method="bounded",
            options={"xatol": xatol},
        )
        solutions[i] = res.x
    return solutions


def cart2ruv_std_refrac_cubature(
    z_c: ArrayLike,
    sqrt_cov: ArrayLike,
    use_half_range: bool = False,
    z_tx: Optional[ArrayLike] = None,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
    ns: float = 313.0,
    points: Optional[ArrayLike] = None,
    weights: Optional[ArrayLike] = None,
    ce: Optional[float] = None,
    r_e: Optional[float] = None,
    sphere_center: Optional[ArrayLike] = None,
) -> CubatureConversionResult:
    """
    Cubature-based Gaussian conversion of Cartesian states to r-u-v.

    Propagates Gaussian state estimates through
    :func:`cart2ruv_std_refrac` by cubature integration, returning the
    converted means and covariances.

    Port of ``Cart2RuvStdRefracCubature.m``.

    Parameters
    ----------
    z_c : array_like
        Cartesian means, shape (3, N) or (3,).
    sqrt_cov : array_like
        Lower-triangular square roots of the covariances, shape (3, 3)
        (shared) or (3, 3, N).
    use_half_range, z_tx, z_rx, m, ns, ce, r_e, sphere_center
        As in :func:`cart2ruv_std_refrac`.
    points, weights : array_like, optional
        Cubature points (num_points, 3) and weights for N(0, I).
        Default: :func:`fifth_order_cubature_points`.

    Returns
    -------
    result : CubatureConversionResult
        Converted means (3, N) and covariances (3, 3, N).

    Examples
    --------
    >>> z_rx = np.array([6378137.0, 0.0, 0.0])
    >>> z_tar = np.array([6428137.0, 100e3, 0.0])
    >>> sr = np.diag([100.0, 100.0, 100.0])
    >>> res = cart2ruv_std_refrac_cubature(z_tar, sr, True, z_rx, z_rx)
    >>> print(f"{res.mean[0, 0]:.1f}")
    111808.3
    """
    z_c = np.atleast_2d(np.asarray(z_c, dtype=np.float64))
    if z_c.shape[0] == 1:
        z_c = z_c.T
    num_meas = z_c.shape[1]
    sqrt_cov = np.asarray(sqrt_cov, dtype=np.float64)
    if sqrt_cov.ndim == 2:
        sqrt_cov = np.repeat(sqrt_cov[:, :, None], num_meas, axis=2)
    if points is None:
        points, weights = fifth_order_cubature_points(3)
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    z_out = np.zeros((3, num_meas))
    cov_out = np.zeros((3, 3, num_meas))
    for cur in range(num_meas):
        cub, _ = transform_cubature_points(
            points, weights, z_c[:, cur], sqrt_cov[:, :, cur]
        )
        converted = cart2ruv_std_refrac(
            cub.T,
            use_half_range,
            z_tx,
            z_rx,
            m,
            ns,
            False,
            ce,
            r_e,
            sphere_center,
        ).z
        mean, cov = cubature_point_moments(converted.T, weights, lambda p: p)
        z_out[:, cur] = mean
        cov_out[:, :, cur] = cov
    return CubatureConversionResult(z_out, cov_out)


def ruv2cart_std_refrac_cubature(
    z_ruv: ArrayLike,
    sqrt_cov: ArrayLike,
    use_half_range: bool = False,
    z_tx: Optional[ArrayLike] = None,
    z_rx: Optional[ArrayLike] = None,
    m: Optional[ArrayLike] = None,
    ns: float = 313.0,
    points: Optional[ArrayLike] = None,
    weights: Optional[ArrayLike] = None,
    ce: Optional[float] = None,
    r_e: Optional[float] = None,
    sphere_center: Optional[ArrayLike] = None,
    x_max: float = 1000e3,
) -> CubatureConversionResult:
    """
    Cubature-based Gaussian conversion of r-u-v measurements to Cartesian.

    Propagates Gaussian measurements through
    :func:`ruv2cart_std_refrac` by cubature integration.

    Port of ``ruv2CartStdRefracCubature.m``.

    Parameters
    ----------
    z_ruv : array_like
        Measurement means, shape (3, N) or (3,).
    sqrt_cov : array_like
        Lower-triangular square roots of the measurement covariances,
        shape (3, 3) or (3, 3, N).
    use_half_range, z_tx, z_rx, m, ns, ce, r_e, sphere_center, x_max
        As in :func:`ruv2cart_std_refrac`.
    points, weights : array_like, optional
        Cubature points and weights for N(0, I). Default:
        :func:`fifth_order_cubature_points`.

    Returns
    -------
    result : CubatureConversionResult
        Converted means (3, N) and covariances (3, 3, N).

    Examples
    --------
    >>> z_rx = np.array([6378137.0, 0.0, 0.0])
    >>> z_tar = z_rx + np.array([1e3, 5e3, 50e3])
    >>> ruv = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx).z[:, 0]
    >>> sr = np.diag([10.0, 1e-4, 1e-4])
    >>> res = ruv2cart_std_refrac_cubature(ruv, sr, True, z_rx, z_rx)
    >>> np.allclose(res.mean[:, 0], z_tar, atol=5.0)
    True
    """
    z_ruv = np.atleast_2d(np.asarray(z_ruv, dtype=np.float64))
    if z_ruv.shape[0] == 1:
        z_ruv = z_ruv.T
    num_meas = z_ruv.shape[1]
    sqrt_cov = np.asarray(sqrt_cov, dtype=np.float64)
    if sqrt_cov.ndim == 2:
        sqrt_cov = np.repeat(sqrt_cov[:, :, None], num_meas, axis=2)
    if points is None:
        points, weights = fifth_order_cubature_points(z_ruv.shape[0])
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    z_out = np.zeros((3, num_meas))
    cov_out = np.zeros((3, 3, num_meas))
    for cur in range(num_meas):
        cub, _ = transform_cubature_points(
            points, weights, z_ruv[:, cur], sqrt_cov[:, :, cur]
        )
        converted = ruv2cart_std_refrac(
            cub.T,
            use_half_range,
            z_tx,
            z_rx,
            m,
            ns,
            ce,
            r_e,
            sphere_center,
            x_max,
        )
        mean, cov = cubature_point_moments(converted.T, weights, lambda p: p)
        z_out[:, cur] = mean
        cov_out[:, :, cur] = cov
    return CubatureConversionResult(z_out, cov_out)
