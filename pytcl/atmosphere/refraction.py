"""
Atmospheric refractivity for radar and optical propagation.

Ports from the MATLAB Tracker Component Library's
``Atmosphere_and_Refraction`` directory. This module currently provides the
refractivity helpers; the astronomical-refraction and exponential-model
ray-tracing functions are planned follow-ons.

Conventions
-----------
Refractivity is ``N = (n - 1) * 1e6`` where ``n`` is the index of
refraction. The standard exponential atmosphere model takes the
refractivity at height ``h`` above sea level to be
``N = Ns * exp(-ce * h)`` where ``Ns`` is the sea-level refractivity and
``ce`` is the decay constant returned by :func:`atmos_exp_decay_const`.
"""

from typing import NamedTuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from pytcl.atmosphere.humidity import H2O_MOLAR_MASS
from pytcl.core.constants import (
    EARTH_ECCENTRICITY_SQ,
    STANDARD_ATMOSPHERE,
    STANDARD_RELATIVE_HUMIDITY,
    STANDARD_TEMPERATURE,
    UNIVERSAL_GAS_CONSTANT,
)
from pytcl.navigation.geodesy import geodetic_to_ecef

__all__ = [
    "AstroRefParams",
    "AstroRefractionResult",
    "ExpDecayConstResult",
    "SinclairAtmosResult",
    "add_astro_refraction",
    "approx_refractivity",
    "atmos_exp_decay_const",
    "remove_astro_refraction",
    "simple_astro_ref_params",
    "sinclair_atmosphere",
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
