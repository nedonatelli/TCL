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

__all__ = [
    "ExpDecayConstResult",
    "approx_refractivity",
    "atmos_exp_decay_const",
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
