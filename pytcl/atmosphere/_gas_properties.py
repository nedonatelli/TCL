"""
Thermophysical properties of atmospheric constituent gases.

Transcription of the MATLAB TCL's ``Constants.gasProp``: standard
atomic weights (CIAAW 2013, range endpoints averaged), ideal-gas
specific heats at constant pressure (Touloukian & Makita polynomial
fits, cal/(g*K) converted to J/(kg*K)), and second virial coefficients
with first and second temperature derivatives (Kaye & Laby exponential
models; a not-a-knot cubic spline over the AIP Handbook table for
helium; the Hyland correlation for water).

Used by :func:`pytcl.atmosphere.models.speed_of_sound_gas_table`.
"""

import math
import warnings
from typing import NamedTuple, Optional

from scipy.interpolate import CubicSpline

# Standard atomic weights in amu (CIAAW 2013; where CIAAW quotes a
# range, the mean of its endpoints, exactly as the MATLAB source).
_H = (1.00784 + 1.00811) / 2
_HE = 4.002602
_C = (12.0096 + 12.0116) / 2
_N = (14.00643 + 14.00728) / 2
_O = (15.99903 + 15.99977) / 2
_NE = 20.1797
_AR = 39.948
_KR = 83.798
_XE = 131.293

# Second virial coefficient of helium: AIP Handbook tabulation
# (cm^3/mol against K), interpolated with a not-a-knot cubic spline --
# the same boundary condition as the MATLAB cubSplineInterpSimp
# helper, so the piecewise polynomial is identical.
_HE_VIRIAL_T = [
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    22.0,
    22.64,
    24.0,
    26.0,
    28.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    60.0,
    80.0,
    100.0,
    120.0,
    160.0,
    200.0,
    273.15,
    373.15,
    400.0,
    600.0,
    800.0,
    1000.0,
    1200.0,
    1400.0,
]
_HE_VIRIAL_B = [
    -26.0,
    -21.7,
    -18.1,
    -15.2,
    -12.7,
    -10.5,
    -8.7,
    -7.1,
    -5.6,
    -4.3,
    -3.2,
    -2.2,
    -0.5,
    0.0,
    0.9,
    2.0,
    3.0,
    3.8,
    5.4,
    6.6,
    7.5,
    8.2,
    9.2,
    10.6,
    11.4,
    11.8,
    12.3,
    12.3,
    12.0,
    11.3,
    11.1,
    10.4,
    9.8,
    9.3,
    8.8,
    8.4,
]
_HE_SPLINE = CubicSpline(_HE_VIRIAL_T, _HE_VIRIAL_B, bc_type="not-a-knot")

# Per-gas data: molar mass (amu), specific-heat polynomial pieces as
# (T_break, low coeffs, high coeffs) in cal/(g*K) with coefficients in
# ascending power order (a constant c0p is a one-piece polynomial),
# the validated specific-heat range, the (a, b, c) of the exponential
# virial model B = a - b*exp(c/T) in cm^3/mol (None for the two
# special-cased gases), and the validated virial range.
_GASES = {
    "N2": (
        2 * _N,
        (
            775.0,
            (0.259934, -8.42119e-5, 1.72117e-7, -6.72914e-11),
            (0.201678, 1.08013e-4, -3.32212e-8, 2.45228e-12),
        ),
        (250.0, 1500.0),
        (185.4, 141.8, 88.7),
        (75.0, 700.0),
    ),
    "O2": (
        2 * _O,
        (
            760.0,
            (0.222081, -7.69230e-5, 2.78765e-7, -1.70107e-10),
            (0.177100, 1.49509e-4, -8.44940e-8, 1.83236e-11),
        ),
        (250.0, 1500.0),
        (152.8, 117.0, 108.8),
        (90.0, 400.0),
    ),
    "Ar": (
        _AR,
        (None, (0.12436,), None),
        (10.0, 6000.0),
        (154.2, 119.3, 105.1),
        (80.0, 1024.0),
    ),
    "CO2": (
        _C + 2 * _O,
        (
            590.0,
            (0.105914, 4.03552e-4, -3.03235e-7, 8.29431e-11),
            (0.135069, 2.89483e-4, -1.64998e-7, 3.53157e-11),
        ),
        (200.0, 1500.0),
        (137.6, 87.7, 325.7),
        (220.0, 1100.0),
    ),
    "Ne": (
        _NE,
        (None, (0.24615,), None),
        (10.0, 8000.0),
        (81.0, 63.6, 30.7),
        (44.0, 973.0),
    ),
    "Kr": (
        _KR,
        (None, (0.059284,), None),
        (10.0, 6200.0),
        (189.6, 148.0, 145.3),
        (110.0, 700.0),
    ),
    "CH4": (
        _C + 4 * _H,
        (
            790.0,
            (0.458066, -2.61341e-4, 2.07904e-6, -1.25017e-9),
            (0.0258866, 1.60802e-3, -6.67069e-7, 1.06432e-10),
        ),
        (270.0, 1500.0),
        (206.4, 159.5, 133.0),
        (110.0, 600.0),
    ),
    "He": (
        _HE,
        (None, (1.2412,), None),
        (10.0, 6000.0),
        None,
        (9.0, 1400.0),
    ),
    "N2O": (
        2 * _N + _O,
        (
            600.0,
            (0.103451, 4.89293e-4, -5.19278e-7, 2.44839e-10),
            (0.149343, 2.67192e-4, -1.47619e-7, 3.01604e-11),
        ),
        (200.0, 1500.0),
        (180.7, 114.8, 305.4),
        (200.0, 423.0),
    ),
    "NO": (
        _N + _O,
        (
            590.0,
            (0.282183, -3.16841e-4, 6.88734e-7, -4.22833e-10),
            (0.192074, 1.22287e-4, -5.05602e-8, 6.90346e-12),
        ),
        (100.0, 1500.0),
        (15.9, 11.0, 372.3),
        (122.0, 311.0),
    ),
    "Xe": (
        _XE,
        (None, (0.037837,), None),
        (10.0, 5200.0),
        (245.6, 190.9, 200.2),
        (160.0, 650.0),
    ),
    "CO": (
        _C + _O,
        (
            615.0,
            (0.256859, -6.46329e-5, 1.31865e-7, -2.65440e-11),
            (0.210345, 9.44224e-5, -1.94071e-8, -2.35385e-12),
        ),
        (250.0, 1500.0),
        (202.6, 154.2, 94.2),
        (90.0, 573.0),
    ),
    "H2": (
        2 * _H,
        (
            400.0,
            (1.46910, 1.60057e-2, -4.44048e-5, 4.21220e-8),
            (3.56903, -4.89590e-4, 6.22549e-7, -1.19686e-10),
        ),
        (100.0, 1500.0),
        (315.0, 289.7, 9.47),
        (14.0, 400.0),
    ),
    "H2O": (
        2 * _H + _O,
        (
            800.0,
            (0.452219, -1.29224e-4, 4.17008e-7, -2.00401e-10),
            (0.378278, 1.53443e-4, 3.31531e-8, -1.78435e-11),
        ),
        (270.0, 1500.0),
        None,
        (223.15, 363.15),
    ),
}


class GasProperties(NamedTuple):
    """Thermophysical properties of one gas at one temperature."""

    molar_mass: float  #: amu (g/mol)
    c0p: float  #: ideal-gas specific heat at constant pressure, J/(kg K)
    b: float  #: second virial coefficient, m^3/mol
    db_dt: float  #: dB/dT, m^3/(mol K)
    d2b_dt2: float  #: d^2B/dT^2, m^3/(mol K^2)


def molar_mass(name: str) -> Optional[float]:
    """Molar mass in amu of a constituent gas, or None if unknown.

    Examples
    --------
    >>> round(molar_mass("N2"), 5)
    28.01371
    >>> molar_mass("O*") is None
    True
    """
    entry = _GASES.get(name)
    return entry[0] if entry is not None else None


def gas_properties(name: str, temperature: float) -> Optional[GasProperties]:
    """Properties of a constituent gas at a temperature, None if unknown.

    Warnings are emitted when the temperature falls outside the
    validated range of a gas's specific-heat fit or virial data,
    exactly as the MATLAB source does.

    Examples
    --------
    >>> p = gas_properties("N2", 300.0)
    >>> round(p.molar_mass, 5)
    28.01371
    >>> gas_properties("O*", 300.0) is None
    True
    """
    entry = _GASES.get(name)
    if entry is None:
        return None
    amu, (t_break, lo, hi), c0p_range, virial, virial_range = entry

    coeffs = lo if t_break is None or temperature < t_break else hi
    c0p = sum(c * temperature**k for k, c in enumerate(coeffs))
    if temperature < c0p_range[0] or temperature > c0p_range[1]:
        warnings.warn(
            f"A temperature outside of the modelled range "
            f"({c0p_range[0]:f}K-{c0p_range[1]:f}K) for the ideal gas "
            f"specific heat constant of {name} was provided. The "
            f"results might be unreliable.",
            stacklevel=2,
        )
    c0p *= 4.184 * 1000.0  # cal/(g K) -> J/(kg K)

    if temperature < virial_range[0] or temperature > virial_range[1]:
        warnings.warn(
            f"A temperature outside of the modelled range "
            f"({virial_range[0]:f}K-{virial_range[1]:f}K) for the "
            f"second virial coefficient of {name} was provided. The "
            f"results might be unreliable.",
            stacklevel=2,
        )

    t = temperature
    if virial is not None:
        a, bb, cc = virial
        b = a - bb * math.exp(cc / t)
        db = bb * cc * math.exp(cc / t) / t**2
        d2b = -bb * cc * (cc + 2.0 * t) * math.exp(cc / t) / t**4
    elif name == "He":
        b = float(_HE_SPLINE(t))
        db = float(_HE_SPLINE(t, 1))
        d2b = float(_HE_SPLINE(t, 2))
    else:  # H2O: the Hyland (1975) correlation, transcribed exactly.
        ln10 = math.log(10.0)
        b = 33.97 - (55306.0 / t) * 10.0 ** (72000.0 / t**2)
        db = (
            27653.0
            * 2.0 ** (1.0 + 72000.0 / t**2)
            * 5.0 ** (72000.0 / t**2)
            * (t**2 + 144000.0 * ln10)
        ) / t**4
        # The MATLAB source scales by cosh(2 log 2) + sinh(2 log 2),
        # which is exactly exp(2 log 2) = 4.
        d2b = (
            -(
                27653.0
                * 10.0 ** (72000.0 / t**2)
                * (t**4 + 360000.0 * t**2 * ln10 + 10368000000.0 * ln10**2)
                * 4.0
            )
            / t**7
        )

    # cm^3/mol -> m^3/mol.
    return GasProperties(amu, c0p, b * 1e-6, db * 1e-6, d2b * 1e-6)
