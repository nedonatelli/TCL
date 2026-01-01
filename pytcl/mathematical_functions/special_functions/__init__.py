"""
Special mathematical functions.

This module provides special functions commonly used in mathematical
physics, signal processing, and statistical applications:
- Bessel functions (cylindrical and spherical)
- Gamma and beta functions
- Error functions
- Elliptic integrals
- Marcum Q function (radar detection)
- Hypergeometric functions
- Lambert W function
- Debye functions (thermodynamics)
"""

from pytcl.mathematical_functions.special_functions.bessel import airy
from pytcl.mathematical_functions.special_functions.bessel import bessel_deriv
from pytcl.mathematical_functions.special_functions.bessel import bessel_ratio
from pytcl.mathematical_functions.special_functions.bessel import bessel_zeros
from pytcl.mathematical_functions.special_functions.bessel import besselh
from pytcl.mathematical_functions.special_functions.bessel import besseli
from pytcl.mathematical_functions.special_functions.bessel import besselj
from pytcl.mathematical_functions.special_functions.bessel import besselk
from pytcl.mathematical_functions.special_functions.bessel import bessely
from pytcl.mathematical_functions.special_functions.bessel import kelvin
from pytcl.mathematical_functions.special_functions.bessel import spherical_in
from pytcl.mathematical_functions.special_functions.bessel import spherical_jn
from pytcl.mathematical_functions.special_functions.bessel import spherical_kn
from pytcl.mathematical_functions.special_functions.bessel import spherical_yn
from pytcl.mathematical_functions.special_functions.bessel import struve_h
from pytcl.mathematical_functions.special_functions.bessel import struve_l
from pytcl.mathematical_functions.special_functions.debye import debye
from pytcl.mathematical_functions.special_functions.debye import debye_1
from pytcl.mathematical_functions.special_functions.debye import debye_2
from pytcl.mathematical_functions.special_functions.debye import debye_3
from pytcl.mathematical_functions.special_functions.debye import debye_4
from pytcl.mathematical_functions.special_functions.debye import debye_entropy
from pytcl.mathematical_functions.special_functions.debye import debye_heat_capacity
from pytcl.mathematical_functions.special_functions.elliptic import ellipe  # noqa: E501
from pytcl.mathematical_functions.special_functions.elliptic import ellipeinc
from pytcl.mathematical_functions.special_functions.elliptic import ellipk
from pytcl.mathematical_functions.special_functions.elliptic import ellipkinc
from pytcl.mathematical_functions.special_functions.elliptic import ellipkm1
from pytcl.mathematical_functions.special_functions.elliptic import elliprc
from pytcl.mathematical_functions.special_functions.elliptic import elliprd
from pytcl.mathematical_functions.special_functions.elliptic import elliprf
from pytcl.mathematical_functions.special_functions.elliptic import elliprg
from pytcl.mathematical_functions.special_functions.elliptic import elliprj
from pytcl.mathematical_functions.special_functions.error_functions import (  # noqa: E501
    dawsn,
)
from pytcl.mathematical_functions.special_functions.error_functions import erf
from pytcl.mathematical_functions.special_functions.error_functions import erfc
from pytcl.mathematical_functions.special_functions.error_functions import erfcinv
from pytcl.mathematical_functions.special_functions.error_functions import erfcx
from pytcl.mathematical_functions.special_functions.error_functions import erfi
from pytcl.mathematical_functions.special_functions.error_functions import erfinv
from pytcl.mathematical_functions.special_functions.error_functions import fresnel
from pytcl.mathematical_functions.special_functions.error_functions import voigt_profile
from pytcl.mathematical_functions.special_functions.error_functions import wofz
from pytcl.mathematical_functions.special_functions.gamma_functions import (  # noqa: E501
    beta,
)
from pytcl.mathematical_functions.special_functions.gamma_functions import betainc
from pytcl.mathematical_functions.special_functions.gamma_functions import betaincinv
from pytcl.mathematical_functions.special_functions.gamma_functions import betaln
from pytcl.mathematical_functions.special_functions.gamma_functions import comb
from pytcl.mathematical_functions.special_functions.gamma_functions import digamma
from pytcl.mathematical_functions.special_functions.gamma_functions import factorial
from pytcl.mathematical_functions.special_functions.gamma_functions import factorial2
from pytcl.mathematical_functions.special_functions.gamma_functions import gamma
from pytcl.mathematical_functions.special_functions.gamma_functions import gammainc
from pytcl.mathematical_functions.special_functions.gamma_functions import gammaincc
from pytcl.mathematical_functions.special_functions.gamma_functions import gammaincinv
from pytcl.mathematical_functions.special_functions.gamma_functions import gammaln
from pytcl.mathematical_functions.special_functions.gamma_functions import perm
from pytcl.mathematical_functions.special_functions.gamma_functions import polygamma
from pytcl.mathematical_functions.special_functions.hypergeometric import (
    falling_factorial,
)
from pytcl.mathematical_functions.special_functions.hypergeometric import (
    generalized_hypergeometric,
)
from pytcl.mathematical_functions.special_functions.hypergeometric import hyp0f1
from pytcl.mathematical_functions.special_functions.hypergeometric import hyp1f1
from pytcl.mathematical_functions.special_functions.hypergeometric import (
    hyp1f1_regularized,
)
from pytcl.mathematical_functions.special_functions.hypergeometric import hyp2f1
from pytcl.mathematical_functions.special_functions.hypergeometric import hyperu
from pytcl.mathematical_functions.special_functions.hypergeometric import pochhammer
from pytcl.mathematical_functions.special_functions.lambert_w import lambert_w
from pytcl.mathematical_functions.special_functions.lambert_w import lambert_w_real
from pytcl.mathematical_functions.special_functions.lambert_w import omega_constant
from pytcl.mathematical_functions.special_functions.lambert_w import (
    solve_exponential_equation,
)
from pytcl.mathematical_functions.special_functions.lambert_w import time_delay_equation
from pytcl.mathematical_functions.special_functions.lambert_w import wright_omega
from pytcl.mathematical_functions.special_functions.marcum_q import log_marcum_q
from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q
from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q1
from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q_inv
from pytcl.mathematical_functions.special_functions.marcum_q import nuttall_q
from pytcl.mathematical_functions.special_functions.marcum_q import (
    swerling_detection_probability,
)

__all__ = [
    # Bessel functions
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "besselh",
    "spherical_jn",
    "spherical_yn",
    "spherical_in",
    "spherical_kn",
    "airy",
    "bessel_ratio",
    "bessel_deriv",
    "bessel_zeros",
    "struve_h",
    "struve_l",
    "kelvin",
    # Gamma functions
    "gamma",
    "gammaln",
    "gammainc",
    "gammaincc",
    "gammaincinv",
    "digamma",
    "polygamma",
    "beta",
    "betaln",
    "betainc",
    "betaincinv",
    "factorial",
    "factorial2",
    "comb",
    "perm",
    # Error functions
    "erf",
    "erfc",
    "erfcx",
    "erfi",
    "erfinv",
    "erfcinv",
    "dawsn",
    "fresnel",
    "wofz",
    "voigt_profile",
    # Elliptic integrals
    "ellipk",
    "ellipkm1",
    "ellipe",
    "ellipeinc",
    "ellipkinc",
    "elliprd",
    "elliprf",
    "elliprg",
    "elliprj",
    "elliprc",
    # Marcum Q function (radar detection)
    "marcum_q",
    "marcum_q1",
    "log_marcum_q",
    "marcum_q_inv",
    "nuttall_q",
    "swerling_detection_probability",
    # Lambert W function
    "lambert_w",
    "lambert_w_real",
    "omega_constant",
    "wright_omega",
    "solve_exponential_equation",
    "time_delay_equation",
    # Debye functions
    "debye",
    "debye_1",
    "debye_2",
    "debye_3",
    "debye_4",
    "debye_heat_capacity",
    "debye_entropy",
    # Hypergeometric functions
    "hyp0f1",
    "hyp1f1",
    "hyp2f1",
    "hyperu",
    "hyp1f1_regularized",
    "pochhammer",
    "falling_factorial",
    "generalized_hypergeometric",
]
