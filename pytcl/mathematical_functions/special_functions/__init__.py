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

from pytcl.mathematical_functions.special_functions.bessel import (
    airy,
    bessel_deriv,
    bessel_ratio,
    bessel_zeros,
    besselh,
    besseli,
    besselj,
    besselk,
    bessely,
    kelvin,
    spherical_in,
    spherical_jn,
    spherical_kn,
    spherical_yn,
    struve_h,
    struve_l,
)
from pytcl.mathematical_functions.special_functions.debye import (
    debye,
    debye_1,
    debye_2,
    debye_3,
    debye_4,
    debye_entropy,
    debye_heat_capacity,
)
from pytcl.mathematical_functions.special_functions.elliptic import ellipe  # noqa: E501
from pytcl.mathematical_functions.special_functions.elliptic import (
    ellipeinc,
    ellipk,
    ellipkinc,
    ellipkm1,
    elliprc,
    elliprd,
    elliprf,
    elliprg,
    elliprj,
)
from pytcl.mathematical_functions.special_functions.error_functions import (  # noqa: E501
    dawsn,
    erf,
    erfc,
    erfcinv,
    erfcx,
    erfi,
    erfinv,
    fresnel,
    voigt_profile,
    wofz,
)
from pytcl.mathematical_functions.special_functions.gamma_functions import (  # noqa: E501
    beta,
    betainc,
    betaincinv,
    betaln,
    comb,
    digamma,
    factorial,
    factorial2,
    gamma,
    gammainc,
    gammaincc,
    gammaincinv,
    gammaln,
    perm,
    polygamma,
)
from pytcl.mathematical_functions.special_functions.hypergeometric import (
    falling_factorial,
    generalized_hypergeometric,
    hyp0f1,
    hyp1f1,
    hyp1f1_regularized,
    hyp2f1,
    hyperu,
    pochhammer,
)
from pytcl.mathematical_functions.special_functions.lambert_w import (
    lambert_w,
    lambert_w_real,
    omega_constant,
    solve_exponential_equation,
    time_delay_equation,
    wright_omega,
)
from pytcl.mathematical_functions.special_functions.marcum_q import (
    log_marcum_q,
    marcum_q,
    marcum_q1,
    marcum_q_inv,
    nuttall_q,
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
