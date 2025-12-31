"""
Special mathematical functions.

This module provides special functions commonly used in mathematical
physics, signal processing, and statistical applications:
- Bessel functions (cylindrical and spherical)
- Gamma and beta functions
- Error functions
- Elliptic integrals
"""

from pytcl.mathematical_functions.special_functions.bessel import (
    airy,
    besselh,
    besseli,
    besselj,
    besselk,
    bessely,
    spherical_in,
    spherical_jn,
    spherical_kn,
    spherical_yn,
)
from pytcl.mathematical_functions.special_functions.elliptic import (  # noqa: E501
    ellipe,
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
]
