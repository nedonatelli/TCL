"""
Special mathematical functions.

This module provides special functions commonly used in mathematical
physics, signal processing, and statistical applications:
- Bessel functions (cylindrical and spherical)
- Gamma and beta functions
- Error functions
- Elliptic integrals
"""

from tracker_component_library.mathematical_functions.special_functions.bessel import (
    besselj,
    bessely,
    besseli,
    besselk,
    besselh,
    spherical_jn,
    spherical_yn,
    spherical_in,
    spherical_kn,
    airy,
)

from tracker_component_library.mathematical_functions.special_functions.gamma_functions import (  # noqa: E501
    gamma,
    gammaln,
    gammainc,
    gammaincc,
    gammaincinv,
    digamma,
    polygamma,
    beta,
    betaln,
    betainc,
    betaincinv,
    factorial,
    factorial2,
    comb,
    perm,
)

from tracker_component_library.mathematical_functions.special_functions.error_functions import (  # noqa: E501
    erf,
    erfc,
    erfcx,
    erfi,
    erfinv,
    erfcinv,
    dawsn,
    fresnel,
    wofz,
    voigt_profile,
)

from tracker_component_library.mathematical_functions.special_functions.elliptic import (  # noqa: E501
    ellipk,
    ellipkm1,
    ellipe,
    ellipeinc,
    ellipkinc,
    elliprd,
    elliprf,
    elliprg,
    elliprj,
    elliprc,
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
