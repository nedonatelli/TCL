"""
Relativistic and sidereal time scales.

Completes the astronomical time-scale chains beyond the UTC/TAI/TT
basics of :mod:`pytcl.astronomical.time_systems`: barycentric and
geocentric dynamical/coordinate time (TDB, TCB, TCG), Besselian and
Julian epochs, and Greenwich/local sidereal time.

The MATLAB TCL implements these as MEX stubs over the IAU SOFA
library; this module uses **pyerfa** (ERFA is the liberated SOFA with
identical algorithms, shipped with astropy in the ``astronomy``
extra), so the numerical core is the same code the MATLAB functions
call [1]_. Times follow the SOFA two-part Julian date convention
``(jd1, jd2)`` for full double precision, exactly as the MATLAB
signatures do.

References
----------
.. [1] IAU SOFA Board, "SOFA Time Scale and Calendar Tools,"
   Software version 18, Document revision 1.63, 2021.
"""

from typing import Optional, Tuple

from pytcl.core.exceptions import DependencyError


def _erfa():
    try:
        import erfa

        return erfa
    except ImportError as e:  # pragma: no cover - exercised without the extra
        raise DependencyError(
            "pytcl.astronomical.time_scales requires pyerfa (installed with "
            'the astropy dependency: pip install "nrl-tracker[astronomy]")'
        ) from e


def tt2tdb(
    jd1: float,
    jd2: float,
    delta_t_ut1: float = 0.0,
    clock_loc: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float]:
    """
    Convert terrestrial time (TT) to barycentric dynamical time (TDB).

    Uses the Fairhead & Bretagnon 1990 series (SOFA ``iauDtdb``) for
    the periodic difference, including the topocentric terms when a
    clock location is supplied -- the same computation the MATLAB
    ``TT2TDB`` MEX performs.

    Parameters
    ----------
    jd1, jd2 : float
        TT as a two-part Julian date.
    delta_t_ut1 : float, optional
        TT-UT1 in seconds, used for the topocentric terms' Earth
        rotation angle. Irrelevant (default 0) for a geocentric clock.
    clock_loc : tuple of (east_longitude_rad, u_km, v_km), optional
        Clock site: east longitude in radians, distance from the
        Earth's rotation axis in kilometers, and distance north of the
        equatorial plane in kilometers. None (default) evaluates the
        geocentric series only, as the MATLAB default does. The
        topocentric terms derive the time of day from the
        smaller-magnitude JD part, so use the canonical half-integer
        day / fraction split when passing a clock location (a
        nonstandard split perturbs only the +/-2 microsecond
        topocentric terms).

    Returns
    -------
    jd1, jd2 : float
        TDB as a two-part Julian date.

    Examples
    --------
    >>> tdb1, tdb2 = tt2tdb(2453750.5, 0.892482639)
    >>> round((tdb1 - 2453750.5) + (tdb2 - 0.892482639), 12)  # within +/-1.7 ms
    4.311e-09

    Notes
    -----
    Counterpart of the MATLAB TCL ``TT2TDB``.
    """
    erfa = _erfa()
    if clock_loc is None:
        elong, u, v = 0.0, 0.0, 0.0
    else:
        elong, u, v = clock_loc
    # UT1 fraction of day for the topocentric terms, per the SOFA
    # cookbook: TT minus (TT-UT1).
    ut = ((jd2 if abs(jd1) >= abs(jd2) else jd1) - delta_t_ut1 / 86400.0) % 1.0
    dtr = erfa.dtdb(jd1, jd2, ut, elong, u, v)
    j1, j2 = erfa.tttdb(jd1, jd2, dtr)
    return float(j1), float(j2)


def tdb2tt(
    jd1: float,
    jd2: float,
    delta_t_ut1: float = 0.0,
    clock_loc: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float]:
    """
    Convert barycentric dynamical time (TDB) to terrestrial time (TT).

    Parameters
    ----------
    jd1, jd2 : float
        TDB as a two-part Julian date.
    delta_t_ut1, clock_loc
        As in :func:`tt2tdb`.

    Returns
    -------
    jd1, jd2 : float
        TT as a two-part Julian date.

    Examples
    --------
    >>> tt = tdb2tt(*tt2tdb(2453750.5, 0.892482639))
    >>> round(tt[0] - 2453750.5 + tt[1] - 0.892482639, 14)
    0.0

    Notes
    -----
    Counterpart of the MATLAB TCL ``TDB2TT``. The TDB-TT difference is
    evaluated at the given epoch (the sub-2-millisecond difference in
    the evaluation epoch is far below the series' own accuracy), as in
    SOFA's own usage.
    """
    erfa = _erfa()
    if clock_loc is None:
        elong, u, v = 0.0, 0.0, 0.0
    else:
        elong, u, v = clock_loc
    ut = ((jd2 if abs(jd1) >= abs(jd2) else jd1) - delta_t_ut1 / 86400.0) % 1.0
    dtr = erfa.dtdb(jd1, jd2, ut, elong, u, v)
    j1, j2 = erfa.tdbtt(jd1, jd2, dtr)
    return float(j1), float(j2)


def tt2tcg(jd1: float, jd2: float) -> Tuple[float, float]:
    """
    Convert terrestrial time (TT) to geocentric coordinate time (TCG).

    Parameters
    ----------
    jd1, jd2 : float
        TT as a two-part Julian date.

    Returns
    -------
    jd1, jd2 : float
        TCG as a two-part Julian date.

    Examples
    --------
    >>> tcg = tt2tcg(2453750.5, 0.892482639)
    >>> bool(tcg[1] > 0.892482639)  # TCG runs ahead of TT
    True

    Notes
    -----
    Counterpart of the MATLAB TCL ``TT2TCG`` (SOFA ``iauTttcg``).
    """
    j1, j2 = _erfa().tttcg(jd1, jd2)
    return float(j1), float(j2)


def tcg2tt(jd1: float, jd2: float) -> Tuple[float, float]:
    """
    Convert geocentric coordinate time (TCG) to terrestrial time (TT).

    Parameters
    ----------
    jd1, jd2 : float
        TCG as a two-part Julian date.

    Returns
    -------
    jd1, jd2 : float
        TT as a two-part Julian date.

    Examples
    --------
    >>> tt = tcg2tt(*tt2tcg(2453750.5, 0.892482639))
    >>> round(tt[0] - 2453750.5 + tt[1] - 0.892482639, 14)
    0.0

    Notes
    -----
    Counterpart of the MATLAB TCL ``TCG2TT`` (SOFA ``iauTcgtt``).
    """
    j1, j2 = _erfa().tcgtt(jd1, jd2)
    return float(j1), float(j2)


def tdb2tcb(jd1: float, jd2: float) -> Tuple[float, float]:
    """
    Convert barycentric dynamical time (TDB) to barycentric coordinate
    time (TCB).

    Parameters
    ----------
    jd1, jd2 : float
        TDB as a two-part Julian date.

    Returns
    -------
    jd1, jd2 : float
        TCB as a two-part Julian date.

    Examples
    --------
    >>> tcb = tdb2tcb(2453750.5, 0.892482639)
    >>> bool(tcb[1] > 0.892482639)  # TCB runs ahead of TDB
    True

    Notes
    -----
    Counterpart of the MATLAB TCL ``TDB2TCB`` (SOFA ``iauTdbtcb``).
    """
    j1, j2 = _erfa().tdbtcb(jd1, jd2)
    return float(j1), float(j2)


def tcb2tdb(jd1: float, jd2: float) -> Tuple[float, float]:
    """
    Convert barycentric coordinate time (TCB) to barycentric dynamical
    time (TDB).

    Parameters
    ----------
    jd1, jd2 : float
        TCB as a two-part Julian date.

    Returns
    -------
    jd1, jd2 : float
        TDB as a two-part Julian date.

    Examples
    --------
    >>> tdb = tcb2tdb(*tdb2tcb(2453750.5, 0.892482639))
    >>> round(tdb[0] - 2453750.5 + tdb[1] - 0.892482639, 14)
    0.0

    Notes
    -----
    Counterpart of the MATLAB TCL ``TCB2TDB`` (SOFA ``iauTcbtdb``).
    """
    j1, j2 = _erfa().tcbtdb(jd1, jd2)
    return float(j1), float(j2)


def tdb2besselian_epoch(jd1: float, jd2: float) -> float:
    """
    Convert a TDB Julian date to a Besselian epoch.

    Parameters
    ----------
    jd1, jd2 : float
        TDB as a two-part Julian date.

    Returns
    -------
    epoch : float
        Besselian epoch (e.g. 1950.0).

    Examples
    --------
    >>> round(tdb2besselian_epoch(2433282.42345905, 0.0), 6)  # B1950.0
    1950.0

    Notes
    -----
    Counterpart of the MATLAB TCL ``TDB2BesselEpoch`` (SOFA
    ``iauEpb``).
    """
    return float(_erfa().epb(jd1, jd2))


def besselian_epoch2tdb(epoch: float) -> Tuple[float, float]:
    """
    Convert a Besselian epoch to a TDB Julian date.

    Parameters
    ----------
    epoch : float
        Besselian epoch (e.g. 1950.0).

    Returns
    -------
    jd1, jd2 : float
        TDB as a two-part Julian date.

    Examples
    --------
    >>> round(tdb2besselian_epoch(*besselian_epoch2tdb(1975.25)), 9)
    1975.25

    Notes
    -----
    Counterpart of the MATLAB TCL ``BesselEpoch2TDB`` (SOFA
    ``iauEpb2jd``).
    """
    j1, j2 = _erfa().epb2jd(epoch)
    return float(j1), float(j2)


def jul_date2jul_epoch(jd1: float, jd2: float) -> float:
    """
    Convert a Julian date to a Julian epoch.

    Parameters
    ----------
    jd1, jd2 : float
        Two-part Julian date (any uniform time scale).

    Returns
    -------
    epoch : float
        Julian epoch (e.g. 2000.0).

    Examples
    --------
    >>> jul_date2jul_epoch(2451545.0, 0.0)  # J2000.0 by definition
    2000.0

    Notes
    -----
    Counterpart of the MATLAB TCL ``JulDate2JulEpoch`` (SOFA
    ``iauEpj``).
    """
    return float(_erfa().epj(jd1, jd2))


def jul_epoch2jul_date(epoch: float) -> Tuple[float, float]:
    """
    Convert a Julian epoch to a Julian date.

    Parameters
    ----------
    epoch : float
        Julian epoch (e.g. 2000.0).

    Returns
    -------
    jd1, jd2 : float
        Two-part Julian date.

    Examples
    --------
    >>> sum(jul_epoch2jul_date(2000.0))
    2451545.0

    Notes
    -----
    Counterpart of the MATLAB TCL ``JulEpoch2JulDate`` (SOFA
    ``iauEpj2jd``).
    """
    j1, j2 = _erfa().epj2jd(epoch)
    return float(j1), float(j2)


def tt2gmst(jd1: float, jd2: float, delta_t_ut1: float = 0.0) -> float:
    """
    Greenwich mean sidereal time (IAU 2006) from TT.

    Parameters
    ----------
    jd1, jd2 : float
        TT as a two-part Julian date.
    delta_t_ut1 : float, optional
        TT-UT1 in seconds (default 0), used to form the UT1 argument.

    Returns
    -------
    gmst : float
        Greenwich mean sidereal time in radians, [0, 2*pi).

    Examples
    --------
    >>> g = tt2gmst(2453750.5, 0.892482639)
    >>> bool(0.0 <= g < 6.2831854)
    True

    Notes
    -----
    Counterpart of the MATLAB TCL ``TT2GMST`` (SOFA ``iauGmst06``);
    the module-level :func:`pytcl.astronomical.reference_frames.gmst_iau82`
    remains the older IAU 1982 model.
    """
    erfa = _erfa()
    ut1_1, ut1_2 = erfa.ttut1(jd1, jd2, delta_t_ut1)
    return float(erfa.gmst06(ut1_1, ut1_2, jd1, jd2))


def tt2gast(jd1: float, jd2: float, delta_t_ut1: float = 0.0) -> float:
    """
    Greenwich apparent sidereal time (IAU 2006/2000A) from TT.

    Parameters
    ----------
    jd1, jd2 : float
        TT as a two-part Julian date.
    delta_t_ut1 : float, optional
        TT-UT1 in seconds (default 0).

    Returns
    -------
    gast : float
        Greenwich apparent sidereal time in radians, [0, 2*pi).

    Examples
    --------
    >>> import math
    >>> g = tt2gast(2453750.5, 0.892482639)
    >>> m = tt2gmst(2453750.5, 0.892482639)
    >>> bool(abs(g - m) < 2e-4)  # the equation of the origins is small
    True

    Notes
    -----
    Counterpart of the MATLAB TCL ``TT2GAST`` (SOFA ``iauGst06a``).
    """
    erfa = _erfa()
    ut1_1, ut1_2 = erfa.ttut1(jd1, jd2, delta_t_ut1)
    return float(erfa.gst06a(ut1_1, ut1_2, jd1, jd2))


def tt2lmst(
    jd1: float, jd2: float, east_longitude: float, delta_t_ut1: float = 0.0
) -> float:
    """
    Local mean sidereal time from TT.

    Parameters
    ----------
    jd1, jd2 : float
        TT as a two-part Julian date.
    east_longitude : float
        Observer's east longitude in radians.
    delta_t_ut1 : float, optional
        TT-UT1 in seconds (default 0).

    Returns
    -------
    lmst : float
        Local mean sidereal time in radians, [0, 2*pi).

    Examples
    --------
    >>> import math
    >>> g = tt2gmst(2453750.5, 0.892482639)
    >>> loc = tt2lmst(2453750.5, 0.892482639, math.pi / 4)
    >>> round((loc - g) % (2 * math.pi), 10) == round(math.pi / 4, 10)
    True

    Notes
    -----
    Counterpart of the MATLAB TCL ``TT2LMST``: GMST plus the east
    longitude, wrapped to [0, 2*pi).
    """
    erfa = _erfa()
    return float(erfa.anp(tt2gmst(jd1, jd2, delta_t_ut1) + east_longitude))


def tt2last(
    jd1: float, jd2: float, east_longitude: float, delta_t_ut1: float = 0.0
) -> float:
    """
    Local apparent sidereal time from TT.

    Parameters
    ----------
    jd1, jd2 : float
        TT as a two-part Julian date.
    east_longitude : float
        Observer's east longitude in radians.
    delta_t_ut1 : float, optional
        TT-UT1 in seconds (default 0).

    Returns
    -------
    last : float
        Local apparent sidereal time in radians, [0, 2*pi).

    Examples
    --------
    >>> import math
    >>> g = tt2gast(2453750.5, 0.892482639)
    >>> loc = tt2last(2453750.5, 0.892482639, -math.pi / 3)
    >>> round((loc - g) % (2 * math.pi), 10) == round(2 * math.pi - math.pi / 3, 10)
    True

    Notes
    -----
    Counterpart of the MATLAB TCL ``TT2LAST``: GAST plus the east
    longitude, wrapped to [0, 2*pi).
    """
    erfa = _erfa()
    return float(erfa.anp(tt2gast(jd1, jd2, delta_t_ut1) + east_longitude))


__all__ = [
    "besselian_epoch2tdb",
    "jul_date2jul_epoch",
    "jul_epoch2jul_date",
    "tcb2tdb",
    "tcg2tt",
    "tdb2besselian_epoch",
    "tdb2tcb",
    "tdb2tt",
    "tt2gast",
    "tt2gmst",
    "tt2last",
    "tt2lmst",
    "tt2tcg",
    "tt2tdb",
]
