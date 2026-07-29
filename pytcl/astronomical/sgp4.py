"""
SGP4/SDP4 Satellite Propagation Models.

This module implements the Simplified General Perturbations model (SGP4)
and its deep-space extension (SDP4) for propagating satellite orbits
from Two-Line Element (TLE) sets.

SGP4 models the effects of:

- Atmospheric drag (via the B* term)
- J2, J3, J4 gravitational harmonics
- Secular and periodic variations

SDP4 additionally models (for orbital periods >= 225 min, computed from
the recovered mean motion):

- Lunar gravitational perturbations (secular and periodic)
- Solar gravitational perturbations (secular and periodic)
- Geopotential resonance for 12-hour and 24-hour orbits, integrated with
  the Euler-Maclaurin scheme of the reference algorithm

These are implemented as the four standard deep-space routines DSCOM,
DSINIT, DSPACE and DPPER.

The output is in the TEME (True Equator, Mean Equinox) reference frame,
which is a quasi-inertial frame used by NORAD.

Limitations
-----------
- Only the WGS-72 gravity model is supported; the reference implementation
  also offers WGS-72 (old) and WGS-84.
- Only the "improved" operation mode (``opsmode = 'i'``) is implemented,
  which is the recommended mode; the legacy AFSPC mode differs in the
  sidereal-time formula and in two angle-wrapping conventions.

References
----------
- Hoots, F. R. and Roehrich, R. L., "Spacetrack Report No. 3:
  Models for Propagation of NORAD Element Sets," 1980.
- Vallado, D. A., Crawford, P., Hujsak, R., and Kelso, T.S.,
  "Revisiting Spacetrack Report #3," AIAA 2006-6753.
- Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
  4th ed., Microcosm Press, 2013.
"""

from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.astronomical.tle import TLE, is_deep_space, tle_epoch_to_jd

# =============================================================================
# Constants (WGS-72 values used by SGP4)
# =============================================================================

# Earth parameters (WGS-72, as used in original SGP4)
MU_EARTH = 398600.8  # km^3/s^2 (WGS-72 value)
RADIUS_EARTH = 6378.135  # km (WGS-72)
J2 = 1.082616e-3
J3 = -2.53881e-6
J4 = -1.65597e-6

# Derived constants
# KE relates mean motion (rad/min) to semi-major axis (Earth radii)
KE = 60.0 / np.sqrt(RADIUS_EARTH**3 / MU_EARTH)  # (1/min)

# In SGP4, semi-major axis is in Earth radii, so K2, K4 are dimensionless
# (not multiplied by RADIUS_EARTH^2 or RADIUS_EARTH^4)
K2 = 0.5 * J2
K4 = -0.375 * J4
A30_OVER_K2 = -J3 / K2

# Atmospheric parameters
Q0 = 120.0  # km
S0 = 78.0  # km
QOMS2T = ((Q0 - S0) / RADIUS_EARTH) ** 4

# Earth rotation rate (rad/min)
OMEGA_EARTH = 7.29211514670698e-5 * 60.0  # rad/min

# Time constants
MINUTES_PER_DAY = 1440.0

# Two-thirds
TWO_THIRDS = 2.0 / 3.0

TWO_PI = 2.0 * np.pi

# -----------------------------------------------------------------------------
# Deep-space (SDP4) constants
#
# Symbols follow the published algorithm (Spacetrack Report No. 3, Appendix 1;
# Vallado et al., AIAA 2006-6753, routines DSCOM/DSINIT/DSPACE/DPPER).
# -----------------------------------------------------------------------------

# Solar and lunar mean motions (rad/min) and eccentricities
ZNS = 1.19459e-5
ZES = 0.01675
ZNL = 1.5835218e-4
ZEL = 0.05490

# Solar perturbation scale factor and the Sun's orientation at the reference
# epoch (obliquity-rotated ecliptic elements)
C1SS = 2.9864797e-6
ZCOSIS = 0.91744867
ZSINIS = 0.39785416
ZCOSGS = 0.1945905
ZSINGS = -0.98088458

# Lunar perturbation scale factor
C1L = 4.7968065e-7

# Tesseral harmonic amplitudes used by the resonance terms
Q22 = 1.7891679e-6
Q31 = 2.1460748e-6
Q33 = 2.2123015e-7
ROOT22 = 1.7891679e-6
ROOT32 = 3.7393792e-7
ROOT44 = 7.3636953e-9
ROOT52 = 1.1428639e-7
ROOT54 = 2.1765803e-9

# Earth rotation rate used by the resonance terms (rad/min)
RPTIM = 4.37526908801129966e-3

# Resonance phase angles
FASX2 = 0.13130908
FASX4 = 2.8843198
FASX6 = 0.37448087
G22 = 5.7686396
G32 = 0.95240898
G44 = 1.8014998
G52 = 1.0508330
G54 = 4.4108898

# Euler-Maclaurin integration step for the resonance integrator (minutes)
STEPP = 720.0
STEPN = -720.0
STEP2 = 259200.0


def _gstime(jdut1: float) -> float:
    """Greenwich mean sidereal time (rad) from a UT1 Julian date.

    Implements the IAU-82 GMST polynomial used by the reference SGP4
    initialization (``opsmode = 'i'``).

    Parameters
    ----------
    jdut1 : float
        Julian date (UT1).

    Returns
    -------
    float
        Greenwich mean sidereal time in radians, on [0, 2*pi).
    """
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (
        -6.2e-6 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 67310.54841
    )
    # Seconds of time -> radians (240 s per degree)
    temp = np.deg2rad(temp / 240.0) % TWO_PI
    if temp < 0.0:
        temp += TWO_PI
    return float(temp)


def unkozai_mean_motion(no: float, inclination: float, eccentricity: float) -> float:
    """Recover the un-Kozai'd (Brouwer) mean motion from a TLE mean motion.

    TLE mean motions are Kozai mean motions; SGP4 works with the Brouwer
    mean motion obtained by removing the secular J2 contribution. The
    recovered value also decides whether the deep-space model applies.

    Parameters
    ----------
    no : float
        Kozai mean motion from the TLE (rad/min).
    inclination : float
        Inclination (rad).
    eccentricity : float
        Eccentricity.

    Returns
    -------
    float
        Un-Kozai'd mean motion (rad/min).

    Examples
    --------
    >>> n = unkozai_mean_motion(0.0676, 0.9013, 0.0006703)
    >>> bool(n < 0.0676)
    True
    """
    a1 = (KE / no) ** TWO_THIRDS
    x3thm1 = 3.0 * np.cos(inclination) ** 2 - 1.0
    betao2 = 1.0 - eccentricity * eccentricity
    betao = np.sqrt(betao2)
    delta1 = 1.5 * K2 * x3thm1 / (a1 * a1 * betao * betao2)
    a0 = a1 * (1.0 - delta1 * (1.0 / 3.0 + delta1 * (1.0 + 134.0 / 81.0 * delta1)))
    delta0 = 1.5 * K2 * x3thm1 / (a0 * a0 * betao * betao2)
    return float(no / (1.0 + delta0))


class SGP4State(NamedTuple):
    """State vector from SGP4 propagation.

    Attributes
    ----------
    r : ndarray
        Position in TEME frame (km), shape (3,).
    v : ndarray
        Velocity in TEME frame (km/s), shape (3,).
    error : int
        Error code (0 = success).
    """

    r: NDArray[np.floating]
    v: NDArray[np.floating]
    error: int


def _failed_state(code: int) -> SGP4State:
    """Build a state for a propagation that could not be completed.

    Parameters
    ----------
    code : int
        Reference SGP4 error code: 1 = mean eccentricity out of range,
        2 = mean motion non-positive, 3 = perturbed eccentricity out of
        range, 4 = negative semi-latus rectum.
    """
    return SGP4State(r=np.full(3, np.nan), v=np.full(3, np.nan), error=code)


class SGP4Satellite:
    """SGP4 satellite propagator initialized from a TLE.

    This class encapsulates the initialization and propagation logic
    for a satellite using the SGP4/SDP4 models.

    Parameters
    ----------
    tle : TLE
        Two-Line Element set.

    Attributes
    ----------
    tle : TLE
        Original TLE data.
    epoch_jd : float
        Julian date of TLE epoch.
    is_deep_space : bool
        True if SDP4 (deep-space) propagation is used.

    Examples
    --------
    >>> from pytcl.astronomical.tle import parse_tle
    >>> line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
    >>> line2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003"
    >>> tle = parse_tle(line1, line2, name="ISS")
    >>> sat = SGP4Satellite(tle)
    >>> state = sat.propagate(0.0)  # At epoch
    >>> bool(6700 < np.linalg.norm(state.r) < 6900)  # ISS orbital radius (km)
    True
    >>> state = sat.propagate(60.0)  # 60 minutes later
    """

    def __init__(self, tle: TLE):
        """Initialize SGP4 satellite from TLE."""
        self.tle = tle
        self.epoch_jd = tle_epoch_to_jd(tle)
        self.is_deep_space = is_deep_space(tle)

        # Initialize orbital elements
        self._initialize()

    def _initialize(self) -> None:
        """Initialize SGP4/SDP4 orbital elements and propagation constants."""
        tle = self.tle

        # Extract TLE elements
        self.inclo = tle.inclination  # rad
        self.nodeo = tle.raan  # rad
        self.ecco = tle.eccentricity
        self.argpo = tle.arg_perigee  # rad
        self.mo = tle.mean_anomaly  # rad
        self.no = tle.mean_motion  # rad/min
        self.bstar = tle.bstar

        # Recover mean motion and semi-major axis
        # First guess for a1
        a1 = (KE / self.no) ** TWO_THIRDS

        # Iterate to get better estimate
        cosi = np.cos(self.inclo)
        theta2 = cosi * cosi
        x3thm1 = 3.0 * theta2 - 1.0
        eosq = self.ecco * self.ecco
        betao2 = 1.0 - eosq
        betao = np.sqrt(betao2)

        delta1 = 1.5 * K2 * x3thm1 / (a1 * a1 * betao * betao2)
        a0 = a1 * (1.0 - delta1 * (1.0 / 3.0 + delta1 * (1.0 + 134.0 / 81.0 * delta1)))
        delta0 = 1.5 * K2 * x3thm1 / (a0 * a0 * betao * betao2)

        # Recovered mean motion and semi-major axis. The semi-major axis is
        # taken as the Kepler radius of the un-Kozai'd mean motion (Vallado's
        # form) rather than the equivalent-to-third-order a0 / (1 - delta0):
        # the two differ by O(delta0^3), and only the former stays exactly
        # consistent with the mean motion used during propagation.
        self.no_kozai = float(self.no / (1.0 + delta0))
        self.ao = (KE / self.no_kozai) ** TWO_THIRDS

        # Store commonly used values
        self.sinio = np.sin(self.inclo)
        self.cosio = cosi
        self.theta2 = theta2
        self.x3thm1 = x3thm1
        self.eosq = eosq
        self.betao = betao
        self.betao2 = betao2

        # For convenience
        self.x1mth2 = 1.0 - theta2
        self.x7thm1 = 7.0 * theta2 - 1.0

        # Compute s and qoms2t based on perigee height
        perigee = (self.ao * (1.0 - self.ecco) - 1.0) * RADIUS_EARTH
        if perigee < 156.0:
            s4 = perigee - 78.0
            if perigee < 98.0:
                s4 = 20.0
            qzms24 = ((120.0 - s4) / RADIUS_EARTH) ** 4
            s4 = s4 / RADIUS_EARTH + 1.0
        else:
            s4 = 1.0 + S0 / RADIUS_EARTH
            qzms24 = QOMS2T

        self.s4 = s4
        self.qzms24 = qzms24

        # Compute constants
        pinvsq = 1.0 / (self.ao * self.ao * self.betao2 * self.betao2)
        tsi = 1.0 / (self.ao - s4)
        self.eta = self.ao * self.ecco * tsi
        etasq = self.eta * self.eta
        eeta = self.ecco * self.eta
        psisq = abs(1.0 - etasq)
        coef = qzms24 * (tsi**4)
        coef1 = coef / (psisq**3.5)

        c2 = (
            coef1
            * self.no_kozai
            * (
                self.ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
                + 0.75
                * K2
                * tsi
                / psisq
                * self.x3thm1
                * (8.0 + 3.0 * etasq * (8.0 + etasq))
            )
        )
        self.c1 = self.bstar * c2

        self.c4 = (
            2.0
            * self.no_kozai
            * coef1
            * self.ao
            * self.betao2
            * (
                self.eta * (2.0 + 0.5 * etasq)
                + self.ecco * (0.5 + 2.0 * etasq)
                - 2.0
                * K2
                * tsi
                / (self.ao * psisq)
                * (
                    -3.0 * self.x3thm1 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                    + 0.75
                    * self.x1mth2
                    * (2.0 * etasq - eeta * (1.0 + etasq))
                    * np.cos(2.0 * self.argpo)
                )
            )
        )

        self.c5 = (
            2.0
            * coef1
            * self.ao
            * self.betao2
            * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
        )

        theta4 = theta2 * theta2
        temp1 = 3.0 * K2 * pinvsq * self.no_kozai
        temp2 = temp1 * K2 * pinvsq
        temp3 = 1.25 * K4 * pinvsq * pinvsq * self.no_kozai

        self.mdot = (
            self.no_kozai
            + 0.5 * temp1 * self.betao * self.x3thm1
            + 0.0625 * temp2 * self.betao * (13.0 - 78.0 * theta2 + 137.0 * theta4)
        )

        # con42 = 1 - 5*cos^2(i) (Vallado's notation)
        con42 = 1.0 - 5.0 * theta2
        self.argpdot = (
            -0.5 * temp1 * con42
            + 0.0625 * temp2 * (7.0 - 114.0 * theta2 + 395.0 * theta4)
            + temp3 * (3.0 - 36.0 * theta2 + 49.0 * theta4)
        )

        xhdot1 = -temp1 * self.cosio
        self.nodedot = (
            xhdot1
            + (0.5 * temp2 * (4.0 - 19.0 * theta2) + 2.0 * temp3 * (3.0 - 7.0 * theta2))
            * self.cosio
        )

        self.xnodcf = 3.5 * self.betao2 * xhdot1 * self.c1
        self.t2cof = 1.5 * self.c1

        # Additional constants for non-simplified propagation
        if abs(1.0 + self.cosio) > 1.5e-12:
            self.xlcof = (
                0.125
                * A30_OVER_K2
                * self.sinio
                * (3.0 + 5.0 * self.cosio)
                / (1.0 + self.cosio)
            )
        else:
            self.xlcof = (
                0.125 * A30_OVER_K2 * self.sinio * (3.0 + 5.0 * self.cosio) / 1.5e-12
            )

        self.aycof = 0.25 * A30_OVER_K2 * self.sinio
        self.x7thm1 = 7.0 * theta2 - 1.0

        # Drag periodic coefficients (Vallado's omgcof/xmcof/delmo/sinmao)
        if self.ecco > 1.0e-4:
            c3 = coef * tsi * A30_OVER_K2 * self.no_kozai * self.sinio / self.ecco
            self.xmcof = -TWO_THIRDS * coef * self.bstar / eeta
        else:
            c3 = 0.0
            self.xmcof = 0.0
        self.omgcof = self.bstar * c3 * np.cos(self.argpo)
        self.delmo = (1.0 + self.eta * np.cos(self.mo)) ** 3
        self.sinmao = np.sin(self.mo)

        # Simplified drag flag: skip higher-order drag terms for very low
        # perigees (< 220 km) and for deep-space orbits (Vallado's isimp)
        rp = self.ao * (1.0 - self.ecco)
        self.isimp = self.is_deep_space or rp < (220.0 / RADIUS_EARTH + 1.0)

        # Higher-order drag coefficients (d2-d4, t3cof-t5cof)
        if not self.isimp:
            c1sq = self.c1 * self.c1
            self.d2 = 4.0 * self.ao * tsi * c1sq
            temp = self.d2 * tsi * self.c1 / 3.0
            self.d3 = (17.0 * self.ao + s4) * temp
            self.d4 = (
                0.5 * temp * self.ao * tsi * (221.0 * self.ao + 31.0 * s4) * self.c1
            )
            self.t3cof = self.d2 + 2.0 * c1sq
            self.t4cof = 0.25 * (
                3.0 * self.d3 + self.c1 * (12.0 * self.d2 + 10.0 * c1sq)
            )
            self.t5cof = 0.2 * (
                3.0 * self.d4
                + 12.0 * self.c1 * self.d3
                + 6.0 * self.d2 * self.d2
                + 15.0 * c1sq * (2.0 * self.d2 + c1sq)
            )
        else:
            self.d2 = self.d3 = self.d4 = 0.0
            self.t3cof = self.t4cof = self.t5cof = 0.0

        # For deep space
        self._ds_initialized = False
        if self.is_deep_space:
            self._init_deep_space()

    def _init_deep_space(self) -> None:
        """Initialize deep-space (SDP4) constants.

        Runs the ``DSCOM`` and ``DSINIT`` steps of the published algorithm:
        the solar/lunar ephemeris coefficients at epoch, the secular rates
        due to lunar-solar gravity, and (for 12-hour and 24-hour orbits)
        the geopotential resonance coefficients and integrator seed.
        """
        self.jd_epoch = self.epoch_jd

        # Greenwich mean sidereal time at epoch (opsmode 'i')
        self.gsto = _gstime(self.epoch_jd)

        # Days since 1949 December 31 00:00 UT
        epoch = self.epoch_jd - 2433281.5

        self._dscom(epoch, tc=0.0)
        # DPPER is called once at initialization with init='y'; with the
        # zeroed peo/pinco/plo/pgho/pho of DSCOM it leaves the elements
        # unchanged, but it is retained for fidelity with the reference.
        (self.ecco, self.inclo, self.nodeo, self.argpo, self.mo) = self._dpper(
            0.0, self.ecco, self.inclo, self.nodeo, self.argpo, self.mo, init=True
        )
        self._dsinit()

        self._ds_initialized = True
        self.resonance_flag = self.irez != 0
        self.synchronous_flag = self.irez == 1

    # -------------------------------------------------------------------------
    # Deep-space routines (Spacetrack Report No. 3 / Vallado AIAA 2006-6753)
    # -------------------------------------------------------------------------

    def _dscom(self, epoch: float, tc: float) -> None:
        """DSCOM -- deep-space common quantities from the solar/lunar ephemeris.

        Parameters
        ----------
        epoch : float
            Days since 1949 December 31 00:00 UT.
        tc : float
            Time offset from epoch (minutes); zero at initialization.
        """
        ep = self.ecco
        argpp = self.argpo
        inclp = self.inclo
        nodep = self.nodeo
        np_ = self.no_kozai

        nm = np_
        em = ep
        snodm = np.sin(nodep)
        cnodm = np.cos(nodep)
        sinomm = np.sin(argpp)
        cosomm = np.cos(argpp)
        sinim = np.sin(inclp)
        cosim = np.cos(inclp)
        emsq = em * em
        betasq = 1.0 - emsq
        rtemsq = np.sqrt(betasq)

        # Long-period periodic offsets, zero for the current formulation
        self.peo = 0.0
        self.pinco = 0.0
        self.plo = 0.0
        self.pgho = 0.0
        self.pho = 0.0

        day = epoch + 18261.5 + tc / 1440.0
        xnodce = np.fmod(4.5236020 - 9.2422029e-4 * day, TWO_PI)
        stem = np.sin(xnodce)
        ctem = np.cos(xnodce)
        zcosil = 0.91375164 - 0.03568096 * ctem
        zsinil = np.sqrt(1.0 - zcosil * zcosil)
        zsinhl = 0.089683511 * stem / zsinil
        zcoshl = np.sqrt(1.0 - zsinhl * zsinhl)
        gam = 5.8351514 + 0.0019443680 * day
        zx = 0.39785416 * stem / zsinil
        zy = zcoshl * ctem + 0.91744867 * zsinhl * stem
        zx = np.arctan2(zx, zy)
        zx = gam + zx - xnodce
        zcosgl = np.cos(zx)
        zsingl = np.sin(zx)

        # Solar terms on the first pass, lunar terms on the second
        zcosg = ZCOSGS
        zsing = ZSINGS
        zcosi = ZCOSIS
        zsini = ZSINIS
        zcosh = cnodm
        zsinh = snodm
        cc = C1SS
        xnoi = 1.0 / nm

        ss1 = ss2 = ss3 = ss4 = ss5 = ss6 = ss7 = 0.0
        sz1 = sz2 = sz3 = 0.0
        sz11 = sz12 = sz13 = 0.0
        sz21 = sz22 = sz23 = 0.0
        sz31 = sz32 = sz33 = 0.0
        s1 = s2 = s3 = s4 = s5 = s6 = s7 = 0.0
        z1 = z2 = z3 = 0.0
        z11 = z12 = z13 = 0.0
        z21 = z22 = z23 = 0.0
        z31 = z32 = z33 = 0.0

        for lsflg in (1, 2):
            a1 = zcosg * zcosh + zsing * zcosi * zsinh
            a3 = -zsing * zcosh + zcosg * zcosi * zsinh
            a7 = -zcosg * zsinh + zsing * zcosi * zcosh
            a8 = zsing * zsini
            a9 = zsing * zsinh + zcosg * zcosi * zcosh
            a10 = zcosg * zsini
            a2 = cosim * a7 + sinim * a8
            a4 = cosim * a9 + sinim * a10
            a5 = -sinim * a7 + cosim * a8
            a6 = -sinim * a9 + cosim * a10

            x1 = a1 * cosomm + a2 * sinomm
            x2 = a3 * cosomm + a4 * sinomm
            x3 = -a1 * sinomm + a2 * cosomm
            x4 = -a3 * sinomm + a4 * cosomm
            x5 = a5 * sinomm
            x6 = a6 * sinomm
            x7 = a5 * cosomm
            x8 = a6 * cosomm

            z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3
            z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4
            z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4
            z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq
            z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq
            z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq
            z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
            z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * (
                -24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5)
            )
            z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
            z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7)
            z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * (
                24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8)
            )
            z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8)
            z1 = z1 + z1 + betasq * z31
            z2 = z2 + z2 + betasq * z32
            z3 = z3 + z3 + betasq * z33
            s3 = cc * xnoi
            s2 = -0.5 * s3 / rtemsq
            s4 = s3 * rtemsq
            s1 = -15.0 * em * s4
            s5 = x1 * x3 + x2 * x4
            s6 = x2 * x3 + x1 * x4
            s7 = x2 * x4 - x1 * x3

            if lsflg == 1:
                ss1, ss2, ss3, ss4, ss5, ss6, ss7 = s1, s2, s3, s4, s5, s6, s7
                sz1, sz2, sz3 = z1, z2, z3
                sz11, sz12, sz13 = z11, z12, z13
                sz21, sz22, sz23 = z21, z22, z23
                sz31, sz32, sz33 = z31, z32, z33
                zcosg = zcosgl
                zsing = zsingl
                zcosi = zcosil
                zsini = zsinil
                zcosh = zcoshl * cnodm + zsinhl * snodm
                zsinh = snodm * zcoshl - cnodm * zsinhl
                cc = C1L

        self.zmol = np.fmod(4.7199672 + 0.22997150 * day - gam, TWO_PI)
        self.zmos = np.fmod(6.2565837 + 0.017201977 * day, TWO_PI)

        # Solar periodic coefficients
        self.se2 = 2.0 * ss1 * ss6
        self.se3 = 2.0 * ss1 * ss7
        self.si2 = 2.0 * ss2 * sz12
        self.si3 = 2.0 * ss2 * (sz13 - sz11)
        self.sl2 = -2.0 * ss3 * sz2
        self.sl3 = -2.0 * ss3 * (sz3 - sz1)
        self.sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * ZES
        self.sgh2 = 2.0 * ss4 * sz32
        self.sgh3 = 2.0 * ss4 * (sz33 - sz31)
        self.sgh4 = -18.0 * ss4 * ZES
        self.sh2 = -2.0 * ss2 * sz22
        self.sh3 = -2.0 * ss2 * (sz23 - sz21)

        # Lunar periodic coefficients
        self.ee2 = 2.0 * s1 * s6
        self.e3 = 2.0 * s1 * s7
        self.xi2 = 2.0 * s2 * z12
        self.xi3 = 2.0 * s2 * (z13 - z11)
        self.xl2 = -2.0 * s3 * z2
        self.xl3 = -2.0 * s3 * (z3 - z1)
        self.xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * ZEL
        self.xgh2 = 2.0 * s4 * z32
        self.xgh3 = 2.0 * s4 * (z33 - z31)
        self.xgh4 = -18.0 * s4 * ZEL
        self.xh2 = -2.0 * s2 * z22
        self.xh3 = -2.0 * s2 * (z23 - z21)

        # Retained for DSINIT
        self._ds_sinim = sinim
        self._ds_cosim = cosim
        self._ds_emsq = emsq
        self._ds_s = (s1, s2, s3, s4, s5, s6, s7)
        self._ds_ss = (ss1, ss2, ss3, ss4, ss5, ss6, ss7)
        self._ds_z = (z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33)
        self._ds_sz = (
            sz1,
            sz2,
            sz3,
            sz11,
            sz12,
            sz13,
            sz21,
            sz22,
            sz23,
            sz31,
            sz32,
            sz33,
        )

    def _dsinit(self) -> None:
        """DSINIT -- lunar-solar secular rates and geopotential resonance setup."""
        sinim = self._ds_sinim
        cosim = self._ds_cosim
        emsq = self._ds_emsq
        s1, s2, s3, s4, s5, s6, s7 = self._ds_s
        ss1, ss2, ss3, ss4, ss5, ss6, ss7 = self._ds_ss
        (z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33) = self._ds_z
        (
            sz1,
            sz2,
            sz3,
            sz11,
            sz12,
            sz13,
            sz21,
            sz22,
            sz23,
            sz31,
            sz32,
            sz33,
        ) = self._ds_sz

        em = self.ecco
        eccsq = self.ecco * self.ecco
        inclm = self.inclo
        nm = self.no_kozai
        no = self.no_kozai
        tc = 0.0

        self.irez = 0
        if 0.0034906585 < nm < 0.0052359877:
            self.irez = 1
        if 8.26e-3 <= nm <= 9.24e-3 and em >= 0.5:
            self.irez = 2

        # Solar secular terms
        ses = ss1 * ZNS * ss5
        sis = ss2 * ZNS * (sz11 + sz13)
        sls = -ZNS * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
        sghs = ss4 * ZNS * (sz31 + sz33 - 6.0)
        shs = -ZNS * ss2 * (sz21 + sz23)
        if inclm < 5.2359877e-2 or inclm > np.pi - 5.2359877e-2:
            shs = 0.0
        if sinim != 0.0:
            shs = shs / sinim
        sgs = sghs - cosim * shs

        # Lunar secular terms, combined with the solar ones
        self.dedt = ses + s1 * ZNL * s5
        self.didt = sis + s2 * ZNL * (z11 + z13)
        self.dmdt = sls - ZNL * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
        sghl = s4 * ZNL * (z31 + z33 - 6.0)
        shll = -ZNL * s2 * (z21 + z23)
        if inclm < 5.2359877e-2 or inclm > np.pi - 5.2359877e-2:
            shll = 0.0
        self.domdt = sgs + sghl
        self.dnodt = shs
        if sinim != 0.0:
            self.domdt = self.domdt - cosim / sinim * shll
            self.dnodt = self.dnodt + shll / sinim

        # Resonance coefficients
        self.d2201 = self.d2211 = 0.0
        self.d3210 = self.d3222 = 0.0
        self.d4410 = self.d4422 = 0.0
        self.d5220 = self.d5232 = self.d5421 = self.d5433 = 0.0
        self.del1 = self.del2 = self.del3 = 0.0
        self.xfact = 0.0
        self.xlamo = 0.0
        self.xli = 0.0
        self.xni = 0.0
        self.atime = 0.0

        theta = np.fmod(self.gsto + tc * RPTIM, TWO_PI)

        if self.irez != 0:
            aonv = (nm / KE) ** TWO_THIRDS

            # 12-hour (2:1) geopotential resonance
            if self.irez == 2:
                cosisq = cosim * cosim
                em = self.ecco
                emsq = eccsq
                eoc = em * emsq
                g201 = -0.306 - (em - 0.64) * 0.440

                if em <= 0.65:
                    g211 = 3.616 - 13.2470 * em + 16.2900 * emsq
                    g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc
                    g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc
                    g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc
                    g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.435 * eoc
                    g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.276 * eoc
                else:
                    g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc
                    g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc
                    g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc
                    g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc
                    g422 = (
                        -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc
                    )
                    if em > 0.715:
                        g520 = (
                            -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc
                        )
                    else:
                        g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq

                if em < 0.7:
                    g533 = (
                        -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc
                    )
                    g521 = (
                        -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc
                    )
                    g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc
                else:
                    g533 = (
                        -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc
                    )
                    g521 = (
                        -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc
                    )
                    g532 = (
                        -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc
                    )

                sini2 = sinim * sinim
                f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
                f221 = 1.5 * sini2
                f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
                f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
                f441 = 35.0 * sini2 * f220
                f442 = 39.3750 * sini2 * sini2
                f522 = (
                    9.84375
                    * sinim
                    * (
                        sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                        + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq)
                    )
                )
                f523 = sinim * (
                    4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                    + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq)
                )
                f542 = (
                    29.53125
                    * sinim
                    * (
                        2.0
                        - 8.0 * cosim
                        + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq)
                    )
                )
                f543 = (
                    29.53125
                    * sinim
                    * (
                        -2.0
                        - 8.0 * cosim
                        + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq)
                    )
                )

                xno2 = nm * nm
                ainv2 = aonv * aonv
                temp1 = 3.0 * xno2 * ainv2
                temp = temp1 * ROOT22
                self.d2201 = temp * f220 * g201
                self.d2211 = temp * f221 * g211
                temp1 = temp1 * aonv
                temp = temp1 * ROOT32
                self.d3210 = temp * f321 * g310
                self.d3222 = temp * f322 * g322
                temp1 = temp1 * aonv
                temp = 2.0 * temp1 * ROOT44
                self.d4410 = temp * f441 * g410
                self.d4422 = temp * f442 * g422
                temp1 = temp1 * aonv
                temp = temp1 * ROOT52
                self.d5220 = temp * f522 * g520
                self.d5232 = temp * f523 * g532
                temp = 2.0 * temp1 * ROOT54
                self.d5421 = temp * f542 * g521
                self.d5433 = temp * f543 * g533
                self.xlamo = np.fmod(
                    self.mo + self.nodeo + self.nodeo - theta - theta, TWO_PI
                )
                self.xfact = (
                    self.mdot
                    + self.dmdt
                    + 2.0 * (self.nodedot + self.dnodt - RPTIM)
                    - no
                )

            # 24-hour (1:1) synchronous resonance
            if self.irez == 1:
                g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
                g310 = 1.0 + 2.0 * emsq
                g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
                f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
                f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (
                    1.0 + cosim
                )
                f330 = 1.0 + cosim
                f330 = 1.875 * f330 * f330 * f330
                self.del1 = 3.0 * nm * nm * aonv * aonv
                self.del2 = 2.0 * self.del1 * f220 * g200 * Q22
                self.del3 = 3.0 * self.del1 * f330 * g300 * Q33 * aonv
                self.del1 = self.del1 * f311 * g310 * Q31 * aonv
                self.xlamo = np.fmod(self.mo + self.nodeo + self.argpo - theta, TWO_PI)
                xpidot = self.argpdot + self.nodedot
                self.xfact = (
                    self.mdot
                    + xpidot
                    - RPTIM
                    + self.dmdt
                    + self.domdt
                    + self.dnodt
                    - no
                )

            # Seed the resonance integrator
            self.xli = self.xlamo
            self.xni = no
            self.atime = 0.0

    def _dspace(
        self,
        t: float,
        nm: float,
        em: float,
        inclm: float,
        argpm: float,
        mm: float,
        nodem: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """DSPACE -- deep-space secular effects, including resonance integration.

        Parameters
        ----------
        t : float
            Time since epoch (minutes).
        nm, em, inclm, argpm, mm, nodem : float
            Mean motion (rad/min), eccentricity, inclination, argument of
            perigee, mean anomaly and right ascension before the deep-space
            secular update.

        Returns
        -------
        tuple of float
            Updated ``(nm, em, inclm, argpm, mm, nodem)``.
        """
        tc = t
        theta = np.fmod(self.gsto + tc * RPTIM, TWO_PI)

        em = em + self.dedt * t
        inclm = inclm + self.didt * t
        argpm = argpm + self.domdt * t
        nodem = nodem + self.dnodt * t
        mm = mm + self.dmdt * t

        if self.irez == 0:
            return nm, em, inclm, argpm, mm, nodem

        no = self.no_kozai

        # Restart the integration if the requested time is not reachable by
        # continuing from the stored integrator state.
        if self.atime == 0.0 or t * self.atime <= 0.0 or abs(t) < abs(self.atime):
            self.atime = 0.0
            self.xni = no
            self.xli = self.xlamo

        delt = STEPP if t > 0.0 else STEPN

        # Euler-Maclaurin integration of the resonance equations
        ft = 0.0
        xndot = 0.0
        xldot = 0.0
        xnddt = 0.0
        stepping = True
        while stepping:
            if self.irez != 2:
                xndot = (
                    self.del1 * np.sin(self.xli - FASX2)
                    + self.del2 * np.sin(2.0 * (self.xli - FASX4))
                    + self.del3 * np.sin(3.0 * (self.xli - FASX6))
                )
                xldot = self.xni + self.xfact
                xnddt = (
                    self.del1 * np.cos(self.xli - FASX2)
                    + 2.0 * self.del2 * np.cos(2.0 * (self.xli - FASX4))
                    + 3.0 * self.del3 * np.cos(3.0 * (self.xli - FASX6))
                )
                xnddt = xnddt * xldot
            else:
                xomi = self.argpo + self.argpdot * self.atime
                x2omi = xomi + xomi
                x2li = self.xli + self.xli
                xndot = (
                    self.d2201 * np.sin(x2omi + self.xli - G22)
                    + self.d2211 * np.sin(self.xli - G22)
                    + self.d3210 * np.sin(xomi + self.xli - G32)
                    + self.d3222 * np.sin(-xomi + self.xli - G32)
                    + self.d4410 * np.sin(x2omi + x2li - G44)
                    + self.d4422 * np.sin(x2li - G44)
                    + self.d5220 * np.sin(xomi + self.xli - G52)
                    + self.d5232 * np.sin(-xomi + self.xli - G52)
                    + self.d5421 * np.sin(xomi + x2li - G54)
                    + self.d5433 * np.sin(-xomi + x2li - G54)
                )
                xldot = self.xni + self.xfact
                xnddt = (
                    self.d2201 * np.cos(x2omi + self.xli - G22)
                    + self.d2211 * np.cos(self.xli - G22)
                    + self.d3210 * np.cos(xomi + self.xli - G32)
                    + self.d3222 * np.cos(-xomi + self.xli - G32)
                    + self.d5220 * np.cos(xomi + self.xli - G52)
                    + self.d5232 * np.cos(-xomi + self.xli - G52)
                    + 2.0
                    * (
                        self.d4410 * np.cos(x2omi + x2li - G44)
                        + self.d4422 * np.cos(x2li - G44)
                        + self.d5421 * np.cos(xomi + x2li - G54)
                        + self.d5433 * np.cos(-xomi + x2li - G54)
                    )
                )
                xnddt = xnddt * xldot

            if abs(t - self.atime) >= STEPP:
                self.xli = self.xli + xldot * delt + xndot * STEP2
                self.xni = self.xni + xndot * delt + xnddt * STEP2
                self.atime = self.atime + delt
            else:
                ft = t - self.atime
                stepping = False

        nm = self.xni + xndot * ft + xnddt * ft * ft * 0.5
        xl = self.xli + xldot * ft + xndot * ft * ft * 0.5
        if self.irez != 1:
            mm = xl - 2.0 * nodem + 2.0 * theta
        else:
            mm = xl - nodem - argpm + theta
        dndt = nm - no
        nm = no + dndt

        return nm, em, inclm, argpm, mm, nodem

    def _dpper(
        self,
        t: float,
        ep: float,
        inclp: float,
        nodep: float,
        argpp: float,
        mp: float,
        init: bool = False,
    ) -> Tuple[float, float, float, float, float]:
        """DPPER -- lunar-solar periodic contributions to the osculating elements.

        Parameters
        ----------
        t : float
            Time since epoch (minutes).
        ep, inclp, nodep, argpp, mp : float
            Eccentricity, inclination, right ascension, argument of perigee
            and mean anomaly before the periodic update.
        init : bool, optional
            True for the single call made during initialization, where the
            periodics are evaluated at epoch and not applied.

        Returns
        -------
        tuple of float
            Updated ``(ep, inclp, nodep, argpp, mp)``.
        """
        zm = self.zmos if init else self.zmos + ZNS * t
        zf = zm + 2.0 * ZES * np.sin(zm)
        sinzf = np.sin(zf)
        f2 = 0.5 * sinzf * sinzf - 0.25
        f3 = -0.5 * sinzf * np.cos(zf)
        ses = self.se2 * f2 + self.se3 * f3
        sis = self.si2 * f2 + self.si3 * f3
        sls = self.sl2 * f2 + self.sl3 * f3 + self.sl4 * sinzf
        sghs = self.sgh2 * f2 + self.sgh3 * f3 + self.sgh4 * sinzf
        shs = self.sh2 * f2 + self.sh3 * f3

        zm = self.zmol if init else self.zmol + ZNL * t
        zf = zm + 2.0 * ZEL * np.sin(zm)
        sinzf = np.sin(zf)
        f2 = 0.5 * sinzf * sinzf - 0.25
        f3 = -0.5 * sinzf * np.cos(zf)
        sel = self.ee2 * f2 + self.e3 * f3
        sil = self.xi2 * f2 + self.xi3 * f3
        sll = self.xl2 * f2 + self.xl3 * f3 + self.xl4 * sinzf
        sghl = self.xgh2 * f2 + self.xgh3 * f3 + self.xgh4 * sinzf
        shll = self.xh2 * f2 + self.xh3 * f3

        pe = ses + sel
        pinc = sis + sil
        pl = sls + sll
        pgh = sghs + sghl
        ph = shs + shll

        if init:
            return ep, inclp, nodep, argpp, mp

        pe = pe - self.peo
        pinc = pinc - self.pinco
        pl = pl - self.plo
        pgh = pgh - self.pgho
        ph = ph - self.pho

        inclp = inclp + pinc
        ep = ep + pe
        sinip = np.sin(inclp)
        cosip = np.cos(inclp)

        if inclp >= 0.2:
            ph = ph / sinip
            pgh = pgh - cosip * ph
            argpp = argpp + pgh
            nodep = nodep + ph
            mp = mp + pl
        else:
            # Lyddane modification for near-equatorial orbits
            sinop = np.sin(nodep)
            cosop = np.cos(nodep)
            alfdp = sinip * sinop
            betdp = sinip * cosop
            dalf = ph * cosop + pinc * cosip * sinop
            dbet = -ph * sinop + pinc * cosip * cosop
            alfdp = alfdp + dalf
            betdp = betdp + dbet
            nodep = np.fmod(nodep, TWO_PI)
            xls = mp + argpp + cosip * nodep
            dls = pl + pgh - pinc * nodep * sinip
            xls = xls + dls
            xnoh = nodep
            nodep = np.arctan2(alfdp, betdp)
            if abs(xnoh - nodep) > np.pi:
                if nodep < xnoh:
                    nodep = nodep + TWO_PI
                else:
                    nodep = nodep - TWO_PI
            mp = mp + pl
            argpp = xls - mp - cosip * nodep

        return ep, inclp, nodep, argpp, mp

    def propagate(self, tsince: float) -> SGP4State:
        """Propagate satellite to specified time.

        Parameters
        ----------
        tsince : float
            Time since epoch (minutes). Positive = after epoch.

        Returns
        -------
        state : SGP4State
            Position and velocity in TEME frame.

        Examples
        --------
        >>> from pytcl.astronomical.tle import parse_tle
        >>> tle = parse_tle(
        ...     "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
        ...     "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003")
        >>> sat = SGP4Satellite(tle)
        >>> state = sat.propagate(0.0)  # At TLE epoch
        >>> state = sat.propagate(60.0)  # 60 minutes later
        >>> state = sat.propagate(-30.0)  # 30 minutes before epoch
        """
        if self.is_deep_space:
            return self._propagate_sdp4(tsince)
        return self._propagate_sgp4(tsince)

    def _propagate_sgp4(self, tsince: float) -> SGP4State:
        """SGP4 propagation (near-Earth satellites)."""
        return self._propagate_core(tsince)

    def _propagate_sdp4(self, tsince: float) -> SGP4State:
        """SDP4 propagation (deep-space satellites).

        Applies the lunar-solar secular and periodic contributions and the
        12-hour/24-hour resonance integration around the shared core.
        """
        return self._propagate_core(tsince)

    def _propagate_core(self, tsince: float) -> SGP4State:
        """Shared SGP4 propagation core (Vallado's reference algorithm)."""
        # Secular effects of atmospheric drag and gravitational perturbations
        xmdf = self.mo + self.mdot * tsince
        argpdf = self.argpo + self.argpdot * tsince
        xnoddf = self.nodeo + self.nodedot * tsince

        tsq = tsince * tsince
        xnode = xnoddf + self.xnodcf * tsq
        tempa = 1.0 - self.c1 * tsince
        tempe = self.bstar * self.c4 * tsince
        templ = self.t2cof * tsq

        argpm = argpdf
        mm = xmdf

        # Higher-order drag effects (skipped for low perigee / deep space)
        if not self.isimp:
            delomg = self.omgcof * tsince
            delmtemp = 1.0 + self.eta * np.cos(xmdf)
            delm = self.xmcof * (delmtemp**3 - self.delmo)
            temp = delomg + delm
            mm = xmdf + temp
            argpm = argpdf - temp
            tcube = tsq * tsince
            tfour = tcube * tsince
            tempa = tempa - self.d2 * tsq - self.d3 * tcube - self.d4 * tfour
            tempe = tempe + self.bstar * self.c5 * (np.sin(mm) - self.sinmao)
            templ = (
                templ + self.t3cof * tcube + tfour * (self.t4cof + tsince * self.t5cof)
            )

        # Deep-space secular effects (lunar-solar gravity and resonance)
        if self.is_deep_space:
            nm_ds, em_ds, inclm, argpm, mm, xnode = self._dspace(
                tsince,
                nm=self.no_kozai,
                em=self.ecco,
                inclm=self.inclo,
                argpm=argpm,
                mm=mm,
                nodem=xnode,
            )
            if nm_ds <= 0.0:
                return _failed_state(2)
            a = (KE / nm_ds) ** TWO_THIRDS * tempa * tempa
            nm = KE / a**1.5
            e = em_ds - tempe
        else:
            inclm = self.inclo
            a = self.ao * tempa * tempa
            nm = KE / a**1.5  # Mean motion for current (drag-decayed) orbit
            e = self.ecco - tempe

        if e >= 1.0 or e < -0.001:
            return _failed_state(1)
        # Limit eccentricity
        if e < 1.0e-6:
            e = 1.0e-6

        mm = mm + self.no_kozai * templ

        ep = e
        inclp = inclm
        nodep = xnode
        argpp = argpm
        mp = mm

        if self.is_deep_space:
            # Normalize before the periodic update, as in the reference
            xlm = np.fmod(mm + argpm + xnode, TWO_PI)
            nodep = np.fmod(xnode, TWO_PI)
            argpp = np.fmod(argpm, TWO_PI)
            mp = np.fmod(xlm - argpp - nodep, TWO_PI)

            ep, inclp, nodep, argpp, mp = self._dpper(
                tsince, ep, inclp, nodep, argpp, mp
            )
            if inclp < 0.0:
                inclp = -inclp
                nodep = nodep + np.pi
                argpp = argpp - np.pi
            if ep < 0.0 or ep > 1.0:
                return _failed_state(3)

        # Inclination-dependent coefficients; for deep space the inclination
        # has been perturbed, so they must be recomputed.
        if self.is_deep_space:
            sinip = np.sin(inclp)
            cosip = np.cos(inclp)
            cosisq = cosip * cosip
            x3thm1 = 3.0 * cosisq - 1.0
            x1mth2 = 1.0 - cosisq
            x7thm1 = 7.0 * cosisq - 1.0
            aycof = 0.25 * A30_OVER_K2 * sinip
            if abs(1.0 + cosip) > 1.5e-12:
                xlcof = (
                    0.125 * A30_OVER_K2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip)
                )
            else:
                xlcof = 0.125 * A30_OVER_K2 * sinip * (3.0 + 5.0 * cosip) / 1.5e-12
        else:
            sinip = self.sinio
            cosip = self.cosio
            x3thm1 = self.x3thm1
            x1mth2 = self.x1mth2
            x7thm1 = self.x7thm1
            aycof = self.aycof
            xlcof = self.xlcof

        # Long-period periodics
        axnl = ep * np.cos(argpp)
        temp = 1.0 / (a * (1.0 - ep * ep))
        aynl = ep * np.sin(argpp) + temp * aycof
        xlt = mp + argpp + nodep + temp * xlcof * axnl

        # Solve Kepler's equation
        u = np.fmod(xlt - nodep, TWO_PI)
        eo1 = u
        for _ in range(10):
            sineo1 = np.sin(eo1)
            coseo1 = np.cos(eo1)
            fp = 1.0 - coseo1 * axnl - sineo1 * aynl
            delta = (u - aynl * coseo1 + axnl * sineo1 - eo1) / fp
            # Limit Newton step for robustness (as in reference code)
            if abs(delta) >= 0.95:
                delta = 0.95 if delta > 0.0 else -0.95
            eo1 = eo1 + delta
            if abs(delta) < 1.0e-12:
                break

        # Short-period preliminary quantities
        ecose = axnl * coseo1 + aynl * sineo1
        esine = axnl * sineo1 - aynl * coseo1
        elsq = axnl * axnl + aynl * aynl
        temp = 1.0 - elsq
        pl = a * temp
        if pl < 0.0:
            return _failed_state(4)
        r = a * (1.0 - ecose)
        rdot = KE * np.sqrt(a) * esine / r
        rvdot = KE * np.sqrt(pl) / r

        betal = np.sqrt(temp)
        sinu = a / r * (sineo1 - aynl - axnl * esine / (1.0 + betal))
        cosu = a / r * (coseo1 - axnl + aynl * esine / (1.0 + betal))
        u = np.arctan2(sinu, cosu)

        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0
        temp = 1.0 / pl
        # J2 short-period coefficient: 0.5 * J2 / p (note K2 = J2 / 2)
        temp1 = K2 * temp
        temp2 = temp1 * temp

        # Update for short-period periodics
        rk = r * (1.0 - 1.5 * temp2 * betal * x3thm1) + 0.5 * temp1 * x1mth2 * cos2u
        uk = u - 0.25 * temp2 * x7thm1 * sin2u
        xnodek = nodep + 1.5 * temp2 * cosip * sin2u
        xinck = inclp + 1.5 * temp2 * cosip * sinip * cos2u
        rdotk = rdot - nm * temp1 * x1mth2 * sin2u
        rvdotk = rvdot + nm * temp1 * (x1mth2 * cos2u + 1.5 * x3thm1)

        # Orientation vectors
        sinuk = np.sin(uk)
        cosuk = np.cos(uk)
        sinik = np.sin(xinck)
        cosik = np.cos(xinck)
        sinnok = np.sin(xnodek)
        cosnok = np.cos(xnodek)

        xmx = -sinnok * cosik
        xmy = cosnok * cosik

        ux = xmx * sinuk + cosnok * cosuk
        uy = xmy * sinuk + sinnok * cosuk
        uz = sinik * sinuk

        vx = xmx * cosuk - cosnok * sinuk
        vy = xmy * cosuk - sinnok * sinuk
        vz = sinik * cosuk

        # Position and velocity in TEME
        # Position: rk is in Earth radii, multiply by RADIUS_EARTH for km
        # Velocity: rdotk/rvdotk are in ER/min, convert to km/s
        r_teme = rk * np.array([ux, uy, uz]) * RADIUS_EARTH
        v_teme = (
            (rdotk * np.array([ux, uy, uz]) + rvdotk * np.array([vx, vy, vz]))
            * RADIUS_EARTH
            / 60.0
        )

        # Satellite has decayed below the Earth's surface
        error = 6 if rk < 1.0 else 0

        return SGP4State(r=r_teme, v=v_teme, error=error)

    def propagate_jd(self, jd: float) -> SGP4State:
        """Propagate satellite to specified Julian date.

        Parameters
        ----------
        jd : float
            Julian date.

        Returns
        -------
        state : SGP4State
            Position and velocity in TEME frame.
        """
        tsince = (jd - self.epoch_jd) * MINUTES_PER_DAY
        return self.propagate(tsince)


def sgp4_propagate(tle: TLE, tsince: float) -> SGP4State:
    """Propagate TLE using SGP4/SDP4 model.

    Convenience function that creates an SGP4Satellite and propagates.

    Parameters
    ----------
    tle : TLE
        Two-Line Element set.
    tsince : float
        Time since epoch (minutes).

    Returns
    -------
    state : SGP4State
        Position and velocity in TEME frame.

    Examples
    --------
    >>> from pytcl.astronomical.tle import parse_tle
    >>> tle = parse_tle(
    ...     "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
    ...     "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003")
    >>> state = sgp4_propagate(tle, 60.0)  # 60 minutes after epoch
    >>> bool(6700 < np.linalg.norm(state.r) < 6900)  # ISS orbital radius (km)
    True
    """
    sat = SGP4Satellite(tle)
    return sat.propagate(tsince)


def sgp4_propagate_batch(
    tle: TLE,
    times: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Propagate TLE to multiple times.

    Parameters
    ----------
    tle : TLE
        Two-Line Element set.
    times : ndarray
        Times since epoch (minutes), shape (n,).

    Returns
    -------
    positions : ndarray
        Positions in TEME frame (km), shape (n, 3).
    velocities : ndarray
        Velocities in TEME frame (km/s), shape (n, 3).

    Examples
    --------
    >>> from pytcl.astronomical.tle import parse_tle
    >>> tle = parse_tle(
    ...     "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
    ...     "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003")
    >>> times = np.linspace(0, 90, 100)  # 0 to 90 minutes
    >>> r, v = sgp4_propagate_batch(tle, times)
    >>> r.shape
    (100, 3)
    """
    sat = SGP4Satellite(tle)
    n = len(times)

    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))

    for i, t in enumerate(times):
        state = sat.propagate(t)
        positions[i] = state.r
        velocities[i] = state.v

    return positions, velocities


__all__ = [
    # Constants
    "MU_EARTH",
    "RADIUS_EARTH",
    "J2",
    "J3",
    "J4",
    # Types
    "SGP4State",
    "SGP4Satellite",
    # Functions
    "sgp4_propagate",
    "sgp4_propagate_batch",
    "unkozai_mean_motion",
]
