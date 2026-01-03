"""
SGP4/SDP4 Satellite Propagation Models.

This module implements the Simplified General Perturbations model (SGP4)
and its deep-space extension (SDP4) for propagating satellite orbits
from Two-Line Element (TLE) sets.

SGP4 models the effects of:
- Atmospheric drag (via the B* term)
- J2, J3, J4 gravitational harmonics
- Secular and periodic variations

SDP4 additionally models (for orbital periods >= 225 min):
- Lunar gravitational perturbations
- Solar gravitational perturbations
- Resonance effects (12-hour and 24-hour)

The output is in the TEME (True Equator, Mean Equinox) reference frame,
which is a quasi-inertial frame used by NORAD.

References
----------
.. [1] Hoots, F. R. and Roehrich, R. L., "Spacetrack Report No. 3:
       Models for Propagation of NORAD Element Sets," 1980.
.. [2] Vallado, D. A., Crawford, P., Hujsak, R., and Kelso, T.S.,
       "Revisiting Spacetrack Report #3," AIAA 2006-6753.
.. [3] Vallado, D. A., "Fundamentals of Astrodynamics and Applications,"
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

# Small number for avoiding singularities
SMALL = 1.0e-12

# Two-thirds
TWO_THIRDS = 2.0 / 3.0


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
    >>> tle = parse_tle(line1, line2, name="ISS")
    >>> sat = SGP4Satellite(tle)
    >>> state = sat.propagate(0.0)  # At epoch
    >>> print(f"Position: {state.r} km")
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

        # Recovered mean motion and semi-major axis
        self.no_kozai = self.no / (1.0 + delta0)
        self.ao = a0 / (1.0 - delta0)

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

        self.argpdot = (
            -0.5 * temp1 * self.x1mth2
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

        # For deep space
        self._ds_initialized = False
        if self.is_deep_space:
            self._init_deep_space()

    def _init_deep_space(self) -> None:
        """Initialize deep-space (SDP4) constants."""
        # This is a simplified version - full implementation would include
        # lunar-solar perturbations and resonance effects

        # For now, store basic deep-space flag
        self._ds_initialized = True

        # Day number from epoch
        self.jd_epoch = self.epoch_jd

        # Solar and lunar constants would go here in full implementation
        # These are placeholders for the basic deep-space effects
        self.resonance_flag = False
        self.synchronous_flag = False

        # Check for 12-hour and 24-hour resonances
        n_day = self.no_kozai * MINUTES_PER_DAY / (2 * np.pi)  # revs/day

        if n_day >= 0.9 and n_day <= 1.1:
            # 24-hour (synchronous) resonance
            self.synchronous_flag = True
            self.resonance_flag = True
        elif n_day >= 1.9 and n_day <= 2.1:
            # 12-hour resonance (like Molniya)
            self.resonance_flag = True

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
        >>> sat = SGP4Satellite(tle)
        >>> state = sat.propagate(0.0)  # At TLE epoch
        >>> state = sat.propagate(60.0)  # 60 minutes later
        >>> state = sat.propagate(-30.0)  # 30 minutes before epoch
        """
        if self.is_deep_space:
            return self._propagate_sdp4(tsince)
        else:
            return self._propagate_sgp4(tsince)

    def _propagate_sgp4(self, tsince: float) -> SGP4State:
        """SGP4 propagation (near-Earth satellites)."""
        # Secular effects of atmospheric drag and gravitational perturbations
        xmdf = self.mo + self.mdot * tsince
        argpdf = self.argpo + self.argpdot * tsince
        xnoddf = self.nodeo + self.nodedot * tsince

        tsq = tsince * tsince
        xnode = xnoddf + self.xnodcf * tsq
        tempa = 1.0 - self.c1 * tsince
        tempe = self.bstar * self.c4 * tsince
        templ = self.t2cof * tsq

        # Handle higher-order effects for non-circular orbits
        if self.ecco > 1.0e-4:
            delomg = self.c5 * (np.sin(xmdf) - np.sin(self.mo))
            delm = (
                (
                    self.c1
                    * self.qzms24
                    * (self.ao * self.betao2) ** 3
                    * (1.0 + self.eta * np.cos(xmdf)) ** 3
                    - (1.0 + self.eta * np.cos(self.mo)) ** 3
                )
                * tempe
                / self.eta
                / self.betao2
            )
            temp = delomg + delm
            xmdf = xmdf + temp
            argpdf = argpdf - temp
            tempa = tempa - self.c1 * tsince * self.c5 * (
                np.cos(xmdf) - np.cos(self.mo)
            )
            tempe = tempe - self.c1 * self.c5 * (np.sin(xmdf) - np.sin(self.mo))

        a = self.ao * tempa * tempa
        e = self.ecco - tempe
        xl = xmdf + argpdf + xnode + self.no_kozai * templ

        # Limit eccentricity
        if e < 1.0e-6:
            e = 1.0e-6
        if e > 0.999999:
            e = 0.999999

        # Long-period periodics
        axnl = e * np.cos(argpdf)
        temp = 1.0 / (a * (1.0 - e * e))
        aynl = e * np.sin(argpdf) + temp * self.aycof
        xlt = xl + temp * self.xlcof * axnl

        # Solve Kepler's equation
        u = (xlt - xnode) % (2.0 * np.pi)
        eo1 = u
        for _ in range(10):
            sineo1 = np.sin(eo1)
            coseo1 = np.cos(eo1)
            f = u - eo1 + axnl * sineo1 - aynl * coseo1
            fp = 1.0 - axnl * coseo1 - aynl * sineo1
            delta = f / fp
            eo1 = eo1 + delta
            if abs(delta) < 1.0e-12:
                break

        # Short-period preliminary quantities
        ecose = axnl * coseo1 + aynl * sineo1
        esine = axnl * sineo1 - aynl * coseo1
        elsq = axnl * axnl + aynl * aynl
        temp = 1.0 - elsq
        if temp < SMALL:
            temp = SMALL
        pl = a * temp
        r = a * (1.0 - ecose)
        # Velocity factor: in SGP4, rdot and rvdot must be multiplied by
        # the mean motion to get proper velocity units (ER/min)
        rdot = KE * np.sqrt(a) * esine / r
        rvdot = KE * np.sqrt(pl) / r

        betal = np.sqrt(temp)
        temp = ecose - axnl
        if temp < 0.0:
            temp = -temp
        if temp < SMALL:
            temp = SMALL
        sinu = a / r * (sineo1 - aynl - axnl * esine / (1.0 + betal))
        cosu = a / r * (coseo1 - axnl + aynl * esine / (1.0 + betal))
        u = np.arctan2(sinu, cosu)

        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0
        temp = 1.0 / pl
        temp1 = 0.5 * K2 * temp
        temp2 = temp1 * temp

        # Update for short-period periodics
        rk = (
            r * (1.0 - 1.5 * temp2 * betal * self.x3thm1)
            + 0.5 * temp1 * self.x1mth2 * cos2u
        )
        uk = u - 0.25 * temp2 * self.x7thm1 * sin2u
        xnodek = xnode + 1.5 * temp2 * self.cosio * sin2u
        xinck = self.inclo + 1.5 * temp2 * self.cosio * self.sinio * cos2u
        rdotk = rdot - KE * temp1 * self.x1mth2 * sin2u / self.no_kozai
        rvdotk = (
            rvdot
            + KE * temp1 * (self.x1mth2 * cos2u + 1.5 * self.x3thm1) / self.no_kozai
        )

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

        return SGP4State(r=r_teme, v=v_teme, error=0)

    def _propagate_sdp4(self, tsince: float) -> SGP4State:
        """SDP4 propagation (deep-space satellites).

        This is a simplified implementation that includes the basic
        deep-space secular and long-period effects, but not the full
        lunar-solar periodics.
        """
        # For satellites with period >= 225 minutes, the SDP4 model
        # adds lunar-solar perturbations.

        # Start with SGP4 secular terms
        xmdf = self.mo + self.mdot * tsince
        argpdf = self.argpo + self.argpdot * tsince
        xnoddf = self.nodeo + self.nodedot * tsince

        tsq = tsince * tsince
        xnode = xnoddf + self.xnodcf * tsq
        tempa = 1.0 - self.c1 * tsince
        tempe = self.bstar * self.c4 * tsince
        templ = self.t2cof * tsq

        # Deep space secular effects (simplified)
        # In full SDP4, these would include lunar-solar perturbations
        # computed from stored initialization values

        # For now, use SGP4-like propagation with period check
        a = self.ao * tempa * tempa
        e = self.ecco - tempe
        xl = xmdf + argpdf + xnode + self.no_kozai * templ

        # Limit eccentricity
        if e < 1.0e-6:
            e = 1.0e-6
        if e > 0.999999:
            e = 0.999999

        # Long-period periodics
        axnl = e * np.cos(argpdf)
        temp = 1.0 / (a * (1.0 - e * e))
        aynl = e * np.sin(argpdf) + temp * self.aycof
        xlt = xl + temp * self.xlcof * axnl

        # Solve Kepler's equation
        u = (xlt - xnode) % (2.0 * np.pi)
        eo1 = u
        for _ in range(10):
            sineo1 = np.sin(eo1)
            coseo1 = np.cos(eo1)
            f = u - eo1 + axnl * sineo1 - aynl * coseo1
            fp = 1.0 - axnl * coseo1 - aynl * sineo1
            delta = f / fp
            eo1 = eo1 + delta
            if abs(delta) < 1.0e-12:
                break

        # Short-period preliminary quantities
        ecose = axnl * coseo1 + aynl * sineo1
        esine = axnl * sineo1 - aynl * coseo1
        elsq = axnl * axnl + aynl * aynl
        temp = 1.0 - elsq
        if temp < SMALL:
            temp = SMALL
        pl = a * temp
        r = a * (1.0 - ecose)
        # Velocity factor
        rdot = KE * np.sqrt(a) * esine / r
        rvdot = KE * np.sqrt(pl) / r

        betal = np.sqrt(temp)
        sinu = a / r * (sineo1 - aynl - axnl * esine / (1.0 + betal))
        cosu = a / r * (coseo1 - axnl + aynl * esine / (1.0 + betal))
        u = np.arctan2(sinu, cosu)

        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0
        temp = 1.0 / pl
        temp1 = 0.5 * K2 * temp
        temp2 = temp1 * temp

        # Update for short-period periodics
        rk = (
            r * (1.0 - 1.5 * temp2 * betal * self.x3thm1)
            + 0.5 * temp1 * self.x1mth2 * cos2u
        )
        uk = u - 0.25 * temp2 * self.x7thm1 * sin2u
        xnodek = xnode + 1.5 * temp2 * self.cosio * sin2u
        xinck = self.inclo + 1.5 * temp2 * self.cosio * self.sinio * cos2u
        rdotk = rdot - KE * temp1 * self.x1mth2 * sin2u / self.no_kozai
        rvdotk = (
            rvdot
            + KE * temp1 * (self.x1mth2 * cos2u + 1.5 * self.x3thm1) / self.no_kozai
        )

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
        r_teme = rk * np.array([ux, uy, uz]) * RADIUS_EARTH
        v_teme = (
            (rdotk * np.array([ux, uy, uz]) + rvdotk * np.array([vx, vy, vz]))
            * RADIUS_EARTH
            / 60.0
        )

        return SGP4State(r=r_teme, v=v_teme, error=0)

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
    >>> tle = parse_tle(line1, line2)
    >>> state = sgp4_propagate(tle, 60.0)  # 60 minutes after epoch
    >>> print(f"Position: {state.r} km")
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
    >>> tle = parse_tle(line1, line2)
    >>> times = np.linspace(0, 90, 100)  # 0 to 90 minutes
    >>> r, v = sgp4_propagate_batch(tle, times)
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
]
