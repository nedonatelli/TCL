"""
The NRLMSISE-00 empirical atmosphere model.

A faithful transcription of the public-domain C reference
implementation as vendored (and lightly modified) by the MATLAB TCL
(``3rd_Party_Libraries/nrlmsise-00-bc9a2fe``), which is the validation
oracle: every code path here is checked against outputs of that C
code compiled directly. The NRL modifications preserved here are the
``ghp7`` altitude return and the MEX-era diagnostic hooks (which
become warnings).

The model [1]_ computes number densities of He, O, N2, O2, Ar, H, N
and anomalous O, the total mass density, and the exospheric and local
temperatures, for altitudes from the ground into the exosphere.

References
----------
.. [1] J. M. Picone, A. E. Hedin, D. P. Drob, and A. C. Aikin,
   "NRLMSISE-00 empirical model of the atmosphere: Statistical
   comparisons and scientific issues," Journal of Geophysical
   Research: Space Physics, vol. 107, no. A12, Dec. 2002.
"""

import warnings
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

try:  # The compiled reference implementation; built by setup.py.
    from pytcl.atmosphere import _nrlmsise00_c as _c_ext
except ImportError:  # pragma: no cover - exercised only on failed builds
    _c_ext = None
from numpy.typing import NDArray

from pytcl.atmosphere._nrlmsise00_data import (
    PAVGM,
    PD,
    PDL,
    PDM,
    PMA,
    PS,
    PT,
    PTL,
    PTM,
)


class NRLMSISEOutput(NamedTuple):
    """Result of the NRLMSISE-00 model.

    Attributes
    ----------
    d : ndarray
        The nine densities, indexed as the reference implementation
        orders them: d[0] He, d[1] O, d[2] N2, d[3] O2, d[4] Ar,
        d[5] total mass density, d[6] H, d[7] N, d[8] anomalous O.
        Number densities are per cubic meter and d[5] is kg/m^3 (the
        SI switch of the reference is always on here, matching the
        MATLAB wrapper).
    t : ndarray
        t[0] the exospheric temperature and t[1] the temperature at
        altitude, in Kelvin.
    """

    d: NDArray[np.floating]
    t: NDArray[np.floating]


class _Flags:
    """The 24 model switches and their derived sw/swc forms (tselec)."""

    def __init__(self, switches: Sequence[float]):
        self.switches = list(switches)
        self.sw = [0.0] * 24
        self.swc = [0.0] * 24
        for i in range(24):
            if i != 9:
                self.sw[i] = 1.0 if self.switches[i] == 1 else 0.0
                self.swc[i] = 1.0 if self.switches[i] > 0 else 0.0
            else:
                self.sw[i] = self.switches[i]
                self.swc[i] = self.switches[i]


class _Input:
    """Mirror of the C nrlmsise_input struct."""

    def __init__(
        self,
        doy: int,
        sec: float,
        alt: float,
        g_lat: float,
        g_long: float,
        lst: float,
        f107a: float,
        f107: float,
        ap: float,
        ap_a: Optional[Sequence[float]] = None,
    ):
        self.doy = doy
        self.sec = sec
        self.alt = alt
        self.g_lat = g_lat
        self.g_long = g_long
        self.lst = lst
        self.f107A = f107a
        self.f107 = f107
        self.ap = ap
        self.ap_a = None if ap_a is None else list(ap_a)


class _Model:
    """The shared-state block of the reference implementation.

    The C code keeps these as file-static globals (PARMB, GTS3C, DMIX,
    MESO7, LPOLY); one instance per evaluation keeps calls independent
    while preserving the intra-call data flow exactly.
    """

    def __init__(self) -> None:
        self.gsurf = 0.0
        self.re = 0.0
        self.dd = 0.0
        self.dm04 = self.dm16 = self.dm28 = 0.0
        self.dm32 = self.dm40 = self.dm01 = self.dm14 = 0.0
        self.meso_tn1 = [0.0] * 5
        self.meso_tn2 = [0.0] * 4
        self.meso_tn3 = [0.0] * 5
        self.meso_tgn1 = [0.0] * 2
        self.meso_tgn2 = [0.0] * 2
        self.meso_tgn3 = [0.0] * 2
        self.dfa = 0.0
        self.plg = [[0.0] * 9 for _ in range(4)]
        self.ctloc = self.stloc = 0.0
        self.c2tloc = self.s2tloc = 0.0
        self.s3tloc = self.c3tloc = 0.0
        self.apdf = 0.0
        self.apt = [0.0] * 4

    # -- small helpers ------------------------------------------------

    def _glatf(self, lat: float) -> None:
        """Latitude variation of gravity and effective radius."""
        dgtr = 1.74533e-2
        c2 = np.cos(2.0 * dgtr * lat)
        self.gsurf = 980.616 * (1.0 - 0.0026373 * c2)
        self.re = 2.0 * self.gsurf / (3.085462e-6 + 2.27e-9 * c2) * 1.0e-5

    def _scalh(self, alt: float, xm: float, temp: float) -> float:
        rgas = 831.4
        g = self.gsurf / ((1.0 + alt / self.re) ** 2)
        return rgas * temp / (g * xm)

    def _zeta(self, zz: float, zl: float) -> float:
        return (zz - zl) * (self.re + zl) / (self.re + zz)

    # -- density/temperature profile machinery ------------------------

    def _densm(
        self,
        alt: float,
        d0: float,
        xm: float,
        tz: float,
        zn3: Sequence[float],
        tn3: Sequence[float],
        tgn3: Sequence[float],
        zn2: Sequence[float],
        tn2: Sequence[float],
        tgn2: Sequence[float],
    ) -> Tuple[float, float]:
        """Temperature/density profiles for the lower atmosphere.

        Returns (density-or-temperature, tz), the C double return plus
        the pointer output.
        """
        mn3 = len(zn3)
        mn2 = len(zn2)
        rgas = 831.4
        densm_tmp = d0
        if alt > zn2[0]:
            if xm == 0.0:
                return tz, tz
            return d0, tz

        # Stratosphere/mesosphere temperature.
        z = alt if alt > zn2[mn2 - 1] else zn2[mn2 - 1]
        mn = mn2
        z1 = zn2[0]
        z2 = zn2[mn - 1]
        t1 = tn2[0]
        t2 = tn2[mn - 1]
        zg = self._zeta(z, z1)
        zgdif = self._zeta(z2, z1)

        xs = [self._zeta(zn2[k], z1) / zgdif for k in range(mn)]
        ys = [1.0 / tn2[k] for k in range(mn)]
        yd1 = -tgn2[0] / (t1 * t1) * zgdif
        yd2 = -tgn2[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)) ** 2

        y2out = _spline(xs, ys, yd1, yd2)
        x = zg / zgdif
        y = _splint(xs, ys, y2out, x)

        tz = 1.0 / y
        if xm != 0.0:
            glb = self.gsurf / ((1.0 + z1 / self.re) ** 2)
            gamm = xm * glb * zgdif / rgas
            yi = _splini(xs, ys, y2out, x)
            expl = gamm * yi
            if expl > 50.0:
                expl = 50.0
            densm_tmp = densm_tmp * (t1 / tz) * np.exp(-expl)

        if alt > zn3[0]:
            if xm == 0.0:
                return tz, tz
            return densm_tmp, tz

        # Troposphere/stratosphere temperature.
        z = alt
        mn = mn3
        z1 = zn3[0]
        z2 = zn3[mn - 1]
        t1 = tn3[0]
        t2 = tn3[mn - 1]
        zg = self._zeta(z, z1)
        zgdif = self._zeta(z2, z1)

        xs = [self._zeta(zn3[k], z1) / zgdif for k in range(mn)]
        ys = [1.0 / tn3[k] for k in range(mn)]
        yd1 = -tgn3[0] / (t1 * t1) * zgdif
        yd2 = -tgn3[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)) ** 2

        y2out = _spline(xs, ys, yd1, yd2)
        x = zg / zgdif
        y = _splint(xs, ys, y2out, x)

        tz = 1.0 / y
        if xm != 0.0:
            glb = self.gsurf / ((1.0 + z1 / self.re) ** 2)
            gamm = xm * glb * zgdif / rgas
            yi = _splini(xs, ys, y2out, x)
            expl = gamm * yi
            if expl > 50.0:
                expl = 50.0
            densm_tmp = densm_tmp * (t1 / tz) * np.exp(-expl)
        if xm == 0.0:
            return tz, tz
        return densm_tmp, tz

    def _densu(
        self,
        alt: float,
        dlb: float,
        tinf: float,
        tlb: float,
        xm: float,
        alpha: float,
        tz: float,
        zlb: float,
        s2: float,
        zn1: Sequence[float],
        tn1,
        tgn1,
    ) -> Tuple[float, float]:
        """Temperature/density profiles for the MSIS thermosphere.

        Returns (density-or-temperature, tz). tn1/tgn1 are mutated at
        their first entries when below the Bates joining altitude,
        exactly as the C writes through its pointers.
        """
        mn1 = len(zn1)
        rgas = 831.4
        x = 0.0
        z1 = 0.0
        t1 = 0.0
        zgdif = 0.0
        mn = 0
        xs: list = []
        ys: list = []
        y2out: list = []

        za = zn1[0]
        z = alt if alt > za else za

        zg2 = self._zeta(z, zlb)
        tt = tinf - (tinf - tlb) * np.exp(-s2 * zg2)
        ta = tt
        tz = tt
        densu_temp = tz

        if alt < za:
            # Temperature gradient at ZA from the Bates profile.
            dta = (tinf - ta) * s2 * ((self.re + zlb) / (self.re + za)) ** 2
            tgn1[0] = dta
            tn1[0] = ta
            z = alt if alt > zn1[mn1 - 1] else zn1[mn1 - 1]
            mn = mn1
            z1 = zn1[0]
            z2 = zn1[mn - 1]
            t1 = tn1[0]
            t2 = tn1[mn - 1]
            zg = self._zeta(z, z1)
            zgdif = self._zeta(z2, z1)
            xs = [self._zeta(zn1[k], z1) / zgdif for k in range(mn)]
            ys = [1.0 / tn1[k] for k in range(mn)]
            yd1 = -tgn1[0] / (t1 * t1) * zgdif
            yd2 = -tgn1[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)) ** 2
            y2out = _spline(xs, ys, yd1, yd2)
            x = zg / zgdif
            y = _splint(xs, ys, y2out, x)
            tz = 1.0 / y
            densu_temp = tz
        if xm == 0:
            return densu_temp, tz

        # Density above ZA.
        glb = self.gsurf / ((1.0 + zlb / self.re) ** 2)
        gamma = xm * glb / (s2 * rgas * tinf)
        expl = np.exp(-s2 * gamma * zg2)
        if expl > 50.0:
            expl = 50.0
        if tt <= 0:
            expl = 50.0

        densa = dlb * (tlb / tt) ** (1.0 + alpha + gamma) * expl
        densu_temp = densa
        if alt >= za:
            return densu_temp, tz

        # Density below ZA.
        glb = self.gsurf / ((1.0 + z1 / self.re) ** 2)
        gamm = xm * glb * zgdif / rgas

        yi = _splini(xs, ys, y2out, x)
        expl = gamm * yi
        if expl > 50.0:
            expl = 50.0
        if tz <= 0:
            expl = 50.0

        densu_temp = densu_temp * (t1 / tz) ** (1.0 + alpha) * np.exp(-expl)
        return densu_temp, tz

    # -- G(L) machinery ------------------------------------------------

    def _globe7(self, p, inp: _Input, flags: _Flags) -> float:
        """Upper-thermosphere G(L); mutates the LPOLY shared state."""
        t = [0.0] * 15
        sr = 7.2722e-5
        dgtr = 1.74533e-2
        dr = 1.72142e-2
        hr = 0.2618

        tloc = inp.lst

        c = np.sin(inp.g_lat * dgtr)
        s = np.cos(inp.g_lat * dgtr)
        c2 = c * c
        c4 = c2 * c2
        s2 = s * s

        plg = self.plg
        plg[0][1] = c
        plg[0][2] = 0.5 * (3.0 * c2 - 1.0)
        plg[0][3] = 0.5 * (5.0 * c * c2 - 3.0 * c)
        plg[0][4] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0
        plg[0][5] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0
        plg[0][6] = (11.0 * c * plg[0][5] - 5.0 * plg[0][4]) / 6.0
        plg[1][1] = s
        plg[1][2] = 3.0 * c * s
        plg[1][3] = 1.5 * (5.0 * c2 - 1.0) * s
        plg[1][4] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s
        plg[1][5] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s
        plg[1][6] = (11.0 * c * plg[1][5] - 6.0 * plg[1][4]) / 5.0
        plg[2][2] = 3.0 * s2
        plg[2][3] = 15.0 * s2 * c
        plg[2][4] = 7.5 * (7.0 * c2 - 1.0) * s2
        plg[2][5] = 3.0 * c * plg[2][4] - 2.0 * plg[2][3]
        plg[2][6] = (11.0 * c * plg[2][5] - 7.0 * plg[2][4]) / 4.0
        plg[2][7] = (13.0 * c * plg[2][6] - 8.0 * plg[2][5]) / 5.0
        plg[3][3] = 15.0 * s2 * s
        plg[3][4] = 105.0 * s2 * s * c
        plg[3][5] = (9.0 * c * plg[3][4] - 7.0 * plg[3][3]) / 2.0
        plg[3][6] = (11.0 * c * plg[3][5] - 8.0 * plg[3][4]) / 3.0

        if not ((flags.sw[7] == 0 and flags.sw[8] == 0) and flags.sw[14] == 0):
            self.stloc = np.sin(hr * tloc)
            self.ctloc = np.cos(hr * tloc)
            self.s2tloc = np.sin(2.0 * hr * tloc)
            self.c2tloc = np.cos(2.0 * hr * tloc)
            self.s3tloc = np.sin(3.0 * hr * tloc)
            self.c3tloc = np.cos(3.0 * hr * tloc)

        cd32 = np.cos(dr * (inp.doy - p[31]))
        cd18 = np.cos(2.0 * dr * (inp.doy - p[17]))
        cd14 = np.cos(dr * (inp.doy - p[13]))
        cd39 = np.cos(2.0 * dr * (inp.doy - p[38]))

        # F10.7 effect.
        df = inp.f107 - inp.f107A
        self.dfa = inp.f107A - 150.0
        dfa = self.dfa
        t[0] = (
            p[19] * df * (1.0 + p[59] * dfa)
            + p[20] * df * df
            + p[21] * dfa
            + p[29] * dfa**2
        )
        f1 = 1.0 + (p[47] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1]
        f2 = 1.0 + (p[49] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1]

        # Time independent.
        t[1] = (
            (p[1] * plg[0][2] + p[2] * plg[0][4] + p[22] * plg[0][6])
            + (p[14] * plg[0][2]) * dfa * flags.swc[1]
            + p[26] * plg[0][1]
        )

        # Symmetrical annual.
        t[2] = p[18] * cd32

        # Symmetrical semiannual.
        t[3] = (p[15] + p[16] * plg[0][2]) * cd18

        # Asymmetrical annual.
        t[4] = f1 * (p[9] * plg[0][1] + p[10] * plg[0][3]) * cd14

        # Asymmetrical semiannual.
        t[5] = p[37] * plg[0][1] * cd39

        # Diurnal.
        if flags.sw[7]:
            t71 = (p[11] * plg[1][2]) * cd14 * flags.swc[5]
            t72 = (p[12] * plg[1][2]) * cd14 * flags.swc[5]
            t[6] = f2 * (
                (p[3] * plg[1][1] + p[4] * plg[1][3] + p[27] * plg[1][5] + t71)
                * self.ctloc
                + (p[6] * plg[1][1] + p[7] * plg[1][3] + p[28] * plg[1][5] + t72)
                * self.stloc
            )

        # Semidiurnal.
        if flags.sw[8]:
            t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14 * flags.swc[5]
            t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14 * flags.swc[5]
            t[7] = f2 * (
                (p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * self.c2tloc
                + (p[8] * plg[2][2] + p[42] * plg[2][4] + t82) * self.s2tloc
            )

        # Terdiurnal.
        if flags.sw[14]:
            t[13] = f2 * (
                (
                    p[39] * plg[3][3]
                    + (p[93] * plg[3][4] + p[46] * plg[3][6]) * cd14 * flags.swc[5]
                )
                * self.s3tloc
                + (
                    p[40] * plg[3][3]
                    + (p[94] * plg[3][4] + p[48] * plg[3][6]) * cd14 * flags.swc[5]
                )
                * self.c3tloc
            )

        # Magnetic activity based on daily ap.
        if flags.sw[9] == -1:
            ap = inp.ap_a
            if p[51] != 0:
                exp1 = np.exp(
                    -10800.0
                    * np.sqrt(p[51] * p[51])
                    / (1.0 + p[138] * (45.0 - np.sqrt(inp.g_lat * inp.g_lat)))
                )
                if exp1 > 0.99999:
                    exp1 = 0.99999
                if p[24] < 1.0e-4:
                    # The C mutates the shared coefficient array; the
                    # clamp is idempotent so persistence is harmless.
                    p[24] = 1.0e-4
                self.apt[0] = _sg0(exp1, p, ap)
                if flags.sw[9]:
                    t[8] = self.apt[0] * (
                        p[50]
                        + p[96] * plg[0][2]
                        + p[54] * plg[0][4]
                        + (p[125] * plg[0][1] + p[126] * plg[0][3] + p[127] * plg[0][5])
                        * cd14
                        * flags.swc[5]
                        + (p[128] * plg[1][1] + p[129] * plg[1][3] + p[130] * plg[1][5])
                        * flags.swc[7]
                        * np.cos(hr * (tloc - p[131]))
                    )
        else:
            apd = inp.ap - 4.0
            p44 = p[43]
            p45 = p[44]
            if p44 < 0:
                p44 = 1.0e-5
            self.apdf = apd + (p45 - 1.0) * (apd + (np.exp(-p44 * apd) - 1.0) / p44)
            if flags.sw[9]:
                t[8] = self.apdf * (
                    p[32]
                    + p[45] * plg[0][2]
                    + p[34] * plg[0][4]
                    + (p[100] * plg[0][1] + p[101] * plg[0][3] + p[102] * plg[0][5])
                    * cd14
                    * flags.swc[5]
                    + (p[121] * plg[1][1] + p[122] * plg[1][3] + p[123] * plg[1][5])
                    * flags.swc[7]
                    * np.cos(hr * (tloc - p[124]))
                )

        if flags.sw[10] and inp.g_long > -1000.0:
            # Longitudinal.
            if flags.sw[11]:
                t[10] = (1.0 + p[80] * dfa * flags.swc[1]) * (
                    (
                        p[64] * plg[1][2]
                        + p[65] * plg[1][4]
                        + p[66] * plg[1][6]
                        + p[103] * plg[1][1]
                        + p[104] * plg[1][3]
                        + p[105] * plg[1][5]
                        + flags.swc[5]
                        * (p[109] * plg[1][1] + p[110] * plg[1][3] + p[111] * plg[1][5])
                        * cd14
                    )
                    * np.cos(dgtr * inp.g_long)
                    + (
                        p[90] * plg[1][2]
                        + p[91] * plg[1][4]
                        + p[92] * plg[1][6]
                        + p[106] * plg[1][1]
                        + p[107] * plg[1][3]
                        + p[108] * plg[1][5]
                        + flags.swc[5]
                        * (p[112] * plg[1][1] + p[113] * plg[1][3] + p[114] * plg[1][5])
                        * cd14
                    )
                    * np.sin(dgtr * inp.g_long)
                )

            # UT and mixed UT, longitude.
            if flags.sw[12]:
                t[11] = (
                    (1.0 + p[95] * plg[0][1])
                    * (1.0 + p[81] * dfa * flags.swc[1])
                    * (1.0 + p[119] * plg[0][1] * flags.swc[5] * cd14)
                    * (
                        (p[68] * plg[0][1] + p[69] * plg[0][3] + p[70] * plg[0][5])
                        * np.cos(sr * (inp.sec - p[71]))
                    )
                )
                t[11] += (
                    flags.swc[11]
                    * (p[76] * plg[2][3] + p[77] * plg[2][5] + p[78] * plg[2][7])
                    * np.cos(sr * (inp.sec - p[79]) + 2.0 * dgtr * inp.g_long)
                    * (1.0 + p[137] * dfa * flags.swc[1])
                )

            # UT, longitude magnetic activity.
            if flags.sw[13]:
                if flags.sw[9] == -1:
                    if p[51]:
                        t[12] = (
                            self.apt[0]
                            * flags.swc[11]
                            * (1.0 + p[132] * plg[0][1])
                            * (
                                (
                                    p[52] * plg[1][2]
                                    + p[98] * plg[1][4]
                                    + p[67] * plg[1][6]
                                )
                                * np.cos(dgtr * (inp.g_long - p[97]))
                            )
                            + self.apt[0]
                            * flags.swc[11]
                            * flags.swc[5]
                            * (
                                p[133] * plg[1][1]
                                + p[134] * plg[1][3]
                                + p[135] * plg[1][5]
                            )
                            * cd14
                            * np.cos(dgtr * (inp.g_long - p[136]))
                            + self.apt[0]
                            * flags.swc[12]
                            * (
                                p[55] * plg[0][1]
                                + p[56] * plg[0][3]
                                + p[57] * plg[0][5]
                            )
                            * np.cos(sr * (inp.sec - p[58]))
                        )
                else:
                    t[12] = (
                        self.apdf
                        * flags.swc[11]
                        * (1.0 + p[120] * plg[0][1])
                        * (
                            (p[60] * plg[1][2] + p[61] * plg[1][4] + p[62] * plg[1][6])
                            * np.cos(dgtr * (inp.g_long - p[63]))
                        )
                        + self.apdf
                        * flags.swc[11]
                        * flags.swc[5]
                        * (p[115] * plg[1][1] + p[116] * plg[1][3] + p[117] * plg[1][5])
                        * cd14
                        * np.cos(dgtr * (inp.g_long - p[118]))
                        + self.apdf
                        * flags.swc[12]
                        * (p[83] * plg[0][1] + p[84] * plg[0][3] + p[85] * plg[0][5])
                        * np.cos(sr * (inp.sec - p[75]))
                    )

        tinf = p[30]
        for i in range(14):
            tinf = tinf + abs(flags.sw[i + 1]) * t[i]
        return tinf

    def _glob7s(self, p, inp: _Input, flags: _Flags) -> float:
        """Lower-atmosphere version of the G(L) function."""
        pset = 2.0
        t = [0.0] * 14
        dr = 1.72142e-2
        dgtr = 1.74533e-2
        if p[99] == 0:
            p[99] = pset
        if p[99] != pset:
            warnings.warn("Wrong parameter set for glob7s", stacklevel=2)
            return -1.0
        cd32 = np.cos(dr * (inp.doy - p[31]))
        cd18 = np.cos(2.0 * dr * (inp.doy - p[17]))
        cd14 = np.cos(dr * (inp.doy - p[13]))
        cd39 = np.cos(2.0 * dr * (inp.doy - p[38]))

        plg = self.plg
        t[0] = p[21] * self.dfa
        t[1] = (
            p[1] * plg[0][2]
            + p[2] * plg[0][4]
            + p[22] * plg[0][6]
            + p[26] * plg[0][1]
            + p[14] * plg[0][3]
            + p[59] * plg[0][5]
        )
        t[2] = (p[18] + p[47] * plg[0][2] + p[29] * plg[0][4]) * cd32
        t[3] = (p[15] + p[16] * plg[0][2] + p[30] * plg[0][4]) * cd18
        t[4] = (p[9] * plg[0][1] + p[10] * plg[0][3] + p[20] * plg[0][5]) * cd14
        t[5] = (p[37] * plg[0][1]) * cd39

        if flags.sw[7]:
            t71 = p[11] * plg[1][2] * cd14 * flags.swc[5]
            t72 = p[12] * plg[1][2] * cd14 * flags.swc[5]
            t[6] = (p[3] * plg[1][1] + p[4] * plg[1][3] + t71) * self.ctloc + (
                p[6] * plg[1][1] + p[7] * plg[1][3] + t72
            ) * self.stloc

        if flags.sw[8]:
            t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14 * flags.swc[5]
            t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14 * flags.swc[5]
            t[7] = (p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * self.c2tloc + (
                p[8] * plg[2][2] + p[42] * plg[2][4] + t82
            ) * self.s2tloc

        if flags.sw[14]:
            t[13] = p[39] * plg[3][3] * self.s3tloc + p[40] * plg[3][3] * self.c3tloc

        if flags.sw[9]:
            if flags.sw[9] == 1:
                t[8] = self.apdf * (p[32] + p[45] * plg[0][2] * flags.swc[2])
            if flags.sw[9] == -1:
                t[8] = (
                    p[50] * self.apt[0] + p[96] * plg[0][2] * self.apt[0] * flags.swc[2]
                )

        if not (flags.sw[10] == 0 or flags.sw[11] == 0 or inp.g_long <= -1000.0):
            t[10] = (
                1.0
                + plg[0][1]
                * (
                    p[80] * flags.swc[5] * np.cos(dr * (inp.doy - p[81]))
                    + p[85] * flags.swc[6] * np.cos(2.0 * dr * (inp.doy - p[86]))
                )
                + p[83] * flags.swc[3] * np.cos(dr * (inp.doy - p[84]))
                + p[87] * flags.swc[4] * np.cos(2.0 * dr * (inp.doy - p[88]))
            ) * (
                (
                    p[64] * plg[1][2]
                    + p[65] * plg[1][4]
                    + p[66] * plg[1][6]
                    + p[74] * plg[1][1]
                    + p[75] * plg[1][3]
                    + p[76] * plg[1][5]
                )
                * np.cos(dgtr * inp.g_long)
                + (
                    p[90] * plg[1][2]
                    + p[91] * plg[1][4]
                    + p[92] * plg[1][6]
                    + p[77] * plg[1][1]
                    + p[78] * plg[1][3]
                    + p[79] * plg[1][5]
                )
                * np.sin(dgtr * inp.g_long)
            )
        tt = 0.0
        for i in range(14):
            tt += abs(flags.sw[i + 1]) * t[i]
        return tt

    # -- top-level model ----------------------------------------------

    def gts7(self, inp: _Input, flags: _Flags) -> NRLMSISEOutput:
        """Thermospheric portion (alt > 72.5 km)."""
        zn1 = [120.0, 110.0, 100.0, 90.0, 72.5]
        dgtr = 1.74533e-2
        dr = 1.72142e-2
        alpha = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0]
        altl = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0]
        d = [0.0] * 9
        t = [0.0, 0.0]

        za = PDL[1][15]
        zn1[0] = za

        # TINF variations not important below ZA or ZN1(1).
        if inp.alt > zn1[0]:
            tinf = PTM[0] * PT[0] * (1.0 + flags.sw[16] * self._globe7(PT, inp, flags))
        else:
            tinf = PTM[0] * PT[0]
        t[0] = tinf

        # Gradient variations not important below ZN1(5).
        if inp.alt > zn1[4]:
            g0v = PTM[3] * PS[0] * (1.0 + flags.sw[19] * self._globe7(PS, inp, flags))
        else:
            g0v = PTM[3] * PS[0]
        tlb = PTM[1] * (1.0 + flags.sw[17] * self._globe7(PD[3], inp, flags)) * PD[3][0]
        s = g0v / (tinf - tlb)

        # Lower thermosphere temp variations not significant for
        # density above 300 km.
        if inp.alt < 300.0:
            self.meso_tn1[1] = (
                PTM[6]
                * PTL[0][0]
                / (1.0 - flags.sw[18] * self._glob7s(PTL[0], inp, flags))
            )
            self.meso_tn1[2] = (
                PTM[2]
                * PTL[1][0]
                / (1.0 - flags.sw[18] * self._glob7s(PTL[1], inp, flags))
            )
            self.meso_tn1[3] = (
                PTM[7]
                * PTL[2][0]
                / (1.0 - flags.sw[18] * self._glob7s(PTL[2], inp, flags))
            )
            self.meso_tn1[4] = (
                PTM[4]
                * PTL[3][0]
                / (1.0 - flags.sw[18] * flags.sw[20] * self._glob7s(PTL[3], inp, flags))
            )
            self.meso_tgn1[1] = (
                PTM[8]
                * PMA[8][0]
                * (1.0 + flags.sw[18] * flags.sw[20] * self._glob7s(PMA[8], inp, flags))
                * self.meso_tn1[4]
                * self.meso_tn1[4]
                / (PTM[4] * PTL[3][0]) ** 2
            )
        else:
            self.meso_tn1[1] = PTM[6] * PTL[0][0]
            self.meso_tn1[2] = PTM[2] * PTL[1][0]
            self.meso_tn1[3] = PTM[7] * PTL[2][0]
            self.meso_tn1[4] = PTM[4] * PTL[3][0]
            self.meso_tgn1[1] = (
                PTM[8]
                * PMA[8][0]
                * self.meso_tn1[4]
                * self.meso_tn1[4]
                / (PTM[4] * PTL[3][0]) ** 2
            )

        # N2 variation factor at Zlb.
        g28 = flags.sw[21] * self._globe7(PD[2], inp, flags)

        # Variation of turbopause height.
        zhf = PDL[1][24] * (
            1.0
            + flags.sw[5]
            * PDL[0][24]
            * np.sin(dgtr * inp.g_lat)
            * np.cos(dr * (inp.doy - PT[13]))
        )
        t[0] = tinf
        xmm = PDM[2][4]
        z = inp.alt

        # N2 density.
        db28 = PDM[2][0] * np.exp(g28) * PD[2][0]
        d[2], t[1] = self._densu(
            z,
            db28,
            tinf,
            tlb,
            28.0,
            alpha[2],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zh28 = PDM[2][2] * zhf
        zhm28 = PDM[2][3] * PDL[1][5]
        xmd = 28.0 - xmm
        b28, tz = self._densu(
            zh28,
            db28,
            tinf,
            tlb,
            xmd,
            alpha[2] - 1.0,
            0.0,
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15] and z <= altl[2]:
            self.dm28, tz = self._densu(
                z,
                b28,
                tinf,
                tlb,
                xmm,
                alpha[2],
                tz,
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            d[2] = _dnet(d[2], self.dm28, zhm28, xmm, 28.0)

        # He density.
        g4 = flags.sw[21] * self._globe7(PD[0], inp, flags)
        db04 = PDM[0][0] * np.exp(g4) * PD[0][0]
        d[0], t[1] = self._densu(
            z,
            db04,
            tinf,
            tlb,
            4.0,
            alpha[0],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15] and z < altl[0]:
            zh04 = PDM[0][2]
            b04, t[1] = self._densu(
                zh04,
                db04,
                tinf,
                tlb,
                4.0 - xmm,
                alpha[0] - 1.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm04, t[1] = self._densu(
                z,
                b04,
                tinf,
                tlb,
                xmm,
                0.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            zhm04 = zhm28
            d[0] = _dnet(d[0], self.dm04, zhm04, xmm, 4.0)
            rl = np.log(b28 * PDM[0][1] / b04)
            zc04 = PDM[0][4] * PDL[1][0]
            hc04 = PDM[0][5] * PDL[1][1]
            d[0] = d[0] * _ccor(z, rl, hc04, zc04)

        # O density.
        g16 = flags.sw[21] * self._globe7(PD[1], inp, flags)
        db16 = PDM[1][0] * np.exp(g16) * PD[1][0]
        d[1], t[1] = self._densu(
            z,
            db16,
            tinf,
            tlb,
            16.0,
            alpha[1],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15] and z <= altl[1]:
            zh16 = PDM[1][2]
            b16, t[1] = self._densu(
                zh16,
                db16,
                tinf,
                tlb,
                16.0 - xmm,
                alpha[1] - 1.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm16, t[1] = self._densu(
                z,
                b16,
                tinf,
                tlb,
                xmm,
                0.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            zhm16 = zhm28
            d[1] = _dnet(d[1], self.dm16, zhm16, xmm, 16.0)
            rl = (
                PDM[1][1]
                * PDL[1][16]
                * (1.0 + flags.sw[1] * PDL[0][23] * (inp.f107A - 150.0))
            )
            hc16 = PDM[1][5] * PDL[1][3]
            zc16 = PDM[1][4] * PDL[1][2]
            hc216 = PDM[1][5] * PDL[1][4]
            d[1] = d[1] * _ccor2(z, rl, hc16, zc16, hc216)
            hcc16 = PDM[1][7] * PDL[1][13]
            zcc16 = PDM[1][6] * PDL[1][12]
            rc16 = PDM[1][3] * PDL[1][14]
            d[1] = d[1] * _ccor(z, rc16, hcc16, zcc16)

        # O2 density.
        g32 = flags.sw[21] * self._globe7(PD[4], inp, flags)
        db32 = PDM[3][0] * np.exp(g32) * PD[4][0]
        d[3], t[1] = self._densu(
            z,
            db32,
            tinf,
            tlb,
            32.0,
            alpha[3],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15]:
            if z <= altl[3]:
                zh32 = PDM[3][2]
                b32, t[1] = self._densu(
                    zh32,
                    db32,
                    tinf,
                    tlb,
                    32.0 - xmm,
                    alpha[3] - 1.0,
                    t[1],
                    PTM[5],
                    s,
                    zn1,
                    self.meso_tn1,
                    self.meso_tgn1,
                )
                self.dm32, t[1] = self._densu(
                    z,
                    b32,
                    tinf,
                    tlb,
                    xmm,
                    0.0,
                    t[1],
                    PTM[5],
                    s,
                    zn1,
                    self.meso_tn1,
                    self.meso_tgn1,
                )
                zhm32 = zhm28
                d[3] = _dnet(d[3], self.dm32, zhm32, xmm, 32.0)
                rl = np.log(b28 * PDM[3][1] / b32)
                hc32 = PDM[3][5] * PDL[1][7]
                zc32 = PDM[3][4] * PDL[1][6]
                d[3] = d[3] * _ccor(z, rl, hc32, zc32)
            hcc32 = PDM[3][7] * PDL[1][22]
            hcc232 = PDM[3][7] * PDL[0][22]
            zcc32 = PDM[3][6] * PDL[1][21]
            rc32 = (
                PDM[3][3]
                * PDL[1][23]
                * (1.0 + flags.sw[1] * PDL[0][23] * (inp.f107A - 150.0))
            )
            d[3] = d[3] * _ccor2(z, rc32, hcc32, zcc32, hcc232)

        # Ar density.
        g40 = flags.sw[21] * self._globe7(PD[5], inp, flags)
        db40 = PDM[4][0] * np.exp(g40) * PD[5][0]
        d[4], t[1] = self._densu(
            z,
            db40,
            tinf,
            tlb,
            40.0,
            alpha[4],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15] and z <= altl[4]:
            zh40 = PDM[4][2]
            b40, t[1] = self._densu(
                zh40,
                db40,
                tinf,
                tlb,
                40.0 - xmm,
                alpha[4] - 1.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm40, t[1] = self._densu(
                z,
                b40,
                tinf,
                tlb,
                xmm,
                0.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            zhm40 = zhm28
            d[4] = _dnet(d[4], self.dm40, zhm40, xmm, 40.0)
            rl = np.log(b28 * PDM[4][1] / b40)
            hc40 = PDM[4][5] * PDL[1][9]
            zc40 = PDM[4][4] * PDL[1][8]
            d[4] = d[4] * _ccor(z, rl, hc40, zc40)

        # Hydrogen density.
        g1 = flags.sw[21] * self._globe7(PD[6], inp, flags)
        db01 = PDM[5][0] * np.exp(g1) * PD[6][0]
        d[6], t[1] = self._densu(
            z,
            db01,
            tinf,
            tlb,
            1.0,
            alpha[6],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15] and z <= altl[6]:
            zh01 = PDM[5][2]
            b01, t[1] = self._densu(
                zh01,
                db01,
                tinf,
                tlb,
                1.0 - xmm,
                alpha[6] - 1.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm01, t[1] = self._densu(
                z,
                b01,
                tinf,
                tlb,
                xmm,
                0.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            zhm01 = zhm28
            d[6] = _dnet(d[6], self.dm01, zhm01, xmm, 1.0)
            rl = np.log(b28 * PDM[5][1] * np.sqrt(PDL[1][17] * PDL[1][17]) / b01)
            hc01 = PDM[5][5] * PDL[1][11]
            zc01 = PDM[5][4] * PDL[1][10]
            d[6] = d[6] * _ccor(z, rl, hc01, zc01)
            hcc01 = PDM[5][7] * PDL[1][19]
            zcc01 = PDM[5][6] * PDL[1][18]
            rc01 = PDM[5][3] * PDL[1][20]
            d[6] = d[6] * _ccor(z, rc01, hcc01, zcc01)

        # Atomic nitrogen density.
        g14 = flags.sw[21] * self._globe7(PD[7], inp, flags)
        db14 = PDM[6][0] * np.exp(g14) * PD[7][0]
        d[7], t[1] = self._densu(
            z,
            db14,
            tinf,
            tlb,
            14.0,
            alpha[7],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[15] and z <= altl[7]:
            zh14 = PDM[6][2]
            b14, t[1] = self._densu(
                zh14,
                db14,
                tinf,
                tlb,
                14.0 - xmm,
                alpha[7] - 1.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm14, t[1] = self._densu(
                z,
                b14,
                tinf,
                tlb,
                xmm,
                0.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            zhm14 = zhm28
            d[7] = _dnet(d[7], self.dm14, zhm14, xmm, 14.0)
            rl = np.log(b28 * PDM[6][1] * np.sqrt(PDL[0][2] * PDL[0][2]) / b14)
            hc14 = PDM[6][5] * PDL[0][1]
            zc14 = PDM[6][4] * PDL[0][0]
            d[7] = d[7] * _ccor(z, rl, hc14, zc14)
            hcc14 = PDM[6][7] * PDL[0][4]
            zcc14 = PDM[6][6] * PDL[0][3]
            rc14 = PDM[6][3] * PDL[0][5]
            d[7] = d[7] * _ccor(z, rc14, hcc14, zcc14)

        # Anomalous oxygen density.
        g16h = flags.sw[21] * self._globe7(PD[8], inp, flags)
        db16h = PDM[7][0] * np.exp(g16h) * PD[8][0]
        tho = PDM[7][9] * PDL[0][6]
        dde, t[1] = self._densu(
            z,
            db16h,
            tho,
            tho,
            16.0,
            alpha[8],
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zsht = PDM[7][5]
        zmho = PDM[7][4]
        zsho = self._scalh(zmho, 16.0, tho)
        d[8] = dde * np.exp(-zsht / zsho * (np.exp(-(z - zmho) / zsht) - 1.0))

        # Total mass density.
        d[5] = 1.66e-24 * (
            4.0 * d[0]
            + 16.0 * d[1]
            + 28.0 * d[2]
            + 32.0 * d[3]
            + 40.0 * d[4]
            + d[6]
            + 14.0 * d[7]
        )

        # Temperature.
        z = np.sqrt(inp.alt * inp.alt)
        _, t[1] = self._densu(
            z,
            1.0,
            tinf,
            tlb,
            0.0,
            0.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        if flags.sw[0]:
            for i in range(9):
                d[i] = d[i] * 1.0e6
            d[5] = d[5] / 1000.0
        return NRLMSISEOutput(np.array(d), np.array(t))

    def gtd7(self, inp: _Input, flags: _Flags) -> NRLMSISEOutput:
        """The full model: thermosphere plus lower atmosphere."""
        zn3 = [32.5, 20.0, 15.0, 10.0, 0.0]
        zn2 = [72.5, 55.0, 45.0, 32.5]
        zmix = 62.5

        # Latitude variation of gravity (none for sw[2]=0).
        xlat = inp.g_lat
        if flags.sw[2] == 0:
            xlat = 45.0
        self._glatf(xlat)

        xmm = PDM[2][4]

        # Thermosphere/mesosphere (above zn2[0]).
        altt = inp.alt if inp.alt > zn2[0] else zn2[0]
        tmp = inp.alt
        inp.alt = altt
        soutput = self.gts7(inp, flags)
        inp.alt = tmp
        if flags.sw[0]:  # metric adjustment
            dm28m = self.dm28 * 1.0e6
        else:
            dm28m = self.dm28
        d = [0.0] * 9
        t = [soutput.t[0], soutput.t[1]]
        if inp.alt >= zn2[0]:
            return NRLMSISEOutput(np.array(soutput.d), np.array(t))

        # Lower mesosphere/upper stratosphere (between zn3[0] and zn2[0]).
        self.meso_tgn2[0] = self.meso_tgn1[1]
        self.meso_tn2[0] = self.meso_tn1[4]
        self.meso_tn2[1] = (
            PMA[0][0]
            * PAVGM[0]
            / (1.0 - flags.sw[20] * self._glob7s(PMA[0], inp, flags))
        )
        self.meso_tn2[2] = (
            PMA[1][0]
            * PAVGM[1]
            / (1.0 - flags.sw[20] * self._glob7s(PMA[1], inp, flags))
        )
        self.meso_tn2[3] = (
            PMA[2][0]
            * PAVGM[2]
            / (1.0 - flags.sw[20] * flags.sw[22] * self._glob7s(PMA[2], inp, flags))
        )
        self.meso_tgn2[1] = (
            PAVGM[8]
            * PMA[9][0]
            * (1.0 + flags.sw[20] * flags.sw[22] * self._glob7s(PMA[9], inp, flags))
            * self.meso_tn2[3]
            * self.meso_tn2[3]
            / (PMA[2][0] * PAVGM[2]) ** 2
        )
        self.meso_tn3[0] = self.meso_tn2[3]

        if inp.alt <= zn3[0]:
            # Lower stratosphere and troposphere (below zn3[0]).
            self.meso_tgn3[0] = self.meso_tgn2[1]
            self.meso_tn3[1] = (
                PMA[3][0]
                * PAVGM[3]
                / (1.0 - flags.sw[22] * self._glob7s(PMA[3], inp, flags))
            )
            self.meso_tn3[2] = (
                PMA[4][0]
                * PAVGM[4]
                / (1.0 - flags.sw[22] * self._glob7s(PMA[4], inp, flags))
            )
            self.meso_tn3[3] = (
                PMA[5][0]
                * PAVGM[5]
                / (1.0 - flags.sw[22] * self._glob7s(PMA[5], inp, flags))
            )
            self.meso_tn3[4] = (
                PMA[6][0]
                * PAVGM[6]
                / (1.0 - flags.sw[22] * self._glob7s(PMA[6], inp, flags))
            )
            self.meso_tgn3[1] = (
                PMA[7][0]
                * PAVGM[7]
                * (1.0 + flags.sw[22] * self._glob7s(PMA[7], inp, flags))
                * self.meso_tn3[4]
                * self.meso_tn3[4]
                / (PMA[6][0] * PAVGM[6]) ** 2
            )

        # Linear transition to full mixing below zn2[0].
        dmc = 0.0
        if inp.alt > zmix:
            dmc = 1.0 - (zn2[0] - inp.alt) / (zn2[0] - zmix)
        dz28 = soutput.d[2]

        # N2 density.
        dmr = soutput.d[2] / dm28m - 1.0
        tz = 0.0
        d[2], tz = self._densm(
            inp.alt,
            dm28m,
            xmm,
            tz,
            zn3,
            self.meso_tn3,
            self.meso_tgn3,
            zn2,
            self.meso_tn2,
            self.meso_tgn2,
        )
        d[2] = d[2] * (1.0 + dmr * dmc)

        # He density.
        dmr = soutput.d[0] / (dz28 * PDM[0][1]) - 1.0
        d[0] = d[2] * PDM[0][1] * (1.0 + dmr * dmc)

        # O density.
        d[1] = 0.0
        d[8] = 0.0

        # O2 density.
        dmr = soutput.d[3] / (dz28 * PDM[3][1]) - 1.0
        d[3] = d[2] * PDM[3][1] * (1.0 + dmr * dmc)

        # Ar density.
        dmr = soutput.d[4] / (dz28 * PDM[4][1]) - 1.0
        d[4] = d[2] * PDM[4][1] * (1.0 + dmr * dmc)

        # Hydrogen and atomic nitrogen.
        d[6] = 0.0
        d[7] = 0.0

        # Total mass density.
        d[5] = 1.66e-24 * (
            4.0 * d[0]
            + 16.0 * d[1]
            + 28.0 * d[2]
            + 32.0 * d[3]
            + 40.0 * d[4]
            + d[6]
            + 14.0 * d[7]
        )
        if flags.sw[0]:
            d[5] = d[5] / 1000.0

        # Temperature at altitude.
        self.dd, tz = self._densm(
            inp.alt,
            1.0,
            0.0,
            tz,
            zn3,
            self.meso_tn3,
            self.meso_tgn3,
            zn2,
            self.meso_tn2,
            self.meso_tgn2,
        )
        t[1] = tz
        return NRLMSISEOutput(np.array(d), np.array(t))

    def gtd7d(self, inp: _Input, flags: _Flags) -> NRLMSISEOutput:
        """The full model with anomalous O in the effective density."""
        out = self.gtd7(inp, flags)
        d = out.d.copy()
        d[5] = 1.66e-24 * (
            4.0 * d[0]
            + 16.0 * d[1]
            + 28.0 * d[2]
            + 32.0 * d[3]
            + 40.0 * d[4]
            + d[6]
            + 14.0 * d[7]
            + 16.0 * d[8]
        )
        if flags.sw[0]:
            d[5] = d[5] / 1000.0
        return NRLMSISEOutput(d, out.t)

    def ghp7(
        self, inp: _Input, flags: _Flags, press: float
    ) -> Tuple[float, NRLMSISEOutput]:
        """Altitude at a given pressure (hPa), plus the model output
        there. Preserves the NRL modification returning the computed
        altitude."""
        bm = 1.3806e-19
        rgas = 831.4
        test = 0.00043
        ltest = 12
        pl = np.log10(press)
        if pl >= -5.0:
            if pl > 2.5:
                zi = 18.06 * (3.00 - pl)
            elif pl > 0.075:
                zi = 14.98 * (3.08 - pl)
            elif pl > -1:
                zi = 17.80 * (2.72 - pl)
            elif pl > -2:
                zi = 14.28 * (3.64 - pl)
            elif pl > -4:
                zi = 12.72 * (4.32 - pl)
            else:
                zi = 25.3 * (0.11 - pl)
            cl = inp.g_lat / 90.0
            cl2 = cl * cl
            if inp.doy < 182:
                cd = (1.0 - float(inp.doy)) / 91.25
            else:
                cd = float(inp.doy) / 91.25 - 3.0
            ca = 0.0
            if -1.11 < pl <= -0.23:
                ca = 1.0
            if pl > -0.23:
                ca = (2.79 - pl) / (2.79 + 0.23)
            if -3 < pl <= -1.11:
                ca = (-2.93 - pl) / (-2.93 + 1.11)
            z = zi - 4.87 * cl * cd * ca - 1.64 * cl2 * ca + 0.31 * ca * cl
        else:
            z = 22.0 * (pl + 4.0) ** 2 + 110.0

        out = NRLMSISEOutput(np.zeros(9), np.zeros(2))
        for line in range(1, ltest + 1):
            inp.alt = z
            out = self.gtd7(inp, flags)
            z = inp.alt
            xn = (
                out.d[0]
                + out.d[1]
                + out.d[2]
                + out.d[3]
                + out.d[4]
                + out.d[6]
                + out.d[7]
            )
            p = bm * xn * out.t[1]
            if flags.sw[0]:
                p = p * 1.0e-6
            diff = pl - np.log10(p)
            if np.sqrt(diff * diff) < test:
                return z, out
            if line == ltest:
                warnings.warn(
                    f"ghp7 not converging for press {press:e}, diff {diff:e}",
                    stacklevel=2,
                )
                return z, out
            xm = out.d[5] / xn / 1.66e-24
            if flags.sw[0]:
                xm = xm * 1.0e3
            g = self.gsurf / ((1.0 + z / self.re) ** 2)
            sh = rgas * out.t[1] / (xm * g)

            if line < 6:
                z = z - sh * diff * 2.302
            else:
                z = z - sh * diff
        return z, out


# ---------------------------------------------------------------------
# Stateless helpers (exact transcriptions).
# ---------------------------------------------------------------------


def _ccor(alt: float, r: float, h1: float, zh: float) -> float:
    """Chemistry/dissociation correction."""
    e = (alt - zh) / h1
    if e > 70:
        return np.exp(0)
    if e < -70:
        return np.exp(r)
    ex = np.exp(e)
    e = r / (1.0 + ex)
    return np.exp(e)


def _ccor2(alt: float, r: float, h1: float, zh: float, h2: float) -> float:
    """Chemistry/dissociation correction with two scale lengths."""
    e1 = (alt - zh) / h1
    e2 = (alt - zh) / h2
    if (e1 > 70) or (e2 > 70):
        return np.exp(0)
    if (e1 < -70) and (e2 < -70):
        return np.exp(r)
    ex1 = np.exp(e1)
    ex2 = np.exp(e2)
    ccor2v = r / (1.0 + 0.5 * (ex1 + ex2))
    return np.exp(ccor2v)


def _dnet(dde: float, dm: float, zhm: float, xmm: float, xm: float) -> float:
    """Turbopause correction (root mean density)."""
    a = zhm / (xmm - xm)
    if not ((dm > 0) and (dde > 0)):
        warnings.warn(f"dnet log error {dm:e} {dde:e} {xm:e}", stacklevel=2)
        if (dde == 0) and (dm == 0):
            dde = 1.0
        if dm == 0:
            return dde
        if dde == 0:
            return dm
    ylog = a * np.log(dm / dde)
    if ylog < -10:
        return dde
    if ylog > 10:
        return dm
    return dde * (1.0 + np.exp(ylog)) ** (1.0 / a)


def _splini(xa, ya, y2a, x: float) -> float:
    """Integrate the cubic spline from xa[0] to x."""
    n = len(xa)
    yi = 0.0
    klo = 0
    khi = 1
    while (x > xa[klo]) and (khi < n):
        xx = x
        if khi < (n - 1):
            xx = x if x < xa[khi] else xa[khi]
        h = xa[khi] - xa[klo]
        a = (xa[khi] - xx) / h
        b = (xx - xa[klo]) / h
        a2 = a * a
        b2 = b * b
        yi += (
            (1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + (
                (-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi]
            )
            * h
            * h
            / 6.0
        ) * h
        klo += 1
        khi += 1
    return yi


def _splint(xa, ya, y2a, x: float) -> float:
    """Cubic spline interpolation (Numerical Recipes form)."""
    n = len(xa)
    klo = 0
    khi = n - 1
    while (khi - klo) > 1:
        k = (khi + klo) // 2
        if xa[k] > x:
            khi = k
        else:
            klo = k
    h = xa[khi] - xa[klo]
    if h == 0.0:
        warnings.warn("bad XA input to splint", stacklevel=2)
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    return (
        a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0
    )


def _spline(x, y, yp1: float, ypn: float):
    """Second derivatives of the cubic spline (Numerical Recipes)."""
    n = len(x)
    y2 = [0.0] * n
    u = [0.0] * n
    if yp1 > 0.99e30:
        y2[0] = 0.0
        u[0] = 0.0
    else:
        y2[0] = -0.5
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (
            6.0
            * (
                (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            )
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1]
        ) / p
    if ypn > 0.99e30:
        qn = 0.0
        un = 0.0
    else:
        qn = 0.5
        un = (3.0 / (x[n - 1] - x[n - 2])) * (
            ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
        )
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]
    return y2


def _g0(a: float, p) -> float:
    """3-hour magnetic activity function, Eq. A24d."""
    return (
        a
        - 4.0
        + (p[25] - 1.0)
        * (
            a
            - 4.0
            + (np.exp(-np.sqrt(p[24] * p[24]) * (a - 4.0)) - 1.0)
            / np.sqrt(p[24] * p[24])
        )
    )


def _sumex(ex: float) -> float:
    """Eq. A24c."""
    return 1.0 + (1.0 - ex**19.0) / (1.0 - ex) * ex**0.5


def _sg0(ex: float, p, ap) -> float:
    """Eq. A24a."""
    return (
        _g0(ap[1], p)
        + (
            _g0(ap[2], p) * ex
            + _g0(ap[3], p) * ex * ex
            + _g0(ap[4], p) * ex**3.0
            + (_g0(ap[5], p) * ex**4.0 + _g0(ap[6], p) * ex**12.0)
            * (1.0 - ex**8.0)
            / (1.0 - ex)
        )
    ) / _sumex(ex)


# ---------------------------------------------------------------------
# Low-level public entry points (reference-implementation interface).
# ---------------------------------------------------------------------


def uses_compiled_backend() -> bool:
    """True when the compiled reference C extension is in use.

    The extension is built from the vendored reference implementation
    (csrc/nrlmsise00); when it cannot be imported this module falls
    back to the pure-Python transcription, which is validated against
    the same C code, so results agree to ~1e-14 either way.
    """
    return _c_ext is not None


def _run(
    doy: int,
    sec: float,
    alt_km: float,
    g_lat_deg: float,
    g_long_deg: float,
    lst: float,
    f107a: float,
    f107: float,
    ap: float,
    ap_a: Optional[Sequence[float]],
    kind: str,
    press: float = 0.0,
):
    ap_list = None if ap_a is None else [float(v) for v in ap_a]
    if _c_ext is not None:
        if kind in ("gtd7", "gtd7d"):
            d, t = _c_ext.gtd7(
                doy,
                sec,
                alt_km,
                g_lat_deg,
                g_long_deg,
                lst,
                f107a,
                f107,
                ap,
                ap_list,
                kind == "gtd7d",
            )
            return NRLMSISEOutput(np.array(d), np.array(t))
        if kind == "ghp7":
            alt, (d, t) = _c_ext.ghp7(
                doy,
                sec,
                press,
                g_lat_deg,
                g_long_deg,
                lst,
                f107a,
                f107,
                ap,
                ap_list,
            )
            return alt, NRLMSISEOutput(np.array(d), np.array(t))
        raise ValueError(kind)

    switches = [1.0] * 24
    if ap_a is not None:
        switches[9] = -1.0
    flags = _Flags(switches)
    inp = _Input(doy, sec, alt_km, g_lat_deg, g_long_deg, lst, f107a, f107, ap, ap_list)
    model = _Model()
    if kind == "gtd7":
        return model.gtd7(inp, flags)
    if kind == "gtd7d":
        return model.gtd7d(inp, flags)
    if kind == "ghp7":
        # ghp7 needs gravity set before its first scale-height use;
        # gtd7 sets it on the first iteration, as in the original.
        return model.ghp7(inp, flags, press)
    raise ValueError(kind)


def nrlmsise00(
    doy: int,
    sec: float,
    alt_km: float,
    g_lat_deg: float,
    g_long_deg: float,
    lst: float,
    f107a: float = 150.0,
    f107: float = 150.0,
    ap: float = 4.0,
    ap_array: Optional[Sequence[float]] = None,
    effective_density: bool = False,
) -> NRLMSISEOutput:
    """
    Evaluate the NRLMSISE-00 model (the ``gtd7``/``gtd7d`` interface).

    All 24 model switches are enabled (SI output units), matching the
    MATLAB TCL wrapper; a 7-element ``ap_array`` activates storm mode
    (``switches[9] = -1``) exactly as there.

    Parameters
    ----------
    doy : int
        Day of year, counting from 1.
    sec : float
        Second of the day (UT).
    alt_km : float
        Altitude in kilometers.
    g_lat_deg, g_long_deg : float
        Geodetic latitude and longitude in degrees.
    lst : float
        Local apparent solar time in hours. The reference documents
        the relation ``lst = sec/3600 + g_long/15``; it is an
        independent input here, as in the reference.
    f107a : float, optional
        81-day average 10.7 cm solar flux centered on the day. Default
        150.
    f107 : float, optional
        Daily 10.7 cm solar flux for the previous day. Default 150.
    ap : float, optional
        Daily magnetic index. Default 4.
    ap_array : sequence of 7 floats, optional
        Storm-mode Ap history (daily, 3-hourly now/-3h/-6h/-9h, and
        the 12-33h and 36-59h averages). Activates storm mode.
    effective_density : bool, optional
        If True, use ``gtd7d``: include anomalous oxygen in the total
        mass density d[5] (relevant to drag above ~500 km). Default
        False (``gtd7``).

    Returns
    -------
    result : NRLMSISEOutput

    Examples
    --------
    >>> out = nrlmsise00(172, 29000.0, 400.0, 60.0, -70.0, 16.0)
    >>> float(round(out.t[0], 2))  # exospheric temperature, K
    1250.54
    >>> float(round(out.d[5] * 1e12, 4))  # total mass density, kg/m^3
    4.0747

    Notes
    -----
    Transcribed from the NRL-vendored reference C implementation and
    validated against it record-for-record.
    """
    kind = "gtd7d" if effective_density else "gtd7"
    return _run(
        doy, sec, alt_km, g_lat_deg, g_long_deg, lst, f107a, f107, ap, ap_array, kind
    )


def nrlmsise00_alt_for_pressure(
    doy: int,
    sec: float,
    press_pa: float,
    g_lat_deg: float,
    g_long_deg: float,
    lst: float,
    f107a: float = 150.0,
    f107: float = 150.0,
    ap: float = 4.0,
    ap_array: Optional[Sequence[float]] = None,
) -> Tuple[float, NRLMSISEOutput]:
    """
    Altitude at which the atmosphere has the given pressure (``ghp7``).

    Parameters
    ----------
    doy, sec, g_lat_deg, g_long_deg, lst, f107a, f107, ap, ap_array
        As in :func:`nrlmsise00`.
    press_pa : float
        Pressure in Pascals.

    Returns
    -------
    alt_km : float
        The altitude in kilometers at which the model pressure matches.
    output : NRLMSISEOutput
        The model output at that altitude.

    Examples
    --------
    >>> alt, out = nrlmsise00_alt_for_pressure(
    ...     172, 29000.0, 1000.0, 60.0, -70.0, 16.0)
    >>> bool(15.0 < alt < 35.0)  # ~1000 Pa is upper-stratospheric
    True

    Notes
    -----
    ``ghp7`` iterates :func:`nrlmsise00` with scale-height Newton
    steps; the pressure is converted to the model's hPa convention
    internally. Preserves the NRL modification returning the computed
    altitude.
    """
    press_hpa = press_pa / 100.0
    z, out = _run(
        doy,
        sec,
        0.0,
        g_lat_deg,
        g_long_deg,
        lst,
        f107a,
        f107,
        ap,
        ap_array,
        "ghp7",
        press_hpa,
    )
    return z, out


__all__ = [
    "NRLMSISEOutput",
    "nrlmsise00",
    "nrlmsise00_alt_for_pressure",
    "uses_compiled_backend",
]
