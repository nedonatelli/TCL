"""Compiled NRLMSISE-00 failure reports reach Python's warning machinery.

The vendored C reported ghp7 non-convergence (and dnet's log-domain
error) with a bare ``printf`` to native stdout, which
``warnings.catch_warnings``, ``-W error`` and pytest's capture never
see. The pure-Python fallback has always used ``warnings.warn`` for the
same conditions. The compiled path now issues a ``RuntimeWarning`` per
model call and writes nothing to stdout; the returned values are
unchanged (pinned bit-for-bit below from the pre-fix extension).
"""

import ctypes
import sys
import warnings

import pytest

from pytcl.atmosphere.nrlmsise00 import (
    nrlmsise00_alt_for_pressure,
    uses_compiled_backend,
)

pytestmark = pytest.mark.skipif(
    not uses_compiled_backend(), reason="compiled NRLMSISE-00 extension unavailable"
)

if uses_compiled_backend():
    from pytcl.atmosphere import _nrlmsise00_c


def _flush_c_stdio():
    # printf output sits in the C runtime's stdio buffer when stdout is
    # not a tty; without this a leaked printf would escape capfd and the
    # stdout assertion would pass vacuously.
    libc = ctypes.cdll.ucrtbase if sys.platform == "win32" else ctypes.CDLL(None)
    libc.fflush(None)


# press = 1e-14 hPa (1e-12 Pa): the 12-iteration Newton loop ends with
# a finite residual diff = -0.3189, so ghp7 hits its non-convergence
# exit with a finite (if absurd, 3.19e89 km) altitude.
_GHP7_ARGS = (172, 29000.0, 1e-14, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0, None)
_GHP7_ALT = "0x1.40a2e8de03b3bp+297"
_GHP7_D = (
    "0x1.8947465283396p+6",
    "0x1.0deaa7232c1dcp-84",
    "0x1.86b509c3a3a5ep-185",
    "0x1.40580c47d758bp-223",
    "0x1.256f09da9fc72p-295",
    "0x1.d91bf2d719811p-63",
    "0x1.cc7323949e219p+26",
    "0x1.8414db881f494p-73",
    "0x1.f3444ca7ae90cp-4",
)
_GHP7_T = ("0x1.38a28e6f6fd47p+10", "0x1.38a28e6f6fd47p+10")

# ap = 1e4 is far past the physical Ap ceiling (400), but it is the only
# input class found that reaches dnet's log-domain guard, and it does so
# three times in one gtd7 call -- exercising both the gtd7 entry point
# and the one-warning-per-call aggregation.
_GTD7_ARGS = (172, 29000.0, 0.0, 0.0, -70.0, 16.0, 150.0, 150.0, 1e4, None, False)
_GTD7_D = (
    "nan",
    "0x0.0p+0",
    "0x1.946f6179a85c4p+9",
    "0x1.b1fdbf1354ee3p+7",
    "nan",
    "nan",
    "0x0.0p+0",
    "0x0.0p+0",
    "0x0.0p+0",
)
_GTD7_T = ("0x1.00d461ba9a9b2p+10", "0x1.2b2095bc70ef1p+8")


def _hex(values):
    return tuple(float(v).hex() for v in values)


class TestGhp7NonConvergence:
    def test_warns_runtimewarning_naming_ghp7(self):
        with pytest.warns(RuntimeWarning, match="ghp7 not converging") as rec:
            _nrlmsise00_c.ghp7(*_GHP7_ARGS)
        assert len(rec) == 1
        assert str(rec[0].message) == (
            "ERROR: ghp7 not converging for press 1.000000e-14, diff -3.188880e-01"
        )

    def test_values_bit_identical_to_pre_fix_output(self):
        with pytest.warns(RuntimeWarning, match="ghp7"):
            alt, (d, t) = _nrlmsise00_c.ghp7(*_GHP7_ARGS)
        assert alt.hex() == _GHP7_ALT
        assert _hex(d) == _GHP7_D
        assert _hex(t) == _GHP7_T

    def test_raises_under_error_filter(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeWarning, match="ghp7"):
                _nrlmsise00_c.ghp7(*_GHP7_ARGS)

    def test_nothing_written_to_stdout(self, capfd):
        with pytest.warns(RuntimeWarning, match="ghp7"):
            _nrlmsise00_c.ghp7(*_GHP7_ARGS)
        _flush_c_stdio()
        assert capfd.readouterr().out == ""

    def test_suppressible(self, capfd):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", message=".*ghp7", category=RuntimeWarning)
            alt, _ = _nrlmsise00_c.ghp7(*_GHP7_ARGS)
        _flush_c_stdio()
        assert alt.hex() == _GHP7_ALT
        assert capfd.readouterr().out == ""

    def test_reaches_the_public_wrapper(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            nrlmsise00_alt_for_pressure(
                press_pa=1e-12,
                doy=172,
                sec=29000.0,
                g_lat_deg=60.0,
                g_long_deg=-70.0,
                lst=16.0,
            )
        runtime = [w for w in rec if w.category is RuntimeWarning]
        assert [str(w.message) for w in runtime] == [
            "ERROR: ghp7 not converging for press 1.000000e-14, diff -3.188880e-01"
        ]

    def test_converging_call_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            alt, _ = _nrlmsise00_c.ghp7(
                172, 29000.0, 10.0, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0, None
            )
        assert 15.0 < alt < 35.0


class TestDnetLogError:
    def test_one_aggregated_warning_per_call(self):
        with pytest.warns(RuntimeWarning, match="dnet log error") as rec:
            _nrlmsise00_c.gtd7(*_GTD7_ARGS)
        assert len(rec) == 1
        assert str(rec[0].message) == (
            "dnet log error nan 4.313267e+16 4.000000e+00; "
            "dnet log error nan 3.177709e+16 4.000000e+01; "
            "dnet log error nan 2.553271e+00 1.000000e+00"
        )

    def test_values_bit_identical_to_pre_fix_output(self):
        with pytest.warns(RuntimeWarning, match="dnet"):
            d, t = _nrlmsise00_c.gtd7(*_GTD7_ARGS)
        assert _hex(d) == _GTD7_D
        assert _hex(t) == _GTD7_T

    def test_raises_under_error_filter(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeWarning, match="dnet"):
                _nrlmsise00_c.gtd7(*_GTD7_ARGS)

    def test_nothing_written_to_stdout(self, capfd):
        with pytest.warns(RuntimeWarning, match="dnet"):
            _nrlmsise00_c.gtd7(*_GTD7_ARGS)
        _flush_c_stdio()
        assert capfd.readouterr().out == ""
