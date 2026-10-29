"""Property coverage of the filter-design wrappers.

Each property is a closed-form consequence of the textbook prototype after the
bilinear transform (analog frequency ``Omega = tan(omega / 2)``), evaluated
through ``scipy.signal.freqz`` / direct polynomial evaluation rather than
through the code under test. Frequencies below are in Hz at ``FS = 1000``.
"""

import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb
from scipy import signal, special

from pytcl.mathematical_functions.signal_processing import filters as flt

FS = 1000.0


def _tan_half(f_hz):
    return np.tan(np.pi * np.asarray(f_hz) / FS)


def _gain_db(coeffs, f_hz):
    w = 2 * np.pi * np.atleast_1d(f_hz) / FS
    if isinstance(coeffs, flt.FilterCoefficients):
        _, h = signal.freqz(coeffs.b, coeffs.a, worN=w)
    else:
        _, h = signal.freqz(coeffs, [1.0], worN=w)
    return 20 * np.log10(np.abs(h))


def _within(got, want, tol):
    assert np.max(np.abs(np.asarray(got) - np.asarray(want))) <= tol


def _h(b, a, w):
    k_b = np.arange(len(b))
    k_a = np.arange(len(a))
    num = (np.asarray(b)[:, None] * np.exp(-1j * np.outer(k_b, w))).sum(axis=0)
    den = (np.asarray(a)[:, None] * np.exp(-1j * np.outer(k_a, w))).sum(axis=0)
    return num / den


class TestButterworth:
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 7])
    @pytest.mark.parametrize("output", ["sos", "ba"])
    def test_minus_3_dB_at_cutoff(self, order, output):
        c = flt.butter_design(order, 120.0, FS, output=output)
        assert _gain_db(c, 120.0)[0] == pytest.approx(
            10 * np.log10(0.5), rel=1e-10, abs=0
        )

    @pytest.mark.parametrize("order", [2, 3, 5])
    def test_closed_form_magnitude_squared(self, order):
        c = flt.butter_design(order, 120.0, FS)
        f = np.linspace(1.0, 300.0, 200)
        want = 1.0 / (1.0 + (_tan_half(f) / _tan_half(120.0)) ** (2 * order))
        got = 10 ** (_gain_db(c, f) / 10)
        assert got == pytest.approx(want, rel=1e-8, abs=0)

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_maximally_flat_at_dc(self, order):
        # 1 - |H|^2 = O(omega^(2N)): doubling omega multiplies it by 2^(2N)
        # up to O(omega^2) corrections, i.e. the first 2N-1 derivatives vanish.
        c = flt.butter_design(order, 250.0, FS, output="ba")
        w = np.array([0.05, 0.1])
        _, h = signal.freqz(c.b, c.a, worN=w)
        dev = 1.0 - np.abs(h) ** 2
        assert dev[1] / dev[0] == pytest.approx(2.0 ** (2 * order), rel=0.05, abs=0)

    def test_highpass_and_bandpass_cutoffs(self):
        hp = flt.butter_design(4, 200.0, FS, btype="high")
        assert _gain_db(hp, 200.0)[0] == pytest.approx(
            10 * np.log10(0.5), rel=1e-10, abs=0
        )
        bp = flt.butter_design(3, (100.0, 200.0), FS, btype="band", output="ba")
        gains = _gain_db(bp, [100.0, 200.0])
        assert gains == pytest.approx(np.full(2, 10 * np.log10(0.5)), rel=1e-9, abs=0)

    def test_sos_and_ba_describe_the_same_filter(self):
        s = flt.butter_design(4, 120.0, FS, output="sos")
        b = flt.butter_design(4, 120.0, FS, output="ba")
        assert s.sos is not None and b.sos is None
        w = np.linspace(0.01, 3.0, 64)
        assert np.abs(_h(s.b, s.a, w)) == pytest.approx(
            np.abs(_h(b.b, b.a, w)), rel=1e-9, abs=0
        )

    def test_poles_are_the_bilinear_image_of_the_butterworth_circle(self):
        n = 5
        c = flt.butter_design(n, 120.0, FS)
        _, p, _ = flt.sos_to_zpk(c.sos)
        p = p[np.abs(p) > 1e-9]  # sos padding of the odd-order section
        k = np.arange(n)
        s = _tan_half(120.0) * np.exp(1j * np.pi * (2 * k + n + 1) / (2 * n))
        want = (1 + s) / (1 - s)
        assert len(p) == n
        for root in want:
            assert np.min(np.abs(p - root)) < 1e-9
        assert np.all(np.abs(p) < 1.0)


class TestChebyshevAndElliptic:
    @pytest.mark.parametrize("order", [3, 4, 5])
    @pytest.mark.parametrize("rp", [0.1, 0.5, 2.0])
    @pytest.mark.parametrize("output", ["sos", "ba"])
    def test_cheby1_ripple_equals_requested(self, order, rp, output):
        c = flt.cheby1_design(order, rp, 150.0, FS, output=output)
        eps2 = 10 ** (rp / 10) - 1
        f = np.linspace(0.5, 150.0, 20001)
        t = cheb.chebval(_tan_half(f) / _tan_half(150.0), [0] * order + [1])
        want = 1.0 / (1.0 + eps2 * t**2)
        got = _gain_db(c, f)
        assert 10 ** (got / 10) == pytest.approx(want, rel=1e-8, abs=0)
        assert _gain_db(c, 150.0)[0] == pytest.approx(-rp, rel=1e-9, abs=0)
        assert got.min() >= -rp - 1e-9
        assert got.max() <= 1e-9

    @pytest.mark.parametrize("order", [3, 4])
    @pytest.mark.parametrize("output", ["sos", "ba"])
    def test_cheby2_attenuation_at_stopband_edge(self, order, output):
        c = flt.cheby2_design(order, 40.0, 200.0, FS, output=output)
        assert _gain_db(c, 200.0)[0] == pytest.approx(-40.0, rel=1e-9, abs=0)
        assert _gain_db(c, np.linspace(200.0, 499.0, 500)).max() <= -40.0 + 1e-9

    @pytest.mark.parametrize("output", ["sos", "ba"])
    def test_ellip_ripple_and_floor(self, output):
        c = flt.ellip_design(4, 0.5, 50.0, 150.0, FS, output=output)
        assert _gain_db(c, 150.0)[0] == pytest.approx(-0.5, rel=1e-9, abs=0)
        # Even order: the Nyquist response sits on the stopband floor.
        assert _gain_db(c, 500.0)[0] == pytest.approx(-50.0, rel=1e-9, abs=0)
        passband = _gain_db(c, np.linspace(0.5, 150.0, 5001))
        assert passband.min() >= -0.5 - 1e-9 and passband.max() <= 1e-9

    @pytest.mark.parametrize("output", ["sos", "ba"])
    def test_bessel_phase_norm_midpoint(self, output):
        n = 4
        c = flt.bessel_design(n, 100.0, FS, output=output)
        w = np.linspace(1e-4, 2 * np.pi * 100.0 / FS, 4001)
        _, h = signal.freqz(c.b, c.a, worN=w)
        # scipy normalises the 'phase' Bessel with a fitted scale factor:
        # 1%, observed 1.03% at n=4, not rounding.
        assert np.unwrap(np.angle(h))[-1] == pytest.approx(
            -n * np.pi / 4, rel=0.015, abs=0
        )

    @pytest.mark.parametrize("output", ["sos", "ba"])
    def test_bessel_mag_norm_is_3_dB_at_cutoff(self, output):
        c = flt.bessel_design(4, 100.0, FS, norm="mag", output=output)
        assert _gain_db(c, 100.0)[0] == pytest.approx(
            10 * np.log10(0.5), rel=1e-9, abs=0
        )

    @pytest.mark.parametrize(
        "design, edge_db",
        [
            # cheby1 / ellip: the band edges are where the gain leaves the
            # ripple band; cheby2: the edges are the stopband edges; bessel
            # with norm="mag": the -3 dB points.
            (lambda: flt.cheby1_design(3, 1.0, (100.0, 200.0), FS, btype="band"), -1.0),
            (
                lambda: flt.cheby2_design(3, 30.0, (100.0, 200.0), FS, btype="band"),
                -30.0,
            ),
            (
                lambda: flt.ellip_design(
                    3, 1.0, 30.0, (100.0, 200.0), FS, btype="band"
                ),
                -1.0,
            ),
            (
                lambda: flt.bessel_design(
                    3, (100.0, 200.0), FS, btype="band", norm="mag"
                ),
                10 * np.log10(0.5),
            ),
        ],
    )
    def test_band_designs_hit_their_edge_gain_at_both_edges(self, design, edge_db):
        c = design()
        assert _gain_db(c, [100.0, 200.0]) == pytest.approx(
            [edge_db, edge_db], rel=1e-8, abs=0
        )


class TestFirDesign:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cutoff": 120.0},
            {"cutoff": 120.0, "pass_zero": False},
            {"cutoff": (100.0, 200.0), "pass_zero": False},
            {"cutoff": (100.0, 200.0), "pass_zero": True},
        ],
    )
    def test_symmetric_taps_give_exact_linear_phase(self, kwargs):
        m = 51
        h = flt.fir_design(m, fs=FS, **kwargs)
        assert h == pytest.approx(h[::-1], rel=1e-12, abs=0)
        w = np.linspace(0.05, 3.0, 200)
        amp = _h(h, [1.0], w) * np.exp(1j * w * (m - 1) / 2)
        assert np.max(np.abs(amp.imag)) < 1e-12

    def test_group_delay_is_half_the_length(self):
        h = flt.fir_design(51, 100.0, FS)
        f, gd = flt.group_delay(h, FS)
        _, resp = signal.freqz(h, [1.0], worN=2 * np.pi * f / FS)
        # group delay is ill-defined at the response zeros, so skip them
        keep = np.abs(resp) > 1e-3
        assert gd[keep] == pytest.approx(np.full(keep.sum(), 25.0), rel=1e-9, abs=0)

    def test_window_choice_changes_the_taps(self):
        a = flt.fir_design(31, 100.0, FS, window="hamming")
        b = flt.fir_design(31, 100.0, FS, window="blackman")
        assert np.abs(a - b).max() > 1e-4
        assert a.sum() == pytest.approx(1.0, rel=1e-12, abs=0)

    def test_remez_is_symmetric_and_meets_the_bands(self):
        h = flt.fir_design_remez(
            61, [0, 100, 150, 500], [1, 0], fs=FS, weight=[1.0, 10.0]
        )
        assert h == pytest.approx(h[::-1], rel=1e-12, abs=0)
        assert abs(_gain_db(h, 50.0)[0]) < 1.0
        assert _gain_db(h, 300.0)[0] < -30.0

    def test_remez_default_weight(self):
        h = flt.fir_design_remez(41, [0, 100, 150, 500], [1, 0], fs=FS)
        assert len(h) == 41
        assert h == pytest.approx(h[::-1], rel=1e-12, abs=0)
        assert abs(_gain_db(h, 50.0)[0]) < 1.5
        assert _gain_db(h, 300.0)[0] < -25.0


def _sine(n=1001, f=5.0):
    return np.sin(2 * np.pi * f * np.arange(n) / FS)


class TestApplicationAndAnalysis:
    @pytest.fixture(params=["sos", "ba", "tuple", "fir"])
    def lowpass(self, request):
        if request.param == "fir":
            return flt.fir_design(51, 200.0, FS), request.param
        c = flt.butter_design(
            4, 200.0, FS, output="ba" if request.param != "sos" else "sos"
        )
        return (c if request.param != "tuple" else (c.b, c.a)), request.param

    @staticmethod
    def _ba(coeffs):
        if isinstance(coeffs, flt.FilterCoefficients):
            return coeffs.b, coeffs.a
        if isinstance(coeffs, tuple):
            return coeffs
        return coeffs, np.array([1.0])

    def test_filtfilt_returns_passband_sinusoid_unchanged(self, lowpass):
        coeffs, kind = lowpass
        x = _sine()
        # Odd extension of a sine that starts and ends on a zero crossing is
        # the exact continuation; a pad of one period lets the start-up
        # transient die before the data begins.
        y = flt.filtfilt(coeffs, x, padlen=200)
        b, a = self._ba(coeffs)
        _, h = signal.freqz(b, a, worN=[2 * np.pi * 5.0 / FS])
        # Zero phase: the only change is the squared passband gain.
        want = np.abs(h[0]) ** 2 * x
        _within(y, want, 1e-10)
        if kind != "fir":
            _within(y, x, 1e-10)

    def test_filtfilt_padding_options(self):
        c = flt.butter_design(2, 200.0, FS, output="ba")
        x = _sine()
        y = flt.filtfilt(c, x, padtype="even", padlen=30)
        assert y.shape == x.shape
        _, h = signal.freqz(c.b, c.a, worN=[2 * np.pi * 5.0 / FS])
        _within(y[100:-100], (np.abs(h[0]) ** 2 * x)[100:-100], 1e-9)

    def test_filtfilt_default_padding_interior(self, lowpass):
        coeffs, kind = lowpass
        x = _sine()
        y = flt.filtfilt(coeffs, x)
        b, a = self._ba(coeffs)
        _, h = signal.freqz(b, a, worN=[2 * np.pi * 5.0 / FS])
        _within(y[100:-100], (np.abs(h[0]) ** 2 * x)[100:-100], 1e-9)

    def test_apply_filter_matches_difference_equation(self, lowpass):
        coeffs, _ = lowpass
        x = np.random.default_rng(3).standard_normal(200)
        b, a = self._ba(coeffs)
        b = np.atleast_1d(b).astype(float)
        a = np.atleast_1d(a).astype(float)
        if isinstance(coeffs, flt.FilterCoefficients) and coeffs.sos is not None:
            b, a = coeffs.b, coeffs.a
        y = np.zeros_like(x)
        for n in range(len(x)):
            acc = sum(b[k] * x[n - k] for k in range(len(b)) if n - k >= 0)
            acc -= sum(a[k] * y[n - k] for k in range(1, len(a)) if n - k >= 0)
            y[n] = acc / a[0]
        _within(flt.apply_filter(coeffs, x), y, 1e-9)

    def test_streaming_with_state_equals_one_shot(self, lowpass):
        coeffs, _ = lowpass
        x = np.random.default_rng(4).standard_normal(300)
        if isinstance(coeffs, flt.FilterCoefficients) and coeffs.sos is not None:
            zi = np.zeros((coeffs.sos.shape[0], 2))
        else:
            b, a = self._ba(coeffs)
            zi = np.zeros(max(len(np.atleast_1d(b)), len(np.atleast_1d(a))) - 1)
        y1, zf = flt.apply_filter(coeffs, x[:120], zi=zi)
        y2, _ = flt.apply_filter(coeffs, x[120:], zi=zf)
        one_shot = flt.apply_filter(coeffs, x)
        _within(np.concatenate([y1, y2]), one_shot, 1e-12)

    def test_frequency_response_matches_direct_evaluation(self, lowpass):
        coeffs, _ = lowpass
        r = flt.frequency_response(coeffs, FS, n_points=257)
        b, a = self._ba(coeffs)
        w = r.frequencies * 2 * np.pi / FS
        h = _h(np.atleast_1d(b), np.atleast_1d(a), w)
        _within(r.magnitude, np.abs(h), 1e-12)
        assert r.frequencies[0] == 0.0
        assert r.frequencies[-1] < FS / 2

    def test_frequency_response_whole_circle(self):
        r = flt.frequency_response(flt.butter_design(3, 100.0, FS), FS, 64, whole=True)
        assert r.frequencies[0] == 0.0 and r.frequencies[-1] > FS / 2
        assert r.magnitude[1:] == pytest.approx(r.magnitude[1:][::-1], rel=1e-9, abs=0)

    def test_group_delay_matches_phase_derivative_formula(self, lowpass):
        coeffs, _ = lowpass
        f, gd = flt.group_delay(coeffs, FS, n_points=128)
        b, a = self._ba(coeffs)
        b = np.atleast_1d(b)
        a = np.atleast_1d(a)
        w = f * 2 * np.pi / FS
        kb, ka = np.arange(len(b)), np.arange(len(a))
        eb = np.exp(-1j * np.outer(w, kb))
        ea = np.exp(-1j * np.outer(w, ka))
        want = (eb @ (kb * b)) / (eb @ b) - (ea @ (ka * a)) / (ea @ a)
        mask = np.abs(eb @ b) > 1e-6
        _within(gd[mask], want.real[mask], 1e-6)


class TestFilterOrder:
    wp, ws, rp, rs = 100.0, 160.0, 1.0, 40.0

    def _args(self):
        return _tan_half(self.wp), _tan_half(self.ws)

    def test_butter_closed_form(self):
        op, os_ = self._args()
        n = np.log10((10 ** (self.rs / 10) - 1) / (10 ** (self.rp / 10) - 1)) / (
            2 * np.log10(os_ / op)
        )
        assert flt.filter_order(self.wp, self.ws, self.rp, self.rs, FS) == int(
            np.ceil(n)
        )

    @pytest.mark.parametrize("kind", ["cheby1", "cheby2"])
    def test_chebyshev_closed_form(self, kind):
        op, os_ = self._args()
        n = np.arccosh(
            np.sqrt((10 ** (self.rs / 10) - 1) / (10 ** (self.rp / 10) - 1))
        ) / np.arccosh(os_ / op)
        assert flt.filter_order(self.wp, self.ws, self.rp, self.rs, FS, kind) == int(
            np.ceil(n)
        )

    def test_ellip_degree_equation(self):
        op, os_ = self._args()
        k = op / os_
        k1 = np.sqrt((10 ** (self.rp / 10) - 1) / (10 ** (self.rs / 10) - 1))
        n = (special.ellipk(k**2) * special.ellipk(1 - k1**2)) / (
            special.ellipk(1 - k**2) * special.ellipk(k1**2)
        )
        assert flt.filter_order(self.wp, self.ws, self.rp, self.rs, FS, "ellip") == int(
            np.ceil(n)
        )

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown filter type: bessel"):
            flt.filter_order(100, 160, 1, 40, FS, "bessel")


class TestSosZpk:
    @pytest.mark.parametrize("pairing", ["nearest", "keep_odd", "minimal"])
    def test_zpk_to_sos_preserves_the_transfer_function(self, pairing):
        z = np.array([-1.0, -1.0, -1.0])
        p = np.array([0.5, 0.6 * np.exp(1j * 0.8), 0.6 * np.exp(-1j * 0.8)])
        k = 0.3
        sos = flt.zpk_to_sos(z, p, k, pairing=pairing)
        w = np.linspace(0.1, 3.0, 40)
        zz = np.exp(1j * w)
        want = k * np.prod(zz[:, None] - z, axis=1) / np.prod(zz[:, None] - p, axis=1)
        got = np.ones_like(zz)
        for row in sos:
            got = got * _h(row[:3], row[3:], w)
        assert got == pytest.approx(want, rel=1e-10, abs=0)

    def test_sos_to_zpk_recovers_roots_and_gain(self):
        z = np.array([-1.0, -1.0])
        p = 0.7 * np.exp(1j * np.array([0.9, -0.9]))
        sos = flt.zpk_to_sos(z, p, 0.25)
        z2, p2, k2 = flt.sos_to_zpk(sos)
        assert len(p2) == len(p)
        for root in p:
            assert np.min(np.abs(p2 - root)) < 1e-12
        # a double zero at -1 splits to -1 +- O(sqrt(eps))
        assert np.max(np.abs(z2 - z)) < 1e-6
        assert k2 == pytest.approx(0.25, rel=1e-12, abs=0)
