"""
Correctness audit tests for signal_processing, transforms, statistics,
basic_matrix, and geometry subpackages.

Validates public functions against scipy/numpy/PyWavelets references and,
where conventions differ, against analytic properties (Monte Carlo false-alarm
rates, Parseval relations, quadrature of PDFs, geometric invariants).
"""

import numpy as np
import pytest
import scipy.linalg
import scipy.signal
import scipy.stats
from numpy.testing import assert_allclose
from scipy.integrate import quad

from pytcl.mathematical_functions.basic_matrix.decompositions import (
    matrix_sqrt,
    null_space,
    pinv_truncated,
    range_space,
    rank_revealing_qr,
    tria,
    tria_sqrt,
)
from pytcl.mathematical_functions.basic_matrix.special_matrices import (
    block_diag,
    circulant,
    commutation_matrix,
    companion,
    dft_matrix,
    duplication_matrix,
    elimination_matrix,
    hadamard,
    hankel,
    hilbert,
    invhilbert,
    kron,
    toeplitz,
    unvec,
    vandermonde,
    vec,
)
from pytcl.mathematical_functions.geometry.geometry import (
    barycentric_coordinates,
    bounding_box,
    convex_hull,
    convex_hull_area,
    delaunay_triangulation,
    line_intersection,
    line_plane_intersection,
    minimum_bounding_circle,
    oriented_bounding_box,
    point_in_polygon,
    point_to_line_distance,
    point_to_line_segment_distance,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
    triangle_area,
)
from pytcl.mathematical_functions.signal_processing.detection import (
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    cluster_detections,
    detection_probability,
    snr_loss,
    threshold_factor,
)
from pytcl.mathematical_functions.signal_processing.filters import (
    apply_filter,
    bessel_design,
    butter_design,
    cheby1_design,
    cheby2_design,
    ellip_design,
    filter_order,
    filtfilt,
    fir_design,
    fir_design_remez,
    frequency_response,
    group_delay,
    sos_to_zpk,
    zpk_to_sos,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    ambiguity_function,
    cross_ambiguity,
    generate_lfm_chirp,
    generate_nlfm_chirp,
    matched_filter,
    matched_filter_frequency,
    optimal_filter,
    pulse_compression,
)
from pytcl.mathematical_functions.statistics.distributions import (
    Beta,
    ChiSquared,
    Exponential,
    Gamma,
    Gaussian,
    MultivariateGaussian,
    Poisson,
    StudentT,
    Uniform,
    VonMises,
    Wishart,
)
from pytcl.mathematical_functions.statistics.estimators import (
    iqr,
    kurtosis,
    mad,
    median,
    moment,
    nees,
    nis,
    sample_corr,
    sample_cov,
    sample_mean,
    sample_var,
    skewness,
    weighted_cov,
    weighted_mean,
    weighted_var,
)
from pytcl.mathematical_functions.transforms.fourier import (
    coherence,
    cross_spectrum,
    fft,
    fft2,
    fftshift,
    frequency_axis,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    magnitude_spectrum,
    periodogram,
    phase_spectrum,
    power_spectrum,
    rfft,
    rfft_frequency_axis,
)
from pytcl.mathematical_functions.transforms.stft import (
    get_window,
    istft,
    mel_spectrogram,
    reassigned_spectrogram,
    spectrogram,
    stft,
    window_bandwidth,
)

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    pywt = None
    HAS_PYWT = False

from pytcl.mathematical_functions.transforms.wavelets import (
    cwt,
    dwt,
    dwt_single_level,
    frequencies_to_scales,
    gaussian_wavelet,
    idwt,
    idwt_single_level,
    morlet_wavelet,
    ricker_wavelet,
    scales_to_frequencies,
    threshold_coefficients,
    wpt,
)

# =============================================================================
# CFAR detection
# =============================================================================


class TestThresholdFactor:
    def test_ca_exact_closed_form(self):
        for pfa, n in [(1e-2, 16), (1e-4, 32), (1e-6, 64)]:
            alpha = threshold_factor(pfa, n, method="ca")
            assert_allclose((1 + alpha / n) ** (-n), pfa, rtol=1e-10)

    def test_os_exact_rohling(self):
        # Pfa = prod_{i=0}^{k-1} (N-i)/(N-i+alpha)
        pfa, n, k = 1e-3, 24, 18
        alpha = threshold_factor(pfa, n, method="os", k=k)
        p = 1.0
        for i in range(k):
            p *= (n - i) / (n - i + alpha)
        assert_allclose(p, pfa, rtol=1e-8)

    def test_go_so_alpha_ordering(self):
        # For same pfa, SO needs a larger multiplier than CA-per-window,
        # GO a smaller one (max estimate is already conservative).
        pfa, n = 1e-3, 16
        a_go = threshold_factor(pfa, n, method="go")
        a_so = threshold_factor(pfa, n, method="so")
        assert a_so > a_go > 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            threshold_factor(0.0, 16)
        with pytest.raises(ValueError):
            threshold_factor(1e-3, 0)
        with pytest.raises(ValueError):
            threshold_factor(1e-3, 16, method="bogus")


class TestCFARFalseAlarmRates:
    """Monte Carlo verification that delivered Pfa matches design Pfa."""

    PFA = 1e-2
    GUARD = 2
    REF = 8

    def _mc_pfa(self, fn, **kw):
        rng = np.random.default_rng(12345)
        g, r = self.GUARD, self.REF
        fa = 0
        cells = 0
        for _ in range(60):
            sig = rng.exponential(1.0, 3000)
            res = fn(sig, guard_cells=g, ref_cells=r, pfa=self.PFA, **kw)
            interior = np.zeros(3000, dtype=bool)
            interior[g + r : 3000 - g - r] = True
            fa += np.count_nonzero(res.detections & interior)
            cells += np.count_nonzero(interior)
        return fa / cells

    def test_cfar_ca_pfa(self):
        assert self._mc_pfa(cfar_ca) == pytest.approx(self.PFA, rel=0.2)

    def test_cfar_go_pfa(self):
        assert self._mc_pfa(cfar_go) == pytest.approx(self.PFA, rel=0.2)

    def test_cfar_so_pfa(self):
        assert self._mc_pfa(cfar_so) == pytest.approx(self.PFA, rel=0.2)

    def test_cfar_os_pfa(self):
        assert self._mc_pfa(cfar_os) == pytest.approx(self.PFA, rel=0.2)


class TestCFARDetection:
    def test_strong_target_detected_all_variants(self):
        rng = np.random.default_rng(7)
        sig = rng.exponential(1.0, 1000)
        sig[500] = 200.0
        for fn in (cfar_ca, cfar_go, cfar_so, cfar_os):
            res = fn(sig, guard_cells=2, ref_cells=16, pfa=1e-4)
            assert 500 in res.detection_indices, fn.__name__

    def test_cfar_os_robust_to_interferer(self):
        # An interferer inside the reference window masks CA but not OS.
        sig = np.ones(200)
        sig[100] = 60.0
        sig[105] = 55.0  # interferer within reference cells of index 100
        res_os = cfar_os(sig, guard_cells=2, ref_cells=8, pfa=1e-3)
        assert 100 in res_os.detection_indices
        assert 105 in res_os.detection_indices

    def test_cfar_ca_noise_estimate_constant_signal(self):
        sig = np.full(100, 3.0)
        res = cfar_ca(sig, guard_cells=1, ref_cells=4, pfa=1e-2)
        assert_allclose(res.noise_estimate, 3.0)

    def test_cfar_2d_target_and_threshold(self):
        rng = np.random.default_rng(11)
        img = rng.exponential(1.0, (64, 64))
        img[32, 32] = 500.0
        for method in ("ca", "go", "so"):
            res = cfar_2d(
                img, guard_cells=(2, 2), ref_cells=(4, 4), pfa=1e-4, method=method
            )
            assert res.detections[32, 32], method
            assert res.threshold.shape == img.shape

    def test_cfar_2d_ca_pfa_monte_carlo(self):
        rng = np.random.default_rng(3)
        fa = 0
        cells = 0
        for _ in range(6):
            img = rng.exponential(1.0, (120, 120))
            res = cfar_2d(img, guard_cells=(1, 1), ref_cells=(3, 3), pfa=1e-2)
            interior = res.detections[8:-8, 8:-8]
            fa += np.count_nonzero(interior)
            cells += interior.size
        assert fa / cells == pytest.approx(1e-2, rel=0.3)


class TestDetectionProbability:
    def test_matches_swerling1_monte_carlo(self):
        # The implemented CA-CFAR Pd formula is exact for Swerling 1 targets.
        rng = np.random.default_rng(42)
        snr, pfa, n = 10.0, 1e-2, 16
        alpha = threshold_factor(pfa, n, method="ca")
        trials = 200_000
        z = rng.gamma(n, 1.0 / n, trials)
        cut = rng.exponential(1 + snr, trials)
        pd_mc = np.mean(cut > alpha * z)
        pd = detection_probability(snr, pfa, n)
        assert pd == pytest.approx(pd_mc, abs=0.01)

    def test_monotonic_in_snr(self):
        pds = [detection_probability(s, 1e-4, 32) for s in (1.0, 5.0, 20.0)]
        assert pds[0] < pds[1] < pds[2]


class TestCFARUtilities:
    def test_cluster_detections(self):
        det = np.zeros(100, dtype=bool)
        det[20:24] = True
        det[60] = True
        det[62] = True
        peaks = cluster_detections(det, min_separation=2)
        assert list(peaks) == [21, 61]

    def test_cluster_detections_empty(self):
        assert len(cluster_detections(np.zeros(10, dtype=bool))) == 0

    def test_snr_loss_positive_and_decreasing(self):
        # Only CA has a derived expression; the other three raise (gh-20).
        assert snr_loss(16, pfa=1e-6) > snr_loss(64, pfa=1e-6) > 0


# =============================================================================
# Filters
# =============================================================================


class TestIIRDesign:
    FS = 1000.0

    def test_butter_matches_scipy(self):
        c = butter_design(4, 100, self.FS)
        sos_ref = scipy.signal.butter(4, 100 / 500, btype="low", output="sos")
        assert_allclose(c.sos, sos_ref)

    def test_butter_ba_matches_scipy(self):
        c = butter_design(4, 100, self.FS, output="ba")
        b_ref, a_ref = scipy.signal.butter(4, 100 / 500, btype="low", output="ba")
        assert_allclose(c.b, b_ref)
        assert_allclose(c.a, a_ref)
        assert c.sos is None

    def test_butter_minus_3db_at_cutoff(self):
        c = butter_design(4, 100, self.FS)
        w, h = scipy.signal.sosfreqz(c.sos, worN=[2 * np.pi * 100 / self.FS])
        assert_allclose(np.abs(h[0]), 1 / np.sqrt(2), rtol=1e-6)

    def test_cheby1_matches_scipy(self):
        c = cheby1_design(4, 0.5, 100, self.FS)
        assert_allclose(c.sos, scipy.signal.cheby1(4, 0.5, 100 / 500, output="sos"))

    def test_cheby2_matches_scipy(self):
        c = cheby2_design(4, 40, 100, self.FS)
        assert_allclose(c.sos, scipy.signal.cheby2(4, 40, 100 / 500, output="sos"))

    def test_ellip_matches_scipy(self):
        c = ellip_design(4, 0.5, 40, 100, self.FS)
        assert_allclose(c.sos, scipy.signal.ellip(4, 0.5, 40, 100 / 500, output="sos"))

    def test_bessel_matches_scipy(self):
        c = bessel_design(4, 100, self.FS)
        assert_allclose(
            c.sos, scipy.signal.bessel(4, 100 / 500, norm="phase", output="sos")
        )

    def test_bandpass_cutoffs(self):
        c = butter_design(4, (50, 150), self.FS, btype="band")
        sos_ref = scipy.signal.butter(
            4, [50 / 500, 150 / 500], btype="band", output="sos"
        )
        assert_allclose(c.sos, sos_ref)


class TestFIRDesign:
    def test_fir_matches_scipy_firwin(self):
        h = fir_design(101, 100, 1000)
        assert_allclose(h, scipy.signal.firwin(101, 100 / 500, window="hamming"))

    def test_fir_dc_gain_unity(self):
        h = fir_design(101, 100, 1000)
        assert_allclose(np.sum(h), 1.0, rtol=1e-6)

    def test_remez_matches_scipy(self):
        h = fir_design_remez(101, [0, 100, 150, 500], [1, 0], 1000)
        h_ref = scipy.signal.remez(101, [0, 100, 150, 500], [1, 0], fs=1000)
        assert_allclose(h, h_ref)


class TestFilterApplication:
    FS = 1000.0

    def test_apply_filter_sos_matches_scipy(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        c = butter_design(4, 100, self.FS)
        assert_allclose(apply_filter(c, x), scipy.signal.sosfilt(c.sos, x))

    def test_apply_filter_ba_tuple(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        b, a = scipy.signal.butter(2, 0.2)
        assert_allclose(apply_filter((b, a), x), scipy.signal.lfilter(b, a, x))

    def test_apply_filter_fir_array(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        h = fir_design(21, 100, self.FS)
        assert_allclose(apply_filter(h, x), scipy.signal.lfilter(h, [1.0], x))

    def test_filtfilt_zero_phase(self):
        t = np.arange(0, 1, 1 / self.FS)
        x = np.sin(2 * np.pi * 10 * t)
        c = butter_design(4, 100, self.FS)
        y = filtfilt(c, x)
        # Zero-phase: 10 Hz passband tone comes through with no delay
        assert_allclose(y[100:-100], x[100:-100], atol=1e-3)

    def test_filtfilt_matches_scipy(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(400)
        c = butter_design(4, 100, self.FS)
        assert_allclose(filtfilt(c, x), scipy.signal.sosfiltfilt(c.sos, x))


class TestFrequencyAnalysis:
    FS = 1000.0

    def test_frequency_response_matches_sosfreqz(self):
        c = butter_design(4, 100, self.FS)
        resp = frequency_response(c, self.FS, n_points=256)
        w, h = scipy.signal.sosfreqz(c.sos, worN=256)
        assert_allclose(resp.frequencies, w * self.FS / (2 * np.pi))
        assert_allclose(resp.magnitude, np.abs(h))
        assert_allclose(resp.phase, np.angle(h))

    def test_frequency_response_dc_gain(self):
        c = butter_design(4, 100, self.FS)
        resp = frequency_response(c, self.FS)
        assert_allclose(resp.magnitude[0], 1.0, rtol=1e-8)

    def test_group_delay_symmetric_fir(self):
        h = fir_design(51, 100, self.FS)
        freqs, gd = group_delay(h, self.FS)
        assert_allclose(gd, 25.0, atol=1e-6)

    def test_filter_order_matches_scipy(self):
        order = filter_order(100, 150, 0.5, 40, 1000, "butter")
        ref, _ = scipy.signal.buttord(100 / 500, 150 / 500, 0.5, 40)
        assert order == ref

    def test_sos_zpk_roundtrip(self):
        c = butter_design(4, 100, self.FS)
        z, p, k = sos_to_zpk(c.sos)
        sos2 = zpk_to_sos(z, p, k)
        z2, p2, k2 = sos_to_zpk(sos2)
        assert_allclose(sorted(np.abs(p)), sorted(np.abs(p2)), rtol=1e-8)
        assert_allclose(k, k2, rtol=1e-8)
        assert np.all(np.abs(p) < 1.0)


# =============================================================================
# Matched filtering
# =============================================================================


class TestMatchedFilter:
    def test_matches_scipy_correlate(self):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal(200)
        tpl = rng.standard_normal(16)
        res = matched_filter(sig, tpl, normalize=False)
        assert_allclose(res.output, scipy.signal.correlate(sig, tpl, mode="same"))

    def test_peak_at_target_and_unit_value(self):
        tpl = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        sig = np.zeros(100)
        sig[50:55] = tpl
        res = matched_filter(sig, tpl)
        assert 50 <= res.peak_index <= 54
        assert res.peak_value == pytest.approx(1.0)

    def test_frequency_domain_equals_linear_correlation(self):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(300)
        tpl = rng.standard_normal(32)
        res = matched_filter_frequency(sig, tpl, normalize=False)
        full = scipy.signal.correlate(sig, tpl, mode="full")
        assert_allclose(res.output, full[len(tpl) - 1 :], atol=1e-9)

    def test_frequency_domain_peak_location(self):
        tpl = np.sin(2 * np.pi * 0.1 * np.arange(50))
        sig = np.zeros(200)
        sig[100:150] = tpl
        res = matched_filter_frequency(sig, tpl)
        assert res.peak_index == 100

    def test_optimal_filter_white_noise_peak(self):
        tpl = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        sig = np.zeros(256)
        sig[100:105] = tpl
        out = optimal_filter(sig, tpl, np.ones(256))
        assert np.argmax(np.abs(out)) == 100


class TestPulseCompression:
    FS = 1000.0

    def test_peak_and_compression_ratio(self):
        chirp = generate_lfm_chirp(0.2, 50, 450, self.FS)
        sig = np.zeros(2000)
        sig[500 : 500 + len(chirp)] = chirp
        res = pulse_compression(sig, chirp)
        assert res.peak_index == 500
        # Compression ratio at least on the order of the
        # time-bandwidth product (80); bounded by pulse length (200)
        assert 20 < res.compression_ratio <= len(chirp)

    def test_windowing_reduces_sidelobes(self):
        chirp = generate_lfm_chirp(0.2, 50, 450, self.FS)
        sig = np.zeros(2000)
        sig[500 : 500 + len(chirp)] = chirp
        pslr_none = pulse_compression(sig, chirp).peak_sidelobe_ratio
        pslr_hamming = pulse_compression(sig, chirp, window="hamming")
        assert pslr_hamming.peak_sidelobe_ratio > pslr_none


class TestChirpGeneration:
    def test_lfm_length_and_start(self):
        c = generate_lfm_chirp(0.001, 1000, 5000, 44100)
        assert len(c) == 44
        assert c[0] == pytest.approx(1.0)

    def test_lfm_instantaneous_frequency(self):
        fs = 100000.0
        c = generate_lfm_chirp(0.01, 1000, 5000, fs)
        analytic = scipy.signal.hilbert(c)
        inst_freq = np.diff(np.unwrap(np.angle(analytic))) * fs / (2 * np.pi)
        n = len(inst_freq)
        assert inst_freq[n // 10] == pytest.approx(1000 + 4000 * 0.1, rel=0.05)
        assert inst_freq[9 * n // 10] == pytest.approx(1000 + 4000 * 0.9, rel=0.05)

    def test_nlfm_length_and_band(self):
        fs = 44100.0
        c = generate_nlfm_chirp(0.005, 1000, 5000, fs, beta=2.0)
        assert len(c) == 220
        assert np.max(np.abs(c)) <= 1.0 + 1e-12
        spec = np.abs(np.fft.rfft(c))
        freqs = np.fft.rfftfreq(len(c), 1 / fs)
        band_energy = np.sum(spec[(freqs >= 800) & (freqs <= 5500)] ** 2)
        assert band_energy / np.sum(spec**2) > 0.9


class TestAmbiguityFunction:
    def test_peak_at_origin(self):
        chirp = generate_lfm_chirp(0.005, 500, 2000, 8000)
        delays, dopplers, af = ambiguity_function(
            chirp, 8000, n_delay=33, n_doppler=33, max_doppler=1000
        )
        assert af.shape == (33, 33)
        i, j = np.unravel_index(np.argmax(af), af.shape)
        # Peak at zero delay / zero Doppler (center bins)
        assert abs(int(j) - 16) <= 1
        assert abs(int(i) - 16) <= 1
        assert np.max(af) == pytest.approx(1.0)

    def test_cross_ambiguity_identical_signals(self):
        chirp = generate_lfm_chirp(0.002, 500, 1500, 8000)
        delays, dopplers, caf = cross_ambiguity(
            chirp, chirp, 8000, n_delay=17, n_doppler=17, max_doppler=500
        )
        assert caf.shape == (17, 17)
        assert np.max(caf) == pytest.approx(1.0)
        i, j = np.unravel_index(np.argmax(caf), caf.shape)
        assert abs(int(j) - 8) <= 1 and abs(int(i) - 8) <= 1


# =============================================================================
# Fourier transforms and spectral analysis
# =============================================================================


class TestFFTWrappers:
    def test_fft_ifft_match_numpy(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        assert_allclose(fft(x), np.fft.fft(x))
        assert_allclose(ifft(fft(x)), x, atol=1e-12)

    def test_rfft_irfft_roundtrip(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(64)
        assert_allclose(rfft(x), np.fft.rfft(x))
        assert_allclose(irfft(rfft(x)), x, atol=1e-12)

    def test_fft2_ifft2(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((8, 8))
        assert_allclose(fft2(x), np.fft.fft2(x))
        assert_allclose(ifft2(fft2(x)).real, x, atol=1e-12)

    def test_parseval(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(128)
        X = fft(x)
        assert_allclose(np.sum(np.abs(x) ** 2), np.sum(np.abs(X) ** 2) / 128)

    def test_fftshift_roundtrip(self):
        x = np.arange(10.0)
        assert_allclose(ifftshift(fftshift(x)), x)

    def test_frequency_axes(self):
        assert_allclose(frequency_axis(8, 100.0), np.fft.fftfreq(8, 0.01))
        assert_allclose(
            frequency_axis(8, 100.0, shift=True),
            np.fft.fftshift(np.fft.fftfreq(8, 0.01)),
        )
        assert_allclose(rfft_frequency_axis(8, 100.0), np.fft.rfftfreq(8, 0.01))


class TestSpectralEstimation:
    FS = 1000.0

    def test_power_spectrum_peak_and_power(self):
        t = np.arange(0, 4, 1 / self.FS)
        x = np.sin(2 * np.pi * 100 * t)
        res = power_spectrum(x, fs=self.FS, nperseg=512)
        peak = res.frequencies[np.argmax(res.psd)]
        assert peak == pytest.approx(100, abs=2)
        df = res.frequencies[1] - res.frequencies[0]
        assert np.sum(res.psd) * df == pytest.approx(0.5, rel=0.05)

    def test_power_spectrum_matches_welch(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(2048)
        res = power_spectrum(x, fs=self.FS, nperseg=256)
        f_ref, p_ref = scipy.signal.welch(x, fs=self.FS, nperseg=256)
        assert_allclose(res.frequencies, f_ref)
        assert_allclose(res.psd, p_ref)

    def test_cross_spectrum_matches_csd(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(2048)
        y = rng.standard_normal(2048)
        res = cross_spectrum(x, y, fs=self.FS, nperseg=256)
        f_ref, p_ref = scipy.signal.csd(x, y, fs=self.FS, nperseg=256)
        assert_allclose(res.frequencies, f_ref)
        assert_allclose(res.csd, p_ref)

    def test_coherence_of_linearly_related_signals(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4096)
        y = 2 * x
        res = coherence(x, y, fs=self.FS, nperseg=256)
        assert np.all(res.coherence > 0.99)

    def test_periodogram_matches_scipy_and_parseval(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(1024)
        res = periodogram(x, fs=self.FS)
        f_ref, p_ref = scipy.signal.periodogram(x, fs=self.FS)
        assert_allclose(res.frequencies, f_ref)
        assert_allclose(res.psd, p_ref)
        df = res.frequencies[1] - res.frequencies[0]
        assert np.sum(res.psd) * df == pytest.approx(np.var(x), rel=1e-6)

    def test_magnitude_and_phase_spectrum(self):
        X = np.array([4 + 0j, 0 - 2j, 0 + 0j, 0 + 2j])
        assert_allclose(magnitude_spectrum(X), [4, 2, 0, 2])
        db = magnitude_spectrum(X, scale="dB")
        assert_allclose(db[0], 20 * np.log10(4))
        assert_allclose(phase_spectrum(np.array([1 + 0j, 0 + 1j])), [0, np.pi / 2])
        with pytest.raises(ValueError):
            magnitude_spectrum(X, scale="bogus")


# =============================================================================
# STFT / spectrogram
# =============================================================================


class TestSTFT:
    FS = 1000.0

    def test_stft_matches_scipy(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal(1000)
        res = stft(x, fs=self.FS, nperseg=128)
        f_ref, t_ref, z_ref = scipy.signal.stft(x, fs=self.FS, nperseg=128)
        assert_allclose(res.frequencies, f_ref)
        assert_allclose(res.times, t_ref)
        assert_allclose(res.Zxx, z_ref)

    def test_istft_roundtrip(self):
        t = np.arange(0, 1, 1 / self.FS)
        x = np.sin(2 * np.pi * 50 * t)
        res = stft(x, fs=self.FS, nperseg=128)
        _, x_rec = istft(res.Zxx, fs=self.FS, nperseg=128)
        assert_allclose(x, x_rec[: len(x)], atol=1e-10)

    def test_stft_tone_frequency(self):
        t = np.arange(0, 1, 1 / self.FS)
        x = np.sin(2 * np.pi * 125 * t)
        res = stft(x, fs=self.FS, nperseg=128)
        mean_mag = np.mean(np.abs(res.Zxx), axis=1)
        assert res.frequencies[np.argmax(mean_mag)] == pytest.approx(125, abs=4)

    def test_spectrogram_matches_scipy(self):
        rng = np.random.default_rng(14)
        x = rng.standard_normal(2000)
        res = spectrogram(x, fs=self.FS, nperseg=256)
        # Note: pytcl defaults to a hann window (scipy defaults to tukey)
        f_ref, t_ref, s_ref = scipy.signal.spectrogram(
            x, fs=self.FS, window="hann", nperseg=256
        )
        assert_allclose(res.frequencies, f_ref)
        assert_allclose(res.times, t_ref)
        assert_allclose(res.power, s_ref)

    def test_get_window_matches_scipy(self):
        assert_allclose(get_window("hann", 64), scipy.signal.get_window("hann", 64))
        assert_allclose(
            get_window(("kaiser", 8.0), 64),
            scipy.signal.get_window(("kaiser", 8.0), 64),
        )

    def test_window_bandwidth_known_values(self):
        assert window_bandwidth("boxcar", 256) == pytest.approx(1.0)
        assert window_bandwidth("hann", 4096) == pytest.approx(1.5, rel=1e-3)

    def test_reassigned_spectrogram_structural(self):
        # NOTE: current implementation computes but does not apply the
        # reassignment corrections; output equals the plain spectrogram.
        t = np.arange(0, 0.5, 1 / self.FS)
        x = np.sin(2 * np.pi * 100 * t)
        f, times, power = reassigned_spectrogram(x, fs=self.FS, nperseg=64)
        assert power.shape == (len(f), len(times))
        assert np.all(power >= 0)
        assert f[np.argmax(np.mean(power, axis=1))] == pytest.approx(100, abs=10)

    def test_mel_spectrogram_tone_band(self):
        fs = 8000.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 1000 * t)
        mel_freqs, times, mel_spec = mel_spectrogram(x, fs, n_mels=40, nperseg=512)
        assert mel_spec.shape[0] == 40
        peak_band_freq = mel_freqs[np.argmax(np.mean(mel_spec, axis=1))]
        assert peak_band_freq == pytest.approx(1000, rel=0.15)


# =============================================================================
# Wavelets
# =============================================================================


class TestWaveletGenerators:
    def test_morlet_unit_energy_and_zero_mean(self):
        w = morlet_wavelet(256, w=5.0)
        assert np.sum(np.abs(w) ** 2) == pytest.approx(1.0)
        # Zero mean requires adequate sampling of the oscillation (s > w/pi)
        w_dilated = morlet_wavelet(256, w=5.0, s=8.0)
        assert abs(np.sum(w_dilated)) < 1e-3

    def test_ricker_analytic_peak(self):
        a = 4.0
        w = ricker_wavelet(129, a=a)
        assert w[64] == pytest.approx(2 / (np.sqrt(3 * a) * np.pi**0.25))
        assert np.sum(w) == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_gaussian_wavelet_is_gaussian_derivative(self, order):
        M, sigma = 401, 30.0
        xs = (np.arange(M) - (M - 1) / 2.0) / sigma
        d = np.exp(-0.5 * xs**2)
        for _ in range(order):
            d = np.gradient(d, xs)
        d /= np.sqrt(np.sum(d**2))
        w = gaussian_wavelet(M, order=order, sigma=sigma)
        assert abs(np.dot(d, w)) > 0.999


class TestCWT:
    def test_ridge_scale_matches_frequency_mapping_morlet(self):
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        scales = np.arange(2, 64, dtype=float)
        res = cwt(x, scales, wavelet="morlet", fs=fs)
        power = np.mean(np.abs(res.coefficients) ** 2, axis=1)
        ridge_scale = scales[np.argmax(power)]
        expected = 5.0 / (2 * np.pi) * fs / 50.0  # ~15.9
        assert abs(ridge_scale - expected) <= 2.0
        assert res.frequencies[np.argmax(power)] == pytest.approx(50, rel=0.15)

    def test_ridge_scale_matches_frequency_mapping_ricker(self):
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        scales = np.arange(1, 20, dtype=float)
        res = cwt(x, scales, wavelet="ricker", fs=fs)
        power = np.mean(np.abs(res.coefficients) ** 2, axis=1)
        expected = fs / (np.sqrt(2) * np.pi * 50.0)  # ~4.5
        assert abs(scales[np.argmax(power)] - expected) <= 2.0

    def test_cwt_conv_and_fft_methods_agree(self):
        fs = 500.0
        t = np.arange(0, 0.5, 1 / fs)
        x = np.sin(2 * np.pi * 25 * t)
        scales = np.array([4.0, 8.0, 16.0])
        r_fft = cwt(x, scales, wavelet="morlet", fs=fs, method="fft")
        r_conv = cwt(x, scales, wavelet="morlet", fs=fs, method="conv")
        assert_allclose(
            np.abs(r_fft.coefficients), np.abs(r_conv.coefficients), atol=1e-8
        )

    def test_scale_frequency_roundtrip_all_wavelets(self):
        s = np.array([2.0, 4.0, 8.0])
        for name in ("morlet", "ricker", "gaussian1", "gaussian2"):
            f = scales_to_frequencies(s, name, fs=100.0)
            s2 = frequencies_to_scales(f, name, fs=100.0)
            assert_allclose(s2, s, rtol=1e-12, err_msg=name)


@pytest.mark.skipif(not HAS_PYWT, reason="pywavelets not installed")
class TestDWT:
    def test_dwt_matches_pywt(self):
        rng = np.random.default_rng(15)
        x = rng.standard_normal(256)
        res = dwt(x, wavelet="db4", level=4)
        coeffs = pywt.wavedec(x, "db4", mode="symmetric", level=4)
        assert_allclose(res.cA, coeffs[0])
        # res.cD is finest-to-coarsest; pywt is coarsest-to-finest
        for lvl, c_ref in enumerate(coeffs[1:][::-1]):
            assert_allclose(res.cD[lvl], c_ref)

    def test_dwt_idwt_roundtrip(self):
        rng = np.random.default_rng(16)
        x = rng.standard_normal(300)
        rec = idwt(dwt(x, wavelet="sym5", level=3))
        assert_allclose(rec[: len(x)], x, atol=1e-10)

    def test_single_level_matches_pywt(self):
        rng = np.random.default_rng(17)
        x = rng.standard_normal(128)
        cA, cD = dwt_single_level(x, wavelet="haar")
        cA_ref, cD_ref = pywt.dwt(x, "haar")
        assert_allclose(cA, cA_ref)
        assert_allclose(cD, cD_ref)
        assert_allclose(idwt_single_level(cA, cD, wavelet="haar"), x, atol=1e-12)

    def test_wpt_matches_pywt(self):
        rng = np.random.default_rng(18)
        x = rng.standard_normal(128)
        nodes = wpt(x, wavelet="db2", level=2)
        wp = pywt.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
        for node in wp.get_level(2, "natural"):
            assert_allclose(nodes[node.path], node.data)

    def test_threshold_coefficients_denoises(self):
        rng = np.random.default_rng(19)
        t = np.linspace(0, 1, 512)
        clean = np.sin(2 * np.pi * 5 * t)
        noisy = clean + 0.4 * rng.standard_normal(512)
        coeffs = dwt(noisy, wavelet="db4", level=4)
        rec = idwt(threshold_coefficients(coeffs, threshold="soft"))[: len(clean)]
        err_denoised = np.mean((rec - clean) ** 2)
        err_noisy = np.mean((noisy - clean) ** 2)
        assert err_denoised < err_noisy


# =============================================================================
# Distributions
# =============================================================================


class TestDistributions:
    def _check_univariate(self, d, support, mean, var, points):
        integral, _ = quad(lambda x: float(d.pdf(x)), *support, limit=200)
        assert integral == pytest.approx(1.0, abs=1e-6)
        assert_allclose(d.mean(), mean, rtol=1e-12)
        assert_allclose(d.var(), var, rtol=1e-12)
        q = np.array([0.1, 0.5, 0.9])
        assert_allclose(d.cdf(d.ppf(q)), q, atol=1e-9)
        assert_allclose(np.exp(d.logpdf(points)), d.pdf(points), rtol=1e-10)

    def test_gaussian(self):
        self._check_univariate(
            Gaussian(2.0, 4.0), (-30, 30), 2.0, 4.0, np.array([0.0, 2.0, 5.0])
        )
        assert Gaussian(0, 1).pdf(0) == pytest.approx(1 / np.sqrt(2 * np.pi))

    def test_uniform(self):
        self._check_univariate(
            Uniform(1.0, 3.0), (0.9, 3.1), 2.0, 4.0 / 12, np.array([1.5, 2.5])
        )

    def test_exponential(self):
        self._check_univariate(
            Exponential(2.0), (0, 40), 0.5, 0.25, np.array([0.1, 1.0])
        )

    def test_gamma_rate_and_scale(self):
        self._check_univariate(
            Gamma(3.0, rate=2.0), (0, 60), 1.5, 0.75, np.array([0.5, 2.0])
        )
        g2 = Gamma(3.0, scale=0.5)
        assert g2.mean() == pytest.approx(1.5)
        with pytest.raises(ValueError):
            Gamma(3.0, rate=1.0, scale=1.0)

    def test_chi_squared(self):
        self._check_univariate(ChiSquared(5), (0, 100), 5.0, 10.0, np.array([1.0, 5.0]))

    def test_student_t(self):
        d = StudentT(5.0, loc=1.0, scale=2.0)
        self._check_univariate(d, (-200, 200), 1.0, 4.0 * 5 / 3, np.array([0.0, 1.0]))
        assert np.isinf(StudentT(1.5).var())

    def test_beta(self):
        self._check_univariate(
            Beta(2.0, 3.0), (0, 1), 0.4, 6.0 / (25 * 6), np.array([0.2, 0.7])
        )

    def test_poisson_pmf(self):
        d = Poisson(3.0)
        k = np.arange(0, 60)
        assert np.sum(d.pdf(k)) == pytest.approx(1.0, abs=1e-12)
        assert d.mean() == 3.0 and d.var() == 3.0
        assert d.pdf(2) == pytest.approx(scipy.stats.poisson(3.0).pmf(2))

    def test_von_mises(self):
        d = VonMises(mu=0.5, kappa=2.0)
        integral, _ = quad(lambda x: float(d.pdf(x)), 0.5 - np.pi, 0.5 + np.pi)
        assert integral == pytest.approx(1.0, abs=1e-8)
        from scipy.special import i0, i1

        assert d.var() == pytest.approx(1 - i1(2.0) / i0(2.0))

    def test_multivariate_gaussian(self):
        mean = np.array([1.0, -1.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        d = MultivariateGaussian(mean, cov)
        ref = scipy.stats.multivariate_normal(mean, cov)
        pts = np.array([[0.0, 0.0], [1.0, -1.0]])
        assert_allclose(d.pdf(pts), ref.pdf(pts))
        assert_allclose(d.mean(), mean)
        assert_allclose(d.cov(), cov)
        assert_allclose(d.var(), np.diag(cov))
        # Mahalanobis distance reference
        diff = pts[0] - mean
        expected = np.sqrt(diff @ np.linalg.inv(cov) @ diff)
        assert d.mahalanobis(pts[0]) == pytest.approx(expected)

    def test_sampling_moments(self):
        np.random.seed(0)
        s = Gaussian(2.0, 9.0).sample(200_000)
        assert np.mean(s) == pytest.approx(2.0, abs=0.05)
        assert np.var(s) == pytest.approx(9.0, rel=0.05)

    def test_wishart_mean(self):
        scale = np.array([[1.0, 0.3], [0.3, 2.0]])
        d = Wishart(df=5.0, scale=scale)
        assert_allclose(d.mean(), 5.0 * scale)
        np.random.seed(1)
        samples = d.sample(20_000)
        assert_allclose(np.mean(samples, axis=0), 5.0 * scale, rtol=0.05)


# =============================================================================
# Estimators
# =============================================================================


class TestEstimators:
    def test_weighted_mean(self):
        assert weighted_mean([1, 2, 3], [1, 1, 2]) == pytest.approx(2.25)
        x = np.arange(10.0)
        assert weighted_mean(x, np.ones(10)) == pytest.approx(np.mean(x))

    def test_weighted_var_equal_weights(self):
        rng = np.random.default_rng(20)
        x = rng.standard_normal(50)
        w = np.ones(50)
        assert weighted_var(x, w) == pytest.approx(np.var(x))
        assert weighted_var(x, w, ddof=1) == pytest.approx(np.var(x, ddof=1))

    def test_weighted_var_frequency_weights(self):
        # Integer weights should equal variance of the expanded sample
        x = np.array([1.0, 2.0, 5.0])
        w = np.array([2.0, 3.0, 1.0])
        expanded = np.repeat(x, [2, 3, 1])
        assert weighted_var(x, w) == pytest.approx(np.var(expanded))

    def test_weighted_cov_equal_weights(self):
        rng = np.random.default_rng(21)
        x = rng.standard_normal((60, 3))
        assert_allclose(weighted_cov(x, np.ones(60)), np.cov(x.T, ddof=0))
        assert_allclose(weighted_cov(x, np.ones(60), ddof=1), np.cov(x.T, ddof=1))

    def test_sample_statistics(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert sample_mean(x) == 3.0
        assert sample_var(x) == 2.5
        assert median(x) == 3.0
        assert sample_cov(x) == pytest.approx(np.var(x, ddof=1))
        data = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 6.0]])
        assert_allclose(sample_cov(data), np.cov(data.T, ddof=1))
        assert_allclose(sample_corr(data), np.corrcoef(data.T))

    def test_mad_iqr(self):
        assert mad([1, 2, 3, 4, 5]) == pytest.approx(1.4826)
        assert mad([1, 2, 3, 4, 5], scale=1.0) == pytest.approx(1.0)
        assert iqr(np.arange(1, 10)) == pytest.approx(4.0)

    def test_mad_estimates_normal_std(self):
        rng = np.random.default_rng(22)
        x = rng.normal(0, 3.0, 100_000)
        assert mad(x) == pytest.approx(3.0, rel=0.02)

    def test_skewness_kurtosis_moment(self):
        rng = np.random.default_rng(23)
        x = rng.standard_normal(100)
        assert skewness(x) == pytest.approx(scipy.stats.skew(x))
        assert kurtosis(x) == pytest.approx(scipy.stats.kurtosis(x))
        assert moment(x, 3) == pytest.approx(scipy.stats.moment(x, moment=3))
        assert moment([1, 2, 3], 2, central=False) == pytest.approx(14 / 3)

    def test_nees_nis(self):
        error = np.array([1.0, 0.5])
        cov = np.eye(2)
        assert nees(error, cov) == pytest.approx(1.25)
        assert nis(error, cov) == pytest.approx(1.25)
        errors = np.array([[1.0, 0.0], [0.0, 2.0]])
        assert_allclose(nees(errors, np.diag([1.0, 4.0])), [1.0, 1.0])

    def test_nees_chi_squared_consistency(self):
        rng = np.random.default_rng(24)
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        L = np.linalg.cholesky(cov)
        errors = rng.standard_normal((50_000, 2)) @ L.T
        vals = nees(errors, cov)
        assert np.mean(vals) == pytest.approx(2.0, rel=0.03)


# =============================================================================
# Matrix decompositions
# =============================================================================


class TestDecompositions:
    def test_tria_positive_definite(self):
        rng = np.random.default_rng(25)
        M = rng.standard_normal((4, 4))
        A = M @ M.T + 4 * np.eye(4)
        S = tria(A)
        assert_allclose(S @ S.T, A, atol=1e-10)
        assert_allclose(S, np.tril(S), atol=1e-12)

    def test_tria_sqrt_two_blocks(self):
        rng = np.random.default_rng(26)
        A = rng.standard_normal((3, 5))
        B = rng.standard_normal((3, 2))
        S = tria_sqrt(A, B)
        assert_allclose(S @ S.T, A @ A.T + B @ B.T, atol=1e-10)
        assert_allclose(S, np.tril(S), atol=1e-12)
        assert np.all(np.diag(S) >= 0)

    def test_tria_sqrt_single_block(self):
        rng = np.random.default_rng(27)
        A = rng.standard_normal((4, 6))
        S = tria_sqrt(A)
        assert_allclose(S @ S.T, A @ A.T, atol=1e-10)

    def test_pinv_truncated_matches_numpy(self):
        rng = np.random.default_rng(28)
        A = rng.standard_normal((5, 3))
        assert_allclose(pinv_truncated(A), np.linalg.pinv(A), atol=1e-10)

    def test_pinv_truncated_rank_control(self):
        A = np.diag([10.0, 1.0, 1e-8])
        P = pinv_truncated(A, rank=2)
        assert_allclose(np.diag(P), [0.1, 1.0, 0.0], atol=1e-12)

    def test_matrix_sqrt_all_methods(self):
        rng = np.random.default_rng(29)
        M = rng.standard_normal((4, 4))
        A = M @ M.T + 4 * np.eye(4)
        ref = scipy.linalg.sqrtm(A).real
        for method in ("schur", "eigenvalue", "denman_beavers"):
            S = matrix_sqrt(A, method=method)
            assert_allclose(S @ S, A, atol=1e-8, err_msg=method)
            assert_allclose(S, ref, atol=1e-6, err_msg=method)

    def test_matrix_sqrt_nonsymmetric(self):
        A = np.array([[4.0, 1.0], [0.0, 9.0]])
        S = matrix_sqrt(A, method="schur")
        assert_allclose(S @ S, A, atol=1e-10)

    def test_rank_revealing_qr(self):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        Q, R, P, rank = rank_revealing_qr(A)
        assert rank == 2
        assert_allclose(A[:, P], Q @ R, atol=1e-10)
        assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_null_space(self):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        N = null_space(A)
        assert N.shape == (3, 1)
        assert_allclose(A @ N, 0, atol=1e-10)
        assert_allclose(N.T @ N, np.eye(1), atol=1e-12)
        ref = scipy.linalg.null_space(A)
        assert_allclose(np.abs(N.T @ ref), np.eye(1), atol=1e-10)

    def test_range_space(self):
        A = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        R = range_space(A)
        assert R.shape == (3, 1)
        # Column space of A spanned by R
        proj = R @ R.T
        assert_allclose(proj @ A, A, atol=1e-10)


# =============================================================================
# Special matrices
# =============================================================================


class TestSpecialMatrices:
    def test_vandermonde(self):
        assert_allclose(vandermonde([1, 2, 3]), np.vander([1.0, 2.0, 3.0]))
        assert_allclose(
            vandermonde([1, 2, 3], increasing=True),
            np.vander([1.0, 2.0, 3.0], increasing=True),
        )

    def test_toeplitz_hankel_circulant(self):
        assert_allclose(
            toeplitz([1, 2, 3], [1, 4, 5]),
            scipy.linalg.toeplitz([1, 2, 3], [1, 4, 5]),
        )
        assert_allclose(
            hankel([1, 2, 3], [3, 4, 5]),
            scipy.linalg.hankel([1, 2, 3], [3, 4, 5]),
        )
        C = circulant([1, 2, 3])
        assert_allclose(C, scipy.linalg.circulant([1, 2, 3]))
        # Circulant matrices are diagonalized by the DFT
        eigs = np.sort_complex(np.linalg.eigvals(C))
        assert_allclose(eigs, np.sort_complex(np.fft.fft([1, 2, 3])), atol=1e-10)

    def test_block_diag_companion(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5]])
        assert_allclose(block_diag(A, B), scipy.linalg.block_diag(A, B))
        # Companion matrix eigenvalues are the polynomial roots
        C = companion([1, -6, 11, -6])  # x^3 - 6x^2 + 11x - 6
        assert_allclose(np.sort(np.linalg.eigvals(C).real), [1, 2, 3], atol=1e-8)

    def test_hilbert_and_inverse(self):
        H = hilbert(5)
        assert_allclose(H, scipy.linalg.hilbert(5))
        assert_allclose(H @ invhilbert(5), np.eye(5), atol=1e-6)

    def test_hadamard(self):
        H = hadamard(8)
        assert_allclose(H @ H.T, 8 * np.eye(8))

    def test_dft_matrix(self):
        F = dft_matrix(8)
        x = np.arange(8.0)
        assert_allclose(F @ x, np.fft.fft(x), atol=1e-10)
        Fn = dft_matrix(8, normalized=True)
        assert_allclose(Fn @ Fn.conj().T, np.eye(8), atol=1e-10)

    def test_kron(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.eye(2)
        assert_allclose(kron(a, b), np.kron(a, b))

    def test_vec_unvec(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(vec(A), [1, 3, 2, 4])
        assert_allclose(unvec(vec(A), 2, 2), A)

    def test_commutation_matrix(self):
        rng = np.random.default_rng(30)
        A = rng.standard_normal((3, 4))
        K = commutation_matrix(3, 4)
        assert_allclose(K @ vec(A), vec(A.T))

    def test_duplication_elimination(self):
        n = 3
        A = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 5.0, 6.0]])
        D = duplication_matrix(n)
        L = elimination_matrix(n)
        vech_A = A[np.tril_indices(n)[0], np.tril_indices(n)[1]]
        # vech via elimination and reconstruction via duplication
        vech_order = L @ vec(A)
        assert_allclose(D @ vech_order, vec(A))
        assert_allclose(np.sort(vech_order), np.sort(vech_A))


# =============================================================================
# Geometry
# =============================================================================


class TestGeometry:
    SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def test_point_in_polygon(self):
        assert point_in_polygon([0.5, 0.5], self.SQUARE)
        assert not point_in_polygon([2.0, 2.0], self.SQUARE)
        # Concave polygon (arrow shape): notch point outside
        concave = np.array([[0, 0], [4, 0], [4, 4], [2, 1.5], [0, 4]], dtype=float)
        assert point_in_polygon([2.0, 0.5], concave)
        assert not point_in_polygon([2.0, 3.0], concave)

    def test_points_in_polygon(self):
        pts = np.array([[0.5, 0.5], [2.0, 2.0], [0.1, 0.9]])
        assert list(points_in_polygon(pts, self.SQUARE)) == [True, False, True]

    def test_convex_hull(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        _, indices = convex_hull(pts)
        assert set(indices) == {0, 1, 2, 3}
        assert convex_hull_area(pts) == pytest.approx(1.0)

    def test_polygon_area_and_centroid(self):
        assert polygon_area(self.SQUARE) == pytest.approx(1.0)
        assert_allclose(polygon_centroid(self.SQUARE), [0.5, 0.5])
        # Clockwise ordering must give the same (unsigned behavior)
        assert polygon_area(self.SQUARE[::-1]) == pytest.approx(1.0)
        assert_allclose(polygon_centroid(self.SQUARE[::-1]), [0.5, 0.5])
        # L-shaped polygon reference (computed by decomposition)
        L_shape = np.array(
            [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float
        )
        assert polygon_area(L_shape) == pytest.approx(3.0)
        assert_allclose(polygon_centroid(L_shape), [5 / 6, 5 / 6])

    def test_line_intersection(self):
        p = line_intersection([0, 0], [1, 1], [0, 1], [1, 0])
        assert_allclose(p, [0.5, 0.5])
        assert line_intersection([0, 0], [1, 0], [0, 1], [1, 1]) is None  # parallel
        # Non-intersecting segments (lines would cross outside segments)
        assert line_intersection([0, 0], [1, 1], [2, 3], [3, 2]) is None

    def test_line_plane_intersection(self):
        p = line_plane_intersection([0, 0, 0], [0, 0, 1], [0, 0, 5], [0, 0, 1])
        assert_allclose(p, [0, 0, 5])
        assert (
            line_plane_intersection([0, 0, 0], [1, 0, 0], [0, 0, 5], [0, 0, 1]) is None
        )

    def test_point_line_distances(self):
        assert point_to_line_distance([0, 1], [0, 0], [1, 0]) == pytest.approx(1.0)
        assert point_to_line_distance([5, 3], [0, 0], [1, 0]) == pytest.approx(3.0)
        # Segment: nearest point is an endpoint
        assert point_to_line_segment_distance([2, 1], [0, 0], [1, 0]) == pytest.approx(
            np.sqrt(2)
        )
        assert point_to_line_segment_distance(
            [0.5, 1], [0, 0], [1, 0]
        ) == pytest.approx(1.0)

    def test_triangle_area(self):
        assert triangle_area([0, 0], [1, 0], [0, 1]) == pytest.approx(0.5)
        assert triangle_area([0, 0, 0], [1, 0, 0], [0, 1, 0]) == pytest.approx(0.5)

    def test_barycentric_coordinates(self):
        p1, p2, p3 = [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]
        # Vertices map to unit coordinates
        assert_allclose(barycentric_coordinates(p1, p1, p2, p3), [1, 0, 0])
        assert_allclose(barycentric_coordinates(p2, p1, p2, p3), [0, 1, 0])
        assert_allclose(barycentric_coordinates(p3, p1, p2, p3), [0, 0, 1])
        # Reconstruction property
        pt = np.array([0.2, 0.3])
        lam = barycentric_coordinates(pt, p1, p2, p3)
        rec = lam[0] * np.array(p1) + lam[1] * np.array(p2) + lam[2] * np.array(p3)
        assert_allclose(rec, pt)
        assert lam.sum() == pytest.approx(1.0)

    def test_delaunay_and_bounding_box(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        tri = delaunay_triangulation(pts)
        assert len(tri.simplices) == 2
        mn, mx = bounding_box(np.array([[0, 1], [2, 3], [1, 2]], dtype=float))
        assert_allclose(mn, [0, 1])
        assert_allclose(mx, [2, 3])

    def test_minimum_bounding_circle(self):
        np.random.seed(0)
        pts = np.array([[0, 0], [2, 0], [1, 0.5], [1, 1]], dtype=float)
        center, radius = minimum_bounding_circle(pts)
        dists = np.linalg.norm(pts - center, axis=1)
        assert np.all(dists <= radius + 1e-9)
        # Optimal circle for these points has diameter [0,0]-[2,0]:
        # center (1, 0), radius 1 (both other points lie within/on it)
        assert radius == pytest.approx(1.0, abs=0.05)

    def test_oriented_bounding_box(self):
        # Rotated rectangle: OBB should recover its area
        rng = np.random.default_rng(31)
        rect = np.array(
            [[x, y] for x in np.linspace(0, 4, 9) for y in np.linspace(0, 1, 5)]
        )
        theta = 0.6
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = rect @ R.T + rng.standard_normal(2) * 0.0
        center, extents, angle = oriented_bounding_box(pts)
        area = 4 * extents[0] * extents[1]
        assert area == pytest.approx(4.0, rel=1e-6)
        # All points inside the OBB
        c, s = np.cos(-angle), np.sin(-angle)
        Rm = np.array([[c, -s], [s, c]])
        local = (pts - center) @ Rm.T
        assert np.all(np.abs(local) <= extents + 1e-9)
