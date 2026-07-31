"""``reassigned_spectrogram`` against signals whose localization is known.

The function computed its reassignment corrections into two local variables and
then returned the plain ``|STFT|^2``, with ``# noqa: F841`` suppressing the
unused-variable warning (gh-17). Callers asking for a reassigned spectrogram got
an ordinary one, silently.

It had a test. ``test_reassigned_spectrogram_structural`` checked the output
shape, that power was non-negative, and that a 100 Hz tone peaked near 100 Hz --
all true of a plain spectrogram, and the test even carried a comment saying so.
That is the failure this file is written against: the assertions have to be ones
an unreassigned spectrogram *fails*.

So the central test here compares the two directly and requires them to differ,
and the rest measure concentration rather than position. Reassignment does not
move the energy of a well-behaved signal to a different place; it moves it to a
*sharper* place. A test on the peak location cannot see the difference, which is
exactly how the defect survived.

Oracles are signals with analytically known time-frequency support:

- a pure tone, which occupies one frequency for all time;
- an impulse, which occupies one instant across all frequencies;
- a linear chirp, whose instantaneous frequency ``f0 + k t`` is known exactly.

Reference: Auger, F. and Flandrin, P. (1995), "Improving the readability of
time-frequency and time-scale representations by the reassignment method",
IEEE Trans. Signal Processing 43(5), 1068-1089.
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.transforms.stft import (
    reassigned_spectrogram,
    spectrogram,
)

FS = 1000.0
DURATION = 1.0


def _time_axis(fs: float = FS, duration: float = DURATION) -> np.ndarray:
    return np.arange(0, duration, 1 / fs)


def _frequency_spread(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Mean energy-weighted standard deviation of frequency, per frame.

    The measure of concentration that matters. A representation can have the
    right centroid and still be smeared across many bins; this is what
    reassignment is supposed to reduce.
    """
    weights = power / (power.sum(axis=0, keepdims=True) + 1e-300)
    centroid = (frequencies[:, None] * weights).sum(axis=0)
    variance = (((frequencies[:, None] - centroid) ** 2) * weights).sum(axis=0)
    occupied = power.sum(axis=0) > 1e-9 * power.sum()
    return float(np.sqrt(variance[occupied]).mean())


def _time_spread(times: np.ndarray, power: np.ndarray) -> float:
    """Energy-weighted standard deviation of time, over the whole plane."""
    weights = power / (power.sum() + 1e-300)
    centroid = (times[None, :] * weights).sum()
    return float(np.sqrt((((times[None, :] - centroid) ** 2) * weights).sum()))


class TestItActuallyReassigns:
    """The regression guard for gh-17.

    Every other test in this file would pass on a plain spectrogram if the
    signal happened to be concentrated already. These would not.
    """

    def test_most_cells_are_vacated(self):
        """The defect, stated directly.

        Reassignment moves each cell's energy somewhere else, so the cells it
        leaves behind hold exactly zero. A plain spectrogram has essentially no
        exact zeros -- windowed leakage puts something everywhere. Around two
        thirds of this plane empties out, which no unreassigned result does.

        Comparing the two arrays elementwise would seem more direct and is not:
        this function frames the signal itself while `spectrogram` delegates to
        scipy, which pads the boundaries differently, so the two differ
        slightly even when the reassignment is skipped entirely. That check
        passed against the reintroduced defect. This one does not.
        """
        t = _time_axis()
        signal = np.sin(2 * np.pi * (50.0 * t + 0.5 * 200.0 * t**2))

        _, _, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=128, noverlap=96
        )
        plain = spectrogram(signal, fs=FS, nperseg=128, noverlap=96).power

        assert reassigned.shape == plain.shape
        vacated = np.count_nonzero(reassigned == 0.0) / reassigned.size
        untouched = np.count_nonzero(plain == 0.0) / plain.size

        assert vacated > 0.25, (
            f"only {vacated:.1%} of cells are empty; a reassigned spectrogram "
            f"vacates most of the plane, so the corrections are being computed "
            f"and discarded again (gh-17)"
        )
        assert untouched < 0.01, (
            "the plain spectrogram is already mostly empty, so this test is "
            "not distinguishing anything"
        )

    def test_energy_moves_to_fewer_cells(self):
        """Concentration, measured without reference to where the energy is.

        Reassignment redistributes energy into fewer time-frequency cells. The
        fraction carried by the busiest one percent of cells is a direct measure
        of that, and it cannot be satisfied by a representation that merely has
        the right peak.
        """
        t = _time_axis()
        signal = np.sin(2 * np.pi * (50.0 * t + 0.5 * 200.0 * t**2))

        _, _, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=128, noverlap=96
        )
        plain = spectrogram(signal, fs=FS, nperseg=128, noverlap=96).power

        def busiest_fraction(power: np.ndarray) -> float:
            ordered = np.sort(power.ravel())[::-1]
            top = max(1, int(0.01 * ordered.size))
            return float(ordered[:top].sum() / ordered.sum())

        assert busiest_fraction(reassigned) > busiest_fraction(plain) * 1.2, (
            f"reassigned concentration {busiest_fraction(reassigned):.3f} is "
            f"not meaningfully above the plain spectrogram's "
            f"{busiest_fraction(plain):.3f}"
        )


class TestKnownLocalization:
    """Signals whose time-frequency support is known analytically."""

    def test_a_pure_tone_concentrates_in_frequency(self):
        """A tone occupies one frequency. Windowing smears it; reassignment
        should undo most of that."""
        signal = np.sin(2 * np.pi * 100.0 * _time_axis())

        frequencies, _, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=256, noverlap=192
        )
        plain_result = spectrogram(signal, fs=FS, nperseg=256, noverlap=192)

        sharp = _frequency_spread(frequencies, reassigned)
        blurred = _frequency_spread(plain_result.frequencies, plain_result.power)

        assert sharp < blurred / 10.0, (
            f"reassigned frequency spread {sharp:.4f} Hz is not an order of "
            f"magnitude below the plain spectrogram's {blurred:.4f} Hz"
        )

    def test_a_pure_tone_stays_at_its_own_frequency(self):
        """Sharpening must not move the tone.

        Concentration alone is not enough -- energy piled onto the wrong bin is
        concentrated too. The peak must land within one bin of 100 Hz.
        """
        signal = np.sin(2 * np.pi * 100.0 * _time_axis())
        frequencies, _, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=256, noverlap=192
        )

        peak = frequencies[np.argmax(reassigned.sum(axis=1))]
        resolution = frequencies[1] - frequencies[0]
        assert abs(peak - 100.0) <= resolution, (
            f"reassigned peak at {peak:.2f} Hz, more than one {resolution:.2f} "
            f"Hz bin from the true 100 Hz"
        )

    def test_an_impulse_concentrates_in_time(self):
        """An impulse occupies one instant across all frequencies."""
        signal = np.zeros(int(FS * DURATION))
        signal[len(signal) // 2] = 1.0

        _, times, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=128, noverlap=120
        )
        plain_result = spectrogram(signal, fs=FS, nperseg=128, noverlap=120)

        sharp = _time_spread(times, reassigned)
        blurred = _time_spread(plain_result.times, plain_result.power)

        assert sharp < blurred / 10.0, (
            f"reassigned time spread {sharp * 1000:.3f} ms is not an order of "
            f"magnitude below the plain spectrogram's {blurred * 1000:.3f} ms"
        )

    def test_an_impulse_stays_at_its_own_instant(self):
        signal = np.zeros(int(FS * DURATION))
        signal[len(signal) // 2] = 1.0

        _, times, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=128, noverlap=120
        )
        weights = reassigned / reassigned.sum()
        centroid = float((times[None, :] * weights).sum())

        step = times[1] - times[0]
        assert abs(centroid - 0.5) <= step, (
            f"reassigned energy centers at t={centroid:.4f} s, more than one "
            f"{step * 1000:.1f} ms frame step from the true 0.5 s"
        )

    def test_a_chirp_concentrates_along_its_instantaneous_frequency(self):
        """The strongest case: a chirp is smeared by any fixed window.

        Its instantaneous frequency ``f0 + k t`` is known exactly, so both the
        concentration and the position can be checked at once.
        """
        start_hz, sweep_hz_per_s = 50.0, 200.0
        t = _time_axis()
        signal = np.sin(2 * np.pi * (start_hz * t + 0.5 * sweep_hz_per_s * t**2))

        frequencies, times, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=128, noverlap=120
        )
        plain_result = spectrogram(signal, fs=FS, nperseg=128, noverlap=120)

        sharp = _frequency_spread(frequencies, reassigned)
        blurred = _frequency_spread(plain_result.frequencies, plain_result.power)
        assert sharp < blurred / 5.0, (
            f"reassigned spread {sharp:.3f} Hz vs plain {blurred:.3f} Hz -- the "
            f"chirp is not being sharpened"
        )

        # And the sharpened ridge sits on the true instantaneous frequency.
        weights = reassigned / (reassigned.sum(axis=0, keepdims=True) + 1e-300)
        centroid = (frequencies[:, None] * weights).sum(axis=0)
        expected = start_hz + sweep_hz_per_s * times
        interior = (times > 0.1) & (times < 0.9)  # away from edge effects
        error = np.abs(centroid - expected)[interior]
        assert np.median(error) < 5.0, (
            f"reassigned ridge sits a median {np.median(error):.2f} Hz from the "
            f"chirp's true instantaneous frequency"
        )


class TestConservationAndContract:
    """Properties any spectrogram-shaped return has to satisfy."""

    SIGNALS = {
        "tone": lambda t: np.sin(2 * np.pi * 100.0 * t),
        "chirp": lambda t: np.sin(2 * np.pi * (50.0 * t + 0.5 * 200.0 * t**2)),
        "noise": lambda t: np.random.default_rng(0).standard_normal(t.size),
    }
    IDS = list(SIGNALS)

    @pytest.mark.parametrize("name", IDS)
    def test_total_energy_is_preserved(self, name):
        """Reassignment moves energy; it does not create or destroy it.

        A few tenths of a percent goes missing because energy reassigned past
        the edge of the grid is dropped rather than piled onto the boundary,
        which would invent a peak that is not there.
        """
        t = _time_axis()
        signal = self.SIGNALS[name](t)

        _, _, reassigned = reassigned_spectrogram(
            signal, fs=FS, nperseg=128, noverlap=96
        )
        plain = spectrogram(signal, fs=FS, nperseg=128, noverlap=96).power

        ratio = reassigned.sum() / plain.sum()
        assert 0.95 < ratio < 1.05, (
            f"{name}: reassigned total energy is {ratio:.4f} of the plain "
            f"spectrogram's, so the scaling does not match `spectrogram`"
        )

    @pytest.mark.parametrize("name", IDS)
    def test_power_is_non_negative_and_finite(self, name):
        t = _time_axis()
        _, _, reassigned = reassigned_spectrogram(
            self.SIGNALS[name](t), fs=FS, nperseg=128, noverlap=96
        )
        assert np.all(np.isfinite(reassigned)), f"{name}: non-finite power"
        assert np.all(reassigned >= 0.0), f"{name}: negative power"

    def test_the_axes_match_the_returned_grid(self):
        t = _time_axis()
        frequencies, times, reassigned = reassigned_spectrogram(
            self.SIGNALS["chirp"](t), fs=FS, nperseg=128, noverlap=96
        )
        assert reassigned.shape == (frequencies.size, times.size)
        assert np.all(np.diff(frequencies) > 0)
        assert np.all(np.diff(times) > 0)
        assert frequencies[0] == 0.0
        assert frequencies[-1] == pytest.approx(FS / 2)

    def test_a_signal_shorter_than_one_segment_is_rejected(self):
        """Better than the shape error `sliding_window_view` would raise."""
        with pytest.raises(ValueError, match="fewer than nperseg"):
            reassigned_spectrogram(np.zeros(50), fs=FS, nperseg=128)

    def test_an_explicit_window_array_is_accepted(self):
        """The signature allows a window array, so it has to work."""
        from scipy.signal.windows import hamming

        t = _time_axis()
        _, _, reassigned = reassigned_spectrogram(
            self.SIGNALS["tone"](t),
            fs=FS,
            window=hamming(128, sym=False),
            nperseg=128,
            noverlap=96,
        )
        assert np.all(np.isfinite(reassigned))
        assert reassigned.sum() > 0.0
