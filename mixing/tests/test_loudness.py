"""``mixing.loudness``: array-only meters, group leveling and the peak limiter."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from mixing.loudness import (
    integrated_lufs,
    level_group,
    limit_peaks,
    loudness_range_lu,
    true_peak_db,
    true_peak_envelope,
)
from mixing.util import has_ffmpeg

FS = 44100


def _voice(seconds, amplitude, *, seed=0, channels=1):
    """Speech-like: harmonics under a slow envelope, plus hiss."""
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    wave = sum(np.sin(2 * np.pi * 220 * h * t) / h for h in (1, 2, 3, 4)) / 2
    x = amplitude * envelope * wave + 0.0005 * rng.standard_normal(len(t))
    return np.tile(x[:, None], (1, channels))


def test_importing_the_module_is_light():
    code = (
        "import sys, mixing.loudness;"
        "bad = [m for m in ('pydub', 'moviepy', 'cv2', 'PIL') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stderr


@pytest.mark.parametrize("channels", [1, 2])
def test_meters_agree_with_pyloudnorm(channels):
    pyln = pytest.importorskip("pyloudnorm")
    data = _voice(6, 0.3, channels=channels)
    ours = integrated_lufs(data, FS)
    assert ours == pytest.approx(pyln.Meter(FS).integrated_loudness(data), abs=0.1)
    assert integrated_lufs(data, FS, meter="pyloudnorm") == pytest.approx(ours, abs=0.1)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_ebur128_meter_agrees_with_numpy():
    data = _voice(6, 0.3, channels=2)
    assert integrated_lufs(data, FS, meter="ffmpeg_ebur128") == pytest.approx(
        integrated_lufs(data, FS), abs=0.1
    )


def test_meter_is_a_callable_or_a_known_name():
    data = _voice(1, 0.3)
    assert integrated_lufs(data, FS, meter=lambda d, fs: -23.0) == -23.0
    with pytest.raises(ValueError, match="unknown meter"):
        integrated_lufs(data, FS, meter="nope")


def test_level_group_reaches_the_target_and_keeps_relative_dynamics():
    quiet, loud = _voice(4, 0.05), _voice(4, 0.4, seed=1)
    gain, measured = level_group([quiet, loud], FS, target_lufs=-20.0)
    assert measured == pytest.approx(integrated_lufs(np.vstack([quiet, loud]), FS))
    joined = np.vstack([quiet, loud]) * 10 ** (gain / 20)
    assert integrated_lufs(joined, FS) == pytest.approx(-20.0, abs=0.05)
    # one gain for the group: the members keep their spread (quiet stays quieter)
    spread = integrated_lufs(loud, FS) - integrated_lufs(quiet, FS)
    assert integrated_lufs(loud * 10 ** (gain / 20), FS) - integrated_lufs(
        quiet * 10 ** (gain / 20), FS
    ) == pytest.approx(spread, abs=1e-6)


def test_level_group_edge_cases():
    # silence: nothing to level
    assert level_group([np.zeros((FS, 1))], FS, target_lufs=-20.0) == (0.0, -np.inf)
    # a group of short lines is looped to be measurable
    short = _voice(0.3, 0.3)
    _, measured = level_group([short], FS, target_lufs=-20.0)
    assert np.isfinite(measured)
    # mixed channel counts are downmixed rather than failing
    level_group([_voice(1, 0.3), _voice(1, 0.3, channels=2)], FS, target_lufs=-20.0)
    # a huge gain is capped, with a warning
    with pytest.warns(UserWarning, match="capped"):
        gain, _ = level_group(
            [_voice(2, 0.0005)], FS, target_lufs=-10.0, max_gain_db=6.0
        )
    assert gain == 6.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        level_group([_voice(2, 0.3)], FS, target_lufs=-30.0)  # a cut never warns


def test_limiter_never_exceeds_its_ceiling():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((FS, 2)) * 0.8
    out = limit_peaks(data, ceiling=0.5, window=int(0.005 * FS))
    assert np.abs(out).max() <= 0.5 + 1e-12
    assert out.shape == data.shape


def test_limiter_leaves_quiet_signal_untouched_and_does_not_move_in_time():
    quiet = _voice(1, 0.1)
    assert limit_peaks(quiet, ceiling=1.0, window=100) is quiet
    burst = quiet.copy()
    burst[FS // 2] = 3.0
    out = limit_peaks(burst, ceiling=1.0, window=100)
    assert np.abs(out[FS // 2]).max() <= 1.0
    assert np.argmax(np.abs(out[:, 0])) == FS // 2


def test_true_peak_option_holds_intersample_peaks():
    n = np.arange(FS)
    # fs/4 sine at 45 degrees: samples sit ~3 dB under the true peak
    data = (np.sin(2 * np.pi * n / 4 + np.pi / 4) * 0.99)[:, None]
    ceiling = 10 ** (-3.0 / 20)
    sample_limited = limit_peaks(data, ceiling=ceiling, window=64)
    tp_limited = limit_peaks(data, ceiling=ceiling, window=64, true_peak=True, fs=FS)
    assert true_peak_db(sample_limited, FS) > -3.0 + 1.0  # sample peak misses it
    assert true_peak_db(tp_limited, FS) <= -3.0 + 0.1
    with pytest.raises(ValueError, match="needs fs"):
        limit_peaks(data, ceiling=ceiling, true_peak=True)


def test_loudness_range_of_steady_signal_is_small():
    assert loudness_range_lu(_voice(10, 0.3), FS) < 2.0
    assert loudness_range_lu(np.zeros((0, 1)), FS) == 0.0


def test_true_peak_limiter_holds_the_ceiling_with_its_default_window():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((FS * 2, 2)) * 0.7
    data[FS] = 3.0
    ceiling = 10 ** (-1.0 / 20)
    out = limit_peaks(data, ceiling=ceiling, true_peak=True, fs=FS)
    assert true_peak_db(out, FS) <= -1.0 + 0.1


def test_silence_reads_as_silence_in_every_meter_and_gets_no_gain():
    silent = np.zeros((FS, 2))
    assert integrated_lufs(silent, FS) == -np.inf
    assert level_group([silent], FS, target_lufs=-16.0)[0] == 0.0
    if has_ffmpeg():
        assert integrated_lufs(silent, FS, meter="ffmpeg_ebur128") == -np.inf
        assert level_group([silent], FS, target_lufs=-16.0, meter="ffmpeg_ebur128") == (
            0.0,
            -np.inf,
        )


def test_one_dimensional_arrays_are_mono():
    mono = _voice(3, 0.3)
    flat = mono[:, 0]
    assert integrated_lufs(flat, FS) == integrated_lufs(mono, FS)
    assert true_peak_db(flat, FS) == true_peak_db(mono, FS)
    assert level_group([flat], FS, target_lufs=-20.0) == level_group(
        [mono], FS, target_lufs=-20.0
    )
    assert limit_peaks(flat * 20, ceiling=0.5).shape == (len(flat), 1)


def test_ceiling_must_be_linear_and_min_measure_one_block():
    with pytest.raises(ValueError, match="linear"):
        limit_peaks(np.ones((10, 1)), ceiling=-1.5)
    with pytest.raises(ValueError, match="gate block"):
        level_group([_voice(1, 0.3)], FS, target_lufs=-20.0, min_measure_s=0.1)


def test_mixed_channel_counts_are_downmixed_and_measured():
    mono, stereo = _voice(2, 0.3), _voice(2, 0.3, channels=2)
    _, measured = level_group([mono, stereo], FS, target_lufs=-20.0)
    both_mono = np.vstack([mono, stereo.mean(axis=1, keepdims=True)])
    assert measured == pytest.approx(integrated_lufs(both_mono, FS))


# --- true peak against an ideal sinc reconstruction ---------------------------

TP_FS = 48000
TP_UP = 4  # the meter's factor at 48 kHz
TP_EDGE = 300  # samples kept out of interior-only comparisons
#: The near-ideal reference: a windowed sinc ~100x longer than the meter's.
REF_HALF_SAMPLES, REF_KAISER_BETA = 3000, 14.0


def _db(x):
    return 20 * np.log10(x)


def _sinc_envelope(x, up=TP_UP):
    """``|y|`` on the ``up``-times grid of a near-ideal sinc reconstruction of
    ``x`` (taken as zero outside its span)."""
    from scipy.signal import resample_poly
    from scipy.signal.windows import kaiser

    n = np.arange(-REF_HALF_SAMPLES * up, REF_HALF_SAMPLES * up + 1)
    taps = np.sinc(n / up) * kaiser(len(n), REF_KAISER_BETA) / up
    return np.abs(resample_poly(x, up, 1, window=taps))


def _band_limited_noise(n, cutoff, rng):
    """White noise with nothing above ``cutoff`` (a fraction of fs), peak 1."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n) > cutoff] = 0
    x = np.fft.irfft(spectrum, n)
    return x / np.abs(x).max()


def _interior_peak_db(x):
    """The meter and the reference, away from the edges: dB, dB."""
    inner = slice(TP_EDGE, -TP_EDGE)
    meter = true_peak_envelope(x, TP_FS)[inner].max()
    ideal = _sinc_envelope(x)[TP_EDGE * TP_UP : -TP_EDGE * TP_UP].max()
    return _db(meter), _db(ideal)


def test_true_peak_reads_bright_noise_as_an_ideal_sinc_would():
    # white noise band-limited at 0.45 and 0.47 fs (an ADC's anti-alias edge):
    # the old filter (scipy's default) under-read these by up to 0.1 / 0.2 dB
    rng = np.random.default_rng(2)
    for cutoff, tolerance in ((0.45, 0.005), (0.47, 0.02)):
        for _ in range(3):
            meter, ideal = _interior_peak_db(_band_limited_noise(4000, cutoff, rng))
            assert meter == pytest.approx(ideal, abs=tolerance)


def test_true_peak_of_a_near_nyquist_sine():
    # 21.6 kHz = 9/20 fs at 9 degrees: samples peak at sin(81 deg), 0.1 dB under
    # the crest, which the 4x grid hits; the old filter lost 0.66 dB there and
    # read only the sample peak. The reference is the sine itself.
    n = np.arange(2400)
    phase = np.pi / 20
    x = np.sin(2 * np.pi * 9 / 20 * n + phase)
    grid = np.arange(len(x) * TP_UP) / TP_UP
    ideal = _db(np.abs(np.sin(2 * np.pi * 9 / 20 * grid + phase)).max())
    assert _interior_peak_db(x)[0] == pytest.approx(ideal, abs=0.01)
    assert true_peak_db(x, TP_FS) > _db(np.abs(x).max()) + 0.05


def test_true_peak_does_not_loop_the_signal_at_its_edges():
    # an FFT upsampler treats the clip as looping: the -0.8 -> +0.8 jump reads
    # ~+0.2 dBTP. Taken as silence outside, it reads what the sinc sum reads.
    x = np.linspace(-0.8, 0.8, 2400)
    assert true_peak_db(x, TP_FS) == pytest.approx(
        _db(_sinc_envelope(x).max()), abs=0.01
    )
    assert true_peak_db(x, TP_FS) < -0.5


def test_true_peak_passes_the_samples_unchanged():
    # an impulse's reconstruction peaks at the sample itself; the old filter's
    # centre tap read it 0.0055 dB high
    x = np.zeros(1000)
    x[500] = 0.5
    assert true_peak_envelope(x, TP_FS).max() == 0.5
    rng = np.random.default_rng(7)
    noise = np.clip(rng.standard_normal((TP_FS, 2)), -1, 1)
    assert (true_peak_envelope(noise, TP_FS) >= np.abs(noise).max(axis=1)).all()


def test_true_peak_limiter_holds_the_ceiling_by_an_independent_meter():
    # limited by its own meter, re-measured by the reference: the old filter
    # let bright material through at -0.92 dBTP for a -1 dBTP ceiling
    x = _band_limited_noise(TP_FS, 0.45, np.random.default_rng(4))[:, None]
    out = limit_peaks(x, ceiling=10 ** (-1.0 / 20), true_peak=True, fs=TP_FS)
    inner = slice(TP_EDGE * TP_UP, -TP_EDGE * TP_UP)
    assert _db(_sinc_envelope(out[:, 0])[inner].max()) <= -1.0 + 0.01


def test_a_finer_grid_brings_the_limiter_to_its_true_ceiling():
    # at 4x the peak between grid points is missed: -1 dBTP limits to ~-0.8 by
    # a 16x reference; at 8x it lands within a few hundredths of the ceiling
    x = 1.5 * _band_limited_noise(TP_FS // 2, 0.45, np.random.default_rng(1))[:, None]
    ceiling = 10 ** (-1.0 / 20)
    inner = slice(TP_EDGE * 16, -TP_EDGE * 16)

    def fine_peak_db(oversampling):
        out = limit_peaks(
            x, ceiling=ceiling, true_peak=True, fs=TP_FS, oversampling=oversampling
        )
        return _db(_sinc_envelope(out[:, 0], up=16)[inner].max())

    assert fine_peak_db(None) > -1.0 + 0.1
    assert fine_peak_db(8) <= -1.0 + 0.06
