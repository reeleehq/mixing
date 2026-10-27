"""Voice diagnosis and pitch shifting (``mixing.audio.voice``)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("pydub")
pytest.importorskip("scipy")

from pydub import AudioSegment

from mixing.audio import Audio, has_rubberband, loudness, shift_pitch, voice_profile
from mixing.audio import voice as voice_module
from mixing.util import has_ffmpeg

needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not on PATH")
needs_rubberband = pytest.mark.skipif(
    not has_rubberband(), reason="ffmpeg has no rubberband filter"
)

RATE = 22050


def _pcm(signal, *, sample_width=2, channels=1):
    full = float(1 << (8 * sample_width - 1))
    ints = np.clip(np.round(signal * full), -full, full - 1)
    if channels > 1:
        ints = np.repeat(ints[:, None], channels, axis=1).ravel()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    return AudioSegment(
        ints.astype(dtype).tobytes(),
        frame_rate=RATE,
        sample_width=sample_width,
        channels=channels,
    )


def _harmonic(seconds, f0, *, amplitude=0.3, boom=0.0, n_harmonics=8):
    """A voiced signal: harmonics of ``f0`` (1/h amplitudes), optional 100 Hz boom."""
    t = np.arange(int(seconds * RATE)) / RATE
    wave = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n_harmonics + 1))
    wave = wave / np.max(np.abs(wave))
    wave = wave + boom * np.sin(2 * np.pi * 100.0 * t)
    return amplitude * wave / np.max(np.abs(wave))


def _audio(signal, **kw):
    return Audio(_pcm(signal, **kw))


def _median_f0(audio):
    return voice_profile(audio)["all"]["f0_hz"]


# --- voice_profile --------------------------------------------------------------


def test_profile_without_f0_needs_no_librosa_and_measures_loudness():
    signal = _harmonic(2.0, 150.0)
    audio = _audio(signal)
    profile = voice_profile(audio, [("a", 0.0, 2.0)], f0=False)["a"]
    assert profile["f0_hz"] is None and profile["voiced_fraction"] is None
    assert profile["duration_s"] == pytest.approx(2.0)
    assert profile["loudness_lufs"] == pytest.approx(loudness(audio), abs=1e-9)


@pytest.mark.parametrize("f0", [110.0, 220.0])
def test_median_f0_of_harmonic_series(f0):
    pytest.importorskip("librosa")
    assert _median_f0(_audio(_harmonic(2.0, f0))) == pytest.approx(f0, rel=0.03)


def test_centroid_and_low_band_ordering():
    clean = _audio(_harmonic(2.0, 220.0))
    boomy = _audio(_harmonic(2.0, 220.0, boom=1.5))
    bright = _audio(_harmonic(2.0, 440.0))
    p_clean, p_boomy, p_bright = (
        voice_profile(a, f0=False)["all"] for a in (clean, boomy, bright)
    )
    assert p_boomy["low_band_db"] > p_clean["low_band_db"] + 3
    assert p_boomy["spectral_centroid_hz"] < p_clean["spectral_centroid_hz"]
    assert p_bright["spectral_centroid_hz"] > p_clean["spectral_centroid_hz"]


def test_spans_of_a_label_are_measured_over_their_union():
    pytest.importorskip("librosa")
    low, high = _harmonic(1.5, 110.0), _harmonic(1.5, 220.0, amplitude=0.05)
    signal = np.concatenate([low, high, low, high])  # a b a b, 1.5 s each
    audio = _audio(signal)
    turns = [("a", 0.0, 1.5), ("b", 1.5, 3.0), ("a", 3.0, 4.5), ("b", 4.5, 6.0)]
    profile = voice_profile(audio, turns)
    assert profile["a"]["duration_s"] == pytest.approx(3.0)
    assert profile["a"]["f0_hz"] == pytest.approx(110.0, rel=0.03)
    assert profile["b"]["f0_hz"] == pytest.approx(220.0, rel=0.03)
    union_a = _audio(np.concatenate([low, low]))
    assert profile["a"]["loudness_lufs"] == pytest.approx(loudness(union_a), abs=0.05)
    assert profile["b"]["loudness_lufs"] < profile["a"]["loudness_lufs"] - 10
    # the mapping shape is equivalent, overlapping spans of a label are merged
    mapping = voice_profile(
        audio, {"a": [(0.0, 1.0), (0.5, 1.5), (3.0, 4.5)]}, f0=False
    )
    assert mapping["a"]["duration_s"] == pytest.approx(3.0)
    assert mapping["a"]["loudness_lufs"] == pytest.approx(
        profile["a"]["loudness_lufs"], abs=1e-6
    )


def test_noise_and_silence_have_no_f0():
    pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    noise = _audio(0.1 * rng.standard_normal(2 * RATE))
    assert _median_f0(noise) is None
    silent = voice_profile(Audio(AudioSegment.silent(1000, frame_rate=RATE)))["all"]
    assert silent["f0_hz"] is None
    assert silent["loudness_lufs"] is None and silent["low_band_db"] is None


def test_spans_in_seconds_whatever_time_unit():
    segment = _pcm(_harmonic(2.0, 150.0))
    by_s = voice_profile(Audio(segment), [("a", 0.5, 1.5)], f0=False)
    by_ms = voice_profile(
        Audio(segment, time_unit="milliseconds"), [("a", 0.5, 1.5)], f0=False
    )
    assert by_ms == by_s


def test_bad_span_raises():
    with pytest.raises(ValueError):
        voice_profile(_audio(_harmonic(1.0, 150.0)), [("a", 0.5, 3.0)], f0=False)


def test_mapping_to_a_single_span_is_a_clear_type_error():
    audio = _audio(_harmonic(1.0, 150.0))
    with pytest.raises(TypeError, match=r"list of \(start_s, end_s\) spans"):
        voice_profile(audio, {"a": (0.0, 0.5)}, f0=False)


def test_spans_shorter_than_a_pyin_frame_have_no_f0_and_no_voiced_fraction():
    pytest.importorskip("librosa")
    audio = _audio(_harmonic(1.0, 150.0))
    short = voice_profile(audio, [("a", 0.0, 0.05)])["a"]  # < 128 ms frame
    assert short["f0_hz"] is None and short["voiced_fraction"] is None
    silent = voice_profile(Audio(AudioSegment.silent(1000, frame_rate=RATE)))["all"]
    assert silent["voiced_fraction"] == 0.0


# --- shift_pitch ----------------------------------------------------------------


def test_zero_semitones_is_an_equal_copy():
    audio = _audio(_harmonic(1.0, 150.0))
    out = shift_pitch(audio, 0)
    assert out is not audio
    assert out._get_segment().raw_data == audio._get_segment().raw_data


def test_semitone_range_and_method_validated():
    audio = _audio(_harmonic(0.5, 150.0))
    with pytest.raises(ValueError):
        shift_pitch(audio, 25)
    with pytest.raises(ValueError):
        shift_pitch(audio, 2, method="nope")
    with pytest.raises(ValueError):  # validated before the semitones == 0 shortcut
        shift_pitch(audio, 0, method="nope")


@pytest.mark.parametrize("sample_width,channels", [(2, 1), (1, 1), (4, 2)])
def test_librosa_octave_up_doubles_f0_and_keeps_shape(sample_width, channels):
    pytest.importorskip("librosa")
    audio = Audio(
        _pcm(_harmonic(2.0, 110.0), sample_width=sample_width, channels=channels),
        time_unit="milliseconds",
    )
    out = shift_pitch(audio, 12, method="librosa")
    a, b = audio._get_segment(), out._get_segment()
    assert len(b.raw_data) == len(a.raw_data)
    assert (b.frame_rate, b.channels, b.sample_width) == (
        a.frame_rate,
        a.channels,
        a.sample_width,
    )
    assert out.time_unit == "milliseconds"
    if sample_width >= 2:
        assert _median_f0(out) == pytest.approx(220.0, rel=0.05)


def test_shift_does_not_clip_beyond_input():
    pytest.importorskip("librosa")
    audio = _audio(_harmonic(1.0, 110.0, amplitude=0.999))
    out = shift_pitch(audio, -5, method="librosa")
    samples = np.array(out._get_segment().get_array_of_samples())
    assert np.max(np.abs(samples)) < 32767


@needs_ffmpeg
def test_auto_without_rubberband_falls_back_to_resample_and_warns(monkeypatch):
    calls = []
    real = voice_module.resample_strategy
    monkeypatch.setattr(
        voice_module,
        "resample_strategy",
        lambda *a: calls.append(1) or real(*a),
    )
    monkeypatch.setattr(voice_module, "has_rubberband", lambda ffmpeg=None: False)
    audio = _audio(_harmonic(0.5, 150.0))
    with pytest.warns(UserWarning, match="formants"):
        shift_pitch(audio, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shift_pitch(audio, 3, preserve_formants=False)
    assert len(calls) == 2


def test_forced_rubberband_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(voice_module, "has_rubberband", lambda ffmpeg=None: False)
    with pytest.raises(RuntimeError, match="librubberband"):
        shift_pitch(_audio(_harmonic(0.5, 150.0)), 3, method="rubberband")


def test_callable_strategy_output_is_validated():
    audio = _audio(_harmonic(1.0, 150.0), channels=2)
    slack = int(voice_module.LENGTH_TOLERANCE_S * RATE)
    out = shift_pitch(audio, 2, method=lambda x, fs, s, pf: x[: len(x) - slack])
    assert len(out._get_segment().raw_data) == len(audio._get_segment().raw_data)
    with pytest.raises(ValueError, match="keep the length"):
        shift_pitch(audio, 2, method=lambda x, fs, s, pf: x[: len(x) // 2])
    with pytest.raises(ValueError, match="shape"):
        shift_pitch(audio, 2, method=lambda x, fs, s, pf: x.mean(axis=1))
    with pytest.raises(ValueError, match="shape"):
        shift_pitch(audio, 2, method=lambda x, fs, s, pf: x[:, :1])


def _overshoot_strategy(x, fs, semitones, preserve_formants):
    """Identity plus one spike that lands just over full scale once loudness is
    restored (one sample barely moves the integrated loudness)."""
    out = x.copy()
    out[len(out) // 2] = 1.02
    return out


def test_clip_guard_scales_to_the_ceiling_not_the_input_peak():
    audio = _audio(_harmonic(2.0, 150.0, amplitude=0.3))
    out = shift_pitch(audio, 2, method=_overshoot_strategy)
    samples = np.array(out._get_segment().get_array_of_samples())
    assert np.max(np.abs(samples)) < 32767  # no clipping
    assert np.max(np.abs(samples)) / 32768 == pytest.approx(
        voice_module.CLIP_CEILING, abs=1e-4
    )
    assert abs(loudness(out) - loudness(audio)) < 1.0


def test_preserve_loudness_false_keeps_the_strategy_level():
    audio = _audio(_harmonic(2.0, 150.0))
    halve = lambda x, fs, s, pf: 0.5 * x  # noqa: E731
    kept = shift_pitch(audio, 2, method=halve)
    raw = shift_pitch(audio, 2, method=halve, preserve_loudness=False)
    assert abs(loudness(kept) - loudness(audio)) < 0.1
    assert loudness(raw) - loudness(audio) == pytest.approx(-6.02, abs=0.1)


def test_output_path_writes_the_file(tmp_path):
    audio = _audio(_harmonic(1.0, 150.0))
    target = tmp_path / "shifted.wav"
    got = shift_pitch(audio, 2, method=lambda x, fs, s, pf: x, output=target)
    assert got == target and target.is_file()
    assert len(AudioSegment.from_file(target)) == len(audio._get_segment())


def test_failed_filter_probe_is_not_cached(tmp_path):
    exe = str(tmp_path / "no-such-ffmpeg")
    assert voice_module._ffmpeg_filters(exe) == frozenset()
    assert exe not in voice_module._FFMPEG_FILTERS_CACHE
    assert not voice_module.has_rubberband(exe)


@needs_ffmpeg
def test_shift_latency_raises_when_the_filter_outputs_silence():
    with pytest.raises(RuntimeError, match="search limit"):
        voice_module.shift_latency("volume=0", 16000)


def test_atempo_chain_stays_in_range():
    lo, hi = voice_module.ATEMPO_RANGE
    for factor in (0.25, 0.3, 0.5, 0.9, 1.7, 2.0, 3.1, 4.0):
        stages = voice_module._atempo_chain(factor)
        assert all(lo <= t <= hi for t in stages)
        assert np.prod(stages) == pytest.approx(factor)


@needs_ffmpeg
@pytest.mark.parametrize("semitones", [-5, 7])
def test_loudness_is_preserved(semitones):
    audio = _audio(_harmonic(2.0, 150.0))
    out = shift_pitch(audio, semitones, method="resample")
    assert abs(loudness(out) - loudness(audio)) < 0.5
    silent = Audio(AudioSegment.silent(1000, frame_rate=RATE))
    assert loudness(shift_pitch(silent, semitones, method="resample")) == -np.inf


@needs_ffmpeg
@pytest.mark.parametrize("channels", [1, 2])
def test_resample_octave_up_doubles_f0_and_keeps_shape(channels):
    pytest.importorskip("librosa")
    audio = _audio(_harmonic(2.0, 110.0), channels=channels)
    out = shift_pitch(audio, 12, method="resample")
    a, b = audio._get_segment(), out._get_segment()
    assert len(b.raw_data) == len(a.raw_data) and b.channels == channels
    assert _median_f0(out) == pytest.approx(220.0, rel=0.05)


# --- timing: the ratio-dependent delay is removed ------------------------------

TIMING_RATE = 44100
#: (start_s, length_s) of each harmonic tone burst; the last one runs to the end.
BURST_LENGTH_S = 0.2
BURST_STARTS_S = np.arange(0.37, 5.5, 0.61)
TIMING_SECONDS = 6.0
TAIL_S = 0.15


def _tone_bursts():
    fs = TIMING_RATE
    t = np.arange(int(TIMING_SECONDS * fs)) / fs
    wave = sum(np.sin(2 * np.pi * 220.0 * h * t) / h for h in range(1, 9))
    wave = 0.3 * wave / np.max(np.abs(wave))
    on = np.zeros_like(t, dtype=bool)
    for start in BURST_STARTS_S:
        on |= (t >= start) & (t < start + BURST_LENGTH_S)
    on |= t >= TIMING_SECONDS - TAIL_S
    return np.where(on, wave, 0.0)[:, None]


def _burst_delays_ms(signal, shifted):
    """Per-burst lag (ms) best aligning the energy envelopes of input and output."""
    fs, search = TIMING_RATE, int(0.05 * TIMING_RATE)
    delays = []
    for start in BURST_STARTS_S:
        lo, hi = int((start - 0.1) * fs), int((start + BURST_LENGTH_S + 0.1) * fs)
        ref = voice_module._energy_envelope(signal[lo:hi, 0], fs)
        res = voice_module._energy_envelope(shifted[lo - search : hi + search, 0], fs)
        delays.append((np.argmax(np.correlate(res, ref, "valid")) - search) / fs)
    return 1000 * np.array(delays)


def _check_timing(strategy, semitones, *, tolerance_ms, **kw):
    signal = _tone_bursts()
    shifted = strategy(signal, TIMING_RATE, semitones, **kw)
    assert shifted.shape == signal.shape
    delays = _burst_delays_ms(signal, shifted)
    assert abs(delays.mean()) <= tolerance_ms, delays
    # the burst running to the very end survives (no tail is lost)
    n_tail, mid = int(0.1 * TIMING_RATE), int((BURST_STARTS_S[3] + 0.05) * TIMING_RATE)

    def tail_share(x):
        rms = lambda y: np.sqrt(np.mean(y**2))  # noqa: E731
        return rms(x[-n_tail:]) / rms(x[mid : mid + n_tail])

    assert tail_share(shifted) > 0.5 * tail_share(signal)


@needs_rubberband
@pytest.mark.parametrize("preserve_formants", [True, False])
@pytest.mark.parametrize("semitones", [-12, 12, -24])
def test_rubberband_timing_is_compensated(semitones, preserve_formants):
    # Measured mean timing error differs between rubberband builds (Homebrew
    # ffmpeg@6, Debian apt, Windows): up to 3.1 ms within an octave and 5.9 ms at
    # -24 st, where per-probe delays also spread by several ms. The bounds below
    # keep clear margin over those and still fail on the uncompensated error
    # (22 ms at -12 st, 60 ms at -24 st).
    _check_timing(
        voice_module.rubberband_strategy,
        semitones,
        tolerance_ms=9.0 if abs(semitones) > 12 else 6.0,
        preserve_formants=preserve_formants,
    )


@needs_ffmpeg
@pytest.mark.parametrize("semitones", [-12, 12])
def test_resample_timing_is_compensated(semitones):
    _check_timing(
        voice_module.resample_strategy,
        semitones,
        tolerance_ms=5.0,
        preserve_formants=False,
    )


@needs_rubberband
@pytest.mark.parametrize("preserve_formants", [True, False])
def test_rubberband_octave_up(preserve_formants):
    pytest.importorskip("librosa")
    audio = _audio(_harmonic(2.0, 110.0))
    out = shift_pitch(
        audio, 12, method="rubberband", preserve_formants=preserve_formants
    )
    assert len(out._get_segment().raw_data) == len(audio._get_segment().raw_data)
    assert _median_f0(out) == pytest.approx(220.0, rel=0.05)


@needs_rubberband
def test_rubberband_pitchq_via_partial():
    from functools import partial

    audio = _audio(_harmonic(2.0, 150.0))
    quality = partial(voice_module.rubberband_strategy, pitchq="quality")
    out = shift_pitch(audio, 5, method=quality)
    assert len(out._get_segment().raw_data) == len(audio._get_segment().raw_data)
    assert abs(loudness(out) - loudness(audio)) < 0.5
    _check_timing(quality, 12, tolerance_ms=3.0, preserve_formants=True)
    with pytest.raises(ValueError, match="pitchq"):
        shift_pitch(
            audio, 5, method=partial(voice_module.rubberband_strategy, pitchq="x")
        )


def test_voice_profile_accepts_a_custom_f0_tracker():
    calls = []

    def tracker(y, fs):
        calls.append((len(y), fs))
        return np.array([100.0, 120.0, 140.0, np.nan]), np.array(
            [True, True, True, False]
        )

    audio = Audio(AudioSegment.silent(1000, frame_rate=16000))
    out = voice_profile(audio, [("a", 0.0, 0.5)], f0_tracker=tracker)
    assert calls == [(8000, 16000)]  # 0.5 s of mono, resampled to f0_sample_rate
    assert out["a"]["f0_hz"] == 120.0
    assert out["a"]["voiced_fraction"] == 0.75


def test_require_package_names_the_beats_extra_for_librosa(monkeypatch):
    import importlib

    from mixing.util import require_package

    def deny(name, *a, **k):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", deny)
    with pytest.raises(ImportError, match=r"mixing\[beats\]"):
        require_package("librosa")


def test_custom_f0_tracker_output_is_normalised():
    audio = Audio(AudioSegment.silent(1000, frame_rate=16000))

    def lists_with_nan(y, fs):  # plain lists; NaN flagged voiced must be ignored
        return [100.0, float("nan"), 140.0], [True, True, False]

    out = voice_profile(audio, f0_tracker=lists_with_nan)
    assert out["all"]["f0_hz"] == 100.0
    assert out["all"]["voiced_fraction"] == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="voiced flags"):
        voice_profile(audio, f0_tracker=lambda y, fs: ([1.0, 2.0], [True]))


def test_tracker_min_samples_sets_the_shortest_span_tracked():
    calls = []

    def tracker(y, fs):
        calls.append(len(y))
        return np.array([100.0]), np.array([True])

    audio = Audio(AudioSegment.silent(1000, frame_rate=16000))
    spans = [("a", 0.0, 0.05)]  # 800 samples, under one default pyin frame
    assert voice_profile(audio, spans, f0_tracker=tracker)["a"]["f0_hz"] is None
    tracker.min_samples = 512
    assert voice_profile(audio, spans, f0_tracker=tracker)["a"]["f0_hz"] == 100.0
    assert calls == [800]
    assert voice_module.pyin_tracker(frame_length=1024).min_samples == 1024


def test_pyin_arguments_on_voice_profile_are_deprecated(monkeypatch):
    built = []
    real = voice_module.pyin_tracker
    monkeypatch.setattr(
        voice_module, "pyin_tracker", lambda **kw: built.append(kw) or real(**kw)
    )
    audio = Audio(AudioSegment.silent(1000, frame_rate=16000))
    with pytest.warns(DeprecationWarning, match="pyin_tracker"):
        voice_profile(audio, fmin=80.0, min_voiced_prob=0.3)
    assert built[-1] == {"fmin": 80.0, "min_voiced_prob": 0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        voice_profile(audio)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="pyin_tracker"),
    ):
        voice_profile(audio, fmax=300.0, f0_tracker=lambda y, fs: ([1.0], [True]))
