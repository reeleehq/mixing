"""Loudness meter (checked against ffmpeg's ebur128) and per-turn levelling
(``mixing.audio.leveling``)."""

import re
import subprocess

import numpy as np
import pytest

pytest.importorskip("pydub")
pytest.importorskip("scipy")

from pydub import AudioSegment

from mixing.audio import Audio, level_turns, loudness, process_turns
from mixing.audio.leveling import (
    MAX_GAIN_DB,
    apply_ffmpeg_filter,
    apply_gain,
    noise_floor_db,
)
from mixing.audio.turns import check_turns
from mixing.util import ffmpeg_exe

RATE = 44100


def _voice(seconds, amplitude, *, freq=220.0, seed=0):
    """A speech-like signal: harmonics under a slow amplitude envelope, plus hiss."""
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3, 4))
    signal = amplitude * envelope * wave / 2 + 0.0005 * rng.standard_normal(len(t))
    pcm = np.clip(signal, -1, 1) * 32767
    return AudioSegment(
        pcm.astype(np.int16).tobytes(), frame_rate=RATE, sample_width=2, channels=1
    )


def _silence(seconds):
    return AudioSegment.silent(int(seconds * 1000), frame_rate=RATE)


def _ffmpeg_lufs(path):
    proc = subprocess.run(
        [
            ffmpeg_exe(),
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return float(re.findall(r"I:\s+(-?\d+\.\d+) LUFS", proc.stderr)[-1])


@pytest.mark.parametrize("amplitude", [0.05, 0.3])
def test_loudness_matches_ffmpeg_ebur128(tmp_path, amplitude):
    segment = _voice(6, amplitude)
    path = tmp_path / "v.wav"
    segment.export(str(path), format="wav")
    assert loudness(Audio(segment)) == pytest.approx(_ffmpeg_lufs(path), abs=0.1)


def test_loudness_silence_and_gain():
    assert loudness(Audio(_silence(2))) == -np.inf
    quiet = _voice(4, 0.1)
    assert loudness(Audio(quiet + 6)) - loudness(Audio(quiet)) == pytest.approx(
        6, abs=0.05
    )


def test_check_turns_rejects_overlap_and_out_of_range():
    with pytest.raises(ValueError):
        check_turns([("a", 0, 2), ("b", 1, 3)], 5)
    with pytest.raises(ValueError):
        check_turns([("a", 0, 9)], 5)
    assert check_turns([("b", 2, 3), ("a", 0, 1)], 5)[0][0] == "a"


def test_level_turns_equalises_speakers():
    gaps = _silence(0.8)
    audio = Audio(
        _voice(3, 0.5, seed=1) + gaps + _voice(3, 0.04, freq=140, seed=2) + gaps
        + _voice(3, 0.12, freq=310, seed=3)
    )  # fmt: skip
    turns = [("a", 0.0, 3.0), ("b", 3.8, 6.8), ("c", 7.6, 10.6)]
    before = [loudness(audio[s:e]) for _, s, e in turns]
    assert max(before) - min(before) > 12

    out = level_turns(audio, turns, target_lufs=-16, ceiling_db=None)
    after = [loudness(out[s:e]) for _, s, e in turns]
    assert all(abs(x - -16) < 1.0 for x in after), after
    assert out.duration == pytest.approx(audio.duration, abs=0.05)


def test_chains_and_callables_and_egress(tmp_path):
    seen = []

    def spy(audio):
        seen.append(audio.duration)
        return audio

    audio = Audio(_voice(4, 0.2))
    out = level_turns(
        audio,
        [("a", 1.0, 3.0)],
        chains={"a": spy, "z": "highpass=f=80"},
        compressor="acompressor=ratio=2",
        output=tmp_path / "leveled.wav",
    )
    assert out.exists()
    assert seen and seen[0] == pytest.approx(
        3.0, abs=0.05
    )  # turn + 0.5 s padding each side


def test_noise_floor_db_is_low_for_quiet_gaps():
    mixed = Audio(_voice(1, 0.3) + _silence(2))
    assert noise_floor_db(mixed) < -80


# --- meter accuracy: stereo, 48 kHz ---------------------------------------------


def _stereo(left, right):
    return AudioSegment.from_mono_audiosegments(left, right)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _voice(6, 0.2).set_frame_rate(48000), id="mono-48k"),
        pytest.param(
            lambda: _stereo(_voice(6, 0.3, seed=1), _voice(6, 0.08, freq=150, seed=2)),
            id="stereo-44k",
        ),
        pytest.param(
            lambda: _stereo(_voice(6, 0.1), _voice(6, 0.25, freq=330)).set_frame_rate(
                48000
            ),
            id="stereo-48k",
        ),
    ],
)
def test_loudness_matches_ffmpeg_ebur128_stereo_and_rates(tmp_path, make):
    segment = make()
    path = tmp_path / "v.wav"
    segment.export(str(path), format="wav")
    assert loudness(Audio(segment)) == pytest.approx(_ffmpeg_lufs(path), abs=0.1)


# --- level_turns regressions ----------------------------------------------------


def _peak_dbfs(audio):
    return audio._get_segment().max_dBFS


def _three_speakers():
    gaps = _silence(0.8)
    audio = Audio(
        _voice(3, 0.9, seed=1) + gaps + _voice(3, 0.06, freq=140, seed=2) + gaps
        + _voice(3, 0.2, freq=310, seed=3)
    )  # fmt: skip
    return audio, [("a", 0.0, 3.0), ("b", 3.8, 6.8), ("c", 7.6, 10.6)]


def test_default_path_hits_target_under_ceiling():
    """The default limiter must not auto-level to 0 dBFS (alimiter's default)."""
    audio, turns = _three_speakers()
    out = level_turns(audio, turns, target_lufs=-16)  # default ceiling_db=-1.5
    for _, s, e in turns:
        assert loudness(out[s:e]) == pytest.approx(-16, abs=0.5)
    assert _peak_dbfs(out) <= -1.5 + 0.1


def test_limiter_holds_the_true_peak_not_just_the_sample_peak():
    # an fs/4 tone at 45 degrees: its true peak sits 3 dB above every sample,
    # so a sample-peak limiter (the old alimiter stage) let it 3 dB over
    from mixing.audio import true_peak_db

    n = np.arange(3 * RATE)
    tone = np.sin(2 * np.pi * n / 4 + np.pi / 4) * 0.3
    pcm = (tone * 32767).astype(np.int16).tobytes()
    audio = Audio(AudioSegment(pcm, frame_rate=RATE, sample_width=2, channels=1))
    out = level_turns(
        audio, [("a", 0.0, 3.0)], target_lufs=0, compressor=None, denoise=None
    )
    assert true_peak_db(out) <= -1.5 + 0.1  # the default ceiling_db


@pytest.mark.parametrize("time_unit", ["milliseconds", "samples"])
def test_time_unit_does_not_break_slicing(time_unit):
    audio, turns = _three_speakers()
    ref = level_turns(audio, turns, ceiling_db=None, denoise=None)
    out = level_turns(
        Audio(audio._get_segment(), time_unit=time_unit),
        turns,
        ceiling_db=None,
        denoise=None,
    )
    assert out.time_unit == time_unit
    assert out._get_segment().raw_data == ref._get_segment().raw_data
    assert out.duration == pytest.approx(audio.duration, abs=1e-3)


def _onset_s(audio, start, end, *, fraction=0.3):
    """Time (s, from ``start``) the signal first exceeds ``fraction`` of its peak."""
    seg = audio._get_segment()
    data = np.abs(np.array(seg.get_array_of_samples(), dtype=float))
    window = data[int(start * RATE) : int(end * RATE)]
    return int(np.argmax(window > fraction * window.max())) / RATE


def test_denoise_and_compressor_keep_onsets_in_place():
    """afftdn delays ~25 ms; levelling must not shift a turn's content."""
    rng = np.random.default_rng(0)
    pieces = []
    for i in range(3):
        hiss = (0.003 * rng.standard_normal(int(0.4 * RATE)) * 32767).astype(np.int16)
        pieces.append(
            AudioSegment(hiss.tobytes(), frame_rate=RATE, sample_width=2, channels=1)
            + _voice(1.5, 0.3, seed=i)
            + _silence(0.5)
        )
    audio = Audio(sum(pieces[1:], pieces[0]))
    turns = [("s", i * 2.4 + 0.05, i * 2.4 + 2.2) for i in range(3)]
    out = level_turns(audio, turns, denoise_above=-200, ceiling_db=None)
    for _, s, e in turns:
        before, after = _onset_s(audio, s, e), _onset_s(out, s, e)
        assert abs(after - before) < 0.005, (before, after)


def test_many_turns_join_is_linear():
    """2000 turns over 10 min: a quadratic join (``out += piece``) copies ~20 GB
    and takes seconds; the single ``b"".join`` takes ~25 ms."""
    import time

    audio = Audio(AudioSegment.silent(600_000, frame_rate=16000))
    turns = [("ab"[i % 2], i * 0.3, i * 0.3 + 0.2) for i in range(2000)]
    t0 = time.perf_counter()
    out = process_turns(audio, turns, lambda speaker, a, core: a, pad_s=0.05)
    assert time.perf_counter() - t0 < 1.0
    assert out._get_segment().raw_data == audio._get_segment().raw_data


def test_gain_is_capped_with_a_warning():
    audio = Audio(_voice(3, 0.0005, seed=4) + _silence(0.5))
    with pytest.warns(UserWarning, match="capped"):
        out = level_turns(
            audio, [("q", 0.0, 3.0)], compressor=None, denoise=None, ceiling_db=None
        )
    gained = loudness(out[0:3]) - loudness(audio[0:3])
    assert gained == pytest.approx(MAX_GAIN_DB, abs=0.5)


@pytest.mark.parametrize("delta_s", [-0.7, 0.4])
def test_stage_changing_length_keeps_total_duration(delta_s):
    def shrink_or_grow(audio):
        seg = audio._get_segment()
        if delta_s < 0:
            return Audio(seg[: int((audio.duration + delta_s) * 1000)])
        return Audio(seg + AudioSegment.silent(int(delta_s * 1000), RATE))

    audio, turns = _three_speakers()
    out = level_turns(
        audio,
        turns,
        chains={"a": shrink_or_grow, "b": shrink_or_grow},
        ceiling_db=None,
        denoise=None,
    )
    raw_in, raw_out = audio._get_segment().raw_data, out._get_segment().raw_data
    assert len(raw_out) == len(raw_in)
    # the untouched gaps stay byte-identical, so nothing after a turn shifted
    gap = slice(int(3.2 * RATE) * 2, int(3.6 * RATE) * 2)
    assert raw_out[gap] == raw_in[gap]


def test_process_turns_identity_is_lossless_and_fade_longer_than_turn():
    audio, turns = _three_speakers()
    same = process_turns(audio, turns, lambda speaker, a, core: a)
    assert same._get_segment().raw_data == audio._get_segment().raw_data
    out = level_turns(
        audio, [("x", 1.0, 1.01)], fade_s=5, compressor=None, denoise=None
    )
    assert len(out._get_segment().raw_data) == len(audio._get_segment().raw_data)


def test_apply_ffmpeg_filter_keeps_sample_width():
    seg = _voice(1, 0.2).set_sample_width(4)
    out = apply_ffmpeg_filter(Audio(seg), "volume=0.5")
    out_seg = out._get_segment()
    assert out_seg.sample_width == 4
    assert len(out_seg.raw_data) == len(seg.raw_data)


# --- float headroom: nothing clips before the limiter ---------------------------


def _pcm(signal, *, rate=RATE):
    pcm = np.clip(signal, -1, 1) * 32767
    return AudioSegment(
        pcm.astype(np.int16).tobytes(), frame_rate=rate, sample_width=2, channels=1
    )


def _bursty_speaker():
    """A quiet speaker with 2 ms loud sine bursts each second (very high crest)."""
    t = np.arange(6 * RATE) / RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    x = (
        0.03
        * envelope
        * sum(np.sin(2 * np.pi * 220 * h * t) / h for h in (1, 2, 3, 4))
        / 2
    )
    n = int(0.002 * RATE)
    for k in range(1, 6):
        x[k * RATE : k * RATE + n] += 0.6 * np.sin(
            2 * np.pi * 2900 * np.arange(n) / RATE
        )
    pad = np.zeros(RATE // 2)
    return Audio(_pcm(np.concatenate([pad, x, pad]))), [("q", 0.5, 6.5)]


def _flat_top_count(audio):
    """Samples within 1 LSB of the peak: a hard clip leaves hundreds."""
    samples = np.abs(np.array(audio._get_segment().get_array_of_samples(), dtype=int))
    return int((samples >= samples.max() - 1).sum())


def test_high_crest_gains_do_not_clip_without_ceiling():
    """Pre-gain to -6 LUFS pushes the bursts ~18 dB above 0 dBFS; gaining back
    down to -30 LUFS must restore them intact (float headroom between stages)."""
    audio, turns = _bursty_speaker()
    out = level_turns(
        audio, turns, pregain_lufs=-6, target_lufs=-30,
        compressor=None, denoise=None, ceiling_db=None,
    )  # fmt: skip
    assert _flat_top_count(out) <= _flat_top_count(audio) + 2  # integer path: ~435
    assert _peak_dbfs(out) < 0
    assert loudness(out[0.5:6.5]) == pytest.approx(-30, abs=0.1)


def test_high_crest_default_path_has_only_the_limiters_ceiling():
    audio, turns = _bursty_speaker()
    out = level_turns(audio, turns)
    assert _peak_dbfs(out) <= -1.5 + 0.1
    assert _flat_top_count(out) <= 10  # the integer pipeline left ~435


def test_apply_gain_is_float_then_saturates_once():
    seg = _voice(1, 0.5)
    back = apply_gain(apply_gain(Audio(seg), -12), 12)
    diff = np.abs(
        np.array(back._get_segment().get_array_of_samples(), dtype=int)
        - np.array(seg.get_array_of_samples(), dtype=int)
    )
    assert diff.max() <= 2


# --- latency compensation only where the filter has a fixed delay ---------------


def _onset_hp_s(audio, start, end, *, fraction=0.3, cutoff=150.0):
    """Onset (s, from ``start``) of the content above ``cutoff`` Hz (zero-phase)."""
    from scipy.signal import butter, sosfiltfilt

    seg = audio._get_segment()
    data = np.array(seg.get_array_of_samples(), dtype=float)
    data = sosfiltfilt(butter(4, cutoff, "highpass", fs=RATE, output="sos"), data)
    window = np.abs(data[int(start * RATE) : int(end * RATE)])
    return int(np.argmax(window > fraction * window.max())) / RATE


def test_hum_does_not_misalign_chain_or_denoise():
    """Strong 60 Hz hum makes signal cross-correlation periodic; an EQ chain
    must not be "compensated" at all, and denoise only by its probed delay."""
    t = np.arange(6 * RATE) / RATE
    x = 0.3 * np.sin(2 * np.pi * 60 * t)
    for onset in (1.0, 3.6):
        span = (t >= onset) & (t < onset + 1.5)
        x[span] += (
            0.3
            * sum(np.sin(2 * np.pi * 220 * h * t[span]) / h for h in (1, 2, 3, 4))
            / 2
        )
    audio = Audio(_pcm(x))
    turns = [("s", 0.5, 2.8), ("s", 3.1, 5.4)]
    out = level_turns(audio, turns, chains={"s": "highpass=f=80"}, denoise_above=-200)
    for _, s, e in turns:
        before, after = _onset_hp_s(audio, s, e), _onset_hp_s(out, s, e)
        assert abs(after - before) < 0.005, (before, after)


# --- sample formats -------------------------------------------------------------


def test_8bit_input_round_trips_and_levels():
    seg = _voice(4, 0.2).set_sample_width(1)
    same = apply_ffmpeg_filter(Audio(seg), "volume=1")
    a = np.array(seg.get_array_of_samples(), dtype=int)
    b = np.array(same._get_segment().get_array_of_samples(), dtype=int)
    assert np.abs(a - b).max() <= 1  # pydub's 8-bit is signed; read as such
    out = level_turns(Audio(seg), [("a", 0.5, 3.5)], denoise=None)
    assert out._get_segment().sample_width == 1
    assert len(out._get_segment().raw_data) == len(seg.raw_data)
    assert loudness(out[0.5:3.5]) == pytest.approx(-16, abs=1.0)


def test_stereo_through_level_turns():
    gaps = _silence(0.8)
    left = _voice(3, 0.5, seed=1) + gaps + _voice(3, 0.04, freq=140, seed=2)
    right = _voice(3, 0.3, seed=3) + gaps + _voice(3, 0.06, freq=150, seed=4)
    audio = Audio(_stereo(left, right))
    turns = [("a", 0.0, 3.0), ("b", 3.8, 6.8)]
    out = level_turns(audio, turns)
    assert out._get_segment().channels == 2
    assert out.duration == pytest.approx(audio.duration, abs=1e-3)
    for _, s, e in turns:
        assert loudness(out[s:e]) == pytest.approx(-16, abs=1.0)


def test_24bit_wav_through_level_turns(tmp_path):
    path = tmp_path / "v24.wav"
    (_voice(3, 0.5, seed=1) + _silence(0.8) + _voice(3, 0.06, freq=140, seed=2)).export(
        str(path), format="wav", parameters=["-acodec", "pcm_s24le"]
    )
    audio = Audio(str(path))
    turns = [("a", 0.0, 3.0), ("b", 3.8, 6.8)]
    out = level_turns(audio, turns)
    assert out.duration == pytest.approx(audio.duration, abs=1e-3)
    assert len(out._get_segment().raw_data) == len(audio._get_segment().raw_data)
    for _, s, e in turns:
        assert loudness(out[s:e]) == pytest.approx(-16, abs=1.0)


# --- validation and warnings ----------------------------------------------------


@pytest.mark.parametrize("ceiling_db", [0.5, -30.0])
def test_ceiling_db_out_of_range_raises(ceiling_db):
    audio, turns = _three_speakers()
    with pytest.raises(ValueError, match="ceiling_db"):
        level_turns(audio, turns, ceiling_db=ceiling_db)


def test_turn_below_gate_is_left_untouched_with_warning():
    t = np.arange(2 * RATE) / RATE
    audio = Audio(_pcm(1e-4 * np.sin(2 * np.pi * 1000 * t)))  # ~3 LSB, < -70 LUFS
    with pytest.warns(UserWarning, match="gate"):
        out = level_turns(audio, [("q", 0.2, 1.8)])
    assert out._get_segment().raw_data == audio._get_segment().raw_data


# --- chunked metering, LRA, true peak -----------------------------------------


def test_chunked_loudness_matches_single_pass():
    from mixing.loudness import numpy_lufs as integrated_lufs
    from mixing.audio.samples import audio_samples

    mono, fs = audio_samples(Audio(_voice(4, 0.3)))
    data = np.hstack([mono, 0.5 * mono[::-1]])  # stereo
    whole = integrated_lufs(data, fs, chunk_s=1e6)
    assert integrated_lufs(data, fs, chunk_s=0.37) == pytest.approx(whole, abs=1e-9)


def test_loudness_range_separates_quiet_and_loud_passages():
    from mixing.audio import loudness_range

    quiet, loud = _voice(8, 0.05), _voice(8, 0.4, seed=1)
    assert loudness_range(Audio(quiet)) < 2.0
    mixed = Audio(quiet + loud)
    assert loudness_range(mixed) > 10.0
    # EBU Tech 3342 test 1: 20 s at -20 LUFS then 20 s at -30 LUFS -> 10 LU
    step = Audio(_voice(20, 0.3))
    first = apply_gain(step, -20.0 - loudness(step))
    second = apply_gain(step, -30.0 - loudness(step))
    assert loudness_range(first + second) == pytest.approx(10.0, abs=0.5)
    assert loudness_range(Audio(AudioSegment.silent(5000))) == 0.0
    assert loudness_range(Audio(_voice(1, 0.3))) == 0.0  # shorter than one window


def test_true_peak_exceeds_sample_peak_for_intersample_peaks():
    from mixing.audio import true_peak_db

    # fs/4 sine with a 45 degree phase: samples sit at +-0.707 of the true peak
    n = np.arange(RATE)
    pcm = (np.sin(2 * np.pi * n / 4 + np.pi / 4) * 0.9 * 32767).astype(np.int16)
    seg = AudioSegment(pcm.tobytes(), frame_rate=RATE, sample_width=2, channels=1)
    sample_peak_db = 20 * np.log10(np.abs(pcm).max() / 32768)
    assert true_peak_db(Audio(seg)) > sample_peak_db + 1.0
    assert true_peak_db(Audio(AudioSegment.silent(500))) == -np.inf


def test_true_peak_is_independent_of_chunking(monkeypatch):
    from mixing.audio import true_peak_db
    import mixing.loudness as lv

    seg = _voice(3, 0.9)
    whole = true_peak_db(Audio(seg))
    monkeypatch.setattr(lv, "FILTER_CHUNK_S", 0.37)
    assert true_peak_db(Audio(seg)) == pytest.approx(whole, abs=1e-6)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_sample_conversions_scale_by_the_sample_width(width):
    # AudioSamples and the ndarray conversion divided by 32768 whatever the width
    from mixing.audio.audio_ops import AudioSamples
    from mixing.audio.audio_util import _normalize_audio_source

    half = AudioSegment(
        np.array([1 << (8 * width - 2)], dtype={1: "i1", 2: "<i2", 4: "<i4"}[width])
        .tobytes(),
        frame_rate=RATE, sample_width=width, channels=1,
    )  # fmt: skip
    assert AudioSamples(half)[0] == pytest.approx(0.5)
    assert _normalize_audio_source(half, target_type="ndarray")[0] == pytest.approx(0.5)


def test_measurement_and_gain_helpers_accept_paths(tmp_path):
    path = tmp_path / "voice.wav"
    _voice(1, 0.3).export(path, format="wav")
    audio = Audio(str(path))
    assert noise_floor_db(path) == noise_floor_db(audio)
    assert apply_gain(path, -6).duration == pytest.approx(audio.duration)
    assert apply_ffmpeg_filter(path, "volume=0.5").duration == pytest.approx(
        audio.duration
    )


def test_audio_is_not_padded_to_a_whole_millisecond():
    # 147016 frames at 44.1 kHz is 3333.7 ms; pydub's ms slicing padded it
    # with 13 frames of silence (to 3334 ms), so outputs outgrew their source
    n = 147016
    seg = AudioSegment(
        np.zeros(n, dtype="<i2").tobytes(), frame_rate=RATE, sample_width=2, channels=1
    )
    audio = Audio(seg)
    assert audio._get_segment().frame_count() == n
    assert audio[0 : audio.duration]._get_segment().frame_count() == n
    assert audio[1.0:]._get_segment().frame_count() == n - RATE


def test_turns_may_be_segments_labelled_by_speaker():
    from mixing.audio.segmentation import Segment

    audio, turns = _three_speakers()
    as_segments = [Segment(start, end, label=speaker) for speaker, start, end in turns]
    ref = level_turns(audio, turns, ceiling_db=None, denoise=None)
    out = level_turns(audio, as_segments, ceiling_db=None, denoise=None)
    assert out._get_segment().raw_data == ref._get_segment().raw_data


def test_a_turn_may_end_at_the_reported_duration():
    # 147016 frames: duration reads 3.334 s, the last frame ends at 3.33370 s
    seg = AudioSegment(
        np.zeros(147016, dtype="<i2").tobytes(), frame_rate=RATE, sample_width=2,
        channels=1,
    )  # fmt: skip
    audio = Audio(seg)
    out = process_turns(audio, [("a", 0.0, audio.duration)], lambda s, a, c: a)
    assert out._get_segment().frame_count() == 147016
