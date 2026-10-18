"""Tests for the learned ``"audioset"`` segmentation strategy and its helpers.

Offline: the classifier seam is fed a stub that reads the synthetic signal
(loud tone = music, quiet noise = talk), so the pipeline (thresholding,
smoothing, runs, boundary snapping, song/jingle labels) is tested without
torch. The real AST model is exercised only by the ``slow`` tests at the bottom,
which skip unless ``MIXING_RUN_SLOW=1`` and the ``ml`` extra is installed.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("pydub")
pytest.importorskip("scipy")

from pydub import AudioSegment

from mixing.audio import (
    Segment,
    WindowProbabilities,
    ast_audioset_classifier,
    find_segments,
    segment_by_audioset,
    text_before_segments,
)
from mixing.audio.tagging import window_starts
from mixing.srt import Cue, parse_vtt_words

SR = 8000
WIN, HOP = 4.0, 1.0


def _to_segment(x: np.ndarray, sr: int = SR) -> AudioSegment:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)


def _tone(seconds, freq=440.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _talk(seconds, amp=0.05, sr=SR, seed=0):
    return amp * np.random.default_rng(seed).standard_normal(int(seconds * sr))


def _silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr))


def _show():
    """talk 20 s | song 90 s | talk 20 s | jingle 15 s | talk 15 s, with 0.3 s gaps."""
    parts = [
        _talk(19.7),
        _silence(0.3),
        _tone(90.0),
        _silence(0.3),
        _talk(19.7, seed=1),
        _silence(0.3),
        _tone(15.0, freq=660),
        _silence(0.3),
        _talk(14.4, seed=2),
    ]
    return _to_segment(np.concatenate(parts))


def loudness_classifier(samples, sr):
    """Stub classifier: windows whose RMS is high are music, the rest speech."""
    starts = window_starts(len(samples), sr, window_seconds=WIN, hop_seconds=HOP)
    win = int(WIN * sr)
    rms = np.array([np.sqrt(np.mean(samples[s : s + win] ** 2)) for s in starts])
    music = np.clip(rms / 0.2, 0, 1)
    t = starts / sr
    return WindowProbabilities(t, t + WIN, 1 - music, music)


def test_finds_song_and_jingle_with_stub_classifier():
    segs = find_segments(
        _show(),
        strategy="audioset",
        classifier=loudness_classifier,
        min_song_duration=60,
    )
    assert [s.label for s in segs] == ["speech", "song", "speech", "jingle", "speech"]
    song, jingle = segs[1], segs[3]
    assert song.start == pytest.approx(20.0, abs=0.5)
    assert song.end == pytest.approx(110.0, abs=0.5)
    assert jingle.start == pytest.approx(130.0, abs=0.5)
    assert jingle.end == pytest.approx(145.0, abs=0.5)
    assert 0.5 < song.score <= 1.0


def test_segments_are_back_to_back_and_cover_the_audio():
    audio = _show()
    segs = segment_by_audioset(audio, classifier=loudness_classifier)
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(len(audio) / 1000.0)
    for a, b in zip(segs, segs[1:]):
        assert a.end == b.start
        assert a.label != b.label


def test_without_min_song_duration_music_is_labelled_music():
    segs = segment_by_audioset(_show(), classifier=loudness_classifier)
    assert [s.label for s in segs] == ["speech", "music", "speech", "music", "speech"]


def test_custom_labels():
    segs = segment_by_audioset(
        _show(),
        classifier=loudness_classifier,
        speech_label="talk",
        min_song_duration=60,
        song_label="track",
        jingle_label="sting",
    )
    assert [s.label for s in segs] == ["talk", "track", "talk", "sting", "talk"]


def test_boundaries_snap_to_the_quiet_gap():
    # Coarse windows put the raw music onset at a window edge (18 s or 20 s);
    # snapping must move it into the 0.3 s silence centred on 19.85 s.
    snapped = segment_by_audioset(_show(), classifier=loudness_classifier)
    unsnapped = segment_by_audioset(
        _show(), classifier=loudness_classifier, snap_radius=0
    )
    assert unsnapped[1].start in (16.0, 17.0, 18.0, 19.0, 20.0)
    assert 19.7 <= snapped[1].start <= 20.0


def test_two_boundaries_near_one_quiet_point_coincide():
    from mixing.audio.segmentation import _snap_boundaries

    x = np.concatenate([_tone(10.0), _silence(0.2), _tone(10.0)])
    snapped = _snap_boundaries([9.0, 11.5], x, sr=SR, radius=4.0, total_duration=20.2)
    assert snapped[0] == snapped[1]
    assert 10.0 <= snapped[0] <= 10.2


def test_classifier_receives_mono_samples_at_native_or_requested_rate():
    seen = {}

    def spy(samples, sr):
        seen["shape"], seen["sr"], seen["dtype"] = samples.shape, sr, samples.dtype
        return loudness_classifier(samples, sr)

    stereo = _show().set_channels(2)
    segment_by_audioset(stereo, classifier=spy)  # default: the AST's 16 kHz
    assert seen["sr"] == 16000 and len(seen["shape"]) == 1
    assert seen["dtype"] == np.float32
    segment_by_audioset(stereo, classifier=spy, sample_rate=None)
    assert seen["sr"] == SR
    segment_by_audioset(stereo, classifier=spy, sample_rate=4000)
    assert seen["sr"] == 4000


def _probs_from_flags(flags, *, hop=2.0, win=5.0):
    t = np.arange(len(flags)) * hop
    music = np.asarray(flags, dtype=float)
    return WindowProbabilities(t, t + win, 1 - music, music)


def test_a_run_collapsed_by_snapping_does_not_leave_two_speech_segments():
    # One music window whose only quiet point (a 0.4 s gap) attracts both of its
    # boundaries: the music run vanishes and the talk around it must re-join.
    x = np.concatenate([_talk(20.0, amp=0.3), _silence(0.4), _talk(24.6, amp=0.3)])
    flags = [0] * 10 + [1] + [0] * 11

    segs = segment_by_audioset(
        _to_segment(x),
        classifier=lambda s, sr: _probs_from_flags(flags),
        smooth_windows=1,
    )
    labels = [s.label for s in segs]
    assert all(not (a == b == "speech") for a, b in zip(labels, labels[1:]))
    assert segs[0].start == 0.0 and segs[-1].end == pytest.approx(45.0)


def test_even_smooth_windows_behave_like_the_next_odd_size():
    flags = [0] * 8 + [1] * 2 + [0] * 8
    audio = AudioSegment.silent(40_000, frame_rate=SR)

    def run(size):
        return [
            (s.label, s.start, s.end)
            for s in segment_by_audioset(
                audio,
                classifier=lambda s, sr: _probs_from_flags(flags),
                smooth_windows=size,
                snap_radius=0,
            )
        ]

    assert run(2) == run(3)
    assert run(4) == run(5)


def test_short_talk_blip_inside_music_is_smoothed_away():
    def blip(samples, sr):
        t = np.arange(0.0, 56.0, 2.0)
        music = np.ones_like(t)
        music[10] = 0.0  # one talk window in the middle of a song
        return WindowProbabilities(t, t + 5.0, 1 - music, music)

    audio = AudioSegment.silent(60_000, frame_rate=SR)
    assert [s.label for s in segment_by_audioset(audio, classifier=blip)] == ["music"]
    # Without smoothing the blip survives, but never as a negative-length segment.
    raw = segment_by_audioset(audio, classifier=blip, smooth_windows=1, snap_radius=0)
    assert all(s.duration > 0 for s in raw)
    assert all(a.end == b.start for a, b in zip(raw, raw[1:]))


def test_no_windows_means_one_speech_segment():
    def empty(samples, sr):
        e = np.array([])
        return WindowProbabilities(e, e, e, e)

    segs = segment_by_audioset(AudioSegment.silent(3000), classifier=empty)
    assert [(s.label, s.start, s.end) for s in segs] == [("speech", 0.0, 3.0)]


def test_window_probabilities_validates_lengths():
    with pytest.raises(ValueError, match="same length"):
        WindowProbabilities(np.zeros(2), np.ones(2), np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="ascending"):
        z = np.zeros(2)
        WindowProbabilities(np.array([4.0, 2.0]), np.array([9.0, 7.0]), z, z)
    with pytest.raises(ValueError, match="ends > starts"):
        z = np.zeros(1)
        WindowProbabilities(np.array([4.0]), np.array([4.0]), z, z)


def test_unknown_strategy_lists_audioset():
    with pytest.raises(ValueError, match="audioset"):
        find_segments(AudioSegment.silent(1000), strategy="nope")


def test_default_classifier_names_the_ml_extra_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    classify = ast_audioset_classifier()  # building it is free; calling it needs torch
    with pytest.raises(ImportError, match=r"mixing\[ml\]"):
        classify(np.zeros(SR, dtype=np.float32), SR)


def test_importing_mixing_audio_does_not_import_torch():
    code = (
        "import sys, mixing.audio; "
        "from mixing.audio import find_segments, ast_audioset_classifier; "
        "print([m for m in ('torch', 'transformers') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_text_before_segments_with_dicts_cues_and_tuples():
    words = [
        {"text": "and", "start": 70.0},
        {"text": "now,", "start": 95.0},
        {"text": " ", "start": 95.4, "type": "spacing"},
        {"text": "Moments.", "start": 97.0},
        {"text": "la", "start": 104.0},
    ]
    songs = [Segment(100.0, 280.0, label="song"), Segment(400.0, 500.0, label="song")]
    assert text_before_segments(songs, words) == ["now, Moments.", ""]
    cues = [Cue(1, 90.0, 92.0, "Here is\na song"), Cue(2, 120.0, 121.0, "late")]
    assert text_before_segments([(100.0, 200.0)], cues, before_seconds=15) == [
        "Here is a song"
    ]


def test_parse_vtt_words_youtube_rolling_captions():
    vtt = (
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:12.240 --> 00:00:15.509 align:start position:0%\n \n"
        "Music<00:00:12.639><c> with</c><00:00:12.800><c> meaning</c>\n\n"
        "00:00:15.509 --> 00:00:15.519 align:start position:0%\n"
        "Music with meaning\n \n\n"
        "00:00:15.519 --> 00:00:17.510 align:start position:0%\n"
        "Music with meaning\n"
        "can<00:00:16.560><c> be</c>\n"
    )
    words = parse_vtt_words(vtt)
    assert [w["text"] for w in words] == ["Music", "with", "meaning", "can", "be"]
    assert [w["start"] for w in words] == [12.24, 12.639, 12.8, 15.519, 16.56]
    assert all(w["end"] >= w["start"] for w in words)
    assert words[2]["end"] == pytest.approx(15.509)  # capped at its cue end


def test_parse_vtt_words_ignores_identifiers_and_notes_and_handles_variants():
    vtt = (
        "WEBVTT\n\nNOTE a comment\nspanning lines\n\n"
        "intro\n00:01.000 --> 00:02.500\nRock &amp; roll\n\n"
        "2\r\n00:00:03.000 --> 00:00:06.000\r\n"
        "<c>[Music]</c><00:00:03.500><c.colorE5E5E5> so</c>\r\n"
        "next<00:00:04.500><c> line</c>\r\n"
    )
    words = parse_vtt_words(vtt)
    assert [(w["text"], w["start"]) for w in words] == [
        ("[Music]", 3.0),
        ("so", 3.5),
        ("next", 3.5),  # later line of a cue starts at the latest timing seen
        ("line", 4.5),
    ]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    # No inline timings anywhere: one entry per cue, identifiers and NOTEs dropped.
    plain = parse_vtt_words(
        "WEBVTT\n\nNOTE x\n\n1\n00:01.000 --> 00:02.500\nRock &amp; roll\n\n2\n"
    )
    assert [w["text"] for w in plain] == ["Rock & roll"]


def test_text_before_segments_accepts_any_object_with_start():
    class Span:
        start, end = 10.0, 20.0

    assert text_before_segments([Span()], [{"text": "hi", "start": 9.0}]) == ["hi"]


# ----------------------------------------------------------------------------
# Real model (slow; opt in with MIXING_RUN_SLOW=1 and the ``ml`` extra)
# ----------------------------------------------------------------------------

_slow = pytest.mark.skipif(
    os.environ.get("MIXING_RUN_SLOW") != "1",
    reason="real-model test; set MIXING_RUN_SLOW=1 (needs mixing[ml] + a model download)",
)


@pytest.mark.slow
@_slow
def test_real_ast_classifier_shapes():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    x = _tone(12.0, sr=16000).astype(np.float32)
    probs = ast_audioset_classifier()(x, 16000)
    assert len(probs) == len(
        window_starts(len(x), 16000, window_seconds=5, hop_seconds=2)
    )
    for arr in (probs.speech, probs.music):
        assert np.all((arr >= 0) & (arr <= 1))


@pytest.mark.slow
@_slow
def test_real_radio_tape_song_count():
    """Point MIXING_AUDIOSET_TAPE at a ~30 min radio show (talk, jingles, songs)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tape = os.environ.get("MIXING_AUDIOSET_TAPE")
    if not tape or not os.path.exists(tape):
        pytest.skip("set MIXING_AUDIOSET_TAPE to a radio-show audio file")
    t0 = time.time()
    segs = find_segments(tape, strategy="audioset", min_song_duration=75)
    elapsed = time.time() - t0
    songs = [s for s in segs if s.label == "song"]
    assert 4 <= len(songs) <= 10
    assert elapsed < segs[-1].end / 2  # well under realtime
