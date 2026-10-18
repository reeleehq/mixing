"""Tests for :mod:`mixing.audio.talk_edges` (host talk at the edges of a carved song).

Offline: the classifier seam is fed either hand-built ``WindowProbabilities`` or a
stub that reads the synthetic signal (white noise = talk, a pure tone = music), so
the logic (hysteresis, song body, intro/outro, word snapping, segment offsets) runs
without torch. The real AST model runs only in the ``slow`` test at the bottom,
against the hand labels in ``data/talk_edges_labels.json``; it skips unless
``MIXING_RUN_SLOW=1``, the ``ml`` extra is installed, and ``MIXING_MWM_TAPES``
points at the tapes those labels describe.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydub")
pytest.importorskip("scipy")

from pydub import AudioSegment

from mixing.audio import (
    Segment,
    TalkEdges,
    WindowProbabilities,
    find_talk_edges,
    talk_edges_from_probs,
)
from mixing.audio import talk_edges as te
from mixing.audio.tagging import window_starts
from mixing.srt import Cue

SR = 16000  # the default analysis rate: no resampling in the stub tests
WIN, HOP = te.TALK_WINDOW_SECONDS, te.TALK_HOP_SECONDS
LABELS = Path(__file__).parent / "data" / "talk_edges_labels.json"


def _probs(speech_of_t, duration):
    """Windows every HOP over ``duration`` with P(speech) = speech_of_t(window start)."""
    t = np.arange(0.0, duration - WIN + 1e-9, HOP)
    speech = np.array([speech_of_t(x) for x in t], dtype=float)
    return WindowProbabilities(t, t + WIN, speech, 1 - speech)


def _edges(speech_of_t, duration=120.0, **kw):
    return talk_edges_from_probs(
        _probs(speech_of_t, duration), start=0.0, end=duration, **kw
    )


# ----------------------------------------------------------------------------
# The audio logic, on hand-built probabilities
# ----------------------------------------------------------------------------


def test_intro_and_outro_talk_are_found_and_clean_span_is_between():
    edges = _edges(lambda t: 0.8 if t < 10 or t > 110 else 0.1)
    assert edges.intro.start == 0.0 and edges.intro.label == "talk"
    assert edges.intro.end == pytest.approx(10.0, abs=1.0)
    assert edges.outro.start == pytest.approx(112.0, abs=1.0)
    assert edges.outro.end == 120.0
    assert (edges.clean_start, edges.clean_end) == (edges.intro.end, edges.outro.start)
    assert edges.clean == Segment(edges.clean_start, edges.clean_end, label="music")
    assert not edges.snapped_to_words


def test_a_short_sting_between_talk_stays_inside_the_intro():
    # skit tail 0-1 s, a sting, then the host over a bed 7-14 s, song from 14 s
    def speech(t):
        return 0.7 if t < 1 or 6 <= t < 13 else 0.05

    edges = _edges(speech)
    assert edges.intro.start == 0.0
    assert edges.intro.end == pytest.approx(14.0, abs=1.0)


def test_singing_that_reads_as_some_speech_is_not_talk():
    # sung vocals: P(speech) 0.4-0.55 throughout, never reaching the seed threshold
    edges = _edges(lambda t: 0.4 + 0.15 * (np.sin(t) > 0))
    assert edges.intro is None and edges.outro is None
    assert (edges.clean_start, edges.clean_end) == (0.0, 120.0)


def test_talk_inside_the_song_body_is_never_an_edge():
    edges = _edges(lambda t: 0.9 if 50 <= t < 55 else 0.1)
    assert edges.intro is None and edges.outro is None


def _probs_sm(speech_of_t, music_of_t, duration=120.0):
    t = np.arange(0.0, duration - WIN + 1e-9, HOP)
    return WindowProbabilities(
        t,
        t + WIN,
        np.array([speech_of_t(x) for x in t], dtype=float),
        np.array([music_of_t(x) for x in t], dtype=float),
    )


def test_hysteresis_extends_a_seeded_region_down_to_the_floor():
    # host over a bed at 4-12 s (after a short instrumental opening): mostly
    # P(speech) 0.5, one window at 0.7; the bed keeps P(music) at 0.3
    def speech(t):
        if 4 <= t < 12:
            return 0.7 if t == 8.0 else 0.5
        return 0.1

    probs = _probs_sm(speech, lambda t: 0.3)
    edges = talk_edges_from_probs(probs, start=0.0, end=120.0)
    assert edges.intro.end == pytest.approx(12.0, abs=1.0)
    # 0.5 everywhere: never reaches the seed bar (0.45 + 0.4 * 0.3 = 0.57)
    flat = _probs_sm(lambda t: 0.5 if 4 <= t < 12 else 0.1, lambda t: 0.3)
    assert talk_edges_from_probs(flat, start=0.0, end=120.0).intro is None
    assert talk_edges_from_probs(flat, start=0.0, end=120.0, talk_threshold=0.35).intro


def test_the_seed_bar_rises_with_music():
    # the same P(speech) 0.6 is talk over a quiet bed, singing over a full band
    def speech(t):
        return 0.6 if 4 <= t < 12 else 0.1

    quiet = _probs_sm(speech, lambda t: 0.2)  # bar 0.53
    loud = _probs_sm(speech, lambda t: 0.8)  # bar 0.77
    assert talk_edges_from_probs(quiet, start=0.0, end=120.0).intro is not None
    assert talk_edges_from_probs(loud, start=0.0, end=120.0).intro is None
    no_penalty = talk_edges_from_probs(loud, start=0.0, end=120.0, music_penalty=0.0)
    assert no_penalty.intro is not None


def test_talk_under_way_at_the_cut_needs_only_the_edge_threshold():
    # the host is already talking when the song starts / still talking at its
    # end, over loud music: P(speech) 0.65 with P(music) 0.8
    def speech(t):
        return 0.65 if t < 6 or t > 110 else 0.1

    probs = _probs_sm(speech, lambda t: 0.8)
    edges = talk_edges_from_probs(probs, start=0.0, end=120.0)
    assert edges.intro is not None and edges.outro is not None
    strict = talk_edges_from_probs(probs, start=0.0, end=120.0, edge_threshold=0.7)
    assert strict.intro is None and strict.outro is None


def test_a_late_intro_beyond_max_edge_seconds_is_ignored():
    # talk at 20-30 s after an instrumental opening shorter than min_music_seconds
    def speech(t):
        return 0.8 if 5 <= t < 10 else 0.1

    assert _edges(speech).intro is not None
    assert _edges(speech, max_edge_seconds=4.0).intro is None


def test_no_music_body_means_all_talk_and_an_empty_clean_span():
    edges = _edges(lambda t: 0.8, duration=30.0)
    assert edges.intro == Segment(0.0, 30.0, label="talk")
    assert edges.outro is None
    assert edges.clean_start == edges.clean_end == 30.0


def test_no_windows_and_no_talk():
    empty = np.array([])
    probs = WindowProbabilities(empty, empty, empty, empty)
    edges = talk_edges_from_probs(probs, start=5.0, end=9.0)
    assert edges == TalkEdges(5.0, 9.0, None, None, 5.0, 9.0)


def test_edge_fraction_moves_edges_symmetrically():
    def speech(t):
        return 0.8 if t < 10 or t > 110 else 0.1

    lo, hi = _edges(speech, edge_fraction=0.1), _edges(speech, edge_fraction=0.6)
    assert hi.intro.end - lo.intro.end == pytest.approx(0.5 * WIN)
    assert lo.outro.start - hi.outro.start == pytest.approx(0.5 * WIN)


# ----------------------------------------------------------------------------
# The full function, with a stub classifier reading a synthetic signal
# ----------------------------------------------------------------------------


def _to_segment(x, sr=SR):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)


def _tone(seconds, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return 0.4 * np.sin(2 * np.pi * 440.0 * t)


def _talk(seconds, sr=SR, seed=0):
    return 0.3 * np.random.default_rng(seed).standard_normal(int(seconds * sr))


def zcr_classifier(samples, sr):
    """Stub: white noise (high zero-crossing rate) is speech, a 440 Hz tone is not."""
    starts = window_starts(len(samples), sr, window_seconds=WIN, hop_seconds=HOP)
    win = int(WIN * sr)
    zcr = np.array(
        [np.mean(np.abs(np.diff(np.sign(samples[s : s + win])))) / 2 for s in starts]
    )
    speech = np.clip((zcr - 0.15) / 0.3, 0, 1)
    t = starts / sr
    return WindowProbabilities(t, np.minimum(t + WIN, len(samples) / sr), speech, 1 - speech)


def _song():
    """host 8 s | song 60 s | host 6 s."""
    return np.concatenate([_talk(8.0), _tone(60.0), _talk(6.0, seed=1)])


def test_find_talk_edges_on_a_song_file_like_signal():
    edges = find_talk_edges(_to_segment(_song()), classifier=zcr_classifier)
    assert edges.intro.start == 0.0
    assert edges.intro.end == pytest.approx(8.0, abs=1.0)
    assert edges.outro.start == pytest.approx(68.0, abs=1.0)
    assert edges.outro.end == pytest.approx(74.0)


def test_segment_bounds_return_times_in_the_tape_timeline():
    tape = np.concatenate([_tone(30.0), _song(), _tone(20.0)])  # song at 30-104 s
    seen = {}

    def spy(samples, sr):
        seen["seconds"] = len(samples) / sr
        return zcr_classifier(samples, sr)

    edges = find_talk_edges(
        _to_segment(tape), Segment(30.0, 104.0, label="song"), classifier=spy
    )
    assert seen["seconds"] == pytest.approx(74.0, abs=0.01)  # only the song is classified
    assert (edges.start, edges.end) == (30.0, 104.0)
    assert edges.intro.start == 30.0
    assert edges.intro.end == pytest.approx(38.0, abs=1.0)
    assert edges.outro.start == pytest.approx(98.0, abs=1.0)
    # a (start, end) tuple works the same
    same = find_talk_edges(_to_segment(tape), (30.0, 104.0), classifier=zcr_classifier)
    assert same == edges


def test_words_snap_the_edges_to_the_announcer():
    words = [
        {"text": "here's", "start": 6.0, "end": 6.4},
        {"text": " ", "start": 6.4, "end": 6.5, "type": "spacing"},
        {"text": "Bobo.", "start": 7.3, "end": 30.0},  # caption 'end' far away: capped
        {"text": "la", "start": 20.0, "end": 20.5},  # a lyric deep in the song
        {"text": "that", "start": 68.3, "end": 68.6},
        {"text": "was", "start": 68.7, "end": 69.0},
    ]
    edges = find_talk_edges(_to_segment(_song()), classifier=zcr_classifier, words=words)
    assert edges.snapped_to_words
    assert edges.intro.end == pytest.approx(7.3 + te.MAX_WORD_SECONDS)
    assert edges.outro.start == pytest.approx(68.3)
    assert edges.clean_start == edges.intro.end and edges.clean_end == 68.3


def test_words_out_of_reach_or_cues_are_handled():
    far = [{"text": "la", "start": 30.0, "end": 30.5}]
    edges = find_talk_edges(_to_segment(_song()), classifier=zcr_classifier, words=far)
    plain = find_talk_edges(_to_segment(_song()), classifier=zcr_classifier)
    assert not edges.snapped_to_words
    assert (edges.intro, edges.outro) == (plain.intro, plain.outro)
    # a cue: credited MAX_WORD_SECONDS per word, not up to its (lagging) end
    cues = [Cue(1, 6.5, 12.0, "next up")]
    snapped = find_talk_edges(_to_segment(_song()), classifier=zcr_classifier, words=cues)
    assert snapped.snapped_to_words
    assert snapped.intro.end == pytest.approx(6.5 + 2 * te.MAX_WORD_SECONDS)


def test_empty_segment_is_an_informative_error():
    with pytest.raises(ValueError, match="Empty segment"):
        find_talk_edges(_to_segment(_song()), (50.0, 40.0), classifier=zcr_classifier)


def test_default_classifier_uses_short_dense_windows(monkeypatch):
    made = {}

    def fake_factory(**kw):
        made.update(kw)
        return zcr_classifier

    monkeypatch.setattr(te, "ast_audioset_classifier", fake_factory)
    find_talk_edges(_to_segment(_song()))
    assert made == {"window_seconds": WIN, "hop_seconds": HOP}


def test_labels_fixture_is_well_formed():
    labels = json.loads(LABELS.read_text())
    edges = labels["edges"]
    assert len(edges) >= 15
    assert len({e["tape"] for e in edges}) >= 4
    for e in edges:
        assert e["split"] in ("dev", "heldout", "test")
        for key in ("intro_end", "outro_start"):
            if e[key] is not None:
                assert 0 <= e[key] <= e["duration"]


# ----------------------------------------------------------------------------
# Real model (slow; opt in with MIXING_RUN_SLOW=1, the ``ml`` extra, and the tapes)
# ----------------------------------------------------------------------------

_slow = pytest.mark.skipif(
    os.environ.get("MIXING_RUN_SLOW") != "1",
    reason="real-model test; set MIXING_RUN_SLOW=1 (needs mixing[ml] + a model download)",
)


@pytest.mark.slow
@_slow
def test_real_model_against_hand_labels():
    """Set MIXING_MWM_TAPES to the folder of ``NNN_<id>/`` tapes the labels describe."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    root = os.environ.get("MIXING_MWM_TAPES")
    if not root or not os.path.isdir(root):
        pytest.skip("set MIXING_MWM_TAPES to the tapes folder of the labels fixture")
    from mixing.srt import parse_vtt_words

    labels = json.loads(LABELS.read_text())["edges"]
    errors = []
    for tape_id in sorted({e["tape"] for e in labels}):
        tape_dir = next(Path(root).glob(f"*_{tape_id}"))
        audio = AudioSegment.from_file(str(tape_dir / f"{tape_dir.name}.m4a"))
        words = parse_vtt_words((tape_dir / f"{tape_dir.name}.en.vtt").read_text())
        for e in (e for e in labels if e["tape"] == tape_id):
            s0 = e["song_start"]
            edges = find_talk_edges(audio, (s0, s0 + e["duration"]), words=words)
            if e["intro_end"] is not None:
                errors.append(abs(edges.clean_start - s0 - e["intro_end"]))
            if e["outro_start"] is not None:
                errors.append(abs(edges.clean_end - s0 - e["outro_start"]))
    assert np.median(errors) < 1.0
