"""Tests for mixing.audio.code_switch: utterances, runs, anchored pairs."""

import numpy as np
import pytest

from mixing.audio.code_switch import (
    LanguagePair,
    Utterance,
    anchored_pairs,
    is_junk,
    language_runs,
    language_utterances,
)

SR = 16000
#: Stand-ins for two languages: a tone's pitch tells the fake language id.
PITCH = {"zh": 300.0, "fr": 900.0}


def _lesson(pattern, *, sr=SR, gap_s=0.6):
    """Tones (language, seconds) separated by silences; returns samples and spans."""
    parts, spans, t = [np.zeros(int(gap_s * sr))], [], gap_s
    for lang, dur in pattern:
        n = int(dur * sr)
        parts += [0.3 * np.sin(2 * np.pi * PITCH[lang] * np.arange(n) / sr), np.zeros(int(gap_s * sr))]
        spans.append((lang, t, t + dur))
        t += dur + gap_s
    return np.concatenate(parts), spans


def _pitch_language_id(samples, sr):
    spectrum = np.abs(np.fft.rfft(samples))
    f = np.fft.rfftfreq(len(samples), 1 / sr)[np.argmax(spectrum)]
    return {"zh": 1.0, "fr": 0.0} if abs(f - PITCH["zh"]) < abs(f - PITCH["fr"]) else {"zh": 0.02, "fr": 0.98}


def test_utterances_are_split_at_silences_and_labelled():
    samples, spans = _lesson([("zh", 0.8), ("fr", 1.0), ("zh", 0.5), ("fr", 0.7)])
    said = []
    utts = language_utterances(
        samples, sr=SR, language_id=_pitch_language_id,
        transcribe=lambda clip, sr, lang: said.append((lang, sr)) or f"{lang}!",
    )
    assert [u.language for u in utts] == [lang for lang, *_ in spans]
    for u, (_, start, end) in zip(utts, spans):
        assert u.start == pytest.approx(start, abs=0.1) and u.end == pytest.approx(end, abs=0.1)
    assert said == [(lang, SR) for lang, *_ in spans]  # forced to its language, at 16 kHz
    assert all(u.text.endswith("!") for u in utts)


def test_utterances_from_another_rate_reach_the_seams_at_16k():
    samples, _ = _lesson([("zh", 0.5)], sr=44100)
    rates = []
    language_utterances(
        samples, sr=44100,
        language_id=lambda clip, sr: rates.append(sr) or {"zh": 1.0},
        transcribe=lambda clip, sr, lang: "x",
    )
    assert rates == [SR]


def test_is_junk():
    assert is_junk(Utterance(0, 1, {"fr": 1.0}, ""))
    assert is_junk(Utterance(0, 1, {"fr": 1.0}, "uszusz"))
    assert is_junk(Utterance(0, 0.2, {"fr": 0.6, "zh": 0.4}, "a"))
    assert not is_junk(Utterance(0, 0.2, {"fr": 0.99}, "oui"))
    assert not is_junk(Utterance(0, 1.0, {"fr": 0.6, "zh": 0.4}, "bon"))


def test_runs_merge_one_language_and_uncertain_utterances_join_their_run():
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.1}, "a"),
        Utterance(1.2, 1.9, {"zh": 0.45, "fr": 0.4}, "b"),  # uncertain: joins zh
        Utterance(2.0, 3.0, {"zh": 0.0, "fr": 1.0}, "c"),
        Utterance(5.0, 6.0, {"zh": 0.0, "fr": 1.0}, "d"),  # gap > 1.6 s: new run
    ]
    runs = language_runs(utts)
    assert [(r.start, r.end, r.label) for r in runs] == [
        (0.0, 1.9, "zh"), (2.0, 3.0, "fr"), (5.0, 6.0, "fr"),
    ]


def test_runs_drop_junk_by_default():
    utts = [Utterance(0, 1, {"zh": 1.0}, "a"), Utterance(1.1, 1.2, {"zh": 0.5, "fr": 0.5}, "x")]
    assert [(r.start, r.end) for r in language_runs(utts)] == [(0, 1)]


def test_pairs_anchor_on_the_reliable_language_whatever_the_noisy_labels():
    """The other language's single words come back mislabelled (p_fr high);
    it is the speech between the reliable anchors that is paired."""
    utts = [
        Utterance(0.0, 0.6, {"zh": 0.9, "fr": 0.1}, "ni"),
        Utterance(0.7, 1.0, {"zh": 0.3, "fr": 0.7}, "hao"),  # noisy: looks French
        Utterance(1.2, 2.0, {"zh": 0.0, "fr": 1.0}, "bonjour"),
        Utterance(2.2, 2.8, {"zh": 0.6, "fr": 0.4}, "xie"),
        Utterance(3.0, 3.9, {"zh": 0.01, "fr": 0.99}, "merci"),
        Utterance(4.0, 4.5, {"zh": 0.05, "fr": 0.95}, "beaucoup"),  # within 0.6 s: same anchor
    ]
    pairs = anchored_pairs(utts, anchor="fr")
    assert pairs == [
        LanguagePair(anchor=_seg(1.2, 2.0, "fr"), other=_seg(0.0, 1.0, "zh")),
        LanguagePair(anchor=_seg(3.0, 4.5, "fr"), other=_seg(2.2, 2.8, "zh")),
    ]


def test_pairs_anchor_then_other_and_too_short_other_speech():
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.0, "fr": 1.0}, "bonjour"),
        Utterance(1.2, 2.0, {"zh": 0.9, "fr": 0.1}, "ni hao"),
        Utterance(3.0, 4.0, {"zh": 0.0, "fr": 1.0}, "merci"),
        Utterance(4.2, 4.4, {"zh": 0.9, "fr": 0.1}, "xie"),  # 0.2 s < min_other_s
    ]
    pairs = anchored_pairs(utts, anchor="fr", order="anchor-then-other")
    assert [p.other for p in pairs] == [_seg(1.2, 2.0, "zh"), None]
    with pytest.raises(ValueError, match="order"):
        anchored_pairs(utts, anchor="fr", order="sideways")


def test_pairs_ignore_junk_when_asked():
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.1}, "uszusz"),
        Utterance(1.2, 2.0, {"zh": 0.0, "fr": 1.0}, "bonjour"),
    ]
    assert anchored_pairs(utts, anchor="fr")[0].other is not None
    assert anchored_pairs(utts, anchor="fr", junk=is_junk)[0].other is None


def test_end_to_end_on_a_synthetic_lesson():
    samples, _ = _lesson([("zh", 0.8), ("fr", 1.0), ("zh", 0.5), ("fr", 0.7)])
    utts = language_utterances(
        samples, sr=SR, language_id=_pitch_language_id, transcribe=lambda *a: "t"
    )
    pairs = anchored_pairs(utts, anchor="fr")
    assert len(pairs) == 2 and all(p.other is not None and p.other.label == "zh" for p in pairs)


def _seg(start, end, label):
    from mixing.audio.segmentation import Segment

    return Segment(start, end, label)
