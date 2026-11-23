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


def test_utterances_from_another_rate_reach_the_seams_at_16k_on_time():
    samples, spans = _lesson([("zh", 0.5), ("fr", 0.7)], sr=44100)
    clips = []
    utts = language_utterances(
        samples, sr=44100,
        language_id=lambda clip, sr: clips.append((len(clip), sr)) or _pitch_language_id(clip, sr),
        transcribe=lambda clip, sr, lang: "x",
    )
    assert [sr for _, sr in clips] == [SR, SR]
    for u, (n, _), (_, start, end) in zip(utts, clips, spans):
        assert u.start == pytest.approx(start - 0.08, abs=0.03)  # the default pad
        assert u.end == pytest.approx(end + 0.08, abs=0.03)
        assert n == pytest.approx((u.end - u.start) * SR, abs=2)  # the clip is the span


def test_a_language_the_id_never_names_is_an_error():
    samples, _ = _lesson([("zh", 0.5)])
    with pytest.raises(ValueError, match="never named"):
        language_utterances(
            samples, sr=SR, languages=("en", "zh-cn"),
            language_id=lambda clip, sr: {"en": 0.5, "zh": 0.5},
            transcribe=lambda *a: "x",
        )


def test_is_junk():
    assert is_junk(Utterance(0, 1, {"fr": 1.0}, ""))
    assert is_junk(Utterance(0, 1, {"fr": 1.0}, "uszusz"))
    assert is_junk(Utterance(0, 0.2, {"fr": 0.4, "zh": 0.3}, "a"))  # short, no language sure
    assert not is_junk(Utterance(0, 0.3, {"fr": 0.0, "zh": 0.72}, "是"))  # short but sure
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
    utts = [Utterance(0, 1, {"zh": 1.0}, "a"), Utterance(1.1, 1.2, {"zh": 0.3, "fr": 0.3}, "x")]
    assert [(r.start, r.end) for r in language_runs(utts)] == [(0, 1)]


def test_runs_label_by_the_decided_language_not_the_argmax():
    """Whisper spreads mass over languages not asked about: fr at 0.3 with zh at
    0.05 is French (every other language small), not ambiguous; fr 0.3 / zh 0.25
    is ambiguous and joins the run it follows, even though fr is the argmax."""
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.0}, "a"),
        Utterance(1.1, 2.0, {"zh": 0.25, "fr": 0.3}, "b"),  # ambiguous: stays zh
        Utterance(2.1, 3.0, {"zh": 0.05, "fr": 0.3}, "c"),  # decided: fr
    ]
    assert [r.label for r in language_runs(utts)] == ["zh", "fr"]
    assert language_runs(utts)[0].end == 2.0


def test_runs_take_utterances_in_time_order():
    utts = [Utterance(2.0, 3.0, {"fr": 1.0}, "b"), Utterance(0.0, 1.0, {"fr": 1.0}, "a")]
    assert [(r.start, r.end) for r in language_runs(utts)] == [(0.0, 3.0)]


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


def test_junk_keeps_hallucinations_out_of_anchors_but_not_out_of_the_other_side():
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.1}, "uszusz"),  # hallucinated text, real speech
        Utterance(1.2, 2.0, {"zh": 0.0, "fr": 1.0}, "bonjour"),
        Utterance(2.2, 2.5, {"zh": 0.0, "fr": 0.95}, "uszusz"),  # would-be anchor: junk
    ]
    pairs = anchored_pairs(utts, anchor="fr")
    assert pairs[0] == LanguagePair(_seg(1.2, 2.0, "fr"), _seg(0.0, 1.0, "zh"))
    assert pairs[0].anchor.end == 2.0  # the junk anchor did not extend it
    assert anchored_pairs(utts, anchor="fr", other_junk=is_junk)[0].other is None
    with_junk_anchor = anchored_pairs(utts, anchor="fr", anchor_junk=None)
    assert with_junk_anchor[0].anchor.end == 2.5


def test_other_speech_after_the_last_anchor_comes_back_unpaired():
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.0}, "ni hao"),
        Utterance(1.5, 2.5, {"zh": 0.0, "fr": 1.0}, "bonjour"),
        Utterance(3.0, 4.0, {"zh": 0.9, "fr": 0.0}, "gong zuo"),
    ]
    pairs = anchored_pairs(utts, anchor="fr")
    assert pairs[-1] == LanguagePair(None, _seg(3.0, 4.0, "zh"))
    first = anchored_pairs(utts, anchor="fr", order="anchor-then-other")
    assert first[0] == LanguagePair(None, _seg(0.0, 1.0, "zh"))
    assert first[1].other == _seg(3.0, 4.0, "zh")


def test_anchors_do_not_merge_across_other_speech():
    utts = [
        Utterance(0.0, 1.0, {"zh": 0.0, "fr": 1.0}, "un"),
        Utterance(1.1, 1.5, {"zh": 0.95, "fr": 0.0}, "yi"),
        Utterance(1.55, 2.5, {"zh": 0.0, "fr": 1.0}, "deux"),
    ]
    pairs = anchored_pairs(utts, anchor="fr")
    assert [p.anchor for p in pairs] == [_seg(0.0, 1.0, "fr"), _seg(1.55, 2.5, "fr")]
    assert pairs[1].other == _seg(1.1, 1.5, "zh")


def test_no_anchor_at_all_is_one_unpaired_run_of_speech():
    utts = [Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.0}, "a"), Utterance(1.2, 2.0, {"zh": 0.9}, "b")]
    assert anchored_pairs(utts, anchor="fr") == [LanguagePair(None, _seg(0.0, 2.0, "zh"))]
    assert anchored_pairs([], anchor="fr") == []


def test_default_transcriber_uses_mlx_whisper_when_installed(monkeypatch):
    import importlib.util
    import sys
    import types

    from mixing.audio import code_switch

    calls = []
    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = lambda audio, **kw: calls.append(kw) or {"text": " ni hao "}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: object() if name == "mlx_whisper" else real_find_spec(name, *a),
    )
    text = code_switch.default_transcriber()(np.zeros(SR), SR, "zh")
    assert text == " ni hao "
    assert calls[0]["language"] == "zh" and calls[0]["condition_on_previous_text"] is False
    assert calls[0]["path_or_hf_repo"] == code_switch.DFLT_WHISPER_MODEL


def test_default_transcriber_falls_back_to_scribed(monkeypatch):
    import importlib.util
    import sys
    import types

    from mixing.audio import code_switch

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "mlx_whisper" else real_find_spec(name, *a),
    )
    seen = []
    fake = types.ModuleType("scribed")
    fake.transcribe_text = lambda audio, **kw: seen.append((audio[:4], kw)) or "bonjour"
    monkeypatch.setitem(sys.modules, "scribed", fake)
    assert code_switch.default_transcriber()(np.zeros(SR), SR, "fr") == "bonjour"
    assert seen[0][0] == b"RIFF" and seen[0][1]["language"] == "fr"


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


def test_weak_evidence_everywhere_is_ambiguous_not_the_first_language():
    """All languages at or under max_other_probability: None, so a run can absorb the
    utterance, not the dict's first language (reeleehq/mixing#218)."""
    from mixing.audio.code_switch import Utterance

    assert Utterance(0, 1, {"zh": 0.02, "fr": 0.05}).decided() is None
    assert Utterance(0, 1, {"fr": 0.0, "zh": 0.08}).decided() is None
    assert Utterance(0, 1, {"fr": 0.3, "zh": 0.05}).decided() == "fr"
    assert Utterance(0, 1, {"fr": 0.05, "zh": 0.3}).decided() == "zh"
