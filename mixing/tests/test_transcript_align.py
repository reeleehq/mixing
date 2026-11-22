"""Tests for mixing.transcript.align: authored tokens timed by heard words."""

import pytest

from mixing.transcript import align_words, normalize_word, tokens_of
from mixing.transcript.align import AlignedWords


def _starts(aligned: AlignedWords) -> list[float]:
    return [w[1] for w in aligned.words]


def test_all_matched_take_the_heard_times():
    heard = [("Hello,", 0.5, 0.9), ("there!", 0.9, 1.3)]
    aligned = align_words(["hello", "there"], heard)
    assert aligned.words == [("hello", 0.5, 0.9), ("there", 0.9, 1.3)]
    assert (aligned.matched, aligned.total, aligned.coverage) == (2, 2, 1.0)
    assert aligned.anchors == {0: 0, 1: 1}


def test_misheard_run_is_spread_over_the_misheard_span():
    heard = [("my", 0.0, 0.2), ("fred", 0.2, 0.6), ("came", 0.6, 0.9)]
    aligned = align_words(["my", "friend", "came"], heard)
    assert aligned.words[1] == ("friend", 0.2, 0.6)
    assert aligned.matched == 2
    assert 1 not in aligned.anchors


def test_unheard_run_fills_the_gap_by_token_length():
    heard = [("a", 0.0, 0.1), ("z", 1.1, 1.2)]
    aligned = align_words(["a", "xx", "yyyy", "z"], heard)
    (_, s1, e1), (_, s2, e2) = aligned.words[1:3]
    assert s1 == pytest.approx(0.1) and e2 == pytest.approx(1.1)
    assert (e2 - s2) == pytest.approx(2 * (e1 - s1), abs=1e-5)  # 4 letters vs 2


def test_unheard_tail_reaches_the_bounds():
    aligned = align_words(["a", "b"], [("a", 0.0, 0.5)], bounds=(0.0, 2.0))
    assert aligned.words[1] == ("b", 0.5, 2.0)


def test_inserted_heard_words_let_time_pass():
    heard = [("a", 0.0, 0.1), ("um", 0.1, 0.8), ("b", 0.8, 1.0)]
    aligned = align_words(["a", "b"], heard)
    assert aligned.words == [("a", 0.0, 0.1), ("b", 0.8, 1.0)]
    assert aligned.anchors == {0: 0, 1: 2}


def test_starts_never_decrease_even_with_overlapping_heard_times():
    heard = [("a", 0.0, 0.5), ("b", 0.3, 0.6), ("d", 0.2, 0.4)]
    aligned = align_words(["a", "b", "c", "d"], heard)
    starts = _starts(aligned)
    assert starts == sorted(starts)
    assert all(end >= start for _, start, end in aligned.words)


def test_scribe_dicts_skip_spacing_and_events_and_anchor_original_indices():
    scribe = [
        {"text": "Hi", "start": 0.0, "end": 0.2, "type": "word"},
        {"text": " ", "start": 0.2, "end": 0.3, "type": "spacing"},
        {"text": "(laughs)", "start": 0.3, "end": 0.5, "type": "audio_event"},
        {"text": "there.", "start": 0.5, "end": 0.8, "type": "word"},
    ]
    aligned = align_words(["hi", "there"], scribe)
    assert aligned.anchors == {0: 0, 1: 3}
    assert aligned.words == [("hi", 0.0, 0.2), ("there", 0.5, 0.8)]


def test_punctuation_only_heard_words_are_ignored():
    aligned = align_words(["a", "b"], [("a", 0.0, 0.1), ("—", 0.1, 0.2), ("b", 0.2, 0.3)])
    assert aligned.anchors == {0: 0, 1: 2}


def test_custom_normalize_is_the_matching_key():
    heard = [("colour", 0.0, 0.4)]
    plain = align_words(["color"], heard)
    assert plain.matched == 0
    keyed = align_words(["color"], heard, normalize=lambda w: normalize_word(w).replace("ou", "o"))
    assert keyed.matched == 1 and keyed.words == [("color", 0.0, 0.4)]


def test_nothing_heard():
    aligned = align_words(["a", "b"], [], bounds=(1.0, 3.0))
    assert [w[0] for w in aligned.words] == ["a", "b"]
    assert aligned.words[0][1] == 1.0 and aligned.words[-1][2] == 3.0
    assert aligned.matched == 0 and aligned.coverage == 0.0


def test_empty_reference_has_full_coverage():
    aligned = align_words([], [("a", 0.0, 1.0)])
    assert aligned.words == [] and aligned.coverage == 1.0


def test_tokens_and_normalize():
    assert tokens_of("Hello, there — it's ME!") == ["Hello", "there", "it's", "ME"]
    assert normalize_word(" Là.") == "là"
    assert normalize_word("...") == ""


def test_matches_an_reference_example():
    """The example an's align_words (the starter) documents, unchanged."""
    heard = [("i", 0.0, 0.1), ("did", 0.1, 0.3), ("ask", 0.6, 0.9)]
    assert align_words(["I", "did", "NOT", "ask"], heard).words == [
        ("I", 0.0, 0.1),
        ("did", 0.1, 0.3),
        ("NOT", 0.3, 0.6),
        ("ask", 0.6, 0.9),
    ]
