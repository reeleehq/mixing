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


def test_starts_never_decrease_with_overlapping_heard_times():
    heard = [("a", 0.0, 0.5), ("b", 0.3, 0.6), ("d", 0.4, 0.45)]
    aligned = align_words(["a", "b", "c", "d"], heard)
    starts = _starts(aligned)
    assert starts == sorted(starts)
    assert all(end >= start for _, start, end in aligned.words)


def test_out_of_order_and_non_finite_heard_words_are_refused():
    with pytest.raises(ValueError, match="out of order"):
        align_words(["a", "b"], [("a", 2.0, 2.5), ("b", 0.0, 0.5)])
    with pytest.raises(ValueError, match="non-finite"):
        align_words(["a"], [("a", float("nan"), 0.3)])
    with pytest.raises(ValueError, match="no start"):
        align_words(["a"], [{"text": "a", "start": None, "end": 0.3}])


def test_a_string_reference_is_refused():
    with pytest.raises(TypeError, match="tokens_of"):
        align_words("hello", [("hello", 0.0, 1.0)])


def test_whisper_dicts_carry_their_text_under_word():
    heard = [{"word": " hi", "start": 0.0, "end": 0.2}]
    assert align_words(["hi"], heard).words == [("hi", 0.0, 0.2)]


def test_typographic_apostrophes_match_straight_ones():
    heard = [("It\u2019s", 0.0, 0.3), ("fine", 0.3, 0.6)]
    aligned = align_words(tokens_of("It's fine"), heard)
    assert aligned.coverage == 1.0
    assert align_words(tokens_of("It\u2019s fine"), heard).coverage == 1.0


def test_a_heard_compound_matches_the_scripts_tokens_and_anchors_its_word():
    heard = [("It's", 0.0, 0.2), ("twenty-five", 0.2, 1.0), ("past", 1.0, 1.3)]
    aligned = align_words(tokens_of("It's twenty-five past"), heard)
    assert aligned.coverage == 1.0
    assert aligned.anchors == {0: 0, 1: 1, 2: 1, 3: 2}
    (_, s1, e1), (_, s2, e2) = aligned.words[1:3]
    assert (s1, e2) == (0.2, 1.0) and e1 == s2
    whole = align_words(tokens_of("It's twenty-five past"), heard, split=None)
    assert whole.matched == 2  # split=None keeps each heard word whole


def test_a_misheard_run_ends_where_the_misheard_word_ends():
    """Distinguishes replace (spread over the misheard span) from delete
    (spread up to the next heard start)."""
    heard = [("my", 0.0, 0.2), ("fred", 0.2, 0.5), ("came", 0.9, 1.2)]
    aligned = align_words(["my", "friend", "came"], heard)
    assert aligned.words[1] == ("friend", 0.2, 0.5)


def test_a_token_heard_as_another_takes_that_words_span_not_the_gap():
    """difflib pairs an unheard token with a stray heard word (a replace), so
    the token takes that word's span, not the gap up to the next anchor."""
    heard = [("a", 0.0, 0.1), ("um", 0.1, 0.6), ("c", 1.0, 1.2)]
    aligned = align_words(["a", "b", "c"], heard)
    assert aligned.words[1] == ("b", 0.1, 0.6)


def test_an_unheard_tail_after_several_words_ends_at_the_last_one():
    heard = [("a", 0.0, 0.1), ("b", 0.1, 0.2), ("c", 0.2, 0.9)]
    aligned = align_words(["a", "b", "c", "d"], heard)
    assert aligned.words[3] == ("d", 0.9, 0.9)


def test_bounds_are_floats():
    aligned = align_words(["x"], [], bounds=(0, 1))
    assert all(isinstance(t, float) for _, *ts in aligned.words for t in ts)


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
    # a reference token that normalizes to nothing is spread, never matched
    dash = align_words(["a", "—", "b"], [("a", 0.0, 0.1), ("—", 0.1, 0.2), ("b", 0.2, 0.3)])
    assert dash.anchors == {0: 0, 2: 2} and dash.total == 3


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


def _an_align_words(reference, timed, *, bounds=None):
    """A frozen copy of an's align_words (an/audio/transcripts.py), the starter
    #195 moved here: the differential oracle."""
    import difflib
    import re

    tok = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
    norm = lambda t: " ".join(tok.findall(t.casefold()))
    ref = [norm(w) for w in reference]
    asr = [norm(w[0]) for w in timed]
    first = bounds[0] if bounds else (float(timed[0][1]) if timed else 0.0)
    last = bounds[1] if bounds else (float(timed[-1][2]) if timed else 0.0)
    out, matched, prev_end = [], 0, first

    def spread(indices, lo, hi):
        nonlocal prev_end
        lo = max(lo, prev_end)
        hi = max(hi, lo)
        weights = [max(1, len(ref[k]) or len(reference[k])) for k in indices]
        total = float(sum(weights)) or 1.0
        t = lo
        for k, w in zip(indices, weights):
            length = (hi - lo) * (w / total)
            out.append((reference[k], round(t, 6), round(t + length, 6)))
            t += length
        prev_end = hi

    opcodes = difflib.SequenceMatcher(None, ref, asr, autojunk=False).get_opcodes()
    for n, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            for k in range(i2 - i1):
                start, end = float(timed[j1 + k][1]), float(timed[j1 + k][2])
                start = max(start, prev_end)
                out.append((reference[i1 + k], start, max(end, start)))
                prev_end = max(end, start)
                matched += 1
        elif tag == "replace":
            spread(range(i1, i2), float(timed[j1][1]), float(timed[j2 - 1][2]))
        elif tag == "delete":
            nxt = next((float(timed[op[3]][1]) for op in opcodes[n + 1 :] if op[3] < op[4]), last)
            spread(range(i1, i2), prev_end, nxt)
        else:
            prev_end = max(prev_end, float(timed[j2 - 1][2])) if j2 > j1 else prev_end
    return out, matched, len(ref)


@pytest.mark.parametrize("seed", range(40))
def test_agrees_with_ans_aligner_on_plain_words(seed):
    """On plain words (no compounds, no punctuation-only words) the result is
    an's, word for word."""
    import random

    rng = random.Random(seed)
    vocab = ["a", "bb", "ccc", "dd", "e", "fff", "Gg"]
    reference = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
    heard, t = [], 0.0
    for _ in range(rng.randint(0, 12)):
        start = t + rng.choice([0.0, 0.05, 0.3])
        end = start + rng.choice([0.0, 0.1, 0.4])
        heard.append((rng.choice(vocab), round(start, 3), round(end, 3)))
        t = start + rng.choice([0.0, 0.1, 0.2])  # overlaps allowed, order kept
    bounds = rng.choice([None, (0.0, t + 1.0)])
    got = align_words(reference, heard, bounds=bounds)
    assert (got.words, got.matched, got.total) == _an_align_words(
        reference, heard, bounds=bounds
    )
