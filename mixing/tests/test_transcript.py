"""Tests for ``mixing.transcript`` — pure / no-network units only.

These tests do not call the ElevenLabs API or shell out to ffmpeg, so
they run anywhere CI runs without extra setup.
"""

from __future__ import annotations

import os

import pytest

from mixing.transcript import (
    DEFAULT_AUDIO_EVENTS_TO_CUT,
    DEFAULT_FILLER_TOKENS,
    build_cuts,
    fmt_srt_time,
    is_filler,
    keeps_from_cuts,
    normalize_token,
    remap_time_after_cuts,
    words_to_prose,
    words_to_srt,
    words_to_srt_remapped,
)


def _word(text, start, end, type_="word"):
    return {"text": text, "start": start, "end": end, "type": type_}


@pytest.fixture
def words():
    """A small Scribe-shaped word list with some fillers."""
    # "Hello, uh, world." then long pause then "Bye."
    return [
        _word("Hello,", 0.0, 0.5),
        _word(" ", 0.5, 0.55, type_="spacing"),
        _word("uh,", 0.55, 0.85),
        _word(" ", 0.85, 1.0, type_="spacing"),
        _word("world.", 1.0, 1.5),
        _word(" ", 1.5, 3.0, type_="spacing"),
        _word("Bye.", 3.0, 3.4),
    ]


def test_normalize_token():
    assert normalize_token("Uh,") == "uh"
    assert normalize_token("UM!") == "um"
    assert normalize_token("hello") == "hello"
    assert normalize_token("123") == ""


def test_default_sets_have_expected_members():
    assert "uh" in DEFAULT_FILLER_TOKENS
    assert "um" in DEFAULT_FILLER_TOKENS
    assert "(coughs)" in DEFAULT_AUDIO_EVENTS_TO_CUT
    assert "(laughs)" not in DEFAULT_AUDIO_EVENTS_TO_CUT


def test_is_filler_word_and_event():
    assert is_filler(_word("Uh,", 0, 0.1))
    assert not is_filler(_word("Hello", 0, 0.5))
    assert is_filler({"text": "(coughs)", "start": 0, "end": 0.3, "type": "audio_event"})
    assert not is_filler({"text": "(laughs)", "start": 0, "end": 0.3, "type": "audio_event"})


def test_is_filler_respects_overrides():
    overrides = {"like"}
    assert is_filler(_word("like,", 0, 0.1), fillers=overrides)
    assert not is_filler(_word("uh,", 0, 0.1), fillers=overrides)


def test_build_cuts_absorbs_trailing_space(words):
    cuts = build_cuts(words)
    assert len(cuts) == 1
    c = cuts[0]
    assert c["start"] == pytest.approx(0.55)
    assert c["end"] == pytest.approx(1.0)  # absorbed trailing spacing
    assert c["label"].strip(", ").lower() == "uh"


def test_build_cuts_no_absorb(words):
    cuts = build_cuts(words, absorb_trailing_space=False)
    assert cuts[0]["end"] == pytest.approx(0.85)


def test_build_cuts_merges_adjacent_fillers():
    seq = [
        _word("uh,", 1.0, 1.2),
        _word(" ", 1.2, 1.22, type_="spacing"),
        _word("um,", 1.22, 1.4),
        _word(" ", 1.4, 1.45, type_="spacing"),
    ]
    cuts = build_cuts(seq)
    assert len(cuts) == 1
    assert cuts[0]["start"] == pytest.approx(1.0)
    assert cuts[0]["end"] == pytest.approx(1.45)
    assert "+" in cuts[0]["label"]


def test_keeps_from_cuts_basic():
    cuts = [{"start": 1.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]
    keeps = keeps_from_cuts(cuts, duration=10.0)
    assert keeps == [
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 5.0},
        {"start": 6.0, "end": 10.0},
    ]


def test_keeps_from_cuts_at_boundaries():
    # cut starts at 0
    keeps = keeps_from_cuts([{"start": 0.0, "end": 1.0}], duration=2.0)
    assert keeps == [{"start": 1.0, "end": 2.0}]
    # cut ends at duration
    keeps = keeps_from_cuts([{"start": 1.0, "end": 2.0}], duration=2.0)
    assert keeps == [{"start": 0.0, "end": 1.0}]
    # no cuts
    assert keeps_from_cuts([], duration=5.0) == [{"start": 0.0, "end": 5.0}]


def test_fmt_srt_time():
    assert fmt_srt_time(0) == "00:00:00,000"
    assert fmt_srt_time(1.5) == "00:00:01,500"
    assert fmt_srt_time(3661.123) == "01:01:01,123"


def test_remap_time_after_cuts():
    cuts = [{"start": 1.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]
    # before any cut
    assert remap_time_after_cuts(0.5, cuts) == 0.5
    # after first cut
    assert remap_time_after_cuts(3.0, cuts) == pytest.approx(2.0)
    # inside a cut snaps to the post-cut timeline value at the cut start
    assert remap_time_after_cuts(1.5, cuts) == pytest.approx(1.0)
    # after both cuts
    assert remap_time_after_cuts(7.0, cuts) == pytest.approx(5.0)


def test_words_to_prose_paragraphs(words):
    prose = words_to_prose(words, paragraph_pause=1.0)
    assert "\n\n" in prose  # 1.5s gap exceeds 1.0s threshold
    assert "Hello" in prose and "Bye" in prose


def test_words_to_prose_drop_fillers(words):
    prose = words_to_prose(words, drop_fillers=True)
    assert "uh" not in prose.lower()


def test_words_to_srt_basic(words):
    srt = words_to_srt(words)
    assert "1\n" in srt
    assert "-->" in srt
    assert "world" in srt


def test_words_to_srt_remapped_drops_fillers_and_remaps(words):
    cuts = build_cuts(words)
    srt = words_to_srt_remapped(words, cuts)
    assert "uh" not in srt.lower()
    # the first non-filler word "Hello," starts at 0.0 and is unaffected
    assert "00:00:00,000" in srt


def test_transcribe_raises_without_api_key(monkeypatch):
    """No network call: missing key must raise before any HTTP I/O."""
    from mixing.transcript import transcribe

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY or ELEVEN_API_KEY"):
        transcribe(b"\x00\x00\x00")


def test_pipeline_with_provided_scribe_data_no_network(tmp_path, monkeypatch):
    """Run the pipeline against a synthetic scribe response, mocking ffmpeg.

    Verifies that all transcript outputs are written and that the cleaned
    media path is returned, without hitting the network or invoking ffmpeg.
    """
    from mixing.transcript import remove_fillers
    import mixing.transcript.pipeline as pipeline_mod

    fake_input = tmp_path / "in.mov"
    fake_input.write_bytes(b"")  # path just needs to exist for stem logic

    def fake_apply_keeps(input_path, output_path, keeps, **kw):
        from pathlib import Path

        Path(output_path).write_bytes(b"FAKE_VIDEO")
        return Path(output_path)

    monkeypatch.setattr(pipeline_mod, "apply_keeps", fake_apply_keeps)

    scribe_data = {
        "language_code": "eng",
        "text": "Hello, uh, world.",
        "audio_duration_secs": 3.4,
        "words": [
            _word("Hello,", 0.0, 0.5),
            _word(" ", 0.5, 0.55, "spacing"),
            _word("uh,", 0.55, 0.85),
            _word(" ", 0.85, 1.0, "spacing"),
            _word("world.", 1.0, 1.5),
        ],
    }
    out = tmp_path / "out"
    result = remove_fillers(fake_input, out, scribe_data=scribe_data)

    assert result.cleaned_media.exists()
    assert result.transcript_md.read_text() == "Hello, uh, world."
    assert "uh" not in result.cleaned_md.read_text().lower()
    assert "Hello" in result.transcript_srt.read_text()
    assert len(result.cuts) == 1
    assert result.duration == pytest.approx(3.4)


@pytest.mark.skipif(
    not os.environ.get("MIXING_TEST_LIVE_ELEVENLABS"),
    reason="set MIXING_TEST_LIVE_ELEVENLABS=1 to opt into live API tests",
)
def test_transcribe_live():
    """Opt-in live test (requires ELEVENLABS_API_KEY and ffmpeg)."""
    pytest.skip("live test stub — provide a sample audio file to enable")
