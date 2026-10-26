"""Tests for the public cue-level WebVTT parser ``mixing.srt.parse_vtt``."""

from mixing.srt import parse_vtt

VTT = (
    "WEBVTT\n\nNOTE a comment\n\nSTYLE\n::cue {}\n\n"
    "intro\n01:02.500 --> 01:04.000\n<c>Hi</c> <00:01:03.000>there &amp; you\n\n"
    "00:00:05.000 --> 00:00:06.000\n \nsecond\n"
)


def test_parse_vtt_cleans_by_default():
    cues = parse_vtt(VTT)
    assert [(c.index, c.start, c.end, c.text) for c in cues] == [
        (1, 62.5, 64.0, "Hi there & you"),
        (2, 5.0, 6.0, "second"),
    ]


def test_parse_vtt_raw_keeps_markup():
    cues = parse_vtt(VTT, clean=False)
    assert cues[0].text == "<c>Hi</c> <00:01:03.000>there &amp; you"
    assert cues[1].text == "second"


def test_parse_vtt_empty():
    assert parse_vtt("WEBVTT\n") == []
