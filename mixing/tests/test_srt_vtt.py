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


def test_parse_vtt_crlf_settings_multiline_and_empty_cue():
    vtt = (
        "﻿WEBVTT\r\n\r\n5\r\n00:00:01.000 --> 00:00:02.000 align:start\r\n"
        "line one\r\nline two\r\n\r\n00:00:03.000 --> 00:00:04.000\r\n"
    )
    cues = parse_vtt(vtt)
    assert [(c.index, c.text) for c in cues] == [(1, "line one line two"), (2, "")]
    assert parse_vtt(vtt, clean=False)[0].text == "line one\nline two"


def test_parse_vtt_on_top_level_facade():
    import mixing

    assert mixing.parse_vtt is parse_vtt
