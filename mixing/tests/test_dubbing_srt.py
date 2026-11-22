"""Tests for ``mixing.dubbing.srt``: translation keeps timings and segment count; JSON-array parsing."""

import sys
import types

import pytest

from mixing.dubbing import srt as dsrt
from mixing.srt import parse_srt

SRT = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n2\n00:00:01,000 --> 00:00:02,500\nworld\n"


def test_translate_srt_keeps_timings_and_strips_text():
    seen = {}

    def fn(texts, target, source):
        seen.update(texts=list(texts), target=target, source=source)
        return [f"  {t.upper()}  " for t in texts]

    out = parse_srt(dsrt.translate_srt(SRT, "French", source_language="English", translate_fn=fn))
    assert seen == {"texts": ["hello", "world"], "target": "French", "source": "English"}
    assert [c.text for c in out] == ["HELLO", "WORLD"]
    assert [(c.start, c.end) for c in out] == [(0.0, 1.0), (1.0, 2.5)]


def test_translate_srt_accepts_cue_list():
    cues = parse_srt(SRT)
    out = parse_srt(dsrt.translate_srt(cues, "x", translate_fn=lambda t, a, b: list(t)))
    assert [c.text for c in out] == ["hello", "world"]


def test_translate_srt_rejects_wrong_segment_count():
    with pytest.raises(ValueError, match="preserve segment count"):
        dsrt.translate_srt(SRT, "x", translate_fn=lambda t, a, b: ["only one"])


@pytest.mark.parametrize(
    "raw",
    [
        '["a", "b"]',
        '```json\n["a", "b"]\n```',
        'Sure! Here you go: ["a", "b"] hope that helps',
        '  ["a", "b"]  ',
    ],
)
def test_parse_json_array_variants(raw):
    assert dsrt._parse_json_array(raw, expected=2) == ["a", "b"]


def test_parse_json_array_stringifies_and_rejects_non_arrays():
    assert dsrt._parse_json_array("[1, 2]", expected=2) == ["1", "2"]
    with pytest.raises(ValueError, match="not a JSON array"):
        dsrt._parse_json_array('"x"', expected=1)


def test_default_translate_fn_calls_llm_once_with_all_segments(monkeypatch):
    calls = []

    def chat(prompt, *, model=None, temperature=None):
        calls.append((prompt, model, temperature))
        return '["bonjour", "monde"]'

    monkeypatch.setattr(dsrt, "require_llm_chat", lambda feature, *, alternative: chat)
    out = dsrt.default_translate_fn(["hello", "world"], "French", "English", model="m")
    assert out == ["bonjour", "monde"]
    (prompt, model, temp), = calls
    assert "into French" in prompt and "from English" in prompt and "hello" in prompt
    assert model == "m" and temp == 0.3


def test_default_translate_fn_without_llm_names_the_alternative(monkeypatch):
    monkeypatch.setitem(sys.modules, "aix", None)
    with pytest.raises(ImportError, match="translate_fn to translate_srt"):
        dsrt.default_translate_fn(["x"], "French")
