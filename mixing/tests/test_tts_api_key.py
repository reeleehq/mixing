"""Per-request API-key resolution in :func:`mixing.dubbing.text_to_speech`.

``text_to_speech`` already accepts an explicit ``api_key`` (falling back to
``$ELEVENLABS_API_KEY`` when omitted). These tests lock that contract — the
key sent in the ``xi-api-key`` request header is the explicit one when given,
and the env one otherwise — because a downstream consumer (braidio → reelee)
relies on it for per-user BYO keys. They stub the network layer
(``urllib.request.urlopen``) so no real ElevenLabs call happens.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mixing.dubbing import text_to_speech
from mixing.dubbing import tts as tts_mod


class _FakeResp:
    def __init__(self, data: bytes = b"AUDIO"):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_urlopen(captured: list):
    def _stub(req, timeout=None):
        captured.append(req)
        return _FakeResp()

    return _stub


def _sent_api_key(req) -> str | None:
    """The ``xi-api-key`` header on the captured urllib Request."""
    return req.get_header("Xi-api-key")


def test_explicit_api_key_is_used(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ENV-KEY")
    captured: list = []
    with patch.object(tts_mod.urllib.request, "urlopen", _stub_urlopen(captured)):
        text_to_speech("hello", "voice-x", api_key="EXPLICIT-KEY", cache=False)
    assert len(captured) == 1
    # explicit key wins over the env var
    assert _sent_api_key(captured[0]) == "EXPLICIT-KEY"


def test_omitted_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ENV-KEY")
    captured: list = []
    with patch.object(tts_mod.urllib.request, "urlopen", _stub_urlopen(captured)):
        text_to_speech("hello", "voice-x", cache=False)
    assert len(captured) == 1
    assert _sent_api_key(captured[0]) == "ENV-KEY"


def test_no_key_anywhere_raises(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        text_to_speech("hello", "voice-x", cache=False)


def test_eleven_api_key_alias_and_precedence(monkeypatch):
    from mixing._elevenlabs import resolve_api_key

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("ELEVEN_API_KEY", "ALIAS")
    assert resolve_api_key(None) == "ALIAS"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "PRIMARY")
    assert resolve_api_key(None) == "PRIMARY"
    assert resolve_api_key("explicit") == "explicit"


def test_empty_primary_falls_through_to_alias(monkeypatch):
    from mixing._elevenlabs import resolve_api_key

    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    monkeypatch.setenv("ELEVEN_API_KEY", "ALIAS")
    assert resolve_api_key(None) == "ALIAS"
