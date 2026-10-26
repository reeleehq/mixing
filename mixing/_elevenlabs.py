"""Shared ElevenLabs HTTP client helpers (internal).

The Scribe (STT) and TTS clients both target ElevenLabs over stdlib HTTP and
share the same auth convention: an ``xi-api-key`` taken from the call or the
``ELEVENLABS_API_KEY`` environment variable (``ELEVEN_API_KEY``, the other name
ElevenLabs tooling uses, is accepted too). Centralized here so both clients
resolve credentials identically.
"""

from __future__ import annotations

import os

#: Environment variables holding the ElevenLabs API key, in lookup order.
#: ``ELEVENLABS_API_KEY`` is the documented primary.
ENV_KEYS = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY")


def resolve_api_key(api_key: str | None) -> str:
    """Return ``api_key`` or fall back to the first set of :data:`ENV_KEYS`.

    Raises:
        RuntimeError: Neither an explicit key nor any of the env vars is set.
    """
    key = api_key or next((os.environ[k] for k in ENV_KEYS if os.environ.get(k)), None)
    if not key:
        names = " or ".join(ENV_KEYS)
        raise RuntimeError(f"No ElevenLabs API key. Pass api_key= or set {names}.")
    return key
