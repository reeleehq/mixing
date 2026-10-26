"""Text-to-speech dubbing: re-voice or translate a video from its SRT.

The transcript side of :mod:`mixing` turns speech into SRT (ElevenLabs
Scribe); this subpackage closes the loop — turning SRT back into speech with
an ElevenLabs voice and muxing it over the video, optionally translating the
SRT first to produce a foreign-language dub.

Building blocks:
    - :func:`text_to_speech` / :func:`synthesize_to_file` — ElevenLabs TTS.
    - :func:`list_voices` / :func:`find_voice` — discover account voices.
    - :func:`parse_srt` / :func:`dump_srt` / :func:`translate_srt` — SRT I/O
      and LLM-backed translation.
    - :func:`dub_video_from_srt` — the end-to-end pipeline.

The HTTP clients use stdlib only (no ``requests`` / ``elevenlabs`` SDK), so
this adds no required deps. Needs ffmpeg on PATH and an ElevenLabs API key
(env var ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``) or explicit ``api_key=``). Translation needs
either the ``aix`` package (``pip install 'mixing[llm]'``) or a custom
``translate_fn``.

**Lazy by design.** Like the top-level :mod:`mixing` facade, this subpackage
uses PEP 562 ``__getattr__`` so that the TTS / SRT building blocks
(:func:`text_to_speech`, :func:`list_voices`, :func:`parse_srt`,
:class:`Cue`, …) import with **no** ``moviepy`` dependency. Only
:func:`dub_video_from_srt` — the end-to-end pipeline that muxes audio over a
video — pulls ``moviepy``, and only when it is first accessed.

Quick start:
    >>> from mixing.dubbing import dub_video_from_srt, translate_srt  # doctest: +SKIP
    >>> dub_video_from_srt("promo.mp4", "promo.srt", voice_id="...", output="promo.en.mp4")  # doctest: +SKIP
    >>> fr_srt = translate_srt(open("promo.srt").read(), "French")  # doctest: +SKIP
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# --- Eager, dependency-light exports -------------------------------------
# The TTS (stdlib HTTP) and SRT (pure) building blocks pull no media backend,
# so they stay eager: ``from mixing.dubbing import text_to_speech`` is cheap.
from mixing.dubbing.tts import (
    ELEVENLABS_TTS_URL,
    ELEVENLABS_VOICES_URL,
    DFLT_MODEL_ID,
    DFLT_OUTPUT_FORMAT,
    DFLT_VOICE_SETTINGS,
    default_cache_dir,
    text_to_speech,
    synthesize_to_file,
    list_voices,
    find_voice,
    search_shared_voices,
    add_shared_voice,
)
from mixing.dubbing.srt import (
    Cue,
    parse_srt,
    dump_srt,
    srt_time_to_seconds,
    translate_srt,
    default_translate_fn,
)

# --- Lazy facade ----------------------------------------------------------
# name -> the submodule that defines it. Imported (with its moviepy backend)
# only when the name is first accessed via ``__getattr__`` (PEP 562).
_LAZY: dict[str, str] = {
    # dub.py pulls moviepy via mixing.video.video_ops
    "dub_video_from_srt": "mixing.dubbing.dub",
}

if TYPE_CHECKING:  # help static analyzers see the lazy name
    from mixing.dubbing.dub import dub_video_from_srt  # noqa: F401


def __getattr__(name: str):
    """Resolve a lazy facade name (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is not None:
        value = getattr(importlib.import_module(module_path), name)
        globals()[name] = value  # cache so __getattr__ isn't hit again
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ELEVENLABS_TTS_URL",
    "ELEVENLABS_VOICES_URL",
    "DFLT_MODEL_ID",
    "DFLT_OUTPUT_FORMAT",
    "DFLT_VOICE_SETTINGS",
    "default_cache_dir",
    "text_to_speech",
    "synthesize_to_file",
    "list_voices",
    "find_voice",
    "search_shared_voices",
    "add_shared_voice",
    "Cue",
    "parse_srt",
    "dump_srt",
    "srt_time_to_seconds",
    "translate_srt",
    "default_translate_fn",
    "dub_video_from_srt",
]
