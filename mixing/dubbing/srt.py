"""SRT translation for dubbing — built on the canonical :mod:`mixing.srt`.

SRT parsing/serialization and the :class:`~mixing.srt.Cue` model live in
:mod:`mixing.srt`; this module re-exports them and adds the dubbing-specific
piece: translating cue *text* into another language while preserving timings.

Translation is pluggable: pass any ``translate_fn`` that maps a list of strings
to a list of the same length. The default uses an LLM via the ``aix`` package
(provider-agnostic) when it is importable.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

# Canonical SRT primitives (re-exported for backward compatibility).
from mixing.srt import Cue, dump_srt, parse_srt, srt_time_to_seconds
from mixing.util import require_llm_chat

__all__ = [
    "Cue",
    "parse_srt",
    "dump_srt",
    "srt_time_to_seconds",
    "TranslateFn",
    "translate_srt",
    "default_translate_fn",
]

#: A translator maps (texts, target_language, source_language) -> texts.
TranslateFn = Callable[[Sequence[str], str, "str | None"], "list[str]"]


def translate_srt(
    srt: str | Sequence[Cue],
    target_language: str,
    *,
    source_language: str | None = None,
    translate_fn: TranslateFn | None = None,
) -> str:
    """Translate the cue text of an SRT to ``target_language``, keeping timings.

    Args:
        srt: SRT text or a list of :class:`~mixing.srt.Cue` objects.
        target_language: Human-readable target language (e.g. ``"French"``).
        source_language: Optional source language hint.
        translate_fn: A callable ``(texts, target_language, source_language)
            -> list[str]`` returning one translation per input text, in order.
            Defaults to :func:`default_translate_fn` (LLM-backed).

    Returns:
        Translated SRT text with the original cue timings.

    Raises:
        ValueError: The translator returned a different number of segments than
            it was given.
    """
    cues = parse_srt(srt) if isinstance(srt, str) else list(srt)
    translate_fn = translate_fn or default_translate_fn
    texts = [c.text for c in cues]
    translated = translate_fn(texts, target_language, source_language)
    if len(translated) != len(texts):
        raise ValueError(
            f"Translator returned {len(translated)} segments for {len(texts)} cues; "
            "translate_fn must preserve segment count."
        )
    out = [
        Cue(index=c.index, start=c.start, end=c.end, text=t.strip())
        for c, t in zip(cues, translated)
    ]
    return dump_srt(out)


def default_translate_fn(
    texts: Sequence[str],
    target_language: str,
    source_language: str | None = None,
    *,
    model: str | None = None,
) -> list[str]:
    """LLM-backed translator (segment-count preserving) using ``aix.chat``.

    Translates all segments in a single call, returning a JSON array so the
    one-to-one mapping with the input cues is preserved. Falls back to a clear
    error if ``aix`` is not importable — pass your own ``translate_fn`` then.
    """
    chat = require_llm_chat(
        "default_translate_fn", alternative="translate_fn to translate_srt()"
    )

    import json as _json

    src = f" from {source_language}" if source_language else ""
    numbered = _json.dumps(list(texts), ensure_ascii=False, indent=2)
    prompt = (
        f"You are a professional subtitle translator{src} into {target_language}.\n"
        f"Translate each string in the following JSON array into {target_language}.\n"
        "Preserve meaning and tone (this is marketing copy — keep it natural and "
        "persuasive, not literal). Keep proper nouns and brand names unchanged.\n"
        "Return ONLY a JSON array of the same length, one translated string per "
        "input string, in the same order. No commentary, no code fences.\n\n"
        f"{numbered}"
    )
    raw = chat(prompt, model=model, temperature=0.3)
    return _parse_json_array(raw, expected=len(texts))


def _parse_json_array(raw: str, *, expected: int) -> list[str]:
    """Extract a JSON array of strings from a (possibly fenced) LLM response."""
    import json as _json

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text.startswith("["):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            text = m[0]
    arr = _json.loads(text)
    if not isinstance(arr, list):
        raise ValueError("Translator response was not a JSON array.")
    return [str(x) for x in arr]
