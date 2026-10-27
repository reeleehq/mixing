"""Detect chapter markers from a transcript — platform-agnostic.

A *chapter* is a ``(start_seconds, title)`` pair marking a topic shift. This
module turns a transcript (ElevenLabs Scribe response/words, SRT text, or a
list of cues) into a list of :class:`Chapter` objects, with titles produced by
a pluggable LLM segmenter. The result is intentionally **target-neutral** —
formatting chapters into YouTube description timestamps, podcast PSC, or ID3
chapter frames is the job of a publication layer (e.g. the ``yb`` package).

The detector enforces the constraints common to chapter-aware players:
first chapter at ``0:00``, a minimum spacing between chapters, and a minimum
count below which chapters are not worth showing (an empty list is returned so
callers can cleanly skip them — e.g. for a very short clip).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from mixing.srt import parse_srt as _parse_srt
from mixing.util import require_llm_chat

#: A segmenter maps (segments, target_count) -> [{"start": float, "title": str}].
#: ``segments`` is a list of ``{"start", "end", "text"}`` sentence-ish units.
SegmentFn = Callable[[Sequence[dict], int], list[dict]]

_SENTENCE_ENDINGS = ".?!"

#: Heuristic spacing used to pick a default chapter count when ``target_count``
#: is not given: roughly one chapter per this many seconds of media. ~90s (1.5
#: minutes) is a common chapter cadence for talks/tutorials.
SECONDS_PER_CHAPTER_HEURISTIC = 90.0

#: Small slack (seconds) when comparing chapter starts against the media
#: duration, so a marker sitting essentially *at* the end is dropped despite
#: floating-point jitter in the last timestamp.
_EPSILON_SECONDS = 1e-3


@dataclass
class Chapter:
    """A chapter marker: a start time (seconds) and a short title."""

    start: float
    title: str


def detect_chapters(
    transcript,
    *,
    duration: float | None = None,
    min_chapters: int = 3,
    max_chapters: int = 8,
    min_spacing: float = 10.0,
    target_count: int | None = None,
    segment_fn: SegmentFn | None = None,
    model: str | None = None,
) -> list[Chapter]:
    """Detect chapter markers from a ``transcript``.

    Args:
        transcript: One of — a Scribe response ``dict`` (with ``"words"``), a
            Scribe ``words`` list, SRT text, or a list of cue dicts/objects
            exposing ``start``/``end``/``text``.
        duration: Media duration in seconds. Inferred from the transcript's
            last timestamp when omitted; used to choose a sensible chapter
            count and to bound the final marker.
        min_chapters: Minimum chapters worth showing. If fewer survive the
            constraints, an **empty list** is returned.
        max_chapters: Upper bound on chapter count.
        min_spacing: Minimum seconds between consecutive chapters (players such
            as YouTube require >= 10s).
        target_count: Desired chapter count. When omitted, scales with
            ``duration`` (roughly one chapter per ~1.5 min, clamped to
            ``[min_chapters, max_chapters]``).
        segment_fn: Pluggable segmenter ``(segments, target_count) -> [{start,
            title}]``. Defaults to :func:`default_segment_fn` (LLM-backed).
        model: Optional LLM model override for the default segmenter.

    Returns:
        A list of :class:`Chapter`, first at ``0:00``, spaced by at least
        ``min_spacing`` — or ``[]`` when the media can't support
        ``min_chapters`` (e.g. it is too short).
    """
    segments = _segments_from_transcript(transcript)
    if not segments:
        return []
    if duration is None:
        duration = segments[-1]["end"]

    # Too short to host the minimum number of spaced chapters → no chapters.
    if duration < min_chapters * min_spacing:
        return []

    if target_count is None:
        target_count = max(
            min_chapters,
            min(max_chapters, round(duration / SECONDS_PER_CHAPTER_HEURISTIC)),
        )
    target_count = max(min_chapters, min(max_chapters, target_count))

    segment_fn = segment_fn or default_segment_fn
    raw = segment_fn(segments, target_count)
    chapters = [
        Chapter(start=float(c["start"]), title=str(c["title"]).strip())
        for c in raw
        if c.get("title")
    ]
    return _enforce_constraints(
        chapters,
        segments,
        duration=duration,
        min_chapters=min_chapters,
        max_chapters=max_chapters,
        min_spacing=min_spacing,
    )


def _segments_from_transcript(transcript) -> list[dict]:
    """Normalize any supported transcript form to sentence-ish segments."""
    # Scribe response dict
    if isinstance(transcript, dict) and "words" in transcript:
        return _sentences_from_words(transcript["words"])
    # SRT text
    if isinstance(transcript, str):
        return _sentences_from_srt(transcript)
    if isinstance(transcript, (list, tuple)):
        items = list(transcript)
        if not items:
            return []
        first = items[0]
        # Scribe words list (dicts with a "type" or word-level start/end)
        if (
            isinstance(first, dict)
            and "text" in first
            and "start" in first
            and (
                first.get("type") in {"word", "spacing", "audio_event"}
                or "end" in first
            )
            and _looks_like_words(items)
        ):
            return _sentences_from_words(items)
        # Cue-like (dict or object with start/end/text at sentence granularity)
        out = []
        for it in items:
            start = it["start"] if isinstance(it, dict) else getattr(it, "start")
            end = (
                it.get("end", start)
                if isinstance(it, dict)
                else getattr(it, "end", start)
            )
            text = (it["text"] if isinstance(it, dict) else getattr(it, "text")).strip()
            if text:
                out.append({"start": float(start), "end": float(end), "text": text})
        return out
    return []


def _looks_like_words(items: Sequence[dict]) -> bool:
    """Heuristic: many short single-token entries → a word list, not cues."""
    sample = [it for it in items[:20] if isinstance(it, dict)]
    if not sample:
        return False
    short = sum(1 for it in sample if len(str(it.get("text", "")).split()) <= 1)
    return short >= len(sample) * 0.6


def _sentences_from_words(words: Sequence[dict]) -> list[dict]:
    """Group Scribe words into sentence segments by terminal punctuation."""
    out: list[dict] = []
    cur: list[str] = []
    cur_start: float | None = None
    cur_end = 0.0
    for w in words:
        if w.get("type") == "spacing":
            continue
        text = str(w.get("text", ""))
        if cur_start is None:
            cur_start = float(w.get("start", cur_end))
        cur_end = float(w.get("end", cur_end))
        cur.append(text)
        if text.rstrip().endswith(tuple(_SENTENCE_ENDINGS)):
            out.append(
                {"start": cur_start, "end": cur_end, "text": " ".join(cur).strip()}
            )
            cur, cur_start = [], None
    if cur and cur_start is not None:
        out.append({"start": cur_start, "end": cur_end, "text": " ".join(cur).strip()})
    return _tidy(out)


def _sentences_from_srt(srt_text: str) -> list[dict]:
    """Parse SRT into per-cue segments (cues already ~sentence granularity).

    Uses the canonical :func:`mixing.srt.parse_srt`; multi-line cue text is
    flattened to a single line (chapters work at sentence granularity).
    """
    out = [
        {"start": c.start, "end": c.end, "text": " ".join(c.text.split())}
        for c in _parse_srt(srt_text)
    ]
    return _tidy(out)


def _tidy(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["text"]]


def _enforce_constraints(
    chapters: list[Chapter],
    segments: Sequence[dict],
    *,
    duration: float,
    min_chapters: int,
    max_chapters: int,
    min_spacing: float,
) -> list[Chapter]:
    """Snap to segment boundaries, force 0:00 first, drop too-close markers."""
    if not chapters:
        return []
    boundaries = [s["start"] for s in segments]
    chapters = sorted(chapters, key=lambda c: c.start)
    # Snap each start to the nearest real segment boundary.
    for c in chapters:
        c.start = min(boundaries, key=lambda b: abs(b - c.start))
    # Force the first chapter to 0:00.
    chapters[0].start = 0.0
    # Enforce minimum spacing and the duration bound; keep first occurrence.
    kept: list[Chapter] = []
    for c in chapters:
        if c.start >= duration - _EPSILON_SECONDS:
            continue
        if kept and c.start - kept[-1].start < min_spacing:
            continue
        kept.append(c)
    kept = kept[:max_chapters]
    if len(kept) < min_chapters:
        return []
    return kept


def default_segment_fn(
    segments: Sequence[dict],
    target_count: int,
    *,
    model: str | None = None,
) -> list[dict]:
    """LLM-backed segmenter using ``aix.chat``.

    Presents the timestamped sentences and asks for ``target_count`` chapter
    boundaries as a JSON array of ``{"start", "title"}``. Pass your own
    ``segment_fn`` to :func:`detect_chapters` to avoid the ``aix`` dependency.
    """
    chat = require_llm_chat(
        "default_segment_fn", alternative="segment_fn to detect_chapters()"
    )

    lines = "\n".join(f"[{_fmt(s['start'])}] {s['text']}" for s in segments)
    prompt = (
        f"You are segmenting a media transcript into about {target_count} chapters "
        "for a chapter-aware player.\n"
        "Each line below is a sentence prefixed with its start time [M:SS].\n"
        "Choose chapter boundaries at real topic shifts (not every sentence). "
        "Rules: the first chapter MUST start at 0:00; use a start time taken "
        "verbatim from one of the lines; titles are 2-7 words, no leading numbers.\n"
        'Return ONLY a JSON array of {"start": <seconds:number>, "title": <string>} '
        "objects, ordered by start. No code fences, no commentary.\n\n"
        f"{lines}"
    )
    raw = chat(prompt, model=model, temperature=0.3)
    return _parse_json_array(raw)


def _fmt(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text.startswith("["):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            text = m[0]
    arr = json.loads(text)
    if not isinstance(arr, list):
        raise ValueError("Segmenter response was not a JSON array.")
    return [
        {"start": float(x["start"]), "title": str(x["title"])}
        for x in arr
        if "start" in x and "title" in x
    ]
