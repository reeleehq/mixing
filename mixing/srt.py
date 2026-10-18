"""Canonical SRT / timeline parsing and formatting (pure, dependency-free).

This is the single source of truth for SubRip (``.srt``) time handling in
``mixing``. Before consolidation the same logic was reimplemented in three
places (``dubbing.srt``, ``video.video_subtitles``, ``transcript.formats``)
with subtly different rounding and tolerance; those modules now re-export from
here.

The vocabulary:

- A **timestamp** is the ``HH:MM:SS,mmm`` string SRT uses; :func:`seconds_to_srt_time`
  and :func:`srt_time_to_seconds` convert between it and seconds (a ``float``).
- A :class:`Cue` is one subtitle block: a 1-based ``index``, ``start``/``end``
  in seconds, and ``text`` (which may contain embedded newlines).

Times are always seconds (``float``) everywhere except the timestamp string.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "Cue",
    "TIME_RE",
    "srt_time_to_seconds",
    "seconds_to_srt_time",
    "fmt_srt_time",
    "to_srt_time",
    "parse_srt",
    "dump_srt",
    "shift_srt_timestamps",
    "parse_vtt_words",
]

#: Matches an SRT cue time line, tolerant of ``,`` or ``.`` as the ms separator.
TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class Cue:
    """One SRT subtitle cue.

    Attributes:
        index: 1-based cue number.
        start: Start time in seconds.
        end: End time in seconds.
        text: Cue text (may contain embedded newlines).
    """

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Cue duration in seconds (never negative)."""
        return max(0.0, self.end - self.start)


def srt_time_to_seconds(timestamp: str) -> float:
    """Parse an ``HH:MM:SS,mmm`` (or ``.mmm``) SRT timestamp to seconds.

    >>> srt_time_to_seconds('00:43:12,187')
    2592.187
    >>> srt_time_to_seconds('00:00:01.500')
    1.5
    """
    h, m, sec, ms = re.split(r"[:,.]", timestamp.strip())
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000.0


def seconds_to_srt_time(seconds: float) -> str:
    """Format a time in seconds as an SRT timestamp ``HH:MM:SS,mmm``.

    Milliseconds are *rounded* (not truncated), with carry handled correctly,
    and negative inputs clamp to zero.

    >>> seconds_to_srt_time(2592.187)
    '00:43:12,187'
    >>> seconds_to_srt_time(1.5)
    '00:00:01,500'
    >>> seconds_to_srt_time(-3)
    '00:00:00,000'
    """
    ms_total = max(0, int(round(seconds * 1000)))
    h, ms_total = divmod(ms_total, 3600 * 1000)
    m, ms_total = divmod(ms_total, 60 * 1000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


#: Back-compat aliases — historically these names lived in different modules.
fmt_srt_time = seconds_to_srt_time
to_srt_time = seconds_to_srt_time


def parse_srt(srt_text: str) -> list[Cue]:
    """Parse SRT text into a list of :class:`Cue` objects.

    Tolerant of blank-line spacing variations and of either ``,`` or ``.`` as
    the millisecond separator. Cues without a valid time line are skipped.
    """
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", srt_text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # The time line is the one matching TIME_RE (usually line 0 or 1; the
        # numeric index, when present, is the line just above it).
        time_idx = next((i for i, ln in enumerate(lines) if TIME_RE.search(ln)), None)
        if time_idx is None:
            continue
        m = TIME_RE.search(lines[time_idx])
        start = (
            int(m[1]) * 3600
            + int(m[2]) * 60
            + int(m[3])
            + int(m[4].ljust(3, "0")) / 1000.0
        )
        end = (
            int(m[5]) * 3600
            + int(m[6]) * 60
            + int(m[7])
            + int(m[8].ljust(3, "0")) / 1000.0
        )
        idx_line = lines[time_idx - 1] if time_idx > 0 else ""
        index = int(idx_line) if idx_line.strip().isdigit() else len(cues) + 1
        text = "\n".join(lines[time_idx + 1 :]).strip()
        cues.append(Cue(index=index, start=start, end=end, text=text))
    return cues


def dump_srt(cues: Iterable[Cue]) -> str:
    """Serialize cues back to SRT text, renumbering from 1."""
    parts = [
        f"{i}\n{seconds_to_srt_time(c.start)} --> {seconds_to_srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues, 1)
    ]
    return "\n".join(parts)


def shift_srt_timestamps(srt_text: str, shift_seconds: float = 0.0) -> str:
    """Shift every cue time line in ``srt_text`` by ``shift_seconds``.

    Negative values shift earlier; resulting negative times clamp to zero.

    >>> srt = '1\\n00:43:12,187 --> 00:43:13,817\\nHello world'
    >>> '00:00:00,187 --> 00:00:01,817' in shift_srt_timestamps(srt, -2592)
    True
    """

    def _shift_line(line: str) -> str:
        m = re.match(r"\s*([\d:,.]+)\s*-->\s*([\d:,.]+)", line)
        if not m:
            return line
        start = srt_time_to_seconds(m[1]) + shift_seconds
        end = srt_time_to_seconds(m[2]) + shift_seconds
        return f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}"

    return "\n".join(
        _shift_line(line) if "-->" in line else line for line in srt_text.split("\n")
    )


#: A WebVTT timestamp; hours are optional (``01:02.500`` or ``00:01:02.500``).
_VTT_TS = r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{1,3})"
#: A WebVTT cue timing line.
_VTT_TIME_LINE_RE = re.compile(rf"^\s*{_VTT_TS}\s*-->\s*{_VTT_TS}")
#: An inline timestamp tag inside cue text: ``<00:00:12.639>``.
_VTT_INLINE_TS_RE = re.compile(rf"<{_VTT_TS}>")
_TAG_RE = re.compile(r"<[^>]+>")


def _vtt_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _vtt_clean(text: str) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    return " ".join(html.unescape(_TAG_RE.sub("", text)).split())


def _vtt_cues(vtt_text: str) -> list[Cue]:
    """Cues of a WebVTT file: the timing line plus the text lines up to an empty line.

    Not :func:`parse_srt`: YouTube cues often open with a whitespace-only line,
    which an SRT blank-line split takes for the end of the cue. Here only a truly
    empty line ends a cue, and anything outside a cue (cue identifiers, ``NOTE``
    and ``STYLE`` blocks, the header) is ignored.
    """
    cues: list[Cue] = []
    in_cue = False
    for line in vtt_text.splitlines():
        m = _VTT_TIME_LINE_RE.match(line)
        if m:
            g = m.groups()
            start, end = _vtt_seconds(*g[:4]), _vtt_seconds(*g[4:])
            cues.append(Cue(index=len(cues) + 1, start=start, end=end, text=""))
            in_cue = True
        elif line == "":
            in_cue = False
        elif in_cue and line.strip():
            cues[-1].text += ("\n" if cues[-1].text else "") + line.strip()
    return cues


def parse_vtt_words(vtt_text: str) -> list[dict]:
    r"""Parse a WebVTT caption file into a word list shaped like Scribe's.

    Each entry is ``{"text", "start", "end", "type": "word"}`` (seconds), the same
    shape :func:`mixing.transcript.transcribe` returns under ``"words"``, so the
    result plugs into the word-level helpers (``words_to_srt``,
    :func:`mixing.audio.segmentation.text_before_segments`, ...).

    YouTube auto-captions carry per-word timings as inline ``<hh:mm:ss.mmm>``
    tags; text before a line's first tag takes the cue start (or, on a later
    line of the same cue, the latest timing seen), and the rolling duplicate lines
    (no inline timings) are ignored. A file with no inline timings falls back to
    one entry per cue (its whole text). A word ends where the next one starts
    (capped at its cue end). Tags are stripped and HTML entities unescaped.

    >>> vtt = '''WEBVTT
    ...
    ... 00:00:12.240 --> 00:00:15.509 align:start position:0%
    ... Music<00:00:12.639><c> with</c><00:00:12.800><c> meaning</c>
    ...
    ... 00:00:15.509 --> 00:00:15.519 align:start position:0%
    ... Music with meaning
    ... '''
    >>> [(w["text"], w["start"]) for w in parse_vtt_words(vtt)]
    [('Music', 12.24), ('with', 12.639), ('meaning', 12.8)]
    >>> parse_vtt_words("1\n00:01.000 --> 00:02.500\nRock &amp; roll\n\nNOTE hi\n")
    [{'text': 'Rock & roll', 'start': 1.0, 'end': 2.5, 'type': 'word'}]
    """
    cues = _vtt_cues(vtt_text)
    timed = []  # (start, text, cue_end)
    for cue in cues:
        latest = cue.start
        for line in cue.text.splitlines():
            parts = _VTT_INLINE_TS_RE.split(line)
            if len(parts) == 1:
                continue  # no inline timings: a rolling duplicate line
            # split() interleaves: [lead, h, m, s, ms, chunk, h, m, s, ms, chunk, ...]
            chunks = [(latest, parts[0])]
            for i in range(1, len(parts), 5):
                chunks.append((_vtt_seconds(*parts[i : i + 4]), parts[i + 4]))
            for t, chunk in chunks:
                latest = max(latest, t)
                text = _vtt_clean(chunk)
                if text:
                    timed.append((t, text, cue.end))
    if not timed:
        timed = [
            (c.start, _vtt_clean(c.text), c.end) for c in cues if _vtt_clean(c.text)
        ]
    words = []
    for i, (start, text, cue_end) in enumerate(timed):
        end = min(timed[i + 1][0], cue_end) if i + 1 < len(timed) else cue_end
        words.append(
            {"text": text, "start": start, "end": max(start, end), "type": "word"}
        )
    return words
