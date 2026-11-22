"""Time authored words by a recogniser's timed words of the same audio.

A script, a cue sheet or a line of dialogue says *what* was meant to be said; a
transcriber (ElevenLabs Scribe, Whisper, ...) says *when* each word it heard was
said. :func:`align_words` walks the two in order (:class:`difflib.SequenceMatcher`)
and gives every authored token a time:

- a token a heard word matches (same :func:`normalize_word` form) takes that
  word's times;
- a run heard differently (``friend`` heard as ``fred``) is spread over the
  misheard words' span;
- a run not heard at all is spread over the gap to the next anchor;

spreading in proportion to token length, so every token gets a plausible slot,
and the result says how many were actually anchored (``matched`` / ``total``,
``coverage``) and by which heard word (``anchors``).

Heard words are split into tokens the way the reference is (``split=``,
default :func:`tokens_of`), so a heard ``"twenty-five"`` or ``"U.S."`` matches
the script's ``twenty``, ``five`` / ``U``, ``S``: each piece takes its share of
the word's span, and anchors the word's own index.

Stdlib only. Heard words may be ``(text, start, end)`` triples or mappings
(Scribe's ``text``/``start``/``end``/``type``, or Whisper's ``word``); Scribe's
``spacing`` and ``audio_event`` entries are skipped.

>>> heard = [("i", 0.0, 0.1), ("did", 0.1, 0.3), ("ask", 0.6, 0.9)]
>>> align_words(["I", "did", "NOT", "ask"], heard).words
[('I', 0.0, 0.1), ('did', 0.1, 0.3), ('NOT', 0.3, 0.6), ('ask', 0.6, 0.9)]
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence, Union

WordTiming = tuple[str, float, float]
"""One timed word: ``(text, start_s, end_s)``."""

HeardWord = Union[WordTiming, Mapping]
"""A recogniser's word: a :data:`WordTiming`, or a mapping with ``text`` (or
``word``), ``start``, ``end`` (and optionally ``type``, as in Scribe's list)."""

APOSTROPHES = "'’"
"""Apostrophes kept inside a word (straight and typographic)."""

TOKEN_PATTERN = re.compile(rf"[^\W_]+(?:[{APOSTROPHES}][^\W_]+)*")
"""A word as written: letters and digits, an apostrophe inside a word kept."""

SPOKEN_WORD_TYPE = "word"
"""The Scribe word ``type`` that is speech (``spacing``/``audio_event`` are not)."""

TEXT_KEYS = ("text", "word")
"""Keys a heard-word mapping may carry its text under (Scribe, Whisper)."""

TIME_DECIMALS = 6
"""Decimals kept on spread (interpolated) times, so sums do not drift visibly."""


def tokens_of(text: str) -> list[str]:
    """The words of ``text`` as written: punctuation dropped, an apostrophe
    (straight or typographic) inside a word kept, case kept. What a line's
    authored tokens are.

    >>> tokens_of("Hello, there — it’s ME!")
    ['Hello', 'there', 'it’s', 'ME']
    """
    return TOKEN_PATTERN.findall(text)


def normalize_word(text: str) -> str:
    """The form two words are matched by: :func:`tokens_of` the case-folded
    ``text``, typographic apostrophes made straight, joined by single spaces.

    >>> normalize_word(" It’s,")
    "it's"
    """
    return " ".join(tokens_of(text.casefold().replace("’", "'")))


@dataclass(frozen=True)
class AlignedWords:
    """:func:`align_words`'s result.

    ``words`` are the reference tokens timed, in reference order with
    non-decreasing starts; ``matched`` of the ``total`` (every reference token,
    those that normalize to nothing included) were anchored by a heard word
    (the rest interpolated); ``anchors`` maps a reference index to the index,
    in the heard list as given, of the word that anchored it.
    """

    words: list[WordTiming]
    matched: int
    total: int
    anchors: dict[int, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """The share of reference tokens anchored by a heard word, 0..1."""
        return self.matched / self.total if self.total else 1.0


def _time(value, what: str, i: int) -> float:
    if value is None:
        raise ValueError(f"heard word {i} has no {what} time")
    t = float(value)
    if not math.isfinite(t):
        raise ValueError(f"heard word {i} has a non-finite {what} time: {value!r}")
    return t


def _as_timing(word: HeardWord, i: int) -> WordTiming | None:
    """``word`` as ``(text, start, end)``, or None for a non-speech Scribe entry."""
    if isinstance(word, Mapping):
        if word.get("type", SPOKEN_WORD_TYPE) != SPOKEN_WORD_TYPE:
            return None
        key = next((k for k in TEXT_KEYS if k in word), None)
        if key is None:
            raise ValueError(f"heard word {i} has no text (keys {TEXT_KEYS})")
        text, start, end = word[key], word.get("start"), word.get("end")
    else:
        text, start, end = word
    return (str(text), _time(start, "start", i), _time(end, "end", i))


def _speech(
    heard: Sequence[HeardWord],
    normalize: Callable[[str], str],
    split: Optional[Callable[[str], Sequence[str]]],
) -> tuple[list[WordTiming], list[str], list[int]]:
    """The heard words that are speech, split into pieces: timings, normalized
    forms, and each piece's word's index in ``heard`` as given."""
    timed, forms, origin = [], [], []
    last_start = -math.inf
    for i, word in enumerate(heard):
        timing = _as_timing(word, i)
        if timing is None:
            continue
        text, start, end = timing
        if start < last_start:
            raise ValueError(
                f"heard words are out of order: word {i} starts at {start}, "
                f"before the previous one ({last_start})"
            )
        last_start = start
        pieces = [p for p in (split(text) if split else [text]) if normalize(p)]
        weights = [len(p) for p in pieces]
        t, total = start, float(sum(weights)) or 1.0
        for k, (piece, w) in enumerate(zip(pieces, weights)):
            length = (max(end, start) - start) * (w / total)
            last_piece = k == len(pieces) - 1
            stop = end if last_piece else round(t + length, TIME_DECIMALS)
            timed.append((piece, round(t, TIME_DECIMALS), stop))
            forms.append(normalize(piece))
            origin.append(i)
            t += length
    return timed, forms, origin


def align_words(
    reference: Sequence[str],
    heard: Sequence[HeardWord],
    *,
    bounds: tuple[float, float] | None = None,
    normalize: Callable[[str], str] = normalize_word,
    split: Optional[Callable[[str], Sequence[str]]] = tokens_of,
) -> AlignedWords:
    """Time the ``reference`` tokens (what was meant to be said) by a
    recogniser's ``heard`` words of the same audio.

    ``bounds`` (``(start, end)`` of the audio) time the ends when no anchor
    does; without it the first and last heard words do. ``normalize`` is the
    matching key (default :func:`normalize_word`: case-folded, punctuation
    dropped); pass your own for a language with elisions or a phonetic key.
    ``split`` cuts a heard word into tokens like the reference's (default
    :func:`tokens_of`; ``None`` keeps each heard word whole). Heard words must
    come in time order (overlaps are fine); times must be finite.

    The walk is ``difflib``'s, roughly quadratic: about 0.2 s for 5,000 words,
    4 s for 20,000; align long recordings by section.

    >>> heard = [("i", 0.0, 0.1), ("did", 0.1, 0.3), ("ask", 0.6, 0.9)]
    >>> aligned = align_words(["I", "did", "NOT", "ask"], heard)
    >>> aligned.words
    [('I', 0.0, 0.1), ('did', 0.1, 0.3), ('NOT', 0.3, 0.6), ('ask', 0.6, 0.9)]
    >>> aligned.matched, aligned.total, aligned.anchors
    (3, 4, {0: 0, 1: 1, 3: 2})

    Scribe's word list works as is (spacing and audio events are skipped, and
    ``anchors`` index the list as given):

    >>> scribe = [{"text": "Hi", "start": 0.0, "end": 0.2, "type": "word"},
    ...           {"text": " ", "start": 0.2, "end": 0.3, "type": "spacing"},
    ...           {"text": "there.", "start": 0.3, "end": 0.6, "type": "word"}]
    >>> align_words(["hi", "there"], scribe).anchors
    {0: 0, 1: 2}
    """
    if isinstance(reference, str):
        raise TypeError("reference is a sequence of tokens: pass tokens_of(text)")
    timed, asr, origin = _speech(heard, normalize, split)
    ref = [normalize(w) for w in reference]
    first = float(bounds[0]) if bounds else (timed[0][1] if timed else 0.0)
    last = float(bounds[1]) if bounds else (timed[-1][2] if timed else 0.0)
    out: list[WordTiming] = []
    anchors: dict[int, int] = {}
    prev_end = first

    def spread(indices: range, lo: float, hi: float) -> None:
        nonlocal prev_end
        lo = max(lo, prev_end)
        hi = max(hi, lo)
        weights = [max(1, len(ref[k]) or len(reference[k])) for k in indices]
        total = float(sum(weights)) or 1.0
        t = lo
        for k, w in zip(indices, weights):
            length = (hi - lo) * (w / total)
            out.append(
                (reference[k], round(t, TIME_DECIMALS), round(t + length, TIME_DECIMALS))
            )
            t += length
        prev_end = hi

    opcodes = difflib.SequenceMatcher(None, ref, asr, autojunk=False).get_opcodes()
    for n, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            for k in range(i2 - i1):
                start = max(timed[j1 + k][1], prev_end)
                end = max(timed[j1 + k][2], start)
                out.append((reference[i1 + k], start, end))
                anchors[i1 + k] = origin[j1 + k]
                prev_end = end
        elif tag == "replace":
            spread(range(i1, i2), timed[j1][1], timed[j2 - 1][2])
        elif tag == "delete":
            # Not heard at all: the gap up to the next heard word, or the end.
            nxt = next(
                (timed[op[3]][1] for op in opcodes[n + 1 :] if op[3] < op[4]), last
            )
            spread(range(i1, i2), prev_end, nxt)
        else:  # "insert": words heard that the text lacks; time passes
            prev_end = max(prev_end, timed[j2 - 1][2])
    return AlignedWords(words=out, matched=len(anchors), total=len(ref), anchors=anchors)
