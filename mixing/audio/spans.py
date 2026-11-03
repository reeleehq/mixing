"""Time spans in seconds: the one parser of "where" for span-based edits.

:func:`as_spans` turns the forms a caller has at hand (talk edges, segments, an
aligner's words, pairs, arrays) into checked ``(start, end)`` pairs, for
:func:`~mixing.audio.separation.remove_talk` and :func:`~mixing.audio.censor.bleep`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real

import numpy as np

#: A span in seconds.
Span = tuple[float, float]


def is_number_pair(obj) -> bool:
    """A ``(start, end)`` tuple or list of two real numbers (not bools).

    >>> is_number_pair((1, 2.5)), is_number_pair([(1, 2), (3, 4)]), is_number_pair((True, 1))
    (True, False, False)
    """
    return (
        isinstance(obj, (tuple, list))
        and len(obj) == 2
        and all(isinstance(v, Real) and not isinstance(v, bool) for v in obj)
    )


def _raw_spans(obj) -> list[Span]:
    """``(start, end)`` pairs, unvalidated, from TalkEdges (its intro/outro),
    Segments, pairs, ``(2,)`` / ``(k, 2)`` arrays, or an iterable of these."""
    if obj is None:
        return []
    if hasattr(obj, "intro") and hasattr(obj, "outro"):  # TalkEdges
        return [(s.start, s.end) for s in (obj.intro, obj.outro) if s is not None]
    if isinstance(obj, Mapping) and "start" in obj and "end" in obj:  # a word
        return [(obj["start"], obj["end"])]
    if hasattr(obj, "start") and hasattr(obj, "end"):  # Segment-like
        return [(obj.start, obj.end)]
    if isinstance(obj, np.ndarray):
        if obj.shape == (2,):
            return [tuple(obj)]
        if obj.ndim == 2 and obj.shape[1] == 2:
            return [tuple(row) for row in obj]
        raise ValueError(f"a span array must be (2,) or (k, 2), got {obj.shape}")
    if is_number_pair(obj):
        return [tuple(obj)]
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return [span for item in obj for span in _raw_spans(item)]
    raise TypeError(
        "spans must be TalkEdges, Segments, {start, end} mappings or (start, end) "
        f"pairs, got {obj!r}"
    )


def as_spans(spans_or_edges) -> list[Span]:
    """``(start, end)`` spans in seconds, each checked to be finite with ``end > start``.

    Accepts a :class:`~mixing.audio.talk_edges.TalkEdges` (its intro and outro), a
    ``Segment`` (or anything with ``start`` and ``end``), a ``{"start", "end"}``
    mapping (an aligner's word), a ``(start, end)`` pair, a ``(2,)`` / ``(k, 2)``
    array, or an iterable of these.

    >>> as_spans([(1, 2), {"start": 3.0, "end": 3.5, "word": "x"}])
    [(1.0, 2.0), (3.0, 3.5)]
    """
    spans = [(float(s), float(e)) for s, e in _raw_spans(spans_or_edges)]
    for s, e in spans:
        if not (np.isfinite(s) and np.isfinite(e) and e > s):
            raise ValueError(f"a span needs finite end > start, got {(s, e)}")
    return spans
