"""Splice new audio into a span of a take: cut, insert, fade the joins.

Re-rendering one line of a recorded dialogue, a fluffed word in a narration, a bar
of a song: ``base[:start] + insert + base[end:]``, each join faded so there is no
click and no surviving tail, every cut on a sample boundary.

- :func:`replace_span`: arrays in, array out (numpy only).
- :func:`replace_span_file`: a file in, a file out, streamed through
  :mod:`mixing.decode` in bounded memory, encoded once. Run it on a **lossless
  master** (FLAC, WAV) and encode the deliverable from the result, so lines that
  were not edited do not degrade generation after generation.

Where to cut is the caller's knowledge: cut inside the pause around the span
(mid-pause between two lines, say), not on an aligner's word edge, and give the
insert the room tone it needs; the fades only remove the click. Needs numpy (and
an ffmpeg for files), not the media stack.

>>> import numpy as np
>>> base = np.ones(10)
>>> replace_span(base, (0.3, 0.6), np.zeros(2), sr=10, fade_s=0).tolist()
[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

from mixing.egress import Output, write_egress
from mixing.util import LOUDNESS_EXTRA, install_hint

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.splice needs numpy: {install_hint(LOUDNESS_EXTRA)}", name=e.name
    ) from e

#: Fade out and in at each join (s): long enough to remove a click from a cut
#: through room tone, short enough to sit inside any pause between turns
#: (reeleehq/braidio#178); halved further when the piece is shorter.
JOIN_FADE_S = 0.008
#: Suffix on a written result's stem (``take.flac`` -> ``take_spliced.flac``).
SPLICED_SUFFIX = "_spliced"

Span = tuple[float, float]
#: An insert: float samples ``(n,)`` / ``(n, channels)`` at the base's rate, or
#: (for :func:`replace_span_file`) a file, decoded and resampled to it.
Insert = Union[np.ndarray, str, Path]


def _ramp(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, endpoint=False)[:, None]


def _cut_frames(span: Span, sr: int) -> tuple[int, int]:
    start, end = span
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start <= end):
        raise ValueError(f"the span needs finite 0 <= start <= end, got {span}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    return int(round(start * sr)), int(round(end * sr))


def _faded_insert(insert: np.ndarray, channels: int, fade: int) -> np.ndarray:
    """``insert`` as ``(n, channels)`` float, faded in and out over ``fade``."""
    x = np.asarray(insert, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    if x.ndim != 2 or x.shape[1] not in (1, channels):
        raise ValueError(
            f"the insert must be mono or have the base's {channels} channels, "
            f"got shape {np.shape(insert)}"
        )
    x = np.broadcast_to(x, (len(x), channels)).copy()
    f = min(fade, len(x) // 2)
    if f:
        x[:f] *= _ramp(f)
        x[-f:] *= _ramp(f)[::-1]
    return x


def spliced_chunks(
    chunks: Iterable[np.ndarray],
    cut: tuple[int, int],
    insert: np.ndarray,
    *,
    fade: int,
) -> Iterator[np.ndarray]:
    """Base ``chunks`` (``(k, channels)``, in order) with frames ``[cut[0],
    cut[1])`` replaced by ``insert`` (already shaped and faded): the ``fade``
    frames before the cut fade out, the ``fade`` after it fade in, by absolute
    frame index, so the result does not depend on where the chunks split. One
    chunk is held at a time; a cut at or past the end puts the insert at the end.
    The engine of both :func:`replace_span` and :func:`replace_span_file`.
    """
    a, b = cut
    out_ramp, in_ramp = _ramp(fade)[::-1], _ramp(fade)  # ends at 0 / starts at 0
    pos, inserted = 0, False
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float64)
        chunk = chunk[:, None] if chunk.ndim == 1 else chunk
        if not len(chunk):
            continue
        lo, hi = pos, pos + len(chunk)
        pos = hi
        if lo < a:  # frames before the cut, the last `fade` of them faded out
            head = chunk[: min(hi, a) - lo]
            first_faded = max(lo, a - fade)
            if first_faded < lo + len(head):
                head = head.copy()
                ramp = out_ramp[first_faded - (a - fade) : lo + len(head) - (a - fade)]
                head[first_faded - lo :] *= ramp
            yield head
        if hi >= a and not inserted:
            yield insert
            inserted = True
        if hi > b:  # frames after the cut, the first `fade` of them faded in
            start = max(lo, b)
            after = chunk[start - lo :]
            if start < b + fade:
                after = after.copy()
                n = min(len(after), b + fade - start)
                after[:n] *= in_ramp[start - b : start - b + n]
            yield after
    if not inserted:  # the cut is at or past the end
        yield insert


def replace_span(
    base: np.ndarray,
    span: Span,
    insert: np.ndarray,
    *,
    sr: int,
    fade_s: float = JOIN_FADE_S,
) -> np.ndarray:
    """``base`` with ``span`` (seconds) replaced by ``insert``, the joins faded.

    ``base`` is float ``(n,)`` or ``(n, channels)``; ``insert`` is mono or has
    ``base``'s channels, at the same rate ``sr`` (the result's length changes by
    the insert's minus the span's). The ``fade_s`` fade out ends at the cut and
    the fade in starts after it; the insert fades in and out too. Frames outside
    the fades are copied exactly. Returns float64, ``base``'s shape kind.
    """
    if fade_s < 0:
        raise ValueError(f"fade_s must be >= 0, got {fade_s}")
    x = np.asarray(base, dtype=np.float64)
    x2 = x[:, None] if x.ndim == 1 else x
    fade = int(round(fade_s * sr))
    piece = _faded_insert(insert, x2.shape[1], fade)
    out = np.concatenate(
        [*spliced_chunks([x2], _cut_frames(span, sr), piece, fade=fade)]
        or [np.zeros((0, x2.shape[1]))]
    )
    return out[:, 0] if x.ndim == 1 else out


def _insert_samples(insert: Insert, *, rate: int, decode) -> np.ndarray:
    if isinstance(insert, np.ndarray):
        return insert
    _, chunks = decode(insert, sample_rate=rate)
    parts = list(chunks)
    return np.concatenate(parts) if parts else np.zeros((0, 1))


def replace_span_file(
    src,
    span: Span,
    insert: Insert,
    *,
    output: Output = None,
    fade_s: float = JOIN_FADE_S,
    encode_args: Sequence[str] = (),
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
):
    """Write ``src``'s audio with ``span`` replaced by ``insert``, streamed, once.

    ``src`` is decoded in chunks (:func:`mixing.decode.decode_float_chunks`) and
    re-encoded once (:func:`mixing.decode.encode_float_chunks`, ``encode_args``
    extra options such as ``("-c:a", "pcm_s24le")``), so memory is bounded by a
    chunk and the insert. ``insert`` is float samples at ``src``'s rate or a file
    (decoded and resampled to it), mono or with ``src``'s channels. ``output``
    follows mixing's ``output`` protocol (default: beside ``src`` with
    ``SPLICED_SUFFIX``, same format): give a lossless format and encode the
    deliverable from it. It may be ``src`` itself (replaced once the new file is
    complete), unless ``src`` has more than audio. ``ffmpeg`` and ``timeout``
    are :mod:`mixing.decode`'s (each run bounded).

    Returns the ``Path`` written (or what a callable ``output`` returns).
    """
    from mixing.decode import (
        decode_float_chunks,
        encode_float_chunks,
        refuse_dropping_streams,
    )

    if fade_s < 0:
        raise ValueError(f"fade_s must be >= 0, got {fade_s}")
    src = Path(src)

    def decode(path, **kw):
        return decode_float_chunks(path, ffmpeg=ffmpeg, timeout=timeout, **kw)

    def write(path: Path) -> None:
        refuse_dropping_streams(src, path, ffmpeg=ffmpeg, timeout=timeout)
        rate, chunks = decode(src)
        try:
            first = next(chunks)
        except StopIteration:
            raise ValueError(f"{src.name} has no audio to splice into") from None
        channels = first.shape[1]
        fade = int(round(fade_s * rate))
        piece = _faded_insert(
            _insert_samples(insert, rate=rate, decode=decode), channels, fade
        )
        body = spliced_chunks(
            _chain_first(first, chunks), _cut_frames(span, rate), piece, fade=fade
        )
        encode_float_chunks(
            body, rate, path, args=encode_args, ffmpeg=ffmpeg, timeout=timeout
        )

    default_path = src.with_name(f"{src.stem}{SPLICED_SUFFIX}{src.suffix}")
    return write_egress(output, default_path=default_path, write=write)


def _chain_first(first: np.ndarray, rest: Iterator[np.ndarray]):
    yield first
    yield from rest


__all__ = [
    "JOIN_FADE_S",
    "SPLICED_SUFFIX",
    "replace_span",
    "replace_span_file",
    "spliced_chunks",
]
