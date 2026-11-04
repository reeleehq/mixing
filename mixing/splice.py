"""Splice new audio into a span of a take: cut, insert, fade the joins.

Re-rendering one line of a recorded dialogue, a fluffed word in a narration, a bar
of a song: ``base[:start] + insert + base[end:]``, each join faded so there is no
click and no surviving tail, every cut on a sample boundary.

- :func:`replace_span`: arrays in, array out (numpy only).
- :func:`replace_span_file`: a file in, a file out, streamed through
  :mod:`mixing.decode` in bounded memory, encoded once, in the source's own PCM
  format by default. Run it on a **lossless master** (FLAC, WAV) and encode the
  deliverable from the result, so lines that were not edited do not degrade
  generation after generation.
- :func:`spliced_chunks`: the engine both use, over a stream of chunks.

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

import os
from contextlib import closing
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

from mixing.edit import JOIN_FADE_S  # one join fade for the package
from mixing.egress import Output, write_egress
from mixing.util import LOUDNESS_EXTRA, install_hint

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.splice needs numpy: {install_hint(LOUDNESS_EXTRA)}", name=e.name
    ) from e

#: Suffix on a written result's stem (``take.flac`` -> ``take_spliced.flac``).
SPLICED_SUFFIX = "_spliced"

Span = tuple[float, float]
#: An insert: float samples ``(n,)`` / ``(n, channels)`` at the base's rate, or
#: (for :func:`replace_span_file`) a file, decoded and resampled to it.
Insert = Union[np.ndarray, str, os.PathLike]


def _ramp(n: int) -> np.ndarray:
    """``n`` gains rising from 0 (inclusive) towards 1, as a column."""
    return np.linspace(0.0, 1.0, n, endpoint=False)[:, None]


def _fade_frames(fade_s: float, sr: int) -> int:
    if not fade_s >= 0:
        raise ValueError(f"fade_s must be >= 0, got {fade_s}")
    return int(round(fade_s * sr))


def _cut_frames(span: Span, sr: int) -> tuple[int, int]:
    start, end = span
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start <= end):
        raise ValueError(f"the span needs finite 0 <= start <= end, got {span}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    return int(round(start * sr)), int(round(end * sr))


def _shaped(x, channels: Optional[int]) -> np.ndarray:
    """``x`` as float ``(n, channels)``: a mono insert goes into every channel."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    if channels is None:
        return x
    if x.ndim != 2 or x.shape[1] not in (1, channels):
        raise ValueError(
            f"the insert must be mono or have the base's {channels} channels, "
            f"got shape {x.shape}"
        )
    return np.broadcast_to(x, (len(x), channels)).copy()


def _faded_in_and_out(x: np.ndarray, fade: int) -> np.ndarray:
    f = min(fade, len(x) // 2)  # a short insert gets shorter fades
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
    """Base ``chunks`` (``(k, channels)`` or mono, in order) with frames
    ``[cut[0], cut[1])`` replaced by ``insert`` (mono or the base's channels).

    The ``fade`` frames before the cut fade out, the ``fade`` after it fade in,
    and the insert fades in and out (over at most half its length). A cut at or
    past the end puts the insert after the base's last frames, which still fade
    out. Fades go by absolute frame index, so the result does not depend on
    where the chunks split; one chunk plus ``fade`` frames are held at a time.
    Yields ``(k, channels)`` float64 chunks.
    """
    a, b = cut
    if not 0 <= a <= b:
        raise ValueError(f"the cut needs 0 <= start <= end frames, got {cut}")
    if fade < 0:
        raise ValueError(f"fade must be >= 0 frames, got {fade}")
    held = None  # the last frames before the cut, faded out once the cut is reached
    pos, channels, inserted = 0, None, False

    def through_the_cut():
        nonlocal inserted
        inserted = True
        if held is not None and len(held):
            out = held.copy()
            f = min(fade, len(out))
            if f:
                out[-f:] *= _ramp(f)[::-1]
            yield out
        yield _faded_in_and_out(_shaped(insert, channels), fade)

    for chunk in chunks:
        chunk = _shaped(chunk, None)
        if not len(chunk):
            continue
        channels = chunk.shape[1] if channels is None else channels
        lo, hi = pos, pos + len(chunk)
        pos = hi
        if lo < a:  # before the cut: hold the last `fade` frames back
            head = chunk[: min(hi, a) - lo]
            held = head if held is None else np.concatenate([held, head])
            if len(held) > fade:
                yield held[: len(held) - fade]
                held = held[len(held) - fade :]
        if hi >= a and not inserted:
            yield from through_the_cut()
        if hi > b:  # after the cut: its first `fade` frames fade in
            start = max(lo, b)
            after = chunk[start - lo :]
            if start < b + fade:
                after = after.copy()
                n = min(len(after), b + fade - start)
                after[:n] *= _ramp(fade)[start - b : start - b + n]
            yield after
    if not inserted:  # the cut is at or past the end
        yield from through_the_cut()


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
    the insert's minus the span's). A span past the end appends. Frames outside
    the ``fade_s`` fades are copied exactly. Returns float64, ``base``'s shape kind.
    """
    x = np.asarray(base, dtype=np.float64)
    pieces = spliced_chunks(
        [x], _cut_frames(span, sr), insert, fade=_fade_frames(fade_s, sr)
    )
    out = np.concatenate(list(pieces))
    return out[:, 0] if x.ndim == 1 else out


def same_pcm_args(src, path, *, ffmpeg: Optional[str] = None, timeout=None):
    """ffmpeg output options that keep ``src``'s PCM sample format in ``path``.

    Encoding float chunks to ``.wav`` defaults to 16-bit, so a 24-bit or float
    master would lose depth on a rewrite. When ``path`` has ``src``'s suffix and
    ``src``'s audio is PCM, this is ``("-c:a", <its codec>)``; otherwise ``()``
    (FLAC and lossy formats keep ffmpeg's default encoder).
    """
    from mixing.probe import probe

    if Path(path).suffix.lower() != Path(src).suffix.lower():
        return ()
    found = probe(src, ffmpeg=ffmpeg, timeout=timeout)
    codecs = [s.codec for s in found.streams if s.kind == "audio" and s.codec]
    return ("-c:a", codecs[0]) if codecs and codecs[0].startswith("pcm_") else ()


def _insert_samples(insert: Insert, *, rate: int, decode) -> np.ndarray:
    if not isinstance(insert, (str, os.PathLike)):
        return np.asarray(insert)
    _, chunks = decode(insert, sample_rate=rate)
    with closing(chunks):
        parts = list(chunks)
    return np.concatenate(parts) if parts else np.zeros((0, 1))


def replace_span_file(
    src,
    span: Span,
    insert: Insert,
    *,
    output: Output = None,
    fade_s: float = JOIN_FADE_S,
    encode_args: Optional[Sequence[str]] = None,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
):
    """Write ``src``'s audio with ``span`` replaced by ``insert``, streamed, once.

    ``src`` is decoded in chunks (:func:`mixing.decode.decode_float_chunks`) and
    re-encoded once (:func:`mixing.decode.encode_float_chunks`), so memory is
    bounded by a chunk and the insert. ``encode_args`` are extra ffmpeg output
    options; ``None`` (default) keeps ``src``'s PCM format when writing its own
    format (:func:`same_pcm_args`: a 24-bit WAV master stays 24-bit). ``insert``
    is float samples at ``src``'s rate or a file (decoded and resampled to it),
    mono or with ``src``'s channels. ``output`` follows mixing's ``output``
    protocol (default: beside ``src`` with ``SPLICED_SUFFIX``, same format):
    give a lossless format and encode the deliverable from it. It may be ``src``
    itself (replaced once the new file is complete), unless ``src`` has more
    than audio. ``ffmpeg`` and ``timeout`` are :mod:`mixing.decode`'s: each
    ffmpeg run is bounded on its own (the insert is decoded before ``src`` is
    opened, so it does not spend ``src``'s budget).

    Returns the ``Path`` written (or what a callable ``output`` returns).
    """
    from mixing.decode import (
        decode_float_chunks,
        encode_float_chunks,
        refuse_dropping_streams,
    )
    from mixing.probe import probe

    src = Path(src)

    def decode(path, **kw):
        return decode_float_chunks(path, ffmpeg=ffmpeg, timeout=timeout, **kw)

    def write(path: Path) -> None:
        refuse_dropping_streams(src, path, ffmpeg=ffmpeg, timeout=timeout)
        rate = probe(src, ffmpeg=ffmpeg, timeout=timeout).sample_rate
        if not rate:
            raise ValueError(f"{src.name} has no audio to splice into")
        fade, cut = _fade_frames(fade_s, rate), _cut_frames(span, rate)
        piece = _insert_samples(insert, rate=rate, decode=decode)
        args = (
            same_pcm_args(src, path, ffmpeg=ffmpeg, timeout=timeout)
            if encode_args is None
            else tuple(encode_args)
        )
        decoded_rate, chunks = decode(src, sample_rate=rate)
        with closing(chunks):
            body = spliced_chunks(chunks, cut, piece, fade=fade)
            encode_float_chunks(
                body, decoded_rate, path, args=args, ffmpeg=ffmpeg, timeout=timeout
            )

    default_path = src.with_name(f"{src.stem}{SPLICED_SUFFIX}{src.suffix}")
    return write_egress(output, default_path=default_path, write=write)


__all__ = [
    "JOIN_FADE_S",
    "SPLICED_SUFFIX",
    "replace_span",
    "replace_span_file",
    "same_pcm_args",
    "spliced_chunks",
]
