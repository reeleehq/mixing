"""A media file's audio as a stream of float chunks, and back, through ffmpeg.

:func:`decode_float_chunks` runs ffmpeg once and hands back the sample rate
and an iterator of ``(k, channels)`` float32 arrays in [-1, 1], at the file's
own rate and channel count (no downmix; resampled only when asked). Only a
chunk is held at a time, so a consumer such as
:func:`mixing.loudness.stream_loudness` works on an hour-long track in bounded
memory. :func:`encode_float_chunks` is the way back: chunks piped into one
ffmpeg that writes a file. Needs numpy and an ffmpeg, not the media stack.

Both take ``ffmpeg=`` (the executable to run; default
:func:`mixing.util.ffmpeg_exe`, which prefers imageio-ffmpeg's bundled build:
pass ``ffmpeg="ffmpeg"`` for the one on ``PATH``) and ``timeout=`` (seconds
the ffmpeg run may take in all, time spent waiting on the consumer
included; a watchdog kills it, and what it started, then, and
:class:`FfmpegTimeout` is raised: a ``TimeoutError`` and a ``RuntimeError``).
The runner itself lives in :mod:`mixing.ffmpeg` (stdlib only); ``FfmpegTimeout``
is re-exported here.
:func:`stream_kinds` reads a file's stream kinds from ffmpeg's banner.
"""

from __future__ import annotations

import itertools
from contextlib import suppress
import os
import secrets
import shutil
import struct
import subprocess
import time
import warnings
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from mixing.errors import ProbeError
from mixing.ffmpeg import (
    FfmpegTimeout,
    check_timeout,
    ignore_closed_stdin,
    run_ffmpeg,
)
from mixing.probe import probe
from mixing.util import LOUDNESS_EXTRA, ffmpeg_exe, install_hint

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.decode needs numpy: {install_hint(LOUDNESS_EXTRA)}", name=e.name
    ) from e

#: A half-written file Windows refuses to remove (still in use) is retried this
#: many times, this far apart: half a second in all.
PARTIAL_REMOVE_ATTEMPTS = 10
PARTIAL_REMOVE_WAIT_S = 0.05
#: Windows' error number for a file in use by another process.
WINDOWS_SHARING_VIOLATION = 32

#: Seconds of audio per chunk by default.
DECODE_CHUNK_S = 5.0
#: ffmpeg writes the stream as 32-bit float WAV (its header gives rate and
#: channels).
_WAV_CODEC = "pcm_f32le"
_SAMPLE_BYTES = 4
#: How float samples travel into the encoder.
_RAW_FLOAT = "f32le"
#: Random bytes in a partial file's name, so concurrent writes never share one.
_PARTIAL_TOKEN_BYTES = 6


def _wav_format(stream) -> tuple[int, int]:
    """``(channels, rate)`` from a streamed WAV header, read up to its ``data``
    chunk (ffmpeg writes unknown sizes when piping, so they are not used)."""
    header = stream.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a WAV stream")
    channels = rate = None
    while True:
        chunk = stream.read(8)
        if len(chunk) < 8:
            raise ValueError("WAV stream ended before its data chunk")
        name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
        if name == b"data":
            if channels is None:
                raise ValueError("WAV stream has no fmt chunk")
            return channels, rate
        body = stream.read(size + size % 2)
        if name == b"fmt ":
            channels, rate = struct.unpack("<HI", body[2:8])


def stream_kinds(
    path, *, ffmpeg: Optional[str] = None, timeout: Optional[float] = None
) -> list[str]:
    """The kind of each of ``path``'s streams (``"audio"``, ``"video"`` for a
    picture or cover art, ``"subtitle"``, ``"data"``, ...), from ffmpeg's
    banner; ``[]`` for a file ffmpeg cannot open.

    An adapter over :attr:`mixing.probe.MediaProbe.streams` (the banner
    reading, so ``ffmpeg=`` is the binary that reads it).
    """
    try:
        found = probe(path, ffprobe=None, ffmpeg=ffmpeg, timeout=timeout)
    except ProbeError:
        return []
    return [stream.kind for stream in found.streams]


def _decode(
    path,
    chunk_s: float,
    ffmpeg: Optional[str],
    sample_rate: Optional[int],
    timeout: Optional[float],
):
    """First ``(channels, rate)``, then the float chunks."""
    resample = ["-ar", str(sample_rate)] if sample_rate else []
    cmd = [
        ffmpeg or ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-vn",
        "-map", "0:a:0", *resample, "-c:a", _WAV_CODEC, "-f", "wav", "-",
    ]  # fmt: skip
    what = f"decode the audio of {path!s}"
    with run_ffmpeg(cmd, what=what, stdout=subprocess.PIPE, timeout=timeout) as proc:
        try:
            channels, rate = _wav_format(proc.stdout)
        except ValueError:
            channels = None
        if channels is not None:
            yield channels, rate
            frame = _SAMPLE_BYTES * channels
            step = max(1, int(round(chunk_s * rate))) * frame
            while block := proc.stdout.read(step):
                usable = len(block) - len(block) % frame
                yield np.frombuffer(block[:usable], "<f4").reshape(-1, channels)
    if channels is None:  # ffmpeg exited cleanly yet wrote no WAV stream
        raise RuntimeError(f"ffmpeg could not {what}: no WAV stream came out")


def decode_float_chunks(
    path,
    *,
    chunk_s: float = DECODE_CHUNK_S,
    ffmpeg: Optional[str] = None,
    sample_rate: Optional[int] = None,
    timeout: Optional[float] = None,
) -> tuple[int, Iterator[np.ndarray]]:
    """``(rate, chunks)``: the first audio stream of ``path`` as float32 chunks.

    Each chunk is ``(k, channels)`` with ``k`` up to ``chunk_s * rate``, at the
    file's own rate, or resampled by ffmpeg to ``sample_rate``. ffmpeg
    (``ffmpeg``, default :func:`mixing.util.ffmpeg_exe`) starts at once; it is
    stopped when the iterator is exhausted or closed. Raises ``RuntimeError``
    (with ffmpeg's message) when the file has no decodable audio, here or,
    for a stream that breaks partway, from the iterator. With ``timeout``
    (seconds) ffmpeg is killed that long after it started, and
    :class:`FfmpegTimeout` is raised, here (a stalled input: a FIFO nobody
    writes) or from the iterator.
    """
    chunks = _decode(path, chunk_s, ffmpeg, sample_rate, timeout)
    _, rate = next(chunks)
    return rate, chunks


def encode_float_chunks(
    chunks: Iterable[np.ndarray],
    rate: int,
    path,
    *,
    args: Sequence[str] = (),
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Path:
    """Write float chunks to ``path`` through one ffmpeg; return the ``Path``.

    ``chunks`` are ``(k, channels)`` arrays (or 1-D, mono) at ``rate``, in
    [-1, 1], all with the same channel count; one is held at a time. The
    format is the one ``path``'s suffix names (ffmpeg's default codec for it,
    16-bit PCM for ``.wav``); ``args`` are extra output options, such as
    ``("-c:a", "pcm_s24le")`` or ``("-ar", "44100")``. ffmpeg writes a hidden,
    uniquely named file beside ``path`` (beside its target, for a symlink),
    renamed over it on success with the replaced file's permissions, so
    ``path`` is never left half written and may be the very file the chunks
    are decoded from. Raises ``ValueError`` for no audio at all,
    ``RuntimeError`` (with ffmpeg's message) when ffmpeg fails,
    :class:`FfmpegTimeout` when it is still running ``timeout`` seconds after
    it started, finishing the file included (it is killed then). The timeout
    kills ffmpeg, not Python: ``chunks`` that block for ever still hang.
    """
    check_timeout(timeout)  # checked before the first chunk is read
    path = Path(path)
    target = path.resolve()  # replace a symlink's target, not the link
    chunks = (c[:, None] if c.ndim == 1 else c for c in map(np.asarray, chunks))
    first = next((c for c in chunks if len(c)), None)
    if first is None:
        raise ValueError(f"no audio to write to {path.name}")
    channels = first.shape[1]
    partial = target.with_name(
        f".{target.stem}.{secrets.token_hex(_PARTIAL_TOKEN_BYTES)}.partial{target.suffix}"
    )
    cmd = [
        ffmpeg or ffmpeg_exe(), "-v", "error", "-y", "-f", _RAW_FLOAT,
        "-ar", str(rate), "-ac", str(channels), "-i", "pipe:0", *args, str(partial),
    ]  # fmt: skip
    try:
        with run_ffmpeg(
            cmd, what=f"write {path.name}", stdin=subprocess.PIPE, timeout=timeout
        ) as proc:
            with ignore_closed_stdin():
                for chunk in itertools.chain([first], chunks):
                    if chunk.shape[1] != channels:
                        raise ValueError(
                            f"chunks must all have {channels} channels, got {chunk.shape[1]}"
                        )
                    proc.stdin.write(np.ascontiguousarray(chunk, "<f4").tobytes())
        if target.exists():
            shutil.copymode(target, partial)
        os.replace(partial, target)
    finally:
        _remove_partial(partial)
    return path


def _remove_partial(
    partial: Path,
    *,
    attempts: Optional[int] = None,
    wait_s: Optional[float] = None,
) -> None:
    """Remove a half-written file, never raising: it only exists when the write
    already failed, and that error is the one the caller must see.

    Windows can refuse the removal for a moment after the ffmpeg writing it has
    exited (``WinError 32``: in use by another process, typically a scanner), so
    that refusal is retried, ``attempts`` times ``wait_s`` apart. A file still
    refused after that, or any other refusal, is left behind with a warning
    (suppressed under ``-W error``, so it cannot replace the real error either).
    ``attempts`` and ``wait_s`` default to the module's constants, read at call
    time.
    """
    attempts = PARTIAL_REMOVE_ATTEMPTS if attempts is None else attempts
    wait_s = PARTIAL_REMOVE_WAIT_S if wait_s is None else wait_s
    for attempt in range(attempts):
        try:
            partial.unlink(missing_ok=True)
            return
        except PermissionError as e:
            in_use = getattr(e, "winerror", None) == WINDOWS_SHARING_VIOLATION
            if not in_use or attempt == attempts - 1:
                with suppress(Warning):  # -W error must not mask the real error
                    warnings.warn(
                        f"could not remove the partial file {partial}: {e}",
                        stacklevel=3,
                    )
                return
            time.sleep(wait_s)
