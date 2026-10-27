"""A media file's audio as a stream of float chunks, and back, through ffmpeg.

:func:`decode_float_chunks` runs ffmpeg once and hands back the sample rate
and an iterator of ``(k, channels)`` float32 arrays in [-1, 1], at the file's
own rate and channel count (no downmix; resampled only when asked). Only a
chunk is held at a time, so a consumer such as
:func:`mixing.loudness.stream_loudness` works on an hour-long track in bounded
memory. :func:`encode_float_chunks` is the way back: chunks piped into one
ffmpeg that writes a file. Needs numpy and an ffmpeg
(:func:`mixing.util.ffmpeg_exe`), not the media stack.
"""

from __future__ import annotations

import itertools
import os
import secrets
import shutil
import struct
import subprocess
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from mixing.util import LOUDNESS_EXTRA, ffmpeg_exe, install_hint

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.decode needs numpy: {install_hint(LOUDNESS_EXTRA)}", name=e.name
    ) from e

#: Seconds of audio per chunk by default.
DECODE_CHUNK_S = 5.0
#: ffmpeg writes the stream as 32-bit float WAV (its header gives rate and
#: channels).
_WAV_CODEC = "pcm_f32le"
_SAMPLE_BYTES = 4
#: How float samples travel into the encoder.
_RAW_FLOAT = "f32le"
#: How much of ffmpeg's complaint an error carries.
_STDERR_TAIL_CHARS = 2000
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


@contextmanager
def _ffmpeg(cmd: list[str], *, what: str, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL):
    """Run the ffmpeg ``cmd`` for the block; yield the process.

    Its stderr goes to a temporary file (a pipe nobody reads fills up and
    blocks ffmpeg). Any exception in the block (a generator closed early
    included) kills it. On the way out its pipes are closed, a broken stdin
    pipe ignored (ffmpeg stopped reading: its exit code says why), and it is
    waited for; a non-zero exit raises ``RuntimeError("ffmpeg could not
    {what}: ...")`` with the tail of its message.
    """
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=errors)
        try:
            yield proc
        except BaseException:
            proc.kill()
            raise
        finally:
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None:
                    with suppress(BrokenPipeError):
                        pipe.close()
            proc.wait()
        if proc.returncode != 0:
            errors.seek(0)
            message = errors.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg could not {what}: {message[-_STDERR_TAIL_CHARS:]}")


def _decode(path, chunk_s: float, ffmpeg: Optional[str], sample_rate: Optional[int]):
    """First ``(channels, rate)``, then the float chunks."""
    resample = ["-ar", str(sample_rate)] if sample_rate else []
    cmd = [
        ffmpeg or ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-vn",
        "-map", "0:a:0", *resample, "-c:a", _WAV_CODEC, "-f", "wav", "-",
    ]  # fmt: skip
    what = f"decode the audio of {path!s}"
    with _ffmpeg(cmd, what=what, stdout=subprocess.PIPE) as proc:
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
) -> tuple[int, Iterator[np.ndarray]]:
    """``(rate, chunks)``: the first audio stream of ``path`` as float32 chunks.

    Each chunk is ``(k, channels)`` with ``k`` up to ``chunk_s * rate``, at the
    file's own rate, or resampled by ffmpeg to ``sample_rate``. ffmpeg
    (``ffmpeg``, default :func:`mixing.util.ffmpeg_exe`) starts at once; it is
    stopped when the iterator is exhausted or closed. Raises ``RuntimeError``
    (with ffmpeg's message) when the file has no decodable audio, here or,
    for a stream that breaks partway, from the iterator.
    """
    chunks = _decode(path, chunk_s, ffmpeg, sample_rate)
    _, rate = next(chunks)
    return rate, chunks


def encode_float_chunks(
    chunks: Iterable[np.ndarray],
    rate: int,
    path,
    *,
    args: Sequence[str] = (),
    ffmpeg: Optional[str] = None,
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
    ``RuntimeError`` (with ffmpeg's message) when ffmpeg fails.
    """
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
        with _ffmpeg(cmd, what=f"write {path.name}", stdin=subprocess.PIPE) as proc:
            with suppress(BrokenPipeError):  # ffmpeg stopped reading: see its exit
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
        partial.unlink(missing_ok=True)
    return path
