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
:func:`stream_kinds` reads a file's stream kinds from ffmpeg's banner.
"""

from __future__ import annotations

import errno
import itertools
import math
import numbers
import os
import re
import secrets
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
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
#: A stream's line in ffmpeg's banner: ``Stream #0:1[0x2](und): Video: ...``.
_STREAM_KIND = re.compile(r"^\s*Stream #\d+:\d+.*?: (\w+):", re.M)
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


class FfmpegTimeout(TimeoutError, RuntimeError):
    """ffmpeg ran past its ``timeout`` and was killed.

    A ``TimeoutError``, and a ``RuntimeError`` like every other ffmpeg failure
    here, so a caller catching either sees it.
    """


def _seconds(timeout) -> Optional[float]:
    """``timeout`` checked: ``None`` (or ``inf``) for no bound, else seconds > 0."""
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real):
        raise ValueError(f"timeout is a number of seconds (or None); got {timeout!r}")
    if not timeout > 0:
        raise ValueError(f"timeout is in seconds, > 0 (or None); got {timeout}")
    if timeout == math.inf:
        return None
    if timeout > threading.TIMEOUT_MAX:
        raise ValueError(f"timeout is too long for a timer ({timeout} s); use None")
    return float(timeout)


def _kill(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and what it started: an ``ffmpeg=`` wrapper that does not
    exec (a shell script, a Windows shim) would otherwise leave the real
    ffmpeg running, holding the pipes open."""
    if proc.poll() is not None:
        return  # reaped: its pid (and group id) may already be someone else's
    if os.name == "posix":  # it leads its own process group (start_new_session)
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True
        )
    with suppress(ProcessLookupError):
        proc.kill()


@contextmanager
def _ffmpeg_gone():
    """Ignore writing to (or closing) the stdin of an ffmpeg that stopped
    reading: a broken pipe, or ``EINVAL`` on Windows; its exit code says why."""
    try:
        yield
    except OSError as e:
        if not isinstance(e, BrokenPipeError) and e.errno != errno.EINVAL:
            raise


@contextmanager
def _ffmpeg(
    cmd: list[str],
    *,
    what: str,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    timeout: Optional[float] = None,
    check: bool = True,
):
    """Run the ffmpeg ``cmd`` for the block; yield the process.

    Its stderr goes to a temporary file (a pipe nobody reads fills up and
    blocks ffmpeg); once it is done the text is ``proc.message``. Any
    exception in the block (a generator closed early included) kills it. On
    the way out its pipes are closed, a broken stdin pipe ignored (ffmpeg
    stopped reading: its exit code says why), and it is waited for. With
    ``check`` a non-zero exit raises ``RuntimeError("ffmpeg could not {what}:
    ...")`` with the tail of its message. With ``timeout`` (seconds, see
    :func:`_seconds`) a watchdog kills it, and what it started, that long
    after it started, waiting included, and :class:`FfmpegTimeout` is raised
    (the block sees an ordinary end of stream or broken pipe); an ffmpeg that
    finished in time is never reported as timed out.
    """
    timeout = _seconds(timeout)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=stdout,
            stderr=errors,
            start_new_session=os.name == "posix",
        )
        fired = threading.Event()
        watchdog = threading.Timer(timeout or 0, lambda: (fired.set(), _kill(proc)))
        watchdog.daemon = True
        if timeout is not None:
            watchdog.start()
        try:
            try:
                yield proc
            except BaseException:
                _kill(proc)
                raise
            finally:
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        with _ffmpeg_gone():
                            pipe.close()
                try:
                    proc.wait()  # still bounded: the watchdog runs until it returns
                except BaseException:  # Ctrl-C no longer reaches its own session
                    _kill(proc)
                    proc.wait()
                    raise
        finally:
            watchdog.cancel()
        errors.seek(0)
        proc.message = errors.read().decode(errors="replace").strip()
        if fired.is_set() and proc.returncode != 0:
            raise FfmpegTimeout(
                f"ffmpeg took more than {timeout:g} s to {what}; it was killed"
            )
        if check and proc.returncode != 0:
            tail = proc.message[-_STDERR_TAIL_CHARS:]
            raise RuntimeError(f"ffmpeg could not {what}: {tail}")


def stream_kinds(
    path, *, ffmpeg: Optional[str] = None, timeout: Optional[float] = None
) -> list[str]:
    """The kind of each of ``path``'s streams (``"audio"``, ``"video"`` for a
    picture or cover art, ``"subtitle"``, ``"data"``, ...), from ffmpeg's
    banner; ``[]`` for a file ffmpeg cannot open."""
    cmd = [ffmpeg or ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)]
    what = f"read the streams of {path!s}"
    with _ffmpeg(cmd, what=what, timeout=timeout, check=False) as proc:
        pass  # with no output named, ffmpeg prints the banner and exits 1
    return [kind.lower() for kind in _STREAM_KIND.findall(proc.message)]


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
    with _ffmpeg(cmd, what=what, stdout=subprocess.PIPE, timeout=timeout) as proc:
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
    _seconds(timeout)  # checked before the first chunk is read
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
        with _ffmpeg(
            cmd, what=f"write {path.name}", stdin=subprocess.PIPE, timeout=timeout
        ) as proc:
            with _ffmpeg_gone():
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
