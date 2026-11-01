"""Run one ffmpeg process with a watchdog: the runner :mod:`mixing.decode` and
:mod:`mixing.probe` share, and the light operations built on it.

:func:`run_ffmpeg` is a context manager around ``subprocess.Popen``: ffmpeg's
stderr goes to a temporary file (a pipe nobody reads fills up and blocks it),
``timeout=`` arms a watchdog that kills ffmpeg **and what it started** (a
wrapper script or a Windows shim would otherwise leave the real ffmpeg running,
holding the pipes open) and raises :class:`FfmpegTimeout`, and any exception in
the block kills it. Pure stdlib, so importing it pulls no media library.

:func:`audio_track` takes the sound out of a muxed file (a video that a
video-to-audio model answers with, say) as lossless WAV bytes at the stream's
own rate and channel count.

>>> from mixing.ffmpeg import run_ffmpeg  # doctest: +SKIP
>>> with run_ffmpeg([exe, "-i", "talk.mp4"], what="read", timeout=5, check=False) as p:  # doctest: +SKIP
...     pass
>>> "Duration:" in p.message  # doctest: +SKIP
True
"""

from __future__ import annotations

import errno
import math
import numbers
import os
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Optional, Union

from mixing.egress import Output, PathLike, deliver
from mixing.errors import MediaNotFoundError
from mixing.util import ffmpeg_exe

__all__ = [
    "FfmpegTimeout",
    "run_ffmpeg",
    "check_timeout",
    "ignore_closed_stdin",
    "audio_track",
    "DEFAULT_AUDIO_TRACK_CODEC",
    "DEFAULT_AUDIO_TRACK_TIMEOUT",
]

#: How much of ffmpeg's complaint an error carries.
_STDERR_TAIL_CHARS = 2000
#: :func:`audio_track`'s WAV sample format: 16-bit PCM (``"pcm_f32le"`` keeps a
#: float decoder's samples unquantized, at twice the size).
DEFAULT_AUDIO_TRACK_CODEC = "pcm_s16le"
#: Seconds :func:`audio_track` lets ffmpeg run before killing it.
DEFAULT_AUDIO_TRACK_TIMEOUT = 120.0
#: :func:`audio_track`'s file name when ``output`` is a directory.
_AUDIO_TRACK_NAME = "audio.wav"
#: What marks a location ffmpeg opens that is not a local file (``https://``…).
_URL_MARKER = "://"


class FfmpegTimeout(TimeoutError, RuntimeError):
    """ffmpeg ran past its ``timeout`` and was killed.

    A ``TimeoutError``, and a ``RuntimeError`` like every other ffmpeg failure
    here, so a caller catching either sees it.
    """


def check_timeout(timeout) -> Optional[float]:
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


def _kill_tree(proc: subprocess.Popen) -> None:
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
def ignore_closed_stdin():
    """Ignore writing to (or closing) the stdin of an ffmpeg that stopped
    reading: a broken pipe, or ``EINVAL`` on Windows; its exit code says why."""
    try:
        yield
    except OSError as e:
        if not isinstance(e, BrokenPipeError) and e.errno != errno.EINVAL:
            raise


@contextmanager
def run_ffmpeg(
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
    :func:`check_timeout`) a watchdog kills it, and what it started, that long
    after it started, waiting included, and :class:`FfmpegTimeout` is raised
    (the block sees an ordinary end of stream or broken pipe); an ffmpeg that
    finished in time is never reported as timed out.
    """
    timeout = check_timeout(timeout)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=stdout,
            stderr=errors,
            start_new_session=os.name == "posix",
        )
        fired = threading.Event()
        watchdog = threading.Timer(
            timeout or 0, lambda: (fired.set(), _kill_tree(proc))
        )
        watchdog.daemon = True
        if timeout is not None:
            watchdog.start()
        try:
            try:
                yield proc
            except BaseException:
                _kill_tree(proc)
                raise
            finally:
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        with ignore_closed_stdin():
                            pipe.close()
                try:
                    proc.wait()  # still bounded: the watchdog runs until it returns
                except BaseException:  # Ctrl-C no longer reaches its own session
                    _kill_tree(proc)
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


@contextmanager
def _local_media(media: Union[bytes, bytearray, PathLike]):
    """A local path for ``media``: bytes are written to a temporary file (a
    seekable file, which an mp4 with its index at the end needs); a path or
    URL is yielded as is."""
    if not isinstance(media, (bytes, bytearray)):
        text = os.fspath(media)
        if _URL_MARKER not in text and not Path(text).exists():
            raise MediaNotFoundError(f"no such media file: {media}")
        yield text
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "media"
        path.write_bytes(media)
        yield str(path)


def audio_track(
    media: Union[bytes, bytearray, PathLike],
    *,
    output: Output = None,
    codec: str = DEFAULT_AUDIO_TRACK_CODEC,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_AUDIO_TRACK_TIMEOUT,
):
    """The first audio stream of ``media`` as WAV, at its own rate and channels.

    For a muxed answer (a video-to-audio model that returns a video): the sound
    taken out, decoded to PCM (``codec``, default 16-bit) with no resampling and
    no downmix, unlike :func:`mixing.transcript.media.extract_audio` (16 kHz
    mono mp3, for speech-to-text). ffmpeg writes a file, not a pipe, so the WAV
    header carries the final size.

    Args:
        media: the container, as bytes or a path (or any location ffmpeg opens).
        output: the ``output`` protocol (:mod:`mixing.egress`): ``None``
            (default) returns the WAV bytes; a file or directory path writes
            them there and returns the ``Path``; a callable gets the bytes.
        codec: the WAV sample format, an ffmpeg PCM encoder
            (:data:`DEFAULT_AUDIO_TRACK_CODEC`; ``"pcm_f32le"`` for float).
        ffmpeg: the executable (default :func:`mixing.util.ffmpeg_exe`).
        timeout: seconds ffmpeg may run before it is killed
            (:data:`DEFAULT_AUDIO_TRACK_TIMEOUT`; ``None`` for no bound).

    Raises:
        MediaNotFoundError: ``media`` is a path that does not exist.
        FfmpegTimeout: ffmpeg ran past ``timeout`` (a ``RuntimeError``).
        RuntimeError: ffmpeg cannot read it, or it has no audio stream.
        ValueError: ``timeout`` is not ``None`` or a number of seconds > 0.

    >>> wav = audio_track(answer_mp4_bytes)  # doctest: +SKIP
    >>> wav[:4], wav[8:12]  # doctest: +SKIP
    (b'RIFF', b'WAVE')
    """
    check_timeout(timeout)
    exe = ffmpeg or ffmpeg_exe()
    with _local_media(media) as source, tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / _AUDIO_TRACK_NAME
        cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", source,
               "-map", "0:a:0", "-c:a", codec, str(wav)]  # fmt: skip
        with run_ffmpeg(cmd, what="take the audio track out", timeout=timeout):
            pass
        data = wav.read_bytes()
    return deliver(
        data,
        output,
        write=lambda b, path: Path(path).write_bytes(b),
        default_name=_AUDIO_TRACK_NAME,
    )

