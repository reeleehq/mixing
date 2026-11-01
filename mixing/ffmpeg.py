"""Run one ffmpeg process with a watchdog: the runner :mod:`mixing.decode` and
:mod:`mixing.probe` share.

:func:`run_ffmpeg` is a context manager around ``subprocess.Popen``: ffmpeg's
stderr goes to a temporary file (a pipe nobody reads fills up and blocks it),
``timeout=`` arms a watchdog that kills ffmpeg **and what it started** (a
wrapper script or a Windows shim would otherwise leave the real ffmpeg running,
holding the pipes open) and raises :class:`FfmpegTimeout`, and any exception in
the block kills it. Pure stdlib, so importing it pulls no media library.

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
from typing import Optional

__all__ = [
    "FfmpegTimeout",
    "run_ffmpeg",
    "check_timeout",
    "kill_tree",
    "ignore_closed_stdin",
    "STDERR_TAIL_CHARS",
]

#: How much of ffmpeg's complaint an error carries.
STDERR_TAIL_CHARS = 2000


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


def kill_tree(proc: subprocess.Popen) -> None:
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
        watchdog = threading.Timer(timeout or 0, lambda: (fired.set(), kill_tree(proc)))
        watchdog.daemon = True
        if timeout is not None:
            watchdog.start()
        try:
            try:
                yield proc
            except BaseException:
                kill_tree(proc)
                raise
            finally:
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        with ignore_closed_stdin():
                            pipe.close()
                try:
                    proc.wait()  # still bounded: the watchdog runs until it returns
                except BaseException:  # Ctrl-C no longer reaches its own session
                    kill_tree(proc)
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
            tail = proc.message[-STDERR_TAIL_CHARS:]
            raise RuntimeError(f"ffmpeg could not {what}: {tail}")
