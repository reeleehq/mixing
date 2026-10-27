"""``timeout=`` and ``ffmpeg=`` on the ffmpeg pipes of :mod:`mixing.decode` (#116).

A stalled input (a FIFO nobody writes) used to hang ``file_loudness`` for ever
and leave an orphaned ffmpeg; a watchdog now kills it and a ``TimeoutError``
is raised. An explicit ``ffmpeg=`` is the executable every pipe runs.
"""

import os
import stat
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from mixing.decode import decode_float_chunks, encode_float_chunks  # noqa: E402
from mixing.loudness import file_loudness, level_file  # noqa: E402
from mixing.util import ffmpeg_exe, has_ffmpeg  # noqa: E402

FS = 48000
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not on PATH")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="FIFOs / shell scripts")


def _tone(path, seconds=1.0, channels=2):
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate={FS}", "-t", str(seconds), "-ac", str(channels),
         str(path)],
        check=True,
    )  # fmt: skip
    return path


def _ffmpegs_reading(path) -> list[str]:
    """Running processes whose command line names ``path`` (Linux /proc only)."""
    if not os.path.isdir("/proc"):
        return []
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode(errors="replace")
        except OSError:
            continue
        if str(path) in cmdline and "ffmpeg" in cmdline:
            found.append(pid)
    return found


@needs_ffmpeg
@posix_only
def test_a_stalled_input_times_out_and_leaves_no_ffmpeg(tmp_path):
    fifo = tmp_path / "stalled.wav"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="killed"):
        file_loudness(fifo, timeout=1.0)
    assert time.monotonic() - started < 10
    assert _ffmpegs_reading(fifo) == []


@needs_ffmpeg
def test_the_timeout_bounds_the_whole_decode(tmp_path):
    src = _tone(tmp_path / "long.wav", seconds=30)  # far more than a pipe buffer
    rate, chunks = decode_float_chunks(src, chunk_s=0.05, timeout=0.5)
    with pytest.raises(TimeoutError):
        for _ in chunks:
            time.sleep(0.05)  # a slow consumer: ffmpeg waits on the pipe meanwhile


@needs_ffmpeg
def test_a_decode_within_its_timeout_is_untouched(tmp_path):
    src = _tone(tmp_path / "tone.wav")
    assert file_loudness(src, timeout=60).integrated_lufs == pytest.approx(
        file_loudness(src).integrated_lufs
    )


@needs_ffmpeg
def test_closing_a_stalled_decode_early_does_not_hang(tmp_path):
    src = _tone(tmp_path / "long.wav", seconds=30)
    rate, chunks = decode_float_chunks(src, chunk_s=0.05)
    next(chunks)
    started = time.monotonic()
    chunks.close()  # ffmpeg is blocked writing a full pipe: it is killed, not awaited
    assert time.monotonic() - started < 5
    assert _ffmpegs_reading(src) == []


@needs_ffmpeg
def test_an_encode_fed_too_slowly_times_out_and_leaves_no_file(tmp_path):
    def slow():
        for _ in range(3):
            yield np.zeros((FS // 10, 2))
            time.sleep(1.0)

    with pytest.raises(TimeoutError, match="killed"):
        encode_float_chunks(slow(), FS, tmp_path / "out.wav", timeout=0.5)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [0, -1.0])
def test_timeout_must_be_positive(tmp_path, bad):
    with pytest.raises(ValueError, match="timeout"):
        decode_float_chunks(tmp_path / "x.wav", timeout=bad)
    with pytest.raises(ValueError, match="timeout"):
        encode_float_chunks([np.zeros((10, 1))], FS, tmp_path / "x.wav", timeout=bad)


@needs_ffmpeg
@posix_only
def test_an_explicit_ffmpeg_runs_every_pipe(tmp_path):
    log = tmp_path / "calls.log"
    wrapper = tmp_path / "my-ffmpeg"
    wrapper.write_text(f'#!/bin/sh\necho run >> "{log}"\nexec "{ffmpeg_exe()}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    src = _tone(tmp_path / "tone.wav")

    file_loudness(src, ffmpeg=str(wrapper))
    assert log.read_text().count("run") == 1
    level_file(src, tmp_path / "out.wav", target_lufs=-20.0, ffmpeg=str(wrapper))
    # measure, decode again, encode, meter the written file back
    assert log.read_text().count("run") == 5
    level_file(src, src, target_lufs=-20.0, ffmpeg=str(wrapper))
    assert log.read_text().count("run") == 10  # and the in-place streams check


@needs_ffmpeg
def test_level_file_bounds_each_ffmpeg_run(tmp_path, monkeypatch):
    import mixing.loudness as loudness

    seen = []
    real_decode, real_encode = loudness.decode_float_chunks, loudness.encode_float_chunks

    def decode(*a, **k):
        seen.append(("decode", k.get("timeout")))
        return real_decode(*a, **k)

    def encode(*a, **k):
        seen.append(("encode", k.get("timeout")))
        return real_encode(*a, **k)

    monkeypatch.setattr(loudness, "decode_float_chunks", decode)
    monkeypatch.setattr(loudness, "encode_float_chunks", encode)
    level_file(_tone(tmp_path / "t.wav"), tmp_path / "o.wav", target_lufs=-20.0, timeout=30)
    assert seen == [("decode", 30), ("decode", 30), ("encode", 30), ("decode", 30)]
