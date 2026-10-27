"""Decode a media file's audio as a stream of float chunks, through ffmpeg.

:func:`decode_float_chunks` runs ffmpeg once and hands back the sample rate
and an iterator of ``(k, channels)`` float32 arrays in [-1, 1], at the file's
own rate and channel count (no resample, no downmix). Only a chunk is held
at a time, so a consumer such as :func:`mixing.loudness.stream_loudness`
works on an hour-long track in bounded memory. Needs numpy and an ffmpeg
(:func:`mixing.util.ffmpeg_exe`), not the media stack.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from typing import Iterator, Optional

import numpy as np

from mixing.util import ffmpeg_exe

#: Seconds of audio per chunk by default.
DECODE_CHUNK_S = 5.0
#: ffmpeg writes the stream as 32-bit float WAV (its header gives rate and
#: channels).
_WAV_CODEC = "pcm_f32le"
_SAMPLE_BYTES = 4


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


def _decode(path, chunk_s: float, ffmpeg: Optional[str]):
    """First ``(channels, rate)``, then the float chunks."""
    cmd = [
        ffmpeg or ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-vn",
        "-map", "0:a:0", "-c:a", _WAV_CODEC, "-f", "wav", "-",
    ]  # fmt: skip
    # stderr to a file: a pipe nobody reads fills up and blocks ffmpeg
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors
        ) as proc:
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
        if proc.returncode != 0 or channels is None:
            errors.seek(0)
            message = errors.read().decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg could not decode the audio of {path!s}: {message}"
            )


def decode_float_chunks(
    path, *, chunk_s: float = DECODE_CHUNK_S, ffmpeg: Optional[str] = None
) -> tuple[int, Iterator[np.ndarray]]:
    """``(rate, chunks)``: the first audio stream of ``path`` as float32 chunks.

    Each chunk is ``(k, channels)`` with ``k`` up to ``chunk_s * rate``. ffmpeg
    (``ffmpeg``, default :func:`mixing.util.ffmpeg_exe`) starts at once; it is
    stopped when the iterator is exhausted or closed. Raises ``RuntimeError``
    (with ffmpeg's message) when the file has no decodable audio, here or,
    for a stream that breaks partway, from the iterator.
    """
    chunks = _decode(path, chunk_s, ffmpeg)
    _, rate = next(chunks)
    return rate, chunks
