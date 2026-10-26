"""Conversions between :class:`Audio` / pydub segments and float sample arrays.

The one place that turns pydub's integer PCM into float samples in [-1, 1]
(and back), so measurement, leveling, voice, segmentation and beat analysis
share a single implementation instead of each carrying its own copy.

- :func:`segment_samples` / :func:`audio_samples`: segment / ``Audio`` -> float
  array shaped ``(n, channels)``.
- :func:`to_audio`: float array -> ``Audio`` in the sample width of a template
  segment (values beyond full scale saturate).
- :func:`as_audio`: accept a path or an ``Audio``.
- :func:`resample`: polyphase resampling of a float array.
- :func:`ffmpeg_float`: float samples through an ffmpeg ``-af`` chain, unclipped.

>>> from pydub import AudioSegment
>>> segment_samples(AudioSegment.silent(100, frame_rate=8000)).shape
(800, 1)
"""

from __future__ import annotations

import subprocess
from math import gcd
from pathlib import Path
from typing import Union

import numpy as np

from ..util import ffmpeg_exe
from .audio_ops import Audio

#: ffmpeg raw sample format used to pipe unclipped float samples.
FLOAT_PCM = "f64le"
_INT_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}  # pydub stores 8-bit signed


def segment_samples(segment, *, dtype=np.float64) -> np.ndarray:
    """Float samples ``(n, channels)`` in [-1, 1] of a pydub segment."""
    data = np.array(segment.get_array_of_samples(), dtype=dtype)
    return data.reshape(-1, segment.channels) / np.dtype(dtype).type(
        1 << (8 * segment.sample_width - 1)
    )


def audio_samples(audio: Audio) -> tuple[np.ndarray, int]:
    """``(float samples shaped (n, channels) in [-1, 1], rate)`` of ``audio``."""
    segment = audio._get_segment()
    return segment_samples(segment), segment.frame_rate


def to_audio(data: np.ndarray, like, *, time_unit: str = "seconds") -> Audio:
    """Float ``data`` as an :class:`Audio` with ``like``'s rate, channels, width.

    Values beyond full scale **saturate** (integer PCM has no headroom).
    """
    width = like.sample_width
    full_scale = float(1 << (8 * width - 1))
    ints = np.clip(np.round(data * full_scale), -full_scale, full_scale - 1)
    if width == 3:  # pydub converts 24-bit to 32-bit, but be safe
        raw = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        raw = ints.astype(_INT_DTYPES[width]).tobytes()
    return Audio(like._spawn(raw), time_unit=time_unit)


def as_audio(audio: Union[str, Path, Audio]) -> Audio:
    """``audio`` itself, or the :class:`Audio` loaded from a path."""
    return audio if isinstance(audio, Audio) else Audio(audio)


def resample(data: np.ndarray, fs: int, target: int) -> np.ndarray:
    """``data`` (along axis 0) resampled from ``fs`` to ``target`` Hz (polyphase)."""
    from scipy.signal import resample_poly

    if fs == target:
        return data
    g = gcd(fs, target)
    return resample_poly(data, target // g, fs // g, axis=0)


def ffmpeg_float(data: np.ndarray, fs: int, filters: str) -> np.ndarray:
    """``data`` (float ``(n, channels)``) through an ffmpeg ``-af`` chain, unclipped."""
    ch = data.shape[1]
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", FLOAT_PCM, "-ar", str(fs), "-ac", str(ch),
         "-i", "pipe:0", "-af", filters, "-f", FLOAT_PCM, "-ar", str(fs),
         "-ac", str(ch), "pipe:1"],
        input=np.ascontiguousarray(data, dtype="<f8").tobytes(),
        capture_output=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg filter {filters!r} failed: {proc.stderr.decode(errors='replace')}"
        )
    out = np.frombuffer(proc.stdout, dtype="<f8")
    return out[: len(out) // ch * ch].reshape(-1, ch)
