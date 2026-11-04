"""Conversions between :class:`Audio` / pydub segments and float sample arrays.

The one place that turns pydub's integer PCM into float samples in [-1, 1]
(and back), so measurement, leveling, voice, segmentation and beat analysis
share a single implementation instead of each carrying its own copy.

- :func:`segment_samples` / :func:`audio_samples`: segment / ``Audio`` -> float
  array shaped ``(n, channels)`` (``segment_samples`` and ``resample`` live in
  the numpy-only leaf :mod:`mixing.audio._pcm`, re-exported here).
- :func:`to_audio`: float array -> ``Audio`` in the sample width of a template
  segment (values beyond full scale saturate).
- :func:`as_audio`: accept a path or an ``Audio``.
- :func:`resample`: polyphase resampling of a float array.
- :func:`ffmpeg_float`: float samples through an ffmpeg ``-af`` chain, unclipped.
- :func:`float_input`: any audio input (``Audio``, path, ``AudioSamples``, or an
  array + rate) as float samples, with the way back to the input's kind.

>>> from pydub import AudioSegment
>>> segment_samples(AudioSegment.silent(100, frame_rate=8000)).shape
(800, 1)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np

from ..util import ffmpeg_exe, require_package
from ._pcm import resample, segment_samples  # noqa: F401  (resample re-exported)
from .audio_ops import Audio, AudioSamples

#: ffmpeg raw sample format used to pipe unclipped float samples.
FLOAT_PCM = "f64le"
_INT_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}  # pydub stores 8-bit signed
#: Default name stem of a written result when the input has no file name.
DEFAULT_STEM = "audio"
#: Sample width (bytes) used when an array result is written to a file: 32-bit
#: PCM, so writing adds no audible quantisation.
ARRAY_WRITE_SAMPLE_WIDTH = 4


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


# ---------------------------------------------------------------------------
# Inputs: Audio, path, AudioSamples, or array + sr
# ---------------------------------------------------------------------------

AudioLike = Union[str, Path, Audio, AudioSamples, np.ndarray, Any]


@dataclass(frozen=True)
class FloatInput:
    """Float samples of an input, and how to give a result back in its kind
    (:func:`float_input`)."""

    samples: np.ndarray  # (n, channels), float64 in [-1, 1]
    sr: int
    restore: Callable[[np.ndarray], Any]  # float result -> the input's kind
    as_audio: Callable[[np.ndarray], Audio]  # float result -> a writable Audio
    stem: str  # name stem for a written result


def _array_input(array: np.ndarray, sr: Optional[int]) -> FloatInput:
    if sr is None:
        raise ValueError("an array input needs its sample rate: pass sr=")
    if array.ndim not in (1, 2):
        raise ValueError(f"samples must be (n,) or (n, channels), got {array.shape}")
    dtype = array.dtype
    if np.issubdtype(dtype, np.signedinteger):  # PCM: scale like segment_samples
        full_scale = float(np.iinfo(dtype).max) + 1
    elif np.issubdtype(dtype, np.floating):
        full_scale = None
    else:
        raise TypeError(f"samples must be float or signed-integer PCM, got {dtype}")
    x = np.asarray(array, dtype=np.float64) / (full_scale or 1.0)
    x = x[:, None] if array.ndim == 1 else x

    def restore(y):
        y = y[:, 0] if array.ndim == 1 else y
        if full_scale is None:
            return y.astype(dtype)
        info = np.iinfo(dtype)
        return np.clip(np.round(y * full_scale), info.min, info.max).astype(dtype)

    def audio(y):
        AudioSegment = require_package("pydub").AudioSegment
        like = AudioSegment(
            data=b"",
            sample_width=ARRAY_WRITE_SAMPLE_WIDTH,
            frame_rate=int(sr),
            channels=x.shape[1],
        )
        return to_audio(y, like)

    return FloatInput(x, int(sr), restore, audio, DEFAULT_STEM)


def float_input(audio: AudioLike, sr: Optional[int] = None) -> FloatInput:
    """``audio`` (Audio, path, AudioSamples, pydub segment, or array + ``sr``) as
    float samples ``(n, channels)``, with the way back to its kind.

    An array is float in [-1, 1] or signed-integer PCM (scaled by its full scale)
    and needs ``sr``; for anything else ``sr``, if given, must be its rate.
    Shared by :func:`~mixing.audio.separation.remove_talk` and
    :func:`~mixing.audio.censor.bleep`.
    """
    if isinstance(audio, np.ndarray):
        return _array_input(audio, sr)
    if isinstance(audio, AudioSamples):
        segment = audio._audio.get_sample_slice(audio.start_sample, audio.end_sample)
        audio = Audio(segment)
    audio = as_audio(audio)
    samples, rate = audio_samples(audio)
    if sr is not None and sr != rate:
        raise ValueError(f"sr={sr} given for audio at {rate} Hz")
    segment = audio._get_segment()
    restore = lambda y: to_audio(y, segment, time_unit=audio.time_unit)
    stem = Path(audio.src_path).stem if audio.src_path else DEFAULT_STEM
    return FloatInput(samples, rate, restore, restore, stem)
