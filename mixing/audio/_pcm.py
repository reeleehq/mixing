"""Integer PCM to float samples, and resampling: numpy only, no :class:`Audio`.

The leaf the rest of :mod:`mixing.audio` builds on (``audio_ops`` imports it,
so it must not import ``audio_ops``); :mod:`mixing.audio.samples` re-exports
both functions.

>>> from pydub import AudioSegment
>>> segment_samples(AudioSegment.silent(100, frame_rate=8000)).shape
(800, 1)
"""

from __future__ import annotations

from math import gcd

import numpy as np

from ..util import require_package


def segment_samples(segment, *, dtype=np.float64) -> np.ndarray:
    """Float samples ``(n, channels)`` in [-1, 1] of a pydub segment."""
    data = np.array(segment.get_array_of_samples(), dtype=dtype)
    return data.reshape(-1, segment.channels) / np.dtype(dtype).type(
        1 << (8 * segment.sample_width - 1)
    )


def resample(data: np.ndarray, fs: int, target: int) -> np.ndarray:
    """``data`` (along axis 0) resampled from ``fs`` to ``target`` Hz (polyphase)."""
    resample_poly = require_package("scipy.signal").resample_poly

    if fs == target:
        return data
    g = gcd(fs, target)
    return resample_poly(data, target // g, fs // g, axis=0)
