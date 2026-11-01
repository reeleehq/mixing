"""Integer PCM to float samples, and resampling: numpy only, no :class:`Audio`.

``resample`` lives in :mod:`mixing.resample` (shared with :mod:`mixing.loudness`)
and is re-exported here.

The leaf the rest of :mod:`mixing.audio` builds on (``audio_ops`` imports it,
so it must not import ``audio_ops``); :mod:`mixing.audio.samples` re-exports
both functions.

>>> from pydub import AudioSegment
>>> segment_samples(AudioSegment.silent(100, frame_rate=8000)).shape
(800, 1)
"""

from __future__ import annotations

import numpy as np

from ..resample import resample  # noqa: F401  (re-exported: the leaf owns it)


def segment_samples(segment, *, dtype=np.float64) -> np.ndarray:
    """Float samples ``(n, channels)`` in [-1, 1] of a pydub segment."""
    data = np.array(segment.get_array_of_samples(), dtype=dtype)
    return data.reshape(-1, segment.channels) / np.dtype(dtype).type(
        1 << (8 * segment.sample_width - 1)
    )
