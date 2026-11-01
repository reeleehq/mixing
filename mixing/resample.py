"""Polyphase resampling of float arrays: the one resampler mixing uses.

A leaf (numpy, with scipy imported when a rate actually changes), so both the
media stack (:mod:`mixing.audio`, through ``mixing.audio._pcm``) and the
array-only :mod:`mixing.loudness` share it without the loudness layer importing
the media stack.

>>> import numpy as np
>>> resample(np.zeros((480, 2)), 48000, 16000).shape
(160, 2)
"""

from __future__ import annotations

from math import gcd

import numpy as np

from mixing.util import require_package

__all__ = ["resample"]


def resample(data: np.ndarray, fs: int, target: int) -> np.ndarray:
    """``data`` (along axis 0) resampled from ``fs`` to ``target`` Hz (polyphase,
    anti-aliased); returned as is when the rates are equal."""
    if fs == target:
        return data
    resample_poly = require_package("scipy.signal").resample_poly
    g = gcd(fs, target)
    return resample_poly(data, target // g, fs // g, axis=0)
