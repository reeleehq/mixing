"""Resampling of float arrays: the one resampler mixing uses.

A leaf (numpy, with scipy imported when a rate actually changes), so both the
media stack (:mod:`mixing.audio`, through ``mixing.audio._pcm``) and the
array-only :mod:`mixing.loudness` share it without the loudness layer importing
the media stack.

Two methods: ``"polyphase"`` (``scipy.signal.resample_poly``, anti-aliased,
the default) and ``"fft"`` (numpy only: the spectrum truncated or zero-padded,
the same arithmetic as ``scipy.signal.resample``). ``"auto"`` takes polyphase
when scipy is installed and falls back to the FFT when it is not, so a
consumer that keeps scipy out of its install (``mixing.loudness`` with the
ffmpeg meter) still resamples. Both give ``ceil(n * target / fs)`` samples.

>>> import numpy as np
>>> resample(np.zeros((480, 2)), 48000, 16000).shape
(160, 2)
>>> resample(np.zeros((480, 2)), 48000, 16000, method="fft").shape
(160, 2)
"""

from __future__ import annotations

import importlib.util
from math import gcd
from typing import Literal

import numpy as np

from mixing.util import require_package

__all__ = ["resample", "fft_resample", "ResampleMethod", "RESAMPLE_METHODS"]

ResampleMethod = Literal["auto", "polyphase", "fft"]
RESAMPLE_METHODS: tuple[str, ...] = ("auto", "polyphase", "fft")
"""The ``method=`` values :func:`resample` takes."""


def _has_scipy() -> bool:
    try:
        return importlib.util.find_spec("scipy") is not None
    except (ImportError, ValueError):
        return False


def resample(
    data: np.ndarray, fs: int, target: int, *, method: ResampleMethod = "auto"
) -> np.ndarray:
    """``data`` (along axis 0) resampled from ``fs`` to ``target`` Hz; returned as
    is when the rates are equal.

    ``method``: ``"polyphase"`` (scipy, anti-aliased), ``"fft"`` (numpy only,
    :func:`fft_resample`), or ``"auto"`` (polyphase when scipy is installed,
    else the FFT).
    """
    if method not in RESAMPLE_METHODS:
        raise ValueError(f"method must be one of {RESAMPLE_METHODS}, got {method!r}")
    if fs == target:
        return data
    if method == "fft" or (method == "auto" and not _has_scipy()):
        return fft_resample(data, fs, target)
    resample_poly = require_package("scipy.signal").resample_poly
    g = gcd(fs, target)
    return resample_poly(data, target // g, fs // g, axis=0)


def fft_resample(data: np.ndarray, fs: int, target: int) -> np.ndarray:
    """``data`` (along axis 0) resampled from ``fs`` to ``target`` Hz with numpy
    alone: the real spectrum truncated (down) or zero-padded (up), as
    ``scipy.signal.resample`` does, to ``ceil(n * target / fs)`` samples.

    It treats the signal as periodic (its ends meet), which polyphase does not;
    for measuring loudness that is immaterial (within 0.2 LU on noise).

    >>> import numpy as np
    >>> t = np.arange(800) / 8000
    >>> y = fft_resample(np.sin(2 * np.pi * 500 * t), 8000, 16000)
    >>> y.shape, bool(np.allclose(y[::2], np.sin(2 * np.pi * 500 * t)))
    ((1600,), True)
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    g = gcd(fs, target)
    num = -(-n * (target // g) // (fs // g))  # ceil, as resample_poly's length
    if n == 0 or num == 0:
        return np.zeros((num,) + x.shape[1:])
    X = np.fft.rfft(x, axis=0)
    Y = np.zeros((num // 2 + 1,) + x.shape[1:], dtype=X.dtype)
    m = min(num, n)
    keep = m // 2 + 1
    Y[:keep] = X[:keep]
    if m % 2 == 0 and num != n:
        # The kept Nyquist bin of an even length stands for both its halves:
        # doubled when it becomes an interior bin (down), halved when it is
        # split into two (up), as scipy.signal.resample does.
        Y[m // 2] *= 2.0 if num < n else 0.5
    return np.fft.irfft(Y, num, axis=0) * (num / n)
