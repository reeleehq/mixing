"""The audio-visual lag between a motion signal and a song's onset envelope.

:func:`onset_lag` answers "how late does the dancer move after the music?": the
shift that best correlates a motion envelope (pose speed, frame difference, any
signal sampled at times ``t``) with an onset envelope (spectral flux at a fixed
hop, as :func:`mixing.audio.onset_envelope` or :func:`mixing.audio.beat_grid`
give it). Positive means the motion comes **after** the sound.

Four choices make it a measurement rather than an artefact of the search
(each one measured wrong in a circular, argmax version, reeleehq/mixing#129):

- **No wrapping.** The onset envelope is sampled at ``t - lag`` over its own
  full extent; a shift never brings the end of the window round to its start.
  Only the samples where both signals exist are compared: a gap in the motion
  (a dancer off screen) is left out, never interpolated across.
- **Normalised over the overlap.** Each lag is scored by the Pearson
  correlation of the two signals over the samples they share, so a lag is not
  favoured for overlapping less (or more). An anti-correlated lag scores low.
- **Near-ties go to the smallest ``|lag|``.** On periodic music the lag is
  defined only modulo a beat: every *peak* of the correlation within
  ``tie_tolerance`` of the best is an equal answer, and the peak closest to
  zero wins, at its own best lag (lags on the shoulders of one peak never
  compete, so a smooth peak is not pulled toward zero).
- **The grid is the coarser sampling.** The motion is averaged into bins one
  grid step wide, the step being the coarser of the onset hop and the motion's
  own sampling interval: 60 fps motion against a 23 ms hop is averaged down,
  10 fps motion is used as it was sampled, and neither is invented between its
  samples. Each bin sits at the mean time of its samples. The lag itself is
  searched in onset hops.

>>> import numpy as np
>>> hop = 0.01
>>> rng = np.random.default_rng(0)
>>> onset = rng.random(1000)  # 10 s of onset envelope
>>> t = np.arange(200, 400) * hop  # motion from 2 s to 4 s ...
>>> motion = np.interp(t - 0.3, np.arange(1000) * hop, onset)  # ... 0.3 s late
>>> lag, r = onset_lag(t, motion, onset, onset_hop_s=hop)
>>> round(lag, 2), round(r, 3)
(0.3, 1.0)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

__all__ = [
    "onset_lag",
    "OnsetLag",
    "DEFAULT_MAX_LAG_S",
    "DEFAULT_TIE_TOLERANCE",
    "MIN_OVERLAP_SAMPLES",
]

#: How far either way :func:`onset_lag` searches by default (seconds).
DEFAULT_MAX_LAG_S = 1.0
#: Correlation peaks this close to the best one (absolute, in r) are a tie,
#: broken toward the smallest ``|lag|``.
DEFAULT_TIE_TOLERANCE = 0.01
#: Fewest shared grid bins a lag is scored on; below it the lag is skipped.
MIN_OVERLAP_SAMPLES = 8


class OnsetLag(NamedTuple):
    """What :func:`onset_lag` found: unpacks as ``lag_s, r``.

    Attributes:
        lag_s: the shift in seconds, positive when the motion comes after the
            sound; ``nan`` when no lag could be scored.
        r: the Pearson correlation over the overlap at that lag (``nan`` likewise).
    """

    lag_s: float
    r: float


_NO_LAG = OnsetLag(math.nan, math.nan)


def _binned(t: np.ndarray, x: np.ndarray, step: float):
    """``(times, values)``: the mean time and mean value of the samples in each
    ``step``-wide bin from ``t[0]`` (sorted ``t``); empty bins (gaps) are absent."""
    index = np.floor((t - t[0]) / step + 1e-9).astype(np.int64)
    count = np.bincount(index)
    has = count > 0
    times = np.bincount(index, weights=t)[has] / count[has]
    values = np.bincount(index, weights=x)[has] / count[has]
    return times, values


def _pick_lag(lags: np.ndarray, rs: np.ndarray, tie_tolerance: float):
    """``(lag, r)``: the best peak of the correlation surface ``rs`` (``nan``
    where not scored), or, among peaks within ``tie_tolerance`` of it, the one
    with the smallest ``|lag|``; ``None`` when nothing was scored.

    A peak is a local maximum (its scored neighbours are not higher); the lags
    on its shoulders are not separate answers.
    """
    if not np.isfinite(rs).any():
        return None
    padded = np.concatenate(
        [[-np.inf], np.where(np.isfinite(rs), rs, -np.inf), [-np.inf]]
    )
    inner = padded[1:-1]
    peaks = np.flatnonzero(
        np.isfinite(rs) & (inner >= padded[:-2]) & (inner >= padded[2:])
    )
    best = rs[peaks].max()
    contenders = [k for k in peaks if rs[k] >= best - tie_tolerance]
    k = min(contenders, key=lambda k: (abs(lags[k]), lags[k]))
    return float(lags[k]), float(rs[k])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    norm = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / norm if norm > 0 else math.nan


def onset_lag(
    t,
    motion,
    onset_env,
    *,
    onset_hop_s: float,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    min_overlap: int = MIN_OVERLAP_SAMPLES,
) -> OnsetLag:
    """The lag of ``motion`` behind ``onset_env``, and the correlation there.

    Args:
        t: the times (s) ``motion`` is sampled at, on the song's clock; any
            order, any spacing (it is sorted; non-finite pairs are dropped).
        motion: the motion envelope at ``t``.
        onset_env: the onset envelope, one value every ``onset_hop_s`` from 0 s.
        onset_hop_s: the onset envelope's hop (s).
        max_lag_s: the search runs over ``[-max_lag_s, max_lag_s]``.
        tie_tolerance: correlation peaks within this of the best (absolute r)
            are equal answers; the one with the smallest ``|lag|`` wins.
        min_overlap: fewest shared grid bins a lag is scored on.

    Returns:
        :class:`OnsetLag` ``(lag_s, r)``: positive ``lag_s`` means the motion
        comes after the sound. ``(nan, nan)`` when no lag in range shares
        ``min_overlap`` samples, or either signal is constant there.

    Raises:
        ValueError: ``t`` and ``motion`` differ in length, or ``onset_hop_s``,
            ``max_lag_s`` or ``tie_tolerance`` is negative (``onset_hop_s``: not > 0).
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    motion = np.asarray(motion, dtype=np.float64).ravel()
    onset = np.asarray(onset_env, dtype=np.float64).ravel()
    if t.shape != motion.shape:
        raise ValueError(f"t and motion differ in length: {len(t)} != {len(motion)}")
    if not onset_hop_s > 0:
        raise ValueError(f"onset_hop_s must be > 0, got {onset_hop_s}")
    if not (max_lag_s >= 0 and tie_tolerance >= 0):
        raise ValueError(
            f"max_lag_s and tie_tolerance must be >= 0, got {max_lag_s}, {tie_tolerance}"
        )
    keep = np.isfinite(t) & np.isfinite(motion)
    order = np.argsort(t[keep], kind="stable")
    t, motion = t[keep][order], motion[keep][order]
    onset_t = np.arange(len(onset)) * onset_hop_s
    finite_onset = np.isfinite(onset)
    gaps = np.diff(t)
    gaps = gaps[gaps > 0]  # repeated timestamps say nothing about the rate
    if not len(gaps) or finite_onset.sum() < 2:
        return _NO_LAG
    onset_t, onset = onset_t[finite_onset], onset[finite_onset]

    # the grid: the coarser of the two samplings; motion averaged into its bins
    step = max(onset_hop_s, float(np.median(gaps)))
    times, values = _binned(t, motion, step)

    max_k = int(math.floor(max_lag_s / onset_hop_s + 1e-9))
    lags = np.arange(-max_k, max_k + 1) * onset_hop_s
    rs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        at = times - lag  # the sound that motion at ``times`` follows
        inside = (at >= onset_t[0]) & (at <= onset_t[-1])  # no wrapping
        if inside.sum() >= min_overlap:
            rs[i] = _pearson(values[inside], np.interp(at[inside], onset_t, onset))
    picked = _pick_lag(lags, rs, tie_tolerance)
    if picked is None:
        return _NO_LAG
    lag, r = picked
    return OnsetLag(lag, r)
