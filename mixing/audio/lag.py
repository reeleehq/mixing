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
  Only the samples where both signals exist are compared.
- **Normalised over the overlap.** Each lag is scored by the Pearson
  correlation of the two signals over the samples they share, so a lag is not
  favoured for overlapping less (or more).
- **Near-ties go to the smallest ``|lag|``.** On periodic music the lag is
  defined only modulo a beat: every lag within ``tie_tolerance`` of the best
  correlation is an equal answer, and the one closest to zero is returned.
- **The grid is the coarser sampling.** The signals are compared on a grid
  whose step is the coarser of the onset hop and the motion's own sampling
  interval, the finer one averaged down to it, so a 10 fps motion signal is
  not reduced to scattered points on a 10 ms grid. The lag itself is searched
  in onset hops.

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
#: Correlations this close to the best one (absolute, in r) are a tie, broken
#: toward the smallest ``|lag|``.
DEFAULT_TIE_TOLERANCE = 0.01
#: Fewest shared grid samples a lag is scored on; below it the lag is skipped.
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


def _box_average(x: np.ndarray, width: int) -> np.ndarray:
    """``x`` averaged over a centred window of ``width`` samples (edges use what
    exists), so sampling it every ``width`` samples does not alias."""
    if width <= 1:
        return x
    kernel = np.ones(width)
    total = np.convolve(x, kernel, mode="same")
    count = np.convolve(np.ones_like(x), kernel, mode="same")
    return total / count


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
        tie_tolerance: correlations within this of the best (absolute r) are
            equal answers; the smallest ``|lag|`` among them wins.
        min_overlap: fewest shared grid samples a lag is scored on.

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
    if len(t) < 2 or finite_onset.sum() < 2:
        return _NO_LAG
    onset_t, onset = onset_t[finite_onset], onset[finite_onset]

    # the grid: the coarser of the two samplings, the finer averaged down to it
    motion_step = float(np.median(np.diff(t)))
    step = max(onset_hop_s, motion_step)
    grid = np.arange(t[0], t[-1] + step / 2, step)
    motion_on_grid = np.interp(grid, t, motion)
    onset_smooth = _box_average(onset, int(round(step / onset_hop_s)))

    best = None  # (r, |lag|, lag)
    scored = []
    max_k = int(math.floor(max_lag_s / onset_hop_s + 1e-9))
    for k in range(-max_k, max_k + 1):
        lag = k * onset_hop_s
        at = grid - lag  # the sound that motion at ``grid`` follows
        inside = (at >= onset_t[0]) & (at <= onset_t[-1])  # no wrapping
        if inside.sum() < min_overlap:
            continue
        r = _pearson(
            motion_on_grid[inside], np.interp(at[inside], onset_t, onset_smooth)
        )
        if math.isnan(r):
            continue
        scored.append((r, lag))
        if best is None or r > best[0]:
            best = (r, lag)
    if best is None:
        return _NO_LAG
    ties = [(abs(lag), lag, r) for r, lag in scored if r >= best[0] - tie_tolerance]
    _, lag, r = min(ties)
    return OnsetLag(float(lag), float(r))
