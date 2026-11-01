"""Shot boundaries from frame changes: the array stages, light to import.

What :func:`mixing.video.shot_boundaries.diff_shots` does once its frames are
decoded, for callers that decode frames themselves (a scoring pass that
already holds them): numpy only, so importing it pulls none of the video stack
(``mixing.video`` loads moviepy, looks and burns).

- :func:`frame_changes`: mean absolute difference of consecutive frames
  (:func:`streamed_changes` over frames arriving in chunks).
- :func:`adaptive_threshold`: median change plus robust deviations, clamped.
- :func:`cut_indices`: which changes are cuts, by a ``rule``:

  - ``"level"`` (default; :func:`diff_shots`' since it shipped): every change
    above the threshold. Right for edited video, where cuts can come on
    consecutive samples (a page scrolled every 0.5 s);
  - ``"spike"``: a change above the threshold **and** :data:`SPIKE_RATIO` times
    both neighbouring changes, so sustained motion (handheld footage, a pan, a
    fade) is not a run of cuts, and neither is a flash (a frame that differs
    and comes back changes twice in a row). Measured on 14 single-take phone
    clips (reeleehq/mixing#160): false cuts 481 -> 10 at 2 Hz, 473 -> 11 at
    5 Hz (PySceneDetect: 16). The price is recall: of 62 real cuts between two
    takes of one event it finds 45 % at 2 Hz, 71 % at 5 Hz (at a ratio of 1.5:
    74 % / 90 %, with twice the false cuts), and it misses cuts on consecutive
    samples. For continuous footage, not for edited video.

- :func:`spans_from_changes` / :func:`shots_from_changes`: cuts to spans that
  tile the duration, at a fixed rate or at given frame times.

The thresholds are calibrated on frames shrunk by ffmpeg's ``scale`` to
:data:`SHOT_FRAME_SIZE` (what :func:`mixing.video.shot_boundaries.iter_frames`
gives); frames shrunk another way (cv2's ``INTER_AREA``) have somewhat
different change levels.

>>> import numpy as np
>>> handheld = np.array([20.0, 24.0, 31.0, 28.0, 33.0, 26.0, 90.0, 4.0, 3.0])
>>> shots_from_changes(handheld, fps=2, threshold=30.0)  # level: motion cuts
[(0.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0)]
>>> shots_from_changes(handheld, fps=2, threshold=30.0, rule="spike")
[(0.0, 3.5), (3.5, 5.0)]
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "Span",
    "SHOT_FRAME_SIZE",
    "DEFAULT_SHOT_FPS",
    "ADAPTIVE_THRESHOLD_SIGMAS",
    "MIN_SHOT_THRESHOLD",
    "MAX_SHOT_THRESHOLD",
    "CUT_RULES",
    "SPIKE_RATIO",
    "frame_changes",
    "streamed_changes",
    "adaptive_threshold",
    "cut_indices",
    "spans_from_changes",
    "shots_from_changes",
    "merge_short_spans",
]

Span = tuple[float, float]

DEFAULT_SHOT_FPS = 2.0
#: Frames are shrunk to this (width, height) before differencing: cheap and
#: insensitive to compression noise.
SHOT_FRAME_SIZE = (64, 36)
#: Adaptive threshold = median change + this many robust standard deviations.
ADAPTIVE_THRESHOLD_SIGMAS = 6.0
#: Never go below this mean abs difference (0-255 scale); noise-only videos
#: would otherwise have a threshold near zero.
MIN_SHOT_THRESHOLD = 8.0
#: ...and never above this: when cuts are frequent (a page every 0.5 s) the
#: median change *is* a cut, and an uncapped threshold would sit above all of
#: them. On handheld footage the median change is 14 to 33 at the default
#: size, so the threshold pins here and motion reads as cuts under the
#: ``"level"`` rule: use ``rule="spike"`` there (reeleehq/mixing#160).
MAX_SHOT_THRESHOLD = 30.0
#: Scale turning a median absolute deviation into a standard deviation.
_MAD_TO_SIGMA = 1.4826
#: The ways :func:`cut_indices` decides which changes are cuts.
CUT_RULES = ("level", "spike")
#: Under ``rule="spike"`` a cut's change is at least this many times both
#: neighbouring changes: a hard cut is one large change between two small ones;
#: motion and fades are runs of similar ones.
SPIKE_RATIO = 2.0


def frame_changes(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames (length ``n - 1``)."""
    frames = np.asarray(frames).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def streamed_changes(chunks: Iterable[np.ndarray]) -> tuple[np.ndarray, int]:
    """``(changes, frame count)`` over frames arriving in chunks (each
    ``(k, h, w)``, or one ``(h, w)`` frame), holding one chunk plus the frame
    before it at a time: equal to :func:`frame_changes` of all the frames."""
    changes, count, last = [], 0, None
    for chunk in chunks:
        chunk = np.asarray(chunk)
        if chunk.ndim == 2:  # a single (h, w) frame
            chunk = chunk[None]
        if not len(chunk):
            continue
        joined = chunk if last is None else np.concatenate([last[None], chunk])
        changes.append(frame_changes(joined))
        count, last = count + len(chunk), chunk[-1]
    return (np.concatenate(changes) if changes else np.zeros(0, np.float32)), count


def adaptive_threshold(
    changes: np.ndarray,
    *,
    sigmas: float = ADAPTIVE_THRESHOLD_SIGMAS,
    floor: float = MIN_SHOT_THRESHOLD,
    ceiling: float = MAX_SHOT_THRESHOLD,
) -> float:
    """Median change plus ``sigmas`` robust deviations, clamped to ``[floor, ceiling]``."""
    if len(changes) == 0:
        return floor
    median = float(np.median(changes))
    mad = float(np.median(np.abs(changes - median)))
    return min(ceiling, max(floor, median + sigmas * _MAD_TO_SIGMA * mad))


def cut_indices(
    changes: np.ndarray,
    *,
    threshold: float,
    rule: str = "level",
    spike_ratio: float = SPIKE_RATIO,
) -> np.ndarray:
    """Indices ``i`` of the changes that are cuts (between frames ``i`` and ``i + 1``).

    ``rule`` (:data:`CUT_RULES`): ``"level"`` takes every change above
    ``threshold``; ``"spike"`` also needs it ``spike_ratio`` times both
    neighbouring changes (a missing neighbour, at an end, does not count). That
    rejects flashes too: the frames on either side of a spike differ by at least
    half its change (the triangle inequality on mean absolute differences), so
    no separate "does the picture come back" test can add anything.
    """
    if rule not in CUT_RULES:
        raise ValueError(f"unknown rule {rule!r}; one of {CUT_RULES}")
    changes = np.asarray(changes, dtype=np.float64)
    above = changes > threshold
    if rule == "level" or not len(changes):
        return np.flatnonzero(above)
    before = np.concatenate([[0.0], changes[:-1]])
    after = np.concatenate([changes[1:], [0.0]])
    spike = above & (changes >= spike_ratio * before) & (changes >= spike_ratio * after)
    return np.flatnonzero(spike)


def spans_from_changes(
    changes: np.ndarray,
    *,
    fps: float,
    threshold: float,
    duration: float | None = None,
) -> list[Span]:
    """Turn per-frame changes into ``[(start, end), ...]`` shot spans.

    ``changes[i]`` is the change between frames ``i`` and ``i + 1``; a value
    above ``threshold`` puts a boundary at time ``(i + 1) / fps``. The spans
    tile ``[0, duration]`` with no gaps; ``duration`` defaults to the sampled
    length ``(len(changes) + 1) / fps``. Cuts at or past ``duration`` are dropped.
    """
    end = (len(changes) + 1) / fps if duration is None else float(duration)
    cuts = [(i + 1) / fps for i in np.flatnonzero(changes > threshold)]
    return _tile(cuts, end)


def _tile(cuts: Sequence[float], end: float) -> list[Span]:
    edges = [0.0, *(c for c in cuts if c < end), end]
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def shots_from_changes(
    changes: np.ndarray,
    *,
    fps: Optional[float] = None,
    times: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
    rule: str = "level",
    duration: Optional[float] = None,
    min_shot_s: Optional[float] = None,
) -> list[Span]:
    """Shot spans from frame changes a caller computed (the detector without a decode).

    Args:
        changes: :func:`frame_changes` of the frames (``changes[i]`` between
            frames ``i`` and ``i + 1``).
        fps: the frames' fixed rate, or
        times: each frame's time (s), increasing, ``len(changes) + 1`` of them;
            a cut between frames ``i`` and ``i + 1`` is at ``times[i + 1]``.
        threshold: above which a change can be a cut; ``None`` picks one
            (:func:`adaptive_threshold`).
        rule: as :func:`cut_indices` (``"spike"`` for continuous footage).
        duration: the length (s) the spans tile; default the last frame's time
            plus one frame (``len(changes) + 1`` frames at ``fps``; the last
            interval of ``times`` repeated).
        min_shot_s: merge spans shorter than this (:func:`merge_short_spans`).

    Returns:
        Spans tiling ``[0, duration]``; ``[]`` for no frames and no duration.
    """
    if (fps is None) == (times is None):
        raise ValueError("give the frames' rate (fps=) or their times (times=), one")
    if duration is not None and (
        isinstance(duration, bool) or not math.isfinite(duration) or duration < 0
    ):
        raise ValueError(f"duration must be a number of seconds >= 0, got {duration!r}")
    changes = np.asarray(changes, dtype=np.float64)
    if fps is not None:
        if not fps > 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        frame_times = np.arange(len(changes) + 1) / fps
        end = (len(changes) + 1) / fps
    else:
        frame_times = np.asarray(times, dtype=np.float64)
        if not len(frame_times) and not len(changes):
            return [] if not duration else [(0.0, float(duration))]
        if len(frame_times) != len(changes) + 1:
            raise ValueError(
                f"{len(frame_times)} times for {len(changes)} changes: one time per "
                "frame, one more than the changes"
            )
        if np.any(np.diff(frame_times) <= 0):
            raise ValueError("times must increase, one per frame in order")
        step = frame_times[-1] - frame_times[-2] if len(frame_times) > 1 else 0.0
        end = float(frame_times[-1] + step)  # the last frame lasts one interval
    end = end if duration is None else float(duration)
    if not end:
        return []
    if threshold is None:
        threshold = adaptive_threshold(changes)
    cuts = frame_times[cut_indices(changes, threshold=threshold, rule=rule) + 1]
    spans = _tile(cuts.tolist(), end)
    return merge_short_spans(spans, min_s=min_shot_s) if min_shot_s else spans


def merge_short_spans(spans: list[Span], *, min_s: float) -> list[Span]:
    """``spans`` with every span shorter than ``min_s`` merged into the one
    before it (the first into the one after), so the result still tiles them.

    >>> merge_short_spans([(0.0, 2.0), (2.0, 2.5), (2.5, 6.0)], min_s=1.0)
    [(0.0, 2.5), (2.5, 6.0)]
    """
    merged: list[Span] = []
    for start, end in spans:
        if merged and end - start < min_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_s:
        merged[:2] = [(merged[0][0], merged[1][1])]
    return merged
