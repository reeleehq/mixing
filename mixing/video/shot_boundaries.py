"""Shot-boundary detection for finished videos.

``shots(video)`` returns the shot list ``[(start, end), ...]`` in seconds: where
each cut, card or scrolled page starts and ends. That is what re-editing a
finished video needs, to judge image dwell time and to choose cutaways.

Method: sample the video at a low frame rate (default 2 fps) as tiny grayscale
frames through ffmpeg, take the mean absolute difference between consecutive
frames, and call a boundary wherever that change exceeds a threshold. The
default threshold is adaptive (robust to a video's own noise level).
Boundaries are therefore accurate to about ``1 / fps`` seconds (a cut is
reported at the nearest sample); raise ``fps`` for
finer cuts.

The pure stages (:func:`frame_changes`, :func:`adaptive_threshold`,
:func:`spans_from_changes`) take arrays, so they can be reused on frames from
any source, and ``sampler`` lets a caller swap the frame source.

>>> import numpy as np
>>> changes = np.array([1.0, 1.0, 90.0, 1.0, 1.0])  # a cut at frame 3
>>> spans_from_changes(changes, fps=2, threshold=10.0)
[(0.0, 1.5), (1.5, 3.0)]
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from mixing.egress import Output, deliver
from mixing.util import ffmpeg_exe

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
#: median change *is* a cut, and an uncapped threshold would sit above all of them.
MAX_SHOT_THRESHOLD = 30.0
#: Scale turning a median absolute deviation into a standard deviation.
_MAD_TO_SIGMA = 1.4826

CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_THUMB_WIDTH = 240

Sampler = Callable[..., np.ndarray]


def sample_frames(
    video, *, fps: float = DEFAULT_SHOT_FPS, size: tuple[int, int] = SHOT_FRAME_SIZE
) -> np.ndarray:
    """Decode ``video`` at ``fps`` as tiny grayscale frames, shape ``(n, h, w)``."""
    width, height = size
    cmd = [
        ffmpeg_exe(), "-v", "error", "-i", str(video),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not read {video!s}: {proc.stderr.decode(errors='replace')}"
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width)


def frame_changes(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames (length ``n - 1``)."""
    frames = frames.astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


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


def spans_from_changes(
    changes: np.ndarray, *, fps: float, threshold: float, duration: float | None = None
) -> list[Span]:
    """Turn per-frame changes into ``[(start, end), ...]`` shot spans.

    ``changes[i]`` is the change between frames ``i`` and ``i + 1``; a value
    above ``threshold`` puts a boundary at time ``(i + 1) / fps``. The spans
    tile ``[0, duration]`` with no gaps; ``duration`` defaults to the sampled
    length ``(len(changes) + 1) / fps``. Cuts at or past ``duration`` are dropped.
    """
    end = (len(changes) + 1) / fps if duration is None else float(duration)
    cuts = [(i + 1) / fps for i in np.flatnonzero(changes > threshold)]
    edges = [0.0, *(c for c in cuts if c < end), end]
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(video) -> float | None:
    """Container duration in seconds via ``ffmpeg -i``, or ``None`` if unknown."""
    proc = subprocess.run([ffmpeg_exe(), "-i", str(video)], capture_output=True)
    match = _DURATION_RE.search(proc.stderr.decode(errors="replace"))
    if match is None:
        return None
    h, m, sec = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(sec)


def shots(
    video,
    *,
    fps: float = DEFAULT_SHOT_FPS,
    threshold: float | None = None,
    sampler: Sampler = sample_frames,
) -> list[Span]:
    """Detect the shots of a finished video as ``[(start, end), ...]`` seconds.

    Args:
        video: path to a video file (anything ``sampler`` understands).
        fps: sampling rate; boundaries are resolved to ``1 / fps`` seconds.
        threshold: mean-abs-difference (0-255) above which consecutive frames
            are a cut. ``None`` picks one adaptively (:func:`adaptive_threshold`).
        sampler: ``sampler(video, fps=fps) -> (n, h, w)`` grayscale frames; the
            seam for another frame source or a PySceneDetect-style strategy.

    Returns:
        Spans in seconds that tile the video's duration (the sampled length for
        a custom ``sampler``); a video with no cut is one span. A short dissolve
        shows up as a short fake shot. An empty list if nothing could be decoded.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    frames = sampler(video, fps=fps)
    duration = probe_duration(video) if sampler is sample_frames else None
    if len(frames) == 0:
        return [] if not duration else [(0.0, float(duration))]
    changes = frame_changes(frames)
    if threshold is None:
        threshold = adaptive_threshold(changes)
    return spans_from_changes(changes, fps=fps, threshold=threshold, duration=duration)


def _frame_at(video, t: float, *, width: int):
    from PIL import Image

    cmd = [
        ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", f"scale={width}:-2",
        "-f", "image2pipe", "-vcodec", "png", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not grab a frame of {video!s} at {t:.2f}s")
    import io

    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def contact_sheet(
    video,
    spans: list[Span],
    *,
    output: Output = None,
    columns: int = CONTACT_SHEET_COLUMNS,
    thumb_width: int = CONTACT_SHEET_THUMB_WIDTH,
):
    """One mid-shot thumbnail per span, tiled into a PIL image for review.

    ``output=None`` returns the image; a path (or directory) saves a PNG;
    a callable receives the image.
    """
    from PIL import Image

    if not spans:
        raise ValueError("contact_sheet needs at least one span")
    thumbs = [_frame_at(video, (a + b) / 2, width=thumb_width) for a, b in spans]
    cell_w, cell_h = thumbs[0].size
    columns = min(columns, len(thumbs))
    rows = -(-len(thumbs) // columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * cell_w, (i // columns) * cell_h))

    def write(image, path: Path):
        image.save(path)

    return deliver(
        sheet,
        output,
        write=write,
        default_name=f"{Path(str(video)).stem}_contact_sheet.png",
    )
