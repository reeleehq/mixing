"""Shot-boundary detection for finished videos.

``shots(video)`` returns the shot list ``[(start, end), ...]`` in seconds: where
each cut, card or scrolled page starts and ends. That is what re-editing a
finished video needs, to judge image dwell time and to choose cutaways.

``strategy`` picks the detector (``SHOT_STRATEGIES``, or any
``(video, **kwargs) -> spans`` callable):

- ``"diff"`` (default, ffmpeg + numpy): sample the video at a low frame rate
  (default 2 fps) as tiny grayscale frames, streamed from ffmpeg in chunks,
  take the mean absolute difference between consecutive frames, and call a
  boundary wherever that change exceeds a threshold. The default threshold is
  adaptive (robust to a video's own noise level). Boundaries are accurate to
  about ``1 / fps`` seconds (a cut is reported at the nearest sample); raise
  ``fps`` for finer cuts.
- ``"pyscenedetect"``: PySceneDetect's ``ContentDetector`` (``pip install
  scenedetect``), frame-accurate, slower.

``min_shot_s`` merges spans shorter than that into their neighbour (a dissolve
otherwise shows up as a short fake shot). The pure stages
(:func:`frame_changes`, :func:`adaptive_threshold`, :func:`spans_from_changes`,
:func:`merge_short_spans`) take arrays, so they can be reused on frames from any
source, and ``sampler`` lets a caller swap the diff strategy's frame source.

>>> import numpy as np
>>> changes = np.array([1.0, 1.0, 90.0, 1.0, 1.0])  # a cut at frame 3
>>> spans_from_changes(changes, fps=2, threshold=10.0)
[(0.0, 1.5), (1.5, 3.0)]
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Iterator, Union

import numpy as np

from mixing.egress import Output, deliver
from mixing.util import ffmpeg_exe, require_package

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

#: Frames read from ffmpeg at a time: memory stays flat however long the video.
SAMPLE_CHUNK_FRAMES = 512

CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_THUMB_WIDTH = 240
#: Pixels around a contact-sheet time label.
CONTACT_SHEET_LABEL_PAD = 3

#: ``sampler(video, fps=...)``: grayscale frames as one ``(n, h, w)`` array, or
#: an iterable of frames ``(h, w)`` or chunks ``(k, h, w)``.
Sampler = Callable[..., Union[np.ndarray, Iterable[np.ndarray]]]


def iter_frames(
    video,
    *,
    fps: float = DEFAULT_SHOT_FPS,
    size: tuple[int, int] = SHOT_FRAME_SIZE,
    chunk_frames: int = SAMPLE_CHUNK_FRAMES,
) -> Iterator[np.ndarray]:
    """Decode ``video`` at ``fps`` as tiny grayscale frames, yielded in chunks
    ``(k, h, w)`` of at most ``chunk_frames``."""
    width, height = size
    frame_bytes = width * height
    cmd = [
        ffmpeg_exe(), "-v", "error", "-i", str(video),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]  # fmt: skip
    # stderr goes to a file: a pipe nobody reads fills up (~64 KB of decode
    # errors on a damaged file) and blocks ffmpeg, and stdout with it
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors) as proc:
            while chunk := proc.stdout.read(chunk_frames * frame_bytes):
                usable = len(chunk) - len(chunk) % frame_bytes
                yield np.frombuffer(chunk[:usable], np.uint8).reshape(-1, height, width)
        errors.seek(0)
        stderr = errors.read()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not read {video!s}: {stderr.decode(errors='replace')}"
        )


def sample_frames(
    video, *, fps: float = DEFAULT_SHOT_FPS, size: tuple[int, int] = SHOT_FRAME_SIZE
) -> np.ndarray:
    """Decode ``video`` at ``fps`` as tiny grayscale frames, shape ``(n, h, w)``."""
    chunks = list(iter_frames(video, fps=fps, size=size))
    width, height = size
    return np.concatenate(chunks) if chunks else np.zeros((0, height, width), np.uint8)


def frame_changes(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames (length ``n - 1``)."""
    frames = frames.astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def _streamed_changes(chunks: Iterable[np.ndarray]) -> tuple[np.ndarray, int]:
    """``(frame_changes, frame count)`` over frames arriving in chunks, holding
    one chunk (plus the previous frame) at a time."""
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


DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(video) -> float | None:
    """Container duration in seconds via ``ffmpeg -i``, or ``None`` if unknown."""
    proc = subprocess.run([ffmpeg_exe(), "-i", str(video)], capture_output=True)
    match = DURATION_RE.search(proc.stderr.decode(errors="replace"))
    if match is None:
        return None
    h, m, sec = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(sec)


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


def diff_shots(
    video,
    *,
    fps: float = DEFAULT_SHOT_FPS,
    threshold: float | None = None,
    sampler: Sampler = iter_frames,
) -> list[Span]:
    """The ``"diff"`` strategy: frame differences of a low-rate grayscale stream.

    Args:
        video: path to a video file (anything ``sampler`` understands).
        fps: sampling rate; boundaries are resolved to ``1 / fps`` seconds.
        threshold: mean-abs-difference (0-255) above which consecutive frames
            are a cut. ``None`` picks one adaptively (:func:`adaptive_threshold`).
        sampler: ``sampler(video, fps=fps)`` -> grayscale frames: one ``(n, h,
            w)`` array, or an iterable of frames or chunks (the default
            streams chunks).

    Returns:
        Spans tiling the video's duration (the sampled length for a custom
        ``sampler``); one span if there is no cut; ``[]`` if nothing decoded.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    frames = sampler(video, fps=fps)
    chunks = [frames] if isinstance(frames, np.ndarray) else frames
    changes, count = _streamed_changes(chunks)
    own_sampler = sampler in (iter_frames, sample_frames)
    duration = probe_duration(video) if own_sampler else None
    if count == 0:
        return [] if not duration else [(0.0, float(duration))]
    if threshold is None:
        threshold = adaptive_threshold(changes)
    return spans_from_changes(changes, fps=fps, threshold=threshold, duration=duration)


def pyscenedetect_shots(
    video, *, threshold: float | None = None, **detector_kwargs
) -> list[Span]:
    """The ``"pyscenedetect"`` strategy: PySceneDetect's ``ContentDetector``.

    Frame-accurate (it decodes every frame), so slower than ``"diff"``.
    ``threshold`` (its own scale; ``None`` keeps PySceneDetect's default) and
    ``detector_kwargs`` go to ``ContentDetector``. Needs ``pip install
    scenedetect``.
    """
    scenedetect = require_package("scenedetect")
    if threshold is not None:
        detector_kwargs["threshold"] = threshold
    scenes = scenedetect.detect(
        str(video), scenedetect.ContentDetector(**detector_kwargs)
    )
    if not scenes:  # no cut: one shot, if the video has a length
        duration = probe_duration(video)
        return [(0.0, float(duration))] if duration else []
    return [(float(a.get_seconds()), float(b.get_seconds())) for a, b in scenes]


#: Named shot detectors; ``shots(strategy=...)`` also takes any callable
#: ``(video, **kwargs) -> [(start, end), ...]``.
SHOT_STRATEGIES: dict[str, Callable[..., list[Span]]] = {
    "diff": diff_shots,
    "pyscenedetect": pyscenedetect_shots,
}
ShotStrategy = Union[str, Callable[..., list[Span]]]


def shots(
    video,
    *,
    strategy: ShotStrategy = "diff",
    min_shot_s: float = 0.0,
    **strategy_kwargs,
) -> list[Span]:
    """Detect the shots of a finished video as ``[(start, end), ...]`` seconds.

    Args:
        video: path to a video file.
        strategy: a ``SHOT_STRATEGIES`` name (``"diff"``: :func:`diff_shots`,
            ``"pyscenedetect"``: :func:`pyscenedetect_shots`) or a callable
            ``(video, **strategy_kwargs) -> spans``.
        min_shot_s: merge spans shorter than this into a neighbour
            (:func:`merge_short_spans`); a dissolve otherwise reads as a short
            fake shot. ``0`` keeps every span.
        **strategy_kwargs: the strategy's own options, e.g. ``fps``,
            ``threshold`` and ``sampler`` for ``"diff"``.

    Returns:
        Spans in seconds that tile the video; ``[]`` if nothing could be decoded.
    """
    if callable(strategy):
        detect = strategy
    elif strategy in SHOT_STRATEGIES:
        detect = SHOT_STRATEGIES[strategy]
    else:
        raise ValueError(
            f"unknown shot strategy {strategy!r}; one of {sorted(SHOT_STRATEGIES)} "
            "or a callable"
        )
    spans = detect(video, **strategy_kwargs)
    return merge_short_spans(spans, min_s=min_shot_s) if min_shot_s > 0 else spans


def _clock(seconds: float) -> str:
    """``m:ss.s``, e.g. ``1:05.5`` (``59.96`` is ``1:00.0``, not ``0:60.0``)."""
    minutes, rest = divmod(round(seconds, 1), 60)
    return f"{int(minutes)}:{rest:04.1f}"


def _label(image, text: str) -> None:
    """Draw ``text`` bottom-left on ``image``, light on a dark box."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text)
    pad = CONTACT_SHEET_LABEL_PAD
    top = image.height - (y1 - y0) - 2 * pad
    draw.rectangle((0, top, x1 - x0 + 2 * pad, image.height), fill=(0, 0, 0))
    draw.text((pad - x0, top + pad - y0), text, fill=(255, 255, 255))


def contact_sheet(
    video,
    spans: list[Span],
    *,
    output: Output = None,
    columns: int = CONTACT_SHEET_COLUMNS,
    thumb_width: int = CONTACT_SHEET_THUMB_WIDTH,
    labels: bool = True,
):
    """One mid-shot thumbnail per span, tiled into a PIL image for review.

    ``labels`` writes each span's ``start-end`` time on its thumbnail.
    ``output=None`` returns the image; a path (or directory) saves a PNG;
    a callable receives the image.
    """
    from PIL import Image

    from mixing.video.thumbnail import grab_frame

    if not spans:
        raise ValueError("contact_sheet needs at least one span")
    thumbs = [grab_frame(video, (a + b) / 2, width=thumb_width) for a, b in spans]
    if labels:
        for thumb, (a, b) in zip(thumbs, spans):
            _label(thumb, f"{_clock(a)}-{_clock(b)}")
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
