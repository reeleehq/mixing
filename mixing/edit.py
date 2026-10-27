"""Render a re-edit of a finished video from a sound edit list and its picture clips.

Re-editing a finished talking-head video with an editor's eye means two edit
lists of the same length: the **sound** (which source spans play, and where a
silence is inserted) and the **picture** over it (the source, punched in or
not, or a held still with a slow move, dissolving or cutting between clips).
:func:`render` takes both and renders the film, frame-exact:

>>> from mixing.edit import Span, Gap, Cam, Still, render  # doctest: +SKIP
>>> render(  # doctest: +SKIP
...     [Span(0.0, 4.2, "a"), Span(5.0, 9.0, "b"), Gap(1.5, "card"), Span(9.0, 12.0, "a")],
...     {"a": Cam(), "b": Cam(zoom=1.3, cx=0.4), "card": Still("card.png", dissolve_in=0.5)},
...     source="talk.mp4",
... )

**Sound.** Each segment is a :class:`Span` of source seconds or a :class:`Gap`
of silence. Spans snap to source frames. Consecutive spans that are contiguous
in the source play as one unbroken run; every other join gets a short fade
(:data:`JOIN_FADE_S`) so it does not click.

**Picture.** Each segment names a clip id. Consecutive segments with the same
id form **one** picture clip, so a :class:`Still` keeps its move across a sound
cut (that is how a pause is removed invisibly) and a :class:`Cam` punch-in
hides a jump cut. Under a :class:`Cam` clip:

- a span shows its own source frames;
- a :class:`Gap` **holds the frame next to it**: the last frame of the
  previous span of the same clip, or, when the gap opens the clip, the first
  frame of the next span. (A Cam clip made only of gaps has nothing to show
  and is refused.)

**Frame-exact by construction.** The timeline runs at the source rate as a
:class:`~fractions.Fraction` (``30000/1001`` stays exact; ``fps=`` overrides).
Output frame boundaries are cumulative integer frame counts and each sound
boundary is ``round(frame * sample_rate / fps)`` of the **same** frame boundary
(one rounding per boundary, never accumulated), so sound and picture never
drift and end on the same instant. :func:`timeline` is that arithmetic, pure
and testable without ffmpeg; :func:`render` executes it.

**Dissolves.** ``dissolve_in=D`` on a clip renders it ``D/2`` early and the
outgoing clip ``D/2`` late (a Cam clip shows the source frames beyond its
span, a Still runs its move longer), joined with ffmpeg ``xfade``, so the total
length is unchanged.

**Picture and sound fades.** ``closing_fade=`` fades the last frames to black
and the sound to silence over the same frames (the last frame is black and
silent).

**Overlays.** Each :class:`Overlay` is an image with alpha shown over a time
range (optionally fading), composited in the final pass, which also applies an
optional closing fade and muxes the sound. An overlay starting at or after the
film's end is refused; one running past it is clipped to it, its fades
shortened to fit. A ``tituli`` label is one way to
make the image (``tituli.render(lower_third(...), frame)`` returns a PIL image,
which an :class:`Overlay` accepts); nothing here depends on ``tituli``.

**Stills** go through a strategy slot, ``still_renderer``, defaulting to
:func:`burns_still_renderer`: the move is resolved by ``burns.resolve_move``
(named moves such as ``"push_in"``, or an explicit path), a still of another
aspect is pillar- or letterboxed rather than silently cropped, and windows
stay inside the still's content box. ``burns`` comes from the ``media`` extra
and is imported only when a still is rendered.

**Bounded memory.** The picture is rendered as parts: one small ffmpeg run per
source piece (seek, resample, trim, punch-in) or still slice, one two-input
run per dissolve, each written as a lossless intermediate in a temporary
directory, then read back to back by the concat demuxer in the final pass. No
run opens more than two decoders, however many segments the edit has. The
sound is assembled in memory from one PCM decode of the source (32-bit float:
about 115 MB for five minutes of 48 kHz stereo).

**What is assumed of the source.**

- It has a video and a sound stream (the first of each is used) and an even
  width and height (the delivery is ``yuv420p``); others are refused.
- Source seconds count from the first video frame. A sound stream that starts
  later or earlier (its ``start_time``) is shifted back into sync.
- Frame-exact identity assumes a constant frame rate. A variable-rate source
  (``r_frame_rate`` and ``avg_frame_rate`` disagree) is resampled by time to
  its nominal rate, each frame the nearest source frame, with a warning;
  ``fps=`` resamples the same way.
- A display rotation is applied by ffmpeg on decode; the size used is the
  displayed one.
- The source's length is its counted video packets (``ffprobe``) or, without
  ``ffprobe`` or for a variable-rate source, its duration; a span past it is
  refused. Without ``ffprobe`` the rate is read from ffmpeg's banner (snapped
  by :func:`as_rate`) and both streams are taken to start together.
- The PCM sound is handled as little-endian float (every platform ffmpeg
  ships for in practice).

Everything runs through ``ffmpeg`` subprocesses (:func:`mixing.util.ffmpeg_exe`);
importing this module pulls no media library.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from array import array
from dataclasses import dataclass
from fractions import Fraction
from itertools import count, groupby
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Union

from mixing.egress import Output, PathLike, write_egress
from mixing.util import ffmpeg_exe, require_media_extra

__all__ = [
    "Span",
    "Gap",
    "Cam",
    "Still",
    "Overlay",
    "SourceInfo",
    "SoundPiece",
    "PicturePiece",
    "PictureClip",
    "Timeline",
    "as_rate",
    "punch_in_box",
    "timeline",
    "probe_source",
    "burns_still_renderer",
    "slide_into",
    "render",
    "JOIN_FADE_S",
    "DEFAULT_ENCODE_ARGS",
]

# --- constants -----------------------------------------------------------------

#: Fade (s) at every sound join that is not a continuous source run: long
#: enough to remove the click of a waveform cut, short enough to be inaudible.
JOIN_FADE_S = 0.006
#: The ``xfade`` transition a ``dissolve_in`` renders (a plain cross-dissolve).
DISSOLVE_TRANSITION = "fade"
#: Seek this fraction of a frame before a wanted source frame, so ffmpeg's
#: accurate input seek (which drops frames earlier than the target) lands on it.
SEEK_LEAD_FRAMES = Fraction(1, 2)
#: Decimals of a seek or filter time in seconds (a microsecond: far below a frame).
TIME_DECIMALS = 6
#: Resampler of the punch-in's scale back to the source size.
SCALE_FLAGS = "lanczos"
#: Encode of the per-part intermediates: lossless x264 (``-qp 0``), so the
#: final encode is the only generation that costs quality. Lossless is large
#: (tens of Mbit/s at 1080p); the parts live in a temporary directory.
INTERMEDIATE_ENCODE_ARGS = (
    "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p",
)  # fmt: skip
#: Container of the intermediates (joined by the concat demuxer).
INTERMEDIATE_SUFFIX = ".mp4"
#: Encode of the delivered film.
DEFAULT_ENCODE_ARGS = (
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
)  # fmt: skip
#: Video bitrate ``burns`` renders still moves at (an intermediate: generous).
STILL_BITRATE = "20000k"
#: The ``burns`` backend used for stills (one ffmpeg process per still).
STILL_BACKEND = "ffmpeg"
#: A still's default move: a static hold of the whole (aspect-fitted) picture.
DEFAULT_STILL_MOVE = "hold"
#: The zoom of a ``"hold"`` when ``Still.zoom`` is not given: the untouched frame.
HOLD_ZOOM = 1.0
#: How a still whose aspect differs from the frame's is fitted: ``"pad"`` letter-
#: or pillarboxes it (the whole picture is shown; the padding becomes the
#: content box), ``"fill"`` lets ``burns`` cover-crop it.
STILL_FITS = ("pad", "fill")
DEFAULT_STILL_FIT = "pad"
#: Colour of the padding a ``"pad"`` fit adds.
PAD_COLOR = (0, 0, 0)
#: Relative difference under which two aspect ratios are the same shape.
ASPECT_TOLERANCE = 1e-3
#: Relative disagreement between a stream's ``r_frame_rate`` and its
#: ``avg_frame_rate`` above which the source is taken to be variable-rate.
VFR_TOLERANCE = 0.01
#: Default closing fade (s) of picture and sound; 0 means none.
DEFAULT_CLOSING_FADE_S = 0.0
#: Suffix of the default output name, beside the source.
DEFAULT_OUTPUT_SUFFIX = "_edit.mp4"
#: Raw PCM the sound is assembled in: 32-bit float, native byte order.
PCM_FORMAT = "f32le"
PCM_TYPECODE = "f"
PCM_BYTES = 4
#: NTSC rates are ``k * 1000 / 1001``; a float rate this close to one (or to
#: an integer) is taken to mean it exactly (ffmpeg prints ``29.97``).
NTSC_DENOMINATOR = 1001
RATE_SNAP_TOLERANCE = 0.01
#: Largest denominator a float rate that is neither is approximated with.
MAX_RATE_DENOMINATOR = 1001
#: Channel counts of the layouts ffmpeg's banner names.
CHANNEL_LAYOUTS = {"mono": 1, "stereo": 2, "2.1": 3, "quad": 4, "5.0": 5, "5.1": 6}

Rate = Union[Fraction, int, float, str]
ClipId = Hashable


# --- the edit as data ----------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """Sound from source second ``src_in`` to ``src_out``, under picture clip ``clip``."""

    src_in: float
    src_out: float
    clip: ClipId

    def __post_init__(self):
        if not 0 <= self.src_in < self.src_out:
            raise ValueError(f"a Span needs 0 <= src_in < src_out: {self}")


@dataclass(frozen=True)
class Gap:
    """``duration`` seconds of silence, under picture clip ``clip``."""

    duration: float
    clip: ClipId

    def __post_init__(self):
        if not self.duration > 0:
            raise ValueError(f"a Gap needs duration > 0: {self}")


Segment = Union[Span, Gap]


@dataclass(frozen=True)
class Cam:
    """The source picture, punched in by ``zoom`` centred at ``(cx, cy)``.

    ``cx``, ``cy`` are fractions of the frame; the window is cropped (kept
    inside the frame) then scaled back to the source size. ``dissolve_in`` (s)
    dissolves into this clip from the previous one.
    """

    zoom: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    dissolve_in: float = 0.0

    def __post_init__(self):
        if self.zoom < 1:
            raise ValueError(f"Cam zoom must be >= 1 (a punch-in): {self}")
        if not (0 <= self.cx <= 1 and 0 <= self.cy <= 1):
            raise ValueError(f"Cam cx, cy are fractions of the frame: {self}")
        if self.dissolve_in < 0:
            raise ValueError(f"dissolve_in must be >= 0: {self}")


@dataclass(frozen=True)
class Still:
    """A held still image with a slow move.

    ``image`` is a path or a PIL image. ``path`` is the move, resolved by
    ``burns.resolve_move``: a name from ``burns.MOVES`` (``"hold"``, the
    default, ``"push_in"``, ``"drift_left"``, ...), a ``burns.BurnsPath``, or
    its ``to_dict()`` payload. ``zoom`` is a named move's end zoom (``None``:
    :data:`HOLD_ZOOM` for a hold, ``burns``' default otherwise).

    ``fit`` decides what happens when the image's aspect is not the frame's:
    ``"pad"`` (default) letter- or pillarboxes it, so the whole picture is
    held and never silently cover-cropped; ``"fill"`` lets ``burns`` crop it to
    cover the frame. ``content_box`` (normalized ``(x, y, w, h)`` of the image)
    is the part that is picture, e.g. inside a pillarboxed card: every window
    of the move is slid inside it (centred on it where the window is larger),
    so a zoomed move never shows the bars. Padding added by ``fit="pad"`` is
    outside the content box automatically. ``on_aspect_mismatch`` is passed to
    ``burns.resolve_move`` for an explicit path authored at another aspect
    (``"raise"``, default, or ``"refit"``).

    The move spans the whole picture clip, whatever sound cuts happen under it.
    """

    image: Any
    path: Any = DEFAULT_STILL_MOVE
    zoom: Optional[float] = None
    fit: str = DEFAULT_STILL_FIT
    content_box: Optional[tuple[float, float, float, float]] = None
    on_aspect_mismatch: str = "raise"
    dissolve_in: float = 0.0

    def __post_init__(self):
        if self.dissolve_in < 0:
            raise ValueError(f"dissolve_in must be >= 0: {self}")
        if self.fit not in STILL_FITS:
            raise ValueError(f"Still fit must be one of {STILL_FITS}: {self}")
        if self.content_box is not None:
            x, y, w, h = self.content_box
            if not (
                0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= 1 and y + h <= 1
            ):
                raise ValueError(f"content_box must lie inside the unit square: {self}")


Clip = Union[Cam, Still]


@dataclass(frozen=True)
class Overlay:
    """An image with alpha shown over output seconds ``[start, end)``.

    ``image`` is a path or anything with ``.save(path)`` (a PIL image, e.g. a
    ``tituli`` render). ``x``, ``y`` are its position (pixels or an ffmpeg
    ``overlay`` expression). ``fade`` (s) fades it in and out inside the range.
    """

    image: Any
    start: float
    end: float
    x: Union[int, str] = 0
    y: Union[int, str] = 0
    fade: float = 0.0

    def __post_init__(self):
        if not 0 <= self.start < self.end:
            raise ValueError(f"an Overlay needs 0 <= start < end: {self}")
        if not 0 <= 2 * self.fade <= self.end - self.start:
            raise ValueError(f"an Overlay's fades must fit in its range: {self}")


# --- the timeline (pure) ---------------------------------------------------------


@dataclass(frozen=True)
class SoundPiece:
    """One segment's sound: output samples ``[start, end)``.

    ``src_start`` is the source sample at ``start`` (``None``: silence);
    ``fade_in`` / ``fade_out`` are the join fades, in samples.
    """

    start: int
    end: int
    src_start: Optional[int]
    fade_in: int = 0
    fade_out: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PicturePiece:
    """``frames`` output frames of a Cam clip from source frame ``src_frame`` on.

    ``hold=True`` repeats ``src_frame`` instead of playing on from it.
    """

    src_frame: int
    frames: int
    hold: bool = False


@dataclass(frozen=True)
class PictureClip:
    """A run of consecutive segments under one clip id, in output frames.

    It occupies ``[start, end)`` and is rendered ``head`` frames early and
    ``tail`` frames late for the dissolves around it. ``pieces`` is the source
    frame plan of a :class:`Cam` clip, head and tail included (empty for a
    :class:`Still`). ``dissolve`` is the length of the dissolve into it.
    """

    clip_id: ClipId
    clip: Clip
    start: int
    end: int
    head: int = 0
    tail: int = 0
    dissolve: int = 0
    pieces: tuple[PicturePiece, ...] = ()

    @property
    def render_frames(self) -> int:
        return self.end - self.start + self.head + self.tail


@dataclass(frozen=True)
class Timeline:
    """Frame and sample boundaries of an edit (see :func:`timeline`).

    ``frame_bounds[k]`` / ``sample_bounds[k]`` is where segment ``k`` starts;
    the last entry is the total.
    """

    fps: Fraction
    sample_rate: int
    frame_bounds: tuple[int, ...]
    sample_bounds: tuple[int, ...]
    sound: tuple[SoundPiece, ...]
    picture: tuple[PictureClip, ...]

    @property
    def total_frames(self) -> int:
        return self.frame_bounds[-1]

    @property
    def total_samples(self) -> int:
        return self.sample_bounds[-1]


def as_rate(value: Rate) -> Fraction:
    """A frame rate as an exact :class:`~fractions.Fraction`.

    Strings are parsed exactly; a float near an integer or an NTSC rate
    (``k * 1000 / 1001``) is taken to mean it.

    >>> as_rate("30000/1001"), as_rate(29.97), as_rate(25), as_rate(23.976)
    (Fraction(30000, 1001), Fraction(30000, 1001), Fraction(25, 1), Fraction(24000, 1001))
    """
    if isinstance(value, (Fraction, int)):
        rate = Fraction(value)
    elif isinstance(value, str):
        rate = Fraction(value.strip())
    else:
        rate = _snap_rate(float(value))
    if rate <= 0:
        raise ValueError(f"a frame rate must be > 0, got {value!r}")
    return rate


def _snap_rate(fps: float) -> Fraction:
    if abs(fps - round(fps)) < RATE_SNAP_TOLERANCE:
        return Fraction(round(fps))
    k = round(fps * NTSC_DENOMINATOR / 1000)
    ntsc = Fraction(k * 1000, NTSC_DENOMINATOR)
    if abs(fps - float(ntsc)) < RATE_SNAP_TOLERANCE:
        return ntsc
    return Fraction(fps).limit_denominator(MAX_RATE_DENOMINATOR)


def _frame_at(seconds: float, fps: Fraction) -> int:
    return round(Fraction(seconds) * fps)


def _sample_at(frame: int, fps: Fraction, sample_rate: int) -> int:
    return round(frame * Fraction(sample_rate) / fps)


def _checked_segment(segment) -> Segment:
    if not isinstance(segment, (Span, Gap)):
        raise TypeError(f"a segment is a Span or a Gap, got {segment!r}")
    return segment


def _segment_frames(segments, fps, source_frames) -> list[tuple[Optional[int], int]]:
    """Per segment: (first source frame or None for a gap, output frame count).

    A span reaching past the source's last frame is refused: there is no
    picture or sound there to play.
    """
    out = []
    for seg in segments:
        if isinstance(seg, Span):
            f_in, f_out = _frame_at(seg.src_in, fps), _frame_at(seg.src_out, fps)
            n = f_out - f_in
            if source_frames is not None and f_out > source_frames:
                raise ValueError(
                    f"{seg} ends at frame {f_out}, past the end of the source "
                    f"({source_frames} frames, {float(source_frames / fps):.3f} s "
                    f"at {fps} fps). Shorten the span or use a Gap."
                )
        else:
            f_in, n = None, _frame_at(seg.duration, fps)
        if n <= 0:
            raise ValueError(f"{seg} is shorter than one frame at {fps} fps")
        out.append((f_in, n))
    return out


def _sound_pieces(
    segments, frames, frame_bounds, sample_bounds, *, fps, sample_rate, join_fade
):
    """The sound plan: continuous source runs, silences, fades at real joins."""
    last = len(segments) - 1
    # contiguous[k]: segment k flows on, in the source, into segment k + 1
    contiguous = [
        k < last
        and frames[k][0] is not None
        and frames[k + 1][0] is not None
        and frames[k][0] + frames[k][1] == frames[k + 1][0]
        for k in range(len(segments))
    ]
    fade = round(join_fade * sample_rate)
    pieces: list[SoundPiece] = []
    for k, (f_in, _) in enumerate(frames):
        start, end = sample_bounds[k], sample_bounds[k + 1]
        if f_in is None:
            pieces.append(SoundPiece(start, end, None))
            continue
        if k and contiguous[k - 1]:
            src_start = pieces[-1].src_start + pieces[-1].length
        else:
            src_start = _sample_at(f_in, fps, sample_rate)
        n = min(fade, (end - start) // 2)
        pieces.append(
            SoundPiece(
                start,
                end,
                src_start,
                fade_in=n if k and not contiguous[k - 1] else 0,
                fade_out=n if k < last and not contiguous[k] else 0,
            )
        )
    return tuple(pieces)


def _cam_body(group, segments, frames) -> list[PicturePiece]:
    """Source pieces of a Cam clip's segments (gaps hold the neighbouring frame)."""
    spans = [k for k in group if frames[k][0] is not None]
    if not spans:
        raise ValueError(
            f"Cam clip {segments[group[0]].clip!r} covers only gaps: it has no "
            "source frame to hold. Put a Still under the gap, or give the clip a span."
        )
    pieces: list[PicturePiece] = []
    for k in group:
        f_in, n = frames[k]
        if f_in is not None:
            prev = pieces[-1] if pieces else None
            if prev and not prev.hold and prev.src_frame + prev.frames == f_in:
                pieces[-1] = PicturePiece(prev.src_frame, prev.frames + n)
            else:
                pieces.append(PicturePiece(f_in, n))
            continue
        before = [j for j in spans if j < k]
        ref = (
            frames[before[-1]][0] + frames[before[-1]][1] - 1
            if before
            else frames[next(j for j in spans if j > k)][0]
        )
        pieces.append(PicturePiece(ref, n, hold=True))
    return pieces


def _extend(pieces: list[PicturePiece], head: int, tail: int, source_frames) -> tuple:
    """Extend a Cam clip's pieces by ``head``/``tail`` frames of source handle,
    holding the edge frame where the source runs out."""
    first, last = pieces[0], pieces[-1]
    if head:
        if first.hold:
            pieces[0] = PicturePiece(first.src_frame, first.frames + head, True)
        else:
            take = min(head, first.src_frame)
            pieces[0] = PicturePiece(first.src_frame - take, first.frames + take)
            if head > take:
                pieces.insert(0, PicturePiece(0, head - take, True))
        last = pieces[-1]
    if tail:
        if last.hold:
            pieces[-1] = PicturePiece(last.src_frame, last.frames + tail, True)
        else:
            end = last.src_frame + last.frames
            room = tail if source_frames is None else max(0, source_frames - end)
            take = min(tail, room)
            pieces[-1] = PicturePiece(last.src_frame, last.frames + take)
            if tail > take:
                pieces.append(PicturePiece(end + take - 1, tail - take, True))
    return tuple(pieces)


def _picture_clips(segments, clips, frames, frame_bounds, *, fps, source_frames):
    groups = [
        [k for k, _ in grp]
        for _, grp in groupby(enumerate(segments), key=lambda kv: kv[1].clip)
    ]
    dissolves = []
    for j, group in enumerate(groups):
        clip = clips[segments[group[0]].clip]
        d = _frame_at(clip.dissolve_in, fps)
        if j == 0 and d:
            raise ValueError(
                f"the first picture clip {segments[group[0]].clip!r} cannot "
                "dissolve in: there is nothing before it to dissolve from"
            )
        dissolves.append(d)
    heads = [d // 2 for d in dissolves]
    tails = [d - h for d, h in zip(dissolves[1:], heads[1:])] + [0]
    out = []
    for j, group in enumerate(groups):
        clip_id = segments[group[0]].clip
        clip = clips[clip_id]
        start, end = frame_bounds[group[0]], frame_bounds[group[-1] + 1]
        # frames of its own body each dissolve blends: the incoming one past its
        # start, the outgoing one before its end. They may not overlap.
        blended_in = tails[j - 1] if j else 0
        blended_out = heads[j + 1] if j + 1 < len(groups) else 0
        if end - start < blended_in + blended_out:
            raise ValueError(
                f"picture clip {clip_id!r} is {end - start} frames: too short for "
                f"its dissolves ({blended_in} + {blended_out} frames of it are blended)"
            )
        pieces = ()
        if isinstance(clip, Cam):
            body = _cam_body(group, segments, frames)
            pieces = _extend(body, heads[j], tails[j], source_frames)
        out.append(
            PictureClip(
                clip_id, clip, start, end, heads[j], tails[j], dissolves[j], pieces
            )
        )
    return tuple(out)


def timeline(
    segments: Iterable,
    clips: Mapping[ClipId, Clip],
    *,
    fps: Rate,
    sample_rate: int,
    source_frames: Optional[int] = None,
    join_fade: float = JOIN_FADE_S,
) -> Timeline:
    """The frame-exact plan of an edit: frame and sample boundaries, sound and
    picture pieces. Pure: no file is read.

    Args:
        segments: :class:`Span` / :class:`Gap`.
        clips: clip id -> :class:`Cam` / :class:`Still`.
        fps: the timeline rate (see :func:`as_rate`).
        sample_rate: the sound's sample rate.
        source_frames: the source's frame count at ``fps``: a span past it is
            refused, and dissolve handles past it hold the last frame
            (``None``: assume the source runs on).
        join_fade: fade (s) at each sound join that is not a continuous run.

    >>> plan = timeline([Span(0, 1, "a"), Gap(0.5, "a")], {"a": Cam()},
    ...                 fps="30000/1001", sample_rate=48000)
    >>> plan.frame_bounds, plan.sample_bounds
    ((0, 30, 45), (0, 48048, 72072))
    """
    segments = [_checked_segment(s) for s in segments]
    if not segments:
        raise ValueError("an edit needs at least one segment")
    missing = sorted({repr(s.clip) for s in segments if s.clip not in clips})
    if missing:
        raise ValueError(f"segments name clips not in `clips`: {', '.join(missing)}")
    fps = as_rate(fps)
    frames = _segment_frames(segments, fps, source_frames)
    frame_bounds = [0]
    for _, n in frames:
        frame_bounds.append(frame_bounds[-1] + n)
    sample_bounds = [_sample_at(b, fps, sample_rate) for b in frame_bounds]
    sound = _sound_pieces(
        segments, frames, frame_bounds, sample_bounds,
        fps=fps, sample_rate=sample_rate, join_fade=join_fade,
    )  # fmt: skip
    picture = _picture_clips(
        segments, clips, frames, frame_bounds, fps=fps, source_frames=source_frames
    )
    plan = Timeline(
        fps, sample_rate, tuple(frame_bounds), tuple(sample_bounds), sound, picture
    )
    picture_frames = sum(c.render_frames - c.dissolve for c in picture)
    assert picture_frames == plan.total_frames, (picture_frames, plan.total_frames)
    assert sum(p.length for p in sound) == plan.total_samples
    assert all(
        sum(p.frames for p in c.pieces) == c.render_frames for c in picture if c.pieces
    )
    return plan


def punch_in_box(
    width: int, height: int, *, zoom: float, cx: float = 0.5, cy: float = 0.5
) -> tuple[int, int, int, int]:
    """The ``(x, y, w, h)`` pixel crop of a punch-in, kept inside the frame.

    >>> punch_in_box(64, 36, zoom=2.0, cx=0.0, cy=0.0)
    (0, 0, 32, 18)
    >>> punch_in_box(1920, 1080, zoom=1.5)
    (320, 180, 1280, 720)
    """
    w, h = round(width / zoom), round(height / zoom)
    x = round(min(max(cx * width - w / 2, 0), width - w))
    y = round(min(max(cy * height - h / 2, 0), height - h))
    return x, y, w, h


# --- probing ------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """What a render needs to know about its source (see :func:`probe_source`).

    ``width`` x ``height`` is the **displayed** size (a 90/270-degree display
    rotation swaps them; ffmpeg rotates on decode). ``frames`` is the frame
    count at ``fps``. ``video_start`` is where the first video frame sits after
    the file's start (what an input seek counts from), and ``audio_offset``
    how far the sound starts after the first video frame (negative: before).
    """

    width: int
    height: int
    fps: Fraction
    frames: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    video_start: float = 0.0
    audio_offset: float = 0.0


def _ffprobe_exe() -> Optional[str]:
    sibling = Path(ffmpeg_exe()).with_name("ffprobe")
    return str(sibling) if sibling.is_file() else shutil.which("ffprobe")


def _rotation(stream: dict) -> float:
    for datum in stream.get("side_data_list", ()):
        if "rotation" in datum:
            return float(datum["rotation"])
    return float(stream.get("tags", {}).get("rotate", 0))


def _is_quarter_turn(degrees: float) -> bool:
    return abs(round(degrees)) % 180 == 90


def _ratio(text: Optional[str]) -> Optional[Fraction]:
    if not text or text.endswith("/0"):
        return None
    value = Fraction(text)
    return value if value > 0 else None


def _seconds(text) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _probe_with_ffprobe(binary: str, path) -> SourceInfo:
    proc = subprocess.run(
        [binary, "-v", "error", "-count_packets", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    probed = json.loads(proc.stdout)
    streams, fmt = probed.get("streams", []), probed.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    nominal, average = (
        _ratio(video.get("r_frame_rate")),
        _ratio(video.get("avg_frame_rate")),
    )
    fps = nominal or average
    if fps is None:
        raise ValueError(f"cannot read the frame rate of {path}")
    variable = bool(nominal and average and abs(average / nominal - 1) > VFR_TOLERANCE)
    if variable:
        warnings.warn(
            f"{path} looks variable-rate (r_frame_rate {nominal}, avg_frame_rate "
            f"{average}): its picture is resampled to {fps} fps by time. Pass "
            "fps= to choose the timeline rate.",
            stacklevel=3,
        )
    packets = video.get("nb_read_packets")
    duration = _seconds(video.get("duration")) or _seconds(fmt.get("duration"))
    if packets and packets.isdigit() and not variable:
        frames = int(packets)
    else:
        frames = _frame_at(duration, fps) if duration else None
    origin = _seconds(fmt.get("start_time")) or 0.0
    video_start = (_seconds(video.get("start_time")) or origin) - origin
    audio_start = (
        (_seconds(audio.get("start_time")) or origin) - origin if audio else 0.0
    )
    width, height = int(video["width"]), int(video["height"])
    if _is_quarter_turn(_rotation(video)):
        width, height = height, width
    return SourceInfo(
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        sample_rate=int(audio["sample_rate"]) if audio else None,
        channels=int(audio["channels"]) if audio else None,
        video_start=video_start,
        audio_offset=audio_start - video_start,
    )


_BANNER_VIDEO = re.compile(r"Stream #.*Video:.*?,\s*(\d{2,5})x(\d{2,5}).*?([\d.]+) fps")
_BANNER_AUDIO = re.compile(r"Stream #.*Audio:.*?(\d+) Hz, ([\w.]+)")
_BANNER_DURATION = re.compile(r"Duration: (\d+):(\d+):([\d.]+)")
_BANNER_ROTATION = re.compile(r"rotation of (-?[\d.]+) degrees")


def _probe_with_banner(path) -> SourceInfo:
    """Fallback when there is no ffprobe: parse ``ffmpeg -i``'s banner.

    The banner has no per-stream start times, so both streams are taken to
    start with the file, and the frame count is estimated from its duration.
    """
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )
    video = _BANNER_VIDEO.search(proc.stderr)
    if not video:
        raise ValueError(f"no video stream in {path}")
    fps = as_rate(float(video.group(3)))
    audio = _BANNER_AUDIO.search(proc.stderr)
    duration = _BANNER_DURATION.search(proc.stderr)
    seconds = (
        int(duration.group(1)) * 3600
        + int(duration.group(2)) * 60
        + float(duration.group(3))
        if duration
        else None
    )
    width, height = int(video.group(1)), int(video.group(2))
    rotation = _BANNER_ROTATION.search(proc.stderr)
    if rotation and _is_quarter_turn(float(rotation.group(1))):
        width, height = height, width
    return SourceInfo(
        width=width,
        height=height,
        fps=fps,
        frames=_frame_at(seconds, fps) if seconds else None,
        sample_rate=int(audio.group(1)) if audio else None,
        channels=CHANNEL_LAYOUTS.get(audio.group(2), 2) if audio else None,
    )


def probe_source(path: PathLike) -> SourceInfo:
    """Displayed size, exact frame rate, frame count, sound format and stream
    offsets of ``path``.

    Uses ``ffprobe`` (its ``r_frame_rate`` is the exact rational; packets are
    counted, which reads but does not decode the file) when one is found
    beside ffmpeg or on ``PATH``, else ffmpeg's banner (rate snapped by
    :func:`as_rate`, frame count from the duration, no stream offsets). Warns
    when the source looks variable-rate.
    """
    binary = _ffprobe_exe()
    if binary:
        return _probe_with_ffprobe(binary, path)
    return _probe_with_banner(path)


# --- rendering ------------------------------------------------------------------


def _t(seconds) -> str:
    return f"{float(seconds):.{TIME_DECIMALS}f}"


def _rate(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-2000:]}")


def _stamp(fps: Fraction) -> str:
    """Restamp frames as ``0, 1, 2, ...`` frames at ``fps``, whatever their pts."""
    return f"settb={fps.denominator}/{fps.numerator},setpts=N"


def _intermediate(frames: int, fps: Fraction, output) -> list[str]:
    """The output half of every part's command: exactly ``frames`` frames, lossless."""
    return [
        "-frames:v", str(frames), "-an", "-r", _rate(fps),
        *INTERMEDIATE_ENCODE_ARGS, str(output),
    ]  # fmt: skip


def _slice(pieces, start: int, stop: int) -> list[PicturePiece]:
    """The pieces covering a clip's render frames ``[start, stop)``.

    >>> _slice([PicturePiece(10, 5), PicturePiece(40, 5, hold=True)], 3, 7)
    [PicturePiece(src_frame=13, frames=2, hold=False), PicturePiece(src_frame=40, frames=2, hold=True)]
    """
    out, pos = [], 0
    for piece in pieces:
        lo, hi = max(start, pos), min(stop, pos + piece.frames)
        if lo < hi:
            first = piece.src_frame if piece.hold else piece.src_frame + lo - pos
            out.append(PicturePiece(first, hi - lo, piece.hold))
        pos += piece.frames
    return out


def _parts(plan: Timeline) -> list[tuple]:
    """The bounded render jobs of the picture, in output order.

    ``("solo", k, a, b)``: frames ``[a, b)`` of clip ``k``'s render.
    ``("dissolve", k, d)``: the last ``d`` frames of clip ``k - 1`` blended
    into the first ``d`` of clip ``k``. Their frame counts sum to the film's.
    """
    parts = []
    picture = plan.picture
    for k, clip in enumerate(picture):
        d_out = picture[k + 1].dissolve if k + 1 < len(picture) else 0
        if clip.dissolve:
            parts.append(("dissolve", k, clip.dissolve))
        if clip.render_frames - d_out > clip.dissolve:
            parts.append(("solo", k, clip.dissolve, clip.render_frames - d_out))
    return parts


def _cam_piece_command(
    source, piece: PicturePiece, cam: Cam, *, info: SourceInfo, fps: Fraction, output
) -> list[str]:
    """One ffmpeg run, one decoder: a piece of source picture, exactly ``frames`` long.

    Seeked half a frame before its first frame, re-zeroed, then resampled to
    the timeline rate by time (``fps``), so a variable-rate source plays at
    its speed; a hold repeats one frame; a punch-in crops and scales back.
    """
    seek = info.video_start + (piece.src_frame - SEEK_LEAD_FRAMES) / fps
    inputs = ["-ss", _t(seek)] if seek > 0 else []
    chain = ["setpts=PTS-STARTPTS", f"fps={_rate(fps)}"]
    if piece.hold:
        chain += ["trim=end_frame=1", f"tpad=stop_mode=clone:stop={piece.frames - 1}"]
    else:
        chain += [f"trim=end_frame={piece.frames}"]
    if cam.zoom != 1:
        x, y, w, h = punch_in_box(
            info.width, info.height, zoom=cam.zoom, cx=cam.cx, cy=cam.cy
        )
        chain += [
            f"crop={w}:{h}:{x}:{y}",
            f"scale={info.width}:{info.height}:flags={SCALE_FLAGS}",
        ]
    # a source that ends early holds its last frame rather than shortening the part
    chain += ["setsar=1", "format=yuv420p", "tpad=stop_mode=clone:stop=-1", _stamp(fps)]
    return [
        ffmpeg_exe(), "-y", "-v", "error", *inputs, "-i", str(source),
        "-vf", ",".join(chain), *_intermediate(piece.frames, fps, output),
    ]  # fmt: skip


def _still_slice_command(
    still: Path, start: int, stop: int, *, size, fps: Fraction, output
) -> list[str]:
    """Frames ``[start, stop)`` of a rendered still (its last frame held if short)."""
    w, h = size
    chain = [
        "tpad=stop_mode=clone:stop=-1",
        f"trim=start_frame={start}:end_frame={stop}",
        f"scale={w}:{h}:flags={SCALE_FLAGS}",
        "setsar=1",
        "format=yuv420p",
        _stamp(fps),
    ]
    return [
        ffmpeg_exe(), "-y", "-v", "error", "-i", str(still),
        "-vf", ",".join(chain), *_intermediate(stop - start, fps, output),
    ]  # fmt: skip


def _concat_list(files, path: Path) -> Path:
    """A concat-demuxer list of ``files``, written to ``path``."""

    def quoted(f) -> str:
        return "'" + str(Path(f).resolve()).replace("'", "'\\''") + "'"

    path.write_text("".join(f"file {quoted(f)}\n" for f in files))
    return path


def _sequence_input(files, list_path: Path) -> list[str]:
    """Input arguments reading ``files`` back to back with ONE decoder."""
    if len(files) == 1:
        return ["-i", str(files[0])]
    return ["-f", "concat", "-safe", "0", "-i", str(_concat_list(files, list_path))]


def _dissolve_command(outgoing, incoming, frames: int, *, fps: Fraction, output):
    """Two inputs, ``xfade``: ``frames`` frames blending ``outgoing`` into ``incoming``."""
    graph = (
        f"[0:v]{_stamp(fps)},fps={_rate(fps)}[a];[1:v]{_stamp(fps)},fps={_rate(fps)}[b];"
        f"[a][b]xfade=transition={DISSOLVE_TRANSITION}:"
        f"duration={_t(Fraction(frames) / fps)}:offset=0,{_stamp(fps)}[v]"
    )
    return [
        ffmpeg_exe(), "-y", "-v", "error", *outgoing, *incoming,
        "-filter_complex", graph, "-map", "[v]", *_intermediate(frames, fps, output),
    ]  # fmt: skip


def _render_picture(
    plan: Timeline, *, source, info: SourceInfo, still_renderer, size, workdir: Path
) -> list[Path]:
    """Render the picture as a sequence of part files, each from at most two
    decoders, whatever the number of segments (memory O(1) in their count)."""
    fps = plan.fps
    names = count()
    stills: dict[int, Path] = {}

    def new_path(stem="part") -> Path:
        return workdir / f"{stem}{next(names):05d}{INTERMEDIATE_SUFFIX}"

    def still_file(k: int) -> Path:  # each still is rendered once, when first needed
        if k not in stills:
            clip = plan.picture[k]
            stills[k] = Path(
                still_renderer(
                    clip.clip,
                    frames=clip.render_frames,
                    fps=fps,
                    size=size,
                    output=new_path("still"),
                )
            )
        return stills[k]

    def frames_of(k: int, start: int, stop: int) -> list[Path]:
        clip, files = plan.picture[k], []
        if isinstance(clip.clip, Cam):
            for piece in _slice(clip.pieces, start, stop):
                files.append(new_path())
                _run(
                    _cam_piece_command(
                        source, piece, clip.clip, info=info, fps=fps, output=files[-1]
                    )
                )
        else:
            files.append(new_path())
            _run(
                _still_slice_command(
                    still_file(k), start, stop, size=size, fps=fps, output=files[-1]
                )
            )
        return files

    sequence: list[Path] = []
    for part in _parts(plan):
        if part[0] == "solo":
            _, k, start, stop = part
            sequence += frames_of(k, start, stop)
            continue
        _, k, d = part
        before = plan.picture[k - 1].render_frames
        outgoing = _sequence_input(frames_of(k - 1, before - d, before), new_path("a"))
        incoming = _sequence_input(frames_of(k, 0, d), new_path("b"))
        sequence.append(new_path())
        _run(_dissolve_command(outgoing, incoming, d, fps=fps, output=sequence[-1]))
    return sequence


def _pad_to_aspect(image, aspect: float, Image):
    """``image`` letter- or pillarboxed to ``aspect``, and where it now sits
    (normalized ``(x, y, w, h)``; ``None`` when it already had that aspect)."""
    iw, ih = image.size
    if abs(iw / ih / aspect - 1) <= ASPECT_TOLERANCE:
        return image, None
    if iw / ih < aspect:
        width, height = round(ih * aspect), ih
    else:
        width, height = iw, round(iw / aspect)
    canvas = Image.new("RGB", (width, height), PAD_COLOR)
    x, y = (width - iw) // 2, (height - ih) // 2
    canvas.paste(image, (x, y))
    return canvas, (x / width, y / height, iw / width, ih / height)


def _within(outer, box):
    """``box`` (normalized to an image) in the coordinates ``outer`` places it at."""
    if box is None:
        return outer
    ox, oy, ow, oh = outer
    bx, by, bw, bh = box
    return (ox + bx * ow, oy + by * oh, bw * ow, bh * oh)


def slide_into(window, box) -> tuple[float, float, float, float]:
    """``window`` ``(x, y, w, h)`` slid inside ``box``, never resized.

    On an axis where the window is larger than the box it is centred on the
    box instead (and kept inside the image).

    >>> slide_into((0.0, 0.0, 0.5, 0.5), (0.25, 0.0, 0.5, 1.0))
    (0.25, 0.0, 0.5, 0.5)
    >>> slide_into((0.0, 0.0, 1.0, 1.0), (0.25, 0.0, 0.5, 1.0))
    (0.0, 0.0, 1.0, 1.0)
    """

    def axis(pos, size, lo, span):
        if size <= span:
            return min(max(pos, lo), lo + span - size)
        return min(max(lo + (span - size) / 2, 0.0), 1.0 - size)

    x, y, w, h = window
    bx, by, bw, bh = box
    return (axis(x, w, bx, bw), axis(y, h, by, bh), w, h)


def burns_still_renderer(
    still: Still, *, frames: int, fps: Fraction, size: tuple[int, int], output: Path
) -> Path:
    """Render ``still`` as exactly ``frames`` frames of ``size`` with ``burns``.

    The move is resolved by ``burns.resolve_move`` at the frame's aspect
    (named moves, explicit paths, and its ``on_aspect_mismatch`` refusal);
    the image is padded to that aspect first when ``fit="pad"``, and every
    window is slid inside the content box (see :class:`Still`). ``burns``
    renders integer rates only, so a fractional ``fps`` (29.97) is rendered at
    the next integer with the **same frame count**; each part restamps the
    frames at the edit's rate (the move runs 0.1% slower; no frame is dropped
    or repeated). The default ``still_renderer`` of :func:`render`; any
    callable with this signature can replace it.
    """
    require_media_extra("still clips in mixing.edit", modules=("burns", "PIL"))
    import burns
    from PIL import Image

    image = still.image
    image = Image.open(image) if isinstance(image, (str, os.PathLike)) else image
    image = image.convert("RGB")
    aspect = size[0] / size[1]
    box = still.content_box
    if still.fit == "pad":
        image, placed = _pad_to_aspect(image, aspect, Image)
        if placed is not None:
            box = _within(placed, box)
    zoom = still.zoom
    if zoom is None:
        zoom = HOLD_ZOOM if still.path == DEFAULT_STILL_MOVE else burns.DFLT_ZOOM
    path = burns.resolve_move(
        still.path,
        image=image,
        aspect=aspect,
        zoom=zoom,
        content_box=box,
        on_aspect_mismatch=still.on_aspect_mismatch,
    )
    if box is not None:
        path = burns.BurnsPath(
            keyframes=tuple(
                (t, burns.Rect(*slide_into((r.x, r.y, r.w, r.h), box)))
                for t, r in path.keyframes
            ),
            easing=path.easing,
            interp=path.interp,
            output_aspect=path.output_aspect,
        )
    rate = math.ceil(fps)
    burns.ken_burns_video(
        image,
        path,
        duration=frames / rate,
        fps=rate,
        saveas=str(output),
        output_size=size,
        backend=STILL_BACKEND,
        bitrate=STILL_BITRATE,
    )
    return Path(output)


def _decode_sound(source, *, sample_rate: int, channels: int) -> bytes:
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn",
         "-f", PCM_FORMAT, "-ar", str(sample_rate), "-ac", str(channels), "-"],
        capture_output=True,
    )  # fmt: skip
    if proc.returncode:
        raise RuntimeError(
            f"ffmpeg could not decode the sound of {source}: {proc.stderr[-2000:]!r}"
        )
    return proc.stdout


def _ramp(buf: array, start: int, n: int, channels: int, *, rising: bool) -> None:
    """Fade ``n`` frames of interleaved ``buf`` from frame ``start``, in place, linearly
    (rising from 0, or falling from 1)."""
    for i in range(n):
        gain = i / n if rising else (n - i) / n
        base = (start + i) * channels
        for c in range(channels):
            buf[base + c] *= gain


def _source_chunk(pcm: bytes, first: int, length: int, frame_bytes: int) -> bytes:
    """``length`` sample frames of ``pcm`` from ``first``; silence outside it."""
    lead = min(length, max(0, -first))
    stop = first + length
    body = pcm[max(first, 0) * frame_bytes : max(stop, 0) * frame_bytes]
    chunk = bytes(lead * frame_bytes) + body
    return chunk + bytes(length * frame_bytes - len(chunk))


def _assemble_sound(
    pcm: bytes, plan: Timeline, *, channels: int, closing_frames: int, offset: int = 0
) -> array:
    """The edit's sound, interleaved float samples, exactly ``total_samples`` long.

    ``offset`` is the source sample (counted from the first video frame) that
    ``pcm[0]`` is, so a sound stream starting late or early stays in sync.
    """
    frame_bytes = channels * PCM_BYTES
    out = array(PCM_TYPECODE)
    for piece in plan.sound:
        if piece.src_start is None:
            out.extend(array(PCM_TYPECODE, bytes(piece.length * frame_bytes)))
            continue
        chunk = _source_chunk(pcm, piece.src_start - offset, piece.length, frame_bytes)
        part = array(PCM_TYPECODE, chunk)
        _ramp(part, 0, piece.fade_in, channels, rising=True)
        _ramp(
            part, piece.length - piece.fade_out, piece.fade_out, channels, rising=False
        )
        out.extend(part)
    if closing_frames:
        # the same frames as the picture's fade: full at frame N-1-k, silent from N-1
        last = plan.total_frames - 1
        start = _sample_at(last - closing_frames, plan.fps, plan.sample_rate)
        stop = _sample_at(last, plan.fps, plan.sample_rate)
        _ramp(out, start, stop - start, channels, rising=False)
        for i in range(stop * channels, len(out)):
            out[i] = 0.0
    assert len(out) == plan.total_samples * channels
    return out


def _overlay_file(overlay: Overlay, workdir: Path, j: int) -> str:
    if isinstance(overlay.image, (str, os.PathLike)):
        return str(overlay.image)
    path = workdir / f"overlay{j}.png"
    overlay.image.save(path)
    return str(path)


def _overlay_frames(overlays, plan: Timeline) -> list[tuple[int, int, Fraction]]:
    """Each overlay's output frames ``[a, b)`` and fade (s), checked against
    the film: one starting at or after its end is refused; one running past it
    is clipped to it, its fades shortened to fit what is left."""
    fps, total = plan.fps, plan.total_frames
    out = []
    for overlay in overlays:
        a, b = _frame_at(overlay.start, fps), min(_frame_at(overlay.end, fps), total)
        if a >= total:
            raise ValueError(
                f"{overlay} starts at frame {a}, at or after the end of the film "
                f"({total} frames, {float(total / fps):.3f} s)"
            )
        if b <= a:
            raise ValueError(f"{overlay} is shorter than one frame at {fps} fps")
        fade = min(Fraction(overlay.fade), Fraction(b - a, 2) / fps)
        out.append((a, b, fade))
    return out


def _final_graph(
    plan: Timeline, *, overlays, spans, overlay_inputs: list[int], closing_frames: int
) -> str:
    """The final pass over the joined parts: overlays, then the closing fade."""
    fps = plan.fps
    parts = [f"[0:v]{_stamp(fps)},trim=end_frame={plan.total_frames}[base]"]
    acc = "base"
    for j, (overlay, (a, b, fade), index) in enumerate(
        zip(overlays, spans, overlay_inputs)
    ):
        fades = ""
        if fade:
            fades = (
                f",fade=t=in:st={_t(a / fps)}:d={_t(fade)}:alpha=1"
                f",fade=t=out:st={_t(b / fps - fade)}:d={_t(fade)}:alpha=1"
            )
        parts.append(f"[{index}:v]format=rgba,{_stamp(fps)}{fades}[o{j}]")
        parts.append(
            f"[{acc}][o{j}]overlay=x={overlay.x}:y={overlay.y}:eof_action=pass:"
            f"enable='between(n,{a},{b - 1})'[ov{j}]"
        )
        acc = f"ov{j}"
    tail = ""
    if closing_frames:  # full at frame N-1-k, black on the last frame
        first = plan.total_frames - 1 - closing_frames
        tail = f"fade=t=out:start_frame={first}:nb_frames={closing_frames},"
    parts.append(f"[{acc}]{tail}format=yuv420p[vout]")
    return ";".join(parts)


def _check_source(source: Path, info: SourceInfo) -> None:
    if not (info.sample_rate and info.channels):
        raise ValueError(f"{source} has no sound stream: a sound edit needs one")
    if info.width % 2 or info.height % 2:
        raise ValueError(
            f"{source} is {info.width}x{info.height}: the delivery is yuv420p, "
            "which needs even dimensions. Re-encode the source to an even size "
            "first (e.g. ffmpeg -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2')."
        )


def render(
    segments: Iterable[Segment],
    clips: Mapping[ClipId, Clip],
    *,
    source: PathLike,
    overlays: Iterable[Overlay] = (),
    fps: Optional[Rate] = None,
    output: Output = None,
    still_renderer: Callable[..., Path] = burns_still_renderer,
    closing_fade: float = DEFAULT_CLOSING_FADE_S,
    join_fade: float = JOIN_FADE_S,
    encode_args: tuple[str, ...] = DEFAULT_ENCODE_ARGS,
    probe: Callable[[PathLike], SourceInfo] = probe_source,
) -> Union[Path, Any]:
    """Render the edit ``segments`` / ``clips`` of ``source``, frame-exact.

    Args:
        segments: the sound edit list, :class:`Span` / :class:`Gap`.
        clips: picture clip id -> :class:`Cam` / :class:`Still`.
        source: the finished video being re-edited (must have sound, and an
            even size).
        overlays: :class:`Overlay` images composited in the final pass.
        fps: the timeline rate; default the source's exact rate.
        output: where the film goes (see :mod:`mixing.egress`); default beside
            ``source`` as ``<stem>_edit.mp4``.
        still_renderer: ``still_renderer(still, *, frames, fps, size, output)``
            renders a :class:`Still` as ``frames`` frames (see
            :func:`burns_still_renderer`, the default).
        closing_fade: seconds of fade to black (and to silence) at the end.
        join_fade: seconds of fade at each sound join (see :data:`JOIN_FADE_S`).
        encode_args: the final encode's ffmpeg arguments.
        probe: ``probe(source) -> SourceInfo`` (see :func:`probe_source`).

    Returns:
        The written ``Path`` or, when ``output`` is a callable, what it
        returns for that path.
    """
    source = Path(source)
    overlays = tuple(overlays)
    info = probe(source)
    _check_source(source, info)
    rate = as_rate(fps) if fps is not None else info.fps
    source_frames = info.frames
    if source_frames is not None and rate != info.fps:
        source_frames = math.floor(Fraction(source_frames) * rate / info.fps)
    plan = timeline(
        segments, clips, fps=rate, sample_rate=info.sample_rate,
        source_frames=source_frames, join_fade=join_fade,
    )  # fmt: skip
    spans = _overlay_frames(overlays, plan)
    closing_frames = min(_frame_at(closing_fade, rate), plan.total_frames - 1)
    size = (info.width, info.height)

    def write(target: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="mixing-edit-") as tmp:
            workdir = Path(tmp)
            sequence = _render_picture(
                plan, source=source, info=info, still_renderer=still_renderer,
                size=size, workdir=workdir,
            )  # fmt: skip
            pcm = _decode_sound(
                source, sample_rate=info.sample_rate, channels=info.channels
            )
            sound = _assemble_sound(
                pcm, plan, channels=info.channels, closing_frames=closing_frames,
                offset=round(info.audio_offset * info.sample_rate),
            )  # fmt: skip
            sound_path = workdir / "sound.pcm"
            sound_path.write_bytes(sound.tobytes())
            del pcm, sound
            inputs = _sequence_input(sequence, workdir / "parts.txt")
            inputs += ["-f", PCM_FORMAT, "-ar", str(info.sample_rate),
                       "-ac", str(info.channels), "-i", str(sound_path)]  # fmt: skip
            film_seconds = _t(Fraction(plan.total_frames) / rate)
            overlay_inputs = []
            for j, overlay in enumerate(overlays):
                overlay_inputs.append(2 + j)
                inputs += ["-loop", "1", "-framerate", _rate(rate), "-t", film_seconds,
                           "-i", _overlay_file(overlay, workdir, j)]  # fmt: skip
            graph = _final_graph(
                plan, overlays=overlays, spans=spans,
                overlay_inputs=overlay_inputs, closing_frames=closing_frames,
            )  # fmt: skip
            _run([
                ffmpeg_exe(), "-y", "-v", "error", *inputs, "-filter_complex", graph,
                "-map", "[vout]", "-map", "1:a", "-r", _rate(rate),
                *encode_args, str(target),
            ])  # fmt: skip

    default_path = source.with_name(source.stem + DEFAULT_OUTPUT_SUFFIX)
    return write_egress(output, default_path=default_path, write=write)
