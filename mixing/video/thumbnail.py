"""Thumbnail / cover-image generation from a video frame.

Extracts a representative frame (:func:`grab_frame`, also used by
:func:`mixing.video.shot_boundaries.contact_sheet`) and renders it to a 16:9
1280x720 image (a good default for video thumbnails and podcast-cover-over-video alike),
optionally overlaying a short title with a legible gradient band. Uses ffmpeg
for frame extraction and Pillow for compositing. Platform-neutral — the size
is a sensible default, not a YouTube-specific coupling.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from ..egress import Output, write_egress
from ..util import ffmpeg_exe

PathLike = str | Path

#: Default 16:9 thumbnail/cover resolution (also YouTube's recommended size).
THUMBNAIL_SIZE = (1280, 720)
#: Backwards-compatible alias.
YOUTUBE_THUMB_SIZE = THUMBNAIL_SIZE

#: Fraction of the video duration used for the default frame grab when
#: ``at_time`` is omitted — 85% lands near the end, typically on the closing
#: brand/logo shot, which makes a good default thumbnail.
DEFAULT_FRAME_TIME_FRACTION = 0.85


def make_thumbnail(
    video: PathLike,
    *,
    at_time: float | None = None,
    text: str | None = None,
    output: Output = None,
    size: tuple[int, int] = THUMBNAIL_SIZE,
) -> Path:
    """Create a thumbnail image from a frame of ``video``.

    Args:
        video: Source video path.
        at_time: Timestamp (seconds) of the frame to grab. Defaults to 85% of
            the video duration (typically the closing brand/logo shot).
        text: Optional short overlay text (e.g. the title). Rendered bottom-left
            over a dark gradient band for legibility.
        output: Where to put the result — None (save beside the input as
            ``<video-stem>.thumb.jpg``), a file path, a directory (auto-named),
            or a callable sink. See mixing.egress.
        size: Output size (width, height). Defaults to 1280x720.

    Returns:
        Path to the written JPEG.
    """
    from PIL import Image

    video = Path(video)
    default_path = video.with_suffix("").with_name(f"{video.stem}.thumb.jpg")

    if at_time is None:
        at_time = DEFAULT_FRAME_TIME_FRACTION * _media_duration(video)

    def _write(out: Path) -> None:
        img = _cover_resize(grab_frame(video, at_time), size)
        if text:
            _overlay_text(img, text)
        img.save(out, quality=90)

    return write_egress(output, default_path=default_path, write=_write)


def _seek_time(at_time: float) -> str:
    """``at_time`` for ``-ss``: to the nanosecond (rounding to milliseconds moved
    a time computed to land on a 29.97 fps frame past it, onto the next one),
    minus :data:`_TIME_EPS_S` so it sits on that frame, as the one pass does;
    plain decimals, which ffmpeg's time parser reads (it refuses ``1e-06``)."""
    return f"{max(0.0, at_time - _TIME_EPS_S):.9f}"


def grab_frame(video: PathLike, at_time: float, *, width: int | None = None):
    """The frame of ``video`` at ``at_time`` seconds as an RGB PIL image.

    ``width`` scales it (height follows the aspect ratio, kept even); ``None``
    keeps the video's size. One ffmpeg seek, piped as PNG (no temp file).
    """
    import io

    from PIL import Image

    scale = ["-vf", f"scale={width}:-2"] if width else []
    cmd = [
        ffmpeg_exe(), "-v", "error", "-ss", _seek_time(at_time), "-i", str(video),
        "-frames:v", "1", *scale, "-f", "image2pipe", "-vcodec", "png", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg could not grab a frame of {video!s} at {at_time:.2f}s: "
            f"{proc.stderr.decode(errors='replace')}"
        )
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


#: Shots per minute of video above which :func:`grab_frames` decodes the video
#: once rather than seeking once per frame. Measured on 30 min of 720p h264
#: (reeleehq/mixing#94): one pass ~20 s whatever the count, a seek ~0.085 s
#: each (3 s at 1/min, 34 s at 15/min, 153 s at 60/min). Measured on synthetic
#: ``testsrc2`` at x264's default keyframe interval: a seek's cost follows the
#: keyframe interval, so real footage may move the crossover; re-measure with
#: ``misc/bench_contact_sheet.py``.
ONE_PASS_SHOTS_PER_MINUTE = 8.0
#: :func:`grab_frames` methods: one ffmpeg seek per frame, one decoding pass,
#: or the faster of the two for the frame density.
GRAB_METHODS = ("auto", "seek", "pass")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_END = b"IEND"
_PNG_CHUNK_OVERHEAD = 12  # length + type + crc
#: Seconds the one pass decodes past the last time asked for (then stops).
_PASS_TAIL_S = 1.0
#: A frame whose time equals an asked time to this many seconds counts as at
#: it (a time computed as ``k * 1001 / 30000`` lands an ulp off the frame's).
_TIME_EPS_S = 1e-6


class _PassUnavailable(RuntimeError):
    """The one pass cannot answer (two times in one frame interval, a time past
    the end, or an ffmpeg older than 7): seeking still can."""


def _split_pngs(stream: bytes) -> list[bytes]:
    """The PNG files concatenated in ``stream`` (walked chunk by chunk, so a
    signature-like run inside compressed data cannot split one)."""
    images, at = [], 0
    try:
        while at < len(stream):
            if stream[at : at + len(_PNG_SIGNATURE)] != _PNG_SIGNATURE:
                raise RuntimeError("ffmpeg's frame stream is not a run of PNG images")
            end = at + len(_PNG_SIGNATURE)
            while True:
                (length,) = struct.unpack(">I", stream[end : end + 4])
                kind = stream[end + 4 : end + 8]
                end += _PNG_CHUNK_OVERHEAD + length
                if kind == _PNG_END:
                    break
            if end > len(stream):
                raise struct.error("truncated")
            images.append(stream[at:end])
            at = end
    except struct.error as e:
        raise RuntimeError(f"ffmpeg's PNG stream is truncated ({e})") from e
    return images


def _num_s(seconds: float) -> str:
    return f"{seconds:.6f}"


def _frames_in_one_pass(video, times: Sequence[float], width: Optional[int]):
    """The first frame at or after each of ``times`` (sorted, distinct), from one
    decode (up to just past the last time): a ``select`` of the frames whose interval
    ``[prev_t, t]`` crosses a time, so two times in one frame interval give one
    frame (the caller checks the count)."""
    import io

    from PIL import Image

    # on the file's clock, as ``-ss`` seeks: a frame is picked when the interval
    # since the one before it crosses a time (an epsilon below it, so a time
    # computed to land on a frame picks that frame, not the next)
    at = [repr(t - _TIME_EPS_S) for t in times]
    terms = "+".join(f"gte(t,{a})*lt(prev_t,{a})" for a in at)
    # the first frame has no prev_t (NaN): it is picked when a time is <= its own
    first = f"isnan(prev_t)*gte(t,{at[0]})"
    scale = f",scale={width}:-2" if width else ""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "select.txt"
        # a file, not an argument: thousands of terms exceed a Windows command line
        script.write_text(f"select='{first}+{terms}'{scale}")
        proc = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-to", _num_s(times[-1] + _PASS_TAIL_S),
             "-i", str(video), "-/filter:v", str(script),
             "-fps_mode", "passthrough", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        )  # fmt: skip
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        if "/filter:v" in stderr:  # ffmpeg < 7 knows no ``-/filter``
            raise _PassUnavailable(
                "the one pass needs ffmpeg >= 7 (for -/filter:v); upgrade ffmpeg "
                "or use method='seek'"
            )
        raise RuntimeError(
            f"ffmpeg could not select frames of {video!s}: {stderr[-500:]}"
        )
    return [
        Image.open(io.BytesIO(png)).convert("RGB") for png in _split_pngs(proc.stdout)
    ]


def grab_frames(
    video: PathLike,
    times: Sequence[float],
    *,
    width: int | None = None,
    method: str = "auto",
    duration_s: Optional[float] = None,
) -> list:
    """The frames of ``video`` at ``times`` (seconds, any order) as RGB PIL images.

    ``method`` (:data:`GRAB_METHODS`): ``"seek"`` runs :func:`grab_frame` per
    time (one ffmpeg seek each: cheap for a few frames of a long video);
    ``"pass"`` decodes the video once and selects every frame (cheap for many
    frames; it stops decoding just past the last time); ``"auto"`` (default)
    takes the pass above :data:`ONE_PASS_SHOTS_PER_MINUTE` frames per minute of
    ``duration_s`` (default: the last time) when the frames are scaled
    (``width``: the pass holds every frame in memory at once), and falls back
    to seeking when the pass cannot give one distinct frame per time (two times
    in one frame interval, a time past the end, or an ffmpeg older than 7).
    Both give the first frame at or after each time, on the file's clock.
    """
    if method not in GRAB_METHODS:
        raise ValueError(f"unknown method {method!r}; one of {GRAB_METHODS}")
    times = [float(t) for t in times]
    if not times:
        return []
    if method == "auto":
        minutes = max(duration_s or max(times), 1e-9) / 60
        dense = len(times) / minutes > ONE_PASS_SHOTS_PER_MINUTE
        method = "pass" if dense and width is not None else "seek"
        if method == "pass":
            try:
                return grab_frames(video, times, width=width, method="pass")
            except _PassUnavailable:
                method = "seek"
    if method == "seek":
        return [grab_frame(video, t, width=width) for t in times]
    order = sorted(range(len(times)), key=times.__getitem__)
    ordered = [times[i] for i in order]
    frames = _frames_in_one_pass(video, ordered, width)
    if len(frames) != len(ordered):
        raise _PassUnavailable(
            f"one pass over {video!s} gave {len(frames)} frames for {len(times)} "
            "times (two in one frame interval, or past the end); use method='seek'"
        )
    out = [None] * len(times)
    for i, frame in zip(order, frames):
        out[i] = frame
    return out


def _media_duration(path: PathLike) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def _cover_resize(img, size):
    """Resize+center-crop ``img`` to exactly ``size`` (object-fit: cover)."""
    from PIL import Image

    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h = img.size
    left, top = (w - tw) // 2, (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _overlay_text(img, text: str) -> None:
    """Draw ``text`` bottom-left over a dark gradient band, in place."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = img.size
    band_h = int(h * 0.32)
    band = Image.new("RGBA", (w, band_h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    for y in range(band_h):
        alpha = int(200 * (y / band_h))  # transparent at top -> dark at bottom
        bd.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
    img.paste(Image.new("RGB", (w, band_h), (0, 0, 0)), (0, h - band_h), band)

    draw = ImageDraw.Draw(img)
    font = _load_font(int(h * 0.085))
    margin = int(w * 0.04)
    text = _wrap(draw, text, font, w - 2 * margin)
    _, _, _, text_h = draw.multiline_textbbox((0, 0), text, font=font)
    draw.multiline_text(
        (margin, h - margin - text_h),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        spacing=6,
    )


def _load_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_width: int) -> str:
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)
