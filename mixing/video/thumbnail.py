"""Thumbnail / cover-image generation from a video frame.

Extracts a representative frame (:func:`grab_frame`, also used by
:func:`mixing.video.shot_boundaries.contact_sheet`) and renders it to a 16:9
1280x720 image (a good default for video thumbnails and podcast-cover-over-video alike),
optionally overlaying a short title with a legible gradient band. Uses ffmpeg
for frame extraction and Pillow for compositing. Platform-neutral — the size
is a sensible default, not a YouTube-specific coupling.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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


def grab_frame(video: PathLike, at_time: float, *, width: int | None = None):
    """The frame of ``video`` at ``at_time`` seconds as an RGB PIL image.

    ``width`` scales it (height follows the aspect ratio, kept even); ``None``
    keeps the video's size. One ffmpeg seek, piped as PNG (no temp file).
    """
    import io

    from PIL import Image

    scale = ["-vf", f"scale={width}:-2"] if width else []
    cmd = [
        ffmpeg_exe(), "-v", "error", "-ss", f"{at_time:.3f}", "-i", str(video),
        "-frames:v", "1", *scale, "-f", "image2pipe", "-vcodec", "png", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg could not grab a frame of {video!s} at {at_time:.2f}s: "
            f"{proc.stderr.decode(errors='replace')}"
        )
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


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
