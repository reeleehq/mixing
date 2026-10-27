"""Characterization tests pinning the CURRENT behavior of thumbnail generation.

Targets ``mixing.video.thumbnail``:

- :func:`make_thumbnail` — extracts a frame from a video and renders it to a
  cover-fit image (with an optional text overlay band).
- Constants :data:`THUMBNAIL_SIZE` and :data:`YOUTUBE_THUMB_SIZE`.

These tests pin OBSERVABLE behavior (output exists, dimensions, format,
default save path, no-crash on text overlay) so an upcoming refactor cannot
silently change semantics. They do NOT assert on private helpers or internal
implementation details.

NOTE on current signature (to be preserved or deliberately changed by the
refactor): the first positional parameter is ``video`` (not ``media``) and the
output path keyword is ``output`` (the canonical egress parameter). The default
output is ``<video-stem>.thumb.jpg`` written next to the source video, and the
default frame timestamp is 85% of the source duration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Frame extraction (ffmpeg) + Pillow compositing are required for every test.
pytest.importorskip("PIL")

from mixing.video.thumbnail import (  # noqa: E402
    make_thumbnail,
    THUMBNAIL_SIZE,
    YOUTUBE_THUMB_SIZE,
)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def test_thumbnail_size_constant_is_1280x720():
    """The default 16:9 size is pinned at 1280x720."""
    assert THUMBNAIL_SIZE == (1280, 720)


def test_youtube_thumb_size_is_alias_of_thumbnail_size():
    """``YOUTUBE_THUMB_SIZE`` is the backwards-compatible alias and is equal."""
    assert YOUTUBE_THUMB_SIZE == THUMBNAIL_SIZE
    assert YOUTUBE_THUMB_SIZE == (1280, 720)


# --------------------------------------------------------------------------
# make_thumbnail: basic production of an image file
# --------------------------------------------------------------------------


def test_make_thumbnail_produces_image_file(color_video, tmp_path):
    """A thumbnail is written to the requested path and is a readable image."""
    from PIL import Image

    out = tmp_path / "thumb.jpg"
    result = make_thumbnail(color_video, output=out)

    assert isinstance(result, Path)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
    # Pillow can open it -> it really is an image.
    with Image.open(out) as im:
        im.verify()


def test_make_thumbnail_default_size_is_youtube_preset(color_video, tmp_path):
    """With no ``size`` override, output is exactly the 1280x720 preset."""
    out = tmp_path / "thumb.jpg"
    make_thumbnail(color_video, output=out)
    assert _image_size(out) == (1280, 720)
    assert _image_size(out) == YOUTUBE_THUMB_SIZE


def test_make_thumbnail_respects_custom_size(color_video, tmp_path):
    """An explicit ``size`` is honored exactly (cover-fit to that box)."""
    out = tmp_path / "thumb.jpg"
    make_thumbnail(color_video, output=out, size=(640, 360))
    assert _image_size(out) == (640, 360)


def test_make_thumbnail_default_saveas_is_thumb_jpg_next_to_video(
    make_color_video, tmp_path
):
    """Default output path is ``<stem>.thumb.jpg`` beside the source video."""
    # Place a video in an isolated dir so we can assert on the derived name.
    src = make_color_video(1.0)
    expected = src.with_suffix("").with_name(f"{src.stem}.thumb.jpg")
    try:
        result = make_thumbnail(src)
        assert result == expected
        assert expected.exists()
        assert _image_size(expected) == THUMBNAIL_SIZE
    finally:
        expected.unlink(missing_ok=True)


def test_make_thumbnail_writes_only_the_thumbnail(color_video, tmp_path):
    """No intermediate frame file is left beside the output (the frame is now
    piped from ffmpeg, so there is none to clean up)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "thumb.jpg"
    make_thumbnail(color_video, output=out)
    assert list(out_dir.iterdir()) == [out]


# --------------------------------------------------------------------------
# make_thumbnail: at_time
# --------------------------------------------------------------------------


def test_make_thumbnail_explicit_at_time(color_video, tmp_path):
    """An explicit ``at_time`` grabs a frame without error, sized to preset."""
    out = tmp_path / "thumb.jpg"
    make_thumbnail(color_video, output=out, at_time=0.1)
    assert out.exists()
    assert _image_size(out) == THUMBNAIL_SIZE


# --------------------------------------------------------------------------
# make_thumbnail: text overlay path does not crash
# --------------------------------------------------------------------------


def test_make_thumbnail_with_text_overlay_does_not_crash(color_video, tmp_path):
    """Passing ``text`` exercises the overlay band; output stays a valid image."""
    from PIL import Image

    out = tmp_path / "thumb_text.jpg"
    result = make_thumbnail(color_video, output=out, text="Hello World")
    assert result == out
    assert out.exists()
    assert _image_size(out) == THUMBNAIL_SIZE
    with Image.open(out) as im:
        im.verify()


def test_make_thumbnail_with_long_wrapping_text_does_not_crash(color_video, tmp_path):
    """Long text exercises the word-wrap path without raising."""
    out = tmp_path / "thumb_long.jpg"
    long_text = "A rather long title that should wrap across multiple lines nicely"
    make_thumbnail(color_video, output=out, text=long_text)
    assert out.exists()
    assert _image_size(out) == THUMBNAIL_SIZE


def test_make_thumbnail_empty_text_is_falsy_so_no_overlay(color_video, tmp_path):
    """Empty string is falsy: the overlay branch is skipped, image still made."""
    out = tmp_path / "thumb_empty.jpg"
    make_thumbnail(color_video, output=out, text="")
    assert out.exists()
    assert _image_size(out) == THUMBNAIL_SIZE


# --------------------------------------------------------------------------
# make_thumbnail: different output formats
# --------------------------------------------------------------------------


def test_make_thumbnail_png_output(color_video, tmp_path):
    """A ``.png`` output path produces a PNG image of the preset size."""
    from PIL import Image

    out = tmp_path / "thumb.png"
    make_thumbnail(color_video, output=out)
    assert out.exists()
    with Image.open(out) as im:
        assert im.format == "PNG"
        assert im.size == THUMBNAIL_SIZE


def test_make_thumbnail_jpg_output(color_video, tmp_path):
    """A ``.jpg`` output path produces a JPEG image of the preset size."""
    from PIL import Image

    out = tmp_path / "thumb.jpg"
    make_thumbnail(color_video, output=out)
    assert out.exists()
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.size == THUMBNAIL_SIZE


def test_make_thumbnail_accepts_str_path(color_video, tmp_path):
    """``video`` and ``output`` accept str paths (coerced via Path)."""
    out = tmp_path / "thumb_strpath.jpg"
    result = make_thumbnail(str(color_video), output=str(out))
    assert isinstance(result, Path)
    assert result == out
    assert out.exists()
