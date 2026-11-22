"""The pure SRT half of ``mixing.video.video_subtitles``: parsing, shifting, file fixing.

No ffmpeg or moviepy: these are the text-and-path helpers, which a coverage run
showed were reached by nothing (the module sat at 25%).
"""

from pathlib import Path

import pytest

from mixing.video.video_subtitles import (
    SubtitleStyle,
    _get_first_timestamp,
    _get_output_path,
    _parse_srt_subtitle,
    _parse_srt_timestamp,
    _read_srt_content,
    auto_shift_srt_to_start,
    fix_srt_file,
)

SRT = """1
00:00:10,000 --> 00:00:12,000
First

2
00:00:20,500 --> 00:00:22,000
Second
line"""


def test_timestamps_go_from_srt_to_moviepy_form():
    assert _parse_srt_timestamp("00:00:01,500") == "00:00:01.500"


def test_a_block_parses_and_joins_its_text_lines():
    assert _parse_srt_subtitle("2\n00:00:20,500 --> 00:00:22,000\nSecond\nline") == (
        "00:00:20.500",
        "00:00:22.000",
        "Second line",
    )


@pytest.mark.parametrize(
    "block", ["", "1\n00:00:01,000 --> 00:00:02,000", "1\nnot a time range\ntext"]
)
def test_a_malformed_block_is_none(block):
    assert _parse_srt_subtitle(block) is None


def test_first_timestamp_or_not_found():
    assert _get_first_timestamp(SRT) == "00:00:10,000"
    assert _get_first_timestamp("no cues here") == "Not found"


def test_shifting_to_a_start_time_moves_every_cue():
    out = auto_shift_srt_to_start(SRT, start_time=3.0)
    assert "00:00:03,000 --> 00:00:05,000" in out
    assert "00:00:13,500 --> 00:00:15,000" in out


def test_no_start_time_and_no_cues_leave_the_text_alone():
    assert auto_shift_srt_to_start(SRT) == SRT
    assert auto_shift_srt_to_start("nothing", start_time=1.0) == "nothing"


def test_auto_detection_needs_a_video():
    with pytest.raises(ValueError, match="video_path is required"):
        auto_shift_srt_to_start(SRT, start_time=True)


def test_a_callable_start_time_finds_the_target_from_the_video():
    """The docstring promises this; a callable on its own used to be ignored."""
    out = auto_shift_srt_to_start(SRT, video_path="v.mp4", start_time=lambda v: 1.0)
    assert "00:00:01,000 --> 00:00:03,000" in out


def test_fix_srt_file_names_its_output_by_what_it_did(tmp_path):
    src = tmp_path / "a.srt"
    src.write_text(SRT)
    unchanged = fix_srt_file(src)
    assert unchanged.name == "a_fixed.srt" and unchanged.read_text() == SRT
    shifted = fix_srt_file(src, start_time=0.0)
    assert shifted.name == "a_audio_aligned.srt"
    assert "00:00:00,000 --> 00:00:02,000" in shifted.read_text()
    explicit = fix_srt_file(src, tmp_path / "out.srt", start_time=0.0)
    assert explicit == tmp_path / "out.srt"


def test_subtitles_resolve_from_path_content_or_a_sibling_file(tmp_path):
    video = tmp_path / "v.mp4"
    with pytest.raises(FileNotFoundError, match="No subtitle file"):
        _read_srt_content(None, video)
    (tmp_path / "v.srt").write_text("sibling")
    assert _read_srt_content(None, video) == "sibling"
    assert _read_srt_content(str(tmp_path / "v.srt"), video) == "sibling"
    assert _read_srt_content("1\n00:00:01,000 --> 00:00:02,000\nx", video).startswith("1")


def test_output_path_defaults_beside_the_video():
    v = Path("/x/clip.mp4")
    assert _get_output_path(None, v) == Path("/x/clip_with_subtitles.mp4")
    assert _get_output_path(None, v, "_s") == Path("/x/clip_s.mp4")
    assert _get_output_path("/y/o.mkv", v) == Path("/y/o.mkv")


def test_style_renders_ffmpeg_and_clip_arguments():
    style = SubtitleStyle(font_size=30, color="Yellow", position=("center", "top"))
    assert style.to_ffmpeg_style() == (
        "FontSize=30,FontName=Arial,PrimaryColour=&HFFFF00&,Alignment=8"
    )
    assert style.to_text_clip_kwargs() == {"font_size": 30, "color": "Yellow"}
    assert "Alignment=2" in SubtitleStyle(position=("nowhere", "x")).to_ffmpeg_style()
