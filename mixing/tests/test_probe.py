"""``mixing.probe``: ffprobe first, the ``ffmpeg -i`` banner as the fallback.

The media is made with ffmpeg ``lavfi`` sources in a temporary directory: an
NTSC-rate (``30000/1001``) clip with sound, the same clip with a 90-degree
display rotation, a variable-rate clip (every 4th frame dropped), an
audio-only file and a still. Each is read both ways (``ffprobe=None`` forces
the banner) where the two must agree, and the exact-only facts are pinned on
the ffprobe reading.
"""

import json
import subprocess
import sys
from fractions import Fraction

import pytest

from mixing.errors import MediaNotFoundError, ProbeError
from mixing.probe import (
    BANNER_METHOD,
    FFPROBE_METHOD,
    MediaProbe,
    find_ffprobe,
    parse_banner,
    probe,
    snap_rate,
)
from mixing.util import has_ffmpeg

NTSC = Fraction(30000, 1001)
W, H = 64, 48
SR = 44100
SECONDS = 2  # 2 s at 30000/1001 is 59.94 frames: ffmpeg writes 60
NTSC_FRAMES = 60
VFR_RATE = 30
VFR_SECONDS = 3

needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="needs ffmpeg on PATH")
needs_ffprobe = pytest.mark.skipif(
    not has_ffmpeg() or find_ffprobe() is None, reason="needs ffprobe"
)
BOTH_WAYS = pytest.mark.parametrize(
    "ffprobe", [pytest.param("auto", marks=needs_ffprobe), None], ids=["ffprobe", "banner"]
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    if not has_ffmpeg():
        pytest.skip("needs ffmpeg on PATH")
    d = tmp_path_factory.mktemp("probe")
    files = {k: d / name for k, name in [
        ("ntsc", "ntsc.mp4"), ("rotated", "rotated.mp4"), ("vfr", "vfr.mp4"),
        ("audio", "tone.wav"), ("still", "still.png"),
    ]}  # fmt: skip
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=s={W}x{H}:r=30000/1001:d={SECONDS}",
        "-f", "lavfi", "-i", f"sine=r={SR}:d={SECONDS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2",
        str(files["ntsc"]),
    )  # fmt: skip
    _ffmpeg("-display_rotation", "90", "-i", str(files["ntsc"]), "-c", "copy",
            str(files["rotated"]))  # fmt: skip
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=s={W}x{H}:r={VFR_RATE}:d={VFR_SECONDS}",
        "-vf", "select='not(eq(mod(n\\,4)\\,3))'", "-fps_mode", "vfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(files["vfr"]),
    )  # fmt: skip
    _ffmpeg("-f", "lavfi", "-i", "sine=r=22050:d=1.5", "-ac", "1", str(files["audio"]))
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=s={W}x{H}", "-frames:v", "1",
            str(files["still"]))  # fmt: skip
    return files


# --- both ways -------------------------------------------------------------------


@BOTH_WAYS
def test_odd_ntsc_rate_is_exact(media, ffprobe):
    found = probe(media["ntsc"], ffprobe=ffprobe)
    assert isinstance(found, MediaProbe)
    assert found.fps == NTSC  # exact from ffprobe; the banner's 29.97 snapped
    assert (found.width, found.height, found.rotation) == (W, H, 0.0)
    assert found.frames == NTSC_FRAMES
    assert found.has_video and found.has_audio
    assert (found.sample_rate, found.channels) == (SR, 2)
    assert found.duration == pytest.approx(SECONDS, abs=0.01)
    assert found.audio_offset == pytest.approx(0.0, abs=0.05)


@BOTH_WAYS
def test_rotated_file_reports_its_displayed_size(media, ffprobe):
    found = probe(media["rotated"], ffprobe=ffprobe)
    assert abs(found.rotation) == 90
    assert (found.width, found.height) == (H, W)  # displayed: swapped
    assert found.coded_size == (W, H)


@BOTH_WAYS
def test_audio_only_file_has_no_video_fields(media, ffprobe):
    found = probe(media["audio"], ffprobe=ffprobe)
    assert not found.has_video
    assert (found.width, found.height, found.fps, found.frames) == (None,) * 4
    assert found.coded_size is None
    assert found.has_audio and (found.sample_rate, found.channels) == (22050, 1)
    assert found.duration == pytest.approx(1.5, abs=0.01)


@BOTH_WAYS
def test_still_image_is_a_one_picture_video(media, ffprobe):
    found = probe(media["still"], ffprobe=ffprobe)
    assert (found.width, found.height) == (W, H)
    assert not found.has_audio and found.audio_offset is None


@BOTH_WAYS
def test_missing_file_is_named_in_the_error(tmp_path, ffprobe):
    missing = tmp_path / "nowhere.mp4"
    with pytest.raises(MediaNotFoundError, match="no such media file: .*nowhere.mp4"):
        probe(missing, ffprobe=ffprobe)
    # one error, three ways to catch it
    with pytest.raises(FileNotFoundError):
        probe(missing, ffprobe=ffprobe)
    with pytest.raises(ValueError):
        probe(missing, ffprobe=ffprobe)


@pytest.mark.skipif(not has_ffmpeg(), reason="needs ffmpeg on PATH")
@pytest.mark.parametrize(
    "ffprobe", [pytest.param("auto", marks=needs_ffprobe), None], ids=["ffprobe", "banner"]
)
def test_unreadable_file_raises_probe_error(tmp_path, ffprobe):
    junk = tmp_path / "junk.mp4"
    junk.write_text("not a media file")
    with pytest.raises(ProbeError, match="cannot probe .*junk.mp4") as caught:
        probe(junk, ffprobe=ffprobe)
    assert not isinstance(caught.value, MediaNotFoundError)


# --- what only ffprobe knows --------------------------------------------------------


@needs_ffprobe
def test_ffprobe_reading_is_exact(media):
    found = probe(media["ntsc"])
    assert found.method == FFPROBE_METHOD and found.exact
    assert found.avg_fps == NTSC and found.packets == NTSC_FRAMES
    assert not found.variable_rate


@needs_ffprobe
def test_header_frame_count_without_counting_packets(media):
    found = probe(media["ntsc"], count_packets=False)
    assert found.packets == NTSC_FRAMES  # mp4's nb_frames header
    assert found.frames == NTSC_FRAMES


@needs_ffprobe
def test_variable_rate_is_detected_and_frames_come_from_the_duration(media):
    found = probe(media["vfr"])
    assert found.variable_rate
    assert found.fps == VFR_RATE  # r_frame_rate: the nominal rate
    assert found.avg_fps != found.fps  # what gives it away
    kept = sum(n % 4 != 3 for n in range(VFR_RATE * VFR_SECONDS))  # 68 of 90
    assert found.packets == kept
    # not the packets: the duration (to the last kept frame) at the nominal rate
    assert found.frames == round(Fraction(found.video_duration) * found.fps)
    assert found.frames > kept


def test_explicit_ffprobe_path_is_used(media, monkeypatch):
    import mixing.probe

    def no_banner(media):
        raise AssertionError("the banner must not be used when ffprobe is named")

    monkeypatch.setattr(mixing.probe, "_probe_with_banner", no_banner)
    binary = find_ffprobe()
    if binary is None:
        pytest.skip("needs ffprobe")
    assert probe(media["ntsc"], ffprobe=binary).method == FFPROBE_METHOD


# --- the banner fallback is approximate ------------------------------------------------


def test_banner_fallback_is_approximate(media):
    found = probe(media["vfr"], ffprobe=None)
    assert found.method == BANNER_METHOD and not found.exact
    # no second rate and no packet count: a variable rate goes unnoticed, and
    # the frame count is the container duration times the printed (average) rate
    assert not found.variable_rate
    assert found.avg_fps is None and found.packets is None
    assert found.frames == round(Fraction(found.duration) * found.fps)


def test_parse_banner_is_pure():
    found = parse_banner(
        "Duration: 00:01:02.50, start: 0\n"
        "Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 23.98 fps\n"
        "Side data:\n Display Matrix: rotation of -90.00 degrees\n"
        "Stream #0:1: Audio: aac, 48000 Hz, 5.1(side), fltp"
    )
    assert found.fps == Fraction(24000, 1001)
    assert (found.width, found.height, found.rotation) == (1080, 1920, -90.0)
    assert (found.sample_rate, found.channels) == (48000, 6)
    assert found.duration == 62.5 and found.frames == round(Fraction(62.5) * found.fps)
    assert not parse_banner("Stream #0:0: Audio: aac, 8000 Hz, mono").has_video
    assert parse_banner("nothing here") == MediaProbe(method=BANNER_METHOD)


def test_snap_rate():
    assert snap_rate(29.97) == NTSC
    assert snap_rate(59.94) == Fraction(60000, 1001)
    assert snap_rate(24.0) == 24
    assert snap_rate(12.5) == Fraction(25, 2)


def test_banner_fallback_when_no_ffprobe_beside_ffmpeg(media, monkeypatch):
    """The default finds no ffprobe (imageio-ffmpeg's case): the banner is read."""
    import mixing.probe

    monkeypatch.setattr(mixing.probe, "find_ffprobe", lambda: None)
    assert probe(media["ntsc"]).method == BANNER_METHOD


# --- consumers --------------------------------------------------------------------------


def test_edit_source_info_is_the_probe_view(media):
    from mixing.edit import probe_source

    found, info = probe(media["ntsc"]), probe_source(media["ntsc"])
    assert (info.width, info.height, info.fps, info.frames) == (
        found.width, found.height, found.fps, found.frames,
    )  # fmt: skip
    assert (info.sample_rate, info.channels) == (found.sample_rate, found.channels)


def test_cutaways_probe_media_is_the_probe_view(media):
    pytest.importorskip("moviepy")
    from mixing.video.cutaways import probe_media

    found, info = probe(media["rotated"], count_packets=False), probe_media(media["rotated"])
    assert (info.width, info.height, info.has_audio) == (H, W, True)
    assert info.fps == float(found.fps) and info.duration == found.duration
    with pytest.raises(ValueError, match="no video stream"):
        probe_media(media["audio"])


@pytest.mark.parametrize("name", ["ntsc", "rotated", "vfr"])
def test_get_video_dimensions_agrees_with_the_probe(media, name):
    """``get_video_dimensions`` (cv2, untouched: see .claude/CLAUDE.md, "looks owns
    the geometry vocabulary") reports the same displayed size as the probe."""
    pytest.importorskip("cv2")
    pytest.importorskip("moviepy")
    from mixing.video.video_util import get_video_dimensions

    found = probe(media[name], ffprobe=None)
    assert get_video_dimensions(str(media[name])) == (found.width, found.height)


# --- stays light -------------------------------------------------------------------------


def test_import_mixing_probe_loads_no_media_library():
    code = (
        "import json, sys, mixing.probe; "
        "print(json.dumps([n for n in ('numpy', 'moviepy', 'cv2', 'PIL', 'pydub', 'scipy')"
        " if n in sys.modules]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []
