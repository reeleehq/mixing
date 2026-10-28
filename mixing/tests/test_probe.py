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
#: Seconds the sound of ``early_sound.mkv`` starts before its picture (mkv
#: timestamps are milliseconds).
AUDIO_LEAD = 0.27

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
        ("audio", "tone.wav"), ("still", "still.png"), ("early", "early_sound.mkv"),
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
    # video at 0.0, sound from -AUDIO_LEAD (and the file with it): both starts
    # are stated, one of them exactly 0.0
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=s={W}x{H}:r=25:d={SECONDS}",
        "-itsoffset", str(-AUDIO_LEAD), "-f", "lavfi", "-i", f"sine=r={SR}:d={SECONDS}",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-avoid_negative_ts", "disabled", str(files["early"]),
    )  # fmt: skip
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
    found = probe(media["ntsc"], ffprobe=None)
    assert found.method == BANNER_METHOD and not found.exact
    # no packet count: the frame count is the container duration times the rate
    assert found.packets is None
    assert found.frames == round(Fraction(found.duration) * found.fps)
    # no stream start times: both are assumed to be the file's
    assert (found.video_start, found.audio_start) == (0.0, 0.0)


def test_banner_reads_vfr_from_its_two_rates(media):
    """``tbr`` is the nominal rate, the printed ``fps`` the average: ``fps``
    means the same on both methods, and a VFR source is seen without ffprobe."""
    found = probe(media["vfr"], ffprobe=None)
    assert found.fps == VFR_RATE  # tbr, like ffprobe's r_frame_rate
    assert found.avg_fps == Fraction(573, 25)  # the printed 22.92 fps
    assert found.variable_rate
    assert found.frames == round(Fraction(found.duration) * found.fps)


@needs_ffprobe
def test_fps_means_the_nominal_rate_on_both_methods(media):
    for name in ("ntsc", "vfr", "still"):
        assert probe(media[name]).fps == probe(media[name], ffprobe=None).fps, name


@needs_ffprobe
def test_a_stated_zero_start_is_a_start_not_a_missing_value(media):
    """Regression: video at exactly 0.0 under a file starting at -0.27 s sits
    0.27 s in, and the sound 0.27 s before it (``or origin`` read 0.0 as unset)."""
    found = probe(media["early"])
    assert found.video_start == pytest.approx(AUDIO_LEAD, abs=2e-3)
    assert found.audio_start == pytest.approx(0.0, abs=2e-3)
    assert found.audio_offset == pytest.approx(-AUDIO_LEAD, abs=2e-3)
    # mkv's DURATION tag is an end timestamp: the length is not cut by the lead
    assert not found.variable_rate and found.frames == found.packets == 25 * SECONDS


def test_parse_banner_is_pure():
    found = parse_banner(
        "Duration: 00:01:02.50, start: 0\n"
        "Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 23.98 fps, 23.98 tbr\n"
        "Side data:\n Display Matrix: rotation of -90.00 degrees\n"
        "Stream #0:1: Audio: aac, 48000 Hz, 5.1(side), fltp"
    )
    assert found.fps == Fraction(24000, 1001)
    assert (found.width, found.height, found.rotation) == (1080, 1920, -90.0)
    assert (found.sample_rate, found.channels) == (48000, 6)
    assert found.duration == 62.5 and found.frames == round(Fraction(62.5) * found.fps)
    assert not parse_banner("Stream #0:0: Audio: aac, 8000 Hz, mono").has_video
    assert parse_banner("nothing here") == MediaProbe(method=BANNER_METHOD)
    assert not found.variable_rate and found.avg_fps == found.fps


def test_parse_banner_rates():
    def video(rates):
        return parse_banner(f"Stream #0:0: Video: h264, yuv420p, 64x48, {rates}")

    vfr = video("15.52 fps, 30 tbr, 15360 tbn")
    assert (vfr.fps, vfr.avg_fps, vfr.variable_rate) == (30, Fraction(388, 25), True)
    fast = video("90k tbr, 90k tbn")  # no fps figure: tbr alone, k-suffixed
    assert (fast.fps, fast.avg_fps, fast.variable_rate) == (90000, None, False)
    only_fps = video("25 fps")  # what parse_media_info's callers pass
    assert (only_fps.fps, only_fps.avg_fps) == (25, 25)
    near = video("29.97 fps, 30 tbr")  # 0.1 % apart: within VFR_TOLERANCE
    assert near.variable_rate is False


def test_snap_rate():
    assert snap_rate(29.97) == NTSC
    assert snap_rate(59.94) == Fraction(60000, 1001)
    assert snap_rate(24.0) == 24
    assert snap_rate(12.5) == Fraction(25, 2)


def test_a_banner_rate_snapping_to_zero_is_no_rate():
    found = parse_banner("Stream #0:0: Video: png, rgb24, 64x48, 0.0033 fps, 0.0033 tbr")
    assert found.has_video and (found.fps, found.avg_fps) == (None, None)


def test_cover_art_is_not_the_picture_in_a_banner():
    found = parse_banner(
        "Duration: 00:00:02.00\n"
        "Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s\n"
        "Stream #0:1: Video: mjpeg (Baseline), yuvj420p, 300x300, 90k tbr, 90k tbn"
        " (attached pic)\n"
    )
    assert not found.has_video and found.has_audio


@BOTH_WAYS
def test_an_mp3_with_cover_art_is_audio_only(media, tmp_path, ffprobe):
    song = tmp_path / "song.mp3"
    _ffmpeg(
        "-f", "lavfi", "-i", f"sine=r={SR}:d=1", "-i", str(media["still"]),
        "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame", "-c:v", "png",
        "-disposition:v", "attached_pic", str(song),
    )  # fmt: skip
    found = probe(song, ffprobe=ffprobe)
    assert not found.has_video and found.fps is None
    assert found.has_audio and found.sample_rate == SR


@needs_ffprobe
def test_first_audio_frame_not_first_packet(tmp_path):
    """opus drops its pre-skip: its first packet (-0.007 s) plays from 0."""
    clip = tmp_path / "opus.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=s={W}x{H}:r=25:d=1",
        "-f", "lavfi", "-i", "sine=r=48000:d=1", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "libopus", str(clip),
    )  # fmt: skip
    found = probe(clip)
    assert found.audio_offset == pytest.approx(0.0, abs=1e-3)


def test_find_ffprobe_ignores_a_non_exe_suffix(tmp_path, monkeypatch):
    """imageio's Linux ffmpeg is ``ffmpeg-linux-x86_64-v7.0.2``: its sibling is
    ``ffprobe``, not ``ffprobe.2``."""
    import mixing.probe

    (tmp_path / "ffmpeg-linux-x86_64-v7.0.2").touch()
    (tmp_path / "ffprobe").touch()
    monkeypatch.setattr(
        mixing.probe, "ffmpeg_exe", lambda: str(tmp_path / "ffmpeg-linux-x86_64-v7.0.2")
    )
    assert find_ffprobe() == str(tmp_path / "ffprobe")


def test_banner_error_states_the_exit_code(tmp_path, monkeypatch):
    import mixing.probe

    junk = tmp_path / "crash.ts"
    junk.touch()

    def crashed(argv, **kwargs):
        return subprocess.CompletedProcess(argv, -11, stdout="", stderr="")

    monkeypatch.setattr(mixing.probe.subprocess, "run", crashed)
    with pytest.raises(ProbeError, match=r"could not read it \(exit code -11\)"):
        probe(junk, ffprobe=None)


def test_find_ffprobe_takes_ffmpegs_suffix(tmp_path, monkeypatch):
    """Beside ``ffmpeg.exe`` the sibling is ``ffprobe.exe`` (Windows)."""
    import mixing.probe

    (tmp_path / "ffmpeg.exe").touch()
    (tmp_path / "ffprobe.exe").touch()
    monkeypatch.setattr(mixing.probe, "ffmpeg_exe", lambda: str(tmp_path / "ffmpeg.exe"))
    assert find_ffprobe() == str(tmp_path / "ffprobe.exe")


def test_a_bad_ffprobe_path_names_the_binary(media, tmp_path):
    bogus = tmp_path / "no-such-ffprobe"
    with pytest.raises(ProbeError, match="cannot run ffprobe .*no-such-ffprobe") as caught:
        probe(media["ntsc"], ffprobe=str(bogus))
    assert not isinstance(caught.value, FileNotFoundError)  # the media is there


def test_an_empty_ffprobe_is_refused(media):
    with pytest.raises(ValueError, match="names no binary"):
        probe(media["ntsc"], ffprobe="")


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
