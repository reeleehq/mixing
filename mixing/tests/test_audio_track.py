"""``mixing.demux.audio_track``: a muxed file's sound as PCM WAV, own rate and channels.

The media is made with ffmpeg ``lavfi`` sources: a video with a 48 kHz stereo
aac track, the same video with no sound, and a 22.05 kHz mono wav.
"""

import io
import struct
import subprocess
import sys
import wave

import pytest

from mixing.errors import MediaNotFoundError
from mixing.demux import audio_track
from mixing.ffmpeg import FfmpegTimeout
from mixing.util import ffmpeg_exe, has_ffmpeg

SECONDS = 2
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="needs ffmpeg on PATH")


def _ffmpeg(*args):
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    if not has_ffmpeg():
        pytest.skip("needs ffmpeg on PATH")
    d = tmp_path_factory.mktemp("audio_track")
    files = {k: d / name for k, name in [
        ("muxed", "answer.mp4"), ("silent", "silent.mp4"), ("mono", "mono.wav"),
    ]}  # fmt: skip
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=s=64x48:r=25:d={SECONDS}",
        "-f", "lavfi", "-i", f"sine=r=48000:d={SECONDS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2",
        "-shortest", str(files["muxed"]),
    )  # fmt: skip
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=s=64x48:r=25:d={SECONDS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(files["silent"]))  # fmt: skip
    _ffmpeg("-f", "lavfi", "-i", "sine=r=22050:d=1", "-ac", "1", str(files["mono"]))
    return files


def _chunk(data: bytes, name: bytes) -> bytes:
    """The body of a RIFF chunk (not assuming fmt sits at a fixed offset)."""
    at = 12
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], struct.unpack_from("<I", data, at + 4)[0]
        if tag == name:
            return data[at + 8 : at + 8 + size]
        at += 8 + size + size % 2
    raise AssertionError(f"no {name!r} chunk")


def _wav(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()


def test_the_sound_of_a_muxed_video_at_its_own_rate_and_channels(media):
    rate, channels, width, frames = _wav(audio_track(media["muxed"]))
    assert (rate, channels, width) == (48000, 2, 2)  # 16-bit PCM, no resample/downmix
    assert frames == pytest.approx(SECONDS * 48000, rel=0.03)


def test_bytes_in_gives_the_same_wav_as_a_path(media):
    by_path = audio_track(media["muxed"])
    assert audio_track(media["muxed"].read_bytes()) == by_path
    assert by_path[:4] == b"RIFF" and by_path[8:12] == b"WAVE"


def test_a_mono_file_stays_mono_and_codec_picks_the_sample_format(media):
    assert _wav(audio_track(media["mono"]))[:3] == (22050, 1, 2)
    as_float = audio_track(media["mono"], codec="pcm_f32le")
    # the fmt chunk (stdlib ``wave`` reads integer PCM only): format 3 is float
    fmt, channels, rate, _, _, bits = struct.unpack_from(
        "<HHIIHH", _chunk(as_float, b"fmt ")
    )
    assert (fmt, channels, rate, bits) == (3, 1, 22050, 32)


def test_output_follows_the_egress_protocol(media, tmp_path):
    to_dir = audio_track(media["muxed"], output=tmp_path)
    assert to_dir == tmp_path / "answer.wav" and to_dir.read_bytes()[:4] == b"RIFF"
    from_bytes = audio_track(media["muxed"].read_bytes(), output=tmp_path)
    assert from_bytes == tmp_path / "audio.wav"
    assert audio_track(media["mono"], output=tmp_path) == tmp_path / "mono.wav"
    to_file = audio_track(media["muxed"], output=tmp_path / "x.wav")
    assert to_file.read_bytes() == to_dir.read_bytes()
    assert audio_track(media["muxed"], output=len) == len(to_dir.read_bytes())


def test_a_video_with_no_sound_is_an_error_naming_ffmpeg(media):
    with pytest.raises(RuntimeError, match="could not take the audio track out"):
        audio_track(media["silent"])


def test_a_local_file_with_a_colon_is_a_file_not_a_protocol(media, tmp_path):
    colon = tmp_path / "take:2.wav"
    colon.write_bytes(media["mono"].read_bytes())
    assert audio_track(colon) == audio_track(media["mono"])
    out = tmp_path / "out"
    out.mkdir()  # a directory: the WAV is named after the input
    assert audio_track(colon, output=out) == out / "take:2.wav"
    with pytest.raises(MediaNotFoundError):
        audio_track(tmp_path / "take:9.mp4")


def test_a_missing_file_is_media_not_found(tmp_path):
    with pytest.raises(MediaNotFoundError):
        audio_track(tmp_path / "missing.mp4")


@pytest.mark.skipif(
    sys.platform == "win32", reason="a shell-script stand-in for ffmpeg"
)
def test_a_stalled_ffmpeg_is_killed_at_the_timeout(tmp_path):
    stalled = tmp_path / "ffmpeg"
    stalled.write_text("#!/bin/sh\nsleep 30\n")
    stalled.chmod(0o755)
    with pytest.raises(FfmpegTimeout):
        audio_track(b"\x00" * 64, ffmpeg=str(stalled), timeout=0.5)


@pytest.mark.parametrize("timeout", [0, -1, "5"])
def test_a_bad_timeout_is_refused_before_anything_runs(tmp_path, timeout):
    ran = tmp_path / "ran"
    recorder = tmp_path / "ffmpeg"
    recorder.write_text(f"#!/bin/sh\ntouch {ran}\n")
    recorder.chmod(0o755)
    with pytest.raises(ValueError, match="timeout"):
        audio_track(b"x", ffmpeg=str(recorder), timeout=timeout)
    assert not ran.exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="a shell-script stand-in for ffmpeg"
)
def test_an_ffmpeg_that_writes_nothing_is_a_runtime_error(tmp_path):
    silent = tmp_path / "ffmpeg"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    with pytest.raises(RuntimeError, match="wrote no WAV") as caught:
        audio_track(b"x", ffmpeg=str(silent))
    assert not isinstance(caught.value, FileNotFoundError)


def test_bytearray_and_memoryview_are_bytes_too(media):
    raw = media["mono"].read_bytes()
    assert (
        audio_track(bytearray(raw)) == audio_track(memoryview(raw)) == audio_track(raw)
    )


def test_import_mixing_demux_loads_no_media_library():
    code = (
        "import json, sys, mixing.demux; "
        "print(json.dumps([n for n in ('numpy', 'moviepy', 'cv2', 'PIL', 'pydub', 'scipy')"
        " if n in sys.modules]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip().splitlines()[-1] == "[]"
