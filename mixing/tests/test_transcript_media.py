"""``mixing.transcript.media``: audio extraction and keep-cutting through the
shared ffmpeg runner (``ffmpeg=``, ``timeout=``), with real ffmpeg runs."""

import subprocess
import sys

import pytest

from mixing.ffmpeg import FfmpegTimeout
from mixing.probe import probe
from mixing.transcript.media import apply_keeps, extract_audio
from mixing.util import ffmpeg_exe, has_ffmpeg

pytestmark = pytest.mark.skipif(not has_ffmpeg(), reason="needs ffmpeg")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=s=160x90:r=25:d=4", "-f", "lavfi", "-i", "sine=r=48000:d=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2",
         "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_extract_audio_writes_a_mono_16k_mp3(clip, tmp_path):
    out = extract_audio(clip, tmp_path / "speech.mp3")
    found = probe(out, ffprobe=None)
    assert (found.sample_rate, found.channels, found.has_video) == (16000, 1, False)
    assert found.duration == pytest.approx(4.0, abs=0.2)


def test_apply_keeps_keeps_only_the_ranges(clip, tmp_path):
    keeps = [{"start": 0.0, "end": 1.0}, {"start": 2.5, "end": 3.5}]
    out = apply_keeps(clip, tmp_path / "kept.mp4", keeps, preset="ultrafast")
    assert probe(out, ffprobe=None).duration == pytest.approx(2.0, abs=0.2)
    with pytest.raises(ValueError, match="at least one"):
        apply_keeps(clip, tmp_path / "none.mp4", [])


def test_a_failed_run_is_a_runtime_error_and_a_called_process_error(clip, tmp_path):
    import pickle

    from mixing.ffmpeg import FfmpegError

    silent = tmp_path / "silent.mp4"  # a picture and no sound: nothing to extract
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-i", str(clip), "-an", "-c", "copy",
         str(silent)],
        check=True,
    )  # fmt: skip
    with pytest.raises(FfmpegError) as caught:
        extract_audio(silent, tmp_path / "x.mp3")
    error = caught.value
    assert isinstance(error, RuntimeError)
    assert isinstance(error, subprocess.CalledProcessError)  # callers of the old API
    assert error.returncode != 0 and b"does not contain any stream" in error.stderr
    assert "could not extract the audio" in str(error)
    assert "ffmpeg version" not in str(
        error
    )  # -hide_banner: the complaint, not the build
    again = pickle.loads(pickle.dumps(error))
    assert (str(again), again.returncode, again.stderr) == (
        str(error),
        error.returncode,
        error.stderr,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="a shell-script stand-in")
def test_a_stalled_ffmpeg_is_killed_at_the_timeout(clip, tmp_path):
    stalled = tmp_path / "ffmpeg"
    stalled.write_text("#!/bin/sh\nsleep 30\n")
    stalled.chmod(0o755)
    with pytest.raises(FfmpegTimeout):
        extract_audio(clip, tmp_path / "x.mp3", ffmpeg=str(stalled), timeout=0.5)
    with pytest.raises(FfmpegTimeout):
        apply_keeps(
            clip, tmp_path / "y.mp4", [{"start": 0, "end": 1}],
            ffmpeg=str(stalled), timeout=0.5,
        )  # fmt: skip


def test_importing_it_stays_light():
    code = (
        "import sys, mixing.transcript.media;"
        "print([m for m in ('numpy', 'moviepy', 'cv2', 'PIL', 'pydub') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stderr
