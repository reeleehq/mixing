"""``level_file`` (#130): one measured gain and a peak limiter, file to file, streaming.

Also the pieces it is built from: :func:`mixing.loudness.stream_limit_peaks`
(the limiter over chunks, equal to :func:`~mixing.loudness.limit_peaks` on the
joined signal) and :func:`mixing.decode.encode_float_chunks` (chunks into one
ffmpeg, written atomically).
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

import mixing.loudness as loudness  # noqa: E402
from mixing.loudness import (  # noqa: E402
    LEVELED_SUFFIX,
    file_loudness,
    level_file,
    limit_peaks,
    stream_limit_peaks,
    true_peak_db,
)
from mixing.util import has_ffmpeg  # noqa: E402

FS = 44100
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not on PATH")


def _pieces(data, sizes):
    """``data`` cut into consecutive chunks of the given sizes, cycled."""
    out, i, k = [], 0, 0
    while i < len(data):
        out.append(data[i : i + sizes[k % len(sizes)]])
        i, k = i + sizes[k % len(sizes)], k + 1
    return out


@pytest.mark.parametrize("true_peak", [False, True])
@pytest.mark.parametrize("sizes", [(1,), (777, 5, 0, 3001), (50_000,)])
def test_stream_limiter_equals_the_array_limiter(monkeypatch, true_peak, sizes):
    rng = np.random.default_rng(4)
    data = rng.standard_normal((20_000, 2)) * 0.4
    data[4999] = 2.0  # a peak at a block join
    kwargs = dict(ceiling=0.6, window=40, true_peak=true_peak, fs=FS)
    whole = limit_peaks(data, **kwargs)
    monkeypatch.setattr(loudness, "LIMITER_CHUNK_FRAMES", 5000)
    blocks = list(stream_limit_peaks(_pieces(data, sizes), **kwargs))
    assert [len(b) for b in blocks] == [5000] * 4
    np.testing.assert_allclose(np.concatenate(blocks), whole, rtol=0, atol=1e-12)


def test_stream_limiter_checks_settings_at_once_and_channels_as_it_goes():
    with pytest.raises(ValueError, match="not dB"):
        stream_limit_peaks(iter(()), ceiling=-1.0)  # raised before any chunk
    with pytest.raises(ValueError, match="needs fs"):
        stream_limit_peaks(iter(()), ceiling=0.5, true_peak=True)
    with pytest.raises(ValueError, match="integer >= 1"):
        stream_limit_peaks(iter(()), ceiling=0.5, true_peak=True, fs=FS, oversampling=0)
    with pytest.raises(ValueError, match="channels"):
        list(stream_limit_peaks([np.zeros((5, 2)), np.zeros((5, 1))], ceiling=0.5))
    assert list(stream_limit_peaks([], ceiling=0.5)) == []
    mono = list(stream_limit_peaks([np.array([2.0, 0.5])], ceiling=1.0))
    assert mono[0].shape == (2, 1) and mono[0].max() <= 1.0


def test_stream_limiter_memory_does_not_grow_with_the_track(monkeypatch):
    import tracemalloc

    monkeypatch.setattr(loudness, "LIMITER_CHUNK_FRAMES", FS)

    def peak_bytes(seconds):
        rng = np.random.default_rng(0)
        chunks = (rng.standard_normal((FS // 2, 2)) for _ in range(2 * seconds))
        tracemalloc.start()
        for _ in stream_limit_peaks(chunks, ceiling=0.5, fs=FS, true_peak=True):
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    short, long = peak_bytes(5), peak_bytes(60)
    assert long < short * 1.2  # a block and its context, not the track


# --- level_file -------------------------------------------------------------------


def _lavfi(path, source, *, seconds, rate=FS, channels=2, extra=()):
    from mixing.util import ffmpeg_exe

    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", source, "-t", str(seconds),
         "-ar", str(rate), "-ac", str(channels), *extra, str(path)],
        check=True,
    )  # fmt: skip
    return path


def _read(path):
    from mixing.decode import decode_float_chunks

    rate, chunks = decode_float_chunks(path)
    return np.concatenate(list(chunks)).astype(np.float64), rate


def _speechy(tmp_path, name="voice.wav", *, volume_db=-30, seconds=8, rate=FS):
    """Harmonics with a slow tremolo and a burst: dynamics a gain must keep."""
    source = (
        f"aevalsrc='(0.5+0.4*sin(2*PI*3*t))*(sin(2*PI*220*t)+0.5*sin(2*PI*660*t))"
        f"*(1+2*between(t,3,3.5))':s={rate},volume={volume_db}dB"
    )
    return _lavfi(tmp_path / name, source, seconds=seconds, rate=rate)


@needs_ffmpeg
def test_level_file_reaches_the_target_with_one_gain(tmp_path):
    src = _speechy(tmp_path, volume_db=-35)
    result = level_file(src, tmp_path / "out.wav", target_lufs=-20.0)
    assert result.output == tmp_path / "out.wav"
    assert result.gain_db == pytest.approx(-20.0 - result.measured_lufs)
    assert file_loudness(result.output).integrated_lufs == pytest.approx(-20.0, abs=0.1)
    assert result.output_lufs == pytest.approx(-20.0, abs=0.1)
    # one gain: every sample scaled alike (the limiter had nothing to do here)
    before, _ = _read(src)
    after, _ = _read(result.output)
    ratio = after[1000:-1000] / np.where(before == 0, np.nan, before)[1000:-1000]
    loud = np.abs(before[1000:-1000]) > 1e-3
    np.testing.assert_allclose(ratio[loud], 10 ** (result.gain_db / 20), rtol=2e-3)


@needs_ffmpeg
def test_a_big_positive_gain_is_limited_not_clipped(tmp_path):
    # a quiet signal with a full-scale-to-be burst: the gain alone would clip
    src = _speechy(tmp_path, volume_db=-20)
    result = level_file(src, tmp_path / "out.wav", target_lufs=-9.0, ceiling_db=-1.5)
    samples, rate = _read(result.output)
    assert true_peak_db(samples, rate) <= -1.5 + 0.1
    assert np.abs(samples).max() < 10 ** (-1.4 / 20)  # nothing at full scale
    # the limiter took loudness off, and the result says so
    assert result.output_lufs < -9.0 - 0.5
    assert result.output_lufs == pytest.approx(file_loudness(result.output).integrated_lufs)


@needs_ffmpeg
def test_sample_rate_is_applied_before_the_limiter(tmp_path):
    src = _speechy(tmp_path, volume_db=-20, rate=48000)
    out = level_file(
        src, tmp_path / "out.wav", target_lufs=-10.0, ceiling_db=-2.0, sample_rate=22050
    ).output
    samples, rate = _read(out)
    assert rate == 22050
    assert true_peak_db(samples, rate) <= -2.0 + 0.1


@needs_ffmpeg
def test_sample_peak_mode_and_encode_args(tmp_path):
    src = _speechy(tmp_path, volume_db=-20)
    out = level_file(
        src,
        tmp_path / "out.wav",
        target_lufs=-9.0,
        ceiling_db=-3.0,
        true_peak=False,
        encode_args=("-c:a", "pcm_s24le"),
    ).output
    samples, _ = _read(out)
    assert 20 * np.log10(np.abs(samples).max()) <= -3.0 + 1e-3
    from mixing.util import ffmpeg_exe

    probe = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(out)], capture_output=True, text=True
    )
    assert "pcm_s24le" in probe.stderr


@needs_ffmpeg
def test_output_follows_the_egress_protocol(tmp_path):
    src = _speechy(tmp_path, seconds=3)
    beside = level_file(src, target_lufs=-20.0).output
    assert beside == tmp_path / f"voice{LEVELED_SUFFIX}.wav" and beside.exists()
    folder = tmp_path / "out"
    folder.mkdir()
    assert level_file(src, folder, target_lufs=-20.0).output == folder / beside.name
    sunk = level_file(src, lambda path: ("sunk", path.name), target_lufs=-20.0)
    assert sunk.output == ("sunk", beside.name)


@needs_ffmpeg
def test_level_file_may_overwrite_its_source(tmp_path):
    src = _speechy(tmp_path, volume_db=-35)
    result = level_file(src, src, target_lufs=-20.0)
    assert result.output == src
    assert file_loudness(src).integrated_lufs == pytest.approx(-20.0, abs=0.1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["voice.wav"]


@needs_ffmpeg
def test_silence_gets_no_gain_and_a_far_target_is_capped(tmp_path):
    silent = _lavfi(tmp_path / "silent.wav", f"anullsrc=r={FS}:cl=mono", seconds=2)
    result = level_file(silent, tmp_path / "out.wav", target_lufs=-20.0)
    assert result.gain_db == 0.0 and result.measured_lufs == -np.inf
    assert result.output.exists()
    quiet = _speechy(tmp_path, volume_db=-70)
    with pytest.warns(UserWarning, match="quiet.wav|voice.wav") as caught:
        capped = level_file(quiet, tmp_path / "q.wav", target_lufs=-10.0, max_gain_db=6)
    assert capped.gain_db == 6.0
    assert caught[0].filename == __file__  # the warning points at the caller


@needs_ffmpeg
def test_bad_settings_fail_before_any_decoding(tmp_path, monkeypatch):
    def no_decode(*a, **k):
        raise AssertionError("decoded before checking the settings")

    monkeypatch.setattr(loudness, "decode_float_chunks", no_decode)
    for bad in (
        dict(ceiling_db=0.5),
        dict(ceiling_db=-np.inf),
        dict(ceiling_db=np.nan),
        dict(target_lufs=np.nan),
        dict(method="long", true_peak=False),
        dict(oversampling=0),
        dict(window_s=0),
        dict(window_s=np.inf),
    ):
        with pytest.raises(ValueError):
            level_file(tmp_path / "x.wav", **{"target_lufs": -20.0, **bad})
    with pytest.raises(ValueError, match="finite"):
        loudness.level_group([np.ones((FS, 1)) * 0.1], FS, target_lufs=np.nan)


@needs_ffmpeg
def test_in_place_refuses_a_file_with_a_picture(tmp_path):
    from mixing.util import ffmpeg_exe

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x36:d=2:r=10",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration=2:sample_rate={FS}",
         "-shortest", "-c:a", "aac", str(clip)],
        check=True,
    )  # fmt: skip
    before = clip.read_bytes()
    with pytest.raises(ValueError, match="video"):
        level_file(clip, clip, target_lufs=-20.0)
    hard = tmp_path / "same.mp4"  # one file under another name (as on a
    hard.hardlink_to(clip)  # case-insensitive disk): still the source
    with pytest.raises(ValueError, match="video"):
        level_file(clip, hard, target_lufs=-20.0)
    assert clip.read_bytes() == before
    beside = level_file(clip, target_lufs=-20.0).output  # elsewhere: audio only
    assert beside.name == f"clip{LEVELED_SUFFIX}.mp4"
    assert loudness._other_streams(beside, None) == []


@needs_ffmpeg
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and symlinks")
def test_in_place_keeps_permissions_and_symlinks(tmp_path):
    import os
    import stat

    src = _speechy(tmp_path, seconds=2)
    src.chmod(0o640)
    link = tmp_path / "link.wav"
    link.symlink_to(src.name)
    level_file(link, link, target_lufs=-20.0)
    assert link.is_symlink() and stat.S_IMODE(os.stat(src).st_mode) == 0o640
    assert file_loudness(src).integrated_lufs == pytest.approx(-20.0, abs=0.1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.wav", "voice.wav"]


@needs_ffmpeg
def test_a_failed_encode_leaves_no_file_behind(tmp_path):
    from mixing.decode import encode_float_chunks

    with pytest.raises(RuntimeError, match="could not write"):
        encode_float_chunks([np.zeros((100, 1))], FS, tmp_path / "x.wav", args=("-c:a", "nope"))
    with pytest.raises(ValueError, match="no audio"):
        encode_float_chunks([np.zeros((0, 2))], FS, tmp_path / "y.wav")

    def broken():
        yield np.zeros((100, 2))
        raise KeyError("upstream")

    with pytest.raises(KeyError):
        encode_float_chunks(broken(), FS, tmp_path / "z.wav")
    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
@pytest.mark.skipif(sys.platform == "win32", reason="a shell-script ffmpeg")
@pytest.mark.parametrize("frames", [1, 100, FS])
def test_an_ffmpeg_that_dies_midway_raises_its_message_and_leaves_no_file(tmp_path, frames):
    # it stops reading with data still buffered: closing stdin used to raise
    # BrokenPipeError past the cleanup, leaving the partial file behind
    from mixing.decode import encode_float_chunks

    fake = tmp_path / "fake-ffmpeg"
    fake.write_text(
        '#!/bin/sh\nhead -c 300000 > /dev/null\nfor last; do :; done\n'
        'echo partial > "$last"\necho "fake ffmpeg gave up" >&2\nexit 1\n'
    )
    fake.chmod(0o755)
    out = tmp_path / "out" / "x.wav"
    out.parent.mkdir()
    chunks = (np.zeros((frames, 2)) for _ in range(max(1, FS * 4 // frames)))
    with pytest.raises(RuntimeError, match="fake ffmpeg gave up"):
        encode_float_chunks(chunks, FS, out, ffmpeg=str(fake))
    assert list(out.parent.iterdir()) == []


@needs_ffmpeg
def test_level_file_memory_does_not_grow_with_the_file(tmp_path):
    import tracemalloc

    def peak_bytes(seconds):
        src = _lavfi(
            tmp_path / f"n{seconds}.wav", f"anoisesrc=r={FS}:a=0.1:seed=1", seconds=seconds
        )
        tracemalloc.start()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            level_file(src, tmp_path / f"o{seconds}.wav", target_lufs=-14.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    short, long = peak_bytes(20), peak_bytes(180)
    assert long < short * 1.2  # a 3 min stereo float track alone is ~127 MB
