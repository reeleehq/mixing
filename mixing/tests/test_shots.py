"""Shot-boundary detection: pure stages plus an ffmpeg-rendered video."""

import subprocess
import sys

import numpy as np
import pytest

from mixing.video.shot_boundaries import (
    probe_duration,
    adaptive_threshold,
    contact_sheet,
    frame_changes,
    shots,
    spans_from_changes,
)


def test_no_cut_is_one_span():
    assert spans_from_changes(np.array([0.5, 0.4]), fps=2, threshold=8) == [(0.0, 1.5)]


def test_cut_boundaries_tile_the_duration():
    spans = spans_from_changes(np.array([0.0, 50.0, 0.0, 60.0]), fps=2, threshold=8)
    assert spans == [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)]


def test_adaptive_threshold_floor_and_noise():
    assert adaptive_threshold(np.array([])) == 8.0
    assert adaptive_threshold(np.zeros(10)) == 8.0
    noisy = np.array([20.0, 22.0, 21.0, 19.0, 20.0])
    assert adaptive_threshold(noisy) > 22.0


def test_frame_changes():
    frames = np.stack([np.zeros((2, 2)), np.full((2, 2), 10), np.full((2, 2), 10)])
    assert list(frame_changes(frames)) == [10.0, 0.0]


def test_sampler_seam_and_empty():
    assert shots("x", sampler=lambda v, fps: np.zeros((0, 4, 4), np.uint8)) == []
    frames = np.zeros((6, 4, 4), np.uint8)
    frames[3:] = 200
    assert shots("x", fps=1, sampler=lambda v, fps: frames) == [(0.0, 3.0), (3.0, 6.0)]


def test_real_video_cuts(tmp_path):
    import subprocess

    from mixing.util import ffmpeg_exe

    out = tmp_path / "ab.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=red:s=160x90:d=2:r=10",
         "-f", "lavfi", "-i", "color=blue:s=160x90:d=2:r=10",
         "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]", str(out)],
        check=True,
    )  # fmt: skip
    spans = shots(out)
    assert len(spans) == 2
    assert spans[0][1] == pytest.approx(2.0, abs=0.5)
    sheet = contact_sheet(out, spans)
    assert sheet.size[0] == 2 * 240  # two thumbnails, two columns
    assert contact_sheet(out, spans, output=tmp_path).exists()


def test_dense_cuts_are_not_swallowed_by_the_adaptive_threshold():
    changes = np.array([50.0] * 7 + [0.5] * 3)
    assert adaptive_threshold(changes) < 50.0
    assert (
        len(spans_from_changes(changes, fps=2, threshold=adaptive_threshold(changes)))
        == 8
    )


def test_last_span_ends_at_duration():
    spans = spans_from_changes(np.array([0.0, 50.0]), fps=1, threshold=8, duration=2.7)
    assert spans == [(0.0, 2.0), (2.0, 2.7)]
    assert spans_from_changes(np.array([]), fps=1, threshold=8, duration=0.12) == [
        (0.0, 0.12)
    ]


def test_bad_fps():
    with pytest.raises(ValueError):
        shots("x", fps=0)


def test_real_video_duration_and_dense_cuts(tmp_path):
    import subprocess

    from mixing.util import ffmpeg_exe

    out = tmp_path / "pages.mp4"
    colors = ["red", "blue", "green", "white", "black", "yellow"]
    inputs = [
        a
        for c in colors
        for a in ("-f", "lavfi", "-i", f"color={c}:s=160x90:d=0.5:r=10")
    ]
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", *inputs, "-filter_complex",
         f"concat=n={len(colors)}:v=1[v]", "-map", "[v]", str(out)],
        check=True,
    )  # fmt: skip
    assert probe_duration(out) == pytest.approx(3.0, abs=0.1)
    spans = shots(out, fps=4)
    assert len(spans) == len(colors)
    assert spans[-1][1] == pytest.approx(3.0, abs=0.1)
    sheet = contact_sheet(out, spans, columns=3)
    assert sheet.size[0] == 3 * 240 and sheet.size[1] % 2 == 0


def _two_colour_video(path, *, seconds=2, fps=10):
    import subprocess

    from mixing.util import ffmpeg_exe

    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", f"color=red:s=160x90:d={seconds}:r={fps}",
         "-f", "lavfi", "-i", f"color=blue:s=160x90:d={seconds}:r={fps}",
         "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]", str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_merge_short_spans_keeps_the_tiling():
    from mixing.video.shot_boundaries import merge_short_spans

    spans = [(0.0, 0.2), (0.2, 3.0), (3.0, 3.3), (3.3, 6.0)]
    assert merge_short_spans(spans, min_s=1.0) == [(0.0, 3.3), (3.3, 6.0)]
    assert merge_short_spans([(0.0, 0.5)], min_s=1.0) == [(0.0, 0.5)]
    assert merge_short_spans([], min_s=1.0) == []


def test_strategy_seam_and_min_shot_s():
    fake = lambda video, **kw: [(0.0, 2.0), (2.0, 2.3), (2.3, 5.0)]  # noqa: E731
    assert shots("x", strategy=fake) == fake("x")
    assert shots("x", strategy=fake, min_shot_s=0.5) == [(0.0, 2.3), (2.3, 5.0)]
    with pytest.raises(ValueError, match="unknown shot strategy"):
        shots("x", strategy="nope")


def test_chunked_sampler_matches_one_array():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 30, (40, 4, 4)).astype(np.uint8)
    frames[17:] += 150
    whole = shots("x", fps=2, sampler=lambda v, fps: frames)
    chunked = shots(
        "x", fps=2, sampler=lambda v, fps: (frames[i : i + 3] for i in range(0, 40, 3))
    )
    assert chunked == whole == [(0.0, 8.5), (8.5, 20.0)]


def test_streamed_frames_are_whole_and_in_order(tmp_path):
    from mixing.video.shot_boundaries import iter_frames

    video = _two_colour_video(tmp_path / "ab.mp4")  # 2 s red, 2 s blue
    chunks = list(iter_frames(video, fps=5, chunk_frames=3))
    assert max(len(c) for c in chunks) == 3
    frames = np.concatenate(chunks)
    assert len(frames) == 20
    levels = frames.mean(axis=(1, 2))
    red, blue = levels[:10], levels[10:]  # gray levels: red ~76, blue ~29
    assert red.min() > 60 and blue.max() < 45


def test_unreadable_video_raises(tmp_path):
    with pytest.raises(RuntimeError, match="could not read"):
        shots(tmp_path / "missing.mp4")


@pytest.mark.skipif(
    sys.platform == "win32", reason="the fake ffmpeg is a shebang script"
)
def test_ffmpeg_error_flood_does_not_hang(tmp_path, monkeypatch):
    # a damaged file can make ffmpeg write far more than a pipe holds (~64 KB)
    # to stderr; read only after stdout ended, that pipe blocked it forever
    import mixing.video.shot_boundaries as sb

    fake = tmp_path / "ffmpeg"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.write('x' * 1_000_000)\n"
        "sys.stdout.buffer.write(bytes(64 * 36 * 4))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(sb, "ffmpeg_exe", lambda: str(fake))
    frames = list(sb.iter_frames("any.mp4"))
    assert sum(len(c) for c in frames) == 4


def test_a_sampler_may_yield_single_frames():
    frames = np.zeros((6, 4, 4), np.uint8)
    frames[3:] = 200
    assert shots("x", fps=1, sampler=lambda v, fps: iter(frames)) == [
        (0.0, 3.0),
        (3.0, 6.0),
    ]


def test_clock_rounds_before_splitting_minutes():
    from mixing.video.shot_boundaries import _clock

    assert _clock(65.5) == "1:05.5"
    assert _clock(59.96) == "1:00.0"


def test_pyscenedetect_strategy_maps_scenes_to_spans(monkeypatch):
    import sys
    import types

    class Timecode:
        def __init__(self, s):
            self.s = s

        def get_seconds(self):
            return self.s

    seen = {}
    stub = types.ModuleType("scenedetect")
    stub.ContentDetector = lambda **kw: seen.setdefault("kwargs", kw)
    stub.detect = lambda path, detector: [
        (Timecode(0.0), Timecode(2.0)),
        (Timecode(2.0), Timecode(4.0)),
    ]
    monkeypatch.setitem(sys.modules, "scenedetect", stub)
    assert shots("v.mp4", strategy="pyscenedetect", threshold=20.0) == [
        (0.0, 2.0),
        (2.0, 4.0),
    ]
    assert seen["kwargs"] == {"threshold": 20.0}
    # no cut: one shot over the probed duration
    stub.detect = lambda path, detector: []
    monkeypatch.setattr("mixing.video.shot_boundaries.probe_duration", lambda v: 7.5)
    assert shots("v.mp4", strategy="pyscenedetect") == [(0.0, 7.5)]


def test_contact_sheet_labels_each_thumbnail(tmp_path):
    video = _two_colour_video(tmp_path / "ab.mp4")
    spans = [(0.0, 2.0), (2.0, 4.0)]
    plain = contact_sheet(video, spans, labels=False)
    labelled = contact_sheet(video, spans)
    corner = (2, plain.height - 2)  # inside the first thumbnail's label box
    assert plain.getpixel(corner)[0] > 200  # red frame
    assert labelled.getpixel(corner) == (0, 0, 0)


def test_grab_frame_scales_and_raises(tmp_path):
    from mixing.video import grab_frame

    video = _two_colour_video(tmp_path / "ab.mp4")
    assert grab_frame(video, 1.0).size == (160, 90)
    assert grab_frame(video, 3.0, width=80).size == (80, 46)  # height kept even
    with pytest.raises(RuntimeError, match="could not grab"):
        grab_frame(tmp_path / "missing.mp4", 0.0)


# --- grab_frames: one seek per frame, or one pass (reeleehq/mixing#94) ----------


def _counter_video(path, seconds=6, rate=10):
    """A video whose every frame differs (testsrc draws a running counter)."""
    from mixing.util import ffmpeg_exe

    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc=s=160x90:r={rate}:d={seconds}", "-c:v", "libx264", "-g", "25",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_one_pass_gives_the_frames_the_seeks_give(tmp_path):
    from mixing.video.thumbnail import grab_frames

    video = _counter_video(tmp_path / "count.mp4")
    times = [5.05, 0.0, 0.33, 1.0, 2.71, 4.2]  # any order, between and on frames
    seeked = grab_frames(video, times, width=80, method="seek")
    passed = grab_frames(video, times, width=80, method="pass")
    assert [np.asarray(a).tolist() for a in passed] == [
        np.asarray(b).tolist() for b in seeked
    ]
    assert len({np.asarray(a).tobytes() for a in passed}) == len(times)  # distinct


def test_one_pass_refuses_two_times_in_one_frame_and_auto_falls_back(tmp_path):
    from mixing.video.thumbnail import grab_frames

    video = _counter_video(tmp_path / "count.mp4")
    close = [1.01, 1.02] + [0.1 * k for k in range(30, 50)]  # dense, two in one frame
    with pytest.raises(RuntimeError, match="method='seek'"):
        grab_frames(video, close, method="pass")
    assert len(grab_frames(video, close, method="auto", duration_s=6)) == len(close)


def test_auto_picks_by_frames_per_minute(tmp_path, monkeypatch):
    import mixing.video.thumbnail as thumb

    used = []
    real_pass = thumb._frames_in_one_pass
    monkeypatch.setattr(
        thumb, "grab_frame", lambda v, t, width=None: used.append("seek")
    )
    monkeypatch.setattr(
        thumb,
        "_frames_in_one_pass",
        lambda v, ts, w: used.append("pass") or real_pass(v, ts, w),
    )
    video = _counter_video(tmp_path / "count.mp4")
    thumb.grab_frames(video, [30.0], duration_s=60)  # 1 per minute: seek
    assert used == ["seek"]
    used.clear()
    dense = [0.5 * k for k in range(1, 10)]  # 90 per minute
    thumb.grab_frames(video, dense, duration_s=6)  # unscaled: held in memory, seek
    assert used == ["seek"] * len(dense)
    used.clear()
    thumb.grab_frames(video, dense, width=40, duration_s=6)
    assert used == ["pass"]
    with pytest.raises(ValueError, match="unknown method"):
        thumb.grab_frames(video, [1.0], method="fast")
    assert thumb.grab_frames(video, []) == []


def test_contact_sheet_is_the_same_either_way(tmp_path):
    video = _counter_video(tmp_path / "count.mp4")
    spans = [(0.0, 1.0), (1.0, 2.5), (2.5, 6.0)]
    a = contact_sheet(video, spans, method="seek")
    b = contact_sheet(video, spans, method="pass")
    assert a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))


def test_one_pass_is_on_the_files_clock_when_the_video_starts_late(tmp_path):
    """Sound from 0 s, picture from 0.5 s: both methods count from the file."""
    from mixing.util import ffmpeg_exe
    from mixing.video.thumbnail import grab_frames

    late = tmp_path / "late.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-itsoffset", "0.5", "-f", "lavfi", "-i",
         "testsrc=s=160x90:r=10:d=4", "-f", "lavfi", "-i", "sine=d=4.5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(late)],
        check=True,
    )  # fmt: skip
    times = [1.0, 2.05, 3.3]
    seeked = grab_frames(late, times, width=80, method="seek")
    passed = grab_frames(late, times, width=80, method="pass")
    assert [np.asarray(a).tolist() for a in passed] == [
        np.asarray(b).tolist() for b in seeked
    ]


def test_a_time_on_a_frame_picks_that_frame_not_the_next(tmp_path):
    from mixing.util import ffmpeg_exe
    from mixing.video.thumbnail import grab_frames

    ntsc = tmp_path / "ntsc.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=s=160x90:r=30000/1001:d=2", "-c:v", "libx264", "-pix_fmt",
         "yuv420p", str(ntsc)],
        check=True,
    )  # fmt: skip
    on = [k * 1001 / 30000 for k in range(3, 40, 4)]
    just_before = [t - 0.3 * 1001 / 30000 for t in on]  # first frame at/after: same
    a = grab_frames(ntsc, on, width=80, method="pass")
    b = grab_frames(ntsc, just_before, width=80, method="pass")
    assert [np.asarray(x).tolist() for x in a] == [np.asarray(x).tolist() for x in b]


def test_a_truncated_frame_stream_is_a_runtime_error():
    import io

    from PIL import Image

    from mixing.video.thumbnail import _split_pngs

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    png = buffer.getvalue()
    assert _split_pngs(png + png) == [png, png]
    with pytest.raises(RuntimeError, match="truncated"):
        _split_pngs(png + png[:-5])


# --- diff_shots(duration=) (reeleehq/mixing#95) ----------------------------------


def test_a_wrapped_sampler_still_gets_the_probed_duration(monkeypatch):
    import functools

    import mixing.video.shot_boundaries as sb

    monkeypatch.setattr(sb, "probe_duration", lambda v: 6.4)
    frames = np.zeros((6, 4, 4), np.uint8)
    wrapped = functools.partial(lambda v, fps, size: frames, size=(8, 8))
    # a sampler that is not iter_frames itself: the probe still runs for a path
    assert sb.diff_shots("v.mp4", fps=1, sampler=wrapped) == [(0.0, 6.4)]


def test_duration_given_is_used_and_zero_keeps_the_sampled_length(monkeypatch):
    import mixing.video.shot_boundaries as sb

    probed = []
    monkeypatch.setattr(sb, "probe_duration", lambda v: probed.append(v) or 6.4)
    frames = np.zeros((6, 4, 4), np.uint8)
    sampler = lambda v, fps: frames  # noqa: E731
    assert sb.diff_shots("v.mp4", fps=1, sampler=sampler, duration=9.0) == [(0.0, 9.0)]
    zero = sb.diff_shots("v.mp4", fps=1, sampler=sampler, duration=sb.SAMPLED_DURATION)
    assert zero == [(0.0, 6.0)] and probed == []  # 6 frames at 1 fps, no probe
    assert sb.diff_shots(frames, fps=1, sampler=lambda v, fps: v) == [(0.0, 6.0)]
    with pytest.raises(ValueError, match="duration"):
        sb.diff_shots("v.mp4", fps=1, sampler=sampler, duration=-1)
