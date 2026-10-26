"""Shot-boundary detection: pure stages plus an ffmpeg-rendered video."""

import numpy as np
import pytest

from mixing.video.shot_boundaries import (
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
    assert sheet.size[0] > 0
    assert contact_sheet(out, spans, output=tmp_path).exists()
