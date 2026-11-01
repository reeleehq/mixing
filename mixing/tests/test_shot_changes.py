"""``mixing.shot_changes``: the shot detector's array stages, light to import,
and the opt-in ``"spike"`` rule for continuous footage (reeleehq/mixing#160)."""

import subprocess
import sys

import numpy as np
import pytest

from mixing.shot_changes import (
    cut_indices,
    frame_changes,
    shots_from_changes,
    spans_from_changes,
    streamed_changes,
)


def _scene(seed, n, size=(36, 64)):
    """``n`` frames of one textured scene (noise, so every shift changes pixels)."""
    return np.repeat(
        np.random.default_rng(seed).integers(0, 256, size, dtype=np.uint8)[None], n, 0
    )


def _pan(n, step=3, size=(36, 64)):
    """A textured scene panning ``step`` px a frame: steady motion, no cut."""
    wide = np.random.default_rng(9).integers(0, 256, (size[0], size[1] + n * step))
    return np.stack([wide[:, k * step : k * step + size[1]] for k in range(n)]).astype(
        np.uint8
    )


def test_importing_it_loads_no_video_stack():
    code = (
        "import sys, mixing.shot_changes;"
        "print([m for m in ('moviepy', 'cv2', 'PIL', 'looks', 'burns', 'pydub', "
        "'mixing.video') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stderr


def test_the_video_module_reexports_the_same_objects():
    import mixing.shot_changes as light
    import mixing.video.shot_boundaries as sb

    for name in ("frame_changes", "adaptive_threshold", "spans_from_changes",
                 "merge_short_spans", "SHOT_FRAME_SIZE", "MAX_SHOT_THRESHOLD"):  # fmt: skip
        assert getattr(sb, name) is getattr(light, name)


def test_a_hard_cut_is_found_by_both_rules():
    frames = np.concatenate([_scene(1, 6), _scene(2, 6)])
    changes = frame_changes(frames)
    for rule in ("level", "spike"):
        assert shots_from_changes(changes, fps=2, rule=rule) == [(0.0, 3.0), (3.0, 6.0)]


def test_a_pan_is_a_run_of_cuts_by_level_and_none_by_spike():
    changes = frame_changes(_pan(12))
    assert len(cut_indices(changes, threshold=30.0)) > 3  # motion reads as cuts
    assert len(cut_indices(changes, threshold=30.0, rule="spike")) == 0


def _smooth_pan(n, size=(36, 64)):
    """A smooth scene (handheld-like) drifting a pixel a frame."""
    y, x = np.mgrid[0 : size[0], 0 : size[1] + n]
    wide = 128 + 90 * np.sin(x / 6.0) * np.cos(y / 5.0)
    return np.stack([wide[:, k : k + size[1]] for k in range(n)]).astype(np.uint8)


def test_a_cut_inside_motion_is_still_found_by_spike():
    frames = np.concatenate([_smooth_pan(8), _scene(5, 8)])
    changes = frame_changes(frames)
    cuts = cut_indices(changes, threshold=30.0, rule="spike")
    assert list(cuts) == [7]  # between frame 7 (last of the pan) and 8


def test_a_flash_is_not_a_cut_under_spike():
    frames = _scene(3, 10).copy()
    frames[5] = 255  # one white frame: it changes in, and back
    changes = frame_changes(frames)
    assert len(cut_indices(changes, threshold=20.0)) == 2  # level: two cuts
    assert len(cut_indices(changes, threshold=20.0, rule="spike")) == 0


def test_spike_ratio_is_what_separates_cuts_from_motion():
    changes = np.array([10.0, 35.0, 10.0, 35.0, 30.0, 35.0])
    assert list(cut_indices(changes, threshold=20.0, rule="spike")) == [1]
    # a lower ratio admits the motion bumps at 3 and 5 (35 against 30)
    loose = cut_indices(changes, threshold=20.0, rule="spike", spike_ratio=1.1)
    assert list(loose) == [1, 3, 5]


@pytest.mark.parametrize("sizes", [(1,), (2, 3), (5, 1, 0, 1, 6), (13,)])
def test_streamed_changes_equal_the_whole(sizes):
    frames = np.concatenate([_pan(7), _scene(4, 6)])
    pieces, at = [], 0
    for k in sizes:
        pieces.append(frames[at : at + k] if k != 1 else frames[at])
        at += k
    pieces.append(frames[at:])
    changes, count = streamed_changes(pieces)
    assert count == len(frames)
    np.testing.assert_array_equal(changes, frame_changes(frames))


def test_frames_at_given_times():
    changes = np.array([1.0, 90.0, 1.0])  # a cut between frames 1 and 2
    times = [0.0, 0.4, 1.3, 1.9]
    # the last frame lasts the last interval (0.6 s), as at a fixed rate
    assert shots_from_changes(changes, times=times, threshold=20) == [
        (0.0, 1.3),
        (1.3, pytest.approx(2.5)),
    ]
    assert shots_from_changes(changes, times=times, threshold=20, duration=3.0)[-1] == (
        1.3,
        3.0,
    )
    # a cut between the last two frames is kept, as with fps=
    last = np.array([1.0, 1.0, 90.0])
    assert shots_from_changes(last, fps=1, threshold=20) == [(0.0, 3.0), (3.0, 4.0)]
    assert shots_from_changes(last, times=[0, 1, 2, 3], threshold=20) == [
        (0.0, 3.0),
        (3.0, 4.0),
    ]
    assert shots_from_changes(np.zeros(0), times=[], threshold=20) == []


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(times=[0.0, 1.0]), "one time per frame"),
        (dict(times=[0.0, 2.0, 1.0, 3.0]), "increase"),
        (dict(times=[0.0, 1.0, 1.0, 3.0]), "increase"),
        (dict(fps=2, duration=-1), "duration"),
        (dict(fps=2, duration=float("nan")), "duration"),
        (dict(), "or their times"),
    ],
)
def test_bad_times_or_durations_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        shots_from_changes(np.array([1.0, 90.0, 1.0]), threshold=20, **kwargs)


def test_level_rule_is_spans_from_changes_and_min_shot_merges():
    changes = np.array([0.0, 50.0, 0.0, 60.0, 0.0, 0.0])
    assert shots_from_changes(changes, fps=2, threshold=8) == spans_from_changes(
        changes, fps=2, threshold=8
    )
    merged = shots_from_changes(changes, fps=2, threshold=8, min_shot_s=1.5)
    assert merged == [(0.0, 2.0), (2.0, 3.5)]
    with pytest.raises(ValueError, match="unknown rule"):
        cut_indices(changes, threshold=8, rule="fancy")
