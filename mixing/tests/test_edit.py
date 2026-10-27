"""Tests for ``mixing.edit``: the frame-exact timeline (pure) and real tiny renders.

The render tests build their own source with ffmpeg (``lavfi``) rather than the
``conftest`` colour-clip fixture, because they need what that fixture cannot
make: an NTSC rate (``30000/1001``), a picture that encodes its own frame index
(the top-left quadrant's grey is ``4 * N``), distinct quadrant colours for the
punch-in, and a tone to tell silence from sound. Everything is 64x36 and ~3 s.
"""

from __future__ import annotations

import subprocess
import sys
from fractions import Fraction

import pytest

from mixing.edit import (
    Cam,
    Gap,
    Overlay,
    PicturePiece,
    Span,
    Still,
    as_rate,
    probe_source,
    punch_in_box,
    render,
    timeline,
)

NTSC = Fraction(30000, 1001)
SR = 48000
FADE = round(0.006 * SR)


# --- timeline arithmetic (pure) ----------------------------------------------------


def test_ntsc_boundaries_are_integer_and_never_drift():
    # 300 spans of awkward lengths, with gaps between some: the classic drift setup
    segments = []
    t = 0.0
    for k in range(300):
        length = 0.37 + (k % 7) * 0.113
        segments.append(Span(t, t + length, "a"))
        t += length + (0.05 if k % 3 else 0.0)
        if k % 11 == 0:
            segments.append(Gap(0.21, "a"))
    plan = timeline(segments, {"a": Cam()}, fps="30000/1001", sample_rate=SR)
    assert plan.fps == NTSC
    assert all(isinstance(b, int) for b in plan.frame_bounds + plan.sample_bounds)
    for frame, sample in zip(plan.frame_bounds, plan.sample_bounds):
        # one rounding per boundary, from the frame boundary itself
        assert sample == round(frame * Fraction(SR) / NTSC)
        assert abs(sample - frame * Fraction(SR) / NTSC) <= Fraction(1, 2)
    assert sum(p.length for p in plan.sound) == plan.total_samples
    assert [p.start for p in plan.sound] == list(plan.sample_bounds[:-1])
    assert plan.total_samples == round(plan.total_frames * Fraction(SR) / NTSC)


def test_spans_snap_to_source_frames():
    plan = timeline(
        [Span(0.2, 1.0, "a"), Gap(0.3, "a")], {"a": Cam()}, fps=NTSC, sample_rate=SR
    )
    assert plan.frame_bounds == (0, 24, 33)  # 0.2 s -> frame 6, 1.0 s -> 30; 0.3 s -> 9
    assert plan.sound[0].src_start == round(6 * Fraction(SR) / NTSC)
    assert plan.sound[1].src_start is None


def test_contiguous_spans_play_as_one_run_without_fades():
    plan = timeline(
        [Span(0, 1, "a"), Span(1, 2, "b"), Span(3, 4, "b")],
        {"a": Cam(), "b": Cam(zoom=1.5)},
        fps=NTSC,
        sample_rate=SR,
    )
    first, second, third = plan.sound
    assert second.src_start == first.src_start + first.length  # sample-continuous
    assert (first.fade_out, second.fade_in) == (0, 0)  # a picture cut, not a sound join
    assert (second.fade_out, third.fade_in) == (FADE, FADE)
    assert (first.fade_in, third.fade_out) == (0, 0)  # film start / end


def test_gap_fades_the_sound_on_both_sides():
    plan = timeline(
        [Span(0, 1, "a"), Gap(0.5, "a"), Span(1, 2, "a")],
        {"a": Cam()},
        fps=NTSC,
        sample_rate=SR,
    )
    assert plan.sound[0].fade_out == FADE and plan.sound[2].fade_in == FADE


def test_gap_under_cam_holds_the_neighbouring_frame():
    plan = timeline(
        [Span(1, 2, "a"), Gap(0.5, "a"), Gap(0.5, "b"), Span(3, 4, "b")],
        {"a": Cam(), "b": Cam()},
        fps=30,
        sample_rate=SR,
    )
    a, b = plan.picture
    # trailing gap: hold the previous span's last frame
    assert a.pieces == (PicturePiece(30, 30), PicturePiece(59, 15, hold=True))
    # leading gap: hold the next span's first frame
    assert b.pieces == (PicturePiece(90, 15, hold=True), PicturePiece(90, 30))


def test_consecutive_same_clip_segments_form_one_picture_clip():
    plan = timeline(
        [Span(0, 1, "s"), Span(2, 3, "s"), Span(3, 4, "a"), Span(5, 6, "s")],
        {"s": Still("x.png"), "a": Cam()},
        fps=30,
        sample_rate=SR,
    )
    assert [(c.clip_id, c.start, c.end) for c in plan.picture] == [
        ("s", 0, 60),
        ("a", 60, 90),
        ("s", 90, 120),
    ]


def test_dissolve_renders_half_early_half_late_and_keeps_total():
    plan = timeline(
        [Span(1, 2, "a"), Span(3, 4, "b")],
        {"a": Cam(), "b": Cam(dissolve_in=0.2)},
        fps=30,
        sample_rate=SR,
    )
    a, b = plan.picture
    assert b.dissolve == 6 and b.head == 3 and a.tail == 3
    assert a.pieces == (PicturePiece(30, 33),)  # 3 frames of source handle past 2 s
    assert b.pieces == (PicturePiece(87, 33),)  # 3 frames before 3 s
    assert a.render_frames + b.render_frames - b.dissolve == plan.total_frames == 60


def test_dissolve_handles_hold_where_the_source_runs_out():
    plan = timeline(
        [Span(0, 1, "a"), Span(2, 3, "b")],
        {"a": Cam(dissolve_in=0), "b": Cam(dissolve_in=0.2)},
        fps=30,
        sample_rate=SR,
        source_frames=31,
    )
    a, _ = plan.picture
    assert a.pieces == (PicturePiece(0, 31), PicturePiece(30, 2, hold=True))


@pytest.mark.parametrize(
    "segments, clips, match",
    [
        ([Gap(1, "a")], {"a": Cam()}, "only gaps"),
        ([Span(0, 1, "a")], {"a": Cam(dissolve_in=0.5)}, "cannot dissolve in"),
        ([Span(0, 1, "x"), Span(1, 1.1, "b"), Span(2, 3, "a")],
         {"x": Cam(), "a": Cam(dissolve_in=0.4), "b": Cam(dissolve_in=0.4)}, "too short"),
        ([Span(0, 1, "zz")], {"a": Cam()}, "not in `clips`"),
        ([Span(0, 0.01, "a")], {"a": Cam()}, "shorter than one frame"),
        ([], {"a": Cam()}, "at least one segment"),
    ],
)  # fmt: skip
def test_impossible_edits_are_refused(segments, clips, match):
    with pytest.raises(ValueError, match=match):
        timeline(segments, clips, fps=30, sample_rate=SR)


def test_tuple_segment_forms():
    plan = timeline(
        [(0, 1, "a"), ("gap", 0.5, "a")], {"a": Cam()}, fps=30, sample_rate=SR
    )
    assert plan.frame_bounds == (0, 30, 45)


def test_as_rate_and_punch_in_box():
    assert as_rate("30000/1001") == as_rate(29.97) == NTSC
    assert as_rate(59.94) == Fraction(60000, 1001) and as_rate(24.0) == 24
    assert punch_in_box(64, 36, zoom=2, cx=0.25, cy=0.25) == (0, 0, 32, 18)
    assert punch_in_box(64, 36, zoom=2, cx=1.0, cy=1.0) == (
        32,
        18,
        32,
        18,
    )  # kept inside


def test_importing_mixing_edit_stays_light():
    code = (
        "import sys, mixing, mixing.edit; "
        "print([m for m in ('moviepy', 'cv2', 'pydub', 'PIL', 'numpy', 'burns') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


# --- real renders ------------------------------------------------------------------

W, H = 64, 36
GREY_STEP = 4  # the source's top-left grey is GREY_STEP * frame index (luma, mod 256)
PCM_MKV = ("-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le")


def _ffmpeg():
    from mixing.util import ffmpeg_exe

    try:
        return ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """A 3 s 30000/1001 source (frame-indexed top-left, red/green/blue other
    quadrants, 440 Hz tone), a horizontal-ramp still and a green-corner overlay."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    ffmpeg = _ffmpeg()
    d = tmp_path_factory.mktemp("edit")
    top, left = f"lt(Y,{H // 2})", f"lt(X,{W // 2})"
    grey = f"mod(N*{GREY_STEP},256)"
    geq = (
        f"r='if({top}*{left},{grey},if({top},255,0))':"
        f"g='if({top}*{left},{grey},if({top},0,if({left},255,0)))':"
        f"b='if({top}*{left},{grey},if({top},0,if({left},0,255)))'"
    )
    source = d / "src.mkv"
    subprocess.run(
        [ffmpeg, "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=black:s={W}x{H}:r=30000/1001:d=3,format=rgb24,geq={geq}",
         "-f", "lavfi", "-i", f"sine=f=440:r={SR}:d=3",
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-c:a", "pcm_s16le",
         str(source)],
        check=True,
    )  # fmt: skip
    ramp = np.tile(
        np.linspace(0, 255, 4 * W).astype(np.uint8)[None, :, None], (H, 1, 3)
    )
    still = d / "ramp.png"
    Image.fromarray(ramp).save(still)
    overlay = np.zeros((H, W, 4), np.uint8)
    overlay[H // 2 + 4 :, W // 2 + 8 :] = (0, 255, 0, 255)  # over the blue quadrant
    overlay_path = d / "overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    return {"dir": d, "source": source, "still": still, "overlay": overlay_path}


def _frames(path):
    import numpy as np

    raw = subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(int)


def _samples(path):
    import numpy as np

    raw = subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.int16)


def _ffprobe_frame_count(path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


#: The grey wraps at 256, so a frame index is recovered modulo this.
GREY_PERIOD = 256 // GREY_STEP


def _source_frame(top_left_grey: float) -> float:
    """Recover a source frame index (modulo :data:`GREY_PERIOD`) from its grey."""
    return top_left_grey / GREY_STEP


# The composite edit every render test reads from (frames at 30000/1001):
#   a  [ 0, 33): span 0.2-1.0 s (source frames 6..29), then a 0.3 s gap (holds 29)
#   s  [33, 57): a still panning across a ramp, under TWO sound spans (cut at 45),
#                dissolving in over 6 frames (30..35)
#   p  [57, 72): source 2.4-2.9 s punched in 2x on the top-left quadrant,
#                dissolving in over 6 frames (54..59)
#   overlay over 0.5-0.8 s -> frames [15, 24)
SEGMENTS = [
    Span(0.2, 1.0, "a"),
    Gap(0.3, "a"),
    Span(1.2, 1.6, "s"),
    Span(2.0, 2.4, "s"),
    Span(2.4, 2.9, "p"),
]


@pytest.fixture(scope="module")
def edit(media):
    pytest.importorskip("burns")
    from burns import BurnsPath, Rect

    pan = BurnsPath.from_start_end(
        Rect(0, 0, 0.5, 1), Rect(0.5, 0, 0.5, 1), easing="linear"
    )
    clips = {
        "a": Cam(),
        "s": Still(media["still"], pan, dissolve_in=0.2),
        "p": Cam(zoom=2, cx=0.25, cy=0.25, dissolve_in=0.2),
    }
    info = probe_source(media["source"])
    plan = timeline(
        SEGMENTS,
        clips,
        fps=info.fps,
        sample_rate=info.sample_rate,
        source_frames=info.frames,
    )
    out = render(
        SEGMENTS,
        clips,
        source=media["source"],
        overlays=[Overlay(media["overlay"], 0.5, 0.8)],
        output=media["dir"] / "out.mkv",
        encode_args=PCM_MKV,
    )
    return {"plan": plan, "out": out, "frames": _frames(out), "samples": _samples(out)}


def test_probe_reads_the_exact_ntsc_rate(media):
    info = probe_source(media["source"])
    assert info.fps == NTSC and (info.width, info.height) == (W, H)
    assert (info.sample_rate, info.channels) == (SR, 1)


def test_render_frame_and_sample_counts_match_the_plan(edit):
    plan = edit["plan"]
    assert plan.frame_bounds == (0, 24, 33, 45, 57, 72)
    assert _ffprobe_frame_count(edit["out"]) == plan.total_frames == 72
    assert len(edit["frames"]) == plan.total_frames
    assert len(edit["samples"]) == plan.total_samples


def test_cam_frames_are_the_planned_source_frames_and_gap_holds(edit):
    tl = edit["frames"][:, 4, 4, 0]  # the frame-index quadrant
    for out_frame in range(24):
        assert _source_frame(tl[out_frame]) == pytest.approx(6 + out_frame, abs=0.3)
    assert all(
        abs(tl[k] - tl[23]) <= 1 for k in range(24, 31)
    )  # the gap holds frame 29


def test_dissolve_keeps_the_length_and_blends_across_the_join(edit):
    frames = edit["frames"]
    red = frames[:, H - 4, 4, 0]  # bottom-left: green in the source, ramp in the still
    assert len(frames) == 72  # two dissolves, total unchanged
    # 30 is still pure Cam (xfade weight 0); 36 is pure still; between, a blend
    assert red[30] < 10 and all(red[k] < red[k + 1] for k in range(30, 35))


def test_still_keeps_one_continuous_move_across_the_sound_cut(edit):
    brightness = edit["frames"][:, :, :, 0].mean(axis=(1, 2))
    steps = [
        brightness[k + 1] - brightness[k] for k in range(37, 52)
    ]  # the cut is at 45
    assert all(s > 0 for s in steps)  # one monotone pan, no restart
    assert (
        max(steps) - min(steps) <= 1.5
    )  # linear (to pixel rounding): no jump at the cut


def test_punch_in_crops_to_the_top_left_quadrant(edit):
    frames = edit["frames"]
    tr, bl = (4, W - 4), (H - 4, 4)
    assert frames[10][tr].tolist()[0] > 200 and frames[10][tr].tolist()[1] < 50  # red
    for k in range(60, 72):
        corners = [frames[k][tr], frames[k][bl], frames[k][4, 4]]
        assert all(abs(c - corners[2]).max() <= 4 for c in corners)  # all top-left grey
        expected = (k + 15) % GREY_PERIOD  # output 60 is source 75 (2.4 s - 3 frames)
        assert _source_frame(frames[k][4, 4, 0]) == pytest.approx(expected, abs=0.3)


def test_gap_is_silence_of_the_planned_length(edit):
    import numpy as np

    plan, samples = edit["plan"], edit["samples"]
    gap_start, gap_end = plan.sample_bounds[1], plan.sample_bounds[2]
    assert gap_end - gap_start == round(33 * Fraction(SR) / NTSC) - round(
        24 * Fraction(SR) / NTSC
    )
    assert not samples[gap_start:gap_end].any()
    rms = lambda a: float(np.sqrt(np.mean(a.astype(float) ** 2)))  # noqa: E731
    assert rms(samples[gap_start - 1000 : gap_start - FADE]) > 1000
    assert rms(samples[gap_end + FADE : gap_end + 1000]) > 1000


def test_overlay_appears_only_in_its_range(edit):
    frames = edit["frames"]
    corner = frames[:, H - 3, W - 3]
    green = [k for k in range(len(frames)) if corner[k][1] > 200 and corner[k][2] < 50]
    assert green == list(range(15, 24))


def test_default_output_beside_source_and_custom_still_renderer(media):
    calls = []

    def flat_still(still, *, frames, fps, size, output):
        calls.append(frames)
        w, h = size
        subprocess.run(
            [_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
             "-i", f"color=white:s={w}x{h}:r={fps.numerator}/{fps.denominator}",
             "-frames:v", str(frames - 2),  # short on purpose: the final pass holds
             "-pix_fmt", "yuv420p", str(output)],
            check=True,
        )  # fmt: skip
        return output

    out = render(
        [Span(0.0, 0.5, "a"), Gap(0.25, "w")],
        {"a": Cam(), "w": Still("unused.png")},
        source=media["source"],
        still_renderer=flat_still,
        closing_fade=0.1,
    )
    try:
        assert out == media["source"].with_name("src_edit.mp4")
        assert calls == [round(0.25 * NTSC)]
        frames = _frames(out)
        assert (
            len(frames)
            == _ffprobe_frame_count(out)
            == round(0.5 * NTSC) + round(0.25 * NTSC)
        )
        assert (
            frames[-5].mean() > 150 and frames[-1].mean() < 30
        )  # white, faded to black
    finally:
        out.unlink(missing_ok=True)


def test_fps_override_resamples_to_the_requested_rate(media):
    segments = [Span(0.5, 1.5, "a"), Span(2.0, 2.4, "b")]
    clips = {"a": Cam(), "b": Cam(zoom=1.5)}
    out = render(
        segments, clips, source=media["source"], fps=25,
        output=media["dir"] / "at25.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    plan = timeline(segments, clips, fps=25, sample_rate=SR)
    assert (
        _ffprobe_frame_count(out) == plan.total_frames == 36
    )  # 0.5 s -> 12 (half-even)
    assert len(_samples(out)) == plan.total_samples == 36 * SR // 25


def test_static_still_cropped_to_its_content_box_shows_no_bars(media):
    pytest.importorskip("burns")
    Image = pytest.importorskip("PIL.Image")
    card = Image.new("RGB", (W, H), (0, 0, 0))
    card.paste((255, 0, 0), (16, 0, 48, H))  # a pillarboxed card: bars left and right
    segments = [Span(0.0, 0.5, "a"), Gap(0.5, "card")]
    out = render(
        segments,
        {"a": Cam(), "card": Still(card, content_box=(0.25, 0, 0.5, 1))},
        source=media["source"],
        output=media["dir"] / "card.mkv",
        encode_args=PCM_MKV,
    )
    frames = _frames(out)
    first_card = round(0.5 * NTSC)
    assert len(frames) == first_card + round(0.5 * NTSC)
    for frame in frames[first_card:]:
        assert (
            frame[:, :, 0].min() > 200 and frame[:, :, 1].max() < 60
        )  # red edge to edge
