"""Tests for ``mixing.edit``: the frame-exact timeline (pure) and real tiny renders.

The render tests build their own source with ffmpeg (``lavfi``) rather than the
``conftest`` colour-clip fixture, because they need what that fixture cannot
make: an NTSC rate (``30000/1001``), a picture that encodes its own frame index
(the top-left quadrant's grey is ``4 * N``), distinct quadrant colours for the
punch-in, and a tone to tell silence from sound. Everything is 64x36 and ~3 s.
"""

from __future__ import annotations

import re
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
    SourceInfo,
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
        [Span(1, 2, "a"), Span(0, 1, "b")],
        {"a": Cam(dissolve_in=0), "b": Cam(dissolve_in=0.2)},
        fps=30,
        sample_rate=SR,
        source_frames=61,
    )
    a, b = plan.picture
    # 3 frames of handle wanted past frame 60, one exists: the last is held
    assert a.pieces == (PicturePiece(30, 31), PicturePiece(60, 2, hold=True))
    # 3 frames wanted before frame 0, none exist: frame 0 is held
    assert b.pieces == (PicturePiece(0, 3, hold=True), PicturePiece(0, 30))


def test_parts_tile_the_film_exactly():
    from mixing.edit import _parts

    plan = timeline(
        [Span(0, 1, "a"), Span(2, 3, "b"), Span(3, 4, "c"), Gap(1, "d")],
        {
            "a": Cam(),
            "b": Cam(dissolve_in=0.3),
            "c": Cam(),
            "d": Still("x.png", dissolve_in=0.5),
        },
        fps=NTSC,
        sample_rate=SR,
    )
    parts = _parts(plan)
    assert sum(p.stop - p.start for p in parts) == plan.total_frames
    assert [(p.clip, p.blend) for p in parts] == [
        (0, False), (1, True), (1, False), (2, False), (3, True), (3, False),
    ]  # fmt: skip


@pytest.mark.parametrize(
    "segments, clips, match",
    [
        ([Gap(1, "a")], {"a": Cam()}, "only gaps"),
        ([Span(0, 1, "a")], {"a": Cam(dissolve_in=0.5)}, "cannot dissolve in"),
        ([Span(0, 1, "x"), Span(1, 1.1, "b"), Span(1.2, 1.9, "a")],
         {"x": Cam(), "a": Cam(dissolve_in=0.4), "b": Cam(dissolve_in=0.4)}, "too short"),
        ([Span(0, 1, "zz")], {"a": Cam()}, "not in `clips`"),
        ([Span(0, 0.01, "a")], {"a": Cam()}, "shorter than one frame"),
        ([], {"a": Cam()}, "at least one segment"),
        ([Span(0, 2.5, "a")], {"a": Cam()}, "past the end of the source"),
    ],
)  # fmt: skip
def test_impossible_edits_are_refused(segments, clips, match):
    with pytest.raises(ValueError, match=match):
        timeline(segments, clips, fps=30, sample_rate=SR, source_frames=60)


def test_segments_are_spans_and_gaps_only():
    with pytest.raises(TypeError, match="a Span or a Gap"):
        timeline([(0, 1, "a")], {"a": Cam()}, fps=30, sample_rate=SR)


def test_slide_into_keeps_windows_inside_the_content_box():
    from mixing.edit import slide_into

    box = (0.25, 0.0, 0.5, 1.0)
    assert slide_into((0.6, 0.2, 0.5, 0.5), box) == (0.25, 0.2, 0.5, 0.5)
    assert slide_into((0.3, 0.0, 0.25, 1.0), box) == (0.3, 0.0, 0.25, 1.0)


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


def _frames(path, size=(W, H)):
    import numpy as np

    w, h = size
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
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(int)


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
        "s": Still(media["still"], pan, fit="fill", dissolve_in=0.2),
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
        # p's render starts at output 54 with source frame 69 (2.4 s is frame 72,
        # less the 3-frame dissolve head), so output k shows source k + 15
        expected = (k + 15) % GREY_PERIOD
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


# --- review round: sources the composite edit does not cover ------------------------


def _make_source(path, *, rate="30000/1001", select=None, audio_offset=None, mp4=False):
    """The frame-indexed quadrant source, as mp4/AAC or mkv/PCM, optionally
    variable-rate (``select`` drops frames) or with the sound starting late."""
    top, left = f"lt(Y,{H // 2})", f"lt(X,{W // 2})"
    grey = f"mod(N*{GREY_STEP},256)"
    geq = (
        f"r='if({top}*{left},{grey},if({top},255,0))':"
        f"g='if({top}*{left},{grey},if({top},0,if({left},255,0)))':"
        f"b='if({top}*{left},{grey},if({top},0,if({left},0,255)))'"
    )
    picture = f"color=black:s={W}x{H}:r={rate}:d=3,format=rgb24,geq={geq}"
    if select:
        picture += f",select='{select}'"
    argv = [_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", picture]
    if audio_offset:
        argv += ["-itsoffset", str(audio_offset)]
    argv += ["-f", "lavfi", "-i", f"sine=f=440:r={SR}:d=3"]
    if select:
        argv += ["-fps_mode", "vfr"]
    if mp4:
        argv += ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", "-c:a", "aac"]
    else:
        argv += ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"]
        argv += ["-c:a", "pcm_s16le"]
    subprocess.run([*argv, str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def mp4_source(media):
    return _make_source(media["dir"] / "src.mp4", mp4=True)


def _top_left_frames(path, size=(W, H)):
    return [_source_frame(f[4, 4, 0]) for f in _frames(path, size)]


def test_mp4_source_renders_exact_frames_picture_and_samples(media, mp4_source):
    info = probe_source(mp4_source)
    assert (info.fps, info.frames, info.channels) == (NTSC, 90, 1)
    segments = [Span(0.5, 1.0, "a"), Gap(0.2, "a"), Span(1.5, 2.0, "b")]
    clips = {"a": Cam(), "b": Cam(dissolve_in=0.2)}
    out = render(
        segments, clips, source=mp4_source,
        output=media["dir"] / "from_mp4.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    plan = timeline(
        segments, clips, fps=NTSC, sample_rate=SR, source_frames=info.frames
    )
    assert _ffprobe_frame_count(out) == plan.total_frames
    assert len(_samples(out)) == plan.total_samples
    seen = _top_left_frames(out)
    for k in range(15):  # source frames 15..29 (0.5 s is frame 15)
        assert seen[k] == pytest.approx(15 + k, abs=0.3)


def test_banner_probe_without_ffprobe(monkeypatch, media, mp4_source):
    import mixing.edit

    monkeypatch.setattr(mixing.edit, "_ffprobe_exe", lambda: None)
    info = probe_source(mp4_source)
    assert (info.fps, info.width, info.height, info.frames) == (NTSC, W, H, 90)
    assert (info.sample_rate, info.channels) == (SR, 1)
    segments = [Span(0.2, 0.8, "a"), Span(1.0, 1.4, "a")]
    out = render(
        segments, {"a": Cam()}, source=mp4_source,
        output=media["dir"] / "banner.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    plan = timeline(segments, {"a": Cam()}, fps=NTSC, sample_rate=SR)
    assert _ffprobe_frame_count(out) == plan.total_frames
    assert len(_samples(out)) == plan.total_samples


def test_render_refuses_a_span_past_the_source_end(media):
    # an mkv has no frame count: the duration gives it
    with pytest.raises(ValueError, match="past the end of the source"):
        render([Span(2.5, 3.5, "a")], {"a": Cam()}, source=media["source"])
    with pytest.raises(ValueError, match="past the end of the source"):
        render(
            [Span(0.0, 0.5, "a"), Span(4.0, 4.5, "b")],
            {"a": Cam(), "b": Cam()},
            source=media["source"],
        )


def test_sound_starting_late_stays_in_sync(media):
    import numpy as np

    source = _make_source(media["dir"] / "late_sound.mkv", audio_offset=0.5)
    assert probe_source(source).audio_offset == pytest.approx(0.5)
    out = render(
        [Span(0.0, 1.0, "a")], {"a": Cam()}, source=source,
        output=media["dir"] / "late_sound_out.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    samples = _samples(out).astype(float)
    quiet, loud = samples[: int(0.45 * SR)], samples[int(0.55 * SR) :]
    assert not quiet.any()  # the source is silent before 0.5 s
    assert np.sqrt(np.mean(loud**2)) > 1000


def test_variable_rate_source_plays_at_its_speed(media):
    # every 4th frame dropped: decoded-frame counting would play 4/3 too fast
    source = _make_source(
        media["dir"] / "vfr.mp4", select="not(eq(mod(n\\,4)\\,3))", mp4=True
    )
    with pytest.warns(UserWarning, match="variable-rate"):
        info = probe_source(source)
    assert info.fps == NTSC
    with pytest.warns(UserWarning, match="variable-rate"):
        out = render(
            [Span(1.0, 2.0, "a")], {"a": Cam()}, source=source,
            output=media["dir"] / "vfr_out.mkv", encode_args=PCM_MKV,
        )  # fmt: skip
    seen = _top_left_frames(out)
    assert len(seen) == 30
    # a dropped frame is filled by its neighbour: never more than one frame off
    assert all(abs(f - (30 + k)) <= 1.3 for k, f in enumerate(seen))


def test_rotated_source_keeps_its_displayed_shape(media, mp4_source):
    rotated = media["dir"] / "rotated.mp4"
    subprocess.run(
        [_ffmpeg(), "-v", "error", "-y", "-display_rotation", "90",
         "-i", str(mp4_source), "-c", "copy", str(rotated)],
        check=True,
    )  # fmt: skip
    info = probe_source(rotated)
    assert (info.width, info.height) == (H, W)
    out = render(
        [Span(0.0, 0.5, "a")], {"a": Cam()}, source=rotated,
        output=media["dir"] / "rotated_out.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    produced = probe_source(out)
    assert (produced.width, produced.height) == (H, W)
    upright = _frames(rotated, (H, W))[:15]
    assert abs(_frames(out, (H, W)) - upright).max() <= 12  # same picture, not squashed


def test_overlay_from_a_pil_image_fades_and_is_clipped_to_the_film(media):
    Image = pytest.importorskip("PIL.Image")
    green = Image.new("RGBA", (W, H), (0, 255, 0, 255))
    out = render(
        [Span(0.0, 1.0, "a")], {"a": Cam()}, source=media["source"],
        overlays=[Overlay(green, 0.2, 9.0, fade=0.2)],  # runs far past the film
        output=media["dir"] / "overlay_fade.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    g = _frames(out)[:, H - 4, W - 4, 1]  # the blue quadrant: no green of its own
    assert len(g) == 30
    assert all(v < 10 for v in g[:7])  # before 0.2 s (frame 6, alpha 0)
    assert all(g[k] < g[k + 1] for k in range(6, 11))  # fading in
    assert all(v > 245 for v in g[12:24])
    assert all(g[k] > g[k + 1] for k in range(24, 29)) and g[29] < 245  # fading out
    with pytest.raises(ValueError, match="after the end of the film"):
        render(
            [Span(0.0, 1.0, "a")], {"a": Cam()}, source=media["source"],
            overlays=[Overlay(green, 5.0, 6.0)],
        )  # fmt: skip


def test_gap_opening_a_cam_clip_holds_the_next_frame(media):
    out = render(
        [Gap(0.3, "a"), Span(1.0, 1.5, "a")], {"a": Cam()}, source=media["source"],
        output=media["dir"] / "leading_gap.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    seen = _top_left_frames(out)
    assert len(seen) == 9 + 15
    assert all(f == pytest.approx(30, abs=0.3) for f in seen[:9])
    assert all(f == pytest.approx(30 + k, abs=0.3) for k, f in enumerate(seen[9:]))


def test_output_directory_and_callable(media, tmp_path):
    segments, clips = [Span(0.0, 0.3, "a")], {"a": Cam()}
    out = render(segments, clips, source=media["source"], output=tmp_path)
    assert out == tmp_path / "src_edit.mp4" and out.stat().st_size
    got = render(segments, clips, source=media["source"], output=lambda p: ("sunk", p))
    try:
        assert got == ("sunk", media["source"].with_name("src_edit.mp4"))
        assert got[1].exists()
    finally:
        got[1].unlink(missing_ok=True)


def test_many_spans_never_open_more_than_two_decoders(monkeypatch, media):
    import mixing.edit

    calls = []
    real_run = mixing.edit._run

    def counting_run(*args):
        calls.append(args.count("-i"))
        real_run(*args)

    monkeypatch.setattr(mixing.edit, "_run", counting_run)
    segments = [Span(0.1 * k, 0.1 * k + 0.05, "a") for k in range(16)]
    segments += [Span(2.0, 2.5, "b")]
    clips = {"a": Cam(zoom=1.5), "b": Cam(dissolve_in=0.2)}
    out = render(
        segments, clips, source=media["source"],
        output=media["dir"] / "many.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    assert max(calls) == 2  # parts: one input; a dissolve and the final pass: two
    assert len(calls) > 16  # one bounded run per piece
    plan = timeline(segments, clips, fps=NTSC, sample_rate=SR)
    assert _ffprobe_frame_count(out) == plan.total_frames


def test_square_still_is_pillarboxed_not_cropped(media):
    pytest.importorskip("burns")
    Image = pytest.importorskip("PIL.Image")
    square = Image.new("RGB", (H, H), (0, 0, 255))
    square.paste((255, 0, 0), (0, 0, H, H // 4))  # the top quarter is red
    segments = [Span(0.0, 0.3, "a"), Gap(0.3, "s")]
    held = _frames(
        render(
            segments,
            {"a": Cam(), "s": Still(square)},
            source=media["source"],
            output=media["dir"] / "square.mkv",
            encode_args=PCM_MKV,
        )  # fmt: skip
    )[-1]
    red_rows = [y for y in range(H) if held[y, W // 2, 0] > 128]
    assert red_rows == list(range(H // 4))  # the whole picture: nothing cropped
    assert held[:, 2].max() < 30 and held[:, W - 3].max() < 30  # pillarbox bars
    filled = _frames(
        render(
            segments,
            {"a": Cam(), "s": Still(square, fit="fill")},
            source=media["source"],
            output=media["dir"] / "square_fill.mkv",
            encode_args=PCM_MKV,
        )  # fmt: skip
    )[-1]
    assert filled[:, 2].max(axis=1).min() > 100  # cover-cropped, as asked: no bars


def test_a_move_is_kept_inside_the_content_box(media):
    pytest.importorskip("burns")
    Image = pytest.importorskip("PIL.Image")
    from burns import BurnsPath, Rect

    card = Image.new("RGB", (W, H), (0, 0, 0))
    card.paste((255, 0, 0), (16, 0, 48, H))  # a pillarboxed card: bars left and right
    # a pan from the left bar to the right one, were it not kept inside the box
    pan = BurnsPath.from_start_end(
        Rect.from_center_zoom(0.2, 0.5, 2.0), Rect.from_center_zoom(0.8, 0.5, 2.0)
    )
    out = render(
        [Span(0.0, 0.5, "a"), Gap(0.5, "card")],
        {"a": Cam(), "card": Still(card, pan, content_box=(0.25, 0, 0.5, 1))},
        source=media["source"], output=media["dir"] / "card.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    frames = _frames(out)
    first_card = round(0.5 * NTSC)
    assert len(frames) == first_card + round(0.5 * NTSC)
    for frame in frames[first_card:]:
        assert (
            frame[:, :, 0].min() > 200 and frame[:, :, 1].max() < 60
        )  # red edge to edge


def test_named_move_and_aspect_mismatch_refusal(media):
    pytest.importorskip("burns")
    from burns import BurnsPath, Rect

    segments = [Span(0.0, 0.3, "a"), Gap(0.5, "s")]
    out = render(
        segments, {"a": Cam(), "s": Still(media["still"], "push_in")},
        source=media["source"], output=media["dir"] / "push.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    frames = _frames(out)
    assert len(frames) == round(0.3 * NTSC) + round(0.5 * NTSC)
    assert abs(frames[-1] - frames[round(0.3 * NTSC)]).max() > 20  # it moved
    vertical = BurnsPath.from_start_end(
        Rect(0, 0, 1, 1), Rect(0.1, 0.1, 0.5, 0.5), output_aspect=9 / 16
    )
    with pytest.raises(ValueError, match="output_aspect"):
        render(
            segments, {"a": Cam(), "s": Still(media["still"], vertical)},
            source=media["source"],
        )  # fmt: skip


@pytest.mark.parametrize(
    "info, match",
    [
        (SourceInfo(65, 37, NTSC, 90, SR, 1), "even dimensions"),
        (SourceInfo(64, 36, NTSC, 90), "no sound stream"),
    ],
)
def test_unsupported_sources_are_refused_before_any_work(monkeypatch, info, match):
    import mixing.edit

    monkeypatch.setattr(mixing.edit, "_probe", lambda _: info)
    with pytest.raises(ValueError, match=match):
        render([Span(0, 1, "a")], {"a": Cam()}, source="x.mp4")


def test_overlay_position_expressions_with_commas(media):
    Image = pytest.importorskip("PIL.Image")
    dot = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
    out = render(
        [Span(0.0, 0.3, "a")], {"a": Cam()}, source=media["source"],
        overlays=[Overlay(dot, 0.0, 0.3, x="min(60,W-w)", y="max(H-h,0)")],
        output=media["dir"] / "overlay_expr.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    frame = _frames(out)[0]
    assert frame[H - 4, W - 4, 1] > 200 and frame[H - 4, W - 4, 2] < 60  # bottom right
    assert frame[4, W - 4, 1] < 60  # not at the top


def test_variable_rate_mkv_is_detected_by_its_packet_count(media):
    # mkv barely fills avg_frame_rate: 60 packets over 3 s at a nominal 30 fps
    source = _make_source(
        media["dir"] / "vfr.mkv", rate="30", select="not(eq(mod(n\\,3)\\,2))"
    )
    with pytest.warns(UserWarning, match="variable-rate"):
        info = probe_source(source)
    # from the duration (the last kept frame is 88), not the 60 packets
    assert info.variable_rate and info.frames == 89
    with pytest.warns(UserWarning, match="variable-rate"):
        out = render(
            [Span(2.2, 2.8, "a")], {"a": Cam()}, source=source,  # past packet 60
            output=media["dir"] / "vfr_mkv_out.mkv", encode_args=PCM_MKV,
        )  # fmt: skip
    seen = _top_left_frames(out)
    assert len(seen) == 18
    assert all(abs(f - (66 + k) % GREY_PERIOD) <= 1.3 for k, f in enumerate(seen))


def test_transparent_still_is_flattened_onto_the_bars_colour(media):
    pytest.importorskip("burns")
    Image = pytest.importorskip("PIL.Image")
    card = Image.new("RGBA", (W, H), (255, 255, 255, 0))  # white, but transparent
    card.paste((255, 0, 0, 255), (24, 12, 40, 24))
    held = _frames(
        render(
            [Span(0.0, 0.3, "a"), Gap(0.3, "card")],
            {"a": Cam(), "card": Still(card)},
            source=media["source"],
            output=media["dir"] / "alpha_card.mkv",
            encode_args=PCM_MKV,
        )  # fmt: skip
    )[-1]
    assert held[2, 2].max() < 30  # transparent: the bars colour, not the hidden white
    assert held[H // 2, W // 2, 0] > 200 and held[H // 2, W // 2, 1] < 60


def test_sound_starting_before_the_picture_stays_in_sync(media):
    """Regression (#122 review): video at exactly 0.0, sound from -0.27 s. A
    stated 0.0 start was read as unset, losing the 270 ms lead: the sound's
    loud half would land at 1.0 s instead of 0.73 s."""
    import numpy as np

    lead, loud_from = 0.27, 1.0  # the sound is silent for its first second
    source = media["dir"] / "early_sound.mkv"
    tone = f"sine=f=440:r={SR}:d=3,volume=0:enable='lt(t,{loud_from})'"
    subprocess.run(
        [_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=black:s={W}x{H}:r=25:d=3",
         "-itsoffset", str(-lead), "-f", "lavfi", "-i", tone,
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "pcm_s16le", "-avoid_negative_ts", "disabled", str(source)],
        check=True,
    )  # fmt: skip
    info = probe_source(source)
    assert info.video_start == pytest.approx(lead, abs=2e-3)
    assert info.audio_offset == pytest.approx(-lead, abs=2e-3)
    out = render(
        [Span(0.0, 1.5, "a")], {"a": Cam()}, source=source,
        output=media["dir"] / "early_sound_out.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    samples = _samples(out).astype(float)
    onset = loud_from - lead  # 0.73 s into the picture
    quiet = samples[: int((onset - 0.03) * SR)]
    loud = samples[int((onset + 0.03) * SR) : int((onset + 0.3) * SR)]
    assert not quiet.any()
    assert np.sqrt(np.mean(loud**2)) > 1000


#: A white flash on frame FLASH_FRAME and a click at the same instant: the
#: rendered film keeps them together when the sound's start is read right.
FLASH_FRAME, FLASH_RATE, CLICK_S = 25, 25, 0.02
#: A/V tolerance (s) of the measured sync: a quarter of a 1 kHz cycle's slack
#: either side, far under the 7-23 ms codec offsets the probe must get right.
SYNC_TOLERANCE_S = 0.003


#: First ffmpeg major version whose decoder drops aac/mp3 priming samples.
PRIMING_TRIMMED_FROM = 9


def _major_version(binary) -> str:
    banner = subprocess.run([binary, "-version"], capture_output=True, text=True).stdout
    return re.search(r"version\s+[nN]?(\d+)", banner).group(1)


@pytest.mark.parametrize(
    "codec, container",
    [("aac", "mkv"), ("libmp3lame", "mkv"), ("libopus", "mkv"), ("libopus", "webm")],
)
def test_compressed_sound_starts_where_it_plays(media, codec, container):
    """Regression (#122 re-review): a stream's ``start_time`` is its first
    packet. aac/mp3 in mkv really play from it (-21/-23 ms); opus drops its
    pre-skip, so its first packet at -7 ms plays from 0. The probe reads the
    first decoded frame, so the flash and the click stay together for all."""
    import numpy as np

    from mixing.probe import find_ffprobe

    ffprobe = find_ffprobe()
    if ffprobe is None:  # the banner fallback assumes 0 starts (documented)
        pytest.skip("no ffprobe: the banner fallback cannot read stream starts")
    trims = [int(_major_version(b)) >= PRIMING_TRIMMED_FROM for b in (ffprobe, _ffmpeg())]
    if trims[0] != trims[1]:
        # ffmpeg 9 trims aac/mp3 priming when it decodes (first frame at 0.000);
        # 7 keeps it (-0.023). The probe cannot know which decoder renders.
        pytest.skip("ffprobe and ffmpeg disagree on trimming codec priming")

    flash_s = FLASH_FRAME / FLASH_RATE
    picture = (
        f"color=black:s={W}x{H}:r={FLASH_RATE}:d=3,"
        f"drawbox=enable='eq(n,{FLASH_FRAME})':c=white:t=fill"
    )
    click = (
        f"aevalsrc='0.9*sin(2*PI*1000*t)*between(t,{flash_s},{flash_s + CLICK_S})'"
        f":s={SR}:d=3"
    )
    video_codec = "libvpx-vp9" if container == "webm" else "libx264"
    source = media["dir"] / f"click_{codec}.{container}"
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", picture,
         "-f", "lavfi", "-i", click, "-c:v", video_codec, "-pix_fmt", "yuv420p",
         "-c:a", codec, str(source)],
        capture_output=True, text=True,
    )  # fmt: skip
    if proc.returncode:
        pytest.skip(f"this ffmpeg cannot encode {codec}/{container}: {proc.stderr[-200:]}")
    out = render(
        [Span(0.0, 2.0, "a")], {"a": Cam()}, source=source,
        output=media["dir"] / f"click_{codec}_{container}_out.mkv", encode_args=PCM_MKV,
    )  # fmt: skip
    sound = np.abs(_samples(out).astype(float))
    onset = int(np.argmax(sound > 0.1 * sound.max())) / SR
    brightness = _frames(out).mean(axis=(1, 2, 3))
    flash = int(np.argmax(brightness > 128)) / FLASH_RATE
    assert flash == pytest.approx(flash_s)
    assert onset == pytest.approx(flash, abs=SYNC_TOLERANCE_S)
