"""``mixing.video.cutaways``: the pure plan, and a real single-pass render."""

import pytest

pytest.importorskip("moviepy")
pytest.importorskip("cv2")

from mixing.util import has_ffmpeg
from mixing.video.cutaways import (
    CutawayPlan,
    Cutaway,
    MediaInfo,
    cutaways,
    parse_media_info,
    plan_cutaways,
    probe_media,
)

INFO = MediaInfo(width=320, height=240, fps=24.0, duration=3.0, has_audio=True)
SILENT = MediaInfo(width=320, height=240, fps=24.0, duration=3.0, has_audio=False)


def _plan(items, *, base=INFO, infos=None, **kw):
    """Plan with a fake probe: ``base.mp4`` is ``base``; others come from ``infos``."""
    infos = infos or {}

    def probe(path):
        return base if str(path) == "base.mp4" else infos.get(str(path), INFO)

    return plan_cutaways("base.mp4", items, probe=probe, **kw)


def test_insert_validates_its_window():
    with pytest.raises(ValueError):
        Cutaway("a.mp4", 0.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        Cutaway("a.mp4", -1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Cutaway("a.mp4", 0.0, 0.0, 1.0, audio_gain=-0.1)


def test_plan_places_insert_on_the_base_timeline_and_gates_the_overlay():
    plan = _plan([Cutaway("a.mp4", 0.5, 1.0, 2.0)])
    fc = plan.filter_complex
    assert "setpts=PTS+1/TB" in fc
    assert "overlay=eof_action=pass:enable='gte(t,1)*lt(t,2)'" in fc
    assert plan.maps == ("[b1]", "0:a")  # base audio passes through untouched


def test_insert_audio_is_mixed_only_with_a_gain_and_without_normalising():
    silent_insert = _plan([Cutaway("a.mp4", 0, 1, 2)])
    assert "amix" not in silent_insert.filter_complex
    mixed = _plan([Cutaway("a.mp4", 0, 1, 2, audio_gain=0.3)])
    assert "volume=0.3" in mixed.filter_complex
    assert "adelay=1000:all=1" in mixed.filter_complex
    assert "amix=inputs=2:duration=first:normalize=0" in mixed.filter_complex
    assert mixed.maps == ("[b1]", "[aout]")


def test_stills_and_audioless_inserts_never_reach_the_audio_mix():
    plan = _plan(
        [Cutaway("s.png", 0, 1, 2, audio_gain=1.0)],
        infos={"s.png": SILENT},
    )  # a still never has audio, whatever its gain
    assert "amix" not in plan.filter_complex
    assert plan.inputs[2:4] == ("-loop", "1")  # the still is looped for its window


def test_silent_base_gets_synthesised_silence_when_inserts_bring_audio():
    plan = _plan([Cutaway("a.mp4", 0, 1, 2, audio_gain=0.5)], base=SILENT)
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=3" in plan.filter_complex


def test_overlay_shows_only_while_an_insert_does():
    plan = _plan(
        [Cutaway("a.mp4", 0, 1, 2), Cutaway("b.mp4", 0, 2.5, 3)], overlay="cap.png"
    )
    assert "enable='gte(t,1)*lt(t,2)+gte(t,2.5)*lt(t,3)'" in plan.filter_complex
    assert plan.maps[0] == "[vout]"


def test_fit_modes_and_validation():
    wide = {"a.mp4": MediaInfo(640, 240, 24.0, 3.0, False)}
    cut = [Cutaway("a.mp4", 0, 1, 2)]
    assert "pad=" in _plan(cut, infos=wide, fit="fit").filter_complex
    assert "crop=" in _plan(cut, infos=wide).filter_complex  # default: fill
    assert "pad=" not in _plan(cut, infos=wide).filter_complex
    with pytest.raises(ValueError, match="at least one"):
        _plan([])


def test_insert_grade_applies_to_inserts_only():
    plan = _plan([Cutaway("a.mp4", 0, 1, 2)], insert_grade="eq=saturation=0")
    assert plan.filter_complex.count("eq=saturation=0") == 1


def test_src_in_past_the_clip_is_refused_and_a_short_clip_warns():
    short = {"a.mp4": MediaInfo(320, 240, 24.0, 1.0, False)}
    with pytest.raises(ValueError, match="src_in"):
        _plan([Cutaway("a.mp4", 5, 1, 2)], infos=short)
    with pytest.warns(UserWarning, match="base shows through"):
        _plan([Cutaway("a.mp4", 0.5, 1, 3)], infos=short)


def test_overlay_position_is_quoted_so_commas_survive():
    plan = _plan(
        [Cutaway("a.mp4", 0, 1, 2)], overlay="c.png", overlay_xy=("max(0,W-w)", "0")
    )
    assert "overlay='max(0,W-w)':'0'" in plan.filter_complex


def test_parse_media_info_swaps_size_for_a_rotated_display_matrix():
    info = parse_media_info(
        "Duration: 00:00:02.00\n"
        "Stream #0:0: Video: h264, yuv444p, 640x360 [SAR 1:1 DAR 16:9], 25 fps\n"
        "Side data:\n Display Matrix: rotation of 90.00 degrees"
    )
    assert (info.width, info.height) == (360, 640)


def test_parse_media_info():
    info = parse_media_info(
        "Duration: 00:01:02.50, start: 0\n"
        "Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 30 fps\n"
        "Stream #0:1: Audio: aac, 48000 Hz"
    )
    assert info == MediaInfo(1920, 1080, 30.0, 62.5, True)
    with pytest.raises(ValueError):
        parse_media_info("Audio: aac")


def test_plan_command_writes_the_given_output(tmp_path):
    plan = plan_cutaways("base.mp4", [Cutaway("a.mp4", 0, 1, 2)], probe=lambda p: INFO)
    assert isinstance(plan, CutawayPlan)
    assert plan.command(tmp_path / "o.mp4")[-1].endswith("o.mp4")


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_cuts_away_and_keeps_base_timeline_and_audio(
    make_color_video, tmp_path
):
    import cv2

    base = make_color_video(3.0, color=(255, 0, 0), with_audio=True)
    cut = make_color_video(1.5, color=(0, 0, 255), size=(160, 120), with_audio=True)
    out = cutaways(
        base,
        [Cutaway(cut, 0.0, 1.0, 2.0, audio_gain=0.5)],
        output=tmp_path / "out.mp4",
    )
    info = probe_media(out)
    assert (info.width, info.height) == (320, 240)
    assert info.duration == pytest.approx(3.0, abs=0.2)
    assert info.has_audio

    def mean_bgr(t):
        cap = cv2.VideoCapture(str(out))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        cap.release()
        assert ok
        return frame.reshape(-1, 3).mean(axis=0)

    before, during, after = mean_bgr(0.5), mean_bgr(1.5), mean_bgr(2.5)
    assert before[2] > 150 and during[0] > 150 and after[2] > 150  # red, blue, red


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_with_a_still_and_a_caption_overlay(make_color_video, tmp_path):
    import cv2
    import numpy as np

    base = make_color_video(3.0, color=(255, 0, 0), with_audio=True)
    still, caption = tmp_path / "still.png", tmp_path / "cap.png"
    cv2.imwrite(str(still), np.full((100, 200, 3), (0, 255, 0), np.uint8))  # green
    cv2.imwrite(str(caption), np.full((20, 20, 4), (255, 255, 255, 255), np.uint8))
    out = cutaways(
        base,
        [Cutaway(still, 0.0, 1.0, 2.0)],
        overlay=caption,
        output=tmp_path / "out.mp4",
    )

    def frame(t):
        cap = cv2.VideoCapture(str(out))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        assert ok
        return f

    during, outside = frame(1.5), frame(0.5)
    assert during[100, 200, 1] > 150  # the still (green) covers the frame
    assert (during[5, 5] > 200).all()  # caption present in the window
    assert not (outside[5, 5] > 200).all()  # ...and absent outside it
    assert probe_media(out).duration == pytest.approx(3.0, abs=0.2)


def _ffmpeg(*args):
    import subprocess

    from mixing.util import ffmpeg_exe

    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", *map(str, args)], check=True)


def _frame_bgr(path, t):
    import cv2

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame.reshape(-1, 3).mean(axis=0)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_seeks_to_src_in(make_color_video, tmp_path):
    base = make_color_video(4.0, color=(0, 255, 0))
    red_then_blue = tmp_path / "rb.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=25:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=25:d=2",
        "-filter_complex", "[0][1]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p",
        red_then_blue,
    )  # fmt: skip
    out = cutaways(
        base, [Cutaway(red_then_blue, 1.0, 1.0, 3.0)], output=tmp_path / "o.mp4"
    )
    # the window shows the clip from 1.0 s: red until 2.0 s on the base, then blue
    assert _frame_bgr(out, 1.5)[2] > 150  # red
    assert _frame_bgr(out, 2.5)[0] > 150  # blue
    assert _frame_bgr(out, 3.5)[1] > 150  # base (green) again


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_of_a_rotated_base_is_not_rotated_twice(tmp_path):
    from mixing.util import ffmpeg_exe
    import subprocess

    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rot.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:d=2", "-pix_fmt",
            "yuv420p", plain)  # fmt: skip
    probe = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-display_rotation", "90", "-i", plain,
         "-c", "copy", rotated], capture_output=True)  # fmt: skip
    if probe.returncode != 0:
        pytest.skip("this ffmpeg cannot write a display matrix")
    still = tmp_path / "s.png"
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", still)
    out = cutaways(rotated, [Cutaway(still, 0, 0.5, 1.5)], output=tmp_path / "o.mp4")
    info = probe_media(out)
    assert (info.width, info.height) == (180, 320)  # displayed size, no 2nd rotation


def test_cutaway_audio_is_downmixed_before_amix():
    plan = _plan([Cutaway("a.mp4", 0, 1, 2, audio_gain=1.0)])
    assert "aresample=ochl=stereo:rematrix_maxval=1" in plan.filter_complex
    assert "sidedata=mode=delete:type=DISPLAYMATRIX" in plan.filter_complex


def test_cutaway_starting_after_the_base_warns():
    with pytest.warns(UserWarning, match="after the base ends"):
        _plan([Cutaway("a.mp4", 0, 10, 12)])


# --- entrances, per-item overlays, post (reeleehq/mixing#115) ---------------------


def test_an_entrance_xfades_the_screen_into_the_item_then_cuts_to_it():
    from looks import Transition

    plan = _plan([Cutaway("a.mp4", 0, 1, 3, transition_in=Transition(0.5, "wipeleft"))])
    fc = plan.filter_complex
    assert "xfade=transition=wipeleft:duration=0.5:offset=0" in fc
    assert "enable='gte(t,1)*lt(t,1.5)'" in fc  # the blend over the entrance
    assert "enable='gte(t,1.5)*lt(t,3)'" in fc  # the item itself after it
    with pytest.raises(ValueError, match="shorter than the window"):
        Cutaway("a.mp4", 0, 1, 1.5, transition_in=Transition(0.5))


def test_an_items_own_overlay_replaces_the_plans_on_its_window():
    plan = _plan(
        [
            Cutaway("a.mp4", 0, 1, 2),
            Cutaway("b.mp4", 0, 2, 3, overlay="big.png", overlay_xy=("(W-w)/2", "0")),
        ],
        overlay="small.png",
    )
    fc, inputs = plan.filter_complex, plan.inputs
    assert "small.png" in inputs and "big.png" in inputs
    assert "overlay='0':'0':shortest=1:enable='gte(t,1)*lt(t,2)'" in fc
    assert "overlay='(W-w)/2':'0':shortest=1:enable='gte(t,2)*lt(t,3)'" in fc
    assert plan.maps[0] == "[vout]"


def test_post_runs_last_on_the_finished_picture():
    plan = _plan([Cutaway("a.mp4", 0, 1, 2)], overlay="c.png", post="negate")
    assert plan.filter_complex.endswith("[c0]negate[vout]")
    assert _plan([Cutaway("a.mp4", 0, 1, 2)]).maps[0] == "[b1]"  # no post: as before


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_dissolves_and_wipes_in(make_color_video, tmp_path):
    from looks import Transition

    base = make_color_video(4.0, color=(255, 0, 0))  # red (BGR: channel 2)
    item = make_color_video(4.0, color=(0, 255, 0))  # green (BGR: channel 1)
    fade = cutaways(
        base,
        [Cutaway(item, 0.0, 1.0, 3.0, transition_in=Transition(1.0, "fade"))],
        output=tmp_path / "fade.mp4",
    )
    before, middle, after = (_frame_bgr(fade, t) for t in (0.5, 1.5, 2.5))
    assert before[2] > 200 and before[1] < 60
    assert 60 < middle[1] < 200 and 60 < middle[2] < 200  # half way: both
    assert after[1] > 200 and after[2] < 60
    wipe = cutaways(
        base,
        [Cutaway(item, 0.0, 1.0, 3.0, transition_in=Transition(1.0, "wipeleft"))],
        output=tmp_path / "wipe.mp4",
    )
    import cv2

    cap = cv2.VideoCapture(str(wipe))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1500)
    ok, frame = cap.read()
    cap.release()
    assert ok
    left, right = frame[:, : frame.shape[1] // 4], frame[:, -frame.shape[1] // 4 :]
    assert left[..., 1].mean() != pytest.approx(right[..., 1].mean(), abs=60)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_per_item_captions_and_post(make_color_video, tmp_path):
    import cv2
    import numpy as np

    base = make_color_video(3.0, color=(255, 0, 0))
    item = make_color_video(3.0, color=(0, 255, 0))
    small, big = tmp_path / "small.png", tmp_path / "big.png"
    cv2.imwrite(str(small), np.full((20, 20, 4), (255, 255, 255, 255), np.uint8))
    cv2.imwrite(str(big), np.full((40, 40, 4), (255, 0, 255, 255), np.uint8))
    out = cutaways(
        base,
        [
            Cutaway(item, 0.0, 0.5, 1.5),
            Cutaway(item, 0.0, 1.5, 2.5, overlay=big, overlay_xy=("W-w", "H-h")),
        ],
        overlay=small,
        post="negate",
        output=tmp_path / "out.mp4",
    )

    def frame(t):
        cap = cv2.VideoCapture(str(out))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        assert ok
        return f

    first, second, outside = frame(1.0), frame(2.0), frame(2.8)
    assert (first[5, 5] < 60).all()  # small white caption, negated to black
    assert not (second[5, 5] < 60).all()  # ...not on the second window
    assert second[-5, -5, 1] > 200  # big magenta caption bottom right, negated green
    assert outside[50, 50, 0] > 200 and outside[50, 50, 2] < 60  # red base negated
