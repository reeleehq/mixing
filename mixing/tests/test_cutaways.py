"""``mixing.video.cutaways``: the pure plan, and a real single-pass render."""

import pytest

pytest.importorskip("moviepy")
pytest.importorskip("cv2")

from mixing.util import has_ffmpeg
from mixing.video.cutaways import (
    CutawayPlan,
    Insert,
    MediaInfo,
    cutaways,
    parse_media_info,
    plan_cutaways,
    probe_media,
)

INFO = MediaInfo(width=320, height=240, fps=24.0, duration=3.0, has_audio=True)
SILENT = MediaInfo(width=320, height=240, fps=24.0, duration=3.0, has_audio=False)


def _plan(inserts, *, base=INFO, infos=None, **kw):
    infos = infos or {str(i.src): INFO for i in inserts}
    return plan_cutaways("base.mp4", inserts, base_info=base, insert_infos=infos, **kw)


def test_insert_validates_its_window():
    with pytest.raises(ValueError):
        Insert("a.mp4", 0.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        Insert("a.mp4", -1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Insert("a.mp4", 0.0, 0.0, 1.0, audio_gain=-0.1)


def test_plan_places_insert_on_the_base_timeline_and_gates_the_overlay():
    plan = _plan([Insert("a.mp4", 0.5, 1.0, 2.0)])
    fc = plan.filter_complex
    assert "trim=start=0.5:duration=1" in fc
    assert "setpts=PTS+1/TB" in fc
    assert "overlay=eof_action=pass:enable='gte(t,1)*lt(t,2)'" in fc
    assert plan.maps == ("[b1]", "0:a")  # base audio passes through untouched


def test_insert_audio_is_mixed_only_with_a_gain_and_without_normalising():
    silent_insert = _plan([Insert("a.mp4", 0, 1, 2)])
    assert "amix" not in silent_insert.filter_complex
    mixed = _plan([Insert("a.mp4", 0, 1, 2, audio_gain=0.3)])
    assert "volume=0.3" in mixed.filter_complex
    assert "adelay=1000|1000" in mixed.filter_complex
    assert "amix=inputs=2:duration=first:normalize=0" in mixed.filter_complex
    assert mixed.maps == ("[b1]", "[aout]")


def test_stills_and_audioless_inserts_never_reach_the_audio_mix():
    plan = _plan(
        [Insert("s.png", 0, 1, 2, audio_gain=1.0)],
        infos={"s.png": SILENT},
    )
    assert "amix" not in plan.filter_complex
    assert plan.inputs[2:4] == ("-loop", "1")  # the still is looped for its window


def test_silent_base_gets_synthesised_silence_when_inserts_bring_audio():
    plan = _plan([Insert("a.mp4", 0, 1, 2, audio_gain=0.5)], base=SILENT)
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=3" in plan.filter_complex


def test_overlay_shows_only_while_an_insert_does():
    plan = _plan(
        [Insert("a.mp4", 0, 1, 2), Insert("b.mp4", 0, 2.5, 3)], overlay="cap.png"
    )
    assert "enable='gte(t,1)*lt(t,2)+gte(t,2.5)*lt(t,3)'" in plan.filter_complex
    assert plan.maps[0] == "[vout]"


def test_fit_modes_and_validation():
    assert "pad=320:240" in _plan([Insert("a.mp4", 0, 1, 2)], fit="fit").filter_complex
    assert "crop=320:240" in _plan([Insert("a.mp4", 0, 1, 2)]).filter_complex
    with pytest.raises(ValueError, match="fit"):
        _plan([Insert("a.mp4", 0, 1, 2)], fit="cover")
    with pytest.raises(ValueError, match="at least one"):
        _plan([])


def test_insert_grade_applies_to_inserts_only():
    plan = _plan([Insert("a.mp4", 0, 1, 2)], insert_grade="eq=saturation=0")
    assert plan.filter_complex.count("eq=saturation=0") == 1


def test_parse_media_info():
    info = parse_media_info(
        "Duration: 00:01:02.50, start: 0\n"
        "Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 30 fps\n"
        "Stream #0:1: Audio: aac, 48000 Hz"
    )
    assert info == MediaInfo(1920, 1080, 30.0, 62.5, True)
    with pytest.raises(ValueError):
        parse_media_info("Audio: aac")


def test_dry_run_returns_the_plan_without_running(tmp_path):
    out = cutaways(
        "base.mp4",
        [Insert("a.mp4", 0, 1, 2)],
        probe=lambda p: INFO,
        dry_run=True,
    )
    assert isinstance(out, CutawayPlan)
    assert out.command(tmp_path / "o.mp4")[-1].endswith("o.mp4")


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_real_render_cuts_away_and_keeps_base_timeline_and_audio(
    make_color_video, tmp_path
):
    import cv2

    base = make_color_video(3.0, color=(255, 0, 0), with_audio=True)
    cut = make_color_video(1.5, color=(0, 0, 255), size=(160, 120), with_audio=True)
    out = cutaways(
        base,
        [Insert(cut, 0.0, 1.0, 2.0, audio_gain=0.5)],
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
        [Insert(still, 0.0, 1.0, 2.0)],
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
