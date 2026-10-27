---
name: mixing-video
description: >
  Use for VIDEO editing with the `mixing` Python package: trim/crop a clip,
  loop it, speed it up or down, replace or mix in a new audio track, normalize
  audio levels, pan/zoom a still image (Ken Burns), build a film from panels,
  concatenate clips, re-edit a finished video from a sound edit list (punch-ins,
  held stills with moves, dissolves, overlays, frame-exact), grab/extract frames, make a YouTube-style thumbnail, burn
  in subtitles, or resize for shorts/tiktok/square. Trigger on phrasings like
  "trim this video", "loop the intro", "speed up / slow down this clip",
  "replace the audio with this music", "lay room tone / ambience under the cut", "add a pan-zoom over this photo", "make a
  thumbnail", "burn the subtitles in", "resize for YouTube Shorts", or "stitch
  these clips together". For audio-only work use mixing-audio; for transcription
  / filler removal use mixing-transcript; for TTS re-voicing use mixing-dubbing.
---

# mixing-video — edit video from Python

Slice, transform, and assemble video. Most ops need **ffmpeg** on PATH.

## Quick start

```python
import mixing

assert mixing.has_ffmpeg(), "install ffmpeg (brew install ffmpeg / apt install ffmpeg)"

from mixing.video import Video, crop_video, replace_audio, make_thumbnail

Video("clip.mp4")[5:15].save(output="cut.mp4")  # trim 5s–15s
crop_video("clip.mp4", 5, 15, output="cut.mp4")  # same, one call
replace_audio("clip.mp4", "music.mp3", output="scored.mp4")
make_thumbnail("clip.mp4", text="My Title", output="thumb.jpg")
```

`import mixing` is light (no moviepy/opencv until you touch a heavy name). Use
the facade (`mixing.Video`, `mixing.crop_video`, …) or import the subpackage
(`from mixing.video import ...`). Note `THUMBNAIL_SIZE` / `YOUTUBE_THUMB_SIZE` /
`SOCIAL_SIZES` live on `mixing.video`, not the top-level facade.

## The `output` egress protocol (applies to every writer below)

One `output` param: `None` → save beside input, return `Path`; a **file path** →
write+return `Path`; a **directory** → auto-named file; a **callable** → sink.
Time is **seconds (float)** unless noted. Exceptions flagged inline:
`ken_burns_*` and `concatenate_videos` predate the protocol (see below).

## `Video` — sliceable, lazy, path-backed

```python
v = Video("movie.mp4")                       # time_unit="seconds" (default)
v.duration, v.fps, v.frame_count             # lazy props (full_duration = source length)
v[10:20]                  -> Video           # a 10s segment view (no copy)
v[-30:]                   -> Video           # last 30s
v[15]                     -> np.ndarray       # single frame at 15s (BGR, from cv2)
v[10:20].save(output="seg.mp4")              # render the segment
v.save_frame(15, output="f.png")             # write one frame
v.to_clip()                                  # moviepy VideoFileClip (caller closes)
for frame in v.frames[0:100]: ...            # .frames -> VideoFrames Mapping[int, ndarray]

Video("movie.mp4", time_unit="frames")[100:500]   # slice by frame number instead
with Video("movie.mp4") as v: d = v.duration       # context-managed (releases handles)
```

- Slices return new `Video` views; slicing is bounds-clamped; a `step` raises.
- An int/float index returns a **frame** (numpy BGR); a slice returns a `Video`.
- `save(output=None, *, codec="libx264", audio_codec="aac", **write_kwargs)`.
- `save_frame(time_or_frame=None, output=None, *, image_format="png",
  copy_to_clipboard=False)` — `output=False` means clipboard-only (needs
  `copy_to_clipboard=True`).

## Core transforms (all file→file, honor `output`)

```python
crop_video("in.mp4", 5, 15, output="out.mp4")  # trim; start==end -> single frame
crop_video("in.mp4", 100, 500, time_unit="frames")  # by frame number
loop_video("intro.mp4", 3, output="x3.mp4")  # n_loops>=1, repeats the clip
change_speed("in.mp4", 2.0, output="fast.mp4")  # 2.0=2x faster, 0.5=half (audio too)
normalize_audio("lecture.mp4", output="leveled.mp4")  # even out volume swings
```

### replace_audio — swap or blend the soundtrack

```python
replace_audio(
    "v.mp4", "music.mp3", output="out.mp4"
)  # mix_ratio=1.0 (default): only new
replace_audio("v.mp4", "bgm.mp3", mix_ratio=0.0, output="out.mp4")  # keep original only
replace_audio("v.mp4", "bgm.mp3", mix_ratio=0.5, output="out.mp4")  # equal blend
replace_audio("v.mp4", "voice.mp3", mix_ratio=0.7)  # 70% new / 30% original
```

`mix_ratio` (0.0–1.0): `1.0`=only new, `0.0`=keep original, `0.5`=blend.
`match_duration=True` (default) loops/trims the audio to the video length.

### overlay_ambient_bed — loop a bed to length and duck it under dialogue

```python
from mixing.video import overlay_ambient_bed

# Loop `room_tone.wav` to the cut's exact length and mix it 25% under the
# existing track. Works on a VIDEO or an AUDIO file (kind chosen by extension);
# file-first, so output=None writes `cut_ambient.mp4` beside the input.
overlay_ambient_bed("cut.mp4", "room_tone.wav", output="cut_amb.mp4")
overlay_ambient_bed("cut.mp4", "rain.wav", mix_ratio=0.2)  # quieter bed
overlay_ambient_bed("cut.mp4", "rain.wav", duck_under_dialogue=True)  # sidechain duck
overlay_ambient_bed("cut.mp4", "sting.wav", loop=False)  # lay it once at the head
```

`mix_ratio` means exactly what it does in `overlay_audio`: the **bed's
prominence** (0.25 ≈ -12 dB under the cut). Under the hood it is
`mixing.audio.loop_audio` → optional `mixing.audio.duck_audio` →
`mixing.audio.overlay_audio` → mux; reach for those directly when you want the
intermediate `Audio` objects.

## Ken Burns (pan/zoom) — mixing wrappers over the `burns` package

mixing wraps burns' renderers so they speak the **`output` protocol** like
everything else (`output=None` lets burns auto-name beside the image).
`output_size=(w,h)` sets the render size; the pan/zoom path is a `BurnsPath`.

```python
from mixing.video import ken_burns_video, ken_burns_film

# Animate a still image into a clip (default path zooms in slightly)
ken_burns_video(
    "photo.jpg", duration=6.0, fps=30, output_size=(1920, 1080), output="pan.mp4"
)

# Stitch a multi-panel film; mux a pre-built audio track over the panels.
# Each panel is an (image, BurnsPath, duration_s) TRIPLE — the path is required.
from burns import BurnsPath

panels = [
    ("p1.jpg", BurnsPath(), 5.0),
    ("p2.jpg", BurnsPath(), 4.0),
]
ken_burns_film(panels, fps=30, audio_path="film_audio.wav", output="film.mp4")
```

### assemble_audio_track — build the audio that lines up with a film

`output` is **required** (no input file to derive a name from). Returns `None`
and writes nothing when every slot is silent.

```python
from mixing.video import assemble_audio_track

assemble_audio_track(
    [
        ("voice1.mp3", 5.0),
        (None, 3.0),
        ("voice2.mp3", 4.0),
    ],  # (audio|None, duration_s) per panel
    output="film_audio.wav",  # WAV pcm_s16le; slot-for-slot with ken_burns_film panels
)
# then: ken_burns_film(panels, audio_path="film_audio.wav", output="film.mp4")
```

Each segment occupies exactly its `duration_s` (audio then silence padding; an
over-long clip is trimmed). `sample_rate=44100` by default.

## Cutaways — timed inserts over an unbroken base video

`cutaways(base, inserts, overlay=...)` (documentary-style B-roll): the base keeps its timeline and audio; each `Cutaway(src, src_in, t0, t1, audio_gain=None)` covers `[t0, t1)` with a clip (or a still image), scaled by `fit` (`"fill"` crops to cover, `"fit"` letterboxes, `"stretch"`). One ffmpeg pass, no moviepy. An insert's own audio is mixed under the base only when `audio_gain` is set (linear; `amix normalize=0`, so the base level is kept). `overlay` (a caption PNG) shows only while an insert does. `insert_grade` / `base_grade` take a `looks.vf(...)` fragment.

```python
from mixing.video import cutaways, plan_cutaways, Cutaway

cutaways(
    "talk.mp4",
    [
        Cutaway("imagined.mp4", 0.0, 12.0, 16.5, audio_gain=0.3),
        Cutaway("still.png", 0.0, 20.0, 23.0),
    ],
    overlay="ai_imagining.png",
    output="talk_cutaways.mp4",
)
plan = plan_cutaways(
    "talk.mp4", items
)  # inspect first: plan.filter_complex, plan.command(out)
```

## Re-edit a finished video — sound and picture edit lists (`mixing.edit`)

`mixing.edit.render(segments, clips, *, source, overlays=(), fps=None, output=None, ...)` re-cuts a finished talking-head video, frame-exact. Two edit lists of the same length: the **sound** (`Span(src_in, src_out, clip)` source seconds, `Gap(duration, clip)` silence) and the **picture** each segment names by clip id (`Cam(zoom=1.0, cx=0.5, cy=0.5, dissolve_in=0.0)` the source, optionally punched in; `Still(image, path="hold", zoom=None, fit="fit", content_box=None, on_aspect_mismatch="raise", dissolve_in=0.0)` a held still with a slow move: `path` is a `burns` move name such as `"push_in"`/`"drift_left"`, a `BurnsPath` or its dict, resolved by `burns.resolve_move`; a still of another aspect is pillar/letterboxed (`fit="fit"`) unless you ask for `fit="fill"`; windows are slid inside `content_box`). Consecutive segments with the same clip id are ONE picture clip, so a still keeps its move across a sound cut (remove a pause invisibly) and a punch-in hides a jump cut. A `Gap` under a `Cam` holds the neighbouring frame (previous span's last frame; the next span's first when the gap opens the clip). `dissolve_in=D` renders the clip D/2 early and the outgoing one D/2 late (`xfade`), so the length is unchanged. `Overlay(image, start, end, x=0, y=0, fade=0.0)` composites a PNG/PIL image (e.g. a `tituli` render) in the final pass; `closing_fade=` fades picture and sound out. Stdlib-only import, ffmpeg subprocesses; stills need `burns` (extra `media`) unless you pass your own `still_renderer`.

```python
from mixing.edit import Span, Gap, Cam, Still, Overlay, render, timeline

segments = [Span(0.0, 4.2, "a"), Span(5.1, 9.0, "b"), Gap(1.5, "card"), Span(9.0, 12.0, "a")]
clips = {"a": Cam(), "b": Cam(zoom=1.3, cx=0.4), "card": Still("card.png", dissolve_in=0.5)}
render(segments, clips, source="talk.mp4", overlays=[Overlay("lower_third.png", 1.0, 5.0, fade=0.3)])
# -> talk_edit.mp4 beside the source. Inspect the arithmetic first (pure, no ffmpeg):
plan = timeline(segments, clips, fps="30000/1001", sample_rate=48000)
plan.frame_bounds, plan.sample_bounds, plan.picture
```

The timeline runs at the source's exact rate (a `Fraction`; `fps=` overrides); every sound boundary is `round(frame * sr / fps)` of the same frame boundary, so sound and picture never drift. Contiguous source spans play as one run; other sound joins get a 6 ms fade (`JOIN_FADE_S`). Memory is bounded: one small ffmpeg run per piece (lossless intermediates in a temp dir, so budget disk), at most two decoders per run, joined by the concat demuxer.

Source assumptions (refused or warned otherwise): a video AND a sound stream, even width/height; frame-exact identity needs a constant frame rate (a VFR source is resampled to its nominal rate by time, with a warning); a span past the source's end is refused; a sound stream starting late/early is re-synced; display rotation is honoured; overlays past the film's end are clipped (starting after it: refused). Without `ffprobe`, the rate comes from ffmpeg's banner and both streams are assumed to start together.

## Thumbnails

```python
from mixing.video import make_thumbnail, THUMBNAIL_SIZE, YOUTUBE_THUMB_SIZE

make_thumbnail("video.mp4", output="thumb.jpg")  # frame at 85% of duration
make_thumbnail("video.mp4", at_time=12.5, text="Episode 4", output="thumb.jpg")
make_thumbnail(
    "video.mp4", size=YOUTUBE_THUMB_SIZE, output="t/"
)  # 1280x720 (== THUMBNAIL_SIZE)
```

`make_thumbnail(video, *, at_time=None, text=None, output=None, size=(1280,720))`
— all keyword-only after `video`. `text` is overlaid bottom-left on a dark
gradient band. Default `output=None` writes `<stem>.thumb.jpg` beside the video.

## Shots (cuts in a finished video)

```python
from mixing.video import shots, contact_sheet, grab_frame

spans = shots("talk.mp4")  # [(start, end), ...] seconds, tiling the video
spans = shots("talk.mp4", fps=5, min_shot_s=0.5)  # finer cuts; merge dissolve fragments
spans = shots(
    "talk.mp4", strategy="pyscenedetect"
)  # frame-accurate (pip install scenedetect)
contact_sheet(
    "talk.mp4", spans, output="sheet.png"
)  # mid-shot thumbnails, start-end labels
grab_frame("talk.mp4", 12.5, width=320)  # one frame as a PIL image
```

- `strategy`: `"diff"` (default: 2 fps grayscale frame differences streamed from ffmpeg, adaptive threshold, cuts accurate to `1/fps`), `"pyscenedetect"`, or any `(video, **kwargs) -> spans` callable. Strategy options go as keywords (`fps`, `threshold`, `sampler` for `"diff"`).
- `min_shot_s` merges shorter spans into a neighbour (`merge_short_spans`); without it a dissolve reads as a short fake shot.

## Subtitles

```python
from mixing.video import write_subtitles_in_video

write_subtitles_in_video("v.mp4", "subs.srt", output="captioned.mp4")  # SRT path
write_subtitles_in_video("v.mp4", srt_content_string)  # or raw SRT text
write_subtitles_in_video("v.mp4")  # None -> sibling <stem>.srt
```

`write_subtitles_in_video(video, subtitles=None, output=None, *, style=None,
use_ffmpeg=True, start_time=None, auto_detect_audio_start=False, ...)`. Burns
captions in via ffmpeg (~100x faster than the moviepy fallback). `style` is a
`SubtitleStyle(font_size=, color=, position=, font_name=)`. `start_time` (float
seconds, or `True` to auto-detect audio onset) shifts the first cue.

## Dimensions & social presets

`SOCIAL_SIZES` maps names → `(w, h)`: `youtube` (1920x1080), `shorts`/`story`/
`tiktok` (1080x1920), `square` (1080x1080). It is **`looks`' dict, imported** —
`mixing.video.SOCIAL_SIZES is looks.geometry.SOCIAL_SIZES` — along with the
`stretch`/`fit`/`fill` mode names and the two constants that parameterise the
`social` backdrop. One place says what a "shorts" is; the resizing itself stays
here in moviepy. `get_video_dimensions` deliberately did **not** move: it is a
probe (it opens a file), not geometry.

These resize helpers operate on **moviepy clips** and **return clips** (not the
`output` protocol) — get a clip with `Video.to_clip()`, resize, then
`write_videofile`.

```python
from mixing.video import Video, SOCIAL_SIZES, resize_to_dimensions

w, h = SOCIAL_SIZES["shorts"]
with Video("wide.mp4").to_clip() as clip:
    vertical = resize_to_dimensions(clip, w, h, method="social")  # blurred-bg letterbox
    vertical.write_videofile("shorts.mp4")
# methods: "fit" (pad), "fill" (crop), "stretch" (distort), "social" (blurred bg)
# get_video_dimensions(clip) -> (w, h);  normalize_video_dimensions([clips], ...) -> [clips]
```

## Concatenating clips

`concatenate_videos` uses `output=` (a path, or `True` to name from a folder),
but note it **returns a moviepy clip** (not a Path) which the caller must
`.close()`.

```python
from mixing.video import concatenate_videos

final = concatenate_videos(["a.mp4", "b.mp4", "c.mp4"], output="joined.mp4")
final.close()
final = concatenate_videos("clips_folder/", output=True)  # auto-name from folder
final.close()
```

Different sizes are reconciled via `normalize_dimensions` (`"social"` default;
also `"fit"`/`"fill"`/`"stretch"`/`False`). `transform_clips=` lets you inject
transitions (e.g. `crossfade_transition`, `fade_through_black` from the module).

### The join is chosen from the transition, not defaulted

A crossfade lives in the **join**, not in either clip: `CrossFadeIn`/`CrossFadeOut`
only set a mask, and a mask does nothing unless the clips are composited *and*
overlap. moviepy 2.x defaults to neither (`method="chain"`, `padding=0`), which
is why these used to render a silent hard cut (issue #33). `concatenate_videos`
now reads what the transform declares:

| transition | declares | join |
|---|---|---|
| `crossfade_transition` (`duration=0.5`) | 0.5 s overlap | `method="compose", padding=-0.5` |
| `trim_and_crossfade` (`duration=0.4`) | 0.4 s overlap | `method="compose", padding=-0.4` |
| `overlap_blend` (`overlap=0.5`) | 0.5 s overlap | `method="compose", padding=-0.5` |
| `fade_through_black`, `slow_motion_blend` | nothing | back-to-back (they bake the effect into their own frames) |

Consequences worth knowing: an overlapped join makes the output **shorter** than
the sum of its clips by the overlap per join, and **crossfades** the overlapping
audio (the video-mask effects touch no sound, so each of the three transitions
also applies `afx.AudioFadeIn`/`AudioFadeOut` — without them moviepy's
`CompositeAudioClip` would simply *sum* the two tracks at full level, measured
+3.0 dB and clipping on ordinary material). Passing `method=`/`padding=`
yourself always wins, and takes the ceiling below with it.

#### The overlap has a ceiling, and going past it used to delete clips

A crossfade eats the overlap off **each end it touches**, so the first and last
clip pay once and every clip between them pays twice: the ceiling is
`min(duration / joins_it_takes_part_in)`, which is
`mixing.max_overlap_for_clips(clips)`. Past it two things happen at once and
both are silent — a middle clip's fade-in and fade-out masks multiply, so it never
reaches full opacity, and moviepy's `cumsum(durations) + padding * arange`
layout puts clip *i* and clip *i+2* on the same instant (or in the wrong order),
so the later one paints over the middle one. Measured before the fix: three
1 s clips through `overlap_blend` gave a **1 s** video with the middle clip
nowhere in it, and three 0.6 s clips gave 0.5 s from 1.8 s of source.

`concatenate_videos` now **clamps to the ceiling and warns**, rather than
refusing: the clamped value is the largest overlap at which every clip still
reaches full strength, and refusing would make the documented one-liner raise
for any set of sub-second panels — whereupon the caller drops the transition and
goes back to hard cuts. The clamp goes back **through the transform** as well as
into the padding (`@needs_crossfade_overlap` names a *parameter*, which is what
makes that possible), so the ramps and the join stay one number; clamping only
the padding leaves the middle clip peaking at 0.43x its own level.

`functools.partial(crossfade_transition, duration=0.8)` is seen through —
keyword *and* positional binds — so the overlap tracks the duration you asked
for. Writing your own transition:

```python
from mixing import needs_crossfade_overlap, crossfade_overlap


@needs_crossfade_overlap("fade_seconds")  # names the PARAMETER, not a number
def my_transition(clips, *, fade_seconds=0.25): ...


crossfade_overlap(my_transition)  # -> 0.25
```

The name is checked **at decoration time**: a typo or a `**kwargs` signature
raises `TypeError` at import, because there would be nothing to read and the
join would silently go back to back — restoring the exact hard cut issue #33 was
filed for. A declaration that resolves to no value at *render* time (no default,
nothing bound, or a non-number) warns for the same reason. The one case that
stays invisible is a wrapper built without `functools.wraps`, which copies
neither `__dict__` nor the declaration — so wrap transitions with it.

`concatenate_videos` never learns a transition by name, so a new one is a
decoration, not an edit somewhere else.

## AI video generation (Veo) — optional

Needs `pip install 'mixing[gen]'` + Google Cloud auth (`GOOGLE_CLOUD_PROJECT` +
ADC or a service-account JSON). Not on the lazy facade — import explicitly.

```python
from mixing.video.genai import generate_video

path = generate_video(
    "A serene forest at dawn", output="forest.mp4"
)  # output protocol + path
op = generate_video("A serene forest at dawn", output=False)  # raw operation, no save
```

`generate_video(prompt, first_frame=None, last_frame=None, *, output=save_generated_videos,
model=..., aspect_ratio="16:9", duration_seconds=5, project_id=..., ...)`. `output`
follows the egress protocol with two sentinels: default auto-saves to temp and
returns path(s); `output=False` returns the raw operation.

## Gotchas

- `output=None` for file producers writes **next to the input** — pass an
  explicit `output` for a known location.
- `concatenate_videos` follows `output=` but **returns a moviepy clip** (not a
  Path) — you must `.close()` it. `ken_burns_film`'s `output` is required.
- `assemble_audio_track`'s `output` is **required** and it returns `None` (writes
  nothing) when all segments are silent — guard before muxing.
- A `Video` integer/float index returns a **frame** (numpy BGR); a slice returns
  a `Video`. `v[15]` is the frame at 15 *seconds* (or 15 *frames* under
  `time_unit="frames"`).
- `overlay_ambient_bed` is **file-first** (`output=None` writes beside the
  input) even for an audio input — compose `loop_audio`/`duck_audio`/
  `overlay_audio` yourself if you want the `Audio` object back. Its bed is
  attenuated by `mix_ratio` even when the media is silent (a silent base is
  synthesized), and `duck_under_dialogue=True` is a no-op when the media has no
  audio to duck against.
- `change_speed` and `replace_audio(match_duration=True)` change/retime audio too.
- Subtitle `start_time=True` (or `auto_detect_audio_start=True`) needs the video
  to detect audio onset; the moviepy fallback (`use_ffmpeg=False`) is much slower.
