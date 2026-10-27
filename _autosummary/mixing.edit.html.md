# mixing.edit

Render a re-edit of a finished video from a sound edit list and its picture clips.

Re-editing a finished talking-head video with an editor’s eye means two edit
lists of the same length: the **sound** (which source spans play, and where a
silence is inserted) and the **picture** over it (the source, punched in or
not, or a held still with a slow move, dissolving or cutting between clips).
[`render()`](#mixing.edit.render) takes both and renders the film, frame-exact:

```pycon
>>> from mixing.edit import Span, Gap, Cam, Still, render
>>> render(
...     [Span(0.0, 4.2, "a"), Span(5.0, 9.0, "b"), Gap(1.5, "card"), Span(9.0, 12.0, "a")],
...     {"a": Cam(), "b": Cam(zoom=1.3, cx=0.4), "card": Still("card.png", dissolve_in=0.5)},
...     source="talk.mp4",
... )
```

**Sound.** Each segment is a [`Span`](#mixing.edit.Span) of source seconds or a [`Gap`](#mixing.edit.Gap)
of silence. Spans snap to source frames. Consecutive spans that are contiguous
in the source play as one unbroken run; every other join gets a short fade
([`JOIN_FADE_S`](#mixing.edit.JOIN_FADE_S)) so it does not click.

**Picture.** Each segment names a clip id. Consecutive segments with the same
id form **one** picture clip, so a [`Still`](#mixing.edit.Still) keeps its move across a sound
cut (that is how a pause is removed invisibly) and a [`Cam`](#mixing.edit.Cam) punch-in
hides a jump cut. Under a [`Cam`](#mixing.edit.Cam) clip:

- a span shows its own source frames;
- a [`Gap`](#mixing.edit.Gap) **holds the frame next to it**: the last frame of the
  previous span of the same clip, or, when the gap opens the clip, the first
  frame of the next span. (A Cam clip made only of gaps has nothing to show
  and is refused.)

**Frame-exact by construction.** The timeline runs at the source rate as a
[`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction) (`30000/1001` stays exact; `fps=` overrides).
Output frame boundaries are cumulative integer frame counts and each sound
boundary is `round(frame * sample_rate / fps)` of the **same** frame boundary
(one rounding per boundary, never accumulated), so sound and picture never
drift and end on the same instant. [`timeline()`](#mixing.edit.timeline) is that arithmetic, pure
and testable without ffmpeg; [`render()`](#mixing.edit.render) executes it.

**Dissolves.** `dissolve_in=D` on a clip renders it `D/2` early and the
outgoing clip `D/2` late (a Cam clip shows the source frames beyond its
span, a Still runs its move longer), joined with ffmpeg `xfade`, so the total
length is unchanged.

**Picture and sound fades.** `closing_fade=` fades the last frames to black
and the sound to silence over the same frames (the last frame is black and
silent).

**Overlays.** Each [`Overlay`](#mixing.edit.Overlay) is an image with alpha shown over a time
range (optionally fading), composited in the final pass, which also applies an
optional closing fade and muxes the sound. An overlay starting at or after the
film’s end is refused; one running past it is clipped to it, its fades
shortened to fit. A `tituli` label is one way to
make the image (`tituli.render(lower_third(...), frame)` returns a PIL image,
which an [`Overlay`](#mixing.edit.Overlay) accepts); nothing here depends on `tituli`.

**Stills** go through a strategy slot, `still_renderer`, defaulting to
[`burns_still_renderer()`](#mixing.edit.burns_still_renderer): the move is resolved by `burns.resolve_move`
(named moves such as `"push_in"`, or an explicit path), a still of another
aspect is pillar- or letterboxed rather than silently cropped, and windows
stay inside the still’s content box. `burns` comes from the `media` extra
and is imported only when a still is rendered.

**Bounded memory.** The picture is rendered as parts: one small ffmpeg run per
source piece (seek, resample, trim, punch-in) or still slice, one two-input
run per dissolve, each written as a lossless intermediate in a temporary
directory, then read back to back by the concat demuxer in the final pass. No
run opens more than two decoders, however many segments the edit has. The
sound is assembled in memory from one PCM decode of the source (32-bit float:
about 115 MB for five minutes of 48 kHz stereo).

**What is assumed of the source.**

- It has a video and a sound stream (the first of each is used) and an even
  width and height (the delivery is `yuv420p`); others are refused.
- Source seconds count from the first video frame. A sound stream that starts
  later or earlier (its `start_time`) is shifted back into sync.
- Frame-exact identity assumes a constant frame rate. A variable-rate source
  (`r_frame_rate` and `avg_frame_rate` disagree) is resampled by time to
  its nominal rate, each frame the nearest source frame, with a warning;
  `fps=` resamples the same way.
- A display rotation is applied by ffmpeg on decode; the size used is the
  displayed one.
- The source’s length is its counted video packets (`ffprobe`) or, without
  `ffprobe` or for a variable-rate source, its duration; a span past it is
  refused. Without `ffprobe` the rate is read from ffmpeg’s banner (snapped
  by [`as_rate()`](#mixing.edit.as_rate)) and both streams are taken to start together.
- The PCM sound is handled as little-endian float (every platform ffmpeg
  ships for in practice).

Everything runs through `ffmpeg` subprocesses ([`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe));
importing this module pulls no media library.

### Module Attributes

| [`JOIN_FADE_S`](#mixing.edit.JOIN_FADE_S)         | long enough to remove the click of a waveform cut, short enough to be inaudible.   |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`DEFAULT_ENCODE_ARGS`](#mixing.edit.DEFAULT_ENCODE_ARGS) | Encode of the delivered film.                                                      |

### Functions

| [`as_rate`](#mixing.edit.as_rate)(value)                                    | A frame rate as an exact [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction).   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [`punch_in_box`](#mixing.edit.punch_in_box)(width, height, \*, zoom[, cx, cy])   | The `(x, y, w, h)` pixel crop of a punch-in, kept inside the frame.                                           |
| [`timeline`](#mixing.edit.timeline)(segments, clips, \*, fps, sample_rate)   | The frame-exact plan of an edit: frame and sample boundaries, sound and picture pieces.                       |
| [`probe_source`](#mixing.edit.probe_source)(path)                                | Displayed size, exact frame rate, frame count, sound format and stream offsets of `path`.                     |
| [`burns_still_renderer`](#mixing.edit.burns_still_renderer)(still, \*, frames, fps, ...) | Render `still` as exactly `frames` frames of `size` with `burns`.                                             |
| [`render`](#mixing.edit.render)(segments, clips, \*, source[, ...])        | Render the edit `segments` / `clips` of `source`, frame-exact.                                                |

### Classes

| [`Span`](#mixing.edit.Span)(src_in, src_out, clip)                       | Sound from source second `src_in` to `src_out`, under picture clip `clip`.                                          |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`Gap`](#mixing.edit.Gap)(duration, clip)                               | `duration` seconds of silence, under picture clip `clip`.                                                           |
| [`Cam`](#mixing.edit.Cam)([zoom, cx, cy, dissolve_in])                  | The source picture, punched in by `zoom` centred at `(cx, cy)`.                                                     |
| [`Still`](#mixing.edit.Still)(image[, path, zoom, fit, content_box, ...]) | A held still image with a slow move.                                                                                |
| [`Overlay`](#mixing.edit.Overlay)(image, start, end[, x, y, fade])          | An image with alpha shown over output seconds `[start, end)`.                                                       |
| [`SourceInfo`](#mixing.edit.SourceInfo)(width, height, fps[, frames, ...])     | What a render needs to know about its source (see [`probe_source()`](#mixing.edit.probe_source)). |
| [`Timeline`](#mixing.edit.Timeline)(fps, sample_rate, frame_bounds, ...)     | Frame and sample boundaries of an edit (see [`timeline()`](#mixing.edit.timeline)).           |

### *class* mixing.edit.Cam(zoom=1.0, cx=0.5, cy=0.5, dissolve_in=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The source picture, punched in by `zoom` centred at `(cx, cy)`.

`cx`, `cy` are fractions of the frame; the window is cropped (kept
inside the frame) then scaled back to the source size. `dissolve_in` (s)
dissolves into this clip from the previous one.

### mixing.edit.DEFAULT_ENCODE_ARGS *= ('-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart')*

Encode of the delivered film.

### *class* mixing.edit.Gap(duration, clip)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

`duration` seconds of silence, under picture clip `clip`.

### mixing.edit.JOIN_FADE_S *= 0.006*

long
enough to remove the click of a waveform cut, short enough to be inaudible.

* **Type:**
  Fade (s) at every sound join that is not a continuous source run

### *class* mixing.edit.Overlay(image, start, end, x=0, y=0, fade=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An image with alpha shown over output seconds `[start, end)`.

`image` is a path or anything with `.save(path)` (a PIL image, e.g. a
`tituli` render). `x`, `y` are its position (pixels or an ffmpeg
`overlay` expression). `fade` (s) fades it in and out inside the range.

### *class* mixing.edit.SourceInfo(width, height, fps, frames=None, sample_rate=None, channels=None, video_start=0.0, audio_offset=0.0, variable_rate=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What a render needs to know about its source (see [`probe_source()`](#mixing.edit.probe_source)).

`width` x `height` is the **displayed** size (a 90/270-degree display
rotation swaps them; ffmpeg rotates on decode). `frames` is the frame
count at `fps`. `video_start` is where the first video frame sits after
the file’s start (what an input seek counts from), and `audio_offset`
how far the sound starts after the first video frame (negative: before).
`variable_rate` says the picture is not at a constant rate (it is then
resampled by time, and `frames` comes from the duration).

### *class* mixing.edit.Span(src_in, src_out, clip)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sound from source second `src_in` to `src_out`, under picture clip `clip`.

### *class* mixing.edit.Still(image, path='hold', zoom=None, fit='fit', content_box=None, on_aspect_mismatch='raise', dissolve_in=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A held still image with a slow move.

`image` is a path or a PIL image. `path` is the move, resolved by
`burns.resolve_move`: a name from `burns.MOVES` (`"hold"`, the
default, `"push_in"`, `"drift_left"`, …), a `burns.BurnsPath`, or
its `to_dict()` payload. `zoom` is a named move’s end zoom (`None`:
`HOLD_ZOOM` for a hold, `burns`’ default otherwise).

`fit` decides what happens when the image’s aspect is not the frame’s:
`"fit"` (default) letter- or pillarboxes it, so the whole picture is
held and never silently cover-cropped; `"fill"` lets `burns` crop it to
cover the frame. `content_box` (normalized `(x, y, w, h)` of the image)
is the part that is picture, e.g. inside a pillarboxed card: every window
of the move is slid inside it (centred on it where the window is larger),
so a zoomed move never shows the bars. Bars added by `fit="fit"` are
outside the content box automatically. A transparent still is flattened
onto `PAD_COLOR`, its alpha honoured. `on_aspect_mismatch` is passed to
`burns.resolve_move` for an explicit path authored at another aspect
(`"raise"`, default, or `"refit"`).

The move spans the whole picture clip, whatever sound cuts happen under it.

### *class* mixing.edit.Timeline(fps, sample_rate, frame_bounds, sample_bounds, sound, picture)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Frame and sample boundaries of an edit (see [`timeline()`](#mixing.edit.timeline)).

`frame_bounds[k]` / `sample_bounds[k]` is where segment `k` starts;
the last entry is the total.

### mixing.edit.as_rate(value)

A frame rate as an exact [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction).

Strings are parsed exactly; a float near an integer or an NTSC rate
(`k * 1000 / 1001`) is taken to mean it.

```pycon
>>> as_rate("30000/1001"), as_rate(29.97), as_rate(25), as_rate(23.976)
(Fraction(30000, 1001), Fraction(30000, 1001), Fraction(25, 1), Fraction(24000, 1001))
```

* **Return type:**
  [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction)

### mixing.edit.burns_still_renderer(still, , frames, fps, size, output)

Render `still` as exactly `frames` frames of `size` with `burns`.

The move is resolved by `burns.resolve_move` at the frame’s aspect
(named moves, explicit paths, and its `on_aspect_mismatch` refusal);
the image is letter- or pillarboxed to that aspect first when
`fit="fit"`, and every window is slid inside the content box (see
[`Still`](#mixing.edit.Still)). `burns` renders integer rates only, so a fractional
`fps` (29.97) is rendered at the next integer with the \*\*same frame
count\*\*; each part restamps the frames at the edit’s rate (the move runs
0.1% slower; no frame is dropped or repeated). The default
`still_renderer` of [`render()`](#mixing.edit.render); any callable with this signature
can replace it.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.edit.probe_source(path)

Displayed size, exact frame rate, frame count, sound format and stream
offsets of `path`.

Uses `ffprobe` (its `r_frame_rate` is the exact rational; packets are
counted, which reads but does not decode the file) when one is found
beside ffmpeg or on `PATH`, else ffmpeg’s banner (rate snapped by
[`as_rate()`](#mixing.edit.as_rate), frame count from the duration, no stream offsets). Warns
when the source looks variable-rate.

* **Return type:**
  [`SourceInfo`](#mixing.edit.SourceInfo)

### mixing.edit.punch_in_box(width, height, , zoom, cx=0.5, cy=0.5)

The `(x, y, w, h)` pixel crop of a punch-in, kept inside the frame.

```pycon
>>> punch_in_box(64, 36, zoom=2.0, cx=0.0, cy=0.0)
(0, 0, 32, 18)
>>> punch_in_box(1920, 1080, zoom=1.5)
(320, 180, 1280, 720)
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### mixing.edit.render(segments, clips, \*, source, overlays=(), fps=None, output=None, still_renderer=<function burns_still_renderer>, closing_fade=0.0, join_fade=0.006, encode_args=('-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-c: a', 'aac', '-b: a', '192k', '-movflags', '+faststart'))

Render the edit `segments` / `clips` of `source`, frame-exact.

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Span`](#mixing.edit.Span), [`Gap`](#mixing.edit.Gap)]]) – the sound edit list, [`Span`](#mixing.edit.Span) / [`Gap`](#mixing.edit.Gap).
  * **clips** ([`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable), [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Cam`](#mixing.edit.Cam), [`Still`](#mixing.edit.Still)]]) – picture clip id -> [`Cam`](#mixing.edit.Cam) / [`Still`](#mixing.edit.Still).
  * **source** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – the finished video being re-edited (must have sound, and an
    even size).
  * **overlays** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Overlay`](#mixing.edit.Overlay)]) – [`Overlay`](#mixing.edit.Overlay) images composited in the final pass.
  * **fps** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – the timeline rate; default the source’s exact rate.
  * **output** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – where the film goes (see [`mixing.egress`](mixing.egress.html.md#module-mixing.egress)); default beside
    `source` as `<stem>_edit.mp4`.
  * **still_renderer** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `still_renderer(still, *, frames, fps, size, output)`
    renders a [`Still`](#mixing.edit.Still) as `frames` frames (see
    [`burns_still_renderer()`](#mixing.edit.burns_still_renderer), the default).
  * **closing_fade** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – seconds of fade to black (and to silence) at the end.
  * **join_fade** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – seconds of fade at each sound join (see [`JOIN_FADE_S`](#mixing.edit.JOIN_FADE_S)).
  * **encode_args** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]) – the final encode’s ffmpeg arguments.
* **Return type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  The written `Path` or, when `output` is a callable, what it
  returns for that path.

### mixing.edit.timeline(segments, clips, , fps, sample_rate, source_frames=None, join_fade=0.006)

The frame-exact plan of an edit: frame and sample boundaries, sound and
picture pieces. Pure: no file is read.

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – [`Span`](#mixing.edit.Span) / [`Gap`](#mixing.edit.Gap).
  * **clips** ([`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable), [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Cam`](#mixing.edit.Cam), [`Still`](#mixing.edit.Still)]]) – clip id -> [`Cam`](#mixing.edit.Cam) / [`Still`](#mixing.edit.Still).
  * **fps** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the timeline rate (see [`as_rate()`](#mixing.edit.as_rate)).
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – the sound’s sample rate.
  * **source_frames** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – the source’s frame count at `fps`: a span past it is
    refused, and dissolve handles past it hold the last frame
    (`None`: assume the source runs on).
  * **join_fade** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – fade (s) at each sound join that is not a continuous run.
* **Return type:**
  [`Timeline`](#mixing.edit.Timeline)

```pycon
>>> plan = timeline([Span(0, 1, "a"), Gap(0.5, "a")], {"a": Cam()},
...                 fps="30000/1001", sample_rate=48000)
>>> plan.frame_bounds, plan.sample_bounds
((0, 30, 45), (0, 48048, 72072))
```
